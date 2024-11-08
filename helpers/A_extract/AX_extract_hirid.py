# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.helper_filepaths import HiRIDPaths
from helpers.helper import GlobalHelpers


class HiRIDExtractor(HiRIDPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

        self.other_lab_values = [
            "Creatinine [Moles/volume] in Blood",
            "Creatinine [Moles/volume] in Urine",
            "Glucose [Moles/volume] in Serum or Plasma",
            "Urea nitrogen [Moles/volume] in Serum or Plasma",
            "Alkaline phosphatase [Enzymatic activity/volume] in Blood",
            "Creatine kinase panel - Serum or Plasma",
            "Erythrocyte sedimentation rate",
            "Ferritin [Mass/volume] in Blood",
            "INR in Blood by Coagulation assay",
            "MCHC [Mass/volume] in Cord blood",
            "Magnesium [Moles/volume] in Blood",
            "Phosphate [Moles/volume] in Blood",
            "aPTT in Blood by Coagulation assay",
            "Creatine kinase.MB [Mass/volume] in Serum or Plasma",
            "Lactate [Mass/volume] in Arterial blood",
            "Urea [Moles/volume] in Venous blood",
            "Lymphocytes [#/volume] in Blood",
        ]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        # The general table contains neither the height nor the weight of the patients.
        # Also, the length of stay is not specified.
        # All must be fetched from the timeseries table.

        return (
            self._extract_admissions()
            .join(
                self._extract_length_of_stay(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                self._extract_patient_height_weight(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                self.extract_admit_diagnoses(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                # Set care site
                pl.lit("Universitätsspital Bern").alias(self.care_site_col),
                # Set unit type
                # NOTE: the Bern University Hospital only has one unit type
                # -> all ICU patients are cared for within a interdisciplinary 60-bed unit in the Department of Intensive Care Medicine
                pl.lit("Medical-Surgical")
                .replace(self.UNIT_TYPES_MAP)
                .cast(self.unit_types_dtype)
                .first()
                .alias(self.unit_type_col),
                # Get admission time
                pl.col("admissiontime")
                .dt.time()
                .alias(self.admission_time_col),
            )
        )

    # endregion

    # region admissions
    def _extract_admissions(self) -> pl.LazyFrame:
        return (
            pl.scan_csv(
                self.general_table_path, schema_overrides={"admissiontime": str}
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientid": self.icu_stay_id_col,
                    "sex": self.gender_col,
                    "age": self.age_col,
                }
            )
            .with_columns(
                pl.col(self.icu_stay_id_col).cast(str),
                # Convert the admission time to datetime
                pl.col("admissiontime").str.to_datetime(
                    "%Y-%m-%d %H:%M:%S%.9f"
                ),
                # Convert the gender to the established format
                pl.col(self.gender_col)
                .replace({"M": "Male", "F": "Female"})
                .cast(self.gender_dtype),
                # Convert the age to int
                pl.col(self.age_col).cast(int),
                # Convert the discharge status to the established format
                pl.when(pl.col("discharge_status") != "")
                .then(pl.col("discharge_status") == "dead")
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_icu_col),
                pl.when(pl.col("discharge_status") == "dead")
                .then(True)
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_hosp_col),
            )
            .drop("discharge_status")
        )

    # endregion

    # region len of stay
    def _extract_length_of_stay(self) -> pl.LazyFrame:
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_lengths_of_stay.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "HiRID_lengths_of_stay.parquet"
            )

        print("HiRID   - Processing patient length of stay data...")

        # The length of stay is derived from the last measurement of a timeseries variable.
        lengths_of_stay = (
            pl.scan_parquet(self.imputed_stage_path + "*.parquet")
            .select("patientid", "reldatetime")
            .drop_nulls()
            .rename(
                {
                    "patientid": self.icu_stay_id_col,
                    "reldatetime": self.icu_length_of_stay_col,
                }
            )
            .cast({self.icu_stay_id_col: str})
            .group_by(self.icu_stay_id_col)
            .max()
            # Convert the length of stay to days
            .with_columns(
                pl.duration(seconds=pl.col(self.icu_length_of_stay_col))
                .truediv(pl.duration(days=1))
                .alias(self.icu_length_of_stay_col)
            )
        )

        # Save precalculated data
        lengths_of_stay.sink_parquet(
            self.precalc_path + "HiRID_lengths_of_stay.parquet"
        )

        return lengths_of_stay

    # endregion

    # region h/weight
    def _extract_patient_height_weight(self) -> pl.LazyFrame:
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_height_weight.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "HiRID_height_weight.parquet"
            )

        print("HiRID   - Processing patient height and weight data...")

        # The height and weight are derived from the last measurement of a timeseries variable.
        variables = {10000400: self.weight_col, 10000450: self.height_col}
        admissiontimes = (
            pl.scan_csv(self.general_table_path)
            .select(["patientid", "admissiontime"])
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str, "admissiontime": str})
        )

        # Create an empty DataFrame to store the height and weight data
        height_weight = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in os.listdir(self.timeseries_path):
            # Extract the data from the file
            data = (
                pl.scan_parquet(self.timeseries_path + file)
                # Select the relevant columns
                .select(["patientid", "datetime", "value", "variableid"])
                # Rename the columns for consistency
                .rename(
                    {"patientid": self.icu_stay_id_col, "datetime": "valuedate"}
                )
                .cast(
                    {
                        self.icu_stay_id_col: str,
                        "valuedate": str,
                        "variableid": int,
                    }
                )
                # Drop rows with missing values
                .drop_nulls()
                # Join the data with the admission times
                .join(admissiontimes, on=self.icu_stay_id_col)
                # Convert the admission time and the value date to datetime
                .with_columns(
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("valuedate").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    # Replace the variableid with the corresponding variable name
                    pl.col("variableid").replace(variables, default=None),
                )
                # Filter for variables of interest within the cutoff time
                .filter(
                    (pl.col("valuedate") - pl.col("admissiontime"))
                    < pl.duration(hours=self.ADMISSION_WEIGHT_HEIGHT_CUTOFF),
                    pl.col("variableid").is_in(variables.values()),
                )
                .drop(["admissiontime", "valuedate"])
            )

            # Append the data to the DataFrame
            height_weight = pl.concat(
                [height_weight, data], how="diagonal_relaxed"
            )

        height_weight = (
            height_weight.collect(streaming=True)
            .pivot(
                on="variableid",
                index=self.icu_stay_id_col,
                values="value",
                aggregate_function="max",
            )
            .select([self.icu_stay_id_col, self.weight_col, self.height_col])
        )

        height_weight.write_parquet(
            self.precalc_path + "HiRID_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region admitDX
    def extract_admit_diagnoses(self) -> pl.LazyFrame:
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_admitDX.parquet"):
            return pl.scan_parquet(self.precalc_path + "HiRID_admitDX.parquet")

        print("HiRID   - Extracting admission diagnoses...")

        # Load the mapping of the diagnoses
        hirid_diagnosis_mapping = self.load_mapping(self.apache_mapping_path)

        # Create an empty DataFrame to store the admission diagnoses data
        admitDX = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in os.listdir(self.timeseries_path):
            # Extract the data from the file
            data = (
                pl.scan_parquet(self.timeseries_path + file)
                .select("patientid", "datetime", "variableid", "value")
                .rename({"patientid": self.icu_stay_id_col})
                .filter(pl.col("variableid").is_in([9990002, 9990004]))
            )

            # Append the data to the DataFrame
            admitDX = pl.concat([admitDX, data], how="diagonal_relaxed")

        admitDX = (
            admitDX.sort(self.icu_stay_id_col, "datetime")
            .group_by(self.icu_stay_id_col)
            .agg(pl.col("value").first())
            .with_columns(
                pl.col("value")
                .replace(hirid_diagnosis_mapping, default=None)
                .alias(self.admission_diagnosis_col)
            )
            .select(self.icu_stay_id_col, self.admission_diagnosis_col)
            .cast({self.icu_stay_id_col: str})
        )

        admitDX.sink_parquet(self.precalc_path + "HiRID_admitDX.parquet")

        return admitDX

    # region timeseries
    # Extract timeseries information from the timeseries file directory
    def extract_timeseries(self) -> pl.LazyFrame:
        # observation_mapping = self.load_mapping(self.observation_mapping_path)
        admissiontime = (
            self._extract_admissions()
            .select([self.icu_stay_id_col, "admissiontime"])
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        # Create an empty DataFrame to store the timeseries data
        timeseries = pl.LazyFrame()
        timeseries_labs = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in os.listdir(self.timeseries_path):
            data_ = pl.scan_parquet(self.timeseries_path + file).pipe(
                self._extract_timeseries_helper,
                admissiontime,
                length_of_stay,
                # observation_mapping,
            )

            # Separate the lab values from the rest
            data_labs = data_.pipe(self._extract_timeseries_labs_helper)
            data = data_.filter(
                ~pl.col("variableid").is_in(
                    self.relevant_lab_values + self.other_lab_values
                )
            )

            # Append the data to the DataFrame
            timeseries = pl.concat([timeseries, data], how="diagonal_relaxed")
            timeseries_labs = pl.concat(
                [timeseries_labs, data_labs], how="diagonal_relaxed"
            )

        return timeseries, timeseries_labs

    def _extract_timeseries_helper(
        self,
        data: pl.LazyFrame,
        admissiontime: pl.LazyFrame,
        length_of_stay: pl.LazyFrame,
        # observation_mapping: dict,
    ) -> pl.LazyFrame:
        return (
            data.select(["patientid", "datetime", "variableid", "value"])
            # Rename columns for consistency
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str, "datetime": str})
            .join(admissiontime, on=self.icu_stay_id_col)
            .join(length_of_stay, on=self.icu_stay_id_col)
            .with_columns(
                pl.col("admissiontime").str.to_datetime(
                    "%Y-%m-%d %H:%M:%S%.9f"
                ),
                pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                # Replace the variableid with the corresponding variable name
                # then the reprodICU mapping
                pl.col("variableid")
                .cast(int)
                .replace_strict(
                    self._get_observation_variables(), default=None
                ),
                # .replace_strict(observation_mapping, default=None),
                pl.col("value").cast(float),
            )
            .with_columns(
                (
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .truediv(pl.duration(seconds=1))
                    .round(0)
                ).alias(self.timeseries_time_col)
            )
            .drop(["admissiontime", "datetime"])
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (
                    pl.col(self.timeseries_time_col)
                    < pl.duration(
                        days=pl.col(self.icu_length_of_stay_col)
                    ).truediv(pl.duration(seconds=1))
                )
                & (
                    pl.col(self.timeseries_time_col)
                    > pl.duration(
                        days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    ).truediv(pl.duration(seconds=1))
                )
            )
            # Filter for names of interest
            .filter(
                pl.col("variableid").is_in(
                    self.all_values + self.other_lab_values
                )
            )
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("value").is_not_null())
            # Remove rows with empty lab results
            .filter(
                pl.col("variableid").is_not_null()
                & (pl.col("variableid") != "")
            )
        )

    # endregion

    # region ts labs
    def _extract_timeseries_labs_helper(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        return (
            data.filter(
                pl.col("variableid").is_in(
                    self.relevant_lab_values + self.other_lab_values
                )
            )
            # MAKE STRUCT
            .with_columns(
                pl.col("variableid")
                .str.split_exact(by=" by ", n=1)
                .struct.rename_fields(["variable_source", "method"])
                .alias("fields1")
            )
            .unnest("fields1")
            .with_columns(
                pl.col("variable_source")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "source"])
                .alias("fields2")
            )
            .unnest("fields2")
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                pl.col("variable")
                .str.replace("inHDL", "in HDL")
                .str.replace("inLDL", "in LDL")
                .alias("variableid"),
                pl.struct(
                    value="value", source="source", method="method"
                ).alias("value_struct"),
            )
        )

    # region pharma
    # Extract pharma information from the pharma file directory
    def extract_medications(self) -> pl.LazyFrame:
        print("HiRID   - Extracting medications...")

        hirid_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "hirid"
            )
        )
        hirid_drug_class_mapping = self.load_mapping(
            self.drug_administration_route_mapping_path
        )
        hirid_drug_administration_route_mapping = self.load_mapping(
            self.drug_administration_route_mapping_path
        )

        admissiontime = (
            self._extract_admissions()
            .select([self.icu_stay_id_col, "admissiontime"])
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        # Create an empty DataFrame to store the pharma data
        pharma = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in os.listdir(self.pharma_path):
            print(f"Processing file {file}...", end="\r")
            data = (
                pl.scan_parquet(self.pharma_path + file)
                .select(
                    [
                        "patientid",
                        "pharmaid",
                        "givenat",
                        "givendose",
                        "doseunit",
                        "route",
                        "infusionid",
                        "subtypeid",
                    ]
                )
                # Rename columns for consistency
                .rename(
                    {
                        "patientid": self.icu_stay_id_col,
                        "givendose": self.drug_amount_col,
                        "doseunit": self.drug_amount_unit_col,
                        "route": self.drug_admin_route_col,
                        "subtypeid": self.drug_class_col,
                    }
                )
                # Cast the datetime to string to avoid the following error:
                # polars.exceptions.SchemaError: invalid series dtype: expected `String`, got `datetime[ns]`
                .cast({self.icu_stay_id_col: str, "givenat": str})
                .join(admissiontime, on=self.icu_stay_id_col)
                .join(length_of_stay, on=self.icu_stay_id_col)
                .with_columns(
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("givenat").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                    pl.col(self.drug_amount_col).cast(float),
                    # Replace the pharmaid with the corresponding medication name
                    pl.col("pharmaid")
                    .cast(int)
                    .replace_strict(self._get_pharma_variables(), default=None)
                    .alias(self.drug_name_col),
                )
                .with_columns(
                    (pl.col("givenat") - pl.col("admissiontime"))
                    .truediv(pl.duration(seconds=1))
                    .round(0)
                    .alias(self.drug_end_col),
                    # Map the medication names to the ingredients
                    pl.col(self.drug_name_col)
                    .replace_strict(hirid_medication_mapping, default=None)
                    .alias(self.drug_ingredient_col),
                    # Map the medication classes
                    pl.col(self.drug_class_col)
                    .cast(str)
                    .replace_strict(
                        hirid_drug_class_mapping,
                        default=None,
                        return_dtype=pl.String,
                    ),
                    # Map the medication administration routes
                    pl.col(self.drug_admin_route_col).replace_strict(
                        hirid_drug_administration_route_mapping,
                        default=None,
                        return_dtype=pl.String,
                    ),
                )
                .drop("admissiontime", "givenat")
                # Remove duplicate rows
                .unique()
                # Remove rows with empty lab names
                .filter(pl.col(self.drug_amount_col).is_not_null())
                # Keep only drugs within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
                .filter(
                    (
                        pl.col(self.drug_end_col)
                        < pl.duration(
                            days=pl.col(self.icu_length_of_stay_col)
                        ).truediv(pl.duration(seconds=1))
                    )
                    & (
                        pl.col(self.drug_end_col)
                        > pl.duration(
                            days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).truediv(pl.duration(seconds=1))
                    )
                )
                .drop(self.icu_length_of_stay_col)
            )

            # Append the data to the DataFrame
            pharma = pl.concat([pharma, data.lazy()], how="diagonal_relaxed")

        # Get infusion duration where possible, by checking whether the drugname reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        pharma_offsets = (
            pharma.select(self.icu_stay_id_col, self.drug_end_col, "infusionid")
            .unique()
            .sort([self.icu_stay_id_col, "infusionid", self.drug_end_col])
            .with_columns(
                pl.col(self.drug_end_col)
                .shift(1)
                .over(self.icu_stay_id_col, "infusionid")
                .alias("prev_drug_end"),
                pl.col(self.drug_end_col)
                .shift(-1)
                .over(self.icu_stay_id_col, "infusionid")
                .alias("next_drug_end"),
            )
        )

        pharma = (
            pharma.join(
                pharma_offsets,
                on=[self.icu_stay_id_col, self.drug_end_col, "infusionid"],
                how="left",
            )
            # Sort by patient ID, drug name and drug start time
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                "infusionid",
                self.drug_end_col,
                "prev_drug_end",  # sometimes, there is the same drug given twice at the same time
            )
            # NOTE: Convert drug_amount to drug_rates
            .with_columns(
                (
                    (pl.col(self.drug_amount_col) * 60 * 60)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                )
                .round(1)
                .alias(self.drug_rate_col),
                pl.col(self.drug_amount_unit_col).str.replace("µ", "mc"),
                (pl.col(self.drug_amount_unit_col) + pl.lit("/hr"))
                .str.replace("µ", "mc")
                .alias(self.drug_rate_unit_col),
            )
            # 2. Check if drug is continued from the previous log entry
            #    and if it is continued in the next log entry
            .with_columns(
                # Check if drug is continued from the previous log entry
                pl.when(pl.col("prev_drug_end").is_not_null())
                .then(
                    pl.when(
                        # Check if the previous drug is the same as the current drug
                        pl.col(self.drug_name_col)
                        == pl.col(self.drug_name_col).shift(1),
                        # Check if the previous drug end time is the previous log entry time
                        pl.col("prev_drug_end")
                        == pl.col(self.drug_end_col).shift(1),
                        # Check if the drug rate is the same as the previous drug rate
                        pl.col(self.drug_rate_col)
                        == pl.col(self.drug_rate_col).shift(1),
                    )
                    .then(pl.lit("continued"))
                    .otherwise(pl.lit("started"))
                )
                .otherwise(None)
                .alias("drug_status_prev"),
                # Check if drug is continued in the next log entry
                pl.when(
                    # Check if the next drug is the same as the current drug
                    pl.col(self.drug_name_col)
                    == pl.col(self.drug_name_col).shift(-1),
                    # Check if the next drug end time is the next log entry time
                    pl.col("next_drug_end")
                    == pl.col(self.drug_end_col).shift(-1),
                    # Check if the drug rate is the same as the next drug rate
                    pl.col(self.drug_rate_col)
                    == pl.col(self.drug_rate_col).shift(-1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("discontinued"))
                .alias("drug_status_next"),
            )
            # Filter for rows where the drug status changes
            .filter(pl.col("drug_status_prev") != pl.col("drug_status_next"))
            # 3. Get the end time of the drug if it is discontinued
            .with_columns(
                pl.when(pl.col("drug_status_next") == "discontinued")
                .then("prev_drug_end")
                .otherwise(None)
                .alias(self.drug_start_col)
            )
            # 4. Combine rows where the drug is started, continued, then discontinued in the next row
            .with_columns(
                pl.when(
                    pl.col("drug_status_prev").shift(1) == "started",
                    pl.col("drug_status_next").shift(1) == "continued",
                    pl.col("drug_status_prev") == "continued",
                    pl.col("drug_status_next") == "discontinued",
                    # Check if the previous drug is the same as the current drug
                    pl.col(self.drug_name_col)
                    == pl.col(self.drug_name_col).shift(1),
                    # Check if the drug amount is the same as the previous drug amount
                    pl.col(self.drug_rate_col)
                    == pl.col(self.drug_rate_col).shift(1),
                )
                .then(pl.col("prev_drug_end").shift(1))
                .otherwise(pl.col("prev_drug_end"))
                .alias(self.drug_start_col)
            )
            # 5. filter out duplicate rows (same drug, same start time, same rate, different end time)
            .filter(pl.col(self.drug_start_col).is_not_null())
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_start_col,
                self.drug_rate_col,
                self.drug_end_col,
            )
            .group_by(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_start_col,
                self.drug_rate_col,
                maintain_order=True,
            )
            .last()
            # 6. Remove the helper columns
            .drop(
                "prev_drug_end",
                "next_drug_end",
                "drug_status_prev",
                "drug_status_next",
                "infusionid",
            )
        )

        return pharma

    # endregion

    # region helpers
    def _get_variable_reference(self) -> pl.DataFrame:
        return pl.read_csv(
            self.variable_reference_path,
            # separator=";",
            # encoding="unicode_escape",
            columns=["Source Table", "ID", "Variable Name"],
        )

    def _get_observation_variables(self) -> pl.DataFrame:
        observation_variables = (
            self._get_variable_reference()
            .filter(pl.col("Source Table") == "Observation")
            .drop("Source Table")
        ).with_columns(
            # Fix bad mappings (wrong units or spelling unequal to reprodICU harmonization)
            pl.col("Variable Name").replace(
                {
                    "Bilirubin.direct [Mass/volume] in Serum or Plasma": "Bilirubin.direct [Moles/volume] in Serum or Plasma",
                    "Glasgow Coma Score verbal response subscore": "Glasgow Coma Score verbal",
                    "Glasgow Coma Score motor response subscore": "Glasgow Coma Score motor",
                    "Glasgow Coma Score eye opening subscore": "Glasgow Coma Score eye opening",
                }
            )
        )

        return dict(
            zip(
                observation_variables["ID"].to_numpy(),
                observation_variables["Variable Name"].to_numpy(),
            )
        )

    def _get_pharma_variables(self) -> pl.DataFrame:
        pharma_variables = (
            self._get_variable_reference()
            .filter(pl.col("Source Table") == "Pharma")
            .drop("Source Table")
        )

        return dict(
            zip(
                pharma_variables["ID"].to_numpy(),
                pharma_variables["Variable Name"].to_numpy(),
            )
        )
