# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts data from HiRID source files and converts them into a structured format for harmonization.

import os
from pathlib import Path

import polars as pl

from ..helper import GlobalHelpers
from ..helper_filepaths import HiRIDPaths
from ..helper_OMOP import Vocabulary


class HiRIDExtractor(HiRIDPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extract and harmonize patient demographics and clinical information.

        Steps:
            1. Extract admissions data with basic demographics and mortality flags.
            2. Join with length of stay computed from timeseries.
            3. Join with height and weight extracted from timeseries.
            4. Join with admission diagnoses and specialty mapping.
            5. Assign constant columns (care site, unit type, admission time).

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {gender_col}: Patient gender.
                - {age_col}: Patient age (years).
                - {height_col}: Patient height (cm).
                - {weight_col}: Patient weight (kg).
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_hosp_col}: Hospital mortality flag.
                - {icu_length_of_stay_col}: ICU length of stay (days).
                - {admission_diagnosis_col}: Admission diagnosis mapped to APACHE group.
                - {specialty_col}: Treating specialty.
                - {care_site_col}: Care site identifier.
                - {unit_type_col}: ICU unit type.
                - {admission_time_col}: ICU admission time of day.
        """
        return (
            self._extract_admissions()
            .join(
                self._extract_length_of_stay(),
                on=self.icu_stay_id_col,
                how="left",
                coalesce=True,
            )
            .join(
                self._extract_patient_height_weight(),
                on=self.icu_stay_id_col,
                how="left",
                coalesce=True,
            )
            .join(
                self.extract_admit_diagnoses(),
                on=self.icu_stay_id_col,
                how="left",
                coalesce=True,
            )
            .with_columns(
                # Set care site
                pl.lit("Inselspital - Universitätsspital Bern").alias(
                    self.care_site_col
                ),
                # Set unit type
                # NOTE: the Bern University Hospital only has one unit type
                # -> all ICU patients are cared for within a interdisciplinary 60-bed unit in the Department of Intensive Care Medicine
                pl.lit("Intensive care unit")
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
        """
        Load and process admissions data.

        Steps:
            1. Read CSV from {general_table_path} with admissiontime as string.
            2. Rename columns to standardized names for ID, gender, and age.
            3. Convert admissiontime to datetime.
            4. Map gender values (M→Male, F→Female).
            5. Compute mortality flags based on discharge status.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {gender_col}: Patient gender.
                - {age_col}: Patient age (years).
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_hosp_col}: Hospital mortality flag.
                - admissiontime: ICU admission datetime.
        """
        return (
            pl.scan_csv(
                self.general_table_path,
                schema_overrides={"admissiontime": str},
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
        """
        Compute ICU length of stay from timeseries data.

        Steps:
            1. Check for precomputed parquet file; load if available.
            2. Otherwise, scan timeseries parquet files from {timeseries_path}.
            3. Extract maximum relative time offset per patient.
            4. Convert duration from seconds to days.
            5. Save result as parquet file for future reuse.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {icu_length_of_stay_col}: ICU length of stay (days).
        """
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_lengths_of_stay.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "HiRID_lengths_of_stay.parquet"
            )

        print("HiRID   - Processing patient length of stay data...")

        # The length of stay is derived from the last measurement of a timeseries variable.
        lengths_of_stay = (
            pl.scan_parquet(self.timeseries_path + "*.parquet")
            .select("patientid", "datetime")
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str})
            .join(
                self._extract_admissions().select(
                    self.icu_stay_id_col, "admissiontime"
                ),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                pl.col("datetime")
                .sub(pl.col("admissiontime"))
                .dt.total_seconds()
                .alias(self.icu_length_of_stay_col)
            )
            .drop_nulls()
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
        """
        Extract patient height and weight from timeseries data.

        Steps:
            1. Check for precomputed parquet file; load if available.
            2. Otherwise, read admissiontime from {general_table_path}.
            3. For each timeseries file, extract height and weight within cutoff window.
            4. Pivot data so each patient maps to a single height and weight value.
            5. Save result as parquet file for future reuse.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {weight_col}: Patient weight (kg).
                - {height_col}: Patient height (cm).
        """
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
            .select("patientid", "admissiontime")
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str, "admissiontime": str})
        )

        # Create an empty DataFrame to store the height and weight data
        height_weight = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in Path(self.timeseries_path).glob("*.parquet"):
            # Extract the data from the file
            data = (
                pl.scan_parquet(file)
                # Select the relevant columns
                .select("patientid", "datetime", "value", "variableid")
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
                .drop("admissiontime", "valuedate")
            )

            # Append the data to the DataFrame
            height_weight = pl.concat(
                [height_weight, data], how="diagonal_relaxed"
            )

        height_weight = (
            height_weight.collect()
            .pivot(
                on="variableid",
                index=self.icu_stay_id_col,
                values="value",
                aggregate_function="max",
            )
            .select(self.icu_stay_id_col, self.weight_col, self.height_col)
        )

        height_weight.write_parquet(
            self.precalc_path + "HiRID_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region admitDX
    def extract_admit_diagnoses(self) -> pl.LazyFrame:
        """
        Extract and map admission diagnoses and specialty information.

        Steps:
            1. Check for precomputed parquet file; load if available.
            2. Otherwise, load diagnosis and specialty mappings.
            3. For each timeseries file, extract diagnosis events.
            4. Select first diagnosis occurrence per patient, grouped by time.
            5. Map diagnosis and specialty values using reference mappings.
            6. Save result as parquet file for future reuse.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {admission_diagnosis_col}: Admission diagnosis mapped to APACHE group.
                - {specialty_col}: Treating specialty.
        """
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_admitDX.parquet"):
            return pl.scan_parquet(self.precalc_path + "HiRID_admitDX.parquet")

        print("HiRID   - Extracting admission diagnoses...")

        # Load the mapping of the diagnoses
        hirid_diagnosis_mapping = self.load_mapping(self.apache_mapping_path)
        hirid_specialty_mapping = self.load_mapping(self.specialty_mapping_path)

        # Create an empty DataFrame to store the admission diagnoses data
        admitDX = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in Path(self.timeseries_path).glob("*.parquet"):
            # Extract the data from the file
            data = (
                pl.scan_parquet(file)
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
                .alias(self.admission_diagnosis_col),
                pl.col("value")
                .replace(hirid_specialty_mapping, default=None)
                .alias(self.specialty_col),
            )
            .select(
                self.icu_stay_id_col,
                self.admission_diagnosis_col,
                self.specialty_col,
            )
            .cast({self.icu_stay_id_col: str})
        )

        # admitDX.sink_parquet(self.precalc_path + "HiRID_admitDX.parquet")
        admitDX.collect().write_parquet(
            self.precalc_path + "HiRID_admitDX.parquet"
        )

        return admitDX

    # region timeseries
    # Extract timeseries information from the timeseries file directory
    def _extract_timeseries_helper(
        self,
        data: pl.LazyFrame,
        admissiontime: pl.LazyFrame,
        length_of_stay: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """
        Process and align timeseries events relative to ICU admission.

        Steps:
            1. Join data with admission time and length of stay.
            2. Convert admission time and event datetime to datetime objects.
            3. Cast measurement value to float.
            4. Compute time offset in seconds from ICU admission.
            5. Filter for events within ICU stay window.
            6. Remove duplicates and null values.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - variable: Measurement variable name.
                - value: Measurement value (float).
        """
        return (
            data.select("patientid", "datetime", "variableid", "value")
            # Rename columns for consistency
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str, "datetime": str})
            .join(admissiontime, on=self.icu_stay_id_col)
            .join(length_of_stay, on=self.icu_stay_id_col)
            .join(self._get_observation_variables(), on="variableid")
            .with_columns(
                pl.col("admissiontime").str.to_datetime(
                    "%Y-%m-%d %H:%M:%S%.9f"
                ),
                pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                pl.col("value").cast(float),
            )
            .with_columns(
                (pl.col("datetime") - pl.col("admissiontime"))
                .dt.total_seconds()
                .alias(self.timeseries_time_col)
            )
            .drop("admissiontime", "datetime")
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("value").is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col("variable").is_not_null(), pl.col("variable") != "")
        )

    # endregion

    # region ts labs
    def _extract_timeseries_labs_helper(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Map laboratory timeseries data to LOINC concepts.

        Steps:
            1. Extract unique laboratory test names from data.
            2. Derive LOINC components (component, system, method, time aspect, code).
            3. Join LOINC details back to original data.
            4. Filter for relevant lab components and systems.
            5. Create struct column containing LOINC details and lab result.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - variable: Laboratory test name.
                - labstruct: Struct with value, system, method, time, LOINC code.
        """

        LOINC_data = data.select("variable").unique()
        labnames = LOINC_data.collect().to_series().to_list()

        LOINC_data = (
            data.select("variable").unique()
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_system"),
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_method"),
                pl.col("variable").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("variable")
                .replace_strict(
                    self.omop.get_concept_codes_from_names(labnames),
                    default=None,
                )
                .alias("LOINC_code"),
            )
        )

        return (
            data.join(LOINC_data, on="variable")
            # Filter for lab names of interest
            .filter(
                pl.col("LOINC_component").is_in(
                    self.relevant_lab_LOINC_components
                )
            )
            # Filter for systems of interest
            .filter(
                pl.col("LOINC_system").is_in(
                    pl.col("LOINC_component").replace_strict(
                        self.relevant_lab_LOINC_systems,
                        return_dtype=pl.List(str),
                        default=None,
                    )
                )
            )
            # MAKE STRUCT
            .with_columns(pl.col("LOINC_component").alias("variable"))
            .with_columns(
                pl.struct(
                    value=pl.col("value"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "variable",
                "labstruct",
            )
        )

    # region pharma
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract and process medication administration events.

        Steps:
            1. Load medication and route mappings.
            2. Extract pharmaceutical administration records from {pharma_path}.
            3. Map medication IDs to standardized names and ingredients.
            4. Compute infusion rates and durations from log entries.
            5. Extract fluid administration (saline and colloids) from observations.
            6. Combine pharmaceutical and fluid administration data.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_name_col}: Medication name.
                - {drug_ingredient_col}: Active drug ingredient.
                - {drug_amount_col}: Drug amount.
                - {drug_amount_unit_col}: Amount unit.
                - {drug_rate_col}: Administration rate.
                - {drug_rate_unit_col}: Rate unit.
                - {drug_start_col}: Relative start time (seconds).
                - {drug_end_col}: Relative end time (seconds).
                - Additional administrative columns.
        """
        print("HiRID   - Extracting medications...")

        # Extract medication mappings by building a chain of references
        # 1. Get drug name references from our mapping files
        drug_references = self._extract_drug_references(return_ids=True)
        concept_ids = drug_references.values()

        # 2. Retrieve active ingredients for these concept IDs
        ingredients = self.omop.get_ingredient(concept_ids, return_dict=False)

        # 3. Create a mapping from drug names to their active ingredients
        # Convert drug_references dictionary to DataFrame
        drug_references_df = pl.from_dict(
            {
                "pharmaid": list(drug_references.keys()),
                "drug_concept_id": list(drug_references.values()),
            }
        )

        # Join drug references with ingredients to get all drug-ingredient mappings
        # This preserves one-to-many relationships (one drug to multiple ingredients)
        pharmaid_to_ingredient = (
            drug_references_df.join(
                ingredients, on="drug_concept_id", how="inner"
            )
            .rename({"ingredient_name": self.drug_ingredient_col})
            .select("pharmaid", self.drug_ingredient_col)
            .lazy()
        )

        # Load additional mappings
        hirid_drug_class_mapping = self.load_mapping(
            self.drug_administration_route_mapping_path
        )
        hirid_drug_administration_route_mapping = self.load_mapping(
            self.drug_administration_route_mapping_path
        )

        admissiontime = (
            self._extract_admissions()
            .select(self.icu_stay_id_col, "admissiontime")
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        ########################################################################
        # OBSERVATION
        # -> saline and colloid infusions are tracked as observations
        ########################################################################
        # Create an empty DataFrame to store the observation data
        observation = pl.LazyFrame()

        # Filter for the relevant observation variables
        # Observation   30005075    Infusion of saline solution	cummulative
        # Observation   30005080    Intravenous fluid colloid administration
        # -> both are cumulative variables and reset at midnight
        # to calculate the infusion rate, we need to calculate the difference
        # between the current and the previous value
        fluid_ids = [
            30005075,  # Infusion of saline solution
            30005080,  # Intravenous fluid colloid administration
        ]

        # Since each case has it's data in only one file, iterating over the
        # files allows for a more efficient processing of the data.
        for file in Path(self.timeseries_path).glob("*.parquet"):
            print(f"Processing file {os.path.basename(file)}...", end="\r")
            data = (
                pl.scan_parquet(file)
                .filter(pl.col("variableid").is_in(fluid_ids))
                # Select the relevant columns
                .select("patientid", "datetime", "variableid", "value")
                # Rename the columns for consistency
                .rename({"patientid": self.icu_stay_id_col})
                .with_columns(
                    pl.col("variableid")
                    .replace_strict(
                        {
                            30005075: "normal saline (0.9%)",
                            30005080: "colloid",
                        },
                        default=None,
                        return_dtype=pl.String,
                    )
                    .alias(self.fluid_group_col)
                )
                .cast({self.icu_stay_id_col: str, "datetime": str})
                # Join the data with the admission times
                .join(admissiontime, on=self.icu_stay_id_col)
                .join(length_of_stay, on=self.icu_stay_id_col)
                .with_columns(
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                    pl.col("value").cast(float),
                    pl.lit("intravenous").alias(self.drug_admin_route_col),
                    pl.lit(True).alias(self.drug_continuous_col),
                )
                # Calculate the difference between the current and the previous
                # value, store each timestamp in a separate column
                .with_columns(
                    pl.col("datetime")
                    .shift(1)
                    .over(
                        self.icu_stay_id_col,
                        self.fluid_group_col,
                        order_by="datetime",
                    )
                    .alias("prev_datetime"),
                )
                .with_columns(
                    # Check if dates are different (midnight crossed)
                    pl.when(
                        pl.col("datetime").dt.date()
                        != pl.col("prev_datetime").dt.date()
                    )
                    # If midnight crossed, use current value (assuming reset to 0)
                    .then(pl.col("value"))
                    # Otherwise calculate difference from previous value
                    .otherwise(
                        pl.col("value").sub(
                            pl.col("value")
                            .shift(1)
                            .over(
                                self.icu_stay_id_col,
                                self.fluid_group_col,
                                order_by="datetime",
                            )
                        )
                    ).alias(self.fluid_amount_col),
                )
                # Calculate the rate
                .with_columns(
                    (
                        pl.col(self.fluid_amount_col)
                        / pl.col("datetime")
                        .sub(pl.col("prev_datetime"))
                        .dt.total_seconds()
                    )
                    .round_sig_figs(2)
                    .alias(self.fluid_rate_col)
                )
                .with_columns(
                    (pl.col("prev_datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias(self.drug_start_col),
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias(self.drug_end_col),
                    # Add a column to indicate the administration type
                    pl.lit("given").alias(self.drug_admin_type_col),
                )
                .drop("admissiontime", "datetime")
                # Remove duplicate rows
                .unique()
                # Remove rows with empty values
                .filter(pl.col(self.fluid_amount_col) > 0)
                .select(
                    self.icu_stay_id_col,
                    self.drug_admin_route_col,
                    self.drug_admin_type_col,
                    self.drug_continuous_col,
                    self.fluid_group_col,
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                    self.drug_start_col,
                    self.drug_end_col,
                )
            )

            # Append the data to the DataFrame
            observation = pl.concat(
                [observation, data.lazy()], how="diagonal_relaxed"
            )

        ########################################################################
        # PHARMA
        ########################################################################
        # Create an empty DataFrame to store the pharma data
        pharma = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the
        # files allows for a more efficient processing of the data.
        for file in Path(self.pharma_path).glob("*.parquet"):
            print(f"Processing file {os.path.basename(file)}...", end="\r")
            data = (
                pl.scan_parquet(file)
                .select(
                    "patientid",
                    "pharmaid",
                    "givenat",
                    "givendose",
                    "doseunit",
                    "route",
                    "infusionid",
                    "subtypeid",
                    "recordstatus",
                    "fluidamount_calc",
                )
                # Rename columns for consistency
                .rename(
                    {
                        "patientid": self.icu_stay_id_col,
                        "givendose": self.drug_amount_col,
                        "doseunit": self.drug_amount_unit_col,
                        "route": self.drug_admin_route_col,
                        "infusionid": self.drug_mixture_id_col,
                        "subtypeid": self.drug_class_col,
                        "recordstatus": self.drug_admin_type_col,
                        "fluidamount_calc": self.fluid_amount_col,
                    }
                )
                # Cast the datetime to string to avoid the following error:
                # polars.exceptions.SchemaError: invalid series dtype: expected `String`, got `datetime[ns]`
                .cast({self.icu_stay_id_col: str, "givenat": str})
                .join(admissiontime, on=self.icu_stay_id_col)
                .join(length_of_stay, on=self.icu_stay_id_col)
                .with_columns(
                    # Add a column to indicate the administration type
                    # 2 = invalidated, 32 = notified, not administered
                    pl.when((pl.col(self.drug_admin_type_col) & 2) != 0)
                    .then(pl.lit("cancelled"))
                    .when((pl.col(self.drug_admin_type_col) & 32) != 0)
                    .then(pl.lit("ordered"))
                    .otherwise(pl.lit("given"))
                    .alias(self.drug_admin_type_col),
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("givenat").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                    pl.col(self.drug_amount_col).cast(float),
                )
                # Replace the pharmaid with the corresponding medication name
                .join(self._get_pharma_variables(), on="pharmaid", how="left")
                .join(pharmaid_to_ingredient, on="pharmaid", how="left")
                .with_columns(
                    (pl.col("givenat") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias(self.drug_end_col),
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
                .drop(self.icu_length_of_stay_col)
            )

            # Append the data to the DataFrame
            pharma = pl.concat([pharma, data.lazy()], how="diagonal_relaxed")

        # Get infusion duration where possible, by checking whether the drugname reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        pharma_offsets = (
            pharma.select(
                self.icu_stay_id_col,
                self.drug_end_col,
                self.drug_mixture_id_col,
            )
            .unique()
            .sort(
                self.icu_stay_id_col,
                self.drug_mixture_id_col,
                self.drug_end_col,
            )
            .with_columns(
                pl.col(self.drug_end_col)
                .shift(1)
                .over(self.icu_stay_id_col, self.drug_mixture_id_col)
                .alias("prev_drug_end"),
                pl.col(self.drug_end_col)
                .shift(-1)
                .over(self.icu_stay_id_col, self.drug_mixture_id_col)
                .alias("next_drug_end"),
            )
        )

        pharma = (
            pharma.join(
                pharma_offsets,
                on=[
                    self.icu_stay_id_col,
                    self.drug_end_col,
                    self.drug_mixture_id_col,
                ],
                how="left",
            )
            # Sort by patient ID, drug name and drug start time
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_mixture_id_col,
                self.drug_end_col,
                "prev_drug_end",  # sometimes, there is the same drug given twice at the same time
            )
            # NOTE: Convert drug_amount to drug_rates, fluid_amount to fluid_rates
            .with_columns(
                (
                    pl.col(self.drug_amount_col)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                    * 3600
                )
                .round_sig_figs(2)
                .alias(self.drug_rate_col),
                pl.col(self.drug_amount_unit_col).str.replace("µ", "mc"),
                (pl.col(self.drug_amount_unit_col) + pl.lit("/hr"))
                .str.replace("µ", "mc")
                .alias(self.drug_rate_unit_col),
                (
                    pl.col(self.fluid_amount_col)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                    * 3600
                )
                .round_sig_figs(2)
                .alias(self.fluid_rate_col),
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
                self.drug_admin_type_col,
                maintain_order=True,
            )
            .last()
            # 6. Remove the helper columns
            .drop(
                "prev_drug_end",
                "next_drug_end",
                "drug_status_prev",
                "drug_status_next",
            )
        )

        return pl.concat([pharma, observation], how="diagonal_relaxed")

    # endregion

    # region helpers
    def _get_observation_variables(self) -> pl.DataFrame:
        """
        Retrieve and filter observation variables from reference mapping.

        Steps:
            1. Load complete variable reference.
            2. Filter for "Observation" source table.
            3. Map variable IDs to standardized names using internal dictionaries.

        Returns:
            pl.DataFrame: Columns:
                - variableid: Observation variable identifier.
                - variable: Observation variable name.
        """

        references = (
            pl.read_csv(self.variable_reference_path)
            .filter(pl.col("Source Table") == "Observation")
            .select("ID", "Variable Name")
        )

        extracted_references = dict(
            zip(
                references.get_column("ID").to_list(),
                references.get_column("Variable Name").to_list(),
            )
        )

        extracted_references.update(
            {
                # Fix bad mappings (wrong units)
                24000560: "Bilirubin.direct [Moles/volume] in Serum or Plasma", # was "Bilirubin.direct [Mass/volume] in Serum or Plasma"
                # "/100 leukocytes" obselete in v20250827
                24000480: "Lymphocytes/Leukocytes in Blood", # was "Lymphocytes [#/volume] in Blood"
                24000550: "Neutrophils/Leukocytes in Blood", # was "Neutrophils/100 leukocytes in Blood"
                24000556: "Segmented neutrophils/Leukocytes in Blood", # was "Segmented neutrophils/100 leukocytes in Blood"
                24000557: "Band form neutrophils/Leukocytes in Blood", # was "Band form neutrophils/100 leukocytes in Blood"
                # Update mappings for better clarity
                20001000: "Oxygen saturation in Central venous blood",  # was "Central venous oxygenation saturation"
                24000737: "Oxygen saturation in Central venous blood",  # was "Central venous oxygenation saturation"
                # Update mappings for duplicate names
                # -> Respiratory rate appears multiple times with different IDs
                300: "Respiratory rate",  # Atemfrequenz
                310: "Respiratory rate (spontaneous)",  # RRsp(m)
                5685: None,  # RR Caresc
            } # fmt: skip
        )

        return (
            pl.read_csv(self.variable_reference_path)
            .select("ID")
            .with_columns(
                pl.col("ID")
                .replace_strict(extracted_references, default=None)
                # Replace the variable names with the reprodICU mapping
                .replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                        **self.timeseries_extracorporeal_mapping,
                    }
                )
                .alias("Variable Name")
            )
            .drop_nulls()
            .rename({"ID": "variableid", "Variable Name": "variable"})
            .lazy()
        )

    def _get_pharma_variables(self) -> pl.LazyFrame:
        """
        Retrieve pharmaceutical variable mappings.

        Returns:
            pl.LazyFrame: Columns:
                - pharmaid: Pharmaceutical identifier.
                - {drug_name_col}: Medication name.
                - {drug_name_OMOP_col}: OMOP concept name.
        """
        return (
            pl.read_csv(self.MEDICATION_MAPPING_PATH + "HiRID.usagi.csv")
            .filter(pl.col("conceptName") != "Unmapped")
            .select("sourceCode", "sourceName", "conceptName")
            .drop_nulls("sourceCode")
            .unique()
            .rename(
                {
                    "sourceCode": "pharmaid",
                    "sourceName": self.drug_name_col,
                    "conceptName": self.drug_name_OMOP_col,
                }
            )
            .lazy()
        )

    # Extract the information from the HiRID.usagi.csv files
    def _extract_drug_references(self, return_ids: bool = False) -> dict:
        """
        Extract drug concept reference mappings from USAGI file.

        Returns:
            dict: Mapping from source drug code to concept name (or ID if return_ids=True).
        """

        value_col = "conceptName" if not return_ids else "conceptId"
        references = (
            pl.read_csv(self.MEDICATION_MAPPING_PATH + "HiRID.usagi.csv")
            .filter(pl.col("conceptName") != "Unmapped")
            .select("sourceCode", value_col)
            .drop_nulls("sourceCode")
            .unique()
        )

        return dict(
            zip(
                references["sourceCode"].to_numpy(),
                references[value_col].to_numpy(),
            )
        )
