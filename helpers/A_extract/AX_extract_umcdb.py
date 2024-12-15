# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl
import os.path

from helpers.helper_filepaths import UMCdbPaths
from helpers.helper import GlobalHelpers


class UMCdbExtractor(UMCdbPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.umcdb_source_path
        self.helpers = GlobalHelpers()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

        self.other_lab_values = [
            "Bilirubin.conjugated [Moles/volume]",
            "Billirubin.total [Moles/volume]",
            "Creatinine [Moles/volume]",
            "Cholesterol in HDL [Moles/volume]",
            "Cholesterol in LDL [Moles/volume]",
            "Cholesterol [Moles/volume]",
            "Cortisol [Moles/volume]",
            "Creatine kinase.MB [Mass/volume]",
            "Folate [Moles/volume]",
            "Glucose [Moles/volume]",
            "Hemoglobin [Moles/volume]",
            "MCHC [Moles/volume]",
            "Triglyceride [Moles/volume]",
            "Urate [Moles/volume]",
            "Hematocrit [Pure volume fraction]",
            "MCH [Entitic substance]",
            "Band form neutrophils [#/volume]",
            "Basophils [#/volume]",
            "Eosinophils [#/volume]",
            "Lymphocytes [#/volume]",
            "Monocytes [#/volume]",
            "Neutrophils [#/volume]",
            "Segmented neutrophils [#/volume]",
            "Reticulocytes [#/volume]",
        ]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        return (
            pl.scan_parquet(self.admissions_path)
            .select(
                "patientid",
                "admissionid",
                "admissioncount",
                "agegroup",
                "weightgroup",
                "heightgroup",
                "gender",
                "origin",
                "location",
                "urgency",
                "destination",
                "specialty",
                "admittedat",
                "lengthofstay",
                "dischargedat",
                "dateofdeath",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientid": self.person_id_col,
                    "admissionid": self.icu_stay_id_col,
                    "admissioncount": self.icu_stay_seq_num_col,
                }
            )
            .join(
                self.extract_APACHE_admission(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                # for age, weight and height, assume average of the group
                pl.col("agegroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(
                    lambda s: np.mean([int(i) for i in s if i]),
                    return_dtype=float,
                )
                .cast(int)
                .alias(self.age_col),
                pl.col("weightgroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(
                    lambda s: np.mean([int(i) for i in s if i]),
                    return_dtype=float,
                )
                .cast(int)
                .alias(self.weight_col),
                pl.col("heightgroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(
                    lambda s: np.mean([int(i) for i in s if i]),
                    return_dtype=float,
                )
                .cast(int)
                .alias(self.height_col),
                # Convert categorical mortality to binary
                pl.when(pl.col("destination") != "")
                .then(pl.col("destination") == "Overleden")
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_icu_col),
                pl.when(pl.col("destination") == "Overleden")
                .then(pl.lit(True))
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_hosp_col),
                # NOTE: pre-ICU length of stay is not available in the UMCdb dataset,
                # as there is no known hospital admission / discharge data
                # # Calculate pre-ICU length of stay in days
                # pl.duration(milliseconds=pl.col("admittedat"))
                # .truediv(pl.duration(days=1))
                # .cast(float)
                # .alias(self.pre_icu_length_of_stay_col),
                # Calculate ICU length of stay in days
                pl.duration(hours=pl.col("lengthofstay"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.icu_length_of_stay_col),
                # Calculate mortality after discharge
                pl.duration(
                    milliseconds=(
                        pl.col("dateofdeath") - pl.col("dischargedat")
                    )
                )
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.mortality_after_col),
                # Convert categorical gender to enum
                pl.col("gender")
                .replace_strict(
                    {"Man": "Male", "Vrouw": "Female"}, default="Unknown"
                )
                .cast(self.gender_dtype)
                .alias(self.gender_col),
                # Convert categorical admission location to enum
                pl.col("origin")
                .replace_strict(self.ADMISSION_LOCATIONS_MAP, default="Unknown")
                .cast(self.admission_locations_dtype)
                .alias(self.admission_loc_col),
                # Convert categorical discharge location to enum
                pl.col("destination")
                .replace_strict(self.DISCHARGE_LOCATIONS_MAP, default="Unknown")
                .cast(self.discharge_locations_dtype)
                .alias(self.discharge_loc_col),
                # Convert categorical unit type to enum
                pl.col("location")
                .replace_strict(self.UNIT_TYPES_MAP, default="Unknown")
                .cast(self.unit_types_dtype)
                .alias(self.unit_type_col),
                # Convert categorical specialty to enum
                pl.col("specialty")
                .replace_strict(self.SPECIALTIES_MAP, default="Unknown")
                .cast(self.specialties_dtype)
                .alias(self.specialty_col),
                # Convert categorical admission type to enum
                pl.col("urgency")
                .cast(str)
                .replace_strict(self.ADMISSION_TYPES_MAP, default="Unknown")
                .cast(self.admission_types_dtype)
                .alias(self.admission_type_col),
                # Set hospital stay ID to none
                pl.lit(None).alias(self.hospital_stay_id_col),
                # Set care site to the hospital name
                pl.lit("Amsterdam Universitair Medische Centra").alias(
                    self.care_site_col
                ),
            )
            .drop(
                "agegroup",
                "weightgroup",
                "heightgroup",
                # "gender",
                "origin",
                "destination",
                "specialty",
                "dateofdeath",
                "dischargedat",
                "admittedat",
            )
        )

    # endregion

    # region listitems
    # Extract timeseries information from the listitems.csv file
    def extract_timeseries_listitems(self) -> pl.LazyFrame:
        listitems = (
            pl.scan_parquet(self.listitems_path)
            .select(
                "admissionid",
                # "item",
                "itemid",
                "value",
                "valueid",
                "measuredat",
            )
            .rename({"admissionid": self.icu_stay_id_col})
            .with_columns(
                # Replace item names with standardized names
                pl.col("itemid")
                .replace_strict(self._extract_list_references(), default=None)
                .replace(
                    {
                        **self.relevant_vital_values_mapping,
                        **self.relevant_lab_values_mapping,
                        **self.relevant_intakeoutput_values_mapping,
                        **self.relevant_respiratory_values_mapping,
                    }
                )
                .alias("item"),
            )
            .pipe(self._extract_timeseries_helper)
        )

        gcs = self._compute_gcs(listitems)

        return listitems.drop(["valueid", "itemid"]).join(
            gcs, on=self.index_cols
        )

    # endregion

    # region numeric
    def extract_timeseries_numericitems(self) -> pl.LazyFrame:
        return self._extract_timeseries_numericitems().filter(
            # pl.col("item").is_in(self.all_values + self.other_lab_values),
            ~pl.col("item")
            .str.replace("in HDL", "inHDL")
            .str.replace("in LDL", "inLDL")
            .str.replace(" (in|of) ", " INOF ")
            .str.split_exact(by=" INOF ", n=1)
            .struct.rename_fields(["variable", "_"])
            .struct.field("variable")
            .is_in(self.relevant_lab_values + self.other_lab_values)
        )

    # Separate the lab values from the rest
    def extract_timeseries_labs(self) -> pl.LazyFrame:
        return self._extract_timeseries_labs_helper(
            self._extract_timeseries_numericitems()
        )

    # Extract timeseries information from the numericitems.csv file
    def _extract_timeseries_numericitems(self) -> pl.LazyFrame:
        return (
            pl.scan_parquet(self.numericitems_path)
            .select("admissionid", "itemid", "value", "measuredat")
            .rename({"admissionid": self.icu_stay_id_col})
            .with_columns(
                # Replace item names with standardized names
                pl.col("itemid")
                .replace(self._extract_numeric_references(), default=None)
                .replace(
                    {
                        **self.relevant_vital_values_mapping,
                        **self.relevant_lab_values_mapping,
                        **self.relevant_intakeoutput_values_mapping,
                        **self.relevant_respiratory_values_mapping,
                    }
                )
                .alias("item"),
            )
        ).pipe(self._extract_timeseries_helper)

    # endregion

    # region ts helper
    # filter and rename columns for timeseries data
    def _extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        intimes = (
            pl.scan_parquet(self.admissions_path)
            .select("admissionid", "admittedat", "dischargedat")
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        return (
            data.join(intimes, on=self.icu_stay_id_col)
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col("measuredat") < pl.col("outtime"))
                & (
                    pl.col("measuredat")
                    > (
                        pl.col("intime")
                        - pl.duration(
                            days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).truediv(pl.duration(milliseconds=1))
                    )
                )
            )
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("measuredat") - pl.col("intime"))
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.timeseries_time_col),
            )
            # Filter only relevant timeseries values
            .filter(
                # pl.col("item").is_in(self.all_values + self.other_lab_values),
                pl.col("item")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "_"])
                .struct.field("variable")
                .is_in(self.all_values + self.other_lab_values)
            )
            .drop(["measuredat", "intime", "outtime"])
            # Convert values to numbers, if possible, ignore if not
            .cast({"value": float}, strict=False)
        )

    # endregion

    # region ts labs
    # Extract lab information from the numericitems.csv file
    def _extract_timeseries_labs_helper(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        return (
            data.filter(
                # pl.col("item").is_in(self.all_values + self.other_lab_values),
                pl.col("item")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "_"])
                .struct.field("variable")
                .is_in(self.relevant_lab_values + self.other_lab_values)
            )
            # MAKE STRUCT
            .with_columns(
                pl.col("item")
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
            .filter(
                # remove spO2 (-> vitals)
                pl.col("method").ne_missing("Pulse oximetry"),
                # remove etCO2 (-> respiratory)
                pl.col("source").ne_missing("Exhaled gas --at end expiration"),
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                pl.col("variable")
                .str.replace("inHDL", "in HDL")
                .str.replace("inLDL", "in LDL")
                .alias("item"),
                pl.struct(
                    value="value", source="source", method="method"
                ).alias("value_struct"),
            )
        )

    # region gcs
    # compute Glasgow Coma Scale (GCS) from listitems data
    # Implementation using item IDs as in BlendedICU
    # https://github.com/USM-CHU-FGuyon/BlendedICU/blob/master/amsterdam_preprocessing/AmsterdamPreparator.py#L131
    def _compute_gcs(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if os.path.isfile(self.precalc_path + "UMCdb_A_gcs.parquet"):
            return pl.scan_parquet(self.precalc_path + "UMCdb_A_gcs.parquet")

        data = data.sort(self.index_cols).select(
            self.icu_stay_id_col,
            self.timeseries_time_col,
            "valueid",
            "itemid",
        )

        data_eye = (
            data.filter(
                pl.col("itemid").is_in(
                    [6732, 13077, 14470, 16628, 19635, 19638]
                )
            )
            .with_columns(
                pl.when(pl.col("itemid") == 6732)
                .then(5 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14470, 16628, 19635]))
                .then(pl.col("valueid") - 4)
                .when(pl.col("itemid") == 19638)
                .then(pl.col("valueid") - 8)
                .alias("eyes_score"),
            )
            .drop(["itemid", "valueid"])
        )

        data_motor = (
            data.filter(
                pl.col("itemid").is_in(
                    [6734, 13072, 14476, 16634, 19636, 19639]
                )
            )
            .with_columns(
                "valueid",
                pl.when(pl.col("itemid") == 6734)
                .then(5 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14476, 16634, 19636]))
                .then(pl.col("valueid") - 6)
                .when(pl.col("itemid") == 19639)
                .then(pl.col("valueid") - 12)
                .alias("motor_score"),
            )
            .drop(["itemid", "valueid"])
        )

        data_verbal = (
            data.filter(
                pl.col("itemid").is_in(
                    [6735, 13066, 14482, 16640, 19637, 19640]
                )
            )
            .with_columns(
                "valueid",
                pl.when(pl.col("itemid") == 6735)
                .then(6 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14482, 16640, 19637]))
                .then(pl.col("valueid") - 5)
                .when(pl.col("itemid") == 19640)
                .then(pl.col("valueid") - 15)
                .alias("verbal_score"),
            )
            .drop(["itemid", "valueid"])
        )

        data_gcs = (
            data_eye.join(data_motor, on=self.index_cols)
            .join(data_verbal, on=self.index_cols)
            .collect(streaming=True)
        )
        data_gcs = data_gcs.with_columns(
            (
                data_gcs.select(
                    "eyes_score", "motor_score", "verbal_score"
                ).sum_horizontal(ignore_nulls=False)
            ).alias("gcs_score"),
        )

        data_gcs.write_parquet(self.precalc_path + "UMCdb_A_gcs.parquet")

        return data_gcs.lazy()

    # endregion

    # region medication
    # Extract medication information from the drugitems.csv file
    def extract_medications(self) -> pl.LazyFrame:
        print("UMCdb   - Extracting medications...")

        umcdb_medication_mapping = (
            self.helpers.load_many_to_many_to_one_mapping(
                self.mapping_path + "MEDICATIONS.yaml", "amsterdam"
            )
        )
        umcdb_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )
        umcdb_drug_class_mapping = self.helpers.load_mapping(
            self.drug_class_mapping_path
        )

        intimes = (
            pl.scan_parquet(self.admissions_path)
            .select("admissionid", "admittedat", "dischargedat")
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        return (
            pl.scan_parquet(self.drugitems_path)
            .select(
                "admissionid",
                "item",
                "start",
                "stop",
                "ordercategory",
                # "administered",
                # "administeredunit", -> actual doses, to be integrated
                "doserateperkg",
                "dose",
                "doseunit",
                "doserateunit",
                # "rate",
                # "rateunit",
            )
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "item": self.drug_name_col,
                    "start": self.drug_start_col,
                    "stop": self.drug_end_col,
                    # "administered": self.drug_amount_col,
                    # "administeredunit": self.drug_amount_unit_col,
                    # "rate": self.drug_rate_col,
                    # "rateunit": self.drug_rate_unit_col,
                }
            )
            .join(intimes, on=self.icu_stay_id_col)
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col(self.drug_start_col) < pl.col("outtime"))
                & (
                    pl.col(self.drug_end_col)
                    > (
                        pl.col("intime")
                        - pl.duration(
                            days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).truediv(pl.duration(milliseconds=1))
                    )
                )
            )
            .with_columns(
                # Calculate drug start times relative to ICU admission
                pl.duration(
                    milliseconds=(
                        pl.col(self.drug_start_col) - pl.col("intime")
                    )
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.drug_start_col),
                # Calculate drug end times relative to ICU admission
                pl.duration(
                    milliseconds=(pl.col(self.drug_end_col) - pl.col("intime"))
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.drug_end_col),
                # Replace drug names with standardized ingredient names
                pl.col(self.drug_name_col)
                .replace(umcdb_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
                # Replace drug names with OMOP concepts
                pl.col(self.drug_name_col)
                .replace_strict(self._extract_drug_references(), default=None)
                .alias(self.drug_name_OMOP_col),
                # Convert administered unit to enum
                # pl.col(self.drug_amount_unit_col)
                # .replace(self.DRUG_UNIT_MAPPING)
                # .cast(self.drug_unit_dtype),
                # Replace drug rate units
                pl.col("doseunit").replace({"µg": "mcg"}),
                pl.col("doserateunit").replace(
                    {"uur": "hr", "dag": "day", "min": "min"}
                ),
                # Replace drug administration routes
                pl.col("ordercategory")
                .replace(umcdb_drug_administration_route_mapping, default=None)
                .alias(self.drug_admin_route_col),
                # Replace drug classes
                pl.col("ordercategory")
                .replace(umcdb_drug_class_mapping, default=None)
                .alias(self.drug_class_col),
            )
            # assign to rate or amount column based on availability
            .with_columns(
                # drug amounts
                pl.when(pl.col("doserateunit").is_null())
                .then(pl.col("dose"))
                .otherwise(None)
                .alias(self.drug_amount_col),
                pl.when(pl.col("doserateunit").is_null())
                .then(pl.col("doseunit"))
                .otherwise(None)
                .alias(self.drug_amount_unit_col),
                # drug rates
                pl.when(pl.col("doserateunit").is_not_null())
                .then(pl.col("dose"))
                .otherwise(None)
                .alias(self.drug_rate_col),
                pl.when(pl.col("doserateunit").is_not_null())
                .then(
                    pl.concat_str(
                        pl.col("doseunit"),
                        pl.lit("/"),
                        pl.when(pl.col("doserateperkg") == 1)
                        .then(pl.lit("kg/"))
                        .otherwise(pl.lit("")),
                        pl.col("doserateunit"),
                    )
                )
                .otherwise(None)
                .alias(self.drug_rate_unit_col),
            )
            .cast({self.drug_amount_col: float, self.drug_rate_col: float})
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col(self.drug_start_col).is_not_null())
            # Remove rows with empty lab results
            .filter(
                pl.col(self.drug_name_col).is_not_null()
                & (pl.col(self.drug_name_col) != "")
            )
            .drop("intime", "outtime")
        )

    # endregion

    # region procedures
    # Extract procedure information from the procedures.csv file
    def extract_procedures(self) -> pl.LazyFrame:
        print("UMCdb   - Extracting procedures...")
        intimes = (
            pl.scan_parquet(self.admissions_path)
            .select("patientid", "admissionid", "admittedat", "dischargedat")
            .rename(
                {
                    "patientid": self.person_id_col,
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        procedureorderitems = (
            pl.scan_parquet(self.procedureorderitems_path)
            .select("admissionid", "itemid", "registeredat")
            .rename(
                {"admissionid": self.icu_stay_id_col, "registeredat": "start"}
            )
        )

        processitems = (
            pl.scan_parquet(self.processitems_path)
            .select("admissionid", "itemid", "start", "stop")
            .rename({"admissionid": self.icu_stay_id_col})
        )

        return (
            pl.concat(
                [procedureorderitems, processitems], how="diagonal_relaxed"
            )
            .join(intimes, on=self.icu_stay_id_col, how="left")
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col("start") < pl.col("outtime"))
                & (
                    pl.col("start")
                    > (
                        pl.col("intime")
                        - pl.duration(
                            days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).truediv(pl.duration(milliseconds=1))
                    )
                )
            )
            .with_columns(
                # Calculate procedure start / end times relative to ICU admission
                pl.duration(milliseconds=(pl.col("start") - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.procedure_start_col),
                pl.duration(milliseconds=(pl.col("stop") - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.procedure_end_col),
                # Replace procedure ids with standardized names
                pl.col("itemid")
                .replace_strict(
                    self._extract_procedure_references(), default=None
                )
                .alias(self.procedure_description_col),
            )
            .drop(["start", "stop", "intime", "outtime"])
        )

    # endregion

    # region APACHE
    # Extract APACHE admission information from the listitems.csv file
    def extract_APACHE_admission(self) -> pl.LazyFrame:
        APACHE_mapping = self.helpers.load_mapping(self.apache_mapping_path)

        NICE = [18669, 18671]
        LEVEL0_ITEMIDS = [
            13110,  # D_Hoofdgroep
            16651,  # DMC_Hoofdgroep, Medium Care
            18588,  # Apache II Hoofdgroep
            16997,  # APACHE IV Groepen
            18669,  # NICE APACHEII diagnosen
            18671,  # NICE APACHEIV diagnosen
        ]
        LEVEL1_ITEMIDS = [
            13111,  # D_Subgroep_Thoraxchirurgie
            16669,  # DMC_Subgroep_Thoraxchirurgie
            13112,  # D_Subgroep_Algemene chirurgie
            16665,  # DMC_Subgroep_Algemene chirurgie
            13113,  # D_Subgroep_Neurochirurgie
            16667,  # DMC_Subgroep_Neurochirurgie
            13114,  # D_Subgroep_Neurologie
            16668,  # DMC_Subgroep_Neurologie
            13115,  # D_Subgroep_Interne geneeskunde
            16666,  # DMC_Subgroep_Interne geneeskunde
        ]
        SURGICAL_ITEMIDS = [
            13116,  # D_Thoraxchirurgie_CABG en Klepchirurgie
            16671,  # DMC_Thoraxchirurgie_CABG en Klepchirurgie
            13117,  # D_Thoraxchirurgie_Cardio anders
            16672,  # DMC_Thoraxchirurgie_Cardio anders
            13118,  # D_Thoraxchirurgie_Aorta chirurgie
            16670,  # DMC_Thoraxchirurgie_Aorta chirurgie
            13119,  # D_Thoraxchirurgie_Pulmonale chirurgie
            16673,  # DMC_Thoraxchirurgie_Pulmonale chirurgie
            13121,  # D_Algemene chirurgie_Buikchirurgie
            16643,  # DMC_Algemene chirurgie_Buikchirurgie
            13123,  # D_Algemene chirurgie_Endocrinologische chirurgie
            16644,  # DMC_Algemene chirurgie_Endocrinologische chirurgi
            13145,  # D_Algemene chirurgie_KNO/Overige
            16645,  # DMC_Algemene chirurgie_KNO/Overige
            13125,  # D_Algemene chirurgie_Orthopedische chirurgie
            16646,  # DMC_Algemene chirurgie_Orthopedische chirurgie
            13122,  # D_Algemene chirurgie_Transplantatie chirurgie
            16647,  # DMC_Algemene chirurgie_Transplantatie chirurgie
            13124,  # D_Algemene chirurgie_Trauma
            16648,  # DMC_Algemene chirurgie_Trauma
            13126,  # D_Algemene chirurgie_Urogenitaal
            16649,  # DMC_Algemene chirurgie_Urogenitaal
            13120,  # D_Algemene chirurgie_Vaatchirurgie
            16650,  # DMC_Algemene chirurgie_Vaatchirurgie
            13128,  # D_Neurochirurgie _Vasculair chirurgisch
            16661,  # DMC_Neurochirurgie _Vasculair chirurgisch
            13129,  # D_Neurochirurgie _Tumor chirurgie
            16660,  # DMC_Neurochirurgie _Tumor chirurgie
            13130,  # D_Neurochirurgie_Overige
            16662,  # DMC_Neurochirurgie_Overige
            18596,  # Apache II Operatief  Gastr-intenstinaal
            18597,  # Apache II Operatief Cardiovasculair
            18598,  # Apache II Operatief Hematologisch
            18599,  # Apache II Operatief Metabolisme
            18600,  # Apache II Operatief Neurologisch
            18601,  # Apache II Operatief Renaal
            18602,  # Apache II Operatief Respiratoir
            17008,  # APACHEIV Post-operative cardiovascular
            17009,  # APACHEIV Post-operative gastro-intestinal
            17010,  # APACHEIV Post-operative genitourinary
            17011,  # APACHEIV Post-operative hematology
            17012,  # APACHEIV Post-operative metabolic
            17013,  # APACHEIV Post-operative musculoskeletal /skin
            17014,  # APACHEIV Post-operative neurologic
            17015,  # APACHEIV Post-operative respiratory
            17016,  # APACHEIV Post-operative transplant
            17017,  # APACHEIV Post-operative trauma
        ]
        LEVEL2_ITEMIDS = SURGICAL_ITEMIDS + [
            13141,  # D_Algemene chirurgie_Algemeen
            16642,  # DMC_Algemene chirurgie_Algemeen
            13133,  # D_Interne Geneeskunde_Cardiovasculair
            16653,  # DMC_Interne Geneeskunde_Cardiovasculair
            13134,  # D_Interne Geneeskunde_Pulmonaal
            16658,  # DMC_Interne Geneeskunde_Pulmonaal
            13135,  # D_Interne Geneeskunde_Abdominaal
            16652,  # DMC_Interne Geneeskunde_Abdominaal
            13136,  # D_Interne Geneeskunde_Infectieziekten
            16655,  # DMC_Interne Geneeskunde_Infectieziekten
            13137,  # D_Interne Geneeskunde_Metabool
            16656,  # DMC_Interne Geneeskunde_Metabool
            13138,  # D_Interne Geneeskunde_Renaal
            16659,  # DMC_Interne Geneeskunde_Renaal
            13139,  # D_Interne Geneeskunde_Hematologisch
            16654,  # DMC_Interne Geneeskunde_Hematologisch
            13140,  # D_Interne Geneeskunde_Overige
            16657,  # DMC_Interne Geneeskunde_Overige
            13131,  # D_Neurologie_Vasculair neurologisch
            16664,  # DMC_Neurologie_Vasculair neurologisch
            13132,  # D_Neurologie_Overige
            16663,  # DMC_Neurologie_Overige
            13127,  # D_KNO/Overige
            18589,  # Apache II Non-Operatief Cardiovasculair
            18590,  # Apache II Non-Operatief Gastro-intestinaal
            18591,  # Apache II Non-Operatief Hematologisch
            18592,  # Apache II Non-Operatief Metabolisme
            18593,  # Apache II Non-Operatief Neurologisch
            18594,  # Apache II Non-Operatief Renaal
            18595,  # Apache II Non-Operatief Respiratoir
            16998,  # APACHE IV Non-operative cardiovascular
            16999,  # APACHE IV Non-operative Gastro-intestinal
            17000,  # APACHE IV Non-operative genitourinary
            17001,  # APACHEIV  Non-operative haematological
            17002,  # APACHEIV  Non-operative metabolic
            17003,  # APACHEIV Non-operative musculo-skeletal
            17004,  # APACHEIV Non-operative neurologic
            17005,  # APACHEIV Non-operative respiratory
            17006,  # APACHEIV Non-operative transplant
            17007,  # APACHEIV Non-operative trauma
            # # Both NICE APACHEII/IV also count towards surgical if valueid in correct range
            18669,  # NICE APACHEII diagnosen
            18671,  # NICE APACHEIV diagnosen
        ]

        listitems = (
            pl.scan_parquet(self.listitems_path)
            .rename({"admissionid": self.icu_stay_id_col})
            .with_columns(
                pl.when(pl.col("itemid") == 18671)  # NICE APACHEIV diagnosen
                .then(6)
                .when(pl.col("itemid") == 18669)  # NICE APACHEII diagnosen
                .then(5)
                .when(pl.col("itemid").is_between(16998, 17017))  # APACHE IV
                .then(4)
                .when(pl.col("itemid").is_between(18589, 18602))  # Apache II
                .then(3)
                .when(pl.col("itemid").is_between(13116, 13145))  # D_Hoofdgroep
                .then(2)
                .when(
                    pl.col("itemid").is_between(16642, 16673)
                )  # DMC_Hoofdgroep
                .then(1)
                .otherwise(None)
                .cast(int, strict=False)
                .alias("typeid"),
            )
        )

        diagnoses = (
            listitems.filter(pl.col("itemid").is_in(LEVEL2_ITEMIDS))
            .rename(
                {
                    "value": "diagnosis",
                    "valueid": "diagnosis_id",
                }
            )
            .sort(self.icu_stay_id_col, "updatedat", descending=True)
            .with_columns(
                pl.when(pl.col("itemid").is_in(NICE))
                .then(
                    pl.col("diagnosis")
                    .str.replace(" -Coronair", " - Coronair")
                    .str.split(" - ")
                    .list.get(0)
                )
                .otherwise(pl.col("diagnosis"))
                .alias("diagnosis"),
                pl.int_range(pl.len())
                .over(self.icu_stay_id_col)
                .alias("rownum"),
                pl.when(pl.col("itemid").is_in(SURGICAL_ITEMIDS))
                .then(True)
                .when(
                    pl.col("itemid") == 18669,
                    pl.col("diagnosis_id").is_between(1, 26),
                )
                .then(True)
                .when(
                    pl.col("itemid") == 18671,
                    pl.col("diagnosis_id").is_between(222, 452),
                )
                .then(True)
                .otherwise(False)
                .alias("surgical"),
            )
            .cast({"diagnosis": str, "diagnosis_id": str, "surgical": bool})
            .group_by(self.icu_stay_id_col, "typeid", "updatedat")
            .agg(
                pl.col("diagnosis"),
                pl.col("diagnosis_id"),
                pl.col("surgical").first(),
            )
            .explode("diagnosis", "diagnosis_id")
            .with_columns(
                pl.col("typeid")
                .cast(str)
                .replace(
                    {
                        "6": "NICE APACHE IV",
                        "5": "NICE APACHE II",
                        "4": "APACHE IV",
                        "3": "APACHE II",
                        "2": "Legacy ICU",
                        "1": "Legacy MCU",
                    }
                )
                .alias("diagnosis_type"),
            )
            .unique()
            .sort(self.icu_stay_id_col, "typeid", "updatedat", descending=True)
            .with_columns(
                pl.int_range(pl.len())
                .over(self.icu_stay_id_col)
                .alias("rownum")
            )
            .drop("typeid")
        )

        return (
            diagnoses.group_by(self.icu_stay_id_col)
            .first()
            .select(self.icu_stay_id_col, "diagnosis")
            .rename({"diagnosis": self.admission_diagnosis_col})
            .with_columns(
                pl.col(self.admission_diagnosis_col).replace(
                    APACHE_mapping, default=None
                )
            )
        )

    # endregion

    # region references
    # Extract the information from the numericitems_XXX.usagi.csv files
    def _extract_numeric_references(self) -> dict:
        references = (
            pl.concat(
                [
                    pl.read_csv(self.numericitems_lab_mapping_path),
                    pl.read_csv(self.numericitems_other_mapping_path),
                    pl.read_csv(self.numericitems_tag_mapping_path),
                    pl.read_csv(self.numericitems_unit_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            .with_columns(
                pl.col("conceptName").replace(
                    {
                        "Invasive Systolic blood pressure": "Invasive systolic arterial pressure",
                        "Invasive Diastolic blood pressure": "Invasive diastolic arterial pressure",
                        "Invasive Mean blood pressure": "Invasive mean arterial pressure",
                        "Systolic blood pressure by Noninvasive": "Non-invasive systolic arterial pressure",
                        "Diastolic blood pressure by Noninvasive": "Non-invasive diastolic arterial pressure",
                        "Mean blood pressure by Noninvasive": "Non-invasive mean arterial pressure",
                        "Heart rate.beat-to-beat by EKG": "Heart rate",
                    }
                )
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceCode", "conceptName")
            .cast({"sourceCode": int}, strict=False)
        )

        references = references.filter(
            # pl.col("conceptName").is_in(self.all_values + self.other_lab_values),
            pl.col("conceptName")
            .str.replace("in HDL", "inHDL")
            .str.replace("in LDL", "inLDL")
            .str.replace(" (in|of) ", " INOF ")
            .str.split_exact(by=" INOF ", n=1)
            .struct.rename_fields(["variable", "_"])
            .struct.field("variable")
            .is_in(self.all_values + self.other_lab_values)
        )

        return dict(
            zip(
                references["sourceCode"].to_numpy(),
                references["conceptName"].to_numpy(),
            )
        )

    # Extract the information from the listitems_XXX.usagi.csv file
    def _extract_list_references(self) -> dict:
        references = (
            pl.concat(
                [
                    pl.read_csv(self.listitems_item_mapping_path),
                    pl.read_csv(self.listitems_value_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceCode", "conceptName").filter(
                # pl.col("conceptName").is_in(self.all_values + self.other_lab_values),
                pl.col("conceptName")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "_"])
                .struct.field("variable")
                .is_in(self.all_values + self.other_lab_values)
            )
        )

        return dict(
            zip(
                references["sourceCode"].to_numpy(),
                references["conceptName"].to_numpy(),
            )
        )

    # Extract the information from the drugitems_XXX.usagi.csv files
    def _extract_drug_references(self) -> dict:
        references = (
            pl.concat(
                [
                    pl.read_csv(self.drug_administration_route_mapping_path),
                    pl.read_csv(self.drugitems_item_mapping_path),
                    pl.read_csv(self.drug_class_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceName", "conceptName")
        )

        return dict(
            zip(
                references["sourceName"].to_numpy(),
                references["conceptName"].to_numpy(),
            )
        )

    # Extract the information from the processitems_item.usagi.csv
    # and procedureorderitems_item.usagi.csv files
    def _extract_procedure_references(self) -> dict:
        references = (
            pl.concat(
                [
                    pl.read_csv(self.procedureorderitems_item_mapping_path),
                    pl.read_csv(self.processitems_item_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceCode", "conceptName")
        )

        return dict(
            zip(
                references["sourceCode"].to_numpy(),
                references["conceptName"].to_numpy(),
            )
        )

    # endregion
