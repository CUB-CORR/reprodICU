# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl
from ..helper import GlobalVars


class PatientInformationCleaner(GlobalVars):
    def __init__(self, paths) -> None:
        super().__init__(paths)
        self.data_availability_cols = [
            "Table: Diagnoses",
            "Table: Medications",
            "Table: Procedures",
            "Table: Timeseries (Laboratory results)",
            "Table: Timeseries (Vitals)",
            "Table: Timeseries (Respiratory data)",
            "Table: Timeseries (In/Out data)",
        ]

    def clean_patient_information(self, data) -> pl.LazyFrame:
        """
        Round patient anthropometric and duration values to reduce precision.

        Steps:
            1. Round height and weight to nearest integer.
            2. Round stay durations to 4 decimal places (approximately 1-minute resolution).
            3. Round mortality_after to nearest integer.

        Returns:
            pl.LazyFrame: Cleaned patient information with same columns.
        """

        return data.with_columns(
            # Round the height and weight to the nearest integer
            pl.col(self.height_col).round(decimals=0).cast(int),
            pl.col(self.weight_col).round(decimals=0).cast(int),
            # Round the stay durations to the nearest 4 significant digits
            # 4 significant digits are chosen to keep the data at about minute resolution
            pl.col(self.pre_icu_length_of_stay_col).round(decimals=4),
            pl.col(self.icu_length_of_stay_col).round(decimals=4),
            pl.col(self.hospital_length_of_stay_col).round(decimals=4),
            # Round the mortality after X days to the nearest integer
            pl.col(self.mortality_after_col).round(decimals=0).cast(int),
        )

    def add_primary_diagnoses(
        self, data: pl.LazyFrame, diagnoses: str
    ) -> pl.LazyFrame:
        """
        Add primary diagnoses to patient information.

        Steps:
            1. Load primary diagnoses (priority=1) from diagnoses table.
            2. Split by ICU vs hospital scope using global IDs.
            3. Select first ICD-10 code per ICU/hospital stay.
            4. Map ICD-10 code to ICD subchapter.
            5. Join enriched diagnoses back to patient data.

        Returns:
            pl.LazyFrame: Patient information with added diagnosis columns.
        """

        primary_diagnoses = pl.scan_parquet(diagnoses).filter(
            pl.col(self.diagnosis_priority_col) == 1
        )
        primary_diagnoses_icu = (
            primary_diagnoses.filter(
                pl.col(self.global_icu_stay_id_col).is_not_null()
            )
            .unique()
            .group_by(self.global_icu_stay_id_col)
            .agg(
                pl.col(self.diagnosis_icd10_code_col)
                .sort_by(self.diagnosis_start_col)
                .first()
            )
        )
        primary_diagnoses_hosp = (
            primary_diagnoses.filter(
                pl.col(self.global_icu_stay_id_col).is_null()
            )
            .unique()
            .group_by(self.global_hospital_stay_id_col)
            .agg(
                pl.col(self.diagnosis_icd10_code_col)
                .sort_by(self.diagnosis_start_col)
                .first()
            )
        )

        return data.join(
            (
                data.select(
                    self.global_icu_stay_id_col,
                    self.global_hospital_stay_id_col,
                )
                .join(
                    primary_diagnoses_icu,
                    on=self.global_icu_stay_id_col,
                    how="left",
                )
                .join(
                    primary_diagnoses_hosp,
                    on=self.global_hospital_stay_id_col,
                    how="left",
                )
                .with_columns(
                    pl.coalesce(
                        pl.col(self.diagnosis_icd10_code_col),
                        pl.col(self.diagnosis_icd10_code_col + "_right"),
                    )
                )
            )
            .select(self.global_icu_stay_id_col, self.diagnosis_icd10_code_col)
            .rename({self.diagnosis_icd10_code_col: "ICD"})
            .pipe(self.ICD_TO_ICDSUBCHAPTER),
            on=self.global_icu_stay_id_col,
            how="left",
        )

    def remove_bad_patient_information(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Remove and flag invalid patient data.

        Steps:
            1. Set post-ICU mortality to null for patients who died in ICU.
            2. Flag stays with ICU duration <15 minutes or missing data as bad.
            3. Mark for exclusion or special handling.

        Returns:
            pl.LazyFrame: Patient information with added flag_bad_data column.
        """

        return data.with_columns(
            # Remove negative / zero values for mortality after ICU discharge
            # when patient died in ICU
            pl.when(pl.col(self.mortality_icu_col))
            .then(None)
            .otherwise(pl.col(self.mortality_after_col))
            .alias(self.mortality_after_col),
            # FLAG patients with negative / almost zero values for ICU stay durations
            pl.when(
                # less than approx. 15 minutes
                pl.col(self.icu_length_of_stay_col).le(0.01)
                # or no data available
                | pl.any_horizontal(self.data_availability_cols)
            )
            .then(True)
            .otherwise(False)
            .alias(self.flag_bad_data_col),
        )

    def add_good_patient_information(self, data) -> pl.LazyFrame:
        """
        Impute missing mortality flags using logical derivation.

        Steps:
            1. Impute missing ICU mortality from post-discharge survival info.
            2. Impute missing hospital mortality from ICU mortality.
            3. Impute remaining missing values using time-based logic.

        Returns:
            pl.LazyFrame: Patient information with imputed mortality columns.
        """

        return (
            data.with_columns(
                # Add missing values for the ICU mortality if the patient survived
                pl.when(
                    pl.col(self.mortality_icu_col).is_null(),
                    pl.col(self.mortality_after_col) > 1,
                )
                .then(False)
                .otherwise(pl.col(self.mortality_icu_col))
                .alias(self.mortality_icu_col),
                # Add missing values for the Hospital mortality if the patient died in the ICU
                pl.when(
                    pl.col(self.mortality_hosp_col).is_null(),
                    pl.col(self.mortality_icu_col).cast(bool),
                )
                .then(pl.col(self.mortality_icu_col))
                .otherwise(pl.col(self.mortality_hosp_col))
                .alias(self.mortality_hosp_col),
            )
            .with_columns(
                # Add missing values for the ICU mortality if the patient survived
                # to the hospital discharge
                pl.when(
                    pl.col(self.mortality_icu_col).is_null(),
                    pl.col(self.mortality_hosp_col).cast(bool).not_(),
                )
                .then(pl.col(self.mortality_hosp_col))
                .otherwise(pl.col(self.mortality_icu_col))
                .alias(self.mortality_icu_col),
            )
            .with_columns(
                # Add missing values for the Hospital and ICU mortality
                # if the patient died long after the ICU discharge
                pl.when(
                    pl.col(self.mortality_hosp_col).is_null(),
                    pl.col(self.mortality_after_col)
                    > (
                        pl.col(self.hospital_length_of_stay_col)
                        - pl.col(self.icu_length_of_stay_col)
                    ),
                )
                .then(False)
                .otherwise(pl.col(self.mortality_hosp_col))
                .alias(self.mortality_hosp_col),
                pl.when(
                    pl.col(self.mortality_icu_col).is_null(),
                    pl.col(self.mortality_after_col) > 1,
                )
                .then(False)
                .otherwise(pl.col(self.mortality_icu_col))
                .alias(self.mortality_icu_col),
            )
        )

    def add_data_availability_information(
        self,
        data: pl.LazyFrame,
        diagnoses: str,
        medications: str,
        procedures: str,
        timeseries_labs: str,
        timeseries_vitals: str,
        timeseries_resp: str,
        timeseries_inout: str,
    ) -> pl.LazyFrame:
        """
        Add data availability counts from other tables.

        Steps:
            1. For each data table (diagnoses, medications, procedures, timeseries):
               - Count records per ICU stay.
               - Add as new availability column.
            2. Join all availability columns to patient data.
            3. Return enriched patient information.

        Returns:
            pl.LazyFrame: Patient information with data availability columns.
        """

        for table, table_name in zip(
            [
                diagnoses,
                medications,
                procedures,
                timeseries_labs,
                timeseries_vitals,
                timeseries_resp,
                timeseries_inout,
            ],
            self.data_availability_cols,
        ):
            data = data.join(
                pl.scan_parquet(table)
                .group_by(self.global_icu_stay_id_col)
                .agg(pl.len())
                .rename({"len": table_name}),
                on=self.global_icu_stay_id_col,
                how="left",
            )

        return data

    def sort_columns(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Reorder columns in standardized sequence.

        Steps:
            1. Select columns in fixed order: IDs first, flags, then demographics, outcomes.
            2. Append data availability columns at end.

        Returns:
            pl.LazyFrame: Reorganized patient information with standard column order.
        """
        return data.select(
            [
                self.global_person_id_col,
                self.global_hospital_stay_id_col,
                self.global_icu_stay_id_col,
                self.icu_stay_seq_num_col,
                self.icu_time_rel_to_first_col,
                self.flag_bad_data_col,
                self.dataset_col,
                self.dataset_version_col,
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                self.age_col,
                self.gender_col,
                self.height_col,
                self.weight_col,
                self.ethnicity_col,
                self.admission_diagnosis_col,
                self.admission_diagnosis_icd_col,
                self.admission_type_col,
                self.admission_urgency_col,
                self.admission_time_col,
                self.admission_year_col,
                self.admission_loc_col,
                self.specialty_col,
                self.care_site_col,
                self.unit_type_col,
                self.pre_icu_length_of_stay_col,
                self.icu_length_of_stay_col,
                self.hospital_length_of_stay_col,
                self.discharge_loc_col,
                self.mortality_hosp_col,
                self.mortality_icu_col,
                self.mortality_after_col,
                self.mortality_after_cutoff_col,
            ]
            + self.data_availability_cols
        )
