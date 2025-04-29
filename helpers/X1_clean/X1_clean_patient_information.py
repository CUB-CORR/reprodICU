# Author: Finn Fassbender
# Last modified: 2024-09-11

# Description: This script imputes the data to remove missing values.
# It is available as a module for piping in the main script.
# It can be called with command line arguments to specify the source datasets to be imputed. ! NOT IMPLEMENTED YET !

import polars as pl
from helpers.helper import GlobalVars


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
        Cleans the height, weight, duration data by rounding the values.
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
        Adds primary diagnoses from the datasets where they need to be extracted
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
        Removes obviously wrong values from the patient information.
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
        Adds information that can easily be derived from the existing data.
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
        Adds information about the availability of the data in the other
        tables of the dataset.
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
        Sort columns
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


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not yet implemented as a command line tool."
    )
