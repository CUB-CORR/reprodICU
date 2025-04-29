# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Received ANY Antibiotics" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
import os

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RECEIVED_ANY_ANTIBIOTICS(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RECEIVED_ANY_ANTIBIOTICS(self) -> pl.DataFrame:
        """
        Returns the magic concept RECEIVED_ANY_ANTIBIOTICS

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - ICU stay ID
        - RECEIVED_ANY_ANTIBIOTICS (bool)

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        # region eICU
        # print("MAGIC_CONCEPTS: Received ANY Antibiotics - eICU")
        eicu_infusiondrug = (
            pl.scan_csv(self.eicu_paths.infusionDrug_path)
            .select("patientunitstayid", "drugname")
            # Filter for antibiotics
            .filter(
                pl.col("drugname")
                .str.to_lowercase()
                .str.contains(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "eicu"
                    ][0]["regex"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "eicu-",
                "patientunitstayid",
                "drugname",
            )
        )
        eicu_medication = (
            pl.scan_csv(self.eicu_paths.medication_path)
            .select("patientunitstayid", "drugname")
            # Filter for antibiotics
            .filter(
                pl.col("drugname")
                .str.to_lowercase()
                .str.contains(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "eicu"
                    ][1]["regex"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "eicu-",
                "patientunitstayid",
                "drugname",
            )
        )

        eicu_RECEIVED_ANY_ANTIBIOTICS = eicu_infusiondrug.sort(
            by=self.column_names["global_icu_stay_id_col"]
        ).merge_sorted(
            eicu_medication.sort(
                by=self.column_names["global_icu_stay_id_col"]
            ),
            key=self.column_names["global_icu_stay_id_col"],
        )

        # endregion

        # region HiRID
        # print("MAGIC_CONCEPTS: Received ANY Antibiotics - HiRID")
        hirid_RECEIVED_ANY_ANTIBIOTICS = pl.LazyFrame()

        for file in os.listdir(self.hirid_paths.pharma_path):
            hirid_pharma = (
                pl.scan_parquet(self.hirid_paths.pharma_path + file)
                .select("patientid", "pharmaid")
                .filter(
                    pl.col("pharmaid").is_in(
                        self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                            "hirid"
                        ][0]["ids"]
                    )
                )
                .pipe(
                    self._received_any_antibiotics_bool,
                    "hirid-",
                    "patientid",
                    "pharmaid",
                )
            )

            hirid_RECEIVED_ANY_ANTIBIOTICS = pl.concat(
                [hirid_RECEIVED_ANY_ANTIBIOTICS, hirid_pharma],
                how="diagonal_relaxed",
            )

        # endregion

        # region MIMIC-III
        # print("MAGIC_CONCEPTS: Received ANY Antibiotics - MIMIC-III")
        mimic3_prescriptions = (
            pl.scan_csv(self.mimic3_paths.prescriptions_path)
            .select("ICUSTAY_ID", "DRUG")
            # Filter for antibiotics
            .filter(
                pl.col("DRUG")
                .str.to_lowercase()
                .str.contains(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "mimic"
                    ][0]["regex"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "mimic3-",
                "ICUSTAY_ID",
                "DRUG",
            )
        )

        mimic3_inputevents_mv = (
            pl.scan_csv(self.mimic3_paths.inputevents_mv_path)
            .select("ICUSTAY_ID", "ITEMID")
            # Filter for antibiotics
            .filter(
                pl.col("ITEMID").is_in(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "mimic"
                    ][1]["ids"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "mimic3-",
                "ICUSTAY_ID",
                "ITEMID",
            )
        )

        mimic3_RECEIVED_ANY_ANTIBIOTICS = mimic3_prescriptions.sort(
            by=self.column_names["global_icu_stay_id_col"]
        ).merge_sorted(
            mimic3_inputevents_mv.sort(
                by=self.column_names["global_icu_stay_id_col"]
            ),
            key=self.column_names["global_icu_stay_id_col"],
        )

        # endregion

        # region MIMIC-IV
        # print("MAGIC_CONCEPTS: Received ANY Antibiotics - MIMIC-IV")
        # NOTE: MIMIC-IV prescriptions do not include ICU stay IDs
        # mimic4_prescriptions = (
        #     pl.scan_csv(self.mimic4_paths.prescriptions_path)
        #     .select("icustay_id", "drug")
        #     # Filter for antibiotics
        #     .filter(
        #         pl.col("drug").str.to_lowercase().str.contains(
        #             self.ricu_mappings.ricu_concept_dict["abx"]["sources"]["miiv"][0]["regex"],
        #
        #         )
        #     )
        #     .pipe(self._received_any_antibiotics_bool, "mimic4-", "icustay_id", "drug")
        # )

        mimic4_inputevents = (
            pl.scan_csv(self.mimic4_paths.inputevents_path)
            .select("stay_id", "itemid")
            # Filter for antibiotics
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "miiv"
                    ][1]["ids"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "mimic4-",
                "stay_id",
                "itemid",
            )
        )

        mimic4_RECEIVED_ANY_ANTIBIOTICS = mimic4_inputevents

        # endregion

        # region SICdb
        # print("MAGIC_CONCEPTS: Received ANY Antibiotics - SICdb")
        sicdb_RECEIVED_ANY_ANTIBIOTICS = (
            pl.scan_csv(self.sicdb_paths.medication_path)
            .select("CaseID", "DrugID")
            # Filter for antibiotics
            .filter(
                pl.col("DrugID").is_in(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "sic"
                    ][0]["ids"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "sicdb-",
                "CaseID",
                "DrugID",
            )
        )

        # endregion

        # region UMCdb
        # print("MAGIC_CONCEPTS: Received ANY Antibiotics - UMCdb")
        umcdb_RECEIVED_ANY_ANTIBIOTICS = (
            pl.scan_parquet(self.umcdb_paths.drugitems_path)
            .select("admissionid", "itemid")
            # Filter for antibiotics
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["abx"]["sources"][
                        "aumc"
                    ][0]["ids"],
                )
            )
            .pipe(
                self._received_any_antibiotics_bool,
                "umcdb-",
                "admissionid",
                "itemid",
            )
        )

        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Received ANY Antibiotics")

        RECEIVED_ANY_ANTIBIOTICS = pl.concat(
            [
                eicu_RECEIVED_ANY_ANTIBIOTICS,
                hirid_RECEIVED_ANY_ANTIBIOTICS,
                mimic3_RECEIVED_ANY_ANTIBIOTICS,
                mimic4_RECEIVED_ANY_ANTIBIOTICS,
                sicdb_RECEIVED_ANY_ANTIBIOTICS,
                umcdb_RECEIVED_ANY_ANTIBIOTICS,
            ],
            how="vertical_relaxed",
        )
        # endregion

        return RECEIVED_ANY_ANTIBIOTICS

    # region helper functions
    def _received_any_antibiotics_bool(
        self, data, source_dataset, patient_id_col, drug_col_id
    ):
        return (
            data.group_by(patient_id_col)
            .first()
            .with_columns(
                pl.lit(True).alias("Received ANY Antibiotics"),
                # add global ICU stay ID
                pl.concat_str(
                    [pl.lit(source_dataset), pl.col(patient_id_col)]
                ).alias(self.column_names["global_icu_stay_id_col"]),
            )
            .drop(patient_id_col, drug_col_id)
        )

    # endregion
