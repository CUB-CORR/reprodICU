# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Severity Scores" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
import os

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class SEVERITY_SCORES(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def SEVERITY_SCORES(self) -> pl.DataFrame:
        """
        Returns the magic concept SEVERITY_SCORES

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - Global ICU stay ID
        - Time Relative to Admission (seconds)
        - SEVERITY_SCORES:
            - APACHE II (MIMIC-III, MIMIC-IV, UMCdb)
            - APACHE III (MIMIC-III, MIMIC-IV, UMCdb)
            - APACHE IV (eICU, UMCdb)
            - APS III (eICU, MIMIC-III, MIMIC-IV)
            - SOFA (MIMIC-III, MIMIC-IV)
            - SAPS II (UMCdb)
            - SAPS III (SICdb)

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        # region eICU
        print("MAGIC_CONCEPTS: Severity Scores - eICU")
        eicu_SEVERITY_SCORES = (
            pl.scan_csv(self.eicu_paths.apachePatientResult_path)
            .select(
                "patientunitstayid",
                "acutephysiologyscore",
                "apachescore",
                "apacheversion",
            )
            .filter(
                pl.col("apacheversion") == "IV",
                pl.col("acutephysiologyscore").ne_missing(-1),
            )
            .drop("apacheversion")
            .rename(
                {"acutephysiologyscore": "APS III", "apachescore": "APACHE IV"}
            )
            .pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")
        )

        # endregion

        # region HiRID
        # NOTE: No data available
        # endregion

        # region MIMIC-III
        print("MAGIC_CONCEPTS: Severity Scores - MIMIC-III")
        mimic_SCORES = {
            226743: "APACHE II",
            226991: "APACHE III",
            226996: "APS III",
            227428: "SOFA",
        }
        mimic_SCORE_IDS = list(mimic_SCORES.keys())

        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic3_SEVERITY_SCORES = (
            pl.scan_csv(self.mimic3_paths.chartevents_path)
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUENUM")
            # Filter for scores
            .filter(pl.col("ITEMID").is_in(mimic_SCORE_IDS))
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ITEMID").replace_strict(mimic_SCORES, default=None),
            )
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            .with_columns(
                (pl.col("CHARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)")
            )
            .collect(streaming=True)
            .pivot(
                on="ITEMID",
                index=["ICUSTAY_ID", "Time Relative to Admission (seconds)"],
                values="VALUENUM",
                aggregate_function="first",
            )
            .lazy()
            # Make datetime relative to admission in seconds
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
        )

        # endregion

        # region MIMIC-IV
        print("MAGIC_CONCEPTS: Severity Scores - MIMIC-IV")
        mimic4_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic4_SEVERITY_SCORES = (
            pl.scan_csv(self.mimic4_paths.chartevents_path)
            .select("stay_id", "charttime", "itemid", "valuenum")
            # Filter for scores
            .filter(pl.col("itemid").is_in(mimic_SCORE_IDS))
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("itemid").replace_strict(mimic_SCORES, default=None),
            )
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
            # Make datetime relative to admission in seconds
            .with_columns(
                (pl.col("charttime") - pl.col("intime"))
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)")
            )
            .collect(streaming=True)
            .pivot(
                on="itemid",
                index=["stay_id", "Time Relative to Admission (seconds)"],
                values="valuenum",
                aggregate_function="first",
            )
            .lazy()
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
        )

        # endregion

        # region SICdb
        print("MAGIC_CONCEPTS: Severity Scores - SICdb")
        sicdb_SEVERITY_SCORES = (
            pl.scan_csv(self.sicdb_paths.cases_path)
            .select("CaseID", "saps3")
            .rename({"saps3": "SAPS III"})
            .with_columns(
                pl.lit(0).alias("Time Relative to Admission (seconds)")
            )
            .pipe(self._add_global_id_stay_id, "sicdb", "CaseID")
        )

        # endregion

        # region UMCdb
        print("MAGIC_CONCEPTS: Severity Scores - UMCdb")
        umcdb_SCORES = {
            19499: "APACHE II",
            19750: "APACHE III",
            19500: "APACHE IV",
            19503: "SAPS II",
        }
        umcdb_SCORE_IDS = list(umcdb_SCORES.keys())

        umcdb_INTIMES = (
            pl.scan_parquet(self.umcdb_paths.admissions_path)
            .select("admissionid", "admittedat", "dischargedat")
            .rename({"admittedat": "intime", "dischargedat": "outtime"})
        )

        umcdb_SEVERITY_SCORES = (
            pl.scan_parquet(self.umcdb_paths.numericitems_path)
            .select("admissionid", "itemid", "value", "measuredat")
            # Filter for scores
            .filter(pl.col("itemid").is_in(umcdb_SCORE_IDS))
            .join(umcdb_INTIMES, on="admissionid", how="left")
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("measuredat") - pl.col("intime"))
                )
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)"),
                pl.col("itemid").replace_strict(umcdb_SCORES, default=None),
            )
            .collect(streaming=True)
            .pivot(
                on="itemid",
                index=["admissionid", "Time Relative to Admission (seconds)"],
                values="value",
                aggregate_function="first",
            )
            .lazy()
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
        )

        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Severity Scores")

        SEVERITY_SCORES = (
            pl.concat(
                [
                    eicu_SEVERITY_SCORES,
                    # hirid_SEVERITY_SCORES,
                    mimic3_SEVERITY_SCORES,
                    mimic4_SEVERITY_SCORES,
                    sicdb_SEVERITY_SCORES,
                    umcdb_SEVERITY_SCORES,
                ],
                how="diagonal_relaxed",
            )
            .select(
                self.column_names["global_icu_stay_id_col"],
                "Time Relative to Admission (seconds)",
                "APACHE II",
                "APACHE III",
                "APACHE IV",
                "APS III",
                "SOFA",
                "SAPS II",
                "SAPS III",
            )
            .cast(
                {
                    k: pl.Float32
                    for k in [
                        "APACHE II",
                        "APACHE III",
                        "APACHE IV",
                        "APS III",
                        "SOFA",
                        "SAPS II",
                        "SAPS III",
                    ]
                }
            )
        )
        # endregion

        return SEVERITY_SCORES

    # region helpers
    def _add_global_id_stay_id(self, data, source_dataset, stay_id_col):
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str(
                [pl.lit(source_dataset), pl.col(stay_id_col).cast(str)]
            ).alias(self.column_names["global_icu_stay_id_col"])
        ).drop(stay_id_col)

    # endregion
