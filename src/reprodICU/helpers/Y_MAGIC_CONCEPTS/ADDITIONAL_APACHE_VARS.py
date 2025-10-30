# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the so called MAGIC CONCEPT "Severity Scores" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
import os

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class ADDITIONAL_APACHE_VARS(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def ADDITIONAL_APACHE_VARS(self) -> pl.DataFrame:
        """
        Returns the magic concept ADDITIONAL_APACHE_VARS

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - Global ICU stay ID
        - ADDITIONAL_APACHE_VARS:
            - Acute MI location
            - PTCA done within 24 hours
            - Thrombolytic Therapy received within 24 hours
            - Number of grafts performed
            - Internal mammary artery graft?
            - Saphenous vein graft?
            - Pre-op MI during current hospitalization
            - Pre-op cardiac catheterization during this hospitalization
            - Pre-op ejection fraction (%)

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        # region eICU
        print("MAGIC_CONCEPTS: Severity Scores - eICU")
        eicu_ADDITIONAL_APACHE_VARS = (
            pl.scan_csv(self.eicu_paths.apachePredVar_path)
            .select(
                "patientunitstayid",
                "ima",
                "thrombolytics",
                "graftcount",
            )
            .filter(pl.col("apacheversion") == "IV")
            .drop("apacheversion")
            .rename(
                {"acutephysiologyscore": "SAPS III", "apachescore": "APACHE IV"}
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

        mimic3_ADDITIONAL_APACHE_VARS = (
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
            .collect()
            .pivot(
                on="ITEMID",
                index=["ICUSTAY_ID", "Time Relative to Admission (seconds)"],
                values="VALUENUM",
                aggregate_function=pl.first(),
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

        mimic4_ADDITIONAL_APACHE_VARS = (
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
            .collect()
            .pivot(
                on="itemid",
                index=["stay_id", "Time Relative to Admission (seconds)"],
                values="valuenum",
                aggregate_function=pl.first(),
            )
            .lazy()
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
        )

        # endregion

        # region SICdb
        print("MAGIC_CONCEPTS: Severity Scores - SICdb")
        sicdb_ADDITIONAL_APACHE_VARS = (
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

        umcdb_ADDITIONAL_APACHE_VARS = (
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
            .collect()
            .pivot(
                on="itemid",
                index=["admissionid", "Time Relative to Admission (seconds)"],
                values="value",
                aggregate_function=pl.first(),
            )
            .lazy()
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
        )

        # endregion

        eicu_ADDITIONAL_APACHE_VARS.sink_parquet(
            "eicu_ADDITIONAL_APACHE_VARS.parquet"
        )
        mimic3_ADDITIONAL_APACHE_VARS.sink_parquet(
            "mimic3_ADDITIONAL_APACHE_VARS.parquet"
        )
        mimic4_ADDITIONAL_APACHE_VARS.sink_parquet(
            "mimic4_ADDITIONAL_APACHE_VARS.parquet"
        )
        sicdb_ADDITIONAL_APACHE_VARS.sink_parquet(
            "sicdb_ADDITIONAL_APACHE_VARS.parquet"
        )
        umcdb_ADDITIONAL_APACHE_VARS.sink_parquet(
            "umcdb_ADDITIONAL_APACHE_VARS.parquet"
        )

        # region ALL
        print("MAGIC_CONCEPTS: Severity Scoress")

        ADDITIONAL_APACHE_VARS = pl.concat(
            [
                eicu_ADDITIONAL_APACHE_VARS,
                # hirid_ADDITIONAL_APACHE_VARS,
                mimic3_ADDITIONAL_APACHE_VARS,
                mimic4_ADDITIONAL_APACHE_VARS,
                sicdb_ADDITIONAL_APACHE_VARS,
                umcdb_ADDITIONAL_APACHE_VARS,
            ],
            how="diagonal_relaxed",
        )
        # endregion

        return ADDITIONAL_APACHE_VARS

    # region helpers
    def _add_global_id_stay_id(self, data, source_dataset, stay_id_col):
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str(
                [pl.lit(source_dataset), pl.col(stay_id_col).cast(str)]
            ).alias(self.column_names["global_icu_stay_id_col"])
        ).drop(stay_id_col)

    # endregion
