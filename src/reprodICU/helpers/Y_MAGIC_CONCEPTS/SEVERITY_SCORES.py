# Author: Finn Fassbender
# Last modified: 2026-10-29

# Description: This script extracts the so called MAGIC CONCEPT "Severity Scores" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl

from .MAGIC_CONCEPTS import MAGIC_CONCEPTS


class SEVERITY_SCORES(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def SEVERITY_SCORES(self) -> pl.DataFrame:
        """
        Extract severity/acuity scores from multiple ICU databases.

        Steps:
            1. For each available database: extract the scores the database stores (HiRID, MIMIC-III and MIMIC-IV store none).
            2. Map database-specific item IDs to standardized score names.
            3. Calculate time relative to ICU admission.
            4. Concatenate scores from all databases.

        Returns:
            pl.DataFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - APACHE II: Acute Physiology and Chronic Health Evaluation II score.  (UMCdb)
                - APACHE III: Acute Physiology and Chronic Health Evaluation III score.  (UMCdb)
                - APACHE IV: Acute Physiology and Chronic Health Evaluation IV score. (eICU, UMCdb)
                - APS III: Acute Physiology Score III. (eICU)
                - SOFA: Sequential Organ Failure Assessment. (SICdb)
                - SAPS II: Simplified Acute Physiology Score II. (UMCdb)
                - SAPS III: Simplified Acute Physiology Score III. (SICdb)
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

        # region MIMIC-III, MIMIC-IV
        # NOTE: No data available. The MetaVision score items (226743, 226991, 226996,
        # 227428) are charted for under 0.1 % of the stays; mimic-code derives the scores
        # from raw data.
        # endregion

        # region SICdb
        print("MAGIC_CONCEPTS: Severity Scores - SICdb")
        sicdb_cases = pl.scan_csv(self.sicdb_paths.cases_path)

        sicdb_SAPS3 = (
            sicdb_cases.select("CaseID", "saps3")
            .rename({"saps3": "SAPS III"})
            .with_columns(
                pl.lit(0).alias("Time Relative to Admission (seconds)")
            )
        )

        # SOFA: one value per 24 hours, the first at ICU admission
        sicdb_SOFA = (
            pl.scan_csv(self.sicdb_paths.data_float_h_path, infer_schema_length=10000)
            .filter(pl.col("DataID") == 3139)
            .select("CaseID", "Offset", pl.col("Val").alias("SOFA"))
            .join(sicdb_cases.select("CaseID", "ICUOffset"), on="CaseID")
            # Make offset (from the start of the case) relative to admission
            .with_columns(
                (pl.col("Offset") - pl.col("ICUOffset")).alias(
                    "Time Relative to Admission (seconds)"
                )
            )
            .select("CaseID", "Time Relative to Admission (seconds)", "SOFA")
        )

        sicdb_SEVERITY_SCORES = pl.concat(
            [sicdb_SAPS3, sicdb_SOFA], how="diagonal_relaxed"
        ).pipe(self._add_global_id_stay_id, "sicdb-", "CaseID")

        # endregion

        # region UMCdb
        print("MAGIC_CONCEPTS: Severity Scores - UMCdb")
        # NOTE: A_Apache_Score (13081) and MCA_Apache_Score (16624) are entered by staff
        # and not documented. They are read as APACHE II: the AMSTEL mappings categorise
        # them with the APACHE II variables, and the mortality rises with them.
        umcdb_SCORES = {
            19499: "APACHE II",
            13081: "APACHE II",
            16624: "APACHE II",
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
            # Filter for scores; the staff-entered ones hold -1 and values above 71
            .filter(
                pl.col("itemid").is_in(umcdb_SCORE_IDS),
                ~pl.col("itemid").is_in([13081, 16624])
                | pl.col("value").cast(pl.Float64, strict=False).is_between(0, 71),
            )
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
