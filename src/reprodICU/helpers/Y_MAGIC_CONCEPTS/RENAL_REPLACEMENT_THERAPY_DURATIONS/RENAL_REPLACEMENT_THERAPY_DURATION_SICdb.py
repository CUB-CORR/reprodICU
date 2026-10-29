import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RENAL_REPLACEMENT_THERAPY_DURATION_SICdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        """
        Extract renal replacement therapy episodes from SICdb monitoring data.

        Steps:
            1. Extract RRT data from float measurements.
            2. Join with admission times.
            3. Filter for RRT parameter IDs.
            4. Calculate RRT duration.
            5. Compute time relative to admission.

        Returns:
            pl.DataFrame: Contains columns:
                - CaseID: Case identifier.
                - Renal Replacement Therapy Start Relative to Admission (seconds): Start time.
                - Renal Replacement Therapy End Relative to Admission (seconds): End time.
                - Renal Replacement Therapy Type: RRT modality.
                - Renal Replacement Therapy Duration (hours): Episode duration.
        """
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - SICdb")

        ADMISSION_TIMES = pl.scan_csv(self.sicdb_paths.cases_path).select(
            "CaseID", "ICUOffset"
        )

        # Hours without rawdata only exist in the hourly table
        cols = ["CaseID", "Offset", "DataID", "Val"]
        minutely = pl.scan_parquet(self.sicdb_paths.data_float_m_path).select(cols)
        hourly = (
            pl.scan_csv(
                self.sicdb_paths.data_float_h_path,
                infer_schema_length=10000,
                schema_overrides={"rawdata": pl.String},
            )
            .filter(pl.col("rawdata").is_null())
            .select(cols)
        )

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat([minutely, hourly], how="vertical_relaxed")
            .join(ADMISSION_TIMES, on="CaseID", how="left")
            # Filter for RRT IDs
            .filter(
                pl.col("DataID").is_in(
                    [
                        723,  # CRRT Bloodflow # -> use only one since all variables are logged at the same time
                        # 730,  # CRRT DialysateFlow
                        # 731,  # CRRT SubstitutePrae
                        # 732,  # CRRT SubstitutePost
                        # 2022,  # CRRT Withdrawal
                        # 3071,  # CRRT CalciumSubstitution
                    ]
                )
            )
            .with_columns(
                (pl.col("Offset") - pl.col("ICUOffset")).alias("RRT Start")
            )
            # a run is a series of rows less than 2 h apart; the therapy lasts from its first
            # to its last row (a single row has no duration), the gap to the next run is not RRT
            .sort("CaseID", "RRT Start")
            .with_columns(
                (
                    pl.col("RRT Start") - pl.col("RRT Start").shift(1).over("CaseID")
                    >= 60 * 60 * 2
                )
                .fill_null(True)
                .cum_sum()
                .over("CaseID")
                .alias("RRT Run")
            )
            .group_by("CaseID", "RRT Run")
            .agg(
                pl.col("RRT Start").min(),
                pl.col("RRT Start").max().alias("RRT End"),
            )
            .drop("RRT Run")
            .rename(
                {
                    "RRT Start": "Renal Replacement Therapy Start Relative to Admission (seconds)",
                    "RRT End": "Renal Replacement Therapy End Relative to Admission (seconds)",
                } # fmt: skip
            )
            .collect()
        )

        return (
            RENAL_REPLACEMENT_THERAPY_DURATION.unique()
            .pipe(self._add_global_id_stay_id, "sicdb-", "CaseID")
            .lazy()
        )

    # region helpers
    def _add_global_id_stay_id(
        self, data, source_dataset, stay_id_col
    ) -> pl.LazyFrame:
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str([pl.lit(source_dataset), pl.col(stay_id_col)]).alias(
                self.column_names["global_icu_stay_id_col"]
            )
        ).drop(stay_id_col)

    # endregion
