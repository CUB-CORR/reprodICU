import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RENAL_REPLACEMENT_THERAPY_DURATION_SICdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - SICdb")

        ADMISSION_TIMES = pl.scan_csv(self.sicdb_paths.cases_path).select(
            "CaseID", "ICUOffset"
        )

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.scan_parquet(self.sicdb_paths.data_float_m_path)
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
            # drop rows that are one minute apart from the previous row and the next row
            .sort("CaseID", "RRT Start")
            .filter(
                (
                    (
                        pl.col("RRT Start").shift(-1).over("CaseID")
                        - pl.col("RRT Start")
                    )
                    < (60 * 60 * 2)
                ).not_()
                | (
                    (
                        pl.col("RRT Start")
                        - pl.col("RRT Start").shift(1).over("CaseID")
                    )
                    < (60 * 60 * 2)
                ).not_()
                | pl.col("RRT Start").shift(-1).over("CaseID").is_null()
                | pl.col("RRT Start").shift(1).over("CaseID").is_null()
            )
            .with_columns(
                pl.when(pl.col("CaseID") == pl.col("CaseID").shift(-1))
                .then(pl.col("RRT Start").shift(-1))
                .otherwise(None)
                .alias("RRT End")
            )
            .rename(
                {
                    "RRT Start": "Renal Replacement Therapy Start Relative to Admission (seconds)",
                    "RRT End": "Renal Replacement Therapy End Relative to Admission (seconds)",
                }
            )
            .collect(streaming=True)
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
