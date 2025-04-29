import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class RENAL_REPLACEMENT_THERAPY_DURATION_UMCdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - UMCdb")

        ADMISSION_TIMES = pl.scan_parquet(
            self.umcdb_paths.admissions_path
        ).select("admissionid", "admittedat")

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.scan_parquet(self.umcdb_paths.processitems_path)
            .join(ADMISSION_TIMES, on="admissionid", how="left")
            # Filter for renal replacement therapy IDs
            .filter(
                pl.col("itemid").is_in(
                    [
                        12465,  # CVVH
                        16363,  # Hemodialyse
                    ]
                )
            )
            .drop("itemid")
            # replace renal replacement therapy concepts
            .with_columns(pl.col("item").replace({"Hemodialyse": "CVVHD"}))
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("start") - pl.col("admittedat"))
                )
                .dt.total_seconds()
                .alias("start"),
                pl.duration(
                    milliseconds=(pl.col("stop") - pl.col("admittedat"))
                )
                .dt.total_seconds()
                .alias("stop"),
            )
            .drop("admittedat")
            # Rename columns
            .rename(
                {
                    "item": "Renal Replacement Therapy Type",
                    "start": "Renal Replacement Therapy Start Relative to Admission (seconds)",
                    "stop": "Renal Replacement Therapy End Relative to Admission (seconds)",
                }
            )
            .collect(streaming=True)
        )

        return (
            RENAL_REPLACEMENT_THERAPY_DURATION.unique()
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
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
