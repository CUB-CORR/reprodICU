import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_SICdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Ventilation Duration - SICdb")

        ADMISSION_TIMES = pl.scan_csv(self.sicdb_paths.cases_path).select(
            "CaseID", "ICUOffset", "TimeOfStay"
        )

        return (
            pl.scan_csv(self.sicdb_paths.data_range_path)
            .join(ADMISSION_TIMES, on="CaseID", how="left")
            # End must be before Discharge
            .filter(pl.col("OffsetEnd").le(pl.col("TimeOfStay")))
            # Filter for ventilation IDs
            .filter(pl.col("DataID").is_in([720, 3043]))
            .with_columns(
                pl.col("DataID")
                .cast(str)
                .replace(
                    {"720": "invasive ventilation", "3043": "tracheostomy"}
                )
                .alias("Ventilation Type"),
                pl.col("Offset").alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                pl.col("OffsetEnd").alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
            )
            .drop("DataID", "ICUOffset", "Offset", "OffsetEnd", "TimeOfStay")
            .pipe(self._add_global_id_stay_id, "sicdb-", "CaseID")
            .collect(streaming=True)
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
