import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_SICdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        """
        Extract ventilation episodes from SICdb data range records.

        Steps:
            1. Extract ventilation data ranges with timestamps.
            2. Join with admission times for time reference.
            3. Keep episodes starting before discharge and truncate their
               end at discharge.
            4. Calculate episode duration.
            5. Compute time relative to admission.

        Returns:
            pl.DataFrame: Contains columns:
                - CaseID: Case identifier.
                - {timeseries_time_col}: Ventilation start time (seconds from admission).
                - Ventilation Type: Classification (invasive ventilation, etc.).
                - Ventilation Duration (hours): Episode duration.
        """
        print("MAGIC_CONCEPTS: Ventilation Duration - SICdb")

        ADMISSION_TIMES = pl.scan_csv(self.sicdb_paths.cases_path).select(
            "CaseID", "ICUOffset", "TimeOfStay"
        )

        return (
            pl.scan_csv(self.sicdb_paths.data_range_path)
            .join(ADMISSION_TIMES, on="CaseID", how="left")
            # optiflow fix: an episode whose end was charted after the closing
            # of the case (TimeOfStay, counted like Offset from the MetaVision
            # admission) was dropped entirely, which removed every intubation
            # of 759 of the 5,300 SICdb cases with an endotracheal tube. The
            # episode is now kept and its end truncated at the closing of the
            # case; episodes starting after it are still dropped.
            .filter(pl.col("Offset").le(pl.col("TimeOfStay")))
            .with_columns(
                pl.min_horizontal("OffsetEnd", "TimeOfStay").alias("OffsetEnd")
            )
            # Filter for ventilation IDs
            .filter(pl.col("DataID").is_in([720, 3041]))
            # Make offsets (from the start of the case) relative to admission
            .with_columns(pl.col("Offset", "OffsetEnd") - pl.col("ICUOffset"))
            .with_columns(
                pl.col("DataID")
                .cast(str)
                .replace(
                    {"720": "invasive ventilation", "3041": "tracheostomy"}
                )
                .alias("Ventilation Type"),
            )
            # Remove duplicates with slighty different offsets
            .group_by("CaseID", "Offset", "Ventilation Type")
            .agg(pl.col("OffsetEnd").max())
            .group_by("CaseID", "Ventilation Type", "OffsetEnd")
            .agg(pl.col("Offset").min())
            # Fix overlapping periods
            .with_columns(
                pl.col("OffsetEnd")
                .shift(1)
                .over(
                    partition_by=["CaseID", "Ventilation Type"],
                    order_by="Offset",
                )
                .alias("PrevOffsetEnd"),
            )
            .with_columns(
                pl.when(pl.col("PrevOffsetEnd") < pl.col("Offset"))
                .then(True)
                .otherwise(False)
                .fill_null(True)
                .alias("IsNewVentilationPeriod"),
            )
            .with_columns(
                pl.col("IsNewVentilationPeriod")
                .cum_sum()
                .over(
                    partition_by=["CaseID", "Ventilation Type"],
                    order_by="Offset",
                )
                .alias("VentilationPeriodID"),
            )
            .group_by("CaseID", "Ventilation Type", "VentilationPeriodID")
            .agg(
                pl.col("Offset").min().alias("Offset"),
                pl.col("OffsetEnd").max().alias("OffsetEnd"),
            )
            # Rename columns for clarity
            .rename(
                {
                    "Offset": "Ventilation Start Relative to Admission (seconds)",
                    "OffsetEnd": "Ventilation End Relative to Admission (seconds)"
                } # fmt: skip
            )
            .pipe(self._add_global_id_stay_id, "sicdb-", "CaseID")
            .collect()
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
