import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_UMCdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        """
        Extract ventilation episodes from UMCdb processitems.

        Steps:
            1. Extract ventilation process items with start/stop times.
            2. Join with admission times.
            3. Filter for ventilation-related process IDs.
            4. Calculate episode duration.
            5. Compute time relative to admission.

        Returns:
            pl.DataFrame: Contains columns:
                - admissionid: Admission identifier.
                - {timeseries_time_col}: Ventilation start time (seconds from admission).
                - Ventilation Type: Classification (invasive ventilation, etc.).
                - Ventilation Duration (hours): Episode duration.
        """
        print("MAGIC_CONCEPTS: Ventilation Duration - UMCdb")

        ADMISSION_TIMES = pl.scan_parquet(
            self.umcdb_paths.admissions_path
        ).select("admissionid", "admittedat")

        return (
            pl.scan_parquet(self.umcdb_paths.processitems_path)
            .join(ADMISSION_TIMES, on="admissionid", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["aumc"][0]["ids"]
                    + [9671]  # CPAP
                )
            )
            .drop("itemid")
            # replace ventilation concepts
            .with_columns(
                pl.col("item")
                .replace(
                    {
                        "Beademen": "invasive ventilation",
                        "Beademen non-invasief": "non-invasive ventilation",
                        "CPAP": "non-invasive ventilation",
                        "Tracheostoma": "tracheostomy",
                    }
                )
                .cast(str)
                .alias("item")
            )
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
            # # Fix overlapping periods
            # .with_columns(
            #     pl.col("stop")
            #     .shift(1)
            #     .over(
            #         partition_by=["admissionid", "Ventilation Type"],
            #         order_by="start",
            #     )
            #     .alias("prevstop"),
            # )
            # .with_columns(
            #     pl.when(pl.col("prevstop") < pl.col("start"))
            #     .then(True)
            #     .otherwise(False)
            #     .fill_null(True)
            #     .alias("isnewventilationperiod"),
            # )
            # .with_columns(
            #     pl.col("isnewventilationperiod")
            #     .cum_sum()
            #     .over(
            #         partition_by=["admissionid", "Ventilation Type"],
            #         order_by="start",
            #     )
            #     .alias("ventilationperiodid"),
            # )
            # .group_by("admissionid", "Ventilation Type", "ventilationperiodid")
            # .agg(
            #     pl.col("start").min().alias("start"),
            #     pl.col("stop").max().alias("stop"),
            # )
            # Rename columns
            .rename(
                {
                    "item": "Ventilation Type",
                    "start": (
                        "Ventilation Start Relative to Admission (seconds)"
                    ),
                    "stop": "Ventilation End Relative to Admission (seconds)",
                }
            )
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
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
