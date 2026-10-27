import polars as pl

from ..MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_UMCdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

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
            # The AmsterdamUMCdb notebooks have no pause rule, MAX_VENTILATION_PAUSE_HOURS is the
            # value of the concept (8 h, also the one of ventilation_durations.sql of MIMIC-III):
            # rows of a stay and type that overlap or are at most that far apart are one episode
            .sort("admissionid", "item", "start")
            .with_columns(
                (
                    pl.col("start")
                    > pl.col("stop").cum_max().shift(1).over("admissionid", "item")
                    + self.MAX_VENTILATION_PAUSE_HOURS * 60 * 60
                )
                .fill_null(True)
                .cum_sum()
                .over("admissionid", "item")
                .alias("episode")
            )
            .group_by("admissionid", "item", "episode")
            .agg(pl.col("start").min(), pl.col("stop").max())
            .drop("episode")
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
