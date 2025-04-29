import os
import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_HiRID(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets, MAX_VENTILATION_PAUSE_HOURS):
        super().__init__(paths, datasets)
        self.MAX_VENTILATION_PAUSE_HOURS = MAX_VENTILATION_PAUSE_HOURS

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Ventilation Duration - HiRID")

        # get admission times for HiRID
        ADMISSIONTIMES = (
            pl.scan_csv(self.hirid_paths.general_table_path)
            .select("patientid", "admissiontime")
            .with_columns(
                pl.col("patientid").cast(str),
                pl.col("admissiontime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # Scan all files in the timeseries folder
        VENTILATION_DURATION = pl.LazyFrame()

        for file in os.listdir(self.hirid_paths.timeseries_path):
            timeseries_AIRWAYTYPE = (
                pl.scan_parquet(self.hirid_paths.timeseries_path + file)
                .select("datetime", "patientid", "value", "variableid")
                .cast({"datetime": str, "patientid": str})
                # Filter for ventilation IDs
                .filter(
                    pl.col("variableid")
                    == self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["hirid"][0]["ids"]
                )
                .drop("variableid")
                # replace ventilation concepts
                .with_columns(
                    pl.col("value")
                    .cast(int)
                    .cast(str)
                    .replace(
                        {
                            "1": "invasive ventilation",
                            "2": "tracheostomy",
                            "3": "non-invasive ventilation",
                            "4": "non-invasive ventilation",
                            "5": "non-invasive ventilation",
                            "6": "other",  # TODO: check if this is correct
                        }
                    )
                    .alias("Ventilation Type"),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                )
                # Make datetime relative to admission in seconds
                .join(ADMISSIONTIMES, on="patientid", how="left")
                .with_columns(
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias("Ventilation Start Relative to Admission (seconds)")
                )
                .drop("admissiontime", "datetime", "value")
                # Rename columns
            )

            timeseries_VENTMODE = (
                pl.scan_parquet(self.hirid_paths.timeseries_path + file)
                .select("datetime", "patientid", "value", "variableid")
                .cast({"datetime": str, "patientid": str})
                # Filter for ventilation IDs
                .filter(pl.col("variableid") == 3845)  # Ventilator mode
                .drop("variableid")
                # replace ventilation concepts
                .with_columns(
                    pl.when(pl.col("value").gt(1))
                    .then(pl.lit("active"))
                    .otherwise(pl.lit("inactive"))
                    .alias("Ventilator Mode"),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                )
                # Make datetime relative to admission in seconds
                .join(ADMISSIONTIMES, on="patientid", how="left")
                .with_columns(
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias("Ventilation Start Relative to Admission (seconds)")
                )
                .drop("admissiontime", "datetime", "value")
                # Rename columns
            )

            timeseries = (
                pl.concat(
                    [timeseries_AIRWAYTYPE, timeseries_VENTMODE],
                    how="align",
                )
                .select(pl.all().forward_fill())
                .sort(
                    "patientid",
                    "Ventilation Start Relative to Admission (seconds)",
                )
                # drop rows where both columns are staying the same
                .filter(
                    pl.col("Ventilator Mode").ne_missing(
                        pl.col("Ventilator Mode")
                        .shift(1)
                        .over("patientid", "Ventilation Type")
                    )
                    | pl.col("Ventilator Mode").ne_missing(
                        pl.col("Ventilator Mode")
                        .shift(-1)
                        .over("patientid", "Ventilation Type")
                    )
                    | pl.col("Ventilation Type")
                    .shift(1)
                    .over("patientid")
                    .is_null()
                    | pl.col("Ventilation Type")
                    .shift(-1)
                    .over("patientid")
                    .is_null()
                )
                .with_columns(
                    pl.col("Ventilation Start Relative to Admission (seconds)")
                    .shift(-1)
                    .over("patientid")
                    .alias("Ventilation End Relative to Admission (seconds)")
                )
                .drop_nulls("Ventilation End Relative to Admission (seconds)")
                .filter(pl.col("Ventilator Mode") == "active")
            )

            VENTILATION_DURATION = pl.concat(
                [VENTILATION_DURATION, timeseries],
                how="diagonal_relaxed",
            )

        return VENTILATION_DURATION.pipe(
            self._add_global_id_stay_id, "hirid-", "patientid"
        ).collect(streaming=True)

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
