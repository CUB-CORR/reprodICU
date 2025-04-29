import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION_UMCdb(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.DataFrame:
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
            # Rename columns
            .rename(
                {
                    "item": "Ventilation Type",
                    "start": "Ventilation Start Relative to Admission (seconds)",
                    "stop": "Ventilation End Relative to Admission (seconds)",
                }
            )
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
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
