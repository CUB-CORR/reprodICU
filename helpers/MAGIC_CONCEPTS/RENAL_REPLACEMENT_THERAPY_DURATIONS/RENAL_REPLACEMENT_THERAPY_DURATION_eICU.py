import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS
from helpers.A_extract.A_extract_eicu import EICUExtractor


class RENAL_REPLACEMENT_THERAPY_DURATION_eICU(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self) -> pl.DataFrame:
        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration - eICU")
        
        eicu_extractor = EICUExtractor(self.paths, DEMO=False)
        RENAL_REPLACEMENT_THERAPY_DURATION = (
            eicu_extractor.extract_treatments(verbose=False)
            .rename(
                {
                    self.column_names[
                        "procedure_start_col"
                    ]: "Renal Replacement Therapy Start Relative to Admission (seconds)",
                    self.column_names[
                        "procedure_end_col"
                    ]: "Renal Replacement Therapy End Relative to Admission (seconds)",
                    self.column_names["procedure_description_col"]: "RRT Type",
                }
            )
            .filter(
                pl.col("RRT Type").str.contains("Renal - Dialysis"),
                pl.col("RRT Type")
                .str.contains_any(["Arteriovenous Shunt", "Venous Catheter"])
                .not_(),
            )
            .with_columns(
                pl.when(pl.col("RRT Type").str.contains("C A V H D"))
                .then(pl.lit("CAVHD"))
                .when(pl.col("RRT Type").str.contains("C V V H"))
                .then(pl.lit("CVVH"))
                .when(pl.col("RRT Type").str.contains("C V V H D"))
                .then(pl.lit("CVVHD"))
                .when(pl.col("RRT Type").str.contains("Hemodialysis"))
                .then(pl.lit("CVVHDF"))
                .when(pl.col("RRT Type").str.contains("Peritoneal Dialysis"))
                .then(pl.lit("Peritoneal dialysis"))
                .when(pl.col("RRT Type").str.contains("Ultrafiltration"))
                .then(pl.lit("SCUF"))
                .when(pl.col("RRT Type").str.contains("SLED"))
                .then(pl.lit("SLED"))
                .otherwise(None)
                .alias("Renal Replacement Therapy Type"),
            )
            .collect(streaming=True)
        )
        
        return RENAL_REPLACEMENT_THERAPY_DURATION.unique().pipe(
                self._add_global_id_stay_id,
                "eicu-",
                self.column_names["icu_stay_id_col"],
            ).lazy()

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
