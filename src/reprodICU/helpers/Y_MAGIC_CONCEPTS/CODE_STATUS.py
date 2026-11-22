# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the so called MAGIC CONCEPT "code status" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl

from .MAGIC_CONCEPTS import MAGIC_CONCEPTS


class CODE_STATUS(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def CODE_STATUS(self) -> pl.DataFrame:
        """
        Extract patient code status (resuscitation orders) from multiple databases.

        Steps:
            1. For each database: extract code status records and timestamps.
            2. Standardize code status values to common categories.
            3. Calculate time relative to ICU admission.
            4. Concatenate from all databases.

        Returns:
            pl.DataFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - CODE_STATUS: Code status category (full code, DNCPR, DNI, DNR, DNR / DNI, CMO).
        """

        # region eICU
        print("MAGIC_CONCEPTS: Code Status - eICU")
        eicu_CODE_STATUS = (
            pl.scan_csv(self.eicu_paths.carePlanGeneral_path)
            .select(
                "patientunitstayid",
                "cplitemoffset",
                "cplgroup",
                "cplitemvalue",
                # "activeupondischarge", # is true for last value
            )
            .filter(pl.col("cplgroup") == "Care Limitation")
            .drop("cplgroup")
            .with_columns(
                # Convert minutes to seconds
                (pl.col("cplitemoffset") * 60).alias(
                    "Time Relative to Admission (seconds)"
                ),
                # Replace values with standardized values
                pl.col("cplitemvalue")
                .replace_strict(
                    {
                        "Comfort measures only": "CMO",
                        "Do not resuscitate": "DNR",
                        "Full therapy": "full code",
                        "No CPR": "DNCPR",
                        "No intubation": "DNI",
                    },
                    # filter out other values
                    default=None,
                )
                .alias("CODE_STATUS"),
            )
            .drop_nulls("CODE_STATUS")
            .pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")
            .collect()
        )

        # endregion

        # region MIMIC-III
        print("MAGIC_CONCEPTS: Code Status - MIMIC-III")

        mimic_CODE_IDS = [128, 223758, 228687]

        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic3_CODE_STATUS_MAP = pl.LazyFrame(
            {
                "VALUE": [
                    "Full Code",
                    "Full code",
                    "Comfort Measures",
                    "Comfort measures only",
                    "Do Not Intubate",
                    "DNI (do not intubate)",
                    "Do Not Resuscita",
                    "DNR (do not resuscitate)",
                    "DNAR (Do Not Attempt Resuscitation)  [DNR]",
                    "DNR / DNI",
                    "DNAR (Do Not Attempt Resuscitation) [DNR] / DNI",
                    "CPR Not Indicate",
                ],
                "CODE_STATUS": [
                    "full code",
                    "full code",
                    "CMO",
                    "CMO",
                    "DNI",
                    "DNI",
                    "DNR",
                    "DNR",
                    "DNR",
                    "DNR / DNI",
                    "DNR / DNI",
                    "DNCPR",
                ],
            }
        )

        if "parquet" in self.mimic3_paths.chartevents_path:
            mimic3_chartevents = pl.scan_parquet(
                self.mimic3_paths.chartevents_path
            )
        else:
            mimic3_chartevents = pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )

        mimic3_CODE_STATUS = (
            mimic3_chartevents.select(
                "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE"
            )
            # Filter for scores
            .filter(pl.col("ITEMID").is_in(mimic_CODE_IDS))
            .drop("ITEMID")
            .with_columns(
                pl.col("CHARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            .with_columns(
                # Make datetime relative to admission in seconds
                (pl.col("CHARTTIME") - pl.col("INTIME"))
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)"),
            )
            # Replace values with standardized values
            .join(mimic3_CODE_STATUS_MAP, on="VALUE", how="left")
            .drop("VALUE")
            .drop_nulls("CODE_STATUS")
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
            .collect()
        )

        # endregion

        # region MIMIC-IV
        print("MAGIC_CONCEPTS: Code Status - MIMIC-IV")
        mimic4_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic4_CODE_STATUS_MAP = pl.LazyFrame(
            {
                "value": [
                    "Full code",
                    "DNI (do not intubate)",
                    "DNR (do not resuscitate)",
                    "DNAR (Do Not Attempt Resuscitation)  [DNR]",
                    "DNR / DNI",
                    "DNAR (Do Not Attempt Resuscitation) [DNR] / DNI",
                    "Comfort measures only",
                ],
                "CODE_STATUS": [
                    "full code",
                    "DNI",
                    "DNR",
                    "DNR",
                    "DNR / DNI",
                    "DNR / DNI",
                    "CMO",
                ],
            }
        )

        if "parquet" in self.mimic4_paths.chartevents_path:
            mimic4_chartevents = pl.scan_parquet(
                self.mimic4_paths.chartevents_path
            )
        else:
            mimic4_chartevents = pl.scan_csv(
                self.mimic4_paths.chartevents_path,
                schema_overrides={"VALUE": str, "VALUENUM": float},
            )

        mimic4_CODE_STATUS = (
            mimic4_chartevents.select("stay_id", "charttime", "itemid", "value")
            # Filter for scores
            .filter(pl.col("itemid").is_in(mimic_CODE_IDS))
            .drop("itemid")
            .with_columns(
                pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
            # Make datetime relative to admission in seconds
            .with_columns(
                (pl.col("charttime") - pl.col("intime"))
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)"),
            )
            # Replace values with standardized values
            .join(mimic4_CODE_STATUS_MAP, on="value", how="left")
            .drop("value")
            .drop_nulls("CODE_STATUS")
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
            .collect()
        )

        # Hospital-wide provider orders (as in mimic-code code_status.sql),
        # kept for the ICU stay that the order time falls into
        if "parquet" in self.mimic4_paths.poe_path:
            mimic4_poe = pl.scan_parquet(self.mimic4_paths.poe_path)
            mimic4_poe_detail = pl.scan_parquet(self.mimic4_paths.poe_detail_path) # fmt: skip
        else:
            mimic4_poe = pl.scan_csv(self.mimic4_paths.poe_path)
            mimic4_poe_detail = pl.scan_csv(self.mimic4_paths.poe_detail_path)

        mimic4_POE_CODE_STATUS_MAP = pl.LazyFrame(
            {
                "field_value": [
                    "Resuscitate (Full code)",
                    "Full code  (attempt resuscitation)",
                    "DNAR (DO NOT attempt resuscitation for cardiac arrest) ",
                    "Do not resuscitate (DNR/DNI)",
                ],
                "CODE_STATUS": ["full code", "full code", "DNR", "DNR / DNI"],
            }
        )

        mimic4_POE_CODE_STATUS = (
            mimic4_poe.filter(
                pl.col("order_type") == "General Care",
                pl.col("order_subtype") == "Code status",
            )
            .select("poe_id", "hadm_id", "ordertime")
            .join(
                mimic4_poe_detail.filter(
                    pl.col("field_name") == "Code status"
                ).select("poe_id", "field_value"),
                on="poe_id",
            )
            .join(mimic4_POE_CODE_STATUS_MAP, on="field_value")
            .with_columns(
                pl.col("ordertime").str.to_datetime("%Y-%m-%d %H:%M:%S")
            )
            .join(
                pl.scan_csv(self.mimic4_paths.icustays_path)
                .select("stay_id", "hadm_id", "intime", "outtime")
                .with_columns(
                    pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                    pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                ),
                on="hadm_id",
            )
            .filter(
                pl.col("ordertime").is_between(
                    pl.col("intime"), pl.col("outtime")
                )
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                (pl.col("ordertime") - pl.col("intime"))
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)"),
            )
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
            .collect()
        )

        # endregion

        # region UMCdb
        print("MAGIC_CONCEPTS: Code Status - UMCdb")

        umcdb_INTIMES = (
            pl.scan_parquet(self.umcdb_paths.admissions_path)
            .select("admissionid", "admittedat", "dischargedat")
            .rename({"admittedat": "intime", "dischargedat": "outtime"})
        )

        umcdb_CODE_STATUS = (
            pl.scan_parquet(self.umcdb_paths.listitems_path)
            .select("admissionid", "itemid", "value", "measuredat")
            # Filter for scores
            .filter(pl.col("itemid") == 10673)
            .drop("itemid")
            .join(umcdb_INTIMES, on="admissionid", how="left")
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("measuredat") - pl.col("intime"))
                )
                .dt.total_seconds()
                .alias("Time Relative to Admission (seconds)"),
                pl.col("value")
                .replace_strict(
                    # see https://github.com/AmsterdamUMC/AmsterdamUMCdb/issues/90
                    {"I": "full code", "II": "DNR / DNI", "III": "CMO"},
                    default=None,
                )
                .alias("CODE_STATUS"),
            )
            .drop_nulls("CODE_STATUS")
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
            .collect()
        )

        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Code Status")
        CODE_STATUS = self._collapse(
            pl.concat(
                [
                    eicu_CODE_STATUS,
                    mimic3_CODE_STATUS,
                    mimic4_CODE_STATUS,
                    mimic4_POE_CODE_STATUS,
                    umcdb_CODE_STATUS,
                ],
                how="diagonal_relaxed",
            ).lazy()
        )
        # endregion

        return CODE_STATUS

    # region helpers
    def _collapse(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """One status per stay and time (the most restrictive), consecutive repeats dropped."""
        CODE_STATUS_ENUM = pl.Enum(
            ["full code", "DNCPR", "DNI", "DNR", "DNR / DNI", "CMO"]
        )
        return (
            data.select(
                self.column_names["global_icu_stay_id_col"],
                "Time Relative to Admission (seconds)",
                "CODE_STATUS",
            )
            .cast({"CODE_STATUS": CODE_STATUS_ENUM})
            .group_by(
                self.column_names["global_icu_stay_id_col"],
                "Time Relative to Admission (seconds)",
            )
            .agg(
                # most restrictive status; a DNR together with a DNI is "DNR / DNI"
                pl.col("CODE_STATUS").max(),
                pl.col("CODE_STATUS")
                .is_in(["DNR", "DNR / DNI"])
                .any()
                .alias("dnr"),
                pl.col("CODE_STATUS")
                .is_in(["DNI", "DNR / DNI"])
                .any()
                .alias("dni"),
            )
            .with_columns(
                pl.when(
                    (pl.col("CODE_STATUS") != "CMO")
                    & pl.col("dnr")
                    & pl.col("dni")
                )
                .then(pl.lit("DNR / DNI", dtype=CODE_STATUS_ENUM))
                .otherwise(pl.col("CODE_STATUS"))
                .alias("CODE_STATUS")
            )
            .drop("dnr", "dni")
            .sort("Global ICU Stay ID", "Time Relative to Admission (seconds)")
            # Remove consecutive duplicates (needs the sorted order)
            .filter(
                pl.col("CODE_STATUS").ne_missing(
                    pl.col("CODE_STATUS").shift(1).over("Global ICU Stay ID")
                ),
            )
        )

    def _add_global_id_stay_id(
        self, data: pl.LazyFrame, source_dataset: str, stay_id_col: str
    ) -> pl.LazyFrame:
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str(
                [pl.lit(source_dataset), pl.col(stay_id_col).cast(str)]
            ).alias(self.column_names["global_icu_stay_id_col"])
        ).drop(stay_id_col)

    # endregion
