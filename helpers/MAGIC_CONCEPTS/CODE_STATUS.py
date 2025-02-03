# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "code status" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
import os

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class CODE_STATUS(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def CODE_STATUS(self) -> pl.DataFrame:
        """
        Returns the magic concept CODE_STATUS

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - Global ICU stay ID
        - CODE_STATUS (ordinal scale):
            - full code
            - DNCPR (do not attempt CPR)
            - DNI (do not intubate)
            - DNR (do not resuscitate)
            - DNR / DNI (do not resuscitate / do not intubate)
            - CMO (comfort measures only)

        :return: DataFrame
        :rtype: pl.DataFrame
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

        mimic_CODE_IDS = [128, 223758]

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
                    "Do Not Resuscita",
                    "Do Not Intubate",
                    "DNR (do not resuscitate)",
                    "DNI (do not intubate)",
                    "DNR / DNI",
                    "CPR Not Indicate",
                ],
                "CODE_STATUS": [
                    "full code",
                    "full code",
                    "CMO",
                    "CMO",
                    "DNR",
                    "DNI",
                    "DNR",
                    "DNI",
                    "DNR / DNI",
                    "DNCPR",
                ],
            }
        )

        mimic3_CODE_STATUS = (
            pl.scan_csv(
                self.mimic3_paths.chartevents_path,
                schema_overrides={"VALUE": str},
            )
            .select("ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUE")
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
            .collect(streaming=True)
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
                    "Comfort measures only",
                    "DNI (do not intubate)",
                    "DNR (do not resuscitate)",
                    "DNR / DNI",
                ],
                "CODE_STATUS": [
                    "full code",
                    "CMO",
                    "DNI",
                    "DNR",
                    "DNR / DNI",
                ],
            }
        )

        mimic4_CODE_STATUS = (
            pl.scan_csv(
                self.mimic4_paths.chartevents_path,
                schema_overrides={"value": str},
            )
            .select("stay_id", "charttime", "itemid", "value")
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
            .collect(streaming=True)
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
            pl.scan_parquet(self.umcdb_paths.numericitems_path)
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
                    {"I": "full code", "II": "DNCPR", "III": "CMO"},
                    default=None,
                )
                .alias("CODE_STATUS"),
            )
            .drop_nulls("CODE_STATUS")
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
            .collect(streaming=True)
        )

        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Code Status")
        CODE_STATUS_ENUM = pl.Enum(
            ["full code", "DNCPR", "DNI", "DNR", "DNR / DNI", "CMO"]
        )

        CODE_STATUS = (
            pl.concat(
                [
                    eicu_CODE_STATUS,
                    mimic3_CODE_STATUS,
                    mimic4_CODE_STATUS,
                    umcdb_CODE_STATUS,
                ],
                how="diagonal_relaxed",
            )
            .select(
                self.column_names["global_icu_stay_id_col"],
                "Time Relative to Admission (seconds)",
                "CODE_STATUS",
            )
            .lazy()
            .cast({"CODE_STATUS": CODE_STATUS_ENUM})
            .group_by(
                self.column_names["global_icu_stay_id_col"],
                "Time Relative to Admission (seconds)",
            )
            .agg(
                pl.col("CODE_STATUS")
                .sort_by(pl.col("CODE_STATUS"))
                .first()
                .alias("CODE_STATUS")
            )
            # Remove duplicates
            .filter(
                pl.col("CODE_STATUS").ne_missing(
                    pl.col("CODE_STATUS").shift(1).over("Global ICU Stay ID")
                ),
            )
            .sort("Global ICU Stay ID", "Time Relative to Admission (seconds)")
        )
        # endregion

        return CODE_STATUS

    # region helpers
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
