# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Ventilation Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        """
        Returns the magic concept VENTILATION_DURATION

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - ICU stay ID
        - Ventilation Type, one of
            - tracheostomy
            - invasive ventilation
            - non-invasive ventilation
            - weaning
            - other
            - unknown
        - Ventilation Start Relative to Admission (seconds)
        - Ventilation End Relative to Admission (seconds)
        - Ventilation Duration (hours)

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        print("MAGIC_CONCEPTS: Ventilation Duration - approx. 40 min")

        MAX_VENTILATION_PAUSE_HOURS = 8
        SECONDS_IN_1H = 60 * 60
        SECONDS_IN_1D = 24 * 60 * 60

        RESPIRATORY = (
            pl.scan_parquet(
                self.paths.reprodICU_files_path
                + "timeseries_respiratory.parquet"
            )
            .head()
            .collect()
        )

        # region ALL
        print("MAGIC_CONCEPTS: Ventilation Duration")

        VENTILATION_TYPE_ENUM = pl.Enum(
            [
                "tracheostomy",
                "invasive ventilation",
                "non-invasive ventilation",
                "weaning",
                "supplemental oxygen",
                "unknown",
                "other",
            ]
        )
        VENTILATION_DURATION = (
            VENTILATION_DURATION_.filter(
                pl.col("Ventilation Start Relative to Admission (seconds)").lt(
                    pl.col("Ventilation End Relative to Admission (seconds)")
                ),
                pl.col("Ventilation End Relative to Admission (seconds)").gt(
                    -self.global_vars.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    * (SECONDS_IN_1D)
                ),
            )
            .unique()
            .select(
                "Global ICU Stay ID",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
            # .cast({"Ventilation Type": VENTILATION_TYPE_ENUM})
            .group_by(
                "Global ICU Stay ID",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
            )
            .agg(pl.col("Ventilation Type").max())
            .with_columns(
                (
                    pl.col("Ventilation End Relative to Admission (seconds)")
                    - pl.col(
                        "Ventilation Start Relative to Admission (seconds)"
                    )
                )
                .truediv(SECONDS_IN_1H)
                .round(2)
                .alias("Ventilation Duration (hours)")
            )
            .lazy()
        )

        return VENTILATION_DURATION
