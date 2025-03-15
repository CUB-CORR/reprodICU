# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Ventilation Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATIONS.VENTILATION_DURATION_eICU import \
    VENTILATION_DURATION_eICU
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATIONS.VENTILATION_DURATION_HiRID import \
    VENTILATION_DURATION_HiRID
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATIONS.VENTILATION_DURATION_MIMIC3 import \
    VENTILATION_DURATION_MIMIC3
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATIONS.VENTILATION_DURATION_MIMIC4 import \
    VENTILATION_DURATION_MIMIC4
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATIONS.VENTILATION_DURATION_SICdb import \
    VENTILATION_DURATION_SICdb
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATIONS.VENTILATION_DURATION_UMCdb import \
    VENTILATION_DURATION_UMCdb


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

        eicu_VENTILATION_DURATION = VENTILATION_DURATION_eICU(
            self.paths, self.datasets, MAX_VENTILATION_PAUSE_HOURS=24
        ).VENTILATION_DURATION()

        hirid_VENTILATION_DURATION = VENTILATION_DURATION_HiRID(
            self.paths, self.datasets, MAX_VENTILATION_PAUSE_HOURS
        ).VENTILATION_DURATION()

        mimic3_VENTILATION_DURATION = VENTILATION_DURATION_MIMIC3(
            self.paths, self.datasets, MAX_VENTILATION_PAUSE_HOURS=8
        ).VENTILATION_DURATION()

        mimic4_VENTILATION_DURATION = VENTILATION_DURATION_MIMIC4(
            self.paths, self.datasets, MAX_VENTILATION_PAUSE_HOURS=14
        ).VENTILATION_DURATION()

        sicdb_VENTILATION_DURATION = VENTILATION_DURATION_SICdb(
            self.paths, self.datasets, MAX_VENTILATION_PAUSE_HOURS
        ).VENTILATION_DURATION()

        umcdb_VENTILATION_DURATION = VENTILATION_DURATION_UMCdb(
            self.paths, self.datasets
        ).VENTILATION_DURATION()

        # region ALL
        print("MAGIC_CONCEPTS: Ventilation Duration")

        VENTILATION_DURATION = (
            pl.concat(
                [
                    eicu_VENTILATION_DURATION.lazy(),
                    hirid_VENTILATION_DURATION.lazy(),
                    mimic3_VENTILATION_DURATION.lazy(),
                    mimic4_VENTILATION_DURATION.lazy(),
                    sicdb_VENTILATION_DURATION.lazy(),
                    umcdb_VENTILATION_DURATION.lazy(),
                ],
                how="diagonal_relaxed",
            )
            .filter(
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
