# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the so called MAGIC CONCEPT "Ventilation Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl

from .MAGIC_CONCEPTS import MAGIC_CONCEPTS
from .VENTILATION_DURATIONS.VENTILATION_DURATION_eICUv1 import (
    VENTILATION_DURATION_eICUv1,
)
from .VENTILATION_DURATIONS.VENTILATION_DURATION_HiRID import (
    VENTILATION_DURATION_HiRID,
)
from .VENTILATION_DURATIONS.VENTILATION_DURATION_MIMIC3 import (
    VENTILATION_DURATION_MIMIC3,
)
from .VENTILATION_DURATIONS.VENTILATION_DURATION_MIMIC4 import (
    VENTILATION_DURATION_MIMIC4,
)
from .VENTILATION_DURATIONS.VENTILATION_DURATION_SICdb import (
    VENTILATION_DURATION_SICdb,
)
from .VENTILATION_DURATIONS.VENTILATION_DURATION_UMCdb import (
    VENTILATION_DURATION_UMCdb,
)


class VENTILATION_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        """
        Extract mechanical ventilation periods and types.

        Steps:
            1. For each database: call database-specific VENTILATION_DURATION extractors.
            2. Extract ventilation start/end times and type classification.
            3. Calculate ventilation duration (hours).
            4. Standardize ventilation types across databases.
            5. Concatenate results from all databases.

        Returns:
            pl.DataFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - Ventilation Type: Ventilation category (tracheostomy, invasive, non-invasive, weaning, other, unknown).
                - Ventilation Start Relative to Admission (seconds): Start time.
                - Ventilation End Relative to Admission (seconds): End time.
                - Ventilation Duration (hours): Total duration (float).
        """

        print("MAGIC_CONCEPTS: Ventilation Duration - approx. 40 min")

        MAX_VENTILATION_PAUSE_HOURS = 8
        SECONDS_IN_1H = 60 * 60
        SECONDS_IN_1D = 24 * 60 * 60

        eicu_VENTILATION_DURATION = VENTILATION_DURATION_eICUv1(
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
        VENTILATION_TYPE_ENUM = pl.Enum(
            [
                "unknown",
                "other",
                "supplemental oxygen",
                "weaning",
                "non-invasive ventilation",
                "invasive ventilation",
                "tracheostomy",
            ]
        )
        vent_start_col = "Ventilation Start Relative to Admission (seconds)"
        vent_end_col = "Ventilation End Relative to Admission (seconds)"

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
            # Combine consecutive rows where end time equals next start time
            .with_columns(
                pl.struct(
                    pl.col("Global ICU Stay ID"),
                    pl.col(vent_start_col).eq_missing(
                        pl.col(vent_end_col)
                        .shift(1)
                        .fill_null(pl.col(vent_start_col).min())
                        .over(
                            partition_by=[
                                "Global ICU Stay ID",
                                "Ventilation Type",
                            ],
                            order_by=vent_start_col,
                        )
                    ),
                )
                .rle_id()
                .alias("is_consecutive"),
            )
            .group_by(
                "Global ICU Stay ID",
                "Ventilation Type",
                "is_consecutive",
            )
            .agg(
                pl.col(vent_start_col).min(),
                pl.col(vent_end_col).max(),
            )
            # Filter out rows where start time is after end time
            .filter(
                pl.col(vent_start_col).lt(pl.col(vent_end_col)),
                pl.col(vent_end_col).gt(
                    -self.global_vars.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    * (SECONDS_IN_1D)
                ),
            )
            .unique()
            .select(
                "Global ICU Stay ID",
                "Ventilation Type",
                vent_start_col,
                vent_end_col,
            )
            # .cast({"Ventilation Type": VENTILATION_TYPE_ENUM})
            .group_by(
                "Global ICU Stay ID",
                vent_start_col,
                vent_end_col,
            )
            .agg(pl.col("Ventilation Type").max())
            .with_columns(
                (pl.col(vent_end_col) - pl.col(vent_start_col))
                .truediv(SECONDS_IN_1H)
                .round(2)
                .alias("Ventilation Duration (hours)")
            )
            .lazy()
        )

        return VENTILATION_DURATION
