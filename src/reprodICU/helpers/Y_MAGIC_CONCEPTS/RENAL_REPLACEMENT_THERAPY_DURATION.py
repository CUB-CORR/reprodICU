# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the so called MAGIC CONCEPT "Renal Replacement Therapy Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl

from .MAGIC_CONCEPTS import MAGIC_CONCEPTS
from .RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_eICU import (
    RENAL_REPLACEMENT_THERAPY_DURATION_eICU,
)
from .RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3 import (
    RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3,
)
from .RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC4 import (
    RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC4,
)
from .RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_SICdb import (
    RENAL_REPLACEMENT_THERAPY_DURATION_SICdb,
)
from .RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_UMCdb import (
    RENAL_REPLACEMENT_THERAPY_DURATION_UMCdb,
)


class RENAL_REPLACEMENT_THERAPY_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self):
        """
        Extract renal replacement therapy (dialysis) periods and types.

        Steps:
            1. For each database: call database-specific RRT duration extractors.
            2. Extract RRT start/end times and modality classification.
            3. Calculate RRT duration (hours).
            4. Standardize RRT types across databases (CVVH, CVVHD, CVVHDF, IHD, etc.).
            5. Concatenate results from all databases.

        Returns:
            pl.DataFrame: Contains columns:
                - {global_icu_stay_id_col}: Global ICU stay identifier.
                - Renal Replacement Therapy Type: RRT modality (CVVH, CAVHD, CVVHD, CVVHDF, IHD, Peritoneal dialysis, SCUF, SLED, unknown).
                - Renal Replacement Therapy Start Relative to Admission (seconds): Start time.
                - Renal Replacement Therapy End Relative to Admission (seconds): End time.
                - Renal Replacement Therapy Duration (hours): Total duration (float).
        """

        print("MAGIC_CONCEPTS: Renal Replacement Therapy Duration")

        SECONDS_IN_1H = 60 * 60
        SECONDS_IN_1D = 24 * 60 * 60

        eicu_RENAL_REPLACEMENT_THERAPY_DURATION = (
            RENAL_REPLACEMENT_THERAPY_DURATION_eICU(
                self.paths, self.datasets
            ).RENAL_REPLACEMENT_THERAPY_DURATION()
        )

        mimic3_RENAL_REPLACEMENT_THERAPY_DURATION = (
            RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3(
                self.paths, self.datasets
            ).RENAL_REPLACEMENT_THERAPY_DURATION()
        )

        mimic4_RENAL_REPLACEMENT_THERAPY_DURATION = (
            RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC4(
                self.paths, self.datasets
            ).RENAL_REPLACEMENT_THERAPY_DURATION()
        )

        sicdb_RENAL_REPLACEMENT_THERAPY_DURATION = (
            RENAL_REPLACEMENT_THERAPY_DURATION_SICdb(
                self.paths, self.datasets
            ).RENAL_REPLACEMENT_THERAPY_DURATION()
        )

        umcdb_RENAL_REPLACEMENT_THERAPY_DURATION = (
            RENAL_REPLACEMENT_THERAPY_DURATION_UMCdb(
                self.paths, self.datasets
            ).RENAL_REPLACEMENT_THERAPY_DURATION()
        )

        RENAL_REPLACEMENT_THERAPY_DURATION = (
            pl.concat(
                [
                    eicu_RENAL_REPLACEMENT_THERAPY_DURATION.lazy(),
                    # hirid_RENAL_REPLACEMENT_THERAPY_DURATION,
                    mimic3_RENAL_REPLACEMENT_THERAPY_DURATION.lazy(),
                    mimic4_RENAL_REPLACEMENT_THERAPY_DURATION.lazy(),
                    sicdb_RENAL_REPLACEMENT_THERAPY_DURATION.lazy(),
                    umcdb_RENAL_REPLACEMENT_THERAPY_DURATION.lazy(),
                ],
                how="diagonal_relaxed",
            )
            .filter(
                pl.col(
                    "Renal Replacement Therapy Start Relative to Admission (seconds)"
                ).lt(
                    pl.col(
                        "Renal Replacement Therapy End Relative to Admission (seconds)"
                    )
                ),
                pl.col(
                    "Renal Replacement Therapy End Relative to Admission (seconds)"
                ).gt(
                    -self.global_vars.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                    * (SECONDS_IN_1D)
                ),
            )
            .unique()
            .select(
                "Global ICU Stay ID",
                "Renal Replacement Therapy Type",
                "Renal Replacement Therapy Start Relative to Admission (seconds)",
                "Renal Replacement Therapy End Relative to Admission (seconds)",
            )
            .group_by(
                "Global ICU Stay ID",
                "Renal Replacement Therapy Start Relative to Admission (seconds)",
                "Renal Replacement Therapy End Relative to Admission (seconds)",
            )
            .agg(pl.col("Renal Replacement Therapy Type").max())
            .with_columns(
                (
                    pl.col(
                        "Renal Replacement Therapy End Relative to Admission (seconds)"
                    )
                    - pl.col(
                        "Renal Replacement Therapy Start Relative to Admission (seconds)"
                    )
                )
                .truediv(SECONDS_IN_1H)
                .round(2)
                .alias("Renal Replacement Therapy Duration (hours)")
            )
            .lazy()
        )
        # endregion

        return RENAL_REPLACEMENT_THERAPY_DURATION
