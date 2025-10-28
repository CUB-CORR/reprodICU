# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Renal Replacement Therapy Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import polars as pl
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS
from helpers.MAGIC_CONCEPTS.RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_eICU import \
    RENAL_REPLACEMENT_THERAPY_DURATION_eICU
from helpers.MAGIC_CONCEPTS.RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3 import \
    RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC3
from helpers.MAGIC_CONCEPTS.RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC4 import \
    RENAL_REPLACEMENT_THERAPY_DURATION_MIMIC4
from helpers.MAGIC_CONCEPTS.RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_SICdb import \
    RENAL_REPLACEMENT_THERAPY_DURATION_SICdb
from helpers.MAGIC_CONCEPTS.RENAL_REPLACEMENT_THERAPY_DURATIONS.RENAL_REPLACEMENT_THERAPY_DURATION_UMCdb import \
    RENAL_REPLACEMENT_THERAPY_DURATION_UMCdb


class RENAL_REPLACEMENT_THERAPY_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def RENAL_REPLACEMENT_THERAPY_DURATION(self):
        """
        Returns the magic concept RENAL_REPLACEMENT_THERAPY_DURATION

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - ICU stay ID
        - renal replacement therapy type "Renal Replacement Therapy Type", one of
            - "CVVH" (Continuous venovenous hemofiltration),
            - "CAVHD" (Continuous arteriovenous hemodialysis),
            - "CVVHD" (Continuous venovenous hemodialysis),
            - "CVVHDF" (Continuous venovenous hemodiafiltration)
            - "IHD" (Intermittent hemodialysis)
            - "Peritoneal dialysis"
            - "SCUF" (Slow continuous ultra filtration)
            - "SLED" (Sustained low-efficiency dialysis)
            - None (if the type could not be determined)
        - renal replacement therapy start "Renal Replacement Therapy Start Relative to Admission (seconds)"
        - renal replacement therapy end "Renal Replacement Therapy End Relative to Admission (seconds)"
        - renal replacement therapy duration "Renal Replacement Therapy Duration (hours)"

        :return: DataFrame
        :rtype: pl.DataFrame
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
