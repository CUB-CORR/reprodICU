# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the SICdb data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.AX_extract_sicdb import SICdbExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class SICdbProcessor(SICdbExtractor):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.sicdb_source_path
        self.helpers = GlobalHelpers()
        self.convert = SICdbConverter()
        self.icu_stay_id = self.extract_patient_information().select(
            [
                self.icu_stay_id_col,
                self.hospital_stay_id_col,
                self.person_id_col,
            ]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region timeseries
    # # Processes and combines the time series data of the eICU dataset.
    # def process_timeseries(self) -> pl.LazyFrame:
    #     # Load preexisting data if available
    #     if os.path.isfile(self.precalc_path + "SICdb_B_timeseries.parquet"):
    #         return pl.scan_parquet(
    #             self.precalc_path + "SICdb_B_timeseries.parquet"
    #         )

    #     # Load the time series data
    #     print("SICdb   - Loading time series data...")

    #     ts_float = self._process_timeseries_data_float()
    #     ts_labs = self._process_timeseries_data_labs()

    #     timeseries = ts_float.sort(self.index_cols)
    #     timeseries.sink_parquet(
    #         self.precalc_path + "SICdb_B_timeseries.parquet"
    #     )
    #     return timeseries

    def process_timeseries_data_float(self) -> pl.LazyFrame:
        if os.path.isfile(self.precalc_path + "SICdb_B_timeseries.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "SICdb_B_timeseries.parquet"
            )

        print("SICdb   - Processing time series data...")

        # Process timeseries data
        timeseries = (
            self.extract_timeseries()
            # Pivot the timeseries data
            .collect(streaming=True).pivot(
                on="DataID",
                index=self.index_cols,
                values="Val",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(
            set(timeseries.collect_schema().names()) - set(self.index_cols)
        )
        timeseries = (
            timeseries.pipe(self.helpers.dropna, "all", droplist, False)
            .lazy()
            .sort(self.index_cols)
            .unique()
        )

        # Save the preprocessed data
        timeseries.sink_parquet(
            self.precalc_path + "SICdb_B_timeseries.parquet"
        )

        return timeseries

    # endregion

    # region lab values
    def process_timeseries_data_labs(self) -> pl.LazyFrame:
        if os.path.isfile(
            self.precalc_path + "SICdb_B_timeseries_labs.parquet"
        ):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "SICdb_B_timeseries_labs.parquet"
            )

        print("SICdb   - Processing laboratory data...")

        # Process timeseries data
        timeseries = (
            self.extract_laboratory_timeseries()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="LaboratoryID",
                valuecol="LaboratoryValue",
            )
            # Pivot the timeseries data
            .collect(streaming=True).pivot(
                on="LaboratoryID",
                index=self.index_cols,
                values="LaboratoryValue",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(
            set(timeseries.collect_schema().names()) - set(self.index_cols)
        )
        timeseries = (
            timeseries.pipe(self.helpers.dropna, "all", droplist, False)
            .lazy()
            .sort(self.index_cols)
            .unique()
        )

        # Save the preprocessed data
        timeseries.sink_parquet(
            self.precalc_path + "SICdb_B_timeseries_labs.parquet"
        )

        return timeseries

    # endregion


# region convert
class SICdbConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "LaboratoryID",
        valuecol: str = "LaboratoryValue",
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the SICdb dataset.
        """

        # Convert the lab values to the correct units.
        return (
            data.pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.direct [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.total [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea [Mass/volume] in Serum or Plasma",
                itemid_BUN="Urea nitrogen [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .with_columns(
                pl.col(labelcol).replace(
                    {
                        "Bilirubin.direct [Mass/volume] in Serum or Plasma": "Bilirubin.direct [Moles/volume] in Serum or Plasma",
                        "Bilirubin.total [Mass/volume] in Serum or Plasma": "Bilirubin.total [Moles/volume] in Serum or Plasma",
                        "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma": "Cobalamin (Vitamin B12) [Moles/volume] in Serum or Plasma",
                        "Iron [Mass/volume] in Serum or Plasma": "Iron [Mass/volume] in Blood",
                        # NOTE: rename for consistency
                        "Anion gap 4 in Arterial blood": "Anion gap in Blood",
                        "Band form neutrophils/100 leukocytes in Blood by Manual count": "Band form neutrophils/100 leukocytes in Blood",
                        "Basophils/100 leukocytes in Blood by Manual count": "Basophils/100 leukocytes in Blood",
                        "Eosinophils/100 leukocytes in Blood by Manual count": "Eosinophils/100 leukocytes in Blood",
                        "Lymphocytes/100 leukocytes in Blood by Manual count": "Lymphocytes/100 leukocytes in Blood",
                        "Monocytes/100 leukocytes in Blood by Manual count": "Monocytes/100 leukocytes in Blood",
                        "Segmented neutrophils/100 leukocytes in Blood by Manual count": "Segmented neutrophils/100 leukocytes in Blood",
                        "Calcium [Moles/volume] in Serum or Plasma": "Calcium [Moles/volume] in Blood",
                        "Calcium.ionized [Moles/volume] in Arterial blood": "Calcium.ionized [Moles/volume] in Blood",
                        "Chloride [Moles/volume] in Arterial blood": "Chloride [Moles/volume] in Blood",
                        "Chloride [Moles/volume] in Serum or Plasma": "Chloride [Moles/volume] in Blood",
                        "Erythrocyte distribution width [Ratio] by Automated count": "Erythrocyte distribution width [Ratio]",
                        "Hematocrit [Volume Fraction] of Arterial blood": "Hematocrit [Volume Fraction] of Blood",
                        "Monocytes/100 leukocytes in Blood by Manual count": "Monocytes/100 leukocytes in Blood",
                        "Fractional oxyhemoglobin in Arterial blood": "Oxyhemoglobin/Hemoglobin.total in Arterial blood",
                        "Potassium [Moles/volume] in Arterial blood": "Potassium [Moles/volume] in Blood",
                        "Potassium [Moles/volume] in Serum or Plasma": "Potassium [Moles/volume] in Blood",
                        "Sodium [Moles/volume] in Arterial blood": "Sodium [Moles/volume] in Blood",
                        "Sodium [Moles/volume] in Serum or Plasma": "Sodium [Moles/volume] in Blood",
                        # NOTE: fixing wrong unit
                        "Thyroxine (T4) free [Mass/volume] in Serum or Plasma": "Thyroxine (T4) free [Moles/volume] in Serum or Plasma",
                    }
                )
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Convert the lab values of the SICdb dataset.
        """

        return (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Band form neutrophils [#/volume] in Blood by Manual count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Band form neutrophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Basophils [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Basophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils [#/volume] in Blood by Manual count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Eosinophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes [#/volume] in Blood by Manual count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Lymphocytes/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Monocytes [#/volume] in Blood by Manual count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Monocytes/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Neutrophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils [#/volume] in Blood by Manual count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Neutrophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes [#/volume] in Blood",
                total_itemcol="Erythrocytes [#/volume] in Blood",
                goal_itemcol="Reticulocytes/100 erythrocytes in Blood",
            )
        )


# endregion
