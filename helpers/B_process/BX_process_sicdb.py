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
            self.icu_stay_id_col,
            self.hospital_stay_id_col,
            self.person_id_col,
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    def process_timeseries_data_float(self) -> pl.LazyFrame:
        """
        Processes the time series data of the SICdb dataset.
        """
        ts_float_path = self.precalc_path + "SICdb_timeseries.parquet"
        ts_float_path_unsorted = self.precalc_path + "SICdb_ts.parquet"

        if os.path.isfile(ts_float_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_float_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
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
        timeseries.sink_parquet(ts_float_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_float_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_float_path)
        )
        os.remove(ts_float_path_unsorted)

        return pl.scan_parquet(ts_float_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region lab values
    def process_timeseries_data_labs(self) -> pl.LazyFrame:
        """
        Processes the laboratory time series data of the SICdb dataset.
        """
        ts_labs_path = self.precalc_path + "SICdb_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "SICdb_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("SICdb   - Processing laboratory data...")

        # Process timeseries data
        timeseries = (
            self.extract_laboratory_timeseries()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="LaboratoryID",
                valuecol="value_struct",
            )
            .with_columns(
                pl.col("value_struct")
                .struct.json_encode()
                .alias("value_struct")
            )
            # Pivot the timeseries data
            .collect(streaming=True)
            .pivot(
                on="LaboratoryID",
                index=self.index_cols,
                values="value_struct",
                aggregate_function="first",  # NOTE: mean is used here -> check if this is sensible
            )
            .lazy()
        )

        # Save the preprocessed data
        timeseries.sink_parquet(ts_labs_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_labs_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_labs_path)
        )
        os.remove(ts_labs_path_unsorted)

        return pl.scan_parquet(ts_labs_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

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
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the SICdb dataset.
        """

        # Convert the lab values to the correct units.
        return (
            data.pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.direct [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.total [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="Cobalamin (Vitamin B12) [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea [Mass/volume]",
                itemid_BUN="Urea nitrogen [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .with_columns(
                pl.col(labelcol).replace(
                    {
                        "Bilirubin.direct [Mass/volume]": "Bilirubin.direct [Moles/volume]",
                        "Bilirubin.total [Mass/volume]": "Bilirubin.total [Moles/volume]",
                        "Cobalamin (Vitamin B12) [Mass/volume]": "Cobalamin (Vitamin B12) [Moles/volume]",
                        "Iron [Mass/volume]": "Iron [Mass/volume]",
                        # NOTE: rename for consistency
                        "Anion gap 4": "Anion gap",
                        "Fractional oxyhemoglobin": "Oxyhemoglobin/Hemoglobin.total",
                        # NOTE: fixing wrong unit
                        "Thyroxine (T4) free [Mass/volume]": "Thyroxine (T4) free [Moles/volume]",
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
                itemcol="Band form neutrophils [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Band form neutrophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Basophils [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Basophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Eosinophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Lymphocytes/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Monocytes [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Monocytes/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Neutrophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Neutrophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes [#/volume]",
                total_itemcol="Erythrocytes [#/volume]",
                goal_itemcol="Reticulocytes/100 erythrocytes",
            )
        )


# endregion
