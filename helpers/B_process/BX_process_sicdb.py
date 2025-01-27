# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the SICdb data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl
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

    # region vitals
    def process_timeseries_data_float(self) -> pl.LazyFrame:
        """
        Processes the time series data of the SICdb dataset.
        """
        ts_float_path = self.precalc_path + "SICdb_timeseries.parquet"
        ts_float_path_unsorted = self.precalc_path + "SICdb_ts.parquet"
        ts_float_path_cache = self.precalc_path + "SICdb_ts_cache.parquet"

        if os.path.isfile(ts_float_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_float_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("SICdb   - Collecting time series data...")

        # "Cache" the data before pivoting
        if not os.path.isfile(ts_float_path_cache):
            (
                self.extract_timeseries()
                .collect()
                .write_parquet(ts_float_path_cache)
            )

        print("SICdb   - Processing numeric time series data...")

        # Process timeseries data
        timeseries = (
            pl.scan_parquet(ts_float_path_cache)
            # Pivot the timeseries data
            .collect().pivot(
                on="DataID",
                index=self.index_cols,
                values="Val",
                aggregate_function="first",  # NOTE: first is used here to allow for string values
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
        os.remove(ts_float_path_cache)

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
                labelcol="LaboratoryName",
                valuecol="labstruct",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the timeseries data
            .collect(streaming=True)
            .pivot(
                on="LaboratoryName",
                index=self.index_cols,
                values="labstruct",
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
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="Cobalamin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea",
                itemid_BUN="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # .with_columns(
            #     pl.col(labelcol).replace(
            #         {
            #             "Cobalamin (Vitamin B12) [Mass/volume]": "Cobalamin (Vitamin B12) [Moles/volume]",
            #             "Iron [Mass/volume]": "Iron [Moles/volume]",
            #             # NOTE: rename for consistency
            #             "Anion gap 4": "Anion gap",
            #             "Fractional oxyhemoglobin": "Oxyhemoglobin/Hemoglobin.total",
            #         }
            #     )
            # )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Convert the lab values of the SICdb dataset.
        """

        return (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Basophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Basophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Eosinophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes",
                total_itemcol="Leukocytes",
                goal_itemcol="Lymphocytes/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Monocytes",
                total_itemcol="Leukocytes",
                goal_itemcol="Monocytes/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Band form neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils.band form/100 leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes",
                total_itemcol="Erythrocytes",
                goal_itemcol="Reticulocytes/100 erythrocytes",
            )
        )


# endregion
