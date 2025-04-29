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
        """
        Initializes the SICdbProcessor instance.

        Args:
            paths: Object containing source and destination paths.

        Sets:
            self.path: Source path for SICdb data ({sicdb_source_path}).
            self.helpers: Instance of GlobalHelpers.
            self.convert: Instance of SICdbConverter.
            self.icu_stay_id: LazyFrame with columns {icu_stay_id_col}, {hospital_stay_id_col}, and {person_id_col}.
            self.icu_length_of_stay: LazyFrame with columns {icu_stay_id_col} and {icu_length_of_stay_col}.
            self.index_cols: List containing {icu_stay_id_col} and {timeseries_time_col}.
        """
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
        Processes numerical time series data for SICdb.

        Steps:
          1. Check if a preprocessed numeric file exists in {precalc_path}.
          2. If it exists, load the data with sorted index columns.
          3. Otherwise, cache the raw data via extract_timeseries() and save to a temporary cache file.
          4. Pivot the cached data on "DataID" using the first-occurrence aggregation.
          5. Drop rows where all non-{icu_stay_id_col} and non-{timeseries_time_col} columns are null.
          6. Save the unsorted result, sort by {icu_stay_id_col} and {timeseries_time_col}, remove temporary files.
          7. Return the sorted data.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset (in seconds) from ICU admission.
          - Additional columns: Numeric measurements pivoted from "DataID" with values in "Val".

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with numerical data.
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
        Processes laboratory time series data for SICdb.

        Steps:
          1. Check if a preprocessed lab data file exists in {precalc_path}.
          2. If it exists, load the data with sorted index columns.
          3. Otherwise, extract raw lab data using extract_laboratory_timeseries().
          4. Convert lab values to canonical units using _convert_lab_values.
          5. JSON encode the "labstruct" field.
          6. Pivot the data on "LaboratoryName" using first-occurrence aggregation.
          7. Save the unsorted file, sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset (seconds) from ICU admission.
          - "LaboratoryName": Lab test name used as pivot key.
          - "labstruct": JSON-encoded lab result structure (including {value}).

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with laboratory measurements.
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

    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "LaboratoryID",
        valuecol: str = "LaboratoryValue",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Converts laboratory measurement values of SICdb to canonical units.

        Steps:
          1. Apply the conversion for {Cobalamin} using convert_VitB12_pg_mL_to_pmol_L.
          2. Convert {Iron} using convert_iron_ug_dL_to_umol_L.
          3. Convert {Urea} to {Urea nitrogen} using convert_urea_nitrogen_from_urea followed by convert_blood_urea_nitrogen_mmol_L_to_mg_dL.

        Expected input columns:
          - {labelcol}: Contains lab test identifiers.
          - {valuecol}: Contains measurement values or structured values with key {structfield}.

        Returns:
            pl.LazyFrame: Data with lab values converted.
        """
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
        Converts wide-format lab counts into relative percentages for SICdb.

        Steps:
          1. For each lab analyte ("Basophils", "Eosinophils", "Lymphocytes", "Monocytes", "Neutrophils", "Reticulocytes"),
             compute the relative count per 100 of total {Leukocytes} or {Erythrocytes}.

        Columns produced include:
          - "Eosinophils/100 leukocytes"
          - "Lymphocytes/100 leukocytes"
          - "Reticulocytes/100 erythrocytes"

        Returns:
            pl.LazyFrame: Data with selected lab count columns converted to relative values.
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
