# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the UMCdb data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl

from ..A_extract.A_extract_umcdb import UMCdbExtractor
from ..helper import GlobalHelpers
from ..helper_batch import batch_process_timeseries
from ..helper_conversions import UnitConverter


class UMCdbProcessor(UMCdbExtractor):
    def __init__(self, paths):
        """
        Initializes the UMCdbProcessor instance.

        Args:
            paths: An object containing various source and destination paths.

        Sets:
            self.path: Source path for UMCdb data ({umcdb_source_path}).
            self.helpers: Instance of GlobalHelpers.
            self.convert: Instance of UMCdbConverter.
            self.icu_stay_id: LazyFrame with columns {icu_stay_id_col}, {hospital_stay_id_col}, and {person_id_col}.
            self.icu_length_of_stay: LazyFrame with columns {icu_stay_id_col} and {icu_length_of_stay_col}.
            self.index_cols: List containing {icu_stay_id_col} and {timeseries_time_col}.
        """
        super().__init__(paths)
        self.path = paths.umcdb_source_path
        self.helpers = GlobalHelpers()
        self.convert = UMCdbConverter()
        self.icu_stay_id = self.extract_patient_information().select(
            self.icu_stay_id_col,
            self.hospital_stay_id_col,
            self.person_id_col,
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region time series
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self):
        """
        Process and combine all time series data.

        Steps:
            1. Check for preprocessed combined timeseries file; load if available.
            2. Extract numeric timeseries data.
            3. Extract listitems (categorical) timeseries data.
            4. Join both datasets on index columns.
            5. Save sorted combined data and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Numeric and categorical measurement columns.
        """
        ts_path = self.precalc_path + "UMCdb_timeseries.parquet"
        ts_path_unsorted = self.precalc_path + "UMCdb_ts.parquet"

        # Load preexisting data if available
        if os.path.isfile(ts_path):
            return pl.scan_parquet(ts_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        # Load the time series data.
        print("UMCdb   - Loading time series data...")

        ts_numeric = self._process_timeseries_numeric()
        ts_listitems = self._process_timeseries_listitems()

        # Save the preprocessed data
        (
            ts_numeric.join(
                ts_listitems, on=self.index_cols, how="full", coalesce=True
            ).sink_parquet(ts_path_unsorted)
        )

        # Sort the data
        (
            pl.scan_parquet(ts_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_path)
        )
        os.remove(ts_path_unsorted)

        return pl.scan_parquet(ts_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region numeric
    def _process_timeseries_numeric_batch(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Helper method to process numeric timeseries data for batch processing.

        Steps:
            1. Pivot measurements on "item" using mean aggregation.
            2. Return processed timeseries.

        Returns:
            pl.LazyFrame: Processed timeseries with pivoted numeric measurements.
        """
        return (
            data.collect()
            .pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
            .lazy()
        )

    def _process_timeseries_numeric(self) -> pl.LazyFrame:
        """
        Process numeric time series measurements.

        Steps:
            1. Check for preprocessed numeric file; load if available.
            2. Extract numeric measurements from {extract_timeseries_numericitems}.
            3. Process in batches of 500 patients: pivot measurements on "item".
            4. Combine all batches and save sorted result.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Numeric measurement columns (pivoted from item).
        """
        ts_numeric_path = self.precalc_path + "UMCdb_timeseries_numeric.parquet"
        ts_numeric_path_unsorted = self.precalc_path + "UMCdb_ts_numeric.parquet" # fmt: skip

        if os.path.isfile(ts_numeric_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_numeric_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Preparing numeric time series data...")

        # Create the raw numeric timeseries parquet file if it doesn't exist
        if not os.path.isfile(ts_numeric_path_unsorted):
            self.extract_timeseries_numericitems().sink_parquet(
                ts_numeric_path_unsorted
            )

        print("UMCdb   - Processing numeric time series data...")

        # Process in batches using batch_process_timeseries
        batch_process_timeseries(
            input_file=ts_numeric_path_unsorted,
            output_file=ts_numeric_path,
            tempfiles_path=self.precalc_path,
            operation="process",
            method=self._process_timeseries_numeric_batch,
            id_col=self.icu_stay_id_col,
            batch_size=500,
            delete_after=True,
        )

        # Clean up unsorted file
        if os.path.isfile(ts_numeric_path_unsorted):
            os.remove(ts_numeric_path_unsorted)

        return pl.scan_parquet(ts_numeric_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region labs
    def _process_timeseries_labs(self) -> pl.LazyFrame:
        """
        Process laboratory time series measurements.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. Extract lab measurements and cache.
            3. Align unit representations for specific analytes.
            4. Convert lab values to canonical units.
            5. Apply LOINC component mapping and JSON encode structured fields.
            6. Pivot data on "item" to create wide-format dataset.
            7. Apply post-pivot unit conversions and percentage calculations.
            8. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from item, JSON-encoded).
        """
        ts_labs_path = self.precalc_path + "UMCdb_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "UMCdb_ts_labs.parquet"
        ts_labs_path_cache = self.precalc_path + "UMCdb_ts_labs_cache.parquet"

        if os.path.isfile(ts_labs_path):
            # load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Collecting lab time series data...")

        # "Cache" the data before pivoting
        if not os.path.isfile(ts_labs_path_cache):
            (
                self.extract_timeseries_labs()
                .collect()
                .write_parquet(ts_labs_path_cache)
            )

        print("UMCdb   - Processing lab time series data...")

        # Process labs data
        ts_labs = (
            pl.scan_parquet(ts_labs_path_cache)
            # Align the units of the lab values
            .pipe(self.convert._align_units)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="item",
                valuecol="labstruct",
                structfield="value",
            )
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="item",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the labs data
            .collect()
            .pivot(
                on="item",
                index=self.index_cols,
                values="labstruct",
                aggregate_function="first",
            )
            # Convert the wide lab values to the correct units
            .pipe(self.convert._convert_wide_lab_values)
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=[
                    "Basophils/100 leukocytes",
                    "Eosinophils/100 leukocytes",
                    "Lymphocytes/100 leukocytes",
                    "Monocytes/100 leukocytes",
                    "Neutrophils/100 leukocytes",
                    "Neutrophils.band form/100 leukocytes",
                    "Neutrophils.segmented/100 leukocytes",
                    "Reticulocytes/100 erythrocytes",
                ],
            )
            .lazy()
        )

        # Save the preprocessed data
        # ts_labs.sink_parquet(ts_labs_path_unsorted)
        ts_labs.collect().write_parquet(ts_labs_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_labs_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_labs_path)
        )
        os.remove(ts_labs_path_unsorted)
        os.remove(ts_labs_path_cache)

        return pl.scan_parquet(ts_labs_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region listitems
    def _process_timeseries_listitems(self) -> pl.LazyFrame:
        """
        Process listitem (categorical) time series measurements.

        Steps:
            1. Check for preprocessed listitems file; load if available.
            2. Extract listitem measurements and cache.
            3. Pivot data on "item" using first aggregation for wide-format.
            4. Drop rows where all non-index columns are null.
            5. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Categorical measurement columns (pivoted from item).
        """
        ts_list_path = self.precalc_path + "UMCdb_timeseries_list.parquet"
        ts_list_path_unsorted = self.precalc_path + "UMCdb_ts_list.parquet"
        ts_list_path_cache = self.precalc_path + "UMCdb_ts_list_cache.parquet"

        if os.path.isfile(ts_list_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_list_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Collecting list time series data...")

        # "Cache" the data before pivoting
        if not os.path.isfile(ts_list_path_cache):
            self.extract_timeseries_listitems().sink_parquet(ts_list_path_cache)

        print("UMCdb   - Processing list time series data...")

        # Process list data
        ts_listitems = (
            pl.scan_parquet(ts_list_path_cache)
            # Pivot the list data
            .collect().pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="first",
            )
        )

        # Drop empty rows
        droplist = list(
            set(ts_listitems.collect_schema().names()) - set(self.index_cols)
        )
        ts_listitems = (
            ts_listitems.pipe(self.helpers.dropna, "all", droplist, False)
            .lazy()
            .unique()
        )

        # Save the preprocessed data
        ts_listitems.sink_parquet(ts_list_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_list_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_list_path)
        )
        os.remove(ts_list_path_unsorted)
        os.remove(ts_list_path_cache)

        return pl.scan_parquet(ts_list_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion


# region convert
class UMCdbConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "variableid",
        valuecol: str = "value_struct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """Convert raw lab values to canonical units.

        Applies sequential unit conversions for multiple lab tests including:
        ratios to percentages, bilirubin, creatinine, cholesterol, hormones,
        fibrinogen, glucose, hemoglobin, triglycerides, and urea.

        Expected columns:
            - {labelcol}: Lab test identifier.
            - {valuecol}: Lab measurement value.

        Returns:
            pl.LazyFrame: Lab data with unit-converted values.
        """

        print("UMCdb   - Converting lab values...")

        # Convert the lab values to the correct units.
        return (
            data.pipe(
                self.convert_ratio_to_percentage,
                itemid="Hematocrit",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ratio_to_percentage,
                itemid="Oxygen saturation",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.conjugated",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_creatinine_mmol_L_to_mg_dL,
                itemid="Creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol in HDL",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cortisol_nmol_L_to_ug_dL,
                itemid="Cortisol",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_CKMB_ng_mL_to_U_L,
                itemid="Creatine kinase.MB",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_FEU_to_DDU,
                itemid="Fibrin D-dimer FEU",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="Fibrinogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_folate_nmol_L_to_ng_mL,
                itemid="Folate",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="Glucose",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="Hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_mg_L_to_mg_dL,
                itemid="Microalbumin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="Erythrocyte mean corpuscular hemoglobin concentration",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_triglycerides_mmol_L_to_mg_dL,
                itemid="Triglyceride",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ug_L_to_ng_L,
                itemid="Troponin T.cardiac",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urate_umol_L_to_mg_dL,
                itemid="Urate",
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
            .pipe(
                self.convert_blood_urea_nitrogen_mmol_L_to_mg_dL,
                itemid="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
        )

    def _align_units(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Align lab unit representations for creatinine and reticulocytes.
        Converts creatinine from umol/L to mmol/L and reticulocytes from
        percentage to absolute counts (10^12/L).

        Returns:
            pl.LazyFrame: Lab data with aligned unit values.
        """

        print("UMCdb   - Aligning lab value units...")

        # some paO2 / paCO2 values are given in kPa, convert to mmHg for consistency
        # Creatinine in Serum or Plasma is in umol/L, convert to mmol/L for consistency
        # Reticulocytes are given in 10^9/L (percentage), convert to 10^12/L for consistency

        return (
            data.unnest("labstruct")
            .with_columns(
                pl.when(pl.col("itemid").is_in([21213, 21214]))
                .then(pl.col("value").mul(7.50061683))
                .when(
                    pl.col("item") == "Creatinine",
                    pl.col("system") == "Serum or Plasma",
                )
                .then(pl.col("value").truediv(1000))
                .when(pl.col("item") == "Reticulocytes")
                .then(pl.col("value").truediv(1000))
                .otherwise(pl.col("value"))
                .alias("value")
            )
            .select(
                pl.exclude("value", "system", "method", "time", "LOINC"),
                pl.struct(
                    value="value",
                    system="system",
                    method="method",
                    time="time",
                    LOINC="LOINC",
                ).alias("labstruct"),
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Convert wide-format lab values to relative percentages.

        Transforms absolute counts to percentages for differential cell counts
        (basophils, eosinophils, lymphocytes, monocytes, neutrophils,
        band form neutrophils, segmented neutrophils, reticulocytes).

        Returns:
            pl.LazyFrame: Lab data with calculated percentage values.
        """

        print("UMCdb   - Converting wide lab values...")

        return (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Basophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Basophils/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Eosinophils/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes",
                total_itemcol="Leukocytes",
                goal_itemcol="Lymphocytes/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Monocytes",
                total_itemcol="Leukocytes",
                goal_itemcol="Monocytes/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Band form neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils.band form/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Segmented neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils.segmented/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes",
                total_itemcol="Erythrocytes",
                goal_itemcol="Reticulocytes/100 erythrocytes",
                structfield="value",
                structstring=True,
            )
        )


# endregion
