# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the SICdb data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl

from ..A_extract.A_extract_sicdb import SICdbExtractor
from ..helper import GlobalHelpers
from ..helper_batch import batch_process_timeseries
from ..helper_conversions import UnitConverter


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
    def _process_timeseries_batch(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Helper method to process timeseries data for batch processing.

        Steps:
            1. Extract and align timeseries data relative to ICU admission.
            2. Pivot measurements on "DataID" using first aggregation.
            3. Drop rows where all non-index columns are null.

        Returns:
            pl.LazyFrame: Processed timeseries with pivoted measurements.
        """
        # Extract & pivot the timeseries data
        timeseries = (
            data.pipe(self._extract_timeseries_helper)
            .collect()
            .pivot(
                on="DataID",
                index=self.index_cols,
                values="Val",
                aggregate_function="first",  # NOTE: first is used here to allow for string values
            )
            .lazy()
        )

        # Drop empty rows
        droplist = list(
            set(timeseries.collect_schema().names()) - set(self.index_cols)
        )

        return timeseries.pipe(self.helpers.dropna, "all", droplist, False)

    def process_timeseries_data_float(self) -> pl.LazyFrame:
        """
        Process numerical time series data.

        Steps:
            1. Check for preprocessed numeric timeseries file; load if available.
            2. Extract raw timeseries data from {data_float_m_path}.
            3. Process in batches of 500 patients: extract, pivot, and filter measurements.
            4. Combine all batches and save sorted result.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Numeric measurement columns (pivoted from DataID).
        """
        ts_float_path = self.precalc_path + "SICdb_timeseries.parquet"
        ts_float_path_unsorted = self.precalc_path + "SICdb_ts_unsorted.parquet"

        if os.path.isfile(ts_float_path):
            # Load the preprocessed data
            return pl.scan_parquet(
                ts_float_path, parallel="prefiltered"
            ).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("SICdb   - Preparing raw timeseries data...")

        # Create the raw timeseries parquet file if it doesn't exist
        if not os.path.isfile(ts_float_path_unsorted):
            (
                pl.scan_parquet(self.data_float_m_path, parallel="prefiltered")
                .select("CaseID", "Offset", "DataID", "Val")
                # Round values to 2 decimal places due to precision issues of IEEE 754 floats
                .with_columns(pl.col("Val").cast(float).round(2))
                .rename({"CaseID": self.icu_stay_id_col})
                .sink_parquet(ts_float_path_unsorted)
            )

        print("SICdb   - Processing numeric time series data...")

        # Process in batches using batch_process_timeseries
        batch_process_timeseries(
            input_file=ts_float_path_unsorted,
            output_file=ts_float_path,
            tempfiles_path=self.precalc_path,
            operation="process",
            method=self._process_timeseries_batch,
            id_col=self.icu_stay_id_col,
            batch_size=500,
            delete_after=True,
        )

        # Clean up unsorted file
        if os.path.isfile(ts_float_path_unsorted):
            os.remove(ts_float_path_unsorted)

        return pl.scan_parquet(ts_float_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region lab values
    def process_timeseries_data_labs(self) -> pl.LazyFrame:
        """
        Process laboratory time series data.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. Extract lab measurements and align unit representations.
            3. Convert lab values to canonical units.
            4. Apply LOINC component mapping and JSON encode structured fields.
            5. Pivot data on "LaboratoryName" to create wide-format dataset.
            6. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from LaboratoryName, JSON-encoded).
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
            # Align the units of the lab values
            .pipe(self.convert._align_units)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="LaboratoryName",
                valuecol="labstruct",
            )
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="LaboratoryName",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the timeseries data
            .collect()
            .pivot(
                on="LaboratoryName",
                index=self.index_cols,
                values="labstruct",
                aggregate_function="first",
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
        """Convert raw lab values to canonical units.

        Applies sequential unit conversions for multiple lab tests including:
        cobalamin, iron, urea nitrogen, and thyroxine.

        Expected columns:
            - {labelcol}: Lab test identifier.
            - {valuecol}: Lab measurement value.

        Returns:
            pl.LazyFrame: Lab data with unit-converted values.
        """
        return (
            data.pipe(
                self.rename_anion_gap,  # "Anion gap 4" -> "Anion gap"
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="Cobalamin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="C reactive protein",
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
                self.convert_g_dL_to_g_L,
                itemid="Protein",
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
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="Thyroxine free",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
        )

    def _align_units(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Align lab unit representations for protein measurements.
        Converts protein values from mg/dL and mg/L to standardized units.

        Returns:
            pl.LazyFrame: Lab data with aligned unit values.
        """

        print("SICdb   - Aligning lab value units...")

        # 342 = Eiweiss im Liquor (ZL) -> Protein in Cerebral spinal fluid
        # 343 = Eiweiss im Urin (ZL) -> Protein in Urine
        # 406 = Gesamteiweiss im Urin (mg/l) (ZL) -> Protein in Urine

        return (
            data.unnest("labstruct")
            .with_columns(
                # 342 and 343 are in mg/dl, convert to g/dL
                pl.when(pl.col("LaboratoryID").is_in([342, 343]))
                .then(pl.col("value") * 1000)
                # 406 is in mg/L, convert to g/dL
                .when(pl.col("LaboratoryID") == 406)
                .then(pl.col("value") * 10000)
                .otherwise(pl.col("value"))
                .alias("value"),
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
        (basophils, eosinophils, lymphocytes, monocytes, neutrophils, reticulocytes).

        Returns:
            pl.LazyFrame: Lab data with calculated percentage values.
        """
        return (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Basophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Basophils/leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Eosinophils/leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes",
                total_itemcol="Leukocytes",
                goal_itemcol="Lymphocytes/leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Monocytes",
                total_itemcol="Leukocytes",
                goal_itemcol="Monocytes/leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils/leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Band form neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils.band form/leukocytes",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes",
                total_itemcol="Erythrocytes",
                goal_itemcol="Reticulocytes/Erythrocytes",
            )
        )


# endregion
