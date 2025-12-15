# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl

from ..A_extract.A_extract_hirid import HiRIDExtractor
from ..helper import GlobalHelpers
from ..helper_batch import batch_process_timeseries
from ..helper_conversions import UnitConverter


class HiRIDProcessor(HiRIDExtractor):
    def __init__(self, paths):
        """
        Initializes the HiRIDProcessor instance.

        Args:
            paths: Object containing various source and destination paths.

        Sets:
            path: Source path for HiRID data ({hirid_source_path}).
            helpers: Instance of GlobalHelpers.
            convert: Instance of HiRIDConverter.
            index_cols: List containing {icu_stay_id_col} and {timeseries_time_col}.
        """
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.convert = HiRIDConverter()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]
        self.admissiontime = (
            self._extract_admissions()
            .select(self.icu_stay_id_col, "admissiontime")
            .cast({"admissiontime": str})
        )
        self.length_of_stay = self._extract_length_of_stay()

    def _get_labname_components(self) -> pl.LazyFrame:
        """
        Retrieves the lab names from the HiRID dataset.

        Returns:
            pl.LazyFrame: LazyFrame containing the LOINC data.
        """
        data = (
            pl.scan_parquet(self.timeseries_path, parallel="prefiltered")
            .select("variableid")
            .unique()
            .join(self._get_observation_variables(), on="variableid")
            .select("variable")
        )

        # Remove duplicates and sort the LOINC data
        labnames = data.collect().to_series().to_list()
        return data.with_columns(
            pl.col("variable")
            .replace_strict(
                self.omop.get_lab_component_from_name(labnames),
                default=None,
            )
            .alias("LOINC_component")
        )

    # region time series
    def _pivot_timeseries_batch(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Helper to pivot timeseries batches during batch processing."""
        timeseries = data.collect().pivot(
            on="variable",
            index=self.index_cols,
            values="value",
            aggregate_function="mean",
        )

        # Drop empty rows
        droplist = list(set(timeseries.columns) - set(self.index_cols))
        return (
            timeseries.pipe(self.helpers.dropna, "all", droplist, False)
            .unique(self.index_cols)
            .sort(self.index_cols)
            .lazy()
        )

    def process_timeseries(self) -> pl.LazyFrame:
        """
        Process non-laboratory time series data.

        Steps:
            1. Check for preprocessed non-lab timeseries file; load if available.
            2. Scan all raw timeseries files: filter to non-lab variables, extract measurements.
            3. Save extracted data to temporary file before batch pivoting.
            4. Batch pivot data on variable name to create wide-format dataset.
            5. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Non-laboratory measurement columns (pivoted from variable).
        """
        ts_path = self.precalc_path + "HiRID_timeseries.parquet"
        ts_path_unsorted = self.precalc_path + "HiRID_ts.parquet"

        if os.path.isfile(ts_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_path, parallel="prefiltered").select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Processing timeseries data...")

        # Process timeseries data
        (
            pl.scan_parquet(
                self.timeseries_path + "*.parquet", parallel="prefiltered"
            )
            # Drop the lab values from the timeseries data
            .filter(~pl.col("variableid").is_between(20000000, 25000000)).pipe(
                self._extract_timeseries_helper,
                self.admissiontime,
                self.length_of_stay,
            )
            # Save extracted data before pivoting
            .sink_parquet(ts_path_unsorted)
        )

        # Batch pivot the data
        batch_process_timeseries(
            input_file=ts_path_unsorted,
            output_file=ts_path,
            tempfiles_path=self.precalc_path,
            operation="pivot",
            method=self._pivot_timeseries_batch,
            id_col=self.icu_stay_id_col,
            delete_after=True,
        )
        os.remove(ts_path_unsorted)

        # Load the preprocessed data
        return pl.scan_parquet(ts_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    def _pivot_timeseries_labs_batch(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Helper to pivot labs batches during batch processing."""
        timeseries = (
            data.collect()
            .pivot(
                on="variable",
                index=self.index_cols,
                values="labstruct",
                aggregate_function="first",
            )
            .unique(self.index_cols)
            .sort(self.index_cols)
            .lazy()
        )
        return timeseries

    # endregion

    # region timeseries labs
    def process_timeseries_labs(self) -> pl.LazyFrame:
        """
        Process laboratory time series data.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. For each raw timeseries file: filter to lab variables, extract measurements.
            3. Combine all files, convert lab values to canonical units.
            4. Apply LOINC component mapping and JSON encode structured fields.
            5. Save extracted data to temporary file before batch pivoting.
            6. Batch pivot data on variable name to create wide-format dataset.
            7. Apply post-pivot unit conversions for derived measurements.
            8. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from variable, JSON-encoded).
        """
        ts_labs_path = self.precalc_path + "HiRID_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "HiRID_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Collecting lab time series data...")
        labname_components = self._get_labname_components()

        # Process timeseries data
        (
            pl.scan_parquet(self.timeseries_path + "*.parquet")
            # Keep the lab values from the timeseries data
            .filter(pl.col("variableid").is_between(20000000, 25000000))
            .pipe(
                self._extract_timeseries_helper,
                self.admissiontime,
                self.length_of_stay,
            )
            # Drop the non-lab values from the timeseries data
            .join(labname_components, on="variable")
            .filter(pl.col("LOINC_component").is_not_null())
            # Extract and transform lab data
            .pipe(self._extract_timeseries_labs_helper)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="variable",
                valuecol="labstruct",
            )
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="variable",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Save extracted data before pivoting
            .sink_parquet(ts_labs_path_unsorted)
        )

        # Batch pivot the data
        batch_process_timeseries(
            input_file=ts_labs_path_unsorted,
            output_file=ts_labs_path,
            tempfiles_path=self.precalc_path,
            operation="pivot",
            method=self._pivot_timeseries_labs_batch,
            id_col=self.icu_stay_id_col,
            delete_after=True,
        )
        os.remove(ts_labs_path_unsorted)

        # Load the preprocessed data
        return pl.scan_parquet(ts_labs_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion


# region convert
class HiRIDConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "variableid",
        valuecol: str = "value_struct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """Convert raw lab values to canonical units.

        Applies sequential unit conversions for multiple lab tests including:
        bilirubin, creatinine, cortisol, fibrinogen, glucose, hemoglobin, and urea.

        Expected columns:
            - {labelcol}: Lab test identifier.
            - {valuecol}: Lab measurement value.

        Returns:
            pl.LazyFrame: Lab data with unit-converted values.
        """
        return (
            data.pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.direct",
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
                self.convert_creatinine_umol_L_to_mg_dL,
                itemid="Creatinine",
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
                self.convert_g_L_to_mg_dL,
                itemid="Fibrinogen",
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
                self.convert_g_L_to_g_dL,
                itemid="Hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_g_L_to_g_dL,
                itemid="Erythrocyte mean corpuscular hemoglobin concentration",
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

# endregion
