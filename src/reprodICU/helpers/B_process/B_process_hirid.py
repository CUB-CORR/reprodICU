# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import os
import sys
import time

import polars as pl

from ..A_extract.A_extract_hirid import HiRIDExtractor
from ..helper import GlobalHelpers
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
    def process_timeseries(self) -> pl.LazyFrame:
        """
        Process non-laboratory time series data.

        Steps:
            1. Check for preprocessed non-lab timeseries file; load if available.
            2. For each raw timeseries file: filter to non-lab variables, extract and pivot measurements.
            3. Combine all files into single wide-format dataset.
            4. Sort by index columns and save.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Non-laboratory measurement columns (pivoted from variable).
        """
        ts_path = self.precalc_path + "HiRID_timeseries.parquet"

        if os.path.isfile(ts_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_path, parallel="prefiltered").select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Processing timeseries data...")

        # Create an empty DataFrame to store the timeseries data
        timeseries_processed = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        os_listdir_files = os.listdir(self.timeseries_path)
        counter, counter_max, cases, times = 0, len(os_listdir_files), 0, []
        for file in os.listdir(self.timeseries_path):
            start = time.time()
            # Process timeseries data
            timeseries = (
                pl.scan_parquet(
                    self.timeseries_path + file, parallel="prefiltered"
                )
                # Drop the lab values from the timeseries data
                .filter(~pl.col("variableid").is_between(20000000, 25000000))
                .pipe(
                    self._extract_timeseries_helper,
                    self.admissiontime,
                    self.length_of_stay,
                )
                # Pivot the timeseries data
                .collect()
                .pivot(
                    on="variable",
                    index=self.index_cols,
                    values="value",
                    aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
                )
                .sort(self.index_cols)
            )

            # Append the data to the DataFrame
            timeseries_processed = pl.concat(
                [timeseries_processed, timeseries.lazy()],
                how="diagonal_relaxed",
            )

            # Update the counter and timings
            elapsed = time.time() - start
            times.append(elapsed)
            avg = sum(times) / len(times)
            eta_min = int(avg * (counter_max - counter) / 60 + 0.5)
            counter += 1

            sys.stdout.write("\033[K")  # Clear to the end of line
            cases += timeseries.select(self.icu_stay_id_col).unique().shape[0]
            print(
                f"Processing file {file}... \t{counter:3.0f} / {counter_max:3.0f} ({cases:5.0f} cases)"
                f" (last: {elapsed:.2f}s, avg: {avg:.2f}s, ETA: {eta_min:d} min)",
                end="\r",
            )

        # Save the preprocessed data
        timeseries_processed.sink_parquet(ts_path)

        # Load the preprocessed data
        return pl.scan_parquet(ts_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region timeseries labs
    def process_timeseries_labs(self) -> pl.LazyFrame:
        """
        Process laboratory time series data.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. For each raw timeseries file: filter to lab variables, extract measurements.
            3. Combine all files and map to LOINC components.
            4. Convert lab values to canonical units.
            5. Apply LOINC component mapping and JSON encode structured fields.
            6. Pivot data on variable name to create wide-format dataset.
            7. Apply post-pivot unit conversions for derived measurements.
            8. Save, sort by index columns.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from variable, JSON-encoded).
        """
        ts_labs_path = self.precalc_path + "HiRID_timeseries_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Processing lab data...")

        # Create an empty DataFrame to store the timeseries data
        timeseries_labs_filtered = pl.LazyFrame()
        labname_components = self._get_labname_components()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        os_listdir_files = os.listdir(self.timeseries_path)
        counter, counter_max, cases, times = 0, len(os_listdir_files), 0, []
        for file in os.listdir(self.timeseries_path):
            start = time.time()
            # Process timeseries data
            timeseries = (
                pl.scan_parquet(
                    self.timeseries_path + file, parallel="prefiltered"
                )
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
                .collect()
            )

            timeseries_labs_filtered = pl.concat(
                [timeseries_labs_filtered, timeseries.lazy()],
                how="diagonal_relaxed",
            )

            # Update the counter and timings
            elapsed = time.time() - start
            times.append(elapsed)
            avg = sum(times) / len(times)
            eta_min = int(avg * (counter_max - counter) / 60 + 0.5)
            counter += 1

            sys.stdout.write("\033[K")  # Clear to the end of line
            cases += timeseries.select(self.icu_stay_id_col).unique().shape[0]
            print(
                f"Processing file {file}... \t{counter:3.0f} / {counter_max:3.0f} ({cases:5.0f} cases)"
                f" (last: {elapsed:.2f}s, avg: {avg:.2f}s, ETA: {eta_min:d} min)",
                end="\r",
            )

        # Process the timeseries labs data all at once
        timeseries_labs_processed = (
            timeseries_labs_filtered.pipe(self._extract_timeseries_labs_helper)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="variable",
                valuecol="labstruct",
            )
            # Divide by 100 for percentage conversion
            .pipe(
                self.convert._divide_by_100,
                labelcol="variable",
                valuecol="labstruct",
                structfield="value",
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
            # Pivot the timeseries data
            .collect()
            .pivot(
                on="variable",
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
                struct_cols=["Lymphocytes/leukocytes"],
            )
            .sort(self.index_cols)
            .lazy()
        )

        # Save the preprocessed data
        timeseries_labs_processed.sink_parquet(ts_labs_path)

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

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Convert wide-format lab values to relative percentages.

        Transforms absolute counts to percentages for differential cell counts
        (lymphocytes).

        Returns:
            pl.LazyFrame: Lab data with calculated percentage values.
        """
        return data.pipe(
            self.convert_absolute_count_to_relative,
            itemcol="Lymphocytes",
            total_itemcol="Leukocytes",
            goal_itemcol="Lymphocytes/leukocytes",
            structfield="value",
            structstring=True,
        )

    def _divide_by_100(
        self,
        data: pl.LazyFrame,
        labelcol: str = "variableid",
        valuecol: str = "value_struct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """Divide specified lab values by 100 for percentage conversion.

        Steps:
            1. Divide struct field by 100 for items containing "/100".
            2. Update item labels to remove "100" prefix.

        Returns:
            pl.LazyFrame: Lab data with adjusted values and updated labels.
        """
        print("HiRID   - Dividing lab values by 100...")

        items_to_divide = [
            "Lymphocytes/100 leukocytes",
        ]

        return data.with_columns(
            pl.when(pl.col(labelcol).is_in(items_to_divide))
            .then(
                pl.col(valuecol).struct.with_fields(
                    structfield=pl.col(valuecol)
                    .struct.field(structfield)
                    .truediv(100)
                )
            )
            .otherwise(pl.col(valuecol))
            .alias(valuecol),
            pl.when(pl.col(labelcol).is_in(items_to_divide))
            .then(pl.col(labelcol).str.replace("/100 ", "/"))
            .otherwise(pl.col(labelcol))
            .alias(labelcol),
        )


# endregion
