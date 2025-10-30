# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import os
import sys
import time

import polars as pl
from helpers.A_extract.AX_extract_hirid import HiRIDExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


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
        Processes and combines non-laboratory time series data for HiRID.

        Steps:
          1. Check if the preprocessed non-lab parquet file exists in {precalc_path}.
          2. If available, load the LazyFrame with sorted index columns {icu_stay_id_col} and {timeseries_time_col}.
          3. If not, for each raw timeseries file:
             a. Use pre-extracted admissions and length of stay data ({admissiontime}, {length_of_stay}).
             b. Process timeseries data with _extract_timeseries_helper.
             c. Separate lab measurements by mapping "variable" to "LOINC_component" and DROP rows where "LOINC_component" is not null (keep non-lab only).
             d. Pivot the remaining (non-lab) data to wide format (on "variable").
             e. Concatenate results into a single LazyFrame.
          4. Save and return the non-lab timeseries sorted by the index.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time (as string) after conversion.
          - Additional columns: Pivoted non-lab measurement variables from raw files.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with non-lab measurements.
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
        Processes laboratory time series data for HiRID.

        Steps:
          1. Check if a preprocessed lab parquet file exists in {precalc_path}; load it if available.
          2. Otherwise, for each raw timeseries file:
             a. Use pre-extracted admissions and length of stay data ({admissiontime}, {length_of_stay}).
             b. Process timeseries data with _extract_timeseries_helper.
             c. Identify lab measurements by mapping "variable" to "LOINC_component" and KEEP rows where "LOINC_component" is not null.
             d. Convert to structured lab rows via _extract_timeseries_labs_helper and concatenate across files.
          3. Convert lab values to canonical units via _convert_lab_values.
          4. Assign LOINC codes pre-pivot using _assign_LOINC_codes (component_col="variable"); JSON-encode "labstruct".
          5. Pivot on "variable" to create a wide-format dataset (values="labstruct").
          6. Apply wide-format adjustments via _convert_wide_lab_values.
          7. (Optional) Assign LOINC codes again to derived columns (e.g., "Lymphocytes/100 leukocytes").
          8. Save and return the lab timeseries sorted by {icu_stay_id_col} and {timeseries_time_col}.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time (as string) after conversion.
          - Additional columns: JSON-encoded "labstruct" per lab variable.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame of lab measurements.
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
                struct_cols=["Lymphocytes/100 leukocytes"],
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
        """
        Convert laboratory measurement values to canonical units for the HiRID dataset.

        Applies a series of conversion functions sequentially to the input lab values. The conversion is performed
        for the following lab tests:
          - {Bilirubin.direct}
          - {Bilirubin}
          - {Creatinine}
          - {Cortisol}
          - {Fibrinogen}
          - {Glucose}
          - {Hemoglobin}
          - {Erythrocyte mean corpuscular hemoglobin concentration}
          - {Urea} to {Urea nitrogen} conversions

        Each function converts the lab value from an original unit to a canonical unit. The input data must include
        a column with the lab label (default: {labelcol}) and lab value stored in the field specified by {structfield}.

        Args:
            data (pl.LazyFrame): Input lab data containing lab values.
            labelcol (str, optional): Name of the column with lab identifiers. Defaults to "variableid".
            valuecol (str, optional): Name of the column containing lab values or structured lab data. Defaults to "value_struct".
            structfield (str, optional): Field within the structured lab data to extract for conversion. Defaults to "value".

        Returns:
            pl.LazyFrame: The input LazyFrame with lab values converted to canonical units.
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
        """
        Convert wide-format lab values to relative units for the HiRID dataset.

        Specifically, this method converts absolute lab counts into relative values. For example, it converts
        the absolute value of {Lymphocytes} into a relative value per 100 {Leukocytes}. The conversion is applied
        to the following columns (if available):
          - {Lymphocytes} relative to {Leukocytes}.

        Args:
            data (pl.LazyFrame): Lab data in wide format after pivoting.

        Returns:
            pl.LazyFrame: A LazyFrame with the applicable lab columns converted to relative units.
        """
        return data.pipe(
            self.convert_absolute_count_to_relative,
            itemcol="Lymphocytes",
            total_itemcol="Leukocytes",
            goal_itemcol="Lymphocytes/100 leukocytes",
            structfield="value",
            structstring=True,
        )


# endregion
