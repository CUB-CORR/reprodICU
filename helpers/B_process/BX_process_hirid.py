# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import os
import sys

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

    # region time series
    def process_timeseries(self) -> pl.LazyFrame:
        """
        Processes and combines time series data for HiRID.

        Steps:
          1. Check if preprocessed main and lab parquet files exist in {precalc_path}.
          2. If available, load both LazyFrames (main and lab data) with sorted index columns {icu_stay_id_col} and {timeseries_time_col}.
          3. If not, for each raw timeseries file:
             a. Extract admissions and length of stay data.
             b. Process timeseries data with _extract_timeseries_helper.
             c. Separate lab measurements by mapping "variable" to "LOINC_component".
             d. Pivot main and lab datasets separately.
             e. Concatenate results into two LazyFrames.
          4. Save and return a tuple of (main timeseries, lab timeseries) sorted by the index.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time (as string) after conversion.
          - Additional columns: Pivoted measurement variables from raw files.

        Returns:
            tuple: Two LazyFrames (main, lab) sorted by [{icu_stay_id_col}, {timeseries_time_col}].
        """
        ts_path = self.precalc_path + "HiRID_timeseries.parquet"
        ts_labs_path = self.precalc_path + "HiRID_timeseries_labs.parquet"

        if os.path.isfile(ts_path) and os.path.isfile(ts_labs_path):
            # Load the preprocessed data
            return (
                pl.scan_parquet(ts_path).select(
                    pl.col(self.index_cols).set_sorted(),
                    pl.exclude(self.index_cols),
                ),
                pl.scan_parquet(ts_labs_path).select(
                    pl.col(self.index_cols).set_sorted(),
                    pl.exclude(self.index_cols),
                ),
            )

        print("HiRID   - Processing time series data...")

        admissiontime = (
            self._extract_admissions()
            .select(self.icu_stay_id_col, "admissiontime")
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        # Create an empty DataFrame to store the timeseries data
        timeseries_processed = pl.LazyFrame()
        timeseries_labs_processed = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        os_listdir_files = os.listdir(self.timeseries_path)
        counter, counter_max, cases = 0, len(os_listdir_files), 0
        for file in os.listdir(self.timeseries_path):
            # Update the counter
            counter += 1
            sys.stdout.write("\033[K")  # Clear to the end of line
            print(
                f"Processing file {file}... \t{counter:3.0f} / {counter_max:3.0f} ({cases:5.0f} cases)",
                end="\r",
            )

            # Process timeseries data
            timeseries = pl.scan_parquet(self.timeseries_path + file).pipe(
                self._extract_timeseries_helper, admissiontime, length_of_stay
            )
            cases += (
                timeseries.select(self.icu_stay_id_col)
                .unique()
                .collect()
                .shape[0]
            )

            # Separate the lab values from the rest
            LOINC_data = timeseries.select("variable").unique()
            labnames = LOINC_data.collect().to_series().to_list()
            LOINC_data = LOINC_data.with_columns(
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component")
            )
            timeseries_labs = (
                timeseries.join(LOINC_data, on="variable")
                .filter(
                    pl.col("LOINC_component").is_in(
                        self.relevant_lab_LOINC_components
                    )
                )
                .pipe(self._extract_timeseries_labs_helper)
                # Convert the lab values to the correct units
                .pipe(
                    self.convert._convert_lab_values,
                    labelcol="variable",
                    valuecol="labstruct",
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
            )

            timeseries_labs_columns = timeseries_labs.collect_schema().names()
            if ("Lymphocytes" in timeseries_labs_columns) and (
                "Leukocytes" in timeseries_labs_columns
            ):
                timeseries_labs = (
                    timeseries_labs
                    # Convert the wide lab values to the correct units
                    .pipe(self.convert._convert_wide_lab_values)
                )

            timeseries_labs = timeseries_labs.sort(self.index_cols).lazy()

            # Drop the lab values from the timeseries data
            timeseries = (
                timeseries.join(LOINC_data, on="variable")
                .filter(pl.col("LOINC_component").is_null())
                # Pivot the timeseries data
                .collect()
                .pivot(
                    on="variable",
                    index=self.index_cols,
                    values="value",
                    aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
                )
                .sort(self.index_cols)
                .lazy()
            )

            # Append the data to the DataFrame
            timeseries_processed = pl.concat(
                [timeseries_processed, timeseries],
                how="diagonal_relaxed",
            )
            timeseries_labs_processed = pl.concat(
                [timeseries_labs_processed, timeseries_labs],
                how="diagonal_relaxed",
            )

        # Save the preprocessed data
        timeseries_processed.sink_parquet(ts_path)
        timeseries_labs_processed.sink_parquet(ts_labs_path)

        # Load the preprocessed data
        return (
            pl.scan_parquet(ts_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            ),
            pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            ),
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
          - {Bilirubin.total}
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
                itemid="Bilirubin.total",
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
