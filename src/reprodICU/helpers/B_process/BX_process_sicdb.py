# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the SICdb data and stores it in a structured format for further
# processing and harmonization.


import os
import shutil
import sys

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
        ts_float_path_cache = self.precalc_path + "SICdb_ts_cache/"

        if os.path.isfile(ts_float_path):
            # Load the preprocessed data
            return pl.scan_parquet(
                ts_float_path, parallel="prefiltered"
            ).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("SICdb   - Collecting time series data...")

        # "Cache" the data before pivoting
        if not os.path.isdir(ts_float_path_cache):
            self.partition_timeseries(ts_float_path_cache)

        print("SICdb   - Processing numeric time series data...")

        # Create an empty DataFrame to store the timeseries data
        timeseries_processed = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        os_listdir_files = os.listdir(ts_float_path_cache)
        counter, counter_max, cases = 0, len(os_listdir_files), 0

        for file in os.listdir(ts_float_path_cache):
            # Update the counter
            counter += 1
            sys.stdout.write("\033[K")  # Clear to the end of line
            print(
                f"Processing file {file}... \t{counter:3.0f} / {counter_max:3.0f} ({cases:5.0f} cases)",
                end="\r",
            )

            # Process timeseries data
            timeseries = pl.scan_parquet(ts_float_path_cache + file).pipe(
                self._extract_timeseries_helper
            )
            cases += (
                timeseries.select(self.icu_stay_id_col)
                .unique()
                .collect()
                .shape[0]
            )

            # Pivot the timeseries data
            timeseries = timeseries.collect().pivot(
                on="DataID",
                index=self.index_cols,
                values="Val",
                aggregate_function="first",  # NOTE: first is used here to allow for string values
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

            # Append the processed timeseries data
            timeseries_processed = pl.concat(
                [timeseries_processed, timeseries], how="diagonal_relaxed"
            )

        # Save the preprocessed data
        timeseries_processed.sink_parquet(ts_float_path)
        shutil.rmtree(ts_float_path_cache, ignore_errors=True)

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
        Aligns lab unit measurements for {Protein} in SICdb.

        Returns:
            pl.LazyFrame: The LazyFrame with the units for {Protein} adjusted.
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
