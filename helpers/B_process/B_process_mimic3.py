# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.

import os

import polars as pl
from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class MIMIC3Processor(MIMIC3Extractor):
    def __init__(self, paths, DEMO=False):
        """
        Initialize the MIMIC3Processor.

        Args:
            paths: Object containing file paths.
            DEMO (bool): If True, use demo mode parameters.

        Attributes:
            {icu_stay_id_col}: ICU stay identifier.
            {hospital_stay_id_col}: Hospital stay identifier.
            {person_id_col}: Patient identifier.
            {icu_length_of_stay_col}: ICU length of stay.
            index_cols (list): Index columns used for pivoting, specifically [{icu_stay_id_col}, {timeseries_time_col}].
        """
        super().__init__(paths, DEMO)
        self.path = paths.mimic3_source_path
        self.helpers = GlobalHelpers()
        self.convert = MIMIC3Converter()
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
    # Processes the vital data of the MIMIC3 dataset.
    def process_timeseries_vitals(self):
        """
        Processes vital and respiratory time series data for MIMIC-III.

        Steps:
          1. Check if a sorted parquet file exists in {precalc_path}.
          2. If it exists, load and return data with index columns ({icu_stay_id_col} and {timeseries_time_col}) sorted.
          3. Otherwise, extract chart events and:
             • Convert temperature from Fahrenheit to Celsius.
             • Convert fractions to percentages.
          4. Pivot the data using "LABEL" as the key.
          5. Drop rows where all non-index columns are null.
          6. Save the unsorted data, sort it, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset in seconds relative to ICU admission.
          - "LABEL": Pivot key for measurement types.
          - Other columns: Individual vital sign measurements.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with vital and respiratory measurements.
        """
        ts_vitals_path = self.precalc_path + "MIMIC3_timeseries_vitals.parquet"
        ts_vitals_path_unsorted = self.precalc_path + "MIMIC3_ts_vitals.parquet"

        if os.path.isfile(ts_vitals_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_vitals_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("MIMIC3  - Processing vitals & respiratory data...")

        # Process vitals data
        ts_vitals = (
            self.extract_chartevents()
            # Convert temperature from Fahrenheit to Celsius
            .pipe(
                self.convert.convert_temperature_F_to_C,
                itemid_F="Temperature Fahrenheit",
                itemid_C="Temperature",
                labelcol="LABEL",
                valuecol="VALUENUM",
            )
            # Convert fractions to percentages
            .pipe(
                self.convert.convert_ratio_to_percentage,
                itemid="Oxygen/Total gas setting [Volume Fraction] Ventilator",
                labelcol="LABEL",
                valuecol="VALUENUM",
            )
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="LABEL",
                index=self.index_cols,
                values="VALUENUM",
                aggregate_function="first",
            )
            # Replace the integerized values with the original values
            .with_columns(
                pl.col("Heart rate rhythm").replace_strict(
                    self.heart_rhythm_enum_map_inverted,
                    return_dtype=pl.String,
                ),
                pl.col("Oxygen delivery system").replace_strict(
                    self.oxygen_delivery_system_enum_map_inverted,
                    return_dtype=pl.String,
                ),
                pl.col("Ventilation mode Ventilator").replace_strict(
                    self.ventilator_mode_enum_map_inverted,
                    return_dtype=pl.String,
                ),
            )
        )

        print("MIMIC3  - Dropping empty rows...")

        # Drop empty rows
        ts_vitals_cols = ts_vitals.collect_schema().names()
        droplist = list(set(ts_vitals_cols) - set(self.index_cols))
        ts_vitals = (
            ts_vitals.lazy()
            .pipe(self.helpers.dropna, "all", droplist, False)
            .unique()
            .sort(self.index_cols)
        )

        # Save the preprocessed data
        ts_vitals.sink_parquet(ts_vitals_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_vitals_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_vitals_path)
        )
        os.remove(ts_vitals_path_unsorted)

        return pl.scan_parquet(ts_vitals_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region lab
    def process_timeseries_labevents(self):
        """
        Processes laboratory time series data for MIMIC-III.

        Steps:
          1. Check if a preprocessed lab file exists in {precalc_path}; load it if available.
          2. Otherwise, extract raw lab data using extract_lab_measurements().
          3. Convert lab values to canonical units via _convert_lab_values.
          4. JSON encode the "labstruct" field.
          5. Pivot on "LABEL" to create a wide-format dataset.
          6. Apply wide-format adjustments via _convert_wide_lab_values.
          7. Save, sort the data by {icu_stay_id_col} and {timeseries_time_col}, and remove any temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset (in seconds).
          - "LABEL": Lab test name used as pivot key.
          - "labstruct": JSON-encoded structured lab result (including {value}).

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame of lab measurements.
        """
        ts_labs_path = self.precalc_path + "MIMIC3_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "MIMIC3_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("MIMIC3  - Processing lab data...")

        # Process lab data
        ts_lab = (
            self.extract_lab_measurements()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="LABEL",
                valuecol="labstruct",
                structfield="value",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the lab data
            .collect()
            .pivot(
                on="LABEL",
                index=self.index_cols,
                values="labstruct",
                aggregate_function="first",
            )
            # Convert the wide lab values to the correct units
            .pipe(self.convert._convert_wide_lab_values)
            .lazy()
        )

        # Save the preprocessed data
        ts_lab.sink_parquet(ts_labs_path_unsorted)

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

    # region input/output
    def process_timeseries_inputoutput(self):
        """
        Processes input/output measurement data for MIMIC-III.

        Steps:
          1. Check for an existing preprocessed input/output file in {precalc_path}.
          2. If found, load it with sorted index columns.
          3. Otherwise, extract output measurements.
          4. Pivot the data using "LABEL" as key with mean aggregation.
          5. Drop rows where all non-index columns are null.
          6. Save the unsorted data, sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset (seconds).
          - "LABEL": Measurement label for input/output events.
          - "VALUENUM": Aggregated value (mean).

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame containing input/output data.
        """
        ts_inout_path = self.precalc_path + "MIMIC3_timeseries_inout.parquet"
        ts_inout_path_unsorted = self.precalc_path + "MIMIC3_ts_inout.parquet"

        if os.path.isfile(ts_inout_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_inout_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("MIMIC3  - Processing inout data...")

        # Process inout data
        ts_inout = (
            self.extract_output_measurements()
            # Pivot the inout data
            .collect(streaming=True).pivot(
                on="LABEL",
                index=self.index_cols,
                values="VALUENUM",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        ts_inout_cols = ts_inout.collect_schema().names()
        droplist = list(set(ts_inout_cols) - set(self.index_cols))
        ts_inout = (
            ts_inout.lazy()
            .pipe(self.helpers.dropna, "all", droplist, False)
            .unique()
            .sort(self.index_cols)
        )

        # Save the preprocessed data
        ts_inout.sink_parquet(ts_inout_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_inout_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_inout_path)
        )
        os.remove(ts_inout_path_unsorted)

        return pl.scan_parquet(ts_inout_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region helpers
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
        """
        Prints the number of unique ICU cases in the given timeseries data.

        Args:
            data (LazyFrame): The input timeseries data containing at least the column {icu_stay_id_col}.
            name (str): Descriptor for the timeseries data type (e.g., "vitals", "labs").

        Returns:
            LazyFrame: The unmodified input data.

        Columns:
            - {icu_stay_id_col}: ICU stay identifier used for uniqueness count.
        """
        unique_count = (
            data.select(self.icu_stay_id_col)
            .unique()
            .count()
            .collect(streaming=True)
            .to_numpy()[0][0]
        )
        print(
            f"reprodICU - {unique_count:6.0f} unique cases with timeseries data in {name}."
        )

        return data


# region convert
class MIMIC3Converter(UnitConverter):
    def __init__(self):
        super().__init__()

    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Converts raw lab values for MIMIC-III into canonical units.

        Applies a series of sequential unit conversions to adjust lab measurements for tests such as:
          • {Calcium} and {Calcium.ionized}: mg/dL to mmol/L.
          • {Creatine kinase.MB}: ng/mL to U/L.
          • {C reactive protein}: mg/dL to mg/L.
          • Additional tests: {Fibrin D-dimer}, {Iron}, {Magnesium}, {Myoglobin}, {Triiodothyronine}, {Thyroxine}, {Troponin}, and {Cobalamin}.

        Expected columns:
          - {labelcol}: Lab test identifier.
          - {valuecol}: Numeric lab measurement or a structured value with key {structfield}.

        Returns:
            pl.LazyFrame: Lab data with adjusted unit values.
        """
        return (
            data.pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium.ionized",
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
                self.convert_mg_dL_to_mg_L,
                itemid="C reactive protein",
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
                self.convert_ng_mL_to_mg_L,
                itemid="Fibrin D-dimer DDU",
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
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron binding capacity",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_magnesium_mg_dL_to_mmol_L,
                itemid="Magnesium",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ng_mL_to_ug_L,
                itemid="Myoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # MCHC is in %, however this is equal to g/dL due to the definition of MCHC
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="Phosphate",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # Potassium is mEq/L, however as a univalent ion, this is equal to mmol/L
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="Albumin",
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
            # Sodium is mEq/L, however as a univalent ion, this is equal to mmol/L
            .pipe(
                self.convert_T3_ng_dL_to_nmol_L,
                itemid="Triiodothyronine",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="Thyroxine",
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
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="Troponin I.cardiac",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="Troponin T.cardiac",
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
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Converts lab values in wide format to relative percentages.

        Args:
            data (LazyFrame): Input pivoted lab data after initial unit conversion.

        Returns:
            LazyFrame: Lab data with absolute counts converted to relative percentages.

        Columns modified:
            - "Eosinophils/100 leukocytes"
            - "Lymphocytes/100 leukocytes"
            - "Reticulocytes/100 erythrocytes"
        """
        return (
            data.pipe(
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
                itemcol="Reticulocytes",
                total_itemcol="Erythrocytes",
                goal_itemcol="Reticulocytes/100 erythrocytes",
                structfield="value",
                structstring=True,
            )
        )


# endregion
