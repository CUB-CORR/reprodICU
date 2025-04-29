# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl
from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class MIMIC4Processor(MIMIC4Extractor):
    def __init__(self, paths, DEMO=False):
        """
        Initialize the MIMIC4Processor.

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
        self.path = paths.mimic4_source_path
        self.helpers = GlobalHelpers()
        self.convert = MIMIC4Converter()
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
    # Processes the vital data of the MIMIC4 dataset.
    def process_timeseries_vitals(self):
        """
        Processes vital and respiratory data for MIMIC-IV.

        Steps:
          1. Check if a sorted vital data file exists in {precalc_path}.
          2. If not, extract chart events and perform:
             • Temperature conversion from Fahrenheit to Celsius.
          3. Pivot the data using "label" as the key with mean aggregation.
          4. Drop rows with all null non-index values.
          5. Save the unsorted data, sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time (seconds) from admission.
          - "label": Measurement identifier for pivoting.
          - Additional columns: Vital and respiratory measurements.

        Returns:
            pl.LazyFrame: A sorted LazyFrame aggregating vital and respiratory data.
        """
        ts_vitals_path = self.precalc_path + "MIMIC4_timeseries_vitals.parquet"
        ts_vitals_path_unsorted = self.precalc_path + "MIMIC4_ts_vitals.parquet"

        if os.path.isfile(ts_vitals_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_vitals_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("MIMIC4  - Processing vitals & respiratory data...")

        # Process vitals data
        ts_vitals = (
            self.extract_chartevents()
            # Convert temperature from Fahrenheit to Celsius
            .pipe(
                self.convert.convert_temperature_F_to_C,
                itemid_F="Temperature Fahrenheit",
                itemid_C="Temperature",
                labelcol="label",
                valuecol="valuenum",
            )
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="label",
                index=self.index_cols,
                values="valuenum",
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
    # Processes the lab data of the MIMIC4 dataset.
    def process_timeseries_labevents(self):
        """
        Processes laboratory measurement data for MIMIC-IV.

        Steps:
          1. Check for a preprocessed lab file in {precalc_path}.
          2. If not present, extract lab measurements with extract_lab_measurements().
          3. Convert lab values using _convert_lab_values and JSON encode "labstruct".
          4. Pivot on "label" to form a wide-format dataset.
          5. Apply wide-format unit adjustments.
          6. Save, sort by {icu_stay_id_col} and {timeseries_time_col}, and clean up temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset (seconds) for observations.
          - "label": Lab test name pivot key.
          - "labstruct": JSON-encoded lab result structure (including {value}).

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame containing laboratory data.
        """
        ts_labs_path = self.precalc_path + "MIMIC4_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "MIMIC4_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("MIMIC4  - Processing lab data...")

        # Process lab data
        ts_lab = (
            self.extract_lab_measurements()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="label",
                valuecol="labstruct",
                structfield="value",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the lab data
            .collect()
            .pivot(
                on="label",
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
    # Processes the input/output data of the MIMIC4 dataset.
    def process_timeseries_inputoutput(self):
        """
        Processes input/output measurement data for MIMIC-IV.

        Steps:
          1. Check if a preprocessed input/output file exists.
          2. If not, extract output event measurements.
          3. Pivot using "label" with mean aggregation.
          4. Remove rows entirely null aside from index.
          5. Save and sort by {icu_stay_id_col} and {timeseries_time_col}, then remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time (seconds) relative to admission.
          - "label": Key for identifying input/output events.
          - "VALUENUM": Mean aggregated measurement value.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame containing input/output data.
        """
        ts_inout_path = self.precalc_path + "MIMIC4_timeseries_inout.parquet"
        ts_inout_path_unsorted = self.precalc_path + "MIMIC4_ts_inout.parquet"

        if os.path.isfile(ts_inout_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_inout_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("MIMIC4  - Processing inout data...")

        # Process inout data
        ts_inout = (
            self.extract_output_measurements()
            # Pivot the inout data
            .collect(streaming=True).pivot(
                on="label",
                index=self.index_cols,
                values="valuenum",
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
        Prints the count of unique ICU cases present in the provided timeseries data.

        Args:
            data (pl.LazyFrame): The timeseries data containing at least the {icu_stay_id_col} column.
            name (str): Name of the dataset (for display purposes).

        Returns:
            pl.LazyFrame: The input data, unchanged.

        The function counts uniqueness based on the {icu_stay_id_col}.
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


# endregion


# region convert
class MIMIC4Converter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the MIMIC-IV dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "label",
        valuecol: str = "valuenum",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Convert lab measurement values to canonical units.

        This method applies a series of unit conversion pipes to fix discrepancies in lab measurements.
        The conversions target columns specified by:
          - {labelcol}: Column containing the lab label.
          - {valuecol}: Column (or struct field) where the numerical measurement (named {structfield}) is stored.

        Returns:
            pl.LazyFrame: LazyFrame with converted lab values.
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
            # NOTE: Experience from clinical practice:
            # Creatinine is more commonly referred to in mg/L, so this conversion seems necessary
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
        Convert wide-format lab values to relative measurements where appropriate.

        The conversion targets absolute count values and converts them to relative values based on totals.
        This is done for lab analytes such as:
          - "Basophils" relative to "Leukocytes" -> resulting in {Basophils/100 leukocytes}
          - "Eosinophils" relative to "Leukocytes" -> resulting in {Eosinophils/100 leukocytes}
          - "Lymphocytes" relative to "Leukocytes" -> resulting in {Lymphocytes/100 leukocytes}
          - "Monocytes" relative to "Leukocytes" -> resulting in {Monocytes/100 leukocytes}
          - "Neutrophils" relative to "Leukocytes" -> resulting in {Neutrophils/100 leukocytes}
          - "Reticulocytes" relative to "Erythrocytes" -> resulting in {Reticulocytes/100 erythrocytes}

        Args:
            data (pl.LazyFrame): Input lab data in wide format.

        Returns:
            pl.LazyFrame: LazyFrame with converted relative lab values.
        """

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
            # .pipe(
            #     self.convert_absolute_count_to_relative,
            #     itemcol="Band form neutrophils",
            #     total_itemcol="Leukocytes",
            #     goal_itemcol="Neutrophils.band form/100 leukocytes",
            #     structfield="value",
            #     structstring=True,
            # )
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
