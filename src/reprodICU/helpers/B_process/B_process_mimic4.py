# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl

from ..A_extract.A_extract_mimic4 import MIMIC4Extractor
from ..helper import GlobalHelpers
from ..helper_conversions import UnitConverter


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
        Process vital and respiratory time series measurements.

        Steps:
            1. Check for preprocessed vitals file; load if available.
            2. Extract chart events and apply temperature conversion (F→C).
            3. Pivot data on "label" to create wide-format dataset with first value aggregation.
            4. Replace categorical value codes with text descriptions.
            5. Drop rows where all non-index columns are null.
            6. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Vital sign measurement columns (pivoted from label).
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
            .collect().pivot(
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
                pl.col("Continuous renal replacement therapy mode Renal replacement therapy circuit").replace_strict(
                    self.rrt_mode_enum_map_inverted,
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
        Process laboratory time series measurements.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. Extract lab measurements and align unit representations.
            3. Convert lab values to canonical units.
            4. Apply LOINC component mapping and JSON encode structured fields.
            5. Pivot data on "label" to create wide-format dataset.
            6. Apply post-pivot unit conversions and percentage calculations.
            7. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from label, JSON-encoded).
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
            # Align the units of the lab values
            .pipe(self.convert._align_units)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="label",
                valuecol="labstruct",
                structfield="value",
            )
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="label",
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
                    "Reticulocytes/100 erythrocytes",
                ],
            )
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
        Process input/output measurement time series data.

        Steps:
            1. Check for preprocessed input/output file; load if available.
            2. Extract output measurement events.
            3. Pivot data on "label" using mean aggregation for wide-format.
            4. Drop rows where all non-index columns are null.
            5. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Input/output measurement columns (pivoted from label).
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
            .collect().pivot(
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
        """Convert raw lab values to canonical units.

        Applies sequential unit conversions for multiple lab tests including:
        calcium, creatinine kinase, proteins, hormones, and cardiac markers.

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
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium",
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

    def _align_units(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Align lab unit representations for count measurements.
        Converts absolute counts to consistent units (K/uL → #/uL).

        Returns:
            pl.LazyFrame: Lab data with aligned unit values.
        """

        print("MIMIC4  - Aligning lab value units...")

        # 51199: Eosinophil Count -> #/uL (52073 "Absolute Eosinophil Count" in K/uL)
        # 51253: Monocyte Count -> #/uL (52074 "Absolute Monocyte Count" in K/uL)
        # 51697: Neutrophil Count -> #/uL (52075 "Absolute Neutrophil Count" in K/uL)
        # 52769: Absolute Lymphocyte Count -> #/uL (51133 "Absolute Lymphocyte Count" in K/uL)

        return (
            data.unnest("labstruct")
            .with_columns(
                pl.when(pl.col("itemid").is_in([51199, 51253, 51697, 52769]))
                .then(pl.col("value").mul(1000))
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
        (basophils, eosinophils, lymphocytes, monocytes, neutrophils, reticulocytes).

        Returns:
            pl.LazyFrame: Lab data with calculated percentage values.
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
