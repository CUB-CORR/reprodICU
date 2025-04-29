# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl
from helpers.A_extract.A_extract_nwicu import NWICUExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class NWICUProcessor(NWICUExtractor):
    def __init__(self, paths):
        """
        Initialize the NWICUProcessor instance.

        Args:
            paths: Object containing source file paths.

        Sets:
            path: Path to the NWICU source data ({nwicu_source_path}).
            helpers: Instance of GlobalHelpers.
            convert: Instance of NWICUConverter for unit conversions.
            icu_stay_id: LazyFrame with columns [{icu_stay_id_col}, {hospital_stay_id_col}, {person_id_col}].
            icu_length_of_stay: LazyFrame with columns [{icu_stay_id_col}, {icu_length_of_stay_col}].
            index_cols: List of columns used for pivoting (i.e. [{icu_stay_id_col}, {timeseries_time_col}]).
        """
        super().__init__(paths)
        self.path = paths.nwicu_source_path
        self.helpers = GlobalHelpers()
        self.convert = NWICUConverter()
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
    def process_timeseries_vitals(self):
        """
        Processes and pivots vital sign measurements for NWICU.

        Steps:
          1. Check for an existing cached vital data file in {precalc_path}.
          2. If found, load the file with sorted index columns ({icu_stay_id_col} and {timeseries_time_col}).
          3. Otherwise, extract vital measurements via extract_chartevents(), then:
             • Convert temperature from Fahrenheit to Celsius.
             • Pivot the data on "label" using mean aggregation.
          4. Drop rows where all non-index columns are null.
          5. Save the unsorted result, sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time offset (seconds) since ICU admission.
          - "label": Vital sign label.
          - Other columns: Numerical measurements.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with vital data.
        """
        ts_vitals_path = self.precalc_path + "NWICU_timeseries_vitals.parquet"
        ts_vitals_path_unsorted = self.precalc_path + "NWICU_ts_vitals.parquet"

        if os.path.isfile(ts_vitals_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_vitals_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("NWICU  - Processing vitals data...")

        # Process vitals data
        ts_vitals = (
            self.extract_chartevents()
            # Convert temperature from Fahrenheit to Celsius
            .pipe(
                self.convert.convert_temperature_F_to_C,
                itemid_F="Temperature",
                itemid_C="Temperature",
                labelcol="label",
                valuecol="valuenum",
            )
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="label",
                index=self.index_cols,
                values="valuenum",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
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
    def process_timeseries_labevents(self):
        """
        Processes laboratory measurement data for NWICU.

        Steps:
          1. Check if a preprocessed laboratory file exists in {precalc_path} and load it if so.
          2. Otherwise, extract lab measurements via extract_lab_measurements().
          3. Convert lab values to canonical units using _convert_lab_values.
          4. JSON encode the "labstruct" field.
          5. Pivot on "label" so that each lab test becomes its own column.
          6. Adjust wide-format lab values using _convert_wide_lab_values.
          7. Save the unsorted result, sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Observation time (seconds).
          - "label": Lab test pivot key.
          - "labstruct": JSON-encoded lab result structure containing {value}.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with laboratory data.
        """
        ts_labs_path = self.precalc_path + "NWICU_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "NWICU_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("NWICU  - Processing lab data...")

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

    # # region input/output
    # # Processes the input/output data of the NWICU dataset.
    # def process_timeseries_inputoutput(self):
    #     """
    #     Processes the input/output data of the NWICU dataset.
    #     """
    #     ts_inout_path = self.precalc_path + "NWICU_timeseries_inout.parquet"
    #     ts_inout_path_unsorted = self.precalc_path + "NWICU_ts_inout.parquet"

    #     if os.path.isfile(ts_inout_path):
    #         # Load the preprocessed data
    #         return pl.scan_parquet(ts_inout_path).select(
    #             pl.col(self.index_cols).set_sorted(),
    #             pl.exclude(self.index_cols),
    #         )

    #     print("NWICU  - Processing inout data...")

    #     # Process inout data
    #     ts_inout = (
    #         self.extract_output_measurements()
    #         # Pivot the inout data
    #         .collect(streaming=True).pivot(
    #             on="label",
    #             index=self.index_cols,
    #             values="valuenum",
    #             aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
    #         )
    #     )

    #     # Drop empty rows
    #     ts_inout_cols = ts_inout.collect_schema().names()
    #     droplist = list(set(ts_inout_cols) - set(self.index_cols))
    #     ts_inout = (
    #         ts_inout.lazy()
    #         .pipe(self.helpers.dropna, "all", droplist, False)
    #         .unique()
    #         .sort(self.index_cols)
    #     )

    #     # Save the preprocessed data
    #     ts_inout.sink_parquet(ts_inout_path_unsorted)

    #     # Sort the data
    #     (
    #         pl.scan_parquet(ts_inout_path_unsorted)
    #         .sort(self.index_cols)
    #         .sink_parquet(ts_inout_path)
    #     )
    #     os.remove(ts_inout_path_unsorted)

    #     return pl.scan_parquet(ts_inout_path).select(
    #         pl.col(self.index_cols).set_sorted(),
    #         pl.exclude(self.index_cols),
    #     )

    # # endregion

    # region helpers
    # Print the number of unique cases in the timeseries data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
        """
        Print the count of unique ICU cases in the timeseries data.

        Args:
            data (pl.LazyFrame): Timeseries data that must include the {icu_stay_id_col} column.
            name (str): Name or description of the dataset (e.g. 'vitals', 'labs').

        Returns:
            pl.LazyFrame: The unchanged input data.

        The function counts unique cases based on the column {icu_stay_id_col} and logs the count.
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
class NWICUConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "label",
        valuecol: str = "valuenum",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Converts lab measurement values for NWICU to canonical units.

        Applies a series of conversion functions (via .pipe calls) on specific lab tests based on their item IDs.
        Conversions include temperature, inflammation markers, and electrolyte-related values.

        Expected input columns:
          - {labelcol}: Lab test identifier.
          - {valuecol}: Field storing numerical value or a struct containing {structfield}.

        Returns:
            pl.LazyFrame: Lab data with converted values.
        """

        # Convert the lab values to the correct units.
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
        Convert wide-format lab values to relative measurements.

        For lab analytes reported as absolute counts, this method converts them to relative values
        (e.g. counts per 100 {total_itemcol}) for:
          - "Basophils" relative to "Leukocytes" as {Basophils/100 leukocytes}
          - "Eosinophils" relative to "Leukocytes" as {Eosinophils/100 leukocytes}
          - "Lymphocytes" relative to "Leukocytes" as {Lymphocytes/100 leukocytes}
          - "Monocytes" relative to "Leukocytes" as {Monocytes/100 leukocytes}
          - "Neutrophils" relative to "Leukocytes" as {Neutrophils/100 leukocytes}
          - "Reticulocytes" relative to "Erythrocytes" as {Reticulocytes/100 erythrocytes}

        Args:
            data (pl.LazyFrame): Wide-format lab data with already pivoted columns.

        Returns:
            pl.LazyFrame: LazyFrame with the applicable lab columns converted to relative measurements.
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
        )


# endregion
