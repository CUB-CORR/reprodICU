# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.

import os
from numbers import Number

import polars as pl
from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class EICUProcessor(EICUExtractor):
    def __init__(self, paths, DEMO=False):
        """
        Initializes the EICUProcessor instance.

        Args:
            paths: Object containing file paths.
            DEMO (bool): If True, use demo mode parameters.

        Sets:
            {icu_stay_id_col}: ICU stay identifier.
            {hospital_stay_id_col}: Hospital stay identifier.
            {person_id_col}: Patient identifier.
            {icu_length_of_stay_col}: ICU length of stay.
            index_cols (list): List of index columns used for pivoting (i.e., [{icu_stay_id_col}, {timeseries_time_col}]).
        """
        super().__init__(paths, DEMO)
        self.path = paths.eicu_source_path
        self.helpers = GlobalHelpers()
        self.convert = EICUConverter()
        self.icu_stay_id = self.extract_patient_information().select(
            self.icu_stay_id_col,
            self.hospital_stay_id_col,
            self.person_id_col,
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region time series
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self) -> pl.LazyFrame:
        """
        Processes and combines time series data from multiple eICU sources.

        Steps:
          1. Check if a preprocessed parquet file exists in {precalc_path}.
          2. If found, load the data with sorted index columns ({icu_stay_id_col} and {timeseries_time_col}).
          3. Otherwise, extract numeric data via _process_timeseries_nurse(), _process_periodics()
             and _process_timeseries_resp(), then join the results.
          4. Save the unsorted data, sort it by {icu_stay_id_col} and {timeseries_time_col},
             remove temporary files, and return the final LazyFrame.

        Columns:
          - {icu_stay_id_col}: Unique ICU stay identifier.
          - {timeseries_time_col}: Time (in seconds) since admission.
          - Additional columns: Measurements from nurse charting, periodic, and respiratory data.

        Returns:
            pl.LazyFrame: A wide-format LazyFrame sorted by [{icu_stay_id_col}, {timeseries_time_col}].
        """
        timeseries_path = self.precalc_path + "EICU_timeseries.parquet"
        timeseries_path_unsorted = (
            self.precalc_path + "EICU_timeseries_unsorted.parquet"
        )

        # Load preexisting data if available
        if os.path.isfile(timeseries_path):
            return pl.scan_parquet(timeseries_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        # Check if the preprocessed data is available
        if not os.path.isfile(timeseries_path_unsorted):
            # Load the time series data.
            print("eICU    - Loading time series data...")
            ts_nurse = self._process_timeseries_nurse()
            ts_periodics = self._process_periodics()
            ts_resp = self._process_timeseries_resp()

            # Join the time series data on the patient unit stay ID.
            print("eICU    - Joining wide time series data...")
            timeseries = pl.concat(
                [ts_nurse, ts_periodics, ts_resp], how="diagonal_relaxed"
            )

            # Save the preprocessed data
            timeseries.sink_parquet(timeseries_path_unsorted)

        # NOTE: if process stops due to insufficient memory, use the following
        # lines instead within a terminal at the precalc_path:
        # (
        #     pl.scan_parquet("EICU_B_timeseries_unsorted.parquet")
        #     .group_by(self.icu_stay_id_col, self.timeseries_time_col)
        #     .first()
        #     .sort(self.index_cols)
        #     .sink_parquet("EICU_B_timeseries.parquet")
        # )
        print("eICU    - Sorting wide time series data...")
        (
            pl.scan_parquet(timeseries_path_unsorted)
            .group_by(self.icu_stay_id_col, self.timeseries_time_col)
            .first()
            .sort(self.index_cols)
            .sink_parquet(timeseries_path)
        )
        os.remove(timeseries_path_unsorted)

        return pl.scan_parquet(timeseries_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region lab
    # Process lab data, i.e. extract and pivot lab data.
    # Keep only the relevant lab values.
    def process_timeseries_lab(self):
        """
        Processes laboratory time series data for eICU.

        Steps:
          1. Check if a preprocessed lab data file exists in {precalc_path}. If so, load it.
          2. Otherwise, extract raw lab data using extract_time_series_lab().
          3. Combine duplicate lab columns (e.g. "Base excess" and "Base deficit") via _combine_base_excess_and_deficit.
          4. Convert lab measurement units using _convert_lab_values.
          5. JSON encode the "labstruct" field.
          6. Pivot on "labname" to obtain one column per lab test.
          7. Save, then sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Observation time (in seconds).
          - "labname": Laboratory test name.
          - "labstruct": JSON-encoded structured lab result (including key {value}).

        Returns:
            pl.LazyFrame: A wide-format, sorted LazyFrame of laboratory data.
        """
        ts_labs_path = self.precalc_path + "EICU_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "EICU_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("eICU    - Processing lab data...")

        ts_lab = (
            self.extract_time_series_lab()
            # Combine base_excess and base_deficit into one column base_excess_deficit
            .pipe(
                self.convert._combine_base_excess_and_deficit,
                base_excess_name="Base excess",
                base_deficit_name="Base deficit",
                labelcol="labname",
                valuecol="labstruct",
                structfield="value",
            )
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="labname",
                valuecol="labstruct",
                structfield="value",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the lab values to wide format
            .collect(streaming=True)
            .pivot(
                on="labname",
                index=self.index_cols,
                values="labstruct",
                aggregate_function="first",
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

    # region resp
    # Process resp data, i.e. extract and pivot respiratory data.
    # Keep only the relevant resp values.
    def _process_timeseries_resp(self):
        """
        Process and pivot respiratory time series data.

        This function extracts respiratory data using extract_time_series_resp, pivots the data by the
        "respchartvaluelabel" column with corresponding {timeseries_time_col} and {icu_stay_id_col}, and cleans the dataset.

        Returns:
            pl.LazyFrame: The respiratory data in wide format indexed by {icu_stay_id_col} and {timeseries_time_col},
            with columns representing distinct respiratory measurements.
        """
        ts_resp_path = self.precalc_path + "EICU_timeseries_resp.parquet"
        ts_resp_path_unsorted = self.precalc_path + "EICU_ts_resp.parquet"

        if os.path.isfile(ts_resp_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_resp_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("eICU    - Processing resp data...")

        ts_resp = (
            self.extract_time_series_resp()
            # Pivot the respiratory values to wide format
            .collect(streaming=True).pivot(
                on="respchartvaluelabel",
                index=self.index_cols,
                values="respchartvalue",
                aggregate_function="first",  # NOTE: first is used here to allow for strings
            )
        )

        # Drop empty rows
        ts_resp_cols = ts_resp.collect_schema().names()
        droplist = list(set(ts_resp_cols) - set(self.index_cols))
        ts_resp = (
            ts_resp.pipe(self.helpers.dropna, "all", droplist, False)
            .cast(
                {
                    col: float
                    for col in droplist
                    if isinstance(ts_resp[col].drop_nulls().first(), Number)
                }
            )
            .unique()
            .sort(self.index_cols)
            .lazy()
        )

        # Save the preprocessed data
        ts_resp.sink_parquet(ts_resp_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_resp_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_resp_path)
        )
        os.remove(ts_resp_path_unsorted)

        return pl.scan_parquet(ts_resp_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region nurse
    # Process nurse charting data, i.e. extract and pivot nurse charting data.
    def _process_timeseries_nurse(self):
        """
        Extract and pivot nurse charting time series data.

        This function processes the nurse charting data from extract_time_series_nurse,
        pivots it using the "nursingchartcelltypevalname" as the key along with {icu_stay_id_col} and {timeseries_time_col},
        and removes empty rows.

        Returns:
            pl.LazyFrame: The nurse charting data in wide format indexed by {icu_stay_id_col} and {timeseries_time_col}.
        """
        ts_nurse_path = self.precalc_path + "EICU_ts_nurse.parquet"
        ts_nurse_path_unsorted = (
            self.precalc_path + "EICU_ts_nurse_unsorted.parquet"
        )

        print("eICU    - Processing nurse charting data...")

        if os.path.isfile(ts_nurse_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_nurse_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        ts_nurse = (
            self.extract_time_series_nurse()
            # Pivot the nurse values to wide format
            .collect(streaming=True).pivot(
                on="nursingchartcelltypevalname",
                index=self.index_cols,
                values="nursingchartvalue",
                aggregate_function="first",  # NOTE: first is used here to not run into issues with strings -> check if this is sensible
            )
        )

        # Drop empty rows
        ts_nurse_cols = ts_nurse.collect_schema().names()
        droplist = list(set(ts_nurse_cols) - set(self.index_cols))
        ts_nurse = (
            ts_nurse.pipe(self.helpers.dropna, "all", droplist, False)
            .unique()
            .lazy()
        )

        # Save the preprocessed data
        ts_nurse.sink_parquet(ts_nurse_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_nurse_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_nurse_path)
        )
        os.remove(ts_nurse_path_unsorted)

        return pl.scan_parquet(ts_nurse_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region inout
    # Process inout data, i.e. extract and pivot intake/output data.
    # Keep only the relevant inout values.
    def process_timeseries_inout(self):
        """
        Extract and pivot intake/output time series data.

        This function extracts intake/output data using extract_time_series_intake_output,
        pivots the data with "celllabel" as the key along with {icu_stay_id_col} and {timeseries_time_col},
        and calculates the mean value for duplicate entries.

        Returns:
            pl.LazyFrame: The processed intake/output data in wide format indexed by {icu_stay_id_col} and {timeseries_time_col}.
        """
        ts_inout_path = self.precalc_path + "EICU_timeseries_inout.parquet"
        ts_inout_path_unsorted = self.precalc_path + "EICU_ts_inout.parquet"

        # Process inout data
        if os.path.isfile(ts_inout_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_inout_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("eICU    - Processing intake/output data...")

        ts_inout = (
            self.extract_time_series_intake_output()
            # Pivot the intake/output values to wide format
            .collect(streaming=True).pivot(
                on="celllabel",
                index=self.index_cols,
                values="cellvaluenumeric",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        ts_inout_cols = ts_inout.collect_schema().names()
        droplist = list(set(ts_inout_cols) - set(self.index_cols))
        ts_inout = (
            ts_inout.pipe(self.helpers.dropna, "all", droplist, False)
            .unique()
            .sort(self.index_cols)
            .lazy()
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

    # region periodics
    # Process periodic data, i.e. extract and combine (a)periodic data.
    def _process_periodics(self):
        """
        Extract and combine periodic and aperiodic time series data.

        This function extracts periodic data from vitalPeriodic.csv and aperiodic data from vitalAperiodic.csv,
        concatenates them, and pivots based on processed measurements.

        Key columns:
            {icu_stay_id_col}: ICU stay identifier.
            {timeseries_time_col}: Time of observation.
            Plus additional columns from periodic/aperiodic measurements.

        Returns:
            pl.LazyFrame: Processed (a)periodic data in wide format indexed by {icu_stay_id_col} and {timeseries_time_col}.
        """
        ts_period_path = self.precalc_path + "EICU_ts_periodics.parquet"
        ts_period_path_unsorted = (
            self.precalc_path + "EICU_ts_periodics_unsorted.parquet"
        )

        # Process (a)periodic data
        if os.path.isfile(ts_period_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_period_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("eICU    - Processing (a)periodic data...")

        ts_periodics = self.extract_and_combine_periodics()

        # Drop empty rows
        ts_periodics_cols = ts_periodics.collect_schema().names()
        droplist = list(set(ts_periodics_cols) - set(self.index_cols))
        ts_periodics = ts_periodics.pipe(
            self.helpers.dropna, "all", droplist, False
        )

        # Save the preprocessed data
        ts_periodics.sink_parquet(ts_period_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_period_path_unsorted)
            .cast(float)
            .cast({self.icu_stay_id_col: int})
            .unique(self.index_cols)
            .sort(self.index_cols)
            .sink_parquet(ts_period_path)
        )
        os.remove(ts_period_path_unsorted)

        return pl.scan_parquet(ts_period_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region diagnoses
    # Processes the diagnoses data of the eICU dataset.
    def process_diagnoses(self):
        """
        Extracts and processes diagnosis data from eICU.

        Steps:
          1. Extract diagnosis data from file (e.g. diagnosis.csv.gz).
          2. Remove dots from ICD codes.
          3. Determine ICD version (9 or 10) based on mappings.
          4. Add diagnosis description accordingly.

        Columns:
          - {diagnosis_icd_code_col}: ICD diagnosis code.
          - {diagnosis_start_col}: Onset time.
          - {diagnosis_priority_col}: Numeric priority.
          - {diagnosis_discharge_col}: Active status upon discharge.
          - {diagnosis_description_col}: Mapped description for the ICD code.

        Returns:
            pl.LazyFrame: Processed diagnosis data.
        """
        ICD9_descriptions = dict(
            zip(
                self.ICD9_TO_ICD10_DIAGS["icd9"],
                self.ICD9_TO_ICD10_DIAGS["description"],
            )
        )
        ICD10_descriptions = dict(
            zip(
                self.ICD10_TO_ICD9_DIAGS["icd10"],
                self.ICD10_TO_ICD9_DIAGS["description"],
            )
        )

        # Return the processed diagnoses data.
        print("eICU    - Processing diagnoses data...")
        return (
            self.extract_diagnoses()
            # Remove the dots from the ICD codes.
            .with_columns(
                pl.col(self.diagnosis_icd_code_col).str.replace_all("\.", "")
            )
            # Determine the ICD version of the diagnoses.
            .with_columns(
                pl.when(
                    pl.col(self.diagnosis_icd_code_col).is_in(
                        ICD9_descriptions.keys()
                    ),
                )
                .then(pl.lit(9))
                .otherwise(
                    pl.when(
                        pl.col(self.diagnosis_icd_code_col).is_in(
                            ICD10_descriptions.keys()
                        ),
                    )
                    .then(pl.lit(10))
                    .otherwise(pl.lit(None))
                )
                .alias(self.diagnosis_icd_version_col)
            )
            # Add the description of the diagnoses, depending on the ICD version.
            .with_columns(
                pl.when(pl.col(self.diagnosis_icd_version_col) == 9)
                .then(
                    pl.col(self.diagnosis_icd_code_col).replace_strict(
                        ICD9_descriptions, default=None
                    )
                )
                .otherwise(
                    pl.when(pl.col(self.diagnosis_icd_version_col) == 10)
                    .then(
                        pl.col(self.diagnosis_icd_code_col).replace_strict(
                            ICD10_descriptions, default=None
                        )
                    )
                    .otherwise(pl.lit(None))
                )
                .alias(self.diagnosis_description_col)
            )
        )

    # endregion


# region convert
class EICUConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    def _combine_base_excess_and_deficit(
        self,
        data: pl.DataFrame,
        base_excess_name: str,
        base_deficit_name: str,
        labelcol: str = "labname",
        valuecol: str = "labstruct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Combines base excess and base deficit into a unified column.

        Steps:
          1. Unnest the {labstruct} column.
          2. For rows where {labname} equals {base_deficit_name}, multiply {value} by -1.
          3. Rename both {base_excess_name} and {base_deficit_name} to "Base excess".
          4. Assign the standard LOINC code ({base_excess_LOINC}) for "Base excess".
          5. Reassemble the struct with keys {value}, {system}, {method}, {time}, and {LOINC}.

        Returns:
            pl.LazyFrame: DataFrame with combined "Base excess" column.
        """
        base_excess_LOINC = "11555-0"  # Base excess in Blood by calculation

        return (
            data.unnest(valuecol).with_columns(
                pl.when(
                    pl.col(labelcol) == base_deficit_name,
                )
                .then(pl.col(structfield) * -1)
                .otherwise(pl.col(structfield))
                .alias(structfield),
            )
            # Rename base_excess and base_deficit to base_excess_deficit
            .with_columns(
                pl.when(
                    pl.col(labelcol).is_in(
                        [base_excess_name, base_deficit_name]
                    ),
                )
                .then(pl.lit("Base excess"))
                .otherwise(pl.col(labelcol))
                .alias(labelcol),
                pl.when(
                    pl.col(labelcol).is_in(
                        [base_excess_name, base_deficit_name]
                    ),
                )
                .then(pl.lit(base_excess_LOINC))
                .otherwise(pl.col("LOINC"))
                .alias("LOINC"),
            )
            # Combine the columns back into a struct again
            .select(
                pl.exclude("value", "system", "method", "time", "LOINC"),
                pl.struct(
                    value="value",
                    system="system",
                    method="method",
                    time="time",
                    LOINC="LOINC",
                ).alias(valuecol),
            )
        )

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "labname",
        valuecol: str = "labstruct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Converts raw lab values to canonical units for eICU.

        Applies a series of unit conversion pipes targeting multiple lab tests including:
          - {Calcium} and {Calcium.ionized}: mg/dL to mmol/L.
          - {Creatine kinase.MB}: ng/mL to U/L.
          - {C reactive protein}: mg/dL to mg/L.
          - {Fibrin D-dimer FEU}, {Iron}, {Magnesium}, {Myoglobin}, {Prealbumin}, {Protein}, {Triiodothyronine},
            {Thyroxine} (and free), {Troponin I.cardiac}, {Troponin T.cardiac}, {Iron binding capacity}, and {Cobalamin}.

        Expected columns:
          - {labelcol}: Lab test identifier.
          - {valuecol}: Lab result struct with key {structfield}.

        Returns:
            pl.LazyFrame: Converted lab data.
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
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron",
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
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="Phosphate",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="Albumin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="Prealbumin",
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
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron binding capacity",
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


# endregion
