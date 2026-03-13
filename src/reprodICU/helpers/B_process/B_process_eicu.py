# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.

import os

import polars as pl

from ..A_extract.A_extract_eicu import EICUExtractor
from ..helper import GlobalHelpers
from ..helper_batch import batch_process_timeseries
from ..helper_conversions import UnitConverter


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
        Process and combine time series data from multiple eICU sources.

        Steps:
            1. Check for preprocessed timeseries file; load if available.
            2. Extract nurse charting, periodic/aperiodic, and respiratory measurements.
            3. Join all three datasets on index columns, preferring later sources when overlapping.
            4. Check data sortedness and sort if necessary.
            5. Save sorted data and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Measurement columns from nurse charting, periodic, and respiratory data.
        """
        timeseries_path = self.precalc_path + "EICU_timeseries.parquet"
        timeseries_path_unsorted = (
            self.precalc_path + "EICU_timeseries_unsorted.parquet"
        )

        # Load preexisting data if available
        if os.path.isfile(timeseries_path):
            return pl.scan_parquet(
                timeseries_path, parallel="prefiltered"
            ).select(
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

            # Common columns for each pairwise combination, excluding the index columns.
            tsn_cols = set(ts_nurse.collect_schema().names())
            tsp_cols = set(ts_periodics.collect_schema().names())
            tsr_cols = set(ts_resp.collect_schema().names())

            common_np = tsn_cols.intersection(tsp_cols).difference(self.index_cols) # fmt: skip
            common_nr = tsn_cols.intersection(tsr_cols).difference(self.index_cols) # fmt: skip
            common_pr = tsp_cols.intersection(tsr_cols).difference(self.index_cols) # fmt: skip
            common_npr = common_nr.union(common_pr)

            # Join the time series data on the patient unit stay ID.
            print("eICU    - Joining wide time series data...")
            (
                ts_periodics.join(
                    ts_nurse,
                    on=self.index_cols,
                    how="full",
                    suffix="_nurse",
                    coalesce=True,
                )
                # prefer periodic values over nurse values
                .with_columns(
                    pl.coalesce(col, col + "_nurse").alias(col)
                    for col in common_np
                )
                .drop([col + "_nurse" for col in common_np])
                .join(
                    ts_resp,
                    on=self.index_cols,
                    how="full",
                    suffix="_resp",
                    coalesce=True,
                )
                # prefer respiratory values over periodic and nurse values
                .with_columns(
                    pl.coalesce(col + "_resp", col).alias(col)
                    for col in common_npr
                )
                .drop(col + "_resp" for col in common_npr)
                .sink_parquet(timeseries_path_unsorted)
            )

        # Check sortedness of index columns
        # 1. Check if the ID column is sorted
        # 2. Check for each ID if the time column is sorted
        print("eICU    - Checking sortedness of index columns...")
        timeseries_is_sorted = (
            pl.scan_parquet(timeseries_path_unsorted, parallel="prefiltered")
            .group_by(self.icu_stay_id_col, maintain_order=True)
            .agg(
                pl.col(self.timeseries_time_col)
                .eq(pl.col(self.timeseries_time_col).sort())
                .all()
                .alias("is_sorted")
            )
            .filter(pl.col("is_sorted").not_())
            .collect()
            .height
            == 0
        )

        if not timeseries_is_sorted:
            print("eICU    - Sorting wide time series data...")
            batch_process_timeseries(
                input_file=timeseries_path_unsorted,
                output_file=timeseries_path,
                tempfiles_path=self.precalc_path,
                operation="sort",
                method=lambda df: df,  # identity -> batch processor is sorting
                id_col=self.icu_stay_id_col,
                delete_after=True,
            )
            os.remove(timeseries_path_unsorted)
        else:
            print("eICU    - Time series data is already sorted.")
            os.rename(timeseries_path_unsorted, timeseries_path)

        # Delete all EICU_ts_* files
        for file in os.listdir(self.precalc_path):
            if file.startswith("EICU_ts_"):
                os.remove(os.path.join(self.precalc_path, file))

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
        Process laboratory time series measurements.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. Extract lab measurements and combine related fields (base excess/deficit).
            3. Convert lab values to canonical units.
            4. Apply LOINC component mapping and JSON encode structured fields.
            5. Pivot data on "labname" to create wide-format dataset using batch processing.
            6. Sort by index columns and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from labname, JSON-encoded).
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

        # Extract and transform lab data
        (
            self.extract_timeseries_lab()
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
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="labname",
            ).with_columns(pl.col("labstruct").struct.json_encode())
            # Save extracted data before pivoting
            .sink_parquet(ts_labs_path_unsorted)
        )

        # Batch pivot the data
        batch_process_timeseries(
            input_file=ts_labs_path_unsorted,
            output_file=ts_labs_path,
            tempfiles_path=self.precalc_path,
            operation="pivot",
            method=lambda df: self.pivot_numeric_or_string(
                df,
                dataset="eICU_labs",
                on_col="labname",
                index_cols=self.index_cols,
                string_col="labstruct",
            ).sort(self.index_cols),
            id_col=self.icu_stay_id_col,
            delete_after=True,
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

        Steps:
            1. Check for preprocessed respiratory file; load if available.
            2. Extract respiratory chart value measurements.
            3. Pivot data on "respchartvaluelabel" using first aggregation in batch processing.
            4. Drop rows where all non-index columns are null.
            5. Cast to float for numeric data and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Respiratory measurement columns (pivoted from respchartvaluelabel).
        """
        ts_resp_path = self.precalc_path + "EICU_ts_resp.parquet"
        ts_resp_path_unsorted = (
            self.precalc_path + "EICU_ts_resp_unsorted.parquet"
        )

        if os.path.isfile(ts_resp_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_resp_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("eICU    - Processing resp data...")

        # Extract respiratory data and save for batch processing
        self.extract_timeseries_resp().sink_parquet(ts_resp_path_unsorted)

        # Batch pivot the respiratory data
        batch_process_timeseries(
            input_file=ts_resp_path_unsorted,
            output_file=ts_resp_path,
            tempfiles_path=self.precalc_path,
            operation="pivot",
            method=lambda df: self.pivot_numeric_or_string(
                df,
                dataset="eICU_resp",
                on_col="respchartvaluelabel",
                index_cols=self.index_cols,
                numeric_col="respchartvaluefloat",
                string_col="respchartvaluestr",
            ).sort(self.index_cols),
            id_col=self.icu_stay_id_col,
            delete_after=True,
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
        Process and pivot nurse charting time series data.

        Steps:
            1. Check for preprocessed nurse charting file; load if available.
            2. Extract nurse charting measurements.
            3. Pivot data on "nursingchartcelltypevalname" using first aggregation in batch processing.
            4. Drop rows where all non-index columns are null.
            5. Clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Nurse charting measurement columns (pivoted from nursingchartcelltypevalname).
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

        # Extract nurse charting data and save for batch processing
        self.extract_timeseries_nurse().sink_parquet(ts_nurse_path_unsorted)

        # Batch pivot the nurse charting data
        batch_process_timeseries(
            input_file=ts_nurse_path_unsorted,
            output_file=ts_nurse_path,
            tempfiles_path=self.precalc_path,
            operation="pivot",
            method=lambda df: self.pivot_numeric_or_string(
                df,
                dataset="eICU_nurse",
                on_col="nursingchartcelltypevalname",
                index_cols=self.index_cols,
                numeric_col="nursingchartvaluefloat",
                string_col="nursingchartvaluestr",
            ).sort(self.index_cols),
            id_col=self.icu_stay_id_col,
            delete_after=True,
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
        Process and pivot intake/output time series data.

        Steps:
            1. Check for preprocessed intake/output file; load if available.
            2. Extract intake/output measurements.
            3. Pivot data on "celllabel" using mean aggregation for wide-format.
            4. Drop rows where all non-index columns are null.
            5. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Intake/output measurement columns (pivoted from celllabel).
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

        (
            self.extract_timeseries_intake_output()
            # Pivot the intake/output values to wide format
            .pipe(
                self.pivot_numeric_or_string,
                dataset="eICU_inout",
                on_col="celllabel",
                index_cols=self.index_cols,
                numeric_col="cellvaluenumeric",
            )
            # Save the preprocessed data
            .sink_parquet(ts_inout_path_unsorted)
        )

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
        Process and pivot periodic and aperiodic time series data.

        Steps:
            1. Check for preprocessed (a)periodic file; load if available.
            2. Extract and combine periodic and aperiodic vital measurements.
            3. Drop rows where all non-index columns are null.
            4. Cast to float for normalization in batch processing.
            5. Clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Periodic/aperiodic vital measurement columns.
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

        # Extract and combine periodic and aperiodic data
        self.extract_and_combine_periodics().sink_parquet(ts_period_path_unsorted) # fmt: skip

        # Batch process: cast to float and deduplicate
        batch_process_timeseries(
            input_file=ts_period_path_unsorted,
            output_file=ts_period_path,
            tempfiles_path=self.precalc_path,
            operation="process",
            method=lambda df: df.unique(self.index_cols).sort(self.index_cols),
            id_col=self.icu_stay_id_col,
            delete_after=True,
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
        Process diagnosis data with ICD version detection and descriptions.

        Steps:
            1. Extract diagnosis data.
            2. Remove dots from ICD codes for standardization.
            3. Determine ICD version (9 or 10) using known code mappings.
            4. Map codes to standardized descriptions.

        Returns:
            pl.LazyFrame: Contains columns:
                - {diagnosis_icd_code_col}: ICD code.
                - {diagnosis_icd_version_col}: ICD version (9 or 10).
                - {diagnosis_description_col}: Mapped diagnosis description.
                - Additional diagnosis metadata columns.
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
                        ICD10_descriptions.keys()
                    )
                )
                .then(pl.lit(10))
                .when(
                    pl.col(self.diagnosis_icd_code_col).is_in(
                        ICD9_descriptions.keys()
                    )
                )
                .then(pl.lit(9))
                .otherwise(pl.lit(None))
                .alias(self.diagnosis_icd_version_col)
            )
            # Add the description of the diagnoses, depending on the ICD version.
            .with_columns(
                pl.when(pl.col(self.diagnosis_icd_version_col) == 10)
                .then(
                    pl.col(self.diagnosis_icd_code_col).replace_strict(
                        ICD10_descriptions, default=None
                    )
                )
                .when(pl.col(self.diagnosis_icd_version_col) == 9)
                .then(
                    pl.col(self.diagnosis_icd_code_col).replace_strict(
                        ICD9_descriptions, default=None
                    )
                )
                .otherwise(pl.lit(None))
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
        Combine base excess and base deficit measurements into unified column.

        Steps:
            1. Unnest structured lab value column.
            2. Negate values for base deficit entries.
            3. Rename both fields to standardized "Base excess" label.
            4. Assign standard LOINC code for base excess.
            5. Reassemble into structured lab value format.

        Returns:
            pl.LazyFrame: Lab data with combined base excess column.
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
