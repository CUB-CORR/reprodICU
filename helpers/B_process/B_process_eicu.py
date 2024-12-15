# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.

from numbers import Number
import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.A_extract_eicu import EICUExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class EICUProcessor(EICUExtractor):
    def __init__(self, paths, DEMO=False):
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
        Process and combine the time series data of the eICU dataset.

        Specifics for the eICU dataset:
        - Extract and pivot lab data.
        - Extract and pivot respiratory data.
        - Extract and pivot intake/output data.
        - Extract and pivot nurse charting data.
        - Extract and combine (a)periodic data.

        The time series data in wide format is indexed by the ICU stay ID and time.

        :return: The processed time series data in wide format.
        :rtype: pl.LazyFrame
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
        # pl.scan_parquet("EICU_B_timeseries_unsorted.parquet").sort(
        #     "icu_stay_id", "time_relative_to_admission"
        # ).sink_parquet("HiRID_B_timeseries.parquet")
        print("eICU    - Sorting wide time series data...")
        (
            pl.scan_parquet(timeseries_path_unsorted)
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
        Process lab data, i.e. extract and pivot lab data.
        Keep only the relevant lab values.

        The processed lab data in wide format is indexed by the ICU stay ID and time.

        :return: The processed lab data in wide format.
        :rtype: pl.LazyFrame
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
                base_excess_name="base_excess",
                base_deficit_name="base_deficit",
                labelcol="labname",
                valuecol="value_struct",
                structfield="value",
            )
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="labname",
                valuecol="value_struct",
                structfield="value",
            )
            .with_columns(
                pl.col("value_struct")
                .struct.json_encode()
                .alias("value_struct")
            )
            # Pivot the lab values to wide format
            .collect(streaming=True)
            .pivot(
                on="labname",
                index=self.index_cols,
                values="value_struct",
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
        Process resp data, i.e. extract and pivot respiratory data.
        Keep only the relevant respiratory values.

        The processed respiratory data in wide format is indexed by the ICU stay ID and time.

        :return: The processed respiratory data in wide format.
        :rtype: pl.LazyFrame
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
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        ts_resp_cols = ts_resp.collect_schema().names()
        droplist = list(set(ts_resp_cols) - set(self.index_cols))
        ts_resp = (
            ts_resp.pipe(self.helpers.dropna, "all", droplist, False)
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
        Process nurse charting data, i.e. extract and pivot nurse charting data.
        NOTE: All values are deemed relevant.

        The processed nurse charting data in wide format is indexed by the ICU stay ID and time.

        :return: The processed nurse charting data in wide format.
        :rtype: pl.LazyFrame
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
            .sort(self.index_cols)
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
        Process inout data, i.e. extract and pivot intake/output data.
        Keep only the relevant inout values.

        The processed intake/output data in wide format is indexed by the ICU stay ID and time.

        :return: The processed intake/output data in wide format.
        :rtype: pl.LazyFrame
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
        Process inout data, i.e. extract (a)periodic data.
        Keep only the relevant inout values.

        The processed intake/output data in wide format is indexed by the ICU stay ID and time.

        :return: The processed intake/output data in wide format.
        :rtype: pl.LazyFrame
        """
        ts_period_path = self.precalc_path + "EICU_ts_periodics.parquet"
        ts_period_path_unsorted = (
            self.precalc_path + "EICU_ts_periodics_unsorted.parquet"
        )

        print("eICU    - Processing (a)periodic data...")

        # Process inout data
        if os.path.isfile(ts_period_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_period_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        ts_periodics = self.extract_and_combine_periodics()

        # Drop empty rows
        ts_periodics_cols = ts_periodics.collect_schema().names()
        droplist = list(set(ts_periodics_cols) - set(self.index_cols))
        ts_periodics = (
            ts_periodics.pipe(self.helpers.dropna, "all", droplist, False)
            .unique()
            .sort(self.index_cols)
        )

        # Save the preprocessed data
        ts_periodics.sink_parquet(ts_period_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_period_path_unsorted)
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
        Processes the diagnoses data of the eICU dataset, i.e. extracts the diagnoses data,
        determines the ICD version of the diagnoses and adds the description of the diagnoses.

        The processed diagnoses data contains the following columns:
        - diagnosis_icd_code: The ICD code of the diagnosis.
        - diagnosis_icd_version: The ICD version of the diagnosis.
        - diagnosis_start: The start time of the diagnosis.
        - diagnosis_priority: The priority of the diagnosis.
        - diagnosis_discharge: The discharge status of the diagnosis.

        The processed diagnoses data is indexed by the ICU stay ID.

        :return: The processed diagnoses data.
        :rtype: pl.LazyFrame
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
        valuecol: str = "labresult",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Combine base_excess and base_deficit into one column base_excess_deficit.
        """

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
            )
            # Combine the columns back into a struct again
            .select(
                pl.exclude("value", "source", "method"),
                pl.struct("value", "source", "method").alias(valuecol),
            )
        )

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "labname",
        valuecol: str = "labresult",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the eICU dataset.
        """

        # Convert the lab values to the correct units.
        return (
            data.pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.direct [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.total [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # prefer mg/dL over mmol/L
            # .pipe(
            #     self.convert_blood_urea_nitrogen_mg_dL_to_mmol_L,
            #     itemid="Urea nitrogen [Mass/volume]",
            #     labelcol=labelcol,
            #     valuecol=valuecol,
            # )
            .pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium.ionized [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_CKMB_ng_mL_to_U_L,
                itemid="Creatine kinase.MB [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # NOTE: Experience from clinical practice:
            # Creatinine is more commonly referred to in mg/dL, so this conversion is not necessary
            # .pipe(
            #     self.convert_creatinine_mg_dL_to_umol_L,
            #     itemid="creatinine",
            #     labelcol=labelcol,
            #     valuecol=valuecol,
            # )
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="C reactive protein [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # NOTE: Experience from clinical practice:
            # Glucose is more commonly referred to in mg/dL, so this conversion is not necessary
            # .pipe(
            #     self.convert_glucose_mg_dL_to_mmol_L,
            #     itemid="glucose",
            #     labelcol=labelcol,
            #     valuecol=valuecol,
            # )
            # .pipe(
            #     self.convert_glucose_mg_dL_to_mmol_L,
            #     itemid="glucose_bedside",
            #     labelcol=labelcol,
            #     valuecol=valuecol,
            # )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_magnesium_mg_dL_to_mmol_L,
                itemid="Magnesium [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ng_mL_to_ug_L,
                itemid="Myoglobin [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="Phosphate [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="Albumin [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="Prealbumin [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="Protein [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_T3_ng_dL_to_nmol_L,
                itemid="Triiodothyronine (T3) [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="Thyroxine (T4) [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="Thyroxine (T4) free [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="Troponin I.cardiac [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="Troponin T.cardiac [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron binding capacity [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="Cobalamin (Vitamin B12) [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .with_columns(
                pl.col(labelcol).replace(
                    {
                        "Bilirubin.direct [Mass/volume]": "Bilirubin.direct [Moles/volume]",
                        "Bilirubin.indirect [Mass/volume]": "Bilirubin.indirect [Moles/volume]",
                        "Bilirubin.total [Mass/volume]": "Bilirubin.total [Moles/volume]",
                        # "Urea nitrogen [Mass/volume]": "Urea nitrogen [Moles/volume]",
                        "Calcium [Mass/volume]": "Calcium [Moles/volume]",
                        "Calcium.ionized [Mass/volume]": "Calcium.ionized [Moles/volume]",
                        "Creatine kinase.MB [Mass/volume]": "Creatine kinase.MB [Enzymatic activity/volume]",
                        "Iron [Mass/volume]": "Iron [Moles/volume]",
                        "Iron binding capacity [Mass/volume]": "Iron binding capacity [Moles/volume]",
                        "Magnesium [Mass/volume]": "Magnesium [Moles/volume]",
                        "Phosphate [Mass/volume]": "Phosphate [Moles/volume]",
                        "Triiodothyronine (T3) [Mass/volume]": "Triiodothyronine (T3) [Moles/volume]",
                        "Thyroxine (T4) [Mass/volume]": "Thyroxine (T4) [Moles/volume]",
                        "Thyroxine (T4) free [Mass/volume]": "Thyroxine (T4) free [Moles/volume]",
                        "Cobalamin (Vitamin B12) [Mass/volume]": "Cobalamin (Vitamin B12) [Moles/volume]",
                    }
                )
            )
        )


# endregion
