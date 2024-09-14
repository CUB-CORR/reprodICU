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
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.eicu_source_path
        self.helpers = GlobalHelpers()
        self.convert = EICUConverter()
        self.icu_stay_id = self.extract_patient_information().select(
            [
                self.icu_stay_id_col,
                self.hospital_stay_id_col,
                self.person_id_col,
            ]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
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
        # Load the time series data.
        print("eICU    - Loading time series data...")
        ts_lab = self._process_timeseries_lab()
        ts_resp = self._process_timeseries_resp()
        ts_nurse = self._process_timeseries_nurse()
        ts_inout = self._process_timeseries_inout()
        ts_periodics = self.extract_and_combine_periodics()

        # Join the time series data on the patient unit stay ID.
        print("eICU    - Joining wide time series data...")
        on = self.index_cols

        # NOTE: Nurse and Periodics are merged first due to duplicate columns
        if not os.path.isfile(
            self.precalc_path + "EICU_B_ts_nurse_periodics.parquet"
        ):
            ts_nurse_periodics = pl.concat(
                [ts_nurse, ts_periodics], how="diagonal_relaxed"
            )

            # Save the preprocessed data
            ts_nurse_periodics.sink_parquet(
                self.precalc_path + "EICU_B_ts_nurse_periodics.parquet"
            )

        else:
            # Load the preprocessed data
            ts_nurse_periodics = pl.scan_parquet(
                self.precalc_path + "EICU_B_ts_nurse_periodics.parquet"
            )

        print("eICU    - Returning wide time series data...")
        return pl.concat(
            [ts_lab, ts_resp, ts_inout, ts_nurse_periodics],
            how="diagonal_relaxed",
        ).unique()

    # endregion

    # region lab
    # Process lab data, i.e. extract and pivot lab data.
    # Keep only the relevant lab values.
    def _process_timeseries_lab(self):
        """
        Process lab data, i.e. extract and pivot lab data.
        Keep only the relevant lab values.

        The processed lab data in wide format is indexed by the ICU stay ID and time.

        :return: The processed lab data in wide format.
        :rtype: pl.LazyFrame
        """

        print("eICU    - Processing lab data...")

        if not os.path.isfile(self.precalc_path + "EICU_B_ts_lab.parquet"):
            ts_lab = (
                self.extract_time_series_lab()
                # Combine base_excess and base_deficit into one column base_excess_deficit
                .pipe(
                    self.convert._combine_base_excess_and_deficit,
                    base_excess_name="base_excess",
                    base_deficit_name="base_deficit",
                    labelcol="labname",
                    valuecol="labresult",
                )
                # Convert the lab values to the correct units
                .pipe(
                    self.convert._convert_lab_values,
                    labelcol="labname",
                    valuecol="labresult",
                )
                # Reverse the base deficit to be negative base excess
                # Pivot the lab values to wide format
                .collect(streaming=True).pivot(
                    on="labname",
                    index=self.index_cols,
                    values="labresult",
                    aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
                )
            )

            # Drop empty rows
            droplist = list(
                set(ts_lab.collect_schema().names()) - set(self.index_cols)
            )
            ts_lab = (
                ts_lab.pipe(self.helpers.dropna, subset=droplist, how="all")
                .unique()
                .lazy()
            )

            # Save the preprocessed data
            ts_lab.sink_parquet(self.precalc_path + "EICU_B_ts_lab.parquet")

        else:
            # Load the preprocessed data
            ts_lab = pl.scan_parquet(
                self.precalc_path + "EICU_B_ts_lab.parquet"
            )

        return ts_lab

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

        print("eICU    - Processing resp data...")

        if not os.path.isfile(self.precalc_path + "EICU_B_ts_resp.parquet"):
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
            droplist = list(
                set(ts_resp.collect_schema().names()) - set(self.index_cols)
            )
            ts_resp = (
                ts_resp.pipe(self.helpers.dropna, subset=droplist, how="all")
                .unique()
                .lazy()
            )

            # Save the preprocessed data
            ts_resp.sink_parquet(self.precalc_path + "EICU_B_ts_resp.parquet")

        else:
            # Load the preprocessed data
            ts_resp = pl.scan_parquet(
                self.precalc_path + "EICU_B_ts_resp.parquet"
            )

        return ts_resp

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

        print("eICU    - Processing nurse charting data...")

        if not os.path.isfile(self.precalc_path + "EICU_B_ts_nurse.parquet"):
            ts_nurse = (
                self.extract_time_series_nurse()
                # Split oxygen_flow / FiO2 into oxygen_flow and FiO2
                .with_columns(
                    pl.when(
                        pl.col("oxygen_delivery_device").is_in(
                            [
                                "ambu_bag",
                                "high_flow_nasal_cannula",
                                "facemask",
                                "nasal cannula",
                                "nebulizer",
                                "non_rebreather_mask",
                            ]
                        )
                        & pl.col("oxygen_flow").is_null()
                    )
                    .then(pl.col("oxygen_flow / FiO2"))
                    .otherwise(pl.col("oxygen_flow"))
                    .alias("oxygen_flow"),
                    pl.when(
                        pl.col("oxygen_delivery_device").is_in(
                            [
                                "BiPAP",
                                "CPAP",
                                "t_piece",
                                "tracheostomy",
                                "ventilator",
                            ]
                        )
                        & pl.col("FiO2").is_null()
                    )
                    .then(pl.col("oxygen_flow / FiO2"))
                    .otherwise(pl.col("FiO2"))
                    .alias("FiO2"),
                )
                .drop("oxygen_flow / FiO2")
                # Pivot the nurse values to wide format
                .collect(streaming=True)
                .pivot(
                    on="nursingchartcelltypevalname",
                    index=self.index_cols,
                    values="nursingchartvalue",
                    aggregate_function="first",  # NOTE: first is used here to not run into issues with strings -> check if this is sensible
                )
            )

            # Drop empty rows
            droplist = list(
                set(ts_nurse.collect_schema().names()) - set(self.index_cols)
            )
            ts_nurse = (
                ts_nurse.pipe(self.helpers.dropna, subset=droplist, how="all")
                .unique()
                .lazy()
            )

            # Save the preprocessed data
            ts_nurse.sink_parquet(self.precalc_path + "EICU_B_ts_nurse.parquet")

        else:
            # Load the preprocessed data
            ts_nurse = pl.scan_parquet(
                self.precalc_path + "EICU_B_ts_nurse.parquet"
            )

        return ts_nurse

    # endregion

    # region inout
    # Process inout data, i.e. extract and pivot intake/output data.
    # Keep only the relevant inout values.
    def _process_timeseries_inout(self):
        """
        Process inout data, i.e. extract and pivot intake/output data.
        Keep only the relevant inout values.

        The processed intake/output data in wide format is indexed by the ICU stay ID and time.

        :return: The processed intake/output data in wide format.
        :rtype: pl.LazyFrame
        """

        print("eICU    - Processing intake/output data...")

        # Process inout data
        print("eICU    - Processing inout data...")
        if not os.path.isfile(self.precalc_path + "EICU_B_ts_inout.parquet"):
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
            droplist = list(
                set(ts_inout.collect_schema().names()) - set(self.index_cols)
            )
            ts_inout = (
                ts_inout.pipe(self.helpers.dropna, subset=droplist, how="all")
                .unique()
                .lazy()
            )

            # Save the preprocessed data
            ts_inout.sink_parquet(self.precalc_path + "EICU_B_ts_inout.parquet")

        else:
            # Load the preprocessed data
            ts_inout = pl.scan_parquet(
                self.precalc_path + "EICU_B_ts_inout.parquet"
            )

        return ts_inout

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
    ) -> pl.LazyFrame:
        """
        Combine base_excess and base_deficit into one column base_excess_deficit.
        """

        return (
            data.with_columns(
                pl.when(
                    pl.col(labelcol) == base_deficit_name,
                )
                .then(pl.col(valuecol) * -1)
                .otherwise(pl.col(valuecol))
                .alias(valuecol),
            )
            # Rename base_excess and base_deficit to base_excess_deficit
            .with_columns(
                pl.when(
                    pl.col(labelcol).is_in(
                        [base_excess_name, base_deficit_name]
                    ),
                )
                .then(pl.lit("base_excess_deficit"))
                .otherwise(pl.col(labelcol))
                .alias(labelcol),
            )
        )

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self, data, labelcol: str = "labname", valuecol: str = "labresult"
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the eICU dataset.
        """

        # Convert the lab values to the correct units.
        (
            data.pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="bilirubin_direct",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="bilirubin_total",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_mg_dL_to_mmol_L,
                itemid="blood_urea_nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="calcium",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_CKMB_ng_mL_to_U_L,
                itemid="CKMB",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            # NOTE: Experience from clinical practice:
            # Creatinine is more commonly referred to in mg/dL, so this conversion is not necessary
            # .pipe(
            #     self.convert_creatinine_mg_dL_to_umol_L,
            #     itemid="creatinine",
            #     labelcol=labelcol,
            #     valuecol=valuecol,
            # )
            # NOTE: Experience from clinical practice:
            # Creatinine is more commonly referred to in mg/L, so this conversion seems necessary
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="CRP",
                labelcol=labelcol,
                valuecol=valuecol,
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
                itemid="iron",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_magnesium_mg_dL_to_mmol_L,
                itemid="magnesium",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_ug_L,
                itemid="myoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="phosphate",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="protein_albumin",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="protein_prealbumin",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="protein_total",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_T3_ng_dL_to_nmol_L,
                itemid="T3",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="T4",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="T4_free",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="troponin_I",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="troponin_T",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="total_iron_binding_capacity",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="vitamin_B12",
                labelcol=labelcol,
                valuecol=valuecol,
            )
        )

        return data


# endregion
