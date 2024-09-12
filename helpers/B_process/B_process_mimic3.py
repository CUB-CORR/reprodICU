# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.A_extract_mimic3 import MIMIC3Extractor
from helpers.helper import GlobalHelpers
from helpers.helper_unit_conversions import UnitConverter


class MIMIC3Processor(MIMIC3Extractor):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.mimic3_source_path
        self.helpers = GlobalHelpers()
        self.convert = MIMIC3Converter()
        self.icu_stay_id = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.hospital_stay_id_col, self.person_id_col]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region time series
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self):
        # Load the time series data.
        print("MIMIC3 — Loading time series data...")

        ts_vitals = self._process_timeseries_vitals()
        ts_lab = self._process_timeseries_labevents()
        ts_inout = self._process_timeseries_inputoutput()

        # Combine all time series data
        print("MIMIC3 - Combining time series data...")
        return pl.concat([ts_vitals, ts_lab, ts_inout], how="diagonal_relaxed")

    # endregion

    # region vitals
    # Processes the vital data of the MIMIC3 dataset.
    def _process_timeseries_vitals(self):
        """
        Processes the vital data of the MIMIC3 dataset.
        """

        if os.path.isfile(self.precalc_path + "MIMIC3_B_ts_vitals.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "MIMIC3_B_ts_vitals.parquet")

        print("MIMIC3 - Processing vitals data...")

        # Process vitals data
        ts_vitals = (
            self.extract_chartevents()
            # Convert temperature from Fahrenheit to Celsius
            .pipe(
                self.convert.convert_temperature_F_to_C,
                itemid_F="temperature_F",
                itemid_C="temperature_C",
                labelcol="LABEL",
                valuecol="VALUENUM",
            )
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="LABEL",
                index=self.index_cols,
                values="VALUENUM",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(set(ts_vitals.collect_schema().names()) - set(self.index_cols))
        ts_vitals = ts_vitals.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()

        # Save the preprocessed data
        ts_vitals.sink_parquet(self.precalc_path + "MIMIC3_B_ts_vitals.parquet")

        return ts_vitals

    # endregion

    # region lab
    # Processes the lab data of the MIMIC3 dataset.
    def _process_timeseries_labevents(self):
        """
        Processes the lab data of the MIMIC3 dataset.
        """

        if os.path.isfile(self.precalc_path + "MIMIC3_B_ts_lab.parquet"):
            # load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "MIMIC3_B_ts_lab.parquet")

        print("MIMIC3 - Processing lab data...")

        # Process lab data
        ts_lab = (
            self.extract_lab_measurements()
            # Convert the lab values to the correct units
            .pipe(self.convert._convert_lab_values, labelcol="LABEL", valuecol="VALUENUM")
            # Pivot the lab data
            .collect(streaming=True).pivot(
                on="LABEL",
                index=self.index_cols,
                values="VALUENUM",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # drop empty rows
        droplist = list(set(ts_lab.collect_schema().names()) - set(self.index_cols))
        ts_lab = ts_lab.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()
        # Save the preprocessed data
        ts_lab.sink_parquet(self.precalc_path + "MIMIC3_B_ts_lab.parquet")

        return ts_lab

    # endregion

    # region input/output
    # Processes the input/output data of the MIMIC3 dataset.
    def _process_timeseries_inputoutput(self):
        """
        Processes the input/output data of the MIMIC3 dataset.
        """

        if os.path.isfile(self.precalc_path + "MIMIC3_B_ts_inout.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(self.precalc_path + "MIMIC3_B_ts_inout.parquet")

        print("MIMIC3 - Processing inout data...")

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
        droplist = list(set(ts_inout.collect_schema().names()) - set(self.index_cols))
        ts_inout = ts_inout.pipe(self.helpers.dropna, subset=droplist, how="all").lazy().unique()

        # Save the preprocessed data
        ts_inout.sink_parquet(self.precalc_path + "MIMIC3_B_ts_inout.parquet")

        return ts_inout

    # endregion


# region convert
class MIMIC3Converter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self, data, labelcol: str = "LABEL", valuecol: str = "VALUENUM"
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the MIMIC dataset.
        """

        # Convert the lab values to the correct units.
        (
            data.pipe(
                self.convert_ammonia_ug_dL_to_umol_L,
                itemid="ammonia",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="bilirubin_direct",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="bilirubin_indirect",
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
            .pipe(
                self.convert_ng_mL_to_mg_L,
                itemid="d_dimers",
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
            # MCHC is in %, however this is equal to g/dL due to the definition of MCHC
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="phosphate",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            # Potassium is mEq/L, however as a univalent ion, this is equal to mmol/L
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
            # Sodium is mEq/L, however as a univalent ion, this is equal to mmol/L
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
