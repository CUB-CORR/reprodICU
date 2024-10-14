# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the eICU data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.A_extract_mimic4 import MIMIC4Extractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class MIMIC4Processor(MIMIC4Extractor):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.mimic3_source_path
        self.helpers = GlobalHelpers()
        self.convert = MIMIC4Converter()
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
    def process_timeseries(self):
        # Load preexisting data if available
        if os.path.isfile(self.precalc_path + "MIMIC4_B_timeseries.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "MIMIC4_B_timeseries.parquet"
            )

        # Load the time series data.
        print("MIMIC4 — Loading time series data...")

        ts_vitals = self._process_timeseries_vitals()
        ts_lab = self._process_timeseries_labevents()
        # ts_inout = self._process_timeseries_inputoutput()

        # Combine all time series data
        print("MIMIC4  - Combining time series data...")
        timeseries = pl.concat(
            [ts_vitals, ts_lab], how="diagonal_relaxed"
        ).sort(self.index_cols)
        timeseries.sink_parquet(
            self.precalc_path + "MIMIC4_B_timeseries.parquet"
        )
        return timeseries

    # endregion

    # region vitals
    # Processes the vital data of the MIMIC4 dataset.
    def _process_timeseries_vitals(self):
        """
        Processes the vital data of the MIMIC4 dataset.
        """
        pl.Config.set_verbose(True)
        if os.path.isfile(self.precalc_path + "MIMIC4_B_ts_vitals.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "MIMIC4_B_ts_vitals.parquet"
            )

        print("MIMIC4  - Processing vitals data...")

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
        ts_vitals.sink_parquet(self.precalc_path + "MIMIC4_B_ts_vitals.parquet")

        return ts_vitals

    # endregion

    # region lab
    # Processes the lab data of the MIMIC4 dataset.
    def _process_timeseries_labevents(self):
        """
        Processes the lab data of the MIMIC4 dataset.
        """

        if os.path.isfile(self.precalc_path + "MIMIC4_B_ts_lab.parquet"):
            # load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "MIMIC4_B_ts_lab.parquet"
            )

        print("MIMIC4  - Processing lab data...")

        # Process lab data
        ts_lab = (
            self.extract_lab_measurements()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="label",
                valuecol="valuenum",
            )
            # Pivot the lab data
            .collect(streaming=True).pivot(
                on="label",
                index=self.index_cols,
                values="valuenum",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # drop empty rows
        ts_lab_cols = ts_lab.collect_schema().names()
        droplist = list(set(ts_lab_cols) - set(self.index_cols))
        ts_lab = (
            ts_lab.lazy()
            .pipe(self.helpers.dropna, "all", droplist, False)
            .unique()
            .sort(self.index_cols)
        )

        # Save the preprocessed data
        ts_lab.sink_parquet(self.precalc_path + "MIMIC4_B_ts_lab.parquet")

        return ts_lab

    # endregion

    # region input/output
    # Processes the input/output data of the MIMIC4 dataset.
    def _process_timeseries_inputoutput(self):
        """
        Processes the input/output data of the MIMIC4 dataset.
        """

        if os.path.isfile(self.precalc_path + "MIMIC4_B_ts_inout.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "MIMIC4_B_ts_inout.parquet"
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
        ts_inout.sink_parquet(self.precalc_path + "MIMIC4_B_ts_inout.parquet")

        return ts_inout

    # endregion

    # region helpers
    # Print the number of unique cases in the timeseries data
    def _print_unique_cases(
        self, data: pl.LazyFrame, name: str
    ) -> pl.LazyFrame:
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
class MIMIC4Converter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the MIMIC-IV dataset.
    def _convert_lab_values(
        self, data, labelcol: str = "label", valuecol: str = "valuenum"
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the MIMIC dataset.
        """

        # Convert the lab values to the correct units.
        (
            data.pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.direct [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.indirect [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_mg_dL_to_umol_L,
                itemid="Bilirubin.total [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_mg_dL_to_mmol_L,
                itemid="Urea nitrogen [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium [Mass/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_calcium_mg_dL_to_mmol_L,
                itemid="Calcium.ionized [Mass/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_CKMB_ng_mL_to_U_L,
                itemid="Creatine kinase.MB [Mass/volume] in Serum or Plasma",
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
                itemid="C reactive protein [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_FEU_to_DDU,
                itemid="Fibrin D-dimer FEU [Mass/volume] in Platelet poor plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_mg_L,
                itemid="Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma",
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
                itemid="Iron [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron binding capacity [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_magnesium_mg_dL_to_mmol_L,
                itemid="Magnesium [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_ug_L,
                itemid="Myoglobin [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            # MCHC is in %, however this is equal to g/dL due to the definition of MCHC
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="Phosphate [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            # Potassium is mEq/L, however as a univalent ion, this is equal to mmol/L
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="Albumin [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_mg_dL_to_mg_L,
                itemid="Prealbumin [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_dL_to_g_L,
                itemid="Protein [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            # Sodium is mEq/L, however as a univalent ion, this is equal to mmol/L
            .pipe(
                self.convert_T3_ng_dL_to_nmol_L,
                itemid="Triiodothyronine (T3) [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="Thyroxine (T4) [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_T4_ug_dL_to_nmol_L_or_ng_dL_to_pmol_L,
                itemid="Thyroxine (T4) free [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="Troponin I.cardiac [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ng_mL_to_ng_L,
                itemid="Troponin T.cardiac [Mass/volume] in Serum or Plasma by High sensitivity method",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_VitB12_pg_mL_to_pmol_L,
                itemid="Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .with_columns(
                pl.col(labelcol).replace(
                    {
                        "Bilirubin.direct [Mass/volume] in Serum or Plasma": "Bilirubin.direct [Moles/volume] in Serum or Plasma",
                        "Bilirubin.indirect [Mass/volume] in Serum or Plasma": "Bilirubin.indirect [Moles/volume] in Serum or Plasma",
                        "Bilirubin.total [Mass/volume] in Serum or Plasma": "Bilirubin.total [Moles/volume] in Serum or Plasma",
                        "Urea nitrogen [Mass/volume] in Serum or Plasma": "Urea nitrogen [Moles/volume] in Serum or Plasma",
                        "Calcium [Mass/volume] in Blood": "Calcium [Moles/volume] in Blood",
                        "Calcium.ionized [Mass/volume] in Blood": "Calcium.ionized [Moles/volume] in Blood",
                        "Creatine kinase.MB [Mass/volume] in Serum or Plasma": "Creatine kinase.MB [Enzymatic activity/volume] in Serum or Plasma",
                        "Iron [Mass/volume] in Serum or Plasma": "Iron [Moles/volume] in Serum or Plasma",
                        "Iron binding capacity [Mass/volume] in Serum or Plasma": "Iron binding capacity [Moles/volume] in Serum or Plasma",
                        "Magnesium [Mass/volume] in Serum or Plasma": "Magnesium [Moles/volume] in Serum or Plasma",
                        "Phosphate [Mass/volume] in Serum or Plasma": "Phosphate [Moles/volume] in Serum or Plasma",
                        "Triiodothyronine (T3) [Mass/volume] in Serum or Plasma": "Triiodothyronine (T3) [Moles/volume] in Serum or Plasma",
                        "Thyroxine (T4) [Mass/volume] in Serum or Plasma": "Thyroxine (T4) [Moles/volume] in Serum or Plasma",
                        "Thyroxine (T4) free [Mass/volume] in Serum or Plasma": "Thyroxine (T4) free [Moles/volume] in Serum or Plasma",
                        "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma": "Cobalamin (Vitamin B12) [Moles/volume] in Serum or Plasma",
                    }
                )
            )
        )

        return data


# endregion
