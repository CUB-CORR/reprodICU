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
from helpers.helper_conversions import UnitConverter


class MIMIC3Processor(MIMIC3Extractor):
    def __init__(self, paths, DEMO=False):
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
        Processes the vital data of the MIMIC3 dataset.
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
    # Processes the lab data of the MIMIC3 dataset.
    def process_timeseries_labevents(self):
        """
        Processes the lab data of the MIMIC3 dataset.
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
                valuecol="value_struct",
                structfield="value",
            )
            .with_columns(
                pl.col("value_struct")
                .struct.json_encode()
                .alias("value_struct")
            )
            # Pivot the lab data
            .collect()
            .pivot(
                on="LABEL",
                index=self.index_cols,
                values="value_struct",
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
    # Processes the input/output data of the MIMIC3 dataset.
    def process_timeseries_inputoutput(self):
        """
        Processes the input/output data of the MIMIC3 dataset.
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
class MIMIC3Converter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the MIMIC-III dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "LABEL",
        valuecol: str = "VALUENUM",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the MIMIC dataset.
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
                itemid="Bilirubin.indirect [Mass/volume]",
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
            .pipe(
                self.convert_FEU_to_DDU,
                itemid="Fibrin D-dimer FEU [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ng_mL_to_mg_L,
                itemid="Fibrin D-dimer DDU [Mass/volume]",
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
                self.convert_iron_ug_dL_to_umol_L,
                itemid="Iron binding capacity [Mass/volume]",
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
            # MCHC is in %, however this is equal to g/dL due to the definition of MCHC
            .pipe(
                self.convert_phosphate_mg_dL_to_mmol_L,
                itemid="Phosphate [Mass/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            # Potassium is mEq/L, however as a univalent ion, this is equal to mmol/L
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
            # Sodium is mEq/L, however as a univalent ion, this is equal to mmol/L
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
                        # NOTE: do sth with this
                        # Protein [Mass/time] in 24 hour Urine
                    }
                )
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Convert the lab values of the MIMIC3 dataset.
        """

        return (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Eosinophils/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes [#/volume]",
                total_itemcol="Leukocytes [#/volume]",
                goal_itemcol="Lymphocytes/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes [#/volume]",
                total_itemcol="Erythrocytes [#/volume]",
                goal_itemcol="Reticulocytes/100 erythrocytes",
                structfield="value",
                structstring=True,
            )
        )


# endregion
