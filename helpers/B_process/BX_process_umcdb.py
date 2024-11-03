# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the UMCdb data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.AX_extract_umcdb import UMCdbExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class UMCdbProcessor(UMCdbExtractor):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.umcdb_source_path
        self.helpers = GlobalHelpers()
        self.convert = UMCdbConverter()
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
        if os.path.isfile(self.precalc_path + "UMCdb_B_timeseries.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "UMCdb_B_timeseries.parquet"
            )

        # Load the time series data.
        print("UMCdb   - Loading time series data...")

        ts_numeric = self._process_timeseries_numeric()
        # ts_listitems = self._process_timeseries_listitems()

        # timeseries = pl.concat([ts_numeric, ts_listitems], how="diagonal_relaxed")
        timeseries = ts_numeric.sort(self.index_cols)
        timeseries.sink_parquet(
            self.precalc_path + "UMCdb_B_timeseries.parquet"
        )
        return timeseries

    def _process_timeseries_numeric(self) -> pl.LazyFrame:
        """
        Process the numeric timeseries data of the UMCdb dataset.
        """

        if os.path.isfile(self.precalc_path + "UMCdb_B_ts_numeric.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "UMCdb_B_ts_numeric.parquet"
            )

        print("UMCdb   - Processing numeric time series data...")

        # Process vitals data
        ts_numeric = (
            self.extract_timeseries_numericitems()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="item",
                valuecol="value",
            )
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
            # Convert the wide lab values to the correct units
            .pipe(self.convert._convert_wide_lab_values)
        )

        # Drop empty rows
        droplist = list(
            set(ts_numeric.collect_schema().names()) - set(self.index_cols)
        )
        ts_numeric = (
            ts_numeric.pipe(
                self.helpers.dropna, subset_cols=droplist, how="all"
            )
            .lazy()
            .unique()
        )

        # Save the preprocessed data
        ts_numeric.sink_parquet(
            self.precalc_path + "UMCdb_B_ts_numeric.parquet"
        )

        return ts_numeric

    def _process_timeseries_listitems(self) -> pl.LazyFrame:
        """
        Process the listitems timeseries data of the UMCdb dataset.
        """

        if os.path.isfile(self.precalc_path + "UMCdb_B_ts_listitems.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "UMCdb_B_ts_listitems.parquet"
            )

        print("UMCdb   - Processing list time series data...")

        # Process vitals data
        ts_listitems = (
            self.extract_timeseries_listitems()
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="item",
                valuecol="value",
            )
            # Pivot the vitals data
            .collect(streaming=True).pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
        )

        # Drop empty rows
        droplist = list(
            set(ts_listitems.collect_schema().names()) - set(self.index_cols)
        )
        ts_listitems = (
            ts_listitems.pipe(
                self.helpers.dropna, subset_cols=droplist, how="all"
            )
            .lazy()
            .unique()
        )

        # Save the preprocessed data
        ts_listitems.sink_parquet(
            self.precalc_path + "UMCdb_B_ts_listitems.parquet"
        )

        return ts_listitems

    # endregion


# region convert
class UMCdbConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    # Convert the lab values of the eICU dataset.
    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "variableid",
        valuecol: str = "value",
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the UMCdb dataset.
        """

        # Convert the lab values to the correct units.
        return (
            data.pipe(
                self.convert_ratio_to_percentage,
                itemid="Hematocrit [Volume Fraction] of Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ratio_to_percentage,
                itemid="Oxygen saturation in Arterial blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.conjugated [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.total [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_mmol_L_to_mg_dL,
                itemid="Creatinine [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_mmol_L_to_mg_dL,
                itemid="Creatinine [Moles/volume] in Urine",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol in HDL [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol in LDL [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_cortisol_nmol_L_to_ug_dL,
                itemid="Cortisol [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_CKMB_ng_mL_to_U_L,
                itemid="Creatine kinase.MB [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_folate_nmol_L_to_ng_mL,
                itemid="Folate [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="Glucose [Moles/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="Hemoglobin [Moles/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="MCHC [Moles/volume]",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_triglycerides_mmol_L_to_mg_dL,
                itemid="Triglyceride [Moles/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_ug_L_to_ng_L,
                itemid="Troponin T.cardiac [Mass/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_urate_umol_L_to_mg_dL,
                itemid="Urate [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea [Moles/volume] in Blood",
                itemid_BUN="Urea nitrogen [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_mmol_L_to_mg_dL,
                itemid="Urea nitrogen [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .with_columns(
                pl.col(labelcol).replace(
                    {
                        "Bilirubin.conjugated [Moles/volume] in Serum or Plasma": "Bilirubin.direct [Mass/volume] in Serum or Plasma",
                        "Billirubin.total [Moles/volume] in Serum or Plasma": "Bilirubin.total [Mass/volume] in Serum or Plasma",
                        "Creatinine [Moles/volume] in Serum or Plasma": "Creatinine [Mass/volume] in Serum or Plasma",
                        "Creatinine [Moles/volume] in Urine": "Creatinine [Mass/volume] in Urine",
                        "Cholesterol in HDL [Moles/volume] in Serum or Plasma": "Cholesterol in HDL [Mass/volume] in Serum or Plasma",
                        "Cholesterol in LDL [Moles/volume] in Serum or Plasma": "Cholesterol in LDL [Mass/volume] in Serum or Plasma",
                        "Cholesterol [Moles/volume] in Serum or Plasma": "Cholesterol [Mass/volume] in Serum or Plasma",
                        "Cortisol [Moles/volume] in Serum or Plasma": "Cortisol [Mass/volume] in Serum or Plasma",
                        "Creatine kinase.MB [Mass/volume] in Serum or Plasma": "Creatine kinase.MB [Enzymatic activity/volume] in Serum or Plasma",
                        "Folate [Moles/volume] in Serum or Plasma": "Folate [Mass/volume] in Serum or Plasma",
                        "Glucose [Moles/volume] in Blood": "Glucose [Mass/volume] in in Serum or Plasma",
                        "Hemoglobin [Moles/volume] in Blood": "Hemoglobin [Mass/volume] in Blood",
                        "MCHC [Moles/volume]": "MCHC [Mass/volume]",
                        "Triglyceride [Moles/volume] in Blood": "Triglyceride [Mass/volume] in Serum or Plasma",
                        "Urate [Moles/volume] in Serum or Plasma": "Urate [Mass/volume] in Serum or Plasma",
                        # NOTE: rename for consistency with other datasets
                        "Carboxyhemoglobin/Hemoglobin.total in Blood": "Carboxyhemoglobin/Hemoglobin.total in Arterial blood",
                        "Methehemoglobin/Hemoglobin.total in Blood": "Methehemoglobin/Hemoglobin.total in Arterial blood",
                        "Oxyhemoglobin/Hemoglobin.total in Blood": "Oxyhemoglobin/Hemoglobin.total in Arterial blood",
                        "Base excess in Blood by calculation": "Base excess in Arterial blood by calculation",
                        "Bicarbonate [Moles/volume] in Blood": "Bicarbonate [Moles/volume] in Arterial blood",
                        "Calcium [Moles/volume] in Serum or Plasma": "Calcium [Moles/volume] in Blood",
                        "Erythrocyte sedimentation rate": "Erythrocyte sedimentation rate by Westergren method",
                        "Ferritin [Mass/volume] in Blood": "Ferritin [Mass/volume] in Serum or Plasma",
                        "Hematocrit [Pure volume fraction] of Blood by Automated count": "Hematocrit [Volume Fraction] of Blood",
                        "INR in Blood by Coagulation assay": "INR in Platelet poor plasma by Coagulation assay",
                        "Lactate [Moles/volume] in Blood": "Lactate [Moles/volume] in Arterial blood",
                        "MCH [Entitic substance]": "MCH [Entitic mass]",
                        "MCV [Entitic volume] by Automated count": "MCV [Entitic volume]",
                        "Neutrophils/100 leukocytes in Blood by Automated count": "Neutrophils/100 leukocytes in Blood",
                        "Carbon dioxide [Partial pressure] in Blood": "Carbon dioxide [Partial pressure] in Arterial blood",
                        "Oxygen [Partial pressure] in Blood": "Oxygen [Partial pressure] in Arterial blood",
                        "Oxygen saturation [Pure mass fraction] in Blood": "Oxygen saturation in Arterial blood",
                        "Transferrin [Mass/volume] in Blood": "Transferrin [Mass/volume] in Serum or Plasma",
                        "Troponin T.cardiac [Mass/volume] in Serum or Plasma": "Troponin T.cardiac [Mass/volume] in Serum or Plasma by High sensitivity method",
                        "Troponin T.cardiac [Mass/volume] in Blood": "Troponin T.cardiac [Mass/volume] in Serum or Plasma by High sensitivity method",
                        "aPTT in Blood by Coagulation assay": "aPTT in Platelet poor plasma by Coagulation assay",
                        "pH of Blood": "pH of Arterial blood",
                        # NOTE: fix wrong units
                        # NOTE: FIXED
                        # "Cobalamin (Vitamin B12) [Mass/volume] in Blood": "Cobalamin (Vitamin B12) [Moles/volume] in Blood",
                    }
                )
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Convert the lab values of the UMCdb dataset.
        """

        return (
            data.pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Band form neutrophils [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Band form neutrophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Basophils [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Basophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Eosinophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils [#/volume] in Blood by Automated count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Eosinophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Eosinophils [#/volume] in Blood by Manual count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Eosinophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Lymphocytes [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Lymphocytes/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Monocytes [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Monocytes/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Neutrophils [#/volume] in Blood by Automated count",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Neutrophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Segmented neutrophils [#/volume] in Blood",
                total_itemcol="Leukocytes [#/volume] in Blood",
                goal_itemcol="Segmented neutrophils/100 leukocytes in Blood",
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes [#/volume] in Blood",
                total_itemcol="Erythrocytes [#/volume] in Blood",
                goal_itemcol="Reticulocytes/100 erythrocytes in Blood",
            )
        )


# endregion
