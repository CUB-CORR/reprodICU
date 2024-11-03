# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import numpy as np
import pandas as pd
import polars as pl
import os

from helpers.A_extract.AX_extract_hirid import HiRIDExtractor
from helpers.helper import GlobalHelpers
from helpers.helper_conversions import UnitConverter


class HiRIDProcessor(HiRIDExtractor):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.convert = HiRIDConverter()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

    # region time series
    # Processes and combines the time series data of the eICU dataset.
    def process_timeseries(self) -> pl.LazyFrame:
        if os.path.isfile(self.precalc_path + "HiRID_B_timeseries.parquet"):
            # Load the preprocessed data
            return pl.scan_parquet(
                self.precalc_path + "HiRID_B_timeseries.parquet"
            )

        print("HiRID   - Processing time series data...")

        # COPY THE NEEDED DATAFRAMES FROM HiRIDExtractor.extract_timeseries() HERE
        # observation_mapping = self.load_mapping(self.observation_mapping_path)
        admissiontime = (
            self._extract_admissions()
            .select([self.icu_stay_id_col, "admissiontime"])
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        if not os.path.isfile(
            self.precalc_path + "HiRID_B_timeseries_unsorted.parquet"
        ):
            # Create an empty DataFrame to store the timeseries data
            timeseries_processed = pl.LazyFrame()

            # Since each case has it's data in only one file, iterating over the files specifically allows
            # for a more efficient processing of the data.
            os_listdir_files = os.listdir(self.timeseries_path)
            counter, counter_max = 0, len(os_listdir_files)
            for file in os.listdir(self.timeseries_path):

                # Update the counter
                counter += 1
                print(
                    f"Processing file {file}... \t{counter} / {counter_max}",
                    end="\r",
                )

                # Process timeseries data
                timeseries = (
                    pl.scan_parquet(self.timeseries_path + file)
                    .pipe(
                        self._extract_timeseries_helper,
                        admissiontime,
                        length_of_stay,
                        # observation_mapping,
                    )
                    # Convert the lab values to the correct units
                    .pipe(
                        self.convert._convert_lab_values,
                        labelcol="variableid",
                        valuecol="value",
                    )
                    # Pivot the timeseries data
                    .collect(streaming=True)
                    .pivot(
                        on="variableid",
                        index=self.index_cols,
                        values="value",
                        aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
                    )
                    # Convert the wide lab values to the correct units
                    .pipe(self.convert._convert_wide_lab_values)
                )

                # Drop empty rows
                droplist = list(
                    set(timeseries.collect_schema().names())
                    - set(self.index_cols)
                )
                timeseries = timeseries.pipe(
                    self.helpers.dropna, "all", droplist, False
                ).unique()

                # Append the data to the DataFrame
                timeseries_processed = pl.concat(
                    [timeseries_processed, timeseries.lazy()],
                    how="diagonal_relaxed",
                )

            # Save the preprocessed data
            timeseries_processed.sink_parquet(
                self.precalc_path + "HiRID_B_timeseries_unsorted.parquet"
            )

        # NOTE: if process stops due to insufficient memory, use the following
        # lines instead within a terminal at the precalc_path:
        # pl.scan_parquet("HiRID_B_timeseries_unsorted.parquet").sort(
        #     "icu_stay_id", "time_relative_to_admission"
        # ).sink_parquet("HiRID_B_timeseries.parquet")
        timeseries = pl.scan_parquet(
            self.precalc_path + "HiRID_B_timeseries_unsorted.parquet"
        ).sort(self.index_cols)
        timeseries.sink_parquet(
            self.precalc_path + "HiRID_B_timeseries.parquet"
        )
        return timeseries

    # endregion


# region convert
class HiRIDConverter(UnitConverter):
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
        Convert the lab values of the HiRID dataset.
        """

        # Convert the lab values to the correct units.
        return (
            data.pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea [Moles/volume] in Venous blood",
                itemid_BUN="Urea nitrogen [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_umol_L_to_mg_dL,
                itemid="Creatinine [Moles/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_creatinine_umol_L_to_mg_dL,
                itemid="Creatinine [Moles/volume] in Urine",
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
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="Glucose [Moles/volume] in Serum or Plasma",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_g_L_to_g_dL,
                itemid="Hemoglobin [Mass/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_g_L_to_g_dL,
                itemid="MCHC [Mass/volume] in Blood",
                labelcol=labelcol,
                valuecol=valuecol,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea [Moles/volume] in Venous blood",
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
                        "Creatinine [Moles/volume] in Blood": "Creatinine [Mass/volume] in Blood",
                        "Creatinine [Moles/volume] in Urine": "Creatinine [Mass/volume] in Urine",
                        "Glucose [Moles/volume] in Serum or Plasma": "Glucose [Mass/volume] in Serum or Plasma",
                        "Urea nitrogen [Moles/volume] in Serum or Plasma": "Urea nitrogen [Mass/volume] in Serum or Plasma",
                        # NOTE: rename for consistency
                        "Alkaline phosphatase [Enzymatic activity/volume] in Blood": "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma",
                        "Creatine kinase panel - Serum or Plasma": "Creatine kinase [Enzymatic activity/volume] in Serum or Plasma",
                        "Erythrocyte sedimentation rate": "Erythrocyte sedimentation rate by Westergren method",
                        "Ferritin [Mass/volume] in Blood": "Ferritin [Mass/volume] in Serum or Plasma",
                        "INR in Blood by Coagulation assay": "INR in Platelet poor plasma by Coagulation assay",
                        "MCHC [Mass/volume] in Cord blood": "MCHC [Mass/volume] in Blood",
                        "Magnesium [Moles/volume] in Blood": "Magnesium [Mass/volume] in Serum or Plasma",
                        "Phosphate [Moles/volume] in Blood": "Phosphate [Mass/volume] in Serum or Plasma",
                        "aPTT in Blood by Coagulation assay": "aPTT in Platelet poor plasma by Coagulation assay",
                        # NOTE: fix wrong unit
                        "Creatine kinase.MB [Mass/volume] in Serum or Plasma": "Creatine kinase.MB [Enzymatic activity/volume] in Serum or Plasma",
                        "Lactate [Mass/volume] in Arterial blood": "Lactate [Moles/volume] in Arterial blood",
                    }
                )
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Convert the lab values of the HiRID dataset.
        """

        return data.pipe(
            self.convert_absolute_count_to_relative,
            itemcol="Lymphocytes [#/volume] in Blood",
            total_itemcol="Leukocytes [#/volume] in Blood",
            goal_itemcol="Lymphocytes/100 leukocytes in Blood",
        )


# endregion
