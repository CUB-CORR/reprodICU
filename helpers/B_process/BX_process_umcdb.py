# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script processes the UMCdb data and stores it in a structured format for further
# processing and harmonization.


import os

import polars as pl
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
    def process_timeseries(self):
        """
        Processes the time series data of the UMCdb dataset.
        """
        ts_path = self.precalc_path + "UMCdb_timeseries.parquet"
        ts_path_unsorted = self.precalc_path + "UMCdb_ts.parquet"

        # Load preexisting data if available
        if os.path.isfile(ts_path):
            return pl.scan_parquet(ts_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        # Load the time series data.
        print("UMCdb   - Loading time series data...")

        ts_numeric = self._process_timeseries_numeric()
        ts_listitems = self._process_timeseries_listitems()

        timeseries = ts_numeric.join(
            ts_listitems, on=self.index_cols, how="full", coalesce=True
        )
        # Save the preprocessed data
        timeseries.collect(streaming=True).write_parquet(ts_path_unsorted)
        # ts_numeric.sink_parquet(ts_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_path)
        )
        os.remove(ts_path_unsorted)

        return pl.scan_parquet(ts_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region numeric
    def _process_timeseries_numeric(self) -> pl.LazyFrame:
        """
        Process the numeric timeseries data of the UMCdb dataset.
        """
        ts_numeric_path = self.precalc_path + "UMCdb_timeseries_numeric.parquet"
        ts_numeric_path_unsorted = (
            self.precalc_path + "UMCdb_ts_numeric.parquet"
        )
        ts_numeric_path_cache = (
            self.precalc_path + "UMCdb_ts_numeric_cache.parquet"
        )

        if os.path.isfile(ts_numeric_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_numeric_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Collecting numeric time series data...")

        # "Cache" the data before pivoting
        if not os.path.isfile(ts_numeric_path_cache):
            (
                self.extract_timeseries_numericitems()
                .collect(streaming=True)
                .write_parquet(ts_numeric_path_cache)
            )

        print("UMCdb   - Processing numeric time series data...")

        # Process numeric data
        ts_numeric = (
            pl.scan_parquet(ts_numeric_path_cache)
            # Pivot the numeric data
            .collect(streaming=True)
            .pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="mean",  # NOTE: mean is used here -> check if this is sensible
            )
            .lazy()
        )

        # Save the preprocessed data
        ts_numeric.sink_parquet(ts_numeric_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_numeric_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_numeric_path)
        )
        os.remove(ts_numeric_path_unsorted)
        os.remove(ts_numeric_path_cache)

        return pl.scan_parquet(ts_numeric_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region labs
    def _process_timeseries_labs(self) -> pl.LazyFrame:
        """
        Process the labs timeseries data of the UMCdb dataset.
        """
        ts_labs_path = self.precalc_path + "UMCdb_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "UMCdb_ts_labs.parquet"
        ts_labs_path_cache = self.precalc_path + "UMCdb_ts_labs_cache.parquet"

        if os.path.isfile(ts_labs_path):
            # load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Collecting lab time series data...")

        # "Cache" the data before pivoting
        if not os.path.isfile(ts_labs_path_cache):
            (
                self.extract_timeseries_labs()
                .collect(streaming=True)
                .write_parquet(ts_labs_path_cache)
            )

        print("UMCdb   - Processing lab time series data...")

        # Process labs data
        ts_labs = (
            pl.scan_parquet(ts_labs_path_cache)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="item",
                valuecol="labstruct",
                structfield="value",
            )
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the labs data
            .collect()
            .pivot(
                on="item",
                index=self.index_cols,
                values="labstruct",
                aggregate_function="first",
            )
            .lazy()
        )

        ts_labs = (
            ts_labs
            # Align the units of the lab values
            .pipe(self.convert._align_units)
            # Convert the wide lab values to the correct units
            .pipe(self.convert._convert_wide_lab_values)
        )

        # Save the preprocessed data
        # ts_labs.sink_parquet(ts_labs_path_unsorted)
        ts_labs.collect(streaming=True).write_parquet(ts_labs_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_labs_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_labs_path)
        )
        os.remove(ts_labs_path_unsorted)
        os.remove(ts_labs_path_cache)

        return pl.scan_parquet(ts_labs_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region listitems
    def _process_timeseries_listitems(self) -> pl.LazyFrame:
        """
        Process the listitems timeseries data of the UMCdb dataset.
        """
        ts_list_path = self.precalc_path + "UMCdb_timeseries_list.parquet"
        ts_list_path_unsorted = self.precalc_path + "UMCdb_ts_list.parquet"

        if os.path.isfile(ts_list_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_list_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Processing list time series data...")

        # Process list data
        ts_listitems = (
            self.extract_timeseries_listitems()
            # Pivot the list data
            .collect(streaming=True).pivot(
                on="item",
                index=self.index_cols,
                values="value",
                aggregate_function="first",
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
        ts_listitems.sink_parquet(ts_list_path_unsorted)

        # Sort the data
        (
            pl.scan_parquet(ts_list_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_list_path)
        )
        os.remove(ts_list_path_unsorted)

        return pl.scan_parquet(ts_list_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

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
        valuecol: str = "value_struct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """
        Convert the lab values of the UMCdb dataset.
        """

        print("UMCdb   - Converting lab values...")

        # Convert the lab values to the correct units.
        return (
            data
            # .with_columns(
            #     pl.col(labelcol).replace(
            #         {
            #             # NOTE: rename for consistency with other datasets
            #             "Hematocrit [Pure volume fraction]": "Hematocrit [Volume Fraction]",
            #             "MCH [Entitic substance]": "MCH [Entitic mass]",
            #             "Oxygen saturation [Pure mass fraction]": "Oxygen saturation",
            #         }
            #     )
            # )
            .pipe(
                self.convert_ratio_to_percentage,
                itemid="Hematocrit",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ratio_to_percentage,
                itemid="Oxygen saturation",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.conjugated",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.total",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_creatinine_mmol_L_to_mg_dL,
                itemid="Creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol in HDL",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cholesterol_mmol_L_to_mg_dL,
                itemid="Cholesterol",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cortisol_nmol_L_to_ug_dL,
                itemid="Cortisol",
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
                self.convert_FEU_to_DDU,
                itemid="Fibrin D-dimer FEU",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="Fibrinogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_folate_nmol_L_to_ng_mL,
                itemid="Folate",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="Glucose",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="Hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_mg_L_to_mg_dL,
                itemid="Microalbumin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_hemoglobin_mmol_L_to_g_dL,
                itemid="Erythrocyte mean corpuscular hemoglobin concentration",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_triglycerides_mmol_L_to_mg_dL,
                itemid="Triglyceride",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_ug_L_to_ng_L,
                itemid="Troponin T.cardiac",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urate_umol_L_to_mg_dL,
                itemid="Urate",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea",
                itemid_BUN="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_mmol_L_to_mg_dL,
                itemid="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
        )

    def _align_units(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Align the units of different sources of the lab values of the UMCdb dataset.
        """

        print("UMCdb   - Aligning lab value units...")

        labstructdtype = pl.Struct(
            [
                pl.Field("value", pl.Float64),
                pl.Field("system", pl.String),
                pl.Field("method", pl.String),
                pl.Field("time", pl.String),
                pl.Field("LOINC", pl.String),
            ]
        )

        return (
            data
            # Creatinine in Serum or Plasma is in umol/L,
            # convert to mmol/L for consistency
            .with_columns(pl.col("Creatinine").str.json_decode(labstructdtype))
            .unnest("Creatinine")
            .with_columns(
                pl.when(pl.col("system") == "Serum or Plasma")
                .then(pl.col("value").truediv(1000))
                .otherwise(pl.col("value"))
                .alias("value")
            )
            .select(
                pl.exclude("value", "system", "method", "time", "LOINC"),
                pl.struct(
                    value="value",
                    system="system",
                    method="method",
                    time="time",
                    LOINC="LOINC",
                ).alias("Creatinine"),
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Convert the lab values of the UMCdb dataset.
        """

        print("UMCdb   - Converting wide lab values...")

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
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Band form neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils.band form/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Segmented neutrophils",
                total_itemcol="Leukocytes",
                goal_itemcol="Neutrophils.segmented/100 leukocytes",
                structfield="value",
                structstring=True,
            )
            .pipe(
                self.convert_absolute_count_to_relative,
                itemcol="Reticulocytes",
                total_itemcol="Erythrocytes",
                goal_itemcol="Reticulocytes/100 erythrocytes",
                structfield="value",
                structstring=True,
            )
        )


# endregion
