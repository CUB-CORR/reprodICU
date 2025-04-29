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
        """
        Initializes the UMCdbProcessor instance.

        Args:
            paths: An object containing various source and destination paths.

        Sets:
            self.path: Source path for UMCdb data ({umcdb_source_path}).
            self.helpers: Instance of GlobalHelpers.
            self.convert: Instance of UMCdbConverter.
            self.icu_stay_id: LazyFrame with columns {icu_stay_id_col}, {hospital_stay_id_col}, and {person_id_col}.
            self.icu_length_of_stay: LazyFrame with columns {icu_stay_id_col} and {icu_length_of_stay_col}.
            self.index_cols: List containing {icu_stay_id_col} and {timeseries_time_col}.
        """
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
        Processes and sorts UMCdb time series data.

        Steps:
          1. Verify if a sorted parquet file exists in {precalc_path}.
          2. If it exists, load the data ensuring that the index columns ({icu_stay_id_col} and {timeseries_time_col}) are sorted.
          3. Otherwise, extract numeric data via _process_timeseries_numeric() and listitems via _process_timeseries_listitems().
          4. Join these two datasets on the index columns.
          5. Save the unsorted result, sort it by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.
          6. Return the final LazyFrame.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Measurement time (in seconds).
          - Additional columns: Numeric and listitem measurements.

        Returns:
            pl.LazyFrame: A wide-format LazyFrame sorted by [{icu_stay_id_col}, {timeseries_time_col}].
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

        # Save the preprocessed data
        (
            ts_numeric.join(
                ts_listitems, on=self.index_cols, how="full", coalesce=True
            ).sink_parquet(ts_path_unsorted)
        )

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
        Processes numeric time series data for UMCdb.

        Steps:
          1. Check if a preprocessed numeric file exists in {precalc_path}.
          2. If absent, cache raw numeric data via extract_timeseries_numericitems().
          3. Pivot the cached data on "item" using mean aggregation.
          4. Save the unsorted result, then sort by {icu_stay_id_col} and {timeseries_time_col}.
          5. Remove temporary cache files.

        Columns:
          - {icu_stay_id_col} and {timeseries_time_col}: Index columns.
          - Other columns: Numeric measurements pivoted from the "item" field.

        Returns:
            pl.LazyFrame: A sorted LazyFrame with numeric data.
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
            self.extract_timeseries_numericitems().sink_parquet(
                ts_numeric_path_cache
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
        Processes laboratory time series data for UMCdb.

        Steps:
          1. Check if a preprocessed lab file exists in {precalc_path}.
          2. If absent, cache raw lab data via extract_timeseries_labs().
          3. Convert lab values to canonical units with _convert_lab_values.
          4. JSON encode the "labstruct" field.
          5. Pivot the data on "item" using first-occurrence aggregation.
          6. Align units with _align_units then adjust wide-format values via _convert_wide_lab_values.
          7. Save the unsorted file, sort by {icu_stay_id_col} and {timeseries_time_col}, and delete temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Time of measurement.
          - Other columns: One column per lab test containing JSON-encoded {value}.

        Returns:
            pl.LazyFrame: Sorted laboratory time series data.
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
        Processes listitems time series data for UMCdb.

        Steps:
          1. Check if a previously preprocessed listitems file exists in {precalc_path}.
          2. If absent, extract raw listitems via extract_timeseries_listitems().
          3. Pivot the data on "item", taking the first occurrence for duplicates.
          4. Drop rows where all non-index columns are null.
          5. Save the unsorted file, sort by {icu_stay_id_col} and {timeseries_time_col}, and remove temporary files.

        Columns:
          - {icu_stay_id_col}: ICU stay identifier.
          - {timeseries_time_col}: Measurement time.
          - Additional columns: Listitem measurements pivoted from the "item" field.

        Returns:
            pl.LazyFrame: A sorted wide-format LazyFrame with listitems.
        """
        ts_list_path = self.precalc_path + "UMCdb_timeseries_list.parquet"
        ts_list_path_unsorted = self.precalc_path + "UMCdb_ts_list.parquet"
        ts_list_path_cache = self.precalc_path + "UMCdb_ts_list_cache.parquet"

        if os.path.isfile(ts_list_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_list_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("UMCdb   - Collecting list time series data...")

        # "Cache" the data before pivoting
        if not os.path.isfile(ts_list_path_cache):
            self.extract_timeseries_listitems().sink_parquet(ts_list_path_cache)

        print("UMCdb   - Processing list time series data...")

        # Process list data
        ts_listitems = (
            pl.scan_parquet(ts_list_path_cache)
            # Pivot the list data
            .collect().pivot(
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
        os.remove(ts_list_path_cache)

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
        Converts laboratory values of UMCdb to canonical units.

        Applies a series of conversion operations (chain of .pipe() calls) targeting specific lab tests:
            • {Hematocrit} and {Oxygen saturation}: convert_ratio_to_percentage.
            • {Bilirubin.conjugated} and {Bilirubin.total}: convert_bilirubin_umol_L_to_mg_dL.
            • {Creatinine}: convert_creatinine_mmol_L_to_mg_dL.
            • {Cholesterol in HDL} and {Cholesterol}: convert_cholesterol_mmol_L_to_mg_dL.
            • {Cortisol}: convert_cortisol_nmol_L_to_ug_dL.
            • {Creatine kinase.MB}: convert_CKMB_ng_mL_to_U_L.
            • {Fibrin D-dimer FEU}: convert_FEU_to_DDU.
            • {Fibrinogen}: convert_g_L_to_mg_dL.
            • {Folate}: convert_folate_nmol_L_to_ng_mL.
            • {Glucose}: convert_glucose_mmol_L_to_mg_dL.
            • {Hemoglobin} and {Erythrocyte mean corpuscular hemoglobin concentration}: convert_hemoglobin_mmol_L_to_g_dL.
            • {Microalbumin}: convert_mg_L_to_mg_dL.
            • {Triglyceride}: convert_triglycerides_mmol_L_to_mg_dL.
            • {Troponin T.cardiac}: convert_ug_L_to_ng_L.
            • {Urate}: convert_urate_umol_L_to_mg_dL.
            • {Urea} and {Urea nitrogen}: convert_urea_nitrogen_from_urea and convert_blood_urea_nitrogen_mmol_L_to_mg_dL.

        Args:
            data (pl.LazyFrame): Input raw lab data.
            labelcol (str): Column with lab test identifier (default "variableid").
            valuecol (str): Column holding lab value structure (default "value_struct").
            structfield (str): Field inside the structured value to convert (default "value").

        Returns:
            pl.LazyFrame: LazyFrame with the lab values converted.
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
        Aligns lab unit measurements for {Creatinine} in UMCdb.

        Steps:
          1. Decode the JSON structure in the "Creatinine" column based on a defined schema.
          2. If the "system" field equals "Serum or Plasma", divide the {value} by 1000.
          3. Reassemble the values into a structured field with keys: {value}, {system}, {method}, {time}, and {LOINC}.

        Returns:
            pl.LazyFrame: The LazyFrame with the {Creatinine} column adjusted.
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
                )
                .struct.json_encode()
                .alias("Creatinine"),
            )
        )

    def _convert_wide_lab_values(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Converts wide-format lab values (absolute counts) to relative counts for UMCdb.

        Steps:
          1. Apply conversion of absolute counts for {Basophils}, {Eosinophils}, {Lymphocytes},
             {Monocytes}, {Neutrophils}, {Band form neutrophils}, {Segmented neutrophils} and {Reticulocytes}
             relative to total {Leukocytes} or {Erythrocytes}.

        Columns produced include:
          - "Basophils/100 leukocytes"
          - "Eosinophils/100 leukocytes"
          - "Lymphocytes/100 leukocytes"
          - "Monocytes/100 leukocytes"
          - "Neutrophils/100 leukocytes"
          - "Neutrophils.band form/100 leukocytes"
          - "Neutrophils.segmented/100 leukocytes"
          - "Reticulocytes/100 erythrocytes"

        Returns:
            pl.LazyFrame: Transformed LazyFrame with relative lab counts.
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
