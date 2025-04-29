# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.


import os.path

import numpy as np
import polars as pl
from helpers.helper import GlobalHelpers
from helpers.helper_filepaths import UMCdbPaths
from helpers.helper_OMOP import Vocabulary


class UMCdbExtractor(UMCdbPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.umcdb_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

        self.other_lab_values = [
            "Bilirubin.conjugated [Moles/volume]",
            "Bilirubin.total [Moles/volume]",
            "Creatinine [Moles/volume]",
            "Cholesterol in HDL [Moles/volume]",
            "Cholesterol in LDL [Moles/volume]",
            "Cholesterol [Moles/volume]",
            "Cortisol [Moles/volume]",
            "Creatine kinase.MB [Mass/volume]",
            "Folate [Moles/volume]",
            "Glucose [Moles/volume]",
            "Hemoglobin [Moles/volume]",
            "MCHC [Moles/volume]",
            "Triglyceride [Moles/volume]",
            "Urate [Moles/volume]",
            "Urea [Moles/volume]",
            "Hematocrit [Pure volume fraction]",
            "MCH [Entitic substance]",
            "Oxygen saturation [Pure mass fraction]",
            "Band form neutrophils [#/volume]",
            "Basophils [#/volume]",
            "Eosinophils [#/volume]",
            "Lymphocytes [#/volume]",
            "Monocytes [#/volume]",
            "Neutrophils [#/volume]",
            "Segmented neutrophils [#/volume]",
            "Reticulocytes [#/volume]",
        ]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extract and transform patient admission information from UMCdb.

        Steps:
            1. Read admission data from a Parquet file at {admissions_path}.
            2. Select and rename columns:
               - "patientid" → {person_id_col}: Patient identifier.
               - "admissionid" → {icu_stay_id_col}: ICU stay identifier.
               - "admissioncount" → {icu_stay_seq_num_col}: ICU admission sequence.
               - Other columns like "agegroup", "weightgroup", "heightgroup" for averages.
            3. Join with APACHE admission data via extract_APACHE_admission().
            4. Convert group values for age, weight, and height to numeric averages.
            5. Transform categorical fields (e.g. {gender_col}, {admission_loc_col}, {discharge_loc_col}, {unit_type_col}).
            6. Compute derived fields:
               - {icu_length_of_stay_col}: ICU length of stay in days.
               - {mortality_after_col}: Days until mortality after discharge.
            7. Set {hospital_stay_id_col} to None and define {care_site_col}.

        Returns:
            pl.LazyFrame: Dataframe with columns:
                - {person_id_col}: Patient ID.
                - {icu_stay_id_col}: ICU stay identifier.
                - {icu_stay_seq_num_col}: ICU stay sequence number.
                - {icu_time_rel_to_first_col}: Time relative to first ICU admission.
                - {age_col}: Patient age.
                - {weight_col}: Patient weight.
                - {height_col}: Patient height.
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_hosp_col}: Hospital mortality flag.
                - {icu_length_of_stay_col}: ICU length of stay in days.
                - {mortality_after_col}: Days until mortality after discharge.
                - {gender_col}: Patient gender.
                - {admission_loc_col}: Admission location.
                - {discharge_loc_col}: Discharge location.
                - {unit_type_col}: ICU unit type.
                - {specialty_col}: Specialty information.
                - {admission_type_col}: Admission type.
                - {admission_urgency_col}: Admission urgency.
                - {hospital_stay_id_col}: Hospital stay identifier.
                - {care_site_col}: Hospital name.
        """

        # calculate mortality after discharge censor cutoff (1 year after last ICU discharge)
        MORTALITY_AFTER_CENSOR_CUTOFF = (
            pl.scan_parquet(self.admissions_path)
            .select("patientid", "dischargedat")
            .rename({"patientid": self.person_id_col})
            .group_by(self.person_id_col)
            .agg(pl.col("dischargedat").max().alias("last_discharge"))
            .with_columns(
                (
                    pl.col("last_discharge")
                    + pl.duration(days=365).dt.total_milliseconds()
                )
                .cast(int)
                .alias("dateofdeathcutoff")
            )
            .select(self.person_id_col, "dateofdeathcutoff")
        )

        return (
            pl.scan_parquet(self.admissions_path)
            .select(
                "patientid",
                "admissionid",
                "admissioncount",
                "agegroup",
                "weightgroup",
                "heightgroup",
                "gender",
                "origin",
                "location",
                "urgency",
                "destination",
                "specialty",
                "admittedat",
                "lengthofstay",
                "dischargedat",
                "dateofdeath",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientid": self.person_id_col,
                    "admissionid": self.icu_stay_id_col,
                    "admissioncount": self.icu_stay_seq_num_col,
                }
            )
            .join(
                self.extract_APACHE_admission(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                MORTALITY_AFTER_CENSOR_CUTOFF, on=self.person_id_col, how="left"
            )
            .with_columns(
                # calculate time since first admission
                pl.col("admittedat")
                .floordiv(1000)
                .alias(self.icu_time_rel_to_first_col),
                # for age, weight and height, assume average of the group
                pl.col("agegroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(
                    lambda s: np.mean([int(i) for i in s if i]),
                    return_dtype=float,
                )
                .cast(int)
                .alias(self.age_col),
                pl.col("weightgroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(
                    lambda s: np.mean([int(i) for i in s if i]),
                    return_dtype=float,
                )
                .cast(int)
                .alias(self.weight_col),
                pl.col("heightgroup")
                .str.replace("-|\+", "–")
                .str.split("–")
                .map_elements(
                    lambda s: np.mean([int(i) for i in s if i]),
                    return_dtype=float,
                )
                .cast(int)
                .alias(self.height_col),
                # Convert categorical mortality to binary
                pl.when(pl.col("destination").is_not_null())
                .then(pl.col("destination") == "Overleden")
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_icu_col),
                pl.when(pl.col("destination") == "Overleden")
                .then(pl.lit(True))
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_hosp_col),
                # NOTE: pre-ICU length of stay is not available in the UMCdb dataset,
                # as there is no known hospital admission / discharge data
                # Calculate ICU length of stay in days
                pl.duration(hours=pl.col("lengthofstay"))
                .truediv(pl.duration(days=1))
                .cast(float)
                .alias(self.icu_length_of_stay_col),
                # Calculate mortality after discharge
                pl.duration(
                    milliseconds=(
                        pl.col("dateofdeath") - pl.col("dischargedat")
                    )
                )
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_col),
                # Calculate mortality after discharge censor cutoff
                pl.duration(
                    milliseconds=(
                        pl.col("dateofdeathcutoff") - pl.col("dischargedat")
                    )
                )
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_cutoff_col),
                # Convert categorical gender to enum
                pl.col("gender")
                .replace_strict(
                    {"Man": "Male", "Vrouw": "Female"}, default="Unknown"
                )
                .cast(self.gender_dtype)
                .alias(self.gender_col),
                # Convert categorical admission location to enum
                pl.col("origin")
                .replace_strict(self.ADMISSION_LOCATIONS_MAP, default="Unknown")
                .cast(self.admission_locations_dtype)
                .alias(self.admission_loc_col),
                # Convert categorical discharge location to enum
                pl.col("destination")
                .replace_strict(self.DISCHARGE_LOCATIONS_MAP, default="Unknown")
                .cast(self.discharge_locations_dtype)
                .alias(self.discharge_loc_col),
                # Convert categorical unit type to enum
                pl.col("location")
                .replace_strict(self.UNIT_TYPES_MAP, default="Unknown")
                .cast(self.unit_types_dtype)
                .alias(self.unit_type_col),
                # Convert categorical specialty to enum
                pl.col("specialty")
                .replace_strict(self.SPECIALTIES_MAP, default="Unknown")
                .cast(self.specialties_dtype)
                .alias(self.specialty_col),
                # Determine Admission Type based on treating specialty
                pl.col("specialty")
                .replace_strict(self.ADMISSION_TYPES_MAP, default=None)
                .cast(self.admission_types_dtype)
                .alias(self.admission_type_col),
                # Convert categorical admission urgency to enum
                pl.col("urgency")
                .cast(str)
                .replace_strict(self.ADMISSION_URGENCY_MAP, default=None)
                .cast(self.admission_urgency_dtype)
                .alias(self.admission_urgency_col),
                # Set hospital stay ID to none
                pl.lit(None).alias(self.hospital_stay_id_col),
                # Set care site to the hospital name
                pl.lit("Amsterdam Universitair Medische Centra").alias(
                    self.care_site_col
                ),
            )
            .drop(
                "agegroup",
                "weightgroup",
                "heightgroup",
                # "gender",
                "origin",
                "destination",
                "specialty",
                "dateofdeath",
                "dischargedat",
                "admittedat",
            )
        )

    # endregion

    # region listitems
    # Extract timeseries information from the listitems.csv file
    def extract_timeseries_listitems(self) -> pl.LazyFrame:
        """
        Extract and process timeseries list items from the UMCdb dataset.

        Steps:
            1. Reads listitems data from a Parquet file and renames columns (e.g. {icu_stay_id_col}).
            2. Joins with a lookup table via _extract_list_references() to map item IDs to standardized names.
            3. Adjusts values for pain and sedation scores.
            4. Computes Glasgow Coma Scale (GCS) using _compute_gcs() and reshapes the data frame.
            5. Filters the data for relevant vital, respiratory, and intake/output items.

        Returns:
            pl.LazyFrame: A LazyFrame with the following columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - "item": Standardized item name.
                - "value": Processed measurement value.
        """

        listitems = (
            pl.scan_parquet(self.listitems_path)
            .select(
                "admissionid",
                "itemid",
                "value",
                "valueid",
                "measuredat",
                "registeredby",
            )
            .rename({"admissionid": self.icu_stay_id_col})
            .join(self._extract_list_references(), on="itemid", how="left")
            .pipe(self._extract_timeseries_helper)
            # Fix the values for RASS and NRS
            .with_columns(
                pl.when(pl.col("item") == "Numeric Pain Rating Scale")
                .then(pl.col("valueid"))
                .when(pl.col("item") == "Richmond agitation-sedation scale")
                .then(5 - pl.col("valueid"))
                .otherwise(pl.col("value"))
                .alias("value"),
            )
            # Fix the values for the mapped listitems
            .with_columns(
                pl.when(pl.col("label") == "Heart rate rhythm")
                .then(
                    pl.col("value").replace_strict(
                        self.HEART_RHYTHM_MAP, default=None
                    )
                )
                .when(pl.col("label") == "Oxygen delivery system")
                .then(
                    pl.col("value").replace_strict(
                        self.OXYGEN_DELIVERY_SYSTEM_MAP, default=None
                    )
                )
                .when(pl.col("label") == "Ventilation mode Ventilator")
                .then(
                    pl.col("value").replace_strict(
                        self.VENTILATOR_MODE_MAP, default=None
                    )
                )
                .otherwise(pl.col("value"))
                .alias("value"),
            )
        )

        gcs = self._compute_gcs(listitems).unpivot(
            index=self.index_cols, variable_name="item", value_name="value"
        )
        listitems = listitems.filter(
            pl.col("item").str.starts_with("Glasgow").not_(),
            pl.col("item").is_in(
                self.relevant_vital_values
                + self.relevant_respiratory_values
                + self.relevant_intakeoutput_values
            ),
        ).drop("valueid", "itemid", "registeredby")

        return pl.concat([listitems, gcs], how="diagonal_relaxed")

    # endregion

    # region numeric
    def extract_timeseries_numericitems(self) -> pl.LazyFrame:
        """
        Extract and filter timeseries numeric items from the dataset.

        Steps:
            1. Invokes helper function _extract_timeseries_numericitems() to read and process numeric data.
            2. Filters the resulting data for specific lists of vital, respiratory, and intake/output measurements.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - "item": Standardized numeric item name.
                - "value": Numeric measurement value.
        """

        return self._extract_timeseries_numericitems().filter(
            pl.col("item").is_in(
                self.relevant_vital_values
                + self.relevant_respiratory_values
                + self.relevant_intakeoutput_values
            )
        )

    # Separate the lab values from the rest
    def extract_timeseries_labs(self) -> pl.LazyFrame:
        """
        Extract timeseries laboratory data from numeric items.

        Steps:
            1. Retrieves numeric lab data via _extract_timeseries_numericitems().
            2. Processes the lab data with _extract_timeseries_labs_helper() to map to LOINC concepts.
            3. Structures the lab details into a single "labstruct" column.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset in seconds from ICU admission.
                - "item": Laboratory test name (mapped from LOINC component).
                - "labstruct": A struct with keys: {value} (lab result), {system} (LOINC system), {method} (LOINC method),
                  {time} (LOINC time aspect), and {LOINC} (LOINC code).
        """

        return self._extract_timeseries_labs_helper(
            self._extract_timeseries_numericitems()
        )

    # Extract timeseries information from the numericitems.csv file
    def _extract_timeseries_numericitems(self) -> pl.LazyFrame:
        """
        Internal helper to extract and process numeric timeseries data from a Parquet file.

        Steps:
            1. Reads numeric items from the file indicated by {numericitems_path}.
            2. Selects required columns and renames {admissionid} to {icu_stay_id_col}.
            3. Joins with numeric references from _extract_numeric_references() to standardize item names.
            4. Processes timeseries data via _extract_timeseries_helper().
            5. Attempts to cast "value" to float.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - "item": Standardized numeric item name.
                - "value": Numeric measurement value.
        """

        return (
            pl.scan_parquet(self.numericitems_path)
            .select("admissionid", "itemid", "value", "measuredat")
            .rename({"admissionid": self.icu_stay_id_col})
            .join(self._extract_numeric_references(), on="itemid", how="left")
            .pipe(self._extract_timeseries_helper)
            # Convert values to numbers, if possible, ignore if not
            .cast({"value": float}, strict=False)
        )

    # endregion

    # region ts helper
    # filter and rename columns for timeseries data
    def _extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Filter and adjust raw timeseries data relative to ICU admission.

        Steps:
            1. Reads admission times from file {admissions_path} and renames columns to 'intime' and 'outtime'.
            2. Joins the provided data with these admission times using {icu_stay_id_col}.
            3. Filters data to retain only records within the ICU stay plus a predefined pre-ICU cutoff.
            4. Computes a time offset {timeseries_time_col} in seconds from ICU admission.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset in seconds from ICU admission.
                - (Other columns from `data` with timeseries alignment applied.)
        """

        intimes = (
            pl.scan_parquet(self.admissions_path)
            .select("admissionid", "admittedat", "dischargedat")
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        return (
            data.join(intimes, on=self.icu_stay_id_col)
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col("measuredat") < pl.col("outtime"))
                & (
                    pl.col("measuredat")
                    > (
                        pl.col("intime")
                        - pl.duration(
                            days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).dt.total_milliseconds()
                    )
                )
            )
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("measuredat") - pl.col("intime"))
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.timeseries_time_col),
            )
            .drop("measuredat", "intime", "outtime")
        )

    # endregion

    # region ts labs
    # Extract lab information from the numericitems.csv file
    def _extract_timeseries_labs_helper(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Process laboratory timeseries data to map numeric items to LOINC concepts.

        Steps:
            1. Reads the lab mapping CSV from {numericitems_lab_mapping_path} and extracts unique lab names.
            2. Creates additional columns mapping lab tests to LOINC component, system, method, time aspect, and code.
            3. Joins the lab mapping with the provided numeric lab data.
            4. Filters the dataframe for labs of interest based on LOINC components and systems.
            5. Creates a structured column "labstruct" that encapsulates detailed lab information.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - "item": Laboratory test name.
                - "labstruct": Struct with keys {value}, {system}, {method}, {time}, and {LOINC}.
        """

        LOINC_data = (
            pl.read_csv(self.numericitems_lab_mapping_path)
            .select("conceptName")
            .rename({"conceptName": "item"})
            .unique()
        )
        labnames = LOINC_data.to_series().to_list()

        LOINC_data = (
            LOINC_data
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("item")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("item")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames), default=None
                )
                .alias("LOINC_system"),
                pl.col("item")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames), default=None
                )
                .alias("LOINC_method"),
                pl.col("item").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("item")
                .replace_strict(
                    self.omop.get_concept_codes_from_names(labnames),
                    default=None,
                )
                .alias("LOINC_code"),
            )
            .with_columns(
                pl.col("LOINC_component")
                .replace_strict(
                    self.relevant_lab_LOINC_systems,
                    return_dtype=pl.List(str),
                    default=None,
                )
                .alias("relevant_LOINC_systems")
            )
            .lazy()
        )

        return (
            data.join(LOINC_data, on="item", how="left")
            # Filter for lab names of interest
            .filter(
                pl.col("LOINC_component").is_in(
                    self.relevant_lab_LOINC_components
                )
            )
            # Filter for systems of interest
            .filter(
                pl.col("LOINC_system").is_in(pl.col("relevant_LOINC_systems"))
            )
            # MAKE STRUCT
            .with_columns(pl.col("LOINC_component").alias("item"))
            .with_columns(
                pl.struct(
                    value=pl.col("value"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "item",
                "labstruct",
            )
        )

    # region gcs
    # compute Glasgow Coma Scale (GCS) from listitems data
    # Implementation based on the SQL query from the AmsterdamUMCdb project
    # https://github.com/AmsterdamUMC/AmsterdamUMCdb/blob/master/amsterdamumcdb/sql/common/legacy/gcs.sql
    def _compute_gcs(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute the Glasgow Coma Scale (GCS) scores from timeseries data.

        Steps:
            1. Determines whether a precomputed GCS file exists and, if so, uses it.
            2. Reads admission times and sorts the data by {icu_stay_id_col} and related time columns.
            3. Filters data separately to compute eye opening, motor, and verbal scores based on item IDs.
            4. Adjusts scores using specific arithmetic transformations.
            5. Aggregates individual scores into a total GCS score and replaces 'registeredby' with a priority order.
            6. Groups the data by {icu_stay_id_col} and selects the first valid scores.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - "Glasgow coma score eye opening": Numeric score for eye opening.
                - "Glasgow coma score motor": Numeric score for motor response.
                - "Glasgow coma score verbal": Numeric score for verbal response.
                - "Glasgow coma score total": Total GCS score.
                - "registeredby": Prioritized identifier for the scorer.
        """

        if os.path.isfile(self.precalc_path + "UMCdb_gcs.parquet"):
            return pl.scan_parquet(self.precalc_path + "UMCdb_gcs.parquet")

        INTIMES = (
            pl.scan_parquet(self.admissions_path)
            .select("admissionid", "admittedat")
            .rename({"admissionid": self.icu_stay_id_col})
        )
        REGISTEREDBY = {
            "ICV_Medisch Staflid": 1,
            "ICV_Medisch": 2,
            "ANES_Anesthesiologie": 3,
            "ICV_Physician assistant": 4,
            "ICH_Neurochirurgie": 5,
            "ICV_IC-Verpleegkundig": 6,
            "ICV_MC-Verpleegkundig": 7,
        }

        data = (
            data.sort(self.index_cols)
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "valueid",
                "itemid",
                "registeredby",
            )
            .join(INTIMES, on=self.icu_stay_id_col)
            .with_columns(
                pl.duration(
                    milliseconds=(
                        pl.col(self.timeseries_time_col) - pl.col("admittedat")
                    )
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            .drop("admittedat")
        )

        data_eye = (
            data.filter(
                pl.col("itemid").is_in(
                    [6732, 13077, 14470, 16628, 19635, 19638]
                )
            )
            .with_columns(
                pl.when(pl.col("itemid") == 6732)
                .then(5 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14470, 16628, 19635]))
                .then(pl.col("valueid") - 4)
                .when(pl.col("itemid") == 19638)
                .then(pl.col("valueid") - 8)
                .otherwise(None)
                .alias("Glasgow coma score eye opening"),
            )
            .drop("itemid", "valueid")
        )

        data_motor = (
            data.filter(
                pl.col("itemid").is_in(
                    [6734, 13072, 14476, 16634, 19636, 19639]
                )
            )
            .with_columns(
                pl.when(pl.col("itemid") == 6734)
                .then(7 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14476, 16634, 19636]))
                .then(pl.col("valueid") - 6)
                .when(pl.col("itemid") == 19639)
                .then(pl.col("valueid") - 12)
                .otherwise(None)
                .alias("Glasgow coma score motor"),
            )
            .drop("itemid", "valueid")
        )

        data_verbal = (
            data.filter(
                pl.col("itemid").is_in(
                    [6735, 13066, 14482, 16640, 19637, 19640]
                )
            )
            .with_columns(
                pl.when(pl.col("itemid") == 6735)
                .then(6 - pl.col("valueid"))
                .when(pl.col("itemid").is_in([14482, 16640]))
                .then(pl.col("valueid") - 5)
                .when(pl.col("itemid") == 19637)
                .then(pl.col("valueid") - 9)
                .when(pl.col("itemid") == 19640)
                .then(pl.col("valueid") - 15)
                .otherwise(None)
                .alias("Glasgow coma score verbal"),
            )
            # handle the special case where the value is <1, which corresponds
            # to intubated patients (assign score 1)
            .with_columns(
                pl.when(pl.col("Glasgow coma score verbal") < 1)
                .then(1)
                .otherwise(pl.col("Glasgow coma score verbal"))
                .alias("Glasgow coma score verbal")
            )
            .drop("itemid", "valueid")
        )

        data_gcs = (
            data_eye.join(data_motor, on=[*self.index_cols + ["registeredby"]])
            .join(data_verbal, on=[*self.index_cols + ["registeredby"]])
            .collect(streaming=True)
        )

        return (
            data_gcs.with_columns(
                (
                    data_gcs.select(
                        "Glasgow coma score eye opening",
                        "Glasgow coma score motor",
                        "Glasgow coma score verbal",
                    ).sum_horizontal(ignore_nulls=False)
                ).alias("Glasgow coma score total"),
                # Replace registeredby with a prioritized order
                pl.col("registeredby")
                .replace_strict(REGISTEREDBY, default=8)
                .alias("registeredby"),
            )
            .group_by(self.index_cols)
            .agg(
                pl.col(
                    "Glasgow coma score eye opening",
                    "Glasgow coma score motor",
                    "Glasgow coma score verbal",
                    "Glasgow coma score total",
                )
                .sort_by("registeredby")
                .first()
            )
            .lazy()
        )

    # endregion

    # region medication
    # Extract medication information from the drugitems.csv file
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract and process medication administration data from UMCdb.

        Steps:
            1. Loads medication mapping files and mapping CSVs.
            2. Reads medication data from a Parquet file specified by {drugitems_path}.
            3. Renames columns (e.g. {icu_stay_id_col}, {drug_name_col}, {drug_start_col}, {drug_end_col}).
            4. Joins with admission times to compute drug start and end times relative to ICU admission.
            5. Converts time values from absolute timestamps to numeric offsets (in seconds).
            6. Standardizes drug names using multiple mapping dictionaries and helper functions.
            7. Creates separate columns for drug amount and rate based on data availability.
            8. Filters out rows with missing values in key columns.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_mixture_admin_id_col}: Mixture administration identifier.
                - {drug_name_col}: Original drug name.
                - {drug_ingredient_col}: Standardized active ingredient.
                - {drug_amount_col}: Drug dose amount (if available).
                - {drug_amount_unit_col}: Unit for the drug amount.
                - {drug_rate_col}: Calculated infusion rate.
                - {drug_rate_unit_col}: Unit for the drug rate.
                - {drug_start_col}: Start time (seconds) relative to ICU admission.
                - {drug_end_col}: End time (seconds) relative to ICU admission.
                - {fluid_amount_col}: Fluid amount administered (if available).
        """

        print("UMCdb   - Extracting medications...")

        # Extract medication mappings by building a chain of references
        # 1. Get drug name references from our mapping files
        drug_references = self._extract_drug_references(return_ids=True)
        concept_ids = drug_references.values()

        # 2. Retrieve active ingredients for these concept IDs
        ingredients = self.omop.get_ingredient(concept_ids)

        # 3. Create a mapping from drug names to their active ingredients
        drug_name_to_ingredient = {}
        for drug_name, concept_id in drug_references.items():
            if concept_id in ingredients:
                drug_name_to_ingredient[drug_name] = ingredients[concept_id]

        # Load additional mappings
        umcdb_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )
        umcdb_drug_class_mapping = self.helpers.load_mapping(
            self.drug_class_mapping_path
        )

        intimes = (
            pl.scan_parquet(self.admissions_path)
            .select("admissionid", "admittedat", "dischargedat")
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        drugitems = (
            pl.scan_parquet(self.drugitems_path)
            .select(
                "admissionid",
                "item",
                "start",
                "stop",
                "orderid",
                "ordercategory",
                "isadditive",
                "administered",
                "administeredunit",
                "doserateperkg",
                "dose",
                "doseunit",
                "doserateunit",
                "solutionitem",
                "solutionadministered",
                "solutionadministeredunit",
            )
            .rename(
                {
                    "admissionid": self.icu_stay_id_col,
                    "item": self.drug_name_col,
                    "start": self.drug_start_col,
                    "stop": self.drug_end_col,
                    "orderid": self.drug_mixture_admin_id_col,
                    "solutionadministered": self.fluid_amount_col,
                }
            )
            .with_columns(
                # Replace drug rate units
                pl.col("doseunit").replace({"µg": "mcg"}),
                pl.col("doserateunit").replace(
                    {"uur": "hr", "dag": "day", "min": "min"}
                ),
            )
            # assign to rate or amount column based on availability
            .with_columns(
                # drug amounts
                pl.col("administered").alias(self.drug_amount_col),
                pl.col("administeredunit").alias(self.drug_amount_unit_col),
                # drug rates
                pl.when(pl.col("doserateunit").is_not_null())
                .then(pl.col("dose"))
                .otherwise(None)
                .alias(self.drug_rate_col),
                pl.when(pl.col("doserateunit").is_not_null())
                .then(
                    pl.concat_str(
                        pl.col("doseunit"),
                        pl.lit("/"),
                        pl.when(pl.col("doserateperkg") == 1)
                        .then(pl.lit("kg/"))
                        .otherwise(pl.lit("")),
                        pl.col("doserateunit"),
                    )
                )
                .otherwise(None)
                .alias(self.drug_rate_unit_col),
            )
            .cast({self.drug_amount_col: float, self.drug_rate_col: float})
        )

        # Separate drug items with and without solution items
        drugitems_with_solutionitem = drugitems.filter(
            pl.col("solutionitem").is_not_null()
        ).rename({"solutionitem": self.fluid_name_col})

        drugitems_without_solutionitem = drugitems.filter(
            pl.col("solutionitem").is_null()
        )

        # Separate fluid infusions from drugs and mixture infusions
        drugitems_without_solutionitem_only_fluid = (
            drugitems_without_solutionitem.filter(
                (pl.col("doseunit") == "ml")
                .all()
                .over(self.drug_mixture_admin_id_col),
                pl.col("ordercategory").str.contains("Infuus"),
            )
            .with_columns(
                # split additive and fluid items
                *[
                    pl.when(pl.col("isadditive") == 0)
                    .then(pl.col(col))
                    .otherwise(None)
                    .alias(col_fluid)
                    for col, col_fluid in zip(
                        [self.drug_name_col, self.drug_amount_col],
                        [self.fluid_name_col, self.fluid_amount_col],
                    )
                ],
                pl.when(pl.col("isadditive") == 0)
                .then(
                    pl.when(pl.col("doserateunit") == "hr")
                    .then(pl.col(self.drug_rate_col))
                    .when(pl.col("doserateunit") == "min")
                    .then(pl.col(self.drug_rate_col) * 60)
                )
                .otherwise(None)
                .alias(self.fluid_rate_col),
                *[
                    pl.when(pl.col("isadditive") == 1)
                    .then(pl.col(col))
                    .otherwise(None)
                    .alias(col)
                    for col in [
                        self.drug_name_col,
                        self.drug_amount_col,
                        self.drug_rate_col,
                    ]
                ],
            )
            .group_by(self.icu_stay_id_col, self.drug_mixture_admin_id_col)
            .max()  # only 2 items (1 fluid, 1 additive) per mixture
        )

        # Handle single drug items without solution items
        drugitems_without_solutionitem_single = (
            drugitems_without_solutionitem.filter(
                (pl.col("doseunit") == "ml")
                .all()
                .over(self.drug_mixture_admin_id_col)
                .not_(),
                pl.col("ordercategory").str.contains("Infuus").not_(),
                pl.col(self.drug_mixture_admin_id_col).is_duplicated().not_(),
            )
        )

        # Handle mixtures with and without fluid items
        drugitems_without_solutionitem_mixtures = (
            drugitems_without_solutionitem.filter(
                (pl.col("doseunit") == "ml")
                .all()
                .over(self.drug_mixture_admin_id_col)
                .not_(),
                pl.col("ordercategory").str.contains("Infuus").not_(),
                pl.col(self.drug_mixture_admin_id_col).is_duplicated(),
            )
            .with_columns(
                # split additive and fluid items
                *[
                    pl.when(pl.col("doseunit") == "ml")
                    .then(pl.col(col))
                    .otherwise(None)
                    .alias(col_fluid)
                    for col, col_fluid in zip(
                        [self.drug_name_col, self.drug_amount_col],
                        [self.fluid_name_col, self.fluid_amount_col],
                    )
                ],
                pl.when(pl.col("doseunit") == "ml")
                .then(
                    pl.when(pl.col("doserateunit") == "hr")
                    .then(pl.col(self.drug_rate_col))
                    .when(pl.col("doserateunit") == "min")
                    .then(pl.col(self.drug_rate_col) * 60)
                )
                .otherwise(None)
                .alias(self.fluid_rate_col),
                *[
                    pl.when(pl.col("doseunit") != "ml")
                    .then(pl.col(col))
                    .otherwise(None)
                    .alias(col)
                    for col in [
                        self.drug_name_col,
                        self.drug_amount_col,
                        self.drug_rate_col,
                    ]
                ],
            )
            .group_by(self.icu_stay_id_col, self.drug_mixture_admin_id_col)
            .agg(
                pl.col(
                    self.drug_start_col,
                    self.drug_end_col,
                    self.fluid_name_col,
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                    "ordercategory",
                ).max(),
                pl.col(
                    self.drug_name_col,
                    self.drug_amount_col,
                    self.drug_amount_unit_col,
                    self.drug_rate_col,
                    self.drug_rate_unit_col,
                ).explode(),
            )
            .explode(
                self.drug_name_col,
                self.drug_amount_col,
                self.drug_amount_unit_col,
                self.drug_rate_col,
                self.drug_rate_unit_col,
            )
        )

        return (
            pl.concat(
                [
                    drugitems_with_solutionitem,
                    drugitems_without_solutionitem_only_fluid,
                    drugitems_without_solutionitem_single,
                    drugitems_without_solutionitem_mixtures,
                ],
                how="diagonal_relaxed",
            )
            .join(intimes, on=self.icu_stay_id_col)
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col(self.drug_start_col) < pl.col("outtime"))
                & (
                    pl.col(self.drug_end_col)
                    > (
                        pl.col("intime")
                        - pl.duration(
                            days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).dt.total_milliseconds()
                    )
                )
            )
            .with_columns(
                # Calculate drug start times relative to ICU admission
                pl.duration(
                    milliseconds=(
                        pl.col(self.drug_start_col) - pl.col("intime")
                    )
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.drug_start_col),
                # Calculate drug end times relative to ICU admission
                pl.duration(
                    milliseconds=(pl.col(self.drug_end_col) - pl.col("intime"))
                )
                .dt.total_seconds()
                .cast(float)
                .alias(self.drug_end_col),
                # Replace drug names with standardized ingredient names
                pl.col(self.drug_name_col)
                .replace_strict(drug_name_to_ingredient, default=None)
                .alias(self.drug_ingredient_col),
                # Replace drug names with OMOP concepts
                pl.col(self.drug_name_col)
                .replace_strict(self._extract_drug_references(), default=None)
                .alias(self.drug_name_OMOP_col),
                # Replace drug administration routes
                pl.col("ordercategory")
                .replace(umcdb_drug_administration_route_mapping, default=None)
                .alias(self.drug_admin_route_col),
                # Replace drug classes
                pl.col("ordercategory")
                .replace(umcdb_drug_class_mapping, default=None)
                .alias(self.drug_class_col),
                # Replace solution items with standardized names
                pl.col(self.fluid_name_col)
                .replace_strict(self.SOLUTION_FLUIDS_MAP, default=None)
                .alias(self.fluid_group_col),
            )
            # Remove duplicate rows
            .unique()
            # Remove rows with empty drug start times
            .filter(pl.col(self.drug_start_col).is_not_null())
            # Remove rows with empty drug names
            .filter(
                pl.col(self.drug_name_col).is_not_null()
                | pl.col(self.fluid_name_col).is_not_null()
            )
            .drop("intime", "outtime")
        )

    # endregion

    # region procedures
    # Extract procedure information from the procedures.csv file
    def extract_procedures(self) -> pl.LazyFrame:
        """
        Extract and process procedure data from UMCdb.

        Steps:
            1. Reads procedure events from two sources: procedure order items and process items.
            2. Retrieves admission times from file and renames columns (e.g. {person_id_col}, {icu_stay_id_col}).
            3. Concatenates both sources of procedure data.
            4. Joins with admission time details to compute procedure start and end times relative to ICU admission.
            5. Filters procedures to include only those within a valid time window.
            6. Replaces procedure IDs with standardized procedure descriptions using a reference mapping.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {procedure_start_col}: Procedure start time (seconds) relative to ICU admission.
                - {procedure_end_col}: Procedure end time (seconds) relative to ICU admission.
                - {procedure_description_col}: Standardized procedure description.
        """

        print("UMCdb   - Extracting procedures...")
        intimes = (
            pl.scan_parquet(self.admissions_path)
            .select("patientid", "admissionid", "admittedat", "dischargedat")
            .rename(
                {
                    "patientid": self.person_id_col,
                    "admissionid": self.icu_stay_id_col,
                    "admittedat": "intime",
                    "dischargedat": "outtime",
                }
            )
        )

        procedureorderitems = (
            pl.scan_parquet(self.procedureorderitems_path)
            .select("admissionid", "itemid", "registeredat")
            .rename(
                {"admissionid": self.icu_stay_id_col, "registeredat": "start"}
            )
        )

        processitems = (
            pl.scan_parquet(self.processitems_path)
            .select("admissionid", "itemid", "start", "stop")
            .rename({"admissionid": self.icu_stay_id_col})
        )

        return (
            pl.concat(
                [procedureorderitems, processitems], how="diagonal_relaxed"
            )
            .join(intimes, on=self.icu_stay_id_col, how="left")
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                (pl.col("start") < pl.col("outtime"))
                & (
                    pl.col("start")
                    > (
                        pl.col("intime")
                        - pl.duration(
                            days=self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                        ).dt.total_milliseconds()
                    )
                )
            )
            .with_columns(
                # Calculate procedure start / end times relative to ICU admission
                pl.duration(milliseconds=(pl.col("start") - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.procedure_start_col),
                pl.duration(milliseconds=(pl.col("stop") - pl.col("intime")))
                .dt.total_seconds()
                .cast(float)
                .alias(self.procedure_end_col),
                # Replace procedure ids with standardized names
                pl.col("itemid")
                .replace_strict(
                    self._extract_procedure_references(), default=None
                )
                .alias(self.procedure_description_col),
            )
            .drop(["start", "stop", "intime", "outtime"])
        )

    # endregion

    # region APACHE
    # Extract APACHE admission information from the listitems.csv file
    def extract_APACHE_admission(self) -> pl.LazyFrame:
        """
        Extract and process APACHE admission diagnoses for ICU patients.

        Steps:
            1. Loads APACHE mapping from file using helpers.load_mapping({apache_mapping_path}).
            2. Reads listitems data and renames {admissionid} to {icu_stay_id_col}.
            3. Uses a series of conditional statements to assign a type identifier based on {itemid}.
            4. Filters, renames, and adjusts diagnosis text and identifiers.
            5. Groups by {icu_stay_id_col} and selects the first record.
            6. Replaces diagnosis values using the APACHE mapping.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {admission_diagnosis_col}: Standardized admission diagnosis.
        """

        APACHE_mapping = self.helpers.load_mapping(self.apache_mapping_path)

        NICE = [18669, 18671]
        LEVEL0_ITEMIDS = [
            13110,  # D_Hoofdgroep
            16651,  # DMC_Hoofdgroep, Medium Care
            18588,  # Apache II Hoofdgroep
            16997,  # APACHE IV Groepen
            18669,  # NICE APACHEII diagnosen
            18671,  # NICE APACHEIV diagnosen
        ]
        LEVEL1_ITEMIDS = [
            13111,  # D_Subgroep_Thoraxchirurgie
            16669,  # DMC_Subgroep_Thoraxchirurgie
            13112,  # D_Subgroep_Algemene chirurgie
            16665,  # DMC_Subgroep_Algemene chirurgie
            13113,  # D_Subgroep_Neurochirurgie
            16667,  # DMC_Subgroep_Neurochirurgie
            13114,  # D_Subgroep_Neurologie
            16668,  # DMC_Subgroep_Neurologie
            13115,  # D_Subgroep_Interne geneeskunde
            16666,  # DMC_Subgroep_Interne geneeskunde
        ]
        SURGICAL_ITEMIDS = [
            13116,  # D_Thoraxchirurgie_CABG en Klepchirurgie
            16671,  # DMC_Thoraxchirurgie_CABG en Klepchirurgie
            13117,  # D_Thoraxchirurgie_Cardio anders
            16672,  # DMC_Thoraxchirurgie_Cardio anders
            13118,  # D_Thoraxchirurgie_Aorta chirurgie
            16670,  # DMC_Thoraxchirurgie_Aorta chirurgie
            13119,  # D_Thoraxchirurgie_Pulmonale chirurgie
            16673,  # DMC_Thoraxchirurgie_Pulmonale chirurgie
            13121,  # D_Algemene chirurgie_Buikchirurgie
            16643,  # DMC_Algemene chirurgie_Buikchirurgie
            13123,  # D_Algemene chirurgie_Endocrinologische chirurgie
            16644,  # DMC_Algemene chirurgie_Endocrinologische chirurgi
            13145,  # D_Algemene chirurgie_KNO/Overige
            16645,  # DMC_Algemene chirurgie_KNO/Overige
            13125,  # D_Algemene chirurgie_Orthopedische chirurgie
            16646,  # DMC_Algemene chirurgie_Orthopedische chirurgie
            13122,  # D_Algemene chirurgie_Transplantatie chirurgie
            16647,  # DMC_Algemene chirurgie_Transplantatie chirurgie
            13124,  # D_Algemene chirurgie_Trauma
            16648,  # DMC_Algemene chirurgie_Trauma
            13126,  # D_Algemene chirurgie_Urogenitaal
            16649,  # DMC_Algemene chirurgie_Urogenitaal
            13120,  # D_Algemene chirurgie_Vaatchirurgie
            16650,  # DMC_Algemene chirurgie_Vaatchirurgie
            13128,  # D_Neurochirurgie _Vasculair chirurgisch
            16661,  # DMC_Neurochirurgie _Vasculair chirurgisch
            13129,  # D_Neurochirurgie _Tumor chirurgie
            16660,  # DMC_Neurochirurgie _Tumor chirurgie
            13130,  # D_Neurochirurgie_Overige
            16662,  # DMC_Neurochirurgie_Overige
            18596,  # Apache II Operatief  Gastr-intenstinaal
            18597,  # Apache II Operatief Cardiovasculair
            18598,  # Apache II Operatief Hematologisch
            18599,  # Apache II Operatief Metabolisme
            18600,  # Apache II Operatief Neurologisch
            18601,  # Apache II Operatief Renaal
            18602,  # Apache II Operatief Respiratoir
            17008,  # APACHEIV Post-operative cardiovascular
            17009,  # APACHEIV Post-operative gastro-intestinal
            17010,  # APACHEIV Post-operative genitourinary
            17011,  # APACHEIV Post-operative hematology
            17012,  # APACHEIV Post-operative metabolic
            17013,  # APACHEIV Post-operative musculoskeletal /skin
            17014,  # APACHEIV Post-operative neurologic
            17015,  # APACHEIV Post-operative respiratory
            17016,  # APACHEIV Post-operative transplant
            17017,  # APACHEIV Post-operative trauma
        ]
        LEVEL2_ITEMIDS = SURGICAL_ITEMIDS + [
            13141,  # D_Algemene chirurgie_Algemeen
            16642,  # DMC_Algemene chirurgie_Algemeen
            13133,  # D_Interne Geneeskunde_Cardiovasculair
            16653,  # DMC_Interne Geneeskunde_Cardiovasculair
            13134,  # D_Interne Geneeskunde_Pulmonaal
            16658,  # DMC_Interne Geneeskunde_Pulmonaal
            13135,  # D_Interne Geneeskunde_Abdominaal
            16652,  # DMC_Interne Geneeskunde_Abdominaal
            13136,  # D_Interne Geneeskunde_Infectieziekten
            16655,  # DMC_Interne Geneeskunde_Infectieziekten
            13137,  # D_Interne Geneeskunde_Metabool
            16656,  # DMC_Interne Geneeskunde_Metabool
            13138,  # D_Interne Geneeskunde_Renaal
            16659,  # DMC_Interne Geneeskunde_Renaal
            13139,  # D_Interne Geneeskunde_Hematologisch
            16654,  # DMC_Interne Geneeskunde_Hematologisch
            13140,  # D_Interne Geneeskunde_Overige
            16657,  # DMC_Interne Geneeskunde_Overige
            13131,  # D_Neurologie_Vasculair neurologisch
            16664,  # DMC_Neurologie_Vasculair neurologisch
            13132,  # D_Neurologie_Overige
            16663,  # DMC_Neurologie_Overige
            13127,  # D_KNO/Overige
            18589,  # Apache II Non-Operatief Cardiovasculair
            18590,  # Apache II Non-Operatief Gastro-intestinaal
            18591,  # Apache II Non-Operatief Hematologisch
            18592,  # Apache II Non-Operatief Metabolisme
            18593,  # Apache II Non-Operatief Neurologisch
            18594,  # Apache II Non-Operatief Renaal
            18595,  # Apache II Non-Operatief Respiratoir
            16998,  # APACHE IV Non-operative cardiovascular
            16999,  # APACHE IV Non-operative Gastro-intestinal
            17000,  # APACHE IV Non-operative genitourinary
            17001,  # APACHEIV  Non-operative haematological
            17002,  # APACHEIV  Non-operative metabolic
            17003,  # APACHEIV Non-operative musculo-skeletal
            17004,  # APACHEIV Non-operative neurologic
            17005,  # APACHEIV Non-operative respiratory
            17006,  # APACHEIV Non-operative transplant
            17007,  # APACHEIV Non-operative trauma
            # # Both NICE APACHEII/IV also count towards surgical if valueid in correct range
            18669,  # NICE APACHEII diagnosen
            18671,  # NICE APACHEIV diagnosen
        ]

        listitems = (
            pl.scan_parquet(self.listitems_path)
            .rename({"admissionid": self.icu_stay_id_col})
            .with_columns(
                pl.when(pl.col("itemid") == 18671)  # NICE APACHEIV diagnosen
                .then(6)
                .when(pl.col("itemid") == 18669)  # NICE APACHEII diagnosen
                .then(5)
                .when(pl.col("itemid").is_between(16998, 17017))  # APACHE IV
                .then(4)
                .when(pl.col("itemid").is_between(18589, 18602))  # Apache II
                .then(3)
                .when(pl.col("itemid").is_between(13116, 13145))  # D_Hoofdgroep
                .then(2)
                .when(
                    pl.col("itemid").is_between(16642, 16673)
                )  # DMC_Hoofdgroep
                .then(1)
                .otherwise(None)
                .cast(int, strict=False)
                .alias("typeid"),
            )
        )

        diagnoses = (
            listitems.filter(pl.col("itemid").is_in(LEVEL2_ITEMIDS))
            .rename(
                {
                    "value": "diagnosis",
                    "valueid": "diagnosis_id",
                }
            )
            .sort(self.icu_stay_id_col, "updatedat", descending=True)
            .with_columns(
                pl.when(pl.col("itemid").is_in(NICE))
                .then(
                    pl.col("diagnosis")
                    .str.replace(" -Coronair", " - Coronair")
                    .str.split(" - ")
                    .list.get(0)
                )
                .otherwise(pl.col("diagnosis"))
                .alias("diagnosis"),
                pl.int_range(pl.len())
                .over(self.icu_stay_id_col)
                .alias("rownum"),
                pl.when(pl.col("itemid").is_in(SURGICAL_ITEMIDS))
                .then(True)
                .when(
                    pl.col("itemid") == 18669,
                    pl.col("diagnosis_id").is_between(1, 26),
                )
                .then(True)
                .when(
                    pl.col("itemid") == 18671,
                    pl.col("diagnosis_id").is_between(222, 452),
                )
                .then(True)
                .otherwise(False)
                .alias("surgical"),
            )
            .cast({"diagnosis": str, "diagnosis_id": str, "surgical": bool})
            .group_by(self.icu_stay_id_col, "typeid", "updatedat")
            .agg(
                pl.col("diagnosis"),
                pl.col("diagnosis_id"),
                pl.col("surgical").first(),
            )
            .explode("diagnosis", "diagnosis_id")
            .with_columns(
                pl.col("typeid")
                .cast(str)
                .replace(
                    {
                        "6": "NICE APACHE IV",
                        "5": "NICE APACHE II",
                        "4": "APACHE IV",
                        "3": "APACHE II",
                        "2": "Legacy ICU",
                        "1": "Legacy MCU",
                    }
                )
                .alias("diagnosis_type"),
            )
            .unique()
            .sort(self.icu_stay_id_col, "typeid", "updatedat", descending=True)
            .with_columns(
                pl.int_range(pl.len())
                .over(self.icu_stay_id_col)
                .alias("rownum")
            )
            .drop("typeid")
        )

        return (
            diagnoses.group_by(self.icu_stay_id_col)
            .first()
            .select(self.icu_stay_id_col, "diagnosis")
            .rename({"diagnosis": self.admission_diagnosis_col})
            .with_columns(
                pl.col(self.admission_diagnosis_col).replace(
                    APACHE_mapping, default=None
                )
            )
        )

    # endregion

    # region references
    # Extract the information from the numericitems_XXX.usagi.csv files
    def _extract_numeric_references(self) -> pl.LazyFrame:
        """
        Extract and process numeric item references from CSV mapping files.

        Steps:
            1. Reads multiple CSV files (lab, other, tag, and unit mappings) using pl.read_csv.
            2. Concatenates the dataframes into one combined reference.
            3. Selects and casts columns such that "sourceCode" becomes int and "conceptName" remains string.
            4. Applies multiple replacement mappings to standardize the concept names.
            5. Drops null values and duplicates.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - "itemid": Numeric item identifier.
                - "item": Standardized item name.
        """

        return (
            pl.concat(
                [
                    pl.read_csv(self.numericitems_lab_mapping_path),
                    pl.read_csv(self.numericitems_other_mapping_path),
                    pl.read_csv(self.numericitems_tag_mapping_path),
                    pl.read_csv(self.numericitems_unit_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceCode", "conceptName")
            .cast({"sourceCode": int}, strict=False)
            .with_columns(
                pl.col("conceptName")
                .replace(
                    {
                        "Tidal volume Ventilator --on ventilator": (
                            "Tidal volume.spontaneous+mechanical --on ventilator"
                        )
                    }
                )
                .replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                    }
                )
            )
            .drop_nulls("sourceCode")
            .unique()
            .rename({"sourceCode": "itemid", "conceptName": "item"})
            .lazy()
        )

    # Extract the information from the listitems_XXX.usagi.csv file
    def _extract_list_references(self) -> pl.LazyFrame:
        """
        Extract and process list item references from CSV mapping files.

        Steps:
            1. Reads two CSV files containing list items mappings.
            2. Concatenates the dataframes using a relaxed diagonal join.
            3. Selects and casts "sourceCode" to int and maps it to "item".
            4. Applies replacement mappings to standardize the concept names.
            5. Drops null values and duplicates.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - "itemid": List item identifier.
                - "item": Standardized list item name.
        """

        return (
            pl.concat(
                [
                    pl.read_csv(self.listitems_item_mapping_path),
                    pl.read_csv(self.listitems_value_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceCode", "conceptName")
            .cast({"sourceCode": int}, strict=False)
            .with_columns(
                pl.col("conceptName").replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                    }
                )
            )
            .drop_nulls("sourceCode")
            .unique()
            .rename({"sourceCode": "itemid", "conceptName": "item"})
            .lazy()
        )

    # Extract the information from the drugitems_XXX.usagi.csv files
    def _extract_drug_references(self, return_ids: bool = False) -> dict:
        """
        Extract and process drug references from CSV mapping files.

        Steps:
            1. Reads CSV files for drug administration routes, drug items, and drug classes.
            2. Concatenates the resulting dataframes.
            3. Selects and drops nulls from "ADD_INFO:source_concept" and "conceptName".
            4. Returns a dictionary mapping source names to concept names.

        Returns:
            dict: A mapping where:
                - Keys: Source names (as strings).
                - Values: Standardized concept names (as strings).
        """

        value_col = "conceptName" if not return_ids else "conceptId"
        references = (
            pl.concat(
                [
                    pl.read_csv(self.drug_administration_route_mapping_path),
                    pl.read_csv(self.drugitems_item_mapping_path),
                    pl.read_csv(self.drug_class_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("ADD_INFO:source_concept", value_col)
            .drop_nulls("ADD_INFO:source_concept")
            .unique()
        )

        return dict(
            zip(
                references["ADD_INFO:source_concept"].to_numpy(),
                references[value_col].to_numpy(),
            )
        )

    # Extract the information from the processitems_item.usagi.csv
    # and procedureorderitems_item.usagi.csv files
    def _extract_procedure_references(self) -> dict:
        """
        Extract and process procedure references from CSV mapping files.

        Steps:
            1. Reads CSV files containing procedure order items and process items mappings.
            2. Concatenates the dataframes.
            3. Selects "sourceCode" and "conceptName", drops nulls, and retains unique mappings.
            4. Returns a dictionary mapping source codes to standardized procedure names.

        Returns:
            dict: A mapping where:
                - Keys: Source codes (as integers).
                - Values: Procedure descriptions (as strings).
        """

        references = (
            pl.concat(
                [
                    pl.read_csv(self.procedureorderitems_item_mapping_path),
                    pl.read_csv(self.processitems_item_mapping_path),
                ],
                how="diagonal_relaxed",
            )
            # .filter(pl.col("equivalence") == "EQUAL")
            .select("sourceCode", "conceptName")
            .drop_nulls("sourceCode")
            .unique()
        )

        return dict(
            zip(
                references["sourceCode"].to_numpy(),
                references["conceptName"].to_numpy(),
            )
        )

    # endregion
