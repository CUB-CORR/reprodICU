# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.


import polars as pl
from helpers.helper import GlobalHelpers
from helpers.helper_filepaths import SICdbPaths
from helpers.helper_OMOP import Vocabulary


class SICdbExtractor(SICdbPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.sicdb_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)

        self.other_lab_values = [
            "Bilirubin.direct [Mass/volume]",
            "Bilirubin.total [Mass/volume]",
            "Cobalamin (Vitamin B12) [Mass/volume]",  # in Serum or Plasma",
            "Creatinine [Mass/time]",  # in 24 hour Urine"
            "Iron [Mass/volume]",
            "Anion gap 4",
            "Fractional oxyhemoglobin",
            "Thyroxine (T4) free [Mass/volume]",  # in Serum or Plasma",
            "Band form neutrophils [#/volume]",
            "Basophils [#/volume]",
            "Eosinophils [#/volume]",
            "Lymphocytes [#/volume]",
            "Monocytes [#/volume]",
            "Neutrophils [#/volume]",
            "Neutrophils [#/volume]",
            "Reticulocytes [#/volume]",
        ]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extract patient information from the SICdb source file.

        This function performs the following steps:
            1. Loads ICD diagnosis mapping from CSV.
            2. Renames raw columns to standardized variable names.
            3. Converts and computes values such as weight (g to kg) and lengths of stay.
            4. Adjusts data types for gender, admission type, urgency and location.
            5. Maps admission diagnosis to an APACHE group.
            6. Sorts by {person_id_col} and time offset, then calculates the ICU stay sequence number.

        Returns:
            pl.LazyFrame: A LazyFrame containing the following columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {person_id_col}: Patient identifier.
                - {age_col}: Patient age on admission.
                - {height_col}: Patient height (cm).
                - {weight_col}: Patient weight in kg.
                - {admission_diagnosis_col}: Admission diagnosis mapped to APACHE group.
                - {icu_length_of_stay_col}: ICU length of stay in days.
                - {pre_icu_length_of_stay_col}: Pre-ICU length of stay in days.
                - {hospital_length_of_stay_col}: Hospital length of stay in days.
                - {gender_col}: Patient gender.
                - {admission_type_col}: Admission type.
                - {admission_urgency_col}: Admission urgency.
                - {admission_loc_col}: Admission location.
                - {specialty_col}: Treating specialty.
                - {unit_type_col}: ICU unit type.
                - {discharge_loc_col}: Discharge location.
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_hosp_col}: Hospital mortality flag.
                - {mortality_after_col}: Post-ICU mortality (in days).
                - {care_site_col}: Care site information.
                - {hospital_stay_id_col}: Empty hospital stay identifier.
                - {icu_stay_seq_num_col}: ICU stay sequence number.
                - {icu_time_rel_to_first_col}: Time relative to first ICU admission.
        """
        diagnosis_mapping = (
            pl.read_csv(
                self.mapping_path + "_icd_codes/icd_diagnoses_apache.csv",
                separator=";",
            )
            .select("ICD", "APACHE_Group")
            .to_pandas()
        )
        diagnosis_mapping_dict = dict(
            zip(diagnosis_mapping["ICD"], diagnosis_mapping["APACHE_Group"])
        )

        return (
            pl.scan_csv(self.cases_path)
            .rename(
                {
                    "CaseID": self.icu_stay_id_col,
                    "PatientID": self.person_id_col,
                    "AgeOnAdmission": self.age_col,
                    "HeightOnAdmission": self.height_col,
                    "WeightOnAdmission": self.weight_col,
                    "ICD10Main": self.admission_diagnosis_col,
                    "EstimatedSurvivalObservationTime": (
                        self.mortality_after_cutoff_col
                    ),
                }
            )
            .with_columns(
                # Convert weight to kg from g
                pl.col(self.weight_col)
                .truediv(1000)
                .cast(float)
                .alias(self.weight_col),
                # Convert length of stay to days
                pl.duration(
                    seconds=(pl.col("TimeOfStay") - pl.col("ICUOffset"))
                )
                .truediv(pl.duration(days=1))
                .alias(self.icu_length_of_stay_col),
                # Get approximate pre-ICU length of stay in days
                (
                    pl.duration(
                        days=pl.col("HospitalStayDays")
                        - (pl.col("HospitalDischargeDay"))
                    )
                )
                .truediv(pl.duration(days=1))
                .round(0)
                .alias(self.pre_icu_length_of_stay_col),
                # Get approximate hospital length of stay in days
                pl.col("HospitalStayDays").alias(
                    self.hospital_length_of_stay_col
                ),
                # Convert gender to established dtype
                pl.col("Sex")
                .replace_strict({735: "Male", 736: "Female"}, default="Unknown")
                .cast(self.gender_dtype)
                .alias(self.gender_col),
                # Convert admission type to established dtype
                pl.coalesce(
                    pl.when(pl.col("SurgicalAdmissionType") == 3124)  # Unknown
                    .then(None)
                    .when(
                        pl.col("SurgicalAdmissionType") == 3125
                    )  # Urgent surgery
                    .then(pl.lit("Surgical"))
                    .when(
                        pl.col("SurgicalAdmissionType") == 3126
                    )  # Elective surgery
                    .then(pl.lit("Surgical"))
                    .when(pl.col("SurgicalAdmissionType") == 3127)  # No surgery
                    .then(pl.lit("Medical"))
                    .otherwise(None)
                    .cast(self.admission_types_dtype),
                    pl.col("ReferringUnit")
                    .replace_strict(self._extract_references("ReferringUnit"))
                    .replace_strict(self.ADMISSION_TYPES_MAP, default=None)
                    .cast(self.admission_types_dtype),
                ).alias(self.admission_type_col),
                # Convert admission urgency to established dtype
                pl.when(pl.col("AdmissionUrgency") == 3136)  # Unknown
                .then(pl.lit("Unknown"))
                .when(pl.col("AdmissionUrgency") == 3137)  # Urgent
                .then(pl.lit("Urgent"))
                .when(pl.col("AdmissionUrgency") == 3138)  # Elective
                .then(pl.lit("Elective"))
                .otherwise(None)
                .cast(self.admission_urgency_dtype)
                .alias(self.admission_urgency_col),
                # Convert admission origin to established dtype
                pl.col("ReferringUnit")
                .replace_strict(self._extract_references("ReferringUnit"))
                .replace_strict(self.ADMISSION_LOCATIONS_MAP, default=None)
                .cast(self.admission_locations_dtype)
                .alias(self.admission_loc_col),
                # Convert specialty to established dtype
                pl.col("ReferringUnit")
                .replace_strict(self._extract_references("ReferringUnit"))
                .replace_strict(self.SPECIALTIES_MAP, default=None)
                .cast(self.specialties_dtype)
                .alias(self.specialty_col),
                # Convert unit type to established dtype
                pl.col("HospitalUnit")
                .replace_strict(self._extract_references("HospitalUnit"))
                .replace_strict(self.UNIT_TYPES_MAP, default=None)
                .cast(self.unit_types_dtype)
                .alias(self.unit_type_col),
                # Convert discharge destination to established dtype
                pl.col("DischargeUnit")
                .replace_strict(self._extract_references("DischargeUnit"))
                .replace_strict(self.DISCHARGE_LOCATIONS_MAP, default=None)
                .cast(self.discharge_locations_dtype)
                .alias(self.discharge_loc_col),
                # Convert mortality to established dtype
                pl.when(pl.col("DischargeState") == 2202)  # "lebend"
                .then(False)
                .when(pl.col("DischargeState") == 2215)  # "verstorben"
                .then(True)
                .otherwise(None)  # "Unknown" -> set to None
                .cast(bool)
                .alias(self.mortality_icu_col),
                pl.when(pl.col("HospitalDischargeType") == 2026)  # "Survived"
                .then(False)
                .when(pl.col("HospitalDischargeType") == 2028)  # "Deceased"
                .then(True)
                .otherwise(None)  # "Unknown" -> set to None
                .cast(bool)
                .alias(self.mortality_hosp_col),
                # Convert post ICU discharge mortality to days
                pl.duration(
                    seconds=pl.col("OffsetOfDeath") - pl.col("ICUOffset")
                )
                .truediv(pl.duration(days=1))
                .alias(self.mortality_after_col),
                # Get mortality after discharge cutoff
                (
                    pl.when(self.mortality_after_cutoff_col == 3076)  # 6 Months
                    .then(pl.duration(days=180))
                    .when(self.mortality_after_cutoff_col == 3077)  # 1 Year
                    .then(pl.duration(days=365))
                    .otherwise(pl.duration(days=365))  # Default to 1 year
                    - pl.duration(seconds=pl.col("TimeOfStay"))
                )
                .truediv(pl.duration(days=1))
                .cast(int)
                .alias(self.mortality_after_cutoff_col),
                # Set care site
                pl.lit(
                    "Landeskrankenhaus Salzburg (SALK) - Universitätsklinikum der PMU"
                ).alias(self.care_site_col),
                # Create empty HospitalStayID column
                pl.lit(None).alias(self.hospital_stay_id_col),
                # Convert admission diagnosis to APACHE group
                pl.col(self.admission_diagnosis_col).replace(
                    diagnosis_mapping_dict, default=None
                ),
            )
            # Calculate ICU stay sequence number
            .sort(self.person_id_col, "OffsetAfterFirstAdmission")
            .with_columns(
                (pl.int_range(pl.len()).over(self.person_id_col) + 1).alias(
                    self.icu_stay_seq_num_col
                ),
                # Calculate time since first admission
                pl.col("OffsetAfterFirstAdmission").alias(self.icu_time_rel_to_first_col)
            )
        )

    # endregion

    # region timeseries
    # Extract timeseries information from the data_float_h.csv file
    def extract_timeseries(self) -> pl.LazyFrame:
        """
        Extract timeseries data from the data source file.

        The function executes these steps:
            1. Loads raw timeseries data from CSV.
            2. Joins data with time offsets computed from another source.
            3. Fixes time offsets by subtracting the case offset.
            4. Converts parameter IDs to descriptive names via mapping.
            5. Applies filtering to keep only data within the ICU stay plus the pre-ICU cutoff.
            6. Filters for non-null parameter names and measurement values.
            7. Removes duplicate rows.

        Returns:
            pl.LazyFrame: A LazyFrame containing the columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (in seconds) from ICU admission.
                - "DataID": Mapped parameter name/identifier.
                - "Val": Measurement value.
        """
        print("SICdb   - Extracting timeseries...")
        extracted_references = self._extract_references("RespiratorSetting")
        extracted_references.update(
            self._extract_references("VentilatorConfiguration")
        )
        extracted_references.update(self._extract_references("SignalFloat"))
        extracted_references.update(self._extract_references("Scores"))
        # fix duplicate names (e.g. RespRate both in SignalFloat and RespiratorSetting)
        extracted_references.update({2282: "RespRateVentilator"})

        offsets = self._get_offsets()
        timeseries = (
            pl.scan_csv(self.data_float_h_path)
            .select("CaseID", "Offset", "DataID", "Val")
            .rename({"CaseID": self.icu_stay_id_col})
        )

        return (
            timeseries.join(offsets, on=self.icu_stay_id_col).with_columns(
                # Fix time offset
                (pl.col("Offset") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.timeseries_time_col),
                # Convert parameter IDs to names, then map them
                pl.col("DataID")
                .replace_strict(extracted_references, default=None)
                .replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                    }
                )
                .alias("DataID"),
            )
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col(self.timeseries_time_col)
                > pl.duration(
                    days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                ).dt.total_seconds(),
                pl.col(self.timeseries_time_col)
                < pl.duration(seconds=pl.col("TimeOfStay")).dt.total_seconds(),
            )
            # Filter only relevant timeseries values
            .filter(pl.col("DataID").is_in(self.all_relevant_values))
            # Remove duplicate rows
            .unique()
            # Remove rows with empty parameter names
            .filter(pl.col(self.timeseries_time_col).is_not_null())
            # Remove rows with empty parameter results
            .filter(pl.col("Val").is_not_null())
            # Drop columns
            .drop("CaseOffset", "Offset")
        )

    # region laboratory
    # Extract laboratory information from the laboratory.csv file
    def extract_laboratory_timeseries(self) -> pl.LazyFrame:
        """
        Extract laboratory timeseries data and map LOINC information.

        The process is as follows:
            1. Loads laboratory data from CSV.
            2. Joins with time offsets and LOINC mappings.
            3. Fixes lab time offsets by subtracting CaseOffset.
            4. Filters for data within ICU stay plus pre-ICU cutoff.
            5. Excludes duplicate or null rows.
            6. Constructs a structured lab result field.

        Returns:
            pl.LazyFrame: A LazyFrame with the columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Lab time offset (in seconds) from ICU admission.
                - "LaboratoryName": Name of the laboratory test (potentially altered to LOINC_component).
                - "labstruct": A struct containing:
                      • value: Laboratory measurement value.
                      • system: LOINC system.
                      • method: LOINC method.
                      • time: LOINC time aspect.
                      • LOINC: LOINC code.
        """
        offsets = self._get_offsets()

        LOINC_data = self._extract_references_LOINC()
        labnames = (
            LOINC_data.select("LaboratoryName").unique().to_series().to_list()
        )
        LOINC_data = (
            LOINC_data
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("LaboratoryName")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("LaboratoryName")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames), default=None
                )
                .alias("LOINC_system"),
                pl.col("LaboratoryName")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames), default=None
                )
                .alias("LOINC_method"),
                pl.col("LaboratoryName").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("LaboratoryName")
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
            pl.scan_csv(self.laboratory_path)
            .rename({"CaseID": self.icu_stay_id_col})
            .join(offsets, on=self.icu_stay_id_col)
            .join(LOINC_data, on="LaboratoryID")
            # Fix lab time offset
            .with_columns(
                (pl.col("Offset") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.timeseries_time_col)
            )
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col(self.timeseries_time_col)
                > pl.duration(
                    days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                ).dt.total_seconds(),
                pl.col(self.timeseries_time_col)
                < pl.duration(seconds=pl.col("TimeOfStay")).dt.total_seconds(),
            )
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("LaboratoryName").is_not_null())
            # Remove rows with empty lab results
            .filter(
                pl.col("LaboratoryValue").is_not_null()
                & (pl.col("LaboratoryName") != "")
            )
            # Drop columns
            .drop("CaseOffset", "LaboratoryType")
            # MAKE STRUCT
            .with_columns(pl.col("LOINC_component").alias("LaboratoryName"))
            .with_columns(
                pl.struct(
                    value=pl.col("LaboratoryValue"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "LaboratoryName",
                "labstruct",
            )
        )

    # endregion

    # region medication
    # Extract medication information from the medication.csv file
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract and process medication events from the SICdb source file.

        Steps performed:
            1. Loads medication data from CSV and renames raw columns.
            2. Joins with time offsets.
            3. Adjusts starting and ending time offsets relative to ICU admission.
            4. Maps raw drug IDs to medication names and units.
            5. Converts rates when required and filters out single dose medication rates.
            6. Filters records for valid timepoints and non-null medication values.
            7. Removes duplicate rows.

        Returns:
            pl.LazyFrame: A LazyFrame containing the following columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_name_col}: Original medication identifier.
                - {drug_amount_col}: Drug amount.
                - {drug_rate_col}: Drug administration rate (appropriately converted).
                - {drug_start_col}: Medication start time offset (seconds).
                - {drug_end_col}: Medication end time offset (seconds).
                - {drug_amount_unit_col}: Drug amount unit.
                - {drug_rate_unit_col}: Rate unit for medication.
                - {drug_ingredient_col}: Mapped active drug ingredient.
        """
        print("SICdb   - Extracting medications...")

        offsets = self._get_offsets()

        return (
            pl.scan_csv(self.medication_path)
            .select(
                "id",
                "CaseID",
                "DrugID",
                "Offset",
                "OffsetDrugEnd",
                "IsSingleDose",
                "Amount",
                "AmountPerMinute",
            )
            .rename(
                {
                    "id": self.drug_mixture_admin_id_col,
                    "CaseID": self.icu_stay_id_col,
                    "Amount": self.drug_amount_col,
                    "AmountPerMinute": self.drug_rate_col,
                }
            )
            .join(offsets, on=self.icu_stay_id_col)
            .with_columns(
                # Fix medication time offset
                (pl.col("Offset") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.drug_start_col),
                (pl.col("OffsetDrugEnd") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.drug_end_col),
                # Convert medication IDs to names, then map them
                pl.col("DrugID")
                .replace_strict(self._extract_references("Drug"), default=None)
                .alias(self.drug_name_col),
            )
            .with_columns(
                # Get drug units
                pl.col(self.drug_name_col)
                .replace_strict(self._extract_drug_units(), default=None)
                .alias(self.drug_amount_unit_col),
                # Get drug rate units
                pl.col(self.drug_name_col)
                .replace_strict(self._extract_drug_units(), default=None)
                .str.replace(r"$", "/min")
                .alias(self.drug_rate_unit_col),
            )
            .with_columns(
                # Change rates from grams per minute to milligrams per minute
                pl.when(pl.col(self.drug_rate_unit_col) == "g/min")
                .then(pl.col(self.drug_rate_col) * 1000)
                .otherwise(pl.col(self.drug_rate_col))
                .alias(self.drug_rate_col),
                pl.when(pl.col(self.drug_rate_unit_col) == "g/min")
                .then(pl.lit("mg/min"))
                .otherwise(pl.col(self.drug_rate_unit_col))
                .alias(self.drug_rate_unit_col),
            )
            .with_columns(
                # Drop rates for single dose medications
                pl.when(pl.col("IsSingleDose") == 1)
                .then(None)
                .otherwise(pl.col(self.drug_rate_col))
                .alias(self.drug_rate_col),
                pl.when(pl.col("IsSingleDose") == 1)
                .then(None)
                .otherwise(pl.col(self.drug_rate_unit_col))
                .alias(self.drug_rate_unit_col),
                (pl.col("IsSingleDose") == 0).alias(self.drug_continous_col),
            )
            # Replace drug names with standardized ingredient names
            .join(
                self._extract_drug_references().lazy(), on="DrugID", how="left"
            )
            # Keep only timepoints within timeframe of ICU stay + PRE_ICU_TIMESERIES_DAYS_CUTOFF
            .filter(
                pl.col(self.drug_start_col)
                > pl.duration(
                    days=-self.PRE_ICU_TIMESERIES_DAYS_CUTOFF
                ).dt.total_seconds(),
                pl.col(self.drug_start_col)
                < pl.duration(seconds=pl.col("TimeOfStay")).dt.total_seconds(),
            )
            # Remove duplicate rows
            .unique()
            # Remove rows with empty medication names
            .filter(pl.col(self.drug_name_col).is_not_null())
            # Remove rows with empty medication results
            .filter(pl.col(self.drug_amount_col).is_not_null())
            # Drop columns
            .drop(
                "CaseOffset",
                "TimeOfStay",
                "Offset",
                "OffsetDrugEnd",
            )
        )

    # endregion

    # region diagnosis
    # Extract diagnosis information from the cases.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        """
        Extract diagnosis information from the cases source file.

        The function carries out these steps:
            1. Loads diagnosis data from a CSV.
            2. Renames raw columns to standard variable names.
            3. Cleans diagnosis ICD codes by removing dots.
            4. Adds default columns for diagnosis start time, priority, and ICD version.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {person_id_col}: Patient identifier.
                - {diagnosis_icd_code_col}: Cleaned ICD diagnosis code (without dots).
                - {diagnosis_start_col}: Diagnosis start time (defaulted to 0).
                - {diagnosis_priority_col}: Diagnosis priority (defaulted to 1).
                - {diagnosis_icd_version_col}: ICD version (defaulted to 10).
        """
        print("SICdb   - Extracting diagnoses...")

        return (
            pl.scan_csv(self.cases_path)
            .select("CaseID", "PatientID", "ICD10Main")
            .rename(
                {
                    "CaseID": self.icu_stay_id_col,
                    "PatientID": self.person_id_col,
                    "ICD10Main": self.diagnosis_icd_code_col,
                }
            )
            .with_columns(
                # Remove dot from ICD code
                pl.col(self.diagnosis_icd_code_col).str.replace("\.", ""),
                # Diagnoses are admission diagnoses
                pl.lit(0).alias(self.diagnosis_start_col),
                pl.lit(1).alias(self.diagnosis_priority_col),
                pl.lit(10).alias(self.diagnosis_icd_version_col),
                # Diagnosis descriptions are available, but only in German
            )
            .drop_nulls(self.diagnosis_icd_code_col)
        )

    # region procedures
    # Extract procedure information from the data_range.csv file
    def extract_procedures(self) -> pl.LazyFrame:
        """
        Extract procedure events from the data source and map device identifiers.

        Steps include:
            1. Loads procedure events and joins them with case identifiers.
            2. Renames columns to standard names.
            3. Maps device identifiers using an external mapping function.

        Returns:
            pl.LazyFrame: A LazyFrame containing the following columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {person_id_col}: Patient identifier.
                - "DataID": Procedure device identifier mapped to descriptive text.
                - {procedure_start_col}: Procedure start time offset (in seconds).
                - {procedure_end_col}: Procedure end time offset (in seconds).
                - {procedure_description_col}: Procedure description.
        """
        print("SICdb   - Extracting procedures...")

        IDs = pl.scan_csv(self.cases_path).select("CaseID", "PatientID")

        return (
            pl.scan_csv(self.data_range_path)
            .join(IDs, on="CaseID")
            .select("CaseID", "PatientID", "DataID", "Offset", "OffsetEnd")
            .rename(
                {
                    "PatientID": self.person_id_col,
                    "CaseID": self.icu_stay_id_col,
                    "Offset": self.procedure_start_col,
                    "OffsetEnd": self.procedure_end_col,
                }
            )
            .with_columns(
                pl.col("DataID")
                .replace(
                    self.load_mapping(self.device_mapping_path), default=None
                )
                .alias(self.procedure_description_col),
            )
        )

    # region mappers
    # Extract the information from the d_references.csv file
    def _extract_references(self, ReferenceName: str) -> dict:
        """
        Extract reference mappings for a given reference category.

        Args:
            ReferenceName (str): Category name for which reference mappings are required.

        Returns:
            dict: A dictionary mapping ReferenceGlobalID to ReferenceValue.
        """
        references = (
            pl.read_csv(self.d_references_path)
            .filter(pl.col("ReferenceName") == ReferenceName)
            .select("ReferenceGlobalID", "ReferenceValue")
        )

        return dict(
            zip(
                references["ReferenceGlobalID"].to_numpy(),
                references["ReferenceValue"].to_numpy(),
            )
        )

    def _extract_references_LOINC(self) -> pl.DataFrame:
        """
        Extract LOINC mapping data for laboratory tests.

        Returns:
            pl.DataFrame: A DataFrame containing:
                - "LaboratoryID": Identifier corresponding to ReferenceGlobalID.
                - "LaboratoryName": The long LOINC description.
        """
        return (
            pl.read_csv(self.d_references_path)
            .filter(pl.col("ReferenceName") == "Laboratory")
            .select("ReferenceGlobalID", "LOINC_long")
            .with_columns(
                pl.col("LOINC_long").replace(
                    {  # NOTE: fixing wrong unit
                        "Creatinine [Mass/time]": "Creatinine [Mass/volume]",
                        "Thyroxine (T4) free [Mass/volume]": "Thyroxine (T4) free [Moles/volume]",
                    }
                )
            )
            .unique()
            .rename(
                {
                    "ReferenceGlobalID": "LaboratoryID",
                    "LOINC_long": "LaboratoryName",
                }
            )
        )

    def _extract_drug_units(self) -> dict:
        """
        Extract drug unit mappings for medications.

        Returns:
            dict: A mapping of drug names to standardized unit strings.
        """
        drug_units = (
            pl.read_csv(self.d_references_path)
            .filter(pl.col("ReferenceName") == "Drug")
            .select("ReferenceValue", "ReferenceUnit")
            .with_columns(
                pl.col("ReferenceUnit")
                .str.replace(r"g\\h", "g/hr")
                .str.replace(r"hr\\kg", "kg/hr")
                .alias("ReferenceUnit")
            )
        )

        return dict(
            zip(
                drug_units["ReferenceValue"].to_numpy(),
                drug_units["ReferenceUnit"].to_numpy(),
            )
        )

    # Extract the information from the SICdb.usagi.csv file
    def _extract_drug_references(self) -> dict:
        """
        Extract and process drug references from CSV mapping files.
        """

        return (
            pl.read_csv(self.MEDICATION_MAPPING_PATH + "SICdb.usagi.csv")
            .filter(pl.col("conceptName") != "Unmapped")
            .select("sourceCode", "conceptName")
            .drop_nulls("sourceCode")
            .unique()
            .rename({
                "sourceCode": "DrugID",
                "conceptName": self.drug_ingredient_col,
            })
        )

    # endregion

    # region timehelper
    def _get_offsets(self) -> float:
        """
        Compute time offsets for SICdb cases as basis for time adjustments.

        The function performs the following steps:
            1. Loads offset components (ICUOffset and OffsetAfterFirstAdmission) from CSV.
            2. Renames the CaseID to {icu_stay_id_col}.
            3. Computes the overall offset ("CaseOffset") as the sum of ICUOffset and OffsetAfterFirstAdmission.
            4. Drops unnecessary columns.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - "CaseOffset": The summed offset value.
                - "TimeOfStay": Total duration of ICU stay.
        """
        return (
            pl.scan_csv(self.cases_path)
            .select(
                "CaseID", "ICUOffset", "OffsetAfterFirstAdmission", "TimeOfStay"
            )
            .rename({"CaseID": self.icu_stay_id_col})
            .with_columns(
                (pl.col("OffsetAfterFirstAdmission") + pl.col("ICUOffset"))
                .cast(float)
                .alias("CaseOffset")
            )
            .drop("ICUOffset", "OffsetAfterFirstAdmission")
        )

    # endregion
