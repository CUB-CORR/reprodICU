# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.


import polars as pl

from ..helper import GlobalHelpers
from ..helper_filepaths import SICdbPaths
from ..helper_OMOP import Vocabulary


class SICdbExtractor(SICdbPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.sicdb_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extract and transform patient demographics and clinical information.

        Steps:
            1. Load ICD diagnosis to APACHE group mapping.
            2. Read cases CSV and rename columns to standardized names.
            3. Convert weight from grams to kilograms.
            4. Compute ICU, pre-ICU, and hospital lengths of stay.
            5. Map categorical fields (gender, admission type, urgency, location).
            6. Compute mortality flags based on discharge status.
            7. Map admission diagnosis to APACHE group.
            8. Sort by patient ID and compute ICU stay sequence number.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {person_id_col}: Patient identifier.
                - {icu_stay_seq_num_col}: ICU stay sequence number.
                - {icu_time_rel_to_first_col}: Time relative to first ICU admission (seconds).
                - {age_col}: Patient age (years).
                - {height_col}: Patient height (cm).
                - {weight_col}: Patient weight (kg).
                - {gender_col}: Patient gender.
                - {admission_year_col}: Admission year.
                - {admission_type_col}: Admission type.
                - {admission_urgency_col}: Admission urgency.
                - {admission_loc_col}: Admission location.
                - {specialty_col}: Treating specialty.
                - {unit_type_col}: ICU unit type.
                - {discharge_loc_col}: Discharge location.
                - {admission_diagnosis_col}: Admission diagnosis mapped to APACHE group.
                - {pre_icu_length_of_stay_col}: Pre-ICU length of stay (days).
                - {icu_length_of_stay_col}: ICU length of stay (days).
                - {hospital_length_of_stay_col}: Hospital length of stay (days).
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_hosp_col}: Hospital mortality flag.
                - {mortality_after_col}: Days between discharge and death.
                - {mortality_after_cutoff_col}: Days from discharge to cutoff.
                - {care_site_col}: Care site identifier.
                - {hospital_stay_id_col}: Hospital stay identifier (always null).
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
                    "AdmissionYear": self.admission_year_col,
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
                pl.col("OffsetAfterFirstAdmission").alias(
                    self.icu_time_rel_to_first_col
                ),
            )
        )

    # endregion

    # region timeseries
    def _extract_timeseries_helper(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Process and align raw timeseries data relative to ICU admission.

        Steps:
            1. Join data with time offsets computed from case information.
            2. Adjust time offsets relative to case offset (ICU admission).
            3. Join with timeseries variable mappings.
            4. Filter for data within ICU stay plus pre-ICU cutoff.
            5. Filter for relevant timeseries variables.
            6. Remove duplicates and null values.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - variable name: Mapped variable identifier.
                - value: Measurement value.
        """
        return (
            data.join(self._get_offsets(), on=self.icu_stay_id_col)
            .with_columns(
                # Fix time offset
                (pl.col("Offset") - pl.col("CaseOffset"))
                .cast(float)
                .alias(self.timeseries_time_col),
            )
            .join(
                self._get_timeseries_mapping(),
                on="DataID",
                how="left",
                coalesce=True,
            )
            .drop("DataID")
            .rename({"DataName": "DataID"})
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
        Extract laboratory measurements and map to LOINC concepts.

        Steps:
            1. Load lab value mappings with LOINC information.
            2. Extract unique laboratory test names and derive LOINC components.
            3. Scan laboratory CSV and align data with ICU admission time.
            4. Join with LOINC mappings and filter for relevant labs.
            5. Create struct column with LOINC details and lab result.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds) from ICU admission.
                - laboratory name: Lab test name (mapped to LOINC component).
                - labstruct: Struct with value, system, method, time, LOINC code.
        """
        offsets = self._get_offsets()

        LOINC_data = self._extract_references_LOINC().lazy()
        labnames = (
            LOINC_data.select("LaboratoryName")
            .unique()
            .collect()
            .to_series()
            .to_list()
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
                    self.omop.get_lab_system_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_system"),
                pl.col("LaboratoryName")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames),
                    default=None,
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
        )

        return (
            pl.scan_csv(self.laboratory_path)
            .rename({"CaseID": self.icu_stay_id_col})
            .join(offsets, on=self.icu_stay_id_col)
            .join(LOINC_data, on="LaboratoryID", how="left")
            # Filter for lab names of interest
            .filter(
                pl.col("LOINC_component").is_in(
                    self.relevant_lab_LOINC_components
                )
            )
            # Filter for systems of interest
            .filter(
                pl.col("LOINC_system").is_in(
                    pl.col("LOINC_component").replace_strict(
                        self.relevant_lab_LOINC_systems,
                        return_dtype=pl.List(str),
                        default=None,
                    )
                )
            )
            .with_columns(
                # Mark only as arterial blood if LaboratoryType explicitly indicates so
                pl.when(pl.col("LOINC_system") == "Blood arterial")
                .then(
                    pl.when(pl.col("LaboratoryType") == 2296)
                    .then(pl.lit("Blood arterial"))
                    .otherwise(pl.lit("Blood"))
                )
                .otherwise(pl.col("LOINC_system"))
                .alias("LOINC_system")
            )
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
            .filter(
                pl.col("LaboratoryName").is_not_null(),
                pl.col("LOINC_component").is_not_null(),
            )
            # Remove rows with empty lab results
            .filter(
                pl.col("LaboratoryValue").is_not_null()
                & (pl.col("LaboratoryName") != "")
            )
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
                "LaboratoryID",
                "LaboratoryName",
                "labstruct",
            )
        )

    # endregion

    # region medication
    # Extract medication information from the medication.csv file
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extract medication administration events.

        Steps:
            1. Read medication data from CSV.
            2. Align data with ICU admission time.
            3. Map drug IDs to standardized medication names and units.
            4. Compute and filter infusion rates for continuous medications.
            5. Map medications to active ingredients.
            6. Filter for valid medication entries within ICU stay window.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_mixture_admin_id_col}: Medication event identifier.
                - {drug_name_col}: Medication name.
                - {drug_ingredient_col}: Active drug ingredient.
                - {drug_amount_col}: Drug amount.
                - {drug_amount_unit_col}: Amount unit.
                - {drug_rate_col}: Administration rate.
                - {drug_rate_unit_col}: Rate unit.
                - {drug_start_col}: Relative start time (seconds).
                - {drug_end_col}: Relative end time (seconds).
                - {drug_admin_type_col}: Administration type.
                - {drug_continuous_col}: Continuous administration flag.
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
                # Add a column to indicate the administration type
                pl.lit("given").alias(self.drug_admin_type_col),
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
                (pl.col("IsSingleDose") == 0).alias(self.drug_continuous_col),
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
        Extract diagnosis information from cases CSV.

        Steps:
            1. Read cases CSV with relevant ICD columns.
            2. Rename columns to standardized names.
            3. Remove dots from ICD codes for standardization.
            4. Set diagnosis start time, priority, and ICD version defaults.
            5. Remove null diagnosis codes.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {person_id_col}: Patient identifier.
                - {diagnosis_icd_code_col}: ICD diagnosis code (cleaned).
                - {diagnosis_start_col}: Diagnosis start time (default: 0).
                - {diagnosis_priority_col}: Diagnosis priority (default: 1).
                - {diagnosis_icd_version_col}: ICD version (default: 10).
        """
        print("SICdb   - Extracting diagnoses...")

        return (
            pl.scan_csv(self.cases_path)
            .select("CaseID", "PatientID", "ICD10Main", "ICD10MainText")
            .rename(
                {
                    "CaseID": self.icu_stay_id_col,
                    "PatientID": self.person_id_col,
                    "ICD10Main": self.diagnosis_icd_code_col,
                    "ICD10MainText": self.diagnosis_description_col,
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
        Extract procedure events from data_range CSV.

        Steps:
            1. Read procedure events from data_range CSV.
            2. Join with case identifiers.
            3. Map procedure device identifiers using reference mappings.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {person_id_col}: Patient identifier.
                - {procedure_start_col}: Procedure start time (seconds) from ICU admission.
                - {procedure_end_col}: Procedure end time (seconds) from ICU admission.
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
        Extract reference mappings for a given category.

        Args:
            ReferenceName (str): Category name for reference mappings.

        Returns:
            dict: Mapping from ReferenceGlobalID to ReferenceValue.
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
        Extract LOINC mapping for laboratory tests.

        Returns:
            pl.DataFrame: Columns:
                - LaboratoryID: Laboratory identifier.
                - LaboratoryName: LOINC long description.
        """
        return (
            pl.read_csv(self.d_references_path)
            .filter(pl.col("ReferenceName") == "Laboratory")
            .select("ReferenceGlobalID", "LOINC_long")
            .drop_nulls("LOINC_long")
            .with_columns(
                pl.col("LOINC_long").replace(
                    {  # NOTE: fixing wrong unit
                        "Creatinine [Mass/time]": "Creatinine [Mass/volume]",
                        "Thyroxine (T4) free [Mass/volume]": "Thyroxine (T4) free [Moles/volume]",
                        "Hematocrit [Volume Fraction] of Arterial blood": "Hematocrit [Volume Fraction] of Blood by Automated count"
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
            .with_columns(
                pl.col("LaboratoryName")
                # "/100 leukocytes" obselete in v20250827
                # -> now without "/100", kept for compatibility and conversion
                .str.replace("/100 leukocytes", "/Leukocytes")
                .str.replace("/100 erythrocytes", "/Erythrocytes")
            )
        ) # fmt: skip

    def _extract_drug_units(self) -> dict:
        """
        Extract drug unit mappings for medications.

        Returns:
            dict: Mapping from drug name to standardized unit string.
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
            .rename(
                {
                    "sourceCode": "DrugID",
                    "conceptName": self.drug_ingredient_col,
                }
            )
        )

    # endregion

    # region timehelper
    def _get_offsets(self) -> float:
        """
        Compute time offsets for cases.

        Steps:
            1. Read offset components from cases CSV.
            2. Rename CaseID to standardized name.
            3. Compute CaseOffset as sum of ICUOffset and OffsetAfterFirstAdmission.

        Returns:
            pl.LazyFrame: Columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - CaseOffset: Computed offset value.
                - TimeOfStay: Total ICU stay duration.
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

    def _get_timeseries_mapping(self) -> pl.LazyFrame:
        """
        Map timeseries data identifiers to descriptive names.

        Returns:
            pl.LazyFrame: Columns:
                - DataID: Original identifier.
                - DataName: Mapped descriptive name.
        """

        extracted_references = {
            **self._extract_references("RespiratorSetting"),
            **self._extract_references("VentilatorConfiguration"),
            **self._extract_references("SignalFloat"),
            **self._extract_references("Scores"),
        }
        # fix duplicate names (e.g. RespRate both in SignalFloat and RespiratorSetting)
        extracted_references.update({2282: "RespRateVentilator"})

        # Convert parameter IDs to names, then map them
        return (
            pl.read_csv(self.d_references_path)
            .select("ReferenceGlobalID")
            .with_columns(
                pl.col("ReferenceGlobalID")
                .replace_strict(extracted_references, default=None)
                .replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                        **self.timeseries_extracorporeal_mapping,
                    }
                )
                .alias("DataName")
            )
            .drop_nulls()
            .rename({"ReferenceGlobalID": "DataID"})
            .lazy()
        )

    # endregion
