# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import polars as pl
from helpers.helper import GlobalHelpers
from helpers.helper_filepaths import EICUPaths
from helpers.helper_OMOP import Vocabulary


class EICUExtractor(EICUPaths):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.eicu_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)
        self.icu_stay_id = self.extract_patient_information().select(
            self.icu_stay_id_col,
            self.hospital_stay_id_col,
            self.person_id_col,
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            self.icu_stay_id_col, self.icu_length_of_stay_col
        )

        self.other_lab_values = [
            "Bilirubin.direct [Mass/volume]",
            "Bilirubin.indirect [Mass/volume]",
            "Bilirubin.total [Mass/volume]",
            "Calcium [Mass/volume]",
            "Calcium.ionized [Mass/volume]",
            "Creatine kinase.MB [Mass/volume]",
            "Iron [Mass/volume]",
            "Iron binding capacity [Mass/volume]",
            "Magnesium [Mass/volume]",
            "Phosphate [Mass/volume]",
            "Triiodothyronine (T3) [Mass/volume]",
            "Thyroxine (T4) [Mass/volume]",
            "Thyroxine (T4) free [Mass/volume]",
            "Cobalamin (Vitamin B12) [Mass/volume]",
        ]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extract and transform patient-level data from the eICU patient CSV file.

        Steps:
            1. Read the raw patient CSV using Polars scan.
            2. Select a set of key columns including:
               - "uniquepid" → {person_id_col}: Unique patient identifier.
               - "patienthealthsystemstayid" → {hospital_stay_id_col}: Hospital admission identifier.
               - "patientunitstayid" → {icu_stay_id_col}: ICU stay identifier.
               - "gender" → {gender_col}: Patient gender.
               - "age" → {age_col}: Patient age.
               - "ethnicity" → {ethnicity_col}: Patient ethnicity.
               - "admissionheight" → {height_col}: Patient height.
               - "admissionweight" → {weight_col}: Patient weight.
               - "unittype" → {unit_type_col}: Type of care unit.
               - "unitadmitsource" → {admission_loc_col}: Admission source/location.
            3. Join with admission diagnoses via extract_admission_diagnoses().
            4. Convert time strings to proper datetime objects.
            5. Derive new columns:
               - {pre_icu_length_of_stay_col}: Pre-ICU length (in days).
               - {icu_length_of_stay_col}: ICU length (in days).
               - {hospital_length_of_stay_col}: Hospital length (in days).
               - {mortality_icu_col}: ICU mortality flag.
            6. Handle zero values in height/weight and calculate {icu_stay_seq_num_col}.

        Returns:
            pl.LazyFrame: Dataframe with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {hospital_stay_id_col}: Hospital admission identifier.
                - {person_id_col}: Patient identifier.
                - {icu_stay_seq_num_col}: ICU stay sequence number.
                - {gender_col}: Patient gender.
                - {age_col}: Patient age.
                - {height_col}: Patient height.
                - {weight_col}: Patient weight.
                - {ethnicity_col}: Patient ethnicity.
                - {admission_loc_col}: Admission location.
                - {pre_icu_length_of_stay_col}: Pre-ICU length of stay.
                - {icu_length_of_stay_col}: ICU length of stay.
                - {hospital_length_of_stay_col}: Hospital length of stay.
                - {mortality_hosp_col}: Hospital mortality flag.
                - {mortality_icu_col}: ICU mortality flag.
                - {mortality_after_col}: Mortality after discharge (days).
                - {unit_type_col}: Unit type.
                - {care_site_col}: Care site.
                - {discharge_loc_col}: Discharge location.
        """
        return (
            pl.scan_csv(self.patient_path)
            .select(  # Select columns of interest
                "uniquepid",
                "patienthealthsystemstayid",
                "patientunitstayid",
                "gender",
                "age",
                "ethnicity",
                "admissionheight",
                "admissionweight",
                "unittype",
                "unitadmitsource",
                "unitadmittime24",
                "unitvisitnumber",
                "unitdischargelocation",
                "unitdischargestatus",
                "unitdischargeoffset",
                "hospitalid",
                "hospitaladmitoffset",
                "hospitaldischargeoffset",
                "hospitaldischargestatus",
                "hospitaldischargeyear",
            )
            # Rename columns for consistency
            .rename(
                {
                    "uniquepid": self.person_id_col,
                    "patienthealthsystemstayid": self.hospital_stay_id_col,
                    "patientunitstayid": self.icu_stay_id_col,
                    "gender": self.gender_col,
                    "age": self.age_col,
                    "ethnicity": self.ethnicity_col,
                    "admissionheight": self.height_col,
                    "admissionweight": self.weight_col,
                    "unittype": self.unit_type_col,
                    "unitadmitsource": self.admission_loc_col,
                    "unitvisitnumber": self.icu_stay_seq_num_col,
                    "unitdischargelocation": self.discharge_loc_col,
                    "unitdischargestatus": self.mortality_icu_col,
                    "unitdischargeoffset": self.icu_length_of_stay_col,
                    "hospitalid": self.care_site_col,
                    "hospitaldischargestatus": self.mortality_hosp_col,
                }
            )
            .sort(self.icu_stay_id_col)
            .join(
                self.extract_specialty_information(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                self.extract_admission_diagnoses(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                # Convert categorical gender to enum
                pl.col(self.gender_col)
                .replace("", "Unknown")
                .cast(self.gender_dtype),
                # Convert categorical ethnicity to enum
                pl.col(self.ethnicity_col)
                .replace(self.ETHNICITY_MAP)
                .cast(self.ethnicity_dtype),
                # NOTE: ASSUMPTION: Replace age values "> 89" with 90 and convert to float
                pl.col(self.age_col)
                .replace("> 89", 90)
                .cast(int, strict=False),
                # Calculate pre ICU length of stay
                # Reverse sign of hospitaladmitoffset to get Pre-ICU length of stay
                (0 - pl.col("hospitaladmitoffset"))
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate ICU mortality
                pl.when(pl.col(self.mortality_icu_col) != "")
                .then(pl.col(self.mortality_icu_col) == "Expired")
                .otherwise(None)
                .cast(bool),
                # # Convert categorical mortality to enum
                # (
                #     pl.col(self.mortality_icu_col)
                #     .replace({"Expired": "Dead", "": "Unknown"})
                #     .cast(self.mortality_dtype)
                # ),
                # Calculate hospital mortality
                pl.when(pl.col(self.mortality_hosp_col) != "")
                .then(pl.col(self.mortality_hosp_col) == "Expired")
                .otherwise(None)
                .cast(bool),
                # Calculate mortality after discharge
                pl.when(
                    (pl.col(self.mortality_icu_col) != "Expired")
                    & (pl.col(self.mortality_hosp_col) == "Expired")
                )
                .then(
                    pl.col("hospitaldischargeoffset")
                    - pl.col(self.icu_length_of_stay_col)
                )
                .otherwise(None)
                .alias(self.mortality_after_col),
                # Calculate hospital_length_of_stay as difference between hospitaldischargeoffset
                # and hospitaladmitoffset
                (
                    pl.col("hospitaldischargeoffset")
                    - pl.col("hospitaladmitoffset")
                ).alias(self.hospital_length_of_stay_col),
                # Convert categorical admission location to enum
                pl.col(self.admission_loc_col)
                .replace(self.ADMISSION_LOCATIONS_MAP)
                .cast(self.admission_locations_dtype),
                # Convert categorical unit type to enum
                pl.col(self.unit_type_col)
                .replace(self.UNIT_TYPES_MAP)
                .cast(self.unit_types_dtype),
                # Convert categorical discharge location to enum
                pl.col(self.discharge_loc_col)
                .replace(self.DISCHARGE_LOCATIONS_MAP)
                .cast(self.discharge_locations_dtype),
                # Convert admssiontime string to datetime
                pl.col("unitadmittime24")
                .str.to_time("%H:%M:%S")
                .alias(self.admission_time_col),
            )
            # Handle zero values for height and weight
            .with_columns(
                pl.when(pl.col(self.height_col) == 0)
                .then(None)
                .otherwise(pl.col(self.height_col))
                .alias(self.height_col),
                pl.when(pl.col(self.weight_col) == 0)
                .then(None)
                .otherwise(pl.col(self.weight_col))
                .alias(self.weight_col),
            )
            # Calculate ICU stay sequence number
            # based on https://github.com/MIT-LCP/eicu-code/issues/145#issuecomment-680487192
            .sort(
                [
                    self.mortality_icu_col,  # mortality must be "increasing" (i.e. alive [= false / 0] first)
                    self.person_id_col,
                    "hospitaldischargeyear",
                    self.age_col,  # age must be increasing
                    self.hospital_stay_id_col,  # keep same hospital stays together
                    self.icu_stay_seq_num_col,
                ],
                descending=False,
                nulls_last=False,
            )
            .with_columns(
                (pl.int_range(pl.len()).over(self.person_id_col) + 1).alias(
                    self.icu_stay_seq_num_col
                )
            )
            # Convert time columns to floating point days for consistency
            .pipe(
                self.helpers._convert_time_to_days_float,
                self.pre_icu_length_of_stay_col,
                base_unit="minutes",
            )
            .pipe(
                self.helpers._convert_time_to_days_float,
                self.icu_length_of_stay_col,
                base_unit="minutes",
            )
            .pipe(
                self.helpers._convert_time_to_days_float,
                self.hospital_length_of_stay_col,
                base_unit="minutes",
            )
            .pipe(
                self.helpers._convert_time_to_days_float,
                self.mortality_after_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region specialty
    # Extract specialty information from the carePlanCareProvider.csv file
    def extract_specialty_information(self) -> pl.LazyFrame:
        """
        Extracts specialty information from the carePlanCareProvider CSV file.

        Reads specialty data, applies name mapping and filters for relevant specialties.
        Steps:
            1. Read specialty data from self.carePlanCareProvider_path.
            2. Replace specialty names using a provided mapping.
            3. Select the first specialty for each ICU stay.
        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {specialty_col}: Mapped specialty name.
        """
        careprovider_mapping = self.helpers.load_mapping(
            self.careprovider_mapping_path
        )

        return (
            pl.scan_csv(self.carePlanCareProvider_path)
            .rename({"patientunitstayid": self.icu_stay_id_col})
            .filter(pl.col("managingphysician") == "Managing")
            .with_columns(
                # Replace specialty names with mapped names
                pl.col("specialty")
                .replace_strict(careprovider_mapping, default=None)
                .alias(self.specialty_col)
            )
            .group_by(self.icu_stay_id_col)
            .agg(
                pl.col(self.specialty_col)
                .sort_by("careprovidersaveoffset")
                .first()
            )
            # Filter for relevant specialties
            .filter(pl.col(self.specialty_col).is_not_null())
            # Remove duplicate rows
            .unique()
        )

    # endregion

    # region admitDX
    # Extract admission diagnosis information from the admissionDx.csv file
    def extract_admission_diagnoses(self) -> pl.LazyFrame:
        """
        Extracts admission diagnosis information from the eICU admissionDx CSV file.

        Reads the admissionDx CSV, cleans and processes the diagnosis text, and infers related admission properties.
        Steps:
            1. Read the CSV file containing admission diagnosis data.
            2. Rename "patientunitstayid" to {icu_stay_id_col} for consistency.
            3. Clean diagnosis strings and harmonize delimiters.
            4. Infer admission type and urgency from text.
            5. Group rows by {icu_stay_id_col} and aggregate the first available non-null value.

        Returns:
            pl.LazyFrame: A LazyFrame with the following columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {admission_type_col}: Admission type (e.g., Surgical or Medical).
                - {admission_urgency_col}: Admission urgency status.
                - {admission_diagnosis_col}: Admission diagnosis description.
        """
        return (
            pl.scan_csv(self.admissionDx_path)
            .select("patientunitstayid", "admitdxpath", "admitdxname")
            .rename({"patientunitstayid": self.icu_stay_id_col})
            .with_columns(
                # Admission Diagnosis
                pl.when(
                    pl.col("admitdxpath").str.starts_with(
                        "admission diagnosis|All Diagnosis|"
                    )
                )
                .then(
                    pl.col("admitdxpath")
                    .str.replace("admission diagnosis\|All Diagnosis\|", "")
                    .str.replace("\|Diagnosis\|", " - ")
                    .str.replace_all("\|", " - ")
                    .str.replace_all(" ,", ",")
                    .str.replace_all(",", ", ")
                    .str.replace_all("  ", " ")
                    # clean comments
                    .str.replace(
                        " \(angina interferes w/quality of life or meds are tolerated poorly\)",
                        "",
                    )
                    .str.replace(
                        " \(with or without respiratory arrest; for respiratory arrest see Respiratory System\)",
                        "",
                    )
                    .str.replace(
                        " \(for gastrointestinal bleeding GI-see GI system\) \(for trauma see Trauma\)",
                        "",
                    )
                    .str.replace(
                        " \(for cerebrovascular accident-see Neurological System\)",
                        "",
                    )
                    .str.replace(
                        ", Do not include shock states",
                        "",
                    )
                    .str.replace(
                        " \(for hepatic see GI, for diabetic see Endocrine, if related to cardiac arrest, see CV\)",
                        "",
                    )
                    .str.replace(
                        " \(excluding vascular shunting-see surgery for portosystemic shunt\)",
                        "",
                    )
                    .str.replace(
                        " \(if related to trauma, see Trauma\)",
                        "",
                    )
                    .str.replace(
                        "-no structural brain disease",
                        "",
                    )
                    .str.replace(
                        ", for fractures due to trauma see Trauma",
                        "",
                    )
                    # harmonize comments
                    .str.replace("Hematoma subdural", "Hematoma, subdural")
                    .str.replace("Hematoma-epidural", "Hematoma, epidural")
                    .str.replace_all("i.e.,", "i.e.", literal=True)
                    .str.replace_all("i.e.", "i.e. ", literal=True)
                    .str.replace_all("i.e.  ", "i.e. ", literal=True)
                    .str.replace("ileal-conduit", "ileal conduit")
                    .str.replace(
                        "Pneumocystic pneumonia", "Pneumocystis pneumonia"
                    )
                    .str.replace("surgery,surgery", "surgery, surgery")
                    .str.replace("; surgery", ", surgery")
                    .str.replace("for;", "for")
                )
                .otherwise(None)
                .alias(self.admission_diagnosis_col),
            )
            .with_columns(
                # Admission Type
                pl.when(
                    pl.col("admitdxpath")
                    == "admission diagnosis|Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?|Yes"
                )
                .then(pl.lit("Surgical"))
                .when(
                    pl.col("admitdxpath")
                    == "admission diagnosis|Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?|No"
                )
                .then(pl.lit("Medical"))
                .when(
                    pl.col(self.admission_diagnosis_col).str.starts_with(
                        "Operative"
                    )
                )
                .then(pl.lit("Surgical"))
                .when(
                    pl.col(self.admission_diagnosis_col).str.starts_with(
                        "Non-operative"
                    )
                )
                .then(pl.lit("Medical"))
                .otherwise(None)
                .alias(self.admission_type_col),
                # Admission Urgency
                pl.when(
                    pl.col("admitdxpath") == "admission diagnosis|Elective|Yes"
                )
                .then(pl.lit("Elective"))
                .when(
                    pl.col("admitdxpath") == "admission diagnosis|Elective|No"
                )
                .then(pl.lit("Emergency"))
                .otherwise(None)
                .alias(self.admission_urgency_col),
            )
            .sort(self.icu_stay_id_col, "admitdxpath")
            .group_by(self.icu_stay_id_col)
            .agg(
                pl.col(self.admission_diagnosis_col).drop_nulls().first(),
                pl.col(self.admission_type_col).drop_nulls().first(),
                pl.col(self.admission_urgency_col).drop_nulls().first(),
            )
            .cast(
                {
                    self.admission_type_col: self.admission_types_dtype,
                    self.admission_urgency_col: self.admission_urgency_dtype,
                }
            )
            .select(
                self.icu_stay_id_col,
                self.admission_type_col,
                self.admission_urgency_col,
                self.admission_diagnosis_col,
            )
        )

    # region lab TS
    # Extract time series information for lab values from the lab.csv file
    def extract_time_series_lab(self) -> pl.LazyFrame:
        """
        Extracts laboratory time series measurements from the lab CSV file.

        Reads lab records, applies name mapping, joins LOINC details and converts times to seconds.
        Steps:
            1. Read lab data from self.lab_path and select key columns.
            2. Rename columns for consistency, replacing "patientunitstayid" with {icu_stay_id_col} and "labresultoffset" with {timeseries_time_col}.
            3. Replace lab names using a mapping.
            4. Join with LOINC components to add: LOINC_component, LOINC_system, LOINC_method, LOINC_time, and LOINC_code.
            5. Filter records based on relevant lab components and systems.
            6. Remove duplicate and null value records.
            7. Convert time offsets from minutes to seconds.
            8. Assemble a structured column "labstruct" containing lab result details.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time of recording (in seconds).
                - "labname": Mapped laboratory test name.
                - "labstruct": A struct with keys:
                    * value: Lab result value.
                    * system: LOINC system.
                    * method: LOINC method.
                    * time: LOINC time aspect.
                    * LOINC: LOINC code.
        """
        lab_names_mapping = self.helpers.load_mapping(self.lab_mapping_path)

        labs = (
            pl.scan_csv(self.lab_path).select(
                "patientunitstayid", "labname", "labresultoffset", "labresult"
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "labresultoffset": self.timeseries_time_col,
                }
            )
            # Replace lab names with mapped names
            .with_columns(
                pl.col("labname")
                .replace_strict(lab_names_mapping, default=None)
                .alias("labname")
            )
        )

        LOINC_data = labs.select("labname").unique()
        labnames = LOINC_data.collect().to_series().to_list()
        LOINC_data = (
            LOINC_data
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("labname")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("labname")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames), default=None
                )
                .alias("LOINC_system"),
                pl.col("labname")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames), default=None
                )
                .alias("LOINC_method"),
                pl.col("labname").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("labname")
                .replace_strict(
                    self.omop.get_concept_codes_from_names(labnames),
                    default=None,
                )
                .alias("LOINC_code"),
            )
        )

        return (
            labs.join(LOINC_data, on="labname", how="left")
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
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("labname").is_not_null())
            # Remove rows with empty lab results
            .filter(pl.col("labresult").is_not_null())
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
            # MAKE STRUCT
            .with_columns(pl.col("LOINC_component").alias("labname"))
            .with_columns(
                pl.struct(
                    value=pl.col("labresult"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "labname",
                "labstruct",
            )
        )

    # endregion

    # region resp TS
    # Extract time series information for respiratory values from the respiratorycharting.csv file
    def extract_time_series_resp(self) -> pl.LazyFrame:
        """
        Extracts respiratory time series measurements from the respiratoryCharting CSV file.

        Reads respiratory data, maps measurement labels, cleans values and converts recording times.
        Steps:
            1. Read respiratory charting data from self.respiratoryCharting_path.
            2. Rename key columns replacing "patientunitstayid" with {icu_stay_id_col} and "respchartoffset" with {timeseries_time_col}.
            3. Apply a mapping for respiratory measurement labels.
            4. Clean measurement values (e.g., removing percentage symbols and extraneous text).
            5. Filter for the defined set of respiratory measurements.
            6. Remove duplicate rows and convert times from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Adjusted time in seconds.
                - "respchartvaluelabel": Respiratory measurement label.
                - "respchartvalue": Recorded respiratory measurement value.
        """
        # NOTE: ASSUMPTION: These are the respiratory values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        keep_resp_names = self.relevant_respiratory_values
        resp_names_mapping = self.helpers.load_mapping(self.resp_mapping_path)
        resp_oxygen_delivery_device_mapping = self.helpers.load_mapping(
            self.resp_oxygen_delivery_device_mapping_path
        )

        return (
            pl.scan_csv(self.respiratoryCharting_path)
            .select(
                "patientunitstayid",
                "respchartoffset",
                "respchartvaluelabel",
                "respchartvalue",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "respchartoffset": self.timeseries_time_col,
                }
            )
            .with_columns(
                # Replace lab names with mapped names
                pl.col("respchartvaluelabel")
                .replace_strict(resp_names_mapping, default=None)
                .alias("respchartvaluelabel"),
                # Remove percentage sign from respchartvalue
                pl.col("respchartvalue")
                .str.replace("%", "")
                .str.replace("Discontinued", "")
                .str.replace("Initiated", "")
                .str.replace("Maintained", "")
                .str.replace("Not applicable", "")
                .str.replace("Refused after education", ""),
                # .cast(float, strict=False),
            )
            .with_columns(
                # Map O2 delivery device values
                pl.when(
                    pl.col("respchartvaluelabel") == "Oxygen delivery system"
                )
                .then(
                    pl.col("respchartvalue").replace_strict(
                        resp_oxygen_delivery_device_mapping, default=None
                    )
                )
                .otherwise(pl.col("respchartvalue"))
                .alias("respchartvalue"),
            )
            # Filter for resp names of interest
            .filter(pl.col("respchartvaluelabel").is_in(keep_resp_names))
            # Remove rows with empty resp values
            .filter(
                pl.col("respchartvalue").is_not_null(),
                pl.col("respchartvalue").ne_missing(""),
            )
            # Remove duplicate rows
            .unique()
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region nurse TS
    # Extract time series information for nurse values from the nurseCharting.csv file
    def extract_time_series_nurse(self) -> pl.LazyFrame:
        """
        Extracts nurse charting time series measurements from the nurseCharting CSV file.

        Reads nurse charting data filtering only for selected measurement types, cleans them and adjusts recording times.
        Steps:
            1. Read nurse charting data from self.nurseCharting_path.
            2. Rename columns to standard names (e.g., replacing "patientunitstayid" with {icu_stay_id_col} and "nursingchartoffset" with {timeseries_time_col}).
            3. Filter rows to keep only nurse measurements of interest.
            4. Clean measurement names and values (e.g., replacing special text or null values).
            5. Convert recording time from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time of recording (in seconds).
                - "nursingchartcelltypevalname": Nurse measurement name.
                - "nursingchartvalue": Nurse measurement value.
        """
        # NOTE: keep only the nurse charting values not covered by the other TS
        keep_nurse_names = [
            # "Non-Invasive BP",
            # "Invasive BP",
            # "Heart Rate",
            # # "Pain Score/Goal",
            # "Respiratory Rate",
            # "O2 Saturation",
            # "Temperature",
            "Glasgow coma score",
            # "Invasive BP",
            "Bedside Glucose",
            "O2 L/%",
            "O2 Admin Device",
            "Sedation Scale/Score/Goal",
            # # "Delirium Scale/Score",
        ]
        nurse_names_mapping = self.load_mapping(self.nurse_mapping_path)
        nurse_oxygen_delivery_device_mapping = self.load_mapping(
            self.nurse_oxygen_delivery_device_mapping_path
        )

        nurseCharting = (
            pl.scan_csv(self.nurseCharting_path)
            .select(
                "patientunitstayid",
                "nursingchartoffset",
                "nursingchartcelltypevallabel",
                "nursingchartcelltypevalname",
                "nursingchartvalue",
            )
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "nursingchartoffset": self.timeseries_time_col,
                }
            )
            # Filter for nurse names of interest
            .filter(
                pl.col("nursingchartcelltypevallabel").is_in(keep_nurse_names)
            )
            # Remove rows with empty nurse values
            .drop_nulls(
                [
                    "nursingchartcelltypevallabel",
                    "nursingchartcelltypevalname",
                    "nursingchartvalue",
                ]
            )
            # Remove duplicate rows
            .unique()
        )

        nurseCharting_RASS = (
            nurseCharting.filter(
                pl.col("nursingchartcelltypevalname") == "Sedation Score",
            )
            .join(
                nurseCharting.filter(
                    pl.col("nursingchartcelltypevalname") == "Sedation Scale",
                    pl.col("nursingchartvalue") == "RASS",
                ).select(self.icu_stay_id_col, self.timeseries_time_col),
                on=[self.icu_stay_id_col, self.timeseries_time_col],
                how="right",
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "nursingchartcelltypevallabel",
                "nursingchartcelltypevalname",
                "nursingchartvalue",
            )
        )

        return (
            pl.concat(
                [
                    nurseCharting.filter(
                        pl.col("nursingchartcelltypevallabel")
                        != "Sedation Scale/Score/Goal"
                    ),
                    nurseCharting_RASS,
                ],
                how="vertical_relaxed",
            )
            .drop("nursingchartcelltypevallabel")
            .with_columns(
                # Replace "Unable to score due to medication" values with None
                pl.when(
                    pl.col("nursingchartvalue")
                    == "Unable to score due to medication"
                )
                .then(None)
                # Replace empty strings with None
                .when(pl.col("nursingchartvalue") == "")
                .then(None)
                .otherwise(pl.col("nursingchartvalue"))
                .alias("nursingchartvalue"),
            )
            .with_columns(
                # Replace nurse names with mapped names
                pl.col("nursingchartcelltypevalname")
                .replace_strict(nurse_names_mapping, default=None)
                .alias("nursingchartcelltypevalname"),
            )
            .with_columns(
                # Map O2 delivery device values
                pl.when(
                    pl.col("nursingchartcelltypevalname")
                    == "Oxygen delivery system"
                )
                .then(
                    pl.col("nursingchartvalue").replace_strict(
                        nurse_oxygen_delivery_device_mapping, default=None
                    )
                )
                .otherwise(pl.col("nursingchartvalue"))
                .alias("nursingchartvalue"),
            )
            # Remove rows with empty nurse values
            .drop_nulls(["nursingchartcelltypevalname", "nursingchartvalue"])
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region in/out TS
    # Extract time series information for intake/output values from the intakeOutput.csv file
    def extract_time_series_intake_output(self) -> pl.LazyFrame:
        """
        Extracts intake/output time series measurements from the intakeOutput CSV file.

        Reads intake/output data, applies mapping to measurement names, filters for relevant values and adjusts time offsets.
        Steps:
            1. Read intake/output data from self.intakeOutput_path.
            2. Rename columns (e.g., "patientunitstayid" to {icu_stay_id_col} and "intakeoutputoffset" to {timeseries_time_col}).
            3. Replace and map intake/output labels using provided mappings.
            4. Filter out null numeric measurement values and remove duplicates.
            5. Convert time offsets from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Adjusted time in seconds.
                - "celllabel": Mapped intake/output variable name.
                - "cellvaluenumeric": Numerical value of the measurement.
        """
        # NOTE: ASSUMPTION: These are the intake/output values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        intakeoutput_mapping = self.load_mapping(self.intakeoutput_mapping_path)

        return (
            pl.scan_csv(self.intakeOutput_path)
            .select(
                "patientunitstayid",
                "intakeoutputoffset",
                # "intaketotal",
                # "outputtotal",
                # "dialysistotal",
                # "nettotal",
                "cellpath",
                "celllabel",
                "cellvaluenumeric",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "intakeoutputoffset": self.timeseries_time_col,
                }
            )
            .with_columns(
                # Replace intakeoutput names with mapped names
                pl.col("cellpath")
                .replace_strict(intakeoutput_mapping, default=None)
                .replace_strict(
                    self.timeseries_intakeoutput_mapping, default=None
                )
                .alias("celllabel"),
            )
            .drop("cellpath")
            # Filter for intakeoutput names of interest
            .filter(
                pl.col("celllabel").is_in(self.relevant_intakeoutput_values)
            )
            # Remove rows with empty intakeoutput values
            .filter(pl.col("cellvaluenumeric").is_not_null())
            # Remove duplicate rows
            .unique()
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region periodic TS
    # Extract time series information for periodic values from the vitalPeriodic.csv file
    def extract_time_series_periodic(self) -> pl.LazyFrame:
        """
        Extracts periodic vital sign measurements from the vitalPeriodic CSV file.

        Reads periodic vital sign data and converts time offsets.
        Steps:
            1. Read periodic data from self.vitalPeriodic_path.
            2. Rename key columns (e.g., "patientunitstayid" to {icu_stay_id_col} and "observationoffset" to {timeseries_time_col}).
            3. Remove duplicate records.
            4. Convert observation times from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time of observation (in seconds).
                - Plus any additional vital sign columns present in the CSV.
        """
        return (
            pl.scan_csv(self.vitalPeriodic_path)
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "observationoffset": self.timeseries_time_col,
                }
            )
            # Remove duplicate rows
            .unique([self.icu_stay_id_col, self.timeseries_time_col])
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region aperiodic TS
    # Extract time series information for aperiodic values from the vitalAperiodic.csv file
    def extract_time_series_aperiodic(self) -> pl.LazyFrame:
        """
        Extracts aperiodic vital sign measurements from the vitalAperiodic CSV file.

        Reads aperiodic vital sign data, selects key blood pressure measurements and converts time offsets.
        Steps:
            1. Read data from self.vitalAperiodic_path.
            2. Select key columns including noninvasive blood pressure measures.
            3. Rename columns for consistency (e.g., "patientunitstayid" to {icu_stay_id_col}, "observationoffset" to {timeseries_time_col}).
            4. Remove duplicate rows.
            5. Convert recording time from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time of observation (in seconds).
                - "noninvasivesystolic": Non-invasive systolic blood pressure.
                - "noninvasivediastolic": Non-invasive diastolic blood pressure.
                - "noninvasivemean": Non-invasive mean blood pressure.
        """
        return (
            pl.scan_csv(self.vitalAperiodic_path).select(
                "patientunitstayid",
                "observationoffset",
                "noninvasivesystolic",
                "noninvasivediastolic",
                "noninvasivemean",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "observationoffset": self.timeseries_time_col,
                }
            )
            # Remove duplicate rows
            .unique([self.icu_stay_id_col, self.timeseries_time_col])
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region combined
    # Combine the aperiodic and periodic time series data
    def extract_and_combine_periodics(self) -> pl.LazyFrame:
        """
        Combines periodic and aperiodic vital sign measurements into one wide-format dataframe.

        Merges periodic and aperiodic data based on time and patient, aggregating and renaming fields according to a mapping.
        Steps:
            1. Extract periodic vital sign data using extract_time_series_periodic.
            2. Extract aperiodic vital sign data using extract_time_series_aperiodic.
            3. Concatenate both datasets in a diagonal-relaxed fashion.
            4. Group by {icu_stay_id_col} and {timeseries_time_col} to aggregate using the first available value.
            5. Rename columns based on a provided mapping.

        Returns:
            pl.LazyFrame: A combined LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time of observation (in seconds).
                - Plus vital sign columns as defined by the mapping.
        """
        periodic_mapping = self.helpers.load_mapping(self.periodic_mapping_path)
        periodic_mapping_keys = list(periodic_mapping.values())

        periodic = self.extract_time_series_periodic()
        aperiodic = self.extract_time_series_aperiodic()

        return (
            pl.concat([periodic, aperiodic], how="diagonal_relaxed")
            .group_by(self.icu_stay_id_col, self.timeseries_time_col)
            .first()
            .rename(periodic_mapping)
            .select(
                [self.icu_stay_id_col, self.timeseries_time_col]
                + periodic_mapping_keys
            )
        )

    # endregion

    # region microbiology
    # Extract microbiology information from the microLab.csv file
    def extract_microbiology(self) -> pl.LazyFrame:
        """
        Extracts microbiology culture information from the microLab CSV file.

        Reads microbiology data, applies mappings to specimen, organism and antibiotic names, and adjusts time offsets.
        Steps:
            1. Read microbiology data from self.microLab_path.
            2. Rename key columns (e.g., "patientunitstayid" to {icu_stay_id_col} and "culturetakenoffset" to {timeseries_time_col}).
            3. Map specimen sites, organisms and antibiotics using provided mappings.
            4. Remove duplicate records.
            5. Convert time offsets from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time of culture measurement (in seconds).
                - {micro_specimen_col}: Mapped specimen type.
                - {micro_organism_col}: Mapped organism name.
                - {micro_antibiotic_col}: Mapped antibiotic name.
                - {micro_sensitivity_col}: Sensitivity shorthand.
        """
        print("eICU    - Extracting microbiology...")

        # NOTE: ASSUMPTION: These are the microbiology values of interest
        eicu_microbiology_culturesite_mapping = self.helpers.load_mapping(
            self.micro_culturesite_mapping_path
        )
        eicu_microbiology_organism_mapping = self.helpers.load_mapping(
            self.micro_organism_mapping_path
        )

        return (
            pl.scan_csv(self.microLab_path)
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "culturetakenoffset": self.timeseries_time_col,
                }
            ).with_columns(
                # Replace culture site names with mapped names
                pl.col("culturesite")
                .replace_strict(
                    eicu_microbiology_culturesite_mapping, default=None
                )
                .alias(self.micro_specimen_col),
                # Replace organism names with mapped names
                pl.col("organism")
                .replace_strict(
                    eicu_microbiology_organism_mapping, default=None
                )
                .alias(self.micro_organism_col),
                # Replace antibiotic names with mapped names
                pl.col("antibiotic")
                .replace(
                    {
                        "amoxicillin/clavulonic acid": (
                            "amoxicillin / clavulanate"
                        ),
                        "ampicillin/sulbactam": "ampicillin / sulbactam",
                        "imipenem/cilastatin": "cilastatin / imipenem",
                        "piperacillin/tazobactam": "piperacillin / tazobactam",
                        "ticarcillin/clavulonic acid": (
                            "clavulanate / ticarcillin"
                        ),
                        "trimethoprim/sulfamethoxazole": (
                            "sulfamethoxazole / trimethoprim"
                        ),
                        "": None,
                    }
                )
                .alias(self.micro_antibiotic_col),
                # Replace sensitivities with shorthands
                pl.col("sensitivitylevel")
                .replace(
                    {
                        "Resistant": "R",
                        "Intermediate": "I",
                        "Sensitive": "S",
                        "": None,
                    }
                )
                .alias(self.micro_sensitivity_col),
            )
            # Remove duplicate rows
            .unique()
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.timeseries_time_col,
                base_unit="minutes",
            )
        )

    # endregion

    # region medication
    # Extract medication information from the different medication files
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extracts medication administration records from infusionDrug and medication CSV files.

        Reads medication data, standardizes names using mappings and computes dosage, infusion rates and durations.
        Steps:
            1. Read medication administration data from infusionDrug and medication CSV files.
            2. Rename columns (e.g., replacing "patientunitstayid" with {icu_stay_id_col}).
            3. Replace and map drug names and derive ingredients.
            4. Calculate dosage, infusion duration, and fluid amounts.
            5. Fix inconsistencies in stop offsets and remove incomplete records.
            6. Convert time offsets from minutes to seconds.

        Returns:
            pl.LazyFrame: A LazyFrame with columns including:
                - {icu_stay_id_col}: ICU stay identifier.
                - {drug_start_col}: Start time of medication (in seconds).
                - {drug_end_col}: End time of medication (in seconds).
                - {drug_name_col}: Original medication name.
                - {drug_ingredient_col}: Mapped medication ingredient.
                - {drug_rate_col}: Dosage or drug rate.
                - {fluid_rate_col}: Infusion fluid rate.
                - {fluid_amount_col}: Calculated infused fluid volume.
                - {drug_patient_weight_col}: Patient weight used for dosing.
                - {drug_rate_unit_col}: Unit extracted from the drug name.
                - {drug_admin_route_col}: Administration route (e.g., intravenous).
        """
        print("eICU    - Extracting medications...")

        eicu_medication_mapping = self.helpers.load_many_to_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS.yaml", "eicu"
        )
        eicu_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )
        SECONDS_IN_1H = 3600

        # NOTE: Extremely infrequently used.
        # cf. w/ Important considerations @ https://eicu-crd.mit.edu/eicutables/admissiondrug/
        # admissiondrug = None

        # NOTE: a lot of calcalations can be done here
        # cf. w/ Important considerations @ https://eicu-crd.mit.edu/eicutables/infusiondrug/
        infusiondrug = (
            pl.scan_csv(
                self.infusionDrug_path,
                schema_overrides={"drugrate": str},
            )
            .select(
                "patientunitstayid",
                "infusionoffset",
                "drugname",
                "drugrate",
                "infusionrate",
                "patientweight",
            )
            # Replace "OFF" values with 0
            .with_columns(pl.col("drugrate").str.replace("OFF", 0))
            .cast({"drugrate": float, "infusionrate": float}, strict=False)
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "infusionoffset": self.drug_start_col,
                    "drugname": self.drug_name_col,
                    "drugrate": self.drug_rate_col,
                    "infusionrate": self.fluid_rate_col,
                    "patientweight": self.drug_patient_weight_col,
                }
            )
            .with_columns(
                # Get unit from drugname
                # e.g. Norepinephrine (mcg/min) -> mcg/min
                pl.col(self.drug_name_col)
                .str.extract(r".*\((.*?)\)$")
                .alias(self.drug_rate_unit_col),
                # Replace drug names with mapped names
                pl.col(self.drug_name_col)
                .replace_strict(eicu_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
                # Set administration route
                pl.lit("intravenous").alias(self.drug_admin_route_col),
                # Add a column to indicate if the drug is continuous
                pl.lit(True).alias(self.drug_continous_col),
            )
            # Remove rows with empty drug names
            .filter(pl.col(self.drug_name_col).is_not_null())
            # Remove rows with empty drug rates
            .filter(pl.col(self.drug_rate_col).is_not_null())
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.drug_start_col,
                base_unit="minutes",
            )
            .sort(self.icu_stay_id_col, self.drug_name_col, self.drug_start_col)
        )

        # Get infusion duration where possible, by checking whether the drugname reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        infusiondrug_offsets = (
            infusiondrug.select(self.icu_stay_id_col, self.drug_start_col)
            .unique()
            .sort(self.icu_stay_id_col, self.drug_start_col)
            .with_columns(
                pl.col(self.drug_start_col)
                .shift(1)
                .over(self.icu_stay_id_col)
                .alias("prev_drug_start"),
                pl.col(self.drug_start_col)
                .shift(-1)
                .over(self.icu_stay_id_col)
                .alias("next_drug_start"),
            )
        )

        infusiondrug = (
            infusiondrug.join(
                infusiondrug_offsets,
                on=[self.icu_stay_id_col, self.drug_start_col],
                how="left",
            )
            # Sort by patient ID, drug name and drug start time
            .sort(self.icu_stay_id_col, self.drug_name_col, self.drug_start_col)
            # 2. Check if drug is continued from the previous log entry
            #    and if it is continued in the next log entry
            .with_columns(
                # Check if drug is continued from the previous log entry
                pl.when(pl.col("prev_drug_start").is_not_null())
                .then(
                    pl.when(
                        # Check if the previous drug is the same as the current drug
                        pl.col(self.drug_name_col)
                        == pl.col(self.drug_name_col).shift(1),
                        # Check if the previous drug start time is the previous log entry time
                        pl.col("prev_drug_start")
                        == pl.col(self.drug_start_col).shift(1),
                        # Check if the drug amount is the same as the previous drug amount
                        pl.col(self.drug_rate_col)
                        == pl.col(self.drug_rate_col).shift(1),
                    )
                    .then(pl.lit("continued"))
                    .otherwise(pl.lit("started"))
                )
                .otherwise(None)
                .alias("drug_status_prev"),
                # Check if drug is continued in the next log entry
                pl.when(pl.col("next_drug_start").is_not_null())
                .then(
                    pl.when(
                        # Check if the next drug is the same as the current drug
                        pl.col(self.drug_name_col)
                        == pl.col(self.drug_name_col).shift(-1),
                        # Check if the next drug start time is the next log entry time
                        pl.col("next_drug_start")
                        == pl.col(self.drug_start_col).shift(-1),
                        # Check if the drug amount is the same as the next drug amount
                        pl.col(self.drug_rate_col)
                        == pl.col(self.drug_rate_col).shift(-1),
                    )
                    .then(pl.lit("continued"))
                    .otherwise(pl.lit("discontinued"))
                )
                .otherwise(None)
                .alias("drug_status_next"),
            )
            # Filter for rows where the drug status changes
            .filter(pl.col("drug_status_prev") != pl.col("drug_status_next"))
            # 3. Get the end time of the drug if it is discontinued
            .with_columns(
                pl.when(pl.col("drug_status_next") == "discontinued")
                .then(pl.col("next_drug_start"))
                .otherwise(None)
                .alias(self.drug_end_col)
            )
            # 4. Combine rows where the drug is started, continued, then discontinued in the next row
            .with_columns(
                pl.when(
                    pl.col("drug_status_prev").shift(1) == "started",
                    pl.col("drug_status_next").shift(1) == "continued",
                    pl.col("drug_status_prev") == "continued",
                    pl.col("drug_status_next") == "discontinued",
                    # Check if the previous drug is the same as the current drug
                    pl.col(self.drug_name_col)
                    == pl.col(self.drug_name_col).shift(1),
                    # Check if the drug amount is the same as the previous drug amount
                    pl.col(self.drug_rate_col)
                    == pl.col(self.drug_rate_col).shift(1),
                )
                .then(pl.col(self.drug_start_col).shift(1))
                .otherwise(pl.col(self.drug_start_col))
                .alias(self.drug_start_col)
            ).filter(pl.col(self.drug_end_col).is_not_null())
            # 5. Remove the helper columns
            .drop(
                "prev_drug_start",
                "next_drug_start",
                "drug_status_prev",
                "drug_status_next",
            )
            # 6. calculate infused fluid volume
            # eICU documentation: "Infusion rate is generally charted as ml/hr."
            .with_columns(
                (
                    pl.col(self.fluid_rate_col)
                    * (pl.col(self.drug_end_col) - pl.col(self.drug_start_col))
                    / SECONDS_IN_1H
                ).alias(self.fluid_amount_col)
            )
        )

        medication = (
            pl.scan_csv(self.medication_path)
            .filter(pl.col("drugordercancelled").ne_missing("Yes"))
            .select(
                "patientunitstayid",
                "drugstartoffset",
                "drugname",
                "dosage",
                "drugstopoffset",
                "routeadmin",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "drugstartoffset": self.drug_start_col,
                    "drugname": self.drug_name_col,
                    "dosage": self.drug_amount_col,
                    "drugstopoffset": self.drug_end_col,
                    "routeadmin": self.drug_admin_route_col,
                }
            )
            # # Dropping drug dosages due to bad data quality
            # .drop(self.drug_amount_col)
            .with_columns(
                # Replace drug names with mapped names
                pl.col(self.drug_name_col)
                .replace_strict(eicu_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
                # Set administration route
                pl.col(self.drug_admin_route_col)
                .replace_strict(
                    eicu_drug_administration_route_mapping, default=None
                )
                .alias(self.drug_admin_route_col),
                # Fix stop offsets (if smaller than start offset)
                pl.when(
                    pl.col(self.drug_end_col) < pl.col(self.drug_start_col),
                )
                .then(pl.col(self.drug_start_col))
                .otherwise(pl.col(self.drug_end_col))
                .alias(self.drug_end_col),
                # Add a column to indicate if the drug is continuous
                # False, since continuous drugs are already in the infusiondrug table
                pl.lit(False).alias(self.drug_continous_col),
            )
            # Remove rows with empty drug names
            .filter(pl.col(self.drug_name_col).is_not_null())
            # # Remove rows with empty ingredient names
            # .filter(pl.col(self.drug_ingredient_col).is_not_null())
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.drug_start_col,
                base_unit="minutes",
            )
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.drug_end_col,
                base_unit="minutes",
            )
        )

        # Concatenate the medication tables
        return pl.concat([infusiondrug, medication], how="diagonal_relaxed")

    # endregion

    # region diagnoses
    # Extract diagnosis information from the diagnosis.csv file
    def extract_diagnoses(self) -> pl.LazyFrame:
        """
        Extracts diagnosis information from the diagnosis CSV file.

        Reads diagnosis records, processes ICD9 codes and calculates diagnosis timing and priority.
        Steps:
            1. Read diagnosis records from diagnosis.csv.gz.
            2. Select and rename key columns where "patientunitstayid" becomes {icu_stay_id_col}.
            3. Split and clean diagnosis ICD codes.
            4. Cast and convert diagnosis offset times (converted to seconds).
            5. Explode list of ICD codes to create one row per code.
            6. Remove duplicate and empty records.
            7. Group rows to keep only the most severe diagnosis when duplicates exist.
            8. Identify continuation of a diagnosis by comparing adjacent log offsets.

        Returns:
            pl.LazyFrame: A LazyFrame with columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {diagnosis_icd_code_col}: One ICD diagnosis code per row.
                - {diagnosis_start_col}: Time of diagnosis onset (in seconds).
                - {diagnosis_priority_col}: Numeric priority of the diagnosis.
                - {diagnosis_discharge_col}: Flag (Boolean) if diagnosis active upon discharge.
                - {diagnosis_end_col}: End time of the diagnosis (if applicable, in seconds).
        """
        diagnosis = (
            pl.scan_csv(self.path + "diagnosis.csv.gz")
            .select(  # Select columns of interest
                "patientunitstayid",
                "diagnosisoffset",
                "icd9code",
                "activeupondischarge",
                "diagnosispriority",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "activeupondischarge": self.diagnosis_discharge_col,
                }
            )
            .join(self.icu_stay_id, on=self.icu_stay_id_col, how="outer")
            .with_columns(  # Convert columns to appropriate data types
                # Split diagnosis codes by comma and rename column
                pl.col("icd9code")
                .str.split(by=", ")
                .alias(self.diagnosis_icd_code_col),
                # Convert diagnosisoffset to float and rename column
                pl.col("diagnosisoffset")
                .cast(float, strict=False)
                .alias(self.diagnosis_start_col),
                # Convert categorical diagnosispriority to float and rename column
                pl.col("diagnosispriority")
                .replace({"Primary": 1, "Major": 2, "Other": 3})
                .cast(float, strict=False)
                .alias(self.diagnosis_priority_col),
            )
            .drop("icd9code", "diagnosisoffset", "diagnosispriority")
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.diagnosis_start_col,
                base_unit="minutes",
            )
            # Explode the icd_code column to have one row per diagnosis code
            .explode(self.diagnosis_icd_code_col)
            # Remove duplicate rows
            .unique()
            # Remove rows with empty diagnosis codes
            .filter(pl.col(self.diagnosis_icd_code_col) != "")
            # Drop the doubled diagnoses with different priorities (keep the most severe one).
            .group_by(
                self.icu_stay_id_col,
                self.diagnosis_icd_code_col,
                self.diagnosis_start_col,
            )
            .agg(
                pl.all().sort_by(self.diagnosis_priority_col).first(),
            )
        )

        # Get continued diagnoses where possible, by checking whether the diagnosis reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        diagnosis_offsets = (
            diagnosis.select(self.icu_stay_id_col, self.diagnosis_start_col)
            .unique()
            .sort(self.icu_stay_id_col, self.diagnosis_start_col)
            .with_columns(
                pl.col(self.diagnosis_start_col)
                .shift(1)
                .over(self.icu_stay_id_col)
                .alias("prev_diag_start"),
                pl.col(self.diagnosis_start_col)
                .shift(-1)
                .over(self.icu_stay_id_col)
                .alias("next_diag_start"),
            )
        )

        diagnosis = (
            diagnosis.join(
                diagnosis_offsets,
                on=[self.icu_stay_id_col, self.diagnosis_start_col],
                how="left",
            )
            # Sort by patient ID, diagnosis code and diagnosis start time
            .sort(
                self.icu_stay_id_col,
                self.diagnosis_icd_code_col,
                self.diagnosis_start_col,
            )
            # 2. Check if diagnosis is continued from the previous log entry
            #    and if it is continued in the next log entry
            .with_columns(
                # Check if diagnosis is continued from the previous log entry
                pl.when(
                    # Check if the previous diagnosis is the same as the current diagnosis
                    pl.col(self.diagnosis_icd_code_col)
                    == pl.col(self.diagnosis_icd_code_col).shift(1),
                    # Check if the previous diagnosis start time is the previous log entry time
                    pl.col("prev_diag_start")
                    == pl.col(self.diagnosis_start_col).shift(1),
                    # Check if the diagnosis priority is the same as the previous diagnosis priority
                    pl.col(self.diagnosis_priority_col)
                    == pl.col(self.diagnosis_priority_col).shift(1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("started"))
                .alias("diag_status_prev"),
                # Check if diagnosis is continued in the next log entry
                pl.when(
                    # Check if the next diagnosis is the same as the current diagnosis
                    pl.col(self.diagnosis_icd_code_col)
                    == pl.col(self.diagnosis_icd_code_col).shift(-1),
                    # Check if the next diagnosis start time is the next log entry time
                    pl.col("next_diag_start")
                    == pl.col(self.diagnosis_start_col).shift(-1),
                    # Check if the diagnosis priority is the same as the previous diagnosis priority
                    pl.col(self.diagnosis_priority_col)
                    == pl.col(self.diagnosis_priority_col).shift(-1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("discontinued"))
                .alias("diag_status_next"),
            )
            # # Filter for rows where the diagnosis status changes
            .filter(pl.col("diag_status_prev") != pl.col("diag_status_next"))
            # 3. Get the end time of the diagnosis if it is discontinued
            .with_columns(
                pl.when(pl.col("diag_status_next") == "discontinued")
                .then(pl.col("next_diag_start"))
                .otherwise(None)
                .alias(self.diagnosis_end_col)
            )
            # Sort by patient ID, diagnosis code and diagnosis start time
            .sort(
                self.icu_stay_id_col,
                self.diagnosis_icd_code_col,
                self.diagnosis_start_col,
            )
            # 4. Combine rows where the diagnosis is started, continued, then discontinued in the next row
            .with_columns(
                pl.when(
                    pl.col("diag_status_prev").shift(1) == "started",
                    pl.col("diag_status_next").shift(1) == "continued",
                    pl.col("diag_status_prev") == "continued",
                    pl.col("diag_status_next") == "discontinued",
                    # Check if the previous diagnosis is the same as the current diagnosis
                    pl.col(self.diagnosis_icd_code_col)
                    == pl.col(self.diagnosis_icd_code_col).shift(1),
                )
                .then(pl.col(self.diagnosis_start_col).shift(1))
                .otherwise(pl.col(self.diagnosis_start_col))
                .alias(self.diagnosis_start_col)
            )
            # 5. Remove the helper columns
            .drop(
                "prev_diag_start",
                "next_diag_start",
                "diag_status_prev",
                "diag_status_next",
            )
            # 6. Set diagnosis active upon discharge to True if it is not discontinued
            .with_columns(
                pl.when(pl.col(self.diagnosis_end_col).is_null())
                .then(True)
                .otherwise(pl.col(self.diagnosis_discharge_col))
                .alias(self.diagnosis_discharge_col)
            ).unique()
        )

        return diagnosis

    # endregion

    # region procedures
    # Extract procedure information from the treatment.csv file
    def extract_treatments(self, verbose=True) -> pl.LazyFrame:
        """
        Extracts procedure/treatment data from the treatment CSV file.

        Reads treatment data, cleans procedure descriptions, and calculates start and end times for each procedure.
        Steps:
            1. Read treatment records from self.treatment_path.
            2. Rename columns (e.g., "patientunitstayid" to {icu_stay_id_col}, "treatmentoffset" to {procedure_start_col}).
            3. Convert treatment offsets from minutes to seconds.
            4. Identify repeated procedure entries and calculate end times.
            5. Clean free-text procedure descriptions.

        Returns:
            pl.LazyFrame: A LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {procedure_start_col}: Start time of the procedure (in seconds).
                - {procedure_description_col}: Cleaned procedure description.
                - {procedure_end_col}: Calculated end time of the procedure (in seconds).
                - {procedure_discharge_col}: Boolean flag indicating active procedure upon discharge.
        """
        if verbose:
            print("eICU    - Extracting procedures...")
        return self._extract_treatments_helper(pl.scan_csv(self.treatment_path))

    def _extract_treatments_helper(
        self, treatment: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Helper function to process treatment/procedure data.

        Processes treatment records by converting time offsets, determining procedure continuation,
        computing end times and cleaning procedure descriptions.
        Steps:
            1. Select and rename treatment columns (e.g., "patientunitstayid" to {icu_stay_id_col}).
            2. Convert treatment offsets from minutes to seconds.
            3. Determine procedure continuation by comparing adjacent rows.
            4. Compute end times for discontinued procedures.
            5. Clean and reformat procedure description text.
            6. Group by appropriate keys to select entries with the longest duration.

        Returns:
            pl.LazyFrame: A processed LazyFrame containing:
                - {icu_stay_id_col}: ICU stay identifier.
                - {procedure_start_col}: Computed start time of the procedure (in seconds).
                - {procedure_description_col}: Re-formatted procedure/treatment description.
                - {procedure_end_col}: End time of the procedure (in seconds), if applicable.
                - {procedure_discharge_col}: Boolean flag indicating if the procedure was active at discharge.
        """
        treatment = (
            treatment
            # Select columns of interest
            .select(
                "patientunitstayid",
                "treatmentoffset",
                "treatmentstring",
                "activeupondischarge",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "treatmentoffset": self.procedure_start_col,
                    "treatmentstring": self.procedure_description_col,
                    "activeupondischarge": self.procedure_discharge_col,
                }
            )
            .join(self.icu_stay_id, on=self.icu_stay_id_col, how="outer")
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.procedure_start_col,
                base_unit="minutes",
            )
        )

        # Get continued procedures where possible, by checking whether the procedure reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        treatment_offsets = (
            treatment.select(self.icu_stay_id_col, self.procedure_start_col)
            .unique()
            .sort(self.icu_stay_id_col, self.procedure_start_col)
            .with_columns(
                pl.col(self.procedure_start_col)
                .shift(1)
                .over(self.icu_stay_id_col)
                .alias("prev_proc_start"),
                pl.col(self.procedure_start_col)
                .shift(-1)
                .over(self.icu_stay_id_col)
                .alias("next_proc_start"),
            )
        )

        treatment = (
            treatment.join(
                treatment_offsets,
                on=[self.icu_stay_id_col, self.procedure_start_col],
                how="left",
            )
            # Sort by patient ID, procedure description and procedure start time
            .sort(
                self.icu_stay_id_col,
                self.procedure_description_col,
                self.procedure_start_col,
            )
            # 2. Check if procedure is continued from the previous log entry
            #    and if it is continued in the next log entry
            .with_columns(
                # Check if procedure is continued from the previous log entry
                pl.when(
                    # Check if the previous procedure is the same as the current procedure
                    pl.col(self.procedure_description_col)
                    == pl.col(self.procedure_description_col).shift(1),
                    # Check if the previous procedure start time is the previous log entry time
                    pl.col("prev_proc_start")
                    == pl.col(self.procedure_start_col).shift(1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("started"))
                .alias("proc_status_prev"),
                # Check if procedure is continued in the next log entry
                pl.when(
                    # Check if the next procedure is the same as the current procedure
                    pl.col(self.procedure_description_col)
                    == pl.col(self.procedure_description_col).shift(-1),
                    # Check if the next procedure start time is the next log entry time
                    pl.col("next_proc_start")
                    == pl.col(self.procedure_start_col).shift(-1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("discontinued"))
                .alias("proc_status_next"),
            )
            # Filter for rows where the procedure status changes
            .filter(pl.col("proc_status_prev") != pl.col("proc_status_next"))
            # 3. Get the end time of the procedure if it is discontinued
            .with_columns(
                pl.when(pl.col("proc_status_next") == "discontinued")
                .then(pl.col("next_proc_start"))
                .otherwise(None)
                .alias(self.procedure_end_col)
            )
            # Sort by patient ID, procedure description and procedure start time
            .sort(
                self.icu_stay_id_col,
                self.procedure_description_col,
                self.procedure_start_col,
            )
            # 4. Combine rows where the procedure is started, continued, then discontinued in the next row
            .with_columns(
                pl.when(
                    pl.col("proc_status_prev").shift(1) == "started",
                    pl.col("proc_status_next").shift(1) == "continued",
                    pl.col("proc_status_prev") == "continued",
                    pl.col("proc_status_next") == "discontinued",
                    # Check if the previous procedure is the same as the current procedure
                    pl.col(self.procedure_description_col)
                    == pl.col(self.procedure_description_col).shift(1),
                )
                .then(pl.col(self.procedure_start_col).shift(1))
                .otherwise(pl.col(self.procedure_start_col))
                .alias(self.procedure_start_col)
            )
            # 5. Continue procedure until discharge if procedure is active upon discharge
            .join(self.icu_length_of_stay, on=self.icu_stay_id_col, how="left")
            .with_columns(
                pl.when(pl.col(self.procedure_discharge_col))
                .then(
                    pl.col(self.procedure_start_col)
                    + pl.duration(
                        days=pl.col(self.icu_length_of_stay_col)
                    ).dt.total_seconds()
                )
                .otherwise(pl.col("next_proc_start"))
                .alias(self.procedure_end_col)
            )
            .filter(pl.col(self.procedure_end_col).is_not_null())
            # 6. Remove the helper columns
            .drop(
                "prev_proc_start",
                "next_proc_start",
                "proc_status_prev",
                "proc_status_next",
                self.icu_length_of_stay_col,
            )
            # 7. Use the rows with the longest duration for each started procedure
            .group_by(
                self.icu_stay_id_col,
                self.procedure_description_col,
                self.procedure_start_col,
            )
            .agg(pl.all().sort_by(self.procedure_end_col).last())
            .with_columns(
                pl.when(
                    pl.col(self.procedure_discharge_col),
                    pl.col(self.procedure_end_col).eq(
                        pl.col(self.procedure_start_col)
                    ),
                )
                .then(None)
                .otherwise(pl.col(self.procedure_end_col))
                .alias(self.procedure_end_col)
            )
            .filter(
                pl.col(self.procedure_end_col).gt(
                    pl.col(self.procedure_start_col)
                )
                | pl.col(self.procedure_end_col).is_null()
            )
            .unique()
        )

        return treatment.with_columns(
            # TODO: make less hacky
            pl.col(self.procedure_description_col)
            .str.replace_all("\|", " - ")
            .str.to_titlecase()
            .str.replace_many(
                {
                    "Ace ": "ACE ",
                    "Afb": "AFB",
                    "Aicd": "AICD",
                    "Arb": "ARB",
                    "Avm": "AVM",
                    "Azt": "AZT",
                    "Bal ": "BAL ",
                    "Bivad": "BIVAD",
                    "Cabg": "CABG",
                    "Ccm": "CCM",
                    "Coa ": "CoA ",
                    "Cpap": "CPAP",
                    "Csf": "CSF",
                    "Ct": "CT",
                    "Ddavp": "DDAVP",
                    "Dvt": "DVT",
                    "Eeg": "EEG",
                    "Emg": "EMG",
                    "Ent": "ENT",
                    "Ercp": "ERCP",
                    "Fio": "FIO",
                    "Gi": "GI",
                    "Hiv": "HIV",
                    "Hmg": "HMG",
                    "Ich": "ICH",
                    "Iiia": "IIIA",
                    "Iii": "III",
                    "Ii": "II",
                    "Iib": "IIB",
                    "Inh ": "INH ",
                    "Iv": "IV",
                    "Ivc": "IVC",
                    "Ivig": "IVIG",
                    "Lr": "LR",
                    "Lvad": "LVAD",
                    "Mri": "MRI",
                    "Mtb": "MTB",
                    "Ns": "NS",
                    "Nsaid": "NSAID",
                    "Okt": "OKT",
                    "Or ": "OR ",
                    "Pbs": "PBS",
                    "Pca": "PCA",
                    "Peep": "PEEP",
                    "Peg": "PEG",
                    "Prbc": "PRBC",
                    "Ppn": "PPN",
                    "Rvad": "RVAD",
                    "Sled": "SLED",
                    "Ssri": "SSRI",
                    "Tc": "TC",
                    "Tips": "TIPS",
                    "Tpn": "TPN",
                    "Tsh": "TSH",
                    "Vii": "VII",
                    "Vk": "VK",
                    "Vte": "VTE",
                    # SPECIAL CASES
                    "pco2": "pCO2",
                    "To ": "to ",
                    "And ": "and ",
                    "Of ": "of ",
                    "Ml": "mL",
                    "Min": "min",
                    "Kg": "kg",
                    "Via ": "via ",
                    ""
                    # and slash without space before
                    "/ ": " / ",
                }
            )
            .str.replace("  / ", " / ")
        ).cast({self.procedure_discharge_col: bool})
