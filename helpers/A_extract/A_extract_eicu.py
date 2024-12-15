# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the data from the source files and provides the extracted data
# in a structured format for further processing and harmonization.

import numpy as np
import pandas as pd
import polars as pl

from helpers.helper_filepaths import EICUPaths
from helpers.helper import GlobalHelpers


class EICUExtractor(EICUPaths):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        self.path = paths.eicu_source_path
        self.helpers = GlobalHelpers()
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
        Extracts patient information from the patient.csv file.

        Return a polars LazyFrame with the extracted patient information, containing the following columns:
        - ICU stay ID
        - Hospital stay ID
        - Person ID
        - Gender
        - Age
        - Height
        - Weight
        - Ethnicity
        - pre-ICU length of stay
        - ICU length of stay
        - Mortality in hospital
        - Mortality in ICU
        - Mortality after ICU discharge
        - Admission location
        - Unit type
        - Care site
        - Discharge location

        :return: A polars LazyFrame with the extracted patient information.
        :rtype: pl.LazyFrame
        """

        return (
            pl.scan_csv(self.patient_path)
            .select(  # Select columns of interest
                [
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
                ]
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
            .select(
                self.icu_stay_id_col,
                self.hospital_stay_id_col,
                self.person_id_col,
                self.icu_stay_seq_num_col,
                self.gender_col,
                self.age_col,
                self.height_col,
                self.weight_col,
                self.ethnicity_col,
                self.admission_type_col,
                self.admission_time_col,
                self.admission_diagnosis_col,
                self.pre_icu_length_of_stay_col,
                self.icu_length_of_stay_col,
                self.hospital_length_of_stay_col,
                self.mortality_hosp_col,
                self.mortality_icu_col,
                self.mortality_after_col,
                self.admission_loc_col,
                self.unit_type_col,
                self.care_site_col,
                self.discharge_loc_col,
            )
        )

    # endregion

    # region admitDX
    # Extract admission diagnosis information from the admissionDx.csv file
    def extract_admission_diagnoses(self) -> pl.LazyFrame:
        """
        Extracts admission diagnosis information from the admissionDx.csv file.

        :return: A polars LazyFrame with the extracted admission diagnosis information.
        :rtype: pl.LazyFrame
        """

        return (
            pl.scan_csv(self.admissionDx_path)
            .select("patientunitstayid", "admitdxpath", "admitdxname")
            .rename({"patientunitstayid": self.icu_stay_id_col})
            .with_columns(
                # Admission Type
                pl.when(
                    pl.col("admitdxpath") == "admission diagnosis|Elective|Yes"
                )
                .then(pl.lit("Elective"))
                .when(
                    pl.col("admitdxpath") == "admission diagnosis|Elective|No"
                )
                .then(pl.lit("Emergency"))
                .when(
                    pl.col("admitdxpath")
                    == "admission diagnosis|Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?|Yes"
                )
                .then(pl.lit("Surgical"))
                .when(
                    pl.col("admitdxpath")
                    == "admission diagnosis|Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?|No"
                )
                .then(pl.lit("Medical"))
                .otherwise(None)
                .alias(self.admission_type_col),
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
                    .str.replace("\|", " - ")
                    # clean comments
                    .str.replace(
                        " (with or without respiratory arrest; for respiratory arrest see Respiratory System)",
                        "",
                    )
                    .str.replace(
                        " (for gastrointestinal bleeding GI-see GI system) (for trauma see Trauma)",
                        "",
                    )
                    .str.replace(
                        " (for cerebrovascular accident-see Neurological System)",
                        "",
                    )
                    .str.replace(", Do not include shock states", "")
                    .str.replace(
                        " (for hepatic see GI, for diabetic see Endocrine, if related to cardiac arrest, see CV)",
                        "",
                    )
                    .str.replace("-no structural brain disease", "")
                    .str.replace(", for fractures due to trauma see Trauma", "")
                    # harmonize comments
                    .str.replace("Hematoma subdural", "Hematoma, subdural")
                    .str.replace("Hematoma-epidural", "Hematoma, epidural")
                )
                .otherwise(None)
                .alias(self.admission_diagnosis_col),
            )
            .sort(self.icu_stay_id_col, "admitdxpath")
            .group_by(self.icu_stay_id_col)
            .agg(
                pl.col(self.admission_diagnosis_col).first(),
                pl.col(self.admission_type_col)
                .str.concat(" ")
                .str.strip_chars(),
            )
            .cast({self.admission_type_col: self.admission_types_dtype})
            .select(
                self.icu_stay_id_col,
                self.admission_type_col,
                self.admission_diagnosis_col,
            )
        )

    # region lab TS
    # Extract time series information for lab values from the lab.csv file
    def extract_time_series_lab(self) -> pl.LazyFrame:
        """
        Extracts time series information for lab values from the lab.csv file.
        Pivoting has to be done in a separate step.

        Return a polars LazyFrame with the extracted lab values in long format, containing the following columns:
        - ICU stay ID
        - Time
        - Lab name
        - Lab result

        :return: A polars LazyFrame with the extracted lab values.
        :rtype: pl.LazyFrame
        """

        lab_names_mapping = self.helpers.load_mapping(self.lab_mapping_path)

        return (
            pl.scan_csv(self.lab_path)
            .select(
                "patientunitstayid", "labname", "labresultoffset", "labresult"
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "labresultoffset": self.timeseries_time_col,
                }
            )
            .with_columns(
                # Replace lab names with mapped names
                pl.col("labname")
                .replace_strict(lab_names_mapping, default=None)
                .alias("labname")
            )
            # Filter for lab names of interest
            .filter(
                # pl.col("labname").is_in(self.all_values + self.other_lab_values)
                pl.col("labname")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "_"])
                .struct.field("variable")
                .is_in(self.relevant_lab_values + self.other_lab_values)
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
            .with_columns(
                pl.col("labname")
                .str.split_exact(by=" by ", n=1)
                .struct.rename_fields(["variable_source", "method"])
                .alias("fields1")
            )
            .unnest("fields1")
            .with_columns(
                pl.col("variable_source")
                .str.replace("in HDL", "inHDL")
                .str.replace("in LDL", "inLDL")
                .str.replace(" (in|of) ", " INOF ")
                .str.split_exact(by=" INOF ", n=1)
                .struct.rename_fields(["variable", "source"])
                .alias("fields2")
            )
            .unnest("fields2")
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                pl.col("variable")
                .str.replace("inHDL", "in HDL")
                .str.replace("inLDL", "in LDL")
                .alias("labname"),
                pl.struct(
                    value="labresult", source="source", method="method"
                ).alias("value_struct"),
            )
        )

    # endregion

    # region resp TS
    # Extract time series information for respiratory values from the respiratorycharting.csv file
    def extract_time_series_resp(self) -> pl.LazyFrame:
        """
        Extracts time series information for respiratory values from the respiratorycharting.csv file.

        Return a polars LazyFrame with the extracted respiratory values in long format, containing the following columns:
        - ICU stay ID
        - Time
        - Respiratory value name
        - Respiratory value

        :return: A polars LazyFrame with the extracted respiratory values.
        :rtype: pl.LazyFrame
        """

        # NOTE: ASSUMPTION: These are the respiratory values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        keep_resp_names = self.relevant_respiratory_values
        resp_names_mapping = self.helpers.load_mapping(self.resp_mapping_path)

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
                .str.replace("Refused after education", "")
                # .cast(float, strict=False),
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
        Extracts time series information for nurse values from the nurseCharting.csv file.

        Return a polars LazyFrame with the extracted nurse values in long format, containing the following columns:
        - ICU stay ID
        - Time
        - Nurse value name
        - Nurse value

        :return: A polars LazyFrame with the extracted nurse values.
        :rtype: pl.LazyFrame
        """

        # NOTE: ASSUMPTION: These are the nurse values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        keep_nurse_names = [
            "Non-Invasive BP",
            "Invasive BP",
            "Heart Rate",
            # "Pain Score/Goal",
            "Respiratory Rate",
            "O2 Saturation",
            "Temperature",
            "Glasgow coma score",
            "Invasive BP",
            "Bedside Glucose",
            "O2 L/%",
            "O2 Admin Device",
            "Sedation Scale/Score/Goal",
            # "Delirium Scale/Score",
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
            )  # Filter for nurse names of interest
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

        nurseCharting_RASS = nurseCharting.filter(
            pl.col("nursingchartvalue") == "Sedation Score",
            pl.col(self.timeseries_time_col).is_in(
                nurseCharting.filter(
                    pl.col("nursingchartcelltypevalname") == "Sedation Score",
                    pl.col("nursingchartvalue") == "RASS",
                )
                .select(self.timeseries_time_col)
                .collect(streaming=True)
                .to_series()
            ),
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
                # Map O2 delivery device values
                pl.when(
                    pl.col("nursingchartcelltypevalname") == "O2 Admin Device"
                )
                .then(
                    pl.col("nursingchartvalue").replace_strict(
                        nurse_oxygen_delivery_device_mapping, default=None
                    )
                )
                .otherwise(pl.col("nursingchartvalue"))
                .alias("nursingchartvalue"),
                # Replace nurse names with mapped names
                pl.col("nursingchartcelltypevalname")
                .replace_strict(nurse_names_mapping, default=None)
                .alias("nursingchartcelltypevalname"),
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
        Extracts time series information for intake/output values from the intakeOutput.csv file.

        Return a polars LazyFrame with the extracted intake/output values in long format, containing the following columns:
        - ICU stay ID
        - Time
        - Intake/output value name
        - Intake/output value

        :return: A polars LazyFrame with the extracted intake/output values.
        :rtype: pl.LazyFrame
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
                    self.relevant_intakeoutput_values_mapping, default=None
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
        Extracts time series information for periodic values from the vitalPeriodic.csv file.

        Return a polars LazyFrame with the extracted periodic values in wide format, containing the following columns:
        - ICU stay ID
        - Time
        - Temperature
        - Oxygen saturation by pulse oximetry
        - Heart rate
        - Respiratory rate
        - Central venous pressure
        - end tidal CO2
        - Invasive systolic blood pressure
        - Invasive diastolic blood pressure
        - Invasive mean blood pressure
        - Intracranial pressure

        :return: A polars LazyFrame with the extracted periodic values.
        :rtype: pl.LazyFrame
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
            .unique()
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
        Extracts time series information for aperiodic values from the vitalAperiodic.csv file.

        Return a polars LazyFrame with the extracted aperiodic values in wide format, containing the following columns:
        - ICU stay ID
        - Time
        - Non-invasive systolic blood pressure
        - Non-invasive diastolic blood pressure
        - Non-invasive mean blood pressure

        :return: A polars LazyFrame with the extracted aperiodic values.
        :rtype: pl.LazyFrame
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
            .unique()
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
        Extracts and combines the aperiodic and periodic time series data.

        Return a polars LazyFrame with the extracted and combined time series data, containing the following columns:
        - ICU stay ID
        - Time
        - Temperature
        - Oxygen saturation by pulse oximetry
        - Heart rate
        - Respiratory rate
        - Central venous pressure
        - end tidal CO2
        - Invasive systolic blood pressure
        - Invasive diastolic blood pressure
        - Invasive mean blood pressure
        - Intracranial pressure
        - Non-invasive systolic blood pressure
        - Non-invasive diastolic blood pressure
        - Non-invasive mean blood pressure

        :return: A polars LazyFrame with the extracted and combined time series data.
        :rtype: pl.LazyFrame
        """

        periodic_mapping = self.helpers.load_mapping(self.periodic_mapping_path)

        periodic = self.extract_time_series_periodic()
        aperiodic = self.extract_time_series_aperiodic()

        periodics = pl.concat(
            [periodic, aperiodic], how="diagonal_relaxed"
        ).rename(periodic_mapping)
        periodics_cols = periodics.collect_schema().names()

        return periodics.select(
            [self.icu_stay_id_col, self.timeseries_time_col]
            + list(
                set(periodics_cols).intersection(
                    set(self.relevant_vital_values + ["Temperature Fahrenheit"])
                )
            )
        )

    # endregion

    # region medication
    # Extract medication information from the different medication files
    # TODO: add administration path
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extracts medication information from the medication files.
        Medication information is extracted from the medication.csv, infusionDrug.csv and admissionDrug.csv files.

        Medication names are mapped to a common set of medication names using the MEDICATIONS.yaml mapping file.

        Return a polars LazyFrame with the extracted medication information, containing the following columns:
        - ICU stay ID
        - Start time
        - End time
        - Medication name
        - Medication ingredient
        - Medication amount
        - Medication amount unit

        :return: A polars LazyFrame with the extracted medication information.
        :rtype: pl.LazyFrame
        """

        print("eICU    - Extracting medications...")

        eicu_medication_mapping = self.helpers.load_many_to_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS.yaml", "eicu"
        )
        eicu_drug_administration_route_mapping = self.helpers.load_mapping(
            self.drug_administration_route_mapping_path
        )

        # # NOTE: Extremely infrequently used.
        # # cf. w/ Important considerations @ https://eicu-crd.mit.edu/eicutables/admissiondrug/
        # admissiondrug = (
        #     pl.scan_csv(self.admissiondrug_path)
        #     .select("patientunitstayid", "drugoffset", "drugname")
        #     .rename(
        #         {
        #             "patientunitstayid": self.icu_stay_id_col,
        #             "drugoffset": self.timeseries_time_col,
        #             "drugname": "medication",
        #         }
        #     )
        #     .pipe(
        #         self.helpers._convert_time_to_seconds_float,
        #         self.timeseries_time_col,
        #         base_unit="minutes",
        #     )
        # )

        # NOTE: a lot of calcalations can be done here
        # cf. w/ Important considerations @ https://eicu-crd.mit.edu/eicutables/infusiondrug/
        infusiondrug = (
            pl.scan_csv(self.infusionDrug_path)
            .select(
                "patientunitstayid",
                "infusionoffset",
                "drugname",
                "infusionrate",
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "infusionoffset": self.drug_start_col,
                    "drugname": self.drug_name_col,
                    "infusionrate": self.drug_rate_col,
                }
            )
            .with_columns(
                # Get unit from drugname
                pl.col(self.drug_name_col)
                .str.extract(r".*\((.*?)\)$")
                .alias(self.drug_rate_unit_col),
                # Replace drug names with mapped names
                pl.col(self.drug_name_col)
                .replace_strict(eicu_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
                # Set administration route
                pl.lit("intravenous").alias(self.drug_admin_route_col),
            )
            # Remove rows with empty drug names
            .filter(pl.col(self.drug_name_col).is_not_null())
            # Remove rows with empty drug rates
            .filter(
                pl.col(self.drug_rate_col).is_not_null()
                | (pl.col(self.drug_rate_col) != "")
            )
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
        )

        medication = (
            pl.scan_csv(self.medication_path)
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
        Extracts diagnosis information from the diagnosis.csv file.

        Return a polars LazyFrame with the extracted diagnosis information, containing the following columns:
        - ICU stay ID
        - Diagnosis ICD code
        - Diagnosis ICD code version (ICD '9' or '10')
        - Diagnosis start time
        - Diagnosis priority (Primary, Major, Other -> 1, 2, 3)
        - Diagnosis active upon discharge (True/False)

        :return: A polars LazyFrame with the extracted diagnosis information.
        :rtype: pl.LazyFrame
        """

        diagnosis = (
            pl.scan_csv(self.path + "diagnosis.csv.gz")
            .select(  # Select columns of interest
                [
                    "patientunitstayid",
                    "diagnosisoffset",
                    "icd9code",
                    "activeupondischarge",
                    "diagnosispriority",
                ]
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
                [
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
                ]
            )
            .drop(["icd9code", "diagnosisoffset", "diagnosispriority"])
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
    def extract_treatments(self) -> pl.LazyFrame:
        """
        Extracts procedure information from the treatment.csv file.

        Procedures in eICU are not well defined and are stored as free text in the treatmentstring column.
        -> TODO: Extract procedure information from the treatmentstring column.

        Return a polars LazyFrame with the extracted procedure information, containing the following columns:
        - ICU stay ID
        - Procedure start time
        - Procedure description
        - Procedure active upon discharge (True/False)

        :return: A polars LazyFrame with the extracted procedure information.
        :rtype: pl.LazyFrame
        """

        print("eICU    - Extracting procedures...")

        IDs = self.extract_patient_information().select(
            self.icu_stay_id_col,
            self.hospital_stay_id_col,
            self.person_id_col,
        )

        return (
            pl.scan_csv(self.treatment_path)
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
            .with_columns(
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
            )
            .join(IDs, on=self.icu_stay_id_col, how="outer")
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.procedure_start_col,
                base_unit="minutes",
            )
            .cast({self.procedure_discharge_col: bool})
            .select(
                self.person_id_col,
                self.hospital_stay_id_col,
                self.icu_stay_id_col,
                self.procedure_start_col,
                self.procedure_description_col,
                self.procedure_discharge_col,
            )
        )
