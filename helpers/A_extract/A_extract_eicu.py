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
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.eicu_source_path
        self.helpers = GlobalHelpers()
        self.icu_stay_id = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.hospital_stay_id_col, self.person_id_col]
        )
        self.icu_length_of_stay = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.icu_length_of_stay_col]
        )

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
                    # "unitvisitnumber",
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
                    "patientunitstayid": self.icu_stay_id_col,
                    "patienthealthsystemstayid": self.hospital_stay_id_col,
                    "gender": self.gender_col,
                    "age": self.age_col,
                    "ethnicity": self.ethnicity_col,
                    "admissionheight": self.height_col,
                    "admissionweight": self.weight_col,
                    "unittype": self.unit_type_col,
                    "unitdischargestatus": self.mortality_icu_col,
                    "unitadmitsource": self.admission_loc_col,
                    "unitdischargelocation": self.discharge_loc_col,
                    "unitdischargeoffset": self.icu_length_of_stay_col,
                    "hospitaldischargestatus": self.mortality_hosp_col,
                    "hospitalid": self.care_site_col,
                }
            )
            .sort(self.icu_stay_id_col)
            .with_columns(
                # Convert categorical gender to enum
                pl.col(self.gender_col).replace("", "Unknown").cast(self.gender_dtype),
                # Convert categorical ethnicity to enum
                pl.col(self.ethnicity_col).replace(self.ETHNICITY_MAP).cast(self.ethnicity_dtype),
                # NOTE: ASSUMPTION: Replace age values "> 89" with 90 and convert to float
                pl.col(self.age_col).replace("> 89", 90).cast(int, strict=False),
                # Calculate pre ICU length of stay
                # Reverse sign of hospitaladmitoffset to get pre_icu_length_of_stay
                (0 - pl.col("hospitaladmitoffset"))
                .cast(float)
                .alias(self.pre_icu_length_of_stay_col),
                # Calculate ICU mortality
                (pl.col(self.mortality_icu_col) == "Expired").cast(bool),
                # # Convert categorical mortality to enum
                # (
                #     pl.col(self.mortality_icu_col)
                #     .replace({"Expired": "Dead", "": "Unknown"})
                #     .cast(self.mortality_dtype)
                # ),
                # Calculate hospital mortality
                (pl.col(self.mortality_hosp_col) == "Expired").cast(bool),
                # Calculate mortality after discharge
                pl.when(
                    (pl.col(self.mortality_icu_col) != "Expired")
                    & (pl.col(self.mortality_hosp_col) == "Expired")
                )
                .then(pl.col("hospitaldischargeoffset") - pl.col(self.icu_length_of_stay_col))
                .otherwise(None)
                .alias(self.mortality_after_col),
                # # Calculate hospital_length_of_stay as difference between hospitaldischargeoffset
                # # and hospitaladmitoffset
                # (pl.col("hospitaldischargeoffset") - pl.col("hospitaladmitoffset")).alias(
                #     self.hospital_length_of_stay_col
                # ),
                # Convert categorical admission location to enum
                pl.col(self.admission_loc_col)
                .replace(self.ADMISSION_LOCATIONS_MAP)
                .cast(self.admission_locations_dtype),
                # Convert categorical unit type to enum
                pl.col(self.unit_type_col).replace(self.UNIT_TYPES_MAP).cast(self.unit_types_dtype),
                # Convert categorical discharge location to enum
                pl.col(self.discharge_loc_col)
                .replace(self.DISCHARGE_LOCATIONS_MAP)
                .cast(self.discharge_locations_dtype),
            )
            .drop(["hospitaladmitoffset", "hospitaldischargeoffset"])
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
                self.mortality_after_col,
                base_unit="minutes",
            )
            .select(
                [
                    self.icu_stay_id_col,
                    self.hospital_stay_id_col,
                    self.person_id_col,
                    self.gender_col,
                    self.age_col,
                    self.height_col,
                    self.weight_col,
                    self.ethnicity_col,
                    self.pre_icu_length_of_stay_col,
                    self.icu_length_of_stay_col,
                    self.mortality_hosp_col,
                    self.mortality_icu_col,
                    self.mortality_after_col,
                    self.admission_loc_col,
                    self.unit_type_col,
                    self.care_site_col,
                    self.discharge_loc_col,
                ]
            )
        )

    # endregion

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

        # NOTE: ASSUMPTION: These are the lab values of interest
        # TODO: Confer with medical experts to confirm these are the correct values
        keep_lab_names = self.relevant_lab_values + ["base_excess", "base_deficit"]
        lab_names_mapping = self.helpers.load_mapping(self.lab_mapping_path)

        return (
            pl.scan_csv(self.lab_path)
            .select(["patientunitstayid", "labname", "labresultoffset", "labresult"])
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
            .filter(pl.col("labname").is_in(keep_lab_names))
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
                [
                    "patientunitstayid",
                    "respchartoffset",
                    "respchartvaluelabel",
                    "respchartvalue",
                ]
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
                # Remove percentage sign from respchartvalue and convert to float
                pl.col("respchartvalue").str.replace("%", "").cast(float, strict=False),
            )
            # Filter for resp names of interest
            .filter(pl.col("respchartvaluelabel").is_in(keep_resp_names))
            # Remove rows with empty resp values
            .filter(pl.col("respchartvalue").is_not_null())
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
            # "Sedation Scale/Score/Goal",
            # "Delirium Scale/Score",
        ]
        nurse_names_mapping = self.load_mapping(self.nurse_mapping_path)
        nurse_oxygen_delivery_device_mapping = self.load_mapping(
            self.nurse_oxygen_delivery_device_mapping_path
        )

        return (
            pl.scan_csv(self.nurseCharting_path)
            .select(
                [
                    "patientunitstayid",
                    "nursingchartoffset",
                    "nursingchartcelltypevallabel",
                    "nursingchartcelltypevalname",
                    "nursingchartvalue",
                ]
            )
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "nursingchartoffset": self.timeseries_time_col,
                }
            )
            # Filter for nurse names of interest
            .filter(pl.col("nursingchartcelltypevallabel").is_in(keep_nurse_names))
            .drop(["nursingchartcelltypevallabel"])
            # Remove rows with empty nurse values
            .filter(pl.col("nursingchartvalue").is_not_null())
            # Remove duplicate rows
            .unique()
            .with_columns(
                # Convert Fahrenheit to Celsius
                pl.when(pl.col("nursingchartcelltypevalname") == "Temperature (F)")
                .then((pl.col("nursingchartvalue").cast(float, strict=False) - 32) * 5 / 9)
                .otherwise(pl.col("nursingchartvalue"))
                .alias("nursingchartvalue"),
            )
            .with_columns(
                # Map O2 delivery device values
                pl.when(pl.col("nursingchartcelltypevalname") == "O2 Admin Device")
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
        keep_inout_names = self.relevant_intakeoutput_values

        return (
            pl.scan_csv(self.intakeOutput_path)
            .select(
                [
                    "patientunitstayid",
                    "intakeoutputoffset",
                    # "intaketotal",
                    # "outputtotal",
                    # "dialysistotal",
                    # "nettotal",
                    "celllabel",
                    "cellvaluenumeric",
                ]
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
                pl.col("celllabel")
                .replace_strict(intakeoutput_mapping, default=None)
                .alias("celllabel"),
            )
            # Filter for intakeoutput names of interest
            .filter(pl.col("celllabel").is_in(keep_inout_names))
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
                [
                    "patientunitstayid",
                    "observationoffset",
                    "noninvasivesystolic",
                    "noninvasivediastolic",
                    "noninvasivemean",
                ]
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

        periodics = periodic.join(
            aperiodic,
            on=[self.icu_stay_id_col, self.timeseries_time_col],
        ).rename(periodic_mapping)

        return periodics.select(
            [
                self.icu_stay_id_col,
                self.timeseries_time_col,
            ]
            + list(
                set(periodics.collect_schema().names()).intersection(
                    set(self.relevant_vital_values + ["temperature_F"])
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

        eicu_medication_mapping = self.helpers.load_many_to_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS.yaml", "eicu"
        )

        # # NOTE: Extremely infrequently used.
        # # cf. w/ Important considerations @ https://eicu-crd.mit.edu/eicutables/admissiondrug/
        # admissiondrug = (
        #     pl.scan_csv(self.admissiondrug_path)
        #     .select(["patientunitstayid", "drugoffset", "drugname"])
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
            pl.scan_csv(self.infusionDrug_path).select(
                [
                    "patientunitstayid",
                    "infusionoffset",
                    "drugname",
                    "infusionrate",
                    "drugamount",
                    "volumeoffluid",
                ]
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "infusionoffset": self.drug_start_col,
                    "drugname": self.drug_name_col,
                    "drugamount": self.drug_amount_col,
                }
            )
            # Calculate infusion duration where possible
            .with_columns(
                (
                    pl.col(self.drug_start_col)
                    + pl.duration(
                        hours=(pl.col("volumeoffluid").truediv(pl.col("infusionrate")))
                    ).truediv(pl.duration(seconds=1))
                ).alias(self.drug_end_col)
            )
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(eicu_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
            )
            # Remove rows with empty drug names
            .filter(pl.col(self.drug_name_col).is_not_null())
            # # Remove rows with empty ingredient names
            # .filter(pl.col(self.drug_ingredient_col).is_not_null())
            .drop(["volumeoffluid", "infusionrate"])
            # Convert time to seconds
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.drug_start_col,
                base_unit="minutes",
            )
        )

        medication = (
            pl.scan_csv(self.medication_path)
            .select(
                ["patientunitstayid", "drugstartoffset", "drugname", "dosage", "drugstopoffset"]
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientunitstayid": self.icu_stay_id_col,
                    "drugstartoffset": self.drug_start_col,
                    "drugname": self.drug_name_col,
                    "dosage": self.drug_amount_col,
                    "drugstopoffset": self.drug_end_col,
                }
            )
            # Dropping drug dosages due to bad data quality
            .drop(self.drug_amount_col)
            # Replace drug names with mapped names
            .with_columns(
                pl.col(self.drug_name_col)
                .replace_strict(eicu_medication_mapping, default=None)
                .alias(self.drug_ingredient_col),
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

        return (
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
                    "activeupondischarge": "active_upon_discharge",
                }
            )
            .join(self.icu_stay_id, on=self.icu_stay_id_col)
            .with_columns(  # Convert columns to appropriate data types
                [
                    # Split diagnosis codes by comma and rename column
                    pl.col("icd9code").str.split(by=", ").alias(self.diagnosis_icd_code_col),
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
        )

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

        IDs = self.extract_patient_information().select(
            [self.icu_stay_id_col, self.hospital_stay_id_col, self.person_id_col]
        )

        return (
            pl.scan_csv(self.treatment_path)
            .select(  # Select columns of interest
                [
                    "patientunitstayid",
                    "treatmentoffset",
                    "treatmentstring",
                    "activeupondischarge",
                ]
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
            .join(IDs, on=self.icu_stay_id_col)
            .pipe(
                self.helpers._convert_time_to_seconds_float,
                self.procedure_start_col,
                base_unit="minutes",
            )
            .cast({self.procedure_discharge_col: bool})
            .select(
                [
                    self.person_id_col,
                    self.hospital_stay_id_col,
                    self.icu_stay_id_col,
                    self.procedure_start_col,
                    self.procedure_description_col,
                    self.procedure_discharge_col,
                ]
            )
        )
