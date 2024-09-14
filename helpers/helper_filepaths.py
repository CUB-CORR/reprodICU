# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy import of the data paths.

import polars as pl

from helpers.helper import GlobalVars


# region eICU
class EICUPaths(GlobalVars):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        eicu_path = paths.eicu_source_path

        # eICU raw data paths
        self.admissionDrug_path = eicu_path + "admissionDrug.csv.gz"
        self.admissionDx_path = eicu_path + "admissionDx.csv.gz"
        self.allergy_path = eicu_path + "allergy.csv.gz"
        self.apacheApsVar_path = eicu_path + "apacheApsVar.csv.gz"
        self.apachePatientResult_path = eicu_path + "apachePatientResult.csv.gz"
        self.apachePredVar_path = eicu_path + "apachePredVar.csv.gz"
        self.carePlanCareProvider_path = (
            eicu_path + "carePlanCareProvider.csv.gz"
        )
        self.carePlanEOL_path = eicu_path + "carePlanEOL.csv.gz"
        self.carePlanGeneral_path = eicu_path + "carePlanGeneral.csv.gz"
        self.carePlanGoal_path = eicu_path + "carePlanGoal.csv.gz"
        self.carePlanInfectiousDisease_path = (
            eicu_path + "carePlanInfectiousDisease.csv.gz"
        )
        self.customLab_path = eicu_path + "customLab.csv.gz"
        self.diagnosis_path = eicu_path + "diagnosis.csv.gz"
        self.hospital_path = eicu_path + "hospital.csv.gz"
        self.infusionDrug_path = eicu_path + "infusionDrug.csv.gz"
        self.intakeOutput_path = eicu_path + "intakeOutput.csv.gz"
        self.lab_path = eicu_path + "lab.csv.gz"
        self.medication_path = eicu_path + "medication.csv.gz"
        self.microLab_path = eicu_path + "microLab.csv.gz"
        self.note_path = eicu_path + "note.csv.gz"
        self.nurseAssessment_path = eicu_path + "nurseAssessment.csv.gz"
        self.nurseCare_path = eicu_path + "nurseCare.csv.gz"
        self.nurseCharting_path = eicu_path + "nurseCharting.csv.gz"
        self.pastHistory_path = eicu_path + "pastHistory.csv.gz"
        self.patient_path = eicu_path + "patient.csv.gz"
        self.physicalExam_path = eicu_path + "physicalExam.csv.gz"
        self.respiratoryCare_path = eicu_path + "respiratoryCare.csv.gz"
        self.respiratoryCharting_path = eicu_path + "respiratoryCharting.csv.gz"
        self.treatment_path = eicu_path + "treatment.csv.gz"
        self.vitalAperiodic_path = eicu_path + "vitalAperiodic.csv.gz"
        self.vitalPeriodic_path = eicu_path + "vitalPeriodic.csv.gz"

        # eICU DEMO data paths
        if DEMO == True:
            eicu_path = paths.eicu_demo_source_path
            self.admissionDrug_path = eicu_path + "admissionDrug.csv"
            self.admissionDx_path = eicu_path + "admissionDx.csv"
            self.allergy_path = eicu_path + "allergy.csv"
            self.apacheApsVar_path = eicu_path + "apacheApsVar.csv"
            self.apachePatientResult_path = (
                eicu_path + "apachePatientResult.csv"
            )
            self.apachePredVar_path = eicu_path + "apachePredVar.csv"
            self.carePlanCareProvider_path = (
                eicu_path + "carePlanCareProvider.csv"
            )
            self.carePlanEOL_path = eicu_path + "carePlanEOL.csv"
            self.carePlanGeneral_path = eicu_path + "carePlanGeneral.csv"
            self.carePlanGoal_path = eicu_path + "carePlanGoal.csv"
            self.carePlanInfectiousDisease_path = (
                eicu_path + "carePlanInfectiousDisease.csv"
            )
            self.customLab_path = eicu_path + "customLab.csv"
            self.diagnosis_path = eicu_path + "diagnosis.csv"
            self.hospital_path = eicu_path + "hospital.csv"
            self.infusionDrug_path = eicu_path + "infusionDrug.csv"
            self.intakeOutput_path = eicu_path + "intakeOutput.csv"
            self.lab_path = eicu_path + "lab.csv"
            self.medication_path = eicu_path + "medication.csv"
            self.microLab_path = eicu_path + "microLab.csv"
            self.note_path = eicu_path + "note.csv"
            self.nurseAssessment_path = eicu_path + "nurseAssessment.csv"
            self.nurseCare_path = eicu_path + "nurseCare.csv"
            self.nurseCharting_path = eicu_path + "nurseCharting.csv"
            self.pastHistory_path = eicu_path + "pastHistory.csv"
            self.patient_path = eicu_path + "patient.csv"
            self.physicalExam_path = eicu_path + "physicalExam.csv"
            self.respiratoryCare_path = eicu_path + "respiratoryCare.csv"
            self.respiratoryCharting_path = (
                eicu_path + "respiratoryCharting.csv"
            )
            self.treatment_path = eicu_path + "treatment.csv"
            self.vitalAperiodic_path = eicu_path + "vitalAperiodic.csv"
            self.vitalPeriodic_path = eicu_path + "vitalPeriodic.csv"

        # eICU custom mapping paths
        self.eICU_mapping_path = self.mapping_path + "eicu/"
        self.lab_mapping_path = self.eICU_mapping_path + "eicu_lab.yaml"
        self.resp_mapping_path = (
            self.eICU_mapping_path + "eicu_respiratoryCharting.yaml"
        )
        self.intakeoutput_mapping_path = (
            self.eICU_mapping_path + "eicu_intakeOutput.yaml"
        )
        self.nurse_mapping_path = (
            self.eICU_mapping_path + "eicu_nurseCharting.yaml"
        )
        self.nurse_oxygen_delivery_device_mapping_path = (
            self.eICU_mapping_path
            + "eicu_nurseCharting_oxygenDeliveryDevices.yaml"
        )
        self.periodic_mapping_path = (
            self.eICU_mapping_path + "eicu_vitalPeriodic.yaml"
        )
        self.medication_mapping_path = (
            self.eICU_mapping_path + "eicu_medication.yaml"
        )


# endregion


# region HiRID
class HiRIDPaths(GlobalVars):
    def __init__(self, paths):
        super().__init__(paths)
        hirid_path = paths.hirid_source_path

        # HiRID raw data paths
        self.reference_data_path = hirid_path + "reference_data/"
        self.raw_stage_path = hirid_path + "raw_stage/"
        self.general_table_path = self.reference_data_path + "general_table.csv"
        self.variable_reference_path = (
            self.reference_data_path + "hirid_variable_reference.csv"
        )
        self.timeseries_path = (
            self.raw_stage_path + "observation_tables/parquet/"
        )
        self.pharma_path = self.raw_stage_path + "pharma_records/parquet/"
        self.imputed_stage_path = (
            hirid_path + "imputed_stage/imputed_stage/parquet/"
        )

        # HiRID custom mapping paths
        self.hirid_mapping_path = self.mapping_path + "hirid/"
        self.observation_mapping_path = (
            self.hirid_mapping_path + "hirid_OBSERVATION.yaml"
        )


# endregion


# region MIMIC-III
class MIMIC3Paths(GlobalVars):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        mimic3_path = paths.mimic3_source_path

        # MIMIC-III raw data paths
        self.admissions_path = mimic3_path + "ADMISSIONS.csv.gz"
        self.chartevents_path = mimic3_path + "CHARTEVENTS.csv.gz"
        self.d_icd_diagnoses_path = mimic3_path + "D_ICD_DIAGNOSES.csv.gz"
        self.d_icd_procedures_path = mimic3_path + "D_ICD_PROCEDURES.csv.gz"
        self.d_items_path = mimic3_path + "D_ITEMS.csv.gz"
        self.d_labitems_path = mimic3_path + "D_LABITEMS.csv.gz"
        self.diagnoses_icd_path = mimic3_path + "DIAGNOSES_ICD.csv.gz"
        self.icustays_path = mimic3_path + "ICUSTAYS.csv.gz"
        self.inputevents_cv_path = mimic3_path + "INPUTEVENTS_CV.csv.gz"
        self.inputevents_mv_path = mimic3_path + "INPUTEVENTS_MV.csv.gz"
        self.labevents_path = mimic3_path + "LABEVENTS.csv.gz"
        self.outputevents_path = mimic3_path + "OUTPUTEVENTS.csv.gz"
        self.patients_path = mimic3_path + "PATIENTS.csv.gz"
        self.prescriptions_path = mimic3_path + "PRESCRIPTIONS.csv.gz"
        self.procedureevents_mv_path = mimic3_path + "PROCEDUREEVENTS_MV.csv.gz"
        self.procedures_icd_path = mimic3_path + "PROCEDURES_ICD.csv.gz"
        self.services_path = mimic3_path + "SERVICES.csv.gz"

        # MIMIC-III DEMO data paths
        if DEMO == True:
            mimic3_path = paths.mimic3_demo_source_path
            self.admissions_path = mimic3_path + "ADMISSIONS.csv"
            self.chartevents_path = mimic3_path + "CHARTEVENTS.csv"
            self.d_icd_diagnoses_path = mimic3_path + "D_ICD_DIAGNOSES.csv"
            self.d_icd_procedures_path = mimic3_path + "D_ICD_PROCEDURES.csv"
            self.d_items_path = mimic3_path + "D_ITEMS.csv"
            self.d_labitems_path = mimic3_path + "D_LABITEMS.csv"
            self.diagnoses_icd_path = mimic3_path + "DIAGNOSES_ICD.csv"
            self.icustays_path = mimic3_path + "ICUSTAYS.csv"
            self.inputevents_cv_path = mimic3_path + "INPUTEVENTS_CV.csv"
            self.inputevents_mv_path = mimic3_path + "INPUTEVENTS_MV.csv"
            self.labevents_path = mimic3_path + "LABEVENTS.csv"
            self.outputevents_path = mimic3_path + "OUTPUTEVENTS.csv"
            self.patients_path = mimic3_path + "PATIENTS.csv"
            self.prescriptions_path = mimic3_path + "PRESCRIPTIONS.csv"
            self.procedureevents_mv_path = (
                mimic3_path + "PROCEDUREEVENTS_MV.csv"
            )
            self.procedures_icd_path = mimic3_path + "PROCEDURES_ICD.csv"
            self.services_path = mimic3_path + "SERVICES.csv"

        # MIMIC-III custom mapping paths
        self.mimic3_mapping_path = self.mapping_path + "mimic3/"
        self.vitals_mapping_path = (
            self.mimic3_mapping_path + "mimic3_chartevents.yaml"
        )
        self.labs_mapping_path = (
            self.mimic3_mapping_path + "mimic3_labevents.yaml"
        )
        self.outputs_mapping_path = (
            self.mimic3_mapping_path + "mimic3_outputevents.yaml"
        )
        # self.medication_mapping_path = self.mimic3_mapping_path + "mimic3_medication.yaml"


# endregion


# region MIMIC-IV
class MIMIC4Paths(GlobalVars):
    def __init__(self, paths, DEMO=False):
        super().__init__(paths, DEMO)
        mimic4_path = paths.mimic4_source_path

        # MIMIC-IV raw data paths
        self.admissions_path = mimic4_path + "hosp/admissions.csv.gz"
        self.chartevents_path = mimic4_path + "icu/chartevents.csv.gz"
        self.d_icd_diagnoses_path = mimic4_path + "hosp/d_icd_diagnoses.csv.gz"
        self.d_icd_procedures_path = (
            mimic4_path + "hosp/d_icd_procedures.csv.gz"
        )
        self.d_items_path = mimic4_path + "icu/d_items.csv.gz"
        self.d_labitems_path = mimic4_path + "hosp/d_labitems.csv.gz"
        self.diagnoses_icd_path = mimic4_path + "hosp/diagnoses_icd.csv.gz"
        self.icustays_path = mimic4_path + "icu/icustays.csv.gz"
        self.inputevents_path = mimic4_path + "icu/inputevents.csv.gz"
        self.labevents_path = mimic4_path + "hosp/labevents.csv.gz"
        self.outputevents_path = mimic4_path + "icu/outputevents.csv.gz"
        self.patients_path = mimic4_path + "hosp/patients.csv.gz"
        self.prescriptions_path = mimic4_path + "hosp/prescriptions.csv.gz"
        self.procedureevents_path = mimic4_path + "icu/procedureevents.csv.gz"
        self.procedures_icd_path = mimic4_path + "hosp/procedures_icd.csv.gz"
        self.services_path = mimic4_path + "hosp/services.csv.gz"

        # MIMIC-IV DEMO data paths
        if DEMO == True:
            mimic4_path = paths.mimic4_demo_source_path
            self.admissions_path = mimic4_path + "hosp/admissions.csv"
            self.chartevents_path = mimic4_path + "icu/chartevents.csv"
            self.d_icd_diagnoses_path = mimic4_path + "hosp/d_icd_diagnoses.csv"
            self.d_icd_procedures_path = (
                mimic4_path + "hosp/d_icd_procedures.csv"
            )
            self.d_items_path = mimic4_path + "icu/d_items.csv"
            self.d_labitems_path = mimic4_path + "hosp/d_labitems.csv"
            self.diagnoses_icd_path = mimic4_path + "hosp/diagnoses_icd.csv"
            self.icustays_path = mimic4_path + "icu/icustays.csv"
            self.inputevents_path = mimic4_path + "icu/inputevents.csv"
            self.labevents_path = mimic4_path + "hosp/labevents.csv"
            self.outputevents_path = mimic4_path + "icu/outputevents.csv"
            self.patients_path = mimic4_path + "hosp/patients.csv"
            self.prescriptions_path = mimic4_path + "hosp/prescriptions.csv"
            self.procedureevents_path = mimic4_path + "icu/procedureevents.csv"
            self.procedures_icd_path = mimic4_path + "hosp/procedures_icd.csv"
            self.services_path = mimic4_path + "hosp/services.csv"

        # MIMIC-IV custom mapping paths
        self.mimic4_mapping_path = self.mapping_path + "mimic4/"
        self.vitals_mapping_path = (
            self.mimic4_mapping_path + "mimic4_chartevents.yaml"
        )
        self.labs_mapping_path = (
            self.mimic4_mapping_path + "mimic4_labevents.yaml"
        )
        self.outputs_mapping_path = (
            self.mimic4_mapping_path + "mimic4_outputevents.yaml"
        )


# endregion


# region SICdb
class SICdbPaths(GlobalVars):
    def __init__(self, paths):
        super().__init__(paths)
        sicdb_path = paths.sicdb_source_path

        # SICdb raw data paths
        self.cases_path = sicdb_path + "cases.csv.gz"
        self.d_references_path = sicdb_path + "d_references.csv.gz"
        self.data_float_h_path = sicdb_path + "data_float_h.csv.gz"
        self.data_range_path = sicdb_path + "data_range.csv.gz"
        self.data_ref_path = sicdb_path + "data_ref.csv.gz"
        self.laboratory_path = sicdb_path + "laboratory.csv.gz"
        self.medication_path = sicdb_path + "medication.csv.gz"
        self.unitlog_path = sicdb_path + "unitlog.csv.gz"

        # SICdb custom mapping paths
        self.sicdb_mapping_path = self.mapping_path + "sicdb/"
        self.laboratory_mapping_path = (
            self.sicdb_mapping_path + "sicdb_Laboratory.yaml"
        )
        self.timeseries_mapping_path = (
            self.sicdb_mapping_path + "sicdb_Timeseries.yaml"
        )


# region UMCdb
class UMCdbPaths(GlobalVars):
    def __init__(self, paths):
        super().__init__(paths)
        umcdb_path = paths.umcdb_source_path

        # UMCdb raw data paths
        self.admissions_path = umcdb_path + "admissions.csv"
        self.drugitems_path = umcdb_path + "drugitems.csv"
        self.freetextitems_path = umcdb_path + "freetextitems.csv"
        self.listitems_path = umcdb_path + "listitems.csv"
        self.numericitems_path = umcdb_path + "numericitems.csv.gz"
        self.procedureorderitems_path = umcdb_path + "procedureorderitems.csv"
        self.processitems_path = umcdb_path + "processitems.csv"

        # UMCdb custom mapping paths
        self.umcdb_mapping_path = self.mapping_path + "umcdb/"
        self.numeric_mapping_path = (
            self.umcdb_mapping_path + "umcdb_numericitems.yaml"
        )
        self.listitems_mapping_path = (
            self.umcdb_mapping_path + "umcdb_listitems.yaml"
        )


# endregion
