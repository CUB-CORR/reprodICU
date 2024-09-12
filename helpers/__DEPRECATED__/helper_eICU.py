# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy import of the eICU data paths and the preloading of the data.
# The paths are stored in a class attribute and can be accessed via the get_paths() method.

import polars as pl

from helpers.helper import GlobalVars


class EICUPaths(GlobalVars):
    def __init__(self, eicu_path, DEMO=False):
        super().__init__()

        # eICU raw data paths
        self.admissionDrug_path = eicu_path + "admissionDrug.csv.gz"
        self.admissionDx_path = eicu_path + "admissionDx.csv.gz"
        self.allergy_path = eicu_path + "allergy.csv.gz"
        self.apacheApsVar_path = eicu_path + "apacheApsVar.csv.gz"
        self.apachePatientResult_path = eicu_path + "apachePatientResult.csv.gz"
        self.apachePredVar_path = eicu_path + "apachePredVar.csv.gz"
        self.carePlanCareProvider_path = eicu_path + "carePlanCareProvider.csv.gz"
        self.carePlanEOL_path = eicu_path + "carePlanEOL.csv.gz"
        self.carePlanGeneral_path = eicu_path + "carePlanGeneral.csv.gz"
        self.carePlanGoal_path = eicu_path + "carePlanGoal.csv.gz"
        self.carePlanInfectiousDisease_path = eicu_path + "carePlanInfectiousDisease.csv.gz"
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
            self.admissionDrug_path = eicu_path + "admissionDrug.csv"
            self.admissionDx_path = eicu_path + "admissionDx.csv"
            self.allergy_path = eicu_path + "allergy.csv"
            self.apacheApsVar_path = eicu_path + "apacheApsVar.csv"
            self.apachePatientResult_path = eicu_path + "apachePatientResult.csv"
            self.apachePredVar_path = eicu_path + "apachePredVar.csv"
            self.carePlanCareProvider_path = eicu_path + "carePlanCareProvider.csv"
            self.carePlanEOL_path = eicu_path + "carePlanEOL.csv"
            self.carePlanGeneral_path = eicu_path + "carePlanGeneral.csv"
            self.carePlanGoal_path = eicu_path + "carePlanGoal.csv"
            self.carePlanInfectiousDisease_path = eicu_path + "carePlanInfectiousDisease.csv"
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
            self.respiratoryCharting_path = eicu_path + "respiratoryCharting.csv"
            self.treatment_path = eicu_path + "treatment.csv"
            self.vitalAperiodic_path = eicu_path + "vitalAperiodic.csv"
            self.vitalPeriodic_path = eicu_path + "vitalPeriodic.csv"

        # eICU custom mapping paths
        self.eICU_mapping_path = self.mapping_path + "eicu/"
        self.lab_mapping_path = self.eICU_mapping_path + "eicu_lab.yaml"
        self.resp_mapping_path = self.eICU_mapping_path + "eicu_respiratoryCharting.yaml"
        self.intakeoutput_mapping_path = self.eICU_mapping_path + "eicu_intakeOutput.yaml"
        self.nurse_mapping_path = self.eICU_mapping_path + "eicu_nurseCharting.yaml"
        self.nurse_oxygen_delivery_device_mapping_path = (
            self.eICU_mapping_path + "eicu_nurseCharting_oxygenDeliveryDevices.yaml"
        )
        self.periodic_mapping_path = self.eICU_mapping_path + "eicu_vitalPeriodic.yaml"
        self.medication_mapping_path = self.eICU_mapping_path + "eicu_medication.yaml"

    def get_paths(self):
        return self.__dict__.values()

    def preload_all_data(self):
        return [pl.scan_csv(path) for path in self.get_paths()]
