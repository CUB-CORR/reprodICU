# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy import of the MIMIC-III data paths and the preloading of the data.
# The paths are stored in a class attribute and can be accessed via the get_paths() method.

import polars as pl

from helpers.helper import GlobalVars


class MIMIC3Paths(GlobalVars):
    def __init__(self, mimic3_path, DEMO=False):
        super().__init__()

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
        self.procedureevents_mv_path = mimic3_path + "PROCEDUREEVENTS_MV.csv.gz"
        self.procedures_icd_path = mimic3_path + "PROCEDURES_ICD.csv.gz"

        # MIMIC-III DEMO data paths
        if DEMO == True:
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
            self.procedureevents_mv_path = mimic3_path + "PROCEDUREEVENTS_MV.csv"
            self.procedures_icd_path = mimic3_path + "PROCEDURES_ICD.csv"

        # MIMIC-III custom mapping paths
        self.mimic3_mapping_path = self.mapping_path + "mimic3/"
        self.vitals_mapping_path = self.mimic3_mapping_path + "mimic3_chartevents.yaml"
        self.labs_mapping_path = self.mimic3_mapping_path + "mimic3_labevents.yaml"
        self.outputs_mapping_path = self.mimic3_mapping_path + "mimic3_outputevents.yaml"
        # self.medication_mapping_path = self.mimic3_mapping_path + "mimic3_medication.yaml"

    def get_paths(self):
        return self.__dict__.values()

    def preload_all_data(self):
        return [pl.scan_csv(path) for path in self.get_paths()]
