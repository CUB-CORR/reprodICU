# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy import of the MIMIC-IV data paths and the preloading of the data.
# The paths are stored in a class attribute and can be accessed via the get_paths() method.

import polars as pl

from helpers.helper import GlobalVars


class MIMIC4Paths(GlobalVars):
    def __init__(self, mimic4_path, DEMO=False):
        super().__init__()

        # MIMIC-IV raw data paths
        self.admissions_path = mimic4_path + "hosp/admissions.csv.gz"
        self.chartevents_path = mimic4_path + "icu/chartevents.csv.gz"
        self.d_icd_diagnoses_path = mimic4_path + "hosp/d_icd_diagnoses.csv.gz"
        self.d_icd_procedures_path = mimic4_path + "hosp/d_icd_procedures.csv.gz"
        self.d_items_path = mimic4_path + "icu/d_items.csv.gz"
        self.d_labitems_path = mimic4_path + "hosp/d_labitems.csv.gz"
        self.diagnoses_icd_path = mimic4_path + "hosp/diagnoses_icd.csv.gz"
        self.icustays_path = mimic4_path + "icu/icustays.csv.gz"
        self.inputevents_path = mimic4_path + "icu/inputevents.csv.gz"
        self.labevents_path = mimic4_path + "hosp/labevents.csv.gz"
        self.outputevents_path = mimic4_path + "icu/outputevents.csv.gz"
        self.patients_path = mimic4_path + "hosp/patients.csv.gz"
        self.procedureevents_path = mimic4_path + "icu/procedureevents.csv.gz"
        self.procedures_icd_path = mimic4_path + "hosp/procedures_icd.csv.gz"

        # MIMIC-IV DEMO data paths
        if DEMO == True:
            self.admissions_path = mimic4_path + "hosp/admissions.csv"
            self.chartevents_path = mimic4_path + "icu/chartevents.csv"
            self.d_icd_diagnoses_path = mimic4_path + "hosp/d_icd_diagnoses.csv"
            self.d_icd_procedures_path = mimic4_path + "hosp/d_icd_procedures.csv"
            self.d_items_path = mimic4_path + "icu/d_items.csv"
            self.d_labitems_path = mimic4_path + "hosp/d_labitems.csv"
            self.diagnoses_icd_path = mimic4_path + "hosp/diagnoses_icd.csv"
            self.icustays_path = mimic4_path + "icu/icustays.csv"
            self.inputevents_path = mimic4_path + "icu/inputevents.csv"
            self.labevents_path = mimic4_path + "hosp/labevents.csv"
            self.outputevents_path = mimic4_path + "icu/outputevents.csv"
            self.patients_path = mimic4_path + "hosp/patients.csv"
            self.procedureevents_path = mimic4_path + "icu/procedureevents.csv"
            self.procedures_icd_path = mimic4_path + "hosp/procedures_icd.csv"

        # MIMIC-IV custom mapping paths
        self.mimic4_mapping_path = self.mapping_path + "mimic4/"
        self.vitals_mapping_path = self.mimic4_mapping_path + "mimic4_chartevents.yaml"
        self.labs_mapping_path = self.mimic4_mapping_path + "mimic4_labevents.yaml"
        self.outputs_mapping_path = self.mimic4_mapping_path + "mimic4_outputevents.yaml"

    def get_paths(self):
        return self.__dict__.values()

    def preload_all_data(self):
        return [pl.scan_csv(path) for path in self.get_paths()]
