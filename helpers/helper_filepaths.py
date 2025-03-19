# Author: Finn Fassbender
# Last modified: 2024-09-05

# Enables the easy import of the data paths.

import os

import polars as pl
from helpers.helper import GlobalVars


# region OMOP
class OMOPPaths(GlobalVars):
    def __init__(self, paths):
        super().__init__(paths)
        omop_path = paths.OMOP_vocab_path

        # OMOP raw data paths
        self.CONCEPT_ANCESTOR_path = omop_path + "CONCEPT_ANCESTOR.csv"
        self.CONCEPT_CLASS_path = omop_path + "CONCEPT_CLASS.csv"
        self.CONCEPT_RELATIONSHIP_path = omop_path + "CONCEPT_RELATIONSHIP.csv"
        self.CONCEPT_SYNONYM_path = omop_path + "CONCEPT_SYNONYM.csv"
        self.CONCEPT_path = omop_path + "CONCEPT.csv"
        self.DOMAIN_path = omop_path + "DOMAIN.csv"
        self.DRUG_STRENGTH_path = omop_path + "DRUG_STRENGTH.csv"
        self.RELATIONSHIP_path = omop_path + "RELATIONSHIP.csv"
        self.VOCABULARY_path = omop_path + "VOCABULARY.csv"

        # PARQUETIZE FOR MORE EFFICIENT DATA PROCESSING
        for path in [
            self.CONCEPT_ANCESTOR_path,
            self.CONCEPT_CLASS_path,
            self.CONCEPT_RELATIONSHIP_path,
            self.CONCEPT_SYNONYM_path,
            self.CONCEPT_path,
            self.DOMAIN_path,
            self.DRUG_STRENGTH_path,
            self.RELATIONSHIP_path,
            self.VOCABULARY_path,
        ]:
            if not os.path.isfile(
                path.replace(".csv", ".parquet").replace(".gz", "")
            ):
                _parquetize_OMOP(path)

            self.CONCEPT_ANCESTOR_path = omop_path + "CONCEPT_ANCESTOR.parquet"
            self.CONCEPT_CLASS_path = omop_path + "CONCEPT_CLASS.parquet"
            self.CONCEPT_RELATIONSHIP_path = (
                omop_path + "CONCEPT_RELATIONSHIP.parquet"
            )
            self.CONCEPT_SYNONYM_path = omop_path + "CONCEPT_SYNONYM.parquet"
            self.CONCEPT_path = omop_path + "CONCEPT.parquet"
            self.DOMAIN_path = omop_path + "DOMAIN.parquet"
            self.DRUG_STRENGTH_path = omop_path + "DRUG_STRENGTH.parquet"
            self.RELATIONSHIP_path = omop_path + "RELATIONSHIP.parquet"
            self.VOCABULARY_path = omop_path + "VOCABULARY.parquet"


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
        if DEMO:
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
        self.careprovider_mapping_path = (
            self.eICU_mapping_path + "eicu_carePlanCareProvider_specialty.yaml"
        )
        self.intakeoutput_mapping_path = (
            self.eICU_mapping_path + "eicu_intakeOutput_cellpath.yaml"
        )
        self.lab_mapping_path = self.eICU_mapping_path + "eicu_lab.yaml"
        self.drug_administration_route_mapping_path = (
            self.eICU_mapping_path + "eicu_medication_routeadmin.yaml"
        )
        self.micro_culturesite_mapping_path = (
            self.eICU_mapping_path + "eicu_microLab_culturesite.yaml"
        )
        self.micro_organism_mapping_path = (
            self.eICU_mapping_path + "eicu_microLab_organism.yaml"
        )
        self.nurse_oxygen_delivery_device_mapping_path = (
            self.eICU_mapping_path
            + "eicu_nurseCharting_oxygenDeliveryDevices.yaml"
        )
        self.nurse_mapping_path = (
            self.eICU_mapping_path + "eicu_nurseCharting.yaml"
        )
        self.resp_airwaytype_mapping_path = (
            self.eICU_mapping_path + "eicu_respiratoryCare_airwaytype.yaml"
        )
        self.resp_oxygen_delivery_device_mapping_path = (
            self.eICU_mapping_path
            + "eicu_respiratoryCharting_oxygenDeliveryDevices.yaml"
        )
        self.resp_mapping_path = (
            self.eICU_mapping_path + "eicu_respiratoryCharting.yaml"
        )
        self.periodic_mapping_path = (
            self.eICU_mapping_path + "eicu_vitalPeriodic.yaml"
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
        self.specialty_mapping_path = (
            self.hirid_mapping_path + "hirid_OBSERVATION_APACHE_specialty.yaml"
        )
        self.apache_mapping_path = (
            self.hirid_mapping_path + "hirid_OBSERVATION_APACHE.yaml"
        )
        self.drug_administration_route_mapping_path = (
            self.hirid_mapping_path
            + "hirid_PHARMA_drug_administration_route.yaml"
        )
        self.drug_class_mapping_path = (
            self.hirid_mapping_path + "hirid_PHARMA_drug_class.yaml"
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
        self.datetimeevents_path = mimic3_path + "DATETIMEEVENTS.csv.gz"
        self.diagnoses_icd_path = mimic3_path + "DIAGNOSES_ICD.csv.gz"
        self.icustays_path = mimic3_path + "ICUSTAYS.csv.gz"
        self.inputevents_cv_path = mimic3_path + "INPUTEVENTS_CV.csv.gz"
        self.inputevents_mv_path = mimic3_path + "INPUTEVENTS_MV.csv.gz"
        self.labevents_path = mimic3_path + "LABEVENTS.csv.gz"
        self.microbiologyevents_path = mimic3_path + "MICROBIOLOGYEVENTS.csv.gz"
        self.outputevents_path = mimic3_path + "OUTPUTEVENTS.csv.gz"
        self.patients_path = mimic3_path + "PATIENTS.csv.gz"
        self.prescriptions_path = mimic3_path + "PRESCRIPTIONS.csv.gz"
        self.procedureevents_mv_path = mimic3_path + "PROCEDUREEVENTS_MV.csv.gz"
        self.procedures_icd_path = mimic3_path + "PROCEDURES_ICD.csv.gz"
        self.services_path = mimic3_path + "SERVICES.csv.gz"

        # MIMIC-III DEMO data paths
        if DEMO:
            mimic3_path = paths.mimic3_demo_source_path
            self.admissions_path = mimic3_path + "ADMISSIONS.csv"
            self.chartevents_path = mimic3_path + "CHARTEVENTS.csv"
            self.d_icd_diagnoses_path = mimic3_path + "D_ICD_DIAGNOSES.csv"
            self.d_icd_procedures_path = mimic3_path + "D_ICD_PROCEDURES.csv"
            self.d_items_path = mimic3_path + "D_ITEMS.csv"
            self.d_labitems_path = mimic3_path + "D_LABITEMS.csv"
            self.datetimeevents_path = mimic3_path + "DATETIMEEVENTS.csv"
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
        self.inputs_mapping_path = (
            self.mimic3_mapping_path + "mimic3_inputevents.yaml"
        )
        # self.medication_mapping_path = self.mimic3_mapping_path + "mimic3_medication.yaml"
        self.drug_administration_route_mapping_path = (
            self.mimic3_mapping_path
            + "mimic3_inputevents_drug_administration_route.yaml"
        )
        self.drug_class_mapping_path = (
            self.mimic3_mapping_path + "mimic3_inputevents_drug_class.yaml"
        )

        # MIMIC-III OMOP mapping paths
        # https://github.com/MIT-LCP/mimic-omop
        self.mimic3_omop_mapping_path = self.mimic3_mapping_path + "mimic-omop/"
        self.care_site_path = self.mimic3_omop_mapping_path + "care_site.csv"
        self.admission_location_to_concept_path = (
            self.mimic3_omop_mapping_path + "admission_location_to_concept.csv"
        )
        self.admission_type_to_concept_path = (
            self.mimic3_omop_mapping_path + "admission_type_to_concept.csv"
        )
        self.admissions_diagnosis_to_concept_path = (
            self.mimic3_omop_mapping_path
            + "admissions_diagnosis_to_concept.csv"
        )
        self.atb_to_concept_path = (
            self.mimic3_omop_mapping_path + "atb_to_concept.csv"
        )
        self.chart_label_to_concept_path = (
            self.mimic3_omop_mapping_path + "chart_label_to_concept.csv"
        )
        self.chart_observation_to_concept_path = (
            self.mimic3_omop_mapping_path + "chart_observation_to_concept.csv"
        )
        self.continuous_unit_carevue_path = (
            self.mimic3_omop_mapping_path + "continuous_unit_carevue.csv"
        )
        self.cpt4_to_concept_path = (
            self.mimic3_omop_mapping_path + "cpt4_to_concept.csv"
        )
        self.cv_input_label_to_concept_path = (
            self.mimic3_omop_mapping_path + "cv_input_label_to_concept.csv"
        )
        self.datetimeevents_to_concept_path = (
            self.mimic3_omop_mapping_path + "datetimeevents_to_concept.csv"
        )
        self.derived_to_concept_path = (
            self.mimic3_omop_mapping_path + "derived_to_concept.csv"
        )
        self.discharge_location_to_concept_path = (
            self.mimic3_omop_mapping_path + "discharge_location_to_concept.csv"
        )
        self.drgcode_to_concept_path = (
            self.mimic3_omop_mapping_path + "drgcode_to_concept.csv"
        )
        self.ethnicity_to_concept_path = (
            self.mimic3_omop_mapping_path + "ethnicity_to_concept.csv"
        )
        self.heart_rhythm_to_concept_path = (
            self.mimic3_omop_mapping_path + "heart_rhythm_to_concept.csv"
        )
        self.inputevents_drug_to_concept_path = (
            self.mimic3_omop_mapping_path + "inputevents_drug_to_concept.csv"
        )
        self.insurance_to_concept_path = (
            self.mimic3_omop_mapping_path + "insurance_to_concept.csv"
        )
        self.lab_label_to_concept_path = (
            self.mimic3_omop_mapping_path + "lab_label_to_concept.csv"
        )
        self.lab_unit_to_concept_path = (
            self.mimic3_omop_mapping_path + "lab_unit_to_concept.csv"
        )
        self.lab_value_to_concept_path = (
            self.mimic3_omop_mapping_path + "lab_value_to_concept.csv"
        )
        self.labs_from_chartevents_to_concept_path = (
            self.mimic3_omop_mapping_path
            + "labs_from_chartevents_to_concept.csv"
        )
        self.labs_specimen_to_concept_path = (
            self.mimic3_omop_mapping_path + "labs_specimen_to_concept.csv"
        )
        self.map_route_to_concept_path = (
            self.mimic3_omop_mapping_path + "map_route_to_concept.csv"
        )
        self.marital_status_to_concept_path = (
            self.mimic3_omop_mapping_path + "marital_status_to_concept.csv"
        )
        self.microbiology_specimen_to_concept_path = (
            self.mimic3_omop_mapping_path
            + "microbiology_specimen_to_concept.csv"
        )
        self.mv_input_label_to_concept_path = (
            self.mimic3_omop_mapping_path + "mv_input_label_to_concept.csv"
        )
        self.note_category_to_concept_path = (
            self.mimic3_omop_mapping_path + "note_category_to_concept.csv"
        )
        self.note_section_to_concept_path = (
            self.mimic3_omop_mapping_path + "note_section_to_concept.csv"
        )
        self.org_name_to_concept_path = (
            self.mimic3_omop_mapping_path + "org_name_to_concept.csv"
        )
        self.output_label_to_concept_path = (
            self.mimic3_omop_mapping_path + "output_label_to_concept.csv"
        )
        self.prescriptions_ndcisnullzero_to_concept_path = (
            self.mimic3_omop_mapping_path
            + "prescriptions_ndcisnullzero_to_concept.csv"
        )
        self.procedure_to_concept_path = (
            self.mimic3_omop_mapping_path + "procedure_to_concept.csv"
        )
        self.religion_to_concept_path = (
            self.mimic3_omop_mapping_path + "religion_to_concept.csv"
        )
        self.resistance_to_concept_path = (
            self.mimic3_omop_mapping_path + "resistance_to_concept.csv"
        )
        self.route_to_concept_path = (
            self.mimic3_omop_mapping_path + "route_to_concept.csv"
        )
        self.seq_num_to_concept_path = (
            self.mimic3_omop_mapping_path + "seq_num_to_concept.csv"
        )
        self.spec_type_to_concept_path = (
            self.mimic3_omop_mapping_path + "spec_type_to_concept.csv"
        )
        self.unit_doseera_concept_id_path = (
            self.mimic3_omop_mapping_path + "unit_doseera_concept_id.csv"
        )

        # MIMIC-III additional OMOP mapping paths
        self.mimic_omop_mapping_additional_path = (
            self.mimic3_mapping_path + "mimic-omop-additional/"
        )
        self.atb_to_concept_additional_path = (
            self.mimic_omop_mapping_additional_path + "atb_to_concept.csv"
        )

        # MIMIC-III LOINC mapping paths
        self.mimic3_loinc_mapping_path = (
            self.mimic3_mapping_path + "mimic-code_mapping/"
        )
        self.d_labitems_to_loinc_path = (
            self.mimic3_loinc_mapping_path + "d_labitems_to_loinc_mimic3.csv"
        )

        # MIMIC-IV additional LOINC mapping paths
        self.mimic_loinc_mapping_additional_path = (
            self.mimic3_mapping_path + "mimic-additional_code_mapping/"
        )
        self.meas_chartevents_main_additional_path = (
            self.mimic_loinc_mapping_additional_path
            + "meas_chartevents_main.csv"
        )

        # MIMIC-IV LOINC mapping paths (additionally used for MIMIC-III)
        self.mimic4_mapping_path = self.mapping_path + "mimic4/"
        self.mimic4_loinc_mapping_path = (
            self.mimic4_mapping_path + "mimic-code_mapping/"
        )
        self.inputevents_to_rxnorm_path = (
            self.mimic4_loinc_mapping_path + "inputevents_to_rxnorm.csv"
        )
        self.lab_itemid_to_loinc_path = (
            self.mimic4_loinc_mapping_path + "lab_itemid_to_loinc.csv"
        )
        self.meas_chartevents_main_path = (
            self.mimic4_loinc_mapping_path + "meas_chartevents_main.csv"
        )
        self.waveforms_summary_path = (
            self.mimic4_loinc_mapping_path + "waveforms-summary.csv"
        )
        self.proc_itemid_path = (
            self.mimic4_loinc_mapping_path + "proc_itemid.csv"
        )
        self.outputevents_to_loinc_path = (
            self.mimic4_loinc_mapping_path + "outputevents_to_loinc.csv"
        )
        self.proc_datetimeevents_path = (
            self.mimic4_loinc_mapping_path + "proc_datetimeevents.csv"
        )
        self.meas_chartevents_value_path = (
            self.mimic4_loinc_mapping_path + "meas_chartevents_value.csv"
        )
        self.numerics_summary_path = (
            self.mimic4_loinc_mapping_path + "numerics-summary.csv"
        )


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
        self.datetimeevents_path = mimic4_path + "icu/datetimeevents.csv.gz"
        self.diagnoses_icd_path = mimic4_path + "hosp/diagnoses_icd.csv.gz"
        self.icustays_path = mimic4_path + "icu/icustays.csv.gz"
        self.inputevents_path = mimic4_path + "icu/inputevents.csv.gz"
        self.labevents_path = mimic4_path + "hosp/labevents.csv.gz"
        self.microbiologyevents_path = (
            mimic4_path + "hosp/microbiologyevents.csv.gz"
        )
        self.outputevents_path = mimic4_path + "icu/outputevents.csv.gz"
        self.patients_path = mimic4_path + "hosp/patients.csv.gz"
        self.prescriptions_path = mimic4_path + "hosp/prescriptions.csv.gz"
        self.procedureevents_path = mimic4_path + "icu/procedureevents.csv.gz"
        self.procedures_icd_path = mimic4_path + "hosp/procedures_icd.csv.gz"
        self.services_path = mimic4_path + "hosp/services.csv.gz"

        # MIMIC-IV DEMO data paths
        if DEMO:
            mimic4_path = paths.mimic4_demo_source_path
            self.admissions_path = mimic4_path + "hosp/admissions.csv"
            self.chartevents_path = mimic4_path + "icu/chartevents.csv"
            self.d_icd_diagnoses_path = mimic4_path + "hosp/d_icd_diagnoses.csv"
            self.d_icd_procedures_path = (
                mimic4_path + "hosp/d_icd_procedures.csv"
            )
            self.d_items_path = mimic4_path + "icu/d_items.csv"
            self.d_labitems_path = mimic4_path + "hosp/d_labitems.csv"
            self.datetimeevents_path = mimic4_path + "icu/datetimeevents.csv"
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
        self.inputs_mapping_path = (
            self.mimic4_mapping_path + "mimic4_inputevents.yaml"
        )
        self.drug_administration_route_mapping_path = (
            self.mimic4_mapping_path
            + "mimic4_inputevents_drug_administration_route.yaml"
        )
        self.drug_class_mapping_path = (
            self.mimic4_mapping_path + "mimic4_inputevents_drug_class.yaml"
        )

        # MIMIC-IV OMOP mapping paths
        # https://github.com/MIT-LCP/mimic-iv-demo-omop
        self.mimic4_omop_mapping_path = (
            self.mimic4_mapping_path + "mimic-iv-omop/"
        )
        self.vis_admission_path = (
            self.mimic4_omop_mapping_path + "gcpt_vis_admission.csv"
        )
        self.cs_place_of_service_path = (
            self.mimic4_omop_mapping_path + "gcpt_cs_place_of_service.csv"
        )
        self.drug_ndc_path = self.mimic4_omop_mapping_path + "gcpt_drug_ndc.csv"
        self.drug_route_path = (
            self.mimic4_omop_mapping_path + "gcpt_drug_route.csv"
        )
        self.meas_chartevents_main_mod_path = (
            self.mimic4_omop_mapping_path + "gcpt_meas_chartevents_main_mod.csv"
        )
        self.meas_chartevents_value_path = (
            self.mimic4_omop_mapping_path + "gcpt_meas_chartevents_value.csv"
        )
        self.meas_lab_loinc_mod_path = (
            self.mimic4_omop_mapping_path + "gcpt_meas_lab_loinc_mod.csv"
        )
        self.meas_unit_path = (
            self.mimic4_omop_mapping_path + "gcpt_meas_unit.csv"
        )
        self.meas_waveforms_path = (
            self.mimic4_omop_mapping_path + "gcpt_meas_waveforms.csv"
        )
        self.micro_antibiotic_path = (
            self.mimic4_omop_mapping_path + "gcpt_micro_antibiotic.csv"
        )
        self.micro_microtest_path = (
            self.mimic4_omop_mapping_path + "gcpt_micro_microtest.csv"
        )
        self.micro_organism_path = (
            self.mimic4_omop_mapping_path + "gcpt_micro_organism.csv"
        )
        self.micro_resistance_path = (
            self.mimic4_omop_mapping_path + "gcpt_micro_resistance.csv"
        )
        self.micro_specimen_path = (
            self.mimic4_omop_mapping_path + "gcpt_micro_specimen.csv"
        )
        self.mimic_generated_path = (
            self.mimic4_omop_mapping_path + "gcpt_mimic_generated.csv"
        )
        self.obs_drgcodes_path = (
            self.mimic4_omop_mapping_path + "gcpt_obs_drgcodes.csv"
        )
        self.obs_insurance_path = (
            self.mimic4_omop_mapping_path + "gcpt_obs_insurance.csv"
        )
        self.obs_marital_path = (
            self.mimic4_omop_mapping_path + "gcpt_obs_marital.csv"
        )
        self.per_ethnicity_path = (
            self.mimic4_omop_mapping_path + "gcpt_per_ethnicity.csv"
        )
        self.proc_datetimeevents_path = (
            self.mimic4_omop_mapping_path + "gcpt_proc_datetimeevents.csv"
        )
        self.proc_itemid_path = (
            self.mimic4_omop_mapping_path + "gcpt_proc_itemid.csv"
        )

        # MIMIC-IV LOINC mapping paths
        self.mimic4_loinc_mapping_path = (
            self.mimic4_mapping_path + "mimic-code_mapping/"
        )
        self.d_labitems_to_loinc_path = (
            self.mimic4_loinc_mapping_path + "d_labitems_to_loinc.csv"
        )
        self.inputevents_to_rxnorm_path = (
            self.mimic4_loinc_mapping_path + "inputevents_to_rxnorm.csv"
        )
        self.lab_itemid_to_loinc_path = (
            self.mimic4_loinc_mapping_path + "lab_itemid_to_loinc.csv"
        )
        self.meas_chartevents_main_path = (
            self.mimic4_loinc_mapping_path + "meas_chartevents_main.csv"
        )
        self.waveforms_summary_path = (
            self.mimic4_loinc_mapping_path + "waveforms-summary.csv"
        )
        self.proc_itemid_path = (
            self.mimic4_loinc_mapping_path + "proc_itemid.csv"
        )
        self.outputevents_to_loinc_path = (
            self.mimic4_loinc_mapping_path + "outputevents_to_loinc.csv"
        )
        self.proc_datetimeevents_path = (
            self.mimic4_loinc_mapping_path + "proc_datetimeevents.csv"
        )
        self.meas_chartevents_value_path = (
            self.mimic4_loinc_mapping_path + "meas_chartevents_value.csv"
        )
        self.numerics_summary_path = (
            self.mimic4_loinc_mapping_path + "numerics-summary.csv"
        )

        # MIMIC-IV additional LOINC mapping paths
        self.mimic_loinc_mapping_additional_path = (
            self.mimic4_mapping_path + "mimic-additional_code_mapping/"
        )
        self.meas_chartevents_main_additional_path = (
            self.mimic_loinc_mapping_additional_path
            + "meas_chartevents_main.csv"
        )


# endregion


# region NWICU
class NWICUPaths(GlobalVars):
    def __init__(self, paths):
        super().__init__(paths)
        nwicu_path = paths.nwicu_source_path

        # NWICU raw data paths
        self.admissions_path = nwicu_path + "nw_hosp/admissions.csv.gz"
        self.chartevents_path = nwicu_path + "nw_icu/chartevents.csv.gz"
        self.d_icd_diagnoses_path = (
            nwicu_path + "nw_hosp/d_icd_diagnoses.csv.gz"
        )
        self.d_items_path = nwicu_path + "nw_icu/d_items.csv.gz"
        self.d_labitems_path = nwicu_path + "nw_hosp/d_labitems.csv.gz"
        self.diagnoses_icd_path = nwicu_path + "nw_hosp/diagnoses_icd.csv.gz"
        self.emar_path = nwicu_path + "nw_hosp/emar.csv.gz"
        self.icustays_path = nwicu_path + "nw_icu/icustays.csv.gz"
        self.labevents_path = nwicu_path + "nw_hosp/labevents.csv.gz"
        self.patients_path = nwicu_path + "nw_hosp/patients.csv.gz"
        self.prescriptions_path = nwicu_path + "nw_hosp/prescriptions.csv.gz"
        self.procedureevents_path = nwicu_path + "nw_icu/procedureevents.csv.gz"

        # NWICU custom mapping paths
        self.nwicu_mapping_path = self.mapping_path + "nwicu/"
        self.vitals_mapping_path = (
            self.nwicu_mapping_path + "nwicu_chartevents.yaml"
        )
        self.drug_administration_route_mapping_path = (
            self.nwicu_mapping_path + "nwicu_prescriptions_route.yaml"
        )
        self.d_labitems_to_loinc_path = (
            self.nwicu_mapping_path + "d_labitems_to_loinc_nwicu.csv"
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

        # SICdb unpacked data paths
        self.data_float_m_path = sicdb_path + "data_float_m.csv.gz"
        # PARQUETIZE FOR MORE EFFICIENT DATA PROCESSING
        path = self.data_float_m_path
        if not os.path.isfile(
            path.replace(".csv", ".parquet").replace(".gz", "")
        ):
            _parquetize(path, "SICdb")

        self.data_float_m_path = sicdb_path + "data_float_m.parquet"

        # SICdb custom mapping paths
        self.sicdb_mapping_path = self.mapping_path + "sicdb/"
        self.device_mapping_path = (
            self.sicdb_mapping_path + "sicdb_Devices.yaml"
        )


# endregion


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

        # PARQUETIZE FOR MORE EFFICIENT DATA PROCESSING
        for path in [
            self.admissions_path,
            self.drugitems_path,
            self.freetextitems_path,
            self.listitems_path,
            self.numericitems_path,
            self.procedureorderitems_path,
            self.processitems_path,
        ]:
            if not os.path.isfile(
                path.replace(".csv", ".parquet").replace(".gz", "")
            ):
                _parquetize(path, "UMCdb")

            self.admissions_path = umcdb_path + "admissions.parquet"
            self.drugitems_path = umcdb_path + "drugitems.parquet"
            self.freetextitems_path = umcdb_path + "freetextitems.parquet"
            self.listitems_path = umcdb_path + "listitems.parquet"
            self.numericitems_path = umcdb_path + "numericitems.parquet"
            self.procedureorderitems_path = (
                umcdb_path + "procedureorderitems.parquet"
            )
            self.processitems_path = umcdb_path + "processitems.parquet"

        # UMCdb custom mapping paths
        self.umcdb_mapping_path = self.mapping_path + "umcdb/"
        self.apache_mapping_path = self.umcdb_mapping_path + "umcdb_APACHE.yaml"
        self.drug_administration_route_mapping_path = (
            self.umcdb_mapping_path
            + "umcdb_drugitems_drug_administration_route.yaml"
        )
        self.drug_class_mapping_path = (
            self.umcdb_mapping_path + "umcdb_drugitems_drug_class.yaml"
        )

        # UMCdb LOINC mapping paths
        self.umcdb_loinc_mapping_path = (
            self.umcdb_mapping_path + "AMSTEL_data_mappings/"
        )
        self.admissions_gender_mapping_path = (
            self.umcdb_loinc_mapping_path + "admissions_gender.usagi.csv"
        )
        self.admissions_origin_mapping_path = (
            self.umcdb_loinc_mapping_path + "admissions_origin.usagi.csv"
        )
        self.admissions_specialty_mapping_path = (
            self.umcdb_loinc_mapping_path + "admissions_specialty.usagi.csv"
        )
        self.drugitems_administeredunit_mapping_path = (
            self.umcdb_loinc_mapping_path
            + "drugitems_administeredunit.usagi.csv"
        )
        self.drugitems_item_mapping_path = (
            self.umcdb_loinc_mapping_path + "drugitems_item.usagi.csv"
        )
        self.drugitems_ordercategory_mapping_path = (
            self.umcdb_loinc_mapping_path + "drugitems_ordercategory.usagi.csv"
        )
        self.freetextitems_item_mapping_path = (
            self.umcdb_loinc_mapping_path + "freetextitems_item.usagi.csv"
        )
        self.freetextitems_value_mapping_path = (
            self.umcdb_loinc_mapping_path + "freetextitems_value.usagi.csv"
        )
        self.listitems_item_mapping_path = (
            self.umcdb_loinc_mapping_path + "listitems_item.usagi.csv"
        )
        self.listitems_value_mapping_path = (
            self.umcdb_loinc_mapping_path + "listitems_value.usagi.csv"
        )
        self.numericitems_lab_mapping_path = (
            self.umcdb_loinc_mapping_path + "numericitems_lab.usagi.csv"
        )
        self.numericitems_other_mapping_path = (
            self.umcdb_loinc_mapping_path + "numericitems_other.usagi.csv"
        )
        self.numericitems_tag_mapping_path = (
            self.umcdb_loinc_mapping_path + "numericitems_tag.usagi.csv"
        )
        self.numericitems_unit_mapping_path = (
            self.umcdb_loinc_mapping_path + "numericitems_unit.usagi.csv"
        )
        self.procedureorderitems_item_mapping_path = (
            self.umcdb_loinc_mapping_path + "procedureorderitems_item.usagi.csv"
        )
        self.processitems_item_mapping_path = (
            self.umcdb_loinc_mapping_path + "processitems_item.usagi.csv"
        )
        self.providers_mapping_path = (
            self.umcdb_loinc_mapping_path + "providers.usagi.csv"
        )
        self.reason_for_admission_mapping_path = (
            self.umcdb_loinc_mapping_path + "reason_for_admission.usagi.csv"
        )
        self.source_to_concept_map_mapping_path = (
            self.umcdb_loinc_mapping_path + "source_to_concept_map.csv"
        )
        self.source_to_value_map_mapping_path = (
            self.umcdb_loinc_mapping_path + "source_to_value_map.csv"
        )
        self.specimen_source_mapping_path = (
            self.umcdb_loinc_mapping_path + "specimen_source.usagi.csv"
        )


def _parquetize(path, db: str):
    print(f"{db}   - parquetizing {path}")
    pl.scan_csv(
        path,
        schema_overrides={"value": str},
    ).sink_parquet(path.replace(".csv", ".parquet").replace(".gz", ""))


def _parquetize_OMOP(path):
    print(f"OMOP   - parquetizing {path}")
    pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
        quote_char=None,
    ).sink_parquet(path.replace(".csv", ".parquet").replace(".gz", ""))


# endregion
