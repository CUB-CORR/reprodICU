# Author: Finn Fassbender
# Last modified: 2024-09-10

# Description: This script extracts data from HiRID source files and converts them into a structured format for harmonization.

import os.path

import polars as pl
from helpers.helper import GlobalHelpers
from helpers.helper_filepaths import HiRIDPaths
from helpers.helper_OMOP import Vocabulary


class HiRIDExtractor(HiRIDPaths):
    def __init__(self, paths):
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.omop = Vocabulary(paths)
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]

        self.other_lab_values = [
            "Creatinine [Moles/volume]",
            "Glucose [Moles/volume]",
            "Urea [Moles/volume]",
            "Creatine kinase panel - Serum or Plasma",
            "Creatine kinase.MB [Mass/volume]",
            "Lactate [Mass/volume]",
            "Lymphocytes [#/volume]",
        ]

    # region patient
    # Extract patient information from the patient.csv file
    def extract_patient_information(self) -> pl.LazyFrame:
        """
        Extracts and harmonizes patient information from the HiRID source data.

        Steps performed:
            1. Extract admissions data.
               - {icu_stay_id_col}: Unique ICU stay identifier.
               - {gender_col}: Patient gender.
               - {age_col}: Patient age.
               - {mortality_icu_col}: Boolean flag indicating ICU mortality.
               - {mortality_hosp_col}: Boolean flag indicating hospital mortality.
            2. Join length of stay data.
               - {icu_stay_id_col}: Unique ICU identifier.
               - {icu_length_of_stay_col}: ICU length of stay in days (computed from timeseries).
            3. Join patient height and weight data.
               - {icu_stay_id_col}: Unique ICU identifier.
               - {weight_col}: Patient weight.
               - {height_col}: Patient height.
            4. Join admission diagnoses data.
               - {icu_stay_id_col}: Unique ICU identifier.
               - {admission_diagnosis_col}: Mapped admission diagnosis.
            5. Assign derived constant information.
               - {care_site_col}: Patient’s care site.
               - {unit_type_col}: ICU unit type.
               - {admission_time_col}: Time component extracted from "admissiontime".

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                {icu_stay_id_col}, {care_site_col}, {unit_type_col}, {admission_time_col},
                and additional columns from the join operations.
        """
        return (
            self._extract_admissions()
            .join(
                self._extract_length_of_stay(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                self._extract_patient_height_weight(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .join(
                self.extract_admit_diagnoses(),
                on=self.icu_stay_id_col,
                how="left",
            )
            .with_columns(
                # Set care site
                pl.lit("Inselspital - Universitätsspital Bern").alias(
                    self.care_site_col
                ),
                # Set unit type
                # NOTE: the Bern University Hospital only has one unit type
                # -> all ICU patients are cared for within a interdisciplinary 60-bed unit in the Department of Intensive Care Medicine
                pl.lit("Intensive care unit")
                .replace(self.UNIT_TYPES_MAP)
                .cast(self.unit_types_dtype)
                .first()
                .alias(self.unit_type_col),
                # Get admission time
                pl.col("admissiontime")
                .dt.time()
                .alias(self.admission_time_col),
            )
        )

    # endregion

    # region admissions
    def _extract_admissions(self) -> pl.LazyFrame:
        """
        Loads and processes admissions data from HiRID.

        Steps performed:
            1. Read CSV data from {general_table_path} with "admissiontime" as a string.
            2. Rename input columns:
               - "patientid" to {icu_stay_id_col}: Unique ICU identifier.
               - "sex" to {gender_col}: Patient gender.
               - "age" to {age_col}: Patient age.
            3. Convert "admissiontime" to a datetime object.
            4. Convert gender values:
               - "M" becomes "Male" and "F" becomes "Female".
            5. Compute Boolean mortality flags:
               - {mortality_icu_col}: True if "discharge_status" is non-empty and equals "dead".
               - {mortality_hosp_col}: True if "discharge_status" equals "dead".
            6. Drop the "discharge_status" column.

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                {icu_stay_id_col}, {gender_col}, {age_col}, {mortality_icu_col},
                {mortality_hosp_col}, and formatted "admissiontime".
        """
        return (
            pl.scan_csv(
                self.general_table_path, schema_overrides={"admissiontime": str}
            )
            # Rename columns for consistency
            .rename(
                {
                    "patientid": self.icu_stay_id_col,
                    "sex": self.gender_col,
                    "age": self.age_col,
                }
            )
            .with_columns(
                pl.col(self.icu_stay_id_col).cast(str),
                # Convert the admission time to datetime
                pl.col("admissiontime").str.to_datetime(
                    "%Y-%m-%d %H:%M:%S%.9f"
                ),
                # Convert the gender to the established format
                pl.col(self.gender_col)
                .replace({"M": "Male", "F": "Female"})
                .cast(self.gender_dtype),
                # Convert the age to int
                pl.col(self.age_col).cast(int),
                # Convert the discharge status to the established format
                pl.when(pl.col("discharge_status") != "")
                .then(pl.col("discharge_status") == "dead")
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_icu_col),
                pl.when(pl.col("discharge_status") == "dead")
                .then(True)
                .otherwise(None)
                .cast(bool)
                .alias(self.mortality_hosp_col),
            )
            .drop("discharge_status")
        )

    # endregion

    # region len of stay
    def _extract_length_of_stay(self) -> pl.LazyFrame:
        """
        Computes the ICU length of stay from timeseries measurements.

        Steps performed:
            1. Check if a precomputed parquet file exists at {precalc_path} ("HiRID_lengths_of_stay.parquet").
            2. If available, load that file; otherwise, read parquet files from {imputed_stage_path}.
            3. Select and rename:
               - "patientid" to {icu_stay_id_col}: Unique ICU identifier.
               - "reldatetime" to {icu_length_of_stay_col}: Raw length of stay in seconds.
            4. Group data by {icu_stay_id_col} and calculate the maximum datetime.
            5. Convert the duration from seconds to days.
            6. Save the computed DataFrame as a parquet file.

        Returns:
            pl.LazyFrame: DataFrame with columns:
                {icu_stay_id_col} (Unique ICU identifier),
                {icu_length_of_stay_col} (ICU length of stay in days).
        """
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_lengths_of_stay.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "HiRID_lengths_of_stay.parquet"
            )

        print("HiRID   - Processing patient length of stay data...")

        # The length of stay is derived from the last measurement of a timeseries variable.
        lengths_of_stay = (
            pl.scan_parquet(self.imputed_stage_path + "*.parquet")
            .select("patientid", "reldatetime")
            .drop_nulls()
            .rename(
                {
                    "patientid": self.icu_stay_id_col,
                    "reldatetime": self.icu_length_of_stay_col,
                }
            )
            .cast({self.icu_stay_id_col: str})
            .group_by(self.icu_stay_id_col)
            .max()
            # Convert the length of stay to days
            .with_columns(
                pl.duration(seconds=pl.col(self.icu_length_of_stay_col))
                .truediv(pl.duration(days=1))
                .alias(self.icu_length_of_stay_col)
            )
        )

        # Save precalculated data
        lengths_of_stay.sink_parquet(
            self.precalc_path + "HiRID_lengths_of_stay.parquet"
        )

        return lengths_of_stay

    # endregion

    # region h/weight
    def _extract_patient_height_weight(self) -> pl.LazyFrame:
        """
        Extracts patient height and weight from timeseries data within a cutoff time window.

        Steps performed:
            1. Check if a precomputed parquet file exists at {precalc_path}("HiRID_height_weight.parquet").
            2. Read admission times from {general_table_path}.
            3. For each file in {timeseries_path}:
               a. Select columns: "patientid", "datetime", "value", "variableid".
               b. Rename "patientid" to {icu_stay_id_col} and "datetime" to "valuedate".
               c. Convert datatypes and join with admission times.
               d. Convert "admissiontime" and "valuedate" to datetime.
               e. Replace {variableid} with variable names for {weight_col} and {height_col}.
               f. Filter rows within the cutoff {ADMISSION_WEIGHT_HEIGHT_CUTOFF}.
            4. Concatenate records from all files and pivot so that each {icu_stay_id_col} maps to {weight_col} and {height_col}.
            5. Save the result as a parquet file.

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                {icu_stay_id_col} (Unique ICU identifier),
                {weight_col} (Patient weight),
                {height_col} (Patient height).
        """
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_height_weight.parquet"):
            return pl.scan_parquet(
                self.precalc_path + "HiRID_height_weight.parquet"
            )

        print("HiRID   - Processing patient height and weight data...")

        # The height and weight are derived from the last measurement of a timeseries variable.
        variables = {10000400: self.weight_col, 10000450: self.height_col}
        admissiontimes = (
            pl.scan_csv(self.general_table_path)
            .select("patientid", "admissiontime")
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str, "admissiontime": str})
        )

        # Create an empty DataFrame to store the height and weight data
        height_weight = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in os.listdir(self.timeseries_path):
            # Extract the data from the file
            data = (
                pl.scan_parquet(self.timeseries_path + file)
                # Select the relevant columns
                .select("patientid", "datetime", "value", "variableid")
                # Rename the columns for consistency
                .rename(
                    {"patientid": self.icu_stay_id_col, "datetime": "valuedate"}
                )
                .cast(
                    {
                        self.icu_stay_id_col: str,
                        "valuedate": str,
                        "variableid": int,
                    }
                )
                # Drop rows with missing values
                .drop_nulls()
                # Join the data with the admission times
                .join(admissiontimes, on=self.icu_stay_id_col)
                # Convert the admission time and the value date to datetime
                .with_columns(
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("valuedate").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    # Replace the variableid with the corresponding variable name
                    pl.col("variableid").replace(variables, default=None),
                )
                # Filter for variables of interest within the cutoff time
                .filter(
                    (pl.col("valuedate") - pl.col("admissiontime"))
                    < pl.duration(hours=self.ADMISSION_WEIGHT_HEIGHT_CUTOFF),
                    pl.col("variableid").is_in(variables.values()),
                )
                .drop(["admissiontime", "valuedate"])
            )

            # Append the data to the DataFrame
            height_weight = pl.concat(
                [height_weight, data], how="diagonal_relaxed"
            )

        height_weight = (
            height_weight.collect(streaming=True)
            .pivot(
                on="variableid",
                index=self.icu_stay_id_col,
                values="value",
                aggregate_function="max",
            )
            .select(self.icu_stay_id_col, self.weight_col, self.height_col)
        )

        height_weight.write_parquet(
            self.precalc_path + "HiRID_height_weight.parquet"
        )

        return height_weight.lazy()

    # endregion

    # region admitDX
    def extract_admit_diagnoses(self) -> pl.LazyFrame:
        """
        Extracts and maps admission diagnoses from timeseries files using HiRID mappings.

        Steps performed:
            1. If a precomputed parquet file exists at {precalc_path} ("HiRID_admitDX.parquet"), load it.
            2. Otherwise, load the diagnosis mapping from {apache_mapping_path}.
            3. For each file in {timeseries_path}:
               a. Select columns: "patientid", "datetime", "variableid", "value".
               b. Rename "patientid" to {icu_stay_id_col} (Unique ICU identifier).
               c. Filter rows where {variableid} is in [9990002, 9990004].
            4. Concatenate data from all files.
            5. Sort by {icu_stay_id_col} and "datetime", then group to take the first diagnosis occurrence.
            6. Replace diagnosis values using the mapping.
            7. Save the resulting DataFrame as a parquet file.

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                {icu_stay_id_col} (Unique ICU identifier),
                {admission_diagnosis_col} (Mapped admission diagnosis).
        """
        # check if precalculated data is available
        if os.path.isfile(self.precalc_path + "HiRID_admitDX.parquet"):
            return pl.scan_parquet(self.precalc_path + "HiRID_admitDX.parquet")

        print("HiRID   - Extracting admission diagnoses...")

        # Load the mapping of the diagnoses
        hirid_diagnosis_mapping = self.load_mapping(self.apache_mapping_path)
        hirid_specialty_mapping = self.load_mapping(self.specialty_mapping_path)

        # Create an empty DataFrame to store the admission diagnoses data
        admitDX = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the files specifically allows
        # for a more efficient processing of the data.
        for file in os.listdir(self.timeseries_path):
            # Extract the data from the file
            data = (
                pl.scan_parquet(self.timeseries_path + file)
                .select("patientid", "datetime", "variableid", "value")
                .rename({"patientid": self.icu_stay_id_col})
                .filter(pl.col("variableid").is_in([9990002, 9990004]))
            )

            # Append the data to the DataFrame
            admitDX = pl.concat([admitDX, data], how="diagonal_relaxed")

        admitDX = (
            admitDX.sort(self.icu_stay_id_col, "datetime")
            .group_by(self.icu_stay_id_col)
            .agg(pl.col("value").first())
            .with_columns(
                pl.col("value")
                .replace(hirid_diagnosis_mapping, default=None)
                .alias(self.admission_diagnosis_col),
                pl.col("value")
                .replace(hirid_specialty_mapping, default=None)
                .alias(self.specialty_col),
            )
            .select(
                self.icu_stay_id_col,
                self.admission_diagnosis_col,
                self.specialty_col,
            )
            .cast({self.icu_stay_id_col: str})
        )

        # admitDX.sink_parquet(self.precalc_path + "HiRID_admitDX.parquet")
        admitDX.collect(streaming=True).write_parquet(
            self.precalc_path + "HiRID_admitDX.parquet"
        )

        return admitDX

    # region timeseries
    # Extract timeseries information from the timeseries file directory
    def _extract_timeseries_helper(
        self,
        data: pl.LazyFrame,
        admissiontime: pl.LazyFrame,
        length_of_stay: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """
        Processes raw timeseries events and computes a time offset from admission.

        Steps performed:
            1. Select columns "patientid", "datetime", "variableid", and "value".
            2. Rename "patientid" to {icu_stay_id_col} (Unique ICU identifier).
            3. Join with admission time and length of stay DataFrames.
            4. Convert "admissiontime" and "datetime" to datetime objects.
            5. Cast "value" to float.
            6. Compute {timeseries_time_col} as the difference in seconds between the event time and admission.
            7. Drop "admissiontime" and "datetime", remove duplicate rows, and filter out null or empty values.

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                {icu_stay_id_col} (Unique ICU identifier),
                {timeseries_time_col} (Time offset in seconds from admission),
                "variable" (Timeseries variable name),
                "value" (Measurement value).
        """
        return (
            data.select("patientid", "datetime", "variableid", "value")
            # Rename columns for consistency
            .rename({"patientid": self.icu_stay_id_col})
            .cast({self.icu_stay_id_col: str, "datetime": str})
            .join(admissiontime, on=self.icu_stay_id_col)
            .join(length_of_stay, on=self.icu_stay_id_col)
            .join(self._get_observation_variables(), on="variableid")
            .with_columns(
                pl.col("admissiontime").str.to_datetime(
                    "%Y-%m-%d %H:%M:%S%.9f"
                ),
                pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                # .replace_strict(observation_mapping, default=None),
                pl.col("value").cast(float),
            )
            .with_columns(
                (
                    (
                        pl.col("datetime") - pl.col("admissiontime")
                    ).dt.total_seconds()
                ).alias(self.timeseries_time_col)
            )
            .drop("admissiontime", "datetime")
            # Remove duplicate rows
            .unique()
            # Remove rows with empty lab names
            .filter(pl.col("value").is_not_null())
            # Remove rows with empty lab results
            .filter(
                pl.col("variable").is_not_null() & (pl.col("variable") != "")
            )
        )

    # endregion

    # region ts labs
    def _extract_timeseries_labs_helper(
        self, data: pl.LazyFrame
    ) -> pl.LazyFrame:
        """
        Structures and enriches laboratory timeseries data with LOINC details.

        Steps performed:
            1. Extract unique laboratory test names from "variable".
            2. For each lab test name, derive LOINC components:
               - "LOINC_component": Derived lab component.
               - "LOINC_system": System information.
               - "LOINC_method": Measurement method.
               - "LOINC_time": Time aspect (with "Point in time (spot)" filtered out).
               - "LOINC_code": Corresponding concept code.
            3. Join the LOINC details back to the original DataFrame.
            4. Filter rows based on {relevant_lab_LOINC_components} and {relevant_lab_LOINC_systems}.
            5. Create a struct column "labstruct" containing:
               - value: Lab result,
               - system: {LOINC_system},
               - method: {LOINC_method},
               - time: {LOINC_time},
               - LOINC: {LOINC_code}.
            6. Select and return relevant columns.

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                - {icu_stay_id_col} (Unique ICU identifier),
                - {timeseries_time_col} (Time offset from admission),
                - "variable" (Lab test name),
                - "labstruct" (Struct with LOINC details and lab result).
        """
        LOINC_data = data.select("variable").unique()
        labnames = LOINC_data.collect().to_series().to_list()
        LOINC_data = (
            data.select("variable").unique()
            # Add columns for LOINC components and systems
            .with_columns(
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_component_from_name(labnames),
                    default=None,
                )
                .alias("LOINC_component"),
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_system_from_name(labnames), default=None
                )
                .alias("LOINC_system"),
                pl.col("variable")
                .replace_strict(
                    self.omop.get_lab_method_from_name(labnames), default=None
                )
                .alias("LOINC_method"),
                pl.col("variable").replace_strict(
                    self.omop.get_lab_time_aspect_from_name(labnames),
                    default=None,
                )
                # remove "Point in time (spot)" values
                .replace({"Point in time (spot)": None}).alias("LOINC_time"),
                pl.col("variable")
                .replace_strict(
                    self.omop.get_concept_codes_from_names(labnames),
                    default=None,
                )
                .alias("LOINC_code"),
            )
        )

        return (
            data.join(LOINC_data, on="variable")
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
            # MAKE STRUCT
            .with_columns(pl.col("LOINC_component").alias("variable"))
            .with_columns(
                pl.struct(
                    value=pl.col("value"),
                    system=pl.col("LOINC_system"),
                    method=pl.col("LOINC_method"),
                    time=pl.col("LOINC_time"),
                    LOINC=pl.col("LOINC_code"),
                ).alias("labstruct")
            )
            .select(
                self.icu_stay_id_col,
                self.timeseries_time_col,
                "variable",
                "labstruct",
            )
        )

    # region pharma
    def extract_medications(self) -> pl.LazyFrame:
        """
        Extracts and processes medication data from HiRID pharmaceutical files.

        Steps performed:
            1. Load medication mapping and drug route mappings.
            2. Retrieve admission time and length of stay.
            3. For each file in {pharma_path}:
               a. Filter out invalid records.
               b. Select columns:
                  - "patientid" renamed to {icu_stay_id_col}: Unique ICU identifier.
                  - "pharmaid": Converted to {drug_name_col} (Medication name).
                  - "givendose" to {drug_amount_col}: Dosage amount.
                  - "doseunit" to {drug_amount_unit_col}: Unit of the dosage.
                  - "route" to {drug_admin_route_col}: Administration route.
                  - "subtypeid" to {drug_class_col}: Drug classification.
               c. Convert datetime columns and join with admission time.
               d. Compute the time offset for the medication as {drug_end_col}.
               e. Map medication names to ingredients and administration routes.
            4. Postprocess to calculate infusion rates:
               a. Compute {drug_rate_col}: Calculated infusion rate.
               b. Compute {fluid_rate_col}: Fluid infusion rate.
               c. Determine drug start ({drug_start_col}) and end times ({drug_end_col})
                  by comparing consecutive log entries.
            5. Remove helper columns and duplicate rows.

        Returns:
            pl.LazyFrame: DataFrame containing columns:
                - {icu_stay_id_col} (Unique ICU identifier),
                - {drug_mixture_id_col}: (Mixture identifier).
                - {drug_name_col} (Medication name),
                - {drug_ingredient_col} (Mapped medication ingredient),
                - {drug_rate_col} (Calculated infusion rate),
                - {drug_rate_unit_col} (Infusion rate unit),
                - {drug_start_col} (Drug start offset from admission),
                - {drug_end_col} (Drug end offset from admission),
                - along with additional derived columns.
        """
        print("HiRID   - Extracting medications...")

        # Extract medication mappings by building a chain of references
        # 1. Get drug name references from our mapping files
        drug_references = self._extract_drug_references(return_ids=True)
        concept_ids = drug_references.values()

        # 2. Retrieve active ingredients for these concept IDs
        ingredients = self.omop.get_ingredient(concept_ids, return_dict=False)

        # 3. Create a mapping from drug names to their active ingredients
        # Convert drug_references dictionary to DataFrame
        drug_references_df = pl.from_dict(
            {
                "pharmaid": list(drug_references.keys()),
                "drug_concept_id": list(drug_references.values()),
            }
        )

        # Join drug references with ingredients to get all drug-ingredient mappings
        # This preserves one-to-many relationships (one drug to multiple ingredients)
        pharmaid_to_ingredient = (
            drug_references_df.join(
                ingredients, on="drug_concept_id", how="inner"
            )
            .rename({"ingredient_name": self.drug_ingredient_col})
            .select("pharmaid", self.drug_ingredient_col)
            .lazy()
        )
        
        # Load additional mappings
        hirid_drug_class_mapping = self.load_mapping(
            self.drug_administration_route_mapping_path
        )
        hirid_drug_administration_route_mapping = self.load_mapping(
            self.drug_administration_route_mapping_path
        )

        admissiontime = (
            self._extract_admissions()
            .select(self.icu_stay_id_col, "admissiontime")
            .cast({"admissiontime": str})
        )
        length_of_stay = self._extract_length_of_stay()

        ########################################################################
        # OBSERVATION
        # -> saline and colloid infusions are tracked as observations
        ########################################################################
        # Create an empty DataFrame to store the observation data
        observation = pl.LazyFrame()

        # Filter for the relevant observation variables
        # Observation   30005075    Infusion of saline solution	cummulative
        # Observation   30005080    Intravenous fluid colloid administration
        # -> both are cumulative variables and reset at midnight
        # to calculate the infusion rate, we need to calculate the difference
        # between the current and the previous value
        fluid_ids = [
            30005075,  # Infusion of saline solution
            30005080,  # Intravenous fluid colloid administration
        ]

        # Since each case has it's data in only one file, iterating over the
        # files allows for a more efficient processing of the data.
        for file in os.listdir(self.timeseries_path):
            print(f"Processing file {file}...", end="\r")
            data = (
                pl.scan_parquet(self.timeseries_path + file)
                .filter(pl.col("variableid").is_in(fluid_ids))
                # Select the relevant columns
                .select("patientid", "datetime", "variableid", "value")
                # Rename the columns for consistency
                .rename({"patientid": self.icu_stay_id_col})
                .with_columns(
                    pl.col("variableid")
                    .replace_strict(
                        {
                            30005075: "normal saline (0.9%)",
                            30005080: "colloid",
                        },
                        default=None,
                        return_dtype=pl.String,
                    )
                    .alias(self.fluid_group_col)
                )
                .cast({self.icu_stay_id_col: str, "datetime": str})
                # Join the data with the admission times
                .join(admissiontime, on=self.icu_stay_id_col)
                .join(length_of_stay, on=self.icu_stay_id_col)
                .with_columns(
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                    pl.col("value").cast(float),
                    pl.lit("intravenous").alias(self.drug_admin_route_col),
                    pl.lit(True).alias(self.drug_continous_col),
                )
                # Calculate the difference between the current and the previous
                # value, store each timestamp in a separate column
                .with_columns(
                    pl.col("datetime")
                    .shift(1)
                    .over(
                        self.icu_stay_id_col,
                        self.fluid_group_col,
                        order_by="datetime",
                    )
                    .alias("prev_datetime"),
                )
                .with_columns(
                    # Check if dates are different (midnight crossed)
                    pl.when(
                        pl.col("datetime").dt.date()
                        != pl.col("prev_datetime").dt.date()
                    )
                    # If midnight crossed, use current value (assuming reset to 0)
                    .then(pl.col("value"))
                    # Otherwise calculate difference from previous value
                    .otherwise(
                        pl.col("value").sub(
                            pl.col("value")
                            .shift(1)
                            .over(
                                self.icu_stay_id_col,
                                self.fluid_group_col,
                                order_by="datetime",
                            )
                        )
                    ).alias(self.fluid_amount_col),
                )
                # Calculate the rate
                .with_columns(
                    (
                        (
                            pl.col(self.fluid_amount_col)
                            / pl.col("datetime")
                            .sub(pl.col("prev_datetime"))
                            .dt.total_seconds()
                        )
                        .round_sig_figs(2)
                        .alias(self.fluid_rate_col)
                    )
                )
                .with_columns(
                    (
                        (
                            pl.col("prev_datetime") - pl.col("admissiontime")
                        ).dt.total_seconds()
                    ).alias(self.drug_start_col),
                    (
                        (
                            pl.col("datetime") - pl.col("admissiontime")
                        ).dt.total_seconds()
                    ).alias(self.drug_end_col),
                )
                .drop("admissiontime", "datetime")
                # Remove duplicate rows
                .unique()
                # Remove rows with empty values
                .filter(pl.col(self.fluid_amount_col) > 0)
                .select(
                    self.icu_stay_id_col,
                    self.drug_admin_route_col,
                    self.drug_continous_col,
                    self.fluid_group_col,
                    self.fluid_amount_col,
                    self.fluid_rate_col,
                    self.drug_start_col,
                    self.drug_end_col,
                )
            )

            # Append the data to the DataFrame
            observation = pl.concat(
                [observation, data.lazy()], how="diagonal_relaxed"
            )

        ########################################################################
        # PHARMA
        ########################################################################
        # Create an empty DataFrame to store the pharma data
        pharma = pl.LazyFrame()

        # Since each case has it's data in only one file, iterating over the
        # files allows for a more efficient processing of the data.
        for file in os.listdir(self.pharma_path):
            print(f"Processing file {file}...", end="\r")
            data = (
                pl.scan_parquet(self.pharma_path + file)
                # Filter out invalidated records
                .filter((pl.col("recordstatus") & 2) == 0)
                .select(
                    "patientid",
                    "pharmaid",
                    "givenat",
                    "givendose",
                    "doseunit",
                    "route",
                    "infusionid",
                    "subtypeid",
                    "fluidamount_calc",
                )
                # Rename columns for consistency
                .rename(
                    {
                        "patientid": self.icu_stay_id_col,
                        "givendose": self.drug_amount_col,
                        "doseunit": self.drug_amount_unit_col,
                        "route": self.drug_admin_route_col,
                        "infusionid": self.drug_mixture_id_col,
                        "subtypeid": self.drug_class_col,
                        "fluidamount_calc": self.fluid_amount_col,
                    }
                )
                # Cast the datetime to string to avoid the following error:
                # polars.exceptions.SchemaError: invalid series dtype: expected `String`, got `datetime[ns]`
                .cast({self.icu_stay_id_col: str, "givenat": str})
                .join(admissiontime, on=self.icu_stay_id_col)
                .join(length_of_stay, on=self.icu_stay_id_col)
                .with_columns(
                    pl.col("admissiontime").str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f"
                    ),
                    pl.col("givenat").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                    pl.col(self.drug_amount_col).cast(float),
                )
                # Replace the pharmaid with the corresponding medication name
                .join(self._get_pharma_variables(), on="pharmaid", how="left")
                .join(pharmaid_to_ingredient, on="pharmaid", how="left")
                .with_columns(
                    (pl.col("givenat") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias(self.drug_end_col),
                    # Map the medication classes
                    pl.col(self.drug_class_col)
                    .cast(str)
                    .replace_strict(
                        hirid_drug_class_mapping,
                        default=None,
                        return_dtype=pl.String,
                    ),
                    # Map the medication administration routes
                    pl.col(self.drug_admin_route_col).replace_strict(
                        hirid_drug_administration_route_mapping,
                        default=None,
                        return_dtype=pl.String,
                    ),
                )
                .drop("admissiontime", "givenat")
                # Remove duplicate rows
                .unique()
                # Remove rows with empty lab names
                .filter(pl.col(self.drug_amount_col).is_not_null())
                .drop(self.icu_length_of_stay_col)
            )

            # Append the data to the DataFrame
            pharma = pl.concat([pharma, data.lazy()], how="diagonal_relaxed")

        # Get infusion duration where possible, by checking whether the drugname reappears
        # on next log entry (as determined by a different offset)
        # 1. Get list of log entry offsets for each patient
        pharma_offsets = (
            pharma.select(
                self.icu_stay_id_col,
                self.drug_end_col,
                self.drug_mixture_id_col,
            )
            .unique()
            .sort(
                self.icu_stay_id_col,
                self.drug_mixture_id_col,
                self.drug_end_col,
            )
            .with_columns(
                pl.col(self.drug_end_col)
                .shift(1)
                .over(self.icu_stay_id_col, self.drug_mixture_id_col)
                .alias("prev_drug_end"),
                pl.col(self.drug_end_col)
                .shift(-1)
                .over(self.icu_stay_id_col, self.drug_mixture_id_col)
                .alias("next_drug_end"),
            )
        )

        pharma = (
            pharma.join(
                pharma_offsets,
                on=[
                    self.icu_stay_id_col,
                    self.drug_end_col,
                    self.drug_mixture_id_col,
                ],
                how="left",
            )
            # Sort by patient ID, drug name and drug start time
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_mixture_id_col,
                self.drug_end_col,
                "prev_drug_end",  # sometimes, there is the same drug given twice at the same time
            )
            # NOTE: Convert drug_amount to drug_rates, fluid_amount to fluid_rates
            .with_columns(
                (
                    pl.col(self.drug_amount_col)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                    * 3600
                )
                .round_sig_figs(2)
                .alias(self.drug_rate_col),
                pl.col(self.drug_amount_unit_col).str.replace("µ", "mc"),
                (pl.col(self.drug_amount_unit_col) + pl.lit("/hr"))
                .str.replace("µ", "mc")
                .alias(self.drug_rate_unit_col),
                (
                    pl.col(self.fluid_amount_col)
                    / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                    * 3600
                )
                .round_sig_figs(2)
                .alias(self.fluid_rate_col),
            )
            # 2. Check if drug is continued from the previous log entry
            #    and if it is continued in the next log entry
            .with_columns(
                # Check if drug is continued from the previous log entry
                pl.when(pl.col("prev_drug_end").is_not_null())
                .then(
                    pl.when(
                        # Check if the previous drug is the same as the current drug
                        pl.col(self.drug_name_col)
                        == pl.col(self.drug_name_col).shift(1),
                        # Check if the previous drug end time is the previous log entry time
                        pl.col("prev_drug_end")
                        == pl.col(self.drug_end_col).shift(1),
                        # Check if the drug rate is the same as the previous drug rate
                        pl.col(self.drug_rate_col)
                        == pl.col(self.drug_rate_col).shift(1),
                    )
                    .then(pl.lit("continued"))
                    .otherwise(pl.lit("started"))
                )
                .otherwise(None)
                .alias("drug_status_prev"),
                # Check if drug is continued in the next log entry
                pl.when(
                    # Check if the next drug is the same as the current drug
                    pl.col(self.drug_name_col)
                    == pl.col(self.drug_name_col).shift(-1),
                    # Check if the next drug end time is the next log entry time
                    pl.col("next_drug_end")
                    == pl.col(self.drug_end_col).shift(-1),
                    # Check if the drug rate is the same as the next drug rate
                    pl.col(self.drug_rate_col)
                    == pl.col(self.drug_rate_col).shift(-1),
                )
                .then(pl.lit("continued"))
                .otherwise(pl.lit("discontinued"))
                .alias("drug_status_next"),
            )
            # Filter for rows where the drug status changes
            .filter(pl.col("drug_status_prev") != pl.col("drug_status_next"))
            # 3. Get the end time of the drug if it is discontinued
            .with_columns(
                pl.when(pl.col("drug_status_next") == "discontinued")
                .then("prev_drug_end")
                .otherwise(None)
                .alias(self.drug_start_col)
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
                .then(pl.col("prev_drug_end").shift(1))
                .otherwise(pl.col("prev_drug_end"))
                .alias(self.drug_start_col)
            )
            # 5. filter out duplicate rows (same drug, same start time, same rate, different end time)
            .filter(pl.col(self.drug_start_col).is_not_null())
            .sort(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_start_col,
                self.drug_rate_col,
                self.drug_end_col,
            )
            .group_by(
                self.icu_stay_id_col,
                self.drug_name_col,
                self.drug_start_col,
                self.drug_rate_col,
                maintain_order=True,
            )
            .last()
            # 6. Remove the helper columns
            .drop(
                "prev_drug_end",
                "next_drug_end",
                "drug_status_prev",
                "drug_status_next",
            )
        )

        return pl.concat([pharma, observation], how="diagonal_relaxed")

    # endregion

    # region helpers
    def _get_variable_reference(self) -> pl.DataFrame:
        """
        Loads the variable reference mapping from a CSV file.

        The CSV must contain columns:
            - "Source Table": Indicates the data source.
            - "ID": The variable identifier.
            - "Variable Name": The human-readable name of the variable.

        Returns:
            pl.DataFrame: DataFrame with columns "Source Table", "ID", and "Variable Name".
        """
        return pl.read_csv(
            self.variable_reference_path,
            # separator=";",
            # encoding="unicode_escape",
            columns=["Source Table", "ID", "Variable Name"],
        )

    def _get_observation_variables(self) -> pl.DataFrame:
        """
        Retrieves and filters observation variables from the variable reference CSV.

        Steps performed:
          1. Load the complete variable reference.
          2. Filter rows where "Source Table" equals "Observation".
          3. Select and rename columns:
              - "ID" to "variableid"
              - "Variable Name" to "variable"
          4. Replace legacy mappings using internal dictionaries.

        Returns:
            pl.DataFrame: DataFrame with columns:
                - "variableid" (Observation variable identifier),
                - "variable" (Observation variable name).
        """
        return (
            self._get_variable_reference()
            .filter(pl.col("Source Table") == "Observation")
            .select("ID", "Variable Name")
            .with_columns(
                pl.col("Variable Name")
                # Fix bad mappings (wrong units)
                .replace(
                    "Bilirubin.direct [Mass/volume] in Serum or Plasma",
                    "Bilirubin.direct [Moles/volume] in Serum or Plasma",
                )
                # Replace the variable names with the reprodICU mapping
                .replace(
                    {
                        **self.timeseries_vitals_mapping,
                        **self.timeseries_intakeoutput_mapping,
                        **self.timeseries_respiratory_mapping,
                    }
                )
            )
            .rename({"ID": "variableid", "Variable Name": "variable"})
            .lazy()
        )

    def _get_pharma_variables(self) -> pl.LazyFrame:
        """
        Retrieves pharmaceutical variable mappings from the variable reference CSV.

        Returns:
            pl.LazyFrame: LazyFrame with columns:
                - "pharmaid"
                - self.drug_name_col
                - self.drug_name_OMOP_col
        """
        return (
            pl.read_csv(self.MEDICATION_MAPPING_PATH + "HiRID.usagi.csv")
            .filter(pl.col("conceptName") != "Unmapped")
            .select("sourceCode", "sourceName", "conceptName")
            .drop_nulls("sourceCode")
            .unique()
            .rename(
                {
                    "sourceCode": "pharmaid",
                    "sourceName": self.drug_name_col,
                    "conceptName": self.drug_name_OMOP_col,
                }
            )
            .lazy()
        )
        
    # Extract the information from the HiRID.usagi.csv files
    def _extract_drug_references(self, return_ids: bool = False) -> dict:
        """
        Extract and process drug references from CSV mapping files.
        """

        value_col = "conceptName" if not return_ids else "conceptId"
        references = (
            pl.read_csv(self.MEDICATION_MAPPING_PATH + "HiRID.usagi.csv")
            .filter(pl.col("conceptName") != "Unmapped")
            .select("sourceCode", value_col)
            .drop_nulls("sourceCode")
            .unique()
        )

        return dict(
            zip(
                references["sourceCode"].to_numpy(),
                references[value_col].to_numpy(),
            )
        )
