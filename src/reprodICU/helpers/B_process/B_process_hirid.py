# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This script processes the HiRID data and stores it in a structured format for further
# processing and harmonization.


import glob
import os
import sys
from pathlib import Path

import polars as pl
import time

from ..A_extract.A_extract_hirid import HiRIDExtractor
from ..helper import GlobalHelpers
from ..helper_conversions import UnitConverter


class HiRIDProcessor(HiRIDExtractor):
    def __init__(self, paths):
        """
        Initializes the HiRIDProcessor instance.

        Args:
            paths: Object containing various source and destination paths.

        Sets:
            path: Source path for HiRID data ({hirid_source_path}).
            helpers: Instance of GlobalHelpers.
            convert: Instance of HiRIDConverter.
            index_cols: List containing {icu_stay_id_col} and {timeseries_time_col}.
        """
        super().__init__(paths)
        self.path = paths.hirid_source_path
        self.helpers = GlobalHelpers()
        self.convert = HiRIDConverter()
        self.index_cols = [self.icu_stay_id_col, self.timeseries_time_col]
        self.admissiontime = (
            self._extract_admissions()
            .select(self.icu_stay_id_col, "admissiontime")
            .cast({"admissiontime": str})
        )
        self.length_of_stay = self._extract_length_of_stay()

    def _get_labname_components(self) -> pl.LazyFrame:
        """
        Retrieves the lab names from the HiRID dataset.

        Returns:
            pl.LazyFrame: LazyFrame containing the LOINC data.
        """
        data = (
            pl.scan_parquet(self.timeseries_path + "*.parquet")
            .select("variableid")
            .unique()
            .join(self._get_observation_variables(), on="variableid")
            .select("variable")
        )

        # Remove duplicates and sort the LOINC data
        labnames = data.collect().to_series().to_list()
        return data.with_columns(
            pl.col("variable")
            .replace_strict(
                self.omop.get_lab_component_from_name(labnames),
                default=None,
            )
            .alias("LOINC_component")
        ).unique()

    # region time series
    def process_timeseries(self) -> pl.LazyFrame:
        """
        Process non-laboratory time series data.

        Steps:
            1. Check for preprocessed non-lab timeseries file; load if available.
            2. For each raw timeseries file: filter to non-lab variables, extract and pivot measurements.
            3. Combine all files into single wide-format dataset.
            4. Sort by index columns and save.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Non-laboratory measurement columns (pivoted from variable).
        """
        ts_path = self.precalc_path + "HiRID_timeseries.parquet"

        if os.path.isfile(ts_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Processing timeseries data...")

        # Since each case has its data in only one file, iterating over the files specifically
        # allows for a more efficient processing of the data.
        files = sorted(list(Path(self.timeseries_path).glob("*.parquet")))
        total_files = len(files)
        batch_size = 10
        total_batches = (total_files + batch_size - 1) // batch_size

        times, cases = [], 0
        batch_id = "HiRID_ts_batch"
        for i in range(0, total_files, batch_size):
            start = time.time()
            index = str(i // batch_size).zfill(4)

            batch_paths = files[i : i + batch_size]

            # Process timeseries data
            timeseries = (
                # Drop the lab values from the timeseries data
                pl.scan_parquet(batch_paths)
                .filter(~pl.col("variableid").is_between(20000000, 25000000))
                .pipe(
                    self._extract_timeseries_helper,
                    self.admissiontime,
                    self.length_of_stay,
                )
            )

            (
                # Pivot the timeseries data
                self.pivot_numeric_or_string(
                    timeseries,
                    dataset="HiRID_ts",
                    on_col="variable",
                    index_cols=self.index_cols,
                    numeric_col="value",
                ).sort(self.index_cols)
                # Sink each intermittent batch result to a parquet
                .sink_parquet(self.precalc_path + f"{batch_id}_{index}.parquet")
            )

            # Update timing information
            elapsed = time.time() - start
            times.append(elapsed)
            avg = sum(times) / len(times)
            eta_min = int(avg * (total_batches - (i // batch_size) - 1) / 60 + 0.5) # fmt: skip

            cases += (
                timeseries.select(self.icu_stay_id_col)
                .unique()
                .collect()
                .shape[0]
            )

            sys.stdout.write("\033[K")  # Clear to the end of line
            print(
                f"Processing batch {i//batch_size + 1:3.0f} of {total_batches} "
                f"with {len(batch_paths):4.0f} files ({cases:5.0f} cases) "
                f"(last: {elapsed:.2f}s, avg: {avg:.2f}s, ETA: {eta_min:d} min)",
                end="\r",
            )

        print("\nBatch processing complete. Concatenating results...")

        # Concatenate and sink all processed frames
        temp_files = sorted(
            glob.glob(self.precalc_path + f"{batch_id}_*.parquet")
        )
        batch_frames = [pl.scan_parquet(file) for file in temp_files]
        pl.concat(batch_frames, how="diagonal_relaxed").sink_parquet(ts_path)

        # Delete the temporary files
        for file in temp_files:
            os.remove(file)

        # Load the preprocessed data
        return pl.scan_parquet(ts_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region timeseries labs
    def process_timeseries_labs(self) -> pl.LazyFrame:
        """
        Process laboratory time series data.

        Steps:
            1. Check for preprocessed labs file; load if available.
            2. For each raw timeseries file: filter to lab variables, extract measurements.
            3. Combine all files, convert lab values to canonical units.
            4. Apply LOINC component mapping and JSON encode structured fields.
            5. Save extracted data to temporary file before batch pivoting.
            6. Batch pivot data on variable name to create wide-format dataset.
            7. Apply post-pivot unit conversions for derived measurements.
            8. Save, sort by index columns, and clean temporary files.

        Returns:
            pl.LazyFrame: Contains columns:
                - {icu_stay_id_col}: ICU stay identifier.
                - {timeseries_time_col}: Time offset (seconds from ICU admission).
                - Laboratory measurement columns (pivoted from variable, JSON-encoded).
        """
        ts_labs_path = self.precalc_path + "HiRID_timeseries_labs.parquet"
        ts_labs_path_unsorted = self.precalc_path + "HiRID_ts_labs.parquet"

        if os.path.isfile(ts_labs_path):
            # Load the preprocessed data
            return pl.scan_parquet(ts_labs_path).select(
                pl.col(self.index_cols).set_sorted(),
                pl.exclude(self.index_cols),
            )

        print("HiRID   - Collecting lab time series data...")
        labname_components = self._get_labname_components()

        # Process timeseries data
        (
            pl.scan_parquet(self.timeseries_path + "*.parquet")
            # Keep the lab values from the timeseries data
            .filter(pl.col("variableid").is_between(20000000, 25000000))
            .pipe(
                self._extract_timeseries_helper,
                self.admissiontime,
                self.length_of_stay,
            )
            # Drop the non-lab values from the timeseries data
            .join(labname_components, on="variable")
            .filter(pl.col("LOINC_component").is_not_null())
            # Extract and transform lab data
            .pipe(self._extract_timeseries_labs_helper)
            # Convert the lab values to the correct units
            .pipe(
                self.convert._convert_lab_values,
                labelcol="variable",
                valuecol="labstruct",
            )
            # Replace the LOINC codes
            .pipe(
                self.convert._assign_LOINC_codes,
                self.omop,
                self.index_cols,
                struct_cols=["labstruct"],
                component_col="variable",
            )
            # JSON encode the structs for pivoting
            .with_columns(pl.col("labstruct").struct.json_encode())
            # Pivot the lab data
            .pipe(
                self.pivot_numeric_or_string,
                dataset="HiRID_labs",
                on_col="variable",
                index_cols=self.index_cols,
                string_col="labstruct",
            )
            # Save extracted data before pivoting
            .sink_parquet(ts_labs_path_unsorted)
        )

        # Sort the data
        (
            pl.scan_parquet(ts_labs_path_unsorted)
            .sort(self.index_cols)
            .sink_parquet(ts_labs_path)
        )
        os.remove(ts_labs_path_unsorted)

        # Load the preprocessed data
        return pl.scan_parquet(ts_labs_path).select(
            pl.col(self.index_cols).set_sorted(),
            pl.exclude(self.index_cols),
        )

    # endregion

    # region medications
    def process_medications(self) -> pl.LazyFrame:
        """
        Process the medication administrations.

        Steps:
            1. Check for preprocessed medications file; load if available.
            2. Extract the pharma records and the fluids (observation tables) in batches of files.
            3. Link the fluid counter readings to the pharma records carrying them.
            4. Derive infusion rates, start and end times of the pharma records.
            5. Move the fluid records from the drug to the fluid columns.
            6. Sink each batch, concatenate them and clean temporary files.

        Returns:
            pl.LazyFrame: Medication administrations in the common medication format.
        """
        medications_path = self.precalc_path + "HiRID_medications.parquet"

        if os.path.isfile(medications_path):
            # Load the preprocessed data
            return pl.scan_parquet(medications_path)

        print("HiRID   - Processing medications...")

        # The counter is a float32, hence the tolerance
        COUNTER_TOLERANCE_ML = 0.01
        COUNTER_TOLERANCE_RELATIVE = 1e-3
        # Significant figures and minimal repeats of a syringe ratio per infusion
        RATIO_SIG_FIGS = 3
        RATIO_MIN_REPEATS = 3

        instant_cols = [self.icu_stay_id_col, "givenat"]
        mixture_cols = [self.icu_stay_id_col, self.drug_mixture_id_col]
        infusion_cols = [*mixture_cols, self.drug_admin_type_col]
        drug_cols = [
            self.icu_stay_id_col,
            "pharmaid",
            self.drug_mixture_id_col,
            self.drug_admin_type_col,
        ]
        amount_cols = [self.drug_amount_col, self.fluid_amount_col]

        # Load the volumes that pharma records add to the fluid counters
        carrier_volumes = pl.read_csv(
            self.carrier_volumes_mapping_path,
            comment_prefix="#",
            columns=[
                "pharmaid",
                "ml",
                "ml_per_ml",
                "ml_per_dose",
                "per_infusion",
                "dose_from",
                "dose_unit",
            ],
            schema_overrides={"pharmaid": pl.Int32},
        ).lazy()
        fluid_classes = self.load_many_to_one_mapping(
            self.mapping_path + "MEDICATIONS_FLUIDS_CLASSES.yaml"
        )

        # Since each case has its data in only one file (with the same name in
        # the observation and pharma folder), iterating over the files in
        # batches allows for a more efficient processing of the data.
        batch_size = 10
        files = sorted(Path(self.pharma_path).glob("*.parquet"))
        total_batches = (len(files) + batch_size - 1) // batch_size
        for i in range(0, len(files), batch_size):
            start = time.time()
            index = str(i // batch_size).zfill(4)
            batch_names = [file.name for file in files[i : i + batch_size]]

            # region EXTRACT
            ###################################################################
            # collected, since both are used several times below
            pharma = (
                pl.scan_parquet([self.pharma_path + name for name in batch_names])
                .pipe(
                    self._extract_medications_pharma,
                    self.admissiontime,
                    self.length_of_stay,
                )
                .collect()
                .lazy()
            )
            fluids = (
                pl.scan_parquet([self.timeseries_path + name for name in batch_names])
                .pipe(
                    self._extract_medications_observation,
                    self.admissiontime,
                    self.length_of_stay,
                )
                .collect()
                .lazy()
                .rename({"datetime": "givenat"})
            )

            # region CARRIER LINKING
            ###################################################################
            # sum the readings at the same instant
            readings = fluids.group_by(instant_cols).agg(
                pl.col(self.fluid_group_col).min(),
                pl.col(self.fluid_amount_col).sum().alias("reading"),
            )

            # a pharma record takes the carrier volume row with the highest
            # dose_from that its dose reaches
            pharma = pharma.with_row_index("record_id")
            volumes = (
                pharma.join(carrier_volumes, on="pharmaid", how="left")
                .filter(pl.col(self.drug_admin_type_col) == "given")
                .filter(
                    pl.col("dose_from").is_null()
                    | (
                        (pl.col(self.drug_amount_col) >= pl.col("dose_from"))
                        & pl.col("dose_unit").eq_missing(
                            pl.col(self.drug_amount_unit_col)
                        )
                    )
                )
                .sort("dose_from", nulls_last=False)
                .unique("record_id", keep="last", maintain_order=True)
            )

            # ratio of the drugs that have one per infusion (syringe
            # concentration): the most frequent ratio between reading and pharma
            # volume of the records alone at their instant
            ratios = (
                volumes.filter(
                    pl.col("per_infusion")
                    & (pl.col(self.fluid_amount_col) > 0)
                    & (pl.len().over(instant_cols) == 1)
                )
                .join(readings, on=instant_cols)
                .with_columns(
                    (pl.col("reading") / pl.col(self.fluid_amount_col))
                    .round_sig_figs(RATIO_SIG_FIGS)
                    .alias("ratio")
                )
                .group_by(*mixture_cols, "pharmaid", "ratio")
                .agg(pl.len().alias("count"))
                .sort("count", "ratio", descending=[True, False])
                .unique([*mixture_cols, "pharmaid"], keep="first", maintain_order=True)
                .filter(pl.col("count") >= RATIO_MIN_REPEATS)
            )

            # volume that each record adds to the counter
            # (injections are excluded, their readings rarely match)
            carrying = (
                volumes.filter(~pl.col("injection"))
                .join(
                    ratios,
                    on=[*mixture_cols, "pharmaid"],
                    how="left",
                    nulls_equal=True,
                )
                .with_columns(
                    pl.coalesce(
                        "ml",
                        pl.col("ml_per_dose") * pl.col(self.drug_amount_col),
                        pl.coalesce("ratio", "ml_per_ml")
                        * pl.col(self.fluid_amount_col),
                    ).alias("carried")
                )
                .filter(pl.col("carried") > 0)
            )

            # readings that equal the summed volume of their instant
            linked = (
                carrying.group_by(instant_cols)
                .agg(pl.col("carried").sum().alias("expected"))
                .join(readings, on=instant_cols)
                .filter(
                    (pl.col("reading") - pl.col("expected")).abs()
                    <= COUNTER_TOLERANCE_ML
                    + COUNTER_TOLERANCE_RELATIVE * pl.col("expected")
                )
            )

            # the records take their share and the fluid group of the reading,
            # the linked readings are removed from the fluids
            pharma = (
                pharma.join(
                    carrying.join(linked, on=instant_cols).select(
                        "record_id",
                        self.fluid_group_col,
                        pl.col("carried").alias("share"),
                    ),
                    on="record_id",
                    how="left",
                )
                .with_columns(
                    pl.col("share")
                    .fill_null(pl.col(self.fluid_amount_col))
                    .alias(self.fluid_amount_col)
                )
                .drop("record_id", "injection", "share", "givenat")
            )
            fluids = fluids.join(linked, on=instant_cols, how="anti").drop(
                "givenat"
            )

            # region INFUSIONS
            ###################################################################
            pharma = pharma.with_columns(
                pl.col(self.drug_amount_unit_col).str.replace("µ", "mc")
            )

            # records outside of an infusion (no mixture) are point administrations
            boluses = pharma.filter(
                pl.col(self.drug_mixture_id_col).is_null()
            ).with_columns(pl.col(self.drug_end_col).alias(self.drug_start_col))
            pharma = pharma.filter(pl.col(self.drug_mixture_id_col).is_not_null())

            # log entries of a drug within the same second are one entry (e.g.
            # the stop of an infusion has a second entry without a dose), which
            # takes the values of the one with the highest amounts
            pharma = (
                pharma.sort(*drug_cols, self.drug_end_col, *amount_cols)
                .group_by(*drug_cols, self.drug_end_col, maintain_order=True)
                .agg(
                    pl.col(amount_cols).sum(),
                    pl.exclude(*drug_cols, self.drug_end_col, *amount_cols).last(),
                )
            )

            # end of the previous log entry of the infusion
            pharma = pharma.join(
                pharma.select(*infusion_cols, self.drug_end_col)
                .unique()
                .sort(*infusion_cols, self.drug_end_col)
                .with_columns(
                    pl.col(self.drug_end_col)
                    .shift(1)
                    .over(infusion_cols)
                    .alias("prev_drug_end")
                ),
                on=[*infusion_cols, self.drug_end_col],
                how="left",
            )

            # the first log entry of an infusion has no interval: it is a point
            # administration if it has a dose or is the only log entry, else it
            # is dropped (the start of the next interval is already its end)
            administrations = (
                pharma.filter(
                    pl.col("prev_drug_end").is_null()
                    & (
                        (pl.len().over(infusion_cols) == 1)
                        | pl.any_horizontal(pl.col(amount_cols) > 0)
                    )
                )
                .with_columns(pl.col(self.drug_end_col).alias(self.drug_start_col))
                .drop("prev_drug_end")
            )

            pharma = (
                pharma.filter(pl.col("prev_drug_end").is_not_null())
                .sort(*drug_cols, self.drug_end_col)
                # NOTE: Convert drug_amount to drug_rates, fluid_amount to fluid_rates
                .with_columns(
                    *[
                        (
                            pl.col(amount)
                            / (pl.col(self.drug_end_col) - pl.col("prev_drug_end"))
                            * 3600
                        )
                        .round_sig_figs(2)
                        .alias(rate)
                        for amount, rate in (
                            (self.drug_amount_col, self.drug_rate_col),
                            (self.fluid_amount_col, self.fluid_rate_col),
                        )
                    ],
                )
                .with_columns(
                    pl.format("{}/hr", self.drug_amount_unit_col).alias(
                        self.drug_rate_unit_col
                    ),
                    # a log entry continues the previous one of the same drug if
                    # it starts where that ended, at the same rate; a run of
                    # continued log entries is one interval
                    (
                        (pl.col("prev_drug_end") == pl.col(self.drug_end_col).shift(1))
                        & (
                            pl.col(self.drug_rate_col)
                            == pl.col(self.drug_rate_col).shift(1)
                        )
                    )
                    .fill_null(False)
                    .not_()
                    .cum_sum()
                    .over(drug_cols)
                    .alias("run"),
                )
                # merge each run: starts with the first log entry, sums the amounts
                .group_by(*drug_cols, "run", maintain_order=True)
                .agg(
                    pl.col("prev_drug_end").first().alias(self.drug_start_col),
                    pl.col(amount_cols).sum(),
                    pl.exclude(*drug_cols, "run", *amount_cols).last(),
                )
                .drop("prev_drug_end", "run")
            )

            pharma = pl.concat(
                [pharma, administrations, boluses], how="diagonal_relaxed"
            ).with_columns(
                # a point administration has no end
                pl.when(pl.col(self.drug_start_col) != pl.col(self.drug_end_col))
                .then(pl.col(self.drug_end_col))
                .alias(self.drug_end_col)
            )

            # region FLUID COLUMNS
            ###################################################################
            # move the fluids (given in ml) from the drug to the fluid columns,
            # and map their class and fluid group
            pharma = pharma.with_columns(
                pl.when(pl.col("fluid_entry"))
                .then(pl.col(self.drug_name_col))
                .alias(self.fluid_name_col),
                *[
                    pl.when(pl.col("fluid_entry"))
                    .then(
                        pl.col(self.drug_name_col).replace_strict(
                            mapping, default=None
                        )
                    )
                    .fill_null(pl.col(column))
                    .alias(column)
                    for column, mapping in (
                        (self.drug_class_col, fluid_classes),
                        (self.fluid_group_col, self.SOLUTION_FLUIDS_MAP),
                    )
                ],
                pl.when(~pl.col("fluid_entry")).then(
                    pl.col(
                        self.drug_name_col,
                        self.drug_amount_col,
                        self.drug_amount_unit_col,
                        self.drug_rate_col,
                        self.drug_rate_unit_col,
                    )
                ),
            ).drop("fluid_entry")

            # region SINK
            ###################################################################
            # sink each intermittent batch result to a parquet
            for table, data in (("observation", fluids), ("pharma", pharma)):
                data.sink_parquet(
                    self.precalc_path
                    + f"HiRID_medications_{table}_batch_{index}.parquet"
                )

            sys.stdout.write("\033[K")  # Clear to the end of line
            print(
                f"Processing batch {i//batch_size + 1:3.0f} of "
                f"{total_batches} (last: {time.time() - start:.2f}s)",
                end="\r",
            )

        print("\nBatch processing complete. Concatenating results...")

        # Concatenate and sink all processed batches
        batch_files = sorted(
            glob.glob(self.precalc_path + "HiRID_medications_*_batch_*.parquet")
        )
        pl.concat(
            [pl.scan_parquet(file) for file in batch_files],
            how="diagonal_relaxed",
        ).sink_parquet(medications_path)

        # Delete the temporary files
        for file in batch_files:
            os.remove(file)

        return pl.scan_parquet(medications_path)

    # endregion


# region convert
class HiRIDConverter(UnitConverter):
    def __init__(self):
        super().__init__()

    def _convert_lab_values(
        self,
        data: pl.LazyFrame,
        labelcol: str = "variableid",
        valuecol: str = "value_struct",
        structfield: str = "value",
    ) -> pl.LazyFrame:
        """Convert raw lab values to canonical units.

        Applies sequential unit conversions for multiple lab tests including:
        bilirubin, creatinine, cortisol, fibrinogen, glucose, hemoglobin, and urea.

        Expected columns:
            - {labelcol}: Lab test identifier.
            - {valuecol}: Lab measurement value.

        Returns:
            pl.LazyFrame: Lab data with unit-converted values.
        """
        return (
            data.pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin.direct",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_bilirubin_umol_L_to_mg_dL,
                itemid="Bilirubin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_creatinine_umol_L_to_mg_dL,
                itemid="Creatinine",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_cortisol_nmol_L_to_ug_dL,
                itemid="Cortisol",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_L_to_mg_dL,
                itemid="Fibrinogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_glucose_mmol_L_to_mg_dL,
                itemid="Glucose",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_g_L_to_g_dL,
                itemid="Hemoglobin",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                # same conversion due to definition of MCHC
                self.convert_g_L_to_g_dL,
                itemid="Erythrocyte mean corpuscular hemoglobin concentration",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_urea_nitrogen_from_urea,
                itemid_urea="Urea",
                itemid_BUN="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
            .pipe(
                self.convert_blood_urea_nitrogen_mmol_L_to_mg_dL,
                itemid="Urea nitrogen",
                labelcol=labelcol,
                valuecol=valuecol,
                structfield=structfield,
            )
        )


# endregion
