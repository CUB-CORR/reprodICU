# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPT "Ventilation Duration" directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import os
import polars as pl

from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS import MAGIC_CONCEPTS


class VENTILATION_DURATION(MAGIC_CONCEPTS):
    def __init__(self, paths, datasets):
        super().__init__(paths, datasets)

    def VENTILATION_DURATION(self) -> pl.DataFrame:
        """
        Returns the magic concept VENTILATION_DURATION

        Description:
        This concept is used to determine whether a patient received any antibiotics during the ICU stay.

        Returns a DataFrame with the following columns:
        - ICU stay ID
        - ventilation type "Ventilation Type"
          (one of "tracheostomy", "invasive ventilation", "non-invasive ventilation", "other")
        - ventilation start "Ventilation Start Relative to Admission (seconds)"
        - ventilation end "Ventilation End Relative to Admission (seconds)"
        - ventilation duration "Ventilation Duration (hours)"

        :return: DataFrame
        :rtype: pl.DataFrame
        """

        # region eICU
        # print("MAGIC_CONCEPTS: Ventilation Duration - eICU")
        eicu_VENTILATION_DURATION = (
            pl.scan_csv(self.eicu_paths.respiratoryCare_path)
            # ventstartoffset and ventendoffset seem not include full ventilation duration
            .select(
                "patientunitstayid",
                "priorventstartoffset",
                "priorventendoffset",
            )
            .with_columns(
                # reltimes in eICU are in minutes
                (pl.col("priorventstartoffset") * 60).alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                (pl.col("priorventendoffset") * 60).alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
            )
            .drop("priorventstartoffset", "priorventendoffset")
            .with_columns(
                # add duration
                pl.duration(
                    seconds=(
                        pl.col(
                            "Ventilation End Relative to Admission (seconds)"
                        )
                        - pl.col(
                            "Ventilation Start Relative to Admission (seconds)"
                        )
                    )
                )
                .truediv(pl.duration(hours=1))
                .alias("Ventilation Duration (hours)")
            )
        ).pipe(self._add_global_id_stay_id, "eicu-", "patientunitstayid")

        # region HiRID
        # print("MAGIC_CONCEPTS: Ventilation Duration - HiRID")

        # get admission times for HiRID
        hirid_ADMISSIONTIMES = (
            pl.scan_csv(self.hirid_paths.general_table_path)
            .select("patientid", "admissiontime")
            .with_columns(
                pl.col("patientid").cast(str),
                pl.col("admissiontime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        # Scan all files in the timeseries folder
        hirid_VENTILATION_DURATION = pl.LazyFrame()

        for file in os.listdir(self.hirid_paths.timeseries_path):
            hirid_timeseries_AIRWAYTYPE = (
                pl.scan_parquet(self.hirid_paths.timeseries_path + file)
                .select("datetime", "patientid", "value", "variableid")
                .cast({"datetime": str, "patientid": str})
                # Filter for ventilation IDs
                .filter(
                    pl.col("variableid")
                    == self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["hirid"][0]["ids"]
                )
                .drop("variableid")
                # replace ventilation concepts
                .with_columns(
                    pl.col("value")
                    .cast(int)
                    .cast(str)
                    .replace(
                        {
                            "1": "invasive ventilation",
                            "2": "tracheostomy",
                            "3": "non-invasive ventilation",
                            "4": "non-invasive ventilation",
                            "5": "non-invasive ventilation",
                            "6": "other",  # TODO: check if this is correct
                        }
                    )
                    .alias("Ventilation Type"),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                )
                # Make datetime relative to admission in seconds
                .join(hirid_ADMISSIONTIMES, on="patientid", how="left")
                .with_columns(
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias("Ventilation Start Relative to Admission (seconds)")
                )
                .drop("admissiontime", "datetime", "value")
                # Rename columns
            )

            hirid_timeseries_VENTMODE = (
                pl.scan_parquet(self.hirid_paths.timeseries_path + file)
                .select("datetime", "patientid", "value", "variableid")
                .cast({"datetime": str, "patientid": str})
                # Filter for ventilation IDs
                .filter(pl.col("variableid") == 3845)  # Ventilator mode
                .drop("variableid")
                # replace ventilation concepts
                .with_columns(
                    pl.when(pl.col("value").ne(1))
                    .then(pl.lit("active"))
                    .otherwise(pl.lit("inactive"))
                    .alias("Ventilator Mode"),
                    pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S%.9f"),
                )
                # Make datetime relative to admission in seconds
                .join(hirid_ADMISSIONTIMES, on="patientid", how="left")
                .with_columns(
                    (pl.col("datetime") - pl.col("admissiontime"))
                    .dt.total_seconds()
                    .alias("Ventilation Start Relative to Admission (seconds)")
                )
                .drop("admissiontime", "datetime", "value")
                # Rename columns
            )

            hirid_timeseries = (
                pl.concat(
                    [hirid_timeseries_AIRWAYTYPE, hirid_timeseries_VENTMODE],
                    how="align",
                )
                .select(pl.all().forward_fill())
                .sort(
                    "patientid",
                    "Ventilation Start Relative to Admission (seconds)",
                )
                # drop rows where both columns are staying the same
                .filter(
                    pl.col("Ventilation Type").ne_missing(
                        pl.col("Ventilation Type").shift(-1)
                    )
                    | pl.col("Ventilator Mode").ne_missing(
                        pl.col("Ventilator Mode").shift(-1)
                    ),
                )
                .with_columns(
                    pl.when(
                        pl.col("patientid") == pl.col("patientid").shift(-1)
                    )
                    .then(
                        pl.col(
                            "Ventilation Start Relative to Admission (seconds)"
                        ).shift(-1)
                    )
                    .otherwise(None)
                    .alias("Ventilation End Relative to Admission (seconds)")
                )
                .with_columns(
                    (
                        (
                            pl.col(
                                "Ventilation End Relative to Admission (seconds)"
                            )
                            - pl.col(
                                "Ventilation Start Relative to Admission (seconds)"
                            )
                        )
                        / (60 * 60)
                    ).alias("Ventilation Duration (hours)")
                )
                .filter(pl.col("Ventilator Mode") == "active")
            )

            hirid_VENTILATION_DURATION = pl.concat(
                [hirid_VENTILATION_DURATION, hirid_timeseries],
                how="diagonal_relaxed",
            )

        hirid_VENTILATION_DURATION = hirid_VENTILATION_DURATION.pipe(
            self._add_global_id_stay_id, "hirid-", "patientid"
        )
        # endregion

        # region MIMIC-III
        # print("MAGIC_CONCEPTS: Ventilation Duration - MIMIC3")

        # get admission times for MIMIC-III
        mimic3_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic3_paths.icustays_path)
            .select("ICUSTAY_ID", "INTIME")
            .with_columns(
                pl.col("INTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic3_VENTILATION_DURATION = (
            pl.scan_csv(self.mimic3_paths.procedureevents_mv_path)
            .select("ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID")
            .join(mimic3_ADMISSIONTIMES, on="ICUSTAY_ID", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("ITEMID").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["miiv"][0]["ids"]
                )
            )
            .cast({"ITEMID": str})
            # replace ventilation concepts
            .with_columns(
                pl.col("ITEMID")
                .replace(
                    {
                        225792: "invasive ventilation",
                        225794: "non-invasive ventilation",
                    }
                )
                .cast(str)
                .alias("Ventilation Type")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("STARTTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("ENDTIME").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("STARTTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("ENDTIME") - pl.col("INTIME"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation End Relative to Admission (seconds)"),
                (pl.col("ENDTIME") - pl.col("STARTTIME"))
                .truediv(pl.duration(hours=1))
                .alias("Ventilation Duration (hours)"),
            )
            .drop("INTIME", "STARTTIME", "ENDTIME", "ITEMID")
            .pipe(self._add_global_id_stay_id, "mimic3-", "ICUSTAY_ID")
        )
        # endregion

        # region MIMIC-IV
        # print("MAGIC_CONCEPTS: Ventilation Duration - MIMIC4")

        # get admission times for MIMIC-IV
        mimic4_ADMISSIONTIMES = (
            pl.scan_csv(self.mimic4_paths.icustays_path)
            .select("stay_id", "intime")
            .with_columns(
                pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
        )

        mimic4_VENTILATION_DURATION = (
            pl.scan_csv(self.mimic4_paths.procedureevents_path)
            .select("stay_id", "starttime", "endtime", "itemid")
            .join(mimic4_ADMISSIONTIMES, on="stay_id", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["miiv"][0]["ids"]
                )
            )
            .cast({"itemid": str})
            # replace ventilation concepts
            .with_columns(
                pl.col("itemid")
                .replace(
                    {
                        225792: "invasive ventilation",
                        225794: "non-invasive ventilation",
                    }
                )
                .cast(str)
                .alias("Ventilation Type")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.col("starttime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("endtime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
            )
            .with_columns(
                (pl.col("starttime") - pl.col("intime"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation Start Relative to Admission (seconds)"),
                (pl.col("endtime") - pl.col("intime"))
                .truediv(pl.duration(seconds=1))
                .alias("Ventilation End Relative to Admission (seconds)"),
                (pl.col("endtime") - pl.col("starttime"))
                .truediv(pl.duration(hours=1))
                .alias("Ventilation Duration (hours)"),
            )
            .drop("intime", "starttime", "endtime", "itemid")
            .pipe(self._add_global_id_stay_id, "mimic4-", "stay_id")
        )
        # endregion

        # region SICdb
        # print("MAGIC_CONCEPTS: Ventilation Duration - SICdb")
        sicdb_ADMISSION_TIMES = pl.scan_csv(self.sicdb_paths.cases_path).select(
            "CaseID", "ICUOffset"
        )

        sicdb_VENTILATION_DURATION = (
            pl.scan_csv(self.sicdb_paths.data_range_path)
            .join(sicdb_ADMISSION_TIMES, on="CaseID", how="left")
            # Filter for ventilation IDs
            .filter(pl.col("DataID").is_in([720, 3043]))
            .with_columns(
                pl.col("DataID")
                .cast(str)
                .replace(
                    {"720": "invasive ventilation", "3043": "tracheostomy"}
                )
                .alias("Ventilation Type"),
                pl.col("Offset").alias(
                    "Ventilation Start Relative to Admission (seconds)"
                ),
                pl.col("OffsetEnd").alias(
                    "Ventilation End Relative to Admission (seconds)"
                ),
                ((pl.col("OffsetEnd") - pl.col("Offset")) / (60 * 60)).alias(
                    "Ventilation Duration (hours)"
                ),
            )
            .drop("DataID", "Offset", "OffsetEnd")
            .pipe(self._add_global_id_stay_id, "sicdb-", "CaseID")
        )

        # endregion

        # region UMCdb
        # print("MAGIC_CONCEPTS: Ventilation Duration - UMCdb")

        umcdb_ADMISSION_TIMES = pl.scan_parquet(
            self.umcdb_paths.admissions_path
        ).select("admissionid", "admittedat")

        umcdb_VENTILATION_DURATION = (
            pl.scan_parquet(self.umcdb_paths.processitems_path)
            .join(umcdb_ADMISSION_TIMES, on="admissionid", how="left")
            # Filter for ventilation IDs
            .filter(
                pl.col("itemid").is_in(
                    self.ricu_mappings.ricu_concept_dict["mech_vent"][
                        "sources"
                    ]["aumc"][0]["ids"]
                )
            )
            .drop("itemid")
            # replace ventilation concepts
            .with_columns(
                pl.col("item")
                .replace(
                    {
                        "Beademen": "invasive ventilation",
                        "Beademen non-invasief": "non-invasive ventilation",
                        "Tracheostoma": "tracheostomy",
                    }
                )
                .cast(str)
                .alias("item")
            )
            # Make datetime relative to admission in seconds
            .with_columns(
                pl.duration(
                    milliseconds=(pl.col("start") - pl.col("admittedat"))
                )
                .dt.total_seconds()
                .alias("start"),
                pl.duration(
                    milliseconds=(pl.col("stop") - pl.col("admittedat"))
                )
                .dt.total_seconds()
                .alias("stop"),
                pl.duration(milliseconds=pl.col("stop") - pl.col("start"))
                .truediv(pl.duration(hours=1))
                .alias("duration"),
            )
            .drop("admittedat")
            # Rename columns
            .rename(
                {
                    "item": "Ventilation Type",
                    "start": "Ventilation Start Relative to Admission (seconds)",
                    "stop": "Ventilation End Relative to Admission (seconds)",
                    "duration": "Ventilation Duration (hours)",
                }
            )
            .pipe(self._add_global_id_stay_id, "umcdb-", "admissionid")
        )
        # endregion

        # region ALL
        print("MAGIC_CONCEPTS: Ventilation Duration")

        VENTILATION_DURATION = (
            pl.concat(
                [
                    eicu_VENTILATION_DURATION,
                    hirid_VENTILATION_DURATION,
                    mimic3_VENTILATION_DURATION,
                    mimic4_VENTILATION_DURATION,
                    sicdb_VENTILATION_DURATION,
                    umcdb_VENTILATION_DURATION,
                ],
                how="diagonal_relaxed",
            )
            .filter(pl.col("Ventilation Duration (hours)").ne_missing(0))
            .unique()
            .select(
                "Global ICU Stay ID",
                "Ventilation Type",
                "Ventilation Start Relative to Admission (seconds)",
                "Ventilation End Relative to Admission (seconds)",
                "Ventilation Duration (hours)",
            )
            .with_columns(pl.col("Ventilation Duration (hours)").round(2))
        )
        # endregion

        return VENTILATION_DURATION

    # region helpers
    def _add_global_id_stay_id(self, data, source_dataset, stay_id_col):
        return data.with_columns(
            # add global ICU stay ID
            pl.concat_str([pl.lit(source_dataset), pl.col(stay_id_col)]).alias(
                self.column_names["global_icu_stay_id_col"]
            )
        ).drop(stay_id_col)

    # endregion
