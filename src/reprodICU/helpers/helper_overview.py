# Description: This file contains helper functions for creating an overview of the data.

import polars as pl


class Overview:
    def __init__(self, save_path: str):
        self.save_path = save_path

    # region overview
    def create_overview(self) -> None:
        """
        Create overview of extracted and harmonized ICU data by ICU stay.

        Steps:
            1. Load patient_information parquet to get global ICU stay IDs and source datasets.
            2. Load each harmonized data table (diagnoses, medications, timeseries variants).
            3. For each table: group by ICU stay ID and count rows.
            4. Join all counts to patient information (left join to preserve all stays).
            5. Write result to overview.parquet file.

        Returns:
            None: Writes to {save_path}/overview.parquet.
        """
        # Create DataFrame to store the overview, initialize columns for each dataset
        overview = pl.scan_parquet(
            self.save_path + "patient_information.parquet"
        ).select("Global ICU Stay ID", "Source Dataset")

        # Add columns for each table
        tables = [
            "diagnoses_imputed",
            # "procedures",
            "medications",
            "timeseries_vitals",
            "timeseries_labs",
            "timeseries_respiratory",
            "timeseries_intakeoutput",
        ]

        for table in tables:
            # print(f"Adding {table} to overview...")
            overview = overview.join(
                pl.scan_parquet(self.save_path + table + ".parquet")
                .select("Global ICU Stay ID")
                .group_by("Global ICU Stay ID")
                .len()
                .rename({"len": table}),
                on="Global ICU Stay ID",
                how="left",
            )

        # Save the overview to a parquet file
        overview.sink_parquet(self.save_path + "overview.parquet")

    # endregion

    # region overview vars
    def create_database_variable_overview(self) -> None:
        """
        Create overview of data variables aggregated by source database.

        Steps:
            1. Load patient_information to map ICU stays to databases.
            2. Group by source dataset and count ICU stays.
            3. Load each timeseries table (vitals, labs, respiratory, intakeoutput).
            4. For each table: join with patient info, sum numeric columns by dataset.
            5. Transpose result to get variables as rows with datasets as columns.
            6. Write result to overview_database_variable.parquet file.

        Returns:
            None: Writes to {save_path}/overview_database_variable.parquet.
        """
        # Create DataFrame to store the overview, initialize columns for each dataset
        ID_TO_DB = pl.scan_parquet(
            self.save_path + "patient_information.parquet"
        ).select("Global ICU Stay ID", "Source Dataset")
        overview = (
            ID_TO_DB.group_by("Source Dataset")
            .len()
            .rename({"len": "Case Count"})
        )

        # Add columns for each table
        tables = [
            "timeseries_vitals",
            "timeseries_labs",
            "timeseries_respiratory",
            "timeseries_intakeoutput",
        ]

        for table in tables:
            print(f"Adding {table} to overview...")
            overview = (
                overview.join(
                    pl.scan_parquet(self.save_path + table + ".parquet")
                    .join(ID_TO_DB, on="Global ICU Stay ID", how="left")
                    .fill_null(0)
                    .group_by("Source Dataset")
                    .sum()
                    .drop(
                        "Global ICU Stay ID",
                        "Time Relative to Admission (seconds)",
                    ),
                    on="Source Dataset",
                    how="left",
                )
                .collect()
                .lazy()
            )

        # Save the overview to a parquet file
        overview = overview.collect().transpose(include_header=True)
        overview = (
            overview.rename(overview.head(1).to_dicts().pop())
            .with_row_index()
            .filter(pl.col("index") != 0)
        )
        overview.write_parquet(
            self.save_path + "overview_database_variable.parquet"
        )

    # endregion
