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
            "diagnoses",
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
                .agg(pl.len().alias(table)),
                on="Global ICU Stay ID",
                how="left",
            )

        # Save the overview to a parquet file, keeping query lazy until sink
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
            4. For each table: iterate through columns and count by dataset using column-wise aggregation.
            5. Save intermediate results to disk between tables to reduce memory pressure.
            6. Transpose result to get variables as rows with datasets as columns.
            7. Write result to overview_database_variable.parquet file.

        Returns:
            None: Writes to {save_path}/overview_database_variable.parquet.
        """
        # Create DataFrame to store the overview, initialize columns for each dataset
        ID_TO_DB = pl.scan_parquet(
            self.save_path + "patient_information.parquet"
        ).select("Global ICU Stay ID", "Source Dataset")
        overview = (
            ID_TO_DB.group_by("Source Dataset")
            .agg(pl.len().alias("Case Count"))
            .collect()
        )

        # Add columns for each table using column-wise counting for performance
        tables = [
            "timeseries_vitals",
            "timeseries_labs",
            "timeseries_respiratory",
            "timeseries_intakeoutput",
        ]

        for table in tables:
            print(f"Adding {table} to overview...")
            table_data = pl.scan_parquet(
                self.save_path + table + ".parquet"
            ).join(ID_TO_DB, on="Global ICU Stay ID", how="left")

            # Get column names, excluding non-data columns
            schema = table_data.collect_schema().names()
            exclude_cols = {
                "Global ICU Stay ID",
                "Time Relative to Admission (seconds)",
                "Source Dataset",
            }
            columns = [col for col in schema if col not in exclude_cols]

            # Collect counts for all columns at once
            _counts = ID_TO_DB.select("Source Dataset").unique().collect()
            for col in columns:
                print(f"  Counting {col}")
                col_counts = (
                    table_data.group_by("Source Dataset")
                    .agg(pl.count(col).alias(col))
                    .collect()
                )
                _counts = _counts.join(
                    col_counts, on="Source Dataset", how="left"
                )

            overview = overview.join(_counts, on="Source Dataset", how="left")

            # Save overview to disk between tables to reduce memory pressure
            overview.write_parquet(
                self.save_path + "overview_database_variable.parquet"
            )
            # Reload from disk to release memory
            overview = pl.read_parquet(
                self.save_path + "overview_database_variable.parquet"
            )

        # Save the overview to a parquet file
        overview = overview.transpose(include_header=True)
        overview = (
            overview.rename(overview.head(1).to_dicts().pop())
            .with_row_index()
            .filter(pl.col("index") != 0)
        )
        overview.write_parquet(
            self.save_path + "overview_database_variable.parquet"
        )

    # endregion
