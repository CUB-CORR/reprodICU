# Description: This file contains helper functions for creating an overview of the data.

import polars as pl


class Overview:
    def __init__(self, save_path: str):
        self.save_path = save_path

    # region overview
    def create_overview(self) -> None:
        """Create an overview of the data extracted and harmonized."""
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
        """Create an overview of the data extracted and harmonized."""
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
