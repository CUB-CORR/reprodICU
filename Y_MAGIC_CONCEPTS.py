# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPTS directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import argparse
import polars as pl
import yaml

from helpers.MAGIC_CONCEPTS._MAGIC_CONCEPTS import MAGIC_CONCEPTS
from helpers.MAGIC_CONCEPTS.RECEIVED_ANY_ANTIBIOTICS import (
    RECEIVED_ANY_ANTIBIOTICS,
)
from helpers.MAGIC_CONCEPTS.VENTILATION_DURATION import VENTILATION_DURATION


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class reprodICUPaths:
    def __init__(self) -> None:
        config = load_mapping("configs/paths_local.yaml")
        for key, value in config.items():
            setattr(self, key, str(value))


# region main
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Select datasets to extract.")
    parser.add_argument(
        "-d",
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="Datasets to extract.",
    )
    parser.add_argument(
        "-c",
        "--concepts",
        type=str,
        nargs="+",
        default=["all"],
        help="MAGIC CONCEPTS to extract.",
    )
    args = parser.parse_args()

    # Select datasets to extract
    if "all" in args.datasets:
        datasets = ["eICU", "HiRID", "MIMIC3", "MIMIC4", "SICdb", "UMCdb"]
    else:
        datasets = args.datasets

    # Select concepts to extract
    if "all" in args.concepts:
        concepts = ["RECEIVED_ANY_ANTIBIOTICS", "VENTILATION_DURATION"]
    else:
        concepts = args.concepts

    # Initialize paths
    paths = reprodICUPaths()
    column_names = load_mapping("configs/COLUMN_NAMES.yaml")
    _MAGIC_CONCEPTS = MAGIC_CONCEPTS(paths, datasets)
    MAGIC_CONCEPTS_PATH = paths.reprodICU_files_path + "MAGIC_CONCEPTS/"

    # Extract concepts
    if "RECEIVED_ANY_ANTIBIOTICS" in concepts:
        concept_instance = RECEIVED_ANY_ANTIBIOTICS(paths, datasets)
        concept_instance.RECEIVED_ANY_ANTIBIOTICS().collect(
            streaming=True
        ).write_parquet(
            MAGIC_CONCEPTS_PATH + "RECEIVED_ANY_ANTIBIOTICS.parquet"
        )

    if "VENTILATION_DURATION" in concepts:
        concept_instance = VENTILATION_DURATION(paths, datasets)
        concept_instance.VENTILATION_DURATION().collect(
            streaming=True
        ).write_parquet(MAGIC_CONCEPTS_PATH + "VENTILATION_DURATION.parquet")

    else:
        raise ValueError(f"reprodICU - No concept found for {concepts}.")
