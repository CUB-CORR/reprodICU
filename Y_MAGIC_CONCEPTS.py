# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script extracts the so called MAGIC CONCEPTS directly from the source datasets.
# The MAGIC CONCEPTS are a set of concepts that are based on the concept dict used in the ricu R package and/or
# available prewritten code snippets where indicated.

import argparse

import yaml
from helpers.MAGIC_CONCEPTS.MAGIC_CONCEPTS_REPOSITORY import \
    MAGIC_CONCEPTS_REPOSITORY


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
        concepts = [
            "CODE_STATUS",
            "RECEIVED_ANY_ANTIBIOTICS",
            "RENAL_REPLACEMENT_THERAPY_DURATION",
            "SEVERITY_SCORES",
            "VENTILATION_DURATION",
        ]
    else:
        concepts = args.concepts

    # Initialize paths
    paths = reprodICUPaths()
    column_names = load_mapping("configs/COLUMN_NAMES.yaml")
    MAGIC_CONCEPTS = MAGIC_CONCEPTS_REPOSITORY(paths, datasets)
    MAGIC_CONCEPTS_PATH = paths.reprodICU_files_path + "MAGIC_CONCEPTS/"

    # Assert concepts exist
    for concept in concepts:
        if concept not in MAGIC_CONCEPTS.magic_concepts_dict:
            raise ValueError(f"reprodICU - No concept found for {concept}.")

    # Extract concepts
    for concept in concepts:
        MAGIC_CONCEPTS.get_magic_concept(concept).collect().write_parquet(
            MAGIC_CONCEPTS_PATH + f"{concept}.parquet"
        )
