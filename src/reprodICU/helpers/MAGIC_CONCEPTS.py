# Author: Finn Fassbender
# Last modified: 2024-10-28

# Description: This module extracts MAGIC CONCEPTS directly from source datasets.
# The MAGIC CONCEPTS are a set of concepts based on the ricu R package and prewritten code snippets.
# All functions are designed to be called programmatically from Python code.

from typing import Dict, List, Optional

import yaml
from config import get_config_manager, reprodICUPaths
from Y_MAGIC_CONCEPTS.MAGIC_CONCEPTS_REPOSITORY import MAGIC_CONCEPTS_REPOSITORY


def load_mapping(path: str) -> dict:
    """Load YAML mapping file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Helper functions for parameter processing


def _normalize_datasets(
    datasets: Optional[List[str]], demo: bool = False
) -> List[str]:
    """Normalize dataset selection.

    Args:
        datasets: None for "all", or list of specific datasets
        demo: If True, restrict to demo-compatible datasets (not used for concepts)

    Returns:
        List of dataset names to process
    """
    if datasets is None or (isinstance(datasets, list) and "all" in datasets):
        all_datasets = [
            "eICU",
            "HiRID",
            "MIMIC3",
            "MIMIC4",
            "SICdb",
            "UMCdb",
        ]
        if demo:
            return ["eICU", "MIMIC3", "MIMIC4"]
        return all_datasets
    return datasets


def _normalize_concepts(concepts: Optional[List[str]]) -> List[str]:
    """Normalize concept selection.

    Args:
        concepts: None for "all", or list of specific concepts

    Returns:
        List of concept names to extract
    """
    if concepts is None or (isinstance(concepts, list) and "all" in concepts):
        return [
            "CODE_STATUS",
            "RECEIVED_ANY_ANTIBIOTICS",
            "RENAL_REPLACEMENT_THERAPY_DURATION",
            "SEVERITY_SCORES",
            "VENTILATION_DURATION",
        ]
    return concepts


# Building functions


def build_magic_concepts(
    paths=None,
    datasets: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
    demo: bool = False,
) -> Dict[str, List[str]]:
    """Build MAGIC CONCEPTS for specified datasets and concepts.

    Args:
        paths: Optional paths object (uses ConfigManager if None)
        datasets: Datasets to process (uses "all" if None)
        concepts: Concepts to extract (uses "all" if None)
        demo: Use demo data if True

    Returns:
        Dict mapping concept names to output file paths

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If invalid concept selection
        RuntimeError: If processing fails
    """
    if paths is None:
        config_manager = get_config_manager()
        paths = reprodICUPaths(config_manager)

    DATASETS = _normalize_datasets(datasets, demo=demo)
    CONCEPTS = _normalize_concepts(concepts)

    # Initialize MAGIC CONCEPTS repository
    MAGIC_CONCEPTS = MAGIC_CONCEPTS_REPOSITORY(paths, DATASETS)
    MAGIC_CONCEPTS_PATH = paths.reprodICU_files_path + "MAGIC_CONCEPTS/"

    # Validate concepts exist
    for concept in CONCEPTS:
        if concept not in MAGIC_CONCEPTS.magic_concepts_dict:
            raise ValueError(
                f"reprodICU - No concept found for {concept}. "
                f"Available concepts: {list(MAGIC_CONCEPTS.magic_concepts_dict.keys())}"
            )

    # Extract concepts
    output_files = {}
    for concept in CONCEPTS:
        print(f"reprodICU - Extracting MAGIC CONCEPT: {concept}...")
        output_path = MAGIC_CONCEPTS_PATH + f"{concept}.parquet"
        MAGIC_CONCEPTS.get_magic_concept(concept).collect().write_parquet(
            output_path
        )
        output_files[concept] = [output_path]

    print("reprodICU - Done extracting MAGIC CONCEPTS.")
    return output_files
