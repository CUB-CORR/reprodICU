# Author: Finn Fassbender
# Last modified: 2025-10-30

# Description: This module extracts MAGIC CONCEPTS directly from source datasets.
# The MAGIC CONCEPTS are a set of concepts based on the ricu R package and prewritten code snippets.
# All functions are designed to be called programmatically from Python code.

from typing import Dict, List, Optional

import yaml

from ..config import get_config_manager, reprodICUPaths
from .Y_MAGIC_CONCEPTS.MAGIC_CONCEPTS_REPOSITORY import (
    MAGIC_CONCEPTS_REPOSITORY,
)


def load_mapping(path: str) -> dict:
    """
    Load YAML mapping configuration file.

    Steps:
        1. Open file at path.
        2. Parse YAML content.

    Returns:
        dict: Parsed YAML configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


# region helpers
def _normalize_datasets(
    datasets: Optional[List[str]], demo: bool = False
) -> List[str]:
    """
    Normalize and expand dataset selection specification.

    Steps:
        1. If datasets is None or contains "all": expand to all 6 databases.
        2. If demo=True: restrict to demo-compatible databases (eICU, MIMIC3, MIMIC4).
        3. Otherwise return specified datasets unchanged.

    Returns:
        List[str]: Normalized dataset names (e.g., ["MIMIC3", "MIMIC4", ...]).
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
    """
    Normalize and expand magic concept selection specification.

    Steps:
        1. If concepts is None or contains "all": expand to all 5 standard concepts.
        2. Otherwise return specified concepts unchanged.

    Returns:
        List[str]: Normalized concept names (CODE_STATUS, RECEIVED_ANY_ANTIBIOTICS, RENAL_REPLACEMENT_THERAPY_DURATION, SEVERITY_SCORES, VENTILATION_DURATION).
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


# region build
def build_magic_concepts(
    paths=None,
    datasets: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
    demo: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract and write MAGIC CONCEPTS for specified databases and concept types.

    Steps:
        1. Load config/paths if not provided.
        2. Normalize dataset and concept selections (expand "all" to full lists).
        3. Initialize MAGIC_CONCEPTS_REPOSITORY with normalized datasets.
        4. Validate all requested concepts exist in repository.
        5. For each concept: call get_magic_concept, collect LazyFrame, write to parquet.
        6. Return dict mapping concept names to output file paths.

    Returns:
        Dict[str, List[str]]: Mapping {concept_name: [output_parquet_path]}.

    Raises:
        ValueError: If requested concept not in repository.
        FileNotFoundError: If output directory does not exist.
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
