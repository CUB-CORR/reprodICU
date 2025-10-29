# Author: Finn Fassbender
# Last modified: 2024-10-28

# Description: reprodICU package initialization.
# Exports public API for building data, extracting concepts, converting formats,
# and implements lazy dataset loading via __getattr__.

import os
import sys
from pathlib import Path
from typing import Any

# Add package directory to path for relative imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

__version__ = "0.1.0"
__author__ = "Institute of Medical Informatics, Charité"

from config import (
    ConfigManager,
    DatasetLoader,
    get_config_manager,
    reprodICUPaths,
)
from helpers.MAGIC_CONCEPTS import build_magic_concepts
from interfaces import convert_to_clif, convert_to_meds, convert_to_omop
from utils import (
    # Core clinical utilities
    SEPSIS,
    SOFA,
    VIS,
    # Data processing utilities
    URINE_OUTPUT,
    FIX_WINDOW_BORDERS,
    # Dataset helpers
    get_patient_information,
    get_timeseries_vitals,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_intakeoutput,
    get_medications,
    get_microbiology,
)

from reprodICU.reprodICU import (
    build_all,
    build_diagnoses,
    build_medications,
    build_microbiology,
    build_notes,
    build_patient_information,
    build_procedures,
    build_timeseries,
)

# Create global dataset loader instance
_dataset_loader: DatasetLoader = None


def _get_dataset_loader() -> DatasetLoader:
    """Get or create the global dataset loader."""
    global _dataset_loader
    if _dataset_loader is None:
        config_manager = get_config_manager()
        _dataset_loader = DatasetLoader(config_manager)
    return _dataset_loader


def __getattr__(name: str) -> Any:
    """
    Module-level attribute access for lazy-loading datasets.

    Allows: reprodICU.timeseries_vitals, reprodICU.diagnoses, etc.
    """
    # Handle dataset access
    loader = _get_dataset_loader()

    if name in loader.DATASET_MAPPING:
        return loader.load_dataset(name)
    
    if name in loader.CONCEPT_MAPPING:
        return loader.load_concept(name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list:
    """Show available attributes in dir(reprodICU)."""
    loader = _get_dataset_loader()
    standard_attrs = [
        # Package metadata
        "__version__",
        "__author__",
        # Configuration and paths
        "get_config_manager",
        "ConfigManager",
        "DatasetLoader",
        "reprodICUPaths",
        # Building functions
        "build_patient_information",
        "build_diagnoses",
        "build_procedures",
        "build_medications",
        "build_microbiology",
        "build_notes",
        "build_timeseries",
        "build_all",
        # Magic concepts
        "build_magic_concepts",
        # Conversion functions
        "convert_to_omop",
        "convert_to_clif",
        "convert_to_meds",
        # Clinical utilities - core (require all data arguments)
        "SEPSIS",
        "SOFA",
        "VIS",
        "URINE_OUTPUT",
        "FIX_WINDOW_BORDERS",
        # Dataset loading helpers
        "get_patient_information",
        "get_timeseries_vitals",
        "get_timeseries_labs",
        "get_timeseries_respiratory",
        "get_timeseries_intakeoutput",
        "get_medications",
        "get_diagnoses",
        "get_procedures",
        "get_notes",
        "get_microbiology",
        # Helper functions
        "available_datasets",
        "dataset_exists",
        "get_dataset_path",
        "available_concepts",
        "concept_exists",
        "get_concept_path",
        "use_demo_mode",
        "use_full_mode",
        "reload_dataset",
        "clear_cache",
    ]
    return standard_attrs + list(loader.DATASET_MAPPING.keys())


# region Public API functions
def available_datasets() -> list:
    """List all available datasets (ones that exist)."""
    return _get_dataset_loader().available_datasets()


def dataset_exists(dataset_name: str) -> bool:
    """Check if a dataset exists."""
    return _get_dataset_loader().dataset_exists(dataset_name)


def get_dataset_path(dataset_name: str) -> Path:
    """Get the full path to a dataset file."""
    return _get_dataset_loader().get_dataset_path(dataset_name)


def available_concepts() -> list:
    """List all available concepts (ones that exist)."""
    return _get_dataset_loader().available_concepts()


def concept_exists(concept_name: str) -> bool:
    """Check if a concept exists."""
    return _get_dataset_loader().concept_exists(concept_name)


def get_concept_path(concept_name: str) -> Path:
    """Get the full path to a concept file."""
    return _get_dataset_loader().get_concept_path(concept_name)


def use_demo_mode() -> None:
    """Switch to demo dataset mode."""
    _get_dataset_loader().set_demo_mode(True)


def use_full_mode() -> None:
    """Switch to full dataset mode."""
    _get_dataset_loader().set_demo_mode(False)


def reload_dataset(dataset_name: str):
    """Reload a dataset, clearing the cache."""
    return _get_dataset_loader().reload_dataset(dataset_name)


def clear_cache() -> None:
    """Clear all cached datasets."""
    _get_dataset_loader().clear_cache()


# endregion

__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    # Configuration and paths
    "get_config_manager",
    "ConfigManager",
    "DatasetLoader",
    "reprodICUPaths",
    # Building functions
    "build_patient_information",
    "build_diagnoses",
    "build_procedures",
    "build_medications",
    "build_microbiology",
    "build_notes",
    "build_timeseries",
    "build_all",
    # Magic concepts
    "build_magic_concepts",
    # Conversion functions
    "convert_to_omop",
    "convert_to_clif",
    "convert_to_meds",
    # Clinical scoring utilities - core (require all data arguments)
    "SEPSIS",
    "SOFA",
    "VIS",
    # Data processing utilities
    "URINE_OUTPUT",
    "FIX_WINDOW_BORDERS",
    # Dataset loading helpers
    "get_patient_information",
    "get_timeseries_vitals",
    "get_timeseries_labs",
    "get_timeseries_respiratory",
    "get_timeseries_intakeoutput",
    "get_medications",
    "get_diagnoses",
    "get_procedures",
    "get_notes",
    "get_microbiology",
    # Helper functions
    "available_datasets",
    "dataset_exists",
    "get_dataset_path",
    "available_concepts",
    "concept_exists",
    "get_concept_path",
    "use_demo_mode",
    "use_full_mode",
    "reload_dataset",
    "clear_cache",
]
