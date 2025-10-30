__version__ = "0.0.0.1"
__author__ = "Finn Fassbender"
__copyright__ = "2024, Institute for Medical Informatics, Charité - Universitätsmedizin Berlin"
__maintainer__ = "Finn Fassbender"
__email__ = "finn.fassbender@charite.de"
__status__ = "Production"

# Description: reprodICU package initialization.
# Exports public API for building data, extracting concepts, converting formats,
# and implements lazy dataset loading via __getattr__.

import os
import sys
from pathlib import Path
from typing import Any

# Add package directory to path for relative imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from config import (
    ConfigManager,
    DatasetLoader,
    get_config_manager,
    reprodICUPaths,
)

# Import submodule namespaces - don't import contents directly
from . import interfaces, utils


# Create namespace aliases for cleaner API
class _BuildNamespace:
    """Namespace for build functions: reprodICU.build.FUNCTION"""

    def __getattr__(self, name: str) -> Any:
        """Dynamically load build functions without method binding issues."""
        from reprodICU import reprodICU
        from helpers import MAGIC_CONCEPTS

        build_functions = {
            "build_all": reprodICU.build_all,
            "build_diagnoses": reprodICU.build_diagnoses,
            "build_medications": reprodICU.build_medications,
            "build_microbiology": reprodICU.build_microbiology,
            "build_notes": reprodICU.build_notes,
            "build_patient_information": reprodICU.build_patient_information,
            "build_procedures": reprodICU.build_procedures,
            "build_timeseries": reprodICU.build_timeseries,
            "build_magic_concepts": MAGIC_CONCEPTS.build_magic_concepts,
        }

        if name in build_functions:
            return build_functions[name]

        raise AttributeError(f"'_BuildNamespace' object has no attribute '{name}'")


class _ConvertNamespace:
    """Namespace for conversion functions: reprodICU.convert.convert_to_X"""

    def __getattr__(self, name: str) -> Any:
        """Dynamically load conversion functions without method binding issues."""
        from interfaces import convert_to_clif, convert_to_meds, convert_to_omop

        convert_functions = {
            "convert_to_clif": convert_to_clif,
            "convert_to_meds": convert_to_meds,
            "convert_to_omop": convert_to_omop,
        }

        if name in convert_functions:
            return convert_functions[name]

        raise AttributeError(f"'_ConvertNamespace' object has no attribute '{name}'")


# Expose namespaces
build = _BuildNamespace()
convert = _ConvertNamespace()

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
        # Submodules (namespaced access)
        "utils",  # utils.scores.SOFA, utils.clinical.URINE_OUTPUT, etc.
        "interfaces",  # interfaces.convert_to_omop, interfaces.convert_to_clif, etc.
        "build",  # build.build_patient_information, build.build_all, etc.
        "convert",  # convert.convert_to_omop, convert.convert_to_clif, etc.
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
    # Submodules
    "utils",
    "interfaces",
    "build",
    "convert",
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
