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
    """
    Namespace for build functions: reprodICU.build.FUNCTION

    Access build functions to construct datasets from raw data sources.
    """

    _FUNCTIONS = {
        "build_all": None,
        "build_diagnoses": None,
        "build_medications": None,
        "build_microbiology": None,
        "build_notes": None,
        "build_patient_information": None,
        "build_procedures": None,
        "build_timeseries": None,
        "build_magic_concepts": None,
    }

    def __dir__(self) -> list:
        """Enable auto-completion for build functions."""
        return sorted(self._FUNCTIONS.keys())

    def __getattr__(self, name: str) -> Any:
        """Dynamically load build functions without method binding issues."""
        if name not in self._FUNCTIONS:
            raise AttributeError(
                f"'_BuildNamespace' object has no attribute '{name}'"
            )

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

        return build_functions[name]


class _ConvertNamespace:
    """
    Namespace for conversion functions: reprodICU.convert.convert_to_X

    Access conversion functions to transform data into standard formats.
    """

    _FUNCTIONS = {
        "convert_to_clif": None,
        "convert_to_meds": None,
        "convert_to_omop": None,
    }

    def __dir__(self) -> list:
        """Enable auto-completion for convert functions."""
        return sorted(self._FUNCTIONS.keys())

    def __getattr__(self, name: str) -> Any:
        """Dynamically load conversion functions without method binding issues."""
        if name not in self._FUNCTIONS:
            raise AttributeError(
                f"'_ConvertNamespace' object has no attribute '{name}'"
            )

        from interfaces import convert_to_clif, convert_to_meds, convert_to_omop

        convert_functions = {
            "convert_to_clif": convert_to_clif,
            "convert_to_meds": convert_to_meds,
            "convert_to_omop": convert_to_omop,
        }

        return convert_functions[name]


class _SetupNamespace:
    """
    Namespace for setup functions: reprodICU.setup.FUNCTION

    Access setup functions to initialize datasets for local development.
    """

    _FUNCTIONS = {
        "setup_mimic3_demo": None,
        "setup_sicdb": None,
        "setup_umcdb": None,
    }

    def __dir__(self) -> list:
        """Enable auto-completion for setup functions."""
        return sorted(self._FUNCTIONS.keys())

    def __getattr__(self, name: str) -> Any:
        """Dynamically load setup functions without method binding issues."""
        if name not in self._FUNCTIONS:
            raise AttributeError(
                f"'_SetupNamespace' object has no attribute '{name}'"
            )

        from setup import setup_mimic3_demo, setup_sicdb, setup_umcdb

        setup_functions = {
            "setup_mimic3_demo": setup_mimic3_demo,
            "setup_sicdb": setup_sicdb,
            "setup_umcdb": setup_umcdb,
        }

        return setup_functions[name]


# Expose namespaces
build = _BuildNamespace()
convert = _ConvertNamespace()
setup = _SetupNamespace()

# Create global dataset loader instance
_dataset_loader: DatasetLoader = None


def _get_dataset_loader() -> DatasetLoader:
    """
    Get or create the global dataset loader.

    Returns
    -------
        DatasetLoader
            Singleton dataset loader instance
    """
    global _dataset_loader
    if _dataset_loader is None:
        config_manager = get_config_manager()
        _dataset_loader = DatasetLoader(config_manager)
    return _dataset_loader


def __getattr__(name: str) -> Any:
    """
    Module-level attribute access for lazy-loading datasets.

    Allows access to datasets and concepts without explicit imports:
        - reprodICU.timeseries_vitals
        - reprodICU.diagnoses
        - reprodICU.patient_information
        - etc.

    Arguments
    ---------
        name : str
            Attribute/dataset name to load

    Returns
    -------
        pl.LazyFrame
            Lazy-loaded dataset
    """
    loader = _get_dataset_loader()

    if name in loader.DATASET_MAPPING:
        return loader.load_dataset(name)

    if name in loader.CONCEPT_MAPPING:
        return loader.load_concept(name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list:
    """
    Show available attributes in dir(reprodICU).

    Enables proper auto-completion and introspection in interactive shells.

    Returns
    -------
        list
            List of all available attributes and functions
    """
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
        "interfaces",  # interfaces.convert_to_omop, interfaces.convert_to_clif
        "build",  # build.build_patient_information, build.build_all
        "convert",  # convert.convert_to_omop, convert.convert_to_clif
        "setup",  # setup.setup_umcdb, setup.setup_sicdb, setup.setup_mimic3_demo
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


# region public API
def available_datasets() -> list:
    """
    List all available datasets (ones that exist).

    Returns
    -------
        list
            Names of available datasets
    """
    return _get_dataset_loader().available_datasets()


def dataset_exists(dataset_name: str) -> bool:
    """
    Check if a dataset exists.

    Arguments
    ---------
        dataset_name : str
            Name of the dataset to check

    Returns
    -------
        bool
            True if dataset exists, False otherwise
    """
    return _get_dataset_loader().dataset_exists(dataset_name)


def get_dataset_path(dataset_name: str) -> Path:
    """
    Get the full path to a dataset file.

    Arguments
    ---------
        dataset_name : str
            Name of the dataset

    Returns
    -------
        Path
            Full path to the dataset file
    """
    return _get_dataset_loader().get_dataset_path(dataset_name)


def available_concepts() -> list:
    """
    List all available concepts (ones that exist).

    Returns
    -------
        list
            Names of available concepts
    """
    return _get_dataset_loader().available_concepts()


def concept_exists(concept_name: str) -> bool:
    """
    Check if a concept exists.

    Arguments
    ---------
        concept_name : str
            Name of the concept to check

    Returns
    -------
        bool
            True if concept exists, False otherwise
    """
    return _get_dataset_loader().concept_exists(concept_name)


def get_concept_path(concept_name: str) -> Path:
    """
    Get the full path to a concept file.

    Arguments
    ---------
        concept_name : str
            Name of the concept

    Returns
    -------
        Path
            Full path to the concept file
    """
    return _get_dataset_loader().get_concept_path(concept_name)


def use_demo_mode() -> None:
    """
    Switch to demo dataset mode.

    Configures the dataset loader to use demo-sized datasets
    instead of full production datasets.
    """
    _get_dataset_loader().set_demo_mode(True)


def use_full_mode() -> None:
    """
    Switch to full dataset mode.

    Configures the dataset loader to use full production datasets
    instead of demo-sized datasets.
    """
    _get_dataset_loader().set_demo_mode(False)


def reload_dataset(dataset_name: str):
    """
    Reload a dataset, clearing the cache.

    Forces a fresh load of the specified dataset, discarding any
    cached version.

    Arguments
    ---------
        dataset_name : str
            Name of the dataset to reload

    Returns
    -------
        pl.LazyFrame
            Fresh lazy-loaded dataset
    """
    return _get_dataset_loader().reload_dataset(dataset_name)


def clear_cache() -> None:
    """
    Clear all cached datasets.

    Removes all cached datasets from memory, forcing fresh loads
    on next access.
    """
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
    "setup",
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
