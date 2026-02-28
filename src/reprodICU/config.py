# Author: Finn Fassbender
# Last modified: 2025-10-30

"""
Configuration management for reprodICU package.

Handles loading of YAML configuration files, user-editable paths, and lazy
loading of datasets and concepts from parquet files. Provides ConfigManager
for configuration management, DatasetLoader for lazy dataset access, and
reprodICUPaths for convenient access to configured directory paths.
"""

import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
import yaml


# region ConfigManager
class ConfigManager:
    """
    Manages reprodICU configuration with user overrides.

    Handles loading of YAML configuration files from both package defaults
    and user-editable configurations in ~/.reprodICU/. Supports caching
    for performance and atomic updates to configuration files.
    """

    PACKAGE_NAME = "reprodICU"
    CONFIG_DIR_NAME = ".reprodICU"

    def __init__(self):
        # Configs are in the package root's 'configs' directory
        self.package_config_dir = Path(
            str(files(self.PACKAGE_NAME).joinpath("configs"))
        )
        self.user_config_dir = Path.home() / self.CONFIG_DIR_NAME
        self._cached_configs: Dict[str, Any] = {}
        self._ensure_user_config_dir()

    def _ensure_user_config_dir(self) -> None:
        """
        Create user config directory and copy templates if needed.

        Creates ~/.reprodICU/ directory and copies PATHS.yaml.template
        to PATHS.yaml if it doesn't exist, allowing users to customize paths.
        """
        self.user_config_dir.mkdir(parents=True, exist_ok=True)

        # Copy PATHS.yaml template if it doesn't exist
        template_path = self.package_config_dir / "PATHS.yaml.template"
        user_paths_file = self.user_config_dir / "PATHS.yaml"

        if not user_paths_file.exists() and template_path.exists():
            shutil.copy(str(template_path), str(user_paths_file))
            print(f"!! Created user config at: {user_paths_file}")
            print("!! Please edit this file with your local paths.")

    def get_config_path(
        self, config_name: str, user_override: bool = True
    ) -> Path:
        """
        Get path to a config file.

        Arguments
        ---------
            config_name : str
                Name of config file (e.g., 'COLUMN_NAMES.yaml', 'PATHS.yaml')
            user_override : bool
                If True, prefer user config over package default

        Returns
        -------
            Path
                Path to the config file

        Raises
        ------
            FileNotFoundError
                If config file not found in user or package directories
        """
        user_path = self.user_config_dir / config_name
        package_path = self.package_config_dir / config_name

        # For PATHS.yaml, prefer user version (they should edit it)
        if user_override and user_path.exists():
            return user_path
        elif package_path.exists():
            return package_path
        else:
            raise FileNotFoundError(
                f"Config file '{config_name}' not found in "
                f"user ({self.user_config_dir}) or package ({self.package_config_dir})"
            )

    def load_config(
        self, config_name: str, user_override: bool = True
    ) -> Dict[str, Any]:
        """
        Load YAML config file.

        Loads configuration from user or package directory, with results
        cached for performance. Uses user configuration by default if it exists.

        Arguments
        ---------
            config_name : str
                Name of config file to load
            user_override : bool
                If True, prefer user config over package default

        Returns
        -------
            dict
                Parsed YAML configuration

        Raises
        ------
            FileNotFoundError
                If config file not found
        """
        if config_name in self._cached_configs:
            return self._cached_configs[config_name]

        config_path = self.get_config_path(
            config_name, user_override=user_override
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self._cached_configs[config_name] = config
        return config

    def update_config(
        self,
        config_name: str,
        updates: Dict[str, Any],
        user_override: bool = True,
    ) -> None:
        """
        Update configuration values and save to file.

        Updates configuration values atomically and clears cache to ensure
        subsequent loads reflect the new values.

        Arguments
        ---------
            config_name : str
                Name of config file (e.g., 'PATHS.yaml')
            updates : dict
                Dictionary of key-value pairs to update
            user_override : bool
                If True, save to user config, otherwise package config

        Raises
        ------
            FileNotFoundError
                If config file not found
        """
        config_path = self.get_config_path(
            config_name, user_override=user_override
        )

        # Load current config
        config = self.load_config(config_name, user_override=user_override)

        # Update with new values
        config.update(updates)

        # Write back to file
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Clear cache to force reload on next access
        self._cached_configs.pop(config_name, None)

    def get_user_config_dir(self) -> Path:
        """
        Return the user config directory.

        Returns
        -------
            Path
                Path to ~/.reprodICU/ directory
        """
        return self.user_config_dir

    def print_config_info(self) -> None:
        """
        Print information about config locations.

        Displays package and user configuration directory paths for debugging.
        """
        print(f"Package config directory: {self.package_config_dir}")
        print(f"User config directory: {self.user_config_dir}")

    def get_clinically_plausible_values(self) -> Dict[str, Any]:
        """
        Load clinically plausible value ranges from package config.

        Returns predefined min/max ranges for clinical measurements
        used for data validation and outlier detection.

        Returns
        -------
            dict
                Dictionary of measurement names with min/max value ranges
        """
        return self.load_config(
            "CLINICALLY_PLAUSIBLE_VALUES.yaml", user_override=False
        )


# endregion


# region DatasetLoader
class DatasetLoader:
    """
    Lazy-loads parquet datasets and concepts from configured paths.

    Provides efficient lazy access to clinical datasets and pre-computed
    concepts using Polars LazyFrame. Supports demo/full mode switching and
    automatic caching for performance.
    """

    # Map attribute names to parquet filenames
    DATASET_MAPPING = {
        # Patient information
        "info": "patient_information.parquet",
        "patient_information": "patient_information.parquet",
        "patient_information_imputed": "patient_information_imputed.parquet",
        # Clinical data
        "diagnoses": "diagnoses.parquet",
        "procedures": "procedures.parquet",
        "medications": "medications.parquet",
        "prescriptions": "medications_prescribed.parquet",
        "medications_prescribed": "medications_prescribed.parquet",
        "microbiology": "microbiology.parquet",
        "notes": "notes.parquet",
        # Timeseries - vitals
        "vitals": "timeseries_vitals.parquet",
        "timeseries_vitals": "timeseries_vitals.parquet",
        "vitals_imputed": "timeseries_vitals_imputed.parquet",
        "timeseries_vitals_imputed": "timeseries_vitals_imputed.parquet",
        "vitals_resampled": "timeseries_vitals_resampled.parquet",
        "timeseries_vitals_resampled": "timeseries_vitals_resampled.parquet",
        # Timeseries - labs
        "labs": "timeseries_labs.parquet",
        "laboratory": "timeseries_labs.parquet",
        "timeseries_labs": "timeseries_labs.parquet",
        "timeseries_labs_winsorized": "timeseries_labs_winsorized.parquet",
        # Timeseries - respiratory
        "resp": "timeseries_respiratory.parquet",
        "respiratory": "timeseries_respiratory.parquet",
        "timeseries_respiratory": "timeseries_respiratory.parquet",
        # Timeseries - intake/output
        "inout": "timeseries_intakeoutput.parquet",
        "intakeoutput": "timeseries_intakeoutput.parquet",
        "timeseries_intakeoutput": "timeseries_intakeoutput.parquet",
    }

    # Map concept names to parquet filenames (pre-computed clinical concepts)
    CONCEPT_MAPPING = {
        # Mechanical ventilation
        "VENT": "VENTILATION_DURATION.parquet",
        "VENTILATION": "VENTILATION_DURATION.parquet",
        "VENTILATION_DURATION": "VENTILATION_DURATION.parquet",
        # Renal replacement therapy
        "RRT": "RENAL_REPLACEMENT_THERAPY_DURATION.parquet",
        "RENAL_REPLACEMENT_THERAPY": "RENAL_REPLACEMENT_THERAPY_DURATION.parquet",
        "RENAL_REPLACEMENT_THERAPY_DURATION": "RENAL_REPLACEMENT_THERAPY_DURATION.parquet",
        # Antibiotic and infection concepts
        "RECEIVED_ANY_ANTIBIOTICS": "RECEIVED_ANY_ANTIBIOTICS.parquet",
        # Severity scores and status
        "SEVERITY_SCORES": "SEVERITY_SCORES.parquet",
        "CODE_STATUS": "CODE_STATUS.parquet",
    } # fmt: skip

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the dataset loader.

        Arguments
        ---------
            config_manager : ConfigManager
                Configuration manager for accessing paths
        """
        self.config_manager = config_manager
        self._lazy_cache: Dict[str, pl.LazyFrame] = {}
        self.demo_mode = False

    def set_demo_mode(self, enabled: bool = True) -> None:
        """
        Switch between demo and full dataset mode.

        Clears cache when switching to ensure fresh data loads.

        Arguments
        ---------
            enabled : bool
                If True, use demo-sized datasets; otherwise full datasets
        """
        self.demo_mode = enabled
        self._lazy_cache.clear()  # Clear cache when switching modes
        mode = "DEMO" if enabled else "FULL"
        print(f"!! Switched to {mode} mode !!")

    def get_data_path(self) -> Path:
        """
        Get the base data path based on current mode.

        Returns
        -------
            Path
                Data directory path

        Raises
        ------
            ValueError
                If data paths not configured in PATHS.yaml
        """
        config = self.config_manager.load_config(
            "PATHS.yaml", user_override=True
        )

        if self.demo_mode:
            base_path = config.get("reprodICU_demo_files_path")
        else:
            base_path = config.get("reprodICU_files_path")

        if not base_path:
            raise ValueError(
                "Data path not configured. "
                "Please set 'reprodICU_files_path' or 'reprodICU_demo_files_path' "
                f"in {self.config_manager.get_user_config_dir() / 'PATHS.yaml'}"
            )

        return Path(base_path)

    def get_concepts_path(self) -> Path:
        """
        Get the base path for pre-computed concepts.

        Returns
        -------
            Path
                MAGIC_CONCEPTS directory path
        """
        base_path = self.get_data_path()
        return Path(base_path / "MAGIC_CONCEPTS")

    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset file exists.

        Arguments
        ---------
            dataset_name : str
                Name of the dataset to check

        Returns
        -------
            bool
                True if dataset exists, False otherwise
        """
        if dataset_name not in self.DATASET_MAPPING:
            return False

        try:
            path = self.get_dataset_path(dataset_name)
            return path.exists()
        except Exception:
            return False

    def get_dataset_path(self, dataset_name: str) -> Path:
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

        Raises
        ------
            ValueError
                If dataset name not found in DATASET_MAPPING
        """
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Available datasets: {', '.join(sorted(self.DATASET_MAPPING.keys()))}"
            )

        filename = self.DATASET_MAPPING[dataset_name]
        return self.get_data_path() / filename

    def available_datasets(self) -> list:
        """
        Get list of available dataset names (ones that actually exist).

        Returns
        -------
            list
                Names of existing datasets
        """
        available = []
        for dataset_name in sorted(self.DATASET_MAPPING.keys()):
            if self.dataset_exists(dataset_name):
                available.append(dataset_name)
        return available

    def concept_exists(self, concept_name: str) -> bool:
        """
        Check if a concept file exists.

        Arguments
        ---------
            concept_name : str
                Name of the concept to check

        Returns
        -------
            bool
                True if concept exists, False otherwise
        """
        if concept_name not in self.CONCEPT_MAPPING:
            return False

        try:
            path = self.get_concept_path(concept_name)
            return path.exists()
        except Exception:
            return False

    def get_concept_path(self, concept_name: str) -> Path:
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

        Raises
        ------
            ValueError
                If concept name not found in CONCEPT_MAPPING
        """
        if concept_name not in self.CONCEPT_MAPPING:
            raise ValueError(
                f"Unknown concept: '{concept_name}'. "
                f"Available concepts: {', '.join(sorted(self.CONCEPT_MAPPING.keys()))}"
            )

        filename = self.CONCEPT_MAPPING[concept_name]
        return self.get_concepts_path() / filename

    def available_concepts(self) -> list:
        """
        Get list of available concept names (ones that actually exist).

        Returns
        -------
            list
                Names of existing concepts
        """
        available = []
        for concept_name in sorted(self.CONCEPT_MAPPING.keys()):
            if self.concept_exists(concept_name):
                available.append(concept_name)
        return available

    def load_dataset(self, dataset_name: str) -> pl.LazyFrame:
        """
        Lazy-load a dataset as a Polars LazyFrame.

        Returns dataset without loading into memory using lazy evaluation.
        Results are cached for performance.

        Arguments
        ---------
            dataset_name : str
                Name of the dataset to load

        Returns
        -------
            pl.LazyFrame
                Scanned parquet file (lazy, not loaded into memory)

        Raises
        ------
            FileNotFoundError
                If dataset file doesn't exist
            ValueError
                If dataset name unknown
        """
        # Return from cache if already loaded
        if dataset_name in self._lazy_cache:
            return self._lazy_cache[dataset_name]

        dataset_path = self.get_dataset_path(dataset_name)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"Dataset not found: {dataset_name}\n"
                f"Expected path: {dataset_path}\n"
                f"\nThis file was not generated. To create it, run:\n"
                f"  reprodicu -t <table_name>\n\n"
                f"Available datasets: {', '.join(self.available_datasets())}\n"
                f"Data location: {self.get_data_path()}\n"
                f"Mode: {'DEMO' if self.demo_mode else 'FULL'}\n"
                f"{'='*70}"
            )

        # Lazy-load and cache
        lazy_frame = pl.scan_parquet(str(dataset_path))
        self._lazy_cache[dataset_name] = lazy_frame

        return lazy_frame

    def reload_dataset(self, dataset_name: str) -> pl.LazyFrame:
        """
        Reload a dataset, clearing the cache.

        Forces a fresh load of the specified dataset, discarding any cached version.

        Arguments
        ---------
            dataset_name : str
                Name of the dataset to reload

        Returns
        -------
            pl.LazyFrame
                Fresh lazy-loaded dataset
        """
        if dataset_name in self._lazy_cache:
            del self._lazy_cache[dataset_name]
        return self.load_dataset(dataset_name)

    def load_concept(self, concept_name: str) -> pl.LazyFrame:
        """
        Lazy-load a concept as a Polars LazyFrame.

        Returns concept without loading into memory using lazy evaluation.
        Results are cached for performance.

        Arguments
        ---------
            concept_name : str
                Name of the concept to load

        Returns
        -------
            pl.LazyFrame
                Scanned parquet file (lazy, not loaded into memory)

        Raises
        ------
            FileNotFoundError
                If concept file doesn't exist
            ValueError
                If concept name unknown
        """
        # Return from cache if already loaded
        if concept_name in self._lazy_cache:
            return self._lazy_cache[concept_name]

        concept_path = self.get_concept_path(concept_name)

        if not concept_path.exists():
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"Concept not found: {concept_name}\n"
                f"Expected path: {concept_path}\n"
                f"\nThis file was not generated. To create it, run:\n"
                f"  reprodicu -c <concept_name>\n\n"
                f"Available concepts: {', '.join(self.available_concepts())}\n"
                f"Data location: {self.get_concepts_path()}\n"
                f"Mode: {'DEMO' if self.demo_mode else 'FULL'}\n"
                f"{'='*70}"
            )

        # Lazy-load and cache
        lazy_frame = pl.scan_parquet(str(concept_path))
        self._lazy_cache[concept_name] = lazy_frame

        return lazy_frame

    def reload_concept(self, concept_name: str) -> pl.LazyFrame:
        """
        Reload a concept, clearing the cache.

        Forces a fresh load of the specified concept, discarding any cached version.

        Arguments
        ---------
            concept_name : str
                Name of the concept to reload

        Returns
        -------
            pl.LazyFrame
                Fresh lazy-loaded concept
        """
        if concept_name in self._lazy_cache:
            del self._lazy_cache[concept_name]
        return self.load_concept(concept_name)

    def clear_cache(self) -> None:
        """
        Clear all cached datasets.

        Removes all cached datasets from memory, forcing fresh loads
        on next access.
        """
        self._lazy_cache.clear()
        print("-> Dataset cache cleared")


# endregion


# region reprodICUPaths
class reprodICUPaths:
    """
    Load and store reprodICU paths from user configuration.

    Loads configured directory paths from PATHS.yaml and provides convenient
    access to all data, output, and configuration directories. Attributes
    are dynamically set from the configuration file.
    """

    def __init__(self, config_manager=None) -> None:
        """
        Initialize paths from user configuration.

        Loads all paths from PATHS.yaml and sets them as instance attributes
        for convenient access throughout the package.

        Arguments
        ---------
            config_manager : ConfigManager, optional
                Configuration manager instance. Uses get_config_manager() if None.

        Raises
        ------
            FileNotFoundError
                If PATHS.yaml configuration is not found
            ValueError
                If required paths are missing from configuration
        """
        if config_manager is None:
            config_manager = get_config_manager()

        config = config_manager.load_config("PATHS.yaml", user_override=True)
        for key, value in config.items():
            setattr(self, key, str(value))

    def validate_paths(self) -> list:
        """
        Validate that all configured paths exist.

        Returns
        -------
            list
                List of tuples (key, path) for paths that don't exist
        """
        missing_paths = []
        for key, value in self.__dict__.items():
            if not Path(value).exists():
                missing_paths.append((key, value))
        return missing_paths


# endregion


# region Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get or create the global config manager.

    Returns
    -------
        ConfigManager
            Singleton configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# endregion
