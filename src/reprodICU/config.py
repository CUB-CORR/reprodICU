# Author: Finn Fassbender
# Last modified: 2024-10-28

# Description: Configuration management for reprodICU package.
# Handles loading of config files, user-editable paths, and lazy loading of datasets.

import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl
import yaml


# region ConfigManager
class ConfigManager:
    """Manages reprodICU configuration with user overrides."""

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
        """Create user config directory and copy templates if needed."""
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

        Args:
            config_name: Name of config file (e.g., 'COLUMN_NAMES.yaml', 'PATHS.yaml')
            user_override: If True, prefer user config over package default

        Returns:
            Path to the config file
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
        """Load YAML config file."""
        if config_name in self._cached_configs:
            return self._cached_configs[config_name]

        config_path = self.get_config_path(
            config_name, user_override=user_override
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self._cached_configs[config_name] = config
        return config

    def get_user_config_dir(self) -> Path:
        """Return the user config directory."""
        return self.user_config_dir

    def print_config_info(self) -> None:
        """Print information about config locations."""
        print(f"Package config directory: {self.package_config_dir}")
        print(f"User config directory: {self.user_config_dir}")


# endregion


# region DatasetLoader
class DatasetLoader:
    """Lazy-loads parquet datasets from configured paths."""

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
        "ventilation": "ventilation.parquet",
        "ventilation_duration": "ventilation.parquet",
        # Renal replacement therapy
        "rrt": "renal_replacement_therapy.parquet",
        "renal_replacement_therapy": "renal_replacement_therapy.parquet",
        "renal_replacement_therapy_duration": "renal_replacement_therapy.parquet",
        # Antibiotic and infection concepts
        "received_any_antibiotics": "received_any_antibiotics.parquet",
        # Severity scores and status
        "severity_scores": "severity_scores.parquet",
        "code_status": "code_status.parquet",
    } # fmt: skip

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._lazy_cache: Dict[str, pl.LazyFrame] = {}
        self.demo_mode = False

    def set_demo_mode(self, enabled: bool = True) -> None:
        """Switch between demo and full dataset mode."""
        self.demo_mode = enabled
        self._lazy_cache.clear()  # Clear cache when switching modes
        mode = "DEMO" if enabled else "FULL"
        print(f"!! Switched to {mode} mode !!")

    def get_data_path(self) -> Path:
        """Get the base data path based on current mode."""
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
        """Get the base data path based on current mode."""
        base_path = self.get_data_path()
        return Path(base_path / "MAGIC_CONCEPTS")

    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset file exists."""
        if dataset_name not in self.DATASET_MAPPING:
            return False

        try:
            path = self.get_dataset_path(dataset_name)
            return path.exists()
        except Exception:
            return False

    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get the full path to a dataset file."""
        if dataset_name not in self.DATASET_MAPPING:
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Available datasets: {', '.join(sorted(self.DATASET_MAPPING.keys()))}"
            )

        filename = self.DATASET_MAPPING[dataset_name]
        return self.get_data_path() / filename

    def available_datasets(self) -> list:
        """Get list of available dataset names (ones that actually exist)."""
        available = []
        for dataset_name in sorted(self.DATASET_MAPPING.keys()):
            if self.dataset_exists(dataset_name):
                available.append(dataset_name)
        return available

    def concept_exists(self, concept_name: str) -> bool:
        """Check if a concept file exists."""
        if concept_name not in self.CONCEPT_MAPPING:
            return False

        try:
            path = self.get_concept_path(concept_name)
            return path.exists()
        except Exception:
            return False

    def get_concept_path(self, concept_name: str) -> Path:
        """Get the full path to a concept file."""
        if concept_name not in self.CONCEPT_MAPPING:
            raise ValueError(
                f"Unknown concept: '{concept_name}'. "
                f"Available concepts: {', '.join(sorted(self.CONCEPT_MAPPING.keys()))}"
            )

        filename = self.CONCEPT_MAPPING[concept_name]
        return self.get_concepts_path() / filename

    def available_concepts(self) -> list:
        """Get list of available concept names (ones that actually exist)."""
        available = []
        for concept_name in sorted(self.CONCEPT_MAPPING.keys()):
            if self.concept_exists(concept_name):
                available.append(concept_name)
        return available

    def load_dataset(self, dataset_name: str) -> pl.LazyFrame:
        """
        Lazy-load a dataset as a Polars LazyFrame.

        Returns:
            pl.LazyFrame: Scanned parquet file (lazy, not loaded into memory)

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset name unknown
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
        """Reload a dataset, clearing the cache."""
        if dataset_name in self._lazy_cache:
            del self._lazy_cache[dataset_name]
        return self.load_dataset(dataset_name)

    def load_concept(self, concept_name: str) -> pl.LazyFrame:
        """
        Lazy-load a concept as a Polars LazyFrame.

        Returns:
            pl.LazyFrame: Scanned parquet file (lazy, not loaded into memory)

        Raises:
            FileNotFoundError: If concept file doesn't exist
            ValueError: If concept name unknown
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
        """Reload a concept, clearing the cache."""
        if concept_name in self._lazy_cache:
            del self._lazy_cache[concept_name]
        return self.load_concept(concept_name)

    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        self._lazy_cache.clear()
        print("✓ Dataset cache cleared")


# endregion


class reprodICUPaths:
    """Load and store reprodICU paths from user configuration.

    This class loads paths from PATHS.yaml and provides convenient
    access to all configured directories for data, output, etc.

    Attributes are dynamically set from the configuration file.
    """

    def __init__(self, config_manager=None) -> None:
        """Initialize paths from user configuration.

        Args:
            config_manager: Optional ConfigManager instance. If not provided,
                          a global instance will be created.

        Raises:
            FileNotFoundError: If PATHS.yaml configuration is not found
            ValueError: If required paths are missing from configuration
        """
        if config_manager is None:
            config_manager = get_config_manager()

        config = config_manager.load_config("PATHS.yaml", user_override=True)
        for key, value in config.items():
            setattr(self, key, str(value))

    def validate_paths(self) -> list:
        """Validate that all configured paths exist.

        Returns:
            List of tuples (key, path) for paths that don't exist

        Example:
            >>> paths = reprodICUPaths()
            >>> missing = paths.validate_paths()
            >>> if missing:
            ...     print(f"Missing paths: {missing}")
        """
        missing_paths = []
        for key, value in self.__dict__.items():
            if not Path(value).exists():
                missing_paths.append((key, value))
        return missing_paths


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
