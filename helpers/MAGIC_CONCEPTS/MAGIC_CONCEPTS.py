# Author: Finn Fassbender
# Last modified: 2024-09-05

import yaml

from helpers.helper import GlobalHelpers, GlobalVars
from helpers.helper_filepaths import (
    EICUPaths,
    HiRIDPaths,
    MIMIC3Paths,
    MIMIC4Paths,
    SICdbPaths,
    UMCdbPaths,
)
from helpers.helper_ricu_mappings import ricuMappings


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MAGIC_CONCEPTS:
    def __init__(self, paths, datasets, DEMO=False):
        self.global_vars = GlobalVars(paths=paths, DEMO=DEMO)
        self.global_helpers = GlobalHelpers()

        # Initialize the paths
        self.paths = paths
        self.eicu_paths = EICUPaths(paths=paths)
        self.hirid_paths = HiRIDPaths(paths=paths)
        self.mimic3_paths = MIMIC3Paths(paths=paths)
        self.mimic4_paths = MIMIC4Paths(paths=paths)
        self.sicdb_paths = SICdbPaths(paths=paths)
        self.umcdb_paths = UMCdbPaths(paths=paths)

        # Initialize the datasets
        self.datasets = datasets

        # Initialize the ricu mappings
        self.ricu_mappings = ricuMappings()
        self.ricu_concept_dict = self.ricu_mappings.ricu_concept_dict

        # Initialize the column names
        self.column_names = load_mapping("configs/COLUMN_NAMES.yaml")
