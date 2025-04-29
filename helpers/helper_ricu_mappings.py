# Author: Finn Fassbender
# Last modified: 2024-09-05

# Description: This script contains helper functions to get the ricu mappings used in
# eth-mds/ricu (https://github.com/eth-mds/ricu)

import json
import polars as pl
from pathlib import Path

from helpers.helper import GlobalVars, GlobalHelpers


class ricuMappings:
    def __init__(self):
        self.ricu_concept_dict_file = (
            Path(__file__).parent / "../mappings/_ricu/concept-dict.json"
        )
        self.ricu_concept_dict = self._load_ricu_concept_dict()

    def _load_ricu_concept_dict(self):
        with open(self.ricu_concept_dict_file, "r") as f:
            return json.load(f)
