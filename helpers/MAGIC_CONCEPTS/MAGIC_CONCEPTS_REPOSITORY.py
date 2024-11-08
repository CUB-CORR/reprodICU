# Author: Finn Fassbender
# Last modified: 2024-09-05

import yaml

from helpers.MAGIC_CONCEPTS import (
    RECEIVED_ANY_ANTIBIOTICS as ANY_ABX,
    VENTILATION_DURATION as MV_DURATION,
    RRT_DURATION as RRT_DURATION,
)


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MAGIC_CONCEPTS_REPOSITORY:
    def __init__(self, paths, datasets):
        self.ANY_ABX = ANY_ABX.RECEIVED_ANY_ANTIBIOTICS(
            paths=paths, datasets=datasets
        )
        self.MV_DURATION = MV_DURATION.VENTILATION_DURATION(
            paths=paths, datasets=datasets
        )
        self.RRT_DURATION = RRT_DURATION.RENAL_REPLACEMENT_THERAPY_DURATION(
            paths=paths, datasets=datasets
        )

        self.magic_concepts_dict = {
            "RECEIVED_ANY_ANTIBIOTICS": self.ANY_ABX.RECEIVED_ANY_ANTIBIOTICS,
            "VENTILATION_DURATION": self.MV_DURATION.VENTILATION_DURATION,
            "RENAL_REPLACEMENT_THERAPY_DURATION": self.RRT_DURATION.RENAL_REPLACEMENT_THERAPY_DURATION,
        }

    def get_magic_concept(self, concept: str):
        return self.magic_concepts_dict[concept]()

    def get_all_magic_concepts(self):
        return {
            concept: self.get_magic_concept(concept)
            for concept in self.magic_concepts_dict.keys()
        }
