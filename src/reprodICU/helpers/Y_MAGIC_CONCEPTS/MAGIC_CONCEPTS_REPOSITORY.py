# Author: Finn Fassbender
# Last modified: 2025-10-30

import yaml


def load_mapping(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class MAGIC_CONCEPTS_REPOSITORY:
    def __init__(self, paths, datasets):
        # Lazy imports to avoid circular dependency with helpers.MAGIC_CONCEPTS
        from .CODE_STATUS import CODE_STATUS
        from .RECEIVED_ANY_ANTIBIOTICS import (
            RECEIVED_ANY_ANTIBIOTICS as ANY_ABX,
        )
        from .RENAL_REPLACEMENT_THERAPY_DURATION import (
            RENAL_REPLACEMENT_THERAPY_DURATION as RRT_DURATION,
        )
        from .SEVERITY_SCORES import SEVERITY_SCORES
        from .VENTILATION_DURATION import VENTILATION_DURATION as MV_DURATION

        self.CODE_STATUS = CODE_STATUS(paths=paths, datasets=datasets)
        self.ANY_ABX = ANY_ABX(paths=paths, datasets=datasets)
        self.MV_DURATION = MV_DURATION(paths=paths, datasets=datasets)
        self.RRT_DURATION = RRT_DURATION(paths=paths, datasets=datasets)
        self.SEVERITY_SCORES = SEVERITY_SCORES(paths=paths, datasets=datasets)

        self.magic_concepts_dict = {
            "CODE_STATUS": self.CODE_STATUS.CODE_STATUS,
            "RECEIVED_ANY_ANTIBIOTICS": self.ANY_ABX.RECEIVED_ANY_ANTIBIOTICS,
            "RENAL_REPLACEMENT_THERAPY_DURATION": (
                self.RRT_DURATION.RENAL_REPLACEMENT_THERAPY_DURATION
            ),
            "SEVERITY_SCORES": self.SEVERITY_SCORES.SEVERITY_SCORES,
            "VENTILATION_DURATION": self.MV_DURATION.VENTILATION_DURATION,
        }

    def get_magic_concept(self, concept: str):
        return self.magic_concepts_dict[concept]()

    def get_all_magic_concepts(self):
        return {
            concept: self.get_magic_concept(concept)
            for concept in self.magic_concepts_dict.keys()
        }
