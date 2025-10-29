# Author: Finn Fassbender
# Last modified: 2025-10-29

# Description: Utility functions for clinical score calculations and data processing.
# These utilities provide specialized implementations for:
# - SOFA score (Sequential Organ Failure Assessment)
# - Sepsis detection (Sepsis-3 and surveillance criteria)
# - VIS score (Vasoactive-Inotropic Score)
# - Urine output calculations and time window processing
#
# Core functions support automatic dataset loading from package defaults:
# - SOFA(), SEPSIS(), VIS(), URINE_OUTPUT()
# All parameters are optional - if not provided, datasets load from package
# configuration automatically.

# Clinical scoring utilities
from .scores.SOFA import SOFA
from .scores.VIS import VIS
from .sepsis.SEPSIS import SEPSIS

# Data processing utilities
from .URINE_OUTPUT import URINE_OUTPUT
from .FIX_WINDOW_BORDERS import FIX_WINDOW_BORDERS

# Helper utilities for manual dataset loading
from .common import (
    get_patient_information,
    get_timeseries_vitals,
    get_timeseries_labs,
    get_timeseries_respiratory,
    get_timeseries_intakeoutput,
    get_medications,
    get_diagnoses,
    get_procedures,
    get_notes,
    get_microbiology,
)

__all__ = [
    # Clinical scoring functions (auto-load datasets if not provided)
    "SOFA",
    "SEPSIS",
    "VIS",
    # Data processing functions
    "URINE_OUTPUT",
    "FIX_WINDOW_BORDERS",
    # Dataset loading helpers (for manual control)
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
]
