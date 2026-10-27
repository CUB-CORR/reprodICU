# Author: Finn Fassbender
# Last modified: 2026-10-29

# Description: Derives sicdb_fluid_carriers.csv (next to this script) from the
# medication table of SICdb. It is read by SICdbExtractor._add_volumes. Run it
# again for other releases of SICdb:
#   python derive_sicdb_fluid_carriers.py SICDB_PATH [--output CSV]

import argparse
from pathlib import Path

import polars as pl

# A fluid or food in ml is a carrier if this share of its rows has a drug at its time
CARRIER_MIN_SHARE = 0.85
# A drug is a usual one if this share of its rows alone at their time (at least
# USUAL_MIN_ROWS) has a carrier at its time, and the control (times shifted by
# CONTROL_SHIFT_SECONDS) has at most USUAL_MAX_CONTROL of the hits
USUAL_MIN_SHARE = 0.5
USUAL_MIN_ROWS = 5
USUAL_MAX_CONTROL = 0.3
CONTROL_SHIFT_SECONDS = 3600

HEADER = f"""\
# Carriers of SICdb drugs and the drugs that usually get one. A carrier is a fluid in ml that is the diluent of a drug (e.g. "NaCl 0,9% Perfusor"): it has the case, start and end of a drug.
# role carrier: fluid or food in ml that has a drug at its time in {CARRIER_MIN_SHARE * 100:.0f} % or more of its rows. rows: its rows with Amount > 0; hits: rows with a drug at their time; share: hits / rows.
# role usual: drug that gets a carrier in {USUAL_MIN_SHARE * 100:.0f} % or more of its rows that are alone at their time (at least {USUAL_MIN_ROWS} such rows). rows: its rows that are alone at their time; hits: rows with a carrier at their time.
# control_hits: hits with the times shifted by {CONTROL_SHIFT_SECONDS // 3600} h (chance). Usual drugs with control_hits above {USUAL_MAX_CONTROL * 100:.0f} % of hits are not listed.
# Used in SICdbExtractor._add_volumes: a carrier is linked to the one usual drug at its time. Derived from SICdb 1.0.8 by derive_sicdb_fluid_carriers.py (re-derive for other SICdb releases).
"""

KEY = ["CaseID", "Offset", "OffsetDrugEnd"]


def get_hits(
    rows: pl.DataFrame, others: pl.DataFrame, shift: int
) -> pl.DataFrame:
    """Count the rows of each DrugID that have a row of others at their time (shifted)."""
    at_time = (
        others.select(KEY)
        .unique()
        .with_columns(
            pl.col("Offset") + shift,
            pl.col("OffsetDrugEnd") + shift,
            pl.lit(True).alias("found"),
        )
    )
    return (
        rows.join(at_time, on=KEY, how="left")
        .group_by("drugid")
        .agg(
            pl.len().alias("rows"),
            pl.col("found").fill_null(False).sum().alias("hits"),
        )
    )


def get_evidence(rows: pl.DataFrame, others: pl.DataFrame) -> pl.DataFrame:
    """Get rows, hits, share and the hits with the times shifted (control) of each DrugID."""
    control = get_hits(rows, others, CONTROL_SHIFT_SECONDS).select(
        "drugid", pl.col("hits").alias("control_hits")
    )
    return (
        get_hits(rows, others, 0)
        .join(control, on="drugid")
        .with_columns((pl.col("hits") / pl.col("rows")).round(3).alias("share"))
    )


def derive_fluid_carriers(sicdb_path: Path) -> pl.DataFrame:
    """Derive the carriers and the usual drugs from the medication table."""
    references = pl.read_csv(sicdb_path / "d_references.csv.gz").select(
        pl.col("ReferenceGlobalID").alias("drugid"),
        pl.col("ReferenceValue").alias("drug_name"),
        "ReferenceName",
        "ReferenceUnit",
    )
    fluids = references.filter(
        pl.col("ReferenceName").is_in(["Fluid", "Food"]),
        pl.col("ReferenceUnit") == "ml",
    )
    drug_names = references.filter(pl.col("ReferenceName") == "Drug")
    medication = pl.read_csv(sicdb_path / "medication.csv.gz").select(
        "CaseID",
        pl.col("DrugID").alias("drugid"),
        "Offset",
        "OffsetDrugEnd",
        "Amount",
    )
    drugs = medication.filter(
        pl.col("drugid").is_in(drug_names["drugid"].implode())
    )
    volumes = medication.filter(
        pl.col("drugid").is_in(fluids["drugid"].implode()), pl.col("Amount") > 0
    )

    # Carriers have a drug at their time in nearly all of their rows
    carriers = (
        get_evidence(volumes, drugs)
        .filter(pl.col("share") >= CARRIER_MIN_SHARE)
        .join(fluids.select("drugid", "drug_name"), on="drugid")
    )

    # Usual drugs have a carrier in many of their rows that are alone at their time
    usual = (
        get_evidence(
            drugs.filter(pl.len().over(KEY) == 1),
            volumes.join(carriers.select("drugid"), on="drugid", how="semi"),
        )
        .filter(
            pl.col("share") >= USUAL_MIN_SHARE,
            pl.col("rows") >= USUAL_MIN_ROWS,
            pl.col("control_hits") <= USUAL_MAX_CONTROL * pl.col("hits"),
        )
        .join(drug_names.select("drugid", "drug_name"), on="drugid")
    )

    columns = [
        "drugid",
        "drug_name",
        "role",
        "rows",
        "hits",
        "share",
        "control_hits",
    ]
    return pl.concat(
        [
            frame.with_columns(pl.lit(role).alias("role"))
            .select(columns)
            .sort("rows", descending=True)
            for role, frame in (("carrier", carriers), ("usual", usual))
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sicdb_path", type=Path, help="Folder of the SICdb files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("sicdb_fluid_carriers.csv"),
    )
    args = parser.parse_args()

    table = derive_fluid_carriers(args.sicdb_path)
    args.output.write_text(HEADER + table.write_csv())
