#!/usr/bin/env python3
"""
build_db.py: builds the harmonized ICU database with reprodICU, from raw data to concepts.

Commands (run through ./run.sh, which sets up the Python environment and the log):

    ./run.sh check      status of the raw data, the OMOP vocabularies and the outputs
    ./run.sh verify     checks the downloads against the PhysioNet SHA256SUMS.txt files
    ./run.sh prepare    one-off preparation of the raw data (safe to rerun):
                          verify : SHA256 check of the downloads (once)
                          HiRID  : extraction of the .tar.gz archives
                          SICdb  : unpacking of data_float_h into data_float_m (2 to 3 h)
                          UMCdb  : re-encoding to UTF-8
                          OMOP   : unzipping of the Athena archive
                          all    : conversion of the large CSV files to parquet (reprodICU cache)
    ./run.sh build      builds the harmonized tables for every dataset that is ready
    ./run.sh concepts   reprodICU derived concepts (ventilation durations, scores...)
    ./run.sh summary    counts per dataset in the output tables

Everything can be rerun. `build` rebuilds the combined tables but reuses the
per-dataset intermediate files cached in <output>/_tempfiles/: adding a dataset
means `prepare`, then `build`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

# ----------------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------------
TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent  # root of the reprodICU repository
CONFIG_FILE = (TOOL_DIR / "config.local.yaml") if (TOOL_DIR / "config.local.yaml").exists() \
    else (TOOL_DIR / "config.yaml")
CONFIG = yaml.safe_load(CONFIG_FILE.read_text())


def _p(path: str) -> Path:
    """Absolute path; relative paths are resolved against the repository root."""
    p = Path(path).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


RAW = _p(CONFIG["paths"]["raw_data"])
OMOP = _p(CONFIG["paths"]["omop_vocabulary"])
OUT = _p(CONFIG["paths"]["output"])
OUT_DEMO = _p(CONFIG["paths"]["output_demo"])
PHYSIONET = RAW / "physionet.org" / "files"
UMCDB_RAW = RAW / "UMCdb"
UMCDB_FIXED = UMCDB_RAW / "ENCODINGFIX"
# Marker files written when a preparation step is done. Earlier versions of this
# tool used the prefix ".optiflow_done_", still recognised so that done steps are
# not redone.
MARK = ".build_done_"
MARKS = (MARK, ".optiflow_done_")

ALL_DATASETS = ["eICU", "HiRID", "MIMIC3", "MIMIC4", "NWICU", "SICdb", "UMCdb"]

# Files expected by reprodICU (see helpers/helper_filepaths.py)
DATASETS = {
    "eICU": {
        "dir": PHYSIONET / "eicu-crd" / "2.0",
        "files": [f"{t}.csv.gz" for t in (
            "admissionDrug admissionDx allergy apacheApsVar apachePatientResult "
            "apachePredVar carePlanCareProvider carePlanEOL carePlanGeneral carePlanGoal "
            "carePlanInfectiousDisease customLab diagnosis hospital infusionDrug "
            "intakeOutput lab medication microLab note nurseAssessment nurseCare "
            "nurseCharting pastHistory patient physicalExam respiratoryCare "
            "respiratoryCharting treatment vitalAperiodic vitalPeriodic").split()],
    },
    "HiRID": {
        "dir": PHYSIONET / "hirid" / "1.1.1",
        "files": [
            "reference_data/general_table.csv",
            "reference_data/hirid_variable_reference.csv",
            "raw_stage/observation_tables/parquet",
            "raw_stage/pharma_records/parquet",
        ],
    },
    "MIMIC3": {
        "dir": PHYSIONET / "mimiciii" / "1.4",
        "files": [f"{t}.csv.gz" for t in (
            "ADMISSIONS CHARTEVENTS D_ICD_DIAGNOSES D_ICD_PROCEDURES D_ITEMS D_LABITEMS "
            "DATETIMEEVENTS DIAGNOSES_ICD ICUSTAYS INPUTEVENTS_CV INPUTEVENTS_MV LABEVENTS "
            "MICROBIOLOGYEVENTS NOTEEVENTS OUTPUTEVENTS PATIENTS PRESCRIPTIONS "
            "PROCEDUREEVENTS_MV PROCEDURES_ICD SERVICES").split()],
    },
    "MIMIC4": {
        "dir": PHYSIONET / "mimiciv" / "3.1",
        "files": [f"{t}.csv.gz" for t in (
            "hosp/admissions icu/chartevents hosp/d_icd_diagnoses hosp/d_icd_procedures "
            "icu/d_items hosp/d_labitems icu/datetimeevents hosp/diagnoses_icd hosp/emar "
            "hosp/emar_detail icu/icustays icu/inputevents hosp/labevents "
            "hosp/microbiologyevents icu/outputevents hosp/patients hosp/prescriptions "
            "icu/procedureevents hosp/procedures_icd hosp/services").split()],
        "notes_dir": PHYSIONET / "mimic-iv-note" / "2.2",
        "notes_files": [f"note/{t}.csv.gz" for t in
                        "discharge discharge_detail radiology radiology_detail".split()],
    },
    "NWICU": {
        "dir": PHYSIONET / "nwicu-northwestern-icu" / "0.1.0" / "data",
        "files": [f"{t}.csv.gz" for t in (
            "nw_hosp/admissions nw_icu/chartevents nw_hosp/d_icd_diagnoses nw_icu/d_items "
            "nw_hosp/d_labitems nw_hosp/diagnoses_icd nw_hosp/emar nw_icu/icustays "
            "nw_hosp/labevents nw_hosp/patients nw_hosp/prescriptions "
            "nw_icu/procedureevents").split()],
    },
    "SICdb": {
        "dir": PHYSIONET / "sicdb" / "1.0.8",
        "files": [f"{t}.csv.gz" for t in (
            "cases d_references data_float_h data_range data_ref laboratory medication "
            "unitlog").split()],
    },
    "UMCdb": {
        "dir": UMCDB_FIXED,
        "files": ["admissions.csv", "drugitems.csv", "freetextitems.csv", "listitems.csv",
                  "numericitems.csv.gz", "procedureorderitems.csv", "processitems.csv"],
    },
}

OMOP_TABLES = ["CONCEPT", "CONCEPT_ANCESTOR", "CONCEPT_CLASS", "CONCEPT_RELATIONSHIP",
               "CONCEPT_SYNONYM", "DOMAIN", "DRUG_STRENGTH", "RELATIONSHIP", "VOCABULARY"]

HIRID_ARCHIVES = {  # archive -> (target folder, marker)
    "reference_data.tar.gz": ("reference_data", "reference_data"),
    "raw_stage/observation_tables_parquet.tar.gz": ("raw_stage", "observation_tables"),
    "raw_stage/pharma_records_parquet.tar.gz": ("raw_stage", "pharma_records"),
}

BUILD_STEPS = ["patient_information", "diagnoses", "procedures", "medications",
               "microbiology", "notes", "timeseries", "availability", "overview"]

# Datasets whose free-text notes reprodICU harmonizes
NOTES_DATASETS = {"MIMIC3", "MIMIC4"}

# Datasets read by the reprodICU MAGIC_CONCEPTS (not NWICU)
CONCEPT_DATASETS = ["eICU", "HiRID", "MIMIC3", "MIMIC4", "SICdb", "UMCdb"]


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def log(msg: str = "") -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def header(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


def d(path: Path) -> str:
    """Folder path with a trailing '/' (reprodICU concatenates strings)."""
    return str(path) + "/"


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"


def duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def missing_files(base: Path, files: list[str]) -> list[str]:
    out = []
    for f in files:
        p = base / f
        if not p.exists() or (p.is_dir() and not any(p.iterdir())):
            out.append(f)
    return out


def enabled(name: str) -> bool:
    return bool(CONFIG.get("datasets", {}).get(name, True))


def done(folder: Path, step: str) -> bool:
    """True if the marker of a preparation step exists (current or earlier prefix)."""
    return any((folder / f"{m}{step}").exists() for m in MARKS)


def mark_done(folder: Path, step: str) -> None:
    (folder / f"{MARK}{step}").write_text(dt.datetime.now().isoformat())


# ----------------------------------------------------------------------------
# reprodICU configuration (~/.reprodICU/PATHS.yaml)
# ----------------------------------------------------------------------------
def write_reprodicu_paths(umcdb_source: Path | None = None) -> Path:
    """Writes ~/.reprodICU/PATHS.yaml from the configuration (absolute paths)."""
    if umcdb_source is None:
        umcdb_source = UMCDB_FIXED
    paths = {
        "reprodICU_files_path": d(OUT),
        "reprodICU_demo_files_path": d(OUT_DEMO),
        "OMOP_vocab_path": d(OMOP),
        "eicu_source_path": d(DATASETS["eICU"]["dir"]),
        "hirid_source_path": d(DATASETS["HiRID"]["dir"]),
        "mimic3_source_path": d(DATASETS["MIMIC3"]["dir"]),
        "mimic4_source_path": d(DATASETS["MIMIC4"]["dir"]),
        "mimic4_notes_source_path": d(DATASETS["MIMIC4"]["notes_dir"]),
        "nwicu_source_path": d(DATASETS["NWICU"]["dir"]),
        "sicdb_source_path": d(DATASETS["SICdb"]["dir"]),
        "umcdb_source_path": d(umcdb_source),
        # demo datasets (not used by this tool)
        "eicu_demo_source_path": d(PHYSIONET / "eicu-crd-demo" / "2.0.1"),
        "mimic3_demo_source_path": d(PHYSIONET / "mimiciii-demo" / "headerfix"),
        "mimic4_demo_source_path": d(PHYSIONET / "mimic-iv-demo" / "2.2"),
    }
    text = (f"# Written by {TOOL_DIR / 'build_db.py'}\n"
            f"# Do not edit here: edit {CONFIG_FILE.name} in {TOOL_DIR} and rerun.\n"
            + yaml.safe_dump(paths, sort_keys=False, allow_unicode=True))

    user_dir = Path.home() / ".reprodICU"
    user_dir.mkdir(parents=True, exist_ok=True)
    target = user_dir / "PATHS.yaml"
    managed = [user_dir / ".managed_by_build_db", user_dir / ".managed_by_optiflow"]
    if target.exists() and not any(m.exists() for m in managed):
        # first run: keep a copy of the existing PATHS.yaml
        backup = target.with_name(f"PATHS.yaml.bak-{dt.datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy(target, backup)
        log(f"Previous PATHS.yaml saved as {backup}")
    target.write_text(text)
    managed[0].write_text(str(TOOL_DIR))
    for p in (OUT, OMOP):
        p.mkdir(parents=True, exist_ok=True)
    _keep_data_out_of_git()
    return target


def _keep_data_out_of_git() -> None:
    """Data folders configured inside the repository get a .gitignore ('*'), so that
    credentialed data and outputs can never be committed by mistake."""
    for p in (RAW, OMOP, OUT, OUT_DEMO):
        if p == REPO_ROOT or REPO_ROOT not in p.parents or not p.exists():
            continue
        ignore = p / ".gitignore"
        if not ignore.exists():
            ignore.write_text("# Written by build_db.py: data folder, never committed\n*\n")
            log(f"{p} is inside the repository: .gitignore added to keep it out of git")


def reprodicu_paths():
    """Fresh reprodICUPaths object (no cached configuration)."""
    from reprodICU.config import ConfigManager, reprodICUPaths
    return reprodICUPaths(ConfigManager())


# ----------------------------------------------------------------------------
# Data status
# ----------------------------------------------------------------------------
def omop_status() -> tuple[bool, list[str]]:
    miss = [t for t in OMOP_TABLES
            if not (OMOP / f"{t}.csv").exists() and not (OMOP / f"{t}.parquet").exists()]
    return (not miss, miss)


def dataset_status(name: str) -> tuple[str, str]:
    """Returns (status, detail): READY | TO PREPARE | INCOMPLETE | MISSING | DISABLED."""
    if not enabled(name):
        return "DISABLED", f"disabled in {CONFIG_FILE.name}"
    spec = DATASETS[name]

    if name == "UMCdb":
        raw = UMCDB_RAW
        has_raw = raw.exists() and any(
            (raw / f).exists() for f in ("admissions.csv", "numericitems.zip",
                                         "numericitems.csv", "numericitems.csv.gz"))
        if done(UMCDB_FIXED, "encodingfix") and not missing_files(UMCDB_FIXED, spec["files"]):
            return "READY", str(UMCDB_FIXED)
        if has_raw:
            return "TO PREPARE", "UTF-8 re-encoding to do (./run.sh prepare)"
        return "MISSING", f"put the AmsterdamUMCdb files in {UMCDB_RAW}"

    base = spec["dir"]
    if name == "HiRID":
        archives_ok = all((base / a).exists() for a in HIRID_ARCHIVES)
        todo = [m for _, (dest, m) in HIRID_ARCHIVES.items() if not done(base / dest, m)]
        if not base.exists():
            return "MISSING", str(base)
        if todo:
            return ("TO PREPARE" if archives_ok else "INCOMPLETE",
                    "extraction to do: " + ", ".join(todo))
    if not base.exists():
        return "MISSING", str(base)
    miss = missing_files(base, spec["files"])
    if miss:
        return "INCOMPLETE", "missing: " + ", ".join(miss[:6]) + (" ..." if len(miss) > 6 else "")
    if name == "MIMIC4":
        nmiss = missing_files(spec["notes_dir"], spec["notes_files"])
        if nmiss:
            return "INCOMPLETE", f"MIMIC-IV-Note missing ({spec['notes_dir']})"
    if name == "SICdb":
        if not (done(base, "data_float_m") or (base / "data_float_m.parquet").exists()):
            return "TO PREPARE", "unpacking of data_float_h into data_float_m to do (2 to 3 h)"
    return "READY", str(base)


def ready_datasets() -> list[str]:
    return [n for n in ALL_DATASETS if dataset_status(n)[0] == "READY"]


def cmd_check(args) -> int:
    target = write_reprodicu_paths()
    header("reprodICU build: status")
    print(f"Configuration       : {CONFIG_FILE}")
    print(f"reprodICU config    : {target}")
    print(f"reprodICU (code)    : {REPO_ROOT}  [branch {_git_branch()}: {_git_commit()}]")
    print()
    print(f"{'Dataset':8} {'Status':12} Detail")
    print("-" * 78)
    for name in ALL_DATASETS:
        state, detail = dataset_status(name)
        print(f"{name:8} {state:12} {detail}")
    ok, miss = omop_status()
    print("-" * 78)
    print(f"{'OMOP':8} {'READY' if ok else 'MISSING':12} "
          + (str(OMOP) if ok else f"missing tables: {', '.join(miss)}"))
    zips = list(OMOP.glob("*.zip"))
    if not ok and zips:
        print(f"{'':8} {'':12} archive found ({zips[0].name}): ./run.sh prepare will unzip it")

    print(f"\nOutputs ({OUT}):")
    files = sorted(OUT.glob("*.parquet")) + sorted((OUT / "MAGIC_CONCEPTS").glob("*.parquet"))
    if not files:
        print("  (none yet)")
    for f in files:
        print(f"  {f.relative_to(OUT)!s:55} {human(f.stat().st_size):>10}  "
              f"{dt.datetime.fromtimestamp(f.stat().st_mtime):%Y-%m-%d %H:%M}")

    print("\nNext step:")
    states = {n: dataset_status(n)[0] for n in ALL_DATASETS}
    if any(s == "TO PREPARE" for s in states.values()) or (not ok and zips):
        print("  ./run.sh prepare")
    elif not ok:
        print("  download the OMOP vocabularies from Athena (see README), then ./run.sh prepare")
    elif ready_datasets():
        print(f"  ./run.sh build   (ready datasets: {', '.join(ready_datasets())})")
    return 0


# ----------------------------------------------------------------------------
# Preparation of the raw data
# ----------------------------------------------------------------------------
def verify_checksums(force: bool = False) -> bool:
    """Checks the downloaded files against the SHA256SUMS.txt files provided by PhysioNet."""
    import hashlib
    all_ok = True
    for sums in sorted(PHYSIONET.glob("*/*/SHA256SUMS.txt")):
        base = sums.parent
        name = f"{base.parent.name}/{base.name}"
        if done(base, "sha256") and not force:
            log(f"SHA256 - {name}: already checked")
            continue
        bad, missing, n, size = [], [], 0, 0
        t0 = time.time()
        for line in sums.read_text().splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            expected, rel = parts[0], parts[1].lstrip("*")
            f = base / rel
            if not f.exists():
                missing.append(rel)
                continue
            sha = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(16 << 20), b""):
                    sha.update(chunk)
            n += 1
            size += f.stat().st_size
            if sha.hexdigest() != expected:
                bad.append(rel)
        msg = f"SHA256 - {name}: {n} files ({human(size)}) in {duration(time.time() - t0)}"
        if bad:
            all_ok = False
            log(msg + f", !! CORRUPTED: {', '.join(bad)} (download them again)")
        else:
            log(msg + ", OK")
            mark_done(base, "sha256")
        if missing:
            log(f"SHA256 - {name}: not downloaded (for information): {', '.join(missing)}")
    return all_ok


def prepare_hirid() -> None:
    base = DATASETS["HiRID"]["dir"]
    if not base.exists():
        log("HiRID  - missing, skipped")
        return
    for archive, (dest, marker) in HIRID_ARCHIVES.items():
        target = base / dest
        if done(target, marker):
            log(f"HiRID  - {marker}: already extracted")
            continue
        src = base / archive
        if not src.exists():
            log(f"HiRID  - !! archive missing: {src}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        log(f"HiRID  - extracting {archive} ({human(src.stat().st_size)}) to {target}")
        t0 = time.time()
        subprocess.run(["tar", "-xzf", str(src), "-C", str(target)], check=True)
        mark_done(target, marker)
        log(f"HiRID  - {marker} extracted in {duration(time.time() - t0)}")


def prepare_sicdb() -> None:
    base = DATASETS["SICdb"]["dir"]
    if not base.exists():
        log("SICdb  - missing, skipped")
        return
    if done(base, "data_float_m") or (base / "data_float_m.parquet").exists():
        log("SICdb  - data_float_m: already unpacked")
        return
    log("SICdb  - unpacking data_float_h into data_float_m (about 2.6 billion values, "
        "2 to 3 h; can be rerun if interrupted)")
    t0 = time.time()
    from reprodICU.config import ConfigManager
    from reprodICU.setup import setup_sicdb
    setup_sicdb(config_manager=ConfigManager())
    mark_done(base, "data_float_m")
    log(f"SICdb  - unpacking done in {duration(time.time() - t0)}")


def _umcdb_convert(src: Path, dst: Path, gz_in: bool, gz_out: bool) -> None:
    """Re-encodes an AmsterdamUMCdb CSV from unicode-escape to UTF-8 (same logic as reprodICU)."""
    import csv
    import gzip
    opener_in = (lambda: gzip.open(src, "rt", encoding="unicode-escape", errors="ignore")) if gz_in \
        else (lambda: open(src, "r", encoding="unicode-escape", errors="ignore"))
    opener_out = (lambda: gzip.open(dst, "wt", encoding="utf-8")) if gz_out \
        else (lambda: open(dst, "w", encoding="utf-8"))
    with opener_in() as fin, opener_out() as fout:
        writer = csv.writer(fout)
        for row in csv.reader(fin):
            writer.writerow(row)


def prepare_umcdb() -> None:
    if not enabled("UMCdb"):
        return
    if done(UMCDB_FIXED, "encodingfix"):
        log("UMCdb  - already re-encoded")
        return
    if not UMCDB_RAW.exists() or not (UMCDB_RAW / "admissions.csv").exists():
        log(f"UMCdb  - missing (put the AmsterdamUMCdb files in {UMCDB_RAW})")
        return
    log("UMCdb  - re-encoding from unicode-escape to UTF-8 (reprodICU.setup.setup_umcdb)")
    t0 = time.time()
    from reprodICU.config import ConfigManager
    from reprodICU.setup import setup_umcdb
    write_reprodicu_paths(umcdb_source=UMCDB_RAW)  # setup_umcdb reads the raw folder
    setup_umcdb(config_manager=ConfigManager())
    # numericitems delivered as .csv or .csv.gz instead of .zip: same conversion
    out = UMCDB_FIXED / "numericitems.csv.gz"
    if not out.exists():
        for name, gz in (("numericitems.csv.gz", True), ("numericitems.csv", False)):
            if (UMCDB_RAW / name).exists():
                log(f"UMCdb  - converting {name}")
                _umcdb_convert(UMCDB_RAW / name, out, gz_in=gz, gz_out=True)
                break
    write_reprodicu_paths()
    miss = missing_files(UMCDB_FIXED, DATASETS["UMCdb"]["files"])
    if miss:
        log(f"UMCdb  - !! files missing after conversion: {', '.join(miss)}")
        return
    mark_done(UMCDB_FIXED, "encodingfix")
    log(f"UMCdb  - re-encoding done in {duration(time.time() - t0)}")


def prepare_omop() -> None:
    ok, miss = omop_status()
    if ok:
        log("OMOP   - vocabularies present")
    else:
        zips = sorted(OMOP.glob("*.zip"))
        if not zips:
            log(f"OMOP   - !! vocabularies missing ({', '.join(miss)}): "
                f"download them from https://athena.ohdsi.org and put the zip in {OMOP}")
            return
        log(f"OMOP   - unzipping {zips[0].name}")
        with zipfile.ZipFile(zips[0]) as zf:
            for member in zf.infolist():
                name = Path(member.filename).name
                if member.is_dir() or not name:
                    continue
                with zf.open(member) as src, open(OMOP / name, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        ok, miss = omop_status()
        if not ok:
            log(f"OMOP   - !! tables still missing after unzipping: {', '.join(miss)}")
            return
    log("OMOP   - conversion to parquet (reprodICU cache)")
    from reprodICU.helpers.helper_filepaths import OMOPPaths
    OMOPPaths(reprodicu_paths())


def prepare_parquet(datasets: list[str]) -> None:
    """Instantiates the reprodICU *Paths classes, which convert the large CSV files to
    parquet (the same code as during the build, which then reuses these files)."""
    from reprodICU.helpers import helper_filepaths as hf
    classes = {"eICU": hf.EICUPaths, "MIMIC3": hf.MIMIC3Paths, "MIMIC4": hf.MIMIC4Paths,
               "SICdb": hf.SICdbPaths, "UMCdb": hf.UMCdbPaths}
    paths = reprodicu_paths()
    for name in datasets:
        if name not in classes:
            continue
        log(f"{name:6} - converting the large CSV files to parquet (skipped if already done)")
        t0 = time.time()
        try:
            classes[name](paths)
        except Exception as e:  # go on with the other datasets
            log(f"{name:6} - !! conversion FAILED: {type(e).__name__}: {e}")
            continue
        log(f"{name:6} - ok ({duration(time.time() - t0)})")


def cmd_prepare(args) -> int:
    write_reprodicu_paths()
    header("Preparation of the raw data")
    steps = args.only or ["verify", "hirid", "sicdb", "umcdb", "omop", "parquet"]
    t0 = time.time()
    if "verify" in steps and not verify_checksums():
        log("!! Some files are corrupted: download them again before going on.")
        return 1
    if "hirid" in steps:
        prepare_hirid()
    if "umcdb" in steps:
        prepare_umcdb()
    if "omop" in steps:
        prepare_omop()
    if "parquet" in steps:
        prepare_parquet([n for n in ready_datasets() if n != "SICdb"])
    if "sicdb" in steps:  # the longest one last
        prepare_sicdb()
        if "parquet" in steps and dataset_status("SICdb")[0] == "READY":
            prepare_parquet(["SICdb"])
    log(f"Preparation done in {duration(time.time() - t0)}")
    return cmd_check(args)


# ----------------------------------------------------------------------------
# Build of the harmonized database
# ----------------------------------------------------------------------------
def _resolve_datasets(requested: list[str] | None) -> list[str]:
    ready = ready_datasets()
    if not requested:
        return ready
    bad = [r for r in requested if r not in ALL_DATASETS]
    if bad:
        sys.exit(f"!! Unknown dataset(s): {bad} (choices: {ALL_DATASETS})")
    not_ready = [r for r in requested if r not in ready]
    if not_ready:
        sys.exit(f"!! Dataset(s) not ready: {not_ready}, run ./run.sh check")
    return requested


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "log", "-1",
                               "--format=%h %ad %s", "--date=short"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def cmd_build(args) -> int:
    write_reprodicu_paths()
    ok, miss = omop_status()
    if not ok:
        print(f"!! OMOP vocabularies missing ({', '.join(miss)}). See README, then ./run.sh prepare")
        return 1
    datasets = _resolve_datasets(args.datasets)
    if not datasets:
        print("!! No dataset is ready. Run ./run.sh check")
        return 1
    steps = BUILD_STEPS
    if args.steps:
        steps = [s for s in BUILD_STEPS if s in args.steps]
    elif args.from_step:
        steps = BUILD_STEPS[BUILD_STEPS.index(args.from_step):]

    opts = CONFIG.get("build", {}) or {}
    winsorize, impute, resample = bool(opts.get("winsorize")), bool(opts.get("impute")), opts.get("resample")

    header(f"reprodICU build, datasets: {', '.join(datasets)}")
    log(f"Steps: {', '.join(steps)}")
    log(f"Output: {OUT}")

    from reprodICU import build as B
    paths = reprodicu_paths()
    # reprodICU sometimes writes conflict files to the current folder
    conflicts_dir = OUT / "_conflicts"
    conflicts_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(conflicts_dir)

    actions = {
        "patient_information": lambda: B.build_patient_information(
            paths=paths, datasets=datasets, winsorize=winsorize, add_availability=False),
        "diagnoses": lambda: B.build_diagnoses(paths=paths, datasets=datasets),
        "procedures": lambda: B.build_procedures(paths=paths, datasets=datasets),
        "medications": lambda: B.build_medications(paths=paths, datasets=datasets),
        "microbiology": lambda: B.build_microbiology(paths=paths, datasets=datasets),
        "notes": lambda: B.build_notes(paths=paths, datasets=datasets),
        "timeseries": lambda: B.build_timeseries(
            paths=paths, datasets=datasets, timeseries=None,
            winsorize=winsorize, impute=impute, resample=resample),
        "availability": lambda: B.add_patient_information_availability(paths=paths, impute=impute),
        "overview": lambda: B.build_overview(paths=paths),
    }
    timings = {}
    t_all = time.time()
    for step in steps:
        header(f"Step {BUILD_STEPS.index(step) + 1}/{len(BUILD_STEPS)}: {step}")
        if step == "notes" and not set(datasets) & NOTES_DATASETS:
            # reprodICU harmonizes notes for MIMIC-III and MIMIC-IV only
            log("notes: skipped (no dataset with notes among those requested)")
            continue
        t0 = time.time()
        actions[step]()
        timings[step] = round(time.time() - t0, 1)
        log(f"{step}: done in {duration(timings[step])}")
    log(f"Build done in {duration(time.time() - t_all)}")
    _write_manifest(datasets, steps, timings, opts)
    return cmd_summary(args)


def _write_manifest(datasets, steps, timings, opts) -> None:
    import polars as pl
    from importlib.metadata import distributions
    freeze = sorted({f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()},
                    key=str.lower)
    manifest_path = OUT / "MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    runs = previous.get("runs", [])
    runs.append({
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "steps": steps,
        "timings_seconds": timings,
        "build_options": opts,
        "reprodICU_commit": _git_commit(),
        "reprodICU_branch": _git_branch(),
        "python": sys.version.split()[0],
        "polars": pl.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
    })
    manifest = {
        "description": "Harmonized ICU database built with reprodICU: history of the builds",
        "last_run": runs[-1],
        "files": {f.name: {"size": f.stat().st_size,
                           "modified": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")}
                  for f in sorted(OUT.glob("*.parquet"))},
        "python_packages": freeze,
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log(f"Manifest written: {manifest_path}")


def cmd_concepts(args) -> int:
    write_reprodicu_paths()
    # The reprodICU MAGIC_CONCEPTS read the six datasets below, whatever the list given.
    missing = [x for x in CONCEPT_DATASETS if x not in ready_datasets()]
    if missing:
        print("!! The reprodICU concepts (ventilation durations, scores, code status...) "
              f"need all of these datasets: {', '.join(CONCEPT_DATASETS)}.\n"
              f"   Missing: {', '.join(missing)}. Rerun when they are ready.")
        return 1
    datasets = [x for x in _resolve_datasets(args.datasets) if x in CONCEPT_DATASETS]
    header(f"reprodICU MAGIC_CONCEPTS, datasets: {', '.join(datasets)}")
    from reprodICU import build as B
    os.chdir(OUT)
    t0 = time.time()
    B.build_magic_concepts(paths=reprodicu_paths(), datasets=datasets, concepts=args.concepts or None)
    log(f"Concepts done in {duration(time.time() - t0)}")
    return 0


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def cmd_summary(args) -> int:
    import polars as pl
    header("Summary of the harmonized database")
    pi = OUT / "patient_information.parquet"
    if not pi.exists():
        print("  patient_information.parquet missing, run ./run.sh build")
        return 0
    lf = pl.scan_parquet(pi)
    cols = lf.collect_schema().names()
    src, stay, person = "Source Dataset", "Global ICU Stay ID", "Global Person ID"
    aggs = [pl.len().alias("ICU stays")]
    if person in cols:
        aggs.append(pl.col(person).drop_nulls().n_unique().alias("persons"))
        aggs.append(pl.col(person).null_count().alias("no person ID"))
    with pl.Config(tbl_rows=20, tbl_hide_dataframe_shape=True, tbl_formatting="ASCII_MARKDOWN"):
        print(lf.group_by(src).agg(aggs).sort(src).collect())

    print("\nRows per table and dataset (prefix of the Global ICU Stay ID):")
    parts = []
    for f in sorted(OUT.glob("*.parquet")):
        if f.name.startswith("overview"):
            continue
        try:
            t = pl.scan_parquet(f)
            if stay not in t.collect_schema().names():
                continue
            parts.append(
                t.select(pl.col(stay).cast(pl.String).str.split("-").list.first()
                         .fill_null("(no ID)").alias("dataset"))
                .group_by("dataset").len()
                .with_columns(pl.lit(f.stem).alias("table"))
                .collect())
        except Exception as e:  # best-effort summary
            print(f"  {f.stem}: unreadable ({str(e)[:60]})")
    if parts:
        wide = (pl.concat(parts)
                .pivot(on="dataset", index="table", values="len")
                .fill_null(0).sort("table"))
        with pl.Config(tbl_rows=40, tbl_cols=12, tbl_hide_dataframe_shape=True,
                       tbl_formatting="ASCII_MARKDOWN", tbl_width_chars=200):
            print(wide)
    return 0


# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Builds the harmonized ICU database with reprodICU.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="status of the data and the outputs")

    p = sub.add_parser("prepare", help="one-off preparation of the raw data")
    p.add_argument("--only", nargs="+",
                   choices=["verify", "hirid", "sicdb", "umcdb", "omop", "parquet"],
                   help="only these steps")

    p = sub.add_parser("verify", help="checks the downloads (PhysioNet SHA256SUMS)")
    p.add_argument("--force", action="store_true", help="check again even if already done")

    p = sub.add_parser("build", help="builds the harmonized tables")
    p.add_argument("--datasets", nargs="+", help=f"subset of datasets ({' '.join(ALL_DATASETS)})")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--steps", nargs="+", choices=BUILD_STEPS, help="only these steps")
    g.add_argument("--from-step", choices=BUILD_STEPS, help="resume from this step")

    p = sub.add_parser("concepts", help="derived concepts (ventilation, scores...)")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--concepts", nargs="+",
                   help="e.g. VENTILATION_DURATION SEVERITY_SCORES (default: all)")

    sub.add_parser("summary", help="counts per dataset in the outputs")

    args = parser.parse_args()
    return {"check": cmd_check, "prepare": cmd_prepare, "build": cmd_build,
            "verify": lambda a: 0 if verify_checksums(force=a.force) else 1,
            "concepts": cmd_concepts, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
