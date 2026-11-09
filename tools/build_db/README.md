# build_db: build the harmonized database in a few commands

A small wrapper around the reprodICU build functions. It sets up a Python
environment, writes `~/.reprodICU/PATHS.yaml` from one configuration file, checks and
prepares the raw downloads, then runs the build and the derived concepts, with a
timestamped log of every run. It was used to build the seven datasets (eICU, HiRID,
MIMIC-III, MIMIC-IV, NWICU, SICdb, AmsterdamUMCdb) end to end.

It only calls reprodICU code from this repository (installed in editable mode), so
the build always reflects the checked-out branch.

## Files of the tool

| File | Role | Do you edit it? |
| --- | --- | --- |
| `run.sh` | Entry point: Python environment, then `build_db.py` with a log | No |
| `build_db.py` | All the commands (`check`, `prepare`, `build`...) | No, unless a dataset version changes |
| `config.yaml` | Default configuration (paths, datasets, build options), tracked in git | No, copy it to `config.local.yaml` |
| `config.local.yaml` | Your own configuration, ignored by git, used instead of `config.yaml` | Yes, this is the one to edit |
| `requirements.txt` | Python packages of the environment | Only to change a package version |
| `.gitignore` | Keeps `.venv/`, `logs/` and `config.local.yaml` out of git | No |

### run.sh

The only command you need: `./run.sh <command> [options]`, from any folder.

1. Creates `tools/build_db/.venv/` on the first run, with uv (Python 3.12) if it is
   installed, otherwise with the first `python3.12`, `python3.11` or `python3.10`
   found. It installs `requirements.txt` and reprodICU from this repository in
   editable mode (`pip install -e`), so code changes in `src/reprodICU` are used
   without reinstalling.
2. Reinstalls the packages whenever `requirements.txt` is newer than the
   environment. To start from a clean environment, delete `.venv/`.
3. Runs `build_db.py` with the arguments given and copies everything it prints to
   `logs/<date>_<command>.log`. The exit code is the one of `build_db.py`.

Nothing to configure here.

### build_db.py

Does the actual work, one subcommand per stage (see [Commands](#commands)). At every
run it:

- reads `config.local.yaml` if it exists, otherwise `config.yaml`;
- turns the paths into absolute paths and writes `~/.reprodICU/PATHS.yaml`, the file
  reprodICU reads its paths from (so you never edit `PATHS.yaml` by hand);
- checks for each dataset that the files expected by reprodICU are there.

It is configured through the YAML files only. The one thing set in the script is the
folder of each dataset version (`DATASETS` near the top, for example
`physionet.org/files/mimiciv/3.1`), which follows the versions supported by
reprodICU: edit it only if you use another version.

### config.yaml and config.local.yaml

`config.yaml` is the template kept in git. Copy it to `config.local.yaml` and edit the
copy: when `config.local.yaml` exists it is read instead of `config.yaml` (the whole
file, the two are not merged), and git ignores it, so your paths stay on your
machine.

```yaml
paths:
  raw_data: /Volumes/ICU/raw                  # PhysioNet downloads and UMCdb/ (layout below)
  omop_vocabulary: /Volumes/ICU/OMOP          # Athena vocabularies (zip or CSV files)
  output: /Volumes/ICU/reprodICU              # harmonized database (parquet)
  output_demo: /Volumes/ICU/reprodICU_demo    # reprodICU demo mode, not used by this tool

datasets:          # false leaves a dataset out of check, prepare and build
  eICU: true
  HiRID: true
  MIMIC3: true
  MIMIC4: true
  NWICU: false
  SICdb: true
  UMCdb: true

build:             # passed as is to the reprodICU build functions
  winsorize: false # clip extreme values (labs, height and weight)
  impute: false    # impute vital signs
  resample: null   # resample vital signs, in seconds (e.g. 3600), null = no resampling
```

- Paths can be absolute, start with `~`, or be relative to the root of the
  repository. The defaults point to a `reprodICU-data` folder next to the repository.
  If a data folder is inside the repository anyway, the tool adds a `.gitignore` to it
  so that it is never committed.
- `paths` is required. A dataset missing from `datasets` counts as `true`; a missing
  `build` section means the three defaults above.
- A dataset left out can be added later: set it to `true`, then `./run.sh prepare`
  and `./run.sh build`.
- `./run.sh check` prints which configuration file is in use.

### requirements.txt

Packages installed in `.venv/` next to reprodICU. `polars` is pinned to the version
the seven datasets were built with (1.41.2 crashed with a bus error on the HiRID
medication step, 1.44.2 builds them all). After a change, the next `./run.sh`
reinstalls the environment.

## Requirements

- macOS or Linux, with `bash`, `tar` and `git`.
- [uv](https://docs.astral.sh/uv/) (recommended, creates a Python 3.12 environment)
  or Python 3.10 or later.
- Enough disk space for the raw data, the parquet cache and the outputs (several
  hundred GB for the seven datasets).

## Quick start

```bash
cd tools/build_db
cp config.yaml config.local.yaml   # then edit the paths (optional)
./run.sh check                     # what is there, what is missing, next step
./run.sh prepare                   # one-off preparation of the raw data
./run.sh build                     # harmonized tables
./run.sh concepts                  # derived concepts (ventilation, scores...)
```

The first run creates `.venv/` (and updates it whenever `requirements.txt` changes).
Every command writes its output to `logs/<date>_<command>.log`. Every command can be
rerun: finished steps are detected and skipped.

## Expected layout of the raw data

PhysioNet datasets are expected where `wget` puts them when it is run from the
`raw_data` folder:

```bash
cd <raw_data>
wget -r -N -c -np --user <physionet-user> --ask-password https://physionet.org/files/mimiciv/3.1/
```

```
<raw_data>/
  physionet.org/files/
    eicu-crd/2.0/                     *.csv.gz
    hirid/1.1.1/                      reference_data.tar.gz
                                      raw_stage/observation_tables_parquet.tar.gz
                                      raw_stage/pharma_records_parquet.tar.gz
    mimiciii/1.4/                     *.csv.gz
    mimiciv/3.1/                      hosp/ and icu/
    mimic-iv-note/2.2/                note/  (required for MIMIC-IV)
    nwicu-northwestern-icu/0.1.0/     data/nw_hosp/ and data/nw_icu/
    sicdb/1.0.8/                      *.csv.gz
  UMCdb/                              AmsterdamUMCdb files: admissions.csv, drugitems.csv,
                                      freetextitems.csv, listitems.csv, numericitems
                                      (.zip, .csv or .csv.gz), procedureorderitems.csv,
                                      processitems.csv
<omop_vocabulary>/                    Athena download (zip or unzipped CSV files)
```

`./run.sh check` lists, for each dataset, `READY`, `TO PREPARE`, `INCOMPLETE` (with
the missing files), `MISSING` or `DISABLED`, and the next command to run.

## Commands

| Command | What it does |
| --- | --- |
| `./run.sh check` | Status of every dataset, of the OMOP vocabularies and of the outputs |
| `./run.sh verify [--force]` | Checks the downloads against the PhysioNet `SHA256SUMS.txt` files |
| `./run.sh prepare [--only ...]` | One-off preparation, steps `verify hirid sicdb umcdb omop parquet` (all by default) |
| `./run.sh build [--datasets ...] [--steps ... \| --from-step ...]` | Builds the harmonized tables for the ready datasets |
| `./run.sh concepts [--datasets ...] [--concepts ...]` | reprodICU MAGIC_CONCEPTS |
| `./run.sh summary` | ICU stays and persons per dataset, rows per table and dataset |

`./run.sh --help` and `./run.sh <command> --help` give the details.

### prepare

- `verify`: SHA256 check of the PhysioNet downloads (once per dataset; corrupted files
  stop the preparation).
- `hirid`: extraction of the three HiRID archives.
- `umcdb`: re-encoding of the AmsterdamUMCdb files to UTF-8 (`reprodICU.setup.setup_umcdb`),
  written to `<raw_data>/UMCdb/ENCODINGFIX/`.
- `omop`: unzipping of the Athena archive, then conversion to parquet.
- `parquet`: conversion of the large CSV files to parquet by the reprodICU path
  classes (the files the build reads afterwards).
- `sicdb`: unpacking of `data_float_h` into `data_float_m` (`reprodICU.setup.setup_sicdb`),
  2 to 3 hours, run last.

### build

Steps, in order: `patient_information diagnoses procedures medications microbiology
notes timeseries availability overview`. Use `--steps` to run some of them or
`--from-step` to resume after an interruption. Per-dataset intermediate files are
cached by reprodICU in `<output>/_tempfiles/`, so adding a dataset means `prepare`
then `build`. Notes are only harmonized for MIMIC-III and MIMIC-IV.

### concepts

The reprodICU concepts read eICU, HiRID, MIMIC-III, MIMIC-IV, SICdb and AmsterdamUMCdb
(not NWICU): the command refuses to run until these six datasets are ready.

## Files written outside the output folder

- `~/.reprodICU/PATHS.yaml` is rewritten at every run from the configuration. The
  first time, an existing file is copied to `PATHS.yaml.bak-<date>`.
- Marker files `.build_done_<step>` in the raw data folders record the finished
  preparation steps (markers named `.optiflow_done_<step>`, from an earlier version
  of the tool, are recognised too).
- `<output>/MANIFEST.json`: history of the builds (date, datasets, steps, timings,
  options, reprodICU commit and branch, Python and package versions, machine).
- `<output>/_conflicts/`: working folder of the build, where reprodICU may write
  conflict files.
- `tools/build_db/.venv/` and `tools/build_db/logs/` (both ignored by git).
