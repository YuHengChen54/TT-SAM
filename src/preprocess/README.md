# Data Preprocessing Pipeline

This directory contains the preprocessing scripts used to transform raw seismic observations into the structured HDF5 datasets required for **TT-SAM**.

The content below is aligned with the current code implementation and I/O behavior.

---

## Data Acquisition & Sources

As raw seismic data cannot be redistributed directly via GitHub, users must obtain the following files from the respective authorities:

### 1. Seismic Catalog & Station List (via [CWA GDMS](https://gdms.cwa.gov.tw/))
* **`data/raw/GDMScatalog.csv`**: Downloaded from **Data** → **Event Catalog** on the GDMS website.
* **`data/raw/GDMSstations.csv`**: Downloaded from **Station** on the GDMS website.
* **Raw Waveforms**: Downloaded via the **"Multi-station Waveform Data"** function. 
    * **Time Window**: For each event in the catalog, request a 3-minute segment (from 15s before to 165s after the origin time). Refer to the `origin time` provided in the downloaded `GDMScatalog.csv` for each event.
    * **Naming Convention**: Folders should be named `YYYY_MMDD_HHMMSS`.
    * **⚠️ IMPORTANT: Instrument Response Correction**: 
      The scripts in this repository **do not** include instrument response correction. Users must perform this step independently before running the preprocessing pipeline to ensure data accuracy. Please refer to the [GDMS Technical Documentation](https://gdms.cwa.gov.tw/help.php) for guidelines on instrument correction for TSMIP data.

### 2. Site Conditions ($V_{S30}$) (via [NCREE EGDT](https://egdt.ncree.org.tw/))
* **`data/processed/GDMSstations_Vs30.csv`**: This file is an extension of the GDMS station list. $V_{S30}$ values were obtained by applying to the **Engineering Geological Database for TSMIP (EGDT)** maintained by the National Center for Research on Earthquake Engineering (NCREE).

---

## Workflow

> **Note:** Please ensure that the `data/raw/` and `data/processed/` directories exist before running the scripts.

Run the scripts in this order:

1. `convert_gdms_catalog.py`
2. `generate_tracer_catalog.py`
3. `cut_trace_to_hdf5.py`

---

## 1) `convert_gdms_catalog.py`

### Input
- `data/raw/GDMScatalog.csv`
- `data/raw/GDMSstations.csv`

### Main logic
- Sorts events by `date + time`.
- Converts decimal latitude/longitude to degree + minute fields.
- Computes nearest station distance (`nearest_sta_dist (km)`) with haversine distance.
- Assigns sequential `EQ_ID`.

### Output
- `data/processed/2025_final_catalog_demo.csv`

---

## 2) `generate_tracer_catalog.py`

### Input
- `data/raw/GDMScatalog.csv`
- `data/processed/GDMSstations_Vs30.csv`

### Main logic
- Sorts events by `date + time`.
- Expands each event to all stations in `GDMSstations_Vs30.csv`.
- Parses station code/name from `station_info`.
- Computes:
  - Epicentral distance `epdis (km)`
  - Azimuth `sta_angle`
- Writes per-event-per-station trace metadata, including `Vs30`.

### Output
- `data/processed/2025_final_traces_demo.csv`

---

## 3) `cut_trace_to_hdf5.py`

> **Dependency:** This script requires `obspy` for reading SAC files and performing P-wave detection.

### Input
- `data/processed/2025_final_catalog_demo.csv`
- `data/processed/2025_final_traces_demo.csv`
- Raw waveform folders under `data/raw/`

### Waveform assumptions
- SAC naming pattern:
  - `*.TW.{station_code}.10.HLZ.D.SAC`
  - `*.TW.{station_code}.10.HLN.D.SAC`
  - `*.TW.{station_code}.10.HLE.D.SAC`
- Event folder naming rule from catalog timestamp:
  - `YYYY_MMDD_HHMMSS`

### Main logic
- Reads 3-component traces (Z/N/E) per station.
- Resamples to `TARGET_SAMPLING_RATE = 200` Hz when needed.
- Detects P-picks with `obspy.signal.trigger.ar_pick`.
- Uses earliest valid P-pick in event as reference.
- Cuts window: `[-5s, +25s]` relative to that earliest pick (target 30s window).
- Builds derived channels:
  - `vel_traces`: integrated + bandpass (via `get_integrated_stream`)
  - `vel_lowfreq_traces`: lowpass filtered velocity branch (`freq=0.33`)
  - `dis_traces`: second integration + highpass
  - physical features:
    - `pd`: Peak displacement (Pd)
    - `cvav`: Cumulative Vertical Absolute Velocity (CVAV)
    - `TP`: Predominant Period
- Computes label and timing:
  - `pgv` (log10-scaled peak from vector norm)
  - `pgv_time`
- Sorts stations by P-pick order.

### Output
- `data/processed/TSMIP_2025_demo.hdf5`

**HDF5 structure (per event):**
- Group: `data/{EQ_ID}/`
- Datasets: `acc_traces`, `vel_traces`, `vel_lowfreq_traces`, `dis_traces`, `pd`, `cvav`, `TP`, `p_picks`, `pgv`, `pgv_time`, `station_name`, `start_time`, `station_location`, `Vs30`.
- Metadata (Pandas tables): `metadata/event_metadata`, `metadata/traces_metadata`.

---

## Important Notes (Scope)

- **Time masking** (e.g., 3s/5s/7s...) is not done in these preprocessing scripts; it is applied later in dataset loading/training logic. This ensures the HDF5 dataset remains flexible and can support any time window from 3s to 15s during testing.
- **Magnitude/event oversampling** is also handled in dataset/training stage, not in `src/preprocess`.

---

## Acknowledgements & Citations

If you use the data processed by these scripts in your research, please adhere to the following citation requirements from the data providers:

### CWA GDMS (Mandatory)
> "The authors appreciate the Central Weather Administration (CWA) for providing access to the Geophysical Data Management System (GDMS) at https://gdms.cwa.gov.tw."

**Reference:**
Central Weather Administration (CWA, Taiwan). (2012). *Central Weather Administration Seismographic Network* [Data set]. International Federation of Digital Seismograph Networks. https://doi.org/10.7914/SN/T5

### NCREE EGDT
> "The $V_{S30}$ data were provided by the Engineering Geological Database for TSMIP (EGDT) of the National Center for Research on Earthquake Engineering (NCREE), Taiwan."
