import sys
import re
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import obspy
import pandas as pd
from obspy.signal.trigger import ar_pick
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

sys.path.append("..")
from utils.read_tsmip import get_integrated_stream, get_integrated_stream_second, get_peak_value

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "2025_final_catalog_demo.csv"
TRACES_PATH = PROJECT_ROOT / "data" / "processed" / "2025_final_traces_demo.csv"
WAVEFORM_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_HDF5 = PROJECT_ROOT / "data" / "processed" / "TSMIP_2025_demo.hdf5"
STATION_TRANS_PATH = PROJECT_ROOT / "data" / "raw" / "TSMIP-Instrument-parameters-20240821.txt"

TARGET_SAMPLING_RATE = 200

try:
    station_trans_info = pd.read_csv(STATION_TRANS_PATH, sep="\t")
    station_trans_info.drop(['安裝經度', '安裝緯度', '安裝高程', '啟用日期', '停用日期', 'Z軸極性', 'N軸極性', 'E軸極性', '感測器型號'], axis=1, inplace=True)
    station_trans_info.columns = ["Z_trans", "N_trans", "E_trans", "station_code"]
    station_trans_info = station_trans_info.loc[:675]
    print(f"✓ Loaded station transformation info: {len(station_trans_info)} stations")
except Exception as e:
    print(f"⚠ Warning: Could not load station transformation info: {e}")
    station_trans_info = None

def build_waveform_folder(event_row: pd.Series) -> str:
    """Build waveform folder name from event timestamp fields."""
    year = int(event_row["year"])
    month = int(event_row["month"])
    day = int(event_row["day"])
    hour = int(event_row["hour"])
    minute = int(event_row["minute"])
    second = int(float(event_row["second"]))
    return f"{year}_{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}"


def find_sac_file(waveform_path: str, station_code: str, component: str) -> str:
    """Find the SAC file path for a station component in a waveform folder."""
    pattern = f"*.TW.{station_code}.10.HL{component}.D.SAC"
    matches = list(Path(waveform_path).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Missing SAC file for station {station_code} component {component} in {waveform_path}"
        )
    return str(matches[0])


def extract_station_code_from_sac(filename: str) -> str:
    """Extract station code from SAC filename pattern."""
    match = re.search(r'\.TW\.([A-Za-z0-9]+)\.10', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract station code from {filename} using pattern .TW.{{station_code}}.10")


def get_unique_stations_from_waveform_folder(waveform_path: str) -> set:
    """Return unique station codes from all SAC files in a waveform folder."""
    sac_files = list(Path(waveform_path).glob("*.SAC"))
    stations = set()
    for sac_file in sac_files:
        try:
            station_code = extract_station_code_from_sac(sac_file.name)
            stations.add(station_code)
        except:
            continue
    return stations


def read_station_stream(waveform_path: str, station_code: str) -> obspy.Stream:
    """Read Z, N, and E station components and apply instrument scaling when available."""
    z_file = find_sac_file(waveform_path, station_code, "Z")
    n_file = find_sac_file(waveform_path, station_code, "N")
    e_file = find_sac_file(waveform_path, station_code, "E")

    trace_z = obspy.read(z_file)
    trace_n = obspy.read(n_file)
    trace_e = obspy.read(e_file)

    if station_trans_info is not None:
        trans_rows = station_trans_info[station_trans_info["station_code"] == station_code]
        if len(trans_rows) > 0:
            z_trans = trans_rows["Z_trans"].values[0]
            n_trans = trans_rows["N_trans"].values[0]
            e_trans = trans_rows["E_trans"].values[0]

            trace_z[0].data = trace_z[0].data * z_trans
            trace_n[0].data = trace_n[0].data * n_trans
            trace_e[0].data = trace_e[0].data * e_trans

    stream = obspy.core.stream.Stream()
    stream.append(trace_z[0])
    stream.append(trace_n[0])
    stream.append(trace_e[0])
    return stream


def resample_stream(stream: obspy.Stream, target_rate: int) -> obspy.Stream:
    """Resample stream when sampling rate differs from target rate."""
    sampling_rate = stream[0].stats.sampling_rate
    if sampling_rate != target_rate:
        stream = stream.copy()
        stream.resample(target_rate, window="hann")
    return stream

def process_event(
    eq_id: int,
    catalog_df: pd.DataFrame,
    traces_df: pd.DataFrame,
    waveform_root: Path,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Process one event and return waveform tensors with aligned trace metadata."""
    event_row = catalog_df[catalog_df["EQ_ID"] == eq_id].iloc[0]
    waveform_folder = build_waveform_folder(event_row)
    waveform_path = Path(waveform_root) / waveform_folder

    print(f"Looking for waveforms in: {waveform_path}")
    print(f"Path exists: {waveform_path.exists()}")

    if not Path(waveform_path).exists():
        raise ValueError(f"Waveform folder not found: {waveform_path}")

    stations_in_folder = get_unique_stations_from_waveform_folder(str(waveform_path))

    if not stations_in_folder:
        raise ValueError(f"No SAC files found in {waveform_path}")

    print(f"Found {len(stations_in_folder)} stations in waveform folder for EQ_ID {eq_id}")

    acc_traces = []
    vel_traces = []
    vel_lowfreq_traces = []
    dis_traces = []
    pd_traces = []
    cvav_traces = []
    tp_traces = []
    p_picks = []
    pgv = []
    pgv_time = []
    station_name = []
    start_time = []
    station_location = []
    vs30_list = []
    station_code_list = []

    traces_for_event = traces_df[traces_df["EQ_ID"] == eq_id].copy()

    traces_for_event = traces_for_event[
        (traces_for_event["station_code"].isin(stations_in_folder)) &
        (traces_for_event["latitude"].notna()) &
        (traces_for_event["longitude"].notna()) &
        (traces_for_event["elevation"].notna()) &
        (traces_for_event["Vs30"].notna())
    ]

    if len(traces_for_event) == 0:
        raise ValueError(f"No valid stations found for EQ_ID {eq_id}")

    print(f"Processing {len(traces_for_event)} stations for EQ_ID {eq_id}")

    temp_p_picks_list = []
    for _, trace_row in traces_for_event.iterrows():
        station_code = trace_row["station_code"]
        try:
            stream = read_station_stream(str(waveform_path), station_code)
            stream = resample_stream(stream, TARGET_SAMPLING_RATE)

            try:
                p_pick, _ = ar_pick(
                    stream[0].data,
                    stream[1].data,
                    stream[2].data,
                    samp_rate=TARGET_SAMPLING_RATE,
                    f1=1,
                    f2=20,
                    lta_p=1,
                    sta_p=0.1,
                    lta_s=4.0,
                    sta_s=1.0,
                    m_p=2,
                    m_s=8,
                    l_p=0.1,
                    l_s=0.2,
                    s_pick=False,
                )
                p_pick_sample = int(round(p_pick * TARGET_SAMPLING_RATE))
            except:
                p_pick_sample = -1

            temp_p_picks_list.append((station_code, p_pick_sample))
        except Exception:
            continue

    valid_p_picks = [p for _, p in temp_p_picks_list if p >= 0]
    if len(valid_p_picks) == 0:
        raise ValueError(f"Could not detect P-wave for any station in EQ_ID {eq_id}")

    min_p_pick = min(valid_p_picks)

    window_start = max(0, min_p_pick - 5 * TARGET_SAMPLING_RATE)
    window_end = min_p_pick + 25 * TARGET_SAMPLING_RATE
    window_length = window_end - window_start

    print(f"Time window: samples [{window_start}, {window_end}] (length: {window_length} samples = {window_length/TARGET_SAMPLING_RATE:.2f} sec)")

    for _, trace_row in traces_for_event.iterrows():
        station_code = trace_row["station_code"]

        try:
            stream = read_station_stream(str(waveform_path), station_code)
            stream = resample_stream(stream, TARGET_SAMPLING_RATE)

            try:
                p_pick, _ = ar_pick(
                    stream[0].data,
                    stream[1].data,
                    stream[2].data,
                    samp_rate=TARGET_SAMPLING_RATE,
                    f1=1,
                    f2=20,
                    lta_p=1,
                    sta_p=0.1,
                    lta_s=4.0,
                    sta_s=1.0,
                    m_p=2,
                    m_s=8,
                    l_p=0.1,
                    l_s=0.2,
                    s_pick=False,
                )
                p_pick_sample = int(round(p_pick * TARGET_SAMPLING_RATE))
            except:
                p_pick_sample = -1

            acc_trace_windowed = np.transpose(np.array(stream))
            acc_trace_windowed = acc_trace_windowed[window_start:window_end, :]

            vel_stream = get_integrated_stream(stream)
            vel_trace_windowed = np.transpose(np.array(vel_stream))
            vel_trace_windowed = vel_trace_windowed[window_start:window_end, :]
            vel_peak, vel_peak_time = get_peak_value(vel_stream)

            vel_stream_lowfreq = vel_stream.copy()
            vel_stream_lowfreq.filter('lowpass', freq=0.33)
            vel_lowfreq_windowed = np.transpose(np.array(vel_stream_lowfreq))
            vel_lowfreq_windowed = vel_lowfreq_windowed[window_start:window_end, :]

            dis_stream = get_integrated_stream_second(vel_stream)
            dis_trace_windowed = np.transpose(np.array(dis_stream))
            dis_trace_windowed = dis_trace_windowed[window_start:window_end, :]

            dis_abs = np.abs(dis_trace_windowed[:, 0])
            pd = np.maximum.accumulate(dis_abs)
            pd = np.expand_dims(pd, axis=1)

            cvav = np.add.accumulate(np.abs(vel_trace_windowed[:, 0]))
            cvav = np.expand_dims(cvav, axis=1)

            sample_rate = TARGET_SAMPLING_RATE

            vel_all = np.linalg.norm(vel_trace_windowed, axis=1)
            dis_all = np.linalg.norm(dis_trace_windowed, axis=1)

            vel_int = cumulative_trapezoid(vel_all, dx=1 / sample_rate, initial=0)
            dis_int = cumulative_trapezoid(dis_all, dx=1 / sample_rate, initial=0)

            eps = 1e-10
            ratio = vel_int / (dis_int + eps)
            tau_c = (2 * np.pi) / (np.sqrt(ratio) + eps)
            tp = tau_c * pd[:, 0]
            tp = np.expand_dims(tp, axis=1)

            station_code_list.append(station_code)
            station_name.append(trace_row["station_name"])
            station_location.append([trace_row["latitude"], trace_row["longitude"], trace_row["elevation"]])
            vs30_list.append(trace_row["Vs30"])
            start_time.append(str(stream[0].stats.starttime))
            p_picks.append(p_pick_sample - window_start if p_pick_sample >= 0 else -1)
            pgv.append(vel_peak)
            pgv_time.append(vel_peak_time)

            acc_traces.append(acc_trace_windowed)
            vel_traces.append(vel_trace_windowed)
            vel_lowfreq_traces.append(vel_lowfreq_windowed)
            dis_traces.append(dis_trace_windowed)
            pd_traces.append(pd)
            cvav_traces.append(cvav)
            tp_traces.append(tp)

        except Exception as e:
            print(f"Skipping station {station_code}: {e}")
            continue

    if len(acc_traces) == 0:
        raise ValueError(f"No valid stations found for EQ_ID {eq_id}")

    print(f"Successfully processed {len(acc_traces)} stations for EQ_ID {eq_id}")

    sort_indices = np.argsort(np.asarray(p_picks, dtype=np.int64))

    event_output = {
        "acc_traces": [acc_traces[i] for i in sort_indices],
        "vel_traces": [vel_traces[i] for i in sort_indices],
        "vel_lowfreq_traces": [vel_lowfreq_traces[i] for i in sort_indices],
        "dis_traces": [dis_traces[i] for i in sort_indices],
        "pd": [pd_traces[i] for i in sort_indices],
        "cvav": [cvav_traces[i] for i in sort_indices],
        "TP": [tp_traces[i] for i in sort_indices],
        "p_picks": np.asarray([p_picks[i] for i in sort_indices], dtype=np.int64),
        "pgv": np.asarray([pgv[i] for i in sort_indices], dtype=np.float64),
        "pgv_time": np.asarray([pgv_time[i] for i in sort_indices], dtype=np.int64),
        "station_name": np.array([station_name[i].encode('utf-8') if isinstance(station_name[i], str) else station_name[i] for i in sort_indices], dtype='S50'),
        "start_time": np.array([start_time[i].encode('utf-8') if isinstance(start_time[i], str) else start_time[i] for i in sort_indices], dtype='S50'),
        "station_location": np.asarray([station_location[i] for i in sort_indices], dtype=np.float64),
        "Vs30": np.asarray([vs30_list[i] for i in sort_indices], dtype=np.float64),
    }

    processed_stations = [station_name[i] for i in sort_indices]
    event_traces_metadata = traces_df[(traces_df["EQ_ID"] == eq_id) & (traces_df["station_name"].isin(processed_stations))].copy()

    event_traces_metadata = event_traces_metadata.set_index('station_name').loc[processed_stations].reset_index()

    event_traces_metadata["p_pick_sample"] = [p_picks[i] for i in sort_indices]
    event_traces_metadata["start_time"] = [start_time[i] for i in sort_indices]

    print(f"Metadata shape: {event_traces_metadata.shape}")

    return event_output, event_traces_metadata

def write_event_group(h5_file: h5py.File, eq_id: int, event_data: Dict[str, np.ndarray]) -> None:
    """Write one event dataset group into the HDF5 output file."""
    data_group = h5_file.require_group("data")
    event_group = data_group.create_group(str(eq_id))

    acc_traces_list = event_data["acc_traces"]
    vel_traces_list = event_data["vel_traces"]
    vel_lowfreq_traces_list = event_data["vel_lowfreq_traces"]
    dis_traces_list = event_data["dis_traces"]
    pd_list = event_data["pd"]
    cvav_list = event_data["cvav"]
    tp_list = event_data["TP"]

    max_length = max(acc.shape[0] for acc in acc_traces_list)
    print(f"Max waveform length: {max_length} samples ({max_length/200:.2f} seconds)")

    acc_traces_padded = []
    vel_traces_padded = []
    vel_lowfreq_traces_padded = []
    dis_traces_padded = []
    pd_padded = []
    cvav_padded = []
    tp_padded = []

    for acc_trace, vel_trace, vel_lf_trace, dis_trace, pd_trace, cvav_trace, tp_trace in zip(
        acc_traces_list,
        vel_traces_list,
        vel_lowfreq_traces_list,
        dis_traces_list,
        pd_list,
        cvav_list,
        tp_list,
    ):
        orig_length = acc_trace.shape[0]

        if orig_length < max_length:
            acc_padded = np.pad(acc_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
            vel_padded = np.pad(vel_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
            vel_lf_padded = np.pad(vel_lf_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
            dis_padded = np.pad(dis_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
            pd_pad = np.pad(pd_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
            cvav_pad = np.pad(cvav_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
            tp_pad = np.pad(tp_trace, ((0, max_length - orig_length), (0, 0)), mode='constant')
        else:
            acc_padded = acc_trace
            vel_padded = vel_trace
            vel_lf_padded = vel_lf_trace
            dis_padded = dis_trace
            pd_pad = pd_trace
            cvav_pad = cvav_trace
            tp_pad = tp_trace

        acc_traces_padded.append(acc_padded)
        vel_traces_padded.append(vel_padded)
        vel_lowfreq_traces_padded.append(vel_lf_padded)
        dis_traces_padded.append(dis_padded)
        pd_padded.append(pd_pad)
        cvav_padded.append(cvav_pad)
        tp_padded.append(tp_pad)

    acc_array = np.asarray(acc_traces_padded, dtype=np.float64)
    vel_array = np.asarray(vel_traces_padded, dtype=np.float64)
    vel_lowfreq_array = np.asarray(vel_lowfreq_traces_padded, dtype=np.float64)
    dis_array = np.asarray(dis_traces_padded, dtype=np.float64)
    pd_array = np.asarray(pd_padded, dtype=np.float64)
    cvav_array = np.asarray(cvav_padded, dtype=np.float64)
    tp_array = np.asarray(tp_padded, dtype=np.float64)

    event_group.create_dataset("acc_traces", data=acc_array, dtype=np.float64)
    event_group.create_dataset("vel_traces", data=vel_array, dtype=np.float64)
    event_group.create_dataset("vel_lowfreq_traces", data=vel_lowfreq_array, dtype=np.float64)
    event_group.create_dataset("dis_traces", data=dis_array, dtype=np.float64)
    event_group.create_dataset("pd", data=pd_array, dtype=np.float64)
    event_group.create_dataset("cvav", data=cvav_array, dtype=np.float64)
    event_group.create_dataset("TP", data=tp_array, dtype=np.float64)

    event_group.create_dataset("p_picks", data=event_data["p_picks"], dtype=np.int64)
    event_group.create_dataset("pgv", data=event_data["pgv"], dtype=np.float64)
    event_group.create_dataset("pgv_time", data=event_data["pgv_time"], dtype=np.int64)

    event_group.create_dataset("station_name", data=event_data["station_name"], dtype='S50')
    event_group.create_dataset("start_time", data=event_data["start_time"], dtype='S50')
    
    event_group.create_dataset("station_location", data=event_data["station_location"], dtype=np.float64)
    event_group.create_dataset("Vs30", data=event_data["Vs30"], dtype=np.float64)

def main() -> None:
    """Process all events and export waveforms with metadata to HDF5."""
    catalog_df = pd.read_csv(CATALOG_PATH)
    traces_df = pd.read_csv(TRACES_PATH)

    error_events = []
    processed_eq_ids = []
    processed_traces = []

    output_path = Path(OUTPUT_HDF5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            output_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete old file: {e}")

    with h5py.File(output_path, "w") as h5_file:
        h5_file.require_group("data")

        for eq_id in tqdm(catalog_df["EQ_ID"].tolist(), desc="Processing events"):
            try:
                print(f"\n=== Processing EQ_ID {eq_id} ===")
                event_data, event_traces = process_event(
                    eq_id, catalog_df, traces_df, WAVEFORM_ROOT
                )
                print(f"Event data keys: {event_data.keys()}")
                print(f"Number of stations: {len(event_data['station_name'])}")
                write_event_group(h5_file, eq_id, event_data)
                processed_eq_ids.append(eq_id)
                processed_traces.append(event_traces)
                print(f"Successfully wrote EQ_ID {eq_id} to HDF5")
            except Exception as exc:
                print(f"ERROR processing EQ_ID {eq_id}: {exc}")
                import traceback
                traceback.print_exc()
                error_events.append({"EQ_ID": eq_id, "reason": str(exc)})
                continue

    if processed_traces:
        traces_metadata = pd.concat(processed_traces, ignore_index=True)

        if "p_pick_sec" in traces_metadata.columns:
            traces_metadata["p_pick_sec"] = traces_metadata["p_pick_sec"].astype(str)
        if "p_arrival_abs_time" in traces_metadata.columns:
            traces_metadata["p_arrival_abs_time"] = traces_metadata["p_arrival_abs_time"].astype(str)
        if "start_time" in traces_metadata.columns:
            traces_metadata["start_time"] = traces_metadata["start_time"].astype(str)

        traces_metadata.to_hdf(output_path, key="metadata/traces_metadata", mode="a", format="table")
    
    event_metadata = catalog_df[catalog_df["EQ_ID"].isin(processed_eq_ids)].copy()
    if len(event_metadata) > 0:
        event_metadata.to_hdf(output_path, key="metadata/event_metadata", mode="a", format="table")

    print(f"Output HDF5: {output_path}")
    print(f"Processed events: {len(processed_eq_ids)}")
    print(f"Error events: {len(error_events)}")


if __name__ == "__main__":
    main()
