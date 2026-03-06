from itertools import repeat

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class intensity_classifier:
    """Classify seismic intensity values into discrete categories based on thresholds."""

    def __init__(self, label=None):
        """Initialize intensity thresholds for the given label type."""
        if label == "pga":
            self.threshold = np.log10([0.008, 0.025, 0.08, 0.25, 0.8, 2.5, 8])
            self.label = [0, 1, 2, 3, 4, 5, 6, 7]
        if label == "pgv":
            self.threshold = np.log10([0.002, 0.007, 0.019, 0.057, 0.15, 0.5, 1.4])
            self.label = [0, 1, 2, 3, 4, 5, 6, 7]

    def classify(self, input_array):
        output_array = np.zeros_like(input_array)
        for i in range(len(input_array)):
            if input_array[i] < self.threshold[0]:
                output_array[i] = self.label[0]
            elif input_array[i] < self.threshold[1]:
                output_array[i] = self.label[1]
            elif input_array[i] < self.threshold[2]:
                output_array[i] = self.label[2]
            elif input_array[i] < self.threshold[3]:
                output_array[i] = self.label[3]
            elif input_array[i] < self.threshold[4]:
                output_array[i] = self.label[4]
            elif input_array[i] < self.threshold[5]:
                output_array[i] = self.label[5]
            elif input_array[i] < self.threshold[6]:
                output_array[i] = self.label[6]
            elif input_array[i] >= self.threshold[6]:
                output_array[i] = self.label[7]
        return output_array

class multiple_station_dataset(Dataset):
    """Load multi-station earthquake waveform data with preprocessing and augmentation."""

    def __init__(
        self,
        data_path,
        specific_event_metadata=None,
        sampling_rate=200,
        data_length_sec=30,
        test_year=2018,
        mode="train",
        limit=None,
        input_type="acc",
        label_key="pga",
        mask_waveform_sec=None,
        mask_waveform_random=False,
        dowmsampling=False,
        oversample=1,
        oversample_mag=4,
        max_station_num=25,
        label_target=25,
        sort_by_picks=True,
        oversample_by_labels=False,
        mag_threshold=0,
        part_small_event=False,
        weight_label=False,
        station_blind=False,
        bias_to_closer_station=False,
    ):
        if specific_event_metadata is not None:
            init_event_metadata = specific_event_metadata
        else:
            init_event_metadata = pd.read_hdf(data_path, "metadata/event_metadata")
        trace_metadata = pd.read_hdf(data_path, "metadata/traces_metadata")

        event_metadata = init_event_metadata[
            init_event_metadata["magnitude"] >= mag_threshold
        ]
        if part_small_event:
            small_event = init_event_metadata.query(
                f"magnitude < {mag_threshold} & year!={test_year}"
            ).sample(frac=0.25, random_state=0)
            event_metadata = pd.concat([event_metadata, small_event])

        if mode == "train":
            event_test_mask = [
                int(year) != test_year for year in event_metadata["year"]
            ]
            trace_test_mask = [
                int(year) != test_year for year in trace_metadata["year"]
            ]
            event_metadata = event_metadata[event_test_mask]
            trace_metadata = trace_metadata[trace_test_mask]
        elif mode == "test":
            event_test_mask = [
                int(year) == test_year for year in event_metadata["year"]
            ]
            trace_test_mask = [
                int(year) == test_year for year in trace_metadata["year"]
            ]
            event_metadata = event_metadata[event_test_mask]
            trace_metadata = trace_metadata[trace_test_mask]

        if limit:
            event_metadata = event_metadata.iloc[:limit]
        metadata = {}
        data = {}
        with h5py.File(data_path, "r") as f:
            decimate = 1

            skipped = 0
            contained = []
            events_index = np.zeros((1, 2), dtype=int)
            for _, event in event_metadata.iterrows():
                event_name = str(int(event["EQ_ID"]))
                if (
                    event_name not in f["data"]
                ):
                    skipped += 1
                    contained += [False]
                    continue
                contained += [True]
                g_event = f["data"][event_name]
                for key in g_event:
                    if key not in data:
                        data[key] = []
                    if key == f"{input_type}_traces":
                        index = np.arange(g_event[key].shape[0]).reshape(-1, 1)
                        event_id = (
                            np.array([str(event_name)] * g_event[key].shape[0])
                            .astype(np.int32)
                            .reshape(-1, 1)
                        )
                        single_event_index = np.concatenate([event_id, index], axis=1)
                    if key == label_key:
                        data[key] += [g_event[key][()]]
                    if key == "p_picks":
                        data[key] += [g_event[key][()]]
                    if key == "station_name":
                        data[key] += [g_event[key][()]]
                    if key == "p_picks":
                        data[key][-1] //= decimate
                events_index = np.append(events_index, single_event_index, axis=0)
            events_index = np.delete(events_index, [0], 0)
        labels = np.concatenate(data[label_key], axis=0)
        stations = np.concatenate(data["station_name"], axis=0)
        picks = np.concatenate(data["p_picks"], axis=0)
        mask = (events_index != 0).any(axis=1)
        mask = np.logical_and(mask, ~np.isnan(labels))
        if dowmsampling:
            small_labels_array = labels < np.log10(0.019)
            np.random.seed(0)
            random_array = np.random.choice(
                [True, False], size=small_labels_array.shape
            )
            random_delete_mask = np.logical_and(small_labels_array, random_array)
            mask = np.logical_and(mask, ~random_delete_mask)

        labels = np.expand_dims(np.expand_dims(labels, axis=1), axis=2)
        stations = np.expand_dims(np.expand_dims(stations, axis=1), axis=2)
        p_picks = picks[mask]
        stations = stations[mask]
        labels = labels[mask]

        ok_events_index = events_index[mask]
        ok_event_id = np.intersect1d(
            np.array(event_metadata["EQ_ID"].values), ok_events_index
        )
        if oversample > 1:
            oversampled_catalog = []
            filter = event_metadata["magnitude"] >= oversample_mag
            oversample_catalog = np.intersect1d(
                np.array(event_metadata[filter]["EQ_ID"].values), ok_events_index
            )
            for event_id in oversample_catalog:
                catch_mag = event_metadata["EQ_ID"] == event_id
                mag = event_metadata[catch_mag]["magnitude"]
                repeat_time = int(oversample ** (mag - 1) - 1)
                oversampled_catalog.extend(repeat(event_id, repeat_time))

            oversampled_catalog = np.array(oversampled_catalog)
            ok_event_id = np.concatenate([ok_event_id, oversampled_catalog])
        if oversample_by_labels:
            oversampled_labels = []
            oversampled_picks = []
            labels = labels.flatten()
            filter = labels > np.log10(0.057)
            oversample_events_index = ok_events_index[filter]
            oversample_p_pick = p_picks[filter]
            repeat_times = 4.5 ** (1.5 ** labels[filter]) + 1
            repeat_times = np.round(repeat_times, 0)
            for i in range(len(repeat_times)):
                repeat_time = int(repeat_times[i])
                oversampled_labels.extend(
                    repeat(oversample_events_index[i], repeat_time)
                )
                oversampled_picks.extend(repeat(oversample_p_pick[i], repeat_time))
            oversampled_labels = np.array(oversampled_labels)
            oversampled_picks = np.array(oversampled_picks)
            ok_events_index = np.concatenate(
                (ok_events_index, oversampled_labels), axis=0
            )
            p_picks = np.concatenate((p_picks, oversampled_picks), axis=0)
        if weight_label:
            labels = labels.flatten()
            classifier = intensity_classifier(label=label_key)
            output_array = classifier.classify(labels)
            label_class, counts = np.unique(output_array, return_counts=True)
            label_counts = {}
            for i, label in enumerate(label_class):
                label_counts[int(label)] = counts[i]
            samples_weight = np.array([1 / label_counts[int(i)] for i in output_array])

        events_index_list = []
        weight_list = []
        for event_enum, event in enumerate(ok_event_id):
            single_event_index = ok_events_index[
                np.where(ok_events_index[:, 0] == event)[0]
            ]
            single_event_p_picks = p_picks[np.where(ok_events_index[:, 0] == event)[0]]
            if weight_label:
                single_event_label_weight = samples_weight[
                    np.where(ok_events_index[:, 0] == event)[0]
                ]
            if sort_by_picks:
                sort = single_event_p_picks.argsort()
                single_event_p_picks = single_event_p_picks[sort]
                single_event_index = single_event_index[sort]
                if weight_label:
                    single_event_label_weight = single_event_label_weight[sort]
            if len(single_event_index) > max_station_num:
                time = int(
                    np.ceil(len(single_event_index) / max_station_num)
                )
                split_index = np.array_split(
                    single_event_index,
                    np.arange(max_station_num, max_station_num * time, max_station_num),
                )
                if weight_label:
                    split_weight = np.array_split(
                        single_event_label_weight,
                        np.arange(
                            max_station_num, max_station_num * time, max_station_num
                        ),
                    )
                for i in range(time):
                    events_index_list.append([split_index[0], split_index[i]])
                    if bias_to_closer_station:
                        events_index_list.append([split_index[0], split_index[0]])
                    if weight_label:
                        weight_list.append(np.mean(split_weight[i]))

            else:
                events_index_list.append([single_event_index, single_event_index])
                if weight_label:
                    weight_list.append(np.mean(single_event_label_weight))
        self.data_path = data_path
        self.mode = mode
        self.event_metadata = event_metadata
        self.trace_metadata = trace_metadata
        self.input_type = input_type
        self.label = label_key
        self.labels = labels
        self.ok_events_index = ok_events_index
        self.ok_event_id = ok_event_id
        if weight_label:
            self.weight = weight_list
        self.sampling_rate = sampling_rate
        self.data_length_sec = data_length_sec
        self.metadata = metadata
        self.events_index = events_index_list
        self.p_picks = p_picks
        self.oversample = oversample
        self.max_station_num = max_station_num
        self.label_target = label_target
        self.mask_waveform_sec = mask_waveform_sec
        self.mask_waveform_random = mask_waveform_random
        self.station_blind = station_blind
        self.bias_to_closer_station = bias_to_closer_station

    def __len__(self):
        return len(self.events_index)

    def __getitem__(self, index):
        """Retrieve a batch of waveforms, stations, and targets by index."""
        specific_index = self.events_index[index]
        with h5py.File(self.data_path, "r") as f:
            specific_waveforms = []
            stations_location = []
            label_targets_location = []
            labels = []
            seen_P_picks = []
            labels_time = []
            P_picks = []
            for eventID in specific_index[0]:

                waveform = f["data"][str(eventID[0])][f"{self.input_type}_traces"][
                    eventID[1]
                ][: (self.data_length_sec * self.sampling_rate), :]

                waveform_lowfreq = f["data"][str(eventID[0])][
                    f"{self.input_type}_lowfreq_traces"
                ][eventID[1]][: (self.data_length_sec * self.sampling_rate)]

                waveform_concat = np.append(waveform, waveform_lowfreq, axis=1)

                peak_displacement = f["data"][str(eventID[0])]["pd"][eventID[1]][
                    : (self.data_length_sec * self.sampling_rate), :
                ]

                cvav = f["data"][str(eventID[0])]["cvav"][eventID[1]][
                    : (self.data_length_sec * self.sampling_rate), :
                ]

                tp = f["data"][str(eventID[0])]["TP"][eventID[1]][
                    : (self.data_length_sec * self.sampling_rate), :
                ]

                waveform_concat = np.append(waveform_concat, peak_displacement, axis=1)
                waveform_concat = np.append(waveform_concat, cvav, axis=1)
                waveform_concat = np.append(waveform_concat, tp, axis=1)

                station_location = f["data"][str(eventID[0])]["station_location"][
                    eventID[1]
                ]
                vs30 = f["data"][str(eventID[0])]["Vs30"][eventID[1]]
                station_location = np.append(station_location, vs30)
                waveform_concat = np.pad(
                    waveform_concat,
                    (
                        (
                            0,
                            self.data_length_sec * self.sampling_rate
                            - len(waveform_concat),
                        ),
                        (0, 0),
                    ),
                    "constant",
                )
                p_pick = f["data"][str(eventID[0])]["p_picks"][eventID[1]]
                specific_waveforms.append(waveform_concat)
                stations_location.append(station_location)
                seen_P_picks.append(p_pick)
            for eventID in specific_index[1]:
                station_location = f["data"][str(eventID[0])]["station_location"][
                    eventID[1]
                ]
                vs30 = f["data"][str(eventID[0])]["Vs30"][eventID[1]]
                station_location = np.append(station_location, vs30)
                label = np.array(
                    f["data"][str(eventID[0])][f"{self.label}"][eventID[1]]
                ).reshape(1, 1)
                p_pick = f["data"][str(eventID[0])]["p_picks"][eventID[1]]
                label_time = f["data"][str(eventID[0])][f"{self.label}_time"][
                    eventID[1]
                ]
                label_targets_location.append(station_location)
                labels.append(label)
                P_picks.append(p_pick)
                labels_time.append(label_time)
            if (
                len(stations_location) < self.max_station_num
            ):
                for zero_pad_num in range(
                    self.max_station_num - len(stations_location)
                ):
                    specific_waveforms.append(np.zeros_like(waveform_concat))
                    stations_location.append(np.zeros_like(station_location))
            if (
                len(label_targets_location) < self.label_target
            ):
                for zero_pad_num in range(
                    self.label_target - len(label_targets_location)
                ):
                    label_targets_location.append(np.zeros_like(station_location))
                    labels.append(np.zeros_like(label))
            specific_waveforms_array = np.array(specific_waveforms)
            if self.mask_waveform_random:
                random_mask_sec = np.random.randint(self.mask_waveform_sec, 15)
                specific_waveforms_array[
                    :, seen_P_picks[0] + (random_mask_sec * self.sampling_rate) :, :
                ] = 0
                for i in range(len(seen_P_picks)):
                    if seen_P_picks[i] > seen_P_picks[0] + (
                        random_mask_sec * self.sampling_rate
                    ):
                        specific_waveforms_array[i, :, :] = 0
                        stations_location[i] = np.zeros_like(station_location)
            elif self.mask_waveform_sec:
                specific_waveforms_array[
                    :,
                    seen_P_picks[0] + (self.mask_waveform_sec * self.sampling_rate) :,
                    :,
                ] = 0
                for i in range(len(seen_P_picks)):
                    if seen_P_picks[i] > seen_P_picks[0] + (
                        self.mask_waveform_sec * self.sampling_rate
                    ):
                        specific_waveforms_array[i, :, :] = 0
                        stations_location[i] = np.zeros_like(station_location)
            stations_location_array = np.array(stations_location)
            label_targets_location = np.array(label_targets_location)
            labels = np.array(labels)
            p_picks_array = np.array(P_picks)
            labels_time = np.array(labels_time)
            if self.station_blind:
                nonzero_indices = np.nonzero(stations_location_array.any(axis=1))[0]

                num_indices_to_fill = np.random.randint(0, len(nonzero_indices))

                random_indices = np.random.choice(
                    nonzero_indices, size=num_indices_to_fill, replace=False
                )

                stations_location_array[random_indices] = 0
                specific_waveforms_array[random_indices] = 0
        if self.mode == "train":
            outputs = {
                "waveform": specific_waveforms_array,
                "sta": stations_location_array,
                "target": label_targets_location,
                "label": labels,
            }
            return outputs
        else:
            p_picks_array = np.array(P_picks)
            labels_time = np.array(labels_time)
            outputs = {
                "waveform": specific_waveforms_array,
                "sta": stations_location_array,
                "target": label_targets_location,
                "label": labels,
                "EQ_ID": specific_index[0],
                "p_picks": p_picks_array,
                f"{self.label}_time": labels_time,
            }
            return outputs
