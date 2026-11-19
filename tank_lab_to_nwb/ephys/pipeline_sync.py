"""The lab pipeline's ephys-behavior sync, ported from U19-pipeline-python so its stored vectors can be reproduced.

`BehaviorSync.make` in `u19_pipeline/ephys_pipeline.py` calls `get_iteration_sample_vector_from_digital_lines_pulses`,
`assert_iteration_samples_count`, `evaluate_sync_process`, the "fix" path in `utils/ephys_fix_sync_code.py` when that
fails, and `get_iteration_intertrial_from_virmen_time`. The logic here follows that code line for line, prints left out,
so `pipeline_sync` gives the same `sync_data` as the DB row. That includes its quirks: an iteration index is a few samples
before the edge it was found from (see `pipeline_index_lag`). The converter itself uses the edges, through
`clock.iteration_edges`. Keep this module a faithful copy; fix nothing here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PipelineSync:
    """What BehaviorSync stores in `sync_data`, plus its status flags and the detection mode."""

    trial_idx_vector: np.ndarray
    iteration_idx_vector: np.ndarray  # object array, one int64 array per trial; empty if both sync paths failed
    trial_idx_vector_from_virmen: np.ndarray
    iteration_idx_vector_from_virmen: np.ndarray
    regular_sync_status: int
    fixed_sync_status: int
    mode: str


def pipeline_lines(digital_word, kind, rate, file_time_seconds):
    """(trial line, iteration line) as `load_trial_iteration_signals` picks them, truncated as `ExtractDigital` reads.

    The pipeline reads int(rate * fileTimeSecs) samples. A NIDQ file has trial on line 1 and iteration on line 2. A
    OneBox file has them on lines 0 and 1, and the one with more rising edges is the iteration line.
    """
    n = int(rate * np.float64(file_time_seconds))
    data = np.ascontiguousarray(np.asarray(digital_word)[:n]).view(np.uint8)
    line = lambda k: (data[k // 8 :: 2] >> (k % 8)) & 1  # noqa: E731 - bit k of each little-endian 16-bit word
    if kind == "nidq":
        return line(1), line(2)
    zero, one = line(0), line(1)
    if np.count_nonzero(zero[1:] > zero[:-1]) > np.count_nonzero(one[1:] > one[:-1]):
        return one, zero
    return zero, one


def get_idx_trial_start(trial_pulse_signal):
    trial_start_idx = np.where(np.diff(trial_pulse_signal) == 1)[0]
    trial_start_pulse_end_idx = np.where(np.diff(trial_pulse_signal) == 255)[0]
    fake_trial_init = []
    for idx, sample in enumerate(trial_start_idx):
        if np.mean(trial_pulse_signal[sample + 1 : sample + 10]) < 0.9:
            fake_trial_init.append(idx)
    trial_start_idx = np.delete(trial_start_idx, fake_trial_init)
    trial_start_pulse_end_idx = np.delete(trial_start_pulse_end_idx, fake_trial_init)
    return trial_start_idx, trial_start_pulse_end_idx


def get_idx_iter_start_pulsesignal(iteration_pulse_signal_trial, iteration_pulse_start_signal, trial_start_idx, samples_before_pulse_start, behavior_iterations):
    iter_samples = np.squeeze(np.where(np.diff(iteration_pulse_signal_trial) == 1))
    iter_samples += samples_before_pulse_start
    iter_start_samples = np.squeeze(np.where(np.diff(iteration_pulse_start_signal) == 1))
    if iter_start_samples.size > 0:
        iter_samples = np.insert(iter_samples, 0, trial_start_idx)
    else:
        iter_samples[0] = trial_start_idx
    if behavior_iterations < iter_samples.shape[0]:
        iter_samples = iter_samples[:-1]
    return iter_samples


def get_idx_iter_start_counterbit(iteration_pulse_signal_trial, trial_start_idx):
    iter_samples = np.where(np.diff(iteration_pulse_signal_trial) == 1)
    iter_samples = iter_samples + trial_start_idx
    if iteration_pulse_signal_trial[0] == 1:
        iter_samples = np.insert(iter_samples, 0, trial_start_idx)
    else:
        iter_samples[0, 0] = trial_start_idx
    iter_samples = np.squeeze(iter_samples)
    iter_samples2 = np.squeeze(np.where(np.diff(iteration_pulse_signal_trial) == 255) + trial_start_idx)
    return np.sort(np.concatenate([iter_samples, iter_samples2]))


def get_trial_signal_mode(iteration_pulse_signal_trial, behavior_time_vector_trial):
    iter_samples = np.where(np.diff(iteration_pulse_signal_trial) == 1)
    return "counter_bit0" if iter_samples[0].shape[0] < (behavior_time_vector_trial.shape[0] * 3 / 4) else "pulse_signal"


def mode_offsets(mode, rate):
    """(samples after the trial pulse's rising edge where the window opens, samples before the next trial it closes)."""
    after, before = (1, 1) if mode == "counter_bit0" else (-4, 4)
    return int(rate * (after / 1000)), int(rate * (before / 1000))


def get_iteration_sample_vector_from_digital_lines_pulses(trial_pulse_signal, iteration_pulse_signal, rate, num_behavior_trials, behavior_time_vector, mode=None):
    trial_start_idx, trial_end_pulse_idx = get_idx_trial_start(trial_pulse_signal)
    if mode is None:
        mode = get_trial_signal_mode(iteration_pulse_signal[trial_start_idx[0] : trial_start_idx[1]], behavior_time_vector[0])
    samples_after_pulse_start, samples_before_pulse_end = mode_offsets(mode, rate)
    num_trials_sync = min([trial_start_idx.shape[0], num_behavior_trials])
    iter_start_idx, iter_times_idx = [], []
    for i in range(num_trials_sync):
        idx_start = trial_end_pulse_idx[i]
        idx_start_before = trial_start_idx[i] + samples_after_pulse_start
        if i < trial_start_idx.shape[0] - 1:
            idx_end = trial_start_idx[i + 1] - samples_before_pulse_end
        else:
            idx_end = trial_pulse_signal.shape[0] - samples_before_pulse_end
        if mode == "counter_bit0":
            iter_samples = get_idx_iter_start_counterbit(iteration_pulse_signal[idx_start_before:idx_end], trial_start_idx[i])
        else:
            iter_samples = get_idx_iter_start_pulsesignal(
                iteration_pulse_signal[idx_start:idx_end], iteration_pulse_signal[idx_start_before:idx_start], trial_start_idx[i], idx_start, behavior_time_vector[i].shape[0]
            )
        iter_start_idx.append(iter_samples)
        times = iter_samples / rate
        iter_times_idx.append(times - times[0])
    return dict(iter_start_idx=_objects(iter_start_idx), iter_times_idx=_objects(iter_times_idx), mode=mode)


def _objects(arrays):
    out = np.empty(len(arrays), dtype=object)
    for i, array in enumerate(arrays):
        out[i] = array
    return out


def assert_iteration_samples_count(iteration_sample_idx_output, behavior_time_vector):
    trial_count_diff = np.abs(iteration_sample_idx_output.shape[0] - behavior_time_vector.shape[0])
    small, big = [], []
    for idx_trial, iter_trials in enumerate(iteration_sample_idx_output):
        expected = behavior_time_vector[idx_trial].shape[0]
        if iter_trials.shape[0] != expected:
            (small if np.abs(iter_trials.shape[0] - expected) < 6 else big).append(idx_trial)
    return trial_count_diff, big, small


def evaluate_sync_process(trial_count_diff, big, small, total_trials):
    """1 synced, 0 missed a few pulses and worth fixing, -1 failed. The pipeline's order of checks, kept."""
    if len(big) > 1:
        return -1
    if len(big) == 1 and big[0] == total_trials - 1:
        return 1
    if trial_count_diff == 0 and len(small) == 0:
        return 1
    if trial_count_diff < 2 and len(small) == 0:
        return -1
    if len(small) > 2:
        return -1
    if trial_count_diff < 2 and len(small) <= 2:
        return 0
    return -1


def get_index_trial_vector_from_iteration(iteration_start_idx):
    trial_start_idx = np.zeros(len(iteration_start_idx), dtype=np.int64)
    for i in range(len(iteration_start_idx)):
        trial_start_idx[i] = iteration_start_idx[i][0]
    return trial_start_idx


def get_iteration_intertrial_from_virmen_time(trial_pulse_signal, rate, num_behavior_trials, behavior_time_vector):
    trial_start_idx, _ = get_idx_trial_start(trial_pulse_signal)
    iter_start_idx = [(trial_start_idx[i] + np.int64(behavior_time_vector[i] * rate)).squeeze() for i in range(num_behavior_trials)]
    return trial_start_idx, _objects(iter_start_idx)


# ---- ephys_fix_sync_code.py ----


def get_shift_vector(synced_time_vector, behavior_time_vector, base_size=40, initial_sample=0, samples_shift=100):
    baseline_diff = synced_time_vector[initial_sample:base_size] - behavior_time_vector[initial_sample:base_size]
    median_diff = np.median(baseline_diff)
    max_diff = np.median(baseline_diff) + 0.007
    min_diff = np.median(baseline_diff) - 0.007
    vec_shift = np.zeros((synced_time_vector.shape[0] - base_size), dtype=int)
    for i in range(synced_time_vector.shape[0] - base_size):
        idx_start = initial_sample + i + 1
        idx_end = base_size + i + 1
        this_bt = behavior_time_vector[idx_start:idx_end]
        this_iv = synced_time_vector[idx_start:idx_end]
        median_ori = np.median(this_iv - this_bt)
        if median_ori >= min_diff and median_ori < max_diff:
            vec_shift[i] = 0
            sign = 0
        elif median_ori < min_diff:
            sign = 1
        else:
            sign = -1
        new_sign = sign
        if sign != 0:
            for j in range(1, samples_shift):
                if new_sign == 1:
                    if idx_end + j > synced_time_vector.shape[0]:
                        this_iv = synced_time_vector[idx_start + j :]
                        this_bt = this_bt[:-1]
                    else:
                        this_iv = synced_time_vector[idx_start + j : idx_end + j]
                else:
                    this_iv = synced_time_vector[idx_start - j : idx_end - j]
                median_now = np.median(this_iv - this_bt)
                if median_now >= min_diff and median_now < max_diff:
                    vec_shift[i] = j * new_sign
                    break
                elif median_now < min_diff:
                    new_sign = 1
                else:
                    new_sign = -1
                if new_sign != sign:
                    vec_shift[i] = 0 if np.abs(median_ori - median_diff) < np.abs(median_now - median_diff) else j * sign
                    break
                if j == samples_shift - 1:
                    vec_shift[i] = j * sign
    new_synced_time_vector = synced_time_vector.copy()
    mid_point = int(initial_sample + base_size / 2)
    for i in range(vec_shift.shape[0]):
        new_synced_time_vector[mid_point + i] = synced_time_vector[mid_point + i + vec_shift[i]]
    idx_end = mid_point + vec_shift.shape[0] - 1
    for i in range(idx_end, new_synced_time_vector.shape[0]):
        if i + vec_shift[-1] < new_synced_time_vector.shape[0]:
            new_synced_time_vector[i] = synced_time_vector[i + vec_shift[-1]]
    return new_synced_time_vector, vec_shift, [min_diff, median_diff, max_diff]


def fix_shifted_sync_vector(synced_time_vector, behavior_time_vector, vec_shift, initial_sample=0, base_size=40):
    mid_point = int(initial_sample + base_size / 2)
    new_synced_time_vector = synced_time_vector.copy()
    diff_vec_shift = np.diff(vec_shift)
    where_insert_iteration = np.where(diff_vec_shift != 0)[0]
    correlation_result = np.correlate(diff_vec_shift == 0, np.full(4, 1))
    consecutive_zeros = np.flatnonzero(correlation_result == 4)
    index_shift = 0
    index_borrow_virmen = []
    for _ in range(len(where_insert_iteration)):
        stable_parts = np.where(consecutive_zeros > where_insert_iteration[index_shift])[0]
        if stable_parts.shape[0] > 0:
            index_borrow_virmen.append([where_insert_iteration[index_shift], consecutive_zeros[stable_parts[0]]])
            next_borrow_start = np.where(where_insert_iteration > consecutive_zeros[stable_parts[0]])[0]
            if next_borrow_start.shape[0] > 0:
                index_shift = next_borrow_start[0]
            else:
                break
        else:
            index_borrow_virmen.append([where_insert_iteration[index_shift], diff_vec_shift.shape[0]])
            break
    index_diff_e = -1
    borrowed_indexes = []
    for i in range(len(index_borrow_virmen)):
        if index_diff_e > index_borrow_virmen[i][0] + mid_point:
            continue
        for j in range(100):
            index_diff_s = index_borrow_virmen[i][0] + mid_point - 1
            index_diff_e = index_borrow_virmen[i][1] + mid_point + 1
            if j >= 1:
                index_diff_s = index_diff_s - 1
            if j > 1:
                index_diff_e = index_diff_e + j - 1
            time_iteration = behavior_time_vector[index_diff_s : index_diff_e + 1] - behavior_time_vector[index_diff_s]
            new_synced_time_vector[index_diff_s : index_diff_e + 1] = np.repeat(new_synced_time_vector[index_diff_s], index_diff_e - index_diff_s + 1) + time_iteration
            check_diff = np.diff(new_synced_time_vector[index_diff_s : index_diff_e + 2])
            if np.where(check_diff <= 0.0005)[0].shape[0] == 0:
                break
        borrowed_indexes.append([index_diff_s, index_diff_e + 1])
    return new_synced_time_vector, borrowed_indexes


def complete_last_part_sync_vec(sync_time_vector, behavior_time_vector):
    new_sync_time_vector = sync_time_vector.copy()
    diff_size = behavior_time_vector.shape[0] - new_sync_time_vector.shape[0]
    borrowed_indexes = []
    if diff_size > 0:
        diff_size = diff_size + 1
        last_part_bt = behavior_time_vector[-diff_size:] - behavior_time_vector[-diff_size]
        insert_part = np.repeat(new_sync_time_vector[-1], diff_size) + last_part_bt
        new_sync_time_vector = np.append(new_sync_time_vector, insert_part[1:])
        borrowed_indexes.append([behavior_time_vector.shape[0] - diff_size, behavior_time_vector.shape[0] - 1])
    return new_sync_time_vector, borrowed_indexes


def fix_iter_vector(synced_iteration_vector, synced_time_vector, original_time_vector, rate):
    new_synced_iteration_vector = synced_iteration_vector.copy()
    ori_shape = original_time_vector.shape[0]
    new_shape = synced_time_vector.shape[0]
    diff_iter_times_idx = original_time_vector - synced_time_vector[:ori_shape]
    first_iter = synced_iteration_vector[0]
    for i in range(diff_iter_times_idx.shape[0]):
        if diff_iter_times_idx[i] != 0:
            new_synced_iteration_vector[i] = first_iter + int(synced_time_vector[i] * rate)
    if ori_shape != new_shape:
        last_iterations = np.round(first_iter + synced_time_vector[ori_shape:] * rate)
        new_synced_iteration_vector = np.append(new_synced_iteration_vector, last_iterations.astype(np.int64))
    return new_synced_iteration_vector


def sync_evaluation_process2(synced_time_vector, behavior_time_vector):
    diff_vector = synced_time_vector - behavior_time_vector[: synced_time_vector.shape[0]]
    num_iter = diff_vector.shape[0]
    median_general = np.median(diff_vector)
    median_diff_abs = np.empty([10])
    for j in range(10):
        start_iter, end_iter = int(j * num_iter / 10), int((j + 1) * num_iter / 10)
        median_diff_abs[j] = np.median(diff_vector[start_iter:end_iter]) - median_general
    return 1 if np.max(np.abs(median_diff_abs)) < 0.005 else -1


def main_ephys_fix_sync_code(iter_start_idx, iter_times_idx, behavior_time, rate):
    starts, times = [], []
    for i in range(len(iter_start_idx)):
        behavior_time_vector = behavior_time[i].flatten()
        synced_time_vector, shift_vec, _ = get_shift_vector(iter_times_idx[i], behavior_time_vector)
        synced_time_vector, _ = fix_shifted_sync_vector(synced_time_vector, behavior_time_vector, shift_vec)
        synced_time_vector, _ = complete_last_part_sync_vec(synced_time_vector, behavior_time_vector)
        starts.append(fix_iter_vector(iter_start_idx[i], synced_time_vector, iter_times_idx[i], rate).copy())
        times.append(synced_time_vector.copy())
    iteration_dict = dict(iter_start_idx=_objects(starts), iter_times_idx=_objects(times))
    trial_count_diff, big, small = assert_iteration_samples_count(iteration_dict["iter_start_idx"], behavior_time)
    status = evaluate_sync_process(trial_count_diff, big, small, behavior_time.shape[0])
    for i in range(len(iteration_dict["iter_start_idx"])):
        status = sync_evaluation_process2(iteration_dict["iter_times_idx"][i], behavior_time[i].flatten())
        if status == -1:
            break
    if status == 1:
        iteration_dict["trial_start_idx"] = get_index_trial_vector_from_iteration(iteration_dict["iter_start_idx"])
    return status, iteration_dict


def pipeline_sync(trial_line, iteration_line, rate, behavior_time):
    """BehaviorSync.make from the lines on: the regular path, the fix path if it fails, and the ViRMEn-borrowed vectors.

    `behavior_time` is an object array of each trial's `trial.time`, in session order, as the pipeline fetches it.
    """
    detected = get_iteration_sample_vector_from_digital_lines_pulses(trial_line, iteration_line, rate, behavior_time.shape[0], behavior_time, None)
    trial_count_diff, big, small = assert_iteration_samples_count(detected["iter_start_idx"], behavior_time)
    status = evaluate_sync_process(trial_count_diff, big, small, behavior_time.shape[0])
    iteration_dict = dict(detected)
    if status == 1:
        iteration_dict["trial_start_idx"] = get_index_trial_vector_from_iteration(iteration_dict["iter_start_idx"])
    status_regular, status_fix = 1, 0
    if status < 1:
        status_regular = 0
        status_fix, iteration_dict = main_ephys_fix_sync_code(detected["iter_start_idx"], detected["iter_times_idx"], behavior_time, rate)
    if status_regular > 0 or status_fix > 0:
        trial_idx, iteration_idx = iteration_dict["trial_start_idx"], iteration_dict["iter_start_idx"]
    else:
        trial_idx, iteration_idx = np.zeros(0, dtype=np.int64), _objects([])
    virmen_trials, virmen_iterations = get_iteration_intertrial_from_virmen_time(trial_line, rate, behavior_time.shape[0], behavior_time)
    return PipelineSync(trial_idx, iteration_idx, virmen_trials, virmen_iterations, status_regular, int(status_fix), detected["mode"])


def pipeline_index_lag(mode, rate):
    """Samples from a stored iteration index to the first sample of the edge it was found from: (first iteration, later ones).

    The first iteration's index is the trial line's diff index, one sample before its rising edge. In counter_bit0 mode a
    later index is the diff index within a window opened `mode_offsets` samples after the trial edge, but the pipeline
    adds the trial's start, not the window's, so the index lands that many samples early, plus one for the diff.
    """
    after, _ = mode_offsets(mode, rate)
    return 1, (1 + after) if mode == "counter_bit0" else 1


# ---- ephys_pipeline.get_full_vectors_from_key and ephys_sync.CuratedClustersIteration ----


def get_full_vector_samples(iter_start_idx_vectors, nidq_sampling_rate, total_samples):
    framenumber_vector_samples = np.zeros(total_samples) * np.nan
    trialnumber_vector_samples = np.zeros(total_samples) * np.nan
    for i in range(len(iter_start_idx_vectors)):
        this_trial_iter_vector = iter_start_idx_vectors[i]
        if i < len(iter_start_idx_vectors) - 1:
            next_trial_iter_vector = iter_start_idx_vectors[i + 1]
        for j in range(this_trial_iter_vector.shape[0] - 1):
            framenumber_vector_samples[this_trial_iter_vector[j] : this_trial_iter_vector[j + 1]] = j + 1
        if i < len(iter_start_idx_vectors) - 1:
            trialnumber_vector_samples[this_trial_iter_vector[0] : next_trial_iter_vector[0]] = i + 1
            framenumber_vector_samples[this_trial_iter_vector[-1] : next_trial_iter_vector[0]] = this_trial_iter_vector.shape[0]
        else:
            trialnumber_vector_samples[this_trial_iter_vector[0] : this_trial_iter_vector[-1] + int(nidq_sampling_rate)] = i + 1
            framenumber_vector_samples[this_trial_iter_vector[-1] : this_trial_iter_vector[-1] + int(nidq_sampling_rate)] = this_trial_iter_vector.shape[0]
    return trialnumber_vector_samples, framenumber_vector_samples


def curated_clusters_iteration_times(iteration_idx_vector, nidq_sampling_rate, total_samples):
    """The iteration times CuratedClustersIteration digitizes spikes into, from the per-sample iteration vector.

    It expands the vectors to one value per NIDQ sample, then takes the samples where the value changes, by np.diff,
    so each time is the sample before the index, and the session's first iteration (after NaN) is dropped. Its spike
    times come from the element, `spike_times.npy / params.py sample_rate`.
    """
    _, iteration_index_nidq = get_full_vector_samples([np.asarray(v, dtype=np.int64).ravel() for v in iteration_idx_vector], nidq_sampling_rate, total_samples)
    ls = np.diff(iteration_index_nidq)
    ls[ls < 0] = 1
    ls[np.isnan(ls)] = 0
    return np.where(ls)[0] / nidq_sampling_rate
