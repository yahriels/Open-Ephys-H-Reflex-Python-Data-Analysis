"""
helpers.py - Shared utilities for EMG analysis notebooks.

Contains constants, data loading helpers, analysis utilities, and the
trial initiation simulation engine used by both:
  - EMG_characterization_stage_Offline_Analysis.ipynb
  - EMG_Trial_Initiation_Simulator.ipynb
"""

import os
import re
import time
import struct
import glob as globmod
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import BinaryIO, List, Optional
from random import Random
from scipy.signal import butter, lfilter


# ====================================================================
# CONSTANTS
# ====================================================================

PROTOCOL_ONLINE = "Online Filtering"
PROTOCOL_OFFLINE = "Offline Filtering"
PROTOCOL_OPTIONS = [PROTOCOL_ONLINE, PROTOCOL_OFFLINE]

BIN_DURATION_MS = 50
SAMPLE_RATE_HINT = 5000
BIN_SAMPLE_COUNT = int((SAMPLE_RATE_HINT / 1000) * BIN_DURATION_MS)

TRIAL_INIT_MIN_MS = 2200
TRIAL_INIT_MAX_MS = 2700
TRIAL_INIT_MIN_UV = 5.0
TRIAL_INIT_MAX_UV = 300.0

MINIMUM_INTERTRIAL_INTERVAL_MS = 10000
TRIAL_RECORDING_DURATION_MS = 100
SIMULATION_BLOCK_SIZE = 50
SIMULATION_RANDOM_SEED = 42

# ====================================================================
# HRS FILE FORMAT CONSTANTS
# ====================================================================

SAMPLE_RATE = 5000.0                     # Hz
BIN_SAMPLES = int(BIN_DURATION_MS * SAMPLE_RATE / 1000)    # 250 samples
TRIAL_RECORD_MS = 100                    # ms post-stim recording window
TRIAL_RECORD_SAMPLES = int(TRIAL_RECORD_MS * SAMPLE_RATE / 1000)  # 500 samples
MS_PER_SAMPLE = 1000.0 / SAMPLE_RATE    # 0.2 ms/sample

STIM_ONSET_THRESHOLD = 4.5  # V -- ADC level marking stim onset rising edge
STIM_END_THRESHOLD   = 1.9  # V -- ADC level below which stim pulse has ended

# Block IDs (mirrors HReflexDataFileBlockIds)
BLOCK_EMG_DATA             = 1
BLOCK_EMG_CHAR_TRIAL       = 2
BLOCK_MH_TRIAL             = 3
BLOCK_EMG_TRIALS_PER_HOUR  = 4

# Low-level type maps (mirrors FileIO_Helpers from hreflex_txbdc)
_HRS_TYPE_FMT  = {'int8': 'b', 'int32': 'i', 'uint64': 'Q', 'uint8': 'B', 'float32': 'f', 'float64': 'd'}
_HRS_TYPE_SIZE = {'int8': 1,  'int32': 4,   'uint64': 8,   'uint8': 1,   'float32': 4,   'float64': 8}


# ====================================================================
# HRS BINARY READER PRIMITIVES
# ====================================================================

def hrs_read_val(fid: BinaryIO, dtype: str):
    raw = fid.read(_HRS_TYPE_SIZE[dtype])
    if len(raw) < _HRS_TYPE_SIZE[dtype]:
        raise EOFError(f"Unexpected end of file reading {dtype}")
    return struct.unpack(_HRS_TYPE_FMT[dtype], raw)[0]


def hrs_read_string(fid: BinaryIO) -> str:
    n = hrs_read_val(fid, 'int32')
    return fid.read(n).decode('utf-8')


def hrs_read_datetime(fid: BinaryIO) -> datetime:
    datenum = hrs_read_val(fid, 'float64')
    days = datenum % 1
    return datetime.fromordinal(int(datenum)) + timedelta(days=days) - timedelta(days=366)


def hrs_read_array(fid: BinaryIO, dtype: str):
    n = hrs_read_val(fid, 'int32')
    fmt = _HRS_TYPE_FMT[dtype]
    size = _HRS_TYPE_SIZE[dtype]
    raw = fid.read(n * size)
    return list(struct.unpack(f'{n}{fmt}', raw))


# ====================================================================
# HRS DATA CLASSES
# ====================================================================

@dataclass
class EmgCharHeader:
    file_version: int = 0
    subject_id: str = ""
    session_datetime: datetime = None
    stage_name: str = ""
    stage_description: str = ""
    stage_type: int = 0
    trial_initiation_uv_min: float = 0.0
    trial_initiation_uv_max: float = 0.0
    trial_initiation_phase_min_ms: int = 0
    trial_initiation_phase_max_ms: int = 0
    bin_duration_ms: int = 0
    # Present in new-format HRS1 files; defaults to 5000.0 for legacy files.
    sample_rate: float = 5000.0
    # Set when an EMG_TRIALS_PER_HOUR block is present in the file; otherwise None.
    trials_per_hour_data: object = None


@dataclass
class EmgTrialsPerHourData:
    sweep_centre_uv: float = 0.0
    elapsed_hours: float = 0.0
    window_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    trials_per_hour: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass
class EmgCharTrial:
    trial_end_datetime: datetime = None
    trial_start_index: int = 0
    trial_start_open_ephys_millis: int = 0
    trial_start_open_ephys_sample_id: int = 0
    grand_mean: float = 0.0
    bins: list = field(default_factory=list)
    monitored_signal: list = field(default_factory=list)


@dataclass
class EmgDataBlock:
    ts_open_ephys_sent: int = 0
    ts_python_received: int = 0
    ts_background_emitted: int = 0
    channel_names: list = field(default_factory=list)
    raw_channels: list = field(default_factory=list)
    diff: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    filtered: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    abs_val: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


@dataclass
class MhRecHeader:
    file_version: int = 0
    subject_id: str = ""
    session_start_time: datetime = None
    stage_name: str = ""
    stage_description: str = ""
    stage_type: int = 0


@dataclass
class MhRecTrial:
    start_time: datetime = None
    min_initiation_threshold: float = 0.0
    max_initiation_threshold: float = 0.0
    stimulation_amplitude_ma: float = 0.0
    trial_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sync_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    # --- file_version >= 2 fields ---
    trigger_wall_time_ms: int = 0
    onset_sample_index: int = -1       # -1 = not found (fallback used)
    onset_detected: int = 0            # 1 = real crossing, 0 = fallback
    stim_end_sample_index: int = -1    # -1 = not found within recording window
    stim_duration_samples: int = 0
    stim_duration_ms: float = 0.0
    sync_peak_voltage: float = 0.0     # max ADC in search window
    n_pre_trigger_frames_discarded: int = 0
    frame_received_timestamps_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint64))
    first_post_trigger_frame_sample_id: int = 0
    # --- file_version >= 3 fields ---
    unipolar_trial_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


# ====================================================================
# HRS FILE READERS
# ====================================================================

def _read_emg_data_block(fid: BinaryIO) -> EmgDataBlock:
    block = EmgDataBlock()
    block.ts_open_ephys_sent     = hrs_read_val(fid, 'uint64')
    block.ts_python_received     = hrs_read_val(fid, 'uint64')
    block.ts_background_emitted  = hrs_read_val(fid, 'uint64')
    n_names = hrs_read_val(fid, 'uint8')
    block.channel_names = [hrs_read_string(fid) for _ in range(n_names)]
    n_ch = hrs_read_val(fid, 'uint8')
    block.raw_channels = [np.array(hrs_read_array(fid, 'float32'), dtype=np.float32) for _ in range(n_ch)]
    block.diff     = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    block.filtered = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    block.abs_val  = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    return block


def _read_mh_trial_block(fid: BinaryIO, file_version: int = 0) -> MhRecTrial:
    t = MhRecTrial()
    t.start_time                = hrs_read_datetime(fid)
    t.min_initiation_threshold  = hrs_read_val(fid, 'float32')
    t.max_initiation_threshold  = hrs_read_val(fid, 'float32')
    t.stimulation_amplitude_ma  = hrs_read_val(fid, 'float32')
    t.trial_data = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    if file_version >= 1:
        t.sync_data = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    if file_version >= 2:
        t.trigger_wall_time_ms              = hrs_read_val(fid, 'uint64')
        t.onset_sample_index                = hrs_read_val(fid, 'int32')
        t.onset_detected                    = hrs_read_val(fid, 'int8')
        t.stim_end_sample_index             = hrs_read_val(fid, 'int32')
        t.stim_duration_samples             = hrs_read_val(fid, 'int32')
        t.stim_duration_ms                  = hrs_read_val(fid, 'float32')
        t.sync_peak_voltage                 = hrs_read_val(fid, 'float32')
        t.n_pre_trigger_frames_discarded    = hrs_read_val(fid, 'int32')
        t.frame_received_timestamps_ms      = np.array(hrs_read_array(fid, 'uint64'), dtype=np.uint64)
        t.first_post_trigger_frame_sample_id = hrs_read_val(fid, 'uint64')
    if file_version >= 3:
        t.unipolar_trial_data = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    return t


def read_hrs1(filepath: str):
    """Read an .hrs1 (EMG Characterization) data file.

    Returns (header, trials, emg_blocks).
    """
    header = EmgCharHeader()
    trials, emg_blocks = [], []

    with open(filepath, 'rb') as fid:
        header.file_version                  = hrs_read_val(fid, 'int32')
        header.subject_id                    = hrs_read_string(fid)
        header.session_datetime              = hrs_read_datetime(fid)
        header.stage_name                    = hrs_read_string(fid)
        header.stage_description             = hrs_read_string(fid)
        header.stage_type                    = hrs_read_val(fid, 'int32')
        header.trial_initiation_uv_min       = hrs_read_val(fid, 'float32')
        header.trial_initiation_uv_max       = hrs_read_val(fid, 'float32')
        header.trial_initiation_phase_min_ms = hrs_read_val(fid, 'int32')
        header.trial_initiation_phase_max_ms = hrs_read_val(fid, 'int32')
        header.bin_duration_ms               = hrs_read_val(fid, 'int32')

        # Auto-detect the sample_rate field added in newer HRS1 files.
        # Peek at the next 4 bytes: if they parse as a float32 in the valid
        # sample-rate range (500–100 000 Hz), consume them; otherwise seek back
        # so the block-ID loop reads correctly (legacy files without this field).
        _pos  = fid.tell()
        _peek = fid.read(4)
        if len(_peek) == 4:
            _as_float = struct.unpack('f', _peek)[0]
            if 500.0 <= _as_float <= 100_000.0:
                header.sample_rate = _as_float
            else:
                fid.seek(_pos)

        while True:
            chunk = fid.read(4)
            if len(chunk) < 4:
                break
            block_id = struct.unpack('i', chunk)[0]
            try:
                if block_id == BLOCK_EMG_CHAR_TRIAL:
                    t = EmgCharTrial()
                    t.trial_end_datetime               = hrs_read_datetime(fid)
                    t.trial_start_index                = hrs_read_val(fid, 'uint64')
                    t.trial_start_open_ephys_millis    = hrs_read_val(fid, 'uint64')
                    t.trial_start_open_ephys_sample_id = hrs_read_val(fid, 'uint64')
                    t.grand_mean        = hrs_read_val(fid, 'float32')
                    t.bins              = hrs_read_array(fid, 'float32')
                    t.monitored_signal  = hrs_read_array(fid, 'float32')
                    trials.append(t)
                elif block_id == BLOCK_EMG_DATA:
                    emg_blocks.append(_read_emg_data_block(fid))
                elif block_id == BLOCK_EMG_TRIALS_PER_HOUR:
                    tph = EmgTrialsPerHourData()
                    tph.sweep_centre_uv = hrs_read_val(fid, 'float32')
                    tph.elapsed_hours   = hrs_read_val(fid, 'float32')
                    tph.window_sizes    = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
                    tph.trials_per_hour = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
                    header.trials_per_hour_data = tph
            except struct.error:
                # Last block was truncated (file closed mid-write); discard it.
                break

    return header, trials, emg_blocks


def read_hrs2(filepath: str):
    """Read an .hrs2 (MH Recruitment Curve) data file.

    Notes:
    - file_version 0: trial blocks contain trial_data only (no sync_data).
    - file_version 1+: trial blocks also contain sync_data (ADC sync line).
    - Known bug: MhRecruitmentCurveTrial.save_to_file writes block_id=1 instead
      of block_id=3. This reader disambiguates by peeking at the first 8 bytes
      (MATLAB datenum float64 ~730000-750000 vs Unix millis uint64 ~1.7e12).

    Returns (header, trials, emg_blocks).
    """
    header = MhRecHeader()
    trials, emg_blocks = [], []

    with open(filepath, 'rb') as fid:
        header.file_version      = hrs_read_val(fid, 'int32')
        header.subject_id        = hrs_read_string(fid)
        header.session_start_time = hrs_read_datetime(fid)
        header.stage_name        = hrs_read_string(fid)
        header.stage_description = hrs_read_string(fid)
        header.stage_type        = hrs_read_val(fid, 'int32')

        while True:
            chunk = fid.read(4)
            if len(chunk) < 4:
                break
            block_id = struct.unpack('i', chunk)[0]

            if block_id == BLOCK_MH_TRIAL:
                trials.append(_read_mh_trial_block(fid, header.file_version))
            elif block_id == BLOCK_EMG_DATA:
                # Could be a trial written with buggy block_id=1 -- peek to disambiguate.
                pos = fid.tell()
                peek = fid.read(8)
                fid.seek(pos)
                if len(peek) < 8:
                    break
                peek_float = struct.unpack('d', peek)[0]
                if 730000 < peek_float < 750000:  # MATLAB datenum range for 2020-2030
                    trials.append(_read_mh_trial_block(fid, header.file_version))
                else:
                    emg_blocks.append(_read_emg_data_block(fid))
            else:
                print(f"Warning: unknown block_id={block_id} at offset {fid.tell()-4}")
                break

    return header, trials, emg_blocks


def find_hrs_files(directory: str):
    """Auto-detect the .hrs1 and .hrs2 files in a recording directory.

    Returns (hrs1_path, hrs2_path). hrs2_path is None if no .hrs2 file is found.
    Raises FileNotFoundError only if the .hrs1 file is missing.
    """
    hrs1_files = globmod.glob(os.path.join(directory, "*.hrs1"))
    hrs2_files = globmod.glob(os.path.join(directory, "*.hrs2"))

    if not hrs1_files:
        raise FileNotFoundError(f"No .hrs1 file found in '{directory}'")
    if len(hrs1_files) > 1:
        print(f"Warning: multiple .hrs1 files found, using: {hrs1_files[0]}")
    if len(hrs2_files) > 1:
        print(f"Warning: multiple .hrs2 files found, using: {hrs2_files[0]}")

    hrs2_path = hrs2_files[0] if hrs2_files else None
    return hrs1_files[0], hrs2_path


# ====================================================================
# HRS SIGNAL HELPERS
# ====================================================================

def detect_stim_onset(sync_data: np.ndarray,
                      bin_samples: int = BIN_SAMPLES,
                      record_samples: int = TRIAL_RECORD_SAMPLES,
                      threshold: float = STIM_ONSET_THRESHOLD) -> int:
    """Return the sample index of stim onset in sync_data.

    Searches from 60% into the pre-stim bin through the end of the recording
    window. Falls back to bin_samples if no onset is detected.
    """
    search_start = int(bin_samples * 0.6)
    search_end   = min(bin_samples + record_samples, len(sync_data))
    if len(sync_data) > search_start:
        window = sync_data[search_start:search_end]
        cands  = np.where(window >= threshold)[0]
        if len(cands) > 0:
            return search_start + int(cands[0])
    return bin_samples


def get_trial_window(trial: MhRecTrial,
                     pre_plot_ms: float,
                     post_plot_ms: float,
                     ms_per_sample: float = MS_PER_SAMPLE,
                     bin_samples: int = BIN_SAMPLES,
                     record_samples: int = TRIAL_RECORD_SAMPLES,
                     onset_threshold: float = STIM_ONSET_THRESHOLD,
                     end_threshold: float = STIM_END_THRESHOLD,
                     use_unipolar: bool = False):
    """Extract the peri-stimulus window for one MhRecTrial.

    For file_version >= 2 trials the app-computed onset_sample_index is used
    directly (it was detected with the live STIM_ONSET_THRESHOLD at record time).
    For older trials onset is re-derived from sync_data.

    Parameters
    ----------
    use_unipolar : bool
        When True, return the unipolar_trial_data slice instead of trial_data.
        Requires file_version >= 3. Falls back to trial_data silently when
        unipolar_trial_data is empty.

    Returns (t_ms, emg, adc_or_None, stim_end_ms_or_None).
    """
    has_sync = len(trial.sync_data) > 1

    # Prefer the pre-computed onset stored in the trial (file_version >= 2)
    if getattr(trial, 'onset_detected', 0) == 1 and getattr(trial, 'onset_sample_index', -1) >= 0:
        onset_idx = trial.onset_sample_index
    elif has_sync:
        onset_idx = detect_stim_onset(trial.sync_data, bin_samples, record_samples,
                                      onset_threshold)
    else:
        onset_idx = bin_samples

    # Select bipolar or unipolar signal
    unipolar = getattr(trial, 'unipolar_trial_data', np.array([], dtype=np.float32))
    signal = unipolar if (use_unipolar and len(unipolar) > 1) else trial.trial_data

    pre_s = int(pre_plot_ms  / ms_per_sample)
    pst_s = int(post_plot_ms / ms_per_sample)
    i0 = max(0, onset_idx - pre_s)
    i1 = min(len(signal), onset_idx + pst_s)

    emg  = signal[i0:i1]
    n    = len(emg)
    t_ms = (np.arange(n) - (onset_idx - i0)) * ms_per_sample

    adc = None
    if has_sync and len(trial.sync_data) >= i1:
        candidate = trial.sync_data[i0:i1]
        if len(candidate) == n:
            adc = candidate

    stim_end_ms = None
    if has_sync and onset_idx < len(trial.sync_data):
        ends = np.where(trial.sync_data[onset_idx:] < end_threshold)[0]
        if len(ends) > 0:
            stim_end_ms = float(ends[0]) * ms_per_sample

    return t_ms, emg, adc, stim_end_ms


def get_trial_context_window(trial: 'MhRecTrial',
                             emg_blocks: list,
                             pre_s: float = 10.0,
                             post_s: float = 10.0,
                             sample_rate: float = SAMPLE_RATE_HINT,
                             bin_samples: int = BIN_SAMPLES):
    """Extract a wide context window (default ±10 s) from the continuous EMG record.

    Uses ``first_post_trigger_frame_sample_id`` (file_version >= 2) to locate the
    trial onset in the continuous ``emg_blocks`` list, then stitches together the
    blocks that overlap the requested window.

    Falls back to matching via ``ts_background_emitted`` wall-clock ms vs
    ``trigger_wall_time_ms`` when sample-ID info is unavailable.

    Parameters
    ----------
    trial      : MhRecTrial – the trial whose onset anchors the window.
    emg_blocks : list[EmgDataBlock] – the continuous EMG blocks (hrs2_emg_blocks).
    pre_s      : seconds of data to include *before* the onset (default 10 s).
    post_s     : seconds of data to include *after*  the onset (default 10 s).
    sample_rate: recording sample rate in Hz.
    bin_samples: number of pre-trigger samples prepended to trial_data.

    Returns
    -------
    (t_s, emg_filt, onset_idx) where
        t_s        – time axis in seconds, zeroed at the detected onset
        emg_filt   – filtered EMG signal over the window
        onset_idx  – index in the returned arrays corresponding to t_s = 0
    Returns None if the onset position cannot be determined or no blocks overlap.
    """
    if not emg_blocks:
        return None

    pre_samp  = int(pre_s  * sample_rate)
    post_samp = int(post_s * sample_rate)

    # ── Locate onset in OE sample space ──────────────────────────────────────
    onset_oe: int | None = None

    first_id = getattr(trial, 'first_post_trigger_frame_sample_id', 0)
    onset_idx_in_trial = getattr(trial, 'onset_sample_index', -1)

    if first_id > 0 and onset_idx_in_trial >= 0:
        # onset_sample_index is relative to the start of trial_data;
        # the first BIN_SAMPLES of trial_data are pre-trigger, so:
        onset_oe = int(first_id) + (onset_idx_in_trial - bin_samples)
    else:
        # Fallback: use trigger wall-clock time to find nearest block by
        # ts_background_emitted (background-thread wall-clock ms).
        tw = getattr(trial, 'trigger_wall_time_ms', 0)
        if tw > 0:
            best_diff = None
            best_block_idx = None
            cumulative = 0
            for bi, blk in enumerate(emg_blocks):
                d = abs(int(getattr(blk, 'ts_background_emitted', 0)) - int(tw))
                if best_diff is None or d < best_diff:
                    best_diff = d
                    best_block_idx = bi
                cumulative += len(blk.filtered)
            if best_block_idx is not None:
                # Use ts_open_ephys_sent as OE sample anchor
                blk = emg_blocks[best_block_idx]
                onset_oe = int(blk.ts_open_ephys_sent) + bin_samples
        if onset_oe is None:
            return None

    target_start = onset_oe - pre_samp
    target_end   = onset_oe + post_samp

    # ── Stitch blocks that overlap [target_start, target_end] ────────────────
    segments: list[np.ndarray] = []
    adc_segments: list[np.ndarray] = []
    collected_start: int | None = None  # OE sample of first collected sample

    for blk in emg_blocks:
        blk_start = int(blk.ts_open_ephys_sent)
        blk_end   = blk_start + len(blk.filtered) - 1

        if blk_end < target_start:
            continue
        if blk_start > target_end:
            break

        clip_lo = max(0, target_start - blk_start)
        clip_hi = min(len(blk.filtered), target_end - blk_start + 1)
        if clip_lo >= clip_hi:
            continue

        chunk = blk.filtered[clip_lo:clip_hi]
        if collected_start is None:
            collected_start = blk_start + clip_lo
        segments.append(chunk)

        # Stitch ADC channel (look for 'ADC' in channel names, fall back to index 2)
        adc_idx = None
        for ci, cn in enumerate(blk.channel_names):
            if 'ADC' in cn.upper() and ci < len(blk.raw_channels):
                adc_idx = ci
                break
        if adc_idx is None and len(blk.raw_channels) >= 3:
            adc_idx = 2
        if adc_idx is not None:
            adc_segments.append(blk.raw_channels[adc_idx][clip_lo:clip_hi])

    if not segments:
        return None

    emg_cat = np.concatenate(segments)
    adc_cat = np.concatenate(adc_segments) if len(adc_segments) == len(segments) else None
    onset_in_window = onset_oe - collected_start
    t_s = (np.arange(len(emg_cat)) - onset_in_window) / sample_rate

    return t_s, emg_cat, onset_in_window, adc_cat


def compute_background_bins(trial,
                            emg_blocks,
                            monitoring_window_ms: float = 2500.0,
                            bin_ms: float = BIN_DURATION_MS,
                            sample_rate: float = SAMPLE_RATE):
    """Reconstruct the pre-stim |EMG| bins that the live H-Reflex App used to
    decide whether to initiate a recruitment-curve trial.

    Walks back through ``emg_blocks`` from the trial's wall-clock trigger time,
    collects ``monitoring_window_ms`` of ``abs_val`` ending at the trigger, splits
    into ``bin_ms``-wide bins, and returns ``(bins, grand_mean)``.

    ``trial.trigger_wall_time_ms`` and ``block.ts_open_ephys_sent`` are both Unix
    millisecond timestamps (see ``HReflexDataFileEmgData`` and
    ``mh_recruitment_curve_data_file``). We match on those, then use sample-count
    arithmetic to look back. Returns ``(None, nan)`` if the trigger time can't be
    located in the block stream.

    Mirrors ``MhRecruitmentCurveStage_TrialInitiationData.process()`` in the app.
    """
    if not emg_blocks:
        return None, float('nan')

    trigger_ms = int(getattr(trial, 'trigger_wall_time_ms', 0) or 0)
    if trigger_ms <= 0:
        return None, float('nan')

    bin_samp = int(bin_ms * sample_rate / 1000.0)
    n_bins = int(monitoring_window_ms / bin_ms)
    if bin_samp <= 0 or n_bins <= 0:
        return None, float('nan')
    needed = n_bins * bin_samp

    trig_idx = None
    best_diff = None
    for i, blk in enumerate(emg_blocks):
        blk_ms = int(blk.ts_open_ephys_sent)
        if blk_ms > trigger_ms:
            break
        diff = trigger_ms - blk_ms
        if best_diff is None or diff < best_diff:
            best_diff = diff
            trig_idx = i

    if trig_idx is None:
        return None, float('nan')

    trig_blk = emg_blocks[trig_idx]
    ms_into_block = trigger_ms - int(trig_blk.ts_open_ephys_sent)
    sample_offset = int(round(ms_into_block * sample_rate / 1000.0))
    sample_offset = max(0, min(sample_offset, len(trig_blk.abs_val)))

    segments: list[np.ndarray] = [trig_blk.abs_val[:sample_offset]]
    collected = len(segments[0])
    j = trig_idx - 1
    while collected < needed and j >= 0:
        chunk = emg_blocks[j].abs_val
        segments.insert(0, chunk)
        collected += len(chunk)
        j -= 1

    if collected < bin_samp:
        return None, float('nan')

    abs_signal = np.concatenate(segments)
    if len(abs_signal) > needed:
        abs_signal = abs_signal[-needed:]
    n_bins_actual = len(abs_signal) // bin_samp
    if n_bins_actual <= 0:
        return None, float('nan')

    abs_signal = abs_signal[-n_bins_actual * bin_samp:]
    bins = abs_signal.reshape(n_bins_actual, bin_samp).mean(axis=1)
    grand_mean = float(bins.mean())
    return bins, grand_mean


def print_hrs1_summary(header, trials, emg_blocks, file_path: str = "") -> None:
    """Print a human-readable summary of an .hrs1 file's header, trials, and EMG blocks."""
    print("=== HRS1 Header ===")
    if file_path:
        print(f"  File:               {os.path.basename(file_path)}")
    print(f"  File version:       {header.file_version}")
    print(f"  Subject ID:         {header.subject_id}")
    print(f"  Session datetime:   {header.session_datetime}")
    print(f"  Stage name:         {header.stage_name}")
    print(f"  Stage description:  {header.stage_description}")
    print(f"  Stage type:         {header.stage_type}")
    print(f"  Trial init uV min:  {header.trial_initiation_uv_min}")
    print(f"  Trial init uV max:  {header.trial_initiation_uv_max}")
    print(f"  Trial init phase min ms: {header.trial_initiation_phase_min_ms}")
    print(f"  Trial init phase max ms: {header.trial_initiation_phase_max_ms}")
    print(f"  Bin duration ms:    {header.bin_duration_ms}")
    print(f"  Sample rate:        {header.sample_rate} Hz")
    tph = header.trials_per_hour_data
    if tph is not None:
        print(f"  Trials/hr data: centre={tph.sweep_centre_uv:.2f} µV, "
              f"elapsed={tph.elapsed_hours:.3f} h, {len(tph.window_sizes)} window sizes")
    else:
        print("  Trials-per-hour data: not present in file")
    print(f"\n  Trials found:       {len(trials)}")
    print(f"  EMG data blocks:    {len(emg_blocks)}")

    if len(trials) > 0:
        t0 = trials[0]
        print("\n=== First Trial ===")
        print(f"  End datetime:       {t0.trial_end_datetime}")
        print(f"  Start index:        {t0.trial_start_index}")
        print(f"  Grand mean:         {t0.grand_mean:.4f}")
        print(f"  Bins count:         {len(t0.bins)}")
        print(f"  Monitored signal N: {len(t0.monitored_signal)}")

    if len(emg_blocks) > 0:
        b0 = emg_blocks[0]
        n_raw = len(b0.raw_channels[0]) if b0.raw_channels else 0
        print("\n=== First EMG Block ===")
        print(f"  Channel names:      {b0.channel_names}")
        print(f"  Raw channels:       {len(b0.raw_channels)} x {n_raw} samples")
        print(f"  Diff samples:       {len(b0.diff)}")
        print(f"  Filtered samples:   {len(b0.filtered)}")
        print(f"  Abs samples:        {len(b0.abs_val)}")


def print_hrs2_summary(header, trials, emg_blocks, file_path: str = "") -> None:
    """Print a human-readable summary of an .hrs2 file's header, trials, and EMG blocks."""
    if header is None:
        print("No HRS2 file loaded.")
        return

    fv_descs = {
        0: "v0: trial_data only",
        1: "v1: + ADC sync_data",
        2: "v2: + timing/sync debug fields",
        3: "v3: + unipolar_trial_data",
    }
    fv_desc = fv_descs.get(header.file_version, f"v{header.file_version}")

    print("=== HRS2 Header ===")
    if file_path:
        print(f"  File:               {os.path.basename(file_path)}")
    print(f"  File version:       {header.file_version}  ({fv_desc})")
    print(f"  Subject ID:         {header.subject_id}")
    print(f"  Session start time: {header.session_start_time}")
    print(f"  Stage name:         {header.stage_name}")
    print(f"  Stage description:  {header.stage_description}")
    print(f"  Stage type:         {header.stage_type}")
    print(f"\n  Trials found:       {len(trials)}")
    print(f"  EMG data blocks:    {len(emg_blocks)}")

    if len(emg_blocks) > 0:
        b0 = emg_blocks[0]
        n_raw = len(b0.raw_channels[0]) if b0.raw_channels else 0
        print("\n=== First EMG Block ===")
        print(f"  Channel names:  {b0.channel_names}")
        print(f"  Raw channels:   {len(b0.raw_channels)} x {n_raw} samples")
        print(f"  Diff samples:   {len(b0.diff)}")


def plot_amplitude_distribution(trials, header):
    """Histogram of stim amplitudes used across an HRS2 session."""
    import matplotlib.pyplot as plt
    if not trials:
        print("No trials to plot.")
        return

    amp_counts: dict = defaultdict(int)
    for t in trials:
        amp_counts[round(t.stimulation_amplitude_ma, 2)] += 1

    amplitudes = sorted(amp_counts.keys())
    counts = [amp_counts[a] for a in amplitudes]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(amplitudes, counts, width=0.01,
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Stimulation Intensity (mA)', fontsize=12)
    ax.set_ylabel('Count / Number of Repeats', fontsize=12)
    ax.set_title(f'Stimulation Intensities Distribution - {header.subject_id}',
                 fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.show()


def plot_background_emg_views(trials, emg_blocks, monitoring_window_ms: float = 2500.0):
    """Interactive 'Most recent background' bar chart + 'Background EMG Level' scatter.

    Mirrors the H-Reflex App recruitment-curve trial plot widgets:
      _update_most_recent_background_plot  (bar chart of pre-stim |EMG| bins)
      _update_emg_level_plot               (scatter of grand means per trial)
    """
    import matplotlib.pyplot as plt
    from ipywidgets import IntSlider, Output, VBox, HBox
    from IPython.display import display

    if len(trials) == 0:
        print("No HRS2 trials to plot.")
        return

    bins_per_trial = []
    grand_means = []
    for trial in trials:
        bins, gm = compute_background_bins(
            trial, emg_blocks, monitoring_window_ms=monitoring_window_ms)
        bins_per_trial.append(bins)
        grand_means.append(gm)

    valid_idx = [i for i, gm in enumerate(grand_means) if not np.isnan(gm)]
    if not valid_idx:
        print("Could not reconstruct any background windows from emg_blocks.")
        return

    out = Output()
    slider = IntSlider(value=valid_idx[-1], min=0, max=len(trials) - 1, step=1,
                       description='Trial:', layout={'width': '600px'},
                       continuous_update=False)

    def _draw(idx):
        with out:
            out.clear_output(wait=True)
            fig, (ax_bg, ax_lvl) = plt.subplots(1, 2, figsize=(15, 5))

            bins = bins_per_trial[idx]
            if bins is None:
                ax_bg.text(0.5, 0.5, f'Trial {idx}: no background reconstructed',
                           ha='center', va='center', transform=ax_bg.transAxes)
            else:
                gm = grand_means[idx]
                x = np.arange(len(bins))
                ax_bg.bar(x, bins, width=0.8, color=(70/255, 130/255, 180/255))
                ax_bg.axhline(gm, color='red', linestyle='--', linewidth=2,
                              label=f'Mean={gm:.2f}')
                ax_bg.legend(loc='upper right')
            ax_bg.set_xlabel('Bin #')
            ax_bg.set_ylabel('EMG (µV)')
            ax_bg.set_title(f'Most Recent Background  (Trial {idx})')
            ax_bg.grid(True, alpha=0.3, axis='y')

            ax_lvl.scatter(valid_idx, [grand_means[i] for i in valid_idx],
                           s=40, color=(0, 0, 200/255),
                           edgecolor='black', linewidth=0.4)
            if not np.isnan(grand_means[idx]):
                ax_lvl.scatter([idx], [grand_means[idx]], s=160, marker='*',
                               color='gold', edgecolor='black', linewidth=0.6,
                               zorder=5, label=f'Selected (Trial {idx})')
            tr = trials[idx]
            ax_lvl.axhline(tr.min_initiation_threshold, color=(0, 160/255, 0),
                           linestyle='--', linewidth=1.5,
                           label=f'Min thresh: {tr.min_initiation_threshold:.1f}')
            ax_lvl.axhline(tr.max_initiation_threshold, color=(200/255, 0, 0),
                           linestyle='--', linewidth=1.5,
                           label=f'Max thresh: {tr.max_initiation_threshold:.1f}')
            ax_lvl.set_xlabel('Trial #')
            ax_lvl.set_ylabel('EMG Mean (µV)')
            ax_lvl.set_title('Background EMG Level')
            ax_lvl.legend(loc='upper right', fontsize=9)
            ax_lvl.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    slider.observe(lambda c: _draw(c['new']) if c['name'] == 'value' else None,
                   names='value')
    print(f"Reconstructed background bins for {len(valid_idx)}/{len(trials)} trials "
          f"(window={monitoring_window_ms} ms).")
    display(VBox([HBox([slider]), out]))
    _draw(slider.value)


def _apply_tiered_ticks(ax, axis: str = 'both'):
    """Three-tier tick sizing: labelled (major) big, midpoint minor medium,
    other minor small. Caller is responsible for setting major ticks; minor
    ticks are auto-populated via AutoMinorLocator(10) only if not preset.
    """
    from matplotlib.ticker import AutoMinorLocator

    axes_iter = []
    if axis in ('x', 'both'):
        axes_iter.append(ax.xaxis)
    if axis in ('y', 'both'):
        axes_iter.append(ax.yaxis)

    for axis_obj in axes_iter:
        if axis_obj.get_minorticklocs().size == 0:
            axis_obj.set_minor_locator(AutoMinorLocator(10))

    ax.tick_params(axis=axis, which='major', length=10, width=1.8)
    ax.tick_params(axis=axis, which='minor', length=3,  width=0.8)

    ax.figure.canvas.draw()

    for axis_obj in axes_iter:
        major_locs = axis_obj.get_majorticklocs()
        if len(major_locs) < 2:
            continue
        step = float(major_locs[1] - major_locs[0])
        mid = step / 2.0
        tol = abs(step) * 1e-3 + 1e-9
        for tick in axis_obj.get_minor_ticks():
            rel = (tick.get_loc() - major_locs[0]) % step
            if abs(rel - mid) < tol:
                for line in (tick.tick1line, tick.tick2line):
                    line.set_markersize(6)
                    line.set_markeredgewidth(1.2)


def plot_hrs2_analysis(trials, header,
                       pre_avg_ms: float = 2.0, post_avg_ms: float = 15.0,
                       n_per_page: int = 6,
                       m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                       h_start_ms: float = 6.0, h_end_ms: float = 10.0):
    """Interactive averaged-waveform paged grid + zoom + recruitment curve.

    Each page shows ``n_per_page`` panels, one per stimulation amplitude, with raw
    bipolar trials, average bipolar, optional |bipolar|/unipolar overlays, ADC sync
    overlay, M/H wave region shading, and peak markers/labels. Click to select an
    amplitude in the dropdown; double-click to zoom. Following the grid we plot
    the normalized recruitment curve and the raw mean ± SEM curve.
    """
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.interpolate import interp1d
    from ipywidgets import (Button, Output, HBox, VBox, Dropdown, Label,
                            Checkbox, ToggleButton, FloatText, HTML)
    from IPython.display import display

    if len(trials) == 0:
        print("No HRS2 trials to group.")
        return

    _groups: dict = defaultdict(list)
    for _trial in trials:
        _key = round(_trial.stimulation_amplitude_ma, 2)
        _t_win, _bip_win, _adc_win, _stim_end = get_trial_window(
            _trial, pre_avg_ms, post_avg_ms)
        _t_uni, _uni_win, _, _ = get_trial_window(
            _trial, pre_avg_ms, post_avg_ms, use_unipolar=True)
        _groups[_key].append((_t_win, _bip_win, _adc_win, _uni_win, _stim_end))

    _sorted_amps = sorted(_groups.keys())

    def _pad_rows(rows, n_pts):
        p = np.full((len(rows), n_pts), np.nan)
        for k, a in enumerate(rows):
            if a is not None and len(a) > 0:
                _n = min(len(a), n_pts)
                p[k, :_n] = np.asarray(a[:_n], dtype=float)
        return p

    _amp_data = []
    for _amp in _sorted_amps:
        _wins  = _groups[_amp]
        _t_ref = _wins[0][0]
        _np    = len(_t_ref)

        _pb  = _pad_rows([w[1] for w in _wins], _np)
        _pa  = _pad_rows([w[2] for w in _wins], _np)
        _pu  = _pad_rows([w[3] for w in _wins], _np)
        _pab = np.abs(_pb)
        _pau = np.abs(_pu)

        _avg_b  = np.nanmean(_pb,  axis=0)
        _avg_a  = np.nanmean(_pa,  axis=0)
        _avg_u  = np.nanmean(_pu,  axis=0)
        _avg_ab = np.abs(_avg_b)
        _avg_au = np.abs(_avg_u)

        _se = [w[4] for w in _wins if w[4] is not None]
        _mse = float(np.mean(_se)) if _se else 0.5

        _mm = (_t_ref >= m_start_ms) & (_t_ref <= m_end_ms)
        _hm = (_t_ref >= h_start_ms) & (_t_ref <= h_end_ms)
        _mi = int(np.argmax(_avg_ab[_mm])) if _mm.any() else 0
        _hi = int(np.argmax(_avg_ab[_hm])) if _hm.any() else 0
        _m_t   = float(_t_ref[_mm][_mi])  if _mm.any() else m_start_ms
        _m_a   = float(_avg_ab[_mm][_mi]) if _mm.any() else float('nan')
        _m_bip = float(_avg_b[_mm][_mi])  if _mm.any() else float('nan')
        _h_t   = float(_t_ref[_hm][_hi])  if _hm.any() else h_start_ms
        _h_a   = float(_avg_ab[_hm][_hi]) if _hm.any() else float('nan')
        _h_bip = float(_avg_b[_hm][_hi])  if _hm.any() else float('nan')

        _amp_data.append({
            'amp': _amp, 't_ref': _t_ref, 'n': len(_wins),
            'mean_stim_end': _mse,
            'padded_bip': _pb,  'avg_bip': _avg_b,
            'padded_adc': _pa,  'avg_adc': _avg_a,
            'padded_uni': _pu,  'avg_uni': _avg_u,
            'padded_abs_bip': _pab, 'avg_abs_bip': _avg_ab,
            'padded_abs_uni': _pau, 'avg_abs_uni': _avg_au,
            'm_peak_time': _m_t, 'm_peak_amp': _m_a, 'm_peak_bip': _m_bip,
            'h_peak_time': _h_t, 'h_peak_amp': _h_a, 'h_peak_bip': _h_bip,
        })

    _show_sigs = {'val': set()}
    _ylim_auto = {'val': True}
    _ylim_man  = {'lo': -5000.0, 'hi': 5500.0}

    def _get_ylim():
        if _ylim_auto['val']:
            _arrays = [d['padded_bip'] for d in _amp_data]
            for _sig in ('abs_bip', 'uni', 'abs_uni'):
                if _sig in _show_sigs['val']:
                    _arrays += [d[f'padded_{_sig}'] for d in _amp_data]
            _all = np.concatenate([a.flatten() for a in _arrays])
            _all = _all[~np.isnan(_all)]
            if len(_all) == 0:
                return (-1000.0, 1500.0)
            _lo, _hi = float(np.nanmin(_all)), float(np.nanmax(_all))
            _pad = max(0.08 * (_hi - _lo), 1.0)
            return (_lo - _pad, _hi + _pad)
        return (_ylim_man['lo'], _ylim_man['hi'])

    def _draw_avg_panel(ax, d, small=True):
        lw  = 0.8 if small else 1.5
        fsz = 8   if small else 11
        t   = d['t_ref']
        end_ms = d['mean_stim_end']
        _text_off = 150 if small else 220
        sigs = _show_sigs['val']

        ax.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.4, zorder=1)

        for _row in d['padded_bip']:
            ax.plot(t, _row, color='red', alpha=0.6, linewidth=lw * 0.7)
        ax.plot(t, d['avg_bip'], color='black', linewidth=lw * 2.5, label='Avg Bipolar')

        if 'abs_bip' in sigs:
            for _row in d['padded_abs_bip']:
                ax.plot(t, _row, color='gray', alpha=0.25, linewidth=lw * 0.5)
            ax.plot(t, d['avg_abs_bip'], color='gray', linewidth=lw * 1.8,
                    label='|Bipolar| avg')

        if 'uni' in sigs:
            for _row in d['padded_uni']:
                ax.plot(t, _row, color='orange', alpha=0.35, linewidth=lw * 0.5)
            ax.plot(t, d['avg_uni'], color='orange', linewidth=lw * 1.8,
                    label='Avg Unipolar')

        if 'abs_uni' in sigs:
            for _row in d['padded_abs_uni']:
                ax.plot(t, _row, color='purple', alpha=0.25, linewidth=lw * 0.5)
            ax.plot(t, d['avg_abs_uni'], color='purple', linewidth=lw * 1.8,
                    label='|Unipolar| avg')

        if 'adc' in sigs:
            _ax2 = ax.twinx()
            for _row in d['padded_adc']:
                _ax2.plot(t, _row, color='green', alpha=0.25, linewidth=lw * 0.4)
            _ax2.plot(t, d['avg_adc'], color='green', linewidth=lw * 1.8,
                      label='ADC sync')
            _ax2.set_ylabel('ADC (V)', color='green', fontsize=fsz - 1)
            _ax2.tick_params(axis='y', labelcolor='green', labelsize=fsz - 2)

        ax.axvspan(0, end_ms, color='red', alpha=0.20)
        ax.axvline(0,      color='red', linestyle='--', linewidth=lw)
        ax.axvline(end_ms, color='red', linestyle='--', linewidth=lw)

        ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.3)
        ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.3)

        m_t, m_a = d['m_peak_time'], d['m_peak_amp']
        h_t, h_a = d['h_peak_time'], d['h_peak_amp']
        m_bip = d.get('m_peak_bip', m_a)
        h_bip = d.get('h_peak_bip', h_a)
        _msize = 8 if small else 14
        ax.axvline(m_t, color='blue',  linestyle=':', linewidth=lw * 1.2,
                   label=f'M-peak: {m_a:.1f} uV')
        ax.axvline(h_t, color='green', linestyle=':', linewidth=lw * 1.2,
                   label=f'H-peak: {h_a:.1f} uV')
        ax.plot(m_t, m_bip, '*', color='blue',  markersize=_msize, zorder=6,
                markeredgecolor='darkblue', markeredgewidth=0.5)
        ax.plot(h_t, h_bip, '*', color='green', markersize=_msize, zorder=6,
                markeredgecolor='darkgreen', markeredgewidth=0.5)
        if not np.isnan(m_bip):
            ax.text(m_t, m_bip + _text_off, f'{m_a:.1f} uV',
                    color='blue', fontsize=fsz - 1, ha='center')
        if not np.isnan(h_bip):
            ax.text(h_t, h_bip + _text_off, f'{h_a:.1f} uV',
                    color='green', fontsize=fsz - 1, ha='center')

        ax.set_xlim(-pre_avg_ms, post_avg_ms)
        ax.set_ylim(_get_ylim())
        ax.set_xlabel('Time re: onset (ms)', fontsize=fsz)
        ax.set_ylabel('EMG (uV)', fontsize=fsz)
        ax.tick_params(labelsize=fsz - 1)
        ax.grid(True, alpha=0.3)
        _min_ms = int(np.floor(t[0]))
        _max_ms = int(np.ceil(t[-1]))
        ax.set_xticks(np.arange(_min_ms, _max_ms + 1, 1))
        if not small:
            ax.legend(fontsize=fsz - 2, loc='upper right')

    _pages    = [_amp_data[i:i+n_per_page] for i in range(0, len(_amp_data), n_per_page)]
    _cur_page = {'idx': 0}
    _out      = Output()
    _amp_drop = Dropdown(description='Amplitude:')

    def _plot_page(page_idx):
        with _out:
            _out.clear_output(wait=True)
            page = _pages[page_idx]
            n    = len(page)

            _amp_drop.options = [
                (f"{d['amp']:.2f} mA (n={d['n']})", page_idx * n_per_page + i)
                for i, d in enumerate(page)
            ]
            if _amp_drop.options:
                _amp_drop.value = _amp_drop.options[0][1]

            fig, axs  = plt.subplots(2, 3, figsize=(15, 7))
            _axs_flat = axs.flatten()

            for j in range(n_per_page):
                ax = _axs_flat[j]
                if j < n:
                    d = page[j]
                    _draw_avg_panel(ax, d, small=True)
                    ax.set_title(f"{d['amp']:.2f} mA  (n={d['n']})", fontsize=24)
                else:
                    ax.axis('off')

            _n_start = n_per_page * page_idx + 1
            _n_end   = min(n_per_page * (page_idx + 1), len(_amp_data))
            fig.suptitle(
                f"{header.subject_id}    "
                f"Amplitudes {_n_start}-{_n_end} of {len(_amp_data)}"
                f"  (Page {page_idx+1}/{len(_pages)})",
                fontsize=11
            )
            plt.tight_layout()
            plt.show()

            try:
                def _on_click(event):
                    if event.inaxes is None:
                        return
                    for k, ax in enumerate(_axs_flat):
                        if event.inaxes is ax and k < n:
                            if getattr(event, 'dblclick', False):
                                _show_zoom(page_idx * n_per_page + k)
                            else:
                                _amp_drop.value = page_idx * n_per_page + k
                            break
                fig.canvas.mpl_connect('button_press_event', _on_click)
            except Exception:
                pass

    def _show_zoom(amp_idx):
        d = _amp_data[amp_idx]
        with _out:
            _out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(13, 5))
            _draw_avg_panel(ax, d, small=False)
            ax.set_title(
                f"Averaged Waveforms (n={d['n']}) | Stim Amp: {d['amp']:.2f} mA | "
                f"{header.subject_id}  "
                f"({header.session_start_time:%Y-%m-%d %H:%M})",
                fontsize=12
            )
            plt.tight_layout()
            plt.show()

            _back_btn = Button(description='Back to grid', button_style='info')
            _back_btn.on_click(lambda b: _plot_page(_cur_page['idx']))
            display(_back_btn)

    def _on_prev(b):
        if _cur_page['idx'] > 0:
            _cur_page['idx'] -= 1
            _page_drop.value = _cur_page['idx']
            _plot_page(_cur_page['idx'])

    def _on_next(b):
        if _cur_page['idx'] < len(_pages) - 1:
            _cur_page['idx'] += 1
            _page_drop.value = _cur_page['idx']
            _plot_page(_cur_page['idx'])

    def _on_page_change(change):
        if change['name'] == 'value' and change['new'] != _cur_page['idx']:
            _cur_page['idx'] = change['new']
            _plot_page(_cur_page['idx'])

    def _on_view(b):
        try:
            _show_zoom(int(_amp_drop.value))
        except Exception as e:
            print(f'Could not zoom amplitude: {e}')

    def _make_sig_cb(key):
        def _cb(change):
            if change['new']:
                _show_sigs['val'].add(key)
            else:
                _show_sigs['val'].discard(key)
            _plot_page(_cur_page['idx'])
        return _cb

    def _on_auto_toggle(change):
        _ylim_auto['val'] = bool(change['new'])
        _ymin_box.disabled = bool(change['new'])
        _ymax_box.disabled = bool(change['new'])
        _plot_page(_cur_page['idx'])

    def _on_ymin_change(change):
        _ylim_man['lo'] = float(change['new'])
        if not _ylim_auto['val']:
            _plot_page(_cur_page['idx'])

    def _on_ymax_change(change):
        _ylim_man['hi'] = float(change['new'])
        if not _ylim_auto['val']:
            _plot_page(_cur_page['idx'])

    _prev_btn  = Button(description='Prev',           button_style='')
    _next_btn  = Button(description='Next',           button_style='primary')
    _page_drop = Dropdown(
        options=[(f'Page {i+1}', i) for i in range(len(_pages))],
        description='Page:', layout={'width': '130px'}
    )
    _view_btn  = Button(description='View amplitude', button_style='info')

    _cb_adc     = Checkbox(value=False, description='ADC sync (green)',    indent=False,
                           layout={'width': '185px'})
    _cb_abs_bip = Checkbox(value=False, description='|Bipolar| (gray)',    indent=False,
                           layout={'width': '175px'})
    _cb_uni     = Checkbox(value=False, description='Unipolar (orange)',   indent=False,
                           layout={'width': '185px'})
    _cb_abs_uni = Checkbox(value=False, description='|Unipolar| (purple)', indent=False,
                           layout={'width': '195px'})

    _auto_toggle = ToggleButton(
        value=True, description='Auto y-scale',
        button_style='success',
        tooltip='Auto-scale shared y-axis from visible signals'
    )
    _ymin_box = FloatText(value=-5000.0, description='Y min:',
                          disabled=True, layout={'width': '145px'})
    _ymax_box = FloatText(value=5500.0,  description='Y max:',
                          disabled=True, layout={'width': '145px'})

    _prev_btn.on_click(_on_prev)
    _next_btn.on_click(_on_next)
    _page_drop.observe(_on_page_change, names='value')
    _view_btn.on_click(_on_view)
    _cb_adc.observe(_make_sig_cb('adc'), names='value')
    _cb_abs_bip.observe(_make_sig_cb('abs_bip'), names='value')
    _cb_uni.observe(_make_sig_cb('uni'), names='value')
    _cb_abs_uni.observe(_make_sig_cb('abs_uni'), names='value')
    _auto_toggle.observe(_on_auto_toggle, names='value')
    _ymin_box.observe(_on_ymin_change, names='value')
    _ymax_box.observe(_on_ymax_change, names='value')

    _nav_row = HBox([_prev_btn, _next_btn, _page_drop,
                     Label('  '), _amp_drop, _view_btn])
    _sig_row = HBox([
        VBox([
            HTML('<b>Signal overlays:</b>'),
            HBox([_cb_adc, _cb_abs_bip, _cb_uni, _cb_abs_uni]),
        ]),
        Label('   '),
        VBox([_auto_toggle, _ymin_box, _ymax_box]),
    ])

    print(f"Loaded {len(_amp_data)} amplitude groups across {len(_pages)} page(s).")
    print("Double-click a subplot to zoom in. Ctrl-click to select multiple signals.")

    display(VBox([_nav_row, _sig_row, _out]))
    _plot_page(0)

    # ---- Recruitment Curve ----
    _PRE_RC  = max(m_start_ms, h_start_ms) + 1.0
    _POST_RC = h_end_ms + 2.0

    m_wave_dict: dict = defaultdict(list)
    h_wave_dict: dict = defaultdict(list)

    for trial in trials:
        amp_key = round(trial.stimulation_amplitude_ma, 2)
        t_ms, emg, _, _ = get_trial_window(trial, _PRE_RC, _POST_RC)

        m_mask = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
        if np.any(m_mask):
            m_wave_dict[amp_key].append(np.max(np.abs(emg[m_mask])))

        h_mask = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
        if np.any(h_mask):
            h_wave_dict[amp_key].append(np.max(np.abs(emg[h_mask])))

    sorted_amps = sorted(set(m_wave_dict.keys()) | set(h_wave_dict.keys()))
    m_wave_data = [m_wave_dict.get(a, [0]) for a in sorted_amps]
    h_wave_data = [h_wave_dict.get(a, [0]) for a in sorted_amps]

    m_means = np.array([np.mean(v) for v in m_wave_data])
    h_means = np.array([np.mean(v) for v in h_wave_data])
    m_sems  = np.array([stats.sem(v) if len(v) > 1 else 0 for v in m_wave_data])
    h_sems  = np.array([stats.sem(v) if len(v) > 1 else 0 for v in h_wave_data])

    M_max = np.max(m_means) if np.max(m_means) > 0 else 1
    m_means_norm = (m_means / M_max) * 100
    h_means_norm = (h_means / M_max) * 100
    m_sems_norm  = (m_sems  / M_max) * 100
    h_sems_norm  = (h_sems  / M_max) * 100

    interp_func = interp1d(m_means_norm, sorted_amps, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    try:
        current_at_50 = float(interp_func(50))
    except Exception:
        current_at_50 = sorted_amps[np.argmax(m_means_norm >= 50)]

    normalized_currents = np.array(sorted_amps) / current_at_50

    H_max = np.max(h_means_norm)
    idx_Hmax = int(np.argmax(h_means_norm))
    current_at_Hmax_norm = normalized_currents[idx_Hmax]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(normalized_currents - 0.02, m_means_norm, yerr=m_sems_norm,
                fmt='o-', color='blue',  label='M-wave (% Mmax)', capsize=3)
    ax.errorbar(normalized_currents + 0.02, h_means_norm, yerr=h_sems_norm,
                fmt='o-', color='green', label='H-wave (% Mmax)', capsize=3)

    ax.axhline(H_max, color='green', linestyle='--', linewidth=1,
               label=f'H_max = {H_max:.1f}% Mmax')
    ax.axvline(current_at_Hmax_norm, color='gray', linestyle='--', linewidth=1,
               label=f'Current at H_max = {current_at_Hmax_norm:.2f}x')

    ax.text(current_at_Hmax_norm + 0.02, H_max + 2, 'b', fontsize=12, color='black')
    ax.text(normalized_currents[idx_Hmax] - 0.08, H_max + 2, 'a', fontsize=12, color='black')

    ax.set_xlabel('Current (normalized to current at 50% Mmax)', fontsize=18)
    ax.set_ylabel('H and M wave amplitude (% of Mmax)', fontsize=18)
    ax.set_title(f'HRS2 Normalized Recruitment Curve - {header.subject_id}',
                 fontsize=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_tiered_ticks(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(sorted_amps))

    ax.errorbar(positions - 0.2, m_means, yerr=m_sems, fmt='o-', color='blue',
                label='M-wave mean ± SEM', capsize=3)
    ax.errorbar(positions + 0.2, h_means, yerr=h_sems, fmt='o-', color='green',
                label='H-wave mean ± SEM', capsize=3)

    labelled_idx = list(range(0, len(sorted_amps), 10))
    other_idx    = [i for i in range(len(sorted_amps)) if i not in labelled_idx]
    ax.set_xticks(labelled_idx)
    ax.set_xticklabels([f'{sorted_amps[i]:.1f}' for i in labelled_idx])
    ax.set_xticks(other_idx, minor=True)
    ax.set_xlabel('Stimulation Amplitude (mA)', fontsize=18)
    ax.set_ylabel('Peak Amplitude (µV)', fontsize=18)
    ax.set_title(f'HRS2 Recruitment Curve - {header.subject_id}', fontsize=15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _apply_tiered_ticks(ax)
    plt.tight_layout()
    plt.show()

    print(f"M_max = {M_max:.2f} µV")
    print(f"H_max = {np.max(h_means):.2f} µV ({H_max:.1f}% of M_max)")
    print(f"Current at 50% M_max = {current_at_50:.2f} mA")
    print(f"Current at H_max = {sorted_amps[idx_Hmax]:.2f} mA "
          f"({current_at_Hmax_norm:.2f}x normalized)")


def analyze_global_background(trials, emg_blocks, header,
                              sample_rate: float = SAMPLE_RATE,
                              blank_pre_ms: float = 5.0,
                              blank_post_ms: float = 20.0,
                              min_valid_frac: float = 0.7,
                              bin_samples: int = BIN_SAMPLES,
                              show_plots: bool = True):
    """Post-hoc global windowing: stitch continuous |EMG|, blank around stims,
    compute background statistics, extract per-trial pre-stim grand means.

    Sections 4a-4d of Post_Hoc_Global_Windowing.ipynb in one call. Returns a
    state dict consumed by ``run_threshold_sweep`` / ``plot_threshold_sweep``.
    """
    import matplotlib.pyplot as plt

    if not emg_blocks:
        print("No HRS2 EMG blocks available — skipping.")
        return None

    # ---- 4a: Sort and stitch blocks ----
    print("=" * 70)
    print("Section 4a: Reconstructing continuous signal from HRS2 EMG blocks")
    print("=" * 70)

    sorted_blks = sorted(emg_blocks, key=lambda b: int(b.ts_open_ephys_sent))
    first_oe = int(sorted_blks[0].ts_open_ephys_sent)
    last_blk = sorted_blks[-1]
    last_oe = int(last_blk.ts_open_ephys_sent) + len(last_blk.abs_val)
    n_total = last_oe - first_oe
    duration_s = n_total / sample_rate

    continuous_abs_emg = np.zeros(n_total, dtype=np.float32)
    for blk in sorted_blks:
        i0 = int(blk.ts_open_ephys_sent) - first_oe
        i1 = i0 + len(blk.abs_val)
        if 0 <= i0 and i1 <= n_total:
            continuous_abs_emg[i0:i1] = blk.abs_val
    continuous_abs_emg = np.abs(continuous_abs_emg)

    timestamps = np.arange(n_total, dtype=np.float64) / sample_rate

    gap_samples = sum(
        max(0, int(sorted_blks[k].ts_open_ephys_sent)
               - (int(sorted_blks[k-1].ts_open_ephys_sent) + len(sorted_blks[k-1].abs_val)))
        for k in range(1, len(sorted_blks))
    )
    print(f"  Blocks stitched : {len(sorted_blks):,}")
    print(f"  Total samples   : {n_total:,}")
    print(f"  Duration        : {duration_s:.1f} s  ({duration_s / 60:.1f} min)")
    print(f"  OE sample range : {first_oe} – {last_oe}")
    print(f"  Gap samples     : {gap_samples:,}  ({100 * gap_samples / n_total:.3f}% of total)")

    # ---- 4b: Blank mask around stim events ----
    print("\n" + "=" * 70)
    print("Section 4b: Building blank mask around stimulation events")
    print("=" * 70)

    stim_rel = []
    n_no_onset = 0
    for tr in trials:
        fid = int(getattr(tr, 'first_post_trigger_frame_sample_id', 0))
        osi = int(getattr(tr, 'onset_sample_index', -1))
        if fid > 0 and osi >= 0:
            rel = fid + (osi - bin_samples) - first_oe
            if 0 <= rel < n_total:
                stim_rel.append(rel)
            else:
                n_no_onset += 1
        else:
            n_no_onset += 1

    stim_times_s = np.array(stim_rel, dtype=np.float64) / sample_rate
    blank_mask = build_blank_mask(timestamps, stim_times_s, sample_rate,
                                  blank_pre_ms, blank_post_ms)
    print(f"  Stim events mapped  : {len(stim_rel)} of {len(trials)}")
    print(f"  Unmapped (no onset) : {n_no_onset}")
    print(f"  Non-blanked samples : {blank_mask.sum():,}  ({100 * blank_mask.mean():.1f}%)")

    # ---- 4c: Background EMG statistics ----
    print("\n" + "=" * 70)
    print("Section 4c: Background EMG statistics (non-blanked samples)")
    print("=" * 70)

    bg_signal = continuous_abs_emg[blank_mask]
    bg_mean = float(np.mean(bg_signal))
    bg_std = float(np.std(bg_signal))
    bg_q1, bg_med, bg_q3 = (float(x) for x in np.percentile(bg_signal, [25, 50, 75]))

    print(f"  n       : {len(bg_signal):,}")
    print(f"  Mean    : {bg_mean:.3f} µV")
    print(f"  Std     : {bg_std:.3f} µV")
    print(f"  Q1      : {bg_q1:.3f} µV")
    print(f"  Median  : {bg_med:.3f} µV")
    print(f"  Q3      : {bg_q3:.3f} µV")

    if show_plots:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.hist(bg_signal, bins=300, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(bg_mean, color='red', linestyle='-', linewidth=2.0,
                   label=f'Global grand mean = {bg_mean:.2f} µV')
        for val, lbl, col in [(bg_q1, 'Q1', 'orange'),
                              (bg_med, 'Median', 'purple'),
                              (bg_q3, 'Q3', 'darkorange')]:
            ax.axvline(val, color=col, linestyle='--', linewidth=1.5,
                       label=f'{lbl} = {val:.2f} µV')
        ax.set_xlabel('Abs EMG amplitude (µV)')
        ax.set_ylabel('Sample count (log scale)')
        ax.set_yscale('log')
        ax.set_title(
            f'Background EMG distribution — {header.subject_id}\n'
            f'(blanked ±{blank_pre_ms}/{blank_post_ms} ms around {len(stim_rel)} stims)'
        )
        ax.legend()
        plt.tight_layout()
        plt.show()

    # ---- 4d: Per-trial pre-stim grand means ----
    print("\n" + "=" * 70)
    print("Section 4d: Per-trial pre-stim background grand means")
    print("=" * 70)

    trial_bg_gm = []
    trial_min_th = []
    trial_max_th = []
    for tr in trials:
        osi = int(getattr(tr, 'onset_sample_index', -1))
        pre = tr.trial_data[:osi] if osi > 0 else tr.trial_data[:bin_samples]
        if len(pre) > 0:
            trial_bg_gm.append(float(np.mean(np.abs(pre))))
        trial_min_th.append(float(tr.min_initiation_threshold))
        trial_max_th.append(float(tr.max_initiation_threshold))

    trial_bg_gm = np.array(trial_bg_gm, dtype=np.float64)
    trial_min_th = np.array(trial_min_th, dtype=np.float64)
    trial_max_th = np.array(trial_max_th, dtype=np.float64)

    gm_q1, gm_med, gm_q3 = (float(x) for x in np.percentile(trial_bg_gm, [25, 50, 75]))

    print(f"  Trials : {len(trial_bg_gm)}")
    print(f"  Min={trial_bg_gm.min():.2f}  Q1={gm_q1:.2f}  Median={gm_med:.2f}  "
          f"Q3={gm_q3:.2f}  Max={trial_bg_gm.max():.2f}")
    print(f"  Recorded thresholds (mean): "
          f"[{trial_min_th.mean():.2f}, {trial_max_th.mean():.2f}] µV")

    if show_plots:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.hist(trial_bg_gm, bins=80, color='mediumseagreen',
                edgecolor='black', linewidth=0.4, alpha=0.85)
        ax.axvline(gm_q1, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Q1 = {gm_q1:.2f} µV')
        ax.axvline(gm_med, color='purple', linestyle='-.', linewidth=1.5,
                   label=f'Median = {gm_med:.2f} µV')
        ax.axvline(gm_q3, color='darkorange', linestyle=':', linewidth=1.5,
                   label=f'Q3 = {gm_q3:.2f} µV')
        ax.axvline(trial_min_th.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Recorded min-thresh = {trial_min_th.mean():.2f} µV')
        ax.axvline(trial_max_th.mean(), color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Recorded max-thresh = {trial_max_th.mean():.2f} µV')
        ax.set_xlabel('Pre-stim grand mean (µV)')
        ax.set_ylabel('Trial count')
        ax.set_title(
            f'Per-trial pre-stim background grand mean — {header.subject_id}  '
            f'(n = {len(trial_bg_gm)})\n'
            f'(window = |trial_data[:onset_sample_index]|)'
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return {
        'continuous_abs_emg': continuous_abs_emg,
        'timestamps': timestamps,
        'sample_rate': sample_rate,
        'blank_mask': blank_mask,
        'first_oe': first_oe,
        'n_total': n_total,
        'duration_s': duration_s,
        'gap_samples': gap_samples,
        'stim_rel': stim_rel,
        'n_no_onset': n_no_onset,
        'bg_mean': bg_mean, 'bg_std': bg_std,
        'bg_q1': bg_q1, 'bg_med': bg_med, 'bg_q3': bg_q3,
        'trial_bg_gm': trial_bg_gm,
        'trial_min_th': trial_min_th,
        'trial_max_th': trial_max_th,
        'gm_q1': gm_q1, 'gm_med': gm_med, 'gm_q3': gm_q3,
        'blank_pre_ms': blank_pre_ms,
        'blank_post_ms': blank_post_ms,
        'min_valid_frac': min_valid_frac,
    }


def run_threshold_sweep(state, sweep_centres=None,
                        half_widths_uv=None,
                        seed: int = 42,
                        min_ms: int = TRIAL_INIT_MIN_MS,
                        max_ms: int = TRIAL_INIT_MAX_MS,
                        n_trials: int = 0):
    """Section 5a: walk the continuous background with sweep_trials() across
    a grid of (centre ± half-width) threshold windows.

    ``state`` is the dict returned by ``analyze_global_background``.
    Defaults sweep_centres to {Q1, Median, Q3} of the per-trial pre-stim
    distribution and half_widths_uv to [5, 10, 20, 30, 50, 75, 100, 150].
    Returns sweep_results dict keyed by centre label.
    """
    if state is None:
        print("No state — run analyze_global_background first.")
        return None

    if sweep_centres is None:
        sweep_centres = {'Q1': state['gm_q1'],
                         'Median': state['gm_med'],
                         'Q3': state['gm_q3']}
    if half_widths_uv is None:
        half_widths_uv = [5, 10, 20, 30, 50, 75, 100, 150]

    print("=" * 70)
    print("Section 5a: Running post-hoc threshold sweep")
    print("=" * 70)
    print(f"Signal duration : {state['duration_s']:.1f} s | "
          f"Non-blanked: {100 * state['blank_mask'].mean():.1f}%")
    print(f"Background stats: mean={state['bg_mean']:.2f}  "
          f"Q1={state['gm_q1']:.2f}  Med={state['gm_med']:.2f}  "
          f"Q3={state['gm_q3']:.2f} µV\n")

    sweep_results: dict = {}
    for centre_label, centre_val in sweep_centres.items():
        sweep_results[centre_label] = []
        for hw in half_widths_uv:
            min_uv = max(0.0, centre_val - hw)
            max_uv = centre_val + hw
            accepted = sweep_trials(
                state['continuous_abs_emg'], state['timestamps'],
                state['blank_mask'], state['sample_rate'],
                min_ms=min_ms, max_ms=max_ms,
                min_uv=min_uv, max_uv=max_uv,
                min_valid_fraction=state['min_valid_frac'],
                seed=seed,
            )
            sweep_results[centre_label].append({
                'hw': hw, 'min_uv': min_uv, 'max_uv': max_uv,
                'n_accepted': len(accepted),
            })

        row_str = '  '.join(
            f'±{r["hw"]}→{r["n_accepted"]}' for r in sweep_results[centre_label]
        )
        print(f"  {centre_label:6s} (centre={centre_val:.2f} µV):  {row_str}")

    if n_trials:
        print(f"\n  Actual HRS2 trials recorded during session: {n_trials}")
    return sweep_results


def plot_threshold_sweep(sweep_results, state, trials, header):
    """Section 5b: visualise sweep results + summary table.

    Three panels: accepted vs half-width, accepted/actual ratio, threshold
    windows overlaid on the per-trial background histogram. Then prints a
    formatted summary table.
    """
    import matplotlib.pyplot as plt
    if not sweep_results:
        print("No sweep results to plot.")
        return

    n_actual = len(trials)
    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette = ['darkorange', 'purple', 'steelblue', 'teal', 'crimson']

    def _colour(label, i):
        return colours.get(label, palette[i % len(palette)])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for i, (label, rows) in enumerate(sweep_results.items()):
        hws = [r['hw'] for r in rows]
        ns = [r['n_accepted'] for r in rows]
        ax.plot(hws, ns, 'o-', label=label, color=_colour(label, i), linewidth=1.8)
    ax.axhline(n_actual, color='gray', linestyle='--', linewidth=1.5,
               label=f'Actual ({n_actual} trials)')
    ax.set_xlabel('Threshold half-width (µV)')
    ax.set_ylabel('Accepted virtual trials (post-hoc)')
    ax.set_title('Post-Hoc Sweep: Accepted Trials vs Threshold Window Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, (label, rows) in enumerate(sweep_results.items()):
        hws = [r['hw'] for r in rows]
        ratio = [r['n_accepted'] / max(n_actual, 1) for r in rows]
        ax2.plot(hws, ratio, 'o-', label=label, color=_colour(label, i), linewidth=1.8)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='1× actual')
    ax2.set_xlabel('Threshold half-width (µV)')
    ax2.set_ylabel('Accepted / Actual ratio')
    ax2.set_title('Post-Hoc Sweep: Accepted / Actual Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{header.subject_id} — Post-Hoc Threshold Sweep', fontsize=13)
    plt.tight_layout()
    plt.show()

    fig2, ax3 = plt.subplots(figsize=(13, 5))
    ax3.hist(state['trial_bg_gm'], bins=80, color='lightgray', edgecolor='black',
             linewidth=0.5, alpha=0.9, label='Per-trial pre-stim BG')
    for i, (label, rows) in enumerate(sweep_results.items()):
        col = _colour(label, i)
        mid_row = rows[len(rows) // 2]
        ax3.axvspan(mid_row['min_uv'], mid_row['max_uv'], alpha=0.25, color=col,
                    label=f"{label} ±{mid_row['hw']} µV → {mid_row['n_accepted']} trials")
    ax3.axvline(state['bg_mean'], color='black', linestyle='-', linewidth=2.0,
                label=f"Global grand mean = {state['bg_mean']:.2f} µV")
    ax3.axvline(state['trial_min_th'].mean(), color='red', linestyle='--', linewidth=1.5,
                label=f"Recorded min-thresh ({state['trial_min_th'].mean():.2f})")
    ax3.axvline(state['trial_max_th'].mean(), color='darkred', linestyle='--', linewidth=1.5,
                label=f"Recorded max-thresh ({state['trial_max_th'].mean():.2f})")
    ax3.set_xlabel('Pre-stim grand mean (µV)')
    ax3.set_ylabel('Trial count')
    ax3.set_title(f'Threshold Windows vs Background Distribution — {header.subject_id}')
    ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"\nGlobal EMG grand mean = {state['bg_mean']:.3f} µV  "
          f"(std = {state['bg_std']:.3f}, Q1 = {state['bg_q1']:.3f}, "
          f"median = {state['bg_med']:.3f}, Q3 = {state['bg_q3']:.3f})")
    print("This is the divisor used downstream for normalising M-/H-wave sizes.")

    print(f"\n{'Centre':>10} {'±hw (µV)':>10} {'Min (µV)':>10} {'Max (µV)':>10} "
          f"{'Accepted':>10} {'vs Actual':>12}")
    print("-" * 67)
    for label, rows in sweep_results.items():
        for r in rows:
            ratio_s = f"{r['n_accepted'] / max(n_actual, 1):.2f}×"
            print(f"{label:>10} {r['hw']:>10} {r['min_uv']:>10.2f} {r['max_uv']:>10.2f} "
                  f"{r['n_accepted']:>10} {ratio_s:>12}")
        print()


def compute_trial_responses(trials, bg_divisor,
                            m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                            h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                            pre_ms: float = 2.0, post_ms: float = 15.0,
                            metric: str = 'ptp'):
    """Per-trial M-/H-response sizes within each window, plus the same sizes
    normalised by ``bg_divisor``.

    bg_divisor:
        scalar -- single global EMG grand mean (recommended; same divisor for
                  every trial, e.g. ``state['bg_mean']``)
        array  -- per-trial background, one value per trial, shape (n_trials,)

    metric:
        'ptp'  -- peak-to-peak  =  max(emg) - min(emg)        (default)
        'peak' -- absolute peak =  max(|emg|)

    Returns a dict with 1-D arrays of length ``len(trials)``:
        m_size, h_size  -- response size in µV (per ``metric``)
        m_norm, h_norm  -- size / bg_divisor  (NaN where divisor <= 0)
    plus the divisor and window/metric used for downstream labelling.
    """
    if metric not in ('ptp', 'peak'):
        raise ValueError(f"metric must be 'ptp' or 'peak', got {metric!r}")

    def _size(values):
        if metric == 'ptp':
            return float(np.ptp(values))
        return float(np.max(np.abs(values)))

    n = len(trials)
    m_size = np.full(n, np.nan)
    h_size = np.full(n, np.nan)
    for i, trial in enumerate(trials):
        t_ms, emg, _, _ = get_trial_window(trial, pre_ms, post_ms)
        m_mask = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
        h_mask = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
        if np.any(m_mask):
            m_size[i] = _size(emg[m_mask])
        if np.any(h_mask):
            h_size[i] = _size(emg[h_mask])

    bg_arr = np.atleast_1d(np.asarray(bg_divisor, dtype=float))
    if bg_arr.size == 1:
        d = float(bg_arr.item())
        safe_bg = d if d > 0 else float('nan')
        bg_for_return = d
    else:
        if bg_arr.size != n:
            raise ValueError(
                f"bg_divisor array length {bg_arr.size} != n_trials {n}")
        safe_bg = np.where(bg_arr > 0, bg_arr, np.nan)
        bg_for_return = bg_arr

    return {
        'm_size': m_size, 'h_size': h_size,
        'm_norm': m_size / safe_bg, 'h_norm': h_size / safe_bg,
        'bg_divisor': bg_for_return,
        'm_start_ms': m_start_ms, 'm_end_ms': m_end_ms,
        'h_start_ms': h_start_ms, 'h_end_ms': h_end_ms,
        'pre_ms': pre_ms, 'post_ms': post_ms,
        'metric': metric,
    }


def print_response_stats(responses, header, label_prefix: str = ''):
    """Pretty-print descriptive statistics (n, mean, median, std, var, sem,
    min, max) for the raw and normalised M-/H-response sizes.

    Pass ``label_prefix='Peak-to-peak'`` or ``'Peak amplitude'`` to disambiguate
    when both metrics are computed in the same notebook.
    """
    metric = responses.get('metric', '?')
    pref = f"{label_prefix} " if label_prefix else ''
    bg_div = responses.get('bg_divisor', None)
    bg_str = (f"{bg_div:.3f} µV (scalar)"
              if isinstance(bg_div, float)
              else f"per-trial array (n={len(bg_div) if bg_div is not None else '?'})")

    print(f"=== {header.subject_id} — {pref}response statistics  "
          f"(metric={metric}, bg_divisor={bg_str}) ===")
    for desc, arr in [
        ('M-size (raw, µV)', responses['m_size']),
        ('H-size (raw, µV)', responses['h_size']),
        ('M-norm (size / bg)', responses['m_norm']),
        ('H-norm (size / bg)', responses['h_norm']),
    ]:
        a = np.asarray(arr, dtype=float)
        valid = a[~np.isnan(a)]
        if len(valid) == 0:
            print(f"  {desc}: no valid data")
            continue
        std = float(np.std(valid))
        sem = std / np.sqrt(len(valid)) if len(valid) > 0 else float('nan')
        print(f"  {desc}:")
        print(f"    n      = {len(valid)}")
        print(f"    mean   = {np.mean(valid):.4f}")
        print(f"    median = {np.median(valid):.4f}")
        print(f"    std    = {std:.4f}")
        print(f"    var    = {np.var(valid):.4f}")
        print(f"    sem    = {sem:.4f}")
        print(f"    min    = {np.min(valid):.4f}")
        print(f"    max    = {np.max(valid):.4f}")
    print()


def compute_response_variability_sweep(trial_bg_gm, responses,
                                        sweep_centres,
                                        half_widths_uv=None,
                                        metric: str = 'std'):
    """For each (centre, half-width) window over the trial-background axis,
    compute variability of the normalized M and H sizes across trials whose
    background falls in [centre - hw, centre + hw].

    metric:
        'std' -- standard deviation of m_norm / h_norm  (default)
        'cv'  -- coefficient of variation (std / mean)

    Returns dict keyed by centre label. Each value is a list of dicts with:
        hw, n, m_var, h_var, m_mean, h_mean, indices  (indices into trials)
    """
    if half_widths_uv is None:
        half_widths_uv = [5, 10, 20, 30, 50, 75, 100, 150]

    bg = np.asarray(trial_bg_gm, dtype=float)
    m_n = np.asarray(responses['m_norm'], dtype=float)
    h_n = np.asarray(responses['h_norm'], dtype=float)

    def _stat(vals):
        if len(vals) < 2:
            return float('nan'), (float(vals[0]) if len(vals) == 1 else float('nan'))
        std = float(np.std(vals))
        mean = float(np.mean(vals))
        if metric == 'cv':
            return (std / mean if mean != 0 else float('nan')), mean
        return std, mean

    results: dict = {}
    for label, centre in sweep_centres.items():
        rows = []
        for hw in half_widths_uv:
            in_win = (bg >= centre - hw) & (bg <= centre + hw)
            indices = np.where(in_win)[0]
            mv = m_n[in_win]; mv = mv[~np.isnan(mv)]
            hv = h_n[in_win]; hv = hv[~np.isnan(hv)]
            m_var, m_mean = _stat(mv)
            h_var, h_mean = _stat(hv)
            rows.append({
                'hw': hw, 'n': int(len(indices)),
                'm_var': m_var, 'h_var': h_var,
                'm_mean': m_mean, 'h_mean': h_mean,
                'indices': indices.tolist(),
            })
        results[label] = rows
    return results


def plot_response_variability_sweep(variability_results, header,
                                     n_markers: int = 10,
                                     metric_label: str = 'std'):
    """Plot Q1/Median/Q3 variability curves (M and H) vs threshold half-width,
    with ``n_markers`` equally-spaced vertical reference lines. Returns the
    list of marker half-widths so the viewer can slot them in.
    """
    import matplotlib.pyplot as plt
    if not variability_results:
        print("No variability results to plot.")
        return []

    first_label = next(iter(variability_results))
    hws_all = [r['hw'] for r in variability_results[first_label]]
    if n_markers >= 2 and len(hws_all) > 1:
        marker_hws = list(np.linspace(min(hws_all), max(hws_all), n_markers))
    else:
        marker_hws = list(hws_all[:n_markers])

    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette = ['darkorange', 'purple', 'steelblue', 'teal', 'crimson']

    def _col(label, i):
        return colours.get(label, palette[i % len(palette)])

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (label, rows) in enumerate(variability_results.items()):
        col = _col(label, i)
        hws_i = [r['hw'] for r in rows]
        m_vars = [r['m_var'] for r in rows]
        h_vars = [r['h_var'] for r in rows]
        ax.plot(hws_i, m_vars, 'o-', label=f'M-wave  ({label} centre)',
                color=col, linewidth=2.0)
        ax.plot(hws_i, h_vars, 's--', label=f'H-wave  ({label} centre)',
                color=col, linewidth=1.5, alpha=0.85)

    ymin, ymax = ax.get_ylim()
    for k, hw in enumerate(marker_hws):
        ax.axvline(hw, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.text(hw, ymax - (ymax - ymin) * 0.03, f'm{k}',
                color='gray', fontsize=8, ha='center', va='top')

    ax.set_xlabel('Threshold half-width  (EMG Window Size, µV)')
    ax.set_ylabel(f'Avg M-/H-Wave Size Variability  ({metric_label} of size/background)')
    ax.set_title(f'Response Variability vs EMG Window Size — {header.subject_id}')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return marker_hws


def compute_quartile_bins(trial_bg_gm, responses, state,
                          metric: str = 'std'):
    """For each centre (Q1 / Median / Q3) compute variability stats over ALL
    trials in that quartile range:
        Q1     -> bg <= state['gm_q1']                        (lower 25%)
        Median -> state['gm_q1'] <= bg <= state['gm_q3']      (interquartile, middle 50%)
        Q3     -> bg >= state['gm_q3']                        (upper 25%)

    Returns the same shape as ``compute_response_variability_sweep`` but with
    one row per centre. Each row carries ``lo`` / ``hi`` (true range bounds)
    and ``description`` so the print/viewer code can label it correctly.
    """
    bg = np.asarray(trial_bg_gm, dtype=float)
    m_n = np.asarray(responses['m_norm'], dtype=float)
    h_n = np.asarray(responses['h_norm'], dtype=float)

    q1_v = float(state['gm_q1'])
    q3_v = float(state['gm_q3'])
    bg_min = float(np.nanmin(bg))
    bg_max = float(np.nanmax(bg))

    quartile_ranges = {
        'Q1':     (bg_min, q1_v,   'Lower quartile (bg ≤ Q1)'),
        'Median': (q1_v,   q3_v,   'Interquartile range (Q1 ≤ bg ≤ Q3)'),
        'Q3':     (q3_v,   bg_max, 'Upper quartile (bg ≥ Q3)'),
    }

    def _stat(vals):
        if len(vals) < 2:
            return float('nan'), (float(vals[0]) if len(vals) == 1 else float('nan'))
        std = float(np.std(vals))
        mean = float(np.mean(vals))
        if metric == 'cv':
            return (std / mean if mean != 0 else float('nan')), mean
        return std, mean

    results: dict = {}
    for label, (lo, hi, desc) in quartile_ranges.items():
        in_win = (bg >= lo) & (bg <= hi)
        indices = np.where(in_win)[0]
        mv = m_n[in_win]; mv = mv[~np.isnan(mv)]
        hv = h_n[in_win]; hv = hv[~np.isnan(hv)]
        m_var, m_mean = _stat(mv)
        h_var, h_mean = _stat(hv)
        results[label] = [{
            'hw': (hi - lo) / 2.0,
            'lo': lo, 'hi': hi,
            'description': desc,
            'n': int(len(indices)),
            'm_var': m_var, 'h_var': h_var,
            'm_mean': m_mean, 'h_mean': h_mean,
            'indices': indices.tolist(),
        }]
    return results


def plot_response_variability_combined(ptp_results, peak_results, header,
                                         n_markers: int = 10):
    """Stacked 2-panel figure for the merged Section 6:
        top    -- peak-to-peak variability (Q1/Median/Q3 × M/H)
        bottom -- peak amplitude variability (Q1/Median/Q3 × M/H)

    Same x-axis (threshold half-width) on both panels with shared markers.
    Returns the list of marker half-widths.
    """
    import matplotlib.pyplot as plt
    if not ptp_results or not peak_results:
        print("Empty results.")
        return []

    first_label = next(iter(ptp_results))
    hws_all = [r['hw'] for r in ptp_results[first_label]]
    if n_markers >= 2 and len(hws_all) > 1:
        marker_hws = list(np.linspace(min(hws_all), max(hws_all), n_markers))
    else:
        marker_hws = list(hws_all[:n_markers])

    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette = ['darkorange', 'purple', 'steelblue', 'teal', 'crimson']

    def _col(label, i):
        return colours.get(label, palette[i % len(palette)])

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    for ax, results, title in [(axes[0], ptp_results, 'Peak-to-peak'),
                                (axes[1], peak_results, 'Peak amplitude')]:
        for i, (label, rows) in enumerate(results.items()):
            col = _col(label, i)
            hws_i = [r['hw'] for r in rows]
            m_vars = [r['m_var'] for r in rows]
            h_vars = [r['h_var'] for r in rows]
            ax.plot(hws_i, m_vars, 'o-', label=f'M ({label})',
                    color=col, linewidth=2.0)
            ax.plot(hws_i, h_vars, 's--', label=f'H ({label})',
                    color=col, linewidth=1.5, alpha=0.85)

        ymin, ymax = ax.get_ylim()
        for k, hw in enumerate(marker_hws):
            ax.axvline(hw, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
            ax.text(hw, ymax - (ymax - ymin) * 0.03, f'm{k}',
                    color='gray', fontsize=8, ha='center', va='top')

        ax.set_ylabel(f'{title} Variability\n(std of size / background)')
        ax.set_title(f'{title} — {header.subject_id}')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Threshold half-width  (EMG Window Size, µV)')
    plt.suptitle('M-/H-Wave Size Variability vs EMG Window Size', fontsize=12)
    plt.tight_layout()
    plt.show()
    return marker_hws


def print_variability_summary(variability_results, sweep_centres, header,
                              metric_name: str = 'ptp'):
    """Section-5b-style table of a variability sweep, one row per (centre, hw).

    Columns: Centre, ±hw (µV), Min (µV), Max (µV), n, M-var, H-var, M-mean, H-mean.
    Use after computing ``variability_results`` at marker half-widths to get a
    table with one row per marker per centre (e.g. 10 markers × 3 centres = 30 rows).
    """
    if not variability_results:
        return
    print(f"\n=== {header.subject_id} — {metric_name} variability summary ===")
    print(f"{'Centre':>10} {'±hw (µV)':>10} {'Min (µV)':>10} {'Max (µV)':>10} "
          f"{'n':>6} {'M-var':>10} {'H-var':>10} {'M-mean':>10} {'H-mean':>10}")
    print("-" * 100)
    for label, centre in sweep_centres.items():
        for r in variability_results.get(label, []):
            if 'lo' in r and 'hi' in r:
                min_uv = float(r['lo'])
                max_uv = float(r['hi'])
            else:
                min_uv = max(0.0, centre - r['hw'])
                max_uv = centre + r['hw']
            print(f"{label:>10} {r['hw']:>10.1f} {min_uv:>10.2f} {max_uv:>10.2f} "
                  f"{r['n']:>6} {r['m_var']:>10.3f} {r['h_var']:>10.3f} "
                  f"{r['m_mean']:>10.3f} {r['h_mean']:>10.3f}")
        print()


def view_variability_bin(variability_results, marker_hws, trials, header,
                          pre_ms: float = 2.0, post_ms: float = 15.0,
                          m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                          h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                          max_overlay: int = 200,
                          bg_mean: float = None,
                          quartile_results=None):
    """Interactive bin browser: pick a centre + marker → see the trials whose
    background falls in [centre ± marker_hw].

    Prints bin stats + trial indices and overlays the bipolar EMG of those
    trials (capped at ``max_overlay``) with average and M/H windows shaded.

    bg_mean: if given, draws ±``bg_mean`` horizontal reference lines on the
             bipolar overlay so each response can be compared to the global
             EMG grand mean.
    quartile_results: optional dict from ``compute_quartile_bins``. When
             provided, a 'Show all in quartile' toggle appears; turning it on
             switches the bin source from the marker grid to the entire
             quartile range for the selected centre.
    """
    import matplotlib.pyplot as plt
    from ipywidgets import Dropdown, IntSlider, ToggleButton, Output, VBox, HBox
    from IPython.display import display

    if not variability_results or not marker_hws:
        print("No variability data or markers.")
        return

    centre_labels = list(variability_results.keys())
    centre_dd = Dropdown(options=centre_labels, value=centre_labels[0],
                         description='Centre:', layout={'width': '200px'})
    marker_slider = IntSlider(
        value=len(marker_hws) // 2, min=0, max=len(marker_hws) - 1, step=1,
        description='Marker:', layout={'width': '450px'},
        continuous_update=False)
    has_quartile = quartile_results is not None
    quartile_toggle = ToggleButton(
        value=False, description='Show all in quartile',
        tooltip='Use the full quartile range instead of the marker bin',
        layout={'width': '220px'},
        disabled=not has_quartile,
    )
    out = Output()

    def _draw(centre, m_idx, q_mode):
        with out:
            out.clear_output(wait=True)
            if q_mode and has_quartile:
                row = quartile_results[centre][0]
            else:
                target_hw = marker_hws[m_idx]
                rows = variability_results[centre]
                hws = np.array([r['hw'] for r in rows], dtype=float)
                best = int(np.argmin(np.abs(hws - target_hw)))
                row = rows[best]
            indices = row['indices']

            if q_mode and has_quartile:
                lo = float(row.get('lo', 0.0))
                hi = float(row.get('hi', 0.0))
                desc = row.get('description', f'{centre} quartile')
                print(f"Centre = {centre}  |  ALL trials in quartile  |  "
                      f"range = [{lo:.2f}, {hi:.2f}] µV")
                print(f"  {desc}")
                title = (f'{header.subject_id} — Centre={centre}, '
                         f'quartile range [{lo:.1f}, {hi:.1f}] µV, '
                         f'n={row["n"]} trials')
            else:
                print(f"Centre = {centre}  |  marker {m_idx}  |  "
                      f"target hw = {marker_hws[m_idx]:.1f} µV  |  "
                      f"closest sweep hw = {row['hw']} µV")
                title = (f'{header.subject_id} — Centre={centre}, '
                         f'hw={row["hw"]} µV, n={row["n"]} trials')

            print(f"  Trials in window: {row['n']}")
            print(f"  M-wave  (size / bg):  mean = {row['m_mean']:.3f}   "
                  f"variability = {row['m_var']:.3f}")
            print(f"  H-wave  (size / bg):  mean = {row['h_mean']:.3f}   "
                  f"variability = {row['h_var']:.3f}")
            preview = indices[:50]
            print(f"  Trial indices ({len(indices)}): {preview}"
                  f"{' …' if len(indices) > len(preview) else ''}")

            if not indices:
                return

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.4)
            ax.axvspan(m_start_ms, m_end_ms, color='blue', alpha=0.15, label='M window')
            ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.15, label='H window')
            if bg_mean is not None and bg_mean > 0:
                ax.axhline(bg_mean, color='black', linestyle='--', linewidth=1.2,
                           label=f'±global mean ({bg_mean:.1f} µV)')
                ax.axhline(-bg_mean, color='black', linestyle='--', linewidth=1.2)

            shown = indices[:max_overlay]
            t_ref = None
            stack = []
            for idx in shown:
                t_ms, emg, _, _ = get_trial_window(trials[idx], pre_ms, post_ms)
                if t_ref is None:
                    t_ref = t_ms
                if len(emg) == len(t_ref):
                    ax.plot(t_ref, emg, color='red', alpha=0.15, linewidth=0.6)
                    stack.append(emg)

            if stack:
                arr = np.full((len(stack), len(t_ref)), np.nan)
                for k, emg in enumerate(stack):
                    n = min(len(emg), len(t_ref))
                    arr[k, :n] = emg[:n]
                avg = np.nanmean(arr, axis=0)
                ax.plot(t_ref, avg, color='black', linewidth=2.5, label='Average')

            extra = f' (showing first {max_overlay} of {len(indices)})' \
                    if len(indices) > max_overlay else ''
            ax.set_xlabel('Time re: stim onset (ms)')
            ax.set_ylabel('Bipolar EMG (µV)')
            ax.set_title(title + extra)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def _on_centre(c):
        if c['name'] == 'value':
            _draw(c['new'], marker_slider.value, quartile_toggle.value)

    def _on_marker(c):
        if c['name'] == 'value':
            _draw(centre_dd.value, c['new'], quartile_toggle.value)

    def _on_toggle(c):
        if c['name'] == 'value':
            marker_slider.disabled = bool(c['new'])
            _draw(centre_dd.value, marker_slider.value, c['new'])

    centre_dd.observe(_on_centre, names='value')
    marker_slider.observe(_on_marker, names='value')
    quartile_toggle.observe(_on_toggle, names='value')

    display(VBox([HBox([centre_dd, marker_slider, quartile_toggle]), out]))
    _draw(centre_dd.value, marker_slider.value, quartile_toggle.value)


def classify_trials(trials, file_version=0,
                    bin_samples=BIN_SAMPLES, record_samples=TRIAL_RECORD_SAMPLES,
                    onset_threshold=STIM_ONSET_THRESHOLD, end_threshold=STIM_END_THRESHOLD):
    """Scan all MhRecTrial objects and return a list of dicts describing each trial's
    ADC-sync quality.

    For file_version >= 2, pre-computed fields on each trial are used directly.
    For file_version < 2, onset detection is re-derived from sync_data.

    Keys in each returned dict:
        idx                              -- 0-based trial index
        amp_ma                           -- stimulation amplitude (mA)
        has_sync                         -- sync_data present (file_version >= 1)
        onset_found                      -- True if a real rising edge was detected
        onset_idx                        -- sample index used for onset
        adc_peak                         -- max ADC value in the onset search window
        adc_noise_std                    -- std of ADC pre-stim baseline
        stim_end_ms                      -- ms after onset where ADC dropped (or None)
        stim_duration_ms                 -- pulse duration in ms (0 if end not found)
        n_pre_trigger_frames_discarded   -- queued frames discarded (v2 only, else 0)
        first_post_trigger_frame_sample_id -- OE sample counter of first RECORD frame
        failed                           -- True when onset_found is False
    """
    search_start = int(bin_samples * 0.6)
    results = []

    for i, tr in enumerate(trials):
        has_sync = len(tr.sync_data) > 1
        rec = {'idx': i, 'amp_ma': tr.stimulation_amplitude_ma, 'has_sync': has_sync}

        if not has_sync:
            rec.update(onset_found=False, onset_idx=bin_samples,
                       adc_peak=float('nan'), adc_noise_std=float('nan'),
                       stim_end_ms=None, stim_duration_ms=0.0,
                       n_pre_trigger_frames_discarded=0,
                       first_post_trigger_frame_sample_id=0,
                       failed=True)
            results.append(rec)
            continue

        if file_version >= 2:
            onset_found = bool(tr.onset_detected)
            onset_idx   = tr.onset_sample_index if onset_found else bin_samples
            adc_peak    = float(tr.sync_peak_voltage)
            stim_end_ms = (float(tr.stim_duration_ms)
                           if tr.stim_end_sample_index >= 0 else None)
            stim_dur_ms = float(tr.stim_duration_ms)
            n_disc      = int(tr.n_pre_trigger_frames_discarded)
            first_sid   = int(tr.first_post_trigger_frame_sample_id)
        else:
            sd = np.asarray(tr.sync_data, dtype=float)
            search_end = min(bin_samples + record_samples, len(sd))
            window = sd[search_start:search_end]
            cands  = np.where(window >= onset_threshold)[0]
            if len(cands) > 0:
                onset_idx   = search_start + int(cands[0])
                onset_found = True
            else:
                onset_idx   = bin_samples
                onset_found = False
            adc_peak    = float(window.max()) if len(window) > 0 else float('nan')
            stim_end_ms = None
            stim_dur_ms = 0.0
            if onset_idx < len(sd):
                ends = np.where(sd[onset_idx:] < end_threshold)[0]
                if len(ends) > 0:
                    stim_end_ms = float(ends[0]) * MS_PER_SAMPLE
                    stim_dur_ms = stim_end_ms
            n_disc    = 0
            first_sid = 0

        sd_for_noise = np.asarray(tr.sync_data, dtype=float)
        pre_window = sd_for_noise[:search_start]
        adc_noise_std = float(pre_window.std()) if len(pre_window) > 0 else float('nan')

        rec.update(onset_found=onset_found, onset_idx=onset_idx,
                   adc_peak=adc_peak, adc_noise_std=adc_noise_std,
                   stim_end_ms=stim_end_ms, stim_duration_ms=stim_dur_ms,
                   n_pre_trigger_frames_discarded=n_disc,
                   first_post_trigger_frame_sample_id=first_sid,
                   failed=not onset_found)
        results.append(rec)
    return results


# ====================================================================
# DATA LOADING UTILITIES
# ====================================================================

def scan_experiment_structure(session_dir, record_node_name="Record Node 106"):
    """Scan an Open Ephys session directory and return experiments/recordings."""
    node_path = os.path.join(session_dir, record_node_name)
    if not os.path.exists(node_path):
        print(f"Record Node directory not found: {node_path}")
        return {}
    experiment_info = defaultdict(list)
    for item in sorted(os.listdir(node_path)):
        exp_path = os.path.join(node_path, item)
        if os.path.isdir(exp_path) and item.startswith("experiment"):
            recordings = [r for r in sorted(os.listdir(exp_path))
                          if os.path.isdir(os.path.join(exp_path, r)) and r.startswith("recording")]
            experiment_info[item] = recordings
    return experiment_info


def process_emg_signal(data, record_node_name, sample_rate=SAMPLE_RATE_HINT):
    """Process raw multi-channel data into filtered differential EMG.

    Returns (differential_filt, differential_emg) where differential_emg
    is the absolute-value rectified signal.
    """
    if record_node_name == "Record Node 106":
        differential_filt = data[:, 3]
        print("Using Online Filtering: assuming channel 3 is prefiltered differential EMG")
    else:
        emg1_raw = data[:, 2]
        emg2_raw = data[:, 3]
        differential = emg2_raw - emg1_raw
        lowcut, highcut = 100.0, 1000.0
        b, a = butter(2, np.array([lowcut, highcut]) / (sample_rate / 2), btype='bandpass')
        differential_filt = lfilter(b, a, differential)
        print(f"Applied offline bandpass ({lowcut}-{highcut} Hz) using lfilter at {sample_rate} Hz")

    differential_emg = np.abs(differential_filt)
    return differential_filt, differential_emg


def load_message_center_events(directory_str, record_node_name=None,
                               experiment_name=None, recording_name=None):
    """Load MessageCenter events (text + timestamps) from a recording directory.

    Tries two common directory layouts:
      1. <session_root>/<record_node>/<experiment>/<recording>/events/MessageCenter
      2. <directory_str>/events/MessageCenter

    Returns list of (timestamp, text) tuples.
    """
    candidates = []
    if record_node_name and experiment_name and recording_name:
        parts = directory_str.split(os.sep)
        session_root = os.sep.join(parts[:-3]) if len(parts) >= 4 else directory_str
        candidates.append(os.path.join(session_root, record_node_name, experiment_name,
                                       recording_name, "events", "MessageCenter"))
    candidates.append(os.path.join(directory_str, "events", "MessageCenter"))

    messagecenter_dir = None
    for c in candidates:
        if os.path.exists(c):
            messagecenter_dir = c
            break

    if messagecenter_dir is None:
        print(f"MessageCenter directory not found. Tried: {candidates}")
        return []

    texts = np.load(os.path.join(messagecenter_dir, "text.npy"), allow_pickle=True)
    timestamps_msg = np.load(os.path.join(messagecenter_dir, "timestamps.npy"))
    decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]
    message_entries = list(zip(timestamps_msg, decoded_texts))
    print(f"Loaded {len(decoded_texts)} MessageCenter entries from {messagecenter_dir}")
    return message_entries


def parse_trigger_times(message_entries):
    """Extract sorted trigger timestamps from MessageCenter entries."""
    trigger_pattern = re.compile(r"RHDCONTROL TRIGGER", re.IGNORECASE)
    trigger_times = sorted(
        float(t) for t, msg in message_entries if trigger_pattern.search(msg)
    )
    print(f"Found {len(trigger_times)} trigger messages")
    return trigger_times


def build_blank_mask(timestamps, trigger_times, sample_rate,
                     blank_pre_ms=10, blank_post_ms=50):
    """Build a boolean mask (True=keep, False=blank) around trigger events."""
    n = len(timestamps)
    blank_mask = np.ones(n, dtype=bool)
    pre_samples = int(np.ceil(blank_pre_ms * sample_rate / 1000.0))
    post_samples = int(np.ceil(blank_post_ms * sample_rate / 1000.0))
    trig_sample_idxs = np.searchsorted(timestamps, np.array(trigger_times))
    for idx in trig_sample_idxs:
        s = max(0, int(idx - pre_samples))
        e = min(n, int(idx + post_samples) + 1)
        blank_mask[s:e] = False
    blanked = int(np.sum(~blank_mask))
    print(f"Blank mask: {len(trig_sample_idxs)} events -> {blanked} samples blanked "
          f"({blanked / n:.4f} fraction)")
    return blank_mask


# ====================================================================
# ANALYSIS UTILITIES
# ====================================================================

def autoscale_ylim(*signals, percentile_lo=1, percentile_hi=99, pad_frac=0.05, symmetric=True):
    """Return (ymin, ymax) for a plot based on the percentile range of one or more signals.

    Pass multiple signals to compute a shared limit across all of them (useful for
    comparing traces on the same scale). Uses percentile clipping to avoid outliers
    dominating the scale.

    Parameters
    ----------
    *signals : array-like
        One or more signal arrays (e.g. differential_1mm_emg, differential_2mm_emg).
    percentile_lo : float
        Lower percentile for the min bound (default 1).
    percentile_hi : float
        Upper percentile for the max bound (default 99).
    pad_frac : float
        Fractional padding added to the computed span (default 0.05 = 5%).
    symmetric : bool
        If True (default), returns (-abs_max, abs_max) — appropriate for zero-centred
        bipolar signals like filtered differential EMG. If False, uses the raw
        percentile bounds directly.

    Returns
    -------
    (ymin, ymax) tuple suitable for plt.ylim() or ax.set_ylim().

    Example
    -------
    ylim = autoscale_ylim(sig1, sig2, sig3)
    plt.ylim(ylim)
    """
    all_data = np.concatenate([np.asarray(s).ravel() for s in signals])
    p_lo = np.percentile(all_data, percentile_lo)
    p_hi = np.percentile(all_data, percentile_hi)
    if symmetric:
        abs_max = max(abs(p_lo), abs(p_hi))
        if abs_max <= 0:
            abs_max = 1.0
        pad = pad_frac * (2 * abs_max)
        return (-abs_max - pad, abs_max + pad)
    else:
        span = p_hi - p_lo
        if span <= 0:
            span = max(1.0, abs(p_hi) * 0.1)
        return (p_lo - pad_frac * span*20, p_hi + pad_frac * span*20)


def round_to_nearest_multiple(x: int, base: int = BIN_DURATION_MS) -> int:
    return int(base * round(x / base))


def compute_bin_means(signal: np.ndarray, blank_mask: np.ndarray, bin_sample_count: int):
    """Compute non-overlapping bin means and per-bin valid fraction."""
    n = len(signal)
    n_full = (n // bin_sample_count) * bin_sample_count
    if n_full == 0:
        return np.array([]), np.array([])
    sig = signal[:n_full]
    mask = blank_mask[:n_full]
    reshaped_sig = sig.reshape(-1, bin_sample_count)
    reshaped_mask = mask.reshape(-1, bin_sample_count)
    valid_counts = reshaped_mask.sum(axis=1)
    means = np.full(reshaped_sig.shape[0], np.nan, dtype=np.float64)
    for i in range(reshaped_sig.shape[0]):
        if valid_counts[i] > 0:
            means[i] = reshaped_sig[i, reshaped_mask[i]].mean()
    valid_frac = valid_counts / float(bin_sample_count)
    return means, valid_frac


def grand_mean_from_bins(bin_means: np.ndarray, valid_frac: np.ndarray,
                         min_valid_fraction: float):
    mask = (valid_frac >= min_valid_fraction) & (~np.isnan(bin_means))
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(bin_means[mask]))


def sweep_trials(signal, timestamps, blank_mask, sample_rate,
                 min_ms=TRIAL_INIT_MIN_MS, max_ms=TRIAL_INIT_MAX_MS,
                 min_uv=TRIAL_INIT_MIN_UV, max_uv=TRIAL_INIT_MAX_UV,
                 bin_duration_ms=BIN_DURATION_MS, min_valid_fraction=0.7,
                 seed=0):
    """Walk through the recording collecting accepted trials."""
    rng = Random(seed)
    trials = []
    n = len(signal)
    i = 0
    bin_sample_count = int(np.round(bin_duration_ms * sample_rate / 1000.0))
    if bin_sample_count <= 0:
        raise ValueError('bin_sample_count computed as 0')

    while i < n:
        dur_ms = rng.randint(min_ms, max_ms)
        dur_ms = round_to_nearest_multiple(dur_ms, base=bin_duration_ms)
        dur_samples = int(np.round(dur_ms * sample_rate / 1000.0))
        if i + dur_samples > n:
            break
        seg = signal[i:i + dur_samples]
        seg_mask = blank_mask[i:i + dur_samples]
        bin_means, valid_frac = compute_bin_means(seg, seg_mask, bin_sample_count)
        gm = grand_mean_from_bins(bin_means, valid_frac, min_valid_fraction)
        if (not np.isnan(gm)) and (min_uv <= gm <= max_uv):
            trials.append({
                'start_sample': i,
                'end_sample': i + dur_samples,
                'dur_ms': dur_ms,
                'bin_means': bin_means,
                'valid_frac': valid_frac,
                'grand_mean': gm,
            })
            i += dur_samples
        else:
            i += bin_sample_count
    return trials


# ====================================================================
# SIMULATION CLASSES
# ====================================================================

@dataclass
class SimulatedTrial:
    """Represents a single simulated trial with all relevant metadata."""
    trial_number: int
    start_sample_idx: int
    start_time: float  # seconds
    monitoring_duration_ms: int
    grand_mean_uv: float
    min_threshold: float
    max_threshold: float
    bin_means: np.ndarray
    pre_stim_data: np.ndarray
    post_stim_data: np.ndarray
    time_since_last_trial_ms: Optional[float] = None
    num_shifts: int = 0
    total_search_duration_ms: float = 0
    needed_shifting: bool = False


class TrialInitiationData:
    """Sliding-window EMG monitor that bins data and checks initiation criteria."""

    def __init__(self, sample_rate: float, bin_duration_ms: float = BIN_DURATION_MS):
        self.sample_rate = sample_rate
        self.bin_sample_count = int((sample_rate / 1000.0) * bin_duration_ms)
        self.monitored_signal_abs: np.ndarray = np.array([])
        self.bins: np.ndarray = np.array([])
        self.monitored_signal_duration_seconds: float = 0.0
        self.monitored_signal_sample_count: int = 0
        self.current_monitored_signal_sample_count: int = 0

    def initialize(self, duration_ms: int) -> None:
        self.monitored_signal_duration_seconds = float(duration_ms) / 1000.0
        self.monitored_signal_sample_count = int(
            self.monitored_signal_duration_seconds * self.sample_rate
        )
        bin_count = int(duration_ms / BIN_DURATION_MS)
        self.monitored_signal_abs = np.zeros(self.monitored_signal_sample_count)
        self.bins = np.zeros(bin_count)
        self.current_monitored_signal_sample_count = 0

    def process(self, data_block_abs, current_min_threshold, current_max_threshold):
        should_initiate = False
        self.current_monitored_signal_sample_count += len(data_block_abs)
        self.monitored_signal_abs = np.concatenate([self.monitored_signal_abs, data_block_abs])
        elements_to_remove = len(self.monitored_signal_abs) - self.monitored_signal_sample_count
        if elements_to_remove > 0:
            self.monitored_signal_abs = self.monitored_signal_abs[elements_to_remove:]
        for bin_idx in range(len(self.bins)):
            bin_start = self.bin_sample_count * bin_idx
            bin_end = self.bin_sample_count * (bin_idx + 1)
            if bin_end > len(self.monitored_signal_abs):
                bin_end = len(self.monitored_signal_abs)
            if bin_end > bin_start:
                self.bins[bin_idx] = np.mean(self.monitored_signal_abs[bin_start:bin_end])
        bin_grand_mean = np.mean(self.bins)
        if (self.current_monitored_signal_sample_count >= self.monitored_signal_sample_count
                and bin_grand_mean >= current_min_threshold
                and bin_grand_mean <= current_max_threshold):
            should_initiate = True
        return should_initiate, bin_grand_mean

    def get_current_buffer(self):
        return self.monitored_signal_abs.copy()

    def get_bin_means(self):
        return self.bins.copy()


# ====================================================================
# SIMULATION ENGINE (with 50 ms bin sliding)
# ====================================================================

STATE_NOT_SETUP = 0
STATE_WAIT_FOR_INITIATION = 1
STATE_NAMES = {STATE_NOT_SETUP: "NOT_SETUP", STATE_WAIT_FOR_INITIATION: "WAIT_FOR_INITIATION"}


def run_trial_initiation_simulation(
    emg_signal: np.ndarray,
    timestamps: np.ndarray,
    sample_rate: float,
    min_init_threshold: float = TRIAL_INIT_MIN_UV,
    max_init_threshold: float = TRIAL_INIT_MAX_UV,
    block_size: int = SIMULATION_BLOCK_SIZE,
    min_inter_trial_ms: int = MINIMUM_INTERTRIAL_INTERVAL_MS,
    verbose: bool = True,
    max_trials: int = None,
    random_seed: int = SIMULATION_RANDOM_SEED,
) -> tuple:
    """Run the trial initiation simulation with 50 ms bin sliding.

    When the monitoring window is full and the grand mean is outside thresholds,
    instead of discarding the attempt the window slides forward by one 50 ms bin
    at a time until the grand mean falls within thresholds.

    Returns (all_trials, statistics) where all_trials is List[SimulatedTrial].
    """
    rng = Random(random_seed)

    current_state = STATE_NOT_SETUP
    trial_initiation_data = None
    current_trial_number = 0

    all_trials: List[SimulatedTrial] = []
    successful_trials: List[SimulatedTrial] = []
    shifted_trials: List[SimulatedTrial] = []

    current_sample_idx = 0
    total_samples = len(emg_signal)
    last_trial_time_ms = -min_inter_trial_ms
    monitoring_duration_ms = 0
    monitoring_start_sample_idx = 0

    total_monitoring_attempts = 0
    inter_trial_blocks_count = 0
    total_shifts = 0

    start_time_wall = time.time()
    last_progress_pct = 0
    current_grand_mean = 0.0

    bin_sample_count = int((sample_rate / 1000.0) * BIN_DURATION_MS)

    if verbose:
        print("=" * 80)
        print("STARTING TRIAL INITIATION SIMULATION (with bin sliding)")
        print("=" * 80)
        print(f"Total samples: {total_samples:,} | Duration: {timestamps[-1] - timestamps[0]:.2f}s")
        print(f"Thresholds: {min_init_threshold:.1f} - {max_init_threshold:.1f} uV")
        print(f"Min ITI: {min_inter_trial_ms} ms | Block: {block_size} samples | "
              f"Bin slide: {BIN_DURATION_MS} ms ({bin_sample_count} samples)")
        print("=" * 80)

    while current_sample_idx < total_samples:
        block_end_idx = min(current_sample_idx + block_size, total_samples)
        data_block = emg_signal[current_sample_idx:block_end_idx]
        block_timestamps = timestamps[current_sample_idx:block_end_idx]
        current_time_ms = block_timestamps[0] * 1000

        if verbose:
            progress_pct = int((current_sample_idx / total_samples) * 100)
            if progress_pct >= last_progress_pct + 10:
                elapsed = time.time() - start_time_wall
                print(f"[{progress_pct}%] Attempts:{total_monitoring_attempts} | "
                      f"Trials:{current_trial_number} "
                      f"(immediate:{len(successful_trials) - len(shifted_trials)} "
                      f"shifted:{len(shifted_trials)}) | "
                      f"GM:{current_grand_mean:.2f} uV | {elapsed:.1f}s")
                last_progress_pct = progress_pct

        # --- STATE MACHINE ---
        if current_state == STATE_NOT_SETUP:
            total_monitoring_attempts += 1
            monitoring_duration_ms = rng.randint(TRIAL_INIT_MIN_MS, TRIAL_INIT_MAX_MS)
            monitoring_duration_ms = round_to_nearest_multiple(monitoring_duration_ms, BIN_DURATION_MS)
            trial_initiation_data = TrialInitiationData(sample_rate)
            trial_initiation_data.initialize(monitoring_duration_ms)
            monitoring_start_sample_idx = current_sample_idx
            current_state = STATE_WAIT_FOR_INITIATION

            if verbose and total_monitoring_attempts <= 10:
                print(f"\n[NOT_SETUP -> WAIT] Attempt #{total_monitoring_attempts} | "
                      f"Window: {monitoring_duration_ms} ms | Time: {block_timestamps[0]:.2f}s")

        elif current_state == STATE_WAIT_FOR_INITIATION:
            time_since_last_trial_ms = current_time_ms - last_trial_time_ms
            if time_since_last_trial_ms < min_inter_trial_ms:
                inter_trial_blocks_count += 1
                current_sample_idx = block_end_idx
                continue

            _, grand_mean = trial_initiation_data.process(
                data_block, min_init_threshold, max_init_threshold)
            current_grand_mean = grand_mean

            if (trial_initiation_data.current_monitored_signal_sample_count
                    >= trial_initiation_data.monitored_signal_sample_count):

                is_in_threshold = (grand_mean >= min_init_threshold
                                   and grand_mean <= max_init_threshold)

                if is_in_threshold:
                    # --- IMMEDIATE SUCCESS ---
                    current_trial_number += 1
                    last_trial_time_ms = current_time_ms
                    trial = SimulatedTrial(
                        trial_number=current_trial_number,
                        start_sample_idx=monitoring_start_sample_idx,
                        start_time=block_timestamps[0],
                        monitoring_duration_ms=monitoring_duration_ms,
                        grand_mean_uv=grand_mean,
                        min_threshold=min_init_threshold,
                        max_threshold=max_init_threshold,
                        bin_means=trial_initiation_data.get_bin_means(),
                        pre_stim_data=np.array([]),
                        post_stim_data=np.array([]),
                        time_since_last_trial_ms=time_since_last_trial_ms,
                        num_shifts=0,
                        total_search_duration_ms=float(monitoring_duration_ms),
                        needed_shifting=False,
                    )
                    all_trials.append(trial)
                    successful_trials.append(trial)

                    if verbose and current_trial_number <= 15:
                        print(f"  [IMMEDIATE] Trial #{current_trial_number} | "
                              f"GM:{grand_mean:.2f} uV | Window:{monitoring_duration_ms} ms | "
                              f"Time:{block_timestamps[0]:.2f}s | ISI:{time_since_last_trial_ms:.0f} ms")

                    current_state = STATE_NOT_SETUP
                    trial_initiation_data = None
                    if max_trials is not None and current_trial_number >= max_trials:
                        if verbose:
                            print(f"\nMax trials ({max_trials}) reached.")
                        break
                else:
                    # --- BIN SLIDING ---
                    num_shifts = 0
                    initial_gm = grand_mean
                    found = False

                    while current_sample_idx < total_samples:
                        shift_start = block_end_idx if num_shifts == 0 else current_sample_idx
                        shift_end = min(shift_start + bin_sample_count, total_samples)
                        if shift_end <= shift_start:
                            break
                        shift_data = emg_signal[shift_start:shift_end]
                        current_sample_idx = shift_end
                        num_shifts += 1
                        _, grand_mean = trial_initiation_data.process(
                            shift_data, min_init_threshold, max_init_threshold)
                        current_grand_mean = grand_mean
                        if grand_mean >= min_init_threshold and grand_mean <= max_init_threshold:
                            found = True
                            break

                    current_trial_number += 1
                    total_shifts += num_shifts
                    final_idx = min(current_sample_idx - 1, total_samples - 1)
                    final_time = timestamps[final_idx]
                    last_trial_time_ms = final_time * 1000
                    search_dur = float(monitoring_duration_ms) + num_shifts * BIN_DURATION_MS

                    trial = SimulatedTrial(
                        trial_number=current_trial_number,
                        start_sample_idx=monitoring_start_sample_idx,
                        start_time=final_time,
                        monitoring_duration_ms=monitoring_duration_ms,
                        grand_mean_uv=grand_mean,
                        min_threshold=min_init_threshold,
                        max_threshold=max_init_threshold,
                        bin_means=trial_initiation_data.get_bin_means(),
                        pre_stim_data=np.array([]),
                        post_stim_data=np.array([]),
                        time_since_last_trial_ms=time_since_last_trial_ms,
                        num_shifts=num_shifts,
                        total_search_duration_ms=search_dur,
                        needed_shifting=True,
                    )
                    all_trials.append(trial)
                    shifted_trials.append(trial)
                    if found:
                        successful_trials.append(trial)

                    if verbose and current_trial_number <= 15:
                        tag = "SHIFTED" if found else "END-OF-REC"
                        print(f"  [{tag}] Trial #{current_trial_number} | "
                              f"GM:{initial_gm:.2f}->{grand_mean:.2f} uV | "
                              f"{num_shifts} shifts ({num_shifts * BIN_DURATION_MS} ms) | "
                              f"Search:{search_dur:.0f} ms | Time:{final_time:.2f}s")

                    current_state = STATE_NOT_SETUP
                    trial_initiation_data = None
                    if max_trials is not None and current_trial_number >= max_trials:
                        if verbose:
                            print(f"\nMax trials ({max_trials}) reached.")
                        break
                    continue  # skip normal block advance

        current_sample_idx = block_end_idx

    # --- STATISTICS ---
    elapsed = time.time() - start_time_wall
    iti_list = []
    if len(all_trials) > 1:
        for i in range(1, len(all_trials)):
            iti_list.append((all_trials[i].start_time - all_trials[i - 1].start_time) * 1000)

    immediate_count = len(all_trials) - len(shifted_trials)
    shifted_durs = [t.total_search_duration_ms for t in shifted_trials]

    statistics = {
        'total_samples_processed': current_sample_idx,
        'total_monitoring_attempts': total_monitoring_attempts,
        'total_trials': len(all_trials),
        'successful_trials': len(successful_trials),
        'immediate_success_trials': immediate_count,
        'shifted_trials': len(shifted_trials),
        'total_shifts': total_shifts,
        'mean_shifts_per_shifted_trial': (
            float(np.mean([t.num_shifts for t in shifted_trials])) if shifted_trials else 0),
        'success_rate_pct': (
            len(successful_trials) / len(all_trials) * 100 if all_trials else 0),
        'inter_trial_blocks_count': inter_trial_blocks_count,
        'all_inter_trial_intervals_ms': iti_list,
        'inter_trial_intervals_ms': iti_list,
        'mean_inter_trial_interval_ms': float(np.mean(iti_list)) if iti_list else 0,
        'elapsed_time_seconds': elapsed,
        'processing_speed_samples_per_sec': (
            current_sample_idx / elapsed if elapsed > 0 else 0),
        'all_monitoring_windows_ms': [t.monitoring_duration_ms for t in all_trials],
        'mean_monitoring_window_ms': (
            float(np.mean([t.monitoring_duration_ms for t in all_trials])) if all_trials else 0),
        'shifted_search_durations_ms': shifted_durs,
        'mean_shifted_search_duration_ms': float(np.mean(shifted_durs)) if shifted_durs else 0,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print(f"Processed {current_sample_idx:,} samples in {elapsed:.2f}s")
        print(f"Trials: {len(all_trials)} total "
              f"(immediate:{immediate_count}, shifted:{len(shifted_trials)}, "
              f"successful:{len(successful_trials)})")
        print(f"Success rate: {statistics['success_rate_pct']:.1f}%")
        if shifted_trials:
            print(f"Shifts: {total_shifts} total | "
                  f"Mean/shifted: {statistics['mean_shifts_per_shifted_trial']:.1f} | "
                  f"Mean search: {statistics['mean_shifted_search_duration_ms']:.1f} ms")
        if iti_list:
            print(f"Mean ITI: {statistics['mean_inter_trial_interval_ms']:.1f} ms")
        print("=" * 80)

    return all_trials, statistics


# ====================================================================
# RESPIRATION_OFFLINE NOTEBOOK HELPERS
# ====================================================================
# Generic loaders + an interactive segmented-viewer widget. Used by
# Respiration_offline.ipynb. Complements the existing DATA LOADING
# UTILITIES section above by accepting arbitrary channel indices and
# arbitrary signal stacks.

import matplotlib.pyplot as plt
from ipywidgets import Button, Output, VBox
from open_ephys.analysis import Session


def load_session_recording(session_dir, recordnode_idx=0, recording_idx=0, verbose=True):
    """Load a Session and pick a (record_node, recording).

    Returns (session, recording, record_node_name, experiment_name, recording_name).
    record_node_name / experiment_name / recording_name are derived from the
    actual recording.directory path so they reflect what was loaded.
    """
    session   = Session(session_dir)
    recording = session.recordnodes[recordnode_idx].recordings[recording_idx]
    parts     = recording.directory.split(os.sep)
    record_node_name, experiment_name, recording_name = parts[-3], parts[-2], parts[-1]
    if verbose:
        print(f"Session: {session_dir}")
        print(f"Loaded:  {recording.directory}")
        print(f"  record node: {record_node_name}")
        print(f"  experiment:  {experiment_name}")
        print(f"  recording:   {recording_name}")
    return session, recording, record_node_name, experiment_name, recording_name


def load_continuous(recording, stream_idx=0, verbose=True):
    """Pull continuous-stream samples + metadata from an Open Ephys recording.

    Returns (timestamps, data, sample_rate, channel_names).
    """
    stream     = recording.continuous[stream_idx]
    metadata   = stream.metadata
    timestamps = stream.timestamps
    data = stream.get_samples(start_sample_index=0, end_sample_index=timestamps.shape[0])
    if verbose:
        print(f"Sample rate: {metadata.sample_rate} Hz")
        print(f"Channels ({metadata.num_channels}): {metadata.channel_names}")
        print(f"Data shape: {data.shape}")
    return timestamps, data, metadata.sample_rate, metadata.channel_names


def bandpass_filter_emg(data, ch1_idx, ch2_idx, sample_rate,
                        lowcut=100.0, highcut=1000.0, order=2):
    """Bandpass-filter two EMG channels (and their differential) with `lfilter`.

    Returns (emg1_filtered, emg2_filtered, differential_filtered).
    """
    emg1_raw     = data[:, ch1_idx]
    emg2_raw     = data[:, ch2_idx]
    differential = emg2_raw - emg1_raw
    nyq = sample_rate / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass')
    return (lfilter(b, a, emg1_raw),
            lfilter(b, a, emg2_raw),
            lfilter(b, a, differential))


def plot_full_trace(timestamps, signal, title, ylabel="Amplitude (μV)",
                    label=None, color="purple", figsize=(15, 4), ylim=None):
    """Single full-trace line plot with a zero baseline."""
    plt.figure(figsize=figsize)
    plt.plot(timestamps, signal, label=(label or title), color=color)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.show()


def plot_emg_full_traces(timestamps, differential_filt, emg1, emg2, directory):
    """Three stacked full-trace plots: filtered differential, EMG1, EMG2."""
    plot_full_trace(timestamps, differential_filt,
                    title=f"{directory} Filtered Differential EMG Signal (EMG1 - EMG2)",
                    label="Filtered EMG1 - EMG2")
    plot_full_trace(timestamps, emg1, title="EMG1 Raw", label="EMG1")
    plot_full_trace(timestamps, emg2, title="EMG2 Raw", label="EMG2")


def make_segment_viewer(timestamps, signals, labels,
                        segment_duration_s=10, title_prefix="",
                        figsize=(15, 4), color="purple"):
    """Build a Prev/Next ipywidgets viewer that shows one fixed-duration window
    at a time, with one stacked plot per (signal, label) pair.

    Returns a VBox; caller wraps in `display(...)`.
    """
    if len(signals) != len(labels):
        raise ValueError("signals and labels must be the same length")
    total_time   = timestamps[-1] - timestamps[0]
    num_segments = int(np.ceil(total_time / segment_duration_s))
    state = {"idx": 0}
    out   = Output()

    def _draw(idx):
        start_t = timestamps[0] + idx * segment_duration_s
        end_t   = min(start_t + segment_duration_s, timestamps[-1])
        mask    = (timestamps >= start_t) & (timestamps < end_t)
        local_t = timestamps[mask] - start_t
        prefix  = f"{title_prefix}, " if title_prefix else ""
        for sig, lbl in zip(signals, labels):
            plt.figure(figsize=figsize)
            plt.plot(local_t, sig[mask], label=lbl, color=color)
            plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
            plt.title(f"{prefix}{lbl}, Segment {idx + 1}/{num_segments}, "
                      f"Time: {start_t:.1f}-{end_t:.1f}s")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (μV)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def _step(delta):
        new_idx = max(0, min(num_segments - 1, state["idx"] + delta))
        if new_idx == state["idx"]:
            return
        state["idx"] = new_idx
        with out:
            out.clear_output(wait=True)
            _draw(state["idx"])

    next_btn = Button(description="Next")
    prev_btn = Button(description="Previous")
    next_btn.on_click(lambda _b: _step(+1))
    prev_btn.on_click(lambda _b: _step(-1))

    with out:
        _draw(state["idx"])
    return VBox([prev_btn, next_btn, out])
