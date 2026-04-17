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
