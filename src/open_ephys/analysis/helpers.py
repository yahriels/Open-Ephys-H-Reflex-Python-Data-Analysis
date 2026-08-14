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
BLOCK_MH_TRIAL             = 3   # V1 .hrs2 / V2 .hrs1  MH Recruitment Curve trials
BLOCK_EMG_TRIALS_PER_HOUR  = 4
BLOCK_DCP_TRIAL            = 5   # V2 .hrs3  Down Condition Pellet trials
BLOCK_CONTROL_MODE_TRIAL   = 6   # V2 .hrs2  Control Mode trials
BLOCK_FREQUENCY_TEST_TRIAL = 8   # V3 .hrsft Frequency Test trials
BLOCK_UP_COND_PELLET_TRIAL = 9   # V3 .hrs4  Up Condition Pellet trials
BLOCK_DOWN_COND_VNS_TRIAL  = 10  # V3 .hrs5  Down Condition VNS trials
BLOCK_UP_COND_VNS_TRIAL    = 11  # V3 .hrs6  Up Condition VNS trials

# Low-level type maps (mirrors FileIO_Helpers from hreflex_txbdc)
_HRS_TYPE_FMT  = {'int8': 'b', 'int16': 'h', 'int32': 'i', 'int64': 'q', 'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'uint64': 'Q', 'float32': 'f', 'float64': 'd'}
_HRS_TYPE_SIZE = {'int8': 1,  'int16': 2,   'int32': 4,   'int64': 8,   'uint8': 1,   'uint16': 2,   'uint32': 4,   'uint64': 8,   'float32': 4,   'float64': 8}


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
    raw = fid.read(8)
    if len(raw) < 8:
        raise EOFError("Unexpected end of file reading datetime")
    as_f64 = struct.unpack('<d', raw)[0]
    if as_f64 > 1.0:
        # Old format: MATLAB datenum stored as float64
        days = as_f64 % 1
        return datetime.fromordinal(int(as_f64)) + timedelta(days=days) - timedelta(days=366)
    else:
        # New format: Unix milliseconds stored as uint64
        unix_ms = struct.unpack('<Q', raw)[0]
        return datetime.fromtimestamp(unix_ms / 1000.0)


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
    # --- file_version >= 1 fields ---
    actual_init_min_threshold: float = 0.0
    actual_init_max_threshold: float = 0.0


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
    app_version: str = ""   # file_version >= 8 (.hrs1/.hrs2) or >= 9 (.hrs3)
    # Derived after reading trials: len(trial_data) / (TRIAL_RECORD_MS / 1000).
    # Defaults to 5000.0 for files with no trials.
    sample_rate: float = 5000.0
    bin_duration_ms: int = BIN_DURATION_MS


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
    onset_detected: int = 0            # 0 = fallback, 1 = ADC crossing, 2 = digital DIGITAL IN (v7+)
    stim_end_sample_index: int = -1    # -1 = not found within recording window
    stim_duration_samples: int = 0
    stim_duration_ms: float = 0.0
    sync_peak_voltage: float = 0.0     # max ADC in search window
    n_pre_trigger_frames_discarded: int = 0
    frame_received_timestamps_ms: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint64))
    first_post_trigger_frame_sample_id: int = 0
    # --- file_version >= 3 fields ---
    unipolar_trial_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    # --- file_version >= 4 fields ---
    stim_adc_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    # --- file_version >= 5 fields ---
    background_emg_mean: float = 0.0
    background_bins: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # --- file_version >= 6 fields ---
    stim_polarity_reversed: int = 0   # 0 = normal, 1 = reversed
    # --- file_version >= 7 fields ---
    digital_onset_sample_num: int = -1   # absolute OE sample of DIGITAL IN rising edge; -1 = none
    digital_onset_channel: int = -1      # OE digital channel index (0-based); -1 = none
    # --- file_version >= 9 fields (S2 Control Mode; inherited by V3 stages) ---
    h_wave_response:          float = float('nan')
    m_wave_response:          float = float('nan')
    hm_ratio:                 float = float('nan')
    m_wave_window_median:     float = float('nan')
    m_wave_set_value_uv:      float = float('nan')
    m_wave_error:             float = float('nan')
    m_wave_adjust_step_ma:    float = float('nan')
    m_wave_min_intensity_ma:  float = float('nan')
    m_wave_max_intensity_ma:  float = float('nan')


@dataclass
class DcpTrial(MhRecTrial):
    """Down Condition Pellet trial (hreflex_txbdc V2/V3 .hrs3).

    h_wave_response, m_wave_response, hm_ratio inherited from MhRecTrial.
    m_wave_window_median … m_wave_max_intensity_ma inherited from MhRecTrial (V10 gate).
    """
    # --- file_version >= 8 fields (.hrs3 only) ---
    success_threshold: float = float('nan')  # H/M ratio threshold for pellet delivery
    is_success:        int   = 0             # 1 if trial met the success threshold
    pellet_delivered:  int   = 0             # 1 if a food pellet was actually dispensed


@dataclass
class FrequencyTestHeader(MhRecHeader):
    """Header for Frequency Test stage (.hrsft, V3)."""
    n_pulses_per_train: int = 0
    event_period_us:    int = 0
    pulse_width_us:     int = 0


@dataclass
class FrequencyTestTrial(MhRecTrial):
    """Frequency Test trial (V3 .hrsft, block_id=8)."""
    pulse_h_wave_mra: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    pulse_m_wave_mra: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


@dataclass
class UpCondPelletTrial(MhRecTrial):
    """Up Condition Pellet trial (V3 .hrs4, block_id=9)."""
    pellet_delivered: int = 0


@dataclass
class DownCondVnsTrial(MhRecTrial):
    """Down Condition VNS trial (V3 .hrs5, block_id=10)."""
    vns_delivered: int = 0


@dataclass
class UpCondVnsTrial(MhRecTrial):
    """Up Condition VNS trial (V3 .hrs6, block_id=11)."""
    vns_delivered: int = 0


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


def _read_mh_trial_block(fid: BinaryIO, file_version: int = 0,
                          block_id: int = BLOCK_MH_TRIAL) -> MhRecTrial:
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
    if file_version >= 4:
        t.stim_adc_data = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    if file_version >= 5:
        t.background_emg_mean = hrs_read_val(fid, 'float32')
        if file_version >= 6:
            # v6 always writes background_bins as a full array.
            t.background_bins = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
        else:
            # Some v5 files write background_bins; others don't.
            # A valid block_id is ≤ 4, so if the next int32 > 4 it must be an array count.
            peek_raw = fid.read(4)
            if len(peek_raw) == 4:
                maybe_count = struct.unpack('<i', peek_raw)[0]
                if maybe_count > 4:
                    n = maybe_count
                    raw = fid.read(n * 4)
                    t.background_bins = np.frombuffer(raw, dtype='<f4').astype(np.float32)
                else:
                    fid.seek(-4, 1)
    if file_version >= 6:
        t.stim_polarity_reversed = hrs_read_val(fid, 'int8')
    if file_version >= 7:
        t.digital_onset_sample_num = hrs_read_val(fid, 'int64')
        t.digital_onset_channel    = hrs_read_val(fid, 'int32')
    if block_id == BLOCK_CONTROL_MODE_TRIAL and file_version >= 9:
        t.h_wave_response         = hrs_read_val(fid, 'float32')
        t.m_wave_response         = hrs_read_val(fid, 'float32')
        t.hm_ratio                = hrs_read_val(fid, 'float32')
        t.m_wave_window_median    = hrs_read_val(fid, 'float32')
        t.m_wave_set_value_uv     = hrs_read_val(fid, 'float32')
        t.m_wave_error            = hrs_read_val(fid, 'float32')
        t.m_wave_adjust_step_ma   = hrs_read_val(fid, 'float32')
        t.m_wave_min_intensity_ma = hrs_read_val(fid, 'float32')
        t.m_wave_max_intensity_ma = hrs_read_val(fid, 'float32')
    return t


def _read_mh_trial_block_full(fid: BinaryIO) -> MhRecTrial:
    """Read all MhRecTrial base fields unconditionally (V3 new stages)."""
    t = MhRecTrial()
    t.start_time                         = hrs_read_datetime(fid)
    t.min_initiation_threshold           = hrs_read_val(fid, 'float32')
    t.max_initiation_threshold           = hrs_read_val(fid, 'float32')
    t.stimulation_amplitude_ma           = hrs_read_val(fid, 'float32')
    t.trial_data                         = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    t.sync_data                          = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    t.trigger_wall_time_ms               = hrs_read_val(fid, 'uint64')
    t.onset_sample_index                 = hrs_read_val(fid, 'int32')
    t.onset_detected                     = hrs_read_val(fid, 'int8')
    t.stim_end_sample_index              = hrs_read_val(fid, 'int32')
    t.stim_duration_samples              = hrs_read_val(fid, 'int32')
    t.stim_duration_ms                   = hrs_read_val(fid, 'float32')
    t.sync_peak_voltage                  = hrs_read_val(fid, 'float32')
    t.n_pre_trigger_frames_discarded     = hrs_read_val(fid, 'int32')
    t.frame_received_timestamps_ms       = np.array(hrs_read_array(fid, 'uint64'), dtype=np.uint64)
    t.first_post_trigger_frame_sample_id = hrs_read_val(fid, 'uint64')
    t.unipolar_trial_data                = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    t.stim_adc_data                      = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    t.background_emg_mean                = hrs_read_val(fid, 'float32')
    t.background_bins                    = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    t.stim_polarity_reversed             = hrs_read_val(fid, 'int8')
    t.digital_onset_sample_num           = hrs_read_val(fid, 'int64')
    t.digital_onset_channel              = hrs_read_val(fid, 'int32')
    return t


def _read_up_cond_pellet_trial_block(fid: BinaryIO, file_version: int) -> UpCondPelletTrial:
    """Read one Up Condition Pellet trial block (V3 .hrs4, block_id=9)."""
    base = _read_mh_trial_block_full(fid)
    t = UpCondPelletTrial.__new__(UpCondPelletTrial)
    t.__dict__.update(base.__dict__)
    t.pellet_delivered = hrs_read_val(fid, 'int8')
    if file_version >= 2:
        t.m_wave_window_median    = hrs_read_val(fid, 'float32')
        t.m_wave_set_value_uv     = hrs_read_val(fid, 'float32')
        t.m_wave_error            = hrs_read_val(fid, 'float32')
        t.m_wave_adjust_step_ma   = hrs_read_val(fid, 'float32')
        t.m_wave_min_intensity_ma = hrs_read_val(fid, 'float32')
        t.m_wave_max_intensity_ma = hrs_read_val(fid, 'float32')
    return t


def _read_down_cond_vns_trial_block(fid: BinaryIO, file_version: int) -> DownCondVnsTrial:
    """Read one Down Condition VNS trial block (V3 .hrs5, block_id=10)."""
    base = _read_mh_trial_block_full(fid)
    t = DownCondVnsTrial.__new__(DownCondVnsTrial)
    t.__dict__.update(base.__dict__)
    t.vns_delivered = hrs_read_val(fid, 'int8')
    if file_version >= 2:
        t.m_wave_window_median    = hrs_read_val(fid, 'float32')
        t.m_wave_set_value_uv     = hrs_read_val(fid, 'float32')
        t.m_wave_error            = hrs_read_val(fid, 'float32')
        t.m_wave_adjust_step_ma   = hrs_read_val(fid, 'float32')
        t.m_wave_min_intensity_ma = hrs_read_val(fid, 'float32')
        t.m_wave_max_intensity_ma = hrs_read_val(fid, 'float32')
    return t


def _read_up_cond_vns_trial_block(fid: BinaryIO, file_version: int) -> UpCondVnsTrial:
    """Read one Up Condition VNS trial block (V3 .hrs6, block_id=11)."""
    base = _read_mh_trial_block_full(fid)
    t = UpCondVnsTrial.__new__(UpCondVnsTrial)
    t.__dict__.update(base.__dict__)
    t.vns_delivered = hrs_read_val(fid, 'int8')
    if file_version >= 2:
        t.m_wave_window_median    = hrs_read_val(fid, 'float32')
        t.m_wave_set_value_uv     = hrs_read_val(fid, 'float32')
        t.m_wave_error            = hrs_read_val(fid, 'float32')
        t.m_wave_adjust_step_ma   = hrs_read_val(fid, 'float32')
        t.m_wave_min_intensity_ma = hrs_read_val(fid, 'float32')
        t.m_wave_max_intensity_ma = hrs_read_val(fid, 'float32')
    return t


def _read_frequency_test_trial_block(fid: BinaryIO, file_version: int) -> FrequencyTestTrial:
    """Read one Frequency Test trial block (V3 .hrsft, block_id=8)."""
    base = _read_mh_trial_block_full(fid)
    t = FrequencyTestTrial.__new__(FrequencyTestTrial)
    t.__dict__.update(base.__dict__)
    t.pulse_h_wave_mra = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    t.pulse_m_wave_mra = np.array(hrs_read_array(fid, 'float32'), dtype=np.float32)
    return t


def _read_dcp_trial_block(fid: BinaryIO, file_version: int = 8) -> DcpTrial:
    """Read one Down Condition Pellet trial block.

    Reads all standard MhRecTrial fields via _read_mh_trial_block, then
    appends the six DcpTrial-specific outcome fields (file_version >= 8).
    """
    base = _read_mh_trial_block(fid, file_version)
    t = DcpTrial.__new__(DcpTrial)
    t.__dict__.update(base.__dict__)
    t.success_threshold = float('nan')
    t.is_success        = 0
    t.pellet_delivered  = 0
    if file_version >= 8:
        t.h_wave_response   = hrs_read_val(fid, 'float32')
        t.m_wave_response   = hrs_read_val(fid, 'float32')
        t.hm_ratio          = hrs_read_val(fid, 'float32')
        t.success_threshold = hrs_read_val(fid, 'float32')
        t.is_success        = hrs_read_val(fid, 'int8')
        t.pellet_delivered  = hrs_read_val(fid, 'int8')
    if file_version >= 10:
        t.m_wave_window_median    = hrs_read_val(fid, 'float32')
        t.m_wave_set_value_uv     = hrs_read_val(fid, 'float32')
        t.m_wave_error            = hrs_read_val(fid, 'float32')
        t.m_wave_adjust_step_ma   = hrs_read_val(fid, 'float32')
        t.m_wave_min_intensity_ma = hrs_read_val(fid, 'float32')
        t.m_wave_max_intensity_ma = hrs_read_val(fid, 'float32')
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
                    if header.file_version >= 1:
                        t.actual_init_min_threshold = hrs_read_val(fid, 'float32')
                        t.actual_init_max_threshold = hrs_read_val(fid, 'float32')
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
    """Read a peri-stimulus trial file (.hrs1 or .hrs2).

    Used for:
    - V1 .hrs1: EMG Characterization stage  (stage_type=0, EmgCharHeader — use read_hrs1)
    - V1 .hrs2: MH Recruitment Curve stage  (sweeps across intensities)
    - V2 .hrs1: MH Recruitment Curve stage  (same binary format as V1 .hrs2)
    - V2 .hrs2: Control Mode stage          (fixed intensity, user-adjustable)

    All of the above share the MhRecHeader + MhRecTrial binary format.
    Block IDs handled: BLOCK_MH_TRIAL (3), BLOCK_CONTROL_MODE_TRIAL (6),
    BLOCK_DCP_TRIAL (5, used by V2 file_version ≥ 8 Control Mode).

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
        if header.file_version >= 8:
            header.app_version = hrs_read_string(fid)

        while True:
            chunk = fid.read(4)
            if len(chunk) < 4:
                break
            block_id = struct.unpack('i', chunk)[0]

            if block_id in (BLOCK_MH_TRIAL, BLOCK_CONTROL_MODE_TRIAL):
                try:
                    trials.append(_read_mh_trial_block(fid, header.file_version, block_id))
                except (struct.error, EOFError):
                    break
            elif block_id == BLOCK_EMG_DATA:
                # App bug: MhRecruitmentCurveTrial.save_to_file writes block_id=1 (EMG_DATA)
                # instead of block_id=3 (MH_TRIAL).  Disambiguate by peeking at the first
                # 24 bytes after the block_id.
                #
                # EMG block layout  (bytes 0-23):
                #   [0:8]   ts_open_ephys_sent  (uint64, wall-clock Unix ms from OE ZMQ)
                #   [8:16]  ts_python_received  (uint64, wall-clock Unix ms from time.time()*1000)
                #   [16:24] ts_background_emitted (uint64, wall-clock Unix ms)
                #   → all three are Unix ms (~1.7e12 for 2026), ascending, within seconds of each other
                #
                # Trial block layout (bytes 0-23):
                #   [0:8]   start_time          (uint64 Unix ms, OR float64 MATLAB datenum in old format)
                #   [8:12]  min_init_threshold  (float32)
                #   [12:16] max_init_threshold  (float32)
                #   [16:20] stimulation_amplitude_ma (float32)
                #   [20:24] first 4 bytes of trial_data length (uint32)
                #   → only bytes[0:8] is a Unix ms timestamp; [8:24] are floats/counts
                #
                # Strategy: read 24 bytes.  If all three uint64 windows are in the Unix-ms
                # range AND they are ascending AND within 1 hour of each other → EMG block.
                # Otherwise → trial block (only one or zero windows will be in range).
                # Old format check: if bytes[0:8] interpreted as float64 is > 1.0 → MATLAB
                # datenum → trial (float64 MATLAB datums for 2026 are ~738xxx, well above 1).
                _UNIX_MS_LO = 5e11   # ~1985-01-01
                _UNIX_MS_HI = 3e12   # ~2065-01-01
                pos = fid.tell()
                peek = fid.read(24)
                fid.seek(pos)
                if len(peek) < 24:
                    break
                peek_f64 = struct.unpack('<d', peek[:8])[0]
                if peek_f64 > 1.0:
                    # Old format: MATLAB datenum stored as float64 → trial block
                    trials.append(_read_mh_trial_block(fid, header.file_version))
                else:
                    ts0 = struct.unpack('<Q', peek[0:8])[0]
                    ts1 = struct.unpack('<Q', peek[8:16])[0]
                    ts2 = struct.unpack('<Q', peek[16:24])[0]
                    _is_emg = (
                        _UNIX_MS_LO < ts0 < _UNIX_MS_HI and
                        _UNIX_MS_LO < ts1 < _UNIX_MS_HI and
                        _UNIX_MS_LO < ts2 < _UNIX_MS_HI and
                        ts1 >= ts0 and ts2 >= ts1 and
                        (ts2 - ts0) < 3_600_000  # all three within 1 hour of each other
                    )
                    if _is_emg:
                        try:
                            emg_blocks.append(_read_emg_data_block(fid))
                        except (struct.error, EOFError):
                            # Truncated final EMG block (recording ended mid-frame); stop.
                            break
                    else:
                        try:
                            trials.append(_read_mh_trial_block(fid, header.file_version))
                        except (struct.error, EOFError):
                            break
            else:
                print(f"Warning: unknown block_id={block_id} at offset {fid.tell()-4}")
                break

    if trials:
        header.sample_rate = len(trials[0].trial_data) / (TRIAL_RECORD_MS / 1000)
    return header, trials, emg_blocks


def read_hrs3(filepath: str):
    """Read an .hrs3 (Down Condition Pellet) data file (hreflex_txbdc V2).

    file_version=8 appends six outcome fields per trial after all standard
    MhRecTrial fields (h_wave_response, m_wave_response, hm_ratio,
    success_threshold, is_success, pellet_delivered).

    Returns (header, trials, emg_blocks).
    """
    header = MhRecHeader()
    trials, emg_blocks = [], []

    with open(filepath, 'rb') as fid:
        header.file_version       = hrs_read_val(fid, 'int32')
        header.subject_id         = hrs_read_string(fid)
        header.session_start_time = hrs_read_datetime(fid)
        header.stage_name         = hrs_read_string(fid)
        header.stage_description  = hrs_read_string(fid)
        header.stage_type         = hrs_read_val(fid, 'int32')
        if header.file_version >= 9:
            header.app_version = hrs_read_string(fid)

        while True:
            chunk = fid.read(4)
            if len(chunk) < 4:
                break
            block_id = struct.unpack('i', chunk)[0]

            if block_id in (BLOCK_MH_TRIAL, BLOCK_DCP_TRIAL):
                try:
                    trials.append(_read_dcp_trial_block(fid, header.file_version))
                except (struct.error, EOFError):
                    break
            elif block_id == BLOCK_EMG_DATA:
                # Same block_id=1 disambiguation as read_hrs2.
                _UNIX_MS_LO = 5e11
                _UNIX_MS_HI = 3e12
                pos = fid.tell()
                peek = fid.read(24)
                fid.seek(pos)
                if len(peek) < 24:
                    break
                peek_f64 = struct.unpack('<d', peek[:8])[0]
                if peek_f64 > 1.0:
                    try:
                        trials.append(_read_dcp_trial_block(fid, header.file_version))
                    except (struct.error, EOFError):
                        break
                else:
                    ts0 = struct.unpack('<Q', peek[0:8])[0]
                    ts1 = struct.unpack('<Q', peek[8:16])[0]
                    ts2 = struct.unpack('<Q', peek[16:24])[0]
                    _is_emg = (
                        _UNIX_MS_LO < ts0 < _UNIX_MS_HI and
                        _UNIX_MS_LO < ts1 < _UNIX_MS_HI and
                        _UNIX_MS_LO < ts2 < _UNIX_MS_HI and
                        ts1 >= ts0 and ts2 >= ts1 and
                        (ts2 - ts0) < 3_600_000
                    )
                    if _is_emg:
                        try:
                            emg_blocks.append(_read_emg_data_block(fid))
                        except (struct.error, EOFError):
                            break
                    else:
                        try:
                            trials.append(_read_dcp_trial_block(fid, header.file_version))
                        except (struct.error, EOFError):
                            break
            else:
                print(f"Warning: unknown block_id={block_id} at offset {fid.tell()-4}")
                break

    if trials:
        header.sample_rate = len(trials[0].trial_data) / (TRIAL_RECORD_MS / 1000)
    return header, trials, emg_blocks


def _make_v3_block_loop(filepath, header, block_id_expected, reader_fn):
    """Shared block-loop body for V3 stage readers (hrs4/hrs5/hrs6)."""
    trials, emg_blocks = [], []
    with open(filepath, 'rb') as fid:
        header.file_version       = hrs_read_val(fid, 'int32')
        header.subject_id         = hrs_read_string(fid)
        header.session_start_time = hrs_read_datetime(fid)
        header.stage_name         = hrs_read_string(fid)
        header.stage_description  = hrs_read_string(fid)
        header.stage_type         = hrs_read_val(fid, 'int32')
        header.app_version        = hrs_read_string(fid)

        _UNIX_MS_LO = 5e11
        _UNIX_MS_HI = 3e12
        while True:
            chunk = fid.read(4)
            if len(chunk) < 4:
                break
            block_id = struct.unpack('i', chunk)[0]
            if block_id == block_id_expected:
                try:
                    trials.append(reader_fn(fid, header.file_version))
                except (struct.error, EOFError):
                    break
            elif block_id == BLOCK_EMG_DATA:
                pos = fid.tell()
                peek = fid.read(24)
                fid.seek(pos)
                if len(peek) < 24:
                    break
                ts0 = struct.unpack('<Q', peek[0:8])[0]
                ts1 = struct.unpack('<Q', peek[8:16])[0]
                ts2 = struct.unpack('<Q', peek[16:24])[0]
                _is_emg = (
                    _UNIX_MS_LO < ts0 < _UNIX_MS_HI and
                    _UNIX_MS_LO < ts1 < _UNIX_MS_HI and
                    _UNIX_MS_LO < ts2 < _UNIX_MS_HI and
                    ts1 >= ts0 and ts2 >= ts1 and
                    (ts2 - ts0) < 3_600_000
                )
                if _is_emg:
                    try:
                        emg_blocks.append(_read_emg_data_block(fid))
                    except (struct.error, EOFError):
                        break
                else:
                    try:
                        trials.append(reader_fn(fid, header.file_version))
                    except (struct.error, EOFError):
                        break
            else:
                print(f"Warning: unknown block_id={block_id} at offset {fid.tell()-4}")
                break
    return trials, emg_blocks


def read_hrs4(filepath: str):
    """Read a .hrs4 (Up Condition Pellet) data file (hreflex_txbdc V3).

    Returns (header, trials, emg_blocks).
    """
    header = MhRecHeader()
    trials, emg_blocks = _make_v3_block_loop(
        filepath, header, BLOCK_UP_COND_PELLET_TRIAL, _read_up_cond_pellet_trial_block)
    if trials:
        header.sample_rate = len(trials[0].trial_data) / (TRIAL_RECORD_MS / 1000)
    return header, trials, emg_blocks


def read_hrs5(filepath: str):
    """Read a .hrs5 (Down Condition VNS) data file (hreflex_txbdc V3).

    Returns (header, trials, emg_blocks).
    """
    header = MhRecHeader()
    trials, emg_blocks = _make_v3_block_loop(
        filepath, header, BLOCK_DOWN_COND_VNS_TRIAL, _read_down_cond_vns_trial_block)
    if trials:
        header.sample_rate = len(trials[0].trial_data) / (TRIAL_RECORD_MS / 1000)
    return header, trials, emg_blocks


def read_hrs6(filepath: str):
    """Read a .hrs6 (Up Condition VNS) data file (hreflex_txbdc V3).

    Returns (header, trials, emg_blocks).
    """
    header = MhRecHeader()
    trials, emg_blocks = _make_v3_block_loop(
        filepath, header, BLOCK_UP_COND_VNS_TRIAL, _read_up_cond_vns_trial_block)
    if trials:
        header.sample_rate = len(trials[0].trial_data) / (TRIAL_RECORD_MS / 1000)
    return header, trials, emg_blocks


def read_hrs_ft(filepath: str):
    """Read a .hrsft (Frequency Test) data file (hreflex_txbdc V3).

    Returns (header, trials, emg_blocks).
    """
    header = FrequencyTestHeader()
    trials, emg_blocks = [], []

    with open(filepath, 'rb') as fid:
        header.file_version       = hrs_read_val(fid, 'int32')
        header.subject_id         = hrs_read_string(fid)
        header.session_start_time = hrs_read_datetime(fid)
        header.stage_name         = hrs_read_string(fid)
        header.stage_description  = hrs_read_string(fid)
        header.stage_type         = hrs_read_val(fid, 'int32')
        header.app_version        = hrs_read_string(fid)
        if header.file_version >= 1:
            header.n_pulses_per_train = hrs_read_val(fid, 'int32')
            header.event_period_us    = hrs_read_val(fid, 'int32')
            header.pulse_width_us     = hrs_read_val(fid, 'int32')

        _UNIX_MS_LO = 5e11
        _UNIX_MS_HI = 3e12
        while True:
            chunk = fid.read(4)
            if len(chunk) < 4:
                break
            block_id = struct.unpack('i', chunk)[0]
            if block_id == BLOCK_FREQUENCY_TEST_TRIAL:
                try:
                    trials.append(_read_frequency_test_trial_block(fid, header.file_version))
                except (struct.error, EOFError):
                    break
            elif block_id == BLOCK_EMG_DATA:
                pos = fid.tell()
                peek = fid.read(24)
                fid.seek(pos)
                if len(peek) < 24:
                    break
                ts0 = struct.unpack('<Q', peek[0:8])[0]
                ts1 = struct.unpack('<Q', peek[8:16])[0]
                ts2 = struct.unpack('<Q', peek[16:24])[0]
                _is_emg = (
                    _UNIX_MS_LO < ts0 < _UNIX_MS_HI and
                    _UNIX_MS_LO < ts1 < _UNIX_MS_HI and
                    _UNIX_MS_LO < ts2 < _UNIX_MS_HI and
                    ts1 >= ts0 and ts2 >= ts1 and
                    (ts2 - ts0) < 3_600_000
                )
                if _is_emg:
                    try:
                        emg_blocks.append(_read_emg_data_block(fid))
                    except (struct.error, EOFError):
                        break
                else:
                    try:
                        trials.append(_read_frequency_test_trial_block(fid, header.file_version))
                    except (struct.error, EOFError):
                        break
            else:
                print(f"Warning: unknown block_id={block_id} at offset {fid.tell()-4}")
                break

    if trials:
        header.sample_rate = len(trials[0].trial_data) / (TRIAL_RECORD_MS / 1000)
    return header, trials, emg_blocks


def find_hrs_files(directory: str):
    """Auto-detect .hrs* files in a recording directory.

    Returns (hrs1_path, hrs2_path, hrs3_path, hrs4_path, hrs5_path, hrs6_path, hrsft_path).
    Any path is None if the corresponding file is absent.  Raises FileNotFoundError
    only when no .hrs* files of any kind are found in the directory.
    """
    hrs1_files  = globmod.glob(os.path.join(directory, "*.hrs1"))
    hrs2_files  = globmod.glob(os.path.join(directory, "*.hrs2"))
    hrs3_files  = globmod.glob(os.path.join(directory, "*.hrs3"))
    hrs4_files  = globmod.glob(os.path.join(directory, "*.hrs4"))
    hrs5_files  = globmod.glob(os.path.join(directory, "*.hrs5"))
    hrs6_files  = globmod.glob(os.path.join(directory, "*.hrs6"))
    hrsft_files = globmod.glob(os.path.join(directory, "*.hrft"))

    if not any([hrs1_files, hrs2_files, hrs3_files, hrs4_files,
                hrs5_files, hrs6_files, hrsft_files]):
        raise FileNotFoundError(f"No .hrs* files found in '{directory}'")
    for ext, files in [('.hrs1', hrs1_files), ('.hrs2', hrs2_files), ('.hrs3', hrs3_files),
                       ('.hrs4', hrs4_files), ('.hrs5', hrs5_files), ('.hrs6', hrs6_files),
                       ('.hrft', hrsft_files)]:
        if len(files) > 1:
            print(f"Warning: multiple {ext} files found, using: {files[0]}")

    return (
        hrs1_files[0]  if hrs1_files  else None,
        hrs2_files[0]  if hrs2_files  else None,
        hrs3_files[0]  if hrs3_files  else None,
        hrs4_files[0]  if hrs4_files  else None,
        hrs5_files[0]  if hrs5_files  else None,
        hrs6_files[0]  if hrs6_files  else None,
        hrsft_files[0] if hrsft_files else None,
    )


def detect_app_version(directory: str) -> int:
    """Detect which H-Reflex app version produced files in a recording directory.

    Returns 1 (hreflex_txbdc 0.0.1), 2 (0.0.2), or 3 (0.0.3).

    V1 convention: .hrs1 = EMG Char; .hrs2 = MH Recruitment.
    V2 convention: .hrs1 = MH Recruitment; .hrs2 = Control Mode; .hrs3 = DCP.
    V3 convention: adds .hrs4 (S4), .hrs5 (S5), .hrs6 (S6), .hrsft (FT);
                   .hrs2 file_version bumped to 9; .hrs3 file_version bumped to 10.

    Detection priority:
    1. .hrs4/.hrs5/.hrs6/.hrsft present → V3
    2. .hrs2 with file_version >= 9 → V3
    3. .hrs3 present → V2
    4. .hrs1 present → stage_type 0 = V1, else V2
    5. .hrs2 only → "Control Mode" in description → V2, else V1
    6. No files → V1 (default)
    """
    for ext in ('*.hrs4', '*.hrs5', '*.hrs6', '*.hrft'):
        if globmod.glob(os.path.join(directory, ext)):
            return 3

    hrs2_files = globmod.glob(os.path.join(directory, "*.hrs2"))
    if hrs2_files:
        try:
            with open(hrs2_files[0], 'rb') as fid:
                fv = hrs_read_val(fid, 'int32')
            if fv >= 9:
                return 3
        except Exception:
            pass

    hrs3_files = globmod.glob(os.path.join(directory, "*.hrs3"))
    if hrs3_files:
        return 2

    hrs1_files = globmod.glob(os.path.join(directory, "*.hrs1"))
    if hrs1_files:
        try:
            with open(hrs1_files[0], 'rb') as fid:
                hrs_read_val(fid, 'int32')   # file_version
                hrs_read_string(fid)          # subject_id
                hrs_read_datetime(fid)        # session_start_time
                hrs_read_string(fid)          # stage_name
                hrs_read_string(fid)          # stage_description
                stage_type = hrs_read_val(fid, 'int32')
            return 1 if stage_type == 0 else 2
        except Exception:
            pass

    if hrs2_files:
        try:
            with open(hrs2_files[0], 'rb') as fid:
                hrs_read_val(fid, 'int32')   # file_version
                hrs_read_string(fid)          # subject_id
                hrs_read_datetime(fid)        # session_start_time
                hrs_read_string(fid)          # stage_name
                stage_description = hrs_read_string(fid)
            return 2 if 'control mode' in stage_description.lower() else 1
        except Exception:
            pass

    return 1  # default


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


def _trial_onset_oe(trial, bin_samples: int) -> 'int | None':
    """Return the absolute OE sample number of the stim onset.

    Prefers digital_onset_sample_num (v7+) over the ADC-derived position.
    Returns None when neither source is available.
    """
    dig = getattr(trial, 'digital_onset_sample_num', -1)
    if dig is not None and dig >= 0:
        return int(dig)
    fid_val = getattr(trial, 'first_post_trigger_frame_sample_id', 0)
    osi = getattr(trial, 'onset_sample_index', -1)
    if fid_val > 0 and osi >= 0:
        return int(fid_val) + (osi - bin_samples)
    return None


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

    Returns (t_ms, emg, adc_or_None, stim_end_ms_or_None, stim_adc_or_None).
    adc_or_None      – ADC sync/TTL line slice (file_version >= 1).
    stim_adc_or_None – raw stimulator output waveform slice (file_version >= 4).
    """
    has_sync = len(trial.sync_data) > 1

    # For file_version >= 2 the app stores onset_sample_index for both ADC
    # (onset_detected=1) and digital DIGITAL IN (onset_detected=2) sources.
    # Trust it directly — recomputing from digital_onset_sample_num would
    # require knowing the exact bin_sample_count used at record time (which
    # depends on sample_rate and may differ from the offline BIN_SAMPLES constant).
    _od  = getattr(trial, 'onset_detected', 0)
    _osi = getattr(trial, 'onset_sample_index', -1)
    if _od >= 1 and _osi >= 0:
        onset_idx = _osi
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

    stim_adc = None
    _stim_adc_raw = getattr(trial, 'stim_adc_data', np.array([], dtype=np.float32))
    if len(_stim_adc_raw) >= i1:
        candidate = _stim_adc_raw[i0:i1]
        if len(candidate) == n:
            stim_adc = candidate

    return t_ms, emg, adc, stim_end_ms, stim_adc


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

    _dig_oe = getattr(trial, 'digital_onset_sample_num', -1)
    if _dig_oe is not None and _dig_oe >= 0:
        onset_oe = int(_dig_oe)
    else:
        first_id = getattr(trial, 'first_post_trigger_frame_sample_id', 0)
        onset_idx_in_trial = getattr(trial, 'onset_sample_index', -1)
        if first_id > 0 and onset_idx_in_trial >= 0:
            # onset_sample_index is relative to the start of trial_data;
            # the first BIN_SAMPLES of trial_data are pre-trigger, so:
            onset_oe = int(first_id) + (onset_idx_in_trial - bin_samples)

    if onset_oe is None:
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
        4: "v4: + stim_adc_data",
        5: "v5: + background_emg_mean + background_bins",
        6: "v6: + stim_polarity_reversed",
        7: "v7: + digital_onset_sample_num/channel",
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

    if header.file_version >= 6 and trials:
        n_normal   = sum(1 for t in trials if getattr(t, 'stim_polarity_reversed', 0) == 0)
        n_reversed = sum(1 for t in trials if getattr(t, 'stim_polarity_reversed', 0) == 1)
        print(f"  Stim polarity:      {n_normal} normal, {n_reversed} reversed"
              + ("  <-- dual-polarity session" if n_reversed > 0 else ""))

    if header.file_version >= 7 and trials:
        n_dig = sum(1 for t in trials if getattr(t, 'digital_onset_sample_num', -1) >= 0)
        print(f"  Digital onsets:     {n_dig}/{len(trials)} trials have digital onset"
              + ("  <-- all digital" if n_dig == len(trials) else
                 "  <-- partial - some trials missing digital event" if n_dig > 0 else
                 "  <-- none detected"))

    if len(emg_blocks) > 0:
        b0 = emg_blocks[0]
        n_raw = len(b0.raw_channels[0]) if b0.raw_channels else 0
        print("\n=== First EMG Block ===")
        print(f"  Channel names:  {b0.channel_names}")
        print(f"  Raw channels:   {len(b0.raw_channels)} x {n_raw} samples")
        print(f"  Diff samples:   {len(b0.diff)}")


def split_trials_by_polarity(trials):
    """Split HRS2 trials by stim_polarity_reversed field (file_version >= 6).

    Returns a dict keyed by a human-readable label:
      - Single polarity detected → {'All trials': trials}
      - Two polarities detected  → {'Normal (0)': [...], 'Reversed (1)': [...]}

    Trials from older files (no stim_polarity_reversed attribute) are treated as
    normal polarity (value 0).
    """
    if not trials:
        return {'All trials': []}

    normal   = [t for t in trials if getattr(t, 'stim_polarity_reversed', 0) == 0]
    reversed_ = [t for t in trials if getattr(t, 'stim_polarity_reversed', 0) == 1]

    if normal and reversed_:
        print(f"Dual-polarity session detected: "
              f"{len(normal)} normal, {len(reversed_)} reversed → analysing separately.")
        return {'Normal (0)': normal, 'Reversed (1)': reversed_}
    elif reversed_:
        print(f"Single-polarity session (config 1 only): {len(reversed_)} reversed trials.")
        return {'Reversed (1)': reversed_}
    else:
        return {'All trials': trials}


def plot_hm_ratio_summary(trials_by_polarity, header,
                           m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                           h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                           sample_rate: float = SAMPLE_RATE,
                           pre_ms: float = 2.0, post_ms: float = 15.0,
                           h_threshold_pct: float = 80.0):
    """Bar-chart summary of M-wave MRA, H-wave MRA, and H:M ratio per polarity group.

    Figure 1 — Grouped bar chart with M / H / H:M subplots
        * Each subplot shows one bar per polarity group (mean ± SD error bar).
        * Stats panel (whitespace): n, SD, CV per group for each metric.

    Figure 2 — H:M ratio by stimulation amplitude
        * Bar per amplitude × polarity (mean ± SD), n labelled above each bar.

    Figure 3 — H-wave MRA distribution histogram per polarity group
        * Overlaid histograms of per-trial H-wave MRA values.
        * Vertical dashed line at the h_threshold_pct percentile (default 80th = top 20%).
        * Trials above threshold are highlighted; count and value annotated.

    All amplitudes use MRA (mean(|signal|)) within the M/H windows.
    h_threshold_pct : percentile for the threshold line (e.g. 80.0 = top 20%).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as _mpatches

    trials_by_polarity = {k: v for k, v in trials_by_polarity.items() if v}
    if not trials_by_polarity:
        print("No trials.")
        return

    colours = {'Normal (0)': '#2196F3', 'Reversed (1)': '#FF5722', 'All trials': '#4CAF50'}
    palette  = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    def _col(label, i):
        return colours.get(label, palette[i % len(palette)])

    _ms_ps = 1000.0 / sample_rate
    _bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
    _rec_s = int(TRIAL_RECORD_MS  * sample_rate / 1000)

    groups = list(trials_by_polarity.keys())
    n_grps = len(groups)

    # Per-trial MRA within M and H windows for every polarity group
    group_stats: dict = {}
    amp_ratio:   dict = {}
    for lbl, trs in trials_by_polarity.items():
        mv_all, hv_all, hm_all = [], [], []
        amp_d: dict = defaultdict(list)
        for tr in trs:
            t_ms, emg, _, _, _ = get_trial_window(tr, pre_ms, post_ms,
                                                   ms_per_sample=_ms_ps,
                                                   bin_samples=_bin_s,
                                                   record_samples=_rec_s)
            mm = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
            hm = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
            mv = float(np.nanmean(np.abs(emg[mm]))) if mm.any() else np.nan
            hv = float(np.nanmean(np.abs(emg[hm]))) if hm.any() else np.nan
            if np.isfinite(mv):
                mv_all.append(mv)
            if np.isfinite(hv):
                hv_all.append(hv)
            if np.isfinite(mv) and np.isfinite(hv) and mv > 0:
                ratio = hv / mv
                hm_all.append(ratio)
                amp_d[round(tr.stimulation_amplitude_ma, 4)].append(ratio)
        group_stats[lbl] = {'m': np.array(mv_all), 'h': np.array(hv_all), 'hm': np.array(hm_all)}
        amp_ratio[lbl]   = amp_d

    # ── Figure 1: Grouped bar chart — M / H / H:M + stats panel ─────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5),
                             gridspec_kw={'width_ratios': [3, 3, 3, 2.5]})
    ax_m, ax_h, ax_hm, ax_txt = axes
    x_pos  = np.arange(n_grps)
    bar_w  = min(0.55, 0.8 / max(n_grps, 1))

    metrics = [
        ('m',  ax_m,  'M-wave MRA (µV)', 'M-wave'),
        ('h',  ax_h,  'H-wave MRA (µV)', 'H-wave'),
        ('hm', ax_hm, 'H:M Ratio (MRA)', 'H:M Ratio'),
    ]
    stats_lines: list = []
    for key, ax, ylabel, title in metrics:
        stats_lines.append(f'── {title} ──')
        for i, lbl in enumerate(groups):
            v = group_stats[lbl][key]
            v = v[np.isfinite(v)] if len(v) else v
            mu = float(np.mean(v)) if len(v) else 0.0
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            cv = sd / mu if mu != 0 else float('nan')
            col = _col(lbl, i)
            ax.bar(i, mu, width=bar_w, color=col, alpha=0.78,
                   yerr=sd, capsize=6,
                   error_kw={'elinewidth': 2.2, 'ecolor': col, 'capthick': 2})
            stats_lines.append(f'{lbl}\n  n={len(v)}  SD={sd:.3f}  CV={cv:.2f}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([lbl.replace(' (', '\n(') for lbl in groups], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
        stats_lines.append('')

    ax_txt.axis('off')
    ax_txt.text(0.05, 0.97, '\n'.join(stats_lines),
                transform=ax_txt.transAxes,
                va='top', ha='left', fontsize=8.5, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#cccccc', alpha=0.95))

    fig.suptitle(
        f'M / H / H:M Summary (MRA)  —  {header.subject_id}\n'
        f'M: {m_start_ms}–{m_end_ms} ms  |  H: {h_start_ms}–{h_end_ms} ms',
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

    # ── Figure 2: H:M ratio by stim amplitude (bar + STD) ────────────────────
    _all_amps = sorted({a for _d in amp_ratio.values() for a in _d})
    if _all_amps:
        _n_amps = len(_all_amps)
        _bw2    = min(0.8 / n_grps, 0.35)
        _offs   = (np.linspace(-0.4 + _bw2 / 2, 0.4 - _bw2 / 2, n_grps)
                   if n_grps > 1 else np.array([0.0]))

        fig2, ax2 = plt.subplots(figsize=(max(10, _n_amps * n_grps * 0.9 + 3), 5))

        for gi, lbl in enumerate(groups):
            col = _col(lbl, gi)
            _ad  = amp_ratio[lbl]
            off  = float(_offs[gi])
            for ai, amp in enumerate(_all_amps):
                _v = np.array(_ad.get(amp, []))
                if not len(_v):
                    continue
                _xp = ai + 1 + off
                _mu = float(np.mean(_v))
                _sd = float(np.std(_v, ddof=1)) if len(_v) > 1 else 0.0
                ax2.bar(_xp, _mu, width=_bw2 * 0.9, color=col, alpha=0.78,
                        yerr=_sd, capsize=4,
                        error_kw={'elinewidth': 1.8, 'ecolor': col, 'capthick': 1.5})
                ax2.text(_xp, _mu + _sd + max(_mu * 0.03, 0.01),
                         f'n={len(_v)}', ha='center', va='bottom', fontsize=6.5, color=col)

        ax2.set_xticks(range(1, _n_amps + 1))
        ax2.set_xticklabels([f'{a:.2f} mA' for a in _all_amps],
                            rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('H:M Ratio (MRA)', fontsize=11)
        ax2.set_title(
            f'H:M Ratio by Stimulation Amplitude — {header.subject_id}\n'
            f'M: {m_start_ms}–{m_end_ms} ms  |  H: {h_start_ms}–{h_end_ms} ms',
            fontsize=11
        )
        ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax2.grid(True, axis='y', alpha=0.3)
        if n_grps > 1:
            patches = [_mpatches.Patch(color=_col(lbl, gi), alpha=0.78, label=lbl)
                       for gi, lbl in enumerate(groups)]
            ax2.legend(handles=patches, fontsize=9)
        plt.tight_layout()
        plt.show()

    # ── Figure 3: Distribution histograms — M-wave, H-wave, H:M ─────────────
    _dist_metrics = [
        ('m',  f'M-wave MRA (µV)\n[{m_start_ms}–{m_end_ms} ms]', 'M-wave MRA'),
        ('h',  f'H-wave MRA (µV)\n[{h_start_ms}–{h_end_ms} ms]', 'H-wave MRA'),
        ('hm', 'H:M Ratio (MRA)',                                  'H:M Ratio'),
    ]
    fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))

    def _draw_dist_ax(ax, key, xlabel, title):
        pooled = np.concatenate([
            gs[key][np.isfinite(gs[key])]
            for gs in group_stats.values() if len(gs[key])
        ]) if group_stats else np.array([])

        if not len(pooled):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=11, fontweight='bold')
            return

        q1  = float(np.percentile(pooled, 25))
        q2  = float(np.percentile(pooled, 50))
        q3  = float(np.percentile(pooled, 75))
        thr = float(np.percentile(pooled, h_threshold_pct))

        _nbins = min(40, max(10, len(pooled) // 8))

        for i, lbl in enumerate(groups):
            v = group_stats[lbl][key]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            ax.hist(v, bins=_nbins, color=_col(lbl, i), alpha=0.55,
                    label=lbl, edgecolor='white', linewidth=0.4)

        # Q1 / Q2 / Q3 vertical lines with inline labels (staggered y to avoid overlap)
        for _val, _name, _ls, _lw, _qc, _yp in [
            (q1,  'Q1', ':',  1.6, '#555555', 0.90),
            (q2,  'Q2', '--', 2.0, '#222222', 0.97),
            (q3,  'Q3', ':',  1.6, '#555555', 0.83),
        ]:
            ax.axvline(_val, color=_qc, linewidth=_lw, linestyle=_ls, zorder=3)
            ax.text(_val, _yp, f'{_name}\n{_val:.2f}',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=7.5, color=_qc,
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.5))

        # threshold line
        ax.axvline(thr, color='crimson', linewidth=2.0, linestyle='--', zorder=4)
        ax.text(thr, 0.72, f'{h_threshold_pct:.0f}th pct\n{thr:.2f}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7.5, color='crimson',
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.5))

        n_above = int(np.sum(pooled >= thr))
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Trial count', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.text(0.98, 0.60,
                f'n={len(pooled)}\n'
                f'Q1 = {q1:.2f}\n'
                f'Q2 = {q2:.2f}\n'
                f'Q3 = {q3:.2f}\n'
                f'≥{h_threshold_pct:.0f}th: {n_above} ({100*n_above/len(pooled):.1f}%)',
                transform=ax.transAxes, va='top', ha='right', fontsize=8,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', alpha=0.95))

    for ax, (key, xlabel, title) in zip(axes3, _dist_metrics):
        _draw_dist_ax(ax, key, xlabel, title)

    fig3.suptitle(
        f'EMG Distribution Summary — {header.subject_id}\n'
        f'Threshold: {h_threshold_pct:.0f}th percentile (top {100 - h_threshold_pct:.0f}%)',
        fontsize=11
    )
    plt.tight_layout()
    plt.show()


def plot_mwave_control_error(trials, header, title_suffix: str = ''):
    """Plot M-wave stabilization control error and stim amplitude over trials (V3 S2+).

    Left Y axis : m_wave_error (µV); falls back to m_wave_window_median when all NaN.
    Right Y axis: stimulation_amplitude_ma (mA, orange).

    Requires trials with file_version >= 9 fields populated (V3 Control Mode / S4/S5/S6).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not trials:
        print("No trials to plot.")
        return

    trial_nums   = list(range(1, len(trials) + 1))
    errors       = [getattr(t, 'm_wave_error',        float('nan')) for t in trials]
    medians      = [getattr(t, 'm_wave_window_median', float('nan')) for t in trials]
    stim_amps    = [t.stimulation_amplitude_ma for t in trials]

    has_error = any(not (e != e) for e in errors)  # nan check: e != e iff nan
    y_left       = errors  if has_error else medians
    y_left_label = 'M-wave Error (µV)' if has_error else 'M-wave Window Median (µV)'

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
        go.Scatter(x=trial_nums, y=y_left, name=y_left_label,
                   mode='lines+markers', marker=dict(size=3), line=dict(color='steelblue')),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=trial_nums, y=stim_amps, name='Stim Amplitude (mA)',
                   mode='lines+markers', marker=dict(size=3), line=dict(color='orange')),
        secondary_y=True)

    title = f'M-Wave Control Error — {header.subject_id}'
    if title_suffix:
        title += f'  {title_suffix}'
    fig.update_layout(title=title, xaxis_title='Trial #', height=400,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text=y_left_label, secondary_y=False)
    fig.update_yaxes(title_text='Stim Amplitude (mA)', secondary_y=True)
    fig.show()


def plot_frequency_test(trials, header, title_suffix: str = ''):
    """Plot per-pulse H-wave and M-wave MRA across a Frequency Test train (V3 FT).

    Shows mean ± std of pulse_h_wave_mra and pulse_m_wave_mra across all trials,
    normalised to pulse 1 to show depression.
    """
    import plotly.graph_objects as go

    if not trials:
        print("No trials to plot.")
        return

    n_pulses = header.n_pulses_per_train if hasattr(header, 'n_pulses_per_train') else 0
    h_arrays = [getattr(t, 'pulse_h_wave_mra', np.array([])) for t in trials]
    m_arrays = [getattr(t, 'pulse_m_wave_mra', np.array([])) for t in trials]
    h_valid  = [a for a in h_arrays if len(a) > 0]
    m_valid  = [a for a in m_arrays if len(a) > 0]

    if not h_valid and not m_valid:
        print("No pulse MRA data available.")
        return

    pulse_idx = list(range(1, (len(h_valid[0]) if h_valid else len(m_valid[0])) + 1))
    fig = go.Figure()
    if h_valid:
        h_mat = np.vstack(h_valid)
        fig.add_trace(go.Scatter(x=pulse_idx, y=h_mat.mean(axis=0).tolist(),
                                  error_y=dict(type='data', array=h_mat.std(axis=0).tolist()),
                                  name='H-wave MRA (µV)', mode='lines+markers',
                                  line=dict(color='royalblue')))
    if m_valid:
        m_mat = np.vstack(m_valid)
        fig.add_trace(go.Scatter(x=pulse_idx, y=m_mat.mean(axis=0).tolist(),
                                  error_y=dict(type='data', array=m_mat.std(axis=0).tolist()),
                                  name='M-wave MRA (µV)', mode='lines+markers',
                                  line=dict(color='firebrick')))

    title = f'Frequency Test — {header.subject_id}'
    if hasattr(header, 'event_period_us') and header.event_period_us > 0:
        hz = round(1e6 / header.event_period_us, 1)
        title += f'  ({hz} Hz train)'
    if title_suffix:
        title += f'  {title_suffix}'
    fig.update_layout(title=title, xaxis_title='Pulse # in train',
                      yaxis_title='MRA (µV)', height=450,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.show()


def plot_ft_depression_curve(trials, header, sample_rate=None, title_suffix=''):
    """Mean ± 1σ H-wave and M-wave MRA per pulse position across all FT trials.

    The most-recent trial is overlaid as a dashed line so you can compare the
    last-observed depression profile against the session average.
    """
    import matplotlib.pyplot as plt

    if not trials:
        print("No trials to plot.")
        return

    h_arrays = [getattr(t, 'pulse_h_wave_mra', np.array([])) for t in trials]
    m_arrays = [getattr(t, 'pulse_m_wave_mra', np.array([])) for t in trials]
    h_valid  = [a for a in h_arrays if len(a) > 0]
    m_valid  = [a for a in m_arrays if len(a) > 0]

    if not h_valid and not m_valid:
        print("No pulse MRA data available.")
        return

    n_pulses  = len(h_valid[0]) if h_valid else len(m_valid[0])
    pulse_idx = np.arange(1, n_pulses + 1)
    hz        = round(1e6 / header.event_period_us, 1) if getattr(header, 'event_period_us', 0) else '?'
    try:
        amp_str = f'{float(getattr(trials[0], "stimulation_amplitude_ma", 0.0)):.3f} mA'
    except (TypeError, ValueError):
        amp_str = '? mA'

    fig, ax = plt.subplots(figsize=(max(8, n_pulses * 0.85), 5))

    if h_valid:
        h_mat  = np.vstack(h_valid)
        h_mean = h_mat.mean(axis=0)
        h_std  = h_mat.std(axis=0)
        ax.plot(pulse_idx, h_mean, 'o-', color='royalblue', lw=2, ms=7,
                label=f'H-wave mean (n={len(h_valid)})', zorder=4)
        ax.fill_between(pulse_idx, h_mean - h_std, h_mean + h_std,
                        color='royalblue', alpha=0.18, zorder=3)
        ax.plot(pulse_idx, h_valid[-1], 'o--', color='royalblue', lw=1.2, ms=4,
                alpha=0.55, label='H-wave (last trial)', zorder=3)

    if m_valid:
        m_mat  = np.vstack(m_valid)
        m_mean = m_mat.mean(axis=0)
        m_std  = m_mat.std(axis=0)
        ax.plot(pulse_idx, m_mean, 's-', color='firebrick', lw=2, ms=7,
                label=f'M-wave mean (n={len(m_valid)})', zorder=4)
        ax.fill_between(pulse_idx, m_mean - m_std, m_mean + m_std,
                        color='firebrick', alpha=0.18, zorder=3)
        ax.plot(pulse_idx, m_valid[-1], 's--', color='firebrick', lw=1.2, ms=4,
                alpha=0.55, label='M-wave (last trial)', zorder=3)

    ax.set_xlabel('Pulse # in train', fontsize=11)
    ax.set_ylabel('MRA (µV)', fontsize=11)
    ax.set_xticks(pulse_idx)
    title = f'H/M Wave Per Pulse  ·  {hz} Hz  ·  {amp_str}  ·  n={len(trials)} trials'
    if title_suffix:
        title += f'  ·  {title_suffix}'
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_ft_averaged_waveforms(trials, header, pre_pulse_ms=2.0, post_pulse_ms=20.0,
                                m_start_ms=2.0, m_end_ms=4.0,
                                h_start_ms=6.0, h_end_ms=10.0,
                                sample_rate=None, n_per_page=6, page=0):
    """Paged 2×3 grid of averaged EMG waveforms per pulse position, mirroring plot_hrs2_analysis.

    Each tile shows individual trial segments (low alpha) + bold mean waveform for one
    pulse position. M-wave window is blue-shaded with dashed borders; H-wave is green.
    MRA annotations (hlines + text) match the HRS2 analysis style.
    A colorbar at the top maps pulse # to colour (coolwarm: blue=1, red=last).

    Returns total_pages (int).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec

    sr       = sample_rate or getattr(header, 'sample_rate', SAMPLE_RATE)
    n_pulses = getattr(header, 'n_pulses_per_train', 0)
    if n_pulses == 0:
        n_pulses = max((len(getattr(t, 'pulse_h_wave_mra', [])) for t in trials), default=0)
    if n_pulses == 0:
        print("No pulse data available.")
        return 1

    period_us = getattr(header, 'event_period_us', 0)
    if period_us == 0:
        print("event_period_us not set — cannot reconstruct pulse timing.")
        return 1

    pulse_period_samples = round(period_us / 1e6 * sr)
    pre_samp  = round(pre_pulse_ms  * sr / 1000)
    post_samp = round(post_pulse_ms * sr / 1000)
    t_ms      = np.linspace(-pre_pulse_ms, post_pulse_ms, pre_samp + post_samp)
    hz        = round(1e6 / period_us, 1)
    colors    = cm.coolwarm(np.linspace(0, 1, max(n_pulses, 2)))

    total_pages = max(1, int(np.ceil(n_pulses / n_per_page)))
    page        = max(0, min(page, total_pages - 1))
    start_k     = page * n_per_page
    end_k       = min(start_k + n_per_page, n_pulses)

    # Extract and average segments for each pulse position on this page
    pulse_data = []
    for k in range(start_k, end_k):
        segs, h_mras, m_mras = [], [], []
        for t in trials:
            onset = getattr(t, 'onset_sample_index', -1)
            if onset < 0:
                continue
            emg     = np.array(t.trial_data, dtype=float)
            onset_k = onset + k * pulse_period_samples
            s, e    = onset_k - pre_samp, onset_k + post_samp
            if s >= 0 and e <= len(emg):
                segs.append(emg[s:e])
            h_arr = getattr(t, 'pulse_h_wave_mra', [])
            m_arr = getattr(t, 'pulse_m_wave_mra', [])
            if k < len(h_arr):
                h_mras.append(float(h_arr[k]))
            if k < len(m_arr):
                m_mras.append(float(m_arr[k]))
        pulse_data.append((k, segs,
                           float(np.mean(h_mras)) if h_mras else 0.0,
                           float(np.mean(m_mras)) if m_mras else 0.0))

    ncols, nrows = 3, 2
    fig = plt.figure(figsize=(15.0, 7.5))
    gs  = GridSpec(nrows + 1, ncols,
                   height_ratios=[0.10] + [1] * nrows,
                   hspace=0.55, wspace=0.35,
                   top=0.93, bottom=0.08, left=0.07, right=0.97)

    # Colorbar spanning all columns — pulse # → colour legend
    cbar_ax = fig.add_subplot(gs[0, :])
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(1, max(n_pulses, 2)))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Pulse # in train  (blue = pulse 1  ·  red = pulse {n_pulses})',
                   fontsize=9)
    ticks = sorted({1, max(1, n_pulses // 2), n_pulses})
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'Pulse {tt}' for tt in ticks])

    for idx, (k, segs, h_mra, m_mra) in enumerate(pulse_data):
        row, col = divmod(idx, ncols)
        ax       = fig.add_subplot(gs[row + 1, col])
        color    = colors[k]

        # M-wave: blue shading + dashed borders
        ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.13, zorder=1)
        ax.axvline(m_start_ms, color='blue',  ls='--', lw=1.2, alpha=0.8, zorder=2)
        ax.axvline(m_end_ms,   color='blue',  ls='--', lw=1.2, alpha=0.8, zorder=2)
        # H-wave: green shading + dashed borders
        ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.13, zorder=1)
        ax.axvline(h_start_ms, color='green', ls='--', lw=1.2, alpha=0.8, zorder=2)
        ax.axvline(h_end_ms,   color='green', ls='--', lw=1.2, alpha=0.8, zorder=2)
        # Stim onset
        ax.axvline(0, color='#aaa', lw=0.8, ls=':', zorder=1)

        # Individual trial traces (low alpha, gray)
        for seg in segs:
            ax.plot(t_ms[:len(seg)], seg, color='#888', lw=0.5, alpha=0.18, zorder=2)

        # Bold mean waveform
        if segs:
            mean_seg = np.mean(np.vstack(segs), axis=0)
            ax.plot(t_ms[:len(mean_seg)], mean_seg, color='black', lw=2.0, zorder=4)

        # MRA annotation: hline at MRA level + text label (matching HRS2 style)
        trans  = ax.get_xaxis_transform()
        m_mid  = (m_start_ms + m_end_ms) / 2
        h_mid  = (h_start_ms + h_end_ms) / 2
        if m_mra != 0:
            ax.hlines(m_mra, m_start_ms, m_end_ms,
                      colors='blue', lw=1.5, ls=':', zorder=5)
            ax.text(m_mid, 0.91, f'M: {m_mra:.1f} µV',
                    transform=trans, ha='center', va='bottom', fontsize=7,
                    color='blue',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))
        if h_mra != 0:
            ax.hlines(h_mra, h_start_ms, h_end_ms,
                      colors='darkgreen', lw=1.5, ls=':', zorder=5)
            ax.text(h_mid, 0.83, f'H: {h_mra:.1f} µV',
                    transform=trans, ha='center', va='bottom', fontsize=7,
                    color='darkgreen',
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))

        ax.set_title(f'Pulse {k + 1} / {n_pulses}', fontsize=9,
                     color=color, fontweight='bold')
        ax.set_xlabel('ms', fontsize=8)
        ax.set_ylabel('µV', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', alpha=0.2, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused tiles
    for idx in range(len(pulse_data), ncols * nrows):
        row, col = divmod(idx, ncols)
        fig.add_subplot(gs[row + 1, col]).set_visible(False)

    try:
        amp_str = f'{float(getattr(trials[0], "stimulation_amplitude_ma", 0.0)):.3f} mA'
    except (TypeError, ValueError):
        amp_str = '? mA'
    fig.suptitle(
        f'Averaged Pulse Waveforms  ·  {hz} Hz  ·  {amp_str}  ·  '
        f'n={len(trials)} trials  ·  Page {page + 1} / {total_pages}',
        fontsize=11, y=0.99)
    plt.show()

    return total_pages


def plot_ft_peak_curve(trials, header, sample_rate=None,
                        m_start_ms=2.0, m_end_ms=4.0,
                        h_start_ms=6.0, h_end_ms=10.0,
                        title_suffix=''):
    """Peak |EMG| within H/M windows per pulse position, averaged across all FT trials.

    Complements plot_ft_depression_curve (which uses pre-stored MRA values).
    Peak = max(|EMG|) extracted directly from trial_data within each wave window.
    """
    import matplotlib.pyplot as plt

    sr       = sample_rate or getattr(header, 'sample_rate', SAMPLE_RATE)
    n_pulses = getattr(header, 'n_pulses_per_train', 0)
    if n_pulses == 0:
        n_pulses = max((len(getattr(t, 'pulse_h_wave_mra', [])) for t in trials), default=0)
    if n_pulses == 0:
        print("No pulse data available.")
        return

    period_us = getattr(header, 'event_period_us', 0)
    if period_us == 0:
        print("event_period_us not set.")
        return

    pulse_period_samples = round(period_us / 1e6 * sr)
    h_peaks_all, m_peaks_all = [], []

    for t in trials:
        onset = getattr(t, 'onset_sample_index', -1)
        if onset < 0:
            continue
        emg_abs = np.abs(np.array(t.trial_data, dtype=float))
        h_row, m_row = [], []
        for k in range(n_pulses):
            onset_k = onset + k * pulse_period_samples
            h_s = onset_k + round(h_start_ms * sr / 1000)
            h_e = onset_k + round(h_end_ms   * sr / 1000)
            m_s = onset_k + round(m_start_ms * sr / 1000)
            m_e = onset_k + round(m_end_ms   * sr / 1000)
            h_seg = emg_abs[h_s:h_e]
            m_seg = emg_abs[m_s:m_e]
            h_row.append(float(np.max(h_seg)) if len(h_seg) > 0 else 0.0)
            m_row.append(float(np.max(m_seg)) if len(m_seg) > 0 else 0.0)
        h_peaks_all.append(h_row)
        m_peaks_all.append(m_row)

    if not h_peaks_all:
        print("No valid trial data for peak calculation.")
        return

    h_mat     = np.array(h_peaks_all)
    m_mat     = np.array(m_peaks_all)
    pulse_idx = np.arange(1, n_pulses + 1)
    hz = round(1e6 / period_us, 1)
    try:
        amp_str = f'{float(getattr(trials[0], "stimulation_amplitude_ma", 0.0)):.3f} mA'
    except (TypeError, ValueError):
        amp_str = '? mA'

    fig, ax = plt.subplots(figsize=(max(8, n_pulses * 0.85), 5))

    h_mean, h_std = h_mat.mean(axis=0), h_mat.std(axis=0)
    m_mean, m_std = m_mat.mean(axis=0), m_mat.std(axis=0)

    ax.plot(pulse_idx, h_mean, 'o-', color='royalblue', lw=2, ms=7,
            label=f'H-wave peak (n={len(h_peaks_all)})', zorder=4)
    ax.fill_between(pulse_idx, h_mean - h_std, h_mean + h_std,
                    color='royalblue', alpha=0.18, zorder=3)
    ax.plot(pulse_idx, h_mat[-1], 'o--', color='royalblue', lw=1.2, ms=4,
            alpha=0.55, label='H-wave peak (last trial)', zorder=3)

    ax.plot(pulse_idx, m_mean, 's-', color='firebrick', lw=2, ms=7,
            label=f'M-wave peak (n={len(m_peaks_all)})', zorder=4)
    ax.fill_between(pulse_idx, m_mean - m_std, m_mean + m_std,
                    color='firebrick', alpha=0.18, zorder=3)
    ax.plot(pulse_idx, m_mat[-1], 's--', color='firebrick', lw=1.2, ms=4,
            alpha=0.55, label='M-wave peak (last trial)', zorder=3)

    ax.set_xlabel('Pulse # in train', fontsize=11)
    ax.set_ylabel('Peak |EMG|  (µV)', fontsize=11)
    ax.set_xticks(pulse_idx)
    title = f'H/M Peak Per Pulse  ·  {hz} Hz  ·  {amp_str}  ·  n={len(trials)} trials'
    if title_suffix:
        title += f'  ·  {title_suffix}'
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_ft_background_emg(trials, header):
    """Histogram of pre-trial background EMG mean across all FT trials."""
    import matplotlib.pyplot as plt

    bg_vals = [v for v in
               (getattr(t, 'background_emg_mean', 0.0) for t in trials) if v > 0]
    if not bg_vals:
        print("No background EMG data available.")
        return

    bg        = np.array(bg_vals, dtype=float)
    q1, med, q3 = np.percentile(bg, [25, 50, 75])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bg, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(q1,  color='orange', lw=1.5, ls='--', label=f'Q1 = {q1:.2f} µV')
    ax.axvline(med, color='crimson', lw=2.0, ls='-',  label=f'Median = {med:.2f} µV')
    ax.axvline(q3,  color='orange', lw=1.5, ls='--', label=f'Q3 = {q3:.2f} µV')
    ax.set_xlabel('Background EMG Mean (µV)', fontsize=11)
    ax.set_ylabel('Trial count', fontsize=11)
    ax.set_title(
        f'Background EMG Distribution  ·  {len(bg_vals)} trials  '
        f'[median = {med:.2f}, IQR = {q1:.2f}–{q3:.2f} µV]',
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_amplitude_distribution(trials, header):
    """Histogram of stim amplitudes used across an HRS2 session."""
    import matplotlib.pyplot as plt
    if not trials:
        print("No trials to plot.")
        return

    amp_counts: dict = defaultdict(int)
    for t in trials:
        amp_counts[round(t.stimulation_amplitude_ma, 4)] += 1

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


def compute_snr_analysis(trials, header,
                         m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                         h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                         bg_pre_ms: float = 15.0,
                         sample_rate: float = SAMPLE_RATE):
    """Compute and plot signal-to-noise ratio for M-wave, H-wave, and H:M ratio.

    For each trial:
      rms_m   = RMS of trial_data within [m_start_ms, m_end_ms]
      rms_h   = RMS of trial_data within [h_start_ms, h_end_ms]
      hm      = rms_h / rms_m
      rms_bg  = RMS of trial_data within [-bg_pre_ms, 0)  (pre-stim background)

    SNR values (dimensionless):
      snr_m  = rms_m  / rms_bg
      snr_h  = rms_h  / rms_bg
      snr_hm = hm     / rms_bg

    Prints a per-group summary table and produces a 3-panel histogram figure.
    Trials are grouped by stimulation amplitude; overall (all-trial) stats are
    always printed and plotted.
    """
    import matplotlib.pyplot as plt

    if not trials:
        print("No trials.")
        return

    ms_ps = 1000.0 / sample_rate
    bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
    rec_s = int(TRIAL_RECORD_MS  * sample_rate / 1000)
    pre_window = max(bg_pre_ms + 2.0, 5.0)   # load enough pre-stim data

    snr_m_all, snr_h_all, snr_hm_all = [], [], []
    amp_labels = []

    for trial in trials:
        t_ms, emg, _, _, _ = get_trial_window(
            trial, pre_window, max(h_end_ms + 2.0, 20.0),
            ms_per_sample=ms_ps, bin_samples=bin_s, record_samples=rec_s)

        m_mask  = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
        h_mask  = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
        bg_mask = (t_ms >= -bg_pre_ms) & (t_ms < 0)

        def _rms(sig, mask):
            seg = sig[mask]
            return float(np.sqrt(np.mean(seg ** 2))) if mask.any() and len(seg) > 0 else np.nan

        rms_m  = _rms(emg, m_mask)
        rms_h  = _rms(emg, h_mask)
        rms_bg = _rms(emg, bg_mask)

        if rms_bg > 0 and np.isfinite(rms_bg):
            snr_m_all.append(rms_m  / rms_bg if np.isfinite(rms_m)  else np.nan)
            snr_h_all.append(rms_h  / rms_bg if np.isfinite(rms_h)  else np.nan)
            hm = (rms_h / rms_m) if (np.isfinite(rms_m) and rms_m > 0
                                      and np.isfinite(rms_h)) else np.nan
            snr_hm_all.append(hm / rms_bg if np.isfinite(hm) else np.nan)
        else:
            snr_m_all.append(np.nan)
            snr_h_all.append(np.nan)
            snr_hm_all.append(np.nan)
        amp_labels.append(round(trial.stimulation_amplitude_ma, 4))

    snr_m_all  = np.array(snr_m_all,  dtype=float)
    snr_h_all  = np.array(snr_h_all,  dtype=float)
    snr_hm_all = np.array(snr_hm_all, dtype=float)

    def _stats(arr):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, cv=np.nan)
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return dict(n=len(v), mean=mu, median=float(np.median(v)),
                    std=sd, cv=sd / mu if mu != 0 else np.nan)

    sid = getattr(header, 'subject_id', '')
    print(f"SNR Analysis — {sid}")
    print(f"  M window : {m_start_ms}–{m_end_ms} ms")
    print(f"  H window : {h_start_ms}–{h_end_ms} ms")
    print(f"  BG window: -{bg_pre_ms:.0f}–0 ms (pre-stim)")
    print()
    hdr = f"{'Metric':<18} {'n':>5} {'Mean':>9} {'Median':>9} {'SD':>9} {'CV':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, arr in [("SNR_M  (M/BG)",    snr_m_all),
                       ("SNR_H  (H/BG)",    snr_h_all),
                       ("SNR_HM (H:M/BG)",  snr_hm_all)]:
        s = _stats(arr)
        print(f"  {label:<16} {s['n']:>5}  {s['mean']:>8.3f}  {s['median']:>8.3f}"
              f"  {s['std']:>8.3f}  {s['cv']:>6.3f}")

    # ── figure: 3-panel histogram ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    configs = [
        (snr_m_all,  "SNR_M\n(RMS_M / RMS_BG)",   "steelblue"),
        (snr_h_all,  "SNR_H\n(RMS_H / RMS_BG)",   "darkorange"),
        (snr_hm_all, "SNR_H:M\n(H:M ratio / RMS_BG)", "mediumseagreen"),
    ]
    for ax, (arr, ylabel, color) in zip(axes, configs):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            ax.hist(v, bins=40, color=color, edgecolor='black', alpha=0.75)
            mu, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            ax.axvline(mu, color='red', linestyle='--', linewidth=2,
                       label=f'Mean {mu:.3f}')
            ax.axvline(float(np.median(v)), color='black', linestyle=':', linewidth=1.5,
                       label=f'Median {float(np.median(v)):.3f}')
            ax.legend(fontsize=8)
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_ylabel('Trial count')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f'Signal-to-Noise Ratio — {sid}\n'
        f'M: {m_start_ms}–{m_end_ms} ms  |  H: {h_start_ms}–{h_end_ms} ms  |  '
        f'BG: -{bg_pre_ms:.0f}–0 ms',
        fontsize=11)
    plt.tight_layout()
    plt.show()

    return dict(snr_m=snr_m_all, snr_h=snr_h_all, snr_hm=snr_hm_all,
                amp_labels=amp_labels)


def compute_mra_snr_analysis(trials, header,
                              m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                              h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                              bg_pre_ms: float = 15.0,
                              sample_rate: float = SAMPLE_RATE):
    """Compute and plot SNR using Mean Rectified Average (MRA) instead of RMS.

    For each trial:
      mra_m   = mean(|trial_data|) within [m_start_ms, m_end_ms]
      mra_h   = mean(|trial_data|) within [h_start_ms, h_end_ms]
      hm      = mra_h / mra_m
      mra_bg  = mean(|trial_data|) within [-bg_pre_ms, 0)  (pre-stim background)

    SNR values (dimensionless):
      snr_m  = mra_m  / mra_bg
      snr_h  = mra_h  / mra_bg
      snr_hm = hm     / mra_bg

    Prints a summary table and produces a 3-panel histogram figure.
    """
    import matplotlib.pyplot as plt

    if not trials:
        print("No trials.")
        return

    ms_ps = 1000.0 / sample_rate
    bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
    rec_s = int(TRIAL_RECORD_MS  * sample_rate / 1000)
    pre_window = max(bg_pre_ms + 2.0, 5.0)

    snr_m_all, snr_h_all, snr_hm_all = [], [], []
    amp_labels = []

    for trial in trials:
        t_ms, emg, _, _, _ = get_trial_window(
            trial, pre_window, max(h_end_ms + 2.0, 20.0),
            ms_per_sample=ms_ps, bin_samples=bin_s, record_samples=rec_s)

        m_mask  = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
        h_mask  = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
        bg_mask = (t_ms >= -bg_pre_ms) & (t_ms < 0)

        def _mra(sig, mask):
            seg = sig[mask]
            return float(np.mean(np.abs(seg))) if mask.any() and len(seg) > 0 else np.nan

        mra_m  = _mra(emg, m_mask)
        mra_h  = _mra(emg, h_mask)
        mra_bg = _mra(emg, bg_mask)

        if mra_bg > 0 and np.isfinite(mra_bg):
            snr_m_all.append(mra_m  / mra_bg if np.isfinite(mra_m)  else np.nan)
            snr_h_all.append(mra_h  / mra_bg if np.isfinite(mra_h)  else np.nan)
            hm = (mra_h / mra_m) if (np.isfinite(mra_m) and mra_m > 0
                                      and np.isfinite(mra_h)) else np.nan
            snr_hm_all.append(hm / mra_bg if np.isfinite(hm) else np.nan)
        else:
            snr_m_all.append(np.nan)
            snr_h_all.append(np.nan)
            snr_hm_all.append(np.nan)
        amp_labels.append(round(trial.stimulation_amplitude_ma, 4))

    snr_m_all  = np.array(snr_m_all,  dtype=float)
    snr_h_all  = np.array(snr_h_all,  dtype=float)
    snr_hm_all = np.array(snr_hm_all, dtype=float)

    def _stats(arr):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, cv=np.nan)
        mu = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return dict(n=len(v), mean=mu, median=float(np.median(v)),
                    std=sd, cv=sd / mu if mu != 0 else np.nan)

    sid = getattr(header, 'subject_id', '')
    print(f"MRA-SNR Analysis — {sid}")
    print(f"  M window : {m_start_ms}–{m_end_ms} ms")
    print(f"  H window : {h_start_ms}–{h_end_ms} ms")
    print(f"  BG window: -{bg_pre_ms:.0f}–0 ms (pre-stim)")
    print()
    hdr = f"{'Metric':<20} {'n':>5} {'Mean':>9} {'Median':>9} {'SD':>9} {'CV':>7}"
    print(hdr)
    print("-" * len(hdr))
    for label, arr in [("MRA-SNR_M  (M/BG)",   snr_m_all),
                       ("MRA-SNR_H  (H/BG)",   snr_h_all),
                       ("MRA-SNR_HM (H:M/BG)", snr_hm_all)]:
        s = _stats(arr)
        print(f"  {label:<18} {s['n']:>5}  {s['mean']:>8.3f}  {s['median']:>8.3f}"
              f"  {s['std']:>8.3f}  {s['cv']:>6.3f}")

    # ── figure: 3-panel histogram ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    configs = [
        (snr_m_all,  "MRA-SNR_M\n(MRA_M / MRA_BG)",    "royalblue"),
        (snr_h_all,  "MRA-SNR_H\n(MRA_H / MRA_BG)",    "tomato"),
        (snr_hm_all, "MRA-SNR_H:M\n(H:M ratio / MRA_BG)", "mediumorchid"),
    ]
    for ax, (arr, ylabel, color) in zip(axes, configs):
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            ax.hist(v, bins=40, color=color, edgecolor='black', alpha=0.75)
            mu = float(np.mean(v))
            med = float(np.median(v))
            ax.axvline(mu,  color='red',   linestyle='--', linewidth=2,
                       label=f'Mean {mu:.3f}')
            ax.axvline(med, color='black', linestyle=':',  linewidth=1.5,
                       label=f'Median {med:.3f}')
            ax.legend(fontsize=8)
        ax.set_xlabel(ylabel, fontsize=10)
        ax.set_ylabel('Trial count')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f'MRA Signal-to-Noise Ratio — {sid}\n'
        f'M: {m_start_ms}–{m_end_ms} ms  |  H: {h_start_ms}–{h_end_ms} ms  |  '
        f'BG: -{bg_pre_ms:.0f}–0 ms',
        fontsize=11)
    plt.tight_layout()
    plt.show()

    return dict(snr_m=snr_m_all, snr_h=snr_h_all, snr_hm=snr_hm_all,
                amp_labels=amp_labels)


def plot_background_emg_views(trials, emg_blocks,
                              monitoring_window_ms: float = 2500.0,
                              sample_rate: float = SAMPLE_RATE,
                              bin_duration_ms: float = BIN_DURATION_MS):
    """Interactive 'Most recent background' bar chart + 'Background EMG Level' scatter.

    Mirrors the H-Reflex App recruitment-curve trial plot widgets:
      _update_most_recent_background_plot  (bar chart of pre-stim |EMG| bins)
      _update_emg_level_plot               (scatter of grand means per trial)

    For file_version >= 5 trials the stored background_bins / background_emg_mean
    are used directly (actual values the app evaluated).  For older trials the bins
    are reconstructed from emg_blocks as before.  Pass bin_duration_ms from
    hrs1_header.bin_duration_ms so the window label is accurate.
    """
    import matplotlib.pyplot as plt
    from ipywidgets import IntSlider, Output, VBox, HBox
    from IPython.display import display

    if len(trials) == 0:
        print("No HRS2 trials to plot.")
        return

    bins_per_trial = []
    grand_means    = []
    window_ms_list = []   # actual monitoring window per trial
    sources        = []   # "stored" or "reconstructed"

    for trial in trials:
        stored_bins = getattr(trial, 'background_bins', None)
        stored_gm   = getattr(trial, 'background_emg_mean', None)
        if stored_bins is not None and len(stored_bins) > 0:
            bins_per_trial.append(stored_bins)
            grand_means.append(float(stored_gm) if stored_gm is not None else float(np.mean(stored_bins)))
            window_ms_list.append(len(stored_bins) * bin_duration_ms)
            sources.append("stored")
        else:
            bins, gm = compute_background_bins(
                trial, emg_blocks, monitoring_window_ms=monitoring_window_ms,
                sample_rate=sample_rate)
            bins_per_trial.append(bins)
            grand_means.append(gm)
            window_ms_list.append(monitoring_window_ms if bins is not None else float('nan'))
            sources.append("reconstructed")

    n_stored = sources.count("stored")
    n_recon  = sources.count("reconstructed")
    valid_idx = [i for i, gm in enumerate(grand_means) if not np.isnan(gm)]
    if not valid_idx:
        print("Could not obtain background bins for any trial.")
        return

    print(f"{n_stored} trials used stored bins  |  {n_recon} trials reconstructed from emg_blocks.")

    # Pre-compute box-plot stats for every valid trial (done once, not per draw).
    # Whiskers = min/max bin value; box = Q1–Q3; median line = median bin value.
    bxp_stats = {}
    for i in valid_idx:
        b = bins_per_trial[i]
        if b is not None and len(b) > 0:
            bxp_stats[i] = {
                'med':    float(np.median(b)),
                'q1':     float(np.percentile(b, 25)),
                'q3':     float(np.percentile(b, 75)),
                'whislo': float(np.min(b)),
                'whishi': float(np.max(b)),
                'fliers': [],
            }

    # Scale box width so boxes don't overlap for large trial counts.
    box_w = min(0.8, max(0.15, 200.0 / max(1, len(valid_idx))))

    out = Output()
    slider = IntSlider(value=valid_idx[-1], min=0, max=len(trials) - 1, step=1,
                       description='Trial:', layout={'width': '600px'},
                       continuous_update=False)

    def _draw(idx):
        with out:
            out.clear_output(wait=True)
            fig, (ax_bg, ax_lvl) = plt.subplots(1, 2, figsize=(15, 5))

            bins   = bins_per_trial[idx]
            win_ms = window_ms_list[idx]
            src    = sources[idx]
            if bins is None:
                ax_bg.text(0.5, 0.5, f'Trial {idx}: no background data available',
                           ha='center', va='center', transform=ax_bg.transAxes)
            else:
                gm = grand_means[idx]
                x  = np.arange(len(bins))
                ax_bg.bar(x, bins, width=0.8, color=(70/255, 130/255, 180/255))
                ax_bg.axhline(gm, color='red', linestyle='--', linewidth=2,
                              label=f'Mean = {gm:.2f} µV')
                ax_bg.legend(loc='upper right')
                win_lbl = f'{win_ms:.0f} ms' if not np.isnan(win_ms) else '? ms'
                ax_bg.set_title(f'Most Recent Background  (Trial {idx},'
                                f' window={win_lbl}, {src})')
            ax_bg.set_xlabel(f'Bin #  ({bin_duration_ms:.0f} ms each)')
            ax_bg.set_ylabel('EMG (µV)')
            ax_bg.grid(True, alpha=0.3, axis='y')

            # --- Background EMG Level: box-and-whisker per trial ---
            # Draw all non-selected trials in blue.
            bg_stats = [bxp_stats[i] for i in valid_idx if i != idx and i in bxp_stats]
            bg_pos   = [i               for i in valid_idx if i != idx and i in bxp_stats]
            if bg_stats:
                ax_lvl.bxp(bg_stats, positions=bg_pos, widths=box_w,
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=(70/255, 130/255, 180/255, 0.5),
                                         edgecolor=(0, 0, 200/255), linewidth=0.6),
                           whiskerprops=dict(color=(0, 0, 200/255), linewidth=0.8),
                           capprops=dict(color=(0, 0, 200/255), linewidth=0.8),
                           medianprops=dict(color='red', linewidth=1.0))

            # Draw selected trial highlighted in gold.
            if idx in bxp_stats and not np.isnan(grand_means[idx]):
                ax_lvl.bxp([bxp_stats[idx]], positions=[idx], widths=box_w * 1.4,
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=(255/255, 215/255, 0, 0.6),
                                         edgecolor='goldenrod', linewidth=2.0),
                           whiskerprops=dict(color='goldenrod', linewidth=2.0),
                           capprops=dict(color='goldenrod', linewidth=2.0),
                           medianprops=dict(color='darkred', linewidth=2.0))
                # Proxy artist for legend entry
                ax_lvl.plot([], [], color='goldenrod', linewidth=2,
                            label=f'Selected (Trial {idx})')

            # Per-trial threshold markers: short horizontal segments at each trial's x-pos.
            # Thresholds vary per trial (stored in trial.min/max_initiation_threshold),
            # so we draw one segment per trial rather than full-width axhlines.
            _vi_arr   = np.array(valid_idx, dtype=float)
            _min_ths  = np.array([trials[i].min_initiation_threshold for i in valid_idx])
            _max_ths  = np.array([trials[i].max_initiation_threshold for i in valid_idx])
            _hw       = box_w * 0.65   # half-width of each threshold segment

            # Non-selected trials: thin, faded segments
            _others = _vi_arr != idx
            if np.any(_others):
                _xo = _vi_arr[_others]
                ax_lvl.hlines(_min_ths[_others], _xo - _hw, _xo + _hw,
                              colors=(0, 160/255, 0), linewidth=0.9,
                              linestyle='--', alpha=0.95)
                ax_lvl.hlines(_max_ths[_others], _xo - _hw, _xo + _hw,
                              colors=(200/255, 0, 0), linewidth=0.9,
                              linestyle='--', alpha=0.95)

            # Selected trial: wider, fully opaque, with legend labels
            tr = trials[idx]
            _sel_hw = _hw * 1.35
            ax_lvl.hlines(tr.min_initiation_threshold, idx - _sel_hw, idx + _sel_hw,
                          colors=(0, 160/255, 0), linewidth=2.0, linestyle='--',
                          label=f'Min thresh: {tr.min_initiation_threshold:.1f} µV')
            ax_lvl.hlines(tr.max_initiation_threshold, idx - _sel_hw, idx + _sel_hw,
                          colors=(200/255, 0, 0), linewidth=2.0, linestyle='--',
                          label=f'Max thresh: {tr.max_initiation_threshold:.1f} µV')
            ax_lvl.set_xlabel('Trial #')
            ax_lvl.set_ylabel('EMG (µV)')
            ax_lvl.set_title('Background EMG Level  '
                             '(box=Q1–Q3, whiskers=min/max, '
                             '── per-trial init. thresholds)')
            ax_lvl.legend(loc='upper right', fontsize=9)
            ax_lvl.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    slider.observe(lambda c: _draw(c['new']) if c['name'] == 'value' else None,
                   names='value')
    display(VBox([HBox([slider]), out]))
    _draw(slider.value)


def plot_actual_trial_timeline(trials, header=None):
    """Plot the actual HRS2 trial timeline and ITI distribution, and print trial rate.

    Panel 1: Trial number vs time (hours from session start).
    Panel 2: Inter-trial interval (ITI) distribution histogram.

    Timestamps come from trigger_wall_time_ms (file_version >= 2) when non-zero,
    otherwise from the start_time datetime field on each trial.
    """
    import matplotlib.pyplot as plt

    if not trials:
        print("No HRS2 trials to plot.")
        return

    # --- extract times as seconds from the first trial ---
    def _trial_sec(t):
        twms = getattr(t, 'trigger_wall_time_ms', 0)
        if twms and twms > 0:
            return twms / 1000.0
        if t.start_time is not None:
            return t.start_time.timestamp()
        return None

    raw_secs = [_trial_sec(t) for t in trials]
    if all(s is None for s in raw_secs):
        print("No usable timestamps found on trials.")
        return

    # normalise to zero at the first trial
    valid_secs = [s for s in raw_secs if s is not None]
    t0 = min(valid_secs)
    rel_secs  = [s - t0 if s is not None else float('nan') for s in raw_secs]
    rel_hours = [s / 3600.0 for s in rel_secs]

    trial_nums = list(range(1, len(trials) + 1))

    # ITI in seconds between consecutive valid trials
    valid_pairs = [(rel_secs[i], rel_secs[i - 1])
                   for i in range(1, len(rel_secs))
                   if not (np.isnan(rel_secs[i]) or np.isnan(rel_secs[i - 1]))]
    iti_s  = [a - b for a, b in valid_pairs]
    iti_ms = [x * 1000.0 for x in iti_s]

    # --- recording duration and trial rate ---
    duration_s = max(valid_secs) - min(valid_secs)
    duration_h = duration_s / 3600.0
    n_trials   = len([s for s in rel_secs if not np.isnan(s)])
    rate_per_h = n_trials / duration_h if duration_h > 0 else float('nan')

    print(f"Actual trial timeline ({n_trials} trials)")
    print(f"  Recording span : {duration_s:.1f} s  ({duration_h:.3f} h)")
    print(f"  Trial rate     : {rate_per_h:.1f} trials / hour")
    if iti_ms:
        print(f"  ITI — mean: {np.mean(iti_ms):.0f} ms  |  "
              f"median: {np.median(iti_ms):.0f} ms  |  "
              f"min: {np.min(iti_ms):.0f} ms  |  "
              f"max: {np.max(iti_ms):.0f} ms")

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: timeline
    ax1.plot(rel_hours, trial_nums, 'o-', markersize=4, linewidth=1, color='steelblue')
    ax1.set_xlabel('Time from first trial (h)')
    ax1.set_ylabel('Trial number')
    title1 = 'Trial Timeline — Actual Initiation Times'
    if header is not None:
        sid = getattr(header, 'subject_id', '')
        if sid:
            title1 += f'\n{sid}'
    ax1.set_title(title1)
    ax1.grid(True, alpha=0.3)

    # Panel 2: ITI distribution (log-scale x-axis)
    if iti_ms:
        _iti_arr = np.array(iti_ms)
        _iti_pos = _iti_arr[_iti_arr > 0]
        if len(_iti_pos) > 0:
            _bins = np.logspace(np.log10(_iti_pos.min()),
                                np.log10(_iti_pos.max()), 41)
            ax2.hist(_iti_pos, bins=_bins, color='steelblue', edgecolor='black', alpha=0.75)
        ax2.axvline(np.mean(iti_ms), color='red', linestyle='--', linewidth=2,
                    label=f'Mean {np.mean(iti_ms):.0f} ms')
        ax2.axvline(np.median(iti_ms), color='orange', linestyle='--', linewidth=2,
                    label=f'Median {np.median(iti_ms):.0f} ms')
        ax2.set_xscale('log')
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Insufficient trials for ITI', ha='center',
                 va='center', transform=ax2.transAxes)
    ax2.set_xlabel('Inter-Trial Interval (ms, log scale)')
    ax2.set_ylabel('Count')
    ax2.set_title('Inter-Trial Interval Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Actual HRS2 Trials  |  Rate: {rate_per_h:.1f} trials/h  '
                 f'|  Duration: {duration_h:.3f} h  ({n_trials} trials)',
                 fontsize=11)
    plt.tight_layout()
    plt.show()


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


def plot_hwave_regression(trials, emg_blocks=None,
                          m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                          h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                          sample_rate: float = SAMPLE_RATE,
                          pre_ms: float = 2.0, post_ms: float = 15.0,
                          monitoring_window_ms: float = 2500.0,
                          bin_duration_ms: float = BIN_DURATION_MS,
                          title_suffix: str = ''):
    """Two-panel H-wave regression plot for a trial subset.

    Panel 1 — scatter + OLS regression: H-wave MRA (per trial) vs pre-stim
              background EMG grand mean.
    Panel 2 — scatter + OLS regression: H-wave MRA vs M-wave MRA (per trial).
    """
    from scipy import stats as _stats
    import matplotlib.pyplot as plt

    if not trials:
        print("No trials for regression.")
        return

    _ms_ps = 1000.0 / sample_rate
    _bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
    _rec_s = int(TRIAL_RECORD_MS  * sample_rate / 1000)

    # ── per-trial H-wave MRA ──────────────────────────────────────────────
    h_mra = []
    for _tr in trials:
        _t, _emg, _, _, _ = get_trial_window(
            _tr, pre_ms, post_ms,
            ms_per_sample=_ms_ps, bin_samples=_bin_s, record_samples=_rec_s)
        _hm = (_t >= h_start_ms) & (_t <= h_end_ms)
        h_mra.append(float(np.nanmean(np.abs(_emg[_hm]))) if _hm.any() else float('nan'))

    # ── per-trial M-wave MRA ──────────────────────────────────────────────
    m_mra = []
    for _tr in trials:
        _t, _emg, _, _, _ = get_trial_window(
            _tr, pre_ms, post_ms,
            ms_per_sample=_ms_ps, bin_samples=_bin_s, record_samples=_rec_s)
        _mm = (_t >= m_start_ms) & (_t <= m_end_ms)
        m_mra.append(float(np.nanmean(np.abs(_emg[_mm]))) if _mm.any() else float('nan'))

    # ── per-trial background bins and grand mean ──────────────────────────
    bins_list = []
    bg_means  = []
    for _tr in trials:
        _sb = getattr(_tr, 'background_bins', None)
        _gm = getattr(_tr, 'background_emg_mean', None)
        if _sb is not None and len(_sb) > 0:
            _b  = np.asarray(_sb, dtype=float)
            _gm = float(_gm) if _gm is not None else float(np.mean(_b))
        elif emg_blocks is not None:
            _b, _gm = compute_background_bins(
                _tr, emg_blocks,
                monitoring_window_ms=monitoring_window_ms,
                sample_rate=sample_rate)
        else:
            _b, _gm = None, float('nan')
        bins_list.append(_b)
        bg_means.append(_gm if _gm is not None else float('nan'))

    # ── filter to valid trials ────────────────────────────────────────────
    _valid = [(i, h_mra[i], bg_means[i])
              for i in range(len(trials))
              if not np.isnan(h_mra[i]) and not np.isnan(bg_means[i])]
    if len(_valid) < 3:
        print(f"Not enough valid trials for regression (need ≥ 3, got {len(_valid)}).")
        return

    h_arr  = np.array([v[1] for v in _valid])
    bg_arr = np.array([v[2] for v in _valid])

    # ── Panel 1: scatter + linear regression ─────────────────────────────
    _sl, _ic, _r, _p, _ = _stats.linregress(bg_arr, h_arr)
    _r2    = _r ** 2
    _xfit  = np.linspace(float(bg_arr.min()), float(bg_arr.max()), 200)
    _yfit  = _sl * _xfit + _ic

    # ── Panel 2: H-wave MRA vs M-wave MRA linear regression ──────────────
    _valid_mh = [(m_mra[i], h_mra[i])
                 for i in range(len(trials))
                 if not np.isnan(h_mra[i]) and not np.isnan(m_mra[i])]

    # ── figure ────────────────────────────────────────────────────────────
    _sfx = f'  —  {title_suffix}' if title_suffix else ''
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 5))

    _ax1.scatter(bg_arr, h_arr, color='steelblue', alpha=0.75, s=55, zorder=3,
                 label=f'n = {len(_valid)} trials')
    _ax1.plot(_xfit, _yfit, color='crimson', linewidth=2.0,
              label=f'y = {_sl:.2f}x + {_ic:.2f}\nR² = {_r2:.3f},  p = {_p:.3g}')
    _ax1.set_xlabel('Pre-stim background EMG (µV)', fontsize=12)
    _ax1.set_ylabel('H-wave MRA (µV)', fontsize=12)
    _ax1.set_title(f'H-wave vs Background EMG{_sfx}', fontsize=13)
    _ax1.legend(fontsize=10)
    _ax1.grid(True, alpha=0.3)

    if len(_valid_mh) >= 3:
        _m_arr  = np.array([v[0] for v in _valid_mh])
        _h_arr2 = np.array([v[1] for v in _valid_mh])
        _sl2, _ic2, _r2v, _p2, _ = _stats.linregress(_m_arr, _h_arr2)
        _r2_mh  = _r2v ** 2
        _xfit2  = np.linspace(float(_m_arr.min()), float(_m_arr.max()), 200)
        _yfit2  = _sl2 * _xfit2 + _ic2
        _ax2.scatter(_m_arr, _h_arr2, color='steelblue', alpha=0.75, s=55, zorder=3,
                     label=f'n = {len(_valid_mh)} trials')
        _ax2.plot(_xfit2, _yfit2, color='crimson', linewidth=2.0,
                  label=f'y = {_sl2:.2f}x + {_ic2:.2f}\nR² = {_r2_mh:.3f},  p = {_p2:.3g}')
        _ax2.set_xlabel('M-wave MRA (µV)', fontsize=12)
        _ax2.set_ylabel('H-wave MRA (µV)', fontsize=12)
        _ax2.set_title(f'H-wave vs M-wave MRA{_sfx}', fontsize=13)
        _ax2.legend(fontsize=10)
        _ax2.grid(True, alpha=0.3)
    else:
        _ax2.text(0.5, 0.5, 'Not enough valid trials for\nH vs M regression',
                  ha='center', va='center', transform=_ax2.transAxes, fontsize=11)
        _ax2.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"  Regression (n={len(_valid)}):  "
          f"slope={_sl:.4f}  intercept={_ic:.2f}  R²={_r2:.4f}  p={_p:.4g}")


def plot_hrs2_analysis(trials, header,
                       pre_avg_ms: float = 2.0, post_avg_ms: float = 15.0,
                       n_per_page: int = 6,
                       m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                       h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                       sample_rate: float = SAMPLE_RATE,
                       fig_width: float = 15.0, fig_height: float = 7.0,
                       emg_blocks=None, monitoring_window_ms: float = 2500.0):
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

    _ms_ps  = 1000.0 / sample_rate
    _bin_s  = int(BIN_DURATION_MS * sample_rate / 1000)
    _rec_s  = int(TRIAL_RECORD_MS  * sample_rate / 1000)

    # ── polarity detection ──────────────────────────────────────────────────
    _pol_split: dict = {}
    for _tr in trials:
        _p = getattr(_tr, 'stim_polarity_reversed', 0)
        _pol_split.setdefault(_p, []).append(_tr)
    _dual_pol   = len(_pol_split) > 1
    _pol_keys   = sorted(_pol_split.keys())
    _pol_labels = {0: 'Normal (0)', 1: 'Reversed (1)'}
    _active_pol = {'val': _pol_keys[0]}
    _pol_change_hooks = []

    def _pad_rows(rows, n_pts):
        p = np.full((len(rows), n_pts), np.nan)
        for k, a in enumerate(rows):
            if a is not None and len(a) > 0:
                _n = min(len(a), n_pts)
                p[k, :_n] = np.asarray(a[:_n], dtype=float)
        return p

    def _build_amp_data(trial_list):
        _groups = defaultdict(list)
        _trial_groups = defaultdict(list)
        for _trial in trial_list:
            _key = round(_trial.stimulation_amplitude_ma, 4)
            _trial_groups[_key].append(_trial)
            _t_win, _bip_win, _adc_win, _stim_end, _stim_adc_win = get_trial_window(
                _trial, pre_avg_ms, post_avg_ms, ms_per_sample=_ms_ps,
                bin_samples=_bin_s, record_samples=_rec_s)
            _t_uni, _uni_win, _, _, _ = get_trial_window(
                _trial, pre_avg_ms, post_avg_ms, ms_per_sample=_ms_ps,
                bin_samples=_bin_s, record_samples=_rec_s, use_unipolar=True)
            _groups[_key].append((_t_win, _bip_win, _adc_win, _uni_win, _stim_end, _stim_adc_win))

        _adata = []
        for _amp in sorted(_groups.keys()):
            _wins  = _groups[_amp]
            _t_ref = _wins[0][0]
            _np    = len(_t_ref)
            _pb   = _pad_rows([w[1] for w in _wins], _np)
            _pa   = _pad_rows([w[2] for w in _wins], _np)
            _pu   = _pad_rows([w[3] for w in _wins], _np)
            _psa  = _pad_rows([w[5] for w in _wins], _np)
            _pab  = np.abs(_pb)
            _pau  = np.abs(_pu)
            _avg_b   = np.nanmean(_pb,  axis=0)
            _avg_a   = np.nanmean(_pa,  axis=0)
            _avg_u   = np.nanmean(_pu,  axis=0)
            _avg_sa  = np.nanmean(_psa, axis=0)
            _avg_ab  = np.abs(_avg_b)
            _avg_au  = np.abs(_avg_u)
            _se  = [w[4] for w in _wins if w[4] is not None]
            _mse = float(np.mean(_se)) if _se else 0.5
            _mm  = (_t_ref >= m_start_ms) & (_t_ref <= m_end_ms)
            _hm  = (_t_ref >= h_start_ms) & (_t_ref <= h_end_ms)
            _m_t   = float((m_start_ms + m_end_ms) / 2)
            _m_a   = float(np.mean(_avg_ab[_mm])) if _mm.any() else float('nan')
            _m_ci  = int(len(_t_ref[_mm]) // 2) if _mm.any() else 0
            _m_bip = float(_avg_b[_mm][_m_ci]) if _mm.any() else float('nan')
            _h_t   = float((h_start_ms + h_end_ms) / 2)
            _h_a   = float(np.mean(_avg_ab[_hm])) if _hm.any() else float('nan')
            _h_ci  = int(len(_t_ref[_hm]) // 2) if _hm.any() else 0
            _h_bip = float(_avg_b[_hm][_h_ci]) if _hm.any() else float('nan')
            _trs = _trial_groups[_amp]
            _bg_mean, _bg_lo, _bg_hi = get_group_bg_stats(_trs)
            _first_tr = _trs[0] if _trs else None
            _is_mg    = getattr(_first_tr, '_is_merged', False) if _first_tr else False
            _amp_lo_v = getattr(_first_tr, '_merged_amp_lo', _amp) if _is_mg else _amp
            _amp_hi_v = getattr(_first_tr, '_merged_amp_hi', _amp) if _is_mg else _amp
            _adata.append({
                'amp': _amp, 't_ref': _t_ref, 'n': len(_wins),
                'mean_stim_end': _mse,
                'padded_bip': _pb,   'avg_bip': _avg_b,
                'padded_adc': _pa,   'avg_adc': _avg_a,
                'padded_uni': _pu,   'avg_uni': _avg_u,
                'padded_stim_adc': _psa, 'avg_stim_adc': _avg_sa,
                'padded_abs_bip': _pab, 'avg_abs_bip': _avg_ab,
                'padded_abs_uni': _pau, 'avg_abs_uni': _avg_au,
                'm_peak_time': _m_t, 'm_peak_amp': _m_a, 'm_peak_bip': _m_bip,
                'h_peak_time': _h_t, 'h_peak_amp': _h_a, 'h_peak_bip': _h_bip,
                'trials': _trs,
                'bg_mean': _bg_mean, 'bg_lo': _bg_lo, 'bg_hi': _bg_hi,
                'is_merged': _is_mg, 'amp_lo': _amp_lo_v, 'amp_hi': _amp_hi_v,
            })
        return _adata

    # mutable view state — rebuilt when polarity toggle changes
    _vst = {'amp_data': _build_amp_data(_pol_split[_active_pol['val']]), 'pages': []}
    _vst['pages'] = [_vst['amp_data'][i:i+n_per_page]
                     for i in range(0, len(_vst['amp_data']), n_per_page)]

    _show_sigs = {'val': set()}
    _ylim_auto = {'val': True}
    _ylim_man  = {'lo': -1000.0, 'hi': 1500.0}
    _figsize   = {'w': fig_width, 'h': fig_height}

    def _get_ylim():
        if _ylim_auto['val']:
            _arrays = [d['padded_bip'] for d in _vst['amp_data']]
            for _sig in ('abs_bip', 'uni', 'abs_uni'):
                if _sig in _show_sigs['val']:
                    _arrays += [d[f'padded_{_sig}'] for d in _vst['amp_data']]
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
        sigs = _show_sigs['val']

        ax.axhline(0, color='black', linewidth=2.0, linestyle='-', alpha=1.0, zorder=3)

        for _row in d['padded_bip']:
            ax.plot(t, _row, color='red', alpha=0.15, linewidth=lw * 0.5)
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

        _ax2 = None
        if 'adc' in sigs or 'stim_adc' in sigs:
            _ax2 = ax.twinx()
            if 'adc' in sigs:
                _adc_avg    = d['avg_adc']
                _adc_center = float(np.nanmean(_adc_avg))
                for _row in d['padded_adc']:
                    _ax2.plot(t, _row - _adc_center, color='green', alpha=0.25, linewidth=lw * 0.4)
                _ax2.plot(t, _adc_avg - _adc_center, color='green', linewidth=lw * 1.8,
                          label='ADC sync (TTL)')
            if 'stim_adc' in sigs:
                _sa_avg = d['avg_stim_adc']
                _sa_center = float(np.nanmean(_sa_avg))
                for _row in d['padded_stim_adc']:
                    _ax2.plot(t, _row - _sa_center, color='magenta', alpha=0.25, linewidth=lw * 0.4)
                _ax2.plot(t, _sa_avg - _sa_center, color='magenta', linewidth=lw * 1.8,
                          label='Stim ADC')
            _ax2.set_ylabel('ADC (V)', fontsize=fsz - 1)
            _ax2.tick_params(axis='y', labelsize=fsz - 2)

        ax.axvspan(0, end_ms, color='red', alpha=0.20)
        ax.axvline(0,      color='red', linestyle='--', linewidth=lw)
        ax.axvline(end_ms, color='red', linestyle='--', linewidth=lw)

        ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.20, zorder=2)
        ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.20, zorder=2)
        ax.axvline(m_start_ms, color='blue',  linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
        ax.axvline(m_end_ms,   color='blue',  linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
        ax.axvline(h_start_ms, color='green', linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
        ax.axvline(h_end_ms,   color='green', linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)

        m_a = abs(d['m_peak_amp'])
        h_a = abs(d['h_peak_amp'])
        if not np.isnan(m_a):
            ax.hlines(m_a, m_start_ms, m_end_ms, colors='blue', linestyles='dotted',
                      linewidth=lw * 2.5, zorder=5, label=f'M-MRA: {m_a:.1f} uV')
            ax.text((m_start_ms + m_end_ms) / 2, 0.93, f'M: {m_a:.1f} uV',
                    transform=ax.get_xaxis_transform(),
                    color='blue', fontsize=fsz - 1, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='blue', alpha=0.85),
                    zorder=8)
        if not np.isnan(h_a):
            ax.hlines(h_a, h_start_ms, h_end_ms, colors='green', linestyles='dotted',
                      linewidth=lw * 2.5, zorder=5, label=f'H-MRA: {h_a:.1f} uV')
            ax.text((h_start_ms + h_end_ms) / 2, 0.93, f'H: {h_a:.1f} uV',
                    transform=ax.get_xaxis_transform(),
                    color='darkgreen', fontsize=fsz - 1, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.85),
                    zorder=8)

        # ── stim pulse peak-to-peak annotation ───────────────────────────
        _sa_avg = d.get('avg_stim_adc')
        if _sa_avg is not None and len(_sa_avg) > 0:
            _stm = (t >= 0) & (t <= end_ms)
            if _stm.sum() >= 2:
                _seg = _sa_avg[_stm]
                _seg = _seg[~np.isnan(_seg)]
                if len(_seg) >= 2:
                    _ptp = float(np.max(_seg) - np.min(_seg))
                    ax.text(0.01, 0.01, f'Stim P2P: {_ptp:.3f} V',
                            transform=ax.transAxes, color='magenta',
                            fontsize=fsz - 1, ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec='magenta', alpha=0.85),
                            zorder=8)

        # ── pre-stim EMG activity annotation ─────────────────────────────
        _pre_mask = t < 0
        if _pre_mask.sum() >= 2:
            _pre_abs = d['padded_abs_bip'][:, _pre_mask]
            _pre_emg = float(np.nanmean(_pre_abs))
            if not np.isnan(_pre_emg):
                ax.text(0.99, 0.01, f'Pre-stim EMG: {_pre_emg:.1f} µV',
                        transform=ax.transAxes, color='dimgray',
                        fontsize=fsz - 1, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='dimgray', alpha=0.85),
                        zorder=8)

        ax.set_xlim(-pre_avg_ms, post_avg_ms)
        ax.set_ylim(_get_ylim())
        if _ax2 is not None:
            _y1_lo, _y1_hi = ax.get_ylim()
            if _y1_hi > _y1_lo:
                _zero_frac = (0.0 - _y1_lo) / (_y1_hi - _y1_lo)
                _y2_lo, _y2_hi = _ax2.get_ylim()
                _y2_span = _y2_hi - _y2_lo
                _ax2.set_ylim(-_zero_frac * _y2_span,
                              (1.0 - _zero_frac) * _y2_span)
        ax.set_xlabel('Time (ms)', fontsize=fsz)
        ax.set_ylabel('EMG (uV)', fontsize=fsz)
        ax.tick_params(labelsize=fsz - 1)
        ax.tick_params(axis='x', width=2.0)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.grid(True, alpha=0.3)
        _min_ms = int(np.floor(t[0]))
        _max_ms = int(np.ceil(t[-1]))
        ax.set_xticks(np.arange(_min_ms, _max_ms + 1, 1))
        if not small:
            ax.legend(fontsize=fsz - 2, loc='upper right')

    _cur_page   = {'idx': 0}
    _zoom_state = {'active': False, 'amp_idx': 0}
    _show_bg    = {'val': False}
    _out        = Output()

    def _amp_label(d, with_bg=False):
        """Build display label for one amplitude group dict."""
        if d.get('is_merged'):
            lbl = (f"{d['amp']:.5f} mA "
                   f"({d['amp_lo']:.5f}–{d['amp_hi']:.5f}, n={d['n']})")
        else:
            lbl = f"{d['amp']:.5f} mA (n={d['n']})"
        if with_bg:
            _blo = d.get('bg_lo', float('nan'))
            _bhi = d.get('bg_hi', float('nan'))
            if np.isfinite(_blo) and np.isfinite(_bhi):
                lbl += f"\nBG: {_blo:.0f}–{_bhi:.0f} µV"
        return lbl

    def _refresh():
        if _zoom_state['active']:
            _show_zoom(_zoom_state['amp_idx'])
        else:
            _plot_page(_cur_page['idx'])
    _amp_drop = Dropdown(description='Amplitude:')

    def _plot_page(page_idx):
        with _out:
            _out.clear_output(wait=True)
            page = _vst['pages'][page_idx]
            n    = len(page)

            _amp_drop.options = [
                (_amp_label(d, _show_bg['val']), page_idx * n_per_page + i)
                for i, d in enumerate(page)
            ]
            if _amp_drop.options:
                _amp_drop.value = _amp_drop.options[0][1]

            fig, axs  = plt.subplots(2, 3, figsize=(_figsize['w'], _figsize['h']))
            _axs_flat = axs.flatten()

            for j in range(n_per_page):
                ax = _axs_flat[j]
                if j < n:
                    d = page[j]
                    _draw_avg_panel(ax, d, small=True)
                    _grid_lbl = _amp_label(d, _show_bg['val'])
                    ax.set_title(_grid_lbl, fontsize=18 if '\n' in _grid_lbl else 24)
                else:
                    ax.axis('off')

            _n_start = n_per_page * page_idx + 1
            _n_end   = min(n_per_page * (page_idx + 1), len(_vst['amp_data']))
            _pol_lbl = _pol_labels.get(_active_pol['val'], '')
            fig.suptitle(
                f"{header.subject_id}    "
                f"Amplitudes {_n_start}-{_n_end} of {len(_vst['amp_data'])}"
                f"  (Page {page_idx+1}/{len(_vst['pages'])})"
                + (f"  [{_pol_lbl}]" if _dual_pol else ""),
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
        _zoom_state['active']  = True
        _zoom_state['amp_idx'] = amp_idx
        d = _vst['amp_data'][amp_idx]
        with _out:
            _out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(_figsize['w'] * 13/15, _figsize['h'] * 5/7))
            _draw_avg_panel(ax, d, small=False)
            _zoom_lbl = (
                f"Averaged Waveforms (n={d['n']}) | "
                f"Stim Amp: {_amp_label(d, False)} | "
                f"{header.subject_id}  "
                f"({header.session_start_time:%Y-%m-%d %H:%M})"
            )
            _blo = d.get('bg_lo', float('nan'))
            _bhi = d.get('bg_hi', float('nan'))
            if _show_bg['val'] and np.isfinite(_blo) and np.isfinite(_bhi):
                _zoom_lbl += f"\nBG: {_blo:.0f}–{_bhi:.0f} µV"
            ax.set_title(_zoom_lbl, fontsize=12)
            plt.tight_layout()
            plt.show()

            def _on_back(b):
                _zoom_state['active'] = False
                _plot_page(_cur_page['idx'])
            _back_btn = Button(description='Back to grid', button_style='info')
            _back_btn.on_click(_on_back)
            display(_back_btn)

    def _on_prev(b):
        if _cur_page['idx'] > 0:
            _cur_page['idx'] -= 1
            _page_drop.value = _cur_page['idx']
            _plot_page(_cur_page['idx'])

    def _on_next(b):
        if _cur_page['idx'] < len(_vst['pages']) - 1:
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
            _refresh()
        return _cb

    def _on_auto_toggle(change):
        _ylim_auto['val'] = bool(change['new'])
        _ymin_box.disabled = bool(change['new'])
        _ymax_box.disabled = bool(change['new'])
        _refresh()

    def _on_ymin_change(change):
        _ylim_man['lo'] = float(change['new'])
        if not _ylim_auto['val']:
            _refresh()

    def _on_ymax_change(change):
        _ylim_man['hi'] = float(change['new'])
        if not _ylim_auto['val']:
            _refresh()

    def _on_figw_change(change):
        _figsize['w'] = float(change['new'])
        _refresh()

    def _on_figh_change(change):
        _figsize['h'] = float(change['new'])
        _refresh()

    _prev_btn  = Button(description='Prev',           button_style='')
    _next_btn  = Button(description='Next',           button_style='primary')
    _page_drop = Dropdown(
        options=[(f'Page {i+1}', i) for i in range(len(_vst['pages']))],
        description='Page:', layout={'width': '130px'}
    )
    _view_btn  = Button(description='View amplitude', button_style='info')

    _cb_adc      = Checkbox(value=False, description='ADC sync (green)',     indent=False,
                            layout={'width': '185px'})
    _cb_stim_adc = Checkbox(value=False, description='Stim ADC (magenta)',   indent=False,
                            layout={'width': '195px'})
    _cb_abs_bip  = Checkbox(value=False, description='|Bipolar| (gray)',     indent=False,
                            layout={'width': '175px'})
    _cb_uni      = Checkbox(value=False, description='Unipolar (orange)',    indent=False,
                            layout={'width': '185px'})
    _cb_abs_uni  = Checkbox(value=False, description='|Unipolar| (purple)',  indent=False,
                            layout={'width': '195px'})
    _cb_bg       = Checkbox(value=False, description='BG EMG range',          indent=False,
                            layout={'width': '160px'})

    _auto_toggle = ToggleButton(
        value=True, description='Auto y-scale',
        button_style='success',
        tooltip='Auto-scale shared y-axis from visible signals'
    )
    _ymin_box = FloatText(value=-1000.0, description='Y min:',
                          disabled=True, layout={'width': '145px'})
    _ymax_box = FloatText(value=1500.0,  description='Y max:',
                          disabled=True, layout={'width': '145px'})
    _figw_box = FloatText(value=fig_width,  description='Fig W:',
                          layout={'width': '130px'})
    _figh_box = FloatText(value=fig_height, description='Fig H:',
                          layout={'width': '130px'})

    _prev_btn.on_click(_on_prev)
    _next_btn.on_click(_on_next)
    _page_drop.observe(_on_page_change, names='value')
    _view_btn.on_click(_on_view)
    _cb_adc.observe(_make_sig_cb('adc'), names='value')
    _cb_stim_adc.observe(_make_sig_cb('stim_adc'), names='value')
    _cb_abs_bip.observe(_make_sig_cb('abs_bip'), names='value')
    _cb_uni.observe(_make_sig_cb('uni'), names='value')
    _cb_abs_uni.observe(_make_sig_cb('abs_uni'), names='value')

    def _on_bg_toggle(change):
        _show_bg['val'] = bool(change['new'])
        _refresh()
    _cb_bg.observe(_on_bg_toggle, names='value')
    _auto_toggle.observe(_on_auto_toggle, names='value')
    _ymin_box.observe(_on_ymin_change, names='value')
    _ymax_box.observe(_on_ymax_change, names='value')
    _figw_box.observe(_on_figw_change, names='value')
    _figh_box.observe(_on_figh_change, names='value')

    # ── polarity toggle (dual-polarity sessions only) ──────────────────────
    _top_rows = []
    if _dual_pol:
        from ipywidgets import ToggleButtons as _TB
        _pol_tog = _TB(
            options=[(_pol_labels[k], k) for k in _pol_keys],
            value=_active_pol['val'],
            description='Polarity:',
            button_style='info',
            tooltips=['Show normal polarity trials', 'Show reversed polarity trials'],
        )
        def _on_pol_change(change):
            _active_pol['val'] = change['new']
            _vst['amp_data'] = _build_amp_data(_pol_split[change['new']])
            _vst['pages'] = [_vst['amp_data'][i:i+n_per_page]
                             for i in range(0, len(_vst['amp_data']), n_per_page)]
            _cur_page['idx'] = 0
            _zoom_state['active'] = False
            _page_drop.options = [(f'Page {i+1}', i) for i in range(len(_vst['pages']))]
            _page_drop.value = 0
            _plot_page(0)
            for _hook in _pol_change_hooks:
                _hook(change['new'])
        _pol_tog.observe(_on_pol_change, names='value')
        _top_rows.append(HBox([HTML('<b>Stim polarity:</b>  '), _pol_tog]))

    _nav_row = HBox([_prev_btn, _next_btn, _page_drop,
                     Label('  '), _amp_drop, _view_btn])
    _sig_row = HBox([
        VBox([
            HTML('<b>Signal overlays:</b>'),
            HBox([_cb_adc, _cb_stim_adc, _cb_abs_bip, _cb_uni, _cb_abs_uni]),
            HBox([_cb_bg, HTML('<i style="color:#555;font-size:0.85em">'
                               ' toggle BG EMG range in labels</i>')]),
        ]),
        Label('   '),
        VBox([_auto_toggle, _ymin_box, _ymax_box]),
        Label('   '),
        VBox([HTML('<b>Figure size:</b>'), _figw_box, _figh_box]),
    ])

    _pol_note = (f"  — dual-polarity session: {len(_pol_split[0])} normal, "
                 f"{len(_pol_split[1])} reversed" if _dual_pol else "")
    print(f"Loaded {len(_vst['amp_data'])} amplitude groups across "
          f"{len(_vst['pages'])} page(s){_pol_note}.")
    print("Double-click a subplot to zoom in. Ctrl-click to select multiple signals.")

    display(VBox(_top_rows + [_nav_row, _sig_row, _out]))
    _plot_page(0)

    # ---- Recruitment Curve ----
    _PRE_RC  = max(m_start_ms, h_start_ms) + 1.0
    _POST_RC = h_end_ms + 2.0

    # ── compute wave data per polarity group ──────────────────────────────
    _rc_data = {}
    for _pk in _pol_keys:
        _md: dict = defaultdict(list)
        _hd: dict = defaultdict(list)
        for trial in _pol_split[_pk]:
            amp_key = round(trial.stimulation_amplitude_ma, 3)
            t_ms, emg, _, _, _ = get_trial_window(trial, _PRE_RC, _POST_RC,
                                                   ms_per_sample=_ms_ps,
                                                   bin_samples=_bin_s,
                                                   record_samples=_rec_s)
            m_mask = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
            if np.any(m_mask):
                _md[amp_key].append(np.mean(np.abs(emg[m_mask])))
            h_mask = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
            if np.any(h_mask):
                _hd[amp_key].append(np.mean(np.abs(emg[h_mask])))
        _sa  = sorted(set(_md.keys()) | set(_hd.keys()))
        _mls = [_md.get(a, [0]) for a in _sa]
        _hls = [_hd.get(a, [0]) for a in _sa]
        _rc_data[_pk] = {
            'sorted_amps': _sa,
            'm_means': np.array([np.mean(v) for v in _mls]),
            'm_stds':  np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in _mls]),
            'h_means': np.array([np.mean(v) for v in _hls]),
            'h_stds':  np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in _hls]),
        }

    # ── interactive recruitment curve output ──────────────────────────────
    _rc_out = Output()

    def _draw_rc_curves(pol_key):
        _d   = _rc_data[pol_key]
        _sa  = np.array(_d['sorted_amps'])
        M_max_k = float(np.max(_d['m_means'])) if np.max(_d['m_means']) > 0 else 1.0
        _norm_m     = (_d['m_means'] / M_max_k) * 100
        _norm_h     = (_d['h_means'] / M_max_k) * 100
        _norm_m_std = (_d['m_stds']  / M_max_k) * 100
        _norm_h_std = (_d['h_stds']  / M_max_k) * 100

        _interp_func_k = interp1d(_norm_m, _sa, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
        try:
            current_at_50_k = float(_interp_func_k(50))
        except Exception:
            current_at_50_k = float(_sa[np.argmax(_norm_m >= 50)])
        _nc = _sa / current_at_50_k

        _pol_lbl = _pol_labels.get(pol_key, str(pol_key)) if _dual_pol else ''
        _lbl_sfx = f'  [{_pol_lbl}]' if _dual_pol else ''

        with _rc_out:
            _rc_out.clear_output(wait=True)

            # normalized recruitment curve
            fig, ax = plt.subplots(figsize=(_figsize['w'] * 10/15, _figsize['h']))
            ax.errorbar(_nc, _norm_m, yerr=_norm_m_std, fmt='o-', color='blue',
                        label='M-wave (% Mmax) ± STD', capsize=3)
            ax.errorbar(_nc, _norm_h, yerr=_norm_h_std, fmt='o-', color='green',
                        label='H-wave (% Mmax) ± STD', capsize=3)
            _H_max_k    = float(np.max(_norm_h))
            _idx_Hmax_k = int(np.argmax(_norm_h))
            _cur_Hmax_k = float(_nc[_idx_Hmax_k])
            ax.axhline(_H_max_k, color='green', linestyle='--', linewidth=1,
                       label=f'H_max = {_H_max_k:.1f}% Mmax')
            ax.axvline(_cur_Hmax_k, color='gray', linestyle='--', linewidth=1,
                       label=f'Current at H_max = {_cur_Hmax_k:.2f}x')
            ax.text(_cur_Hmax_k + 0.02, _H_max_k + 2, 'b', fontsize=12, color='black')
            ax.text(_cur_Hmax_k - 0.08, _H_max_k + 2, 'a', fontsize=12, color='black')
            ax.set_xlabel('Current (normalized to current at 50% Mmax)', fontsize=18)
            ax.set_ylabel('H and M wave amplitude (% of Mmax)', fontsize=18)
            ax.set_title(f'HRS2 Normalized Recruitment Curve - {header.subject_id}{_lbl_sfx}',
                         fontsize=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
            _apply_tiered_ticks(ax)
            plt.tight_layout()
            plt.show()

            # raw (mA) recruitment curve
            fig, ax = plt.subplots(figsize=(_figsize['w'] * 10/15, _figsize['h']))
            ax.errorbar(_sa, _d['m_means'], yerr=_d['m_stds'],
                        fmt='o-', color='blue',  label='M-wave mean ± STD', capsize=3)
            ax.errorbar(_sa, _d['h_means'], yerr=_d['h_stds'],
                        fmt='o-', color='green', label='H-wave mean ± STD', capsize=3)
            ax.set_xticks(_sa)
            ax.set_xticklabels([f'{a:.3f}' for a in _sa], rotation=45, ha='right', fontsize=9)
            ax.set_xlabel('Stimulation Amplitude (mA)', fontsize=18)
            ax.set_ylabel('MRA Amplitude (µV)', fontsize=18)
            ax.set_title(f'HRS2 Recruitment Curve - {header.subject_id}{_lbl_sfx}', fontsize=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
            _apply_tiered_ticks(ax)
            plt.tight_layout()
            plt.show()

            _h_norm_k    = (_d['h_means'] / M_max_k) * 100
            _idx_Hmax_rw = int(np.argmax(_h_norm_k))
            print(f"M_max = {M_max_k:.2f} µV")
            print(f"H_max = {float(np.max(_d['h_means'])):.2f} µV "
                  f"({float(np.max(_h_norm_k)):.1f}% of M_max)")
            print(f"Current at 50% M_max = {current_at_50_k:.3f} mA")
            print(f"Current at H_max = {float(_sa[_idx_Hmax_rw]):.3f} mA "
                  f"({float(_sa[_idx_Hmax_rw] / current_at_50_k):.3f}x normalized)")

    _pol_change_hooks.append(_draw_rc_curves)
    display(_rc_out)
    _draw_rc_curves(_active_pol['val'])

    # ── linked background EMG: updates on amplitude selection or polarity change ──
    if emg_blocks is not None:
        _bg_out_a = Output()

        def _render_bg_a(trial_subset):
            with _bg_out_a:
                _bg_out_a.clear_output(wait=True)
                if trial_subset:
                    plot_background_emg_views(
                        trial_subset, emg_blocks,
                        monitoring_window_ms=monitoring_window_ms,
                        sample_rate=sample_rate)

        def _on_amp_bg(change):
            if change['new'] is not None:
                _idx = change['new']
                if 0 <= _idx < len(_vst['amp_data']):
                    _render_bg_a(_vst['amp_data'][_idx]['trials'])

        _amp_drop.observe(_on_amp_bg, names='value')
        _pol_change_hooks.append(lambda _pk: _render_bg_a(_pol_split[_pk]))

        display(HTML('<hr><b>Background EMG — selected amplitude / polarity group</b>'))
        display(_bg_out_a)
        if _vst['amp_data']:
            _render_bg_a(_vst['amp_data'][0]['trials'])

    # ── linked H-wave regression ──────────────────────────────────────────
    if emg_blocks is not None:
        _reg_out_a = Output()

        def _render_reg_a(trial_subset):
            with _reg_out_a:
                _reg_out_a.clear_output(wait=True)
                if trial_subset:
                    plot_hwave_regression(
                        trial_subset, emg_blocks,
                        m_start_ms=m_start_ms, m_end_ms=m_end_ms,
                        h_start_ms=h_start_ms, h_end_ms=h_end_ms,
                        sample_rate=sample_rate,
                        pre_ms=pre_avg_ms, post_ms=post_avg_ms,
                        monitoring_window_ms=monitoring_window_ms)

        def _on_amp_reg(change):
            if change['new'] is not None:
                _idx = change['new']
                if 0 <= _idx < len(_vst['amp_data']):
                    _render_reg_a(_vst['amp_data'][_idx]['trials'])

        _amp_drop.observe(_on_amp_reg, names='value')
        _pol_change_hooks.append(lambda _pk: _render_reg_a(_pol_split[_pk]))

        display(HTML('<hr><b>H-wave Regression — selected amplitude / polarity group</b>'))
        display(_reg_out_a)
        if _vst['amp_data']:
            _render_reg_a(_vst['amp_data'][0]['trials'])


def plot_hrs2_trials(trials, header,
                     pre_plot_ms: float = 2.0, post_plot_ms: float = 15.0,
                     n_per_page: int = 6,
                     m_start_ms: float = 2.0, m_end_ms: float = 4.0,
                     h_start_ms: float = 6.0, h_end_ms: float = 10.0,
                     sample_rate: float = SAMPLE_RATE,
                     fig_width: float = 15.0, fig_height: float = 7.0,
                     emg_blocks=None, monitoring_window_ms: float = 2500.0):
    """Interactive per-trial paged grid + zoom viewer.

    Each page shows ``n_per_page`` panels, one per individual trial, with the
    single-trial bipolar EMG, optional ADC sync and stim-waveform overlays,
    M/H wave region shading, and stim onset/end markers.
    Mirrors the style of ``plot_hrs2_analysis``.
    """
    import matplotlib.pyplot as plt
    from ipywidgets import (Button, Output, HBox, VBox, Dropdown, Label,
                            Checkbox, ToggleButton, FloatText, HTML)
    from IPython.display import display

    if len(trials) == 0:
        print("No HRS2 trials to display.")
        return

    _ms_ps = 1000.0 / sample_rate
    _bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
    _rec_s = int(TRIAL_RECORD_MS  * sample_rate / 1000)

    # ── polarity split ────────────────────────────────────────────────────
    _pol_split_t: dict = {}
    for _tr in trials:
        _p = int(getattr(_tr, 'stim_polarity_reversed', 0))
        _pol_split_t.setdefault(_p, []).append(_tr)
    _dual_pol_t  = len(_pol_split_t) > 1
    _pol_keys_t  = sorted(_pol_split_t.keys())
    _pol_labels_t = {0: 'Normal polarity (0)', 1: 'Reversed polarity (1)'}
    _active_pol_t = {'val': _pol_keys_t[0]}

    def _build_trial_data(trial_list):
        _td = []
        for _i, _tr in enumerate(trial_list):
            _t, _emg, _adc, _end, _stim_adc = get_trial_window(
                _tr, pre_plot_ms, post_plot_ms,
                ms_per_sample=_ms_ps, bin_samples=_bin_s, record_samples=_rec_s)
            _td.append({
                'idx': _i, 'trial': _tr,
                't_ms': _t, 'emg': _emg, 'adc': _adc,
                'stim_end_ms': _end, 'stim_adc': _stim_adc,
                'amp': _tr.stimulation_amplitude_ma,
                'bg':  get_trial_bg_emg(_tr),
            })
        return _td

    _vst_t = {'trial_data': _build_trial_data(_pol_split_t[_active_pol_t['val']])}
    _vst_t['pages'] = [_vst_t['trial_data'][i:i+n_per_page]
                       for i in range(0, len(_vst_t['trial_data']), n_per_page)]

    _show_sigs = {'val': set()}
    _ylim_auto = {'val': True}
    _ylim_man  = {'lo': -1000.0, 'hi': 1500.0}
    _figsize   = {'w': fig_width, 'h': fig_height}
    _abs_emg   = {'val': False}
    _view_mode = {'val': 'all'}   # 'all' | 'stim'
    _upd_amp   = {'val': False}   # guard against observe re-entrancy

    def _get_ylim():
        if _ylim_auto['val']:
            _raw = np.concatenate([d['emg'] for d in _vst_t['trial_data']])
            _all = np.abs(_raw) if _abs_emg['val'] else _raw
            _all = _all[~np.isnan(_all)]
            if len(_all) == 0:
                return (0.0, 1500.0) if _abs_emg['val'] else (-1000.0, 1500.0)
            _lo, _hi = float(np.nanmin(_all)), float(np.nanmax(_all))
            _pad = max(0.08 * (_hi - _lo), 1.0)
            return (_lo - _pad, _hi + _pad)
        return (_ylim_man['lo'], _ylim_man['hi'])

    def _draw_trial_panel(ax, d, small=True):
        lw  = 0.8 if small else 1.5
        fsz = 8   if small else 11
        t, emg = d['t_ms'], d['emg']
        if _abs_emg['val']:
            emg = np.abs(emg)
        end_ms = d['stim_end_ms']
        sigs   = _show_sigs['val']

        ax.axhline(0, color='black', linewidth=2.0, linestyle='-', alpha=1.0, zorder=3)
        ax.plot(t, emg, color='black', linewidth=lw * 2.0)

        _ax2 = None
        if ('adc' in sigs and d['adc'] is not None) or \
           ('stim_adc' in sigs and d['stim_adc'] is not None):
            _ax2 = ax.twinx()
            if 'adc' in sigs and d['adc'] is not None:
                _adc_center = float(np.nanmean(d['adc']))
                _ax2.plot(t, d['adc'] - _adc_center, color='green',
                          linewidth=lw * 1.8, alpha=0.85, label='ADC sync (TTL)')
            if 'stim_adc' in sigs and d['stim_adc'] is not None:
                _sa_center = float(np.nanmean(d['stim_adc']))
                _ax2.plot(t, d['stim_adc'] - _sa_center, color='magenta',
                          linewidth=lw * 1.8, alpha=0.85, label='Stim ADC')
            _ax2.set_ylabel('ADC (V)', fontsize=fsz - 1)
            _ax2.tick_params(axis='y', labelsize=fsz - 2)

        if end_ms is not None:
            ax.axvspan(0, end_ms, color='red', alpha=0.20)
            ax.axvline(end_ms, color='red', linestyle='--', linewidth=lw)
        ax.axvline(0, color='red', linestyle='--', linewidth=lw)

        ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.20, zorder=2)
        ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.20, zorder=2)
        ax.axvline(m_start_ms, color='blue',  linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
        ax.axvline(m_end_ms,   color='blue',  linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
        ax.axvline(h_start_ms, color='green', linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)
        ax.axvline(h_end_ms,   color='green', linestyle='--', linewidth=1.5, alpha=0.9, zorder=3)

        _mm = (t >= m_start_ms) & (t <= m_end_ms)
        _hm = (t >= h_start_ms) & (t <= h_end_ms)
        _m_mra = float(np.nanmean(np.abs(emg[_mm]))) if _mm.any() else float('nan')
        _h_mra = float(np.nanmean(np.abs(emg[_hm]))) if _hm.any() else float('nan')
        if not np.isnan(_m_mra):
            ax.hlines(_m_mra, m_start_ms, m_end_ms, colors='blue', linestyles='dotted',
                      linewidth=lw * 2.5, zorder=5)
            ax.text((m_start_ms + m_end_ms) / 2, 0.93, f'M: {_m_mra:.1f} uV',
                    transform=ax.get_xaxis_transform(),
                    color='blue', fontsize=fsz - 1, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='blue', alpha=0.85),
                    zorder=8)
        if not np.isnan(_h_mra):
            ax.hlines(_h_mra, h_start_ms, h_end_ms, colors='green', linestyles='dotted',
                      linewidth=lw * 2.5, zorder=5)
            ax.text((h_start_ms + h_end_ms) / 2, 0.93, f'H: {_h_mra:.1f} uV',
                    transform=ax.get_xaxis_transform(),
                    color='darkgreen', fontsize=fsz - 1, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='green', alpha=0.85),
                    zorder=8)

        # ── stim pulse peak-to-peak annotation ───────────────────────────
        _sa = d.get('stim_adc')
        if _sa is not None and len(_sa) > 0:
            _end = end_ms if end_ms is not None else 1.0
            _stm = (t >= 0) & (t <= _end)
            if _stm.sum() >= 2:
                _seg = _sa[_stm]
                _seg = _seg[~np.isnan(_seg)]
                if len(_seg) >= 2:
                    _ptp = float(np.max(_seg) - np.min(_seg))
                    ax.text(0.01, 0.01, f'Stim P2P: {_ptp:.3f} V',
                            transform=ax.transAxes, color='magenta',
                            fontsize=fsz - 1, ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec='magenta', alpha=0.85),
                            zorder=8)

        # ── pre-stim EMG activity annotation ─────────────────────────────
        _pre_mask = t < 0
        if _pre_mask.sum() >= 2:
            _pre_emg = float(np.nanmean(np.abs(d['emg'][_pre_mask])))
            if not np.isnan(_pre_emg):
                ax.text(0.99, 0.01, f'Pre-stim EMG: {_pre_emg:.1f} µV',
                        transform=ax.transAxes, color='dimgray',
                        fontsize=fsz - 1, ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  ec='dimgray', alpha=0.85),
                        zorder=8)

        ax.set_xlim(-pre_plot_ms, post_plot_ms)
        ax.set_ylim(_get_ylim())
        if _ax2 is not None:
            _y1_lo, _y1_hi = ax.get_ylim()
            if _y1_hi > _y1_lo:
                _zero_frac = (0.0 - _y1_lo) / (_y1_hi - _y1_lo)
                _y2_lo, _y2_hi = _ax2.get_ylim()
                _y2_span = _y2_hi - _y2_lo
                _ax2.set_ylim(-_zero_frac * _y2_span,
                              (1.0 - _zero_frac) * _y2_span)

        ax.set_xlabel('Time (ms)', fontsize=fsz)
        ax.set_ylabel('EMG (uV)', fontsize=fsz)
        ax.tick_params(labelsize=fsz - 1)
        ax.tick_params(axis='x', width=2.0)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.grid(True, alpha=0.3)
        _min_ms = int(np.floor(t[0]))
        _max_ms = int(np.ceil(t[-1]))
        ax.set_xticks(np.arange(_min_ms, _max_ms + 1, 1))
        if not small:
            handles, labels = ax.get_legend_handles_labels()
            if _ax2 is not None:
                h2, l2 = _ax2.get_legend_handles_labels()
                handles += h2
                labels  += l2
            if handles:
                ax.legend(handles, labels, fontsize=fsz - 2, loc='upper right')

    _cur_page   = {'idx': 0}
    _zoom_state = {'active': False, 'trial_idx': 0}
    _show_bg_t  = {'val': False}
    _out        = Output()

    def _trial_label(d, with_bg=False):
        """Build display label for one trial dict."""
        lbl = f"Trial {d['idx']+1}  |  {d['amp']:.5f} mA"
        if with_bg:
            _bg = d.get('bg', float('nan'))
            if np.isfinite(_bg):
                lbl += f"\nBG: {_bg:.0f} µV"
        return lbl

    def _refresh():
        if _zoom_state['active']:
            _show_zoom(_zoom_state['trial_idx'])
        else:
            _plot_page(_cur_page['idx'])

    _trial_drop = Dropdown(description='Trial:')

    def _plot_page(page_idx):
        with _out:
            _out.clear_output(wait=True)
            page = _vst_t['pages'][page_idx]
            n    = len(page)

            _trial_drop.options = [
                (_trial_label(d, _show_bg_t['val']), page_idx * n_per_page + i)
                for i, d in enumerate(page)
            ]
            if _trial_drop.options:
                _trial_drop.value = _trial_drop.options[0][1]

            fig, axs  = plt.subplots(2, 3, figsize=(_figsize['w'], _figsize['h']))
            _axs_flat = axs.flatten()

            for j in range(n_per_page):
                ax = _axs_flat[j]
                if j < n:
                    d = page[j]
                    _draw_trial_panel(ax, d, small=True)
                    _grid_lbl_t = _trial_label(d, _show_bg_t['val'])
                    ax.set_title(_grid_lbl_t, fontsize=7.5 if '\n' in _grid_lbl_t else 9)
                else:
                    ax.axis('off')

            _n_start = n_per_page * page_idx + 1
            _n_end   = min(n_per_page * (page_idx + 1), len(_vst_t['trial_data']))
            _pol_lbl_t = (f"  [{_pol_labels_t.get(_active_pol_t['val'], str(_active_pol_t['val']))}]"
                          if _dual_pol_t else "")
            fig.suptitle(
                f"{header.subject_id}    "
                f"Trials {_n_start}–{_n_end} of {len(_vst_t['trial_data'])}"
                f"  (Page {page_idx+1}/{len(_vst_t['pages'])}){_pol_lbl_t}",
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
                                _trial_drop.value = page_idx * n_per_page + k
                            break
                fig.canvas.mpl_connect('button_press_event', _on_click)
            except Exception:
                pass

    def _show_zoom(trial_idx):
        _zoom_state['active']    = True
        _zoom_state['trial_idx'] = trial_idx
        d = _vst_t['trial_data'][trial_idx]
        with _out:
            _out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(_figsize['w'] * 13/15, _figsize['h'] * 5/7))
            _draw_trial_panel(ax, d, small=False)
            _zt_lbl = (
                f"Trial {d['idx']+1}  |  Stim = {d['amp']:.5f} mA"
                f"  |  {header.subject_id}  "
                f"({header.session_start_time:%Y-%m-%d %H:%M})"
            )
            if _show_bg_t['val'] and np.isfinite(d.get('bg', float('nan'))):
                _zt_lbl += f"\nBG: {d['bg']:.0f} µV"
            ax.set_title(_zt_lbl, fontsize=12)
            plt.tight_layout()
            plt.show()

            def _on_back(b):
                _zoom_state['active'] = False
                _plot_page(_cur_page['idx'])
            _back_btn = Button(description='Back to grid', button_style='info')
            _back_btn.on_click(_on_back)
            display(_back_btn)

    _all_amps_t = sorted({round(tr.stimulation_amplitude_ma, 3) for tr in trials})
    _rebuild_hooks_t = []

    def _rebuild_trials():
        pol_trs = list(_pol_split_t[_active_pol_t['val']])
        if _view_mode['val'] == 'stim':
            _pol_amps = sorted({round(t.stimulation_amplitude_ma, 3) for t in pol_trs})
            _upd_amp['val'] = True
            _stim_amp_drop.options = ([(f'{a:.5f} mA', a) for a in _pol_amps]
                                      if _pol_amps else [('—', None)])
            _stim_amp_drop.disabled = not bool(_pol_amps)
            if _stim_amp_drop.value not in _pol_amps and _pol_amps:
                _stim_amp_drop.value = _pol_amps[0]
            _upd_amp['val'] = False
            tgt = _stim_amp_drop.value
            if tgt is not None:
                pol_trs = [t for t in pol_trs
                           if round(t.stimulation_amplitude_ma, 3) == tgt]
        else:
            _stim_amp_drop.disabled = True
        _vst_t['trial_data'] = _build_trial_data(pol_trs)
        _vst_t['pages'] = [_vst_t['trial_data'][i:i + n_per_page]
                           for i in range(0, max(len(_vst_t['trial_data']), 1), n_per_page)]
        _cur_page['idx'] = 0
        _zoom_state['active'] = False
        _page_drop.options = [(f'Page {i+1}', i) for i in range(len(_vst_t['pages']))]
        _page_drop.value = 0
        _plot_page(0)
        for _hook in _rebuild_hooks_t:
            _hook(pol_trs)

    def _on_prev(b):
        if _cur_page['idx'] > 0:
            _cur_page['idx'] -= 1
            _page_drop.value = _cur_page['idx']
            _plot_page(_cur_page['idx'])

    def _on_next(b):
        if _cur_page['idx'] < len(_vst_t['pages']) - 1:
            _cur_page['idx'] += 1
            _page_drop.value = _cur_page['idx']
            _plot_page(_cur_page['idx'])

    def _on_page_change(change):
        if change['name'] == 'value' and change['new'] != _cur_page['idx']:
            _cur_page['idx'] = change['new']
            _plot_page(_cur_page['idx'])

    def _on_view(b):
        try:
            _show_zoom(int(_trial_drop.value))
        except Exception as e:
            print(f'Could not zoom trial: {e}')

    def _make_sig_cb(key):
        def _cb(change):
            if change['new']:
                _show_sigs['val'].add(key)
            else:
                _show_sigs['val'].discard(key)
            _refresh()
        return _cb

    def _on_auto_toggle(change):
        _ylim_auto['val'] = bool(change['new'])
        _ymin_box.disabled = bool(change['new'])
        _ymax_box.disabled = bool(change['new'])
        _refresh()

    def _on_ymin_change(change):
        _ylim_man['lo'] = float(change['new'])
        if not _ylim_auto['val']:
            _refresh()

    def _on_ymax_change(change):
        _ylim_man['hi'] = float(change['new'])
        if not _ylim_auto['val']:
            _refresh()

    def _on_figw_change(change):
        _figsize['w'] = float(change['new'])
        _refresh()

    def _on_figh_change(change):
        _figsize['h'] = float(change['new'])
        _refresh()

    _has_stim_adc = any(d['stim_adc'] is not None for d in _vst_t['trial_data'])

    _prev_btn  = Button(description='Prev',       button_style='')
    _next_btn  = Button(description='Next',       button_style='primary')
    _page_drop = Dropdown(
        options=[(f'Page {i+1}', i) for i in range(len(_vst_t['pages']))],
        description='Page:', layout={'width': '130px'}
    )
    _view_btn  = Button(description='View trial', button_style='info')

    _cb_adc      = Checkbox(value=False, description='ADC sync (green)',   indent=False,
                            layout={'width': '185px'})
    _cb_stim_adc = Checkbox(value=False, description='Stim ADC (magenta)', indent=False,
                            layout={'width': '195px'}, disabled=not _has_stim_adc)
    _cb_bg_t     = Checkbox(value=False, description='BG EMG',             indent=False,
                            layout={'width': '130px'})

    from ipywidgets import ToggleButtons as _TBs_t
    _abs_emg_btn   = ToggleButton(value=False, description='Abs EMG',
                                  button_style='', icon='',
                                  tooltip='Show |EMG| absolute value trace',
                                  layout={'width': '110px'})
    _view_mode_tgl = _TBs_t(options=[('All trials', 'all'), ('By stim intensity', 'stim')],
                             value='all', button_style='',
                             layout={'width': '300px'})
    _stim_amp_drop = Dropdown(
        options=[(f'{a:.5f} mA', a) for a in _all_amps_t],
        description='Amplitude:', layout={'width': '210px', 'display': 'none'}
    )

    def _on_abs_emg_t(change):
        _abs_emg['val'] = bool(change['new'])
        _abs_emg_btn.button_style = 'warning' if change['new'] else ''
        _refresh()

    def _on_view_mode_t(change):
        _view_mode['val'] = change['new']
        _stim_amp_drop.layout.display = 'flex' if change['new'] == 'stim' else 'none'
        _stim_amp_drop.disabled = change['new'] != 'stim'
        _rebuild_trials()

    def _on_stim_amp_t(change):
        if _view_mode['val'] == 'stim' and not _upd_amp['val']:
            _rebuild_trials()

    _auto_toggle = ToggleButton(
        value=True, description='Auto y-scale',
        button_style='success',
        tooltip='Auto-scale shared y-axis from visible signals'
    )
    _ymin_box = FloatText(value=-1000.0, description='Y min:',
                          disabled=True, layout={'width': '145px'})
    _ymax_box = FloatText(value=1500.0,  description='Y max:',
                          disabled=True, layout={'width': '145px'})
    _figw_box = FloatText(value=fig_width,  description='Fig W:',
                          layout={'width': '130px'})
    _figh_box = FloatText(value=fig_height, description='Fig H:',
                          layout={'width': '130px'})

    _prev_btn.on_click(_on_prev)
    _next_btn.on_click(_on_next)
    _page_drop.observe(_on_page_change, names='value')
    _view_btn.on_click(_on_view)
    _cb_adc.observe(_make_sig_cb('adc'), names='value')
    _cb_stim_adc.observe(_make_sig_cb('stim_adc'), names='value')

    def _on_bg_toggle_t(change):
        _show_bg_t['val'] = bool(change['new'])
        _refresh()
    _cb_bg_t.observe(_on_bg_toggle_t, names='value')
    _abs_emg_btn.observe(_on_abs_emg_t, names='value')
    _view_mode_tgl.observe(_on_view_mode_t, names='value')
    _stim_amp_drop.observe(_on_stim_amp_t, names='value')
    _auto_toggle.observe(_on_auto_toggle, names='value')
    _ymin_box.observe(_on_ymin_change, names='value')
    _ymax_box.observe(_on_ymax_change, names='value')
    _figw_box.observe(_on_figw_change, names='value')
    _figh_box.observe(_on_figh_change, names='value')

    _nav_row = HBox([_prev_btn, _next_btn, _page_drop,
                     Label('  '), _trial_drop, _view_btn])
    _view_row = HBox([HTML('<b>View mode:</b>  '), _view_mode_tgl,
                      Label('  '), _stim_amp_drop])
    _sig_row = HBox([
        VBox([
            HTML('<b>Signal overlays:</b>'),
            HBox([_cb_adc, _cb_stim_adc, Label('  '), _abs_emg_btn]),
            HBox([_cb_bg_t, HTML('<i style="color:#555;font-size:0.85em">'
                                 ' show BG EMG in labels</i>')]),
        ]),
        Label('   '),
        VBox([_auto_toggle, _ymin_box, _ymax_box]),
        Label('   '),
        VBox([HTML('<b>Figure size:</b>'), _figw_box, _figh_box]),
    ])

    # ── polarity toggle (dual-polarity sessions only) ──────────────────────
    _top_rows_t = []
    if _dual_pol_t:
        from ipywidgets import ToggleButtons as _TB_t
        _pol_tog_t = _TB_t(
            options=[(_pol_labels_t[k], k) for k in _pol_keys_t],
            value=_active_pol_t['val'],
            description='Polarity:',
            button_style='info',
            tooltips=['Show normal polarity trials', 'Show reversed polarity trials'],
        )
        def _on_pol_change_t(change):
            _active_pol_t['val'] = change['new']
            _rebuild_trials()
        _pol_tog_t.observe(_on_pol_change_t, names='value')
        _top_rows_t.append(HBox([HTML('<b>Stim polarity:</b>  '), _pol_tog_t]))

    _pol_note_t = (f"  — dual-polarity session: "
                   f"{len(_pol_split_t.get(0, []))} normal, "
                   f"{len(_pol_split_t.get(1, []))} reversed" if _dual_pol_t else "")
    print(f"Loaded {len(_vst_t['trial_data'])} trials across "
          f"{len(_vst_t['pages'])} page(s){_pol_note_t}.")
    print(f"Sample rate: {sample_rate:.0f} Hz  |  ms/sample: {_ms_ps:.4f}")
    print(f"Stim waveform: {'available' if _has_stim_adc else 'not present in this file'}")
    print("Double-click a subplot to zoom in, or use the dropdown + 'View trial' button.")

    display(VBox(_top_rows_t + [_view_row, _nav_row, _sig_row, _out]))
    _plot_page(0)

    # ── linked background EMG: updates when group selection changes ────────
    if emg_blocks is not None:
        _bg_out_t = Output()

        def _render_bg_t(trial_subset):
            with _bg_out_t:
                _bg_out_t.clear_output(wait=True)
                if trial_subset:
                    plot_background_emg_views(
                        trial_subset, emg_blocks,
                        monitoring_window_ms=monitoring_window_ms,
                        sample_rate=sample_rate)

        _rebuild_hooks_t.append(_render_bg_t)
        display(HTML('<hr><b>Background EMG — selected group</b>'))
        display(_bg_out_t)
        _render_bg_t(list(_pol_split_t[_active_pol_t['val']]))

    # ── linked H:M ratio summary: updates when group selection changes ─────
    _hm_out_t = Output()

    def _render_hm_t(trial_subset):
        with _hm_out_t:
            _hm_out_t.clear_output(wait=True)
            if trial_subset:
                _tbp = split_trials_by_polarity(trial_subset)
                plot_hm_ratio_summary(
                    _tbp, header,
                    m_start_ms=m_start_ms, m_end_ms=m_end_ms,
                    h_start_ms=h_start_ms, h_end_ms=h_end_ms,
                    sample_rate=sample_rate,
                    pre_ms=pre_plot_ms, post_ms=post_plot_ms)

    _rebuild_hooks_t.append(_render_hm_t)
    display(HTML('<hr><b>H:M Ratio Summary — selected group</b>'))
    display(_hm_out_t)
    _render_hm_t(list(_pol_split_t[_active_pol_t['val']]))

    # ── linked H-wave regression ──────────────────────────────────────────
    _reg_out_t = Output()

    def _render_reg_t(trial_subset):
        with _reg_out_t:
            _reg_out_t.clear_output(wait=True)
            if trial_subset:
                plot_hwave_regression(
                    trial_subset, emg_blocks,
                    m_start_ms=m_start_ms, m_end_ms=m_end_ms,
                    h_start_ms=h_start_ms, h_end_ms=h_end_ms,
                    sample_rate=sample_rate,
                    pre_ms=pre_plot_ms, post_ms=post_plot_ms,
                    monitoring_window_ms=monitoring_window_ms)

    _rebuild_hooks_t.append(_render_reg_t)
    display(HTML('<hr><b>H-wave Regression — selected group</b>'))
    display(_reg_out_t)
    _render_reg_t(list(_pol_split_t[_active_pol_t['val']]))


def analyze_global_background(trials, emg_blocks, header,
                              sample_rate: float = SAMPLE_RATE,
                              blank_pre_ms: float = 5.0,
                              blank_post_ms: float = 20.0,
                              min_valid_frac: float = 0.7,
                              bin_samples: int = BIN_SAMPLES,
                              monitoring_window_ms: float = 2500.0,
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



    # ---- 4d: Per-trial pre-stim grand means ----
    print("\n" + "=" * 70)
    print("Section 4d: Per-trial pre-stim background grand means")
    print("=" * 70)

    # Compute per-trial background grand means by slicing continuous_abs_emg
    # directly using the same stim-position arithmetic as the 4b blank-mask
    # loop. This avoids a per-trial Python scan of emg_blocks.
    bg_bin_samp = int(BIN_DURATION_MS * sample_rate / 1000.0)
    bg_n_bins   = int(monitoring_window_ms / BIN_DURATION_MS)
    bg_needed   = bg_n_bins * bg_bin_samp

    trial_bg_gm = []
    trial_min_th = []
    trial_max_th = []
    n_bg_failed = 0
    for tr in trials:
        fid = int(getattr(tr, 'first_post_trigger_frame_sample_id', 0))
        osi = int(getattr(tr, 'onset_sample_index', -1))
        gm = float('nan')
        if fid > 0 and osi >= 0:
            stim_pos = fid + (osi - bin_samples) - first_oe
            if 0 <= stim_pos <= n_total:
                seg = continuous_abs_emg[max(0, stim_pos - bg_needed):stim_pos]
                if len(seg) >= bg_bin_samp:
                    n_act = len(seg) // bg_bin_samp
                    bins = seg[-n_act * bg_bin_samp:].reshape(n_act, bg_bin_samp).mean(axis=1)
                    gm = float(bins.mean())
        trial_bg_gm.append(gm)
        if np.isnan(gm):
            n_bg_failed += 1
        trial_min_th.append(float(tr.min_initiation_threshold))
        trial_max_th.append(float(tr.max_initiation_threshold))

    trial_bg_gm = np.array(trial_bg_gm, dtype=np.float64)
    trial_min_th = np.array(trial_min_th, dtype=np.float64)
    trial_max_th = np.array(trial_max_th, dtype=np.float64)

    valid_bg = trial_bg_gm[~np.isnan(trial_bg_gm)]
    gm_q1, gm_med, gm_q3 = (float(x) for x in np.percentile(valid_bg, [25, 50, 75]))

    print(f"  Trials : {len(trial_bg_gm)}  (failed reconstruction: {n_bg_failed})")
    print(f"  Min={np.nanmin(trial_bg_gm):.2f}  Q1={gm_q1:.2f}  Median={gm_med:.2f}  "
          f"Q3={gm_q3:.2f}  Max={np.nanmax(trial_bg_gm):.2f}")
    print(f"  Recorded thresholds (mean): "
          f"[{trial_min_th.mean():.2f}, {trial_max_th.mean():.2f}] µV")

    if show_plots:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.hist(valid_bg, bins=80, color='lightgray',
                edgecolor='black', linewidth=0.5, alpha=0.9,
                label='Per-trial pre-stim BG grand mean')
        ax.axvline(gm_q1,  color='darkorange', linestyle=':', linewidth=1.5,
                   label=f'Q1 = {gm_q1:.2f} µV')
        ax.axvline(gm_med, color='purple',     linestyle=':', linewidth=1.5,
                   label=f'Median = {gm_med:.2f} µV')
        ax.axvline(gm_q3,  color='steelblue',  linestyle=':', linewidth=1.5,
                   label=f'Q3 = {gm_q3:.2f} µV')
        ax.set_xlabel('Per-trial pre-stim grand mean (µV)')
        ax.set_ylabel('Trial count')
        ax.set_title(f'Per-Trial Pre-Stim Background Grand Means — {header.subject_id}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
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
        'monitoring_window_ms': monitoring_window_ms,
    }


def _bg_binned_grand_mean(trial_bg, min_uv, max_uv, bin_width_uv=1.0):
    """Compute representative background grand mean using 1 µV bins.

    Mirrors the Section 3/4 approach (time window → 50 ms bins → mean per bin →
    grand mean), but in EMG-level space:
        window  = [min_uv, max_uv]
        bins    = 1 µV each
        per bin = mean(trial_bg_gm values that fall in that bin)
        result  = mean of non-empty bin means (grand mean)

    Returns (grand_mean, n_occupied_bins).
    """
    window_width = max_uv - min_uv
    n_bins = max(1, int(round(window_width / bin_width_uv)))
    bin_edges = np.linspace(min_uv, max_uv, n_bins + 1)
    bin_means = []
    for k in range(n_bins):
        b_lo, b_hi = bin_edges[k], bin_edges[k + 1]
        in_bin = trial_bg[(trial_bg >= b_lo) & (trial_bg < b_hi)]
        if len(in_bin) > 0:
            bin_means.append(float(np.mean(np.abs(in_bin))))
    if not bin_means:
        return float('nan'), 0
    return float(np.mean(bin_means)), len(bin_means)


def run_direct_bg_sweep(state, sweep_centres=None, half_widths_uv=None):
    """Section 5a (direct): count actual HRS2 trials whose per-trial pre-stim
    background grand mean falls within each (centre ± half-width) window.

    Background characterisation uses the same binning pipeline as Sections 3/4:
        window  = [centre - hw, centre + hw]  (µV)
        bins    = 1 µV each
        per bin = mean of trial_bg_gm values in that bin
        result  = grand mean of non-empty bin means  → ``bin_grand_mean``

    No simulation — every count maps 1-to-1 with real recorded trials.
    ``state`` is the dict returned by ``analyze_global_background``;
    ``state['trial_bg_gm']`` must be populated (requires HRS2 trials with
    pre-stim grand mean data in Section 4d).

    Returns a sweep_results dict keyed by centre label, compatible with
    ``plot_direct_bg_sweep``.
    """
    if state is None:
        print("No state — run analyze_global_background first.")
        return None

    trial_bg = np.asarray(state['trial_bg_gm'], dtype=float)
    valid_bg = trial_bg[~np.isnan(trial_bg)]
    n_actual = int(len(valid_bg))

    if sweep_centres is None:
        sweep_centres = {'Q1':     state['gm_q1'],
                         'Median': state['gm_med'],
                         'Q3':     state['gm_q3']}
    if half_widths_uv is None:
        half_widths_uv = [5, 10, 20, 30, 50, 75, 100, 150]

    print("=" * 70)
    print("Section 5a: Direct per-trial background sweep")
    print("  Background window → 1 µV bins → mean per bin → grand mean")
    print("=" * 70)
    print(f"  Actual HRS2 trials : {n_actual}")
    print(f"  Background Q1={state['gm_q1']:.2f}  "
          f"Median={state['gm_med']:.2f}  Q3={state['gm_q3']:.2f} µV\n")

    sweep_results: dict = {}
    for centre_label, centre_val in sweep_centres.items():
        sweep_results[centre_label] = []
        for hw in half_widths_uv:
            min_uv = max(0.0, centre_val - hw)
            max_uv = centre_val + hw
            mask = (trial_bg >= min_uv) & (trial_bg <= max_uv)
            n_matched = int(np.sum(mask))
            bgm, n_occ = _bg_binned_grand_mean(valid_bg, min_uv, max_uv)
            sweep_results[centre_label].append({
                'hw': hw, 'min_uv': min_uv, 'max_uv': max_uv,
                'n_matched': n_matched,
                'matched_indices': np.where(mask)[0].tolist(),
                'bin_grand_mean': bgm,
                'n_occupied_bins': n_occ,
            })

        row_str = '  '.join(
            f'±{r["hw"]}→{r["n_matched"]}' for r in sweep_results[centre_label]
        )
        print(f"  {centre_label:6s} (centre={centre_val:.2f} µV):  {row_str}")

    print(f"\n{'Centre':>10} {'±hw (µV)':>10} {'Min (µV)':>10} "
          f"{'Max (µV)':>10} {'Matched':>10} {'Fraction':>10} {'BG Grand Mean':>14}")
    print("-" * 80)
    for label, rows in sweep_results.items():
        for r in rows:
            frac = r['n_matched'] / max(n_actual, 1)
            bgm_str = f"{r['bin_grand_mean']:.2f}" if np.isfinite(r['bin_grand_mean']) else 'n/a'
            print(f"{label:>10} {r['hw']:>10} {r['min_uv']:>10.2f} "
                  f"{r['max_uv']:>10.2f} {r['n_matched']:>10} {frac:>9.1%} {bgm_str:>14}")
        print()

    return sweep_results


def plot_direct_bg_sweep(sweep_results, state, trials, header):
    """Section 5b (direct): visualise run_direct_bg_sweep results.

    Three panels:
      1. Matched trial count vs half-width for each centre.
      2. Fraction of total trials vs half-width.
      3. Histogram of per-trial pre-stim grand means with threshold windows.
    """
    import matplotlib.pyplot as plt
    if not sweep_results:
        print("No sweep results to plot.")
        return

    n_actual = len(trials)
    trial_bg = np.asarray(state['trial_bg_gm'], dtype=float)
    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette = ['darkorange', 'purple', 'steelblue', 'teal', 'crimson']

    def _col(label, i):
        return colours.get(label, palette[i % len(palette)])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    for i, (label, rows) in enumerate(sweep_results.items()):
        hws = [r['hw'] for r in rows]
        ns  = [r['n_matched'] for r in rows]
        ax.plot(hws, ns, 'o-', label=label, color=_col(label, i), linewidth=1.8)
    ax.axhline(n_actual, color='gray', linestyle='--', linewidth=1.5,
               label=f'Total actual ({n_actual} trials)')
    ax.set_xlabel('EMG Window width (µV)')
    ax.set_ylabel('Matched actual trials')
    ax.set_title('Direct BG Sweep: Matched Trials vs Window Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i, (label, rows) in enumerate(sweep_results.items()):
        hws  = [r['hw'] for r in rows]
        frac = [r['n_matched'] / max(n_actual, 1) for r in rows]
        ax2.plot(hws, frac, 'o-', label=label, color=_col(label, i), linewidth=1.8)
    ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='100% of trials')
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax2.set_xlabel('EMG Window width (µV)')
    ax2.set_ylabel('Fraction of actual trials')
    ax2.set_title('Direct BG Sweep: Fraction of Trials in Window')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{header.subject_id} — Direct Per-Trial Background Sweep', fontsize=13)
    plt.tight_layout()
    plt.show()

    fig2, ax3 = plt.subplots(figsize=(13, 5))
    ax3.hist(trial_bg[~np.isnan(trial_bg)], bins=80, color='lightgray',
             edgecolor='black', linewidth=0.5, alpha=0.9,
             label='Per-trial pre-stim BG grand mean')
    ax3.axvline(state['gm_q1'],  color='darkorange', linestyle=':', linewidth=1.5,
                label=f"Q1 = {state['gm_q1']:.2f} µV")
    ax3.axvline(state['gm_med'], color='purple',     linestyle=':', linewidth=1.5,
                label=f"Median = {state['gm_med']:.2f} µV")
    ax3.axvline(state['gm_q3'],  color='steelblue',  linestyle=':', linewidth=1.5,
                label=f"Q3 = {state['gm_q3']:.2f} µV")
    if 'trial_min_th' in state:
        ax3.axvline(state['trial_min_th'].mean(), color='red', linestyle='--',
                    linewidth=1.5,
                    label=f"Recorded min-thresh ({state['trial_min_th'].mean():.2f})")
        ax3.axvline(state['trial_max_th'].mean(), color='darkred', linestyle='--',
                    linewidth=1.5,
                    label=f"Recorded max-thresh ({state['trial_max_th'].mean():.2f})")

    # ── top bracket bars (data x, axes y) — one per centre ──────────────────
    # Bars stack downward from just inside the top of the axes so they sit
    # above the histogram but below the title.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax3.transData, ax3.transAxes)
    bar_h     = 0.030   # bar height in axes fraction
    gap       = 0.008   # gap between stacked bars
    y_top_ref = 0.980   # top edge of the uppermost bar (axes fraction)

    for i, (label, rows) in enumerate(sweep_results.items()):
        col = _col(label, i)
        mid = rows[len(rows) // 2]
        y_top = y_top_ref - i * (bar_h + gap)
        y_bot = y_top - bar_h
        y_mid = (y_bot + y_top) / 2.0

        ax3.fill_between(
            [mid['min_uv'], mid['max_uv']], [y_bot, y_bot], [y_top, y_top],
            transform=trans, color=col, alpha=0.85, linewidth=0,
            clip_on=True,
            label=f"{label}  EMG window ±{mid['hw']} µV  →  {mid['n_matched']} trials")

        centre_x = (mid['min_uv'] + mid['max_uv']) / 2.0
        ax3.plot(centre_x, y_mid, '|', transform=trans,
                 color='white', markersize=7, markeredgewidth=2.0,
                 clip_on=True)

        ax3.text(mid['max_uv'], y_mid, f'  {label}',
                 transform=trans, color=col, fontsize=8,
                 ha='left', va='center', clip_on=True)

    ax3.set_xlabel('Per-trial pre-stim grand mean (µV)')
    ax3.set_ylabel('Trial count')
    ax3.set_title(f'Threshold Windows vs Actual Trial Background — {header.subject_id}')
    ax3.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def _build_bin_gm_series(signal, blank_mask, sample_rate, min_valid_frac=0.7,
                         n_win_bins=49):
    """Pre-compute a dense array of per-window grand means for fast sweeping.

    Bins the entire continuous signal into BIN_DURATION_MS-wide bins, then
    slides a n_win_bins-wide window across the bin array.  gm_series[k] is
    the grand mean of the valid (non-blanked) bins in window [k, k+n_win_bins).
    Used by run_threshold_sweep so the signal is processed only once instead
    of once per (centre, half-width) pair.
    """
    bin_samp   = int(BIN_DURATION_MS * sample_rate / 1000.0)
    n_complete = (len(signal) // bin_samp) * bin_samp
    if n_complete == 0 or bin_samp == 0:
        return np.array([]), bin_samp

    sig_r  = signal[:n_complete].reshape(-1, bin_samp).astype(np.float64)
    mask_r = blank_mask[:n_complete].reshape(-1, bin_samp)
    valid_counts = mask_r.sum(axis=1)
    bin_means = np.where(
        valid_counts > 0,
        (sig_r * mask_r).sum(axis=1) / np.maximum(valid_counts, 1),
        np.nan,
    )
    bin_valid = (valid_counts / bin_samp) >= min_valid_frac

    n_bins = len(bin_means)
    if n_bins < n_win_bins:
        return np.full(max(0, n_bins - n_win_bins + 1), np.nan), bin_samp

    from numpy.lib.stride_tricks import sliding_window_view
    win_bm = sliding_window_view(bin_means, n_win_bins)
    win_bv = sliding_window_view(bin_valid.view(np.uint8), n_win_bins).astype(bool)

    n_valid          = win_bv.sum(axis=1)
    win_bm_valid_sum = np.where(win_bv, win_bm, 0.0).sum(axis=1)
    gm_series        = np.where(n_valid > 0, win_bm_valid_sum / n_valid, np.nan)
    return gm_series, bin_samp


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
    print("Section 4b: Running post-hoc virtual trial rate sweep")
    print("=" * 70)
    print(f"Signal duration : {state['duration_s']:.1f} s | "
          f"Non-blanked: {100 * state['blank_mask'].mean():.1f}%")
    print(f"Background stats: mean={state['bg_mean']:.2f}  "
          f"Q1={state['gm_q1']:.2f}  Med={state['gm_med']:.2f}  "
          f"Q3={state['gm_q3']:.2f} µV\n")

    duration_h = state['duration_s'] / 3600.0

    # Pre-compute grand means for every bin position once; then simulate the
    # accept/reject walk for each (centre, hw) pair with a lightweight loop
    # instead of re-processing the raw signal 30 times.
    n_win_bins = int(round((min_ms + max_ms) / 2 / BIN_DURATION_MS))
    gm_series, bin_samp = _build_bin_gm_series(
        state['continuous_abs_emg'], state['blank_mask'],
        state['sample_rate'], state['min_valid_frac'],
        n_win_bins=n_win_bins,
    )
    n_gm = len(gm_series)

    sweep_results: dict = {}
    for centre_label, centre_val in sweep_centres.items():
        sweep_results[centre_label] = []
        for hw in half_widths_uv:
            min_uv = max(0.0, centre_val - hw)
            max_uv = centre_val + hw

            rng = Random(seed)
            k = 0
            n_acc = 0
            while k < n_gm:
                dur_ms   = round_to_nearest_multiple(rng.randint(min_ms, max_ms),
                                                     base=BIN_DURATION_MS)
                dur_bins = dur_ms // BIN_DURATION_MS
                gm = gm_series[k]
                if not np.isnan(gm) and min_uv <= gm <= max_uv:
                    k     += dur_bins
                    n_acc += 1
                else:
                    k += 1  # advance one bin on rejection

            tph = n_acc / duration_h if duration_h > 0 else 0.0
            sweep_results[centre_label].append({
                'hw': hw, 'min_uv': min_uv, 'max_uv': max_uv,
                'n_accepted': n_acc,
                'trials_per_hour': tph,
            })

        row_str = '  '.join(
            f'±{r["hw"]}→{r["trials_per_hour"]:.1f}/hr' for r in sweep_results[centre_label]
        )
        print(f"  {centre_label:6s} (centre={centre_val:.2f} µV):  {row_str}")

    if n_trials:
        print(f"\n  Actual HRS2 trials recorded during session: {n_trials}")
    return sweep_results


def plot_threshold_sweep(sweep_results, state, trials=None, header=None):
    """Visualise sweep results from run_threshold_sweep.

    Two panels:
      Left  — virtual trial rate (trials/hour) vs EMG window half-width,
              one line per centre (Q1, Median, Q3).
      Right — per-trial background histogram with quartile-centre dashed lines
              and threshold-window spans at the median half-width.
    Then prints a summary table.
    """
    import matplotlib.pyplot as plt
    if not sweep_results:
        print("No sweep results to plot.")
        return

    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette = ['darkorange', 'purple', 'steelblue', 'teal', 'crimson']
    centre_keys = {'Q1': 'gm_q1', 'Median': 'gm_med', 'Q3': 'gm_q3'}

    def _colour(label, i):
        return colours.get(label, palette[i % len(palette)])

    duration_h = state['duration_s'] / 3600.0

    # ── Left panel: trials/hour vs half-width ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    for i, (label, rows) in enumerate(sweep_results.items()):
        hws = [r['hw'] for r in rows]
        tph = [r.get('trials_per_hour', r['n_accepted'] / duration_h)
               for r in rows]
        ax.plot(hws, tph, 'o-', label=label, color=_colour(label, i), linewidth=1.8)
    ax.set_xlabel('EMG window half-width (±µV)', fontsize=11)
    ax.set_ylabel('Virtual trial rate (trials / hour)', fontsize=11)
    ax.set_title('Post-Hoc Sweep: Trial Rate vs Threshold Window Width')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Right panel: background histogram with window overlays ───────────────
    ax2 = axes[1]
    bg_vals = state['trial_bg_gm']
    bg_valid = bg_vals[~np.isnan(bg_vals)] if hasattr(bg_vals, '__len__') else bg_vals
    ax2.hist(bg_valid, bins=80, color='lightgray', edgecolor='black',
             linewidth=0.5, alpha=0.9, label='Per-trial pre-stim BG')

    for i, (label, rows) in enumerate(sweep_results.items()):
        col = _colour(label, i)
        mid_row = rows[len(rows) // 2]
        tph_mid = mid_row.get('trials_per_hour', mid_row['n_accepted'] / duration_h)
        ax2.axvspan(mid_row['min_uv'], mid_row['max_uv'], alpha=0.20, color=col,
                    label=f"{label} ±{mid_row['hw']} µV → {tph_mid:.0f}/hr")
        centre_val = state.get(centre_keys.get(label, ''))
        if centre_val is not None:
            ax2.axvline(centre_val, color=col, linestyle='--', linewidth=1.5)

    if 'trial_min_th' in state and 'trial_max_th' in state:
        app_lo = float(np.mean(state['trial_min_th']))
        app_hi = float(np.mean(state['trial_max_th']))
        ax2.axvspan(app_lo, app_hi, alpha=0.15, color='red',
                    label=f"App threshold [{app_lo:.0f}–{app_hi:.0f} µV]")

    ax2.axvline(state['bg_mean'], color='black', linestyle='-', linewidth=2.0,
                label=f"Global mean = {state['bg_mean']:.1f} µV")
    ax2.set_xlabel('Pre-stim grand mean (µV)', fontsize=11)
    ax2.set_ylabel('Trial count', fontsize=11)
    ax2.set_title('Background Distribution with Threshold Windows\n'
                  '(shaded spans at median half-width)')
    ax2.legend(fontsize=8)

    title = (f'{header.subject_id} — Post-Hoc Virtual Trial Rate Sweep'
             if header else 'Post-Hoc Virtual Trial Rate Sweep')
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\nSession duration: {state['duration_s']:.0f} s  ({duration_h:.3f} h)")
    print(f"Global EMG grand mean = {state['bg_mean']:.3f} µV  "
          f"(std={state['bg_std']:.3f}, Q1={state['bg_q1']:.3f}, "
          f"median={state['bg_med']:.3f}, Q3={state['bg_q3']:.3f})")

    print(f"\n{'Centre':>10} {'±hw (µV)':>10} {'Min (µV)':>10} {'Max (µV)':>10} "
          f"{'Accepted':>10} {'Trials/hr':>12}")
    print("-" * 67)
    for label, rows in sweep_results.items():
        for r in rows:
            tph = r.get('trials_per_hour', r['n_accepted'] / duration_h)
            print(f"{label:>10} {r['hw']:>10} {r['min_uv']:>10.2f} {r['max_uv']:>10.2f} "
                  f"{r['n_accepted']:>10} {tph:>12.1f}")
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
        'mra'  -- mean rectified average = mean(|emg|)

    Returns a dict with 1-D arrays of length ``len(trials)``:
        m_size, h_size  -- response size in µV (per ``metric``)
        m_norm, h_norm  -- size / bg_divisor  (NaN where divisor <= 0)
    plus the divisor and window/metric used for downstream labelling.
    """
    if metric not in ('ptp', 'peak', 'mra'):
        raise ValueError(f"metric must be 'ptp', 'peak', or 'mra', got {metric!r}")

    def _size(values):
        if metric == 'ptp':
            return float(np.ptp(values))
        if metric == 'mra':
            return float(np.mean(np.abs(values)))
        return float(np.max(np.abs(values)))

    n = len(trials)
    m_size = np.full(n, np.nan)
    h_size = np.full(n, np.nan)
    for i, trial in enumerate(trials):
        t_ms, emg, _, _, _ = get_trial_window(trial, pre_ms, post_ms)
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
                          quartile_results=None,
                          show_stim_adc: bool = False):
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
    _has_stim_adc = any(
        len(getattr(t, 'stim_adc_data', [])) > 1 for t in trials
    )
    stim_adc_toggle = ToggleButton(
        value=show_stim_adc, description='Stim ADC overlay',
        button_style='warning' if _has_stim_adc else '',
        tooltip='Overlay raw stimulator ADC waveform on a secondary axis (file_version >= 4)',
        layout={'width': '190px'},
        disabled=not _has_stim_adc,
    )
    out = Output()

    def _draw(centre, m_idx, q_mode, stim_adc_on=False):
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
            stim_adc_stack = []
            for idx in shown:
                t_ms, emg, _, _, stim_adc = get_trial_window(trials[idx], pre_ms, post_ms)
                if t_ref is None:
                    t_ref = t_ms
                if len(emg) == len(t_ref):
                    ax.plot(t_ref, emg, color='red', alpha=0.15, linewidth=0.6)
                    stack.append(emg)
                    if stim_adc_on and stim_adc is not None and len(stim_adc) == len(t_ref):
                        stim_adc_stack.append(stim_adc)

            if stack:
                arr = np.full((len(stack), len(t_ref)), np.nan)
                for k, emg in enumerate(stack):
                    n = min(len(emg), len(t_ref))
                    arr[k, :n] = emg[:n]
                avg = np.nanmean(arr, axis=0)
                ax.plot(t_ref, avg, color='black', linewidth=2.5, label='Average')

            if stim_adc_on and stim_adc_stack:
                ax2 = ax.twinx()
                sa_arr = np.full((len(stim_adc_stack), len(t_ref)), np.nan)
                for k, sa in enumerate(stim_adc_stack):
                    n = min(len(sa), len(t_ref))
                    sa_arr[k, :n] = sa[:n]
                sa_avg = np.nanmean(sa_arr, axis=0)
                sa_center = float(np.nanmean(sa_avg))
                for row in sa_arr:
                    ax2.plot(t_ref, row - sa_center, color='magenta', alpha=0.12, linewidth=0.5)
                ax2.plot(t_ref, sa_avg - sa_center, color='magenta',
                         linewidth=1.8, label='Stim ADC avg')
                ax2.set_ylabel('Stim ADC (V)', color='magenta', fontsize=9)
                ax2.tick_params(axis='y', labelcolor='magenta', labelsize=8)

            extra = f' (showing first {max_overlay} of {len(indices)})' \
                    if len(indices) > max_overlay else ''
            ax.set_xlabel('Time re: stim onset (ms)')
            ax.tick_params(axis='x', width=2.0)
            ax.spines['bottom'].set_linewidth(2.5)
            ax.set_ylabel('Bipolar EMG (µV)')
            ax.set_title(title + extra)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def _on_centre(c):
        if c['name'] == 'value':
            _draw(c['new'], marker_slider.value, quartile_toggle.value,
                  stim_adc_toggle.value)

    def _on_marker(c):
        if c['name'] == 'value':
            _draw(centre_dd.value, c['new'], quartile_toggle.value,
                  stim_adc_toggle.value)

    def _on_toggle(c):
        if c['name'] == 'value':
            marker_slider.disabled = bool(c['new'])
            _draw(centre_dd.value, marker_slider.value, c['new'],
                  stim_adc_toggle.value)

    def _on_stim_adc(c):
        if c['name'] == 'value':
            _draw(centre_dd.value, marker_slider.value, quartile_toggle.value,
                  c['new'])

    centre_dd.observe(_on_centre, names='value')
    marker_slider.observe(_on_marker, names='value')
    quartile_toggle.observe(_on_toggle, names='value')
    stim_adc_toggle.observe(_on_stim_adc, names='value')

    display(VBox([HBox([centre_dd, marker_slider, quartile_toggle, stim_adc_toggle]), out]))
    _draw(centre_dd.value, marker_slider.value, quartile_toggle.value,
          stim_adc_toggle.value)


# ---------------------------------------------------------------------------
# Section 5 — quartile window comparison
# ---------------------------------------------------------------------------

def plot_quartile_window_comparison(
        trials, emg_blocks, state, header,
        hw_uv=20.0,
        pre_ms=2.0, post_ms=15.0,
        bg_pre_ms=2500.0,
        m_start_ms=3.0, m_end_ms=5.5,
        h_start_ms=6.5, h_end_ms=11.5,
        sample_rate=SAMPLE_RATE,
        max_overlay=200):
    """Section 5: compare trials selected by Q1 / Median / Q3 windows at a fixed EMG window width.

    Plot 1 — histogram of trial background distribution with shaded ±hw_uv windows.
    Interactive viewer — dropdown selects Q1 / Median / Q3 / All; shows:
        Left  : peri-stimulus waveforms  (-pre_ms … post_ms ms, M/H shading)
        Right : bg_pre_ms ms of pre-stimulus background
        Both panels: individual traces (faint, group colour), group avg (black) + SEM,
                     overall avg of ALL recording trials (blue).
    """
    import matplotlib.pyplot as plt
    from ipywidgets import (Dropdown, Output, VBox, HBox,
                            ToggleButton, FloatText)
    from IPython.display import display

    trial_bg = np.asarray(state['trial_bg_gm'], dtype=float)
    valid_bg = trial_bg[~np.isnan(trial_bg)]
    gm_q1    = float(state['gm_q1'])
    gm_med   = float(state['gm_med'])
    gm_q3    = float(state['gm_q3'])

    group_defs = {           # label -> (centre, colour)
        'Q1':     (gm_q1,  'darkorange'),
        'Median': (gm_med, 'purple'),
        'Q3':     (gm_q3,  'steelblue'),
    }

    # ── trial indices per group ───────────────────────────────────────────────
    group_indices: dict = {}
    for lbl, (centre, _) in group_defs.items():
        lo = max(0.0, centre - hw_uv)
        hi = centre + hw_uv
        mask = (trial_bg >= lo) & (trial_bg <= hi) & ~np.isnan(trial_bg)
        group_indices[lbl] = np.where(mask)[0].tolist()
    group_indices['All'] = sorted(
        set(group_indices['Q1']) | set(group_indices['Median']) | set(group_indices['Q3'])
    )

    group_colours = {lbl: v[1] for lbl, v in group_defs.items()}
    group_colours['All'] = 'dimgray'

    # ── Plot 1: histogram with shaded windows ─────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.hist(valid_bg, bins=80, color='lightgray',
             edgecolor='black', linewidth=0.5, alpha=0.9)
    for lbl, (centre, col) in group_defs.items():
        lo = max(0.0, centre - hw_uv)
        hi = centre + hw_uv
        ax1.axvspan(lo, hi, color=col, alpha=0.25,
                    label=f'{lbl}  [{lo:.1f}, {hi:.1f}] µV  '
                          f'n={len(group_indices[lbl])}')
        ax1.axvline(centre, color=col, linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Pre-stim background grand mean (µV)')
    ax1.set_ylabel('Trial count')
    ax1.set_title(
        f'Trial Background Distribution — {header.subject_id}\n'
        f'EMG window width = ±{hw_uv} µV  |  '
        f'Q1={gm_q1:.1f}  Median={gm_med:.1f}  Q3={gm_q3:.1f} µV')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # ── pre-compute abs-val background bins (Most Recent Background) ──────────
    print(f"Pre-computing {bg_pre_ms:.0f} ms background bins for {len(trials)} trials…",
          end=' ', flush=True)
    bins_cache = [None] * len(trials)
    for i, t in enumerate(trials):
        bins_arr, _ = compute_background_bins(
            t, emg_blocks, monitoring_window_ms=bg_pre_ms, sample_rate=sample_rate)
        bins_cache[i] = bins_arr
    print("done.")

    # ── pre-compute overall peri-stim average (all trials) ────────────────────
    print("Pre-computing overall averages…", end=' ', flush=True)
    t_stim_ref = None
    _sum_s = None
    _cnt_s = 0
    for i, t in enumerate(trials):
        tm, emg, _, _, _ = get_trial_window(t, pre_ms, post_ms)
        if t_stim_ref is None:
            t_stim_ref = tm
            _sum_s = np.zeros(len(tm), dtype=float)
        if len(emg) == len(t_stim_ref):
            _sum_s += emg
            _cnt_s += 1
    avg_s_overall = _sum_s / _cnt_s if _cnt_s > 0 else None
    print(f"done  ({_cnt_s} peri-stim trials).")

    # ── widget state ──────────────────────────────────────────────────────────
    _auto = {'val': True}
    _man  = {'lo': -2000.0, 'hi': 2000.0}

    group_dd = Dropdown(options=['Q1', 'Median', 'Q3', 'All'],
                        value='Q1', description='Group:',
                        layout={'width': '180px'})
    auto_btn = ToggleButton(value=True, description='Auto y-scale',
                            button_style='success', layout={'width': '140px'})
    ymin_box = FloatText(value=-2000.0, description='Y min:',
                         disabled=True, layout={'width': '150px'})
    ymax_box = FloatText(value=2000.0,  description='Y max:',
                         disabled=True, layout={'width': '150px'})
    out = Output()

    def _draw(group):
        with out:
            out.clear_output(wait=True)
            indices = group_indices[group]
            col     = group_colours[group]
            shown   = indices[:max_overlay]

            # peri-stim traces
            t_ref = None
            stim_stack = []
            for idx in shown:
                tm, emg, _, _, _ = get_trial_window(trials[idx], pre_ms, post_ms)
                if t_ref is None:
                    t_ref = tm
                if len(emg) == len(t_ref):
                    stim_stack.append(emg)

            # bins (Most Recent Background) stack
            bins_stack = []
            for idx in shown:
                if idx < len(bins_cache) and bins_cache[idx] is not None:
                    bins_stack.append(np.asarray(bins_cache[idx], dtype=float))

            if not stim_stack:
                print(f"No trials in group {group}.")
                return

            sarr  = np.array(stim_stack)
            avg_s = np.nanmean(sarr, axis=0)
            nv    = np.sum(~np.isnan(sarr), axis=0).astype(float)
            sem_s = (np.nanstd(sarr, axis=0, ddof=1)
                     / np.where(nv > 1, np.sqrt(nv), np.nan))

            avg_bins = sem_bins = gm_group = None
            if bins_stack:
                min_len  = min(len(b) for b in bins_stack)
                bins_arr = np.array([b[:min_len] for b in bins_stack])
                avg_bins = np.nanmean(bins_arr, axis=0)
                nv_bins  = np.sum(~np.isnan(bins_arr), axis=0).astype(float)
                sem_bins = (np.nanstd(bins_arr, axis=0, ddof=1)
                            / np.where(nv_bins > 1, np.sqrt(nv_bins), np.nan))
                gm_group = float(np.nanmean(avg_bins))

            if _auto['val']:
                all_v = np.concatenate(stim_stack)
                all_v = all_v[~np.isnan(all_v)]
                if len(all_v):
                    lo  = float(np.nanmin(all_v))
                    hi  = float(np.nanmax(all_v))
                    pad = max(0.08 * (hi - lo), 1.0)
                    ylim = (lo - pad, hi + pad)
                else:
                    ylim = (-2000.0, 2000.0)
            else:
                ylim = (_man['lo'], _man['hi'])

            extra = (f' (first {max_overlay} of {len(indices)})'
                     if len(indices) > max_overlay else '')
            if group in group_defs:
                c = group_defs[group][0]
                title_str = (f'{group}  [{max(0.0, c - hw_uv):.1f}, '
                             f'{c + hw_uv:.1f}] µV  n={len(indices)}{extra}')
            else:
                title_str = f'All windowed (Q1 ∪ Median ∪ Q3, ±{hw_uv} µV)  n={len(indices)}{extra}'

            fig, (ax_bins, ax_s) = plt.subplots(1, 2, figsize=(18, 5))
            fig.suptitle(f'{header.subject_id} — {title_str}', fontsize=11)

            # left: Most Recent Background (group-averaged abs-EMG bins)
            if avg_bins is not None:
                x = np.arange(len(avg_bins))
                bin_ms_each = bg_pre_ms / len(avg_bins)
                for b in bins_stack:
                    ax_bins.plot(np.arange(len(b[:len(avg_bins)])),
                                 b[:len(avg_bins)],
                                 color=col, alpha=0.12, linewidth=0.5)
                ax_bins.bar(x, avg_bins, width=0.8, color=col, alpha=0.75,
                            yerr=sem_bins, capsize=2,
                            error_kw={'linewidth': 0.8, 'ecolor': 'black'},
                            label=f'Group mean ±SEM (n={len(bins_stack)})')
                ax_bins.axhline(gm_group, color='red', linestyle='--',
                                linewidth=2.0,
                                label=f'Grand mean = {gm_group:.1f} µV')
                ax_bins.set_xlim(-0.5, len(avg_bins) - 0.5)
                ax_bins.set_ylim(bottom=0)
                ax_bins.set_xlabel(
                    f'Bin # ({bin_ms_each:.0f} ms each, {bg_pre_ms:.0f} ms total)')
            else:
                ax_bins.text(0.5, 0.5, 'No background bin data',
                             ha='center', va='center',
                             transform=ax_bins.transAxes)
                ax_bins.set_xlabel(f'Bin # ({BIN_DURATION_MS:.0f} ms each)')
            ax_bins.set_ylabel('|EMG| (µV)')
            ax_bins.set_title(f'Most Recent Background ({bg_pre_ms:.0f} ms pre-stim)')
            ax_bins.legend(fontsize=8, loc='upper right')
            ax_bins.grid(True, alpha=0.3, axis='y')

            # right: peri-stimulus
            ax_s.axhline(0, color='black', linewidth=0.5, alpha=0.4)
            ax_s.axvline(0, color='red',   linewidth=1.0, linestyle='--', alpha=0.6)
            ax_s.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.15)
            ax_s.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.15)
            for emg in stim_stack:
                ax_s.plot(t_ref, emg, color=col, alpha=0.15, linewidth=0.6)
            if avg_s_overall is not None:
                ax_s.plot(t_stim_ref, avg_s_overall, color='blue', linewidth=2.0,
                          alpha=0.85, zorder=3,
                          label=f'Overall avg — all {_cnt_s} trials')
            ax_s.fill_between(t_ref, avg_s - sem_s, avg_s + sem_s,
                              color='red', alpha=0.25, linewidth=0,
                              label='±SEM', zorder=3)
            ax_s.plot(t_ref, avg_s, color='black', linewidth=2.5,
                      label=f'Group avg (n={len(stim_stack)})', zorder=4)
            ax_s.set_xlim(-pre_ms, post_ms)
            ax_s.set_ylim(ylim)
            ax_s.set_xlabel('Time re: stim onset (ms)')
            ax_s.set_ylabel('Bipolar EMG (µV)')
            ax_s.set_title('Peri-stimulus')
            ax_s.set_xticks(np.arange(int(np.floor(-pre_ms)),
                                      int(np.ceil(post_ms)) + 1, 1))
            ax_s.tick_params(axis='x', width=2.0)
            ax_s.spines['bottom'].set_linewidth(2.5)
            ax_s.legend(fontsize=8, loc='upper right')
            ax_s.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def _on_group(c):
        if c['name'] == 'value':
            _draw(c['new'])

    def _on_auto(c):
        if c['name'] == 'value':
            _auto['val'] = bool(c['new'])
            ymin_box.disabled = bool(c['new'])
            ymax_box.disabled = bool(c['new'])
            _draw(group_dd.value)

    def _on_ymin(c):
        _man['lo'] = float(c['new'])
        if not _auto['val']:
            _draw(group_dd.value)

    def _on_ymax(c):
        _man['hi'] = float(c['new'])
        if not _auto['val']:
            _draw(group_dd.value)

    group_dd.observe(_on_group, names='value')
    auto_btn.observe(_on_auto,  names='value')
    ymin_box.observe(_on_ymin,  names='value')
    ymax_box.observe(_on_ymax,  names='value')

    display(VBox([HBox([group_dd, auto_btn, ymin_box, ymax_box]), out]))
    _draw(group_dd.value)


# ---------------------------------------------------------------------------
# Section 6 — per-trial MH metrics (per-trial background normalisation)
# ---------------------------------------------------------------------------

def _mh_binned_mra(emg_window, sample_rate, bin_ms=0.2):
    """Compute M or H wave peak using the same binning pipeline as Sections 3/4/5.

    Analogous to:
        Section 3/4: time window (2500 ms) → 50 ms bins → mean(abs_val per bin) → grand mean
        Section 5:   µV window (±hw)       → 1 µV bins  → mean(trial_bg per bin) → grand mean
        Section 6:   M/H time window       → 0.2 ms bins → abs(mean per bin) → grand mean

    abs is applied AFTER averaging within each bin (abs-last principle).
    At 5000 Hz, 0.2 ms = 1 sample per bin, which is equivalent to mean(abs(samples)).
    """
    bin_samples = max(1, int(bin_ms * sample_rate / 1000.0))
    n_bins = len(emg_window) // bin_samples
    if n_bins == 0:
        return float(np.mean(np.abs(emg_window))) if len(emg_window) > 0 else float('nan')
    trimmed = emg_window[:n_bins * bin_samples].reshape(n_bins, bin_samples)
    bin_abs_means = np.abs(trimmed.mean(axis=1))  # abs of mean per bin (abs last)
    return float(bin_abs_means.mean())             # grand mean across bins


def compute_mh_metrics(trials, trial_bg_gm,
                       m_start_ms=3.0, m_end_ms=5.5,
                       h_start_ms=6.5, h_end_ms=11.5,
                       pre_ms=2.0, post_ms=15.0,
                       sample_rate=SAMPLE_RATE, mh_bin_ms=0.2):
    """Per-trial M-/H-wave metrics normalised to each trial's own pre-stim background.

    M and H wave peaks use the same binning pipeline as Sections 3/4/5:
        M/H time window → 0.2 ms bins → abs(mean per bin) → grand mean = MRA peak

    For every trial computes:
        m_mra     -- M-wave peak via 0.2 ms binned grand mean  (µV)
        h_mra     -- H-wave peak via 0.2 ms binned grand mean  (µV)
        hm_ratio  -- h_mra / m_mra  (NaN when m_mra == 0)
        m_size    -- (m_mra - per_trial_bg) / per_trial_bg  (units of background)
        h_size    -- (h_mra - per_trial_bg) / per_trial_bg  (units of background)

    Returns a dict of 1-D arrays of length len(trials).
    """
    n = len(trials)
    bg = np.asarray(trial_bg_gm, dtype=float)
    if bg.size != n:
        raise ValueError(f"trial_bg_gm length {bg.size} != n_trials {n}")

    m_mra    = np.full(n, np.nan)
    h_mra    = np.full(n, np.nan)
    hm_ratio = np.full(n, np.nan)
    m_size   = np.full(n, np.nan)
    h_size   = np.full(n, np.nan)

    for i, trial in enumerate(trials):
        t_ms, emg, _, _, _ = get_trial_window(trial, pre_ms, post_ms)
        m_mask = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
        h_mask = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)

        mm = _mh_binned_mra(emg[m_mask], sample_rate, mh_bin_ms) if m_mask.any() else float('nan')
        hm = _mh_binned_mra(emg[h_mask], sample_rate, mh_bin_ms) if h_mask.any() else float('nan')
        b  = float(bg[i])

        m_mra[i] = mm
        h_mra[i] = hm
        if np.isfinite(mm) and np.isfinite(hm) and mm > 0:
            hm_ratio[i] = hm / mm
        if np.isfinite(mm) and b > 0:
            m_size[i] = (mm - b) / b
        if np.isfinite(hm) and b > 0:
            h_size[i] = (hm - b) / b

    return {
        'm_mra': m_mra, 'h_mra': h_mra,
        'hm_ratio': hm_ratio,
        'm_size': m_size, 'h_size': h_size,
        'trial_bg': bg,
        'm_start_ms': m_start_ms, 'm_end_ms': m_end_ms,
        'h_start_ms': h_start_ms, 'h_end_ms': h_end_ms,
        'pre_ms': pre_ms, 'post_ms': post_ms,
        'mh_bin_ms': mh_bin_ms,
    }


def compute_mh_variability_sweep(trial_bg_gm, metrics, sweep_centres,
                                  half_widths_uv=None):
    """For each (centre, half-width) window, compute STD and CV for all five
    M-/H-wave metrics from ``compute_mh_metrics``.

    sweep_centres : dict  label -> centre value (µV)

    Each row in the returned list contains:
        hw, n, indices
        m_mra, h_mra, hm_ratio, m_size, h_size  -- each a dict {std, cv, mean}
    """
    if half_widths_uv is None:
        half_widths_uv = [5, 10, 20, 30, 50, 75, 100, 150]

    bg = np.asarray(trial_bg_gm, dtype=float)

    def _stat(arr):
        v = arr[~np.isnan(arr)]
        if len(v) < 2:
            return {'std': float('nan'), 'cv': float('nan'),
                    'mean': float(v[0]) if len(v) == 1 else float('nan')}
        std  = float(np.std(v, ddof=1))
        mean = float(np.mean(v))
        cv   = std / mean if mean != 0 else float('nan')
        return {'std': std, 'cv': cv, 'mean': mean}

    metric_keys = ['m_mra', 'h_mra', 'hm_ratio', 'm_size', 'h_size']
    arrays = {k: np.asarray(metrics[k], dtype=float) for k in metric_keys}

    valid_bg = bg[~np.isnan(bg)]

    results: dict = {}
    for label, centre in sweep_centres.items():
        rows = []
        for hw in half_widths_uv:
            min_uv = max(0.0, centre - hw)
            max_uv = centre + hw
            in_win = (bg >= min_uv) & (bg <= max_uv)
            indices = np.where(in_win)[0]
            bgm, n_occ = _bg_binned_grand_mean(valid_bg, min_uv, max_uv)
            row: dict = {'hw': hw, 'n': int(len(indices)),
                         'indices': indices.tolist(),
                         'bin_grand_mean': bgm,
                         'n_occupied_bins': n_occ}
            for k in metric_keys:
                row[k] = _stat(arrays[k][in_win])
            rows.append(row)
        results[label] = rows
    return results


def plot_mh_variability(variability_sweep, header, n_markers=10, marker_hws=None):
    """3-row x 2-col figure: STD (left) and CV (right) for M-H Peak (MRA),
    H/M Ratio, and M-H Size -- one curve per sweep centre (Q1/Median/Q3).

    marker_hws : list of float, optional
        Explicit half-widths at which to draw vertical marker lines.
        When provided, n_markers is ignored.  Pass the same half_widths_uv
        list used in compute_mh_variability_sweep to align markers with the
        sweep points.

    Returns the list of marker half-widths.
    """
    import matplotlib.pyplot as plt
    if not variability_sweep:
        print("No variability data.")
        return []

    first_label = next(iter(variability_sweep))
    hws_all = [r['hw'] for r in variability_sweep[first_label]]
    if marker_hws is not None:
        marker_hws = list(marker_hws)
    elif n_markers >= 2 and len(hws_all) > 1:
        marker_hws = list(np.linspace(min(hws_all), max(hws_all), n_markers))
    else:
        marker_hws = list(hws_all[:n_markers])

    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette  = ['darkorange', 'purple', 'steelblue', 'teal', 'crimson']

    def _col(lbl, i):
        return colours.get(lbl, palette[i % len(palette)])

    rows_cfg = [
        ('m_mra',    'h_mra',    'M-H MRA Peak',  'MRA (µV)'),
        ('hm_ratio', None,       'H/M Ratio',      'H/M Ratio'),
        ('m_size',   'h_size',   'M-H Size',       'Size (× background)'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    for row_i, (m_key, h_key, title, ylabel) in enumerate(rows_cfg):
        for col_i, stat in enumerate(['std', 'cv']):
            ax = axes[row_i][col_i]
            for i, (label, sweep_rows) in enumerate(variability_sweep.items()):
                col = _col(label, i)
                hws_i = [r['hw'] for r in sweep_rows]
                if h_key is None:
                    vals = [r[m_key][stat] for r in sweep_rows]
                    ax.plot(hws_i, vals, 'o-', label=label,
                            color=col, linewidth=2.0)
                else:
                    m_vals = [r[m_key][stat] for r in sweep_rows]
                    h_vals = [r[h_key][stat] for r in sweep_rows]
                    ax.plot(hws_i, m_vals, 'o-', label=f'M ({label})',
                            color=col, linewidth=2.0)
                    ax.plot(hws_i, h_vals, 's--', label=f'H ({label})',
                            color=col, linewidth=1.5, alpha=0.85)

            ymin, ymax = ax.get_ylim()
            span = ymax - ymin if ymax > ymin else 1.0
            for k, hw in enumerate(marker_hws):
                ax.axvline(hw, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
                ax.text(hw, ymax - span * 0.03, f'm{k}',
                        color='gray', fontsize=7, ha='center', va='top')

            stat_label = 'STD' if stat == 'std' else 'CV (STD/Mean)'
            ax.set_ylabel(f'{ylabel}\n({stat_label})')
            ax.set_title(f'{title} Variability — {stat_label}')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)

    for col_i in range(2):
        axes[2][col_i].set_xlabel('EMG Window Half-Width (µV)')

    plt.suptitle(f'M-/H-Wave Variability vs EMG Window Size — {header.subject_id}',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()
    return marker_hws


def print_mh_variability_summary(variability_sweep, sweep_centres, header,
                                  marker_hws=None):
    """Print per-(centre, hw) table for STD and CV of all five metrics.

    If marker_hws is given, only the row closest to each marker is printed.
    """
    if not variability_sweep:
        return

    def _v(r, k, s):
        v = r[k].get(s, float('nan'))
        return f"{v:.3f}" if np.isfinite(v) else '  nan'

    print(f"\n=== {header.subject_id} — M-H variability summary ===")
    hdr = (f"{'Centre':>10} {'±hw (µV)':>10} {'n':>6} "
           f"{'M-MRA STD':>11} {'H-MRA STD':>11} "
           f"{'H/M STD':>9} {'M-sz STD':>10} {'H-sz STD':>10} "
           f"{'M-MRA CV':>10} {'H-MRA CV':>10} {'H/M CV':>8}")
    print(hdr)
    print("-" * len(hdr))

    for label, centre in sweep_centres.items():
        rows = variability_sweep.get(label, [])
        if marker_hws is not None:
            hws_arr = np.array([r['hw'] for r in rows], dtype=float)
            selected = [rows[int(np.argmin(np.abs(hws_arr - mhw)))]
                        for mhw in marker_hws]
        else:
            selected = rows
        for r in selected:
            print(f"{label:>10} {r['hw']:>10.1f} {r['n']:>6} "
                  f"{_v(r,'m_mra','std'):>11} {_v(r,'h_mra','std'):>11} "
                  f"{_v(r,'hm_ratio','std'):>9} {_v(r,'m_size','std'):>10} "
                  f"{_v(r,'h_size','std'):>10} "
                  f"{_v(r,'m_mra','cv'):>10} {_v(r,'h_mra','cv'):>10} "
                  f"{_v(r,'hm_ratio','cv'):>8}")
        print()


def _fetch_bg_filtered(trial, emg_blocks, bg_pre_ms, sample_rate=SAMPLE_RATE):
    """Fetch bg_pre_ms ms of filtered bipolar EMG ending at the trial trigger time.

    Mirrors compute_background_bins but returns raw filtered samples instead of
    binned abs values — used to prepend background context to the peri-stim view.
    Returns an ndarray of shape (needed,), or None if unavailable.
    """
    if not emg_blocks:
        return None
    trigger_ms = int(getattr(trial, 'trigger_wall_time_ms', 0) or 0)
    if trigger_ms <= 0:
        return None

    needed = int(bg_pre_ms * sample_rate / 1000.0)
    if needed <= 0:
        return None

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
        return None

    trig_blk = emg_blocks[trig_idx]
    ms_into = trigger_ms - int(trig_blk.ts_open_ephys_sent)
    sample_offset = int(round(ms_into * sample_rate / 1000.0))
    sample_offset = max(0, min(sample_offset, len(trig_blk.filtered)))

    segments = [trig_blk.filtered[:sample_offset]]
    collected = len(segments[0])
    j = trig_idx - 1
    while collected < needed and j >= 0:
        chunk = emg_blocks[j].filtered
        segments.insert(0, chunk)
        collected += len(chunk)
        j -= 1

    if collected < 1:
        return None

    emg_cat = np.concatenate(segments).astype(float)
    if len(emg_cat) > needed:
        emg_cat = emg_cat[-needed:]
    # Pad with NaN at start if insufficient data
    if len(emg_cat) < needed:
        pad = np.full(needed - len(emg_cat), np.nan)
        emg_cat = np.concatenate([pad, emg_cat])
    return emg_cat


def view_mh_bin(variability_sweep, marker_hws, trials, metrics, header,
                pre_ms=2.0, post_ms=12.0,
                m_start_ms=2.0, m_end_ms=5.5,
                h_start_ms=7.0, h_end_ms=11.5,
                max_overlay=200,
                emg_blocks=None, bg_pre_ms=0.0,
                sample_rate=SAMPLE_RATE):
    """Interactive bin browser for M-H metrics.

    Controls (matching plot_hrs2_analysis style):
      - Centre dropdown + Marker slider to select the bin
      - |Bipolar| checkbox: overlays per-trial |EMG| traces (gray) and
        np.abs(avg_bip) — abs is applied AFTER averaging, so the trace
        correctly returns to zero wherever the average goes to zero
      - Overall avg (blue) checkbox: overlays the grand average across ALL
        trials in the recording as a blue reference line
      - Auto y-scale toggle: disabled reveals manual Y-min / Y-max fields

    Each redraw shows:
      - Red individual bipolar trials (up to max_overlay)
      - Black average bipolar (avg of bin trials)
      - Optional gray per-trial |EMG| + dark-gray |avg_bip| (abs applied last)
      - Optional blue overall grand average (all recording trials)
      - Blue / green M / H window shading
      - Dotted vertical lines at M/H window centres with star markers
        and MRA value annotations

    When emg_blocks is provided, bg_pre_ms of raw filtered background is
    prepended to each trial's waveform so the monitoring window context is
    visible alongside the peri-stimulus response.
    """
    import matplotlib.pyplot as plt
    from ipywidgets import (Dropdown, IntSlider, Output, VBox, HBox,
                            Checkbox, ToggleButton, FloatText, HTML, Label)
    from IPython.display import display

    if not variability_sweep or not marker_hws:
        print("No variability data or markers.")
        return

    centre_labels = list(variability_sweep.keys())

    # ── Step 1: fetch background signals for each trial ──────────────────────
    _bg_cache: list = [None] * len(trials)
    _has_bg = False
    _n_bg_samp = 0
    _t_bg_ref  = np.array([])
    if emg_blocks is not None:
        print(f"Pre-computing {bg_pre_ms:.0f} ms background for each trial…",
              end=' ', flush=True)
        n_bg_ok = 0
        for _i, _t in enumerate(trials):
            _bg = _fetch_bg_filtered(_t, emg_blocks, bg_pre_ms, sample_rate)
            _bg_cache[_i] = _bg
            if _bg is not None:
                n_bg_ok += 1
        _has_bg = n_bg_ok > 0
        _n_bg_samp = int(bg_pre_ms * sample_rate / 1000.0)
        _dt = 1000.0 / sample_rate  # ms per sample
        _t_bg_ref = np.arange(-_n_bg_samp, 0) * _dt - pre_ms
        print(f"done  ({n_bg_ok}/{len(trials)} trials).")

    # ── Step 2: overall averages — peri-stim + full-window (bg + peri-stim) ──
    print("Pre-computing overall averages…", end=' ', flush=True)
    _t_overall   = None
    _sum_stim    = None   # peri-stim only accumulator
    _cnt_all     = 0
    _t_full_overall = None
    _sum_full    = None   # bg + peri-stim accumulator
    _cnt_full    = 0
    for _i, _t in enumerate(trials):
        _tm, _emg, _, _, _ = get_trial_window(_t, pre_ms, post_ms)
        # peri-stim running sum
        if _t_overall is None:
            _t_overall = _tm
            _sum_stim  = np.zeros(len(_tm), dtype=float)
        if len(_emg) == len(_t_overall):
            _sum_stim += _emg
            _cnt_all  += 1
        # full-window running sum (bg + peri-stim)
        if _has_bg and _bg_cache[_i] is not None:
            _bg = np.asarray(_bg_cache[_i], dtype=float)
            if len(_bg) == _n_bg_samp and len(_emg) == len(_t_overall):
                _full_seg = np.concatenate([_bg, _emg])
                if _sum_full is None:
                    _t_full_overall = np.concatenate([_t_bg_ref, _tm])
                    _sum_full  = np.zeros(len(_full_seg), dtype=float)
                    _cnt_full_arr = np.zeros(len(_full_seg), dtype=float)
                if len(_full_seg) == len(_sum_full):
                    nan_ok = ~np.isnan(_full_seg)
                    _sum_full += np.where(nan_ok, _full_seg, 0.0)
                    _cnt_full_arr += nan_ok.astype(float)
                    _cnt_full += 1
    _avg_overall = (_sum_stim / _cnt_all) if _cnt_all > 0 else None
    _avg_full_overall = (
        _sum_full / np.where(_cnt_full_arr > 0, _cnt_full_arr, np.nan)
        if _sum_full is not None else None
    )
    print(f"done  (peri-stim: {_cnt_all}, full-window: {_cnt_full} trials).")

    # ── widget state ──────────────────────────────────────────────────────────
    _show_abs     = {'val': False}
    _show_overall = {'val': False}
    _ylim_auto    = {'val': True}
    _ylim_man     = {'lo': -5000.0, 'hi': 5000.0}

    # ── widgets ───────────────────────────────────────────────────────────────
    centre_dd = Dropdown(options=centre_labels, value=centre_labels[0],
                         description='Centre:', layout={'width': '200px'})
    marker_slider = IntSlider(
        value=len(marker_hws) // 2, min=0, max=len(marker_hws) - 1, step=1,
        description='Marker:', layout={'width': '450px'},
        continuous_update=False)
    _cb_abs_bip  = Checkbox(value=False, description='|Bipolar| (gray)',
                            indent=False, layout={'width': '175px'})
    _cb_overall  = Checkbox(value=False,
                            description=f'Overall avg — all {_cnt_all} trials (blue)',
                            indent=False, layout={'width': '310px'})
    _auto_toggle = ToggleButton(
        value=True, description='Auto y-scale', button_style='success',
        tooltip='Auto-scale y-axis from all visible signals')
    _ymin_box = FloatText(value=-5000.0, description='Y min:',
                          disabled=True, layout={'width': '150px'})
    _ymax_box = FloatText(value=5000.0,  description='Y max:',
                          disabled=True, layout={'width': '150px'})
    out = Output()

    # ── draw function ─────────────────────────────────────────────────────────
    def _draw(centre, m_idx):
        with out:
            out.clear_output(wait=True)
            target_hw = marker_hws[m_idx]
            rows = variability_sweep[centre]
            hws  = np.array([r['hw'] for r in rows], dtype=float)
            best = int(np.argmin(np.abs(hws - target_hw)))
            row  = rows[best]
            indices = row['indices']

            # stats header
            print(f"Centre = {centre}  |  marker {m_idx}  |  "
                  f"hw = {row['hw']:.1f} µV  |  n = {row['n']}")

            def _fmt(k, s):
                v = row[k].get(s, float('nan'))
                return f"{v:.4f}" if np.isfinite(v) else '   nan'

            for mkey, lbl in [('m_mra',    'M-MRA (µV)    '),
                               ('h_mra',    'H-MRA (µV)    '),
                               ('hm_ratio', 'H/M ratio     '),
                               ('m_size',   'M-size (×bg)  '),
                               ('h_size',   'H-size (×bg)  ')]:
                print(f"  {lbl}  mean={_fmt(mkey,'mean')}  "
                      f"std={_fmt(mkey,'std')}  cv={_fmt(mkey,'cv')}")

            preview = indices[:50]
            print(f"  Trial indices ({len(indices)}): {preview}"
                  f"{' ...' if len(indices) > len(preview) else ''}")

            if not indices:
                return

            # ── build waveform stack (bin trials only) ────────────────────────
            shown = indices[:max_overlay]
            t_ref = None
            stack = []         # peri-stim only (for stats / abs / SEM)
            full_stack = []    # background + peri-stim (for plotting)
            for idx in shown:
                t_ms, emg, _, _, _ = get_trial_window(trials[idx], pre_ms, post_ms)
                if t_ref is None:
                    t_ref = t_ms
                if len(emg) != len(t_ref):
                    continue
                stack.append(emg)
                if _has_bg and idx < len(_bg_cache) and _bg_cache[idx] is not None:
                    bg_seg = np.asarray(_bg_cache[idx], dtype=float)
                    if len(bg_seg) == len(_t_bg_ref):
                        full_stack.append(np.concatenate([bg_seg, emg]))
                    else:
                        full_stack.append(emg)
                else:
                    full_stack.append(emg)

            if not stack:
                return

            # Full time axis: background (if available) + peri-stim
            if _has_bg and len(_t_bg_ref) > 0:
                t_plot = np.concatenate([_t_bg_ref, t_ref])
            else:
                t_plot = t_ref

            # Pad full_stack entries that lack background (peri-stim only)
            n_bg_pad = len(t_plot) - len(t_ref)
            padded_stack = []
            for seg in full_stack:
                if len(seg) == len(t_plot):
                    padded_stack.append(seg)
                else:
                    pad = np.full(n_bg_pad, np.nan)
                    padded_stack.append(np.concatenate([pad, seg]))

            # ── full-window array (background + peri-stim) for all averages ────
            full_arr = np.full((len(padded_stack), len(t_plot)), np.nan)
            for k, seg in enumerate(padded_stack):
                n_pts = min(len(seg), len(t_plot))
                full_arr[k, :n_pts] = seg[:n_pts]
            avg_full  = np.nanmean(full_arr, axis=0)
            n_valid_f = np.sum(~np.isnan(full_arr), axis=0).astype(float)
            sem_full  = (np.nanstd(full_arr, axis=0, ddof=1)
                         / np.where(n_valid_f > 1, np.sqrt(n_valid_f), np.nan))

            # Peri-stim slice of the full average (for MRA marker y-positions)
            n_bg_pts = len(t_plot) - len(t_ref)
            avg_bip  = avg_full[n_bg_pts:]

            # abs of full average — applied LAST so trace reaches zero correctly
            avg_abs_full = np.abs(avg_full)

            # ── y-axis limits ─────────────────────────────────────────────────
            if _ylim_auto['val']:
                ylim_arrs = [seg for seg in padded_stack]
                if _show_abs['val']:
                    ylim_arrs += [np.abs(seg) for seg in padded_stack]
                if _show_overall['val']:
                    ref = _avg_full_overall if _avg_full_overall is not None else _avg_overall
                    if ref is not None:
                        ylim_arrs.append(ref)
                all_v = np.concatenate(ylim_arrs)
                all_v = all_v[~np.isnan(all_v)]
                if len(all_v):
                    lo  = float(np.nanmin(all_v))
                    hi  = float(np.nanmax(all_v))
                    pad = max(0.08 * (hi - lo), 1.0)
                    ylim = (lo - pad, hi + pad)
                else:
                    ylim = (-1000.0, 1500.0)
            else:
                ylim = (_ylim_man['lo'], _ylim_man['hi'])

            # ── plot ──────────────────────────────────────────────────────────
            fig_w = 20 if _has_bg else 14
            fig, ax = plt.subplots(figsize=(fig_w, 5))
            ax.axhline(0, color='black', linewidth=0.6, linestyle='-',
                       alpha=0.4, zorder=1)
            ax.axvline(0, color='red', linewidth=1.0, linestyle='--', alpha=0.6)
            ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.20)
            ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.20)

            # individual bipolar traces (red, faint) — full window
            for seg in padded_stack:
                ax.plot(t_plot, seg, color='red', alpha=0.15, linewidth=0.6)

            # overall grand average (blue) — full window when available
            if _show_overall['val']:
                if _avg_full_overall is not None:
                    ax.plot(_t_full_overall, _avg_full_overall, color='blue',
                            linewidth=2.0, alpha=0.85, zorder=3,
                            label=f'Overall avg — all {_cnt_all} trials (full window)')
                elif _avg_overall is not None:
                    ax.plot(_t_overall, _avg_overall, color='blue', linewidth=2.0,
                            alpha=0.85, zorder=3,
                            label=f'Overall avg — all {_cnt_all} trials (peri-stim)')

            # bin group average (black) + SEM — continuous across full window
            ax.fill_between(t_plot, avg_full - sem_full, avg_full + sem_full,
                            color='red', alpha=0.25, linewidth=0,
                            label='± SEM', zorder=3)
            ax.plot(t_plot, avg_full, color='black', linewidth=2.5,
                    label='Group avg (full window)', zorder=4)

            # |Bipolar| overlay — abs applied LAST, across full window
            if _show_abs['val']:
                for seg in padded_stack:
                    ax.plot(t_plot, np.abs(seg), color='gray',
                            alpha=0.15, linewidth=0.5)
                ax.plot(t_plot, avg_abs_full, color='dimgray', linewidth=2.0,
                        label='|Group avg| (abs last)', zorder=5)

            # ── MRA window-centre markers ─────────────────────────────────────
            m_t = (m_start_ms + m_end_ms) / 2.0
            h_t = (h_start_ms + h_end_ms) / 2.0
            m_mra_val = row['m_mra'].get('mean', float('nan'))
            h_mra_val = row['h_mra'].get('mean', float('nan'))

            m_mask2 = (t_ref >= m_start_ms) & (t_ref <= m_end_ms)
            h_mask2 = (t_ref >= h_start_ms) & (t_ref <= h_end_ms)
            m_ci    = int(len(t_ref[m_mask2]) // 2) if m_mask2.any() else 0
            h_ci    = int(len(t_ref[h_mask2]) // 2) if h_mask2.any() else 0
            m_bip   = float(avg_bip[m_mask2][m_ci]) if m_mask2.any() else float('nan')
            h_bip   = float(avg_bip[h_mask2][h_ci]) if h_mask2.any() else float('nan')

            m_lbl = (f'M-MRA: {m_mra_val:.1f} µV'
                     if np.isfinite(m_mra_val) else 'M window centre')
            h_lbl = (f'H-MRA: {h_mra_val:.1f} µV'
                     if np.isfinite(h_mra_val) else 'H window centre')
            ax.axvline(m_t, color='blue',  linestyle=':', linewidth=1.5, label=m_lbl)
            ax.axvline(h_t, color='green', linestyle=':', linewidth=1.5, label=h_lbl)

            text_off = (ylim[1] - ylim[0]) * 0.06
            if np.isfinite(m_bip):
                ax.plot(m_t, m_bip, '*', color='blue', markersize=12,
                        markeredgecolor='darkblue', markeredgewidth=0.5, zorder=6)
                if np.isfinite(m_mra_val):
                    ax.text(m_t, m_bip + text_off, f'{m_mra_val:.1f} µV',
                            color='blue', fontsize=9, ha='center')
            if np.isfinite(h_bip):
                ax.plot(h_t, h_bip, '*', color='green', markersize=12,
                        markeredgecolor='darkgreen', markeredgewidth=0.5, zorder=6)
                if np.isfinite(h_mra_val):
                    ax.text(h_t, h_bip + text_off, f'{h_mra_val:.1f} µV',
                            color='green', fontsize=9, ha='center')

            # ── axes labels, ticks, limits ────────────────────────────────────
            extra = (f' (first {max_overlay} of {len(indices)})'
                     if len(indices) > max_overlay else '')
            x_lo = float(t_plot[0]) if _has_bg else -pre_ms
            ax.set_xlim(x_lo, post_ms)
            ax.set_ylim(ylim)
            ax.set_xlabel('Time re: stim onset (ms)', fontsize=10)
            ax.set_ylabel('Bipolar EMG (µV)', fontsize=10)
            ax.set_title(f'{header.subject_id} — Centre={centre}, '
                         f'hw=±{row["hw"]} µV, n={row["n"]}{extra}', fontsize=11)
            if _has_bg:
                # coarse ticks for background, fine for peri-stim
                bg_ticks = list(np.arange(int(x_lo), 0, 500))
                stim_ticks = list(np.arange(0, int(np.ceil(post_ms)) + 1, 2))
                ax.set_xticks(bg_ticks + stim_ticks)
                # shade background region lightly to distinguish it visually
                ax.axvspan(x_lo, -pre_ms, color='lightyellow', alpha=0.4,
                           zorder=0, label=f'Background ({bg_pre_ms:.0f} ms)')
            else:
                ax.set_xticks(np.arange(int(np.floor(-pre_ms)),
                                        int(np.ceil(post_ms)) + 1, 1))
            ax.tick_params(labelsize=9)
            ax.tick_params(axis='x', width=2.0)
            ax.spines['bottom'].set_linewidth(2.5)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_centre(c):
        if c['name'] == 'value':
            _draw(c['new'], marker_slider.value)

    def _on_marker(c):
        if c['name'] == 'value':
            _draw(centre_dd.value, c['new'])

    def _on_abs_bip(change):
        _show_abs['val'] = bool(change['new'])
        _draw(centre_dd.value, marker_slider.value)

    def _on_overall(change):
        _show_overall['val'] = bool(change['new'])
        _draw(centre_dd.value, marker_slider.value)

    def _on_auto_toggle(change):
        _ylim_auto['val'] = bool(change['new'])
        _ymin_box.disabled = bool(change['new'])
        _ymax_box.disabled = bool(change['new'])
        _draw(centre_dd.value, marker_slider.value)

    def _on_ymin(change):
        _ylim_man['lo'] = float(change['new'])
        if not _ylim_auto['val']:
            _draw(centre_dd.value, marker_slider.value)

    def _on_ymax(change):
        _ylim_man['hi'] = float(change['new'])
        if not _ylim_auto['val']:
            _draw(centre_dd.value, marker_slider.value)

    centre_dd.observe(_on_centre, names='value')
    marker_slider.observe(_on_marker, names='value')
    _cb_abs_bip.observe(_on_abs_bip, names='value')
    _cb_overall.observe(_on_overall, names='value')
    _auto_toggle.observe(_on_auto_toggle, names='value')
    _ymin_box.observe(_on_ymin, names='value')
    _ymax_box.observe(_on_ymax, names='value')

    _ctrl_row = HBox([centre_dd, marker_slider])
    _sig_row  = HBox([
        VBox([HTML('<b>Signal overlays:</b>'),
              HBox([_cb_abs_bip, _cb_overall])]),
        Label('    '),
        VBox([_auto_toggle, _ymin_box, _ymax_box]),
    ])

    display(VBox([_ctrl_row, _sig_row, out]))
    _draw(centre_dd.value, marker_slider.value)


# ---------------------------------------------------------------------------
# Section 7 — EMG Window Optimization
# ---------------------------------------------------------------------------

def compute_optimization_scores(sweep_results, mh_variability,
                                 variability_keys=None,
                                 variability_stat='cv',
                                 alpha=0.5,
                                 session_duration_s=None):
    """Combine trial count (Section 5) and M-H variability (Section 6) into a
    composite optimization score for every (centre, hw) decision-variable pair.

    Decision variables
    ------------------
    centre  : EMG window location — Q1 / Median / Q3
    hw      : EMG window half-width in µV

    Objectives
    ----------
    1. Maximize  trial_rate   — number of accepted trials (proportional to trials/hour)
    2. Minimize  variability  — M-H response variability (STD or CV)

    Composite score
    ---------------
    score = alpha * trial_score + (1 - alpha) * consistency_score
    Both components are independently normalized to [0, 1].
        alpha = 1.0  -> care only about maximizing trial rate
        alpha = 0.0  -> care only about minimizing variability
        alpha = 0.5  -> equal weight (default)

    Pareto front
    ------------
    A (centre, hw) pair is Pareto-optimal if no other pair has BOTH more trials
    AND lower variability (i.e., it is not dominated).

    Parameters
    ----------
    sweep_results     : dict from run_direct_bg_sweep — {label: [{hw, n, ...}]}
    mh_variability    : dict from compute_mh_variability_sweep — {label: [{hw, metric_dicts}]}
    variability_keys  : metric keys to average into the variability score,
                        e.g. ['m_mra', 'h_mra'] (default) or add 'hm_ratio'
    variability_stat  : 'cv' (default, dimensionless) or 'std'
    alpha             : trial-rate weight in [0, 1]
    session_duration_s: if given, converts n -> trials/hour for display

    Returns
    -------
    (rows, info)
        rows -- list of dicts, one per (centre, hw), with composite scores and flags
        info -- normalization metadata dict
    """
    if variability_keys is None:
        variability_keys = ['m_mra', 'h_mra']

    # Detect whether sweep_results already contain trials_per_hour
    _sample_sr = next(
        (r for rows in sweep_results.values() for r in rows), {}
    )
    _has_tph = 'trials_per_hour' in _sample_sr

    rows = []
    for label in sweep_results:
        sweep_rows = sweep_results[label]
        var_rows   = mh_variability.get(label, [])
        var_by_hw  = {r['hw']: r for r in var_rows}

        for sr in sweep_rows:
            hw = sr['hw']
            n  = int(sr.get('n_matched', sr.get('n_accepted', sr.get('n', 0))))
            vr = var_by_hw.get(hw, {})

            vvals = []
            per_metric = {}
            for k in ['m_mra', 'h_mra', 'hm_ratio', 'm_size', 'h_size']:
                kd  = vr.get(k, {})
                val = kd.get(variability_stat, float('nan'))
                per_metric[f'{k}_{variability_stat}'] = val
                if k in variability_keys and np.isfinite(val):
                    vvals.append(val)

            variability = float(np.mean(vvals)) if vvals else float('nan')
            if _has_tph:
                trial_rate = float(sr['trials_per_hour'])
            elif session_duration_s:
                trial_rate = n / (session_duration_s / 3600.0)
            else:
                trial_rate = float(n)
            row = {
                'centre': label, 'hw': hw, 'n': n,
                'trial_rate': trial_rate,
                'variability': variability,
            }
            row.update(per_metric)
            rows.append(row)

    # --- normalise both objectives to [0, 1] --------------------------------
    tr_arr = np.array([r['trial_rate']  for r in rows], dtype=float)
    vr_arr = np.array([r['variability'] for r in rows], dtype=float)
    tr_fin = tr_arr[np.isfinite(tr_arr)]
    vr_fin = vr_arr[np.isfinite(vr_arr)]
    tr_min = float(np.min(tr_fin)) if len(tr_fin) else 0.0
    tr_max = float(np.max(tr_fin)) if len(tr_fin) else 1.0
    vr_min = float(np.min(vr_fin)) if len(vr_fin) else 0.0
    vr_max = float(np.max(vr_fin)) if len(vr_fin) else 1.0

    for r in rows:
        tr = r['trial_rate']
        vr = r['variability']
        ts = ((tr - tr_min) / (tr_max - tr_min)
              if np.isfinite(tr) and tr_max > tr_min else float('nan'))
        vs_raw = ((vr - vr_min) / (vr_max - vr_min)
                  if np.isfinite(vr) and vr_max > vr_min else float('nan'))
        cs = 1.0 - vs_raw if np.isfinite(vs_raw) else float('nan')
        r['trial_score']       = ts
        r['consistency_score'] = cs
        r['composite_score']   = (alpha * ts + (1.0 - alpha) * cs
                                  if np.isfinite(ts) and np.isfinite(cs)
                                  else float('nan'))

    # --- rank -----------------------------------------------------------
    rows_sorted = sorted(
        rows,
        key=lambda r: -(r['composite_score']
                        if np.isfinite(r.get('composite_score', float('nan')))
                        else -1))
    for rank, r in enumerate(rows_sorted, 1):
        r['rank'] = rank

    # --- Pareto front: maximise trial_rate, minimise variability --------
    valid = [(i, r) for i, r in enumerate(rows)
             if np.isfinite(r['trial_rate']) and np.isfinite(r['variability'])]
    pareto = [False] * len(rows)
    for i, ri in valid:
        dominated = any(
            (rj['trial_rate'] >= ri['trial_rate'] and
             rj['variability'] <= ri['variability'] and
             (rj['trial_rate'] > ri['trial_rate'] or
              rj['variability'] < ri['variability']))
            for j, rj in valid if j != i
        )
        pareto[i] = not dominated
    for i, r in enumerate(rows):
        r['pareto'] = pareto[i]

    info = {
        'alpha': alpha,
        'variability_stat': variability_stat,
        'variability_keys': variability_keys,
        'tr_min': tr_min, 'tr_max': tr_max,
        'vr_min': vr_min, 'vr_max': vr_max,
        'session_duration_s': session_duration_s,
        'has_tph': _has_tph,
    }
    return rows, info


def plot_optimization(opt_rows, opt_info, header):
    """Three-panel EMG window optimization visualization.

    Panel 1 (top, full-width):
        Pareto scatter — trial count vs variability, one point per (centre, hw).
        Point size proportional to window half-width. Pareto-optimal points (not
        dominated by any other pair) are highlighted with a connecting Pareto curve.

    Panel 2 (bottom-left):
        Composite score heatmap — rows = window location (Q1/Median/Q3),
        columns = window half-width. Cells annotated with score; best cell starred.

    Panel 3 (bottom-right):
        Dual-axis trade-off lines — trial count (solid) and variability (dashed)
        vs half-width, one line per centre. Shows the conflicting objectives directly.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if not opt_rows:
        print("No optimization data.")
        return

    alpha      = opt_info.get('alpha', 0.5)
    vstat      = opt_info.get('variability_stat', 'cv')
    vkeys      = opt_info.get('variability_keys', ['m_mra', 'h_mra'])
    dur_s      = opt_info.get('session_duration_s')
    rate_label = ('Trial Rate (trials / hour)'
                  if (dur_s or opt_info.get('has_tph'))
                  else 'Accepted Trial Count (n)')
    vlabel     = f'Variability ({vstat.upper()})'

    centres = list(dict.fromkeys(r['centre'] for r in opt_rows))
    hw_vals = sorted(set(r['hw'] for r in opt_rows))
    colours = {'Q1': 'darkorange', 'Median': 'purple', 'Q3': 'steelblue'}
    palette = ['darkorange', 'purple', 'steelblue']

    def _col(c, i=0):
        return colours.get(c, palette[i % len(palette)])

    hw_min = min(hw_vals)
    hw_max = max(hw_vals)

    # build heatmap grid
    heatmap = np.full((len(centres), len(hw_vals)), np.nan)
    for r in opt_rows:
        ci = centres.index(r['centre'])
        hi = hw_vals.index(r['hw'])
        sc = r.get('composite_score', float('nan'))
        if np.isfinite(sc):
            heatmap[ci, hi] = sc

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.36)
    ax_scatter = fig.add_subplot(gs[0, :])
    ax_heat    = fig.add_subplot(gs[1, 0])
    ax_lines   = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Pareto scatter ──────────────────────────────────────────────
    for i, c in enumerate(centres):
        col = _col(c, i)
        pts = [r for r in opt_rows if r['centre'] == c]
        xs  = [r['trial_rate'] for r in pts]
        ys  = [r['variability'] for r in pts]
        sizes = [40 + 130 * ((r['hw'] - hw_min) / max(hw_max - hw_min, 1))
                 for r in pts]
        ax_scatter.scatter(xs, ys, s=sizes, color=col, alpha=0.80,
                           edgecolors='white', linewidths=0.7, label=c, zorder=3)
        for x, y, pt in zip(xs, ys, pts):
            ax_scatter.annotate(f"±{int(pt['hw'])}µV", (x, y),
                                textcoords='offset points', xytext=(5, 4),
                                fontsize=7.5, color=col)

    pareto_pts = sorted(
        [r for r in opt_rows if r.get('pareto') and
         np.isfinite(r['trial_rate']) and np.isfinite(r['variability'])],
        key=lambda r: r['trial_rate'])
    if pareto_pts:
        px = [r['trial_rate'] for r in pareto_pts]
        py = [r['variability'] for r in pareto_pts]
        ax_scatter.plot(px, py, 'k--', linewidth=1.8, alpha=0.5,
                        label='Pareto front', zorder=2)
        ax_scatter.scatter(px, py, s=70, color='black', marker='D',
                           zorder=4, label='_nolegend_')

    # "ideal" zone annotation (high trial rate, low variability)
    tr_range = opt_info['tr_max'] - opt_info['tr_min']
    vr_range = opt_info['vr_max'] - opt_info['vr_min']
    ideal_x  = opt_info['tr_min'] + 0.88 * tr_range
    ideal_y  = opt_info['vr_min'] + 0.08 * vr_range
    arrow_x  = opt_info['tr_min'] + 0.60 * tr_range
    arrow_y  = opt_info['vr_min'] + 0.30 * vr_range
    ax_scatter.annotate(
        'Ideal:\nhigh trials + low variability',
        xy=(ideal_x, ideal_y), xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
        color='green', fontsize=9, alpha=0.85)

    ax_scatter.set_xlabel(rate_label, fontsize=11)
    ax_scatter.set_ylabel(f'{vlabel}  (avg of {" + ".join(vkeys)})\n'
                          f'<-- lower = more consistent', fontsize=10)
    ax_scatter.set_title(
        f'Optimization Trade-off — {header.subject_id}\n'
        f'Point size proportional to window half-width  |  '
        f'Pareto-optimal points (♦) are non-dominated\n'
        f'Composite score = {alpha:.2f} × trial_score + '
        f'{1 - alpha:.2f} × consistency_score  '
        f'(both normalized to [0, 1])',
        fontsize=10)
    ax_scatter.legend(fontsize=9, loc='upper left')
    ax_scatter.grid(True, alpha=0.3)

    # ── Panel 2: Composite score heatmap ────────────────────────────────────
    im = ax_heat.imshow(heatmap, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=1, origin='upper')
    ax_heat.set_xticks(range(len(hw_vals)))
    ax_heat.set_xticklabels([f'±{int(hw)}' for hw in hw_vals],
                             rotation=45, ha='right', fontsize=9)
    ax_heat.set_yticks(range(len(centres)))
    ax_heat.set_yticklabels(centres, fontsize=10)
    ax_heat.set_xlabel('EMG Window Half-Width (µV)', fontsize=10)
    ax_heat.set_ylabel('Window Location (centre)', fontsize=10)
    ax_heat.set_title(
        f'Composite Score Heatmap  (α = {alpha:.2f})\n'
        f'Green = best trade-off | Red = worst', fontsize=10)
    plt.colorbar(im, ax=ax_heat, label='Composite Score [0–1]', fraction=0.046)
    for ci in range(len(centres)):
        for hi in range(len(hw_vals)):
            val = heatmap[ci, hi]
            if np.isfinite(val):
                txt_col = 'black' if 0.25 < val < 0.85 else 'white'
                ax_heat.text(hi, ci, f'{val:.2f}', ha='center', va='center',
                             fontsize=8.5, color=txt_col, fontweight='bold')
    best_r = max(
        [r for r in opt_rows if np.isfinite(r.get('composite_score', float('nan')))],
        key=lambda r: r['composite_score'],
        default=None)
    if best_r is not None:
        bi = centres.index(best_r['centre'])
        bh = hw_vals.index(best_r['hw'])
        ax_heat.text(bh, bi - 0.4, '★', ha='center', va='center',
                     fontsize=15, color='gold')

    # ── Panel 3: Dual-axis trade-off lines ───────────────────────────────────
    ax2 = ax_lines.twinx()
    for i, c in enumerate(centres):
        col = _col(c, i)
        pts = sorted([r for r in opt_rows if r['centre'] == c],
                     key=lambda r: r['hw'])
        hws = [r['hw'] for r in pts]
        ns  = [r['trial_rate'] for r in pts]
        vs  = [r['variability'] for r in pts]
        ax_lines.plot(hws, ns, 'o-', color=col, linewidth=2.2,
                      label=f'{c}  (trials)')
        ax2.plot(hws, vs, 's--', color=col, linewidth=1.5, alpha=0.65)

    ax_lines.set_xlabel('Window Half-Width (µV)', fontsize=10)
    ax_lines.set_ylabel(rate_label, fontsize=10)
    ax2.set_ylabel(f'{vlabel}  [dashed]', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax_lines.set_title(
        'Trial Count (solid) & Variability (dashed)\nvs Window Size per Centre',
        fontsize=10)
    ax_lines.legend(fontsize=8, loc='upper left')
    ax_lines.grid(True, alpha=0.3)

    plt.suptitle(
        f'EMG Window Optimization — {header.subject_id}\n'
        f'Decision variables: Window Location × Window Half-Width',
        fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def print_optimization_summary(opt_rows, opt_info, header, top_n=5):
    """Print the top-N solutions by composite score, then list the Pareto front
    with the full mathematical breakdown for each candidate.
    """
    alpha   = opt_info.get('alpha', 0.5)
    vstat   = opt_info.get('variability_stat', 'cv')
    vkeys   = opt_info.get('variability_keys', ['m_mra', 'h_mra'])
    dur_s   = opt_info.get('session_duration_s')
    rate_lbl = ('Trial rate (trials/hr)'
                if (dur_s or opt_info.get('has_tph'))
                else 'n (accepted trials)')

    sep = '=' * 72
    print(f"\n{sep}")
    print(f"  EMG Window Optimization Summary — {header.subject_id}")
    print(sep)
    print(f"\n  OBJECTIVES")
    print(f"    1. MAXIMIZE  {rate_lbl}")
    print(f"    2. MINIMIZE  variability  "
          f"({vstat.upper()} of {', '.join(vkeys)})")
    print(f"\n  COMPOSITE SCORE")
    print(f"    score = {alpha:.2f} × trial_score + {1 - alpha:.2f} × consistency_score")
    print(f"    (each component normalized to [0, 1] across all (centre, hw) pairs)")
    print(f"    alpha = 1.0 → maximize trials only")
    print(f"    alpha = 0.0 → minimize variability only")
    print(f"    alpha = 0.5 → balanced trade-off")

    tr_min = opt_info.get('tr_min', 0); tr_max = opt_info.get('tr_max', 1)
    vr_min = opt_info.get('vr_min', 0); vr_max = opt_info.get('vr_max', 1)
    print(f"\n  NORMALIZATION RANGES")
    print(f"    Trial rate  : [{tr_min:.1f}, {tr_max:.1f}]")
    print(f"    Variability : [{vr_min:.4f}, {vr_max:.4f}]")

    ranked = sorted(
        [r for r in opt_rows
         if np.isfinite(r.get('composite_score', float('nan')))],
        key=lambda r: -r['composite_score'])

    print(f"\n  TOP {min(top_n, len(ranked))} SOLUTIONS  (by composite score)\n")
    hdr = (f"  {'Rank':>4}  {'Centre':>8}  {'±hw':>6}  {'n':>5}  "
           f"{'Variability':>12}  {'trial_sc':>9}  "
           f"{'consist_sc':>11}  {'composite':>10}  {'Pareto':>6}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in ranked[:top_n]:
        vval = r.get('variability', float('nan'))
        vstr = f'{vval:.4f}' if np.isfinite(vval) else '   nan'
        print(f"  {r['rank']:>4}  {r['centre']:>8}  {r['hw']:>6.0f}  "
              f"{r['n']:>5}  {vstr:>12}  "
              f"{r['trial_score']:>9.4f}  {r['consistency_score']:>11.4f}  "
              f"{r['composite_score']:>10.4f}  "
              f"{'★' if r.get('pareto') else '':>6}")

    pareto = sorted(
        [r for r in opt_rows
         if r.get('pareto') and
         np.isfinite(r.get('composite_score', float('nan')))],
        key=lambda r: r['trial_rate'])
    if pareto:
        print(f"\n  PARETO-OPTIMAL SOLUTIONS ({len(pareto)})")
        print(f"  (No other (centre, hw) pair has both MORE trials AND LOWER variability)\n")
        hdr2 = (f"  {'Centre':>8}  {'±hw':>6}  {'n':>5}  "
                f"{'Variability':>12}  {'Composite':>10}")
        print(hdr2)
        print("  " + "─" * (len(hdr2) - 2))
        for r in pareto:
            vval = r.get('variability', float('nan'))
            vstr = f'{vval:.4f}' if np.isfinite(vval) else '   nan'
            print(f"  {r['centre']:>8}  {r['hw']:>6.0f}  {r['n']:>5}  "
                  f"{vstr:>12}  {r['composite_score']:>10.4f}")

    best = ranked[0] if ranked else None
    if best:
        print(f"\n  RECOMMENDED WINDOW  (alpha = {alpha:.2f})")
        print(f"    Centre   : {best['centre']}")
        print(f"    ±hw      : {best['hw']:.0f} µV")
        print(f"    Trials   : {best['n']}")
        vval = best.get('variability', float('nan'))
        vstr = f'{vval:.4f}' if np.isfinite(vval) else 'nan'
        print(f"    Variab.  : {vstr}  ({vstat.upper()})")
        print(f"    Score    : {best['composite_score']:.4f}")
        print(f"    Pareto   : {'Yes' if best.get('pareto') else 'No'}")
    print(f"\n  Tip: adjust ALPHA in the cell above to shift the priority.")
    print(sep)


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

        dig_oe = getattr(tr, 'digital_onset_sample_num', -1)
        if dig_oe is None:
            dig_oe = -1
        dig_ch = getattr(tr, 'digital_onset_channel', -1)
        has_digital = dig_oe >= 0

        if not has_sync:
            rec.update(onset_found=has_digital, onset_idx=bin_samples,
                       adc_peak=float('nan'), adc_noise_std=float('nan'),
                       stim_end_ms=None, stim_duration_ms=0.0,
                       n_pre_trigger_frames_discarded=0,
                       first_post_trigger_frame_sample_id=0,
                       digital_onset=dig_oe, digital_ch=dig_ch,
                       failed=not has_digital)
            results.append(rec)
            continue

        if file_version >= 2:
            # onset_detected: 0=none, 1=ADC, 2=digital DIGITAL IN.
            # onset_sample_index is correctly set by the app for both 1 and 2.
            onset_found = bool(tr.onset_detected) or has_digital
            onset_idx   = (tr.onset_sample_index
                           if (bool(tr.onset_detected) and tr.onset_sample_index >= 0)
                           else bin_samples)
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
                onset_found = has_digital
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
                   digital_onset=dig_oe, digital_ch=dig_ch,
                   failed=not onset_found)
        results.append(rec)
    return results


def find_adc_pulses_in_stream(emg_blocks,
                              threshold: float = STIM_ONSET_THRESHOLD,
                              min_gap_samples: int = 500):
    """Scan all continuous EMG blocks in order and return the OE sample index of
    each ADC rising edge (low→high crossing above threshold).

    Parameters
    ----------
    emg_blocks       : list[EmgDataBlock]
    threshold        : crossing threshold in volts; uses STIM_ONSET_THRESHOLD by default
    min_gap_samples  : minimum spacing between consecutive pulses — suppresses
                       double-counts from long stim pulses or ringing

    Returns
    -------
    list[int]  OE sample indices of detected rising edges, in chronological order.
               Index N in this list corresponds to the (N+1)th stimulation event.
    """
    rising_edges = []
    prev_was_high = False
    last_edge_oe  = None

    for blk in emg_blocks:
        adc_idx = None
        for ci, cn in enumerate(blk.channel_names):
            if 'ADC' in cn.upper() and ci < len(blk.raw_channels):
                adc_idx = ci
                break
        if adc_idx is None:
            if len(blk.raw_channels) >= 3:
                adc_idx = 2
            else:
                prev_was_high = False
                continue

        blk_start = int(blk.ts_open_ephys_sent)
        adc_abs   = np.abs(np.asarray(blk.raw_channels[adc_idx], dtype=float))
        is_high   = adc_abs >= threshold

        if len(is_high) == 0:
            continue

        # Vectorised rising-edge detection (incorporates cross-block state)
        prev_arr    = np.empty(len(is_high), dtype=bool)
        prev_arr[0] = prev_was_high
        prev_arr[1:] = is_high[:-1]

        for i in np.where((~prev_arr) & is_high)[0]:
            oe = blk_start + int(i)
            if last_edge_oe is None or (oe - last_edge_oe) >= min_gap_samples:
                rising_edges.append(oe)
                last_edge_oe = oe

        prev_was_high = bool(is_high[-1])

    return rising_edges


def detect_and_correct_failed_trials(hrs2_trials, hrs2_header, hrs2_emg_blocks,
                                     pre_ms: float = 2.0,
                                     post_ms: float = 15.0,
                                     m_start_ms: float = 2.5,
                                     m_end_ms: float = 4.5,
                                     h_start_ms: float = 6.0,
                                     h_end_ms: float = 9.0,
                                     sample_rate: float = None,
                                     ctx_pre_s: float = 10.0,
                                     ctx_post_s: float = 10.0,
                                     silent: bool = False):
    """Classify HRS2 trials for ADC-sync failures, realign each failed trial to the
    true stim onset, and display an interactive per-trial context viewer plus a
    corrected-waveform grid.

    Primary onset strategy: count all ADC rising edges in the continuous stream —
    edge #N corresponds to trial #N (0-indexed). This handles cases where the true
    ADC pulse lies *before* the app's sensed onset (which only searches forward).
    Falls back to a forward/backward context-window search when the pulse count
    is out of range or disagrees strongly.

    Parameters
    ----------
    hrs2_trials     : list[MhRecTrial]
    hrs2_header     : MhRecHeader
    hrs2_emg_blocks : list[EmgDataBlock]  continuous recording blocks
    pre_ms          : ms before true onset to display in corrected window
    post_ms         : ms after  true onset to display in corrected window
    m_start_ms/end  : M-wave window (ms re: true onset)
    h_start_ms/end  : H-wave window (ms re: true onset)
    sample_rate     : override Hz; None → header.sample_rate or SAMPLE_RATE
    ctx_pre_s       : seconds of context to pull before sensed onset (default 10 s)
    ctx_post_s      : seconds of context to pull after  sensed onset (default 10 s)

    Returns
    -------
    trial_report : list[dict]
    failed       : list[dict]
    passed       : list[dict]
    realigned    : list[dict]   Keys: trial_num, amp_ma, t_orig, emg_orig,
                                      t_corr, emg_corr, delay_ms,
                                      delay_ms_count, delay_ms_ctx, ctx_data
    """
    import math
    from IPython.display import display as _display
    from ipywidgets import Button, Output, HBox, VBox, Label

    trial_report = classify_trials(hrs2_trials, file_version=hrs2_header.file_version)
    failed = [r for r in trial_report if r['failed']]
    passed = [r for r in trial_report if not r['failed']]

    v2 = hrs2_header.file_version >= 2
    print(f"File version : {hrs2_header.file_version}  "
          f"({'pre-computed fields used' if v2 else 'fields derived from sync_data'})")
    print(f"Total trials : {len(trial_report)}")
    print(f"Passed       : {len(passed)}")
    print(f"Failed       : {len(failed)}")

    if failed:
        print()
        hdr = (f"{'Trial':>6}  {'Amp':>6}  {'ADC peak':>9}  {'Noise std':>10}  "
               f"{'Stim dur ms':>11}  {'Discarded':>9}  Reason")
        print(hdr)
        print("-" * len(hdr))
        for r in failed:
            reason = ("no sync data" if not r['has_sync']
                      else f"ADC peak {r['adc_peak']:.3f} V < {STIM_ONSET_THRESHOLD} V threshold")
            dur_str  = f"{r['stim_duration_ms']:.2f}" if r['stim_duration_ms'] else "—"
            disc_str = str(r['n_pre_trigger_frames_discarded']) if v2 else "n/a"
            print(f"{r['idx']+1:>6}  {r['amp_ma']:>6.2f}  {r['adc_peak']:>9.3f}  "
                  f"{r['adc_noise_std']:>10.4f}  {dur_str:>11}  {disc_str:>9}  {reason}")
    else:
        print("\nNo failed trials — all trials passed ADC-sync check.")
        return trial_report, failed, passed, []

    _sr     = float(sample_rate or getattr(hrs2_header, 'sample_rate', None) or SAMPLE_RATE)
    _ms_per = 1000.0 / _sr
    _bin_s  = int(round(BIN_DURATION_MS / _ms_per))

    # ---- Scan full stream for all ADC rising edges ----
    print("\nScanning ADC stream for all stim pulses ...")
    all_pulse_oe = find_adc_pulses_in_stream(hrs2_emg_blocks,
                                             threshold=STIM_ONSET_THRESHOLD,
                                             min_gap_samples=int(0.1 * _sr))
    n_trials = len(hrs2_trials)
    print(f"  ADC pulses found in stream : {len(all_pulse_oe)}")
    print(f"  Trials in file             : {n_trials}")
    if len(all_pulse_oe) != n_trials:
        print(f"  NOTE: counts differ — pulse-count matching may be imprecise for some trials.")

    # ---- Per-trial realignment ----
    realigned = []
    for r in failed:
        tr     = hrs2_trials[r['idx']]
        trial_n = r['idx']  # 0-based

        t_orig, emg_orig, _, _, _ = get_trial_window(
            tr, pre_ms, post_ms, ms_per_sample=_ms_per, bin_samples=_bin_s
        )

        # -- Primary: pulse-count approach (bidirectional, pre/post onset) --
        delay_ms_count = None
        if trial_n < len(all_pulse_oe):
            onset_oe = _trial_onset_oe(tr, _bin_s)
            if onset_oe is not None:
                pulse_oe = all_pulse_oe[trial_n]
                delay_ms_count = (pulse_oe - onset_oe) / _sr * 1000.0

        # -- Fallback: context window search (nearest rising edge to sensed onset) --
        delay_ms_ctx = None
        ctx = get_trial_context_window(tr, hrs2_emg_blocks,
                                       pre_s=ctx_pre_s, post_s=ctx_post_s,
                                       sample_rate=_sr, bin_samples=_bin_s)
        if ctx is not None:
            _t_s, _, _onset_i, _adc_c = ctx
            if _adc_c is not None:
                _adc_abs = np.abs(_adc_c)
                _is_hi   = _adc_abs >= STIM_ONSET_THRESHOLD
                _prev_hi = np.concatenate([[False], _is_hi[:-1]])
                _edges   = np.where((~_prev_hi) & _is_hi)[0]
                if len(_edges) > 0:
                    _nearest = _edges[np.argmin(np.abs(_edges.astype(int) - _onset_i))]
                    delay_ms_ctx = float(_t_s[_nearest]) * 1000.0

        # Use pulse-count delay when available; fall back to context search
        delay_ms = delay_ms_count if delay_ms_count is not None else delay_ms_ctx

        # -- Slice corrected window from trial_data --
        emg_corr = t_corr = None
        if delay_ms is not None:
            _delay_samp = int(round(delay_ms * _sr / 1000.0))
            _td = np.array(tr.trial_data)

            _od  = getattr(tr, 'onset_detected', 0)
            _osi = getattr(tr, 'onset_sample_index', -1)
            if _od >= 1 and _osi >= 0:
                _align_i = int(_osi)
            elif len(tr.sync_data) > 1:
                _align_i = detect_stim_onset(tr.sync_data, _bin_s)
            else:
                _align_i = _bin_s

            _true_onset = _align_i + _delay_samp
            _pre_samp   = int(round(pre_ms  * _sr / 1000.0))
            _post_samp  = int(round(post_ms * _sr / 1000.0))
            _s = _true_onset - _pre_samp
            _e = _true_onset + _post_samp
            if 0 <= _s and _e <= len(_td):
                emg_corr = _td[_s:_e]
                t_corr   = (np.arange(len(emg_corr)) - _pre_samp) * _ms_per

        realigned.append(dict(
            trial_num      = trial_n + 1,
            amp_ma         = r['amp_ma'],
            t_orig         = t_orig,
            emg_orig       = emg_orig,
            t_corr         = t_corr,
            emg_corr       = emg_corr,
            delay_ms       = delay_ms,
            delay_ms_count = delay_ms_count,
            delay_ms_ctx   = delay_ms_ctx,
            ctx_data       = ctx,
        ))

    n_found = sum(1 for d in realigned if d['delay_ms'] is not None)

    if silent:
        return trial_report, failed, passed, realigned

    print(f"\nRealignment summary:")
    print(f"  Resolved (pulse-count)  : {sum(1 for d in realigned if d['delay_ms_count'] is not None)}")
    print(f"  Resolved (ctx fallback) : {sum(1 for d in realigned if d['delay_ms_count'] is None and d['delay_ms_ctx'] is not None)}")
    print(f"  Unresolved              : {len(realigned) - n_found}")

    # ---- Interactive context viewer ----
    _state = {'idx': 0}
    _out   = Output()
    _lbl   = Label(value='')

    def _update_lbl():
        d = realigned[_state['idx']]
        _lbl.value = (f"Failed trial {_state['idx']+1} / {len(realigned)}"
                      f"  (trial #{d['trial_num']}, {d['amp_ma']:.2f} mA)")

    def _draw_viewer(idx):
        d   = realigned[idx]
        r   = failed[idx]
        ctx = d['ctx_data']
        with _out:
            _out.clear_output(wait=True)

            print(f"Trial #{d['trial_num']}  |  {d['amp_ma']:.2f} mA  |  "
                  f"onset_detected={getattr(hrs2_trials[r['idx']], 'onset_detected', '?')}")
            if d['delay_ms_count'] is not None:
                print(f"  Pulse-count  (ADC pulse #{r['idx']+1} in stream) : "
                      f"{d['delay_ms_count']:+.2f} ms")
            else:
                print(f"  Pulse-count  : unavailable (stream count / trial index mismatch)")
            if d['delay_ms_ctx'] is not None:
                print(f"  Ctx search   : {d['delay_ms_ctx']:+.2f} ms")
            else:
                print(f"  Ctx search   : not found")
            if d['delay_ms'] is not None:
                print(f"  → Applied correction : {d['delay_ms']:+.2f} ms")
            else:
                print(f"  → No correction applied")

            if ctx is None:
                print("  (Context window unavailable — no plot)")
                return

            t_s, emg_ctx, onset_i, adc_ctx = ctx
            adc_abs = np.abs(adc_ctx) if adc_ctx is not None else None

            cnt_t_s = d['delay_ms_count'] / 1000.0 if d['delay_ms_count'] is not None else None
            ctx_t_s = d['delay_ms_ctx']   / 1000.0 if d['delay_ms_ctx']   is not None else None

            fig, (ax_emg, ax_adc) = plt.subplots(
                2, 1, figsize=(16, 6), sharex=True,
                gridspec_kw={'height_ratios': [2, 1]},
            )
            ax_emg.plot(t_s, emg_ctx, color='black', linewidth=0.5, label='Filtered EMG')
            ax_emg.axvline(0, color='red', linestyle='--', linewidth=1.2,
                           label='Sensed onset (t = 0)')
            if cnt_t_s is not None:
                ax_emg.axvline(cnt_t_s, color='purple', linestyle='-', linewidth=1.5,
                               label=f'Pulse-count onset ({d["delay_ms_count"]:+.1f} ms)')
            if ctx_t_s is not None:
                ax_emg.axvline(ctx_t_s, color='orange', linestyle=':', linewidth=1.2,
                               label=f'Ctx-search onset ({d["delay_ms_ctx"]:+.1f} ms)')
            ax_emg.axvspan(-ctx_pre_s, 0,           color='blue',  alpha=0.04)
            ax_emg.axvspan(0,          ctx_post_s,  color='green', alpha=0.04)
            ax_emg.set_ylabel('EMG (µV)')
            ax_emg.set_title(
                f"Trial #{d['trial_num']}  |  {d['amp_ma']:.2f} mA  |  "
                f"±{ctx_pre_s:.0f} s context  —  {hrs2_header.subject_id}"
            )
            ax_emg.legend(fontsize=8, loc='upper right')
            ax_emg.grid(True, alpha=0.3)

            if adc_abs is not None:
                ax_adc.plot(t_s, adc_abs, color='green', linewidth=0.6, label='|ADC| (V)')
                ax_adc.axhline(STIM_ONSET_THRESHOLD, color='red', linestyle='--',
                               linewidth=1.0, label=f'Threshold ({STIM_ONSET_THRESHOLD} V)')
                ax_adc.axvline(0, color='red', linestyle='--', linewidth=1.2)
                if cnt_t_s is not None:
                    ax_adc.axvline(cnt_t_s, color='purple', linestyle='-', linewidth=1.5,
                                   label=f'Pulse #{r["idx"]+1} ({d["delay_ms_count"]:+.1f} ms)')
                    ax_adc.annotate(
                        f'{d["delay_ms_count"]:+.1f} ms',
                        xy=(cnt_t_s, STIM_ONSET_THRESHOLD),
                        xytext=(cnt_t_s + 0.3, STIM_ONSET_THRESHOLD * 1.3),
                        fontsize=8, color='purple',
                        arrowprops=dict(arrowstyle='->', color='purple', lw=1.0),
                    )
                if ctx_t_s is not None:
                    ax_adc.axvline(ctx_t_s, color='orange', linestyle=':', linewidth=1.2,
                                   label=f'Ctx-search ({d["delay_ms_ctx"]:+.1f} ms)')
                ax_adc.set_ylabel('|ADC| (V)')
                ax_adc.legend(fontsize=8, loc='upper right')
                ax_adc.grid(True, alpha=0.3)
            else:
                ax_adc.text(0.5, 0.5, 'ADC channel not available',
                            transform=ax_adc.transAxes, ha='center', va='center', color='gray')

            ax_adc.set_xlabel('Time re: sensed onset (s)')
            plt.tight_layout()
            plt.show()

    def _on_prev(b):
        if _state['idx'] > 0:
            _state['idx'] -= 1
            _update_lbl()
            _draw_viewer(_state['idx'])

    def _on_next(b):
        if _state['idx'] < len(realigned) - 1:
            _state['idx'] += 1
            _update_lbl()
            _draw_viewer(_state['idx'])

    _prev_btn = Button(description='◀ Prev', button_style='')
    _next_btn = Button(description='Next ▶', button_style='primary')
    _prev_btn.on_click(_on_prev)
    _next_btn.on_click(_on_next)
    _update_lbl()
    _display(VBox([HBox([_prev_btn, _next_btn, _lbl]), _out]))
    _draw_viewer(0)

    # ---- Corrected waveform grid ----
    n_cols = 3
    n_rows = math.ceil(len(realigned) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)
    for idx, d in enumerate(realigned):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.plot(d['t_orig'], d['emg_orig'],
                color='gray', alpha=0.7, linewidth=0.9, label='Original (misaligned)')
        if d['emg_corr'] is not None:
            ax.plot(d['t_corr'], d['emg_corr'],
                    color='black', linewidth=1.1,
                    label=f'Corrected ({d["delay_ms"]:+.1f} ms)')
        ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.15, zorder=0)
        ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.15, zorder=0)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.0, label='True onset')
        delay_str = (f'{d["delay_ms"]:+.1f} ms' if d['delay_ms'] is not None else 'no pulse')
        ax.set_title(f'Trial {d["trial_num"]} | {d["amp_ma"]:.2f} mA | {delay_str}', fontsize=8)
        ax.set_xlabel('Time re: true onset (ms)', fontsize=7)
        ax.set_ylabel('EMG (µV)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')

    for j in range(len(realigned), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')

    fig.suptitle(
        f'Failed Trials — Realigned to True Stim Onset\n'
        f'{hrs2_header.subject_id}  |  {len(realigned)} failed trial(s)',
        fontsize=12
    )
    plt.tight_layout()
    plt.show()

    # ---- ADC Stream Pulse Map ----
    # Compute onset_oe for every trial (not just failed ones) so we can build
    # a full delay map: stream_pulse[i] vs trial[i] sensed onset.
    trial_onset_oes = []
    for _tr in hrs2_trials:
        _fid = getattr(_tr, 'first_post_trigger_frame_sample_id', 0)
        _osi = getattr(_tr, 'onset_sample_index', -1)
        if _fid > 0 and _osi >= 0:
            trial_onset_oes.append(int(_fid) + (_osi - _bin_s))
        else:
            trial_onset_oes.append(None)

    # Delay for every trial using the pulse-count match (trial #i → stream pulse #i)
    all_trial_delays_ms = []
    for _i, _oe in enumerate(trial_onset_oes):
        if _oe is not None and _i < len(all_pulse_oe):
            all_trial_delays_ms.append((all_pulse_oe[_i] - _oe) / _sr * 1000.0)
        else:
            all_trial_delays_ms.append(None)

    failed_set = {r['idx'] for r in failed}
    n_extra_pulses = max(0, len(all_pulse_oe) - len(hrs2_trials))
    n_missing      = max(0, len(hrs2_trials) - len(all_pulse_oe))

    # Condensed summary: only print anomalous rows (|delay| > 5 ms)
    _ANOMALY_MS = 5.0
    _anomalous  = [(i, d) for i, d in enumerate(all_trial_delays_ms)
                   if d is not None and abs(d) > _ANOMALY_MS]
    print(f"\n{'='*72}")
    print(f"ADC STREAM PULSE MAP")
    print(f"  Stream pulses detected : {len(all_pulse_oe)}")
    print(f"  Trials in file         : {len(hrs2_trials)}")
    if n_extra_pulses:
        print(f"  Extra pulses (no trial): {n_extra_pulses}")
    if n_missing:
        print(f"  Trials without a matched pulse: {n_missing}")
    if _anomalous:
        print(f"  Anomalous trials (|delay| > {_ANOMALY_MS} ms): {len(_anomalous)}")
        print(f"\n  {'Trial#':>7}  {'Amp mA':>7}  {'Delay ms':>10}  {'Pulse OE':>12}  Status")
        print(f"  {'-'*55}")
        for _i, _d in _anomalous:
            _amp = hrs2_trials[_i].stimulation_amplitude_ma
            _st  = 'FAILED' if _i in failed_set else 'passed-anomalous'
            print(f"  {_i+1:>7}  {_amp:>7.2f}  {_d:>+10.2f}  {all_pulse_oe[_i]:>12}  {_st}")
    else:
        print(f"  All delays within ±{_ANOMALY_MS} ms — pulse-count matching is consistent.")
    print(f"{'='*72}")

    # Scatter plot: delay_ms vs trial# for all trials
    _xall      = [i + 1 for i, d in enumerate(all_trial_delays_ms) if d is not None]
    _yall      = [d     for i, d in enumerate(all_trial_delays_ms) if d is not None]
    _fail_mask = [(i in failed_set) for i, d in enumerate(all_trial_delays_ms) if d is not None]

    _x_pass = [x for x, f in zip(_xall, _fail_mask) if not f]
    _y_pass = [y for y, f in zip(_yall, _fail_mask) if not f]
    _x_fail = [x for x, f in zip(_xall, _fail_mask) if     f]
    _y_fail = [y for y, f in zip(_yall, _fail_mask) if     f]

    fig_sm, ax_sm = plt.subplots(figsize=(14, 4))
    if _x_pass:
        ax_sm.scatter(_x_pass, _y_pass, color='steelblue', s=10, alpha=0.4,
                      label='Passed trials')
    if _x_fail:
        ax_sm.scatter(_x_fail, _y_fail, color='red', s=40, zorder=5,
                      label='Failed trials')
    ax_sm.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Zero delay')
    ax_sm.set_xlabel('Trial number')
    ax_sm.set_ylabel('Stream pulse delay re: sensed onset (ms)')
    ax_sm.set_title(
        f'ADC Stream Pulse Map — {hrs2_header.subject_id}\n'
        f'{len(all_pulse_oe)} stream pulses  ·  {len(hrs2_trials)} trials  '
        f'(blue = passed, red = failed)'
    )
    ax_sm.legend(fontsize=8)
    ax_sm.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Interactive stream-pulse viewer ----
    # Step through every ADC rising edge detected in the continuous stream.
    # Shows the ±ctx_pre_s/ctx_post_s context centred on the MATCHED TRIAL's
    # sensed onset, with the stream pulse marked in purple.
    # Other pulses visible in the same window are marked orange.
    _ps    = {'idx': 0}
    _pout  = Output()
    _plbl  = Label(value='')

    def _upd_plbl():
        _pi = _ps['idx']
        if _pi < len(hrs2_trials):
            _d   = all_trial_delays_ms[_pi]
            _st  = 'FAILED' if _pi in failed_set else 'passed'
            _amp = hrs2_trials[_pi].stimulation_amplitude_ma
            _ds  = f'{_d:+.1f} ms' if _d is not None else 'n/a'
            _plbl.value = (f"Pulse {_pi+1}/{len(all_pulse_oe)}  →  "
                           f"Trial #{_pi+1}  ({_amp:.2f} mA)  "
                           f"delay={_ds}  [{_st}]")
        else:
            _plbl.value = (f"Pulse {_pi+1}/{len(all_pulse_oe)}  "
                           f"→ extra pulse (no matched trial)")

    def _draw_pview(_pi):
        with _pout:
            _pout.clear_output(wait=True)
            _pulse_oe = all_pulse_oe[_pi]

            # Which trial to use for the context window?
            if _pi < len(hrs2_trials):
                _tr_n = _pi
            else:
                # No direct match — use the trial whose onset_oe is nearest
                _cands = [(j, oe) for j, oe in enumerate(trial_onset_oes)
                          if oe is not None]
                _tr_n  = (min(_cands, key=lambda x: abs(x[1] - _pulse_oe))[0]
                          if _cands else 0)

            _tr_obj    = hrs2_trials[_tr_n]
            _onset_oe  = trial_onset_oes[_tr_n]
            _d_ms      = all_trial_delays_ms[_tr_n] if _tr_n == _pi else None
            _st        = 'FAILED' if _tr_n in failed_set else 'passed'

            print(f"Stream pulse #{_pi+1}  →  Trial #{_tr_n+1}  |  "
                  f"{_tr_obj.stimulation_amplitude_ma:.2f} mA  |  [{_st}]")
            print(f"  Pulse OE sample       : {_pulse_oe}")
            print(f"  Trial sensed onset OE : {_onset_oe}")
            if _d_ms is not None:
                print(f"  Delay (pulse − onset) : {_d_ms:+.2f} ms")

            _ctx_p = get_trial_context_window(
                _tr_obj, hrs2_emg_blocks,
                pre_s=ctx_pre_s, post_s=ctx_post_s,
                sample_rate=_sr, bin_samples=_bin_s,
            )
            if _ctx_p is None:
                print("  (Context window unavailable — no plot)")
                return

            _t_p, _emg_p, _oi_p, _adc_p = _ctx_p
            _adc_abs_p = np.abs(_adc_p) if _adc_p is not None else None

            # Time of this stream pulse relative to the trial's sensed onset
            _pt_s = ((_pulse_oe - _onset_oe) / _sr) if _onset_oe is not None else None

            # Other stream pulses visible inside this context window
            _ctx_s_oe = (_onset_oe - int(ctx_pre_s  * _sr)) if _onset_oe else None
            _ctx_e_oe = (_onset_oe + int(ctx_post_s * _sr)) if _onset_oe else None
            _others = []
            if _ctx_s_oe is not None:
                for _pj, _p_oe in enumerate(all_pulse_oe):
                    if _pj == _pi:
                        continue
                    if _ctx_s_oe <= _p_oe <= _ctx_e_oe:
                        _others.append((((_p_oe - _onset_oe) / _sr), _pj))

            fig_p, (ax_ep, ax_ap) = plt.subplots(
                2, 1, figsize=(16, 6), sharex=True,
                gridspec_kw={'height_ratios': [2, 1]},
            )
            ax_ep.plot(_t_p, _emg_p, color='black', linewidth=0.5, label='Filtered EMG')
            ax_ep.axvline(0, color='red', linestyle='--', linewidth=1.2,
                          label=f'Trial #{_tr_n+1} sensed onset (t=0)')
            if _pt_s is not None:
                _lbl_p = (f'Stream pulse #{_pi+1} ({_d_ms:+.1f} ms)'
                          if _d_ms is not None else f'Stream pulse #{_pi+1}')
                ax_ep.axvline(_pt_s, color='purple', linestyle='-', linewidth=2.0,
                              label=_lbl_p)
            for (_ot_s, _pj) in _others:
                ax_ep.axvline(_ot_s, color='orange', linestyle=':', linewidth=1.0, alpha=0.8)
                ax_ep.text(_ot_s, 0.95, f'#{_pj+1}',
                           transform=ax_ep.get_xaxis_transform(),
                           color='orange', fontsize=7, ha='center', va='top')
            ax_ep.axvspan(-ctx_pre_s, 0,           color='blue',  alpha=0.04)
            ax_ep.axvspan(0,           ctx_post_s, color='green', alpha=0.04)
            ax_ep.set_ylabel('EMG (µV)')
            _ttl_d = f'  |  {_d_ms:+.1f} ms' if _d_ms is not None else ''
            ax_ep.set_title(
                f"Stream Pulse #{_pi+1}  →  Trial #{_tr_n+1}  |  "
                f"{_tr_obj.stimulation_amplitude_ma:.2f} mA{_ttl_d}  |  "
                f"[{_st}]  —  {hrs2_header.subject_id}"
            )
            ax_ep.legend(fontsize=8, loc='upper right')
            ax_ep.grid(True, alpha=0.3)

            if _adc_abs_p is not None:
                ax_ap.plot(_t_p, _adc_abs_p, color='green', linewidth=0.6,
                           label='|ADC| (V)')
                ax_ap.axhline(STIM_ONSET_THRESHOLD, color='red', linestyle='--',
                              linewidth=1.0, label=f'Threshold ({STIM_ONSET_THRESHOLD} V)')
                ax_ap.axvline(0, color='red', linestyle='--', linewidth=1.2)
                if _pt_s is not None:
                    ax_ap.axvline(_pt_s, color='purple', linestyle='-', linewidth=2.0,
                                  label=f'Pulse #{_pi+1}')
                    _ann_txt = (f'{_d_ms:+.1f} ms' if _d_ms is not None
                                else f'pulse #{_pi+1}')
                    ax_ap.annotate(
                        _ann_txt,
                        xy=(_pt_s, STIM_ONSET_THRESHOLD),
                        xytext=(_pt_s + 0.5, STIM_ONSET_THRESHOLD * 1.3),
                        fontsize=8, color='purple',
                        arrowprops=dict(arrowstyle='->', color='purple', lw=1.0),
                    )
                for (_ot_s, _pj) in _others:
                    ax_ap.axvline(_ot_s, color='orange', linestyle=':', linewidth=1.0,
                                  alpha=0.8)
                ax_ap.set_ylabel('|ADC| (V)')
                ax_ap.legend(fontsize=8, loc='upper right')
                ax_ap.grid(True, alpha=0.3)
            else:
                ax_ap.text(0.5, 0.5, 'ADC channel not available',
                           transform=ax_ap.transAxes, ha='center', va='center',
                           color='gray')

            ax_ap.set_xlabel('Time re: trial sensed onset (s)')
            plt.tight_layout()
            plt.show()

    def _on_pp(b):
        if _ps['idx'] > 0:
            _ps['idx'] -= 1
            _upd_plbl()
            _draw_pview(_ps['idx'])

    def _on_pn(b):
        if _ps['idx'] < len(all_pulse_oe) - 1:
            _ps['idx'] += 1
            _upd_plbl()
            _draw_pview(_ps['idx'])

    _pp_btn = Button(description='◀ Prev pulse', button_style='')
    _pn_btn = Button(description='Next pulse ▶', button_style='primary')
    _pp_btn.on_click(_on_pp)
    _pn_btn.on_click(_on_pn)
    _upd_plbl()
    _display(VBox([HBox([_pp_btn, _pn_btn, _plbl]), _pout]))
    _draw_pview(0)

    return trial_report, failed, passed, realigned


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


def simulate_trial_initiation_hrs(
    emg_blocks,
    header,
    hrs2_trials=None,
    sample_rate: float = SAMPLE_RATE,
    min_init_uv: float = TRIAL_INIT_MIN_UV,
    max_init_uv: float = TRIAL_INIT_MAX_UV,
    min_inter_trial_ms: int = MINIMUM_INTERTRIAL_INTERVAL_MS,
    blank_pre_ms: float = 5.0,
    blank_post_ms: float = 20.0,
    verbose: bool = True,
):
    """Run the trial-initiation simulator on HRS binary file data.

    Stitches ``emg_blocks`` (``hrs1_emg_blocks`` or ``hrs2_emg_blocks``) into a
    continuous abs-EMG signal, optionally blanks stim events from HRS2, then
    runs ``run_trial_initiation_simulation`` and prints the full results report
    (summary statistics, first-10-trials table, 6-panel figure, trial-rate
    analysis).

    Parameters
    ----------
    emg_blocks : list
        EMG block list from ``read_hrs1`` or ``read_hrs2`` (``hrs1_emg_blocks``
        or ``hrs2_emg_blocks``).
    header : EmgCharHeader or MhRecHeader
        Used for the subject_id display label.
    hrs2_trials : list or None
        When provided (HRS2 mode) stim onset locations are used to build a
        blank mask; blanked samples are zeroed before simulation so stim
        artifacts cannot trigger initiation.
    sample_rate : float
        Recording sample rate in Hz.
    min_init_uv, max_init_uv : float
        Grand-mean amplitude window that counts as a valid initiation (µV).
    min_inter_trial_ms : int
        Minimum inter-trial interval enforced by the state machine (ms).
    blank_pre_ms, blank_post_ms : float
        Samples to zero before/after each stim event (HRS2 mode only).
    verbose : bool
        Print simulation progress and results.

    Returns
    -------
    (simulated_trials, sim_statistics) — same types as
    ``run_trial_initiation_simulation``.
    """
    import matplotlib.pyplot as plt

    subject_id = getattr(header, 'subject_id', 'unknown')

    if not emg_blocks:
        print("simulate_trial_initiation_hrs: no EMG blocks provided.")
        return [], {}

    # ---- 1. Stitch blocks into continuous abs-EMG ----
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

    timestamps = np.arange(n_total, dtype=np.float64) / sample_rate
    source_label = 'HRS1' if hrs2_trials is None else 'HRS2'

    print("=" * 70)
    print(f"Trial Initiation Simulator — {source_label} source | {subject_id}")
    print("=" * 70)
    print(f"  Blocks stitched : {len(sorted_blks):,}")
    print(f"  Total samples   : {n_total:,}  ({duration_s:.1f} s / {duration_s/60:.1f} min)")
    print(f"  Sample rate     : {sample_rate:.0f} Hz")
    print(f"  Thresholds      : {min_init_uv:.1f} – {max_init_uv:.1f} µV")
    print(f"  Min ITI         : {min_inter_trial_ms} ms")

    # ---- 2. Stim blanking (HRS2 only) ----
    emg_sim = continuous_abs_emg.copy()
    n_blanked = 0
    if hrs2_trials is not None:
        _bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
        stim_rel = []
        for tr in hrs2_trials:
            fid = int(getattr(tr, 'first_post_trigger_frame_sample_id', 0))
            osi = int(getattr(tr, 'onset_sample_index', -1))
            if fid > 0 and osi >= 0:
                rel = fid + (osi - _bin_s) - first_oe
                if 0 <= rel < n_total:
                    stim_rel.append(rel)
        stim_times_s = np.array(stim_rel, dtype=np.float64) / sample_rate
        blank_mask = build_blank_mask(timestamps, stim_times_s, sample_rate,
                                      blank_pre_ms, blank_post_ms)
        emg_sim[~blank_mask] = 0.0
        n_blanked = int(np.sum(~blank_mask))
        print(f"  Stim events     : {len(stim_rel)} of {len(hrs2_trials)} mapped")
        print(f"  Blanked samples : {n_blanked:,}  ({100 * n_blanked / n_total:.2f}%)")
    print()

    # ---- 3. Run simulation ----
    simulated_trials, sim_statistics = run_trial_initiation_simulation(
        emg_signal=emg_sim,
        timestamps=timestamps,
        sample_rate=sample_rate,
        min_init_threshold=min_init_uv,
        max_init_threshold=max_init_uv,
        min_inter_trial_ms=min_inter_trial_ms,
        verbose=verbose,
    )

    if not simulated_trials:
        print("No trials simulated.")
        return simulated_trials, sim_statistics

    # ---- 4. Results report (Phase 5c style) ----
    print()
    print("=" * 80)
    print("SIMULATION RESULTS — DETAILED ANALYSIS")
    print("=" * 80)
    print()

    print("SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total Monitoring Attempts:     {sim_statistics['total_monitoring_attempts']:,}")
    print(f"Successful Trial Initiations:  {sim_statistics['successful_trials']:,}")
    print(f"Success Rate:                  {sim_statistics['success_rate_pct']:.2f}%")
    print()
    print(f"Recording Duration:            {duration_s:.2f} seconds  ({duration_s/60:.2f} min)")
    print(f"Total Samples Processed:       {sim_statistics['total_samples_processed']:,}")
    print(f"Simulation Time:               {sim_statistics['elapsed_time_seconds']:.2f} seconds")
    print(f"Processing Speed:              "
          f"{sim_statistics['processing_speed_samples_per_sec']:,.0f} samples/sec")
    print()

    if len(simulated_trials) > 1:
        iti = sim_statistics['inter_trial_intervals_ms']
        print("Inter-Trial Interval Stats (ms):")
        print(f"  Mean:                        {np.mean(iti):,.1f} ms")
        print(f"  Median:                      {np.median(iti):,.1f} ms")
        print(f"  Std Dev:                     {np.std(iti):,.1f} ms")
        print(f"  Min:                         {np.min(iti):,.1f} ms")
        print(f"  Max:                         {np.max(iti):,.1f} ms")
        print()

        gms = [t.grand_mean_uv for t in simulated_trials]
        print("Grand Mean EMG Stats (µV):")
        print(f"  Mean:                        {np.mean(gms):.2f} µV")
        print(f"  Median:                      {np.median(gms):.2f} µV")
        print(f"  Std Dev:                     {np.std(gms):.2f} µV")
        print(f"  Min:                         {np.min(gms):.2f} µV")
        print(f"  Max:                         {np.max(gms):.2f} µV")
        print()

        mons = [t.monitoring_duration_ms for t in simulated_trials]
        print("Monitoring Duration Stats (ms):")
        print(f"  Mean:                        {np.mean(mons):.1f} ms")
        print(f"  Min:                         {np.min(mons):.1f} ms")
        print(f"  Max:                         {np.max(mons):.1f} ms")
        print()

    print("-" * 80)
    print()

    print("FIRST 10 TRIALS (Detailed View)")
    print("-" * 80)
    print(f"{'#':<5} {'Time (s)':<12} {'Grand Mean':<15} {'Monitor (ms)':<15} {'ISI (ms)':<12}")
    print("-" * 80)
    for tr in simulated_trials[:10]:
        isi_str = (f"{tr.time_since_last_trial_ms:.1f}"
                   if tr.time_since_last_trial_ms is not None else "N/A")
        print(f"{tr.trial_number:<5} {tr.start_time:<12.2f} "
              f"{tr.grand_mean_uv:<15.2f} {tr.monitoring_duration_ms:<15} {isi_str:<12}")
    if len(simulated_trials) > 10:
        print(f"... and {len(simulated_trials) - 10} more trials")
    print("-" * 80)
    print()

    # ---- 5. 6-panel figure ----
    trial_times = [t.start_time for t in simulated_trials]
    trial_times_hr = [t / 3600.0 for t in trial_times]
    trial_numbers = [t.trial_number for t in simulated_trials]
    gms = [t.grand_mean_uv for t in simulated_trials]
    mons = [t.monitoring_duration_ms for t in simulated_trials]
    iti = sim_statistics['inter_trial_intervals_ms']

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Trial timeline
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(trial_times_hr, trial_numbers, 'o-', markersize=4, linewidth=1)
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Trial Number')
    ax1.set_title('Trial Timeline — When Trials Were Initiated')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Inter-trial interval distribution
    ax2 = plt.subplot(3, 2, 2)
    if iti:
        ax2.hist(iti, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(min_inter_trial_ms, color='red', linestyle='--', linewidth=2,
                    label=f'Min ITI ({min_inter_trial_ms} ms)')
        ax2.axvline(float(np.mean(iti)), color='green', linestyle='--', linewidth=2,
                    label=f'Mean ({float(np.mean(iti)):.1f} ms)')
        ax2.legend(fontsize=8)
    ax2.set_xlabel('Inter-Trial Interval (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Inter-Trial Interval Distribution')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Grand mean distribution
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(gms, bins=40, edgecolor='black', alpha=0.7, color='orange')
    ax3.axvline(min_init_uv, color='red', linestyle='--', linewidth=2,
                label=f'Min threshold ({min_init_uv} µV)')
    ax3.axvline(max_init_uv, color='red', linestyle='--', linewidth=2,
                label=f'Max threshold ({max_init_uv} µV)')
    ax3.axvline(float(np.mean(gms)), color='blue', linestyle='--', linewidth=2,
                label=f'Mean ({float(np.mean(gms)):.2f} µV)')
    ax3.set_xlabel('Grand Mean EMG (µV)')
    ax3.set_ylabel('Count')
    ax3.set_title('Grand Mean Distribution of Simulated Trials')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Grand mean vs time
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(trial_times_hr, gms, alpha=0.6, s=30,
                c=range(len(trial_times_hr)), cmap='viridis')
    ax4.axhline(min_init_uv, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(max_init_uv, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Time (h)')
    ax4.set_ylabel('Grand Mean EMG (µV)')
    ax4.set_title('Grand Mean EMG Over Time')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Monitoring duration distribution
    ax5 = plt.subplot(3, 2, 5)
    _uniq, _cnt = np.unique(mons, return_counts=True)
    ax5.bar(_uniq, _cnt, width=40, edgecolor='black', alpha=0.7, color='purple')
    ax5.set_xlabel('Monitoring Duration (ms)')
    ax5.set_ylabel('Count')
    ax5.set_title('Monitoring Window Duration Distribution')
    ax5.grid(True, alpha=0.3, axis='y')

    # Panel 6: First trial's monitoring window from continuous signal
    ax6 = plt.subplot(3, 2, 6)
    _ex = simulated_trials[0]
    _bin_s = int(BIN_DURATION_MS * sample_rate / 1000)
    _mon_s = int(_ex.monitoring_duration_ms * sample_rate / 1000)
    _w0 = max(0, _ex.start_sample_idx + _ex.num_shifts * _bin_s)
    _w1 = min(n_total, _w0 + _mon_s)
    _seg = emg_sim[_w0:_w1]
    _tms = (np.arange(len(_seg)) / sample_rate) * 1000
    ax6.plot(_tms, _seg, color='black', linewidth=0.8)
    ax6.axhline(_ex.grand_mean_uv, color='green', linestyle='--', linewidth=2,
                label=f'Grand mean = {_ex.grand_mean_uv:.2f} µV')
    ax6.axhline(min_init_uv, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Min = {min_init_uv:.1f} µV')
    ax6.axhline(max_init_uv, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Max = {max_init_uv:.1f} µV')
    ax6.axhspan(min_init_uv, max_init_uv, color='green', alpha=0.06)
    ax6.set_xlabel('Time within monitoring window (ms)')
    ax6.set_ylabel('|EMG| (µV)')
    ax6.set_title(f'Trial #1 Monitoring Window ({_ex.monitoring_duration_ms} ms)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        f"Trial Initiation Simulation — {source_label} | {subject_id}  "
        f"({min_init_uv:.1f}–{max_init_uv:.1f} µV, ITI ≥ {min_inter_trial_ms} ms)",
        fontsize=11)
    plt.tight_layout()
    plt.show()

    # ---- 6. Trial rate analysis ----
    recording_duration_hours = duration_s / 3600.0
    trials_per_hour = len(simulated_trials) / recording_duration_hours if recording_duration_hours > 0 else 0
    print("TRIAL RATE ANALYSIS")
    print("-" * 80)
    print(f"Recording Duration:            {recording_duration_hours:.3f} hours  ({duration_s:.1f} s)")
    print(f"Total Simulated Trials:        {len(simulated_trials)}")
    print(f"Trials per Hour:               {trials_per_hour:.1f}")
    print("-" * 80)
    print()
    print("=" * 80)
    print("SIMULATION ANALYSIS COMPLETE")
    print("=" * 80)

    return simulated_trials, sim_statistics


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


def plot_emg_full_traces(timestamps, differential_filt, emg1, emg2, directory,
                         color="purple"):
    """Three stacked full-trace plots: filtered differential, EMG1, EMG2."""
    plot_full_trace(timestamps, differential_filt,
                    title=f"{directory} Filtered Differential EMG Signal (EMG1 - EMG2)",
                    label="Filtered EMG1 - EMG2", color=color)
    plot_full_trace(timestamps, emg1, title="EMG1 Raw", label="EMG1", color=color)
    plot_full_trace(timestamps, emg2, title="EMG2 Raw", label="EMG2", color=color)


def make_segment_viewer(timestamps, signals, labels,
                        segment_duration_s=10, title_prefix="",
                        figsize=(15, 4), color="purple"):
    """Build a Prev/Next ipywidgets viewer that shows one fixed-duration window
    at a time, with one stacked plot per (signal, label) pair.

    Each segment has a "Mark Good" toggle button; marked segments accumulate in
    a set that is returned alongside the widget so it can later be passed to
    make_selected_segment_viewer().

    Returns (VBox, selected_set).  Caller: widget, selected = make_segment_viewer(...); display(widget)
    """
    from ipywidgets import ToggleButton, Label, HBox as _HBox
    if len(signals) != len(labels):
        raise ValueError("signals and labels must be the same length")
    total_time   = timestamps[-1] - timestamps[0]
    num_segments = int(np.ceil(total_time / segment_duration_s))
    state    = {"idx": 0}
    selected = set()
    out      = Output()

    status_lbl = Label(value=f"Segment 1 / {num_segments}  |  Selected: 0")
    mark_btn   = ToggleButton(description="Mark Good", value=False,
                              button_style="", layout={"width": "130px"})

    def _update_status():
        idx = state["idx"]
        mark_btn.value        = idx in selected
        mark_btn.button_style = "success" if idx in selected else ""
        status_lbl.value      = (f"Segment {idx + 1} / {num_segments}"
                                 f"  |  Selected: {len(selected)}")

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
        _update_status()
        with out:
            out.clear_output(wait=True)
            _draw(state["idx"])

    def _toggle_mark(change):
        idx = state["idx"]
        if change["new"]:
            selected.add(idx)
        else:
            selected.discard(idx)
        _update_status()

    next_btn = Button(description="Next")
    prev_btn = Button(description="Previous")
    next_btn.on_click(lambda _b: _step(+1))
    prev_btn.on_click(lambda _b: _step(-1))
    mark_btn.observe(_toggle_mark, names="value")

    with out:
        _draw(state["idx"])
    _update_status()
    nav_bar = _HBox([prev_btn, next_btn, mark_btn, status_lbl])
    return VBox([nav_bar, out]), selected


def make_selected_segment_viewer(timestamps, signals, labels, selected_indices,
                                 segment_duration_s=10, title_prefix="",
                                 figsize=(15, 4), color="steelblue"):
    """Viewer that navigates only the chosen segment indices from make_segment_viewer.

    selected_indices — the set/list returned by make_segment_viewer (0-based segment indices).
    Returns a VBox; caller wraps in display(...).
    """
    from ipywidgets import Label, HBox as _HBox
    if len(signals) != len(labels):
        raise ValueError("signals and labels must be the same length")
    seg_list = sorted(selected_indices)
    if not seg_list:
        from ipywidgets import Label as _Lbl
        return VBox([_Lbl(value="No segments selected.")])

    state      = {"pos": 0}
    out        = Output()
    status_lbl = Label(value="")

    def _draw(pos):
        idx     = seg_list[pos]
        start_t = timestamps[0] + idx * segment_duration_s
        end_t   = min(start_t + segment_duration_s, timestamps[-1])
        mask    = (timestamps >= start_t) & (timestamps < end_t)
        local_t = timestamps[mask] - start_t
        prefix  = f"{title_prefix}, " if title_prefix else ""
        status_lbl.value = (f"Good segment {pos + 1} / {len(seg_list)}"
                            f"  (original segment {idx + 1},"
                            f" {start_t:.1f}–{end_t:.1f} s)")
        for sig, lbl in zip(signals, labels):
            plt.figure(figsize=figsize)
            plt.plot(local_t, sig[mask], label=lbl, color=color)
            plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
            plt.title(f"{prefix}{lbl}  |  Good {pos + 1}/{len(seg_list)}"
                      f"  (orig seg {idx + 1}, {start_t:.1f}–{end_t:.1f} s)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (μV)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def _step(delta):
        new_pos = max(0, min(len(seg_list) - 1, state["pos"] + delta))
        if new_pos == state["pos"]:
            return
        state["pos"] = new_pos
        with out:
            out.clear_output(wait=True)
            _draw(state["pos"])

    next_btn = Button(description="Next")
    prev_btn = Button(description="Previous")
    next_btn.on_click(lambda _b: _step(+1))
    prev_btn.on_click(lambda _b: _step(-1))

    with out:
        _draw(0)
    nav_bar = _HBox([prev_btn, next_btn, status_lbl])
    return VBox([nav_bar, out])


# ====================================================================
# OPEN-EPHYS EEG + dEMG (RESPIRATION) PIPELINE
# ====================================================================
# Used by Open-Ephys_LabCode_w-Respiration.ipynb. Column layout assumed:
#   0: None / 1: EEG1 / 2: EEG2 / 3: EEG3 (common reference)
#   4: dEMG1 / 5: dEMG2 / 6-8: ADC1-3
# EEG analysis = ref-subtract(cols 1,2 - col3) + notch(60 Hz) + HP(0.5 Hz).
# dEMG analysis = bandpass 100-1000 Hz, bipolar = dEMG2 - dEMG1.

import pandas as pd
import scipy.signal as _sig
from scipy.signal import iirnotch, filtfilt
from scipy.stats import sem, ttest_rel, linregress
try:
    from numpy import trapezoid as _trapz
except ImportError:
    from numpy import trapz as _trapz
import seaborn as sns
import matplotlib.lines as mlines
from ipywidgets import (Dropdown, Checkbox, ToggleButton, FloatText, HTML,
                        Label, HBox)
from IPython.display import display


EEG_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 50),
}

_BAND_COLORS = {
    "Delta": "lightblue",
    "Theta": "lightgreen",
    "Alpha": "yellow",
    "Beta":  "orange",
    "Gamma": "red",
}


def load_eeg_demg_recording(directory, recordnode_idx=0, recording_idx=0,
                            sham_line=1, vns_line=3, processor_id=100,
                            stream_name='Rhythm Data'):
    """Load a session's continuous data + sync events + MessageCenter into a dict."""
    session, recording, record_node_name, experiment_name, recording_name = \
        load_session_recording(directory, recordnode_idx, recording_idx)
    timestamps, data, samplerate, channel_names = load_continuous(recording)

    events = recording.events
    sham_ts = events[(events.line == sham_line) &
                     (events.processor_id == processor_id) &
                     (events.stream_name == stream_name) &
                     (events.state == 1)]['timestamp'].to_numpy()
    vns_ts  = events[(events.line == vns_line) &
                     (events.processor_id == processor_id) &
                     (events.stream_name == stream_name) &
                     (events.state == 1)]['timestamp'].to_numpy()

    message_entries = load_message_center_events(
        recording.directory,
        record_node_name=record_node_name,
        experiment_name=experiment_name,
        recording_name=recording_name,
    )

    print(f"Unique TTL lines: {np.unique(events.line)}")
    print(f"Total events: {len(events)}  |  VNS: {len(vns_ts)}  |  Sham: {len(sham_ts)}")

    return {
        'directory': directory,
        'session': session,
        'recording': recording,
        'samplerate': samplerate,
        'timestamps': timestamps,
        'data': data,
        'channel_names': channel_names,
        'vns_timestamps': vns_ts,
        'sham_timestamps': sham_ts,
        'message_entries': message_entries,
        'record_node_name': record_node_name,
        'experiment_name': experiment_name,
        'recording_name': recording_name,
    }


def subtract_common_reference(data, target_cols, ref_col):
    """Return a copy of `data` where each col in `target_cols` has data[:, ref_col] subtracted."""
    out = data.copy()
    out[:, list(target_cols)] = data[:, list(target_cols)] - data[:, ref_col][:, None]
    return out


def notch_filter(x, fs, freq=60.0, quality_factor=30.0, axis=0):
    """Zero-phase IIR notch filter."""
    b, a = iirnotch(freq / (fs / 2), quality_factor)
    return filtfilt(b, a, x, axis=axis)


def filter_eeg(data, sample_rate, cols, hp_cutoff=0.5, notch_freq=60.0):
    """Apply 60 Hz notch + 4th-order Butterworth high-pass to selected columns.

    Returns a copy of `data` with the chosen columns notch + HP filtered.
    """
    out = data.copy()
    sub = data[:, list(cols)].astype(float, copy=True)
    sub = notch_filter(sub, sample_rate, freq=notch_freq, axis=0)
    b_hp, a_hp = _sig.butter(4, hp_cutoff / (sample_rate / 2), btype='high')
    sub = filtfilt(b_hp, a_hp, sub, axis=0)
    out[:, list(cols)] = sub
    return out


def build_eeg_demg_stack(data_filt_eeg, demg_bipolar,
                         eeg_cols=(1, 2, 3)):
    """Stack [EEG1, EEG2, EEG3(ref), bipolar_dEMG] into one (n_samples, 4) array."""
    cols = [data_filt_eeg[:, c] for c in eeg_cols]
    cols.append(demg_bipolar)
    return np.column_stack(cols)


def build_stim_trial_table(bundle):
    """Combine VNS+SHAM timestamps from a bundle into a sorted trial DataFrame."""
    vns, sham = bundle['vns_timestamps'], bundle['sham_timestamps']
    all_ts = np.concatenate((vns, sham))
    labels = np.concatenate((np.full(len(vns), 'VNS'),
                             np.full(len(sham), 'SHAM')))
    order = np.argsort(all_ts)
    all_ts = all_ts[order]; labels = labels[order]
    return pd.DataFrame({
        'Trial':     np.arange(1, len(all_ts) + 1),
        'Type':      labels,
        'Timestamp': all_ts,
    })


def extract_trial_windows(trial_table, timestamps, data,
                          pre_time=5.0, post_time=15.0,
                          stimulation_duration=1.0):
    """Slice a per-trial pre/post window for each row of `trial_table`.

    `data` may already be the channel subset you want plotted/analyzed
    (use build_eeg_demg_stack to assemble it). Returns a list of dicts.
    """
    trials = []
    for _, row in trial_table.iterrows():
        evt   = row['Timestamp']
        mask  = (timestamps >= evt - pre_time) & (timestamps <= evt + post_time)
        win_t = timestamps[mask] - evt
        win_d = data[mask, :]
        pre_mask  = win_t < 0
        post_mask = win_t >= stimulation_duration
        trials.append({
            'trial': row['Trial'],
            'type':  row['Type'],
            'data':  win_d,
            'timestamps': win_t,
            'pre_data':  win_d[pre_mask, :],
            'post_data': win_d[post_mask, :],
            'pre_timestamps':  win_t[pre_mask],
            'post_timestamps': win_t[post_mask],
        })
    return trials


def compute_trial_psds(all_trials, sample_rate, eeg_bands=EEG_BANDS,
                       psd_cols=None, nperseg=None):
    """Compute per-channel PSD + band power for pre/post windows of each trial.

    Mutates each trial dict adding {phase}_freqs, {phase}_psd,
    {phase}_band_power, {phase}_band_mean, {phase}_band_sem. If `psd_cols`
    is given, PSD is computed only for those columns of trial['pre_data'] /
    trial['post_data']; otherwise all columns are used.
    """
    if nperseg is None:
        nperseg = int(sample_rate)
    for trial in all_trials:
        for phase in ("pre", "post"):
            d = trial[f"{phase}_data"]
            cols = range(d.shape[1]) if psd_cols is None else psd_cols
            freqs_list, psd_list, bp_list = [], [], []
            for ch in cols:
                f, psd = _sig.welch(d[:, ch], fs=sample_rate, nperseg=nperseg,
                                    detrend="constant")
                freqs_list.append(f); psd_list.append(psd)
                bp_list.append({b: _trapz(psd[(f >= lo) & (f <= hi)],
                                          f[(f >= lo) & (f <= hi)])
                                for b, (lo, hi) in eeg_bands.items()})
            trial[f"{phase}_freqs"]      = freqs_list
            trial[f"{phase}_psd"]        = psd_list
            trial[f"{phase}_band_power"] = bp_list
            trial[f"{phase}_band_mean"]  = {
                b: float(np.mean([c[b] for c in bp_list])) for b in eeg_bands}
            trial[f"{phase}_band_sem"]   = {
                b: float(sem([c[b] for c in bp_list], ddof=0, nan_policy='omit'))
                for b in eeg_bands}
        ref_col = (0 if psd_cols is None else list(psd_cols)[0])
        all_data = np.vstack([trial["pre_data"], trial["post_data"]])
        f, psd = _sig.welch(all_data[:, ref_col], fs=sample_rate,
                            nperseg=nperseg, detrend="constant")
        trial["freqs"] = f
        trial["psd"]   = psd
        trial["power"] = {b: _trapz(psd[(f >= lo) & (f <= hi)],
                                    f[(f >= lo) & (f <= hi)])
                          for b, (lo, hi) in eeg_bands.items()}


def build_band_power_dataframe(all_trials, eeg_bands=EEG_BANDS):
    """Flatten trial dicts into a long-form (trial, type, band, pre/post) DataFrame."""
    rows = [{
        "trial": t["trial"], "type": t["type"], "band": b,
        "pre_mean":  t["pre_band_mean"][b],
        "post_mean": t["post_band_mean"][b],
        "pre_sem":   t["pre_band_sem"][b],
        "post_sem":  t["post_band_sem"][b],
    } for t in all_trials for b in eeg_bands]
    df = pd.DataFrame(rows)
    df["normalized"] = df["post_mean"] / df["pre_mean"]
    return df


def plot_continuous_with_events(bundle, data_filt, num_channels=None):
    """Stacked per-channel continuous plot with VNS/Sham/MessageCenter overlays."""
    timestamps    = bundle['timestamps']
    channel_names = bundle['channel_names']
    vns, sham     = bundle['vns_timestamps'], bundle['sham_timestamps']
    msg_entries   = bundle['message_entries']

    if num_channels is None:
        num_channels = max(1, len(channel_names) - 2)
    fig, axes = plt.subplots(num_channels, 1,
                             figsize=(15, 2 * num_channels), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(timestamps, data_filt[:, idx], label=channel_names[idx])
        for t in vns:
            ax.axvline(t, color='red', linestyle='--', alpha=0.8,
                       label='VNS' if t == vns[0] else "")
        for t in sham:
            ax.axvline(t, color='blue', linestyle=':', alpha=0.9,
                       label='Sham' if t == sham[0] else "")
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(channel_names[idx])
        ax.grid(True)
        if idx == 0:
            for t, msg in msg_entries:
                ax.axvline(t, color='green', linestyle='--', alpha=0.6)
                ax.annotate(msg, xy=(t, ax.get_ylim()[1]),
                            xytext=(t + 0.1, ax.get_ylim()[1] * 1.05),
                            rotation=30, fontsize=9, color='green',
                            arrowprops=dict(arrowstyle='->', color='green', lw=1),
                            ha='left')
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle('Open Ephys Continuous Data with Sync + MessageCenter Annotations',
                 fontsize=14)
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.eventplot(vns,  orientation='horizontal', colors='red')
    plt.eventplot(sham, orientation='horizontal', colors='blue')
    plt.title('Sync events (VNS = red, Sham = blue)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_post_band_power_scatter(df_power, eeg_bands=EEG_BANDS):
    """Figure 1: side-by-side scatter of post band power for SHAM vs VNS."""
    post_sham = {b: df_power.loc[(df_power.type == 'SHAM') &
                                 (df_power.band == b), 'post_mean'].values
                 for b in eeg_bands}
    post_vns  = {b: df_power.loc[(df_power.type == 'VNS') &
                                 (df_power.band == b), 'post_mean'].values
                 for b in eeg_bands}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, label, src, marker in [(axes[0], 'SHAM', post_sham, 'o'),
                                   (axes[1], 'VNS',  post_vns,  'x')]:
        for i, (b, v) in enumerate(src.items()):
            ax.scatter(v, [i] * len(v), alpha=0.6, label=b, marker=marker)
        ax.set_title(f"Post {label} band power")
        ax.set_yticks(range(len(src)))
        ax.set_yticklabels(list(src.keys()))
        ax.set_xlabel(f"Post_mean ({label})")
        ax.grid(linestyle='--', alpha=0.6)
    fig.suptitle("Figure 1: Post-stim band power (SHAM vs VNS)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    return post_sham, post_vns


def plot_band_power_heatmaps(post_sham, post_vns):
    """Figures 2a/2b: heatmaps of post band power for SHAM and VNS trials."""
    for label, src, cmap in [('SHAM', post_sham, 'viridis'),
                             ('VNS',  post_vns,  'plasma')]:
        heat = np.array(list(src.values()))
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.heatmap(heat, cmap=cmap, xticklabels=10,
                    yticklabels=list(src.keys()), ax=ax)
        ax.set_xlabel(f"{label} Trial Index")
        ax.set_ylabel("EEG Band")
        ax.set_title(f"Post-{label} Band Power Heatmap")
        plt.tight_layout()
        plt.show(block=False)


def plot_pre_post_bars(df_power, eeg_bands=EEG_BANDS):
    """Figure 4: pre-vs-post bars (SHAM/VNS) with paired t-test stats in legend."""
    bands = list(eeg_bands.keys())
    sham_df = df_power[df_power.type == 'SHAM']
    std_df  = df_power[df_power.type == 'VNS']

    def grp(df, col):
        return (df.groupby('band')[col].mean().reindex(bands),
                df.groupby('band')[col].sem(ddof=0).reindex(bands))
    pre_sm,  pre_ss  = grp(sham_df, 'pre_mean')
    post_sm, post_ss = grp(sham_df, 'post_mean')
    pre_vm,  pre_vs  = grp(std_df,  'pre_mean')
    post_vm, post_vs = grp(std_df,  'post_mean')

    x = np.arange(len(bands)); w = 0.15
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x - 15 * w / 8, pre_sm,  w, yerr=pre_ss,  capsize=5, label="Pre-Sham",  color="blue")
    ax.bar(x -  5 * w / 8, post_sm, w, yerr=post_ss, capsize=5, label="Post-Sham", color="blue", hatch="///")
    ax.bar(x +  5 * w / 8, pre_vm,  w, yerr=pre_vs,  capsize=5, label="Pre-VNS",   color="red")
    ax.bar(x + 15 * w / 8, post_vm, w, yerr=post_vs, capsize=5, label="Post-VNS",  color="red",  hatch="///")

    h_std, h_sham = [], []
    for b in bands:
        ts, ps   = ttest_rel(std_df.loc[std_df.band == b, 'pre_mean'],
                             std_df.loc[std_df.band == b, 'post_mean'])
        tsh, psh = ttest_rel(sham_df.loc[sham_df.band == b, 'pre_mean'],
                             sham_df.loc[sham_df.band == b, 'post_mean'])
        h_std.append(mlines.Line2D([], [], color='none',
                                   label=f"STD {b}: t={ts:.2f}, p={ps:.3f}"))
        h_sham.append(mlines.Line2D([], [], color='none',
                                    label=f"SHAM {b}: t={tsh:.2f}, p={psh:.3f}"))
    legend_sham = ax.legend(handles=h_sham, loc='upper left', fontsize=10, frameon=False)
    ax.add_artist(legend_sham)
    main_h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=main_h + h_std, loc='upper right', fontsize=10, frameon=False)
    ax.set_xlabel("EEG Frequency Bands")
    ax.set_ylabel("Mean Power +/- SEM (uV^2/Hz)")
    ax.set_title("EEG Band Power: Sham vs Standard VNS")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} ({lo}-{hi})" for b, (lo, hi) in eeg_bands.items()])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False)


def plot_normalized_bars(df_power, eeg_bands=EEG_BANDS):
    """Figure 5: normalized (post/pre) band power, SHAM vs VNS."""
    bands = list(eeg_bands.keys())
    sham_df = df_power[df_power.type == 'SHAM']
    std_df  = df_power[df_power.type == 'VNS']

    def grp(df):
        return (df.groupby('band')['normalized'].mean().reindex(bands),
                df.groupby('band')['normalized'].sem(ddof=0).reindex(bands))
    sm, ss = grp(sham_df); vm, vs = grp(std_df)

    x = np.arange(len(bands)); w = 0.3
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x - w / 2, sm, w, yerr=ss, capsize=5, label='Sham',         color='blue')
    ax.bar(x + w / 2, vm, w, yerr=vs, capsize=5, label='Standard VNS', color='red')
    ax.axhline(1, linestyle='--')
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel("Normalized (Post/Pre)")
    ax.set_title("Figure 5: Normalized Band Power (Post/Pre)")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)


def plot_average_psd(all_trials, eeg_bands=EEG_BANDS, phase='post'):
    """Figure 6: average per-trial PSD across channels for SHAM vs VNS."""
    sham_trials = [t for t in all_trials if t['type'] == 'SHAM']
    std_trials  = [t for t in all_trials if t['type'] == 'VNS']

    def mean_sem_psd(group):
        stack = np.array([np.mean(np.vstack(t[f"{phase}_psd"]), axis=0)
                          for t in group])
        return np.mean(stack, axis=0), sem(stack, axis=0, nan_policy='omit')

    sham_m, sham_s = mean_sem_psd(sham_trials)
    std_m,  std_s  = mean_sem_psd(std_trials)
    freqs = (sham_trials[0][f'{phase}_freqs'][0] if sham_trials
             else std_trials[0][f'{phase}_freqs'][0])

    fig, ax = plt.subplots(figsize=(6, 5))
    for b, (lo, hi) in eeg_bands.items():
        ax.axvspan(lo, hi, color=_BAND_COLORS[b], alpha=0.2, label=f"{b} Band")
    ax.plot(freqs, sham_m, color='blue', label='Sham', linewidth=2)
    ax.fill_between(freqs, sham_m - sham_s, sham_m + sham_s, color='blue', alpha=0.3)
    ax.plot(freqs, std_m,  color='red',  label='Standard VNS', linewidth=2)
    ax.fill_between(freqs, std_m - std_s, std_m + std_s, color='red', alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (uV^2/Hz)")
    ax.set_title(f"Figure 6: Average {phase}-stim PSD (Sham vs VNS)")
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False)


def plot_pre_delta_regression(all_trials, band="Delta"):
    """Figure 8: scatter + regression of pre-band power vs post/pre ratio (VNS only)."""
    std_trials = [t for t in all_trials if t['type'] == 'VNS']
    x = np.array([t['pre_band_mean'][band] for t in std_trials])
    y = np.array([t['post_band_mean'][band] / t['pre_band_mean'][band]
                  for t in std_trials])
    slope, intercept, r, p, _ = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    colors = ['red' if v < 1 else 'blue' for v in y]
    plt.figure(figsize=(16, 12))
    plt.scatter(x, y, c=colors, alpha=0.7)
    plt.plot(x_line, slope * x_line + intercept)
    plt.axhline(1, linestyle='--')
    plt.xlabel(f"Pre {band} Power")
    plt.ylabel(f"Normalized {band} (Post/Pre)")
    plt.title(f"Pre {band} vs Reduction\nR={r:.2f}, p={p:.3f}")
    plt.text(0.95, 0.95,
             f"Normalized < 1: {int(np.sum(y < 1))}\n"
             f"Normalized >= 1: {int(np.sum(y >= 1))}",
             ha='right', va='top', transform=plt.gca().transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
    plt.show()


def compute_windowed_features(trial, sample_rate, eeg_bands=EEG_BANDS,
                              samples_per_sec=None, psd_cols=None):
    """Compute per-second PSD/band-power features for one trial's pre/post data."""
    if samples_per_sec is None:
        samples_per_sec = int(sample_rate)
    out = {"pre": [], "post": []}
    for phase in ("pre", "post"):
        d = trial[f"{phase}_data"]
        cols = range(d.shape[1]) if psd_cols is None else psd_cols
        for start in range(0, d.shape[0], samples_per_sec):
            window = d[start:start + samples_per_sec]
            if window.shape[0] < samples_per_sec:
                continue
            wr = {"freqs": [], "psd": [], "band_power": []}
            for ch in cols:
                f, psd = _sig.welch(window[:, ch], fs=sample_rate,
                                    nperseg=samples_per_sec, detrend="constant")
                wr["freqs"].append(f); wr["psd"].append(psd)
                wr["band_power"].append({b: _trapz(psd[(f >= lo) & (f <= hi)],
                                                   f[(f >= lo) & (f <= hi)])
                                          for b, (lo, hi) in eeg_bands.items()})
            out[phase].append(wr)
    return out


def plot_windowed_band_power(features, band="Delta", channel=0):
    """Plot a single-channel single-band 1-second windowed band power timeline."""
    pre  = [w["band_power"][channel][band] for w in features["pre"]]
    post = [w["band_power"][channel][band] for w in features["post"]]
    pre_t  = np.arange(len(pre))
    post_t = np.arange(len(post)) + len(pre)
    plt.figure()
    plt.plot(pre_t,  pre,  label="Pre")
    plt.plot(post_t, post, label="Post")
    plt.axvline(len(pre) - 1, linestyle="--")
    plt.xlabel("Time (1s bins)")
    plt.ylabel(f"{band} Power")
    plt.title(f"{band} Power Over Time (Channel {channel})")
    plt.legend()
    plt.show()


def make_eeg_demg_trial_viewer(all_trials, samplerate, eeg_bands=EEG_BANDS,
                               eeg_labels=None,
                               demg_label='Bipolar dEMG (dEMG2-dEMG1)',
                               stimulation_duration=1.0):
    """Interactive viewer: 3 EEG panels + bipolar-dEMG panel per trial.

    Each trial in `all_trials` must have `data` of shape (n_samples, 4) where
    columns are [EEG1, EEG2, EEG3(ref), bipolar_dEMG]. PSD/Power for the
    second figure is computed on the fly from the chosen pre/post windows.
    """
    if eeg_labels is None:
        eeg_labels = ['EEG 1', 'EEG 2', 'EEG 3 (ref)']
    panel_labels = list(eeg_labels) + [demg_label]
    n_panels = len(panel_labels)
    n_trials = len(all_trials)
    demg_idx = n_panels - 1

    state = {
        'idx': 0, 'rectify': False,
        'auto_x': True, 'auto_y': True, 'auto_stim': False, 'units': 's',
        'x_min': -5.0, 'x_max': 15.0,
        'y_min': -500.0, 'y_max': 500.0,
        'stim_end': 0.5, 'pre_start': -2.0, 'pre_end': 0.0,
        'post_start': 0.5, 'post_end': 2.75,
        'focus': -1,
        'psd_mode': 'PSD', 'auto_psd_y': True,
        'psd_y_min': 0.0, 'psd_y_max': 1.0,
    }
    trial_out = Output()
    psd_out   = Output()

    def unit_factor(units):
        return 1000.0 if units == 'ms' else 1.0

    def set_silent(box, observer, value):
        box.unobserve(observer, names='value')
        box.value = float(value)
        box.observe(observer, names='value')

    def window_indicator(ax, t1, t2, color, ylim):
        if t1 >= t2:
            return
        y_bot = ylim[0] + 0.04 * (ylim[1] - ylim[0])
        ax.hlines(y_bot, t1, t2, color=color, linewidth=4, alpha=0.9)

    def draw_panel(ax, t_disp, y, title, units, stim_end_disp,
                   x_lim, y_lim, pre_disp, post_disp,
                   color='C0', linewidth=0.7):
        ax.plot(t_disp, y, color=color, linewidth=linewidth)
        ax.axvspan(0, stim_end_disp, color='red', alpha=0.10)
        ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(stim_end_disp, color='red', linestyle='--', linewidth=1, alpha=0.8)
        window_indicator(ax, pre_disp[0],  pre_disp[1],  'darkblue', y_lim)
        window_indicator(ax, post_disp[0], post_disp[1], 'darkred',  y_lim)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(f"Time ({units})", fontsize=18)
        ax.set_ylabel('uV', fontsize=20)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    def auto_y_for(arr):
        a = arr[~np.isnan(arr)] if arr.size else arr
        if not a.size:
            return -1.0, 1.0
        lo, hi = float(np.min(a)), float(np.max(a))
        pad = max(0.05 * (hi - lo), 1.0)
        return lo - pad, hi + pad

    def compute_psd(seg, fs):
        if len(seg) < 16:
            return None, None, None, None
        nperseg  = min(int(fs), len(seg))
        noverlap = nperseg // 2 if nperseg > 1 else 0
        f, _, Sxx = _sig.spectrogram(seg, fs=fs, nperseg=nperseg,
                                     noverlap=noverlap, detrend='constant',
                                     scaling='density')
        if Sxx.shape[1] == 0:
            return None, None, None, None
        m = np.mean(Sxx, axis=1)
        s = (sem(Sxx, axis=1, nan_policy='omit')
             if Sxx.shape[1] >= 2 else np.zeros_like(m))
        df = float(f[1] - f[0]) if len(f) >= 2 else 1.0
        return f, m, s, df

    def draw_psd_panel(ax, f_pre, m_pre, s_pre, f_post, m_post, s_post,
                       title, ylabel, y_lim):
        for b, (lo, hi) in eeg_bands.items():
            ax.axvspan(lo, hi, color=_BAND_COLORS[b], alpha=0.2, label=f"{b} Band")
        if f_pre is not None:
            ax.plot(f_pre, m_pre, color='darkblue', linewidth=2, label='Pre-VNS')
            ax.fill_between(f_pre, m_pre - s_pre, m_pre + s_pre,
                            color='darkblue', alpha=0.3)
        if f_post is not None:
            ax.plot(f_post, m_post, color='darkred', linewidth=2, label='Post-VNS')
            ax.fill_between(f_post, m_post - s_post, m_post + s_post,
                            color='darkred', alpha=0.3)
        ax.set_xlim([0, 50]); ax.set_ylim(y_lim)
        ax.set_xlabel("Frequency (Hz)", fontsize=18)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.tick_params(labelsize=11)
        ax.grid(linestyle='--', alpha=0.7)

    def dedup_legend(ax, **kw):
        h, l = ax.get_legend_handles_labels()
        seen, h2, l2 = set(), [], []
        for hh, ll in zip(h, l):
            if ll in seen:
                continue
            seen.add(ll); h2.append(hh); l2.append(ll)
        ax.legend(h2, l2, **kw)

    def panel_color_lw(i):
        if i == demg_idx:
            return 'black', 1.0
        return 'C0', 0.7

    def plot_trial(idx):
        with trial_out:
            trial_out.clear_output(wait=True)
            trial  = all_trials[idx]
            units  = state['units']
            factor = unit_factor(units)
            t_disp = trial['timestamps'] * factor

            if state['auto_stim']:
                stim_end_int = float(stimulation_duration)
                state['stim_end'] = stim_end_int
                set_silent(stim_end_box, on_stim_end,
                           round(stim_end_int * factor, 4))
            else:
                stim_end_int = float(state['stim_end'])
            stim_end_disp = stim_end_int * factor

            sigs = [trial['data'][:, i] for i in range(n_panels)]
            if state['rectify']:
                sigs = [np.abs(s) for s in sigs]

            if state['auto_x']:
                state['x_min'] = float(t_disp[0])
                state['x_max'] = float(t_disp[-1])
                set_silent(xmin_box, on_xmin, round(state['x_min'], 4))
                set_silent(xmax_box, on_xmax, round(state['x_max'], 4))
            x_lim = (state['x_min'], state['x_max'])

            # Per-panel auto-y when auto_y is on; manual y applies globally otherwise.
            if state['auto_y']:
                y_lims = [auto_y_for(s) for s in sigs]
                lo, hi = y_lims[0]
                set_silent(ymin_box, on_ymin, round(lo, 4))
                set_silent(ymax_box, on_ymax, round(hi, 4))
                state['y_min'], state['y_max'] = lo, hi
            else:
                y_lims = [(state['y_min'], state['y_max'])] * n_panels

            pre_disp  = (state['pre_start']  * factor, state['pre_end']  * factor)
            post_disp = (state['post_start'] * factor, state['post_end'] * factor)
            rec_tag   = '  |rectified|' if state['rectify'] else ''
            focus     = state['focus']
            suptitle  = (f"Trial {trial['trial']} ({trial['type']})  -  "
                         f"filtered + ref-subtracted{rec_tag}   ({idx + 1}/{n_trials})")

            legend_handles = [
                mlines.Line2D([0], [0], color='darkblue', linewidth=4, label='Pre-VNS window'),
                mlines.Line2D([0], [0], color='darkred',  linewidth=4, label='Post-VNS window'),
            ]

            if focus == -1:
                rows = int(np.ceil(n_panels / 2))
                fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows + 2),
                                         sharex=False, sharey=False)
                axes_flat = np.atleast_1d(axes).flatten()
                for i, (sig, label) in enumerate(zip(sigs, panel_labels)):
                    color, lw = panel_color_lw(i)
                    draw_panel(axes_flat[i], t_disp, sig, label, units,
                               stim_end_disp, x_lim, y_lims[i], pre_disp, post_disp,
                               color=color, linewidth=lw)
                for i in range(n_panels, len(axes_flat)):
                    axes_flat[i].axis('off')
                fig.legend(handles=legend_handles, loc='upper right', ncol=2,
                           fontsize=11, frameon=False, bbox_to_anchor=(0.98, 0.985))
                fig.suptitle(suptitle, fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
            else:
                fig, ax = plt.subplots(figsize=(14, 8))
                color, lw = panel_color_lw(focus)
                draw_panel(ax, t_disp, sigs[focus], panel_labels[focus], units,
                           stim_end_disp, x_lim, y_lims[focus], pre_disp, post_disp,
                           color=color, linewidth=lw if state['focus'] != -1 else lw + 0.2)
                ax.legend(handles=legend_handles, loc='upper right',
                          fontsize=12, frameon=False)
                fig.suptitle(suptitle, fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        plot_psd(idx)

    def plot_psd(idx):
        with psd_out:
            psd_out.clear_output(wait=True)
            trial = all_trials[idx]
            ts = trial['timestamps']
            ps, pe = state['pre_start'],  state['pre_end']
            qs, qe = state['post_start'], state['post_end']
            if ps >= pe or qs >= qe:
                print("Pre or Post window is empty (start >= end). Adjust the boxes.")
                return
            pre_mask  = (ts >= ps) & (ts < pe)
            post_mask = (ts >= qs) & (ts < qe)
            if pre_mask.sum() < 16 or post_mask.sum() < 16:
                print("Pre or Post window has too few samples. Widen the window.")
                return

            sigs = [trial['data'][:, i] for i in range(n_panels)]
            if state['rectify']:
                sigs = [np.abs(s) for s in sigs]
            focus  = state['focus']
            mode   = state['psd_mode']
            ylabel = ('Power (V^2)' if mode == 'Power'
                      else 'Power Density (V^2/Hz)')
            units  = state['units']
            factor = unit_factor(units)
            suptitle = (f"Pre vs Post {mode}  -  Trial {trial['trial']} ({trial['type']})  -  "
                        f"Pre [{ps*factor:.2f}, {pe*factor:.2f}]{units}  /  "
                        f"Post [{qs*factor:.2f}, {qe*factor:.2f}]{units}")

            if focus == -1:
                panels_in = list(zip(sigs, panel_labels))
            else:
                panels_in = [(sigs[focus], panel_labels[focus])]

            panels_data = []
            for sig, t in panels_in:
                f_pre,  m_pre,  s_pre,  df_pre  = compute_psd(sig[pre_mask],  samplerate)
                f_post, m_post, s_post, df_post = compute_psd(sig[post_mask], samplerate)
                if mode == 'Power':
                    if m_pre is not None:
                        m_pre  = m_pre  * df_pre;  s_pre  = s_pre  * df_pre
                    if m_post is not None:
                        m_post = m_post * df_post; s_post = s_post * df_post
                panels_data.append((f_pre, m_pre, s_pre, f_post, m_post, s_post, t))

            tops = []
            for f_pre, m_pre, s_pre, f_post, m_post, s_post, _t in panels_data:
                for f_arr, m_arr, s_arr in [(f_pre, m_pre, s_pre),
                                            (f_post, m_post, s_post)]:
                    if f_arr is None:
                        continue
                    vis = (f_arr >= 0) & (f_arr <= 50)
                    if not vis.any():
                        continue
                    tops.append(float(np.nanmax((m_arr + s_arr)[vis])))
            auto_max = (max(tops) * 1.05) if tops else 1.0

            if state['auto_psd_y']:
                set_silent(psd_ymin_box, on_psd_ymin, 0.0)
                set_silent(psd_ymax_box, on_psd_ymax, round(auto_max, 4))
                state['psd_y_min'] = 0.0
                state['psd_y_max'] = auto_max
            y_lim = (state['psd_y_min'], state['psd_y_max'])

            if focus == -1:
                rows = int(np.ceil(n_panels / 2))
                fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows + 2),
                                         sharex=False, sharey=False)
                axes_flat = np.atleast_1d(axes).flatten()
                for i, panel in enumerate(panels_data):
                    f_pre, m_pre, s_pre, f_post, m_post, s_post, t = panel
                    draw_psd_panel(axes_flat[i], f_pre, m_pre, s_pre,
                                   f_post, m_post, s_post, t, ylabel, y_lim)
                    dedup_legend(axes_flat[i], loc='upper right',
                                 fontsize=9, frameon=False)
                for i in range(n_panels, len(axes_flat)):
                    axes_flat[i].axis('off')
                fig.suptitle(suptitle, fontsize=15)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                fig, ax = plt.subplots(figsize=(14, 8))
                f_pre, m_pre, s_pre, f_post, m_post, s_post, t = panels_data[0]
                draw_psd_panel(ax, f_pre, m_pre, s_pre, f_post, m_post, s_post,
                               t, ylabel, y_lim)
                dedup_legend(ax, loc='upper right', fontsize=12, frameon=False)
                fig.suptitle(suptitle, fontsize=15)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

    # ---- handlers ----
    def on_prev(b):
        new = max(state['idx'] - 1, 0)
        if new != state['idx']:
            trial_drop.value = new

    def on_next(b):
        new = min(state['idx'] + 1, n_trials - 1)
        if new != state['idx']:
            trial_drop.value = new

    def on_trial_change(change):
        if change['name'] == 'value':
            state['idx'] = int(change['new'])
            plot_trial(state['idx'])

    def on_focus(change):
        if change['name'] == 'value':
            state['focus'] = int(change['new'])
            plot_trial(state['idx'])

    def on_rectify(change):
        state['rectify'] = bool(change['new'])
        plot_trial(state['idx'])

    def on_units(change):
        new_units = 'ms' if change['new'] else 's'
        if new_units == state['units']:
            return
        old_f = unit_factor(state['units']); new_f = unit_factor(new_units)
        state['x_min'] *= new_f / old_f; state['x_max'] *= new_f / old_f
        state['units'] = new_units
        units_toggle.description   = f'X-axis: {new_units}'
        xmin_box.description       = f'X min ({new_units}):'
        xmax_box.description       = f'X max ({new_units}):'
        stim_end_box.description   = f'Stim end ({new_units}):'
        pre_start_box.description  = f'Pre start ({new_units}):'
        pre_end_box.description    = f'Pre end ({new_units}):'
        post_start_box.description = f'Post start ({new_units}):'
        post_end_box.description   = f'Post end ({new_units}):'
        for box, obs, key in [(stim_end_box, on_stim_end, 'stim_end'),
                              (pre_start_box,  on_pre_start,  'pre_start'),
                              (pre_end_box,    on_pre_end,    'pre_end'),
                              (post_start_box, on_post_start, 'post_start'),
                              (post_end_box,   on_post_end,   'post_end')]:
            set_silent(box, obs, round(state[key] * new_f, 4))
        plot_trial(state['idx'])

    def on_auto_x(change):
        state['auto_x'] = bool(change['new'])
        xmin_box.disabled = bool(change['new'])
        xmax_box.disabled = bool(change['new'])
        auto_x_toggle.description = 'Auto x-scale' if change['new'] else 'Manual x-scale'
        plot_trial(state['idx'])

    def on_auto_y(change):
        state['auto_y'] = bool(change['new'])
        ymin_box.disabled = bool(change['new'])
        ymax_box.disabled = bool(change['new'])
        auto_y_toggle.description = ('Auto y (per panel)' if change['new']
                                     else 'Manual y-scale')
        plot_trial(state['idx'])

    def on_auto_stim(change):
        state['auto_stim'] = bool(change['new'])
        stim_end_box.disabled = bool(change['new'])
        auto_stim_toggle.description = ('Auto stim end' if change['new']
                                        else 'Manual stim end')
        plot_trial(state['idx'])

    def on_xmin(change):
        state['x_min'] = float(change['new'])
        if not state['auto_x']:
            plot_trial(state['idx'])
    def on_xmax(change):
        state['x_max'] = float(change['new'])
        if not state['auto_x']:
            plot_trial(state['idx'])
    def on_ymin(change):
        state['y_min'] = float(change['new'])
        if not state['auto_y']:
            plot_trial(state['idx'])
    def on_ymax(change):
        state['y_max'] = float(change['new'])
        if not state['auto_y']:
            plot_trial(state['idx'])
    def on_stim_end(change):
        state['stim_end'] = float(change['new']) / unit_factor(state['units'])
        if not state['auto_stim']:
            plot_trial(state['idx'])
    def on_pre_start(change):
        state['pre_start'] = float(change['new']) / unit_factor(state['units'])
        plot_trial(state['idx'])
    def on_pre_end(change):
        state['pre_end'] = float(change['new']) / unit_factor(state['units'])
        plot_trial(state['idx'])
    def on_post_start(change):
        state['post_start'] = float(change['new']) / unit_factor(state['units'])
        plot_trial(state['idx'])
    def on_post_end(change):
        state['post_end'] = float(change['new']) / unit_factor(state['units'])
        plot_trial(state['idx'])

    def on_psd_mode(change):
        state['psd_mode'] = 'Power' if change['new'] else 'PSD'
        psd_mode_toggle.description = f"Mode: {state['psd_mode']}"
        plot_psd(state['idx'])
    def on_auto_psd_y(change):
        state['auto_psd_y'] = bool(change['new'])
        psd_ymin_box.disabled = bool(change['new'])
        psd_ymax_box.disabled = bool(change['new'])
        auto_psd_y_toggle.description = ('Auto y-scale (PSD)' if change['new']
                                         else 'Manual y-scale (PSD)')
        plot_psd(state['idx'])
    def on_psd_ymin(change):
        state['psd_y_min'] = float(change['new'])
        if not state['auto_psd_y']:
            plot_psd(state['idx'])
    def on_psd_ymax(change):
        state['psd_y_max'] = float(change['new'])
        if not state['auto_psd_y']:
            plot_psd(state['idx'])

    # ---- widgets ----
    prev_btn = Button(description='Prev')
    next_btn = Button(description='Next', button_style='primary')
    trial_drop = Dropdown(
        options=[(f"Trial {tr['trial']:>3d}  ({tr['type']})", i)
                 for i, tr in enumerate(all_trials)],
        value=0, description='Trial:', layout={'width': '260px'})
    focus_drop = Dropdown(
        options=[('Grid (all)', -1)] + [(lbl, i) for i, lbl in enumerate(panel_labels)],
        value=-1, description='Focus:', layout={'width': '260px'})
    rectify_cb = Checkbox(value=False, description='Rectify (|signal|)',
                          indent=False, layout={'width': '200px'})
    units_toggle = ToggleButton(value=False, description='X-axis: s',
                                button_style='info', layout={'width': '140px'})
    auto_x_toggle = ToggleButton(value=True, description='Auto x-scale',
                                 button_style='success', layout={'width': '160px'})
    auto_y_toggle = ToggleButton(value=True, description='Auto y (per panel)',
                                 button_style='success', layout={'width': '180px'})
    auto_stim_toggle = ToggleButton(value=False, description='Manual stim end',
                                    button_style='success', layout={'width': '160px'})

    u = state['units']
    xmin_box = FloatText(value=state['x_min'], description=f"X min ({u}):",
                         disabled=True, layout={'width': '170px'})
    xmax_box = FloatText(value=state['x_max'], description=f"X max ({u}):",
                         disabled=True, layout={'width': '170px'})
    ymin_box = FloatText(value=state['y_min'], description='Y min:',
                         disabled=True, layout={'width': '155px'})
    ymax_box = FloatText(value=state['y_max'], description='Y max:',
                         disabled=True, layout={'width': '155px'})
    stim_end_box = FloatText(value=state['stim_end'] * unit_factor(u),
                             description=f"Stim end ({u}):",
                             disabled=False, layout={'width': '180px'})
    pre_start_box  = FloatText(value=state['pre_start']  * unit_factor(u),
                               description=f"Pre start ({u}):",
                               layout={'width': '180px'})
    pre_end_box    = FloatText(value=state['pre_end']    * unit_factor(u),
                               description=f"Pre end ({u}):",
                               layout={'width': '180px'})
    post_start_box = FloatText(value=state['post_start'] * unit_factor(u),
                               description=f"Post start ({u}):",
                               layout={'width': '180px'})
    post_end_box   = FloatText(value=state['post_end']   * unit_factor(u),
                               description=f"Post end ({u}):",
                               layout={'width': '180px'})
    psd_mode_toggle    = ToggleButton(value=False, description='Mode: PSD',
                                      button_style='info', layout={'width': '160px'})
    auto_psd_y_toggle  = ToggleButton(value=True, description='Auto y-scale (PSD)',
                                      button_style='success', layout={'width': '180px'})
    psd_ymin_box = FloatText(value=state['psd_y_min'], description='PSD Y min:',
                             disabled=True, layout={'width': '170px'})
    psd_ymax_box = FloatText(value=state['psd_y_max'], description='PSD Y max:',
                             disabled=True, layout={'width': '170px'})

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)
    trial_drop.observe(on_trial_change, names='value')
    focus_drop.observe(on_focus, names='value')
    rectify_cb.observe(on_rectify, names='value')
    units_toggle.observe(on_units, names='value')
    auto_x_toggle.observe(on_auto_x, names='value')
    auto_y_toggle.observe(on_auto_y, names='value')
    auto_stim_toggle.observe(on_auto_stim, names='value')
    xmin_box.observe(on_xmin, names='value')
    xmax_box.observe(on_xmax, names='value')
    ymin_box.observe(on_ymin, names='value')
    ymax_box.observe(on_ymax, names='value')
    stim_end_box.observe(on_stim_end, names='value')
    pre_start_box.observe(on_pre_start, names='value')
    pre_end_box.observe(on_pre_end, names='value')
    post_start_box.observe(on_post_start, names='value')
    post_end_box.observe(on_post_end, names='value')
    psd_mode_toggle.observe(on_psd_mode, names='value')
    auto_psd_y_toggle.observe(on_auto_psd_y, names='value')
    psd_ymin_box.observe(on_psd_ymin, names='value')
    psd_ymax_box.observe(on_psd_ymax, names='value')

    nav_row = HBox([prev_btn, next_btn, trial_drop, focus_drop,
                    Label('  '), rectify_cb, units_toggle])
    axis_row = VBox([
        HTML('<b>Axis controls</b>  (boxes show current auto-scaled values; toggle off to edit)'),
        HBox([auto_x_toggle,    xmin_box, xmax_box]),
        HBox([auto_y_toggle,    ymin_box, ymax_box]),
        HBox([auto_stim_toggle, stim_end_box]),
    ])
    window_row = VBox([
        HTML('<b>PSD analysis windows</b>  '
             '(Pre = dark blue, Post = dark red; values in current units)'),
        HBox([Label('Pre:'),  pre_start_box,  pre_end_box]),
        HBox([Label('Post:'), post_start_box, post_end_box]),
    ])
    psd_row = VBox([
        HTML('<b>PSD figure controls</b>  '
             '(switch between PSD and Power, set y-axis range for the second figure)'),
        HBox([psd_mode_toggle, auto_psd_y_toggle, psd_ymin_box, psd_ymax_box]),
    ])

    print(f"Loaded {n_trials} trials. Use Prev/Next or the dropdown to navigate.")
    ui = VBox([nav_row, axis_row, window_row, psd_row, trial_out, psd_out])
    plot_trial(0)
    return ui


# ====================================================================
# POST-HOC BIN ANALYSIS HELPERS
# ====================================================================


def get_trial_bg_emg(trial):
    """Return per-trial pre-stimulus background EMG grand mean (µV), or NaN."""
    stored_gm = getattr(trial, 'background_emg_mean', None)
    if stored_gm is not None:
        v = float(stored_gm)
        if np.isfinite(v) and v > 0:
            return v
    stored_bins = getattr(trial, 'background_bins', None)
    if stored_bins is not None and len(stored_bins) > 0:
        return float(np.mean(stored_bins))
    return float('nan')


def get_group_bg_stats(trials):
    """Return (mean, lo, hi) background EMG for a list of trials (µV).

    Returns (nan, nan, nan) if no background data is available.
    """
    bgs = [get_trial_bg_emg(t) for t in trials]
    bgs = [x for x in bgs if np.isfinite(x)]
    if not bgs:
        return float('nan'), float('nan'), float('nan')
    return float(np.mean(bgs)), float(min(bgs)), float(max(bgs))


def build_merged_amp_groups(hrs2_trials, merged_specs, tol=0.001):
    """Pool trials from multiple stim-amplitude groups into merged groups.

    Parameters
    ----------
    hrs2_trials : list
        Original HRS2 trial list.
    merged_specs : list of list of float
        Each inner list contains amplitude values (mA) to merge.
        E.g. ``[[0.12, 0.13], [0.15, 0.16, 0.17]]`` creates 2 merged groups.
    tol : float
        Amplitude matching tolerance (mA).

    Returns
    -------
    list
        New trial list.  Trials belonging to a merged group have
        ``stimulation_amplitude_ma`` set to that group's mean amplitude;
        un-merged trials are appended unchanged.  Merged copies carry
        ``_is_merged``, ``_merged_amp_lo``, and ``_merged_amp_hi`` attributes
        which ``plot_hrs2_analysis`` uses to render the range label.

    Example
    -------
    >>> MERGED_GROUPS = [[0.12, 0.13], [0.15, 0.16]]
    >>> merged = build_merged_amp_groups(hrs2_trials, MERGED_GROUPS)
    >>> plot_hrs2_analysis(merged, hrs2_header, ...)
    """
    from copy import copy as _copy
    result = []
    used_ids = set()
    for amp_list in merged_specs:
        amp_list_f = [float(a) for a in amp_list]
        avg_amp = float(np.mean(amp_list_f))
        amp_lo  = min(amp_list_f)
        amp_hi  = max(amp_list_f)
        group_trials = [t for t in hrs2_trials
                        if any(abs(t.stimulation_amplitude_ma - a) <= tol
                               for a in amp_list_f)]
        if not group_trials:
            print(f"Warning: no trials matched merged group "
                  f"{[f'{a:.3f}' for a in amp_list_f]} mA (tol={tol})")
            continue
        for t in group_trials:
            used_ids.add(id(t))
            try:
                t_copy = _copy(t)
                t_copy.stimulation_amplitude_ma = avg_amp
            except Exception:
                import dataclasses
                t_copy = dataclasses.replace(t, stimulation_amplitude_ma=avg_amp)
            t_copy._is_merged     = True
            t_copy._merged_amp_lo = amp_lo
            t_copy._merged_amp_hi = amp_hi
            result.append(t_copy)
        print(f"Merged group {[f'{a:.3f}' for a in amp_list_f]} mA "
              f"-> avg {avg_amp:.3f} mA, {len(group_trials)} trials")
    for t in hrs2_trials:
        if id(t) not in used_ids:
            result.append(t)
    return result


def filter_trials(hrs2_trials, state, filter_polarity=None, filter_intensities=None):
    """Filter an HRS2 trial list by polarity and/or stimulation intensity.

    Parameters
    ----------
    hrs2_trials        : list of MhRecTrial
    state              : dict from plot_background_grand_means (needs 'trial_bg_gm')
    filter_polarity    : 'normal' | 'reversed' | None  (None = keep all)
    filter_intensities : None | (lo, hi) mA range | list of exact mA values

    Returns
    -------
    (filtered_trials, filtered_state)
        filtered_state has 'trial_bg_gm' re-indexed to match filtered_trials.
    """
    if filter_polarity is not None and not isinstance(filter_polarity, str):
        raise ValueError(
            "filter_polarity must be 'normal', 'reversed', or None — did you forget the quotes?"
        )

    result = list(hrs2_trials)

    if filter_polarity is not None:
        pol_val = 0 if filter_polarity.lower().startswith('n') else 1
        result  = [t for t in result if getattr(t, 'stim_polarity_reversed', 0) == pol_val]
        print(f"Polarity filter '{filter_polarity}': {len(result)}/{len(hrs2_trials)} trials kept")

    if filter_intensities is not None:
        n_before = len(result)
        if isinstance(filter_intensities, tuple):
            lo, hi = filter_intensities
            result  = [t for t in result if lo <= t.stimulation_amplitude_ma <= hi]
            print(f"Intensity range filter [{lo}, {hi}] mA: {len(result)}/{n_before} trials kept")
        else:
            target = {round(float(a), 3) for a in filter_intensities}
            result  = [t for t in result
                       if round(t.stimulation_amplitude_ma, 3) in target]
            print(f"Intensity list filter {sorted(target)} mA: {len(result)}/{n_before} trials kept")

    if filter_polarity is None and filter_intensities is None:
        print(f"No filter applied — using all {len(result)} trials")

    bg_by_id       = {id(t): state['trial_bg_gm'][i] for i, t in enumerate(hrs2_trials)}
    filtered_state = {**state, 'trial_bg_gm': np.array([bg_by_id[id(t)] for t in result])}
    return result, filtered_state


def compute_equal_bin_ranges(
    trials, bin_mode, n_bins, state, sample_rate,
    pre_avg_ms, post_avg_ms, m_start_ms, m_end_ms, h_start_ms, h_end_ms,
):
    """Compute bin boundaries that produce ~equal trial counts per bin.

    Uses percentiles of the per-trial metric (EMG background, M-wave MRA, or
    H-wave MRA) to find edges that divide the filtered trial distribution into
    n_bins roughly equal groups.

    Returns
    -------
    list of (lo, hi) tuples suitable for BIN_RANGES in compute_trial_bins.
    The last bin's upper bound is float('inf') to capture all remaining trials.
    """
    ms_per_sample = 1000.0 / sample_rate

    if bin_mode == 'EMG':
        metrics = np.array(state['trial_bg_gm'], dtype=float)
    elif bin_mode in ('M_WAVE', 'H_WAVE'):
        start_ms = m_start_ms if bin_mode == 'M_WAVE' else h_start_ms
        end_ms   = m_end_ms   if bin_mode == 'M_WAVE' else h_end_ms
        vals = []
        for t in trials:
            tm, emg, _, _, _ = get_trial_window(t, pre_avg_ms, post_avg_ms,
                                                ms_per_sample=ms_per_sample)
            mask = (tm >= start_ms) & (tm <= end_ms)
            vals.append(float(np.nanmean(np.abs(emg[mask]))) if mask.any() else float('nan'))
        metrics = np.array(vals, dtype=float)
    else:
        raise ValueError(f"Unknown bin_mode {bin_mode!r}. Choose 'EMG', 'M_WAVE', or 'H_WAVE'.")

    valid = metrics[~np.isnan(metrics)]
    if len(valid) < n_bins:
        raise ValueError(f"Only {len(valid)} valid trials — cannot form {n_bins} bins.")

    edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
    for i in range(1, len(edges)):          # guarantee strictly increasing
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 0.01
    edges = np.round(edges, 2)

    # Last bin's upper bound = actual max + small epsilon so the max-value trial
    # is captured by compute_trial_bins's (lo <= metric < hi) condition.
    last_hi = round(float(valid.max()) + 0.01, 2)

    ranges = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1]) if i < n_bins - 1 else last_hi
        ranges.append((lo, hi))

    unit_map = {'EMG': 'µV bg', 'M_WAVE': 'µV M', 'H_WAVE': 'µV H'}
    unit = unit_map[bin_mode]
    print(f"Auto-bin: {n_bins} equal-count bins  |  {len(valid)} trials  |  {unit}")
    for i, (lo, hi) in enumerate(ranges):
        n      = int(np.sum((metrics >= lo) & (metrics < hi)))
        hi_str = str(hi)
        print(f"  Bin {i + 1}: [{lo}, {hi_str})  →  {n} trials")

    return ranges


def compute_trial_bins(
    hrs2_trials, bin_mode, bin_ranges, state,
    sample_rate, pre_avg_ms, post_avg_ms,
    m_start_ms, m_end_ms, h_start_ms, h_end_ms,
):
    """Classify trials into user-defined bins by background EMG, M-wave MRA, or H-wave MRA.

    Parameters
    ----------
    bin_mode   : 'EMG', 'M_WAVE', or 'H_WAVE'
    bin_ranges : list of (lo, hi) tuples
    state      : dict from analyze_global_background (needs 'trial_bg_gm')

    Returns
    -------
    binned_trials, bin_labels, bin_colors, trial_bg_dict, bin_unit, pol_labels, trials_by_pol
    """
    ms_per_sample = 1000.0 / sample_rate
    trial_bg_gm   = state['trial_bg_gm']
    trial_bg_dict = {id(t): trial_bg_gm[i] for i, t in enumerate(hrs2_trials)}

    if bin_mode == 'EMG':
        bin_unit     = 'µV bg'
        trial_metric = trial_bg_dict.copy()
    elif bin_mode == 'M_WAVE':
        bin_unit     = 'µV M'
        trial_metric = {}
        for t in hrs2_trials:
            tm, emg, _, _, _ = get_trial_window(t, pre_avg_ms, post_avg_ms, ms_per_sample=ms_per_sample)
            mask = (tm >= m_start_ms) & (tm <= m_end_ms)
            trial_metric[id(t)] = float(np.nanmean(np.abs(emg[mask]))) if mask.any() else float('nan')
    elif bin_mode == 'H_WAVE':
        bin_unit     = 'µV H'
        trial_metric = {}
        for t in hrs2_trials:
            tm, emg, _, _, _ = get_trial_window(t, pre_avg_ms, post_avg_ms, ms_per_sample=ms_per_sample)
            mask = (tm >= h_start_ms) & (tm <= h_end_ms)
            trial_metric[id(t)] = float(np.nanmean(np.abs(emg[mask]))) if mask.any() else float('nan')
    else:
        raise ValueError(f"Unknown bin_mode {bin_mode!r}. Choose 'EMG', 'M_WAVE', or 'H_WAVE'.")

    bin_labels = [
        f'Bin {i+1}: {lo}–{"∞" if hi == float("inf") else hi} {bin_unit}'
        + (f'  [c={(lo + hi) / 2:.4g}, Δ{hi - lo:.4g}]' if hi != float('inf') else '')
        for i, (lo, hi) in enumerate(bin_ranges)
    ]
    bin_colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple'][:len(bin_ranges)]

    trials_by_pol = split_trials_by_polarity(hrs2_trials)
    pol_labels    = list(trials_by_pol.keys())

    binned_trials = {
        pol: {
            lbl: [t for t in trs if lo <= trial_metric[id(t)] < hi]
            for lbl, (lo, hi) in zip(bin_labels, bin_ranges)
        }
        for pol, trs in trials_by_pol.items()
    }

    print(f"BIN_MODE = {bin_mode!r}  |  Unit: {bin_unit}")
    hdr = f"{'Bin':<30}  " + "  ".join(f"{p:<12}" for p in pol_labels)
    print(hdr)
    print("-" * len(hdr))
    for lbl in bin_labels:
        row = "  ".join(f"{len(binned_trials[p][lbl]):<12}" for p in pol_labels)
        print(f"{lbl:<30}  {row}")

    return binned_trials, bin_labels, bin_colors, trial_bg_dict, bin_unit, pol_labels, trials_by_pol


def plot_bin_overview(
    binned_trials, bin_labels, bin_colors, pol_labels, hrs2_header,
    sample_rate, pre_avg_ms, post_avg_ms,
    m_start_ms, m_end_ms, h_start_ms, h_end_ms,
    bin_ranges=None, bin_unit='',
):
    """Interactive EMG bin overview with Global and per-panel zoom views.

    A View selector switches between the full 5-panel layout and enlarged
    single-panel views for Waveforms, M-wave, H-wave, H:M Ratio, and Stats.
    Legend component checkboxes let you toggle bin range, center, width, and
    n= independently.

    Extra keyword arguments
    -----------------------
    bin_ranges : list of (lo, hi) tuples, optional
        When provided, enables per-component legend control.
    bin_unit : str, optional
        Unit label for the bin metric (e.g. 'µV bg').
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    from ipywidgets import (ToggleButtons, Checkbox, Output, VBox, HBox, HTML)
    from IPython.display import display as _display

    ms_per_sample = 1000.0 / sample_rate
    ov_out        = Output()

    # ── View selector ─────────────────────────────────────────────────
    VIEW_OPTIONS = ['Global', 'Waveforms', 'M-wave', 'H-wave', 'H:M Ratio', 'Stats']
    view_tgl = ToggleButtons(
        options=VIEW_OPTIONS, value='Global',
        button_style='', description='View:',
        style={'button_width': '90px'},
    )

    # ── Legend component checkboxes ───────────────────────────────────
    _cw = {'width': '120px', 'margin': '0 4px 0 0'}
    cb_range  = Checkbox(value=True, description='Bin range',  indent=False, layout=_cw)
    cb_center = Checkbox(value=True, description='Center',     indent=False, layout=_cw)
    cb_width  = Checkbox(value=True, description='Width (Δ)',  indent=False, layout=_cw)
    cb_n      = Checkbox(value=True, description='n=',         indent=False,
                         layout={'width': '70px', 'margin': '0 4px 0 0'})
    cb_means  = Checkbox(value=True, description='Bar means',  indent=False, layout=_cw)
    legend_row = HBox([
        HTML('<b style="line-height:32px">Legend:</b>&nbsp;'),
        cb_range, cb_center, cb_width, cb_n,
        HTML('&nbsp;&nbsp;<b style="line-height:32px">Bars:</b>&nbsp;'),
        cb_means,
    ])

    current_pol  = [pol_labels[0]]
    current_view = ['Global']

    def _legend_label(i, lo, hi, n):
        parts = [f'Bin {i + 1}']
        if cb_range.value:
            hi_s = '∞' if hi == float('inf') else str(hi)
            parts.append(f'{lo}–{hi_s} {bin_unit}')
        extras = []
        if cb_center.value and hi != float('inf'):
            extras.append(f'c={(lo + hi) / 2:.4g}')
        if cb_width.value and hi != float('inf'):
            extras.append(f'Δ{hi - lo:.4g}')
        if extras:
            parts.append(f'[{", ".join(extras)}]')
        if cb_n.value:
            parts.append(f'n={n}')
        return '  '.join(parts)

    def _make_leg(lbl):
        bi = bin_labels.index(lbl)
        if bin_ranges is not None:
            lo, hi = bin_ranges[bi]
            return _legend_label(bi, lo, hi, ns[lbl])
        return f'{lbl}  (n={ns[lbl]})' if cb_n.value else lbl

    # Mutable stats cache so _make_leg can access ns inside _draw
    avgs     = {}
    stacks_d = {}
    m_mra    = {}
    h_mra    = {}
    hm_mra   = {}
    ns       = {}

    def _collect(pol_lbl):
        bins = binned_trials[pol_lbl]
        for lbl in bin_labels:
            trs = bins[lbl]
            ns[lbl] = len(trs)
            if not trs:
                avgs[lbl] = None; stacks_d[lbl] = None
                m_mra[lbl] = h_mra[lbl] = hm_mra[lbl] = []
                continue
            stacks, mv_l, hv_l, hm_l = [], [], [], []
            t_ref = None
            for tr in trs:
                tm, emg, _, _, _ = get_trial_window(
                    tr, pre_avg_ms, post_avg_ms, ms_per_sample=ms_per_sample)
                if t_ref is None:
                    t_ref = tm
                stacks.append(emg[:len(t_ref)])
                mm = (tm >= m_start_ms) & (tm <= m_end_ms)
                hm = (tm >= h_start_ms) & (tm <= h_end_ms)
                if mm.any():
                    mv_l.append(float(np.nanmean(np.abs(emg[mm]))))
                if hm.any():
                    hv_l.append(float(np.nanmean(np.abs(emg[hm]))))
                if mm.any() and hm.any():
                    mv = float(np.nanmean(np.abs(emg[mm])))
                    hv = float(np.nanmean(np.abs(emg[hm])))
                    if mv > 0:
                        hm_l.append(hv / mv)
            if stacks:
                arr = np.vstack(stacks)
                avgs[lbl]     = (t_ref, np.nanmean(arr, axis=0))
                stacks_d[lbl] = arr
            else:
                avgs[lbl] = None; stacks_d[lbl] = None
            m_mra[lbl]  = mv_l
            h_mra[lbl]  = hv_l
            hm_mra[lbl] = hm_l

    def _draw(pol_lbl, view='Global'):
        with ov_out:
            ov_out.clear_output(wait=True)
            _collect(pol_lbl)

            present    = [b for b in bin_labels if ns[b] > 0]
            cols       = [bin_colors[bin_labels.index(b)] for b in present]
            x_pos      = np.arange(len(present))
            short_lbls = [f'Bin {bin_labels.index(b) + 1}' for b in present]

            # ── GLOBAL ─────────────────────────────────────────────────
            if view == 'Global':
                fig, axes = plt.subplots(1, 5, figsize=(22, 5),
                                         gridspec_kw={'width_ratios': [5, 3, 3, 3, 2.5]})
                ax_w, ax_m, ax_h, ax_hm, ax_txt = axes

                for lbl, col in zip(present, cols):
                    if avgs[lbl] is not None:
                        t_r, avg = avgs[lbl]
                        ax_w.plot(t_r, avg, color=col, linewidth=2.2, label=_make_leg(lbl))
                ax_w.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.12, zorder=0)
                ax_w.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.12, zorder=0)
                ax_w.axvline(0, color='red', linestyle='--', linewidth=1.0, label='Stim onset')
                ax_w.set_xlabel('Time re: stim onset (ms)')
                ax_w.set_ylabel('EMG (µV)')
                ax_w.set_title(f'Avg Waveforms — {pol_lbl}')
                ax_w.legend(fontsize=8, loc='upper right')
                ax_w.grid(True, alpha=0.3)

                def _bar(ax, val_dict, ylabel, title, fmt='.1f'):
                    mus, sds = [], []
                    for xi, (b_lbl, col) in enumerate(zip(present, cols)):
                        v  = np.array([x for x in val_dict[b_lbl] if np.isfinite(x)])
                        mu = float(np.mean(v)) if len(v) else 0.0
                        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                        ax.bar(xi, mu, color=col, alpha=0.78, yerr=sd, capsize=6,
                               error_kw={'elinewidth': 2, 'ecolor': col, 'capthick': 2})
                        mus.append(mu); sds.append(sd)
                    top_vals = [m + s for m, s in zip(mus, sds)]
                    ax_top   = max(top_vals) if top_vals else 1.0
                    ax.set_ylim(0, ax_top * 1.22)
                    if cb_means.value:
                        for xi, (mu, sd) in enumerate(zip(mus, sds)):
                            if mu > 0:
                                ax.text(xi, mu + sd, f'{mu:{fmt}}',
                                        ha='center', va='bottom', fontsize=8.5,
                                        fontweight='bold', color='#222222')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(short_lbls, rotation=0, ha='center', fontsize=9)
                    ax.set_ylabel(ylabel, fontsize=9)
                    ax.set_title(title, fontsize=10, fontweight='bold')
                    ax.grid(True, axis='y', alpha=0.3)

                _bar(ax_m,  m_mra,  'MRA (µV)', 'M-wave',    fmt='.1f')
                _bar(ax_h,  h_mra,  'MRA (µV)', 'H-wave',    fmt='.1f')
                _bar(ax_hm, hm_mra, 'H:M',           'H:M Ratio', fmt='.3f')

                ax_txt.axis('off')
                stat_lines = [f'Stats — {pol_lbl}', '']
                for lbl, slbl in zip(present, short_lbls):
                    stat_lines.append(slbl)
                    for key, vd in [('M', m_mra), ('H', h_mra), ('H:M', hm_mra)]:
                        v = np.array([x for x in vd[lbl] if np.isfinite(x)])
                        if len(v):
                            mu = float(np.mean(v))
                            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                            cv = sd / mu if mu != 0 else float('nan')
                            stat_lines.append(f'  {key}: µ={mu:.2f} SD={sd:.2f} CV={cv:.2f}')
                    stat_lines.append('')
                ax_txt.text(0.05, 0.97, '\n'.join(stat_lines), transform=ax_txt.transAxes,
                            va='top', ha='left', fontsize=7.5, family='monospace',
                            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc'))

                plt.suptitle(f'Combined EMG Bin Overview  —  {hrs2_header.subject_id}', fontsize=11)
                plt.tight_layout()
                plt.show()

            # ── WAVEFORMS (zoomed) ─────────────────────────────────────
            elif view == 'Waveforms':
                fig, ax = plt.subplots(figsize=(14, 7))
                for lbl, col in zip(present, cols):
                    if avgs[lbl] is None:
                        continue
                    t_r, avg = avgs[lbl]
                    ax.plot(t_r, avg, color=col, linewidth=2.5, label=_make_leg(lbl))
                    if stacks_d[lbl] is not None and stacks_d[lbl].shape[0] > 1:
                        sem = np.nanstd(stacks_d[lbl], axis=0) / np.sqrt(stacks_d[lbl].shape[0])
                        ax.fill_between(t_r, avg - sem, avg + sem, color=col, alpha=0.18)
                ax.axvspan(m_start_ms, m_end_ms, color='blue',  alpha=0.12, zorder=0, label='M window')
                ax.axvspan(h_start_ms, h_end_ms, color='green', alpha=0.12, zorder=0, label='H window')
                ax.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Stim onset')
                ax.set_xlabel('Time re: stim onset (ms)', fontsize=12)
                ax.set_ylabel('EMG (µV)', fontsize=12)
                ax.set_title(
                    f'Averaged Waveforms per Bin  —  {pol_lbl}  —  {hrs2_header.subject_id}',
                    fontsize=13, fontweight='bold')
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.text(0.01, 0.99, 'Shading = ±SEM', transform=ax.transAxes,
                        va='top', fontsize=8, color='gray')
                plt.tight_layout()
                plt.show()

            # ── BAR CHART ZOOM (M-wave / H-wave / H:M Ratio) ──────────
            elif view in ('M-wave', 'H-wave', 'H:M Ratio'):
                val_map  = {'M-wave': m_mra, 'H-wave': h_mra, 'H:M Ratio': hm_mra}
                ylbl_map = {'M-wave': 'MRA (µV)', 'H-wave': 'MRA (µV)', 'H:M Ratio': 'H:M'}
                fmt_map  = {'M-wave': '.1f', 'H-wave': '.1f', 'H:M Ratio': '.3f'}
                val_dict = val_map[view]
                fmt      = fmt_map[view]

                fig, ax = plt.subplots(figsize=(max(8, len(present) * 2.8), 7))
                mus, sds, all_vals = [], [], []
                for xi, (b_lbl, col) in enumerate(zip(present, cols)):
                    v  = np.array([x for x in val_dict[b_lbl] if np.isfinite(x)])
                    mu = float(np.mean(v)) if len(v) else 0.0
                    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                    mus.append(mu); sds.append(sd); all_vals.append(v)
                    # Mean bar (semi-transparent background)
                    ax.bar(xi, mu, color=col, alpha=0.45, width=0.55, zorder=2)
                    # Box plot overlay
                    if len(v) > 1:
                        ax.boxplot(
                            [v], positions=[xi], widths=0.38, patch_artist=True,
                            boxprops=dict(facecolor=col, alpha=0.35, linewidth=1.5),
                            medianprops=dict(color='black', linewidth=2.5),
                            whiskerprops=dict(color=col, linewidth=1.5),
                            capprops=dict(color=col, linewidth=1.5),
                            flierprops=dict(marker='o', markersize=3, alpha=0.3,
                                            markerfacecolor=col, markeredgecolor=col),
                            showfliers=(len(v) < 200),
                            zorder=3,
                        )
                    # Mean diamond
                    ax.scatter([xi], [mu], marker='D', color='black', s=60, zorder=5)
                    if cb_means.value and mu > 0:
                        ax.text(xi, mu + sd * 0.05, f'{mu:{fmt}}',
                                ha='center', va='bottom', fontsize=11,
                                fontweight='bold', color='#111111')

                finite_all = [x for v in all_vals for x in v if np.isfinite(x)]
                ax_top = max(finite_all + [m + s for m, s in zip(mus, sds)] + [1.0])
                ax.set_ylim(0, ax_top * 1.2)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [_make_leg(b) for b in present],
                    rotation=15, ha='right', fontsize=10)
                ax.set_ylabel(ylbl_map[view], fontsize=12)
                ax.set_title(
                    f'{view}  —  {pol_lbl}  —  {hrs2_header.subject_id}',
                    fontsize=13, fontweight='bold')
                ax.grid(True, axis='y', alpha=0.3)
                leg_handles = [
                    mpatches.Patch(fc='gray', alpha=0.45, label='Mean (bar)'),
                    Line2D([0], [0], marker='D', color='black', markersize=9,
                           linestyle='None', label='Mean (◆)'),
                    mpatches.Patch(fc='gray', alpha=0.35, label='IQR box'),
                    Line2D([0], [0], color='black', linewidth=2.5, label='Median'),
                ]
                ax.legend(handles=leg_handles, fontsize=9, loc='upper right')
                plt.tight_layout()
                plt.show()

            # ── STATS (zoomed text panel) ──────────────────────────────
            elif view == 'Stats':
                fig, ax = plt.subplots(figsize=(11, max(5, len(present) * 2.2 + 2)))
                ax.axis('off')
                stat_lines = [
                    f'Detailed Statistics  —  {pol_lbl}',
                    f'{hrs2_header.subject_id}',
                    '─' * 58, '',
                ]
                for lbl, slbl in zip(present, short_lbls):
                    stat_lines.append(f'{slbl}   (n={ns[lbl]})')
                    for key, vd in [('M-wave MRA (µV)',  m_mra),
                                    ('H-wave MRA (µV)',  h_mra),
                                    ('H:M Ratio',             hm_mra)]:
                        v = np.array([x for x in vd[lbl] if np.isfinite(x)])
                        if len(v):
                            mu  = float(np.mean(v))
                            sd  = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                            med = float(np.median(v))
                            cv  = sd / mu if mu != 0 else float('nan')
                            p25 = float(np.percentile(v, 25))
                            p75 = float(np.percentile(v, 75))
                            stat_lines.append(
                                f'  {key:<24} µ={mu:.3f}  SD={sd:.3f}  CV={cv:.3f}')
                            stat_lines.append(
                                f'  {"":24} Median={med:.3f}  IQR=[{p25:.3f}, {p75:.3f}]')
                        else:
                            stat_lines.append(f'  {key:<24} no data')
                    stat_lines.append('')
                ax.text(0.03, 0.97, '\n'.join(stat_lines), transform=ax.transAxes,
                        va='top', ha='left', fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.6', fc='#f9f9f9', ec='#aaaaaa'))
                plt.tight_layout()
                plt.show()

    # ── Reactivity wiring ─────────────────────────────────────────────
    def _on_cb(change):
        _draw(current_pol[0], current_view[0])
    for cb in [cb_range, cb_center, cb_width, cb_n, cb_means]:
        cb.observe(_on_cb, names='value')

    def _on_view(change):
        if change['name'] == 'value':
            current_view[0] = change['new']
            _draw(current_pol[0], change['new'])
    view_tgl.observe(_on_view, names='value')

    if len(pol_labels) > 1:
        pol_tgl = ToggleButtons(options=pol_labels, button_style='info', description='Polarity:')
        def _on_pol(change):
            if change['name'] == 'value':
                current_pol[0] = change['new']
                _draw(change['new'], current_view[0])
        pol_tgl.observe(_on_pol, names='value')
        _display(VBox([HBox([pol_tgl]), HBox([view_tgl]), legend_row, ov_out]))
    else:
        _display(VBox([HBox([view_tgl]), legend_row, ov_out]))
    _draw(pol_labels[0])


def create_bin_viewer(
    binned_trials, bin_labels, bin_colors, pol_labels,
    hrs2_header, hrs2_emg_blocks, trial_bg_dict,
    sample_rate, pre_plot_ms, post_plot_ms, pre_avg_ms, post_avg_ms, n_per_page,
    m_start_ms, m_end_ms, h_start_ms, h_end_ms,
):
    """Per-bin interactive viewer: background EMG, averaged waveforms, and trial grid."""
    import matplotlib.pyplot as plt
    from ipywidgets import ToggleButtons, Dropdown, Button, Output, VBox, HBox, Label
    from IPython.display import display as _display

    pb_pol  = ToggleButtons(options=pol_labels, button_style='info', description='Polarity:')
    pb_bin  = Dropdown(options=bin_labels, description='Bin:', layout={'width': '280px'})
    pb_btn  = Button(description='Load bin viewer', button_style='primary')
    pb_stat = Label(value='Select polarity + bin, then click Load.')
    pb_out  = Output()

    def _load(b=None):
        pol_lbl = pb_pol.value
        bin_lbl = pb_bin.value
        bin_trs = binned_trials[pol_lbl][bin_lbl]
        pb_stat.value = f"Loading {bin_lbl} | {pol_lbl} ({len(bin_trs)} trials)…"
        with pb_out:
            pb_out.clear_output(wait=True)
            if not bin_trs:
                print(f"No trials in {bin_lbl} for {pol_lbl}.")
                pb_stat.value = f"{bin_lbl} | {pol_lbl}: no trials in this bin."
                return

            print(f"═══ {bin_lbl}  |  {pol_lbl}  |  {len(bin_trs)} trials ═══")
            print("── Background EMG ──")

            bg_sel = np.array([v for t in bin_trs
                               for v in [trial_bg_dict.get(id(t), float('nan'))]
                               if np.isfinite(v)])

            fig_bg, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(13, 4))

            bi_sel = bin_labels.index(bin_lbl)
            for xi, blbl in enumerate(bin_labels):
                btrs  = binned_trials[pol_lbl][blbl]
                bvals = np.array([v for t in btrs
                                  for v in [trial_bg_dict.get(id(t), float('nan'))]
                                  if np.isfinite(v)])
                mu    = float(np.mean(bvals))         if len(bvals) else 0.0
                sd    = float(np.std(bvals, ddof=1))  if len(bvals) > 1 else 0.0
                col_b = '#E53935' if blbl == bin_lbl else bin_colors[xi]
                ax_bar.bar(xi, mu, yerr=sd, color=col_b, alpha=0.82, capsize=6,
                           error_kw={'elinewidth': 2, 'ecolor': col_b, 'capthick': 2})
                ax_bar.text(xi, mu + sd + max(mu * 0.02, 2), f'n={len(bvals)}',
                            ha='center', va='bottom', fontsize=8, color=col_b)
            ax_bar.set_xticks(range(len(bin_labels)))
            ax_bar.set_xticklabels(bin_labels, rotation=15, ha='right', fontsize=8)
            ax_bar.set_ylabel('Pre-stim BG EMG (µV)', fontsize=10)
            ax_bar.set_title('Background EMG per Bin  (red = selected)', fontsize=10, fontweight='bold')
            ax_bar.grid(True, axis='y', alpha=0.3)

            if len(bg_sel):
                q1b   = float(np.percentile(bg_sel, 25))
                q2b   = float(np.percentile(bg_sel, 50))
                q3b   = float(np.percentile(bg_sel, 75))
                thrb  = float(np.percentile(bg_sel, 80))
                nbins = min(30, max(8, len(bg_sel) // 8))
                ax_hist.hist(bg_sel, bins=nbins, color=bin_colors[bi_sel],
                             alpha=0.65, edgecolor='white', linewidth=0.4)
                for val, name, ls, lw, qc, yp in [
                    (q1b,  'Q1',        ':',  1.6, '#555555', 0.90),
                    (q2b,  'Q2',        '--', 2.0, '#222222', 0.97),
                    (q3b,  'Q3',        ':',  1.6, '#555555', 0.83),
                    (thrb, '80th\npct', '--', 2.0, 'crimson', 0.72),
                ]:
                    ax_hist.axvline(val, color=qc, linewidth=lw, linestyle=ls, zorder=3)
                    ax_hist.text(val, yp, f'{name}\n{val:.1f}',
                                 transform=ax_hist.get_xaxis_transform(),
                                 ha='center', va='top', fontsize=7.5, color=qc,
                                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.5))
                n_top = int(np.sum(bg_sel >= thrb))
                ax_hist.text(0.98, 0.60,
                             f'n={len(bg_sel)}\nQ1  = {q1b:.1f} µV\nQ2  = {q2b:.1f} µV\n'
                             f'Q3  = {q3b:.1f} µV\n80th= {thrb:.1f} µV\n'
                             f'≥80th: {n_top} ({100*n_top/len(bg_sel):.1f}%)',
                             transform=ax_hist.transAxes, va='top', ha='right',
                             fontsize=8, family='monospace',
                             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc', alpha=0.95))
            else:
                ax_hist.text(0.5, 0.5, 'No BG data', ha='center', va='center',
                             transform=ax_hist.transAxes, fontsize=12, color='gray')
            ax_hist.set_xlabel('Pre-stim BG EMG (µV)', fontsize=10)
            ax_hist.set_ylabel('Trial count', fontsize=10)
            ax_hist.set_title(f'BG EMG Distribution — {bin_lbl}', fontsize=10, fontweight='bold')
            ax_hist.grid(True, axis='y', alpha=0.3)

            fig_bg.suptitle(f'Background EMG — {pol_lbl}  |  {bin_lbl}', fontsize=11)
            plt.tight_layout()
            plt.show()

            print("── Averaged waveforms + recruitment curve ──")
            plot_hrs2_analysis(
                bin_trs, hrs2_header,
                pre_avg_ms=pre_avg_ms, post_avg_ms=post_avg_ms,
                n_per_page=n_per_page,
                m_start_ms=m_start_ms, m_end_ms=m_end_ms,
                h_start_ms=h_start_ms, h_end_ms=h_end_ms,
                sample_rate=sample_rate, emg_blocks=hrs2_emg_blocks,
            )

            print("── Individual trial grid ──")
            plot_hrs2_trials(
                bin_trs, hrs2_header,
                pre_plot_ms=pre_plot_ms, post_plot_ms=post_plot_ms,
                n_per_page=n_per_page,
                m_start_ms=m_start_ms, m_end_ms=m_end_ms,
                h_start_ms=h_start_ms, h_end_ms=h_end_ms,
                sample_rate=sample_rate, emg_blocks=hrs2_emg_blocks,
            )
        pb_stat.value = f"Showing: {bin_lbl} | {pol_lbl} ({len(bin_trs)} trials)"

    pb_btn.on_click(_load)
    _display(VBox([HBox([pb_pol, pb_bin, pb_btn]), pb_stat, pb_out]))
    _load()


def print_bin_statistics(
    binned_trials, bin_labels, pol_labels,
    sample_rate, pre_avg_ms, post_avg_ms,
    m_start_ms, m_end_ms, h_start_ms, h_end_ms,
):
    """Print per-bin x per-polarity numeric statistics (trial count, M-MRA, H-MRA, H:M)."""
    ms_per_sample = 1000.0 / sample_rate
    hdr = (f"{'Polarity':<20}  {'Bin':<38}  {'n':>5}  "
           f"{'M-MRA µV':>10}  {'H-MRA µV':>10}  {'H:M':>8}")
    print(hdr)
    print("-" * len(hdr))

    for pol_lbl in pol_labels:
        for bin_lbl in bin_labels:
            bin_trs = binned_trials[pol_lbl][bin_lbl]
            if not bin_trs:
                print(f"{pol_lbl:<20}  {bin_lbl:<38}  {'0':>5}  "
                      f"{'—':>10}  {'—':>10}  {'—':>8}")
                continue
            m_mra, h_mra, hm = [], [], []
            for tr in bin_trs:
                tm, emg, _, _, _ = get_trial_window(tr, pre_avg_ms, post_avg_ms, ms_per_sample=ms_per_sample)
                mm      = (tm >= m_start_ms) & (tm <= m_end_ms)
                hm_mask = (tm >= h_start_ms) & (tm <= h_end_ms)
                if mm.any():
                    m_mra.append(float(np.nanmean(np.abs(emg[mm]))))
                if hm_mask.any():
                    h_mra.append(float(np.nanmean(np.abs(emg[hm_mask]))))
                if mm.any() and hm_mask.any():
                    mv = float(np.nanmean(np.abs(emg[mm])))
                    hv = float(np.nanmean(np.abs(emg[hm_mask])))
                    if mv > 0:
                        hm.append(hv / mv)
            m_s  = f"{np.nanmean(m_mra):.1f}" if m_mra else "—"
            h_s  = f"{np.nanmean(h_mra):.1f}" if h_mra else "—"
            hm_s = f"{np.nanmean(hm):.3f}"    if hm    else "—"
            print(f"{pol_lbl:<20}  {bin_lbl:<38}  {len(bin_trs):>5}  "
                  f"{m_s:>10}  {h_s:>10}  {hm_s:>8}")
        print()


def filter_failed_trials(hrs2_trials, hrs2_header, hrs2_emg_blocks,
                          pre_ms, post_ms, m_start_ms, m_end_ms,
                          h_start_ms, h_end_ms, sample_rate):
    """Silently detect and remove failed trials; return the cleaned trial list.

    Wraps detect_and_correct_failed_trials with silent=True and filters the
    trial list to only the passed trials.
    """
    _, failed, passed, _ = detect_and_correct_failed_trials(
        hrs2_trials, hrs2_header, hrs2_emg_blocks,
        pre_ms=pre_ms, post_ms=post_ms,
        m_start_ms=m_start_ms, m_end_ms=m_end_ms,
        h_start_ms=h_start_ms, h_end_ms=h_end_ms,
        sample_rate=sample_rate,
        silent=True,
    )
    passed_idx = {r['idx'] for r in passed}
    clean = [t for i, t in enumerate(hrs2_trials) if i in passed_idx]
    print(f"Failed trials removed : {len(failed)}")
    print(f"Trials remaining      : {len(clean)}")
    return clean


def plot_background_grand_means(hrs2_trials, emg_blocks, header, sample_rate):
    """Extract per-trial pre-stim background EMG grand means and plot their distribution.

    Uses background_emg_mean or background_bins stored in each trial (file_version >= 5);
    falls back to reconstructing from emg_blocks for older files.

    Returns a state dict with key 'trial_bg_gm' (one float per trial) compatible
    with compute_trial_bins.
    """
    import matplotlib.pyplot as plt

    bg_gm = []
    n_stored, n_recon = 0, 0

    for tr in hrs2_trials:
        stored_gm   = getattr(tr, 'background_emg_mean', None)
        stored_bins = getattr(tr, 'background_bins', None)
        if stored_gm is not None and float(stored_gm) > 0:
            bg_gm.append(float(stored_gm))
            n_stored += 1
        elif stored_bins is not None and len(stored_bins) > 0:
            bg_gm.append(float(np.mean(stored_bins)))
            n_stored += 1
        else:
            _, gm = compute_background_bins(tr, emg_blocks, sample_rate=sample_rate)
            bg_gm.append(float(gm))
            n_recon += 1

    trial_bg_gm = np.array(bg_gm, dtype=np.float64)
    valid_bg    = trial_bg_gm[~np.isnan(trial_bg_gm)]
    gm_q1, gm_med, gm_q3 = (float(x) for x in np.percentile(valid_bg, [25, 50, 75]))

    print(f'Trials: {len(trial_bg_gm)}  (stored: {n_stored}, reconstructed: {n_recon})')
    print(f'Background  Min={np.nanmin(trial_bg_gm):.2f}  Q1={gm_q1:.2f}  '
          f'Median={gm_med:.2f}  Q3={gm_q3:.2f}  Max={np.nanmax(trial_bg_gm):.2f} µV')

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.hist(valid_bg, bins=80, color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.9,
            label='Per-trial pre-stim BG grand mean')
    ax.axvline(gm_q1,  color='darkorange', linestyle=':', linewidth=1.5, label=f'Q1 = {gm_q1:.2f} µV')
    ax.axvline(gm_med, color='purple',     linestyle=':', linewidth=1.5, label=f'Median = {gm_med:.2f} µV')
    ax.axvline(gm_q3,  color='steelblue',  linestyle=':', linewidth=1.5, label=f'Q3 = {gm_q3:.2f} µV')
    ax.set_xlabel('Per-trial pre-stim grand mean (µV)')
    ax.set_ylabel('Trial count')
    ax.set_title(f'Per-Trial Pre-Stim Background Grand Means — {header.subject_id}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {'trial_bg_gm': trial_bg_gm, 'q1': gm_q1, 'median': gm_med, 'q3': gm_q3}


# ── Notebook viewer utilities ──────────────────────────────────────────────────

def make_viewer(all_recordings, active_rec_label, render_fn, stage_filter=None):
    """
    Creates a self-contained interactive viewer widget with per-viewer Recording + Stage dropdowns.

    Parameters
    ----------
    all_recordings : dict
        {label: {stage_map, sample_rate, hrs1_header, ...}} as built by the load cell.
    active_rec_label : str
        Initial recording label to display.
    render_fn : callable
        render_fn(trials, header, emg_blocks, stage_label, rec_label, sample_rate, hrs1_header)
        Called inside an Output context.
    stage_filter : callable, optional
        stage_filter(key, trials, header, emg_blocks, label) -> bool
        Only stages returning True appear in the Stage dropdown.

    Returns
    -------
    (widget, render_fn)
        widget: VBox with dropdowns + output area; call display(widget).
        render_fn: call once after display() to populate the initial view.
    """
    from ipywidgets import Dropdown, VBox, Output

    _out = Output()

    def _stage_opts_for(rec_label):
        sm = all_recordings[rec_label]['stage_map']
        result = []
        for sk, (_t, _h, _e, lbl) in sm.items():
            if not _t:
                continue
            if stage_filter is not None and not stage_filter(sk, _t, _h, _e, lbl):
                continue
            result.append((lbl, sk))
        return result

    _init_opts = _stage_opts_for(active_rec_label)
    _rec_d = Dropdown(
        options=list(all_recordings.keys()), value=active_rec_label,
        description='Recording:', layout={'width': '600px'}
    )
    _stage_d = Dropdown(
        options=_init_opts, description='Stage:', layout={'width': '480px'}
    )
    if _init_opts:
        _stage_d.value = _init_opts[0][1]

    def _render():
        rec_label = _rec_d.value
        stage_key = _stage_d.value
        rec = all_recordings[rec_label]
        sm  = rec['stage_map']
        with _out:
            _out.clear_output(wait=True)
            if not stage_key or stage_key not in sm:
                print(f'No compatible stages available for {rec_label!r}.')
                return
            _t, _h, _e, _lbl = sm[stage_key]
            render_fn(_t, _h, _e, _lbl, rec_label, rec['sample_rate'], rec['hrs1_header'])

    def _on_stage_change(change):
        _render()

    def _on_rec_change(change):
        new_opts = _stage_opts_for(_rec_d.value)
        _stage_d.unobserve(_on_stage_change, names='value')
        _stage_d.options = new_opts
        _stage_d.value   = new_opts[0][1] if new_opts else None
        _stage_d.observe(_on_stage_change, names='value')
        _render()

    _rec_d.observe(_on_rec_change, names='value')
    _stage_d.observe(_on_stage_change, names='value')

    controls = ([_rec_d] if len(all_recordings) > 1 else []) + [_stage_d]
    return VBox(controls + [_out]), _render


def compute_h_comparison_data(all_recordings, pre_ms, post_ms, h_start_ms, h_end_ms,
                               m_start_ms=2.0, m_end_ms=4.0):
    """
    Pre-compute per-trial H-reflex size, M-wave size, and background MRA for the
    cross-recording comparison plot.

    Per trial:
      bg_mra  = mean|emg(t < 0)|
      h_size  = mean|emg(t in [h_start_ms, h_end_ms])| - bg_mra
      m_size  = mean|emg(t in [m_start_ms, m_end_ms])| - bg_mra
      bg_size = bg_mra  (raw pre-stim background level)

    Call once after loading recordings and setting analysis parameters. Returns a
    cache dict that plot_h_reflex_comparison() consumes directly.

    Parameters
    ----------
    all_recordings : dict  — as built by the notebook load cell
    pre_ms, post_ms : float — trial window bounds (ms before/after stim onset)
    h_start_ms, h_end_ms : float — H-wave window bounds
    m_start_ms, m_end_ms : float — M-wave window bounds

    Returns
    -------
    dict : {rec_label: {stage_key: {'h_sizes', 'm_sizes', 'bg_sizes': np.ndarray, 'mean_amp': float}}}
        Each array has one value per trial with a valid pre-stim window.
    """
    result = {}
    for rec_label, rec in all_recordings.items():
        result[rec_label] = {}
        sr = rec['sample_rate'] or getattr(rec['hrs1_header'], 'sample_rate', SAMPLE_RATE)
        ms_per_sample = 1000.0 / sr

        for stage_key, (trials, _, _, _) in rec['stage_map'].items():
            if not trials:
                continue

            h_sizes, m_sizes, bg_sizes = [], [], []
            for t in trials:
                try:
                    t_ms, emg, *_ = get_trial_window(
                        t, pre_ms, post_ms, ms_per_sample=ms_per_sample)
                    bg_mask = t_ms < 0
                    if not bg_mask.any():
                        continue
                    bg_mra = float(np.mean(np.abs(emg[bg_mask])))
                    bg_sizes.append(bg_mra)
                    h_mask = (t_ms >= h_start_ms) & (t_ms <= h_end_ms)
                    if h_mask.any():
                        h_sizes.append(float(np.mean(np.abs(emg[h_mask]))) - bg_mra)
                    m_mask = (t_ms >= m_start_ms) & (t_ms <= m_end_ms)
                    if m_mask.any():
                        m_sizes.append(float(np.mean(np.abs(emg[m_mask]))) - bg_mra)
                except Exception:
                    continue

            if not bg_sizes:
                continue

            amps     = [t.stimulation_amplitude_ma for t in trials]
            mean_amp = float(np.mean(amps))
            std_amp  = float(np.std(amps, ddof=min(1, len(amps) - 1)))
            result[rec_label][stage_key] = {
                'h_sizes':  np.array(h_sizes,  dtype=float),
                'm_sizes':  np.array(m_sizes,  dtype=float),
                'bg_sizes': np.array(bg_sizes, dtype=float),
                'mean_amp': mean_amp,
                'std_amp':  std_amp,
            }

    return result


def plot_h_reflex_comparison(h_cache, recording_dirs, stage_key, stage_labels,
                             metric='h_reflex'):
    """
    Render a cross-recording comparison box-and-whisker plot.

    Each box shows the per-trial distribution across ALL trials in that recording.

    Parameters
    ----------
    h_cache : dict — return value of compute_h_comparison_data()
    recording_dirs : list — RECORDING_DIRS list of (label, path, sr) tuples; sets x-axis order
    stage_key : str — which stage to display (e.g. 'control_mode')
    stage_labels : dict — {stage_key: display_label}
    metric : str — 'h_reflex', 'm_wave', or 'background'
    """
    import matplotlib.pyplot as plt

    stage_lbl = stage_labels.get(stage_key, stage_key)

    _metric_cfg = {
        'h_reflex':   ('h_sizes',  'H-Reflex Size  (µV)  [H-MRA − BG-MRA]',  'H-Reflex Size Across Recordings'),
        'm_wave':     ('m_sizes',  'M-Wave Size  (µV)  [M-MRA − BG-MRA]',     'M-Wave Size Across Recordings'),
        'background': ('bg_sizes', 'Background MRA  (µV)',                      'EMG Background Level Across Recordings'),
    }
    sizes_key, ylabel, title_base = _metric_cfg.get(metric, _metric_cfg['h_reflex'])

    # Collect data in RECORDING_DIRS order; skip recordings missing this stage/data
    ordered = [lbl for lbl, _, _ in recording_dirs if lbl in h_cache]
    valid = []
    for lbl in ordered:
        entry = h_cache[lbl].get(stage_key)
        if entry is not None:
            sizes = entry.get(sizes_key, np.array([]))
            if len(sizes) >= 1:
                valid.append((lbl, sizes, entry['mean_amp'], entry.get('std_amp', 0.0)))

    if not valid:
        print(f'No data for stage {stage_lbl!r} / metric {metric!r} in any recording.')
        return

    n  = len(valid)
    bw = 0.46
    fig, ax = plt.subplots(figsize=(max(6, n * 2.0), 5))

    all_flat = np.concatenate([d for _, d, _, _ in valid])
    y_span   = float(all_flat.max() - all_flat.min()) or 1.0

    means, xlbls = [], []
    for i, (rl, sizes, mean_amp, std_amp) in enumerate(valid):
        n_trials = len(sizes)
        mn = float(np.mean(sizes))
        sd = float(np.std(sizes, ddof=min(1, n_trials - 1)))
        means.append(mn)
        xlbls.append(f'{rl}\nσ={std_amp:.3f} mA')

        if n_trials >= 2:
            q1, med, q3 = (float(v) for v in np.percentile(sizes, [25, 50, 75]))
            ax.bar(i, q3 - q1, bottom=q1, width=bw,
                   color='lightsteelblue', edgecolor='steelblue', lw=1.5,
                   alpha=0.75, zorder=3)
            ax.hlines(med, i - bw / 2, i + bw / 2,
                      colors='steelblue', lw=2.5, zorder=4)
            ax.vlines(i, mn - sd, mn + sd, colors='#1e3a5f', lw=1.5, zorder=4)
            ax.hlines([mn - sd, mn + sd], i - 0.13, i + 0.13,
                      colors='#1e3a5f', lw=1.5, zorder=4)

        ax.plot(i, mn, 'D', color='#1e3a5f', ms=8, zorder=5)

        top = (mn + sd) if n_trials >= 2 else mn
        ax.text(i, top + y_span * 0.04,
                f'n={n_trials}\namp = {mean_amp:.3f} ± {std_amp:.3f} mA\nμ = {mn:.3f} ± {sd:.3f} µV',
                ha='center', va='bottom', fontsize=8, color='#444', zorder=6)

    if len(means) > 1:
        ax.plot(range(n), means, '--', color='#1e3a5f', lw=1.5, alpha=0.65, zorder=2)

    ax.set_xticks(range(n))
    ax.set_xticklabels(xlbls, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(
        f'{title_base}  ·  {stage_lbl}\n'
        f'box = IQR  ·  — = median  ·  ◆ = mean  ·  whiskers = mean ± 1σ  ·  n = trials',
        fontsize=10)
    ax.set_xlim(-0.65, n - 0.35)
    ax.margins(y=0.22)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
