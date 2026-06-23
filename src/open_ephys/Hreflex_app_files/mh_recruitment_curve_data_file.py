from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import BinaryIO
from platformdirs import user_data_dir
import struct
import os
import numpy as np

from .h_reflex_data_file_shared import HReflexDataFileBlockIds, HReflexDataFileEmgData
from ..fileio_helpers import FileIO_Helpers
from ..application_configuration import ApplicationConfiguration

@dataclass
class MhRecruitmentCurveHeader:

    #region Data members

    file_version: int = 0
    subject_id: str = ""
    session_start_time: datetime = datetime.min
    stage_name: str = ""
    stage_description: str = ""
    stage_type: int = 0

    #endregion

    #region File IO

    def read_from_file (self, fid: BinaryIO) -> None:
        #Read the file version
        self.file_version = FileIO_Helpers.read(fid, "int32")

        #Read the subjecft id
        self.subject_id = FileIO_Helpers.read_string(fid)

        #Read the session start time
        self.session_start_time = FileIO_Helpers.read_datetime(fid)

        #Read the stage name
        self.stage_name = FileIO_Helpers.read_string(fid)

        #Read the stage description
        self.stage_description = FileIO_Helpers.read_string(fid)

        #Read the stage type
        self.stage_type = FileIO_Helpers.read(fid, "int32")

        pass

    def save_to_file (self, fid: BinaryIO) -> None:
        #Save the file version
        FileIO_Helpers.write(fid, "int32", self.file_version)

        #Save the subject id
        FileIO_Helpers.write_string(fid, self.subject_id)

        #Save the session date/time
        FileIO_Helpers.write_datetime(fid, self.session_start_time)

        #Save the stage name
        FileIO_Helpers.write_string(fid, self.stage_name)

        #Save the stage description
        FileIO_Helpers.write_string(fid, self.stage_description)

        #Save the stage type
        FileIO_Helpers.write(fid, "int32", self.stage_type)
    
    #endregion

    pass

class MhRecruitmentCurveTrial:

    #region Constructor

    def __init__(self):

        #Create a timestamp
        self.start_time: datetime = datetime.min

        #Define a numpy array to hold trial data (bipolar/differential filtered)
        self.trial_data: np.ndarray = np.zeros(1, dtype=np.float32)

        #Define a numpy array to hold unipolar (non-differential) filtered trial data.
        #Populated at runtime for display only — not persisted to the data file.
        self.unipolar_trial_data: np.ndarray = np.zeros(1, dtype=np.float32)

        #Define a numpy array to hold the raw stimulation ADC waveform for this trial.
        #Signed (not abs'd) so the full biphasic pulse shape is preserved.
        #Populated in file format version 4+.
        self.stim_adc_data: np.ndarray = np.zeros(1, dtype=np.float32)

        #Define a numpy array to hold the ADC sync line data for this trial
        #(used for precise stim-onset alignment; populated when file_version >= 1)
        self.sync_data: np.ndarray = np.zeros(1, dtype=np.float32)

        #Create variables to hold trial parameters
        self.min_initiation_threshold: float = 0.0
        self.max_initiation_threshold: float = 0.0
        self.stimulation_amplitude_ma: float = 0.0

        # --- Timing / sync debug fields (file format version 2) ---

        #Unix epoch milliseconds recorded at the instant trigger_single() was called.
        #Lets you correlate the trigger moment against Open Ephys sample timestamps.
        self.trigger_wall_time_ms: int = 0

        #Index into sync_data of the first onset sample that crossed STIM_ONSET_THRESHOLD.
        #-1 means no threshold crossing was found and the fallback position was used.
        self.onset_sample_index: int = -1

        #1 if onset was positively identified via a threshold crossing, 0 if the fallback
        #position (bin_sample_count) was used because no crossing was found.
        self.onset_detected: int = 0

        #Index into sync_data of the first sample that dropped below STIM_END_THRESHOLD
        #after the onset.  -1 means the end was not detected within the recorded window.
        self.stim_end_sample_index: int = -1

        #Stim duration expressed in samples (stim_end_sample_index − onset_sample_index).
        #0 when end was not detected.
        self.stim_duration_samples: int = 0

        #Stim duration expressed in milliseconds.
        self.stim_duration_ms: float = 0.0

        #Maximum sync voltage observed inside the onset search window.
        #Use this to tune STIM_ONSET_THRESHOLD: if this value is consistently below
        #the threshold the pulse is not being captured.
        self.sync_peak_voltage: float = 0.0

        #Background EMG grand mean (mean rectified abs-filtered EMG) from the monitoring
        #window captured at the moment this trial's initiation criteria were met.
        #Used to compute normalised H-reflex / M-wave size: (window_mra - bg) / bg.
        self.background_emg_mean: float = 0.0

        #Per-bin means from the background monitoring window (one value per 50 ms bin).
        #Preserves the full bin-by-bin background profile for each trial.
        self.background_bins: np.ndarray = np.zeros(0, dtype=np.float32)

        #Software polarity toggle state at the moment the trigger was sent.
        #0 = normal (toggle off, default state).
        #1 = reversed (toggle on, +/- button was active).
        #Physical cathodal/anodal direction depends on cable orientation and is not
        #encoded here — document your electrode wiring separately.
        #Added in file format version 6.
        self.stim_polarity_reversed: int = 0

        #Absolute Open Ephys sample counter (sample_num) from the DIGITAL IN rising-edge
        #event that was used as the primary stim-onset reference for this trial.
        #-1 means no digital event was captured and onset was determined by ADC fallback.
        #Cross-reference with first_post_trigger_frame_sample_id to recompute onset_sample_index
        #offline: onset_sample_index = digital_onset_sample_num - first_post_trigger_frame_sample_id + pre_stim_samples.
        #Added in file format version 7.
        self.digital_onset_sample_num: int = -1

        #Digital input channel number (0-indexed, matching Open Ephys event_channel) from
        #which the digital_onset_sample_num event was received.
        #-1 when no digital event was used.
        #Added in file format version 7.
        self.digital_onset_channel: int = -1

        #Number of data frames that were received by the RECORD state but discarded
        #because their timestamp_emitted pre-dated the trigger call.  A non-zero value
        #here confirms the queued-frame race condition was active on this trial.
        self.n_pre_trigger_frames_discarded: int = 0

        #timestamp_emitted (Unix ms, set by the background thread) of every post-trigger
        #frame that was accumulated during the RECORD state.  Compare against
        #trigger_wall_time_ms to verify the timing of incoming data relative to the trigger.
        self.frame_received_timestamps_ms: np.ndarray = np.zeros(0, dtype=np.uint64)

        #Open Ephys absolute sample counter (sample_id) of the first post-trigger frame.
        #Cross-reference with the Open Ephys recording to find the exact sample position
        #in the continuous data stream where post-trigger accumulation began.
        self.first_post_trigger_frame_sample_id: int = 0

        pass

    #endregion

    #region Methods

    def initialize (self, min_init_threshold: float, max_init_threshold: float, stim_amp: float) -> None:
        self.start_time = datetime.now()
        self.min_initiation_threshold = min_init_threshold
        self.max_initiation_threshold = max_init_threshold
        self.stimulation_amplitude_ma = stim_amp

    def read_from_file (self, fid: BinaryIO, file_version: int = 0) -> None:
        #Read the timestamp for this trial
        self.start_time = FileIO_Helpers.read_datetime(fid)

        #Read the initiation thresholds
        self.min_initiation_threshold = FileIO_Helpers.read(fid, "float32")
        self.max_initiation_threshold = FileIO_Helpers.read(fid, "float32")

        #Read the stimulation amplitude for this trial
        self.stimulation_amplitude_ma = FileIO_Helpers.read(fid, "float32")

        #Read the trial data
        self.trial_data = FileIO_Helpers.read_array(fid, "float32")

        #Read the ADC sync line data (added in file format version 1)
        if (file_version >= 1):
            self.sync_data = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        #Read timing / sync debug fields (added in file format version 2)
        if (file_version >= 2):
            self.trigger_wall_time_ms = FileIO_Helpers.read(fid, "uint64")
            self.onset_sample_index = FileIO_Helpers.read(fid, "int32")
            self.onset_detected = FileIO_Helpers.read(fid, "int8")
            self.stim_end_sample_index = FileIO_Helpers.read(fid, "int32")
            self.stim_duration_samples = FileIO_Helpers.read(fid, "int32")
            self.stim_duration_ms = FileIO_Helpers.read(fid, "float32")
            self.sync_peak_voltage = FileIO_Helpers.read(fid, "float32")
            self.n_pre_trigger_frames_discarded = FileIO_Helpers.read(fid, "int32")
            self.frame_received_timestamps_ms = np.array(
                FileIO_Helpers.read_array(fid, "uint64"), dtype=np.uint64)
            self.first_post_trigger_frame_sample_id = FileIO_Helpers.read(fid, "uint64")

        #Read the unipolar filtered trial data (added in file format version 3)
        if (file_version >= 3):
            self.unipolar_trial_data = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        #Read the stimulation ADC waveform (added in file format version 4)
        if (file_version >= 4):
            self.stim_adc_data = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        #Read per-trial background EMG mean and bins (added in file format version 5)
        if (file_version >= 5):
            self.background_emg_mean = FileIO_Helpers.read(fid, "float32")
            self.background_bins = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        #Read stim polarity (added in file format version 6)
        if (file_version >= 6):
            self.stim_polarity_reversed = FileIO_Helpers.read(fid, "int8")

        #Read digital onset fields (added in file format version 7)
        if (file_version >= 7):
            self.digital_onset_sample_num = FileIO_Helpers.read(fid, "int64")
            self.digital_onset_channel = FileIO_Helpers.read(fid, "int32")

    def save_to_file (self, fid: BinaryIO) -> None:
        #Save a trial block indicator
        FileIO_Helpers.write(fid, "int32", int(1))

        #Save the timestamp for this trial
        FileIO_Helpers.write_datetime(fid, self.start_time)

        #Save the initiation thresholds
        FileIO_Helpers.write(fid, "float32", self.min_initiation_threshold)
        FileIO_Helpers.write(fid, "float32", self.max_initiation_threshold)

        #Save the stimulation amplitude for this trial
        FileIO_Helpers.write(fid, "float32", self.stimulation_amplitude_ma)

        #Save the trial data
        FileIO_Helpers.write_array(fid, "float32", self.trial_data)

        #Save the ADC sync line data (file format version 1+)
        FileIO_Helpers.write_array(fid, "float32", self.sync_data)

        #Save timing / sync debug fields (file format version 2+)
        FileIO_Helpers.write(fid, "uint64", self.trigger_wall_time_ms)
        FileIO_Helpers.write(fid, "int32", self.onset_sample_index)
        FileIO_Helpers.write(fid, "int8", self.onset_detected)
        FileIO_Helpers.write(fid, "int32", self.stim_end_sample_index)
        FileIO_Helpers.write(fid, "int32", self.stim_duration_samples)
        FileIO_Helpers.write(fid, "float32", self.stim_duration_ms)
        FileIO_Helpers.write(fid, "float32", self.sync_peak_voltage)
        FileIO_Helpers.write(fid, "int32", self.n_pre_trigger_frames_discarded)
        FileIO_Helpers.write_array(fid, "uint64", self.frame_received_timestamps_ms)
        FileIO_Helpers.write(fid, "uint64", self.first_post_trigger_frame_sample_id)

        #Save the unipolar (non-differential) filtered trial data (file format version 3+)
        FileIO_Helpers.write_array(fid, "float32", self.unipolar_trial_data)

        #Save the stimulation ADC waveform (file format version 4+)
        FileIO_Helpers.write_array(fid, "float32", self.stim_adc_data)

        #Save per-trial background EMG mean and bins (file format version 5+)
        FileIO_Helpers.write(fid, "float32", self.background_emg_mean)
        FileIO_Helpers.write_array(fid, "float32", self.background_bins)

        #Save stim polarity (file format version 6+)
        FileIO_Helpers.write(fid, "int8", self.stim_polarity_reversed)

        #Save digital onset fields (file format version 7+)
        FileIO_Helpers.write(fid, "int64", self.digital_onset_sample_num)
        FileIO_Helpers.write(fid, "int32", self.digital_onset_channel)

    #endregion

class MhRecruitmentCurveDataFile:

    #region Constructor

    def __init__(self):

        self.header: MhRecruitmentCurveHeader = None
        self.emg_data_blocks: list[HReflexDataFileEmgData] = []
        self.trials: list[MhRecruitmentCurveTrial] = []

        pass

    #endregion

    #region Methods

    def read (self, fid: BinaryIO) -> None:
        '''
        Reads the MH recruitment curve data file from disk.
        '''

        self.header: MhRecruitmentCurveHeader = MhRecruitmentCurveHeader()
        self.header.read_from_file(fid)

        while (True):
            chunk: bytes = fid.read(4)
            if (not chunk):
                break

            block_id: int = struct.unpack(FileIO_Helpers.type_dictionary["int32"], chunk)[0]
            if (block_id == HReflexDataFileBlockIds.MH_RECRUITMENT_CURVE_TRIAL_BLOCK):
                trial: MhRecruitmentCurveTrial = MhRecruitmentCurveTrial()
                trial.read_from_file(fid, self.header.file_version)

                self.trials.append(trial)
            elif (block_id == HReflexDataFileBlockIds.EMG_DATA_BLOCK):
                emg_data_block: HReflexDataFileEmgData = HReflexDataFileEmgData()
                emg_data_block.read_from_file(fid)
                
                self.emg_data_blocks.append(emg_data_block)
        
        pass

    #endregion

    pass