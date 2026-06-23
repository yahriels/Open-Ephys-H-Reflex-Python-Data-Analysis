import numpy as np
from random import Random
from datetime import datetime
import pyqtgraph as pg
from typing import BinaryIO
from platformdirs import user_data_dir
import os

from PySide6 import QtCore

from .stage import Stage
from ..session_message import SessionMessage
from ..application_configuration import ApplicationConfiguration
from ..fileio_helpers import FileIO_Helpers
from ..datafiles.emg_characterization_data_file import EmgCharacterizationDataFile, EmgCharacterizationHeader, EmgCharacterizationTrial, EmgHistogramData
from ..datafiles.mh_recruitment_curve_data_file import MhRecruitmentCurveDataFile, MhRecruitmentCurveHeader, MhRecruitmentCurveTrial
from ..datafiles.h_reflex_data_file_shared import HReflexDataFileEmgData

from ..open_ephys_streamer import OpenEphysDataFrame

class MhRecruitmentCurveStage_TrialInitiationData:

    #region Constructor

    def __init__(self):
        #Declare a variable to hold the monitored signal
        self.monitored_signal: np.ndarray = np.zeros(1)

        #Declare a variable to hold the absolute value monitored signal
        self.monitored_signal_abs: np.ndarray = np.zeros(1)

        #Declare a variable to hold the unipolar (non-differential) filtered rolling buffer
        self.monitored_signal_unipolar: np.ndarray = np.zeros(1)

        #Declare a variable to hold the ADC sync line rolling buffer
        self.sync_signal: np.ndarray = np.zeros(1)

        #Declare a variable to hold the stim ADC waveform rolling buffer (ch3)
        self.stim_adc_signal: np.ndarray = np.zeros(1, dtype=np.float32)

        #Declare a variable to hold the bins
        self.bins: np.ndarray = np.zeros(1)

        #Declare a variable to hold the current monitored signal duration
        self.monitored_signal_duration_seconds: float = 0.0

        #Declare a variable to hold the number of samples that we will
        #store in the monitored signal
        self.monitored_signal_sample_count: int = 0

        #Declare a variable that helps us determine whether we have streamed enough samples
        self.current_monitored_signal_sample_count: int = 0

        pass

    #endregion

    #region Methods

    def initialize (self, dur_milliseconds: int) -> None:
        #Convert to seconds
        self.monitored_signal_duration_seconds = float(dur_milliseconds) / 1000.0

        #Set the number of samples that we care about
        self.monitored_signal_sample_count = int(self.monitored_signal_duration_seconds * ApplicationConfiguration.sample_rate)

        #Get the number of bins we will be collecting
        bin_count: int = int(dur_milliseconds / MhRecruitmentCurveStage.BIN_DURATION_MILLISECONDS)

        #Re-size the appropriate arrays to hold the data we care about
        self.monitored_signal = np.zeros(self.monitored_signal_sample_count)
        self.monitored_signal_abs = np.zeros(self.monitored_signal_sample_count)
        self.monitored_signal_unipolar = np.zeros(self.monitored_signal_sample_count)
        self.sync_signal = np.zeros(self.monitored_signal_sample_count, dtype=np.float32)
        self.stim_adc_signal = np.zeros(self.monitored_signal_sample_count, dtype=np.float32)
        self.bins = np.zeros(bin_count)

        #We are done. return from this function.
        return

    def process (self, data_frame: OpenEphysDataFrame, current_initiation_min: float, current_initiation_max: float) -> bool:
        n: int = self.monitored_signal_sample_count
        n_new: int = len(data_frame.filtered_data_block)

        self.current_monitored_signal_sample_count += n_new

        if n_new <= 0:
            return False

        if n_new >= n:
            # Edge case: block larger than entire window — keep only the tail
            self.monitored_signal[:]          = data_frame.filtered_data_block[-n:]
            self.monitored_signal_abs[:]      = data_frame.abs_data_block[-n:]
            self.monitored_signal_unipolar[:] = data_frame.unipolar_filtered_data_block[-n:]
            self.sync_signal[:]               = data_frame.sync_data_block[-n:]
            if len(data_frame.stim_adc_data_block) >= n:
                self.stim_adc_signal[:] = data_frame.stim_adc_data_block[-n:]
        else:
            # Normal case: shift the fixed buffer left in-place (zero allocations),
            # then write new samples into the freed tail.  This replaces the
            # np.concatenate + trim pattern that was allocating ~61 MB/sec at
            # 12.5 kHz and causing the main-thread GC stalls / UI freezes.
            keep: int = n - n_new
            self.monitored_signal[:keep]          = self.monitored_signal[n_new:]
            self.monitored_signal[keep:]           = data_frame.filtered_data_block

            self.monitored_signal_abs[:keep]      = self.monitored_signal_abs[n_new:]
            self.monitored_signal_abs[keep:]       = data_frame.abs_data_block

            self.monitored_signal_unipolar[:keep] = self.monitored_signal_unipolar[n_new:]
            self.monitored_signal_unipolar[keep:]  = data_frame.unipolar_filtered_data_block

            self.sync_signal[:keep]               = self.sync_signal[n_new:]
            self.sync_signal[keep:]                = data_frame.sync_data_block

            n_stim = len(data_frame.stim_adc_data_block)
            if n_stim == n_new:
                self.stim_adc_signal[:keep] = self.stim_adc_signal[n_new:]
                self.stim_adc_signal[keep:] = data_frame.stim_adc_data_block
            elif n_stim > 0:
                # Length mismatch (e.g. missing channel) — safe fallback
                combined = np.concatenate([self.stim_adc_signal, data_frame.stim_adc_data_block])
                self.stim_adc_signal[:] = combined[-n:]

        #Bin the data
        bin_sz: int = MhRecruitmentCurveStage.bin_sample_count()
        for bin_index in range(0, len(self.bins)):
            bin_start = bin_sz * bin_index
            bin_end   = min(bin_sz * (bin_index + 1), n)
            self.bins[bin_index] = np.mean(self.monitored_signal_abs[bin_start:bin_end])

        #Get the mean of all the bins
        bin_grand_mean: float = float(np.mean(self.bins))

        #A trial is initiated only when every bin is within [min, max]
        #AND the grand mean is also within [min, max].
        if ((self.current_monitored_signal_sample_count >= n) and
            (np.all(self.bins >= current_initiation_min)) and
            (np.all(self.bins <= current_initiation_max)) and
            (bin_grand_mean >= current_initiation_min) and
            (bin_grand_mean <= current_initiation_max)):
            return True

        return False

    #endregion

class MhRecruitmentCurveStage (Stage):

    #region Constants

    #Define a set of trial states
    TRIAL_STATE_NOT_SETUP = 0
    TRIAL_STATE_WAIT_FOR_INITIATION = 1
    TRIAL_STATE_RECORD = 3
    TRIAL_STATE_FINALIZE = 4

    #This defines the duration of an individual bin in miliseconds
    BIN_DURATION_MILLISECONDS: int = 50


    #This defines the trial recording duration in milliseconds
    TRIAL_RECORDING_DURATION_MILLISECONDS: int = 500



    #The minimum duration for which we will scan for trial initiation criteria to be met
    TRIAL_INITIATION_PHASE_MIN_DURATION_MILLISECONDS: int = 2200

    #The maximum duration for which we will scan for trial initiation criteria to be met
    TRIAL_INITIATION_PHASE_MAX_DURATION_MILLISECONDS: int = 2700

    #The target number of trials per hour
    TARGET_TRIALS_PER_HOUR: int = 150

    #The stimulation amplitude bounds
    STIMULATION_AMPLITUDE_MIN: float = 0.0
    STIMULATION_AMPLITUDE_MAX: float = 2.0
    STIMULATION_AMPLITUDE_STEP: float = 0.1

    #The minimum inter-trial interval. We will not allow new trials during the designated timeout period.
    MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS: int = 10000

    #ADC sync line thresholds for stim onset/end detection.
    STIM_ONSET_THRESHOLD: float = 4.5
    STIM_END_THRESHOLD: float = 1.9

    #endregion

    #region Classmethods

    @classmethod
    def bin_sample_count(cls) -> int:
        '''Samples per bin, computed from the live Open Ephys sample rate.'''
        return int(cls.BIN_DURATION_MILLISECONDS * ApplicationConfiguration.sample_rate / 1000)

    @classmethod
    def trial_recording_sample_count(cls) -> int:
        '''Post-stimulus recording samples, computed from the live Open Ephys sample rate.'''
        return int(cls.TRIAL_RECORDING_DURATION_MILLISECONDS * ApplicationConfiguration.sample_rate / 1000)

    @classmethod
    def ms_per_sample(cls) -> float:
        '''Milliseconds per sample, computed from the live Open Ephys sample rate.'''
        return 1000.0 / ApplicationConfiguration.sample_rate

    #endregion

    #region Constructor

    def __init__(self):
        super().__init__()

        #Set the basic stage information
        self.stage_name = "S2"
        self.stage_description = "Mh Recruitment Curve"
        self.stage_type = Stage.STAGE_TYPE_RECRUITMENT_CURVE

        #Instantiate a random-number generator and use the current time as a seed
        self._rng: Random = Random(datetime.now().timestamp())

        #Instantiate a numpy random number generator as well
        self._numpy_rng: np.random.Generator = np.random.default_rng(seed = int(datetime.now().timestamp()))

        #Create a private variable that will be used to store a save-file handle
        self._fid: BinaryIO = None

        #Create variables to track how often we are triggering a trial
        ms_per_hour: int = 1000 * 60 * 60
        self._desired_ms_between_trials: float = ms_per_hour / MhRecruitmentCurveStage.TARGET_TRIALS_PER_HOUR
        self._average_ms_between_trials: float = 0.0
        self._ms_since_last_trial: int = 0

        self._auto_thresholding_enabled: bool = False
        self._current_min_initiation_threshold: float = 0.0
        self._current_max_initiation_threshold: float = 0.0
        self._current_stimulation_amplitude_ma: float = 0.0
        self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_NOT_SETUP
        
        #Create a list to hold all trials
        self._trials: list[MhRecruitmentCurveTrial] = []

        #Create an object to hold the current trial initiation data
        self._current_trial_initiation_data: MhRecruitmentCurveStage_TrialInitiationData = None

        #Create an object to hold the current trial
        self._current_trial: MhRecruitmentCurveTrial = None

        #Create an object to hold a set of stimulation amplitudes that we will sweep through
        self._stimulation_amplitudes: np.ndarray = np.array([], dtype=np.float64)

        #Create a variable that will store the user-defined minimum stimulation amplitude
        self._user_min_stimulation_amplitude: float = MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MIN

        #Create a variable that will store the user-defined maximum stimulation amplitude
        self._user_max_stimulation_amplitude: float = MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MAX

        #Create a variable that will store the user-defined stimulation amplitude step-size
        self._user_stimulation_amplitude_step_size: float = MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_STEP

        #Create a flag to control whether stimulation intensities are swept sequentially (ascending) or randomly
        self._sequential_stimulation: bool = False

        #Create a variable that will store the user-defined minimum stimulation interval
        self._user_minimum_stimulation_interval_milliseconds: int = MhRecruitmentCurveStage.MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS

        #Create variables that will store the user-defined window start/end times for the M wave and the H wave
        self._user_m_wave_window_start_time_milliseconds: float = 0
        self._user_m_wave_window_end_time_milliseconds: float = 0
        self._user_h_wave_window_start_time_milliseconds: float = 0
        self._user_h_wave_window_end_time_milliseconds: float = 0

        #Trial signal visibility flags (maps to get_trial_signal_options() order):
        #  index 0 = EMG signal (bipolar filtered)
        #  index 1 = ADC sync line
        #  index 2 = |EMG| (abs)
        #  index 3 = Tibial Stim ADC waveform
        #  index 4 = Group Avg overlay
        self._trial_signal_flags: list[bool] = [True, False, False, False, False]

        #Background EMG tracking (populated at each trial initiation from the monitoring window)
        self._background_emg_means: list[float] = []
        self._last_background_bins: np.ndarray = np.zeros(1)
        self._last_background_grand_mean: float = 0.0

        #Polarity filter for session and trial plots.
        #-1 = show all trials regardless of polarity (default).
        # 0 = show only polarity-0 (normal/toggle-off) trials.
        # 1 = show only polarity-1 (reversed/toggle-on) trials.
        self._session_polarity_filter: int = -1

        #When True, the Trial Timeline x-axis is shown in hours instead of minutes.
        self._timeline_use_hours: bool = False

        #Secondary ViewBox used to render the ADC sync line on a right-hand Y-axis.
        #Kept as an instance variable so it can be cleaned up before each redraw.
        self._trial_adc_viewbox: pg.ViewBox = None

        #Peak absolute values of the baseline-corrected ADC traces last drawn.
        #Stored so _on_trial_emg_yrange_changed can recompute ADC y-range without
        #redrawing the whole overlay.
        self._trial_adc_peak_pos: float = 1.0
        self._trial_adc_peak_neg: float = 1.0

        #Wall-clock millisecond timestamp recorded at the moment trigger_single() is called.
        #Used in TRIAL_STATE_RECORD to discard queued pre-trigger frames that the Qt signal
        #queue may have buffered before the trigger was sent.
        self._trigger_wall_time_ms: int = 0

        #Per-trial debug accumulators reset at the start of each RECORD state.
        self._n_pre_trigger_frames_discarded: int = 0
        self._frame_received_timestamps_ms: list = []
        self._first_post_trigger_frame_sample_id: int = 0

        #Manual initiation thresholds — set via the 'thresh' command before or during
        #a session to allow S2 to run without a prior S1 (EMG characterization) file.
        #When _manual_thresholds_set is True, initialize() accepts missing HRS1 data
        #and uses these values instead of the histogram-derived ones.
        self._manual_min_threshold: float = 0.0
        self._manual_max_threshold: float = 0.0
        self._manual_thresholds_set: bool = False

        #Software polarity toggle state.
        #False = normal (toggle off, default). True = reversed (toggle on).
        #Physical cathodal/anodal direction depends on cable orientation.
        self._reversed_polarity: bool = False

        pass
        
    #endregion

    #region Overrides

    def initialize (self, subject_id: str) -> tuple[bool, str]:
        #Set the subject id
        self._subject_id: str = subject_id

        #Create a private variable that will be used to store a save-file handle
        self._fid: BinaryIO = None

        #Create variables to track how often we are triggering a trial
        ms_per_hour: int = 1000 * 60 * 60
        self._desired_ms_between_trials: float = ms_per_hour / MhRecruitmentCurveStage.TARGET_TRIALS_PER_HOUR
        self._average_ms_between_trials: float = 0.0
        self._ms_since_last_trial: int = 0

        #Set some values used during this stage
        self._auto_thresholding_enabled = False
        self._current_stimulation_amplitude_ma = 0.0
        self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_NOT_SETUP
        self._trigger_wall_time_ms = 0
        self._n_pre_trigger_frames_discarded = 0
        self._frame_received_timestamps_ms = []
        self._first_post_trigger_frame_sample_id = 0
        self._digital_sync_event = None  # First rising-edge DIGITAL IN event captured during RECORD
        self._record_write_ptr: int = 0  # Write position in the pre-allocated trial data buffers

        #Create a list to hold all trials
        self._trials: list[MhRecruitmentCurveTrial] = []

        #Clear background EMG history for this session
        self._background_emg_means = []

        #Create an object to hold the current trial initiation data
        self._current_trial_initiation_data: MhRecruitmentCurveStage_TrialInitiationData = None

        #Create an object to hold the current trial
        self._current_trial: MhRecruitmentCurveTrial = None

        #Get the current datetime
        current_datetime: datetime = datetime.now()

        #Define the path where we will save data
        app_data_path: str = user_data_dir(ApplicationConfiguration.appname, ApplicationConfiguration.appauthor)
        app_experimental_data_path: str = os.path.join(app_data_path, "Data")
        file_path: str = os.path.join(app_experimental_data_path, self._subject_id)

        #Define a file name for the file to which we will save data
        file_timestamp: str = current_datetime.strftime("%Y%m%dT%H%M%S")
        file_name: str = f"{self._subject_id}_{file_timestamp}.hrs2"

        #Create the folder if it does not yet exist
        if (not os.path.exists(file_path)):
            os.makedirs(file_path)
        
        #Check to see if there is an existing hrs1 file for this subject.
        hrs1_found: bool = False
        files_list: list[str] = os.listdir(file_path)
        hrs1_file_name: str = ""
        for f in files_list:
            if (f.endswith("hrs1")):
                hrs1_file_name = f
                hrs1_found = True
                break

        if (not hrs1_found):
            #Allow the session to proceed without S1 data if the user has manually set
            #both initiation thresholds via the 'thresh' command.
            if (not self._manual_thresholds_set):
                return (False,
                    "No EMG characterization data was found for this subject. "
                    "Run stage S1 first, or set manual thresholds with: "
                    "thresh lb=X ub=Y")

            #Proceed with manually-set thresholds; no histogram data will be available.
            self._emg_histogram_data = None
            self._current_min_initiation_threshold = self._manual_min_threshold
            self._current_max_initiation_threshold = self._manual_max_threshold
        else:
            #If we reach this point in the code, then we have an existing HRS1 file for this animal.
            #Now let's check to see if there is already an existing HRS2 file for this animal.
            for f in files_list:
                if (f.endswith("hrs2")):
                    return (False, "This subject has already completed this stage. EMG sweep data exists for this animal. This stage cannot proceed. If this is an issue, please talk to your PI.")

            #Load in the EMG characterization data from stage 1
            self._emg_characterization_data: EmgCharacterizationDataFile = EmgCharacterizationDataFile()
            fid = open(os.path.join(file_path, hrs1_file_name), "rb")
            self._emg_characterization_data.read(fid)
            fid.close()

            #Calculate the histogram data from the EMG characterization data from stage 1
            self._emg_histogram_data: EmgHistogramData = self._emg_characterization_data.get_histogram_data()

            #Set thresholds from histogram data (manual thresholds are ignored when S1 exists)
            self._current_min_initiation_threshold = self._emg_histogram_data.min
            self._current_max_initiation_threshold = self._emg_histogram_data.max

        #Update the histogram plot (guards internally for missing histogram data)
        self._update_histogram_plot()

        #Open a file for saving data for this stage
        self._fid = open(os.path.join(file_path, file_name), "wb")

        #Save the file header for this session's data file
        self._save_file_header()

        #Display a message to the user with some information about this stage
        s1_source: str = "S1 histogram" if (self._emg_histogram_data is not None) else "manual (no S1 data)"
        commands_messages: list[str] = [
            "This stage supports the following commands: ",
            "lb = x, lb += x, lb -= x (Set the init threshold lower bound)",
            "ub = x, ub += x, ub -= x (Set the init threshold upper bound)",
            "thresh (Show current thresholds)",
            "thresh lb=X ub=Y  |  thresh lb=X  |  thresh ub=Y  (Set thresholds manually)",
            "auto on, auto off (Turn on/off the automated threshold algorithm)",
            f"--- Active thresholds [{s1_source}]: lb={self._current_min_initiation_threshold:.2f}  ub={self._current_max_initiation_threshold:.2f} ---"
        ]

        for message_str in commands_messages:
            message: SessionMessage = SessionMessage(message_str)
            self.signals.new_message.emit(message)

        #Make sure that the proper stimulation parameters are set on the model 4100
        ApplicationConfiguration.set_biphasic_stimulus_pulse_parameters(0.0, self._reversed_polarity)

        #Return from this function
        return (True, "")

    def process (self, data_frame: OpenEphysDataFrame) -> None:
        '''
        Processes the most recent incoming data and takes any actions
        that are necessary based on the incoming data.
        '''
        current_datetime: datetime = datetime.now()

        #Load in the data from the previous stage
        #That will give us our histogram

        #Plot the histogram, we need a min and max bound
        #Set some value within the min and max bound that will be used as the initiation threshold
        #This initiation threshold will vary during the session to try and maintain a certain
        #number of trials per hour.

        #Target trials per hour = 300 stims/day. Two 1 hour sessions. 150 trials/hour.

        #Each trial is a single stimulation
        #Each trial we will randomly choose the stimulation amplitude

        #2 mA max. 0.1 mA step size. Randomly sample that space. Build the recruitment curve.
        #As we are building it, allow the user to adjust the max amplitude to go higher if needed.

        #START HERE

        if (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_NOT_SETUP):
            #Set things up for a new trial
            self._setup_new_trial()

            #Pop the first stimulation amplitude from the list of stim amplitudes
            self._current_stimulation_amplitude_ma = self._stimulation_amplitudes[0]
            self._stimulation_amplitudes = self._stimulation_amplitudes[1:]

            #Set the stimulation parameters on the Model 4100 for the upcoming trial
            #Disarm the stimulator first so the Model 4100 accepts the amplitude change
            if (ApplicationConfiguration.stimulator is not None):
                ApplicationConfiguration.stimulator.set_active(False)

            ApplicationConfiguration.set_stimulation_amplitude(self._current_stimulation_amplitude_ma, self._reversed_polarity)

            if (ApplicationConfiguration.stimulator is not None):
                ApplicationConfiguration.stimulator.set_active(True)

            #Set the trial state
            self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_WAIT_FOR_INITIATION

        elif (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_WAIT_FOR_INITIATION):
            if (len(self._trials) > 0):
                #Update the amount of time that has passed since the last trial
                elapsed_ms: int = int((current_datetime - self._trials[-1].start_time).total_seconds() * 1000.0)

                #If we are still inside of the inter-trial timeout period, then just return immediately
                if (elapsed_ms < self._user_minimum_stimulation_interval_milliseconds):
                    return

            #Check to see if we should initiate a trial
            should_initiate_trial: bool = self._check_for_trial_initiation(data_frame)
            if (should_initiate_trial):
                #If it is determined that we should initiate a trial...

                #Snapshot the background EMG bin state at the moment of initiation
                if self._current_trial_initiation_data is not None:
                    self._last_background_bins = self._current_trial_initiation_data.bins.copy()
                    self._last_background_grand_mean = float(np.mean(self._last_background_bins))
                    self._background_emg_means.append(self._last_background_grand_mean)

                #------------------------------------------------------------------
                # DEBUG: stim criteria met
                #------------------------------------------------------------------
                decision_wall_time_ms: int = int(datetime.now().timestamp() * 1000)
                decision_oe_sample_id: int = data_frame.sample_id
                decision_msg: str = (
                    f"[DEBUG] Stim criteria met | "
                    f"wall_time={decision_wall_time_ms} ms | "
                    f"OE_sample_id={decision_oe_sample_id} | "
                    f"amp={self._current_stimulation_amplitude_ma:.2f} mA"
                )
                print(decision_msg)
                self.signals.new_message.emit(SessionMessage(decision_msg))
                #------------------------------------------------------------------

                #Set the trial state...
                self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_RECORD

                #Create a trial object
                self._current_trial = MhRecruitmentCurveTrial()
                self._current_trial.initialize(
                    self._current_min_initiation_threshold,
                    self._current_max_initiation_threshold,
                    self._current_stimulation_amplitude_ma
                )
                self._current_trial.background_emg_mean = self._last_background_grand_mean
                self._current_trial.background_bins = self._last_background_bins.copy()

                #Pre-allocate trial data buffers for zero-allocation recording.
                #Using in-place writes during RECORD instead of np.concatenate eliminates
                #allocation pressure that caused the main thread to fall behind at high
                #sample rates, keeping the Qt event queue short so the digital event frame
                #arrives before FINALIZE.
                _pre: int  = MhRecruitmentCurveStage.bin_sample_count()
                _post: int = MhRecruitmentCurveStage.trial_recording_sample_count()
                _buf: int  = _pre + _post + 1024  # extra room for one oversized frame

                self._current_trial.trial_data          = np.zeros(_buf, dtype=np.float32)
                self._current_trial.unipolar_trial_data = np.zeros(_buf, dtype=np.float32)
                self._current_trial.sync_data           = np.zeros(_buf, dtype=np.float32)
                self._current_trial.stim_adc_data       = np.zeros(_buf, dtype=np.float32)

                self._current_trial.trial_data[:_pre]          = self._current_trial_initiation_data.monitored_signal[-_pre:]
                self._current_trial.unipolar_trial_data[:_pre] = self._current_trial_initiation_data.monitored_signal_unipolar[-_pre:]
                self._current_trial.sync_data[:_pre]           = self._current_trial_initiation_data.sync_signal[-_pre:]
                self._current_trial.stim_adc_data[:_pre]       = self._current_trial_initiation_data.stim_adc_signal[-_pre:]

                self._record_write_ptr = _pre

                #Reset per-trial debug accumulators before entering RECORD state
                self._n_pre_trigger_frames_discarded = 0
                self._frame_received_timestamps_ms = []
                self._first_post_trigger_frame_sample_id = 0
                self._digital_sync_event = None

                #Record the wall-clock time at which the trigger is sent so that RECORD
                #state can identify and discard any queued pre-trigger frames.
                self._trigger_wall_time_ms = int(datetime.now().timestamp() * 1000)

                #------------------------------------------------------------------
                # DEBUG: trigger actually sent
                #------------------------------------------------------------------
                trigger_msg: str = (
                    f"[DEBUG] trigger_single() sent | "
                    f"trigger_wall_time={self._trigger_wall_time_ms} ms | "
                    f"OE_sample_id={data_frame.sample_id} | "
                    f"decision→trigger latency={self._trigger_wall_time_ms - decision_wall_time_ms} ms"
                )
                print(trigger_msg)
                self.signals.new_message.emit(SessionMessage(trigger_msg))
                #------------------------------------------------------------------

                #Stamp the polarity at the exact moment of the trigger so offline analysis
                #reflects what was actually delivered even if the user toggled mid-wait.
                self._current_trial.stim_polarity_reversed = 1 if self._reversed_polarity else 0

                #Trigger the Model 4100
                if (ApplicationConfiguration.stimulator is not None):
                    ApplicationConfiguration.stimulator.trigger_single()

                #Notify listeners (e.g. live EMG overlay) that a stim was just sent
                self.signals.stim_triggered.emit()

        elif (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_RECORD):

            #Only accumulate frames that were assembled by the background thread AFTER the
            #trigger was sent.  Because data_received_signal crosses a thread boundary it uses
            #Qt's queued connection, meaning several pre-trigger frames can already be waiting
            #in the main-thread event queue at the moment trigger_single() is called.  Those
            #frames were acquired before the stimulus and cannot contain the stim onset; adding
            #them to the trial record would push the real onset outside the search window.
            #
            #We fall through to _save_emg_data_frame at the end of process() regardless, so
            #the continuous background EMG record is never interrupted.
            frame_is_post_trigger: bool = (
                self._trigger_wall_time_ms == 0 or
                data_frame.timestamp_emitted == 0 or
                data_frame.timestamp_emitted >= self._trigger_wall_time_ms
            )

            if frame_is_post_trigger:
                #Capture the first rising-edge DIGITAL IN event on the configured sync channel.
                #Only look for it if we haven't found one yet this trial.
                if self._digital_sync_event is None:
                    target_ch: int = ApplicationConfiguration.sync_digital_channel - 1  # 1-indexed → 0-indexed
                    for ev in data_frame.digital_events:
                        if ev.event_id == 1 and ev.event_channel == target_ch:
                            self._digital_sync_event = ev
                            break

                #Track the Open Ephys sample ID of the very first post-trigger frame so
                #the analyst can locate the exact position in the continuous OE recording.
                if (len(self._frame_received_timestamps_ms) == 0):
                    self._first_post_trigger_frame_sample_id = data_frame.sample_id

                #Record this frame's background-thread emit timestamp for later debug output.
                self._frame_received_timestamps_ms.append(data_frame.timestamp_emitted)

                #In-place append into the pre-allocated buffers (zero allocations).
                _n_new: int   = len(data_frame.filtered_data_block)
                _ptr: int     = self._record_write_ptr
                _cap: int     = len(self._current_trial.trial_data)
                _space: int   = _cap - _ptr
                if _n_new > 0 and _space > 0:
                    _n_write: int = min(_n_new, _space)
                    self._current_trial.trial_data[_ptr:_ptr + _n_write]          = data_frame.filtered_data_block[:_n_write]
                    self._current_trial.unipolar_trial_data[_ptr:_ptr + _n_write] = data_frame.unipolar_filtered_data_block[:_n_write]
                    self._current_trial.sync_data[_ptr:_ptr + _n_write]           = data_frame.sync_data_block[:_n_write]
                    _n_stim: int = len(data_frame.stim_adc_data_block)
                    if _n_stim > 0:
                        self._current_trial.stim_adc_data[_ptr:_ptr + min(_n_write, _n_stim)] = data_frame.stim_adc_data_block[:min(_n_write, _n_stim)]
                    self._record_write_ptr += _n_write

                #Finalize when we have enough samples AND at least 50 ms have elapsed since
                #the trigger wall-time.  The 50 ms guard prevents a race condition at high
                #sample rates (10 kHz+) where many post-trigger frames are already waiting in
                #the Qt event queue — without the guard, recording would complete from those
                #queued frames before the digital event frame (carrying the OE sample_num of
                #the actual stimulus) has had time to arrive, causing onset_detected = 0.
                _total_needed: int = (MhRecruitmentCurveStage.bin_sample_count()
                                      + MhRecruitmentCurveStage.trial_recording_sample_count())
                _has_samples: bool = self._record_write_ptr >= _total_needed
                _has_time: bool    = (data_frame.timestamp_emitted
                                      >= self._trigger_wall_time_ms + 50)
                if _has_samples and _has_time:
                    _w: int = self._record_write_ptr
                    self._current_trial.trial_data          = self._current_trial.trial_data[:_w]
                    self._current_trial.unipolar_trial_data = self._current_trial.unipolar_trial_data[:_w]
                    self._current_trial.sync_data           = self._current_trial.sync_data[:_w]
                    self._current_trial.stim_adc_data       = self._current_trial.stim_adc_data[:_w]
                    self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_FINALIZE
            else:
                #Count discarded pre-trigger frames for debug output
                self._n_pre_trigger_frames_discarded += 1

        elif (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_FINALIZE):
            
            #Append the current trial to the session's list of trials
            self._trials.append(self._current_trial)

            #Compute and attach timing/sync debug fields before writing to disk
            self._compute_trial_debug_fields(self._current_trial)

            #------------------------------------------------------------------
            # DEBUG: onset detection result
            #------------------------------------------------------------------
            t = self._current_trial
            first_frame_delay_ms: float = (
                float(self._frame_received_timestamps_ms[0]) - float(self._trigger_wall_time_ms)
                if len(self._frame_received_timestamps_ms) > 0 else -1.0
            )
            onset_source_str: str = (
                "digital" if t.onset_detected == 2
                else ("ADC" if t.onset_detected == 1 else "none")
            )
            digital_sample_str: str = (
                str(self._digital_sync_event.sample_num)
                if self._digital_sync_event is not None else "N/A"
            )
            onset_msg: str = (
                f"[DEBUG] Trial finalized | "
                f"onset_source={onset_source_str} | "
                f"onset_detected={t.onset_detected} | "
                f"onset_sample={t.onset_sample_index} | "
                f"digital_sample_num={digital_sample_str} | "
                f"end_sample={t.stim_end_sample_index} | "
                f"stim_dur_ms={t.stim_duration_ms:.3f} | "
                f"sync_peak_V={t.sync_peak_voltage:.3f} | "
                f"pre_trigger_frames_discarded={t.n_pre_trigger_frames_discarded} | "
                f"first_frame_delay_ms={first_frame_delay_ms:.1f} | "
                f"first_OE_sample={t.first_post_trigger_frame_sample_id}"
            )
            print(onset_msg)
            self.signals.new_message.emit(SessionMessage(onset_msg))
            #------------------------------------------------------------------

            #Save the data for this trial to the session's data file
            self._current_trial.save_to_file(self._fid)

            #Display a message to the user in the application's message box
            message: SessionMessage = SessionMessage(f"Trial {len(self._trials)} initiated. Stimulation amplitude: {self._current_trial.stimulation_amplitude_ma:.2f} mA, threshold = [{self._current_trial.min_initiation_threshold:.2f}, {self._current_trial.max_initiation_threshold:.2f}]")
            self.signals.new_message.emit(message)

            #Calculate the average time between trials
            trial_isi_list: list[float] = []
            for i in range(1, len(self._trials)):
                isi: float = (self._trials[i].start_time - self._trials[i - 1].start_time).total_seconds() * 1000.0
                trial_isi_list.append(isi)
            
            if (len(trial_isi_list) > 0):
                self._average_ms_between_trials = np.mean(trial_isi_list)

            #Plot data about this trial in the application's charts
            self.update_trial_plot()
            self.update_session_plot()

            #Set the state
            self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_NOT_SETUP

        #Now save out the EMG data frame
        self._save_emg_data_frame(data_frame)

        return
    
    def input (self, user_input: str) -> None:

        #Display a message to the user in the application's message box
        message: SessionMessage = SessionMessage(f"Command received: {user_input}")
        self.signals.new_message.emit(message)

        #Convert the user input to all lowercase
        user_input = user_input.lower()

        #Remove all whitespace
        user_input = "".join(user_input.split())

        if (user_input.startswith("thresh")):
            self._parse_command_thresh(user_input)
        elif (user_input.startswith("lb")) or (user_input.startswith("ub")):
            self._parse_command_lb_ub(user_input)
        elif (user_input.startswith("auto")):
            self._parse_command_auto(user_input)
        else:
            self.signals.new_message.emit(SessionMessage("Command not recognized"))

        return

    def finalize (self) -> None:        
        if (self._fid is not None):
            #Close the data file for this session
            self._fid.close()

    def set_session_polarity_filter(self, value: int) -> None:
        '''Set polarity filter (-1=all, 0=normal, 1=reversed) and refresh both plots.'''
        self._session_polarity_filter = value
        self.update_session_plot()
        self.update_trial_plot()

    def set_timeline_use_hours(self, use_hours: bool) -> None:
        '''Switch the Trial Timeline x-axis between minutes and hours.'''
        self._timeline_use_hours = use_hours
        if self._session_plot_index == 4:
            self._update_trial_timeline_plot()

    def _filtered_trials(self) -> list:
        '''Return trials matching the current polarity filter (-1 = no filter).'''
        if self._session_polarity_filter < 0:
            return self._trials
        return [t for t in self._trials
                if t.stim_polarity_reversed == self._session_polarity_filter]

    def get_trial_plot_options (self) -> list[str]:
        return ["Most recent trial", "Most recent background", "EMG Level", "Group avg"]

    def get_trial_signal_options (self) -> list[str]:
        return ["EMG signal", "ADC sync line", "|EMG| (abs)", "Tibial Stim", "Group Avg"]

    def get_session_plot_options (self) -> list[str]:
        return ["S1 Histogram", "Recruitment Curve", "Stim Count Histogram",
                "H/M Wave Size", "Trial Timeline", "ITI Distribution"]

    def update_trial_plot (self) -> None:
        self._cleanup_trial_adc_viewbox()
        if self._trial_plot_index == 0:
            self._update_trial_plot()
        elif self._trial_plot_index == 1:
            self._update_most_recent_background_plot()
        elif self._trial_plot_index == 2:
            self._update_emg_level_plot()
        elif self._trial_plot_index == 3:
            self._update_group_avg_plot()

    def update_session_plot (self) -> None:

        if (self._session_plot_index == 0):
            self._update_histogram_plot()
        elif (self._session_plot_index == 1):
            self._update_recruitment_curve_plot()
        elif (self._session_plot_index == 2):
            self._update_stim_count_histogram_plot()
        elif (self._session_plot_index == 3):
            self._update_hm_wave_size_plot()
        elif (self._session_plot_index == 4):
            self._update_trial_timeline_plot()
        elif (self._session_plot_index == 5):
            self._update_iti_distribution_plot()

        pass

    def set_stim_params (self, min_amp: float, max_amp: float, step: float, isi_ms: int, sequential: bool) -> str:
        self._user_min_stimulation_amplitude = min_amp
        self._user_max_stimulation_amplitude = max_amp
        self._user_stimulation_amplitude_step_size = step
        self._user_minimum_stimulation_interval_milliseconds = isi_ms
        self._sequential_stimulation = sequential
        #Force regeneration of the amplitude list on the next trial setup
        self._stimulation_amplitudes = np.array([], dtype=np.float64)
        order = "sequential" if sequential else "randomized"
        return (f"Stim params set: min={min_amp:.2f} mA, max={max_amp:.2f} mA, "
                f"step={step:.2f} mA, ISI={isi_ms} ms, order={order}")

    #endregion

    #region Private methods

    def _save_file_header (self) -> None:
        if (self._fid is not None):

            #Create a file header object
            header: MhRecruitmentCurveHeader = MhRecruitmentCurveHeader(
                7,
                self._subject_id,
                datetime.now(),
                self.stage_name,
                self.stage_description,
                self.stage_type
            )

            #Save the header to the data file
            header.save_to_file(self._fid)

            pass

    def _save_emg_data_frame (self, data_frame: OpenEphysDataFrame) -> None:
        #Assimilate the data blocks
        emg_channel_names: list[str] = [x.channel_name for x in data_frame.channel_data_blocks]
        data_to_save: list[np.ndarray] = [x.data for x in data_frame.channel_data_blocks]

        #Create an object to prepare the data to save to a file
        emg_data: HReflexDataFileEmgData = HReflexDataFileEmgData(
            data_frame.timestamp,
            data_frame.channel_data_blocks[0].timestamp_received_millis,
            data_frame.timestamp_emitted,
            emg_channel_names,
            data_to_save,
            data_frame.diff_data_block,
            data_frame.filtered_data_block,
            data_frame.abs_data_block
        )

        #Save the data to the data file
        emg_data.save_to_file(self._fid)

        pass

    def _update_recruitment_curve_plot (self) -> None:
        self._session_widget.clear()
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(None)
        self._session_widget.getPlotItem().setTitle('M/H Recruitment Curve')
        self._session_widget.getPlotItem().setLabel('bottom', 'Stim Amp (mA)')
        self._session_widget.getPlotItem().setLabel('left', 'Mean |EMG| (µV)')

        trials = self._filtered_trials()
        if len(trials) == 0:
            return

        m_start = self._user_m_wave_window_start_time_milliseconds
        m_end   = self._user_m_wave_window_end_time_milliseconds
        h_start = self._user_h_wave_window_start_time_milliseconds
        h_end   = self._user_h_wave_window_end_time_milliseconds

        if m_end <= m_start and h_end <= h_start:
            text = pg.TextItem("Set M/H wave windows to view recruitment curve", anchor=(0.5, 0.5))
            self._session_widget.addItem(text)
            return

        ms_per_sample = MhRecruitmentCurveStage.ms_per_sample()

        m_dict: dict = {}
        h_dict: dict = {}

        for trial in trials:
            amp_key = round(trial.stimulation_amplitude_ma, 2)
            onset_idx = trial.onset_sample_index if trial.onset_sample_index >= 0 else MhRecruitmentCurveStage.bin_sample_count()

            n = len(trial.trial_data)
            t_ms = (np.arange(n) - onset_idx) * ms_per_sample
            emg  = np.abs(trial.trial_data)

            if m_end > m_start:
                m_mask = (t_ms >= m_start) & (t_ms <= m_end)
                if np.any(m_mask):
                    m_dict.setdefault(amp_key, []).append(float(np.mean(emg[m_mask])))

            if h_end > h_start:
                h_mask = (t_ms >= h_start) & (t_ms <= h_end)
                if np.any(h_mask):
                    h_dict.setdefault(amp_key, []).append(float(np.mean(emg[h_mask])))

        all_amps = sorted(set(m_dict.keys()) | set(h_dict.keys()))
        if len(all_amps) == 0:
            return

        all_amps_arr = np.array(all_amps, dtype=float)

        #Pin a tick mark at every amplitude that actually has data so the x-axis
        #always shows the exact mA values rather than auto-generated round numbers.
        amp_ticks = [[(a, f"{a:.2f}") for a in all_amps]]
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(amp_ticks)
        x_pad = 0.05 if len(all_amps) < 2 else (all_amps[-1] - all_amps[0]) / len(all_amps) * 0.6
        self._session_widget.setXRange(all_amps[0] - x_pad, all_amps[-1] + x_pad, padding=0)

        self._session_widget.addLegend(offset=(0, 0))

        if m_end > m_start and len(m_dict) > 0:
            m_means = np.array([np.mean(m_dict.get(a, [0])) for a in all_amps])
            m_stds  = np.array([float(np.std(m_dict[a])) if len(m_dict.get(a, [])) > 1 else 0.0 for a in all_amps])
            m_pen = pg.mkPen(color=(0, 0, 200), width=2)
            self._session_widget.plot(all_amps_arr, m_means, pen=m_pen, symbol='o',
                                      symbolBrush=(0, 0, 200), symbolSize=6, name='M-wave')
            nonzero = m_stds > 0
            if np.any(nonzero):
                m_err = pg.ErrorBarItem(x=all_amps_arr[nonzero], y=m_means[nonzero],
                                        top=m_stds[nonzero], bottom=m_stds[nonzero],
                                        pen=pg.mkPen(color=(0, 0, 200)))
                self._session_widget.addItem(m_err)

        if h_end > h_start and len(h_dict) > 0:
            h_means = np.array([np.mean(h_dict.get(a, [0])) for a in all_amps])
            h_stds  = np.array([float(np.std(h_dict[a])) if len(h_dict.get(a, [])) > 1 else 0.0 for a in all_amps])
            h_pen = pg.mkPen(color=(0, 160, 0), width=2)
            self._session_widget.plot(all_amps_arr, h_means, pen=h_pen, symbol='o',
                                      symbolBrush=(0, 160, 0), symbolSize=6, name='H-wave')
            nonzero = h_stds > 0
            if np.any(nonzero):
                h_err = pg.ErrorBarItem(x=all_amps_arr[nonzero], y=h_means[nonzero],
                                        top=h_stds[nonzero], bottom=h_stds[nonzero],
                                        pen=pg.mkPen(color=(0, 160, 0)))
                self._session_widget.addItem(h_err)

    def _update_histogram_plot (self) -> None:
        #Clear the plot
        self._session_widget.clear()
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(None)
        self._session_widget.getPlotItem().setTitle('S1 EMG Initiation Threshold')
        self._session_widget.getPlotItem().setLabel('bottom', 'EMG (µV)')
        self._session_widget.getPlotItem().setLabel('left', 'Count')

        if self._emg_histogram_data is not None:
            #Plot a histogram of the grand means from the EMG characterization data
            for i in range(0, len(self._emg_histogram_data.histogram_values)):
                hist_val: float = self._emg_histogram_data.histogram_values[i]
                hist_edge_01: float = self._emg_histogram_data.histogram_bin_edges[i]
                hist_edge_02: float = self._emg_histogram_data.histogram_bin_edges[i+1]
                hist_center: float = (hist_edge_01 + hist_edge_02) / 2.0
                hist_width: float = hist_edge_02 - hist_edge_01
                hist_bar: pg.BarGraphItem = pg.BarGraphItem(x = hist_center, height = hist_val, width = hist_width)
                self._session_widget.addItem(hist_bar)
        else:
            #No histogram data — thresholds were set manually; show a label indicating this
            text_item = pg.TextItem(text="Manual thresholds (no EMG characterization data)", anchor=(0.5, 0.5))
            self._session_widget.addItem(text_item)
            text_item.setPos(self._current_min_initiation_threshold, 0.5)

        #Plot vertical lines where the min and max thresholds are
        vert_line_pen = pg.mkPen(color=(255, 0, 0), width = 2.0, style = QtCore.Qt.DashLine)
        min_thresh_line: pg.InfiniteLine = pg.InfiniteLine(self._current_min_initiation_threshold, 90, vert_line_pen, movable=False,
            label='LB', labelOpts={'position': 0.9, 'color': (200, 0, 0)})
        max_thresh_line: pg.InfiniteLine = pg.InfiniteLine(self._current_max_initiation_threshold, 90, vert_line_pen, movable=False,
            label='UB', labelOpts={'position': 0.9, 'color': (200, 0, 0)})
        self._session_widget.addItem(min_thresh_line)
        self._session_widget.addItem(max_thresh_line)

        #Quartile markers (Q1/Q2/Q3) from the S1 histogram data
        if self._emg_histogram_data is not None and len(self._emg_histogram_data.quartiles) >= 3:
            quartile_names  = ['Q1', 'Q2', 'Q3']
            quartile_colors = [(0, 0, 200), (200, 0, 0), (0, 160, 0)]
            for val, name, color in zip(self._emg_histogram_data.quartiles, quartile_names, quartile_colors):
                pen = pg.mkPen(color=color, width=2.0, style=QtCore.Qt.DashLine)
                line = pg.InfiniteLine(
                    pos=val, angle=90, pen=pen, movable=False,
                    label=name,
                    labelOpts={'position': 0.75, 'color': color}
                )
                self._session_widget.addItem(line)

    def _update_stim_count_histogram_plot (self) -> None:
        '''Bar chart: number of trials delivered at each stimulation amplitude.'''
        self._session_widget.clear()
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(None)
        self._session_widget.getPlotItem().setLabel('bottom', 'Stim Amp (mA)')
        self._session_widget.getPlotItem().setLabel('left', 'Trial Count')
        self._session_widget.getPlotItem().setTitle('Stimulation Count by Amplitude')

        trials = self._filtered_trials()
        if len(trials) == 0:
            return

        #Count trials per amplitude (rounded to 2 dp to match recruitment curve keying)
        amp_counts: dict = {}
        for trial in trials:
            key = round(trial.stimulation_amplitude_ma, 2)
            amp_counts[key] = amp_counts.get(key, 0) + 1

        all_amps = sorted(amp_counts.keys())
        counts   = [amp_counts[a] for a in all_amps]

        if len(all_amps) < 2:
            bar_width = 0.05
        else:
            bar_width = (all_amps[-1] - all_amps[0]) / len(all_amps) * 0.8

        bar_item = pg.BarGraphItem(
            x=all_amps, height=counts, width=bar_width,
            brush=(70, 130, 180), pen=pg.mkPen('k', width=0.5)
        )
        self._session_widget.addItem(bar_item)
        self._session_widget.getPlotItem().showGrid(x=False, y=True, alpha=0.4)

        #Force a tick label at every amplitude so the x-axis always shows the exact
        #mA values; BarGraphItem does not report bounds to the ViewBox reliably.
        amp_ticks = [[(a, f"{a:.2f}") for a in all_amps]]
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(amp_ticks)
        x_pad = bar_width * 1.2
        self._session_widget.setXRange(all_amps[0] - x_pad, all_amps[-1] + x_pad, padding=0)
        self._session_widget.setYRange(0, max(counts) * 1.1, padding=0)

    def _update_hm_wave_size_plot (self) -> None:
        '''Box-and-whisker of M-wave peak amplitude, H-wave peak amplitude, and H:M ratio,
        grouped by stimulation amplitude.  Box = Q1–Q3, whiskers = mean ± SD, white median
        line, white diamond at mean, jittered individual dots.  Filtered by polarity.'''
        self._session_widget.clear()
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(None)
        self._session_widget.getPlotItem().setLabel('bottom', 'Stim Amp (mA)')
        self._session_widget.getPlotItem().setLabel('left', 'Mean |EMG| (µV)  /  H:M Ratio')
        self._session_widget.getPlotItem().setTitle('H/M Wave Size & H:M Ratio by Amplitude')
        self._session_widget.getPlotItem().showGrid(x=False, y=True, alpha=0.4)

        m_start = self._user_m_wave_window_start_time_milliseconds
        m_end   = self._user_m_wave_window_end_time_milliseconds
        h_start = self._user_h_wave_window_start_time_milliseconds
        h_end   = self._user_h_wave_window_end_time_milliseconds

        trials = self._filtered_trials()
        if not trials:
            return

        if m_end <= m_start and h_end <= h_start:
            self._session_widget.addItem(
                pg.TextItem("Set M/H wave windows to view size plot", anchor=(0.5, 0.5)))
            return

        ms_per_sample = MhRecruitmentCurveStage.ms_per_sample()

        #Collect per-amplitude data: {amp: {'m': [], 'h': [], 'hm': []}}
        amp_data: dict = {}
        for trial in trials:
            amp_key = round(trial.stimulation_amplitude_ma, 2)
            onset_idx = (trial.onset_sample_index if trial.onset_sample_index >= 0
                         else MhRecruitmentCurveStage.bin_sample_count())
            n = len(trial.trial_data)
            t_ms = (np.arange(n) - onset_idx) * ms_per_sample
            emg  = np.abs(trial.trial_data)

            m_mra = h_mra = np.nan
            if m_end > m_start:
                m_mask = (t_ms >= m_start) & (t_ms <= m_end)
                if np.any(m_mask):
                    m_mra = float(np.mean(emg[m_mask]))
            if h_end > h_start:
                h_mask = (t_ms >= h_start) & (t_ms <= h_end)
                if np.any(h_mask):
                    h_mra = float(np.mean(emg[h_mask]))

            d = amp_data.setdefault(amp_key, {'m': [], 'h': [], 'hm': []})
            if np.isfinite(m_mra):
                d['m'].append(m_mra)
            if np.isfinite(h_mra):
                d['h'].append(h_mra)
            if np.isfinite(m_mra) and np.isfinite(h_mra) and m_mra > 0:
                d['hm'].append(h_mra / m_mra)

        all_amps = sorted(amp_data.keys())
        if not all_amps:
            return

        rng = np.random.default_rng(0)
        bx_w = 0.10  # half-width of one box
        offsets = {'m': -bx_w * 2.5, 'h': 0.0, 'hm': bx_w * 2.5}
        colors  = {'m': (0, 0, 200), 'h': (0, 160, 0), 'hm': (220, 100, 0)}
        labels  = {'m': 'M-wave (µV)', 'h': 'H-wave (µV)', 'hm': 'H:M ratio'}

        self._session_widget.addLegend(offset=(0, 0))
        ax = self._session_widget.getAxis('bottom')
        ax.setTicks([[(i + 1, f'{a:.2f}') for i, a in enumerate(all_amps)]])

        for metric in ('m', 'h', 'hm'):
            col = colors[metric]
            off = offsets[metric]

            wx, wy = [], []
            cx, cy = [], []
            bx, by = [], []
            mdx, mdy = [], []
            mean_xv, mean_yv = [], []
            dot_x, dot_y = [], []

            for ai, amp in enumerate(all_amps):
                vals = np.array(amp_data[amp][metric], dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                xp = (ai + 1) + off
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                q1, med, q3 = (float(np.percentile(vals, 25)),
                               float(np.median(vals)),
                               float(np.percentile(vals, 75)))

                #SD whiskers
                wx.extend([xp, xp, np.nan])
                wy.extend([mu - sd, mu + sd, np.nan])
                cx.extend([xp - bx_w, xp + bx_w, np.nan,
                            xp - bx_w, xp + bx_w, np.nan])
                cy.extend([mu - sd, mu - sd, np.nan, mu + sd, mu + sd, np.nan])
                #Box Q1–Q3
                bx.extend([xp, xp, np.nan])
                by.extend([q1, q3, np.nan])
                #Median tick
                mdx.extend([xp - bx_w, xp + bx_w, np.nan])
                mdy.extend([med, med, np.nan])
                #Mean
                mean_xv.append(xp)
                mean_yv.append(mu)
                #Dots
                jit = rng.uniform(-bx_w * 0.5, bx_w * 0.5, size=len(vals))
                dot_x.extend((xp + jit).tolist())
                dot_y.extend(vals.tolist())

            if not mean_xv:
                continue

            # legend proxy
            self._session_widget.plot([], [], pen=pg.mkPen(col, width=3),
                                      name=labels[metric])
            if wx:
                self._session_widget.plot(wx, wy, pen=pg.mkPen(col, width=1.5), connect='finite')
            if cx:
                self._session_widget.plot(cx, cy, pen=pg.mkPen(col, width=1.5), connect='finite')
            if bx:
                self._session_widget.plot(bx, by, pen=pg.mkPen(col, width=6.0), connect='finite')
            if mdx:
                self._session_widget.plot(mdx, mdy, pen=pg.mkPen('w', width=2.0), connect='finite')
            self._session_widget.plot(mean_xv, mean_yv, pen=None, symbol='d',
                                      symbolBrush='w',
                                      symbolPen=pg.mkPen(col, width=1.5), symbolSize=7)
            self._session_widget.plot(dot_x, dot_y, pen=None, symbol='o',
                                      symbolBrush=pg.mkBrush(col[0], col[1], col[2], 80),
                                      symbolPen=None, symbolSize=5)

        self._session_widget.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(color=(150, 150, 150), width=1, style=QtCore.Qt.DashLine)
        ))

    def _update_trial_timeline_plot (self) -> None:
        '''Trial number (Y) vs elapsed session time (X), colored by polarity.
        X-axis unit is minutes or hours based on _timeline_use_hours.'''
        self._session_widget.clear()
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(None)

        time_label = 'Elapsed Time (hr)' if self._timeline_use_hours else 'Elapsed Time (min)'
        self._session_widget.getPlotItem().setLabel('bottom', time_label)
        self._session_widget.getPlotItem().setLabel('left', 'Trial #')
        self._session_widget.getPlotItem().setTitle('Trial Timeline')
        self._session_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.3)

        trials = self._filtered_trials()
        if not trials:
            return

        # Elapsed time is relative to the very first trial of the whole session
        # so the axis has consistent meaning regardless of polarity filter.
        t0 = self._trials[0].start_time if self._trials else trials[0].start_time
        divisor = 3600.0 if self._timeline_use_hours else 60.0

        self._session_widget.addLegend(offset=(0, 0))

        pol_groups = {0: ([], []), 1: ([], [])}
        for trial_num, trial in enumerate(trials, start=1):
            elapsed = (trial.start_time - t0).total_seconds() / divisor
            pol_groups[trial.stim_polarity_reversed][0].append(elapsed)
            pol_groups[trial.stim_polarity_reversed][1].append(trial_num)

        pol_styles = {
            0: ((0, 0, 200),   'Normal (0)'),
            1: ((220, 100, 0), 'Reversed (1)'),
        }

        # Draw a faint step line through all trials (ordered by time) then overlay polarity dots
        all_elapsed = [(trial.start_time - t0).total_seconds() / divisor for trial in trials]
        all_nums = list(range(1, len(trials) + 1))
        self._session_widget.plot(all_elapsed, all_nums,
                                  pen=pg.mkPen(color=(180, 180, 180), width=1))

        for pol, (xs, ys) in pol_groups.items():
            if not xs:
                continue
            col, lbl = pol_styles[pol]
            self._session_widget.plot(xs, ys, pen=None, symbol='o',
                                      symbolBrush=pg.mkBrush(*col),
                                      symbolPen=None, symbolSize=8, name=lbl)

    def _update_iti_distribution_plot (self) -> None:
        '''Histogram of inter-trial intervals (seconds) for polarity-filtered trials.'''
        self._session_widget.clear()
        self._session_widget.getPlotItem().getAxis('bottom').setTicks(None)
        self._session_widget.getPlotItem().setLabel('bottom', 'ITI (s)')
        self._session_widget.getPlotItem().setLabel('left', 'Count')
        self._session_widget.getPlotItem().setTitle('Inter-Trial Interval Distribution')
        self._session_widget.getPlotItem().showGrid(x=False, y=True, alpha=0.3)

        trials = self._filtered_trials()
        if len(trials) < 2:
            return

        itis = [(trials[i].start_time - trials[i - 1].start_time).total_seconds()
                for i in range(1, len(trials))]
        itis_arr = np.array(itis, dtype=float)
        itis_arr = itis_arr[itis_arr > 0]
        if len(itis_arr) == 0:
            return

        n_bins = max(5, min(30, len(itis_arr) // 2))
        counts, edges = np.histogram(itis_arr, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        bar_w   = float(edges[1] - edges[0]) * 0.85

        bar_item = pg.BarGraphItem(x=centers, height=counts, width=bar_w,
                                   brush=(70, 130, 180), pen=pg.mkPen('k', width=0.5))
        self._session_widget.addItem(bar_item)

        mu, sd = float(np.mean(itis_arr)), float(np.std(itis_arr))
        for val, label, color in [
            (mu,       f'Mean={mu:.1f}s',    (200, 0, 0)),
            (mu - sd,  f'−SD={mu-sd:.1f}s',  (150, 0, 0)),
            (mu + sd,  f'+SD={mu+sd:.1f}s',  (150, 0, 0)),
        ]:
            if val > 0:
                self._session_widget.addItem(pg.InfiniteLine(
                    pos=val, angle=90, movable=False,
                    pen=pg.mkPen(color=color, width=1.5, style=QtCore.Qt.DashLine),
                    label=label, labelOpts={'position': 0.9, 'color': color}))

    def _update_most_recent_background_plot (self) -> None:
        '''Bar chart of the bin means from the most recently captured background EMG window.'''
        self._trial_widget.clear()
        self._trial_widget.getPlotItem().setLabel('bottom', 'Bin #')
        self._trial_widget.getPlotItem().setLabel('left', 'EMG (µV)')
        self._trial_widget.getPlotItem().setTitle('Most Recent Background')

        if len(self._background_emg_means) == 0:
            return

        x_centers = np.arange(len(self._last_background_bins), dtype=float)
        bar_item = pg.BarGraphItem(
            x=x_centers, height=self._last_background_bins, width=0.8,
            brush=(70, 130, 180)
        )
        self._trial_widget.addItem(bar_item)

        mean_pen = pg.mkPen(color=(255, 0, 0), width=2.0, style=QtCore.Qt.DashLine)
        mean_line = pg.InfiniteLine(
            pos=self._last_background_grand_mean, angle=0, pen=mean_pen, movable=False,
            label=f'Mean={self._last_background_grand_mean:.2f}',
            labelOpts={'position': 0.5, 'color': (200, 0, 0)}
        )
        self._trial_widget.addItem(mean_line)

    def _update_emg_level_plot (self) -> None:
        '''Box-and-whisker of per-trial background EMG bins.
        Box = Q1–Q3, whiskers = absolute min and max bin, median line, mean diamond.
        Only trials matching the current polarity filter are shown.'''
        self._trial_widget.clear()
        self._trial_widget.getPlotItem().setLabel('bottom', 'Trial #')
        self._trial_widget.getPlotItem().setLabel('left', 'EMG Mean (µV)')
        self._trial_widget.getPlotItem().setTitle('Background EMG Level')

        trials = self._filtered_trials()
        if not trials:
            return

        bx_w = 0.35  # half-width of box and whisker caps

        wx, wy = [], []   # whisker verticals (min→max)
        cx, cy = [], []   # cap ticks at min and max
        bx, by = [], []   # box segments (Q1→Q3)
        mdx, mdy = [], [] # median horizontal ticks
        mean_x, mean_y = [], []

        for i, trial in enumerate(trials):
            bins = np.array(trial.background_bins, dtype=float)
            if len(bins) < 2:
                mean_x.append(i)
                mean_y.append(float(trial.background_emg_mean))
                continue
            mn, mx = float(np.min(bins)), float(np.max(bins))
            q1, med, q3 = float(np.percentile(bins, 25)), float(np.median(bins)), float(np.percentile(bins, 75))
            gm = float(np.mean(bins))

            wx.extend([i, i, np.nan])
            wy.extend([mn, mx, np.nan])
            cx.extend([i - bx_w, i + bx_w, np.nan, i - bx_w, i + bx_w, np.nan])
            cy.extend([mn, mn, np.nan, mx, mx, np.nan])
            bx.extend([i, i, np.nan])
            by.extend([q1, q3, np.nan])
            mdx.extend([i - bx_w, i + bx_w, np.nan])
            mdy.extend([med, med, np.nan])
            mean_x.append(i)
            mean_y.append(gm)

        col = (0, 0, 200)
        if wx:
            self._trial_widget.plot(wx, wy, pen=pg.mkPen(col, width=1.5), connect='finite')
        if cx:
            self._trial_widget.plot(cx, cy, pen=pg.mkPen(col, width=1.5), connect='finite')
        if bx:
            self._trial_widget.plot(bx, by, pen=pg.mkPen(col, width=6.0), connect='finite')
        if mdx:
            self._trial_widget.plot(mdx, mdy, pen=pg.mkPen('w', width=2.0), connect='finite')
        if mean_x:
            self._trial_widget.plot(mean_x, mean_y, pen=None, symbol='d',
                                    symbolBrush='w', symbolPen=pg.mkPen(col, width=1.5),
                                    symbolSize=8)

        for val, color in [
            (self._current_min_initiation_threshold, (0, 160, 0)),
            (self._current_max_initiation_threshold, (200, 0, 0)),
        ]:
            self._trial_widget.addItem(pg.InfiniteLine(
                pos=val, angle=0, movable=False,
                pen=pg.mkPen(color=color, width=1.5, style=QtCore.Qt.DashLine)
            ))

    def _update_group_avg_plot (self) -> None:
        '''
        Plots the trial-averaged EMG trace for all trials at the same stimulation
        amplitude as the most recently completed trial.  Each trial is aligned to
        its detected stim onset before averaging so the waveforms stack correctly.
        '''
        self._trial_widget.setBackground('w')
        self._trial_widget.clear()
        self._trial_widget.getPlotItem().hideAxis('right')
        self._trial_widget.getPlotItem().setLabel('bottom', 'Time (ms)')
        self._trial_widget.getPlotItem().setLabel('left', 'EMG (µV)')

        if self._current_trial is None or len(self._trials) == 0:
            return

        current_amp = round(self._current_trial.stimulation_amplitude_ma, 2)
        group_trials = [t for t in self._filtered_trials()
                        if round(t.stimulation_amplitude_ma, 2) == current_amp]
        if len(group_trials) == 0:
            return

        ms_per_sample = MhRecruitmentCurveStage.ms_per_sample()
        pre_samples  = int(2.0  / ms_per_sample)
        post_samples = int(15.0 / ms_per_sample)
        window_len   = pre_samples + post_samples

        aligned: list[np.ndarray] = []
        for trial in group_trials:
            onset_idx = (trial.onset_sample_index
                         if trial.onset_sample_index >= 0
                         else MhRecruitmentCurveStage.bin_sample_count())
            i_pre  = max(0, onset_idx - pre_samples)
            i_post = min(len(trial.trial_data), onset_idx + post_samples)
            if (i_post - i_pre) == window_len:
                aligned.append(trial.trial_data[i_pre:i_post])

        if len(aligned) == 0:
            return

        x_data  = (np.arange(window_len) - pre_samples) * ms_per_sample
        avg_trace = np.mean(np.stack(aligned), axis=0)

        m_start = self._user_m_wave_window_start_time_milliseconds
        m_end   = self._user_m_wave_window_end_time_milliseconds
        h_start = self._user_h_wave_window_start_time_milliseconds
        h_end   = self._user_h_wave_window_end_time_milliseconds

        if m_end > m_start:
            m_region = pg.LinearRegionItem(values=[m_start, m_end],
                                           brush=pg.mkBrush(0, 0, 255, 40), movable=False)
            m_region.setZValue(-10)
            self._trial_widget.addItem(m_region)

        if h_end > h_start:
            h_region = pg.LinearRegionItem(values=[h_start, h_end],
                                           brush=pg.mkBrush(0, 200, 0, 40), movable=False)
            h_region.setZValue(-10)
            self._trial_widget.addItem(h_region)

        self._trial_widget.plot(x_data, avg_trace,
                                pen=pg.mkPen(color=(0, 0, 200), width=2.5))

        self._trial_widget.addItem(
            pg.InfiniteLine(0, 90,
                            pg.mkPen(color=(0, 0, 255), width=2.0, style=QtCore.Qt.DashLine),
                            movable=False))

        self._trial_widget.setXRange(-2, 15, padding=0)
        self._trial_widget.getPlotItem().setTitle(
            f'Group Avg | {current_amp:.2f} mA | n={len(aligned)}')
        if getattr(self._trial_widget, '_trial_y_locked', False):
            self._trial_widget.setYRange(
                self._trial_widget._trial_y_min,
                self._trial_widget._trial_y_max,
                padding=0)

    def _update_trial_plot (self) -> None:
        #Remove any ADC ViewBox left from a previous draw before clearing
        self._cleanup_trial_adc_viewbox()

        #Clear the plot and set background: red tint for failed (no ADC pulse), white for success.
        #Background must be set before clear() so it applies to the fresh plot.
        self._trial_widget.setBackground(pg.mkColor(255, 220, 220) if not self._current_trial.onset_detected else 'w')
        self._trial_widget.clear()
        self._trial_widget.getPlotItem().hideAxis('right')
        self._trial_widget.getPlotItem().setLabel('bottom', 'Time (ms)')
        self._trial_widget.getPlotItem().setLabel('left', 'EMG (µV)')

        #Bold x-axis (y=0 reference line) so zero-crossings are easy to spot
        self._trial_widget.addItem(pg.InfiniteLine(
            pos=0, angle=0, movable=False,
            pen=pg.mkPen(color=(0, 0, 0), width=2.0)))

        emg_pen = pg.mkPen(color=(0, 0, 0), width=2.0)

        #-------------------------------------------------------------------
        # Phase 1: Read pre-computed onset from _compute_trial_debug_fields.
        # onset_detected == 2 → digital DIGITAL IN event (most precise)
        # onset_detected == 1 → ADC analog threshold crossing (fallback)
        # onset_detected == 0 → no onset found; zero at trigger boundary
        #-------------------------------------------------------------------
        trigger_sample: int = MhRecruitmentCurveStage.bin_sample_count()

        onset_was_detected: bool = self._current_trial.onset_detected > 0
        stim_onset_idx: int = (
            self._current_trial.onset_sample_index
            if onset_was_detected
            else trigger_sample
        )

        onset_source: str = (
            "digital" if self._current_trial.onset_detected == 2
            else ("ADC" if self._current_trial.onset_detected == 1
                  else "none")
        )

        if not onset_was_detected:
            fail_msg: str = (
                f"[FAILED STIM TRIAL] No stim onset detected (digital or ADC) — "
                f"plot zeroed at trigger boundary (sample {trigger_sample}). "
                f"Check stimulator connection and sync cable/digital input."
            )
            print(fail_msg)
            self.signals.new_message.emit(SessionMessage(fail_msg))
        else:
            print(
                f"[DEBUG] onset | source={onset_source} | "
                f"onset_idx={stim_onset_idx} | trigger_sample={trigger_sample}"
            )

        #-------------------------------------------------------------------
        # Phase 2: Cut peri-stimulus window (-20 ms … +30 ms around onset)
        #-------------------------------------------------------------------
        pre_stim_samples: int = int(2.0 / MhRecruitmentCurveStage.ms_per_sample())
        post_stim_samples: int = int(15.0 / MhRecruitmentCurveStage.ms_per_sample())

        idx_pre: int = max(0, stim_onset_idx - pre_stim_samples)
        idx_post: int = min(len(self._current_trial.trial_data), stim_onset_idx + post_stim_samples)

        data_to_plot = self._current_trial.trial_data[idx_pre:idx_post]
        num_samples: int = len(data_to_plot)

        #-------------------------------------------------------------------
        # Phase 3: Zero the time axis — t=0 is exactly stim onset
        #-------------------------------------------------------------------
        samples_before_onset: int = stim_onset_idx - idx_pre
        x_data = (np.arange(0, num_samples) - samples_before_onset) * MhRecruitmentCurveStage.ms_per_sample()

        #M-wave and H-wave shaded regions (drawn before signal so traces render on top)
        m_start = self._user_m_wave_window_start_time_milliseconds
        m_end   = self._user_m_wave_window_end_time_milliseconds
        h_start = self._user_h_wave_window_start_time_milliseconds
        h_end   = self._user_h_wave_window_end_time_milliseconds

        if m_end > m_start:
            m_region = pg.LinearRegionItem(
                values=[m_start, m_end],
                brush=pg.mkBrush(0, 0, 255, 40),
                movable=False
            )
            m_region.setZValue(-10)
            self._trial_widget.addItem(m_region)

        if h_end > h_start:
            h_region = pg.LinearRegionItem(
                values=[h_start, h_end],
                brush=pg.mkBrush(0, 200, 0, 40),
                movable=False
            )
            h_region.setZValue(-10)
            self._trial_widget.addItem(h_region)

        #Plot |EMG| abs trace in gray (index 2), drawn before bipolar signal so it renders underneath
        if len(self._trial_signal_flags) > 2 and self._trial_signal_flags[2]:
            abs_pen = pg.mkPen(color=(150, 150, 150), width=1.5)
            self._trial_widget.plot(x_data, np.abs(data_to_plot), pen=abs_pen)

        #Plot EMG signal (bipolar/differential filtered) in black (index 0)
        if len(self._trial_signal_flags) == 0 or self._trial_signal_flags[0]:
            self._trial_widget.plot(x_data, data_to_plot, pen=emg_pen)

        #Group Avg overlay in dashed magenta (index 4): mean of all trials at same amp + polarity
        if len(self._trial_signal_flags) > 4 and self._trial_signal_flags[4]:
            cur_amp = round(self._current_trial.stimulation_amplitude_ma, 2)
            cur_pol = self._current_trial.stim_polarity_reversed
            group = [t for t in self._trials
                     if round(t.stimulation_amplitude_ma, 2) == cur_amp
                     and t.stim_polarity_reversed == cur_pol]
            aligned_avg = []
            for gt in group:
                oi = gt.onset_sample_index if gt.onset_sample_index >= 0 else MhRecruitmentCurveStage.bin_sample_count()
                gp = max(0, oi - pre_stim_samples)
                gq = min(len(gt.trial_data), oi + post_stim_samples)
                if (gq - gp) == num_samples:
                    aligned_avg.append(gt.trial_data[gp:gq])
            if len(aligned_avg) > 1:
                avg_trace = np.mean(np.stack(aligned_avg), axis=0)
                avg_pen = pg.mkPen(color=(180, 0, 180), width=2.0, style=QtCore.Qt.DashLine)
                self._trial_widget.plot(x_data, avg_trace, pen=avg_pen)

        self._trial_widget.setXRange(-2, 15, padding=0)

        #Stim onset marker at t=0 (blue dashed)
        onset_pen = pg.mkPen(color=(0, 0, 255), width=2.0, style=QtCore.Qt.DashLine)
        self._trial_widget.addItem(pg.InfiniteLine(0, 90, onset_pen, movable=False))

        #-------------------------------------------------------------------
        # Phase 4: Stim end marker
        # Only drawn when the onset was positively detected.  When the fallback
        # position is used (onset not found), skipping the end marker avoids the
        # misleading "stim period = 0" display that previously appeared.
        #-------------------------------------------------------------------
        if onset_was_detected and len(self._current_trial.sync_data) > stim_onset_idx:
            post_onset_sync = self._current_trial.sync_data[stim_onset_idx:]
            end_candidates = np.where(post_onset_sync < MhRecruitmentCurveStage.STIM_END_THRESHOLD)[0]
            if len(end_candidates) > 0:
                stim_end_time_ms: float = int(end_candidates[0]) * MhRecruitmentCurveStage.ms_per_sample()
                end_pen = pg.mkPen(color=(128, 0, 128), width=2.0, style=QtCore.Qt.DashLine)
                self._trial_widget.addItem(pg.InfiniteLine(stim_end_time_ms, 90, end_pen, movable=False))

        #-------------------------------------------------------------------
        # Phase 5: Optional ADC overlays (sync line and/or stim ADC) on a
        #          shared secondary (right) Y-axis.
        #-------------------------------------------------------------------
        show_adc: bool = len(self._trial_signal_flags) > 1 and self._trial_signal_flags[1]
        show_stim_adc: bool = len(self._trial_signal_flags) > 3 and self._trial_signal_flags[3]

        if show_adc or show_stim_adc:
            sync_to_plot = None
            stim_adc_to_plot = None

            if show_adc and len(self._current_trial.sync_data) >= idx_post:
                adc_slice = self._current_trial.sync_data[idx_pre:idx_post]
                if len(adc_slice) == num_samples:
                    sync_to_plot = adc_slice

            if show_stim_adc and len(self._current_trial.stim_adc_data) >= idx_post:
                stim_slice = self._current_trial.stim_adc_data[idx_pre:idx_post]
                if len(stim_slice) == num_samples:
                    stim_adc_to_plot = stim_slice

            if sync_to_plot is not None or stim_adc_to_plot is not None:
                self._draw_trial_adc_overlay(x_data, sync_to_plot, stim_adc_to_plot,
                                             pre_onset_samples=samples_before_onset)

        #-------------------------------------------------------------------
        # Phase 6: MRA markers (star) at M-wave and H-wave windows
        # Star x = window midpoint, star y = mean(|EMG|) within window,
        # matching exactly what the recruitment curve and H/M size plots use.
        #-------------------------------------------------------------------
        if m_end > m_start:
            m_mask = (x_data >= m_start) & (x_data <= m_end)
            if np.any(m_mask):
                m_mra   = float(np.mean(np.abs(data_to_plot[m_mask])))
                m_mid_x = float((m_start + m_end) / 2.0)
                m_marker = pg.ScatterPlotItem(
                    x=[m_mid_x], y=[m_mra],
                    symbol='star', size=14,
                    pen=pg.mkPen(color=(0, 0, 200), width=1.5),
                    brush=pg.mkBrush(0, 0, 200, 200)
                )
                self._trial_widget.addItem(m_marker)

        if h_end > h_start:
            h_mask = (x_data >= h_start) & (x_data <= h_end)
            if np.any(h_mask):
                h_mra   = float(np.mean(np.abs(data_to_plot[h_mask])))
                h_mid_x = float((h_start + h_end) / 2.0)
                h_marker = pg.ScatterPlotItem(
                    x=[h_mid_x], y=[h_mra],
                    symbol='star', size=14,
                    pen=pg.mkPen(color=(0, 160, 0), width=1.5),
                    brush=pg.mkBrush(0, 160, 0, 200)
                )
                self._trial_widget.addItem(h_marker)

        #Title: trial number, amplitude, polarity, count at this amp, and sync status
        if self._current_trial is not None:
            cur_amp   = round(self._current_trial.stimulation_amplitude_ma, 2)
            cur_pol   = self._current_trial.stim_polarity_reversed
            n_at_amp  = sum(1 for t in self._trials
                            if round(t.stimulation_amplitude_ma, 2) == cur_amp
                            and t.stim_polarity_reversed == cur_pol)
            total     = len(self._trials)
            pol_str   = "Rev" if cur_pol else "Norm"
            fail_str  = " [NO SYNC]" if not onset_was_detected else ""
            self._trial_widget.getPlotItem().setTitle(
                f'Trial #{total} | {cur_amp:.2f} mA ({pol_str}) | n={n_at_amp}{fail_str}')

        #Honour the user-locked Y range if set via the "Set Y" control
        if getattr(self._trial_widget, '_trial_y_locked', False):
            self._trial_widget.setYRange(
                self._trial_widget._trial_y_min,
                self._trial_widget._trial_y_max,
                padding=0)

    def _cleanup_trial_adc_viewbox (self) -> None:
        '''Removes the secondary ADC ViewBox from the scene before a redraw.'''
        try:
            self._trial_widget.getPlotItem().vb.sigYRangeChanged.disconnect(self._on_trial_emg_yrange_changed)
        except Exception:
            pass
        if self._trial_adc_viewbox is not None:
            try:
                scene = self._trial_adc_viewbox.scene()
                if scene is not None:
                    scene.removeItem(self._trial_adc_viewbox)
            except Exception:
                pass
            self._trial_adc_viewbox = None

    def _draw_trial_adc_overlay (self, x_data: np.ndarray, sync_data: np.ndarray = None,
                                  stim_adc_data: np.ndarray = None, pre_onset_samples: int = 0) -> None:
        '''
        Adds ADC signals as traces on an independent right-hand Y-axis.
        sync_data  (green)  — TTL/ADC sync line used for onset detection.
        stim_adc_data (orange) — raw stimulation pulse waveform from ch3.
        Each signal is baseline-corrected using its pre-stim mean so it is
        vertically centered about the x-axis (zero = resting level).
        The X axis is shared with the main EMG plot so timing is pixel-perfect.
        '''
        plot_item = self._trial_widget.getPlotItem()

        #Disconnect any previous resize callback to avoid accumulation
        try:
            plot_item.vb.sigResized.disconnect(self._on_trial_plot_resized)
        except Exception:
            pass

        #Create a new ViewBox for the right axis
        adc_vb = pg.ViewBox()
        plot_item.scene().addItem(adc_vb)

        #Show and configure the right axis
        plot_item.showAxis('right')
        plot_item.getAxis('right').setLabel('ADC (V)', color='#007700')
        plot_item.getAxis('right').linkToView(adc_vb)

        #Link X and set initial geometry
        adc_vb.setXLink(plot_item)
        adc_vb.setGeometry(plot_item.vb.sceneBoundingRect())

        #Keep geometry in sync when the main view is resized
        plot_item.vb.sigResized.connect(self._on_trial_plot_resized)

        #Plot the sync line (green) if provided, zero-centered about the pre-stim baseline
        corrected_signals = []
        if sync_data is not None:
            baseline = float(np.mean(sync_data[:pre_onset_samples])) if pre_onset_samples > 0 else 0.0
            sync_corrected = sync_data - baseline
            sync_pen = pg.mkPen(color=(0, 160, 0), width=1.5)
            adc_vb.addItem(pg.PlotCurveItem(x_data, sync_corrected, pen=sync_pen))
            corrected_signals.append(sync_corrected)

        #Plot the stim ADC waveform (orange) if provided, zero-centered about the pre-stim baseline
        if stim_adc_data is not None:
            baseline = float(np.mean(stim_adc_data[:pre_onset_samples])) if pre_onset_samples > 0 else 0.0
            stim_corrected = stim_adc_data - baseline
            stim_pen = pg.mkPen(color=(255, 140, 0), width=1.5)
            adc_vb.addItem(pg.PlotCurveItem(x_data, stim_corrected, pen=stim_pen))
            corrected_signals.append(stim_corrected)

        #Align ADC y=0 with the bold x-axis (y=0 of the main EMG plot).
        #Store peak values and connect to sigYRangeChanged so the alignment is
        #recalculated whenever the EMG y-range changes (auto-range, Y-lock, view switch).
        if corrected_signals:
            all_corrected = np.concatenate(corrected_signals)
            self._trial_adc_peak_pos = max(float(np.max(all_corrected)), 0.0)
            self._trial_adc_peak_neg = max(-float(np.min(all_corrected)), 0.0)
        else:
            self._trial_adc_peak_pos = 1.0
            self._trial_adc_peak_neg = 1.0

        self._trial_adc_viewbox = adc_vb

        # Initial y-range alignment
        self._on_trial_emg_yrange_changed()

        # Reconnect so alignment updates whenever EMG y-range changes
        try:
            plot_item.vb.sigYRangeChanged.disconnect(self._on_trial_emg_yrange_changed)
        except Exception:
            pass
        plot_item.vb.sigYRangeChanged.connect(self._on_trial_emg_yrange_changed)

    def _on_trial_plot_resized (self) -> None:
        '''Keeps the ADC ViewBox geometry aligned with the main plot after a resize.'''
        if self._trial_adc_viewbox is not None and self._trial_adc_viewbox.scene() is not None:
            self._trial_adc_viewbox.setGeometry(
                self._trial_widget.getPlotItem().vb.sceneBoundingRect()
            )

    def _on_trial_emg_yrange_changed (self, *args) -> None:
        '''
        Recomputes the ADC ViewBox y-range so ADC y=0 stays pinned to the bold
        x-axis (EMG y=0) whenever the EMG y-range changes.
        '''
        if self._trial_adc_viewbox is None:
            return
        plot_item = self._trial_widget.getPlotItem()
        emg_y = plot_item.vb.viewRange()[1]
        emg_min, emg_max = float(emg_y[0]), float(emg_y[1])
        emg_span = emg_max - emg_min
        if emg_span <= 0:
            return
        # Fraction of pixel height that lies BELOW y=0 in the EMG coordinate system
        f = max(0.05, min(0.95, (0.0 - emg_min) / emg_span))
        peak_pos = self._trial_adc_peak_pos
        peak_neg = self._trial_adc_peak_neg
        total_range = max(
            peak_pos * 1.2 / max(1.0 - f, 1e-6),
            peak_neg * 1.2 / max(f, 1e-6),
            1e-6,
        )
        self._trial_adc_viewbox.setYRange(-f * total_range, (1.0 - f) * total_range, padding=0)

    def _determine_min_max_initiation_threshold (self) -> None:

        if (len(self._trials) == 0):
            self._current_min_initiation_threshold = self._emg_histogram_data.min
            self._current_max_initiation_threshold = self._emg_histogram_data.max
        else:
            if (self._average_ms_between_trials < self._desired_ms_between_trials):
                #Tighten things up
                self._current_min_initiation_threshold += (self._emg_histogram_data.step_size_one_percent * 50.0)
                self._current_max_initiation_threshold -= (self._emg_histogram_data.step_size_one_percent * 50.0)
            else:
                #Loosen things up
                self._current_min_initiation_threshold -= (self._emg_histogram_data.step_size_one_percent * 50.0)
                self._current_max_initiation_threshold += (self._emg_histogram_data.step_size_one_percent * 50.0)
            
            #Clamp the min/max initiation threshold values
            self._current_min_initiation_threshold = max(
                self._current_min_initiation_threshold, self._emg_histogram_data.min
            )

            self._current_max_initiation_threshold = min(
                self._current_max_initiation_threshold, self._emg_histogram_data.max
            )

            #Make sure the min is lower than the max
            if (self._current_min_initiation_threshold > self._current_max_initiation_threshold):
                temp_thresh: float = self._current_min_initiation_threshold
                self._current_min_initiation_threshold = self._current_max_initiation_threshold
                self._current_max_initiation_threshold = temp_thresh

            pass

        pass

    def _setup_new_trial (self) -> None:
        #Choose a trial-initiation monitoring duration
        dur_milliseconds: int = self._rng.randint(
            MhRecruitmentCurveStage.TRIAL_INITIATION_PHASE_MIN_DURATION_MILLISECONDS,
            MhRecruitmentCurveStage.TRIAL_INITIATION_PHASE_MAX_DURATION_MILLISECONDS
        )

        #Round the number to the nearest 50-ms
        dur_milliseconds = self._round_special(dur_milliseconds, 50)

        #Create an object to hold the trial initiation data
        self._current_trial_initiation_data = MhRecruitmentCurveStage_TrialInitiationData()
        self._current_trial_initiation_data.initialize(dur_milliseconds)

        #Determine the min/max initiation thresholds
        if (self._auto_thresholding_enabled):
            self._determine_min_max_initiation_threshold()

        #Check to see if we need to regenerate the stimulation amplitudes list
        if (len(self._stimulation_amplitudes) == 0):
            #Generate a list of stimulation amplitudes using user-defined parameters
            self._stimulation_amplitudes = np.arange(
                self._user_min_stimulation_amplitude,
                self._user_max_stimulation_amplitude + self._user_stimulation_amplitude_step_size,
                self._user_stimulation_amplitude_step_size, dtype=np.float64)

            #Shuffle or sort depending on the selected order mode
            if self._sequential_stimulation:
                self._stimulation_amplitudes = np.sort(self._stimulation_amplitudes)
            else:
                self._numpy_rng.shuffle(self._stimulation_amplitudes)

        #We are done. return from this function.
        return

    def _check_for_trial_initiation (self, data: np.ndarray) -> bool:
        if (self._current_trial_initiation_data is not None):
            return self._current_trial_initiation_data.process(
                data, 
                self._current_min_initiation_threshold, 
                self._current_max_initiation_threshold)
        else:
            return False

    def _round_special (self, x: int, base: int = 50) -> int:
        return base * int(round(float(x) / float(base)))

    def _compute_trial_debug_fields (self, trial) -> None:
        '''
        Computes stim-onset timing and writes the results — plus the frame-timing
        accumulators — into the trial object so they are persisted to the data file.

        Primary path: DIGITAL IN rising-edge event captured during RECORD state.
          onset_detected = 2, onset_sample_index derived from digital sample_num.

        Fallback path: ADC analog threshold crossing on the sync channel.
          onset_detected = 1.

        No onset found:
          onset_detected = 0, onset_sample_index = -1.
        '''
        #Carry over the frame-timing accumulators collected during RECORD state
        trial.trigger_wall_time_ms = self._trigger_wall_time_ms
        trial.n_pre_trigger_frames_discarded = self._n_pre_trigger_frames_discarded
        trial.frame_received_timestamps_ms = np.array(
            self._frame_received_timestamps_ms, dtype=np.uint64)
        trial.first_post_trigger_frame_sample_id = self._first_post_trigger_frame_sample_id

        #Defaults when no onset is found
        trial.onset_sample_index = -1
        trial.onset_detected = 0
        trial.stim_end_sample_index = -1
        trial.stim_duration_samples = 0
        trial.stim_duration_ms = 0.0
        trial.sync_peak_voltage = 0.0
        trial.digital_onset_sample_num = -1
        trial.digital_onset_channel = -1

        #------------------------------------------------------------------
        # PRIMARY: DIGITAL IN event
        # Map the absolute OE sample_num → index in trial_data.
        # trial_data layout: [0 : bin_sample_count()] = pre-stim window,
        #                    [bin_sample_count() : ...] = post-stim recording.
        # first_post_trigger_frame_sample_id is the absolute sample of the
        # first sample in the first post-trigger frame (= trial_data[bin_sample_count()]).
        #------------------------------------------------------------------
        if (self._digital_sync_event is not None
                and trial.first_post_trigger_frame_sample_id > 0):
            offset: int = (self._digital_sync_event.sample_num
                           - trial.first_post_trigger_frame_sample_id)
            digital_onset_idx: int = MhRecruitmentCurveStage.bin_sample_count() + offset
            n_trial: int = len(trial.sync_data)
            if 0 <= digital_onset_idx < n_trial:
                trial.onset_sample_index = digital_onset_idx
                trial.onset_detected = 2  # 2 = digital event used
                trial.digital_onset_sample_num = self._digital_sync_event.sample_num
                trial.digital_onset_channel = self._digital_sync_event.event_channel

        #------------------------------------------------------------------
        # FALLBACK: ADC analog threshold search (used when digital is absent)
        #------------------------------------------------------------------
        if trial.onset_detected != 2:
            search_start: int = MhRecruitmentCurveStage.bin_sample_count()
            search_end: int = min(
                search_start + MhRecruitmentCurveStage.trial_recording_sample_count(),
                len(trial.sync_data)
            )

            if len(trial.sync_data) > search_start:
                search_window = trial.sync_data[search_start:search_end]
                trial.sync_peak_voltage = float(np.max(search_window)) if len(search_window) > 0 else 0.0

                above = np.where(search_window >= MhRecruitmentCurveStage.STIM_ONSET_THRESHOLD)[0]
                if len(above) > 0:
                    chosen: int = int(above[0])
                    for i in range(len(above) - 1):
                        if above[i + 1] == above[i] + 1:
                            chosen = int(above[i])
                            break

                    trial.onset_sample_index = search_start + chosen
                    trial.onset_detected = 1

        #------------------------------------------------------------------
        # End-of-pulse detection (runs regardless of onset source)
        #------------------------------------------------------------------
        if trial.onset_sample_index >= 0 and len(trial.sync_data) > trial.onset_sample_index:
            post_onset = trial.sync_data[trial.onset_sample_index:]
            end_candidates = np.where(
                post_onset < MhRecruitmentCurveStage.STIM_END_THRESHOLD)[0]
            if len(end_candidates) > 0:
                trial.stim_end_sample_index = (
                    trial.onset_sample_index + int(end_candidates[0]))
                trial.stim_duration_samples = (
                    trial.stim_end_sample_index - trial.onset_sample_index)
                trial.stim_duration_ms = (
                    trial.stim_duration_samples * MhRecruitmentCurveStage.ms_per_sample())

    def _parse_command_lb_ub (self, user_input: str) -> None:
        #Check to see if we should just report existing values
        if (user_input == "lb"):
            self.signals.new_message.emit(SessionMessage(f"Current lower bound: {self._current_min_initiation_threshold:.2f}"))
            return
        elif (user_input == "ub"):
            self.signals.new_message.emit(SessionMessage(f"Current upper bound: {self._current_max_initiation_threshold:.2f}"))
            return

        #Set a flag to distinguish lower from upper bound command
        is_lower: bool = False
        if (user_input.startswith("lb")):
            user_input = user_input.removeprefix("lb")
            is_lower = True
        elif (user_input.startswith("ub")):
            user_input = user_input.removeprefix("ub")

        #Determine the desired operation
        operations: dict[str, int] = {
            "-=": -1,
            "=": 0,
            "+=": 1
        }
        op_found: bool = False
        desired_operation: int = 0
        for op in operations.keys():
            if (user_input.startswith(op)):
                desired_operation = operations[op]
                user_input = user_input.removeprefix(op)
                op_found = True
                break
        
        #If a valid operation was not found, return immediately
        if (not op_found):
            self.signals.new_message.emit(SessionMessage("Command failed: invalid operation"))
            return

        #Get the value for the operation
        desired_value: float = 0.0
        try:
            desired_value = float(user_input)
        except ValueError:
            #If the user input cannot be converted to a float,
            #then return immediately
            self.signals.new_message.emit(SessionMessage("Command failed: invalid value"))
            return
        
        #Perform the operation
        if (is_lower):
            lb: float = self._current_min_initiation_threshold
            if (desired_operation == -1):
                lb -= desired_value
            elif (desired_operation == 0):
                lb = desired_value
            elif (desired_operation == 1):
                lb += desired_value
            
            if(lb >= self._current_max_initiation_threshold):
                self.signals.new_message.emit(SessionMessage("Command failed: lower bound cannot be higher than the upper bound"))
                return
            
            #Clamp to histogram min only when S1 data is available
            if (self._emg_histogram_data is not None):
                lb = max(lb, self._emg_histogram_data.min)
            self._current_min_initiation_threshold = lb
            self._manual_min_threshold = lb
            self._manual_thresholds_set = True
            self.signals.new_message.emit(SessionMessage(f"Lower bound set: {self._current_min_initiation_threshold:.2f}"))
            self.update_session_plot()
        else:
            ub: float = self._current_max_initiation_threshold

            if (desired_operation == -1):
                ub -= desired_value
            elif (desired_operation == 0):
                ub = desired_value
            elif (desired_operation == 1):
                ub += desired_value

            if (ub <= self._current_min_initiation_threshold):
                self.signals.new_message.emit(SessionMessage("Command failed: upper bound cannot be lower than the lower bound"))
                return

            #Clamp to histogram max only when S1 data is available
            if (self._emg_histogram_data is not None):
                ub = min(ub, self._emg_histogram_data.max)
            self._current_max_initiation_threshold = ub
            self._manual_max_threshold = ub
            self._manual_thresholds_set = True
            self.signals.new_message.emit(SessionMessage(f"Upper bound set: {self._current_max_initiation_threshold:.2f}"))
            self.update_session_plot()
        pass

    def _parse_command_thresh (self, user_input: str) -> None:
        '''
        Handles the 'thresh' command family.

        thresh                  — display current lb and ub
        thresh lb=X             — set lower bound to X
        thresh ub=X             — set upper bound to X
        thresh lb=X ub=Y        — set both at once

        Works before AND during a session.  When used before starting S2, the values
        are stored as manual thresholds so initialize() can proceed without an HRS1 file.
        '''
        #Strip the command verb and any surrounding whitespace
        remainder: str = user_input.removeprefix("thresh").strip()

        #No arguments — just display current values
        if (remainder == ""):
            source: str = "S1 histogram" if (self._emg_histogram_data is not None) else "manual"
            self.signals.new_message.emit(SessionMessage(
                f"Initiation thresholds [{source}]: "
                f"lb={self._current_min_initiation_threshold:.2f}  "
                f"ub={self._current_max_initiation_threshold:.2f}"
            ))
            if (self._manual_thresholds_set):
                self.signals.new_message.emit(SessionMessage(
                    f"Stored manual thresholds: lb={self._manual_min_threshold:.2f}  ub={self._manual_max_threshold:.2f}"
                ))
            return

        #Parse key=value pairs (lb and/or ub)
        new_lb: float = self._current_min_initiation_threshold
        new_ub: float = self._current_max_initiation_threshold
        lb_changed: bool = False
        ub_changed: bool = False

        for token in remainder.split():
            token = token.replace(" ", "")
            if token.startswith("lb="):
                try:
                    new_lb = float(token[3:])
                    lb_changed = True
                except ValueError:
                    self.signals.new_message.emit(SessionMessage("thresh: invalid value for lb"))
                    return
            elif token.startswith("ub="):
                try:
                    new_ub = float(token[3:])
                    ub_changed = True
                except ValueError:
                    self.signals.new_message.emit(SessionMessage("thresh: invalid value for ub"))
                    return
            else:
                self.signals.new_message.emit(SessionMessage(
                    f"thresh: unrecognized token '{token}'. "
                    f"Usage: thresh lb=X ub=Y"))
                return

        if (not lb_changed) and (not ub_changed):
            self.signals.new_message.emit(SessionMessage("thresh: no lb= or ub= found"))
            return

        if (new_lb >= new_ub):
            self.signals.new_message.emit(SessionMessage(
                "thresh: lower bound must be less than upper bound"))
            return

        #Apply, clamping to histogram bounds if S1 data is available
        if (self._emg_histogram_data is not None):
            new_lb = max(new_lb, self._emg_histogram_data.min)
            new_ub = min(new_ub, self._emg_histogram_data.max)

        self._current_min_initiation_threshold = new_lb
        self._current_max_initiation_threshold = new_ub
        self._manual_min_threshold = new_lb
        self._manual_max_threshold = new_ub
        self._manual_thresholds_set = True

        self.signals.new_message.emit(SessionMessage(
            f"Thresholds updated: lb={self._current_min_initiation_threshold:.2f}  "
            f"ub={self._current_max_initiation_threshold:.2f}"
        ))
        print(
            f"[thresh] lb={self._current_min_initiation_threshold:.2f}  "
            f"ub={self._current_max_initiation_threshold:.2f}  "
            f"manual_thresholds_set={self._manual_thresholds_set}"
        )
        self.update_session_plot()

    def _parse_command_auto (self, user_input: str) -> None:
        on_or_off: str = user_input.removeprefix("auto")
        if (on_or_off == "on"):
            self._auto_thresholding_enabled = True
            self.signals.new_message.emit(SessionMessage("Auto thresholding: ENABLED"))
        else:
            self._auto_thresholding_enabled = False
            self.signals.new_message.emit(SessionMessage("Auto thresholding: DISABLED"))

    #endregion

    #region Public methods

    def manual_stim (self) -> None:
        '''
        Manually triggers a stimulation using the next amplitude from the algorithm.
        Only fires when the stage is actively waiting for trial initiation.
        The inter-trial interval is still enforced.
        '''
        #Only allowed while waiting for EMG-triggered initiation
        if (self._current_trial_state != MhRecruitmentCurveStage.TRIAL_STATE_WAIT_FOR_INITIATION):
            self.signals.new_message.emit(SessionMessage("Manual stim ignored: trial already in progress or stage not ready"))
            return

        #Enforce the inter-trial interval
        if (len(self._trials) > 0):
            elapsed_ms: int = int((datetime.now() - self._trials[-1].start_time).total_seconds() * 1000.0)
            if (elapsed_ms < MhRecruitmentCurveStage.MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS):
                remaining_sec: float = (MhRecruitmentCurveStage.MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS - elapsed_ms) / 1000.0
                self.signals.new_message.emit(SessionMessage(f"Manual stim ignored: inter-trial interval not elapsed ({remaining_sec:.1f}s remaining)"))
                return

        #Reset per-trial debug accumulators before entering RECORD state
        self._n_pre_trigger_frames_discarded = 0
        self._frame_received_timestamps_ms = []
        self._first_post_trigger_frame_sample_id = 0
        self._digital_sync_event = None

        #Transition to the record state
        self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_RECORD

        #Create the trial object
        self._current_trial = MhRecruitmentCurveTrial()
        self._current_trial.initialize(
            self._current_min_initiation_threshold,
            self._current_max_initiation_threshold,
            self._current_stimulation_amplitude_ma
        )

        #Transfer the last 50 ms of pre-stim data from the rolling buffer
        self._current_trial.trial_data = self._current_trial_initiation_data.monitored_signal[-MhRecruitmentCurveStage.bin_sample_count():]
        self._current_trial.sync_data = self._current_trial_initiation_data.sync_signal[-MhRecruitmentCurveStage.bin_sample_count():].copy()
        self._current_trial.stim_adc_data = self._current_trial_initiation_data.stim_adc_signal[-MhRecruitmentCurveStage.bin_sample_count():].copy()

        #Record the trigger wall-clock time before firing so the RECORD state
        #can discard any queued pre-trigger frames (same fix as the automatic path).
        self._trigger_wall_time_ms = int(datetime.now().timestamp() * 1000)

        #Trigger the stimulator
        if (ApplicationConfiguration.stimulator is not None):
            ApplicationConfiguration.stimulator.trigger_single()

        #Notify listeners (e.g. live EMG overlay) that a stim was just sent
        self.signals.stim_triggered.emit()

        self.signals.new_message.emit(SessionMessage(f"Manual stim triggered at {self._current_stimulation_amplitude_ma:.2f} mA"))

    def toggle_polarity(self) -> bool:
        '''
        Flips the biphasic pulse polarity between positive-first (normal/anodal) and
        negative-first (reversed/cathodal). Immediately re-applies the full stimulus
        parameter block so the Model 4100 picks up the change before the next trial.

        Returns the new polarity state: False = positive-first, True = negative-first.
        '''
        self._reversed_polarity = not self._reversed_polarity

        ApplicationConfiguration.set_biphasic_stimulus_pulse_parameters(
            self._current_stimulation_amplitude_ma,
            self._reversed_polarity
        )

        #set_biphasic_stimulus_pulse_parameters calls set_active(False) internally to let
        #the Model 4100 accept the parameter change.  Re-arm here so the very next trigger
        #fires correctly; without this the stimulator stays disarmed and the trial fails.
        if ApplicationConfiguration.stimulator is not None:
            ApplicationConfiguration.stimulator.set_active(True)

        polarity_label: str = "reversed (toggle on)" if self._reversed_polarity else "normal (toggle off)"
        self.signals.new_message.emit(SessionMessage(f"Pulse polarity set to: {polarity_label}"))

        return self._reversed_polarity

    def set_manual_thresholds(self, lb: float, ub: float) -> str:
        '''
        Sets manual initiation thresholds directly (e.g. from the GUI threshold panel).
        Works before or during a session. Values are applied as-is with no clamping.
        Returns a result message string (does not emit any signals).
        '''
        if lb >= ub:
            return "Threshold error: lower bound must be less than upper bound."

        self._manual_min_threshold = lb
        self._manual_max_threshold = ub
        self._manual_thresholds_set = True
        self._current_min_initiation_threshold = lb
        self._current_max_initiation_threshold = ub

        #Refresh the session plot so threshold lines update immediately
        self.update_session_plot()

        return f"Init thresholds overridden: lb={lb:.2f} µV   ub={ub:.2f} µV"

    #endregion