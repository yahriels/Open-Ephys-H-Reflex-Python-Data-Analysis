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
        self.bins = np.zeros(bin_count)

        #We are done. return from this function.
        return

    def process (self, data_frame: OpenEphysDataFrame, current_initiation_min: float, current_initiation_max: float) -> bool:
        should_initiate_trial: bool = False

        #Add the number of samples we are pulling in to the current trial sample count
        self.current_monitored_signal_sample_count += len(data_frame.diff_data_block)

        #Add the new data to the monitored signal (filtered differential signal)
        self.monitored_signal = np.concatenate([self.monitored_signal, data_frame.filtered_data_block])
        self.monitored_signal_abs = np.concatenate([self.monitored_signal_abs, data_frame.abs_data_block])
        self.monitored_signal_unipolar = np.concatenate([self.monitored_signal_unipolar, data_frame.unipolar_filtered_data_block])
        self.sync_signal = np.concatenate([self.sync_signal, data_frame.sync_data_block])
        elements_to_remove: int = len(self.monitored_signal) - self.monitored_signal_sample_count
        if (elements_to_remove > 0):
            self.monitored_signal = self.monitored_signal[elements_to_remove:]
            self.monitored_signal_abs = self.monitored_signal_abs[elements_to_remove:]
            self.monitored_signal_unipolar = self.monitored_signal_unipolar[elements_to_remove:]
            self.sync_signal = self.sync_signal[elements_to_remove:]

        #Bin the data
        for bin_index in range(0, len(self.bins)):
            bin_start = MhRecruitmentCurveStage.bin_sample_count() * bin_index
            bin_end = MhRecruitmentCurveStage.bin_sample_count() * (bin_index + 1)

            if (bin_end > len(self.monitored_signal_abs)):
                bin_end = len(self.monitored_signal_abs)

            bin_mean: float = np.mean(self.monitored_signal_abs[bin_start:bin_end])
            self.bins[bin_index] = bin_mean
        
        #Get the mean of all the bins
        bin_grand_mean: float = np.mean(self.bins)

        #If the bin grand mean is within a pre-specified min or max range, then
        #we consider this a trial initiation.
        if ((self.current_monitored_signal_sample_count >= self.monitored_signal_sample_count) and
            (bin_grand_mean >= current_initiation_min) and 
            (bin_grand_mean <= current_initiation_max)):

            should_initiate_trial = True

        return should_initiate_trial

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

        #Trial signal visibility flags:
        #  index 0 = bipolar filtered (diff), index 1 = ADC sync line, index 2 = |bipolar filtered|
        #  index 3 = unipolar filtered, index 4 = |unipolar filtered|
        self._trial_signal_flags: list[bool] = [True, False, False, False, False]

        #Secondary ViewBox used to render the ADC sync line on a right-hand Y-axis.
        #Kept as an instance variable so it can be cleaned up before each redraw.
        self._trial_adc_viewbox: pg.ViewBox = None

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
        
        #Create a list to hold all trials
        self._trials: list[MhRecruitmentCurveTrial] = []

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
        ApplicationConfiguration.set_biphasic_stimulus_pulse_parameters(0.0)

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

            ApplicationConfiguration.set_stimulation_amplitude(self._current_stimulation_amplitude_ma)

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

                #Transfer the last 50 ms of trial initiation data into the trial object
                self._current_trial.trial_data = self._current_trial_initiation_data.monitored_signal[-MhRecruitmentCurveStage.bin_sample_count():]

                #Transfer the last 50 ms of unipolar initiation data into the trial object
                self._current_trial.unipolar_trial_data = self._current_trial_initiation_data.monitored_signal_unipolar[-MhRecruitmentCurveStage.bin_sample_count():].copy()

                #Transfer the last 50 ms of ADC sync line data into the trial object (pre-stim window)
                self._current_trial.sync_data = self._current_trial_initiation_data.sync_signal[-MhRecruitmentCurveStage.bin_sample_count():].copy()

                #Reset per-trial debug accumulators before entering RECORD state
                self._n_pre_trigger_frames_discarded = 0
                self._frame_received_timestamps_ms = []
                self._first_post_trigger_frame_sample_id = 0

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
                #Track the Open Ephys sample ID of the very first post-trigger frame so
                #the analyst can locate the exact position in the continuous OE recording.
                if (len(self._frame_received_timestamps_ms) == 0):
                    self._first_post_trigger_frame_sample_id = data_frame.sample_id

                #Record this frame's background-thread emit timestamp for later debug output.
                self._frame_received_timestamps_ms.append(data_frame.timestamp_emitted)

                #Copy data into the trial object until we have 100 ms of post-stim data
                self._current_trial.trial_data = np.concatenate([self._current_trial.trial_data, data_frame.filtered_data_block])
                self._current_trial.unipolar_trial_data = np.concatenate([self._current_trial.unipolar_trial_data, data_frame.unipolar_filtered_data_block])
                self._current_trial.sync_data = np.concatenate([self._current_trial.sync_data, data_frame.sync_data_block])

                #Check to see if we have enough data
                if (len(self._current_trial.trial_data) >= (MhRecruitmentCurveStage.bin_sample_count() + MhRecruitmentCurveStage.trial_recording_sample_count())):
                    #If so, move on to the next stage
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
            onset_msg: str = (
                f"[DEBUG] Trial finalized | "
                f"onset_detected={t.onset_detected} | "
                f"onset_sample={t.onset_sample_index} | "
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
            self._update_trial_plot()
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

    def get_trial_plot_options (self) -> list[str]:
        return ["Most recent trial"]

    def get_trial_signal_options (self) -> list[str]:
        return ["EMG signal", "ADC sync line", "|EMG| (abs)", "Unipolar EMG signal", "|Unipolar EMG| (abs)"]

    def get_session_plot_options (self) -> list[str]:
        return ["S1 Histogram", "Recruitment Curve", "Unipolar Recruitment Curve"]

    def update_trial_plot (self) -> None:
        #This stage does not support updating the "most recent trial plot" from an external function call.
        pass

    def update_session_plot (self) -> None:

        if (self._session_plot_index == 0):
            self._update_histogram_plot()
        elif (self._session_plot_index == 1):
            self._update_recruitment_curve_plot()
        elif (self._session_plot_index == 2):
            self._update_unipolar_recruitment_curve_plot()

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
                3,
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
        self._session_widget.getPlotItem().setLabel('bottom', 'Stim Amp (mA)')
        self._session_widget.getPlotItem().setLabel('left', 'Pk Amp (µV)')

        if len(self._trials) == 0:
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

        for trial in self._trials:
            amp_key = round(trial.stimulation_amplitude_ma, 2)
            onset_idx = trial.onset_sample_index if trial.onset_sample_index >= 0 else MhRecruitmentCurveStage.bin_sample_count()

            n = len(trial.trial_data)
            t_ms = (np.arange(n) - onset_idx) * ms_per_sample
            emg  = np.abs(trial.trial_data)

            if m_end > m_start:
                m_mask = (t_ms >= m_start) & (t_ms <= m_end)
                if np.any(m_mask):
                    m_dict.setdefault(amp_key, []).append(float(np.max(emg[m_mask])))

            if h_end > h_start:
                h_mask = (t_ms >= h_start) & (t_ms <= h_end)
                if np.any(h_mask):
                    h_dict.setdefault(amp_key, []).append(float(np.max(emg[h_mask])))

        all_amps = sorted(set(m_dict.keys()) | set(h_dict.keys()))
        if len(all_amps) == 0:
            return

        positions = np.arange(len(all_amps), dtype=float)

        self._session_widget.addLegend(offset=(0, 0))

        if m_end > m_start and len(m_dict) > 0:
            m_means = np.array([np.mean(m_dict.get(a, [0])) for a in all_amps])
            m_sems  = np.array([float(np.std(m_dict[a]) / np.sqrt(len(m_dict[a]))) if len(m_dict.get(a, [])) > 1 else 0.0 for a in all_amps])
            m_pen = pg.mkPen(color=(0, 0, 200), width=2)
            self._session_widget.plot(positions - 0.1, m_means, pen=m_pen, symbol='o',
                                      symbolBrush=(0, 0, 200), symbolSize=6, name='M-wave')
            nonzero = m_sems > 0
            if np.any(nonzero):
                m_err = pg.ErrorBarItem(x=positions[nonzero] - 0.1, y=m_means[nonzero],
                                        top=m_sems[nonzero], bottom=m_sems[nonzero],
                                        pen=pg.mkPen(color=(0, 0, 200)))
                self._session_widget.addItem(m_err)

        if h_end > h_start and len(h_dict) > 0:
            h_means = np.array([np.mean(h_dict.get(a, [0])) for a in all_amps])
            h_sems  = np.array([float(np.std(h_dict[a]) / np.sqrt(len(h_dict[a]))) if len(h_dict.get(a, [])) > 1 else 0.0 for a in all_amps])
            h_pen = pg.mkPen(color=(0, 160, 0), width=2)
            self._session_widget.plot(positions + 0.1, h_means, pen=h_pen, symbol='o',
                                      symbolBrush=(0, 160, 0), symbolSize=6, name='H-wave')
            nonzero = h_sems > 0
            if np.any(nonzero):
                h_err = pg.ErrorBarItem(x=positions[nonzero] + 0.1, y=h_means[nonzero],
                                        top=h_sems[nonzero], bottom=h_sems[nonzero],
                                        pen=pg.mkPen(color=(0, 160, 0)))
                self._session_widget.addItem(h_err)

        #X-axis tick labels: show amplitude every 10 steps to avoid crowding
        tick_pairs = [(int(i), f'{a:.1f}' if i % 10 == 0 else '') for i, a in enumerate(all_amps)]
        self._session_widget.getPlotItem().getAxis('bottom').setTicks([tick_pairs])

    def _update_unipolar_recruitment_curve_plot (self) -> None:
        self._session_widget.clear()
        self._session_widget.getPlotItem().setLabel('bottom', 'Stim Amp (mA)')
        self._session_widget.getPlotItem().setLabel('left', 'Pk Amp (µV)')

        if len(self._trials) == 0:
            return

        m_start = self._user_m_wave_window_start_time_milliseconds
        m_end   = self._user_m_wave_window_end_time_milliseconds
        h_start = self._user_h_wave_window_start_time_milliseconds
        h_end   = self._user_h_wave_window_end_time_milliseconds

        if m_end <= m_start and h_end <= h_start:
            text = pg.TextItem("Set M/H wave windows to view unipolar recruitment curve", anchor=(0.5, 0.5))
            self._session_widget.addItem(text)
            return

        ms_per_sample = MhRecruitmentCurveStage.ms_per_sample()

        m_dict: dict = {}
        h_dict: dict = {}

        for trial in self._trials:
            #Skip any trial that was recorded before unipolar data was available
            if len(trial.unipolar_trial_data) <= 1:
                continue

            amp_key = round(trial.stimulation_amplitude_ma, 2)
            onset_idx = trial.onset_sample_index if trial.onset_sample_index >= 0 else MhRecruitmentCurveStage.bin_sample_count()

            n = len(trial.unipolar_trial_data)
            t_ms = (np.arange(n) - onset_idx) * ms_per_sample
            emg  = np.abs(trial.unipolar_trial_data)

            if m_end > m_start:
                m_mask = (t_ms >= m_start) & (t_ms <= m_end)
                if np.any(m_mask):
                    m_dict.setdefault(amp_key, []).append(float(np.max(emg[m_mask])))

            if h_end > h_start:
                h_mask = (t_ms >= h_start) & (t_ms <= h_end)
                if np.any(h_mask):
                    h_dict.setdefault(amp_key, []).append(float(np.max(emg[h_mask])))

        all_amps = sorted(set(m_dict.keys()) | set(h_dict.keys()))
        if len(all_amps) == 0:
            return

        positions = np.arange(len(all_amps), dtype=float)

        self._session_widget.addLegend(offset=(0, 0))

        #Light green dashed — M-wave unipolar
        if m_end > m_start and len(m_dict) > 0:
            m_means = np.array([np.mean(m_dict.get(a, [0])) for a in all_amps])
            m_sems  = np.array([float(np.std(m_dict[a]) / np.sqrt(len(m_dict[a]))) if len(m_dict.get(a, [])) > 1 else 0.0 for a in all_amps])
            m_color = (100, 210, 100)
            m_pen = pg.mkPen(color=m_color, width=2, style=QtCore.Qt.DashLine)
            self._session_widget.plot(positions - 0.1, m_means, pen=m_pen, symbol='o',
                                      symbolBrush=m_color, symbolSize=6, name='M-wave (unipolar)')
            nonzero = m_sems > 0
            if np.any(nonzero):
                m_err = pg.ErrorBarItem(x=positions[nonzero] - 0.1, y=m_means[nonzero],
                                        top=m_sems[nonzero], bottom=m_sems[nonzero],
                                        pen=pg.mkPen(color=m_color))
                self._session_widget.addItem(m_err)

        #Light blue dashed — H-wave unipolar
        if h_end > h_start and len(h_dict) > 0:
            h_means = np.array([np.mean(h_dict.get(a, [0])) for a in all_amps])
            h_sems  = np.array([float(np.std(h_dict[a]) / np.sqrt(len(h_dict[a]))) if len(h_dict.get(a, [])) > 1 else 0.0 for a in all_amps])
            h_color = (100, 180, 240)
            h_pen = pg.mkPen(color=h_color, width=2, style=QtCore.Qt.DashLine)
            self._session_widget.plot(positions + 0.1, h_means, pen=h_pen, symbol='o',
                                      symbolBrush=h_color, symbolSize=6, name='H-wave (unipolar)')
            nonzero = h_sems > 0
            if np.any(nonzero):
                h_err = pg.ErrorBarItem(x=positions[nonzero] + 0.1, y=h_means[nonzero],
                                        top=h_sems[nonzero], bottom=h_sems[nonzero],
                                        pen=pg.mkPen(color=h_color))
                self._session_widget.addItem(h_err)

        #X-axis tick labels: show amplitude every 10 steps to avoid crowding
        tick_pairs = [(int(i), f'{a:.1f}' if i % 10 == 0 else '') for i, a in enumerate(all_amps)]
        self._session_widget.getPlotItem().getAxis('bottom').setTicks([tick_pairs])

    def _update_histogram_plot (self) -> None:
        #Clear the plot
        self._session_widget.clear()
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

        emg_pen = pg.mkPen(color=(0, 0, 0), width=2.0)

        #-------------------------------------------------------------------
        # Phase 1: Detect true stim onset from the ADC sync line
        #
        # Search window design:
        #   - Never look in the pre-stim window (sync_data[0 : bin_sample_count()]).
        #     That data was captured BEFORE trigger_single() was called; any crossing
        #     there is noise and was causing the ±10–50 ms jitter the user observed.
        #   - Start at bin_sample_count() (the exact sample where post-trigger data
        #     begins), then push forward by the measured TCP + frame-delivery delay.
        #     The stim pulse physically cannot arrive before that delay elapses, so
        #     those leading samples are safe to skip.  Capped at 60 ms so a slow
        #     or bursty frame never hides a real onset.
        #   - Search 120 ms of post-trigger data — wide enough to catch the onset
        #     regardless of network / hardware jitter.
        #-------------------------------------------------------------------
        trigger_sample: int = MhRecruitmentCurveStage.bin_sample_count()

        #Search the full 120 ms of post-trigger data starting right at the trigger boundary.
        #Using frame-delivery delay to push search_start forward was causing missed detections
        #because timestamp_emitted reflects frame assembly completion, not stim hardware arrival.
        #The stim pulse (hardware latency << frame latency) lands in the early samples of the
        #first post-trigger frame, which the delay-based offset was skipping past.
        search_start: int = trigger_sample
        search_end: int = min(
            trigger_sample + int(500.0 / MhRecruitmentCurveStage.ms_per_sample()),
            len(self._current_trial.sync_data)
        )

        stim_onset_idx: int = trigger_sample  # fallback: start of post-trigger data
        onset_was_detected: bool = False

        if len(self._current_trial.sync_data) > search_start:
            search_window = self._current_trial.sync_data[search_start:search_end]
            above = np.where(search_window >= MhRecruitmentCurveStage.STIM_ONSET_THRESHOLD)[0]
            if len(above) > 0:
                #Prefer the first pair of consecutive samples above threshold to reject
                #single-sample noise transients.  Fall back to first isolated crossing
                #at low sample rates where the pulse fits in one sample.
                chosen: int = int(above[0])
                for i in range(len(above) - 1):
                    if above[i + 1] == above[i] + 1:
                        chosen = int(above[i])
                        break
                stim_onset_idx = search_start + chosen
                onset_was_detected = True

        print(
            f"[DEBUG] onset search | "
            f"trigger_sample={trigger_sample} | "
            f"search=[{search_start}, {search_end}] | "
            f"stim_onset_idx={stim_onset_idx} | "
            f"onset_detected={onset_was_detected} | "
            f"sync_data_len={len(self._current_trial.sync_data)}"
        )

        if not onset_was_detected:
            fail_msg: str = (
                f"[FAILED STIM TRIAL] No ADC pulse detected above {MhRecruitmentCurveStage.STIM_ONSET_THRESHOLD} V "
                f"in search window [{search_start}, {search_end}] — "
                f"plot zeroed at trigger boundary (sample {trigger_sample}). "
                f"Check stimulator connection and ADC sync cable."
            )
            print(fail_msg)
            self.signals.new_message.emit(SessionMessage(fail_msg))

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

        #Prepare the unipolar data slice (same window indices as the bipolar slice)
        has_unipolar: bool = len(self._current_trial.unipolar_trial_data) > 1
        unipolar_data_to_plot: np.ndarray = None
        if has_unipolar and len(self._current_trial.unipolar_trial_data) >= idx_post:
            unipolar_data_to_plot = self._current_trial.unipolar_trial_data[idx_pre:idx_post]

        #Plot |unipolar EMG| abs trace in grayish-purple (index 4), drawn before other traces
        if (unipolar_data_to_plot is not None and
                len(self._trial_signal_flags) > 4 and self._trial_signal_flags[4]):
            unipolar_abs_pen = pg.mkPen(color=(148, 130, 172), width=1.5)
            self._trial_widget.plot(x_data, np.abs(unipolar_data_to_plot), pen=unipolar_abs_pen)

        #Plot unipolar filtered signal in orange (index 3)
        if (unipolar_data_to_plot is not None and
                len(self._trial_signal_flags) > 3 and self._trial_signal_flags[3]):
            unipolar_pen = pg.mkPen(color=(255, 140, 0), width=1.5)
            self._trial_widget.plot(x_data, unipolar_data_to_plot, pen=unipolar_pen)

        #Plot |EMG| abs trace in gray (index 2), drawn before bipolar signal so it renders underneath
        if len(self._trial_signal_flags) > 2 and self._trial_signal_flags[2]:
            abs_pen = pg.mkPen(color=(150, 150, 150), width=1.5)
            self._trial_widget.plot(x_data, np.abs(data_to_plot), pen=abs_pen)

        #Plot EMG signal (bipolar/differential filtered) in black (index 0)
        if len(self._trial_signal_flags) == 0 or self._trial_signal_flags[0]:
            self._trial_widget.plot(x_data, data_to_plot, pen=emg_pen)

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
        # Phase 5: Optional ADC sync overlay on a secondary (right) Y-axis
        #-------------------------------------------------------------------
        show_adc: bool = len(self._trial_signal_flags) > 1 and self._trial_signal_flags[1]
        if show_adc and len(self._current_trial.sync_data) >= idx_post:
            adc_to_plot = self._current_trial.sync_data[idx_pre:idx_post]
            if len(adc_to_plot) == num_samples:
                self._draw_trial_adc_overlay(x_data, adc_to_plot)

        #-------------------------------------------------------------------
        # Phase 6: Peak markers (star) at M-wave and H-wave detection points
        #-------------------------------------------------------------------
        # M-wave peak marker — blue star at the sample used for recruitment curve
        if m_end > m_start:
            m_mask = (x_data >= m_start) & (x_data <= m_end)
            if np.any(m_mask):
                m_masked_indices = np.where(m_mask)[0]
                peak_local_idx = int(np.argmax(np.abs(data_to_plot[m_mask])))
                peak_global_idx = m_masked_indices[peak_local_idx]
                peak_x = float(x_data[peak_global_idx])
                peak_y = float(data_to_plot[peak_global_idx])
                m_marker = pg.ScatterPlotItem(
                    x=[peak_x], y=[peak_y],
                    symbol='star', size=14,
                    pen=pg.mkPen(color=(0, 0, 200), width=1.5),
                    brush=pg.mkBrush(0, 0, 200, 200)
                )
                self._trial_widget.addItem(m_marker)

        # H-wave peak marker — green star at the sample used for recruitment curve
        if h_end > h_start:
            h_mask = (x_data >= h_start) & (x_data <= h_end)
            if np.any(h_mask):
                h_masked_indices = np.where(h_mask)[0]
                peak_local_idx = int(np.argmax(np.abs(data_to_plot[h_mask])))
                peak_global_idx = h_masked_indices[peak_local_idx]
                peak_x = float(x_data[peak_global_idx])
                peak_y = float(data_to_plot[peak_global_idx])
                h_marker = pg.ScatterPlotItem(
                    x=[peak_x], y=[peak_y],
                    symbol='star', size=14,
                    pen=pg.mkPen(color=(0, 160, 0), width=1.5),
                    brush=pg.mkBrush(0, 160, 0, 200)
                )
                self._trial_widget.addItem(h_marker)

        pass

    def _cleanup_trial_adc_viewbox (self) -> None:
        '''Removes the secondary ADC ViewBox from the scene before a redraw.'''
        if self._trial_adc_viewbox is not None:
            try:
                scene = self._trial_adc_viewbox.scene()
                if scene is not None:
                    scene.removeItem(self._trial_adc_viewbox)
            except Exception:
                pass
            self._trial_adc_viewbox = None

    def _draw_trial_adc_overlay (self, x_data: np.ndarray, adc_data: np.ndarray) -> None:
        '''
        Adds the ADC sync signal as a green trace on an independent right-hand Y-axis.
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
        plot_item.getAxis('right').setLabel('ADC In (V)', color='#007700')
        plot_item.getAxis('right').linkToView(adc_vb)

        #Link X and set initial geometry
        adc_vb.setXLink(plot_item)
        adc_vb.setGeometry(plot_item.vb.sceneBoundingRect())

        #Keep geometry in sync when the main view is resized
        plot_item.vb.sigResized.connect(self._on_trial_plot_resized)

        #Plot the ADC curve inside the secondary ViewBox
        adc_pen = pg.mkPen(color=(0, 160, 0), width=1.5)
        adc_vb.addItem(pg.PlotCurveItem(x_data, adc_data, pen=adc_pen))

        self._trial_adc_viewbox = adc_vb

    def _on_trial_plot_resized (self) -> None:
        '''Keeps the ADC ViewBox geometry aligned with the main plot after a resize.'''
        if self._trial_adc_viewbox is not None and self._trial_adc_viewbox.scene() is not None:
            self._trial_adc_viewbox.setGeometry(
                self._trial_widget.getPlotItem().vb.sceneBoundingRect()
            )

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
        Runs the same onset/end detection used by _update_trial_plot and writes the
        results — plus the frame-timing accumulators — into the trial object so they
        are persisted to the data file for offline analysis.
        '''
        #Carry over the frame-timing accumulators collected during RECORD state
        trial.trigger_wall_time_ms = self._trigger_wall_time_ms
        trial.n_pre_trigger_frames_discarded = self._n_pre_trigger_frames_discarded
        trial.frame_received_timestamps_ms = np.array(
            self._frame_received_timestamps_ms, dtype=np.uint64)
        trial.first_post_trigger_frame_sample_id = self._first_post_trigger_frame_sample_id

        #Onset search boundaries (mirrors _update_trial_plot Phase 1)
        search_start: int = MhRecruitmentCurveStage.bin_sample_count()
        search_end: int = min(
            search_start + MhRecruitmentCurveStage.trial_recording_sample_count(),
            len(trial.sync_data)
        )

        #Defaults when no onset is found
        trial.onset_sample_index = -1
        trial.onset_detected = 0
        trial.stim_end_sample_index = -1
        trial.stim_duration_samples = 0
        trial.stim_duration_ms = 0.0
        trial.sync_peak_voltage = 0.0

        if len(trial.sync_data) > search_start:
            search_window = trial.sync_data[search_start:search_end]
            trial.sync_peak_voltage = float(np.max(search_window)) if len(search_window) > 0 else 0.0

            above = np.where(search_window >= MhRecruitmentCurveStage.STIM_ONSET_THRESHOLD)[0]
            if len(above) > 0:
                #Prefer first consecutive pair to reject single-sample noise (same logic
                #as _update_trial_plot); fall back to first isolated crossing.
                chosen: int = int(above[0])
                for i in range(len(above) - 1):
                    if above[i + 1] == above[i] + 1:
                        chosen = int(above[i])
                        break

                trial.onset_sample_index = search_start + chosen
                trial.onset_detected = 1

                #End detection
                if len(trial.sync_data) > trial.onset_sample_index:
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

        #Record the trigger wall-clock time before firing so the RECORD state
        #can discard any queued pre-trigger frames (same fix as the automatic path).
        self._trigger_wall_time_ms = int(datetime.now().timestamp() * 1000)

        #Trigger the stimulator
        if (ApplicationConfiguration.stimulator is not None):
            ApplicationConfiguration.stimulator.trigger_single()

        #Notify listeners (e.g. live EMG overlay) that a stim was just sent
        self.signals.stim_triggered.emit()

        self.signals.new_message.emit(SessionMessage(f"Manual stim triggered at {self._current_stimulation_amplitude_ma:.2f} mA"))

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