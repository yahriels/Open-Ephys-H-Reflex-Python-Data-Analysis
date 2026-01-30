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
        self.monitored_signal_sample_count = int(self.monitored_signal_duration_seconds * Stage.SAMPLE_RATE)

        #Get the number of bins we will be collecting
        bin_count: int = int(dur_milliseconds / MhRecruitmentCurveStage.BIN_DURATION_MILLISECONDS)

        #Re-size the appropriate arrays to hold the data we care about
        self.monitored_signal = np.zeros(self.monitored_signal_sample_count)
        self.monitored_signal_abs = np.zeros(self.monitored_signal_sample_count)
        self.bins = np.zeros(bin_count)

        #We are done. return from this function.
        return
    
    def process (self, data_frame: OpenEphysDataFrame, current_initiation_min: float, current_initiation_max: float) -> bool:
        should_initiate_trial: bool = False

        #Add the number of samples we are pulling in to the current trial sample count
        self.current_monitored_signal_sample_count += len(data_frame.diff_data_block)

        #Add the new data to the monitored signal
        self.monitored_signal = np.concatenate([self.monitored_signal, data_frame.diff_data_block])
        self.monitored_signal_abs = np.concatenate([self.monitored_signal_abs, data_frame.abs_data_block])
        elements_to_remove: int = len(self.monitored_signal) - self.monitored_signal_sample_count
        if (elements_to_remove > 0):
            self.monitored_signal = self.monitored_signal[elements_to_remove:]
            self.monitored_signal_abs = self.monitored_signal_abs[elements_to_remove:]

        #Bin the data
        for bin_index in range(0, len(self.bins)):
            bin_start = MhRecruitmentCurveStage.BIN_SAMPLE_COUNT * bin_index
            bin_end = MhRecruitmentCurveStage.BIN_SAMPLE_COUNT * (bin_index + 1)

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

    #This defines the number of samples for an individual bin
    #This is: (sample rate / 1000) * BIN_DURATION_MILLISECONDS
    BIN_SAMPLE_COUNT: int = 250

    #This defines the trial recording duration in milliseconds
    TRIAL_RECORDING_DURATION_MILLISECONDS: int = 100

    #This defines the trial recording duration in number of samples
    TRIAL_RECORDING_DURATION_SAMPLE_COUNT: int = 500

    #This defines the number of milliseconds per sample
    MILLISECONDS_PER_SAMPLE: float = 0.2

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

        #Create a variable that will store the user-defined maximum stimulation amplitude
        self._user_max_stimulation_amplitude: float = MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MAX

        #Create a variable that will store the user-defined stimulation amplitude step-size
        self._user_stimulation_amplitude_step_size: float = MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_STEP

        #Create a variable that will store the user-defined minimum stimulation interval
        self._user_minimum_stimulation_interval_milliseconds: int = MhRecruitmentCurveStage.MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS

        #Create variables that will store the user-defined window start/end times for the M wave and the H wave
        self._user_m_wave_window_start_time_milliseconds: float = 0
        self._user_m_wave_window_end_time_milliseconds: float = 0
        self._user_h_wave_window_start_time_milliseconds: float = 0
        self._user_h_wave_window_end_time_milliseconds: float = 0

        

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
            return (False, "No EMG characterization data was found for this subject. Please run stage S1 before running this stage.")

        #If we reach this point in the code, then we have an existing HRS1 file for this animal.
        #Now let's check to see if there is already an existing HRS2 file for this animal.
        for f in files_list:
            if (f.endswith("hrs2")):
                return (False, "This subject has already completed this stage. EMG sweep data exists for this animal. This stage cannot proceed. If this is an issue, please talk to your PI.")

        #If we reach this point in the code, then no prior EMG sweep data exists for this animal.
        #Therefore, this stage can proceed.

        #Load in the EMG characterization data from stage 1
        self._emg_characterization_data: EmgCharacterizationDataFile = EmgCharacterizationDataFile()
        fid = open(os.path.join(file_path, hrs1_file_name), "rb")
        self._emg_characterization_data.read(fid)
        fid.close()

        #Calculate the histogram data from the EMG characterization data from stage 1
        self._emg_histogram_data: EmgHistogramData = self._emg_characterization_data.get_histogram_data()

        #Set some values related to using the histogram data during this stage
        self._current_min_initiation_threshold = self._emg_histogram_data.min
        self._current_max_initiation_threshold = self._emg_histogram_data.max

        #Update the histogram plot
        self._update_histogram_plot()

        #Open a file for saving data for this stage
        self._fid = open(os.path.join(file_path, file_name), "wb")

        #Save the file header for this session's data file
        self._save_file_header()

        #Display a message to the user with some information about this stage
        commands_messages: list[str] = [
            "This stage supports the following commands: ",
            "lb = x, lb += x, lb -= x (Set the init threshold lower bound)",
            "ub = x, ub += x, ub -= x (Set the init threshold upper bound)",
            "auto on, auto off (Turn on/off the automated algorithm for determining the lower and upper bounds of the initiation threshold)"
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

        data = data_frame.diff_data_block

        if (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_NOT_SETUP):
            #Set things up for a new trial
            self._setup_new_trial()

            #Pop the first stimulation amplitude from the list of stim amplitudes
            self._current_stimulation_amplitude_ma = self._stimulation_amplitudes[0]
            self._stimulation_amplitudes = self._stimulation_amplitudes[1:]

            #Set the stimulation parameters on the Model 4100 for the upcoming trial
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
                if (elapsed_ms < MhRecruitmentCurveStage.MINIMUM_INTERTRIAL_INTERVAL_MILLISECONDS):
                    return

            #Check to see if we should initiate a trial
            should_initiate_trial: bool = self._check_for_trial_initiation(data_frame)
            if (should_initiate_trial):
                #If it is determined that we should initiate a trial...

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
                self._current_trial.trial_data = self._current_trial_initiation_data.monitored_signal[-MhRecruitmentCurveStage.BIN_SAMPLE_COUNT:]

                #Trigger the Model 4100
                if (ApplicationConfiguration.stimulator is not None):
                    ApplicationConfiguration.stimulator.trigger_single()

        elif (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_RECORD):

            #Copy data into the trial object until we have 100 ms of post-stim data
            self._current_trial.trial_data = np.concatenate([self._current_trial.trial_data, data])

            #Check to see if we have enough data
            if (len(self._current_trial.trial_data) >= (MhRecruitmentCurveStage.BIN_SAMPLE_COUNT + MhRecruitmentCurveStage.TRIAL_RECORDING_DURATION_SAMPLE_COUNT)):
                #If so, move on to the next stage
                self._current_trial_state = MhRecruitmentCurveStage.TRIAL_STATE_FINALIZE

        elif (self._current_trial_state == MhRecruitmentCurveStage.TRIAL_STATE_FINALIZE):
            
            #Append the current trial to the session's list of trials
            self._trials.append(self._current_trial)

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
            ## TO DO
            self._update_trial_plot()

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

        if (user_input.startswith("lb")) or (user_input.startswith("ub")):
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
    
    def get_session_plot_options (self) -> list[str]:
        return ["S1 Histogram", "Recruitment Curve"]

    def update_trial_plot (self) -> None:
        #This stage does not support updating the "most recent trial plot" from an external function call.
        pass

    def update_session_plot (self) -> None:

        if (self._session_plot_index == 0):
            self._update_histogram_plot()
        elif (self._session_plot_index == 1):
            self._update_recruitment_curve_plot()

        pass

    #endregion

    #region Private methods

    def _save_file_header (self) -> None:
        if (self._fid is not None):

            #Create a file header object
            header: MhRecruitmentCurveHeader = MhRecruitmentCurveHeader(
                0,
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
        pass

    def _update_histogram_plot (self) -> None:
        #Clear the plot
        self._session_widget.clear()

        #Plot a histogram of the grand means from the EMG characterization data
        for i in range(0, len(self._emg_histogram_data.histogram_values)):
            hist_val: float = self._emg_histogram_data.histogram_values[i]
            hist_edge_01: float = self._emg_histogram_data.histogram_bin_edges[i]
            hist_edge_02: float = self._emg_histogram_data.histogram_bin_edges[i+1]
            hist_center: float = (hist_edge_01 + hist_edge_02) / 2.0
            hist_width: float = hist_edge_02 - hist_edge_01
            hist_bar: pg.BarGraphItem = pg.BarGraphItem(x = hist_center, height = hist_val, width = hist_width)
            self._session_widget.addItem(hist_bar)
        
        #Plot vertical lines where the min and max thresholds are
        vert_line_pen = pg.mkPen(color=(255, 0, 0), width = 2.0, style = QtCore.Qt.DashLine)
        min_thresh_line: pg.InfiniteLine = pg.InfiniteLine(self._current_min_initiation_threshold, 90, vert_line_pen, movable=False)
        max_thresh_line: pg.InfiniteLine = pg.InfiniteLine(self._current_max_initiation_threshold, 90, vert_line_pen, movable=False)
        self._session_widget.addItem(min_thresh_line)
        self._session_widget.addItem(max_thresh_line)

    def _update_trial_plot (self) -> None:
        #Clear the plot
        self._trial_widget.clear()

        #Plot the EMG signal for this trial
        pen = pg.mkPen(color=(0, 0, 0), width = 2.0)

        #Calculate how many samples correspond to 5ms pre-stim and 20ms post-stim
        #This gives us a 25ms window to display
        pre_stim_samples = int(5.0 / MhRecruitmentCurveStage.MILLISECONDS_PER_SAMPLE)  # 25 samples for 5ms
        post_stim_samples = int(20.0 / MhRecruitmentCurveStage.MILLISECONDS_PER_SAMPLE)  # 100 samples for 20ms
        total_display_samples = pre_stim_samples + post_stim_samples  # 125 samples total

        #Extract only the first 25ms of data to display (5ms pre + 20ms post)
        data_to_plot = self._current_trial.trial_data[:total_display_samples]

        #Transform each sample index into a millisecond time value for the trial's x-axis
        #Time axis goes from -5ms (pre-stim) to +20ms (post-stim), with stim event at 0ms
        num_samples = len(data_to_plot)
        x_data = (np.arange(0, num_samples) * MhRecruitmentCurveStage.MILLISECONDS_PER_SAMPLE) - 5.0

        #Plot the trial data
        self._trial_widget.plot(x_data, data_to_plot, pen = pen)

        #Set the x-axis range to -5ms to +20ms
        self._trial_widget.setXRange(-5, 20, padding=0)

        #Plot a vertical line annotation showing where the stimulation event occurred (at time 0)
        vert_line_pen = pg.mkPen(color=(255, 0, 0), width = 2.0, style=QtCore.Qt.DashLine)
        vert_line: pg.InfiniteLine = pg.InfiniteLine(0, 90, vert_line_pen, movable=False)
        self._trial_widget.addItem(vert_line)

        pass

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
            #Generate a list of stimulation amplitudes
            self._stimulation_amplitudes = np.arange(
                MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MIN,
                MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_MAX + MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_STEP,
                MhRecruitmentCurveStage.STIMULATION_AMPLITUDE_STEP, dtype=np.float64)
            
            #Shuffle the list
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
            
            self._current_min_initiation_threshold = max(lb, self._emg_histogram_data.min)
            self.signals.new_message.emit(SessionMessage(f"Min threshold set: {self._current_min_initiation_threshold:.2f}"))
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
            
            self._current_max_initiation_threshold = min(ub, self._emg_histogram_data.max)
            self.signals.new_message.emit(SessionMessage(f"Min threshold set: {self._current_max_initiation_threshold:.2f}"))
            self.update_session_plot()
        pass

    def _parse_command_auto (self, user_input: str) -> None:
        on_or_off: str = user_input.removeprefix("auto")
        if (on_or_off == "on"):
            self._auto_thresholding_enabled = True
            self.signals.new_message.emit(SessionMessage("Auto thresholding: ENABLED"))
        else:
            self._auto_thresholding_enabled = False
            self.signals.new_message.emit(SessionMessage("Auto thresholding: DISABLED"))
        
    #endregion