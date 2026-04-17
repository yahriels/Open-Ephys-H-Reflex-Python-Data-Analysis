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
from ..datafiles.emg_characterization_data_file import EmgCharacterizationDataFile, EmgCharacterizationHeader, EmgCharacterizationTrial, EmgTrialsPerHourData
from ..datafiles.h_reflex_data_file_shared import HReflexDataFileEmgData
from ..open_ephys_streamer import OpenEphysDataFrame

class EmgCharacterizationStage (Stage):

    #region Constants

    #This defines the duration of an individual bin in miliseconds
    BIN_DURATION_MILLISECONDS: int = 50

    #The minimum duration for which we will scan for trial initiation criteria to be met
    TRIAL_INITIATION_PHASE_MIN_DURATION_MILLISECONDS: int = 2200

    #The maximum duration for which we will scan for trial initiation criteria to be met
    TRIAL_INITIATION_PHASE_MAX_DURATION_MILLISECONDS: int = 2700

    #The minimum and maximum range for trial initiation
    TRIAL_INITIATION_MIN_RANGE_MICROVOLTS: float = 15.0
    TRIAL_INITIATION_MAX_RANGE_MICROVOLTS: float = 300.0

    #endregion

    #region Classmethods

    @classmethod
    def bin_sample_count(cls) -> int:
        '''Number of samples per bin, computed from the live Open Ephys sample rate.'''
        return int(cls.BIN_DURATION_MILLISECONDS * ApplicationConfiguration.sample_rate / 1000)

    #endregion

    #region Constructor

    def __init__(self):
        super().__init__()

        #Set the basic stage information
        self.stage_name = "S1"
        self.stage_description = "EMG Characterization"
        self.stage_type = Stage.STAGE_TYPE_EMG_CHARACTERIZATION

        #Declare a variable to hold the monitored signal
        self._monitored_signal: np.ndarray = np.zeros(1, dtype=np.float32)

        #Declare a variable to hold the open ephys timestamps associated with each sample in the monitored signal
        self._monitored_signal_open_ephys_millis: np.ndarray = np.zeros(1, dtype=np.uint64)

        #Declare a variable to hold the open ephys sample id associated with each sample in the monitored signal
        self._monitored_signal_open_ephys_sample_ids: np.ndarray = np.zeros(1, dtype=np.uint64)

        #Declare a variable to hold the bins
        self._bins: np.ndarray = np.zeros(1, dtype=np.float32)

        #Declare a variable to hold the current monitored signal duration
        self._monitored_signal_duration_seconds: float = 0.0

        #Declare a variable to hold the number of samples that we will 
        #store in the monitored signal
        self._monitored_signal_sample_count: int = 0

        #Declare a variable that helps us determine whether we have streamed enough samples
        self._current_trial_sample_count: int = 0

        #Declare a variable to hold the total number of samples streamed during the whole session
        self._current_session_sample_count: int = 0

        #Declare a variable to hold the trial state
        self._is_trial_set_up: bool = False

        #Declare a variable to hold the mean of each trial
        self._trial_means: list[float] = []

        #Declare a variable to hold the session start time (used for trials-per-hour calculation)
        self._session_start_time: datetime = None

        #Declare a variable to hold the grand mean of the most recently completed trial
        self._last_bin_grand_mean: float = 0.0

        #Initiation threshold bounds (settable at runtime via set_initiation_thresholds)
        self._init_min_threshold: float = EmgCharacterizationStage.TRIAL_INITIATION_MIN_RANGE_MICROVOLTS
        self._init_max_threshold: float = EmgCharacterizationStage.TRIAL_INITIATION_MAX_RANGE_MICROVOLTS

        #Instantiate a random-number generator and use the current time as a seed
        self._rng: Random = Random(datetime.now().timestamp())

        #Create a private variable that will be used to store a save-file handle
        self._fid: BinaryIO = None

    #endregion

    #region Overrides

    def initialize (self, subject_id: str) -> tuple[bool, str]:
        #Set the subject id
        self._subject_id: str = subject_id

        #Set the "is trial set up" flag to false
        self._is_trial_set_up = False

        #Set the current trial sample count to 0
        self._current_trial_sample_count = 0

        #Clear the list of trial means
        self._trial_means.clear()

        #Record the session start time
        self._session_start_time = datetime.now()

        #Get the current datetime
        current_datetime: datetime = datetime.now()

        #Define the path where we will save data
        app_data_path: str = user_data_dir(ApplicationConfiguration.appname, ApplicationConfiguration.appauthor)
        file_path: str = os.path.join(app_data_path, "Data")
        file_path = os.path.join(file_path, self._subject_id)

        #Define a file name for the file to which we will save data
        file_timestamp: str = current_datetime.strftime("%Y%m%dT%H%M%S")
        file_name: str = f"{self._subject_id}_{file_timestamp}.hrs1"

        #Create the folder if it does not yet exist
        if (not os.path.exists(file_path)):
            os.makedirs(file_path)
        
        #Check to see if there is an existing hrs1 file for this subject.
        files_list: list[str] = os.listdir(file_path)
        for f in files_list:
            if (f.endswith("hrs1")):
                #If we find an existing hrs1 file for this subject, then it means this subject has already completed
                #this stage. Let's return False with an informative message to the user.
                return (False, "This subject has already completed this stage. EMG Characterization data exists for this subject. This stage cannot proceed. If this is an issue, please talk to your PI.")

        #If we reach this point in the code, then no pre-existing hrs1 file exists for this subject, so we may proceed.

        #Open a file for saving data
        self._fid = open(os.path.join(file_path, file_name), "wb")

        #Save the file header for this data file
        self._save_file_header()

        #Return from this function
        return (True, "")

    def process (self, data_frame: OpenEphysDataFrame) -> None:
        '''
        Processes the most recent incoming data and takes any actions
        that are necessary based on the incoming data.
        '''

        #Check to see if we need to set up a new trial
        if (not self._is_trial_set_up):
            #Set up a new trial
            self._setup_new_trial()

        #Now let's proceed
        if (self._is_trial_set_up):
            #Get the length of the incoming data block
            data_block_len: int = len(data_frame.diff_data_block)

            #Get the Open Ephys millis timestamp (the timestamp at which this data was sent to our app by Open Ephys)
            data_block_open_ephys_millis: np.ndarray = np.full(data_block_len, data_frame.timestamp, dtype=np.uint64)

            #Get the Open Ephys sample id
            #(this identifies the FIRST sample in the data block with a specific sample in the Open Ephys recording)
            data_block_open_ephys_sample_id: np.ndarray = np.arange(data_frame.sample_id, data_frame.sample_id + data_block_len, dtype=np.uint64)

            #Add the number of samples we are pulling in to the current session sample count
            self._current_session_sample_count += data_block_len

            #Add the number of samples we are pulling in to the current trial sample count
            self._current_trial_sample_count += data_block_len

            #Pull in new data to the monitored signal
            self._monitored_signal = np.concatenate([self._monitored_signal, data_frame.abs_data_block])
            self._monitored_signal_open_ephys_millis = np.concatenate([self._monitored_signal_open_ephys_millis, data_block_open_ephys_millis])
            self._monitored_signal_open_ephys_sample_ids = np.concatenate([self._monitored_signal_open_ephys_sample_ids, data_block_open_ephys_sample_id])

            #Throw out old values from the monitored signal
            elements_to_remove: int = len(self._monitored_signal) - self._monitored_signal_sample_count
            if (elements_to_remove > 0):
                self._monitored_signal = self._monitored_signal[elements_to_remove:]
                self._monitored_signal_open_ephys_millis = self._monitored_signal_open_ephys_millis[elements_to_remove:]
                self._monitored_signal_open_ephys_sample_ids = self._monitored_signal_open_ephys_sample_ids[elements_to_remove:]
            
            #Bin the data
            for bin_index in range(0, len(self._bins)):
                bin_start = EmgCharacterizationStage.bin_sample_count() * bin_index
                bin_end = EmgCharacterizationStage.bin_sample_count() * (bin_index + 1)

                if (bin_end > len(self._monitored_signal)):
                    bin_end = len(self._monitored_signal)

                bin_mean: float = np.mean(self._monitored_signal[bin_start:bin_end])
                self._bins[bin_index] = bin_mean
            
            #Get the mean of all the bins
            bin_grand_mean: float = np.mean(self._bins)

            #If the bin grand mean is within a pre-specified min or max range, then
            #we consider this a trial initiation.
            if ((self._current_trial_sample_count >= self._monitored_signal_sample_count) and
                (bin_grand_mean >= self._init_min_threshold) and
                (bin_grand_mean <= self._init_max_threshold)):

                #A trial has been initiatied...

                #Add the grand mean to the list of trial means
                self._trial_means.append(bin_grand_mean)

                #Store for redraw when the user switches the trial plot dropdown
                self._last_bin_grand_mean = bin_grand_mean

                #Save the trial to the data file
                self._save_trial(bin_grand_mean)

                #Update the session plot
                self.update_session_plot()

                #Update the trial plot
                self.update_trial_plot()

                #Let's create a message object
                message: SessionMessage = SessionMessage(f"Trial {len(self._trial_means)} initiated")
                self.signals.new_message.emit(message)

                #Reset the trial-is-set-up flag
                self._is_trial_set_up = False

                #Reset the current trial sample count
                self._current_trial_sample_count = 0

                pass

        #Now save out the EMG data frame
        self._save_emg_data_frame(data_frame)

        return

    def finalize (self) -> None:
        if (self._fid is not None):
            #Compute and save the trials-per-hour vs window-size curve before closing
            self._save_trials_per_hour_data()

            #Close the data file for this session
            self._fid.close()

    def get_trial_plot_options (self) -> list[str]:
        return ["Histogram", "Most recent trial"]

    def get_session_plot_options (self) -> list[str]:
        return ["Session history", "Trials per hour"]

    def update_trial_plot (self) -> None:
        if self._trial_plot_index == 0:
            self._update_histogram_plot()
        else:
            self._update_most_recent_trial_plot()

    def update_session_plot (self) -> None:
        if self._session_plot_index == 0:
            self._update_session_plot()
        else:
            self._update_trials_per_hour_plot()

    def set_initiation_thresholds (self, lb: float, ub: float) -> str:
        self._init_min_threshold = lb
        self._init_max_threshold = ub
        return f"EMG Characterization thresholds set: lb={lb:.2f} µV, ub={ub:.2f} µV"

    #endregion

    #region Private methods

    def _setup_new_trial (self) -> None:
        #Choose a trial-initiation monitoring duration
        dur_milliseconds: int = self._rng.randint(
            EmgCharacterizationStage.TRIAL_INITIATION_PHASE_MIN_DURATION_MILLISECONDS,
            EmgCharacterizationStage.TRIAL_INITIATION_PHASE_MAX_DURATION_MILLISECONDS
        )

        #Round the number to the nearest 50-ms
        dur_milliseconds = self._round_special(dur_milliseconds, 50)

        #Convert to seconds
        self._monitored_signal_duration_seconds = float(dur_milliseconds) / 1000.0

        #Set the number of samples that we care about
        self._monitored_signal_sample_count = int(self._monitored_signal_duration_seconds * ApplicationConfiguration.sample_rate)

        #Get the number of bins we will be collecting
        bin_count: int = int(dur_milliseconds / EmgCharacterizationStage.BIN_DURATION_MILLISECONDS)

        #Re-size the appropriate arrays to hold the data we care about
        self._monitored_signal = np.zeros(self._monitored_signal_sample_count, dtype=np.float32)
        self._bins = np.zeros(bin_count, dtype=np.float32)
        self._monitored_signal_open_ephys_millis = np.zeros(self._monitored_signal_sample_count, dtype=np.uint64)
        self._monitored_signal_open_ephys_sample_ids = np.zeros(self._monitored_signal_sample_count, dtype=np.uint64)

        #Set the flag indicating that the trial has been set up
        self._is_trial_set_up = True

        #We are done. return from this function.
        return

    def _round_special (self, x: int, base: int = 50) -> int:
        return base * int(round(float(x) / float(base)))

    def _update_session_plot (self) -> None:

        #Clear the plot
        self._session_widget.clear()
        self._session_widget.getPlotItem().setLabel('bottom', 'Trial #')
        self._session_widget.getPlotItem().setLabel('left', 'EMG Mean (µV)')

        #Plot the trial means
        self._session_widget.plot(range(0, len(self._trial_means)), self._trial_means, pen = None, symbol = 'o', symbolBrush=('b'), symbolSize=12)

        pass

    def _update_histogram_plot (self) -> None:
        self._trial_widget.clear()
        self._trial_widget.getPlotItem().setLabel('bottom', 'Grand mean (µV)')
        self._trial_widget.getPlotItem().setLabel('left', 'Count')

        if len(self._trial_means) < 2:
            return

        means_arr = np.array(self._trial_means)
        n = len(means_arr)
        mn = float(np.min(means_arr))
        mx = float(np.max(means_arr))
        q25, q50, q75 = np.percentile(means_arr, [25, 50, 75])

        # Build bins: 1pct step, enforce minimum 100 bins (matches notebook logic)
        q1_val = float(np.quantile(means_arr, 0.01)) if n > 1 else mn
        step = q1_val - mn
        if step <= 0:
            step = (mx - mn) / 100.0 if (mx - mn) > 0 else max(0.1, abs(mx) * 0.01 if mx != 0 else 0.1)
        edges = np.arange(mn - 0.5 * step, mx + 1.5 * step, step)
        if edges.size < 2:
            edges = np.linspace(mn - 0.5, mx + 0.5, 101)
        hist_vals, hist_edges = np.histogram(means_arr, bins=edges)
        if hist_vals.size < 100:
            hist_vals, hist_edges = np.histogram(means_arr, bins=100, range=(mn, mx))

        # Bars — steelblue fill, black outline
        centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
        widths  = hist_edges[1:] - hist_edges[:-1]
        bar_item = pg.BarGraphItem(
            x=centers, height=hist_vals, width=widths,
            brush=(100, 149, 237), pen=pg.mkPen('k', width=0.5)
        )
        self._trial_widget.addItem(bar_item)

        # Min / max — black dashed vertical lines
        for val in [mn, mx]:
            self._trial_widget.addItem(pg.InfiniteLine(
                pos=val, angle=90, movable=False,
                pen=pg.mkPen(color=(0, 0, 0), width=1, style=QtCore.Qt.DashLine)
            ))

        # Q1 / median / Q3 — orange dotted, purple dash-dot, orange dotted with value labels
        orange = (255, 165, 0)
        purple = (148, 0, 211)
        quartile_specs = [
            (q25, orange, QtCore.Qt.DotLine,     f'Q1={q25:.2f}'),
            (q50, purple, QtCore.Qt.DashDotLine, f'Med={q50:.2f}'),
            (q75, orange, QtCore.Qt.DotLine,     f'Q3={q75:.2f}'),
        ]
        for val, color, style, lbl in quartile_specs:
            self._trial_widget.addItem(pg.InfiniteLine(
                pos=val, angle=90, movable=False,
                pen=pg.mkPen(color=color, width=1.5, style=style),
                label=lbl,
                labelOpts={'position': 0.85, 'color': color}
            ))

        self._trial_widget.getPlotItem().setTitle(
            f'Histogram | n={n:,} | bins={len(hist_vals)}'
        )

        pass

    def _update_most_recent_trial_plot (self) -> None:
        self._trial_widget.clear()
        self._trial_widget.getPlotItem().setLabel('bottom', 'Bin #')
        self._trial_widget.getPlotItem().setLabel('left', 'EMG (µV)')
        self._trial_widget.getPlotItem().setTitle('Most Recent Trial')

        if len(self._trial_means) == 0:
            return

        #Bar chart — one bar per bin
        x_centers = np.arange(len(self._bins), dtype=float)
        bar_item = pg.BarGraphItem(x=x_centers, height=self._bins, width=0.8, brush=(70, 130, 180))
        self._trial_widget.addItem(bar_item)

        #Horizontal dashed line for the grand mean
        mean_pen = pg.mkPen(color=(255, 0, 0), width=2.0, style=QtCore.Qt.DashLine)
        mean_line = pg.InfiniteLine(
            pos=self._last_bin_grand_mean, angle=0, pen=mean_pen, movable=False,
            label=f'Mean={self._last_bin_grand_mean:.2f}',
            labelOpts={'position': 0.5, 'color': (200, 0, 0)}
        )
        self._trial_widget.addItem(mean_line)

        pass

    def _update_trials_per_hour_plot (self) -> None:
        self._session_widget.clear()
        self._session_widget.getPlotItem().setLabel('bottom', 'Window size (µV)')
        self._session_widget.getPlotItem().setLabel('left', 'Trials per hour')
        self._session_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.4)

        if len(self._trial_means) < 2 or self._session_start_time is None:
            return

        means_arr = np.array(self._trial_means)
        sweep_centre = float(np.median(means_arr))
        std = float(np.std(means_arr))

        # Build 20 window sizes centred on the median, from a narrow to wide spread
        n_steps = 20
        min_hw = max(0.5, std * 0.1)
        max_hw = std * 2.0
        half_widths = np.linspace(min_hw, max_hw, n_steps)
        window_sizes = [2.0 * hw for hw in half_widths]

        # Session duration in hours
        elapsed_s = (datetime.now() - self._session_start_time).total_seconds()
        elapsed_hours = max(elapsed_s / 3600.0, 1e-9)

        # Count how many observed trial means fall within each window
        trials_per_hour = []
        for hw in half_widths:
            n = int(np.sum((means_arr >= sweep_centre - hw) & (means_arr <= sweep_centre + hw)))
            trials_per_hour.append(n / elapsed_hours)

        # Plot: steelblue line + circle markers (matches the "o-" style from the notebook)
        steelblue = (70, 130, 180)
        pen = pg.mkPen(color=steelblue, width=2)
        self._session_widget.plot(
            window_sizes,
            trials_per_hour,
            pen=pen,
            symbol='o',
            symbolBrush=steelblue,
            symbolPen=pg.mkPen(color=steelblue),
            symbolSize=8,
        )

        self._session_widget.getPlotItem().setTitle(
            f'Trials/hr vs Window Size | Centre: {sweep_centre:.2f} \u00b5V'
        )

        pass

    #endregion

    #region File output methods

    def _save_file_header (self) -> None:
        if (self._fid is not None):
            #Create a header object
            header: EmgCharacterizationHeader = EmgCharacterizationHeader(
                0,
                self._subject_id,
                datetime.now(),
                self.stage_name,
                self.stage_description,
                self.stage_type,
                EmgCharacterizationStage.BIN_DURATION_MILLISECONDS,
                EmgCharacterizationStage.TRIAL_INITIATION_MIN_RANGE_MICROVOLTS,
                EmgCharacterizationStage.TRIAL_INITIATION_MAX_RANGE_MICROVOLTS,
                EmgCharacterizationStage.TRIAL_INITIATION_PHASE_MIN_DURATION_MILLISECONDS,
                EmgCharacterizationStage.TRIAL_INITIATION_PHASE_MAX_DURATION_MILLISECONDS,
                float(ApplicationConfiguration.sample_rate),
            )

            #Save the header object to the data file
            header.save_to_file(self._fid)

            pass

    def _save_trial (self, bin_grand_mean: float) -> None:
        if (self._fid is not None):
            #Determine the trial's "start index"
            #The "start index" is the index into the TOTAL NUMBER OF SAMPLES that have been
            #streamed into this Python application during this session.
            trial_start_index: int = self._current_session_sample_count - self._current_trial_sample_count

            #Determine the trial's open ephys millis timestamp
            open_ephys_millis: int = self._monitored_signal_open_ephys_millis[0]

            #Determine the sample id of the first sample in the trial
            open_ephys_sample_id: int = self._monitored_signal_open_ephys_sample_ids[0]

            #Create a trial object
            trial: EmgCharacterizationTrial = EmgCharacterizationTrial(
                datetime.now(),
                trial_start_index,
                open_ephys_millis,
                open_ephys_sample_id,
                bin_grand_mean,
                self._bins,
                self._monitored_signal
            )

            #Save the trial object to the data file
            trial.save_to_file(self._fid)
            
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

    def _save_trials_per_hour_data (self) -> None:
        '''Computes the trials-per-hour vs window-size curve and writes it as a single block.'''
        if self._fid is None:
            return
        if len(self._trial_means) < 2 or self._session_start_time is None:
            return

        means_arr = np.array(self._trial_means)
        sweep_centre = float(np.median(means_arr))
        std = float(np.std(means_arr))

        n_steps = 20
        min_hw = max(0.5, std * 0.1)
        max_hw = std * 2.0
        half_widths = np.linspace(min_hw, max_hw, n_steps)
        window_sizes = np.array([2.0 * hw for hw in half_widths], dtype=np.float32)

        elapsed_s = (datetime.now() - self._session_start_time).total_seconds()
        elapsed_hours = max(elapsed_s / 3600.0, 1e-9)

        tph_values = []
        for hw in half_widths:
            n = int(np.sum((means_arr >= sweep_centre - hw) & (means_arr <= sweep_centre + hw)))
            tph_values.append(n / elapsed_hours)
        trials_per_hour = np.array(tph_values, dtype=np.float32)

        tph_data = EmgTrialsPerHourData(
            sweep_centre_uv=sweep_centre,
            elapsed_hours=elapsed_hours,
            window_sizes=window_sizes,
            trials_per_hour=trials_per_hour
        )
        tph_data.save_to_file(self._fid)

    #endregion