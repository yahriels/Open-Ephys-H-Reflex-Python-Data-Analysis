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

## This class represents the header for an "EMG characterization" data file.
@dataclass
class EmgCharacterizationHeader:
    file_version: int = 0
    subject_id: str = ""
    session_datetime: datetime = datetime.min
    stage_name: str = ""
    stage_description: str = ""
    stage_type: int = 0

    bin_duration_ms: int = 0
    trial_initiation_uv_min: float = 0
    trial_initiation_uv_max: float = 0
    trial_initiation_phase_min_ms: int = 0
    trial_initiation_phase_max_ms: int = 0
    sample_rate: float = 0.0

    def read_from_file (self, fid: BinaryIO) -> None:
        self.file_version = FileIO_Helpers.read(fid, "int32")
        self.subject_id = FileIO_Helpers.read_string(fid)
        self.session_datetime = FileIO_Helpers.read_datetime(fid)
        self.stage_name = FileIO_Helpers.read_string(fid)
        self.stage_description = FileIO_Helpers.read_string(fid)
        self.stage_type = FileIO_Helpers.read(fid, "int32")

        self.trial_initiation_uv_min = FileIO_Helpers.read(fid, "float32")
        self.trial_initiation_uv_max = FileIO_Helpers.read(fid, "float32")
        self.trial_initiation_phase_min_ms = FileIO_Helpers.read(fid, "int32")
        self.trial_initiation_phase_max_ms = FileIO_Helpers.read(fid, "int32")
        self.bin_duration_ms = FileIO_Helpers.read(fid, "int32")
        self.sample_rate = FileIO_Helpers.read(fid, "float32")

        pass

    def save_to_file (self, fid: BinaryIO) -> None:

        #Save the file version
        FileIO_Helpers.write(fid, "int32", self.file_version)

        #Save the subject id
        FileIO_Helpers.write_string(fid, self.subject_id)

        #Save the session date/time
        FileIO_Helpers.write_datetime(fid, self.session_datetime)

        #Save the stage name
        FileIO_Helpers.write_string(fid, self.stage_name)

        #Save the stage description
        FileIO_Helpers.write_string(fid, self.stage_description)

        #Save the stage type
        FileIO_Helpers.write(fid, "int32", self.stage_type)

        #Save the min and max range for trial initiation
        FileIO_Helpers.write(fid, "float32", self.trial_initiation_uv_min)
        FileIO_Helpers.write(fid, "float32", self.trial_initiation_uv_max)

        #Save the min and max recording duration for trial initiation criteria
        FileIO_Helpers.write(fid, "int32", self.trial_initiation_phase_min_ms)
        FileIO_Helpers.write(fid, "int32", self.trial_initiation_phase_max_ms)

        #Save the bin width
        FileIO_Helpers.write(fid, "int32", self.bin_duration_ms)

        #Save the sample rate
        FileIO_Helpers.write(fid, "float32", self.sample_rate)

        pass

##This class represents an individual "trial" within an "EMG characterization" data file/session.
@dataclass
class EmgCharacterizationTrial:

    #This datetime represents the END of the monitored signal, NOT the beginning
    trial_end_datetime: datetime = datetime.min

    #This index represents the index into the FULL EMG data for the session at which this trial's monitored signal begins
    trial_start_index: int = 0

    #This timestamp is a millisecond-scale timestamp representing the time at which the first sample in the monitored signal was sent to
    #this application by Open Ephys
    trial_start_open_ephys_millis: int = 0

    #This sample id is an integer index/id into the FULL EMG data as recorded by Open Ephys. It can be used to sync the monitored signal
    #data with the EMG data in an Open Ephys recording file
    trial_start_open_ephys_sample_id: int = 0

    #The grand mean is the mean of all bin values
    grand_mean: float = 0.0

    #The values of each bin are the mean of each 50-millisecond interval in the monitored signal
    bins: list[float] = field(default_factory=list)

    #The monitored signal is the absolute value of the difference between the two EMG channels.
    #So essentially: monitored signal = abs(EMG2 - EMG1)
    monitored_signal: list[float] = field(default_factory=list)

    #The actual runtime initiation thresholds active when this trial was counted.
    #Defaults match the class-level constants for files written before version 1.
    actual_init_min_threshold: float = 0.0
    actual_init_max_threshold: float = 0.0

    def read_from_file (self, fid: BinaryIO, file_version: int = 0) -> None:
        self.trial_end_datetime = FileIO_Helpers.read_datetime(fid)
        self.trial_start_index = FileIO_Helpers.read(fid, "uint64")
        self.trial_start_open_ephys_millis = FileIO_Helpers.read(fid, "uint64")
        self.trial_start_open_ephys_sample_id = FileIO_Helpers.read(fid, "uint64")
        self.grand_mean = FileIO_Helpers.read(fid, "float32")
        self.bins = FileIO_Helpers.read_array(fid, "float32")
        self.monitored_signal = FileIO_Helpers.read_array(fid, "float32")

        #Per-trial runtime thresholds (added in file version 1)
        if file_version >= 1:
            self.actual_init_min_threshold = FileIO_Helpers.read(fid, "float32")
            self.actual_init_max_threshold = FileIO_Helpers.read(fid, "float32")

    def save_to_file (self, fid: BinaryIO) -> None:

        #Save a trial block header
        FileIO_Helpers.write(fid, "int32", HReflexDataFileBlockIds.EMG_CHARACTERIZATION_TRIAL_BLOCK)

        #Save a datetime for the trial initiation
        FileIO_Helpers.write_datetime(fid, self.trial_end_datetime)

        #Save the index into the full EMG data at which the trial begins
        FileIO_Helpers.write(fid, "uint64", self.trial_start_index)

        #Save the Open Ephys millisecond timestamp at which this trial began
        FileIO_Helpers.write(fid, "uint64", self.trial_start_open_ephys_millis)

        #Save the Open Ephyis sample id for the FIRST sample in this trial
        FileIO_Helpers.write(fid, "uint64", self.trial_start_open_ephys_sample_id)

        #Save the bin grand mean
        FileIO_Helpers.write(fid, "float32", self.grand_mean)

        #Save the bin data
        FileIO_Helpers.write_array(fid, "float32", self.bins)

        #Save the monitored signal
        FileIO_Helpers.write_array(fid, "float32", self.monitored_signal)

        #Save the runtime initiation thresholds (file version 1+)
        FileIO_Helpers.write(fid, "float32", self.actual_init_min_threshold)
        FileIO_Helpers.write(fid, "float32", self.actual_init_max_threshold)

## This class holds the computed "trials per hour vs window size" curve and IS saved to the
## data file as a single block written at session end (block ID = EMG_TRIALS_PER_HOUR_BLOCK).
@dataclass
class EmgTrialsPerHourData:

    #Centre of the sweep (median of trial grand means at the time of saving), in µV
    sweep_centre_uv: float = 0.0

    #Total elapsed session time in hours (used to normalise trial counts to a per-hour rate)
    elapsed_hours: float = 0.0

    #Array of window widths (µV) evaluated during the sweep
    window_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    #Corresponding trials-per-hour count for each window width
    trials_per_hour: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def read_from_file (self, fid: BinaryIO) -> None:
        self.sweep_centre_uv = FileIO_Helpers.read(fid, "float32")
        self.elapsed_hours = FileIO_Helpers.read(fid, "float32")
        self.window_sizes = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)
        self.trials_per_hour = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

    def save_to_file (self, fid: BinaryIO) -> None:
        FileIO_Helpers.write(fid, "int32", HReflexDataFileBlockIds.EMG_TRIALS_PER_HOUR_BLOCK)
        FileIO_Helpers.write(fid, "float32", self.sweep_centre_uv)
        FileIO_Helpers.write(fid, "float32", self.elapsed_hours)
        FileIO_Helpers.write_array(fid, "float32", self.window_sizes)
        FileIO_Helpers.write_array(fid, "float32", self.trials_per_hour)


## This class (EmgHistogramData) is NOT saved in the data file. Rather, it is CALCULATED
## from data that is present in the data file.
@dataclass
class EmgHistogramData:
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    n: float = 0.0
    quartiles: list[float] = field(default_factory=list)
    step_size_one_percent: float = 0.0
    histogram_values: np.ndarray = field(default_factory=lambda: np.zeros(1))
    histogram_bin_edges: np.ndarray = field(default_factory=lambda: np.zeros(1))

## This class represents an "EMG characterization" session.
class EmgCharacterizationDataFile:

    #region Constructor

    def __init__(self):
        self.header: EmgCharacterizationHeader = None
        self.emg_data_blocks: list[HReflexDataFileEmgData] = []
        self.trials: list[EmgCharacterizationTrial] = []
        self.trials_per_hour_data: EmgTrialsPerHourData = None

    #endregion

    #region File IO methods

    def read (self, fid: BinaryIO) -> None:
        '''
        Readers the EMG characterization data file from disk.
        '''

        self.header = EmgCharacterizationHeader()
        self.header.read_from_file(fid)

        while (True):
            chunk: bytes = fid.read(4)
            if (not chunk):
                break

            block_id: int = struct.unpack(FileIO_Helpers.type_dictionary["int32"], chunk)[0]
            if (block_id == HReflexDataFileBlockIds.EMG_CHARACTERIZATION_TRIAL_BLOCK):
                trial: EmgCharacterizationTrial = EmgCharacterizationTrial()
                trial.read_from_file(fid, self.header.file_version)

                self.trials.append(trial)
            elif (block_id == HReflexDataFileBlockIds.EMG_DATA_BLOCK):
                emg_data_block: HReflexDataFileEmgData = HReflexDataFileEmgData()
                emg_data_block.read_from_file(fid)

                self.emg_data_blocks.append(emg_data_block)
            elif (block_id == HReflexDataFileBlockIds.EMG_TRIALS_PER_HOUR_BLOCK):
                tph: EmgTrialsPerHourData = EmgTrialsPerHourData()
                tph.read_from_file(fid)
                self.trials_per_hour_data = tph
        
        pass

    #endregion

    #region Other public methods

    def get_all_grandmeans (self) -> list[float]:
        '''
        Generates a list that contains the grandmean from each trial and
        returns the list to the caller.
        '''

        result: list[float] = [t.grand_mean for t in self.trials]
        return result
    
    def get_histogram_data (self) -> EmgHistogramData:
        '''
        Calculates histogram data that is used by other pieces of the application.
        '''

        #Instantiate an object to hold the histogram data
        histogram_data: EmgHistogramData = EmgHistogramData()

        #Get the list of grand means
        grand_means: list[float] = self.get_all_grandmeans()

        #Calculate the min, max, mean, std dev, and quartiles
        histogram_data.min = min(grand_means)
        histogram_data.max = max(grand_means)
        histogram_data.n = len(grand_means)
        histogram_data.std_dev = np.std(grand_means)

        histogram_data.quartiles.clear()
        histogram_data.quartiles.append(np.quantile(grand_means, 0.25))
        histogram_data.quartiles.append(np.quantile(grand_means, 0.50))
        histogram_data.quartiles.append(np.quantile(grand_means, 0.75))

        histogram_data.step_size_one_percent = np.quantile(grand_means, 0.01) - histogram_data.min

        (histogram_data.histogram_values, histogram_data.histogram_bin_edges) = np.histogram(grand_means)

        #Return the histogram data to the caller
        return histogram_data

    #endregion

    #region Public static methods

    @staticmethod
    def find_all_emg_characterization_data_files (subject_id: str) -> list[str]:
        #Generate the path for this subject id
        app_data_path: str = user_data_dir(ApplicationConfiguration.appname, ApplicationConfiguration.appauthor)
        app_experimental_data_path: str = os.path.join(app_data_path, "Data")
        subject_file_path: str = f"{app_experimental_data_path}/{subject_id}/"

        #Find all files with the expected file extension
        files_list: list[str] = os.listdir(subject_file_path)

        #Create an array to hold the result
        result: list[str] = []

        #Find the first file with the "hrs1" file extension
        for f in files_list:
            if (f.endswith("hrs1")):
                result.append(f)

        return result

    #endregion
