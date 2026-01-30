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

        #Define a numpy array to hold trial data
        self.trial_data: np.ndarray = np.zeros(1, dtype=np.float32)

        #Create variables to hold trial parameters
        self.min_initiation_threshold: float = 0.0
        self.max_initiation_threshold: float = 0.0
        self.stimulation_amplitude_ma: float = 0.0

        pass

    #endregion

    #region Methods

    def initialize (self, min_init_threshold: float, max_init_threshold: float, stim_amp: float) -> None:
        self.start_time = datetime.now()
        self.min_initiation_threshold = min_init_threshold
        self.max_initiation_threshold = max_init_threshold
        self.stimulation_amplitude_ma = stim_amp

    def read_from_file (self, fid: BinaryIO) -> None:
        #Read the timestamp for this trial
        self.start_time = FileIO_Helpers.read_datetime(fid)

        #Read the initiation thresholds
        self.min_initiation_threshold = FileIO_Helpers.read(fid, "float32")
        self.max_initiation_threshold = FileIO_Helpers.read(fid, "float32")

        #Read the stimulation amplitude for this trial
        self.stimulation_amplitude_ma = FileIO_Helpers.read(fid, "float32")

        #Read the trial data
        self.trial_data = FileIO_Helpers.read_array(fid, "float32")

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
                trial.read_from_file(fid)

                self.trials.append(trial)
            elif (block_id == HReflexDataFileBlockIds.EMG_DATA_BLOCK):
                emg_data_block: HReflexDataFileEmgData = HReflexDataFileEmgData()
                emg_data_block.read_from_file(fid)
                
                self.emg_data_blocks.append(emg_data_block)
        
        pass

    #endregion

    pass