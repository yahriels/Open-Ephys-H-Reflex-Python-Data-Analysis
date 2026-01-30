from dataclasses import dataclass, field
from datetime import datetime
from typing import BinaryIO

import numpy as np

from ..fileio_helpers import FileIO_Helpers

class HReflexDataFileBlockIds:

    EMG_DATA_BLOCK: int = 1
    EMG_CHARACTERIZATION_TRIAL_BLOCK: int = 2
    MH_RECRUITMENT_CURVE_TRIAL_BLOCK: int = 3

    pass

@dataclass
class HReflexDataFileEmgData:

    timestamp_millis_open_ephys_sent: int = 0
    timestamp_millis_python_received: int = 0
    timestamp_millis_background_emitted: int = 0

    emg_channel_names: list[str] = field(default_factory=lambda: [])
    emg_data_raw: list[np.ndarray] = field(default_factory=lambda: [])
    emg_data_diff: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    emg_data_filtered: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    emg_data_abs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def read_from_file (self, fid: BinaryIO) -> None:
        #Read the Open Ephys millis timestamp
        self.timestamp_millis_open_ephys_sent = FileIO_Helpers.read(fid, "uint64")

        #Read the Python "received" millis timestamp
        self.timestamp_millis_python_received = FileIO_Helpers.read(fid, "uint64")

        #Read the Python "background emitted" millis timestamp
        self.timestamp_millis_background_emitted = FileIO_Helpers.read(fid, "uint64")

        #Read the number of EMG channel names
        n_channel_names: int = FileIO_Helpers.read(fid, "uint8")

        #Read each channel name
        self.emg_channel_names = []
        for i in range(0, n_channel_names):
            channel_name = FileIO_Helpers.read_string(fid)
            self.emg_channel_names.append(channel_name)

        #Read the number of EMG data channels
        n_channels: int = FileIO_Helpers.read(fid, "uint8")

        #Read the data from each EMG channel
        self.emg_data_raw = []
        for i in range(0, n_channels):
            channel_data = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)
            self.emg_data_raw.append(channel_data)
        
        #Read in the diff'd data
        self.emg_data_diff = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        #Read in the filtered data
        self.emg_data_filtered = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        #Read in the abs'd data
        self.emg_data_abs = np.array(FileIO_Helpers.read_array(fid, "float32"), dtype=np.float32)

        pass

    def save_to_file (self, fid: BinaryIO) -> None:
        #Save a data block header
        FileIO_Helpers.write(fid, "int32", HReflexDataFileBlockIds.EMG_DATA_BLOCK)

        #Save the Open Ephys timestamp (this is a Unix timestamp representing the number of MILLISECONDS since Jan 1, 1970)
        FileIO_Helpers.write(fid, "uint64", self.timestamp_millis_open_ephys_sent)

        #Save the timestamp representing when our Python application received the data from Open Ephys
        FileIO_Helpers.write(fid, "uint64", self.timestamp_millis_python_received)

        #Save the timestamp representing when our Python application sent the data from the background thread to the GUI for processing
        FileIO_Helpers.write(fid, "uint64", self.timestamp_millis_background_emitted)

        #Get the number of EMG channel names
        n_channel_names: int = len(self.emg_channel_names)

        #Save the number of EMG channel names
        FileIO_Helpers.write(fid, "uint8", n_channel_names)

        #Save each channel's name
        for i in range(0, n_channel_names):
            FileIO_Helpers.write_string(fid, self.emg_channel_names[i])

        #Get the number of EMG data channels
        n_channels: int = len(self.emg_data_raw)

        #Save the number of EMG data channels
        FileIO_Helpers.write(fid, "uint8", n_channels)

        #Save each channel's data
        for i in range(0, n_channels):
            FileIO_Helpers.write_array(fid, 'float32', self.emg_data_raw[i])
        
        #Save the diff'd data
        FileIO_Helpers.write_array(fid, 'float32', self.emg_data_diff)

        #Save the filtered data
        FileIO_Helpers.write_array(fid, 'float32', self.emg_data_filtered)

        #Save the abs'd data
        FileIO_Helpers.write_array(fid, 'float32', self.emg_data_abs)


