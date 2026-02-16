# %% Imports

from typing import BinaryIO
from platformdirs import user_data_dir
import os
import numpy as np

from hreflex_txbdc.model.application_configuration import ApplicationConfiguration
from hreflex_txbdc.model.datafiles.emg_characterization_data_file import EmgCharacterizationDataFile, EmgCharacterizationHeader, EmgCharacterizationTrial

# %% Define the subject ID and then find all EMG characterization stage data files

subject_id: str = "TEST1"

file_list: list[str] = EmgCharacterizationDataFile.find_all_emg_characterization_data_files(subject_id)

# %% Print the list of files found

print(f"{len(file_list)} files were found: ")
for f in file_list:
    print(f)

# %% Print data from the first file that was found

if (len(file_list) > 0):

    app_data_path: str = user_data_dir(ApplicationConfiguration.appname, ApplicationConfiguration.appauthor)
    app_experimental_data_path: str = os.path.join(app_data_path, "Data")
    subject_data_path: str = os.path.join(app_experimental_data_path, subject_id)
    full_file_path: str = os.path.join(subject_data_path, file_list[0])

    session_data: EmgCharacterizationDataFile = EmgCharacterizationDataFile()
    
    fid: BinaryIO = open(full_file_path, "rb")
    session_data.read(fid)
    fid.close()

    #Display the number of trials
    print(f"Trial count = {len(session_data.trials)}")

    #Display the grand grand mean
    gm: list[float] = session_data.get_all_grandmeans()
    gm_mean: float = np.mean(gm)

    print(f"Grand mean = {gm_mean}")

    #Display the number of EMG data blocks
    emg_data_block_count: int = len(session_data.emg_data_blocks)
    print(f"EMG data block count = {emg_data_block_count}")
