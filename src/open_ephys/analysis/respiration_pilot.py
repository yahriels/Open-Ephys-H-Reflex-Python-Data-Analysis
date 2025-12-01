import matplotlib.pyplot as plt
import numpy as np
from open_ephys.analysis import Session
import os
import re
from scipy.signal import butter, filtfilt, lfilter
from collections import defaultdict
import scipy.stats as stats

from scipy.signal import resample_poly

def downsample_signal(signal, original_rate, target_rate):
    """
    Downsample a 1D signal from original_rate to target_rate
    using polyphase filtering (resample_poly).
    """
    # Compute up/down ratios
    up = target_rate
    down = original_rate
    return resample_poly(signal, up, down)



def scan_experiment_structure(session_dir, record_node_name="Record Node 106"):
    node_path = os.path.join(session_dir, record_node_name)

    if not os.path.exists(node_path):
        print(f"Record Node directory not found: {node_path}")
        return {}

    experiment_info = defaultdict(list)

    for item in sorted(os.listdir(node_path)):
        exp_path = os.path.join(node_path, item)
        if os.path.isdir(exp_path) and item.startswith("experiment"):
            recordings = [
                r for r in sorted(os.listdir(exp_path))
                if os.path.isdir(os.path.join(exp_path, r)) and r.startswith("recording")
            ]
            experiment_info[item] = recordings

    return experiment_info

# ==== Select record node ====
# w = 0 → Record Node 106
# w = 1 → Record Node 111
w = 1

directory = "dEMG_Pilot-0012025-10-31_13-23-53_001"

session = Session(directory)
print("Session loaded successfully.\n")
print(session)


record_node_name = "Record Node 106" if w == 0 else "Record Node 111"

structure = scan_experiment_structure(directory, record_node_name)
structure


recording_index = 0   # choose which recording to load

try:
    recording = session.recordnodes[w].recordings[recording_index]
    print("Loaded:", recording)
except IndexError:
    print(f"Recording index {recording_index} not found.")



directory_str = recording.directory  # full directory path as a string

print("\n\n\n")

print(directory_str)


parts = directory_str.split(os.sep)

# Extract components based on Open Ephys folder structure
record_node_name = parts[-3]   # "Record Node 111"
experiment_name  = parts[-2]   # "experiment1"
recording_name   = parts[-1]   # "recording1"

print(record_node_name)
print(experiment_name)
print(recording_name)

continuous_list = recording.continuous

print(f"Number of continuous streams: {len(continuous_list)}")

stream = continuous_list[0]   # pick first continuous stream
print(stream)

metadata = stream.metadata
channel_names = metadata.channel_names
sample_rate = metadata.sample_rate
num_channels = metadata.num_channels
print("Sample rate:", metadata.sample_rate)
print("Num channels:", metadata.num_channels)
print("Channel names:", metadata.channel_names)


# Load continuous data using global timestamps and data shape
timestamps = recording.continuous[0].timestamps
n_timestamps = timestamps.shape[0]
data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=n_timestamps)
print("Data shape:", data.shape)

# ==== EMG Processing ====
emg1_raw = data[:, 2]
emg2_raw = data[:, 3]
differential = (emg2_raw - emg1_raw)
lowcut = 100
highcut = 5000

b, a = butter(2, np.array([lowcut, highcut])/(sample_rate/2), btype='bandpass')


#y = lfilter(b, a, x) # only filters in the forward direction



differential_filt = lfilter(b, a, differential)
differential_abs = np.abs(differential_filt)
differential_emg = differential_abs

# ==== Downsample from 25 kHz → 2 kHz ====
target_rate = 2000  # desired sampling rate
differential_emg_ds = downsample_signal(differential_emg, sample_rate, target_rate)

# Also downsample timestamps
timestamps_ds = downsample_signal(timestamps, sample_rate, target_rate)




#differential_filt = data[:,3] #Only if online filtered

# ==== Full Trace Plot ====
plt.figure(figsize=(15, 8))
plt.plot(timestamps, differential_filt, label="Filtered EMG1 - EMG2", color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"{directory}, {experiment_name}, {recording_name}, Absolute Value Filtered Differential EMG Signal (EMG1 - EMG2)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.legend()
#plt.ylim(top=5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(timestamps_ds, differential_ds, label="Downsampled Filtered EMG (2 kHz)")
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"{directory}, {experiment_name}, {recording_name}, Downsampled EMG (25kHz→2kHz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
