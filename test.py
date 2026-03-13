import matplotlib.pyplot as plt
import numpy as np
from open_ephys.analysis import Session
import os
import re
from scipy.signal import butter, filtfilt, lfilter

# ==== Select Session ====
w = 1  # recordnode index

# ==== Define Filters ====

def bandpass_filter(data, fs, lowcut=100, highcut=1000, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

def full_filter(data, fs):
    return bandpass_filter(data, fs)

# ==== Load Session ====
directory = '2-24-2026\SEQ-04_Config1_Test1_2026-02-24_13-50-17_004'
session = Session(directory)
print('Pass Initial Test, Moving on to Loading Session...\n')

x = 0
exp = "experiment1"
y = f"recording{x + 1}"
v = f"Record Node 106" if w == 0 else f"Record Node 111"

try:
    recording = session.recordnodes[w].recordings[x]
except IndexError:
    print(f"No recording {x+1} found in {exp}")
    
print(f"\n{exp.upper()}, Recording {x+1}")
print(recording)

if not any(sync.get("main", False) for sync in recording.sync_lines):
    recording.add_sync_line(1, 100, 'Rhythm Data', main=True)

metadata = recording.continuous[0].metadata
channel_names = metadata.channel_names
print("Channels:", channel_names, '\n')

stream = recording.continuous[0]
timestamps = stream.timestamps
n_timestamps = timestamps.shape[0]
data = stream.get_samples(start_sample_index=0, end_sample_index=n_timestamps)
sample_rate = metadata.sample_rate

emg1_raw = data[:, 2]
emg2_raw = data[:, 3]
adc1 = np.abs(data[:, 6])
differential_emg = full_filter(emg2_raw - emg1_raw, sample_rate)

events = recording.events
sync_events = events[(events.line == 1) & (events.processor_id == 100) &
                        (events.stream_name == 'Rhythm Data') & (events.state == 1)]
sync_timestamps = sync_events['timestamp'].to_numpy()

messagecenter_dir = os.path.join(directory, v, exp, y, "events", "MessageCenter")
if not os.path.exists(messagecenter_dir):
    print(f"MessageCenter directory not found for {exp}/{y}")
   
texts = np.load(os.path.join(messagecenter_dir, "text.npy"), allow_pickle=True)
timestamps_msg = np.load(os.path.join(messagecenter_dir, "timestamps.npy"))
decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]
message_entries = list(zip(timestamps_msg, decoded_texts))
print(f"Loaded {len(decoded_texts)} MessageCenter entries")

for text, time in zip(decoded_texts, timestamps_msg):
    print(f"[Time: {time:.6f} s] Message: {text}")

plt.figure(figsize=(15, 4))
plt.plot(timestamps, adc1, label="Filtered EMG1 - EMG2", color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"{directory}, {exp}, {y}, Absolute Value Filtered Differential EMG Signal (EMG1 - EMG2)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.legend()
#plt.ylim(top=6500)
plt.tight_layout()
plt.show()