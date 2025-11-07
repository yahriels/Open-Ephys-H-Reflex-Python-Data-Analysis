import matplotlib.pyplot as plt
import numpy as np
from open_ephys.analysis import Session
import os
import re
from scipy.signal import butter, filtfilt, iirnotch

# ==== Select Session ====
w = 0

for x in range(4):
    y = f"recording{x + 1}"
    v = f"Record Node 106" if w == 0 else f"Record Node 111"

    # ==== Define Filters ====
    def bandpass_filter(data, fs, lowcut=100, highcut=1000, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, data)


    #def notch_filter(data, fs, freq=60.0, Q=30.0):
    #   nyq = 0.5 * fs
    #   b, a = iirnotch(freq / nyq, Q)
    #   return filtfilt(b, a, data)

    def full_filter(data, fs):
        return bandpass_filter(data, fs)

    # ==== Load Session ====
    #directory = 'ABATE-008_2025-05-14_14-34-09_001'
    #directory = 'NHNCE-187_2025-05-16_10-28-20_001'
    #directory = '2NHNCE-187_2025-05-16_11-34-10_001'
    #directory = 'RAtname_2025-05-16_12-13-55_001'
    directory = 'SNARE-32_2025-06-04_11-14-37_001'

    session = Session(directory)
    print('Pass Initial Test, Moving on to Loading Session...\n')
    print(session.recordnodes[w].recordings[x])

    recording = session.recordnodes[w].recordings[x]
    recording.add_sync_line(1, 100, 'Rhythm Data', main=True)
    recording.compute_global_timestamps()

    metadata = recording.continuous[0].metadata
    channel_names = metadata['channel_names']
    print("Channels:", channel_names, '\n')

    timestamps = recording.continuous[0].timestamps
    data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=len(timestamps))
    sample_rate = metadata['sample_rate']



    '''
    #############THIS ONE LOOKS VERY DIFFERENT! MAY BE THE CORRECT WAY TO DO IT? ####
    # ==== Apply Filtering to EMG Channels BEFORE Subtraction ====
    emg1_raw = data[:, 2]
    emg2_raw = data[:, 3]
    emg1 = full_filter(emg1_raw, sample_rate)
    emg2 = full_filter(emg2_raw, sample_rate)
    differential_emg = np.abs((emg1_raw - emg2_raw))  # <-- Absolute value applied here
    '''
     # ==== Apply Filtering to EMG Channels After Subtraction ====
    emg1_raw = data[:, 2]
    emg2_raw = data[:, 3]
    differential = emg1_raw - emg2_raw
    #differential = full_filter(emg1_raw - emg2_raw, sample_rate)
    differential_emg = np.abs(differential)  # <-- Absolute value applied here








    # ==== Extract Sync Events ====
    events = recording.events
    sync_events = events[(events.line == 1) & (events.processor_id == 100) &
                         (events.stream_name == 'Rhythm Data') & (events.state == 1)]
    sync_timestamps = sync_events['timestamp'].to_numpy()

    # ==== Load MessageCenter ====
    messagecenter_dir = os.path.join(directory, v, "experiment1", y, "events", "MessageCenter")
    texts = np.load(os.path.join(messagecenter_dir, "text.npy"), allow_pickle=True)
    timestamps_msg = np.load(os.path.join(messagecenter_dir, "timestamps.npy"))
    decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]
    message_entries = list(zip(timestamps_msg, decoded_texts))
    print(f"Loaded {len(decoded_texts)} MessageCenter entries")

    # ==== Debug Print ====
    for text, time in zip(decoded_texts, timestamps_msg):
        print(f"[Time: {time:.6f} s] Message: {text}")

    # ==== Full Trace Plot ====
    plt.figure(figsize=(15, 4))
    plt.plot(timestamps, differential_emg, label="Filtered EMG1 - EMG2", color='purple')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title(f"{directory}, {y}, Absolute Value Filtered Differential EMG Signal (EMG1 - EMG2)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



    # ==== Peri-Stimulus Analysis ====
    pre_event_time = 5  # ms
    post_event_time = 20  # ms
    window_samples = int(((pre_event_time + post_event_time) / 1000) * sample_rate)
    time_ms = np.linspace(-pre_event_time, post_event_time, window_samples)

    peri_stim_segments = []
    segment_timestamps = []

    for stamp in sync_timestamps:
        idx_start = np.searchsorted(timestamps, stamp - pre_event_time / 1000)
        idx_end = idx_start + window_samples
        if idx_end <= len(differential_emg):
            segment = differential_emg[idx_start:idx_end]
            if len(segment) == window_samples:
                peri_stim_segments.append(segment)
                segment_timestamps.append(stamp)

    def find_latest_amplitude_msg(before_time):
        # Accepts "starting at X mA" or "starting X mA"
        pattern = re.compile(r"starting(?: at)? (\d+\.?\d*)\s*mA", re.IGNORECASE)
        last_amp_msg = "Unknown"
        for t, msg in reversed(message_entries):
            if t < before_time:
                match = pattern.search(msg)
                if match:
                    return f"{match.group(1)} mA"
        return last_amp_msg
    

    group_size = 5
    n_groups = len(peri_stim_segments) // group_size

    for i in range(n_groups):
        group = peri_stim_segments[i * group_size : (i + 1) * group_size]
        group_stamps = segment_timestamps[i * group_size : (i + 1) * group_size]
        avg_trace = np.mean(group, axis=0)
        amplitude = find_latest_amplitude_msg(group_stamps[0])

        plt.figure(figsize=(7, 4))
        for trace in group:
            plt.plot(time_ms, trace, color='red', alpha=0.6)
        plt.plot(time_ms, avg_trace, color='black', linewidth=2, label='Average EMG')
        plt.axvline(x=0, color='blue', linestyle='--', label='Stimulus Onset')
        plt.title(f"{y}, Peri-Stimulus EMG (Events {i*group_size+1}-{(i+1)*group_size}) | Stim Amp: {amplitude}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (μV)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

