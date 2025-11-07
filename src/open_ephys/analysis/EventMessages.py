import os
import numpy as np
import matplotlib.pyplot as plt
from open_ephys.analysis import Session

# Load the session
directory = 'ABATE-008_2025-05-14_14-34-09_001'
session = Session(directory)

print('Pass Initial Test, Moving on to Loading Session...\n')
print(session)
print(session.recordnodes[0].recordings[3])

# Load specific recording
recording = session.recordnodes[0].recordings[3]

# Add sync line and compute global timestamps
recording.add_sync_line(
    1,               # TTL line number
    100,             # processor ID
    'Rhythm Data',   # Stream name (verify this matches your metadata)
    main=True        # Set this stream as reference
)
recording.compute_global_timestamps()

# Extract metadata and global timestamps
metadata = recording.continuous[0].metadata
channel_names = metadata['channel_names']
timestamps = recording.continuous[0].timestamps
data = recording.continuous[0].get_samples(start_sample_index=0, end_sample_index=timestamps.shape[0])
print("Data shape:", data.shape)

# Filter sync events for Line 1
events = recording.events
sync_events = events[
    (events.line == 1) &
    (events.processor_id == 100) &
    (events.stream_name == 'Rhythm Data') &
    (events.state == 1)
]
sync_timestamps = sync_events['timestamp'].to_numpy()

# Load MessageCenter files
messagecenter_dir = os.path.join(directory, "Record Node 106", "experiment1", "recording4", "events", "MessageCenter")
texts = np.load(os.path.join(messagecenter_dir, "text.npy"), allow_pickle=True)
timestamps_msg = np.load(os.path.join(messagecenter_dir, "timestamps.npy"))
sample_numbers = np.load(os.path.join(messagecenter_dir, "sample_numbers.npy"))

# Decode message texts
decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]

# Debug print
print(f"Loaded {len(decoded_texts)} MessageCenter entries")
for text, time in zip(decoded_texts, timestamps_msg):
    print(f"[Time: {time:.6f} s] Message: {text}")

# Plot continuous data
fig, axes = plt.subplots(len(channel_names), 1, figsize=(15, 12), sharex=True)

for idx, ax in enumerate(axes):
    ax.plot(timestamps, data[:, idx], label=channel_names[idx])
    for t in sync_timestamps:
        ax.axvline(x=t, color='red', linestyle='--', alpha=0.7, label='Sync Event' if t == sync_timestamps[0] else "")
    ax.set_ylabel('Amplitude (μV)')
    ax.set_title(channel_names[idx])
    ax.grid(True)
    if idx == 0:  # Only annotate messages on topmost plot (Ch5)
        for time, msg in zip(timestamps_msg, decoded_texts):
            ax.axvline(x=time, color='green', linestyle='--', alpha=0.6)
            ax.annotate(
                msg,
                xy=(time, ax.get_ylim()[1]),
                xytext=(time + 0.1, ax.get_ylim()[1] * 1.05),
                rotation=30,  # angled, not vertical
                fontsize=9,
                color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1),
                ha='left'
            )

axes[-1].set_xlabel('Time (s)')
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space at top
plt.suptitle('Open Ephys Continuous Data with Sync + MessageCenter Annotations', fontsize=14)
plt.show()

# Optional: standalone sync event plot
plt.figure(figsize=(10, 2))
plt.eventplot(sync_timestamps, orientation='horizontal', colors='red')
plt.title(f'Sync Events on Line 1 - Rhythm Data')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()
