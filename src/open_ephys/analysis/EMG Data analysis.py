#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 10:48:45 2026

@author: pariyazare
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button #added by Pariya
import numpy as np
from open_ephys.analysis import Session
import os
import re
from scipy.signal import butter, filtfilt, lfilter
from collections import defaultdict
import scipy.stats as stats
#import ipywidgets as widgets
#from ipywidgets import Button, Output, VBox
from IPython.display import display
def scan_experiment_structure(session_dir, record_node_name="Record Node 111"):
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
w = 0

directory = '/Users/pariyazare/Desktop/emg_data/MDL-145/MDL-145_10P1_011420262026-01-14_11-30-31_001/'
import os

print("directory =", directory)
print("exists?", os.path.exists(directory))
print("is folder?", os.path.isdir(directory))
print("contents:", os.listdir(directory))
session = Session(directory)
print("Session loaded successfully.\n")
print(session)
# === Flatten experiment-recording pairs to loop ===
#flat_recordings = []
#for exp_name, rec_list in structure.items():
#    for rec in rec_list:
#        flat_recordings.append((exp_name, rec))
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
highcut = 999.99

b, a = butter(2, np.array([lowcut, highcut])/(sample_rate/2), btype='bandpass')


#y = lfilter(b, a, x) # only filters in the forward direction

#differential_filt = data[:,3]
#differential_filt = lfilter(b, a, differential) #use it if you are plotting node 111 (raw data)
#differential_abs = np.abs(differential_filt) #use it if you are plotting node 111 (raw data)
#differential_emg = differential_abs #use it if you are plotting node 111 (raw data)
differential_filt = data[:,3] #Only if online filtered, if not then comment out 
# =========================
# ==== Rectification Block ====
# =========================
differential_rect = np.abs(differential_filt)

#%%
# ==== Full Trace Plot ====
plt.figure(figsize=(15, 4))
plt.plot(timestamps, differential_filt, label="Filtered EMG1 - EMG2", color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"{directory}, {experiment_name}, {recording_name}, Absolute Value Filtered Differential EMG Signal (EMG1 - EMG2)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.legend()

#plt.ylim(top=1000)
#plt.ylim(bottom=-1000)
plt.tight_layout()
#plt.savefig('/Users/pariyazare/Desktop/emg_data/MDL-145/MDL-145_10P2_011420262026-01-14_11-35-06_002/test.png')
plt.show()

differential_filt = data[:,3] #Only if online filtered, if not then comment out 

# Segment setup
total_time = timestamps[-1] - timestamps[0]
segment_duration = 50  # seconds
num_segments = int(np.ceil(total_time / segment_duration))
current_segment = 0

def plot_segment(segment_idx):
    start_time = timestamps[0] + segment_idx * segment_duration
    end_time = min(start_time + segment_duration, timestamps[-1])
    mask = (timestamps >= start_time) & (timestamps < end_time)
    plt.figure(figsize=(15, 4))
    plt.plot(timestamps[mask] - start_time, differential_filt[mask], label="Filtered EMG1 - EMG2", color='purple')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title(f"{directory}, {experiment_name}, {recording_name}, Filtered Differential EMG Signal (EMG1 - EMG2), Segment {segment_idx+1}/{num_segments}, Time: {start_time:.1f}-{end_time:.1f}s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (μV)")
    plt.grid(True)
    plt.legend()
    #plt.ylim(top=1000)
    #plt.ylim(bottom=-1000)
    plt.tight_layout()
    plt.show()
##This section has been commented because of using botton 
# output = Output()

# def on_next(b):
#     global current_segment
#     if current_segment < num_segments - 1:
#         current_segment += 1
#         with output:
#             output.clear_output(wait=True)
#             plot_segment(current_segment)

# def on_prev(b):
#     global current_segment
#     if current_segment > 0:
#         current_segment -= 1
#         with output:
#             output.clear_output(wait=True)
#             plot_segment(current_segment)

# next_button = Button(description="Next")
# prev_button = Button(description="Previous")
# next_button.on_click(on_next)
# prev_button.on_click(on_prev)

# # Initial plot
# with output:
#     plot_segment(current_segment)

# display(VBox([prev_button, next_button, output]))

# =====  Segment viewer with Next/Previous buttons_ Pariya=====
segment_duration = 10  # seconds
t0 = float(timestamps[0])
t_end = float(timestamps[-1])
total_time = t_end - t0
num_segments = int(np.ceil(total_time / segment_duration))

state = {"idx": 0}

fig, ax = plt.subplots(figsize=(15, 4))
plt.subplots_adjust(bottom=0.25)  # space for buttons

line, = ax.plot([], [], color='purple', label="Filtered EMG1 - EMG2")
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel("Time within segment (s)")
ax.set_ylabel("Amplitude (μV)")
ax.grid(True)
ax.legend()
title = ax.set_title("")

def update_plot():
    i = state["idx"]
    start_time = t0 + i * segment_duration
    end_time = min(start_time + segment_duration, t_end)
    mask = (timestamps >= start_time) & (timestamps < end_time)

    x = timestamps[mask] - start_time
    y = differential_filt[mask]

    line.set_data(x, y)
    ax.set_xlim(0, end_time - start_time)

    # autoscale y per segment (optional)
    if y.size > 0:
        ypad = 0.05 * (np.max(y) - np.min(y) + 1e-9)
        ax.set_ylim(np.min(y) - ypad, np.max(y) + ypad)

    title.set_text(
        f"{os.path.basename(directory)} | {experiment_name}/{recording_name} "
        f"| Segment {i+1}/{num_segments} | {start_time:.1f}-{end_time:.1f}s"
    )
    fig.canvas.draw_idle()

def next_segment(event):
    if state["idx"] < num_segments - 1:
        state["idx"] += 1
        update_plot()

def prev_segment(event):
    if state["idx"] > 0:
        state["idx"] -= 1
        update_plot()

axprev = plt.axes([0.30, 0.08, 0.15, 0.10])
axnext = plt.axes([0.55, 0.08, 0.15, 0.10])

bprev = Button(axprev, "Previous")
bnext = Button(axnext, "Next")
bprev.on_clicked(prev_segment)
bnext.on_clicked(next_segment)

update_plot()
plt.show()

# ==========================================
# ===== Graph 3: RECTIFIED (full + segmented viewer)
# ==========================================

# ---- (A) Rectified full trace plot ----
plt.figure(figsize=(15, 4))
plt.plot(timestamps, differential_rect, label="Rectified EMG (abs)", color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"{os.path.basename(directory)} | {experiment_name}/{recording_name} | Rectified EMG (Full Trace)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---- (B) Rectified segmented viewer with Next/Previous buttons ----
segment_duration_rect = 10  # seconds (you can change)
t0 = float(timestamps[0])
t_end = float(timestamps[-1])
total_time = t_end - t0
num_segments_rect = int(np.ceil(total_time / segment_duration_rect))

state_rect = {"idx": 0}

fig_rect, ax_rect = plt.subplots(figsize=(15, 4))
plt.subplots_adjust(bottom=0.25)  # space for buttons

line_rect, = ax_rect.plot([], [], color='purple', label="Rectified EMG (abs)")
ax_rect.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax_rect.set_xlabel("Time within segment (s)")
ax_rect.set_ylabel("Amplitude (μV)")
ax_rect.grid(True)
ax_rect.legend()
title_rect = ax_rect.set_title("")

def update_plot_rect():
    i = state_rect["idx"]
    start_time = t0 + i * segment_duration_rect
    end_time = min(start_time + segment_duration_rect, t_end)
    mask = (timestamps >= start_time) & (timestamps < end_time)

    x = timestamps[mask] - start_time
    y = differential_rect[mask]

    line_rect.set_data(x, y)
    ax_rect.set_xlim(0, end_time - start_time)

    # autoscale y per segment
    if y.size > 0:
        ypad = 0.05 * (np.max(y) - np.min(y) + 1e-9)
        ax_rect.set_ylim(np.min(y) - ypad, np.max(y) + ypad)

    title_rect.set_text(
        f"{os.path.basename(directory)} | {experiment_name}/{recording_name} "
        f"| RECTIFIED | Segment {i+1}/{num_segments_rect} | {start_time:.1f}-{end_time:.1f}s"
    )
    fig_rect.canvas.draw_idle()

def next_segment_rect(event):
    if state_rect["idx"] < num_segments_rect - 1:
        state_rect["idx"] += 1
        update_plot_rect()

def prev_segment_rect(event):
    if state_rect["idx"] > 0:
        state_rect["idx"] -= 1
        update_plot_rect()

axprev_rect = plt.axes([0.30, 0.08, 0.15, 0.10])
axnext_rect = plt.axes([0.55, 0.08, 0.15, 0.10])

bprev_rect = Button(axprev_rect, "Previous")
bnext_rect = Button(axnext_rect, "Next")
bprev_rect.on_clicked(prev_segment_rect)
bnext_rect.on_clicked(next_segment_rect)

update_plot_rect()
plt.show()

# ==========================================
# ===== Graph 4: ENVELOPE (full + segmented viewer)
# ==========================================
# Envelope = low-pass(filtered/rectified EMG)

# ---- (0) Compute envelope ----
envelope_lowpass_hz = 2  # <- change this (e.g., 3, 10, 40)
wn_env = envelope_lowpass_hz / (sample_rate / 2)  # normalize to Nyquist
b_env, a_env = butter(4, wn_env, btype="lowpass")

# Use rectified signal you already computed:
# differential_rect = np.abs(differential_filt)
differential_env = filtfilt(b_env, a_env, differential_rect)

## =========================
## There are 2 diff ways to find ON/OFF thresholds: 1) Automatic baseline (uses lowest 20% of the signal), 2) Robust basline (uses 5 Sec widow anywhre with lowest activity)
# Automatic baseline from lowest percentile of the envelope-For ON/OFF threshold calculation
## =========================
# p = 20  # use lowest 20% (try 10, 15, 20)
# cut = np.percentile(differential_env, p)

# base_vals = differential_env[differential_env <= cut]

# baseline_mean = np.mean(base_vals)
# baseline_sd   = np.std(base_vals, ddof=1)

# ON_thr  = baseline_mean + 3 * baseline_sd
# OFF_thr = baseline_mean + 2 * baseline_sd

# print(f"Baseline (lowest {p}%) mean = {baseline_mean:.2f} µV")
# print(f"Baseline (lowest {p}%) SD   = {baseline_sd:.2f} µV")
# print(f"ON threshold  = {ON_thr:.2f} µV")
# print(f"OFF threshold = {OFF_thr:.2f} µV")
##=========================

# =========================
# Robust baseline: find quietest window anywhere
# =========================
win_s = 5.0  # seconds (try 5 or 10)
win_n = int(win_s * sample_rate)

# moving std of the envelope (cheap version)
env = differential_env
# Slide window across entire recording (overlapping windows)
step_s = 0.1  # step size in seconds (100 ms)
step_n = int(step_s * sample_rate)

mov_std = np.array([
    np.std(env[i:i+win_n])
    for i in range(0, len(env) - win_n, step_n)
])

best_idx = int(np.argmin(mov_std))
start_i = best_idx * step_n
end_i = start_i + win_n

base_vals = env[start_i:end_i]

baseline_mean = np.mean(base_vals)
baseline_sd   = np.std(base_vals, ddof=1)

ON_thr  = baseline_mean + 6 * baseline_sd
#OFF_thr = baseline_mean + 1 * baseline_sd

print(f"Quiet window baseline mean = {baseline_mean:.2f} µV")
print(f"Quiet window baseline SD   = {baseline_sd:.2f} µV")
print(f"ON threshold  = {ON_thr:.2f} µV")
#print(f"OFF threshold = {OFF_thr:.2f} µV")

# visualize where the baseline window was selected
plt.figure(figsize=(15,4))
plt.plot(timestamps, env, color='purple', label="Envelope")
plt.axvspan(timestamps[start_i], timestamps[end_i], alpha=0.2, label="Auto baseline window")
plt.axhline(ON_thr,  color='red',  linestyle='--', label="ON")
#plt.axhline(OFF_thr, color='blue', linestyle='--', label="OFF")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

## ---- (A) Envelope full trace plot ----
plt.figure(figsize=(15, 4))
plt.plot(timestamps, differential_env, label=f"EMG Envelope (LP {envelope_lowpass_hz} Hz)", color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"{os.path.basename(directory)} | {experiment_name}/{recording_name} | EMG Envelope (Full Trace)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (μV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# ===== Threshold Check Plot_ for ON threshold section =====
# ==========================================

fig_thr, ax_thr = plt.subplots(figsize=(15, 4))

ax_thr.plot(timestamps, differential_env,
            color='purple',
            label="Envelope")

ax_thr.axhline(ON_thr,
               color='red',
               linestyle='--',
               linewidth=2,
               label=f"ON threshold ({ON_thr:.1f} µV)")

# OFF threshold kept here for future use
# ax_thr.axhline(OFF_thr,
#                color='blue',
#                linestyle='--',
#                linewidth=2,
#                label=f"OFF threshold ({OFF_thr:.1f} µV)")

# ---- show ON threshold value on the figure ----
x_text = timestamps[0] + 0.02 * (timestamps[-1] - timestamps[0])

ax_thr.text(x_text, ON_thr, f"{ON_thr:.1f} µV",
            color='red',
            fontsize=10,
            va='bottom',
            ha='left',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

# ---- info box on top-right ----
file_info_text = (
    f"File: {os.path.basename(directory)}\n"
    f"{experiment_name}/{recording_name}\n"
    f"Envelope LP: {envelope_lowpass_hz} Hz"
)

ax_thr.text(0.98, 0.98, file_info_text,
            transform=ax_thr.transAxes,
            fontsize=10,
            va='top',
            ha='right',
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.85))

ax_thr.set_title("Envelope with ON Threshold")
ax_thr.set_xlabel("Time (s)")
ax_thr.set_ylabel("Amplitude (µV)")
ax_thr.grid(True)
ax_thr.legend()
plt.tight_layout()
plt.show()



# ---- (B) Envelope segmented viewer with Next/Previous buttons ----
segment_duration_env = 10  # seconds (you can change)
t0_env = float(timestamps[0])
t_end_env = float(timestamps[-1])
total_time_env = t_end_env - t0_env
num_segments_env = int(np.ceil(total_time_env / segment_duration_env))

state_env = {"idx": 0}

fig_env, ax_env = plt.subplots(figsize=(15, 4))
plt.subplots_adjust(bottom=0.25)  # space for buttons

line_env, = ax_env.plot([], [], color='purple',
                        label=f"Envelope (LP {envelope_lowpass_hz} Hz)")

# ON threshold line
on_line = ax_env.axhline(ON_thr,
                         color='red',
                         linestyle='--',
                         linewidth=2,
                         label=f"ON threshold ({ON_thr:.1f} µV)")

# OFF threshold kept here for future use
# off_line = ax_env.axhline(OFF_thr,
#                           color='blue',
#                           linestyle='--',
#                           linewidth=2,
#                           label=f"OFF threshold ({OFF_thr:.1f} µV)")

ax_env.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax_env.set_xlabel("Time within segment (s)")
ax_env.set_ylabel("Amplitude (μV)")
ax_env.grid(True)
ax_env.legend()
title_env = ax_env.set_title("")

# ON threshold text
on_text = ax_env.text(0, ON_thr, f"{ON_thr:.1f} µV",
                      color='red',
                      fontsize=10,
                      va='bottom',
                      ha='left',
                      bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))

# ---- info box on top-right ----
info_box_env = ax_env.text(
    0.98, 0.98, "",
    transform=ax_env.transAxes,
    fontsize=10,
    va='top',
    ha='right',
    bbox=dict(facecolor='white', edgecolor='black', alpha=0.85)
)

def update_plot_env():
    i = state_env["idx"]
    start_time = t0_env + i * segment_duration_env
    end_time = min(start_time + segment_duration_env, t_end_env)
    mask = (timestamps >= start_time) & (timestamps < end_time)

    x = timestamps[mask] - start_time
    y = differential_env[mask]

    line_env.set_data(x, y)
    ax_env.set_xlim(0, end_time - start_time)

    # autoscale y per segment while keeping ON threshold visible
    if y.size > 0:
        ypad = 0.05 * (np.max(y) - np.min(y) + 1e-9)
        ymin = min(np.min(y) - ypad, ON_thr - ypad)
        ymax = max(np.max(y) + ypad, ON_thr + ypad)
        ax_env.set_ylim(ymin, ymax)

    # move ON threshold text slightly inside the segment
    x_text_seg = 0.02 * (end_time - start_time)
    on_text.set_position((x_text_seg, ON_thr))

    # title
    title_env.set_text(
        f"ENVELOPE VIEW | Segment {i+1}/{num_segments_env} | {start_time:.1f}-{end_time:.1f}s"
    )

    # info box
    info_box_env.set_text(
        f"File: {os.path.basename(directory)}\n"
        f"{experiment_name}/{recording_name}\n"
        f"LP: {envelope_lowpass_hz} Hz\n"
        f"ON thr: {ON_thr:.1f} µV"
    )

    fig_env.canvas.draw_idle()


def next_segment_env(event):
    if state_env["idx"] < num_segments_env - 1:
        state_env["idx"] += 1
        update_plot_env()


def prev_segment_env(event):
    if state_env["idx"] > 0:
        state_env["idx"] -= 1
        update_plot_env()


axprev_env = plt.axes([0.30, 0.08, 0.15, 0.10])
axnext_env = plt.axes([0.55, 0.08, 0.15, 0.10])

bprev_env = Button(axprev_env, "Previous")
bnext_env = Button(axnext_env, "Next")
bprev_env.on_clicked(prev_segment_env)
bnext_env.on_clicked(next_segment_env)

update_plot_env()
plt.show()

# ==========================================
# ===== FIGURE 4 (Optional): Show baseline points distribution
# ==========================================
# plt.figure(figsize=(6, 4))
# plt.hist(differential_env, bins=80)
# plt.axvline(cut, linestyle='--', linewidth=2, label=f"{p}th percentile cutoff")
# plt.title("Envelope Value Distribution\n(dashed line = baseline cutoff)")
# plt.xlabel("Envelope amplitude (µV)")
# plt.ylabel("Count")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # # ==== Load MessageCenter ====
# messagecenter_dir = os.path.join(directory, record_node_name, experiment_name, recording_name, "events", "MessageCenter")
# if not os.path.exists(messagecenter_dir):
# #Repleaced print with the other line below-Pariya
# #    print(f"MessageCenter directory not found for {exp_name}/{yy}")
#     print(f"MessageCenter directory not found for {experiment_name}/{recording_name}")

# texts = np.load(os.path.join(messagecenter_dir, "text.npy"), allow_pickle=True)
# timestamps_msg = np.load(os.path.join(messagecenter_dir, "timestamps.npy"))
# decoded_texts = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in texts]
# message_entries = list(zip(timestamps_msg, decoded_texts))
# print(f"Loaded {len(decoded_texts)} MessageCenter entries")

# # ==== Debug Print ====
# for text, time in zip(decoded_texts, timestamps_msg):
#     print(f"[Time: {time:.6f} s] Message: {text}")
    
