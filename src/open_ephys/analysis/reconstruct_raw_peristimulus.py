"""
reconstruct_raw_peristimulus.py

Offline reconstruction of RAW (pre-filter) peristimulus EMG windows from any
of this app's stage recordings -- lets you replicate the app's own
peristimulus plots exactly, using your own filtering instead of the app's,
without needing the live app or a booth config.

WORKS FOR: Frequency Test (.hrft), Control Mode (.hrs2), Up/Down Condition
Pellet (.hrs4/.hrs3), Up/Down Condition VNS (.hrs6/.hrs5), Mh Recruitment
Curve (.hrs1), and Conditioning Mode (.hrscm) -- see STAGE_REGISTRY below.
These 7 all share the identical trial/header layout (trial_data,
unipolar_trial_data, stim_adc_data, onset_sample_index, digital_onset_*,
trigger_wall_time_ms, filtering_protocol, booth_snapshot), so one
reconstruction/alignment implementation covers all of them.

DOES NOT WORK FOR: EMG Characterization (.hrs0) -- that stage uses a
completely different trial schema (monitored_signal/bins/grand_mean, no
onset_sample_index or digital-sync detection at all) because it never
involves a stimulator; there's no "stim onset" concept to align to.

WHY THIS EXISTS
----------------
The app records a session's raw per-channel data continuously throughout the
whole session (HReflexDataFileEmgData blocks, one per incoming frame) AND a
per-trial "trial_data"/"unipolar_trial_data" window that's already been run
through the app's own bandpass filter (see EmgDataFilter). If you want to
apply a DIFFERENT filter and get a directly comparable peristimulus plot, you
need the RAW version of that exact same window -- which isn't saved per trial,
only in the continuous stream. This module finds it for you.

THE HARD PART: alignment
-------------------------
The continuous HReflexDataFileEmgData blocks do NOT carry an Open Ephys
sample number (only wall-clock-ish timestamps) -- so there's no direct
arithmetic mapping from a trial's `first_post_trigger_frame_sample_id` (an OE
sample number) to a position in the reconstructed continuous array. Instead,
this module narrows the search using wall-clock proximity to
`trial.trigger_wall_time_ms`, then finds the EXACT alignment by cross-
correlating a zero-phase-filtered version of that narrowed raw window (see
locate_trial_onset()) against `trial.trial_data` (which the app already
computed and saved -- so it's a known-correct fingerprint of the right
window). This is robust to any dropped frames, since it's anchored to actual
recorded content rather than assumed-gapless sample counting.

TIME-ALIGNMENT LOGIC THIS REPRODUCES (see e.g. frequency_test_stage.py,
identical in every stage listed above)
--------------------------------------------------------------------------
1. Trial onset = the first digital edge on the booth's sync DIGIN line,
   captured after trigger_single() was sent (process(), TRIAL_STATE_RECORD).
2. Converted to a sample index via:
       offset = digital_sync_event.sample_num - first_post_trigger_frame_sample_id
       onset_sample_index = bin_sample_count() + offset
   (_compute_trial_debug_fields). This module doesn't need to redo this
   arithmetic -- trial.onset_sample_index is already the answer, saved per
   trial; what's missing is only the RAW signal to slice around it.
3. (Frequency Test only, multi-pulse trains) Per-pulse onsets are refined by
   searching the recorded Stim ADC channel near each pulse's naive expected
   position for the first sample >= STIM_ONSET_THRESHOLD (4.5V)
   (_detect_pulse_onset_indices). Also already computed and saved per trial
   as trial.pulse_onset_sample_indices -- Stim ADC is never filtered by the
   app, so trial.stim_adc_data IS ALREADY the raw window; no reconstruction
   needed for it (see raw_stim_adc_window() below). The other 6 stages only
   ever deliver a single pulse per trial, so there's nothing to refine --
   trial.onset_sample_index is the whole answer.

THE FILTER THIS REPRODUCES (see emg_data_filter.py)
------------------------------------------------------
EmgDataFilter (2nd-order Butterworth bandpass, 100-1000 Hz) runs in two
distinct modes, and this module only reproduces one of them:

- LIVE/continuous stream (Live EMG plot, real-time initiation-threshold
  gating): causal (scipy sosfilt), stateful, filtered frame-by-frame as
  data arrives. This module does NOT reproduce this mode -- it's only
  relevant to the live app's real-time decisions, not to what ends up in
  trial.trial_data, and the continuous HReflexDataFileEmgData blocks this
  module reconstructs from (emg_data_raw) are RAW anyway, not filtered.

- Per-trial peristimulus window (trial.trial_data/unipolar_trial_data):
  zero-phase (scipy sosfiltfilt), stateless, applied ONCE to a raw window
  padded by EmgDataFilter.TRIAL_WINDOW_PAD_MS (200 ms, see
  EmgFilterConfigSnapshot.trial_window_pad_ms in the file's own header
  when present) beyond the trial's real pre/post-onset range, then trimmed
  back down to that real range. THIS is what this module reproduces (see
  locate_trial_onset() and _app_reconstruct_window()) -- filter the SAME
  padded window the app used, then trim the SAME padding off, and the
  result matches trial.trial_data to floating-point precision. Filtering a
  differently-sized window (e.g. the whole multi-second rough search
  window used just for alignment) does NOT bit-match, since sosfiltfilt's
  edge padding/reflection depends on the exact window it's given -- that's
  why locate_trial_onset() only uses its own wide filtered window to FIND
  the onset, and _app_reconstruct_window() re-filters the tight,
  correctly-padded window for the actual reproduction.

Same filter, same code, regardless of which of the 7 stages recorded the
file.

USAGE
-----
    from reconstruct_raw_peristimulus import load_session

    sess = load_session("TESTSUBJ_20260101T000000.hrs4")   # auto-detects
                                                             # Up Condition Pellet
                                                             # from the extension
    trial = sess.data_file.trials[0]

    # Raw (unfiltered) peristimulus window -- apply ANY filter you want to this.
    t_ms, raw_bipolar_diff = sess.raw_peristimulus_window(trial, pre_ms=20, post_ms=60)
    my_filtered = my_own_filter_function(raw_bipolar_diff)

    # Stim ADC needs no reconstruction -- already raw, already saved per trial.
    t_ms_adc, raw_stim_adc = sess.raw_stim_adc_window(trial, pre_ms=20, post_ms=60)

    # Sanity check: reproduce the app's OWN filter and confirm it lines up
    # with what the app itself saved (trial.trial_data).
    sess.validate_against_saved_trial(trial)

Only Frequency Test files carry a per-trial sample rate (added partway
through that stage's file-format history) -- for the other 6 stage types,
pass sample_rate explicitly: load_session(path, sample_rate=30000.0).

See __main__ below for a runnable end-to-end example with plotting.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import correlate

# ── Make the app's own model classes importable ─────────────────────────────
# Reused directly (not reimplemented) so the binary format, filter math, and
# onset-detection fields can never silently drift from what the app actually
# does. Adjust REPO_SRC if you run this script from somewhere else.
REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from hreflex_txbdc.model.datafiles.h_reflex_data_file_shared import (  # noqa: E402
    HReflexDataFileEmgData, HReflexBoothConfigSnapshot,
)
from hreflex_txbdc.model.emg_data_filter import EmgDataFilter          # noqa: E402
from hreflex_txbdc.model.application_configuration import ApplicationConfiguration  # noqa: E402


# ── Stage registry ───────────────────────────────────────────────────────────
# All 7 entries share the identical trial/header layout this module relies on
# (see module docstring). EMG Characterization (.hrs0) is deliberately absent
# -- different schema, no stim-onset concept at all.

@dataclass(frozen=True)
class StageFileSpec:
    description: str
    module: str          # hreflex_txbdc.model.datafiles.<module>
    data_file_cls: str   # class name within that module
    supports_pulse_trains: bool = False   # only Frequency Test does


STAGE_REGISTRY: dict[str, StageFileSpec] = {
    "hrft":  StageFileSpec("Frequency Test",         "frequency_test_data_file",        "FrequencyTestDataFile", supports_pulse_trains=True),
    "hrs2":  StageFileSpec("Control Mode",           "control_mode_data_file",          "ControlModeDataFile"),
    "hrs4":  StageFileSpec("Up Condition Pellet",    "up_condition_pellet_data_file",   "UpConditionPelletDataFile"),
    "hrs3":  StageFileSpec("Down Condition Pellet",  "down_condition_pellet_data_file", "DownConditionPelletDataFile"),
    "hrs6":  StageFileSpec("Up Condition VNS",       "up_condition_vns_data_file",      "UpConditionVnsDataFile"),
    "hrs5":  StageFileSpec("Down Condition VNS",     "down_condition_vns_data_file",    "DownConditionVnsDataFile"),
    "hrs1":  StageFileSpec("Mh Recruitment Curve",   "mh_recruitment_curve_data_file",  "MhRecruitmentCurveDataFile"),
    "hrscm": StageFileSpec("Conditioning Mode",      "conditioning_mode_data_file",     "ConditioningModeDataFile"),
}


def _load_data_file(file_path: str) -> tuple[Any, StageFileSpec]:
    ext = Path(file_path).suffix.lstrip(".").lower()
    spec = STAGE_REGISTRY.get(ext)
    if spec is None:
        raise ValueError(
            f"Unrecognized/unsupported extension '.{ext}'. Supported: "
            f"{', '.join('.' + e for e in STAGE_REGISTRY)}. "
            f"(.hrs0 EMG Characterization files are not supported -- see module docstring.)")

    import importlib
    mod = importlib.import_module(f"hreflex_txbdc.model.datafiles.{spec.module}")
    data_file_cls = getattr(mod, spec.data_file_cls)

    data_file = data_file_cls()
    with open(file_path, "rb") as fid:
        data_file.read(fid)
    return data_file, spec


# ── Channel-role resolution ──────────────────────────────────────────────────

# HReflexDataFileEmgData only saves each channel's NAME (e.g. "CH3"), not its
# numeric index -- Open Ephys channel names are expected to end in the
# absolute (1-based) channel number, matching Booth's own ch_* numbering. If
# your recording uses different channel naming, pass name_to_index_overrides
# to load_session()/ReconstructedTrialSession instead of relying on this parser.
_CHANNEL_NUMBER_RE = re.compile(r"(\d+)\s*$")


def _parse_channel_index_from_name(channel_name: str) -> int | None:
    m = _CHANNEL_NUMBER_RE.search(channel_name)
    return int(m.group(1)) if m else None


@dataclass
class ChannelRoleMap:
    """Absolute (1-based) channel index for each role, read from the file's
    own booth_snapshot header field (added partway through each stage's file-
    format history) -- no live app/booth config needed."""
    unipolar_emg: int = 0
    bipolar_emg: int = 0
    sync_analog: int = 0
    stim_adc: int = 0
    sync_digital: int = 0   # a DIGIN line number, not an ADC channel index

    @staticmethod
    def from_snapshot(snap: HReflexBoothConfigSnapshot) -> "ChannelRoleMap":
        def _role(role_list, role_name) -> int:
            for role, idx in role_list:
                if role == role_name:
                    return idx
            return 0
        return ChannelRoleMap(
            unipolar_emg=snap.ch_emg_data_1,
            bipolar_emg=snap.ch_emg_data_2,
            sync_analog=_role(snap.analog_in_roles, "Sync (analog)"),
            stim_adc=_role(snap.analog_in_roles, "Stim ADC"),
            sync_digital=_role(snap.digital_in_roles, "Sync (digital)"),
        )


# ── Session loading / reconstruction ────────────────────────────────────────

class ReconstructedTrialSession:
    """
    Loads any of this app's trial-based recordings (see STAGE_REGISTRY) and
    reconstructs continuous raw per-role signals from its HReflexDataFileEmgData
    blocks, ready to filter however you like and slice into peristimulus
    windows aligned exactly like the app's own trial plots.

    Prefer load_session() over constructing this directly -- it auto-detects
    which stage type a file is from its extension.
    """

    def __init__(self, file_path: str, sample_rate: float | None = None,
                 name_to_index_overrides: dict[str, int] | None = None,
                 verbose: bool = True):
        self.file_path = file_path
        self.verbose = verbose

        self.data_file, self.stage_spec = _load_data_file(file_path)
        self._log(f"[INFO] Loaded as {self.stage_spec.description} "
                  f"({self.stage_spec.data_file_cls}), file_version="
                  f"{self.data_file.header.file_version}")

        header = self.data_file.header
        booth_snapshot = getattr(header, 'booth_snapshot', None)
        if booth_snapshot is None:
            self._log("[WARN] This file predates the booth_snapshot field -- channel-role "
                      "mapping will be empty. Pass name_to_index_overrides explicitly instead.")
            booth_snapshot = HReflexBoothConfigSnapshot()

        filtering_protocol = getattr(header, 'filtering_protocol', "")
        if not filtering_protocol:
            self._log("[WARN] This file predates the filtering_protocol field -- can't confirm "
                      "this session used OFFLINE FILTERING (the app's own bandpass filter). If it "
                      "was ONLINE FILTERING, trial.trial_data was never passed through "
                      "EmgDataFilter, and validate_against_saved_trial() below will not match.")
        elif filtering_protocol == "ONLINE":
            self._log("[WARN] This session used ONLINE FILTERING -- the app did not apply its own "
                      "bandpass filter (it trusted Open Ephys' own filtering/differencing "
                      "upstream). trial.trial_data is NOT EmgDataFilter's output for this file, "
                      "so validate_against_saved_trial() will not match. The raw channels "
                      "reconstructed here may themselves already be filtered/differenced by Open "
                      "Ephys, depending on your signal chain -- see the ZMQ Interface plugin "
                      "placement.")

        self.roles = ChannelRoleMap.from_snapshot(booth_snapshot)

        #Only Frequency Test trials carry their own sample_rate (added
        #partway through that stage's file-format history). Every other
        #stage's header/trial never recorded it at all -- must be supplied.
        self.sample_rate: float | None = sample_rate
        if self.sample_rate is None:
            self.sample_rate = next(
                (getattr(t, 'sample_rate', 0.0) for t in self.data_file.trials
                 if getattr(t, 'sample_rate', 0.0) > 0), None)
        if self.sample_rate is None:
            raise ValueError(
                "No trial in this file carries its own sample_rate, and none was supplied. Pass "
                "sample_rate=<Hz> to load_session()/ReconstructedTrialSession explicitly -- only "
                "Frequency Test files ever store this themselves.")

        self._filtering_protocol = filtering_protocol
        self._channels: dict[str, np.ndarray] = {}
        self._block_wall_ms: np.ndarray = np.zeros(0)
        self._block_boundaries: np.ndarray = np.zeros(1, dtype=np.int64)
        self._build_continuous_channels(name_to_index_overrides or {})

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def set_sample_rate(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate

    # ── Continuous-channel reconstruction ────────────────────────────────────

    def _build_continuous_channels(self, name_to_index_overrides: dict[str, int]) -> None:
        blocks: list[HReflexDataFileEmgData] = self.data_file.emg_data_blocks
        if not blocks:
            raise ValueError("This file has no continuous EMG data blocks to reconstruct from.")

        names = blocks[0].emg_channel_names
        name_to_index: dict[str, int] = dict(name_to_index_overrides)
        unresolved = []
        for name in names:
            if name in name_to_index:
                continue
            idx = _parse_channel_index_from_name(name)
            if idx is None:
                unresolved.append(name)
            else:
                name_to_index[name] = idx

        self._log(f"[INFO] Channel names found in file: {names}")
        self._log(f"[INFO] Parsed name -> channel index: {name_to_index}")
        if unresolved:
            self._log(f"[WARN] Could not parse a channel index from name(s): {unresolved} -- "
                      f"pass name_to_index_overrides={{name: index, ...}} for these.")

        self._log(f"[INFO] Role -> channel index (from booth_snapshot): "
                  f"unipolar_emg={self.roles.unipolar_emg} bipolar_emg={self.roles.bipolar_emg} "
                  f"sync_analog={self.roles.sync_analog} stim_adc={self.roles.stim_adc} "
                  f"sync_digital(DIGIN)={self.roles.sync_digital}")

        wanted_index = {
            'unipolar':    self.roles.unipolar_emg,
            'bipolar':     self.roles.bipolar_emg,
            'stim_adc':    self.roles.stim_adc,
            'sync_analog': self.roles.sync_analog,
        }
        index_to_name = {v: k for k, v in name_to_index.items()}
        position_by_name = {name: i for i, name in enumerate(names)}

        role_position: dict[str, int] = {}
        for role, idx in wanted_index.items():
            if idx <= 0:
                continue  # role not wired for this booth
            name = index_to_name.get(idx)
            if name is None:
                self._log(f"[WARN] Role '{role}' (channel {idx}) has no matching channel name in "
                          f"this file -- was it actually wired/recorded for this session?")
                continue
            role_position[role] = position_by_name[name]

        lengths = np.array(
            [len(b.emg_data_raw[0]) if b.emg_data_raw else 0 for b in blocks], dtype=np.int64)
        self._block_boundaries = np.concatenate(([0], np.cumsum(lengths)))
        self._block_wall_ms = np.array(
            [b.timestamp_millis_background_emitted for b in blocks], dtype=np.float64)

        for role, pos in role_position.items():
            self._channels[role] = np.concatenate([b.emg_data_raw[pos] for b in blocks])

        total_samples = int(self._block_boundaries[-1])
        duration_s = total_samples / self.sample_rate
        self._log(f"[INFO] Reconstructed {len(role_position)} channel(s) over {total_samples} "
                  f"samples (~{duration_s:.1f}s) from {len(blocks)} continuous blocks.")

    def raw(self, role: str) -> np.ndarray:
        """role is one of 'unipolar', 'bipolar', 'stim_adc', 'sync_analog'."""
        if role not in self._channels:
            raise KeyError(
                f"Role '{role}' wasn't reconstructed -- either not wired for this booth, or its "
                f"channel name wasn't found in this file. See the [INFO]/[WARN] lines printed "
                f"at load time.")
        return self._channels[role]

    def raw_diff(self) -> np.ndarray:
        """Bipolar - unipolar, matching open_ephys_streamer.py's diff_data_block
        (raw1 - raw0) -- the signal that feeds EmgDataFilter.filter()."""
        return self.raw('bipolar') - self.raw('unipolar')

    def _trial_window_pad_samples(self) -> int:
        """
        The extra raw context (samples, each side) the app grabs around a
        trial's real pre/post-onset window before filtering, then trims
        off -- read from the file's OWN header (EmgFilterConfigSnapshot,
        see h_reflex_data_file_shared.py) when present, so this module
        stays correct even if EmgDataFilter.TRIAL_WINDOW_PAD_MS changes in
        the app after this file was recorded. Falls back to the currently-
        installed code's constant for files that predate that header field.
        """
        filter_config = getattr(self.data_file.header, 'filter_config', None)
        pad_ms = getattr(filter_config, 'trial_window_pad_ms', 0.0) if filter_config else 0.0
        if not pad_ms:
            pad_ms = EmgDataFilter.TRIAL_WINDOW_PAD_MS
            self._log(f"[WARN] This file predates the filter_config header field -- assuming the "
                      f"currently-installed EmgDataFilter.TRIAL_WINDOW_PAD_MS ({pad_ms} ms). If the "
                      f"app's padding has changed since this file was recorded, reconstruction will "
                      f"be off.")
        return int(round(pad_ms * self.sample_rate / 1000.0))

    # ── Trial alignment ──────────────────────────────────────────────────────

    def _rough_search_bounds(self, trial: Any, margin_ms: float) -> tuple[int, int]:
        """Narrows the search to continuous blocks within margin_ms of this
        trial's trigger_wall_time_ms, so alignment doesn't have to correlate
        against the entire session for every trial."""
        target = float(trial.trigger_wall_time_ms)
        wall_ms = self._block_wall_ms
        lo = int(np.searchsorted(wall_ms, target - margin_ms, side='left'))
        hi = int(np.searchsorted(wall_ms, target + margin_ms, side='right'))
        lo = max(0, min(lo, len(self._block_boundaries) - 2))
        hi = max(lo + 1, min(hi, len(self._block_boundaries) - 1))
        return int(self._block_boundaries[lo]), int(self._block_boundaries[hi])

    def _app_filter(self) -> EmgDataFilter:
        #EmgDataFilter.initialize_filter() reads ApplicationConfiguration.sample_rate
        #(a class-level config value, same as the live app uses) to build its
        #Butterworth coefficients -- set it to this FILE's actual rate first.
        ApplicationConfiguration.sample_rate = self.sample_rate
        f = EmgDataFilter()
        f.initialize_filter()
        return f

    def locate_trial_onset(self, trial: Any, reference: str = 'bipolar',
                            margin_ms: float = 8000.0, probe_len: int = 2000) -> int:
        """
        Returns the sample index, IN THE RECONSTRUCTED CONTINUOUS RAW
        STREAM, that corresponds to trial.onset_sample_index in
        trial.trial_data (or trial.unipolar_trial_data if
        reference='unipolar').

        Finds it by zero-phase-filtering a wall-clock-narrowed raw window
        (via EmgDataFilter.filter_window()/filter_window_unipolar(), the
        SAME zero-phase mode the app itself uses for trial_data -- see
        module docstring) and cross-correlating it against trial.trial_data
        (a known-correct fingerprint the app already computed). This filtered
        window is wider than what the app itself used for THIS trial (it's
        margin_ms wide, for search purposes), so its values won't bit-match
        trial_data near ITS OWN edges -- that's fine here, correlation only
        needs local similarity to find the right offset. Exact reproduction
        (bit-matching trial_data) happens in _app_reconstruct_window(),
        which re-filters the tight, correctly-padded window once onset is
        known.
        """
        if reference not in ('unipolar', 'bipolar'):
            raise ValueError("reference must be 'unipolar' or 'bipolar'")
        if trial.onset_sample_index < 0:
            raise ValueError(
                "This trial has no detected onset (onset_sample_index < 0, onset_detected="
                f"{trial.onset_detected}) -- nothing to align to.")

        lo, hi = self._rough_search_bounds(trial, margin_ms)
        raw_window = (self.raw_diff() if reference == 'bipolar' else self.raw('unipolar'))[lo:hi]
        filt = self._app_filter()
        filtered_window = (filt.filter_window(raw_window) if reference == 'bipolar'
                            else filt.filter_window_unipolar(raw_window))

        key = np.asarray(
            trial.trial_data if reference == 'bipolar' else trial.unipolar_trial_data,
            dtype=np.float64)
        probe = key[: min(len(key), probe_len)]
        if len(probe) < 8:
            raise ValueError("This trial's saved data is too short to use as an alignment probe.")
        if len(filtered_window) < len(probe):
            raise ValueError(
                f"Search window ({len(filtered_window)} samples) is shorter than the alignment "
                f"probe ({len(probe)} samples) -- increase margin_ms.")

        corr = correlate(filtered_window.astype(np.float64), probe.astype(np.float64), mode='valid')
        best = int(np.argmax(corr))

        candidate = filtered_window[best: best + len(probe)]
        denom = float(np.max(np.abs(probe))) + 1e-9
        rel_err = float(np.max(np.abs(candidate - probe))) / denom
        if rel_err > 0.05:
            self._log(
                f"[WARN] Trial at {trial.start_time}: alignment match is imperfect "
                f"(max relative error {rel_err:.3%} over the first {len(probe)} samples). "
                f"Try a larger margin_ms, or confirm self.sample_rate ({self.sample_rate}) "
                f"matches what this session actually recorded at. Some mismatch here is expected "
                f"-- this search window's own edges don't match the app's tighter, correctly-padded "
                f"window (see _app_reconstruct_window()); only a LARGE error indicates real "
                f"misalignment.")

        trial_window_start_in_continuous = lo + best
        return trial_window_start_in_continuous + trial.onset_sample_index

    def _app_reconstruct_window(self, reference: str, onset: int, pre: int, post: int,
                                 pad: int | None = None) -> np.ndarray:
        """
        Zero-phase-reconstructs a peristimulus window the SAME way the app
        computes trial_data/unipolar_trial_data: raw window padded by `pad`
        samples on both sides (default _trial_window_pad_samples(), the
        app's own padding), filtered ONCE with filter_window()/
        filter_window_unipolar(), padding trimmed back off. When pre/post
        exactly match a trial's own onset_sample_index/window length, this
        matches trial.trial_data/unipolar_trial_data to floating-point
        precision (see validate_against_saved_trial()); for other pre/post
        values it's a general-purpose zero-phase reconstruction of whatever
        window you asked for, just not a bit-exact crop of a saved trial.
        """
        if pad is None:
            pad = self._trial_window_pad_samples()
        final_len = pre + post

        sig = self.raw_diff() if reference == 'bipolar' else self.raw('unipolar')
        window_start = onset - pre - pad
        window_end = onset + post + pad
        if window_start < 0 or window_end > len(sig):
            raise ValueError(
                f"Padded raw window [{window_start}:{window_end}] falls outside the reconstructed "
                f"continuous stream (0:{len(sig)}) -- this trial/window is too close to the "
                f"start/end of the recorded session to reproduce exactly.")

        filt = self._app_filter()
        raw_padded = sig[window_start:window_end]
        filtered_padded = (filt.filter_window(raw_padded) if reference == 'bipolar'
                            else filt.filter_window_unipolar(raw_padded))
        return filtered_padded[pad:pad + final_len]

    # ── Peristimulus extraction ──────────────────────────────────────────────

    def raw_peristimulus_window(self, trial: Any, pre_ms: float, post_ms: float,
                                 reference: str = 'bipolar', **locate_kwargs
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (t_ms, raw_signal) for the RAW (pre-filter) peristimulus
        window around this trial's stim onset -- t_ms=0 at onset, matching
        the app's own trial plots. Apply your own filter to raw_signal.

        Tip: request a bit more pre_ms than you actually intend to plot/filter
        against (e.g. +200-500ms) if your own filter is also a stateful IIR
        design, so IT also has time to settle before the window you actually
        care about -- then trim after filtering.
        """
        onset = self.locate_trial_onset(trial, reference=reference, **locate_kwargs)
        sig = self.raw_diff() if reference == 'bipolar' else self.raw('unipolar')
        return self._slice_window(sig, onset, pre_ms, post_ms)

    def app_filtered_peristimulus_window(self, trial: Any, pre_ms: float, post_ms: float,
                                          reference: str = 'bipolar', **locate_kwargs
                                          ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (t_ms, filtered_signal) for a zero-phase-filtered
        peristimulus window of your requested size, reconstructed the same
        way the app computes trial_data (see _app_reconstruct_window()) --
        useful for a direct side-by-side against your own filtering, without
        re-deriving the app's filter yourself. When pre_ms/post_ms happen to
        match this trial's own saved window exactly (pre_ms ==
        trial.onset_sample_index converted to ms, post_ms == the rest),
        the result matches trial.trial_data to floating-point precision --
        see validate_against_saved_trial() for that exact check.
        """
        onset = self.locate_trial_onset(trial, reference=reference, **locate_kwargs)
        pre = int(round(pre_ms * self.sample_rate / 1000.0))
        post = int(round(post_ms * self.sample_rate / 1000.0))
        window = self._app_reconstruct_window(reference, onset, pre, post)
        t_ms = (np.arange(pre + post) - pre) * (1000.0 / self.sample_rate)
        return t_ms, window

    def raw_stim_adc_window(self, trial: Any, pre_ms: float, post_ms: float
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Stim ADC is never filtered by the app (unlike the EMG channels) --
        trial.stim_adc_data IS ALREADY the raw peristimulus window, indexed
        by trial.onset_sample_index. No continuous-stream reconstruction or
        alignment needed."""
        if trial.onset_sample_index < 0:
            raise ValueError("This trial has no detected onset -- nothing to center the window on.")
        return self._slice_window(trial.stim_adc_data, trial.onset_sample_index, pre_ms, post_ms)

    def pulse_onset_offsets_ms(self, trial: Any) -> np.ndarray:
        """Per-pulse onset times (ms, relative to trial.onset_sample_index),
        exactly as the app computed them via Stim ADC threshold search
        (_detect_pulse_onset_indices) -- already saved per trial. Frequency
        Test only; every other stage in STAGE_REGISTRY delivers a single
        pulse per trial (trial.onset_sample_index is already the whole
        answer for those)."""
        if not self.stage_spec.supports_pulse_trains:
            raise NotImplementedError(
                f"{self.stage_spec.description} trials only ever have a single pulse -- "
                f"there's nothing to refine. Use trial.onset_sample_index directly.")
        if len(getattr(trial, 'pulse_onset_sample_indices', [])) == 0:
            raise ValueError(
                "This trial has no saved per-pulse onsets (file predates the field, or this "
                "trial's onset was never detected).")
        return ((np.asarray(trial.pulse_onset_sample_indices, dtype=np.float64)
                 - trial.onset_sample_index) * 1000.0 / self.sample_rate)

    def _slice_window(self, sig: np.ndarray, center: int, pre_ms: float, post_ms: float
                       ) -> tuple[np.ndarray, np.ndarray]:
        pre_smp = int(round(pre_ms * self.sample_rate / 1000.0))
        post_smp = int(round(post_ms * self.sample_rate / 1000.0))
        start = max(0, center - pre_smp)
        end = min(len(sig), center + post_smp)
        window = sig[start:end]
        t_ms = (np.arange(start, end) - center) * (1000.0 / self.sample_rate)
        return t_ms, window

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_against_saved_trial(self, trial: Any, reference: str = 'bipolar',
                                      **locate_kwargs) -> float:
        """
        Reproduces trial.trial_data/unipolar_trial_data exactly (see
        _app_reconstruct_window() -- the trial's own real pre/post-onset
        window, padded and filtered the same way the app did) and compares
        against what the app actually saved. Returns the max relative error
        found. Only meaningful for OFFLINE FILTERING sessions (see the
        [WARN] printed at load time otherwise) -- prints a PASS/FAIL summary
        either way.
        """
        onset = self.locate_trial_onset(trial, reference=reference, **locate_kwargs)
        key = np.asarray(
            trial.trial_data if reference == 'bipolar' else trial.unipolar_trial_data,
            dtype=np.float64)
        pre = trial.onset_sample_index
        post = len(key) - pre
        reproduced = self._app_reconstruct_window(reference, onset, pre, post)
        n = min(len(reproduced), len(key))
        denom = float(np.max(np.abs(key[:n]))) + 1e-9
        rel_err = float(np.max(np.abs(reproduced[:n].astype(np.float64) - key[:n]))) / denom
        status = "PASS" if rel_err < 0.01 else "FAIL"
        self._log(f"[{status}] validate_against_saved_trial: max relative error "
                  f"{rel_err:.4%} over {n} samples.")
        return rel_err


def load_session(file_path: str, sample_rate: float | None = None,
                  name_to_index_overrides: dict[str, int] | None = None,
                  verbose: bool = True) -> ReconstructedTrialSession:
    """Auto-detects the stage type from the file's extension (see
    STAGE_REGISTRY) and returns a ReconstructedTrialSession for it. Pass
    sample_rate explicitly for every stage except Frequency Test."""
    return ReconstructedTrialSession(
        file_path, sample_rate=sample_rate,
        name_to_index_overrides=name_to_index_overrides, verbose=verbose)


# ── Runnable example ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file_path", help="Path to a session file (see STAGE_REGISTRY for supported extensions)")
    parser.add_argument("--trial", type=int, default=-1, help="Trial index to plot (default: last)")
    parser.add_argument("--pre-ms", type=float, default=20.0)
    parser.add_argument("--post-ms", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=float, default=None,
                         help="Required for every stage except Frequency Test")
    args = parser.parse_args()

    sess = load_session(args.file_path, sample_rate=args.sample_rate)
    trial = sess.data_file.trials[args.trial]

    extra = (f" condition={trial.condition} n_pulses={trial.n_pulses}"
             if sess.stage_spec.supports_pulse_trains else "")
    print(f"\nTrial{extra} onset_detected={trial.onset_detected} "
          f"onset_sample_index={trial.onset_sample_index}")

    # 1) Sanity check: does reproducing the app's own filter match what it saved?
    sess.validate_against_saved_trial(trial)

    # 2) Get the RAW window and apply your own filter -- this example just
    # reuses a differently-configured Butterworth as a stand-in for "your own
    # filtering method"; swap this out for whatever you actually want to try.
    t_ms, raw_window = sess.raw_peristimulus_window(trial, pre_ms=args.pre_ms + 300, post_ms=args.post_ms)

    from scipy.signal import butter, sosfiltfilt
    my_sos = butter(4, [30, 2000], btype='bandpass', output='sos', fs=sess.sample_rate)
    my_filtered = sosfiltfilt(my_sos, raw_window)  # different band/order from the app's filter, as an example

    # Trim off the extra pre-roll now that filtering (yours or the app's) has settled.
    trim = int(round(300 * sess.sample_rate / 1000.0))
    t_ms, raw_window, my_filtered = t_ms[trim:], raw_window[trim:], my_filtered[trim:]

    # 3) A zero-phase-filtered window reconstructed the same way the app
    # computes trial_data, for a direct side-by-side against your own
    # filtering (see validate_against_saved_trial() above for the exact,
    # bit-matching comparison against trial.trial_data itself).
    t_ms_app, app_filtered = sess.app_filtered_peristimulus_window(trial, pre_ms=args.pre_ms, post_ms=args.post_ms)

    t_ms_adc, raw_stim_adc = sess.raw_stim_adc_window(trial, pre_ms=args.pre_ms, post_ms=args.post_ms)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(t_ms, raw_window, color='gray', linewidth=0.8)
        axes[0].set_title("Raw (unfiltered) bipolar EMG")
        axes[1].plot(t_ms, my_filtered, color='tab:blue', linewidth=1.2)
        axes[1].set_title("Your own filter applied offline")
        axes[2].plot(t_ms_app, app_filtered, color='tab:green', linewidth=1.2)
        axes[2].set_title("App-equivalent filter (reconstructed)")
        axes[3].plot(t_ms_adc, raw_stim_adc, color='tab:orange', linewidth=1.0)
        axes[3].set_title("Stim ADC (raw -- never filtered by the app)")
        if sess.stage_spec.supports_pulse_trains and trial.n_pulses > 1 and len(trial.pulse_onset_sample_indices):
            for onset_ms in sess.pulse_onset_offsets_ms(trial):
                for ax in axes:
                    ax.axvline(onset_ms, color='red', alpha=0.3, linestyle='--')
        axes[-1].set_xlabel("Time relative to stim onset (ms)")
        fig.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed -- skipping plot. Arrays are in t_ms/raw_window/my_filtered.")
