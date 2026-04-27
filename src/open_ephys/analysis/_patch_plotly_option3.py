"""Patch Plotly_Read_H-Reflex_App_Simplified.ipynb with pure Plotly Option 3."""
import json, pathlib

NB_PATH = pathlib.Path(__file__).parent / "Plotly_Read_H-Reflex_App_Simplified.ipynb"

# Constants in this notebook (config cell 884f799b — no underscore prefixes):
#   M_WAVE_START_MS, M_WAVE_END_MS, H_WAVE_START_MS, H_WAVE_END_MS
#   PRE_AVG_MS, POST_AVG_MS

NEW_SOURCE = r'''# ---- Plot HRS2: Interactive Averaged Waveforms by Stimulation Amplitude ----
# Pure Plotly — no ipywidgets, fully exportable as standalone HTML.
# Amplitude dropdown (top-right) switches groups. Legend entries toggle overlay signals.
# Switching amplitudes resets overlays to their default (legend-only) state.

import plotly.graph_objects as go
import scipy.stats as stats
from scipy.interpolate import interp1d
from collections import defaultdict

if len(hrs2_trials) > 0:

    # ---- Data prep ----
    _groups = defaultdict(list)
    for _trial in hrs2_trials:
        _key = round(_trial.stimulation_amplitude_ma, 2)
        _t_win, _bip_win, _adc_win, _stim_end = get_trial_window(
            _trial, PRE_AVG_MS, POST_AVG_MS)
        _t_uni, _uni_win, _, _ = get_trial_window(
            _trial, PRE_AVG_MS, POST_AVG_MS, use_unipolar=True)
        _groups[_key].append((_t_win, _bip_win, _adc_win, _uni_win, _stim_end))

    _sorted_amps = sorted(_groups.keys())

    def _pad_rows(rows, n_pts):
        p = np.full((len(rows), n_pts), np.nan)
        for k, a in enumerate(rows):
            if a is not None and len(a) > 0:
                _n = min(len(a), n_pts)
                p[k, :_n] = np.asarray(a[:_n], dtype=float)
        return p

    _amp_data = []
    for _amp in _sorted_amps:
        _wins  = _groups[_amp]
        _t_ref = _wins[0][0]
        _np    = len(_t_ref)

        _pb  = _pad_rows([w[1] for w in _wins], _np)
        _pa  = _pad_rows([w[2] for w in _wins], _np)
        _pu  = _pad_rows([w[3] for w in _wins], _np)
        _pab = np.abs(_pb)
        _pau = np.abs(_pu)

        _avg_b  = np.nanmean(_pb,  axis=0)
        _avg_a  = np.nanmean(_pa,  axis=0)
        _avg_u  = np.nanmean(_pu,  axis=0)
        _avg_ab = np.nanmean(_pab, axis=0)
        _avg_au = np.nanmean(_pau, axis=0)
        _std_b  = np.nanstd(_pb,   axis=0)

        _se  = [w[4] for w in _wins if w[4] is not None]
        _mse = float(np.mean(_se)) if _se else 0.5

        _mm = (_t_ref >= M_WAVE_START_MS) & (_t_ref <= M_WAVE_END_MS)
        _hm = (_t_ref >= H_WAVE_START_MS) & (_t_ref <= H_WAVE_END_MS)
        _mi = int(np.argmax(_avg_ab[_mm])) if _mm.any() else 0
        _hi = int(np.argmax(_avg_ab[_hm])) if _hm.any() else 0
        _m_t   = float(_t_ref[_mm][_mi])  if _mm.any() else M_WAVE_START_MS
        _m_a   = float(_avg_ab[_mm][_mi]) if _mm.any() else float('nan')
        _m_bip = float(_avg_b[_mm][_mi])  if _mm.any() else float('nan')
        _h_t   = float(_t_ref[_hm][_hi])  if _hm.any() else H_WAVE_START_MS
        _h_a   = float(_avg_ab[_hm][_hi]) if _hm.any() else float('nan')
        _h_bip = float(_avg_b[_hm][_hi])  if _hm.any() else float('nan')

        _amp_data.append({
            'amp': _amp, 't_ref': _t_ref, 'n': len(_wins),
            'mean_stim_end': _mse,
            'avg_bip': _avg_b, 'std_bip': _std_b,
            'avg_adc': _avg_a, 'avg_uni': _avg_u,
            'avg_abs_bip': _avg_ab, 'avg_abs_uni': _avg_au,
            'm_peak_time': _m_t, 'm_peak_amp': _m_a, 'm_peak_bip': _m_bip,
            'h_peak_time': _h_t, 'h_peak_amp': _h_a, 'h_peak_bip': _h_bip,
        })

    # ---- Global y-range (covers mean +/- 1 SD across all amplitudes) ----
    _all_bip = np.concatenate([
        np.concatenate([d['avg_bip'] - d['std_bip'], d['avg_bip'] + d['std_bip']])
        for d in _amp_data
    ])
    _all_bip = _all_bip[~np.isnan(_all_bip)]
    if len(_all_bip):
        _Y_LO = float(np.nanmin(_all_bip))
        _Y_HI = float(np.nanmax(_all_bip))
        _Y_PAD = max(0.12 * (_Y_HI - _Y_LO), 50.0)
        _Y_LO -= _Y_PAD;  _Y_HI += _Y_PAD
    else:
        _Y_LO, _Y_HI = -2000.0, 2000.0

    # ---- Build figure (9 traces per amplitude) ----
    # Trace order per amp i (base = 9*i):
    #   +0 sd_lo   +1 sd_hi   +2 avg_bip   +3 m_peak   +4 h_peak
    #   +5 abs_bip  +6 uni  +7 abs_uni  +8 adc
    _N_PER = 9
    fig_wave = go.Figure()

    def _shapes_for(d):
        return [
            dict(type='rect', xref='x', yref='paper', layer='below',
                 x0=0, y0=0, x1=d['mean_stim_end'], y1=1,
                 fillcolor='rgba(255,0,0,0.10)', line_width=0),
            dict(type='rect', xref='x', yref='paper', layer='below',
                 x0=M_WAVE_START_MS, y0=0, x1=M_WAVE_END_MS, y1=1,
                 fillcolor='rgba(0,0,200,0.10)', line_width=0),
            dict(type='rect', xref='x', yref='paper', layer='below',
                 x0=H_WAVE_START_MS, y0=0, x1=H_WAVE_END_MS, y1=1,
                 fillcolor='rgba(0,160,0,0.10)', line_width=0),
            dict(type='line', xref='x', yref='paper',
                 x0=0, y0=0, x1=0, y1=1,
                 line=dict(color='red', dash='dash', width=1.5)),
            dict(type='line', xref='x', yref='paper',
                 x0=d['mean_stim_end'], y0=0, x1=d['mean_stim_end'], y1=1,
                 line=dict(color='red', dash='dash', width=1.5)),
            dict(type='line', xref='x', yref='paper',
                 x0=d['m_peak_time'], y0=0, x1=d['m_peak_time'], y1=1,
                 line=dict(color='royalblue', dash='dot', width=1.2)),
            dict(type='line', xref='x', yref='paper',
                 x0=d['h_peak_time'], y0=0, x1=d['h_peak_time'], y1=1,
                 line=dict(color='green', dash='dot', width=1.2)),
        ]

    for _i, _d in enumerate(_amp_data):
        _t   = _d['t_ref']
        _lo  = _d['avg_bip'] - _d['std_bip']
        _hi  = _d['avg_bip'] + _d['std_bip']
        _v0  = (_i == 0)

        # +0 SD lower band (transparent baseline for fill)
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_lo, line=dict(width=0),
            showlegend=False, hoverinfo='skip',
            visible=_v0,
        ))
        # +1 SD upper band (fills down to previous trace)
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_hi, fill='tonexty', fillcolor='rgba(220,80,80,0.22)',
            line=dict(width=0),
            showlegend=False, hoverinfo='skip',
            visible=_v0,
        ))
        # +2 Avg bipolar
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_d['avg_bip'],
            line=dict(color='black', width=2.5),
            name='Avg Bipolar ±1 SD',
            showlegend=_v0, visible=_v0,
            hovertemplate='%{x:.2f} ms: %{y:.1f} µV<extra></extra>',
        ))
        # +3 M-peak marker
        fig_wave.add_trace(go.Scatter(
            x=[_d['m_peak_time']], y=[_d['m_peak_bip']],
            mode='markers+text',
            marker=dict(symbol='star', size=13, color='royalblue',
                        line=dict(color='darkblue', width=1)),
            text=[f"{_d['m_peak_amp']:.1f} µV"],
            textposition='top center',
            textfont=dict(color='royalblue', size=10),
            name='M-peak',
            showlegend=_v0, visible=_v0,
            hovertemplate=(f"M-peak: {_d['m_peak_amp']:.1f} µV"
                           f" @ {_d['m_peak_time']:.2f} ms<extra></extra>"),
        ))
        # +4 H-peak marker
        fig_wave.add_trace(go.Scatter(
            x=[_d['h_peak_time']], y=[_d['h_peak_bip']],
            mode='markers+text',
            marker=dict(symbol='star', size=13, color='green',
                        line=dict(color='darkgreen', width=1)),
            text=[f"{_d['h_peak_amp']:.1f} µV"],
            textposition='top center',
            textfont=dict(color='green', size=10),
            name='H-peak',
            showlegend=_v0, visible=_v0,
            hovertemplate=(f"H-peak: {_d['h_peak_amp']:.1f} µV"
                           f" @ {_d['h_peak_time']:.2f} ms<extra></extra>"),
        ))
        # +5 |Bipolar| avg overlay
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_d['avg_abs_bip'],
            line=dict(color='gray', width=1.8),
            name='|Bipolar| avg',
            showlegend=_v0,
            visible='legendonly' if _v0 else False,
        ))
        # +6 Unipolar avg overlay
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_d['avg_uni'],
            line=dict(color='darkorange', width=1.8),
            name='Unipolar avg',
            showlegend=_v0,
            visible='legendonly' if _v0 else False,
        ))
        # +7 |Unipolar| avg overlay
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_d['avg_abs_uni'],
            line=dict(color='purple', width=1.8),
            name='|Unipolar| avg',
            showlegend=_v0,
            visible='legendonly' if _v0 else False,
        ))
        # +8 ADC avg (secondary y-axis)
        fig_wave.add_trace(go.Scatter(
            x=_t, y=_d['avg_adc'],
            line=dict(color='mediumseagreen', width=1.8),
            name='ADC sync',
            showlegend=_v0, yaxis='y2',
            visible='legendonly' if _v0 else False,
        ))

    # ---- Dropdown buttons ----
    _n_total = len(fig_wave.data)
    _buttons = []
    for _i, _d in enumerate(_amp_data):
        _vis = [False] * _n_total
        _base = _N_PER * _i
        _vis[_base + 0] = True
        _vis[_base + 1] = True
        _vis[_base + 2] = True
        _vis[_base + 3] = True
        _vis[_base + 4] = True
        _vis[_base + 5] = 'legendonly'
        _vis[_base + 6] = 'legendonly'
        _vis[_base + 7] = 'legendonly'
        _vis[_base + 8] = 'legendonly'

        _buttons.append(dict(
            label=f"{_d['amp']:.2f} mA  (n={_d['n']})",
            method='update',
            args=[
                {'visible': _vis},
                {
                    'title.text': (
                        f"HRS2 Waveforms — {_d['amp']:.2f} mA"
                        f"  (n={_d['n']})  |  {hrs2_header.subject_id}"
                    ),
                    'shapes': _shapes_for(_d),
                },
            ],
        ))

    fig_wave.update_layout(
        title=dict(
            text=(
                f"HRS2 Waveforms — {_amp_data[0]['amp']:.2f} mA"
                f"  (n={_amp_data[0]['n']})  |  {hrs2_header.subject_id}"
            ),
            font=dict(size=14),
        ),
        updatemenus=[dict(
            buttons=_buttons,
            direction='down',
            showactive=True,
            pad=dict(t=10),
            x=1.02, xanchor='left',
            y=1.0,  yanchor='top',
            bgcolor='white', bordercolor='#bbb',
            font=dict(size=11),
        )],
        xaxis=dict(
            title='Time re: stim onset (ms)',
            range=[-PRE_AVG_MS, POST_AVG_MS],
            tickmode='linear', dtick=1,
            showgrid=True, gridcolor='rgba(0,0,0,0.08)',
            zeroline=True, zerolinecolor='rgba(0,0,0,0.4)', zerolinewidth=1,
        ),
        yaxis=dict(
            title='EMG (µV)',
            range=[_Y_LO, _Y_HI],
            showgrid=True, gridcolor='rgba(0,0,0,0.08)',
        ),
        yaxis2=dict(
            title='ADC (V)',
            overlaying='y', side='right',
            showgrid=False,
            color='mediumseagreen',
            tickfont=dict(color='mediumseagreen'),
        ),
        legend=dict(
            title=dict(text='<b>Signal</b><br><sup>click to toggle</sup>'),
            x=1.02, y=0.55, xanchor='left',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc', borderwidth=1,
        ),
        shapes=_shapes_for(_amp_data[0]),
        height=520,
        margin=dict(r=220, t=60),
        hovermode='x unified',
        plot_bgcolor='white',
    )

    print(f"Loaded {len(_amp_data)} amplitude groups.")
    print("Use the dropdown (top-right) to switch amplitudes.")
    print("Click legend entries to show/hide signal overlays (resets on amplitude change).")
    fig_wave.show()

else:
    print("No HRS2 trials to group.")
    fig_wave = go.Figure()


# ---- Recruitment Curves (pure Plotly) ----

if len(hrs2_trials) > 0:

    _rc_groups = defaultdict(list)
    for _trial in hrs2_trials:
        _key = round(_trial.stimulation_amplitude_ma, 2)
        _t_win, _bip_win, _, _ = get_trial_window(_trial, PRE_AVG_MS, POST_AVG_MS)
        _rc_groups[_key].append((_t_win, _bip_win))

    _m_wave_dict = defaultdict(list)
    _h_wave_dict = defaultdict(list)

    for _amp_key in sorted(_rc_groups.keys()):
        _wins  = _rc_groups[_amp_key]
        _t_ref = _wins[0][0]
        _n_pts = len(_t_ref)

        _padded = np.full((len(_wins), _n_pts), np.nan)
        for k, (_, _bip) in enumerate(_wins):
            _n = min(len(_bip), _n_pts)
            _padded[k, :_n] = np.asarray(_bip[:_n], dtype=float)

        _avg_abs = np.nanmean(np.abs(_padded), axis=0)

        _mm = (_t_ref >= M_WAVE_START_MS) & (_t_ref <= M_WAVE_END_MS)
        if np.any(_mm):
            _m_wave_dict[_amp_key].append(float(_avg_abs[_mm][np.argmax(_avg_abs[_mm])]))

        _hm = (_t_ref >= H_WAVE_START_MS) & (_t_ref <= H_WAVE_END_MS)
        if np.any(_hm):
            _h_wave_dict[_amp_key].append(float(_avg_abs[_hm][np.argmax(_avg_abs[_hm])]))

    _sorted_rc_amps = sorted(set(_m_wave_dict.keys()) | set(_h_wave_dict.keys()))
    _m_data = [_m_wave_dict.get(a, [0]) for a in _sorted_rc_amps]
    _h_data = [_h_wave_dict.get(a, [0]) for a in _sorted_rc_amps]

    _m_means = np.array([np.mean(v) for v in _m_data])
    _h_means = np.array([np.mean(v) for v in _h_data])
    _m_sems  = np.array([stats.sem(v) if len(v) > 1 else 0.0 for v in _m_data])
    _h_sems  = np.array([stats.sem(v) if len(v) > 1 else 0.0 for v in _h_data])

    _M_max = float(np.max(_m_means)) if np.max(_m_means) > 0 else 1.0
    _m_norm = (_m_means / _M_max) * 100
    _h_norm = (_h_means / _M_max) * 100
    _ms_norm = (_m_sems  / _M_max) * 100
    _hs_norm = (_h_sems  / _M_max) * 100

    try:
        _interp = interp1d(_m_norm, _sorted_rc_amps, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
        _cur50 = float(_interp(50))
    except Exception:
        _cur50 = _sorted_rc_amps[int(np.argmax(_m_norm >= 50))]

    _norm_cur   = np.array(_sorted_rc_amps) / _cur50
    _H_max_pct  = float(np.max(_h_norm))
    _idx_Hmax   = int(np.argmax(_h_norm))
    _cur_Hmax   = float(_norm_cur[_idx_Hmax])

    # ---- RC Figure 1: raw amplitude (µV) vs stim current (mA) ----
    fig_rc1 = go.Figure()
    fig_rc1.add_trace(go.Scatter(
        x=list(_sorted_rc_amps), y=_m_means.tolist(),
        error_y=dict(type='data', array=_m_sems.tolist(), visible=True),
        mode='lines+markers', name='M-wave mean ± SEM',
        line=dict(color='royalblue'), marker=dict(size=6),
    ))
    fig_rc1.add_trace(go.Scatter(
        x=list(_sorted_rc_amps), y=_h_means.tolist(),
        error_y=dict(type='data', array=_h_sems.tolist(), visible=True),
        mode='lines+markers', name='H-wave mean ± SEM',
        line=dict(color='green'), marker=dict(size=6),
    ))
    fig_rc1.update_layout(
        title=f'HRS2 Recruitment Curve — {hrs2_header.subject_id}',
        xaxis=dict(title='Stimulation Amplitude (mA)',
                   showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(title='Peak Amplitude (µV)',
                   showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        plot_bgcolor='white', height=460,
    )
    fig_rc1.show()

    # ---- RC Figure 2: normalized current vs % Mmax ----
    fig_rc2 = go.Figure()
    fig_rc2.add_trace(go.Scatter(
        x=_norm_cur.tolist(), y=_m_norm.tolist(),
        error_y=dict(type='data', array=_ms_norm.tolist(), visible=True),
        mode='lines+markers', name='M-wave (% Mmax)',
        line=dict(color='royalblue'), marker=dict(size=6),
    ))
    fig_rc2.add_trace(go.Scatter(
        x=_norm_cur.tolist(), y=_h_norm.tolist(),
        error_y=dict(type='data', array=_hs_norm.tolist(), visible=True),
        mode='lines+markers', name='H-wave (% Mmax)',
        line=dict(color='green'), marker=dict(size=6),
    ))
    fig_rc2.update_layout(
        title=f'HRS2 Normalized Recruitment Curve — {hrs2_header.subject_id}',
        xaxis=dict(title='Current (normalized to current at 50% Mmax)',
                   showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(title='H and M wave amplitude (% of Mmax)',
                   showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
        shapes=[
            dict(type='line', xref='paper', yref='y',
                 x0=0, x1=1, y0=_H_max_pct, y1=_H_max_pct,
                 line=dict(color='green', dash='dash', width=1.2)),
            dict(type='line', xref='x', yref='paper',
                 x0=_cur_Hmax, x1=_cur_Hmax, y0=0, y1=1,
                 line=dict(color='gray', dash='dash', width=1.2)),
        ],
        annotations=[
            dict(x=_cur_Hmax + 0.02, y=_H_max_pct + 2,
                 text='b', showarrow=False, font=dict(size=14)),
            dict(x=_norm_cur[_idx_Hmax] - 0.08, y=_H_max_pct + 2,
                 text='a', showarrow=False, font=dict(size=14)),
        ],
        legend=dict(x=0.02, y=0.98, xanchor='left', yanchor='top'),
        plot_bgcolor='white', height=460,
    )
    fig_rc2.show()

    print(f"\nM_max = {_M_max:.2f} µV")
    print(f"H_max = {float(np.max(_h_means)):.2f} µV  ({_H_max_pct:.1f}% of M_max)")
    print(f"Current at 50% M_max = {_cur50:.2f} mA")
    print(f"Current at H_max = {_sorted_rc_amps[_idx_Hmax]:.2f} mA  ({_cur_Hmax:.2f}x normalized)")

else:
    print("No HRS2 trials available for recruitment curve.")
    fig_rc1 = fig_rc2 = go.Figure()
'''

EXPORT_SOURCE = r'''# ---- Export figures as standalone HTML files ----
# Each file is fully self-contained and interactive (no server or kernel needed).
# Requires internet on first open (Plotly.js loaded from CDN).
# Use include_plotlyjs=True for a fully offline file (larger size).

fig_wave.write_html("amplitude_viewer.html",      include_plotlyjs='cdn')
fig_rc1.write_html("recruitment_curve_raw.html",  include_plotlyjs='cdn')
fig_rc2.write_html("recruitment_curve_norm.html", include_plotlyjs='cdn')

print("Saved:")
print("  amplitude_viewer.html")
print("  recruitment_curve_raw.html")
print("  recruitment_curve_norm.html")

# To export the full notebook as a single HTML (run from terminal in this directory):
#   jupyter nbconvert --to html --no-input Plotly_Read_H-Reflex_App_Simplified.ipynb
'''

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for c in nb['cells']:
    cid = c.get('id', '')
    if cid == '3bdf3a5b':
        c['source'] = NEW_SOURCE
        c['outputs'] = []
        c['execution_count'] = None
        patched += 1
    elif cid == 'f1f7b928':
        c['source'] = EXPORT_SOURCE
        c['outputs'] = []
        c['execution_count'] = None
        patched += 1

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Patched {patched} cells in {NB_PATH.name}")
