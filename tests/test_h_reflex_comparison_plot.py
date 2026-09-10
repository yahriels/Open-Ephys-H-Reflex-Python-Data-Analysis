import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from open_ephys.analysis import helpers


def test_plot_h_reflex_comparison_shows_mean_sd_and_stimulation_std(monkeypatch):
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.setattr(plt, 'subplots', lambda *args, **kwargs: (fig, ax))

    h_cache = {
        'recA': {
            'stage1': {
                'h_sizes': np.array([1.0, 3.0, 5.0]),
                'mean_amp': 2.0,
                'std_amp': 0.25,
            }
        }
    }

    helpers.plot_h_reflex_comparison(
        h_cache,
        [('recA', '', 1000)],
        'stage1',
        {'stage1': 'Stage 1'},
        metric='h_reflex',
    )

    text_values = [t.get_text() for t in ax.texts]
    assert any('μ = 3.000' in value for value in text_values)
    assert any('± 2.000' in value for value in text_values)

    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert any('σ=0.250 mA' in label for label in tick_labels)
