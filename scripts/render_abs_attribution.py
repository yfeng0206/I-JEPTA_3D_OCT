"""Re-render slice-level and window-level occlusion attribution as |Δlogit|.

The previous figures plotted *signed* mean Δlogit per slice (and per W=7 window),
which canceled within-volume opposite-sign evidence and made the population
curve look bimodal with a "dip" in the middle. The magnitude view (mean |Δ|)
shows where attribution is concentrated regardless of direction.

Reads existing npz outputs from the interpretability archives — no AML rerun.
Writes:
  results/summary/slice_contribution_curves.png         (single-slice, 64 positions)
  results/summary/04_window_occlusion_W7.png            (multi-slice W=7, 58 windows)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_FP32 = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA_archive\04_interpretability_v2_fp32\slice_contributions'
ARCHIVE_FP16 = r'C:\Users\garyfeng\OneDrive - Microsoft\Desktop\I-JEPA_archive\04_interpretability\window_occlusion'
OUT_DIR      = os.path.join(REPO, 'results', 'summary')

NAMES    = ['meanpool', 'crossattn', 'd1']
DISPLAYS = {'meanpool': 'FT + MeanPool',
            'crossattn': 'FT + CrossAttnPool',
            'd1':        'FT + AttentiveProbe d=1'}
COLORS   = {'meanpool': '#d62728', 'crossattn': '#1f77b4', 'd1': '#2ca02c'}


def render(curves, x, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, abs_pos, abs_neg, n_pos, n_neg in curves:
        ax.plot(x, abs_pos, color=COLORS[name],
                linestyle='-',  linewidth=2.0,
                label=f'{DISPLAYS[name]} (glaucoma, n={n_pos})')
        ax.plot(x, abs_neg, color=COLORS[name],
                linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'{DISPLAYS[name]} (healthy,  n={n_neg})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean |Δlogit| (occlusion magnitude)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def render_single_slice():
    curves = []
    for name in NAMES:
        d = np.load(os.path.join(ARCHIVE_FP32, f'slice_contrib_{name}.npz'))
        c, y = d['contrib'], d['labels']
        curves.append((name,
                       np.abs(c[y == 1]).mean(axis=0),
                       np.abs(c[y == 0]).mean(axis=0),
                       int((y == 1).sum()),
                       int((y == 0).sum())))
    n_pos = curves[0][1].shape[0]
    render(curves,
           x=np.arange(n_pos),
           xlabel=f'Slice index (0..{n_pos-1}, linspace(0, 199, {n_pos}))',
           title='Slice-level occlusion attribution — 3 fine-tune probes\n'
                 '(magnitude of logit change when each slice is zeroed)',
           out_path=os.path.join(OUT_DIR, 'slice_contribution_curves.png'))


def render_window_W7():
    curves = []
    W = None
    for name in NAMES:
        d = np.load(os.path.join(ARCHIVE_FP16, f'window_contrib_{name}_W7.npz'))
        c, y = d['window_contrib'], d['labels']
        W = int(d['W'])
        curves.append((name,
                       np.abs(c[y == 1]).mean(axis=0),
                       np.abs(c[y == 0]).mean(axis=0),
                       int((y == 1).sum()),
                       int((y == 0).sum())))
    n_windows = curves[0][1].shape[0]
    render(curves,
           x=np.arange(n_windows),
           xlabel=f'Window start index (0..{n_windows-1}, W={W} consecutive slices zeroed)',
           title=f'Window occlusion attribution (W={W}) — 3 fine-tune probes\n'
                 f'(magnitude of logit change when {W} consecutive slices are zeroed)',
           out_path=os.path.join(OUT_DIR, '04_window_occlusion_W7.png'))


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    render_single_slice()
    render_window_W7()
