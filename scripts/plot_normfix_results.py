"""
Generate updated figures incorporating normalization fix results.

Reads from results/downstream/ directories.
Outputs PNG files to results/ directory.

Usage:
    python scripts/plot_normfix_results.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DOWNSTREAM_DIR = os.path.join(RESULTS_DIR, 'downstream')

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_run(name):
    """Load results.json and train_log.csv for a downstream run."""
    run_dir = os.path.join(DOWNSTREAM_DIR, name)
    results = None
    log = None
    rpath = os.path.join(run_dir, 'results.json')
    lpath = os.path.join(run_dir, 'train_log.csv')
    if os.path.exists(rpath):
        with open(rpath) as f:
            results = json.load(f)
    if os.path.exists(lpath):
        log = pd.read_csv(lpath)
    return results, log


def plot_frozen_probe_comparison():
    """Hero image: frozen probe val AUC curves — old (buggy) vs normfix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Random-init frozen probe ---
    _, old_log = load_run('frozen_random_d3_s100')
    _, new_log = load_run('frozen_random_d3_s100_normfix')

    if old_log is not None:
        ax1.plot(old_log['epoch'], old_log['val_auc'], '-', color='#bdc3c7',
                 linewidth=1.5, label='Before fix (test=0.734)', alpha=0.7)
    if new_log is not None:
        ax1.plot(new_log['epoch'], new_log['val_auc'], '-', color='#e74c3c',
                 linewidth=2.5, label='After fix (test=0.834)')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation AUC')
    ax1.set_title('Random-init I-JEPA (frozen)')
    ax1.set_ylim(0.55, 0.90)
    ax1.legend(loc='lower right')
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # --- Right: ImageNet-init frozen probe ---
    _, old_im_log = load_run('frozen_imagenet_ep32_d3_s100')
    # normfix not yet available — placeholder
    normfix_im_dir = os.path.join(DOWNSTREAM_DIR, 'frozen_imagenet_ep32_d3_s100_normfix')
    normfix_im_log = None
    if os.path.exists(os.path.join(normfix_im_dir, 'train_log.csv')):
        normfix_im_log = pd.read_csv(os.path.join(normfix_im_dir, 'train_log.csv'))

    if old_im_log is not None:
        ax2.plot(old_im_log['epoch'], old_im_log['val_auc'], '-', color='#bdc3c7',
                 linewidth=1.5, label='Before fix (test=0.774)', alpha=0.7)
    if normfix_im_log is not None:
        im_res_path = os.path.join(normfix_im_dir, 'results.json')
        im_test_auc = '?'
        if os.path.exists(im_res_path):
            with open(im_res_path) as f:
                im_test_auc = '%.3f' % json.load(f).get('test_auc', 0)
        ax2.plot(normfix_im_log['epoch'], normfix_im_log['val_auc'], '-', color='#3498db',
                 linewidth=2.5, label='After fix (test=%s)' % im_test_auc)
    else:
        ax2.text(0.5, 0.5, 'Running...', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=16, color='gray', alpha=0.5)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AUC')
    ax2.set_title('ImageNet-init I-JEPA ep32 (frozen)')
    ax2.set_ylim(0.55, 0.90)
    ax2.legend(loc='lower right')
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    fig.suptitle('Normalization Fix: Frozen Probe Improvement', fontsize=16, y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'normfix_frozen_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved %s' % out)
    plt.close()


def plot_all_results_bar():
    """Bar chart of all test AUCs — corrected numbers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all results
    runs = []

    # Normfix frozen results
    r, _ = load_run('frozen_random_d3_s100_normfix')
    if r:
        runs.append(('Frozen Random\n(normfix)', r['test_auc'], '#e74c3c', True))

    # Check for normfix ImageNet
    normfix_im_dir = os.path.join(DOWNSTREAM_DIR, 'frozen_imagenet_ep32_d3_s100_normfix')
    if os.path.exists(os.path.join(normfix_im_dir, 'results.json')):
        with open(os.path.join(normfix_im_dir, 'results.json')) as f:
            r_im = json.load(f)
        runs.append(('Frozen ImageNet\nep32 (normfix)', r_im['test_auc'], '#3498db', True))

    # Old frozen (shown faded for comparison)
    r, _ = load_run('frozen_random_d3_s100')
    if r:
        runs.append(('Frozen Random\n(old, no norm)', r['test_auc'], '#bdc3c7', False))
    r, _ = load_run('frozen_imagenet_ep32_d3_s100')
    if r:
        runs.append(('Frozen ImageNet\nep32 (old)', r['test_auc'], '#d5dbdb', False))

    # Unfrozen results (these didn't have normalization bug in unfrozen path)
    for name, label in [
        ('unfrozen_imagenet_d3_s32', 'Unfrozen ImageNet\nd=3, 32 slices'),
        ('unfrozen_imagenet_d3_s64', 'Unfrozen ImageNet\nd=3, 64 slices'),
        ('unfrozen_imagenet_d2_s64', 'Unfrozen ImageNet\nd=2, 64 slices'),
        ('unfrozen_imagenet_d2_s32', 'Unfrozen ImageNet\nd=2, 32 slices'),
    ]:
        r, _ = load_run(name)
        if r and 'test_auc' in r:
            runs.append((label, r['test_auc'], '#2ecc71', True))

    # Sort by AUC descending
    runs.sort(key=lambda x: x[1], reverse=True)

    names = [r[0] for r in runs]
    aucs = [r[1] for r in runs]
    colors = [r[2] for r in runs]
    is_current = [r[3] for r in runs]

    bars = ax.bar(range(len(runs)), aucs, color=colors, edgecolor='white', linewidth=1.5)
    for i, (bar, auc, current) in enumerate(zip(bars, aucs, is_current)):
        style = {'fontweight': 'bold'} if current else {'color': 'gray'}
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                '%.3f' % auc, ha='center', va='bottom', fontsize=10, **style)
        if not current:
            bar.set_alpha(0.4)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Test AUC')
    ax.set_title('Glaucoma Classification: All Results (corrected with ImageNet normalization)')
    ax.set_ylim(0.6, 0.90)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'test_auc_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved %s' % out)
    plt.close()


def plot_frozen_training_curves():
    """Training curves for the normfix frozen random run."""
    r, log = load_run('frozen_random_d3_s100_normfix')
    if log is None:
        print('No normfix frozen random train log found, skipping')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Val AUC
    ax1.plot(log['epoch'], log['val_auc'], '-', color='#e74c3c', label='Val AUC')
    if 'train_auc' in log.columns:
        ax1.plot(log['epoch'], log['train_auc'], '--', color='#e74c3c', alpha=0.5, label='Train AUC')
        ax1.fill_between(log['epoch'], log['train_auc'], log['val_auc'],
                         alpha=0.1, color='#e74c3c')
    best_ep = r['best_epoch'] if r else log['val_auc'].idxmax() + 1
    best_auc = r['best_val_auc'] if r else log['val_auc'].max()
    ax1.axvline(x=best_ep, color='gray', linestyle=':', alpha=0.5)
    ax1.annotate('Best: %.3f\n(ep %d)' % (best_auc, best_ep),
                 xy=(best_ep, best_auc), fontsize=9, color='gray',
                 xytext=(best_ep + 2, best_auc - 0.03))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUC')
    ax1.set_title('Frozen Random-init + Normfix: AUC')
    ax1.legend()
    ax1.set_ylim(0.6, 1.0)

    # Loss
    ax2.plot(log['epoch'], log['train_loss'], '-', color='#e74c3c', label='Train Loss')
    ax2.plot(log['epoch'], log['val_loss'], '--', color='#e74c3c', alpha=0.7, label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BCE Loss')
    ax2.set_title('Frozen Random-init + Normfix: Loss')
    ax2.legend()

    fig.suptitle('Frozen Probe Training (Random-init, normalization fixed)', fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'frozen_probe_normfix_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved %s' % out)
    plt.close()


def plot_normfix_impact():
    """Before/after comparison showing the normalization fix impact."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data: old vs new
    categories = ['Frozen Random\n(d=3, 100 slices)']
    old_aucs = [0.734]
    new_aucs = [0.834]

    # Check if ImageNet normfix is available
    normfix_im_dir = os.path.join(DOWNSTREAM_DIR, 'frozen_imagenet_ep32_d3_s100_normfix')
    if os.path.exists(os.path.join(normfix_im_dir, 'results.json')):
        with open(os.path.join(normfix_im_dir, 'results.json')) as f:
            im_new = json.load(f)['test_auc']
        categories.append('Frozen ImageNet\nep32 (d=3, 100 slices)')
        old_aucs.append(0.774)
        new_aucs.append(im_new)

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, old_aucs, width, label='Before (no ImageNet norm)',
                   color='#bdc3c7', edgecolor='white')
    bars2 = ax.bar(x + width/2, new_aucs, width, label='After (ImageNet norm fixed)',
                   color='#e74c3c', edgecolor='white')

    for bar, auc in zip(bars1, old_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                '%.3f' % auc, ha='center', va='bottom', fontsize=11, color='gray')
    for bar, auc in zip(bars2, new_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                '%.3f' % auc, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Draw improvement arrows
    for i in range(len(categories)):
        improvement = new_aucs[i] - old_aucs[i]
        mid_x = x[i]
        ax.annotate('+%.1f%%' % (improvement * 100),
                    xy=(mid_x, max(new_aucs[i], old_aucs[i]) + 0.025),
                    ha='center', fontsize=12, fontweight='bold', color='#27ae60')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Test AUC')
    ax.set_title('Impact of ImageNet Normalization Fix on Frozen Probe')
    ax.set_ylim(0.6, 0.92)
    ax.legend()

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'normfix_impact.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved %s' % out)
    plt.close()


def plot_hero_image():
    """Main README hero image: frozen probe AUC curves with normfix highlight."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Normfix frozen random (the star)
    _, new_log = load_run('frozen_random_d3_s100_normfix')
    if new_log is not None:
        ax.plot(new_log['epoch'], new_log['val_auc'], '-', color='#e74c3c',
                linewidth=2.5, label='Frozen Random I-JEPA (test=0.834)', zorder=5)

    # Old frozen runs (faded)
    for name, label, color in [
        ('frozen_random_d3_s100', 'Old: Frozen Random (0.734)', '#bdc3c7'),
        ('frozen_imagenet_ep32_d3_s100', 'Old: Frozen ImageNet ep32 (0.774)', '#d5dbdb'),
    ]:
        _, log = load_run(name)
        if log is not None:
            ax.plot(log['epoch'], log['val_auc'], '--', color=color,
                    linewidth=1.2, alpha=0.5, label=label)

    # Normfix ImageNet if available
    normfix_im_dir = os.path.join(DOWNSTREAM_DIR, 'frozen_imagenet_ep32_d3_s100_normfix')
    if os.path.exists(os.path.join(normfix_im_dir, 'train_log.csv')):
        im_log = pd.read_csv(os.path.join(normfix_im_dir, 'train_log.csv'))
        im_res_path = os.path.join(normfix_im_dir, 'results.json')
        im_auc = '?'
        if os.path.exists(im_res_path):
            with open(im_res_path) as f:
                im_auc = '%.3f' % json.load(f)['test_auc']
        ax.plot(im_log['epoch'], im_log['val_auc'], '-', color='#3498db',
                linewidth=2.5, label='Frozen ImageNet I-JEPA (test=%s)' % im_auc, zorder=5)

    # Unfrozen reference lines
    for name, label, color in [
        ('unfrozen_imagenet_d3_s32', 'Unfrozen best (0.829)', '#2ecc71'),
    ]:
        r, _ = load_run(name)
        if r and 'test_auc' in r:
            ax.axhline(y=r['test_auc'], color=color, linestyle='--',
                       linewidth=1.5, alpha=0.7, label=label)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUC')
    ax.set_title('I-JEPA Downstream: Frozen Probe with Normalization Fix')
    ax.set_ylim(0.55, 0.90)
    ax.legend(loc='lower right', fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'downstream_auc_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Saved %s (hero image)' % out)
    plt.close()


if __name__ == '__main__':
    print('Generating normfix plots from %s\n' % DOWNSTREAM_DIR)

    plot_hero_image()
    plot_frozen_probe_comparison()
    plot_all_results_bar()
    plot_frozen_training_curves()
    plot_normfix_impact()

    print('\nAll plots saved to %s/' % RESULTS_DIR)
