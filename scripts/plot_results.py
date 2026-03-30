"""
Generate publication-quality figures from training logs.

Usage:
    python scripts/plot_results.py

Outputs PNG files to plots/ directory.
Requires: matplotlib, pandas (pip install matplotlib pandas)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style
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


def plot_pretraining_loss():
    """Plot pretraining val_loss across runs (Run 1 vs Run 2/3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Run 1: LR=0.0005 (too high)
    r1 = pd.read_csv(os.path.join(LOGS_DIR, 'pretraining/run1_epoch_summary.csv'))
    ax1.plot(r1['epoch'], r1['val_loss'], 'o-', color='#e74c3c', label='Val Loss')
    ax1.plot(r1['epoch'], r1['train_loss'], 's--', color='#e74c3c', alpha=0.5, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Smooth L1)')
    ax1.set_title('Run 1: LR=0.0005 (diverged)')
    ax1.legend()
    ax1.axvline(x=11, color='gray', linestyle=':', alpha=0.5, label='Best (ep 11)')
    ax1.axvline(x=15, color='orange', linestyle=':', alpha=0.5, label='Peak LR')
    ax1.annotate('Peak LR\n(0.0005)', xy=(15, 0.22), fontsize=9, color='orange')
    ax1.annotate('Best\n(0.2081)', xy=(11, 0.2081), fontsize=9, color='gray')

    # Run 2+3: LR=0.00025 (converged)
    r2 = pd.read_csv(os.path.join(LOGS_DIR, 'pretraining/run2_epoch_summary.csv'))
    r3 = pd.read_csv(os.path.join(LOGS_DIR, 'pretraining/run3_epoch_summary.csv'))
    combined = pd.concat([r2, r3]).sort_values('epoch')
    ax2.plot(combined['epoch'], combined['val_loss'], 'o-', color='#2ecc71', label='Val Loss')
    ax2.plot(combined['epoch'], combined['train_loss'], 's--', color='#2ecc71', alpha=0.5, label='Train Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (Smooth L1)')
    ax2.set_title('Run 2+3: LR=0.00025 (converged)')
    ax2.legend()
    ax2.axhline(y=0.1586, color='gray', linestyle=':', alpha=0.5)
    ax2.annotate('Best: 0.1586', xy=(12, 0.1586), fontsize=9, color='gray')

    fig.suptitle('I-JEPA Pretraining: Effect of Learning Rate', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'pretraining_loss.png'), dpi=150, bbox_inches='tight')
    print('Saved pretraining_loss.png')
    plt.close()


def plot_pretraining_diagnostics():
    """Plot cos_sim and rep_diversity across pretraining."""
    r2 = pd.read_csv(os.path.join(LOGS_DIR, 'pretraining/run2_epoch_summary.csv'))
    r3 = pd.read_csv(os.path.join(LOGS_DIR, 'pretraining/run3_epoch_summary.csv'))
    combined = pd.concat([r2, r3]).sort_values('epoch')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(combined['epoch'], combined['cos_sim'], 'o-', color='#3498db')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Prediction Quality (cos_sim)')
    ax1.set_ylim(0.6, 0.9)
    ax1.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3)

    ax2.plot(combined['epoch'], combined['rep_diversity'], 's-', color='#e67e22')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Rep Diversity (1.0=collapsed)')
    ax2.set_title('Representation Diversity')
    ax2.set_ylim(0.4, 1.0)
    ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.3, label='Collapse risk')
    ax2.axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
    ax2.legend()

    fig.suptitle('I-JEPA Pretraining Diagnostics (LR=0.00025)', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'pretraining_diagnostics.png'), dpi=150, bbox_inches='tight')
    print('Saved pretraining_diagnostics.png')
    plt.close()


def plot_downstream_auc_comparison():
    """Plot val AUC curves for all downstream runs."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Frozen d2
    d2 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_frozen_d2/train_log.csv'))
    ax.plot(d2['epoch'], d2['val_auc'], '-', color='#3498db', label='Frozen, d=2, 100 slices (test=0.733)')

    # Frozen d3
    d3 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_frozen_d3/train_log.csv'))
    ax.plot(d3['epoch'], d3['val_auc'], '-', color='#2ecc71', label='Frozen, d=3, 100 slices (test=0.734)')

    # Unfrozen d2 s32
    u2 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_unfrozen_d2_s32/train_log.csv'))
    ax.plot(u2['epoch'], u2['val_auc'], '-', color='#e74c3c', label='Unfrozen, d=2, 32 slices (val=0.819)')

    # Unfrozen d3 s64
    u3 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_unfrozen_d3_s64/train_log.csv'))
    ax.plot(u3['epoch'], u3['val_auc'], '-', color='#9b59b6', label='Unfrozen, d=3, 64 slices (val=0.815)')

    # SLIViT baseline
    ax.axhline(y=0.869, color='black', linestyle='--', linewidth=1.5, label='SLIViT baseline (test=0.869)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUC')
    ax.set_title('Downstream Glaucoma Classification: Val AUC per Epoch')
    ax.set_ylim(0.55, 0.90)
    ax.legend(loc='lower right')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'downstream_auc_comparison.png'), dpi=150, bbox_inches='tight')
    print('Saved downstream_auc_comparison.png')
    plt.close()


def plot_downstream_loss():
    """Plot train/val loss for downstream runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    runs = [
        ('downstream_frozen_d2', 'Frozen d=2 s100', '#3498db'),
        ('downstream_frozen_d3', 'Frozen d=3 s100', '#2ecc71'),
        ('downstream_unfrozen_d2_s32', 'Unfrozen d=2 s32', '#e74c3c'),
        ('downstream_unfrozen_d3_s64', 'Unfrozen d=3 s64', '#9b59b6'),
    ]

    for ax, (folder, title, color) in zip(axes.flat, runs):
        df = pd.read_csv(os.path.join(LOGS_DIR, folder, 'train_log.csv'))
        ax.plot(df['epoch'], df['train_loss'], '-', color=color, alpha=0.7, label='Train')
        if 'val_loss' in df.columns and df['val_loss'].max() > 0:
            ax.plot(df['epoch'], df['val_loss'], '--', color=color, label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('BCE Loss')
        ax.set_title(title)
        ax.legend()

    fig.suptitle('Downstream Training/Validation Loss', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'downstream_loss.png'), dpi=150, bbox_inches='tight')
    print('Saved downstream_loss.png')
    plt.close()


def plot_summary_bar():
    """Bar chart comparing test/val AUC across all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        'SLIViT\n(baseline)',
        'Unfrozen d=2\n32 slices',
        'Unfrozen d=3\n64 slices',
        'Frozen d=3\n100 slices',
        'Frozen d=2\n100 slices',
    ]
    aucs = [0.869, 0.819, 0.815, 0.734, 0.733]
    colors = ['#2c3e50', '#e74c3c', '#9b59b6', '#2ecc71', '#3498db']
    labels = ['Test AUC', 'Val AUC*', 'Val AUC*', 'Test AUC', 'Test AUC']

    bars = ax.bar(methods, aucs, color=colors, edgecolor='white', linewidth=1.5)
    for bar, auc, label in zip(bars, aucs, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                '%.3f\n(%s)' % (auc, label), ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('AUC')
    ax.set_title('Glaucoma Classification: I-JEPA vs SLIViT Baseline')
    ax.set_ylim(0.5, 0.95)
    ax.axhline(y=0.869, color='black', linestyle='--', alpha=0.3)
    ax.annotate('* Val AUC (test eval crashed, NCCL bug now fixed)', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=8, color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'summary_comparison.png'), dpi=150, bbox_inches='tight')
    print('Saved summary_comparison.png')
    plt.close()


def plot_frozen_vs_unfrozen():
    """Side-by-side frozen vs unfrozen AUC trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Frozen
    d2 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_frozen_d2/train_log.csv'))
    d3 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_frozen_d3/train_log.csv'))
    ax1.plot(d2['epoch'], d2['val_auc'], '-', color='#3498db', label='Depth 2 (14.3M params)')
    ax1.plot(d3['epoch'], d3['val_auc'], '-', color='#2ecc71', label='Depth 3 (21.3M params)')
    ax1.axhline(y=0.869, color='black', linestyle='--', alpha=0.3, label='SLIViT (0.869)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val AUC')
    ax1.set_title('Frozen Encoder (probe only)')
    ax1.set_ylim(0.55, 0.90)
    ax1.legend()

    # Unfrozen
    u2 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_unfrozen_d2_s32/train_log.csv'))
    u3 = pd.read_csv(os.path.join(LOGS_DIR, 'downstream_unfrozen_d3_s64/train_log.csv'))
    ax2.plot(u2['epoch'], u2['val_auc'], '-', color='#e74c3c', label='d=2, 32 slices (~100M)')
    ax2.plot(u3['epoch'], u3['val_auc'], '-', color='#9b59b6', label='d=3, 64 slices (~107M)')
    ax2.axhline(y=0.869, color='black', linestyle='--', alpha=0.3, label='SLIViT (0.869)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val AUC')
    ax2.set_title('Unfrozen Encoder (full fine-tune, lr=5e-6)')
    ax2.set_ylim(0.55, 0.90)
    ax2.legend()

    fig.suptitle('Frozen Probe vs Full Fine-tuning', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, 'frozen_vs_unfrozen.png'), dpi=150, bbox_inches='tight')
    print('Saved frozen_vs_unfrozen.png')
    plt.close()


def plot_train_vs_val_auc():
    """Plot train AUC vs val AUC to check overfitting (for runs with train_auc)."""
    for folder, title, color in [
        ('downstream_frozen_d2', 'Frozen d=2', '#3498db'),
        ('downstream_frozen_d3', 'Frozen d=3', '#2ecc71'),
    ]:
        csv_path = os.path.join(LOGS_DIR, folder, 'train_log.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if 'train_auc' not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['epoch'], df['train_auc'], '-', color=color, label='Train AUC')
        ax.plot(df['epoch'], df['val_auc'], '--', color=color, alpha=0.7, label='Val AUC')
        ax.fill_between(df['epoch'], df['train_auc'], df['val_auc'],
                         alpha=0.1, color=color, label='Generalization gap')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('%s: Train vs Val AUC (overfitting check)' % title)
        ax.legend()
        ax.set_ylim(0.5, 1.0)

        fig.tight_layout()
        fname = 'overfit_check_%s.png' % folder
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
        print('Saved %s' % fname)
        plt.close()


def plot_predictions_from_npz():
    """Plot ROC curve and histograms from saved prediction files."""
    from sklearn.metrics import roc_curve

    for folder in os.listdir(LOGS_DIR):
        npz_path = os.path.join(LOGS_DIR, folder, 'test_predictions.npz')
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        labels, probs = data['labels'], data['probs']
        auc = roc_auc_score(labels, probs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC
        fpr, tpr, _ = roc_curve(labels, probs)
        ax1.plot(fpr, tpr, 'b-', linewidth=2, label='AUC = %.3f' % auc)
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve — %s' % folder)
        ax1.legend()

        # Histogram
        ax2.hist(probs[labels == 0], bins=30, alpha=0.6, color='blue',
                 label='Non-Glaucoma', density=True)
        ax2.hist(probs[labels == 1], bins=30, alpha=0.6, color='red',
                 label='Glaucoma', density=True)
        ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('P(Glaucoma)')
        ax2.set_ylabel('Density')
        ax2.set_title('Prediction Distribution')
        ax2.legend()

        fig.tight_layout()
        fname = 'roc_%s.png' % folder
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
        print('Saved %s' % fname)
        plt.close()


if __name__ == '__main__':
    print('Generating plots from logs in %s' % LOGS_DIR)
    print('Output: %s\n' % PLOTS_DIR)

    plot_pretraining_loss()
    plot_pretraining_diagnostics()
    plot_downstream_auc_comparison()
    plot_downstream_loss()
    plot_summary_bar()
    plot_frozen_vs_unfrozen()
    plot_train_vs_val_auc()
    plot_predictions_from_npz()

    print('\nAll plots saved to %s/' % PLOTS_DIR)
