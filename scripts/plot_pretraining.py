"""Generate per-run pretraining diagnostic plots.

Usage:
    python scripts/plot_pretraining.py \
        --csv results/pretraining/pretrain_random_posfix/jepa_patch-log.csv \
        --stdout results/pretraining/pretrain_random_posfix/torchrun_stdout.log \
        --output results/pretraining/pretrain_random_posfix/ \
        --title "Random-init 100ep (posfix)"

Generates 4 plots:
  1. train_val_loss.png   - Train & val loss per epoch
  2. rep_diversity.png    - Representation diversity per epoch
  3. cos_sim.png          - Cosine similarity per epoch
  4. diagnostics_all.png  - All 4 metrics in a 2x2 grid
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_csv(csv_path):
    """Parse per-iteration CSV log into per-epoch averages."""
    epoch_data = defaultdict(lambda: {'losses': [], 'ema': 0, 'lr': 0})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ep = int(row['epoch'])
            epoch_data[ep]['losses'].append(float(row['loss']))
            epoch_data[ep]['ema'] = float(row['ema'])
            epoch_data[ep]['lr'] = float(row['lr'])

    result = {}
    for ep in sorted(epoch_data.keys()):
        d = epoch_data[ep]
        result[ep] = {
            'train_loss': np.mean(d['losses']),
            'ema': d['ema'],
            'lr': d['lr'],
        }
    return result


def parse_stdout(stdout_path):
    """Parse DIAG lines and epoch summary from stdout log."""
    data = {}
    diag = {}
    with open(stdout_path) as f:
        for line in f:
            # DIAG line
            m = re.search(
                r'DIAG.*Epoch (\d+).*cos_sim=([\d.]+).*l2_dist=([\d.]+)\s+'
                r'rep_diversity=([\d.]+)', line)
            if m:
                ep = int(m.group(1))
                diag[ep] = {
                    'cos_sim': float(m.group(2)),
                    'l2_dist': float(m.group(3)),
                    'rep_div': float(m.group(4)),
                }

            # Epoch summary
            m = re.search(
                r'Epoch (\d+)/\d+.*train_loss=([\d.]+)\s+val_loss=([\d.]+)',
                line)
            if m:
                ep = int(m.group(1))
                if ep not in data:
                    data[ep] = {}
                data[ep]['train_loss'] = float(m.group(2))
                data[ep]['val_loss'] = float(m.group(3))
                if ep in diag:
                    data[ep].update(diag[ep])

    return data


def plot_loss(epochs, train_loss, val_loss, title, output_path):
    """Plot train and val loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_loss, label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{title} - Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def plot_metric(epochs, values, ylabel, title, output_path, color='tab:green',
                ylim=None):
    """Plot a single metric over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, values, color=color, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def plot_all(data, title, output_path):
    """Generate 2x2 grid of all diagnostics."""
    epochs = sorted(data.keys())
    has_diag = 'cos_sim' in data[epochs[0]] if epochs else False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title} - Pretraining Diagnostics', fontsize=14)

    # 1. Train & Val Loss
    ax = axes[0, 0]
    train = [data[e].get('train_loss', None) for e in epochs]
    val = [data[e].get('val_loss', None) for e in epochs]
    ax.plot(epochs, train, label='Train', linewidth=1.2)
    ax.plot(epochs, val, label='Val', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train & Val Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Rep Diversity
    ax = axes[0, 1]
    if has_diag:
        rd = [data[e].get('rep_div', None) for e in epochs]
        ax.plot(epochs, rd, color='tab:green', linewidth=1.2)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                    label='Collapsed (1.0)')
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5,
                    label='Healthy (~0.2)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('rep_diversity')
    ax.set_title('Representation Diversity (lower = better)')
    ax.grid(True, alpha=0.3)

    # 3. Cosine Similarity
    ax = axes[1, 0]
    if has_diag:
        cs = [data[e].get('cos_sim', None) for e in epochs]
        ax.plot(epochs, cs, color='tab:orange', linewidth=1.2)
        ax.set_ylim(0.5, 1.0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('cos_sim')
    ax.set_title('Predictor-Target Cosine Similarity')
    ax.grid(True, alpha=0.3)

    # 4. L2 Distance
    ax = axes[1, 1]
    if has_diag:
        l2 = [data[e].get('l2_dist', None) for e in epochs]
        ax.plot(epochs, l2, color='tab:purple', linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('l2_dist')
    ax.set_title('Predictor-Target L2 Distance')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate pretraining diagnostic plots')
    parser.add_argument('--csv', required=True,
                        help='Path to jepa_patch-log.csv')
    parser.add_argument('--stdout', required=True,
                        help='Path to torchrun_stdout.log')
    parser.add_argument('--output', required=True,
                        help='Output directory for plots')
    parser.add_argument('--title', default='Pretraining Run',
                        help='Plot title prefix')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Parsing CSV: {args.csv}')
    csv_data = parse_csv(args.csv)
    print(f'  Found {len(csv_data)} epochs in CSV')

    print(f'Parsing stdout: {args.stdout}')
    stdout_data = parse_stdout(args.stdout)
    print(f'  Found {len(stdout_data)} epochs in stdout')

    # Merge: stdout has val_loss + diagnostics, csv has train_loss
    merged = {}
    all_epochs = sorted(set(list(csv_data.keys()) + list(stdout_data.keys())))
    for ep in all_epochs:
        merged[ep] = {}
        if ep in csv_data:
            merged[ep]['train_loss'] = csv_data[ep]['train_loss']
            merged[ep]['ema'] = csv_data[ep]['ema']
        if ep in stdout_data:
            merged[ep].update(stdout_data[ep])

    epochs = sorted(merged.keys())
    if not epochs:
        print('No data found!')
        return

    # Individual plots
    train_loss = [merged[e].get('train_loss', float('nan')) for e in epochs]
    val_loss = [merged[e].get('val_loss', float('nan')) for e in epochs]
    plot_loss(epochs, train_loss, val_loss, args.title,
              os.path.join(args.output, 'train_val_loss.png'))

    has_diag = 'cos_sim' in merged[epochs[-1]]
    if has_diag:
        rd = [merged[e].get('rep_div', float('nan')) for e in epochs]
        plot_metric(epochs, rd, 'rep_diversity',
                    f'{args.title} - Representation Diversity',
                    os.path.join(args.output, 'rep_diversity.png'),
                    color='tab:green', ylim=(0, 1.05))

        cs = [merged[e].get('cos_sim', float('nan')) for e in epochs]
        plot_metric(epochs, cs, 'cos_sim',
                    f'{args.title} - Cosine Similarity',
                    os.path.join(args.output, 'cos_sim.png'),
                    color='tab:orange', ylim=(0.5, 1.0))

    # Combined 2x2
    plot_all(merged, args.title,
             os.path.join(args.output, 'diagnostics_all.png'))

    print('Done!')


if __name__ == '__main__':
    main()
