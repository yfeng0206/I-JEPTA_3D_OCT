"""Regenerate all comparison plots for the experiment docs.

Main README gets clean, simple charts:
  - results/test_auc_comparison.png   (bar chart, all methods)
  - results/imagenet_degradation.png  (test AUC vs pretraining epoch)
  - results/pretraining_loss.png      (combined ep1-100 loss curve)

Experiment detail pages get training curves:
  - results/frozen_probe_val_auc.png  (val AUC curves, 6 runs)
  - results/frozen_probe_loss.png     (train+val loss curves, 6 runs)
"""
import os, json, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base, 'results')
downstream_dir = os.path.join(results_dir, 'downstream')
pretrain_dir = os.path.join(results_dir, 'pretraining')

# ============================================================
# Load data
# ============================================================
all_logs = {}
all_results = {}
run_ids = [
    'frozen_random_d2_s100', 'frozen_random_d3_s100',
    'frozen_imagenet_ep32_d3_s100', 'frozen_imagenet_ep50_d3_s100',
    'frozen_imagenet_ep75_d3_s100', 'frozen_imagenet_ep99_d3_s100',
]

for run_id in run_ids:
    log_path = os.path.join(downstream_dir, run_id, 'train_log.csv')
    res_path = os.path.join(downstream_dir, run_id, 'results.json')
    if os.path.exists(log_path):
        with open(log_path) as f:
            all_logs[run_id] = list(csv.DictReader(f))
    if os.path.exists(res_path):
        with open(res_path) as f:
            all_results[run_id] = json.loads(f.read())

display_names = {
    'frozen_random_d2_s100': 'Random->SSL (d=2, Linear)',
    'frozen_random_d3_s100': 'Random->SSL (d=3, Linear)',
    'frozen_imagenet_ep32_d3_s100': 'ImageNet->SSL ep32 (d=3, MLP)',
    'frozen_imagenet_ep50_d3_s100': 'ImageNet->SSL ep50 (d=3, MLP)',
    'frozen_imagenet_ep75_d3_s100': 'ImageNet->SSL ep75 (d=3, MLP)',
    'frozen_imagenet_ep99_d3_s100': 'ImageNet->SSL ep99 (d=3, MLP)',
}

colors = {
    'frozen_random_d2_s100': '#9467bd',
    'frozen_random_d3_s100': '#8c564b',
    'frozen_imagenet_ep32_d3_s100': '#1f77b4',
    'frozen_imagenet_ep50_d3_s100': '#ff7f0e',
    'frozen_imagenet_ep75_d3_s100': '#2ca02c',
    'frozen_imagenet_ep99_d3_s100': '#d62728',
}

# ============================================================
# MAIN README: Test AUC bar chart (clean, all methods)
# ============================================================
print("=== Main: Test AUC comparison bar chart ===")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Sorted by test AUC descending
bar_data = [
    ('SLIViT\nbaseline', 0.869, '#333333'),
    ('Unfrozen\nIN->SSL d=2', 0.828, '#e65100'),
    ('Frozen\nIN->SSL ep32', 0.774, '#1f77b4'),
    ('Frozen\nRandom d=3', 0.734, '#8c564b'),
    ('Frozen\nRandom d=2', 0.733, '#9467bd'),
    ('Frozen\nIN->SSL ep50', 0.706, '#ff7f0e'),
    ('Frozen\nIN->SSL ep75', 0.695, '#2ca02c'),
    ('Frozen\nIN->SSL ep99', 0.685, '#d62728'),
]

labels = [d[0] for d in bar_data]
aucs = [d[1] for d in bar_data]
bar_colors = [d[2] for d in bar_data]

bars = ax.bar(labels, aucs, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.7)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Test AUC', fontsize=13)
ax.set_title('Glaucoma Classification: Test AUC Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0.5, 0.95)
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'test_auc_comparison.png'), dpi=150, bbox_inches='tight')
print('  Saved test_auc_comparison.png')
plt.close()

# ============================================================
# MAIN README: Degradation plot (clean)
# ============================================================
print("=== Main: Degradation plot ===")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
pretrain_epochs = [32, 50, 75, 99]
epoch_runs = ['frozen_imagenet_ep32_d3_s100', 'frozen_imagenet_ep50_d3_s100',
              'frozen_imagenet_ep75_d3_s100', 'frozen_imagenet_ep99_d3_s100']
test_by_ep = [all_results[r]['test_auc'] for r in epoch_runs]

ax.plot(pretrain_epochs, test_by_ep, 'o-', color='#1f77b4', linewidth=2.5, markersize=10,
        label='ImageNet->SSL (frozen d=3)')
ax.axhline(y=0.734, color='#9467bd', linestyle='--', linewidth=1.5,
           label='Random->SSL frozen d=3 (0.734)')
ax.axhline(y=0.869, color='black', linestyle='--', linewidth=1.5,
           label='SLIViT baseline (0.869)')

for ep, auc in zip(pretrain_epochs, test_by_ep):
    ax.annotate(f'{auc:.3f}', (ep, auc), textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('I-JEPA Pretraining Epoch (ImageNet init)', fontsize=12)
ax.set_ylabel('Downstream Test AUC (frozen probe)', fontsize=12)
ax.set_title('ImageNet Features Degrade with More I-JEPA Pretraining',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.6, 0.95)
ax.set_xlim(25, 105)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'imagenet_degradation.png'), dpi=150, bbox_inches='tight')
print('  Saved imagenet_degradation.png')
plt.close()

# ============================================================
# MAIN README: Pretraining loss (combined ep1-100)
# ============================================================
print("=== Main: Combined pretraining loss ===")

all_epoch_loss = {}
for run_name in ['pretrain_imagenet_init', 'pretrain_warmrestart', 'pretrain_random_100ep']:
    log_path = os.path.join(pretrain_dir, run_name, 'jepa_patch-log.csv')
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            ep = int(r.get('epoch', 0))
            loss = float(r.get('loss', 0))
            if ep not in all_epoch_loss:
                all_epoch_loss[ep] = []
            all_epoch_loss[ep].append(loss)

epochs_sorted = sorted(all_epoch_loss.keys())
avg_losses = [np.mean(all_epoch_loss[ep]) for ep in epochs_sorted]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(epochs_sorted, avg_losses, color='#1f77b4', linewidth=2)

ax.axvline(x=20.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=30.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ymax = max(avg_losses) * 1.02
ax.text(10, ymax, 'Job 1\n(ImageNet init)', ha='center', fontsize=9, color='gray')
ax.text(25.5, ymax, 'Job 2\n(resume)', ha='center', fontsize=9, color='gray')
ax.text(66, ymax, 'Job 3 (resume, 100 ep target)', ha='center', fontsize=9, color='gray')

best_ep = 32
best_loss = np.mean(all_epoch_loss[best_ep])
ax.plot(best_ep, best_loss, 'r*', markersize=15, zorder=5)
ax.annotate(
    'Best ckpt (ep32)\n0.774 frozen / 0.828 fine-tune',
    (best_ep, best_loss), textcoords="offset points", xytext=(60, -25),
    fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('I-JEPA Loss (Smooth L1)', fontsize=12)
ax.set_title('ImageNet-init I-JEPA Pretraining (Combined)', fontsize=14, fontweight='bold')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 102)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'pretraining_loss.png'), dpi=150, bbox_inches='tight')
print('  Saved pretraining_loss.png')
plt.close()

# ============================================================
# DETAIL: Frozen probe Val AUC curves (for experiment page)
# ============================================================
print("=== Detail: Frozen probe val AUC curves ===")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for run_id in run_ids:
    if run_id not in all_logs:
        continue
    rows = all_logs[run_id]
    epochs = [int(r['epoch']) for r in rows]
    val_auc = [float(r['val_auc']) for r in rows]
    ax.plot(epochs, val_auc, label=display_names[run_id], color=colors[run_id], linewidth=2)
ax.axhline(y=0.869, color='black', linestyle='--', linewidth=1.5, label='SLIViT baseline (0.869)')
ax.set_xlabel('Probe Training Epoch', fontsize=12)
ax.set_ylabel('Validation AUC', fontsize=12)
ax.set_title('Frozen Probe: Val AUC During Training', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.55, 0.90)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'frozen_probe_val_auc.png'), dpi=150, bbox_inches='tight')
print('  Saved frozen_probe_val_auc.png')
plt.close()

# ============================================================
# DETAIL: Frozen probe Train/Val loss curves (for experiment page)
# ============================================================
print("=== Detail: Frozen probe train/val loss curves ===")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for run_id in run_ids:
    if run_id not in all_logs:
        continue
    rows = all_logs[run_id]
    epochs = [int(r['epoch']) for r in rows]
    train_loss = [float(r['train_loss']) for r in rows]
    val_loss = [float(r['val_loss']) for r in rows]
    ax.plot(epochs, train_loss, color=colors[run_id], linewidth=1.5, alpha=0.4)
    ax.plot(epochs, val_loss, color=colors[run_id], linewidth=2,
            label=display_names[run_id])
ax.plot([], [], color='gray', linewidth=1.5, alpha=0.4, label='(thin = train)')
ax.plot([], [], color='gray', linewidth=2, label='(thick = val)')
ax.set_xlabel('Probe Training Epoch', fontsize=12)
ax.set_ylabel('Loss (BCE)', fontsize=12)
ax.set_title('Frozen Probe: Train vs Val Loss', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'frozen_probe_loss.png'), dpi=150, bbox_inches='tight')
print('  Saved frozen_probe_loss.png')
plt.close()

# Clean up old combined 3-panel chart
old_combined = os.path.join(results_dir, 'frozen_probe_comparison.png')
if os.path.exists(old_combined):
    os.remove(old_combined)
    print('  Removed old frozen_probe_comparison.png')

print("\n=== All graphs regenerated ===")
