#!/bin/bash
# Run 4 linear probe evaluations in parallel across 4 GPUs.
# Each GPU evaluates a different pretraining checkpoint (ep25, ep50, ep75, ep100).
# Data download happens once (shared), then 4 eval processes run simultaneously.

set -euo pipefail

# ── Shared settings (from AML env vars) ─────────────────────────────
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
PATIENCE=${PATIENCE:-20}
SEED=${SEED:-42}
NUM_SLICES=${NUM_SLICES:-100}
NUM_WORKERS=${NUM_WORKERS:-4}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-50}
ENCODER_NAME=${ENCODER_NAME:-vit_base}
PATCH_SIZE=${PATCH_SIZE:-16}
CROP_SIZE=${CROP_SIZE:-256}
PROBE_NUM_HEADS=${PROBE_NUM_HEADS:-12}
PROBE_DEPTH=${PROBE_DEPTH:-3}
HEAD_TYPE=${HEAD_TYPE:-linear}
LR_PROBE=${LR_PROBE:-0.0001}
LR_HEAD=${LR_HEAD:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
DROPOUT=${DROPOUT:-0.1}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
PROBE_TYPE=${PROBE_TYPE:-attentive}       # 'attentive' | 'cross_attn_pool'
PROBE_HEAD_DIM=${PROBE_HEAD_DIM:-64}      # only used when PROBE_TYPE=cross_attn_pool

IJEPA_BLOB_PREFIX=${IJEPA_BLOB_PREFIX:?'Set IJEPA_BLOB_PREFIX env var'}
BLOB_ACCOUNT=${BLOB_ACCOUNT:?'Set BLOB_ACCOUNT env var'}
BLOB_CONTAINER=${BLOB_CONTAINER:?'Set BLOB_CONTAINER env var'}

DATA_DIR="/tmp/fairvision_data"
CKPT_DIR="/tmp/ijepa_checkpoints"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$CKPT_DIR"

CHECKPOINTS=("jepa_patch-ep25.pth.tar" "jepa_patch-ep50.pth.tar" "jepa_patch-ep75.pth.tar" "jepa_patch-ep100.pth.tar")

echo "=== Linear Probe Sweep ==="
echo "  Checkpoints: ${CHECKPOINTS[*]}"
echo "  Mode: sequential on GPU 0"
echo "  Blob prefix: ${IJEPA_BLOB_PREFIX}"

# ── Install dependencies ────────────────────────────────────────────
echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version" || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install transformers scikit-learn pillow pyyaml azure-storage-blob azure-identity 2>&1 | tail -5

# ── Download data (once, shared across all 4 runs) ─────────────────
echo "=== Downloading data ==="
python -c "
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
import os

account = '${BLOB_ACCOUNT}'
container_name = '${BLOB_CONTAINER}'
prefix = 'fhl-test-data'
local_dir = '${DATA_DIR}/data'
os.makedirs(local_dir, exist_ok=True)

if not os.path.exists(os.path.join(local_dir, 'Training')):
    cred = ManagedIdentityCredential()
    container = ContainerClient(
        account_url='https://%s.blob.core.windows.net' % account,
        container_name=container_name,
        credential=cred,
    )
    blobs = list(container.list_blobs(name_starts_with=prefix))
    print('Found %d blobs' % len(blobs))
    downloaded = 0
    for blob in blobs:
        if not blob.name.endswith('.npz'):
            continue
        rel_path = blob.name[len(prefix):].lstrip('/')
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            with open(local_path, 'wb') as f:
                f.write(container.download_blob(blob).readall())
            downloaded += 1
            if downloaded % 500 == 0:
                print('Downloaded %d files...' % downloaded)
    print('Downloaded %d new files' % downloaded)
else:
    print('Data already exists.')

for split in ['Training', 'Validation', 'Test']:
    split_dir = os.path.join(local_dir, split)
    if os.path.exists(split_dir):
        n = len([f for f in os.listdir(split_dir) if f.endswith('.npz')])
        print('  %s: %d files' % (split, n))
"

# ── Download all 4 checkpoints ──────────────────────────────────────
echo "=== Downloading checkpoints ==="
python -c "
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os

cred = ManagedIdentityCredential()
checkpoints = '${CHECKPOINTS[*]}'.split()
for ckpt_name in checkpoints:
    local_path = os.path.join('${CKPT_DIR}', ckpt_name)
    if os.path.exists(local_path):
        print('Already exists: %s' % ckpt_name)
        continue
    blob_name = '${IJEPA_BLOB_PREFIX}/' + ckpt_name
    print('Downloading: %s' % blob_name)
    blob = BlobClient(
        account_url='https://${BLOB_ACCOUNT}.blob.core.windows.net',
        container_name='${BLOB_CONTAINER}',
        blob_name=blob_name,
        credential=cred,
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(blob.download_blob().readall())
    print('  Downloaded: %d bytes' % os.path.getsize(local_path))
print('All checkpoints ready.')
"

# ── Run 4 probes sequentially on GPU 0 ──────────────────────────────
# Prior parallel version hung: 4 probes each holding ~2 GB of cached
# features in RAM plus forking 4 DataLoader workers apiece blew past
# node memory. Sequential avoids the contention; total wall time
# ~4× single-probe instead of ~1× parallel.
cd "$(dirname "$0")/.."

EXIT_CODES=()
ALL_OK=0

for i in 0 1 2 3; do
    CKPT_NAME="${CHECKPOINTS[$i]}"
    EP_TAG=$(echo "$CKPT_NAME" | sed 's/jepa_patch-//;s/.pth.tar//')
    RUN_TAG="downstream_linear_${EP_TAG}_d${PROBE_DEPTH}_s${NUM_SLICES}"
    RUN_OUTPUT="/tmp/ijepa_outputs/${RUN_TAG}_${TIMESTAMP}"
    BLOB_OUT="ijepa-downstream/${RUN_TAG}_${TIMESTAMP}"
    CONFIG_PATH="${RUN_OUTPUT}/config.yaml"
    CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}"

    mkdir -p "$RUN_OUTPUT"

    # Generate config for this checkpoint
    cat > "${CONFIG_PATH}" << YAMLEOF
mode: patch
data:
  data_dir: ${DATA_DIR}/data
  num_slices: ${NUM_SLICES}
  slice_size: ${CROP_SIZE}
  batch_size: ${BATCH_SIZE}
  num_workers: ${NUM_WORKERS}
  encode_chunk_size: ${ENCODE_CHUNK_SIZE}
model:
  encoder_checkpoint: ${CKPT_PATH}
  encoder_name: ${ENCODER_NAME}
  patch_size: ${PATCH_SIZE}
  crop_size: ${CROP_SIZE}
  freeze_encoder: true
  probe_type: ${PROBE_TYPE}
  probe_num_heads: ${PROBE_NUM_HEADS}
  probe_depth: ${PROBE_DEPTH}
  probe_head_dim: ${PROBE_HEAD_DIM}
  head_type: ${HEAD_TYPE}
training:
  lr_probe: ${LR_PROBE}
  lr_encoder: 0.000001
  lr_head: ${LR_HEAD}
  weight_decay: ${WEIGHT_DECAY}
  dropout: ${DROPOUT}
  epochs: ${EPOCHS}
  patience: ${PATIENCE}
  warmup_epochs: ${WARMUP_EPOCHS}
  accum_steps: 1
  seed: ${SEED}
logging:
  output_dir: ${RUN_OUTPUT}
YAMLEOF

    echo ""
    echo "=== [$((i+1))/4] Running ${EP_TAG} ==="
    echo "  Config: ${CONFIG_PATH}"
    echo "  Output: ${RUN_OUTPUT}"

    set +e
    EVAL_EXIT=0
    CUDA_VISIBLE_DEVICES=0 python src/eval_downstream.py --config "${CONFIG_PATH}" \
        2>&1 | tee "${RUN_OUTPUT}/eval_stdout.log"
    EVAL_EXIT=${PIPESTATUS[0]}
    set -e

    # Upload results (runs even if eval failed)
    python -c "
import glob, os, traceback
output_dir = '${RUN_OUTPUT}'
blob_prefix = '${BLOB_OUT}'
files = [f for f in glob.glob(os.path.join(output_dir, '*')) if os.path.isfile(f)]
print('[${EP_TAG}] Uploading %d files...' % len(files))
try:
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import ContainerClient
    cred = ManagedIdentityCredential()
    container = ContainerClient(
        account_url='https://${BLOB_ACCOUNT}.blob.core.windows.net',
        container_name='${BLOB_CONTAINER}',
        credential=cred,
    )
    for fpath in files:
        fname = os.path.basename(fpath)
        blob_name = '%s/%s' % (blob_prefix, fname)
        with open(fpath, 'rb') as f:
            container.upload_blob(blob_name, f, overwrite=True)
    print('[${EP_TAG}] Upload complete!')
except Exception as e:
    print('[${EP_TAG}] Upload FAILED: %s' % e)
    traceback.print_exc()
"

    # Free RAM held by cached features before next probe starts
    rm -rf "${RUN_OUTPUT}/feature_cache" || true

    EXIT_CODES+=($EVAL_EXIT)
    echo "  ${CKPT_NAME}: exit code ${EVAL_EXIT}"
    if [ $EVAL_EXIT -ne 0 ]; then
        ALL_OK=1
    fi
done

# ── Print summary ───────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Linear Probe Sweep Complete"
echo "========================================"
for i in 0 1 2 3; do
    EP_TAG=$(echo "${CHECKPOINTS[$i]}" | sed 's/jepa_patch-//;s/.pth.tar//')
    RUN_TAG="downstream_linear_${EP_TAG}_d${PROBE_DEPTH}_s${NUM_SLICES}"
    RUN_OUTPUT="/tmp/ijepa_outputs/${RUN_TAG}_${TIMESTAMP}"
    echo ""
    echo "--- ${EP_TAG} (exit: ${EXIT_CODES[$i]}) ---"
    if [ -f "${RUN_OUTPUT}/results.json" ]; then
        cat "${RUN_OUTPUT}/results.json"
    else
        echo "  No results.json found"
    fi
done

echo ""
echo "=== All done ==="
exit ${ALL_OK}
