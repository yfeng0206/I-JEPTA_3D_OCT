#!/bin/bash
# AML entry point for downstream glaucoma classification.
# Downloads data + pretrained I-JEPA checkpoint, runs eval_downstream.py.
#
# Architecture: Frozen ViT-B/16 -> AttentiveProbe (2 blocks) -> LinearHead
# Adapted from I-JEPA attentive probe (Assran et al., 2023) with 2 blocks
# for inter-slice relationship learning (slices are independently encoded).

set -e

MODE=${MODE:-patch}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-64}
PATIENCE=${PATIENCE:-5}
SEED=${SEED:-42}
NUM_SLICES=${NUM_SLICES:-100}
NUM_WORKERS=${NUM_WORKERS:-4}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-50}
FREEZE_ENCODER=${FREEZE_ENCODER:-true}
NPROC=${NPROC:-4}
ACCUM_STEPS=${ACCUM_STEPS:-4}

# Blob path to pretrained I-JEPA checkpoint
IJEPA_BLOB_PREFIX=${IJEPA_BLOB_PREFIX:-"ijepa-results/patch_vit_base_ps16_ep50_bs64_lr0.00025_20260324_205416"}
IJEPA_CHECKPOINT_NAME=${IJEPA_CHECKPOINT_NAME:-"jepa_patch-best.pth.tar"}

# For slice mode: additional parameters
FE_BLOB_NAME=${FE_BLOB_NAME:-"checkpoints/feature_extractor.pth"}
ENC_DEPTH=${ENC_DEPTH:-6}
ENC_DIM=${ENC_DIM:-768}
ENC_HEADS=${ENC_HEADS:-12}

# Patch mode: additional parameters
ENCODER_NAME=${ENCODER_NAME:-vit_base}
PATCH_SIZE=${PATCH_SIZE:-16}
CROP_SIZE=${CROP_SIZE:-256}
PROBE_NUM_HEADS=${PROBE_NUM_HEADS:-12}
PROBE_DEPTH=${PROBE_DEPTH:-2}
HEAD_TYPE=${HEAD_TYPE:-linear}
LR_PROBE=${LR_PROBE:-0.0001}
LR_ENCODER=${LR_ENCODER:-0.000001}
LR_HEAD=${LR_HEAD:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DROPOUT=${DROPOUT:-0.1}

# Azure blob storage
BLOB_ACCOUNT=${BLOB_ACCOUNT:?'Set BLOB_ACCOUNT env var'}
BLOB_CONTAINER=${BLOB_CONTAINER:?'Set BLOB_CONTAINER env var'}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="downstream_${MODE}_s${NUM_SLICES}_ep${EPOCHS}_bs${BATCH_SIZE}_${HEAD_TYPE}"
OUTPUT_DIR="/tmp/ijepa_outputs/${RUN_TAG}_${TIMESTAMP}"
BLOB_PREFIX="ijepa-downstream/${RUN_TAG}_${TIMESTAMP}"
DATA_DIR="/tmp/fairvision_data"
CKPT_DIR="/tmp/ijepa_checkpoints"
mkdir -p $OUTPUT_DIR $CKPT_DIR

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install transformers scikit-learn pillow pyyaml azure-storage-blob azure-identity 2>&1 | tail -5

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

cred = ManagedIdentityCredential()
container = ContainerClient(
    account_url='https://%s.blob.core.windows.net' % account,
    container_name=container_name,
    credential=cred,
)

if not os.path.exists(os.path.join(local_dir, 'Training')):
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

echo "=== Downloading I-JEPA checkpoint ==="
IJEPA_CKPT_PATH="${CKPT_DIR}/${IJEPA_CHECKPOINT_NAME}"
python -c "
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os

blob_name = '${IJEPA_BLOB_PREFIX}/${IJEPA_CHECKPOINT_NAME}'
local_path = '${IJEPA_CKPT_PATH}'

if os.path.exists(local_path):
    print('Checkpoint already exists: %s' % local_path)
else:
    print('Downloading: %s' % blob_name)
    cred = ManagedIdentityCredential()
    blob = BlobClient(
        account_url='https://${BLOB_ACCOUNT}.blob.core.windows.net',
        container_name='${BLOB_CONTAINER}',
        blob_name=blob_name,
        credential=cred,
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(blob.download_blob().readall())
    print('Downloaded: %d bytes' % os.path.getsize(local_path))
"

# Download feature extractor if slice mode
FE_CKPT_PATH="${DATA_DIR}/feature_extractor.pth"
if [ "$MODE" = "slice" ]; then
    echo "=== Downloading feature extractor checkpoint ==="
    python -c "
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os

local_path = '${FE_CKPT_PATH}'
if os.path.exists(local_path):
    print('FE checkpoint already exists.')
else:
    print('Downloading feature_extractor.pth...')
    cred = ManagedIdentityCredential()
    blob = BlobClient(
        account_url='https://${BLOB_ACCOUNT}.blob.core.windows.net',
        container_name='${BLOB_CONTAINER}',
        blob_name='${FE_BLOB_NAME}',
        credential=cred,
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(blob.download_blob().readall())
    print('  Saved to %s (%d bytes)' % (local_path, os.path.getsize(local_path)))
"
fi

echo "=== Generating config ==="
CONFIG_PATH="${OUTPUT_DIR}/config.yaml"

if [ "$MODE" = "patch" ]; then
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
  encoder_checkpoint: ${IJEPA_CKPT_PATH}
  encoder_name: ${ENCODER_NAME}
  patch_size: ${PATCH_SIZE}
  crop_size: ${CROP_SIZE}
  freeze_encoder: ${FREEZE_ENCODER}
  probe_num_heads: ${PROBE_NUM_HEADS}
  probe_depth: ${PROBE_DEPTH}
  head_type: ${HEAD_TYPE}
training:
  lr_probe: ${LR_PROBE}
  lr_encoder: ${LR_ENCODER}
  lr_head: ${LR_HEAD}
  weight_decay: ${WEIGHT_DECAY}
  dropout: ${DROPOUT}
  epochs: ${EPOCHS}
  patience: ${PATIENCE}
  warmup_epochs: 3
  accum_steps: ${ACCUM_STEPS}
  seed: ${SEED}
logging:
  output_dir: ${OUTPUT_DIR}
YAMLEOF
elif [ "$MODE" = "slice" ]; then
cat > "${CONFIG_PATH}" << YAMLEOF
mode: slice
data:
  data_dir: ${DATA_DIR}/data
  num_slices: ${NUM_SLICES}
  slice_size: 256
  batch_size: ${BATCH_SIZE}
  num_workers: ${NUM_WORKERS}
model:
  slice_encoder_checkpoint: ${IJEPA_CKPT_PATH}
  fe_checkpoint: ${FE_CKPT_PATH}
  enc_depth: ${ENC_DEPTH}
  enc_dim: ${ENC_DIM}
  enc_heads: ${ENC_HEADS}
  freeze_encoder: true
training:
  lr: ${LR_HEAD}
  weight_decay: ${WEIGHT_DECAY}
  epochs: ${EPOCHS}
  patience: ${PATIENCE}
  seed: ${SEED}
logging:
  output_dir: ${OUTPUT_DIR}
YAMLEOF
else
    echo "ERROR: Unknown MODE=$MODE (expected 'patch' or 'slice')"
    exit 1
fi

echo "=== Config ==="
cat "${CONFIG_PATH}"

echo "=== Running downstream evaluation ==="
echo "  Mode: ${MODE}"
echo "  Num slices: ${NUM_SLICES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Head type: ${HEAD_TYPE}"
echo "  Output: ${OUTPUT_DIR}"

EVAL_EXIT=0
cd "$(dirname "$0")/.."
if [ "$FREEZE_ENCODER" = "false" ]; then
    echo "  Launching with torchrun (${NPROC} GPUs, encoder unfrozen)"
    torchrun --nproc_per_node=${NPROC} src/eval_downstream.py --config "${CONFIG_PATH}" 2>&1 | tee "${OUTPUT_DIR}/eval_stdout.log" || EVAL_EXIT=$?
else
    echo "  Launching single GPU (encoder frozen, cached features)"
    python src/eval_downstream.py --config "${CONFIG_PATH}" 2>&1 | tee "${OUTPUT_DIR}/eval_stdout.log" || EVAL_EXIT=$?
fi

echo "=== Eval exit code: ${EVAL_EXIT} ==="
echo "=== Output files ==="
ls -la "${OUTPUT_DIR}/"

# Print results if available
if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo "=== Results ==="
    cat "${OUTPUT_DIR}/results.json"
fi

echo "=== Uploading results to blob storage ==="
python -c "
import glob, os, traceback
output_dir = '${OUTPUT_DIR}'
blob_prefix = '${BLOB_PREFIX}'
files = [f for f in glob.glob(os.path.join(output_dir, '*')) if os.path.isfile(f)]
print('Found %d files to upload' % len(files))
if not files:
    print('Nothing to upload!')
    exit(0)
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
        size = os.path.getsize(fpath)
        print('  %s (%s bytes) -> %s' % (fname, format(size, ','), blob_name))
        with open(fpath, 'rb') as f:
            container.upload_blob(blob_name, f, overwrite=True)
    print('Upload complete!')
except Exception as e:
    print('Upload FAILED: %s' % e)
    traceback.print_exc()
"

echo "=== All done (eval exit=${EVAL_EXIT}) ==="
exit ${EVAL_EXIT}
