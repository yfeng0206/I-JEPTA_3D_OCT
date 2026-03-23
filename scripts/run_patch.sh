#!/bin/bash
# AML entry point for patch-level I-JEPA pretraining.
# Reads env vars, downloads data from blob, runs torchrun, uploads results.

set -e

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-0.00025}
START_LR=${START_LR:-0.0001}
FINAL_LR=${FINAL_LR:-0.000001}
WARMUP=${WARMUP:-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.04}
FINAL_WEIGHT_DECAY=${FINAL_WEIGHT_DECAY:-0.4}
EMA_START=${EMA_START:-0.996}
EMA_END=${EMA_END:-1.0}
CROP_SIZE=${CROP_SIZE:-256}
PATCH_SIZE=${PATCH_SIZE:-16}
NUM_SLICES=${NUM_SLICES:-32}
MODEL_NAME=${MODEL_NAME:-vit_base}
PRED_DEPTH=${PRED_DEPTH:-6}
PRED_EMB_DIM=${PRED_EMB_DIM:-384}
NUM_WORKERS=${NUM_WORKERS:-4}
NPROC=${NPROC:-4}
ACCUM_STEPS=${ACCUM_STEPS:-1}
PATIENCE=${PATIENCE:-15}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="patch_${MODEL_NAME}_ps${PATCH_SIZE}_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}"
OUTPUT_DIR="/tmp/ijepa_outputs/${RUN_TAG}_${TIMESTAMP}"
BLOB_PREFIX="ijepa-results/${RUN_TAG}_${TIMESTAMP}"
DATA_DIR="/tmp/fairvision_data"
mkdir -p $OUTPUT_DIR

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install transformers scikit-learn pillow pyyaml azure-storage-blob azure-identity 2>&1 | tail -5

echo "=== Downloading data ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT/src"
python -c "
import sys
sys.path.insert(0, '.')
from setup_data_ijepa import download_data
download_data('${DATA_DIR}')
" 2>/dev/null || python -c "
import os, sys
sys.path.insert(0, '$(dirname "$0")/../../FairVision/src')
from setup_data import main as setup_main
sys.argv = ['setup_data.py', '--output_dir', '${DATA_DIR}']
setup_main()
" 2>/dev/null || python -c "
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
import os

account = 'STORAGE_ACCOUNT_REDACTED'
container_name = 'CONTAINER_REDACTED'
prefix = 'fhl-test-data'
local_dir = '${DATA_DIR}/data'
os.makedirs(local_dir, exist_ok=True)

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url='https://%s.blob.core.windows.net' % account,
    container_name=container_name,
    credential=credential,
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
"
cd "$PROJECT_ROOT"

echo "=== Generating config ==="
CONFIG_PATH="${OUTPUT_DIR}/config.yaml"
cat > "${CONFIG_PATH}" << YAMLEOF
data:
  batch_size: ${BATCH_SIZE}
  crop_size: ${CROP_SIZE}
  crop_scale: [0.3, 1.0]
  num_workers: ${NUM_WORKERS}
  pin_mem: true
  data_dir: ${DATA_DIR}/data
  num_slices: ${NUM_SLICES}
  color_jitter_strength: 0.0
  use_color_distortion: false
  use_gaussian_blur: false
  use_horizontal_flip: false
mask:
  patch_size: ${PATCH_SIZE}
  num_enc_masks: 1
  num_pred_masks: 4
  enc_mask_scale: [0.85, 1.0]
  pred_mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]
  allow_overlap: false
  min_keep: 10
meta:
  model_name: ${MODEL_NAME}
  pred_depth: ${PRED_DEPTH}
  pred_emb_dim: ${PRED_EMB_DIM}
  use_bfloat16: false
  load_checkpoint: false
  read_checkpoint: null
optimization:
  epochs: ${EPOCHS}
  lr: ${LR}
  start_lr: ${START_LR}
  final_lr: ${FINAL_LR}
  warmup: ${WARMUP}
  weight_decay: ${WEIGHT_DECAY}
  final_weight_decay: ${FINAL_WEIGHT_DECAY}
  ema: [${EMA_START}, ${EMA_END}]
  ipe_scale: 1.0
  accum_steps: ${ACCUM_STEPS}
  patience: ${PATIENCE}
logging:
  folder: ${OUTPUT_DIR}
  write_tag: jepa_patch
YAMLEOF

echo "=== Config ==="
cat "${CONFIG_PATH}"

echo "=== Starting patch-level I-JEPA pretraining ==="
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE} x ${NPROC} GPUs"
echo "  LR: ${LR}"
echo "  Output: ${OUTPUT_DIR}"

TRAIN_EXIT=0
torchrun \
  --nproc_per_node=${NPROC} \
  src/train_patch.py \
  --config "${CONFIG_PATH}" 2>&1 | tee "${OUTPUT_DIR}/torchrun_stdout.log" || TRAIN_EXIT=$?

echo "=== Training exit code: ${TRAIN_EXIT} ==="
echo "=== Output files ==="
ls -la "${OUTPUT_DIR}/"

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
        account_url='https://STORAGE_ACCOUNT_REDACTED.blob.core.windows.net',
        container_name='CONTAINER_REDACTED',
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

echo "=== All done (train exit=${TRAIN_EXIT}) ==="
exit ${TRAIN_EXIT}
