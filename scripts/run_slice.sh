#!/bin/bash
# AML entry point for slice-level I-JEPA pretraining.
# Downloads data + feature_extractor checkpoint, runs torchrun, uploads results.

set -e

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-0.0003}
START_LR=${START_LR:-0.00005}
FINAL_LR=${FINAL_LR:-0.000001}
WARMUP=${WARMUP:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.04}
FINAL_WEIGHT_DECAY=${FINAL_WEIGHT_DECAY:-0.4}
EMA_START=${EMA_START:-0.996}
EMA_END=${EMA_END:-1.0}
NUM_SLICES=${NUM_SLICES:-32}
SLICE_SIZE=${SLICE_SIZE:-256}
ENC_DEPTH=${ENC_DEPTH:-6}
ENC_DIM=${ENC_DIM:-768}
ENC_HEADS=${ENC_HEADS:-12}
PRED_DEPTH=${PRED_DEPTH:-6}
PRED_EMB_DIM=${PRED_EMB_DIM:-384}
NUM_WORKERS=${NUM_WORKERS:-4}
NPROC=${NPROC:-4}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="slice_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}"
OUTPUT_DIR="/tmp/ijepa_outputs/${RUN_TAG}_${TIMESTAMP}"
BLOB_PREFIX="ijepa-results/${RUN_TAG}_${TIMESTAMP}"
DATA_DIR="/tmp/fairvision_data"
FE_CHECKPOINT="${DATA_DIR}/feature_extractor.pth"
mkdir -p $OUTPUT_DIR

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install transformers scikit-learn pillow pyyaml azure-storage-blob azure-identity 2>&1 | tail -5

echo "=== Downloading data and feature extractor checkpoint ==="
python -c "
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient, BlobClient
import os

account = 'YOUR_STORAGE_ACCOUNT'
container_name = 'YOUR_CONTAINER_NAME'
prefix = 'YOUR_DATA_PREFIX'
local_dir = '${DATA_DIR}/data'
fe_path = '${FE_CHECKPOINT}'
os.makedirs(local_dir, exist_ok=True)

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url='https://%s.blob.core.windows.net' % account,
    container_name=container_name,
    credential=credential,
)

# Download dataset
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
    print('Data already exists, skipping download.')

# Download feature extractor checkpoint
if not os.path.exists(fe_path):
    print('Downloading feature_extractor.pth...')
    os.makedirs(os.path.dirname(fe_path), exist_ok=True)
    blob = BlobClient(
        account_url='https://%s.blob.core.windows.net' % account,
        container_name=container_name,
        blob_name='checkpoints/feature_extractor.pth',
        credential=credential,
    )
    with open(fe_path, 'wb') as f:
        f.write(blob.download_blob().readall())
    print('  Saved to %s (%d bytes)' % (fe_path, os.path.getsize(fe_path)))
else:
    print('Feature extractor checkpoint already exists.')

# Print summary
for split in ['Training', 'Validation', 'Test']:
    split_dir = os.path.join(local_dir, split)
    if os.path.exists(split_dir):
        n = len([f for f in os.listdir(split_dir) if f.endswith('.npz')])
        print('  %s: %d files' % (split, n))
"

echo "=== Generating config ==="
CONFIG_PATH="${OUTPUT_DIR}/config.yaml"
cat > "${CONFIG_PATH}" << YAMLEOF
data:
  batch_size: ${BATCH_SIZE}
  num_workers: ${NUM_WORKERS}
  pin_mem: true
  data_dir: ${DATA_DIR}/data
  num_slices: ${NUM_SLICES}
  slice_size: ${SLICE_SIZE}
mask:
  num_slices: ${NUM_SLICES}
  num_pred_masks: 4
  enc_mask_scale: [0.75, 0.9]
  pred_mask_scale: [0.1, 0.2]
  min_keep: 10
meta:
  enc_depth: ${ENC_DEPTH}
  enc_dim: ${ENC_DIM}
  enc_heads: ${ENC_HEADS}
  pred_depth: ${PRED_DEPTH}
  pred_emb_dim: ${PRED_EMB_DIM}
  use_bfloat16: false
  load_checkpoint: false
  read_checkpoint: null
  fe_checkpoint: ${FE_CHECKPOINT}
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
logging:
  folder: ${OUTPUT_DIR}
  write_tag: jepa_slice
YAMLEOF

echo "=== Config ==="
cat "${CONFIG_PATH}"

echo "=== Starting slice-level I-JEPA pretraining ==="
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE} x ${NPROC} GPUs"
echo "  LR: ${LR}"
echo "  FE checkpoint: ${FE_CHECKPOINT}"
echo "  Output: ${OUTPUT_DIR}"

TRAIN_EXIT=0
cd "$(dirname "$0")/.."
torchrun \
  --nproc_per_node=${NPROC} \
  src/train_slice.py \
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
        account_url='https://YOUR_STORAGE_ACCOUNT.blob.core.windows.net',
        container_name='YOUR_CONTAINER_NAME',
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
