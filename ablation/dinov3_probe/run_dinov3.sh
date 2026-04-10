#!/bin/bash
# Entry point for DINOv3 ablation on AML or local.
# Downloads data, installs deps, runs eval_dinov3.py.

set -e

MODE=${MODE:-frozen}  # frozen or unfrozen
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-64}
PATIENCE=${PATIENCE:-20}
NUM_SLICES=${NUM_SLICES:-100}
NUM_WORKERS=${NUM_WORKERS:-4}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-100}
PROBE_DEPTH=${PROBE_DEPTH:-3}
HEAD_TYPE=${HEAD_TYPE:-mlp}
LR_PROBE=${LR_PROBE:-0.0001}
LR_HEAD=${LR_HEAD:-0.001}
LR_ENCODER=${LR_ENCODER:-0.000005}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
NPROC=${NPROC:-4}

BLOB_ACCOUNT=${BLOB_ACCOUNT:?'Set BLOB_ACCOUNT env var'}
BLOB_CONTAINER=${BLOB_CONTAINER:?'Set BLOB_CONTAINER env var'}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="dinov3_${MODE}_s${NUM_SLICES}_${HEAD_TYPE}"
OUTPUT_DIR="/tmp/dinov3_outputs/${RUN_TAG}_${TIMESTAMP}"
BLOB_PREFIX="dinov3-ablation/${RUN_TAG}_${TIMESTAMP}"
DATA_DIR="/tmp/fairvision_data"
mkdir -p $OUTPUT_DIR

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

echo "=== Generating config ==="
FREEZE_FLAG="true"
if [ "$MODE" = "unfrozen" ]; then
    FREEZE_FLAG="false"
    WEIGHT_DECAY="0.01"
    NUM_SLICES=${NUM_SLICES:-32}
fi

CONFIG_PATH="${OUTPUT_DIR}/config.yaml"
cat > "${CONFIG_PATH}" << YAMLEOF
mode: dinov3_${MODE}
data:
  data_dir: ${DATA_DIR}/data
  num_slices: ${NUM_SLICES}
  slice_size: 256
  batch_size: ${BATCH_SIZE}
  num_workers: ${NUM_WORKERS}
  encode_chunk_size: ${ENCODE_CHUNK_SIZE}
model:
  encoder_name: dinov3_vitb16
  freeze_encoder: ${FREEZE_FLAG}
  probe_num_heads: 12
  probe_depth: ${PROBE_DEPTH}
  head_type: ${HEAD_TYPE}
training:
  lr_probe: ${LR_PROBE}
  lr_encoder: ${LR_ENCODER}
  lr_head: ${LR_HEAD}
  weight_decay: ${WEIGHT_DECAY}
  dropout: 0.1
  epochs: ${EPOCHS}
  patience: ${PATIENCE}
  warmup_epochs: 3
  seed: 42
logging:
  output_dir: ${OUTPUT_DIR}
YAMLEOF

echo "=== Config ==="
cat "${CONFIG_PATH}"

echo "=== Running DINOv3 evaluation ==="
EVAL_EXIT=0
cd "$(dirname "$0")/../.."

if [ "$MODE" = "unfrozen" ]; then
    echo "  Launching with torchrun (${NPROC} GPUs, encoder unfrozen)"
    torchrun --nproc_per_node=${NPROC} ablation/dinov3_probe/eval_dinov3.py --config "${CONFIG_PATH}" 2>&1 | tee "${OUTPUT_DIR}/eval_stdout.log" || EVAL_EXIT=$?
else
    echo "  Launching single process (encoder frozen, multi-GPU encoding)"
    python ablation/dinov3_probe/eval_dinov3.py --config "${CONFIG_PATH}" 2>&1 | tee "${OUTPUT_DIR}/eval_stdout.log" || EVAL_EXIT=$?
fi

echo "=== Eval exit code: ${EVAL_EXIT} ==="

if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo "=== Results ==="
    cat "${OUTPUT_DIR}/results.json"
fi

echo "=== Uploading results ==="
python -c "
import glob, os, traceback
output_dir = '${OUTPUT_DIR}'
blob_prefix = '${BLOB_PREFIX}'
files = [f for f in glob.glob(os.path.join(output_dir, '*')) if os.path.isfile(f)]
print('Found %d files to upload' % len(files))
if not files:
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

echo "=== Done (exit=${EVAL_EXIT}) ==="
exit ${EVAL_EXIT}
