#!/bin/bash
# AML entry point for the single-job interpretability run.
# Downloads data + 3 fine-tune checkpoints, runs scripts/interpretability.py,
# uploads outputs to blob.
#
# Sequential phases baked into the Python script:
#   1. encode test set (per model) → per-slice feature cache + predictions
#   2. slice-level occlusion attribution (architecture-agnostic)
#   3. patch-level occlusion heatmaps on 20 selected volumes per model
#   4. cross-model plots

set -euo pipefail

NUM_SLICES=${NUM_SLICES:-64}
SLICE_SIZE=${SLICE_SIZE:-256}
CHUNK_SIZE=${CHUNK_SIZE:-16}
N_TP=${N_TP:-10}
N_TN=${N_TN:-10}
TOP_K_SLICES=${TOP_K_SLICES:-3}

# Blob paths — confirmed via `az storage blob list` + results.json probe_type
MEANPOOL_BLOB_PREFIX=${MEANPOOL_BLOB_PREFIX:-"ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260419_060137"}
CROSSATTN_BLOB_PREFIX=${CROSSATTN_BLOB_PREFIX:-"ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260418_192249"}
D1_BLOB_PREFIX=${D1_BLOB_PREFIX:-"ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260418_035940"}

BLOB_ACCOUNT=${BLOB_ACCOUNT:?'Set BLOB_ACCOUNT env var'}
BLOB_CONTAINER=${BLOB_CONTAINER:?'Set BLOB_CONTAINER env var'}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="interpretability_${TIMESTAMP}"
OUTPUT_DIR="/tmp/ijepa_outputs/${RUN_TAG}"
BLOB_PREFIX="ijepa-interpretability/${RUN_TAG}"
DATA_DIR="/tmp/fairvision_data"
MODEL_DIR="/tmp/interp_models"
mkdir -p "$OUTPUT_DIR" "$MODEL_DIR/meanpool" "$MODEL_DIR/crossattn" "$MODEL_DIR/d1"

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version" || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install scikit-learn pillow pyyaml scipy matplotlib azure-storage-blob azure-identity 2>&1 | tail -5

echo "=== Downloading test set (only Test/) ==="
python - <<PY
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
import os
account = '${BLOB_ACCOUNT}'
container = '${BLOB_CONTAINER}'
prefix = 'fhl-test-data/Test'
local_dir = '${DATA_DIR}/data/Test'
os.makedirs(local_dir, exist_ok=True)

cred = ManagedIdentityCredential()
cc = ContainerClient(
    account_url='https://%s.blob.core.windows.net' % account,
    container_name=container, credential=cred,
)
blobs = list(cc.list_blobs(name_starts_with=prefix))
print('Found %d Test blobs' % len(blobs))
downloaded = 0
for b in blobs:
    if not b.name.endswith('.npz'):
        continue
    rel = b.name[len('fhl-test-data/Test'):].lstrip('/')
    dst = os.path.join(local_dir, rel)
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(cc.download_blob(b).readall())
    downloaded += 1
    if downloaded % 500 == 0:
        print('  downloaded %d' % downloaded)
print('Test set ready: %d files total, %d newly downloaded' %
      (len([x for x in os.listdir(local_dir) if x.endswith('.npz')]), downloaded))
PY

echo "=== Downloading 3 fine-tune best_model.pt files ==="
download_ckpt() {
    local prefix=$1
    local dest=$2
    if [ -f "$dest" ]; then
        echo "  already have $dest"
        return
    fi
    python - <<PY
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os
cred = ManagedIdentityCredential()
blob = BlobClient(
    account_url='https://${BLOB_ACCOUNT}.blob.core.windows.net',
    container_name='${BLOB_CONTAINER}',
    blob_name='$prefix/best_model.pt',
    credential=cred,
)
os.makedirs(os.path.dirname('$dest'), exist_ok=True)
with open('$dest', 'wb') as f:
    f.write(blob.download_blob().readall())
print('  downloaded %s (%d MB)' % ('$dest', os.path.getsize('$dest') // (1024*1024)))
PY
}
download_ckpt "$MEANPOOL_BLOB_PREFIX"  "$MODEL_DIR/meanpool/best_model.pt"
download_ckpt "$CROSSATTN_BLOB_PREFIX" "$MODEL_DIR/crossattn/best_model.pt"
download_ckpt "$D1_BLOB_PREFIX"        "$MODEL_DIR/d1/best_model.pt"

echo "=== Running interpretability pipeline ==="
python scripts/interpretability.py \
    --data-dir "${DATA_DIR}/data" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-slices $NUM_SLICES \
    --slice-size $SLICE_SIZE \
    --chunk-size $CHUNK_SIZE \
    --n-tp $N_TP \
    --n-tn $N_TN \
    --top-k-slices $TOP_K_SLICES \
    --skip-existing \
    2>&1 | tee "$OUTPUT_DIR/interp_stdout.log"
EXIT=${PIPESTATUS[0]}
echo "=== Python exit code: $EXIT ==="

echo "=== Output files ==="
ls -lah "$OUTPUT_DIR" || true
for d in "$OUTPUT_DIR"/heatmaps_*; do
    [ -d "$d" ] && echo "  $(basename $d): $(ls "$d" | wc -l) PNGs"
done

echo "=== Uploading outputs to blob ==="
python - <<PY
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os, glob
cred = ManagedIdentityCredential()
account = '${BLOB_ACCOUNT}'
container = '${BLOB_CONTAINER}'
prefix = '${BLOB_PREFIX}'
root = '${OUTPUT_DIR}'

uploaded = 0
total_bytes = 0
for path in glob.glob(os.path.join(root, '**/*'), recursive=True):
    if not os.path.isfile(path):
        continue
    rel = os.path.relpath(path, root).replace('\\\\', '/')
    dst = '%s/%s' % (prefix, rel)
    with open(path, 'rb') as f:
        data = f.read()
    blob = BlobClient(
        account_url='https://%s.blob.core.windows.net' % account,
        container_name=container, blob_name=dst, credential=cred,
    )
    blob.upload_blob(data, overwrite=True)
    total_bytes += len(data)
    uploaded += 1
print('Uploaded %d files (%d MB total) to %s' %
      (uploaded, total_bytes // (1024*1024), prefix))
PY

echo "=== All done (exit=$EXIT) ==="
exit $EXIT
