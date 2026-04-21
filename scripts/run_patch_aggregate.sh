#!/bin/bash
# AML entry point for the aggregate-patch-attribution run.
# Downloads data + 3 fine-tune checkpoints + cached features from blob,
# runs scripts/patch_aggregate.py, uploads outputs.
#
# Per-model cost: ~3000 volumes × {encode 1 slice + run probe+head on
# (256, 64, 768) variants} = ~20 min on 1 T4. Total ~1-1.5 h for all 3
# models + figure generation.

set -euo pipefail

NUM_SLICES=${NUM_SLICES:-64}
SLICE_SIZE=${SLICE_SIZE:-256}
# Comma-separated list of subset indices, default 20,43 (the two peaks in
# the cross-model contribution curve). Passed to Python as space-separated.
TARGET_SLICES=${TARGET_SLICES:-"20 43"}

# Blob paths (confirmed via results.json probe_type check)
MEANPOOL_BLOB_PREFIX=${MEANPOOL_BLOB_PREFIX:-"ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260419_060137"}
CROSSATTN_BLOB_PREFIX=${CROSSATTN_BLOB_PREFIX:-"ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260418_192249"}
D1_BLOB_PREFIX=${D1_BLOB_PREFIX:-"ijepa-downstream/downstream_patch_s64_ep50_bs1_linear_20260418_035940"}

# Previously-computed per-slice feature caches (from plucky_soccer)
INTERP_BLOB_PREFIX=${INTERP_BLOB_PREFIX:-"ijepa-interpretability/interpretability_20260420_002126"}

BLOB_ACCOUNT=${BLOB_ACCOUNT:?'Set BLOB_ACCOUNT env var'}
BLOB_CONTAINER=${BLOB_CONTAINER:?'Set BLOB_CONTAINER env var'}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="patch_aggregate_${TIMESTAMP}"
OUTPUT_DIR="/tmp/ijepa_outputs/${RUN_TAG}"
BLOB_PREFIX="ijepa-interpretability/${RUN_TAG}"
DATA_DIR="/tmp/fairvision_data"
MODEL_DIR="/tmp/pa_models"
FEATURES_DIR="/tmp/pa_features"
mkdir -p "$OUTPUT_DIR" "$MODEL_DIR/meanpool" "$MODEL_DIR/crossattn" "$MODEL_DIR/d1" "$FEATURES_DIR"

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version" || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
pip install scikit-learn pillow pyyaml scipy matplotlib azure-storage-blob azure-identity 2>&1 | tail -5

echo "=== Downloading Test set ==="
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
dl = 0
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
    dl += 1
    if dl % 500 == 0: print('  downloaded %d' % dl)
print('Test set ready: %d total files, %d new' %
      (len([x for x in os.listdir(local_dir) if x.endswith('.npz')]), dl))
PY

echo "=== Downloading 3 FT best_model.pt + 3 feature caches ==="
download_blob() {
    local blob=$1
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
    blob_name='$blob',
    credential=cred,
)
os.makedirs(os.path.dirname('$dest'), exist_ok=True)
with open('$dest', 'wb') as f:
    f.write(blob.download_blob().readall())
print('  got %s (%d MB)' % ('$dest', os.path.getsize('$dest') // (1024*1024)))
PY
}
download_blob "$MEANPOOL_BLOB_PREFIX/best_model.pt"  "$MODEL_DIR/meanpool/best_model.pt"
download_blob "$CROSSATTN_BLOB_PREFIX/best_model.pt" "$MODEL_DIR/crossattn/best_model.pt"
download_blob "$D1_BLOB_PREFIX/best_model.pt"        "$MODEL_DIR/d1/best_model.pt"
download_blob "$INTERP_BLOB_PREFIX/features_meanpool.npz"  "$FEATURES_DIR/features_meanpool.npz"
download_blob "$INTERP_BLOB_PREFIX/features_crossattn.npz" "$FEATURES_DIR/features_crossattn.npz"
download_blob "$INTERP_BLOB_PREFIX/features_d1.npz"        "$FEATURES_DIR/features_d1.npz"

echo "=== Running patch_aggregate.py ==="
# Disable set -e around the Python call so a Python failure still hits the
# upload block below (otherwise we lose the stdout log which is the main
# thing needed to diagnose). Re-enable after.
set +e
python scripts/patch_aggregate.py \
    --data-dir "${DATA_DIR}/data" \
    --model-dir "$MODEL_DIR" \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-slices $NUM_SLICES \
    --slice-size $SLICE_SIZE \
    --target-slices $TARGET_SLICES \
    2>&1 | tee "$OUTPUT_DIR/patch_aggregate_stdout.log"
EXIT=${PIPESTATUS[0]}
set -e
echo "=== Python exit code: $EXIT ==="

echo "=== Output files ==="
ls -lah "$OUTPUT_DIR" || true

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
uploaded = 0; total = 0
for path in glob.glob(os.path.join(root, '**/*'), recursive=True):
    if not os.path.isfile(path):
        continue
    rel = os.path.relpath(path, root).replace('\\\\', '/')
    dst = '%s/%s' % (prefix, rel)
    with open(path, 'rb') as f:
        data = f.read()
    b = BlobClient(
        account_url='https://%s.blob.core.windows.net' % account,
        container_name=container, blob_name=dst, credential=cred,
    )
    b.upload_blob(data, overwrite=True)
    uploaded += 1; total += len(data)
print('Uploaded %d files (%d MB) to %s' %
      (uploaded, total // (1024*1024), prefix))
PY

echo "=== All done (exit=$EXIT) ==="
exit $EXIT
