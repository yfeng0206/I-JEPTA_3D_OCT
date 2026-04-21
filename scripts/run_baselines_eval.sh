#!/bin/bash
# AML entry point for the foundation-model baselines evaluation.
# Downloads data + weights + clones OCTCube repo, installs deps,
# runs scripts/baselines_eval.py for 2 encoders sequentially
# (DINOv3 + OCTCube — both public, no HF auth required).

set -euo pipefail

NUM_SLICES=${NUM_SLICES:-64}
SLICE_SIZE=${SLICE_SIZE:-256}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-15}
LR=${LR:-0.0004}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
BATCH_SIZE=${BATCH_SIZE:-128}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
ENCODERS=${ENCODERS:-"dinov3 octcube"}

BLOB_ACCOUNT=${BLOB_ACCOUNT:?'Set BLOB_ACCOUNT env var'}
BLOB_CONTAINER=${BLOB_CONTAINER:?'Set BLOB_CONTAINER env var'}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="baselines_eval_${TIMESTAMP}"
OUTPUT_DIR="/tmp/ijepa_outputs/${RUN_TAG}"
BLOB_PREFIX="ijepa-baselines/${RUN_TAG}"
DATA_DIR="/tmp/fairvision_data"
OCTCUBE_REPO_DIR="/tmp/OCTCubeM"
OCTCUBE_WEIGHTS="/tmp/octcube_weights/OCTCube.pth"
mkdir -p "$OUTPUT_DIR"

echo "=== Disk space ==="
df -h /tmp 2>/dev/null || df -h

echo "=== Environment Info ==="
python --version
pip show torch 2>/dev/null | grep -E "^Name|^Version" || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo "=== Installing dependencies ==="
# Core: our pipeline + transformers for HF AutoModel.
# OCTCube transitive: simplejson (util/loggings), iopath (util/misc), timm
# (util/video_vit imports timm.models.layers + timm.models.vision_transformer).
# All three are listed in OCTCube's requirement.txt and may or may not be
# in the base AzureML PyTorch image.
pip install scikit-learn pillow pyyaml scipy matplotlib openpyxl \
            transformers accelerate einops \
            simplejson iopath timm \
            azure-storage-blob azure-identity 2>&1 | tail -5
# flash-attn is optional; OCTCube adapter falls back to models_vit_st
# (non-FA variant) if flash_attn import fails.
pip install flash-attn --no-build-isolation 2>&1 | tail -3 || echo "[flash-attn install skipped]"

echo "=== Downloading FairVision Train + Val + Test ==="
python - <<'PY'
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
import os
account = os.environ['BLOB_ACCOUNT']
container = os.environ['BLOB_CONTAINER']
prefix = 'fhl-test-data'
local_dir = '/tmp/fairvision_data/data'
os.makedirs(local_dir, exist_ok=True)
cred = ManagedIdentityCredential()
cc = ContainerClient(
    account_url='https://%s.blob.core.windows.net' % account,
    container_name=container, credential=cred,
)
# Always list blobs and download whatever is missing. The per-blob
# `if os.path.exists(dst): continue` makes this idempotent so we don't
# re-download anything we already have. This also recovers from partial
# downloads (e.g. Training done but Validation interrupted).
blobs = list(cc.list_blobs(name_starts_with=prefix))
print('Found %d blobs under %s/' % (len(blobs), prefix))
dl = 0
for b in blobs:
    if not b.name.endswith('.npz'):
        continue
    rel = b.name[len(prefix):].lstrip('/')
    dst = os.path.join(local_dir, rel)
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        f.write(cc.download_blob(b).readall())
    dl += 1
    if dl % 1000 == 0: print('  downloaded %d' % dl)
print('Total newly downloaded: %d' % dl)

# Sanity: fail fast if any split is empty AFTER the download loop.
for split in ['Training', 'Validation', 'Test']:
    p = os.path.join(local_dir, split)
    n = len([x for x in os.listdir(p) if x.endswith('.npz')]) if os.path.exists(p) else 0
    print('  %s: %d volumes' % (split, n))
    if n == 0:
        raise RuntimeError('split %s is empty at %s - download failed' % (split, p))
PY

# ---- Clone OCTCube repo if needed ----
if echo "$ENCODERS" | grep -q "octcube"; then
    echo "=== Cloning OCTCube repo ==="
    if [ ! -d "$OCTCUBE_REPO_DIR/.git" ]; then
        git clone --depth 1 https://github.com/ZucksLiu/OCTCubeM.git "$OCTCUBE_REPO_DIR" 2>&1 | tail -3
    else
        echo "  already cloned"
    fi
    export OCTCUBE_REPO_DIR
    # Download OCTCube pretrained weights
    if [ ! -f "$OCTCUBE_WEIGHTS" ]; then
        mkdir -p "$(dirname $OCTCUBE_WEIGHTS)"
        echo "=== Downloading OCTCube pretrained weights from HuggingFace ==="
        python - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
# Try the HF hub first; fall back paths if key names differ
for fn in ['OCTCube.pth', 'OCTCube.pt', 'oct_cube.pth']:
    try:
        path = hf_hub_download(repo_id='zucksliu/OCTCubeM', filename=fn)
        shutil.copy(path, '$OCTCUBE_WEIGHTS')
        print('  got', fn, '->', '$OCTCUBE_WEIGHTS')
        break
    except Exception as e:
        print('  tried', fn, '->', type(e).__name__)
else:
    print('[WARN] Could not locate OCTCube weights on HF; OCTCube encoder will fail to load')
PY
    fi
fi

echo "=== Running baselines_eval.py ==="
set +e
python scripts/baselines_eval.py \
    --data-dir "${DATA_DIR}/data" \
    --output-dir "$OUTPUT_DIR" \
    --encoders $ENCODERS \
    --num-slices $NUM_SLICES \
    --slice-size $SLICE_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --batch-size $BATCH_SIZE \
    --warmup-epochs $WARMUP_EPOCHS \
    --octcube-weights "$OCTCUBE_WEIGHTS" \
    2>&1 | tee "$OUTPUT_DIR/baselines_stdout.log"
EXIT=${PIPESTATUS[0]}
set -e
echo "=== Python exit code: $EXIT ==="

echo "=== Output files ==="
find "$OUTPUT_DIR" -type f | head -50

echo "=== Uploading outputs to blob ==="
python - <<PY
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
import os, glob
cred = ManagedIdentityCredential()
account = os.environ['BLOB_ACCOUNT']
container = os.environ['BLOB_CONTAINER']
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
