#!/bin/bash
# Sequential job queue: monitors each job, submits next when done.
# Jobs run one at a time to avoid GPU contention on shared compute.

RG="rg-npesu01"
WS="CrossencoderRankerNPESU01"
CONFIG_DIR="configs"

check_interval=1200  # 20 minutes

wait_for_job() {
  local job_name="$1"
  echo "[$(date)] Waiting for job: $job_name"
  while true; do
    status=$(az ml job show --name "$job_name" --resource-group "$RG" --workspace-name "$WS" --query "status" -o tsv 2>/dev/null)
    echo "[$(date)] Job $job_name status: $status"
    if [ "$status" = "Completed" ]; then
      echo "[$(date)] Job $job_name COMPLETED"
      return 0
    elif [ "$status" = "Failed" ] || [ "$status" = "Canceled" ]; then
      echo "[$(date)] Job $job_name $status — stopping queue"
      return 1
    fi
    sleep $check_interval
  done
}

submit_job() {
  local config="$1"
  echo "[$(date)] Submitting: $config"
  # Use --query to extract just the job name, suppress stderr warnings
  local job_name
  job_name=$(az ml job create --file "$config" --resource-group "$RG" --workspace-name "$WS" --query "name" -o tsv 2>/dev/null)
  echo "[$(date)] Submitted job: $job_name"
  echo "$job_name"
}

echo "=== Queue started (phase 2: frozen ImageNet already submitted) ==="

# Step 1: Wait for frozen ImageNet (already running)
FROZEN_IMAGENET_JOB="dynamic_lobster_yppngqdlg9"
echo "[$(date)] Step 1: Waiting for frozen ImageNet ($FROZEN_IMAGENET_JOB)"
wait_for_job "$FROZEN_IMAGENET_JOB" || exit 1

# Step 2: Submit and wait for unfrozen random
echo "[$(date)] Step 2: Unfrozen random"
job2=$(submit_job "${CONFIG_DIR}/aml_rerun_unfrozen_random_d3_s64.yml")
wait_for_job "$job2" || exit 1

# Step 3: Submit and wait for unfrozen ImageNet
echo "[$(date)] Step 3: Unfrozen ImageNet ep32"
job3=$(submit_job "${CONFIG_DIR}/aml_rerun_unfrozen_imagenet_ep32_d3_s64.yml")
wait_for_job "$job3" || exit 1

# Step 4: Submit and wait for unfrozen random 32 slices
echo "[$(date)] Step 4: Unfrozen random 32 slices"
job4=$(submit_job "${CONFIG_DIR}/aml_rerun_unfrozen_random_d3_s32.yml")
wait_for_job "$job4" || exit 1

# Step 5: Submit and wait for unfrozen ImageNet 32 slices
echo "[$(date)] Step 5: Unfrozen ImageNet ep32 32 slices"
job5=$(submit_job "${CONFIG_DIR}/aml_rerun_unfrozen_imagenet_ep32_d3_s32.yml")
wait_for_job "$job5" || exit 1

echo "=== All 6 rerun jobs completed! ==="
