#!/bin/bash
# Submit slice-level I-JEPA job to AML without git metadata.
# Temporarily hides .git during submission.
cd "$(dirname "$0")"
mv .git .git_hidden 2>/dev/null
az ml job create --file configs/aml_slice_ep100.yml --workspace-name YOUR_WORKSPACE --resource-group YOUR_RESOURCE_GROUP
mv .git_hidden .git 2>/dev/null
