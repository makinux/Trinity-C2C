#!/usr/bin/env sh
# Run all model-free self-tests at once (run from the repo root: sh scripts/selftest.sh)
set -e
for m in train eval datasets c2c c2c_validate c2c_train_fuser c2c_hetero c2c_rope c2c_fuser_hetero; do
  echo "===== trinity.$m --selftest ====="
  python -m trinity."$m" --selftest
done
echo "===== ALL SELFTESTS PASSED ====="
