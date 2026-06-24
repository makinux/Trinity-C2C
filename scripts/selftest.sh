#!/usr/bin/env sh
# Run all model-free self-tests at once (run from the repo root: sh scripts/selftest.sh)
set -e
for m in train eval datasets c2c c2c_validate c2c_train_fuser c2c_hetero c2c_rope c2c_fuser_hetero; do
  echo "===== trinity.$m --selftest ====="
  python -m trinity."$m" --selftest
done

echo "===== trinity.gateway.selftest ====="
if python -c "import fastapi" 2>/dev/null; then
  python -m trinity.gateway.selftest
else
  echo "[skip] fastapi not installed (pip install -r requirements.txt to enable the gateway self-test)"
fi

echo "===== ALL SELFTESTS PASSED ====="
