#!/usr/bin/env sh
# モデル不要のセルフテストを一括実行（リポジトリ root から実行: sh scripts/selftest.sh）
set -e
for m in train eval datasets c2c c2c_validate c2c_train_fuser c2c_hetero c2c_rope c2c_fuser_hetero; do
  echo "===== trinity.$m --selftest ====="
  python -m trinity."$m" --selftest
done
echo "===== ALL SELFTESTS PASSED ====="
