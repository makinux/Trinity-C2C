# Trinity-C2C

**ローカルLLMの協調（Trinity式コーディネーション）＋ Cache-to-Cache（KV潜在通信）の研究用実装骨子。**
すべてローカル/オープンモデルで完結し、クローズドAPIを使いません。

> *A fully-local research skeleton combining Trinity-style multi-LLM coordination with
> Cache-to-Cache (KV-level latent communication). No closed APIs — open models only.*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)

---

## ⚠️ 位置づけ（必ずお読みください）

- 公開論文の**着想を独自に実装した研究/教育用の骨子**です。論文著者や各機関とは**無関係**の非公式実装です。
- 各モジュールは**モデル不要のセルフテストで機構を検証済み**（`scripts/selftest.sh`）。一方で、
  汎化する学習済みモデルを同梱するものではなく、**汎化はデータ規模のWIP**です（[結果](#検証済みの結果honest)参照）。
- 利用するモデル（Qwen/GLM/DeepSeek/SmolLM 等）は**各自のライセンス**に従ってください。

## 概要

「複数モデルの重みマージはアーキ不一致やクローズドAPIで破綻する」——そこで本実装は、
**重みを触らずテスト時に協調させる**Trinityの着想と、**テキストではなくKVキャッシュで潜在通信する**
Cache-to-Cacheの着想を、ローカル前提で組み合わせた骨子を提供します。

- **star型コーディネーション**：小型の Coordinator が各ターンで役割（Thinker / Worker / Verifier）を割当
- **役割→モデルを `config.yml` で設定**（GLM / Qwen-Coder / DeepSeek 等を差し替え可能）
- **Cache-to-Cache**：同一/異種モデル間で KV を潜在融合（トークナイザ整合・GQA・**RoPE-aware** 対応）

設計の詳細は [docs/design.md](docs/design.md) を参照。

## アーキテクチャ

```
                 ┌─────────────────────────┐
   Query ───────►│ Coordinator (router)    │  小型SLM + 線形ヘッド / sep-CMA-ES で学習
                 │  次の役割を選択          │
                 └───┬─────────┬─────────┬──┘
                     ▼         ▼         ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Thinker  │►│ Worker ★ │◄│ Verifier │   ★=中心の統合点(Receiver)
              │ 計画/批評 │ │ 実装/生成 │ │ 検証     │   役割→モデルは config.yml
              └──────────┘ └────┬─────┘ └──────────┘
   通信:  テキスト（P0）  /  KV潜在融合（C2C: Thinker→Worker を潜在化）
```

## リポジトリ構成

```
.
├── config.yml                 # 役割→モデル・各種設定（メインの設定ファイル）
├── trinity/                   # Python パッケージ
│   ├── config.py              # config.yml ローダ（デフォルト内蔵・env上書き）
│   ├── p0.py                  # オーケストレーション（3役割＋ループ制御）
│   ├── coordinator.py         # 特徴量(SLM隠れ状態)＋線形ヘッド＋学習コーディネータ
│   ├── train.py               # sep-CMA-ES によるコーディネータ学習
│   ├── eval.py                # 実行スコアラー＋方策比較（bench）
│   ├── datasets.py            # HumanEval+/MBPP+ ローダ（差分テスト）
│   ├── c2c.py                 # C2C 融合コア（KVCache・層整合・ゲート）
│   ├── c2c_validate.py        # 同一ファミリ最小検証
│   ├── c2c_realrun.py         # 実機C2C転移（小型SLM）
│   ├── c2c_train_fuser.py     # cache fuser 学習（凍結LM越しの勾配）
│   ├── c2c_hetero.py          # 異種対応（トークナイザ整合・GQA射影）
│   ├── c2c_rope.py            # RoPE-aware 整列
│   ├── c2c_fuser_hetero.py    # 異種＋RoPE＋学習可能 fuser（統合）
│   ├── c2c_hetero_realrun.py  # 異種2モデル実機（SmolLM↔Qwen）
│   └── c2c_train_general.py   # データ規模・汎化学習
├── scripts/selftest.sh        # 全セルフテスト一括（モデル不要）
├── docs/                      # 設計書・論文要約・Docker補足
├── Dockerfile / docker-compose.yml
├── requirements.txt
├── LICENSE                    # Apache License 2.0
└── README.md
```

## 必要要件 / インストール

- **Python 3.12 推奨**（torch/transformers の安定ホイール）
- セルフテストは `numpy` のみ。実機C2C/学習は `torch`/`transformers` が必要。

```bash
# CPU構成（セルフテスト＋小型SLMでのC2C実機まで）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# 任意: HumanEval+/MBPP+ ローダ
pip install -r requirements-datasets.txt
```
> 社内SSL検査環境では pip に `--trusted-host pypi.org --trusted-host files.pythonhosted.org` を、
> HuggingFace ダウンロードには `truststore`（同梱）を利用してください。

## クイックスタート

```bash
# 1) 全セルフテスト（モデル不要・機構の健全性確認）
sh scripts/selftest.sh        # 期待: 最終行に "===== ALL SELFTESTS PASSED ====="
# 個別: python -m trinity.eval --selftest など

# 2) 役割→モデルや各種設定を編集
$EDITOR config.yml

# 3) C2C の潜在転移を実機確認（小型SLM・CPUでも数十秒）
python -m trinity.c2c_realrun          # gate↑で送信文脈の情報が受信側生成へ転移

# 4) 異種2モデルのC2C（SmolLM2-135M → Qwen2.5-0.5B）
python -m trinity.c2c_hetero_realrun

# 5) データ規模での汎化学習（held-out評価つき）
python -m trinity.c2c_train_general
```

### Docker（再現環境）

```bash
docker compose build
docker compose run --rm app                       # = scripts/selftest.sh（全緑が期待値）
docker compose run --rm app python -m trinity.c2c_realrun
# GPUで大型モデル(vLLM 3サーバ)を使う評価:
docker compose --profile gpu up -d --wait
docker compose run --rm app python -m trinity.eval --trials 3
```
詳細は [docs/docker.md](docs/docker.md)。

## 設定（config.yml）

役割ごとのモデルと主要設定を一箇所で管理します。

| セクション | 主な項目 |
|---|---|
| `models.<thinker/worker/verifier>` | `base_url` / `model_id` / `api_key` / `temperature` / `max_tokens` |
| `prompts` | 役割別システムプロンプト |
| `orchestration` | `max_turns` / `verbose` |
| `coordinator` | `slm_model` / `featurizer` |
| `training` | sep-CMA-ES の `budget` / `sigma0` / `m_reps` / `seed` |
| `eval` | `trials` / `timeout` / `max_inputs` |
| `c2c` | `sharer_model` / `receiver_model` / `init_gate` / `tau` |

- 上書き優先: `config.yml` < 環境変数（`TRINITY_CONFIG`=パス、`THINKER_URL`等=base_url、`WORKER_API_KEY`等=秘密鍵）
- 設定ファイルや pyyaml が無い場合は内蔵デフォルトで動作します。

## 検証済みの結果（honest）

すべて手元のCPU実機で実走・検証した範囲です（過大主張を避けています）。

- **C2C 潜在転移（同一ファミリ）**：France文脈KVをJapan文脈に融合し gate↑で
  `Δlogp(" Paris") = +10.87`（注入経路は gate=0 で通常forwardと完全一致＝非侵襲を確認）。
- **学習可能性**：凍結LM越しに fuser へ勾配が貫通し loss が低下。
- **異種2モデル実機**：SmolLM2-135M(層30/KVヘッド3/rope1e5) → Qwen2.5-0.5B(層24/2/1e6) を結線。
  トークナイザ整合・GQA・**RoPE-aware**（自前RoPEがQwen rotaryと数値一致）まで機構を検証。
- **汎化（WIP）**：24カ国＋contrastiveで held-out のlearnedが改善し、正しいshare内容を使用
  （learned ≫ shuffled）。ただし `learned < gate0` で**閾値は僅差未達**＝データ/計算のスケールが次段階。

## 参考論文（着想元）

- Trinity: An Evolved LLM Coordinator — arXiv:2512.04695
- Cache-to-Cache: Direct Semantic Communication Between LLMs — arXiv:2510.03215
- Activated LoRA / Multi-Adapter KV-Cache Reuse — arXiv:2504.12397, 2512.17910

## ライセンス

[Apache License 2.0](LICENSE) — Copyright 2026 NSS Co., Ltd.
帰属・非公式である旨・着想元の論文・参照モデルの扱いは [NOTICE](NOTICE) を参照。
利用する各モデル・データセットのライセンスは別途各提供元に従ってください。
