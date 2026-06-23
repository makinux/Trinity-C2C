# ローカル版 Trinity-C2C アーキテクチャ設計書

- 作成日: 2026-06-23
- 前提: ローカル/オープンモデルのみ（Qwen, GLM, DeepSeek など）。クローズドなクラウドAIサービスは使用しない。
- 通信層: Cache-to-Cache (C2C) によるモデル間KV潜在融合を中核に据える。
- 検討経緯: Claude と Codex (gpt-5.5) の共同レビューにより、当初の chain 型から **star 型** に変更。

## 参照論文
- Trinity: An Evolved LLM Coordinator — arXiv:2512.04695
- Cache-to-Cache (C2C): Direct Semantic Communication Between LLMs — arXiv:2510.03215
- Activated LoRA (aLoRA) — arXiv:2504.12397
- Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA — arXiv:2512.17910

---

## 0. 設計の狙い
- 一貫性（解釈ドリフトの抑制） + 低コスト + 誤りの非相関（多様性）を、ローカルのみで両立する。
- モデル間通信を「損失の多いテキスト」から「より豊かな潜在(KV)」へ置き換える（=C2C）。
- ただしC2Cの弱点（ペアワイズ・一方向・履歴分岐に弱い・両モデルprefillが必要）を設計で回避する。

---

## 1. トポロジー（star 型）

```
                         ┌───────────────────────────────┐
        Query Q ───────► │  Coordinator (router)         │
                         │  Qwen3-0.6B + head            │
                         │  学習: sep-CMA-ES（終端報酬）  │
                         └──────┬─────────┬─────────┬─────┘
              （制御: 点線。全役割をルーティング）       ┊
                  ┊            ┊                       ┊
                  ▼            ▼                       ▼
        ┌───────────────┐   ┌───────────────────┐   ┌────────────────────┐
        │ Thinker       │   │ Worker = Receiver ★│   │ Verifier           │
        │ GLM           │──►│ Qwen3-Coder        │◄──│ DeepSeek-R1-distill │
        │ 計画/分解/批評 │   │ 唯一の中心統合点    │   │ 検証 ACCEPT/REVISE  │
        └───────────────┘   └─────────┬──────────┘   └────────────────────┘
           plan(潜在/C2C) ─┘          │   └─ critique(潜在/C2C, REVISE)
                                       │      artifact(テキスト) ─►Verifier
                                       ▼
                               ┌──────────────┐
                               │ Final artifact│  ← Verifier が ACCEPT で確定
                               └──────────────┘
```

凡例:
- 実線(C2C latent bus): ソフトな意図・計画・批評・不確実性を潜在(KV)で運ぶ
- 破線(exact text/artifact): コード・証明・成果物など「正確さが必要な現物」をテキストで運ぶ
- 点線(coordinator control): Coordinator が各役割の起動順・ループ・停止を制御

---

## 2. 構成要素とモデル割当（誤りの非相関を最大化）

| 構成要素 | モデル | 役割 / 理由 |
|---|---|---|
| Coordinator（ルーター） | Qwen3-0.6B + 約1万paramヘッド | 流れの制御のみ。sep-CMA-ESで終端報酬から学習。Trinityの「激安コーディネート」を継承 |
| Thinker（Sharer） | GLM | 計画・分解・批評。Qwen系と異なる帰納バイアス |
| Worker = Receiver ★（中心） | Qwen3-Coder | 唯一の統合点。成果物を生成。最強のローカルコード生成＝安定した融合先 |
| Verifier（Sharer） | DeepSeek-R1-distill | 検証＋ACCEPT/REVISE。WorkerともThinkerとも別系統＝最大の誤り非相関 |

注: GLMをThinkerとVerifierの両方に使わないこと（非相関が崩れる）。

---

## 3. なぜ star か（chain を捨てた理由）
C2Cはペアワイズ・一方向・履歴分岐に弱い。Thinker→Worker→Verifier→Worker と潜在状態を鎖状に渡すと
分布シフトが累積する（"latent telephone" = 伝言ゲーム化）。
→ 中心のReceiver（Qwen3-Coder）を唯一の正準統合点にし、Sharerはそこへ放射状に潜在を注入。

学習が必要な fuser は2本だけ:
- GLM(Thinker) → Qwen3-Coder(Receiver)   … 計画の潜在
- DeepSeek(Verifier) → Qwen3-Coder(Receiver) … 批評の潜在

Worker→Verifier 方向は fuser 不要。成果物は「正確なテキスト」で渡せば足りる。

---

## 4. 二重チャネル（latent + exact text）
- 潜在チャネル(C2C): ソフトな意図・計画事前分布・批評・不確実性 → Receiverへ融合
- 正確チャネル(テキスト): コード・証明・成果物（ロスのある融合に載せてはいけないもの）

原則: 「意図は潜在で、現物はテキストで」運ぶ。純粋な潜在ハンドオフに固執しない。

---

## 5. ターンの流れ（最大5ターン）
1. Q → Coordinator が符号化 → Thinker を起動
2. Thinker(GLM) が Q を prefill し計画を生成（KV = Sharer）
3. Coordinator が統合を指示 → C2C(Thinker→Receiver) が計画潜在を Qwen3-Coder に注入 → Receiver が成果物（正確なコード）を生成
4. Coordinator が検証を指示 → Verifier(DeepSeek) が Q＋正確な成果物テキストを読み、ACCEPT/REVISE＋批評を出力
5. REVISE なら（§6の対策に従い）Receiver を新規 prefill で再構成 → ループ
6. ACCEPT で終了 → 成果物を返す

---

## 6. REVISE逆辺の罠と対策（最重要）
Verifier→Worker で同じWorkerの履歴を継続使用するとC2Cが壊れる
（生成済みWorkerに、別軌道のVerifier由来潜在を注入＝KV不整合）。

対策: REVISEを「継続」ではなく「新規統合パス」として扱う。
- Receiver を正準テキスト状態から再 prefill: `プロンプト + 正確な成果物 + Verifierの正確な批評テキスト`
- ＋任意で Verifier→Receiver 潜在を、まっさらな prefill に注入

---

## 7. 学習順序（co-train しない）
1. テキストのみのローカルTrinityを先に構築（C2C無しのベースライン）
2. Thinker→Receiver fuser を学習・評価
3. Verifier→Receiver 批評 fuser を「新規再構成プロトコル」で学習
4. Worker→Verifier は最後（正確テキストで足りる可能性が高い）
5. 役割・fuser を凍結した上で、最後に Coordinator を sep-CMA-ES で学習

---

## 8. インフラ / GPU
- 現実解: 1ノード 2×80GB（A100/H100）。7B–32B級を複数同時・KV常駐・実験込みで回すならこれが妥当。
- 1×80GB: 量子化＋オフロードで可能だが窮屈。
- 1×24GB: プロトタイプ専用（小型・量子化のみ）。
- サービング: KVへの生アクセスと注入フックを持つ vLLM 系（aLoRA対応エンジン 2512.17910 が近い土台）。
- 量子化: AWQ / GPTQ / FP8。C2Cは両モデルのprefillが常駐する前提でKVメモリを確保。

---

## 9. トップ3リスク
1. 履歴分岐によるKV不整合 → REVISEは新規再構成で回避
2. 弱い/ノイズの多い潜在注入 → 役割モデルを同等以上の強さに保つ＋層ゲート
3. モデル/役割変更時のfuserの脆さ → 役割ラインナップを凍結、入替時はfuser再学習

---

## 10. 段階的構築
- P0: テキストのみローカルTrinity（C2C無し）で動作＆精度の土台
- P1: Thinker→Receiver の1本だけC2C化し、テキスト版とA/B比較
- P2: 2本のfuser＋凍結Coordinatorで完成

---

## 設計の要点（まとめ）
C2Cの弱点（履歴分岐・伝言ゲーム）を「star ＋ 新規再構成 ＋ 二重チャネル」で回避し、
誤り非相関は Qwen / GLM / DeepSeek の系統分散で確保する。
すべてローカル・オープンモデルで完結し、クローズドAIは不要。
