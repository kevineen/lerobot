# Flow Reversal Steering (FRS) — SO-101 実装ガイド

論文: *"Improving Robotic Generalist Policies via Flow Reversal Steering"* (Tang et al., arXiv:2606.13675)

## 手法の概要

フロー・マッチング型VLA（π0 / π0.5 / SmolVLA）のノイズ除去（デノイズ）を**逆向きに**実行します。

- 通常のデノイズ: ノイズ `a0 ~ N(0,I)` から Euler 積分でアクション `a1` へ（t: 1→0）
- **フロー逆転 (FRS)**: 参照アクション `a1` から逆Euler積分でノイズ `â0` を復元（t: 0→1）

復元した `â0` を再度デノイズすると、参照に近い・かつ政策の行動分布内（in-distribution）な
アクションチャンクが得られます。モデルの重みを一切変えず、デノイズと同じ計算量で、
人間/VLMの粗い指示を政策に「接地（グラウンディング）」できます。

```
粗い参照 a1 (キーボード方向 → IK → 関節ターゲット)
   │  フロー逆転 (逆Euler積分, 10ステップ, h=0.1)
   â0 = μθ⁻¹(a1, o)   … 積分誤差により分布内ノイズになる
   │  通常のデノイズ
   â1 = μθ(â0, o)     … 実行するアクションチャンク
```

### 3つの使い方

| モード | 内容 | 必要なもの |
|---|---|---|
| **ゼロショット・ステアリング** | 毎計画ステップでキーボード方向 → 参照チャンク → FRS | 訓練済み政策のみ |
| **DSBC（オフライン）** | 既存デモから復元したノイズで小さなノイズ政策 πφ(a0\|o) をBC学習し、推論時は πφ のサンプルをデノイズ | 訓練済み政策 + デモデータセット |
| **defer（委譲）** | ステアリング入力がなければ通常の政策実行 | — |

## 実装構成

| 要素 | 場所 |
|---|---|
| `reverse_integrate`（逆Euler積分） | `src/lerobot/policies/common/flow_matching.py` |
| FRSヘルパー（参照チャンク構築・ノイズ展開・アクション正規化） | `src/lerobot/policies/common/frs.py` |
| `sample_actions(reference_actions=...)` / `reverse_actions(...)` | pi0 / pi05 / smolvla の各 `modeling_*.py` |
| DSBCノイズ政策（MLP） | `src/lerobot/experiments/frs/noise_policy.py` |
| 政策チェックポイントの読み込み・橋渡し | `src/lerobot/experiments/frs/policy_bridge.py` |
| キーボード状態（W/A/S/D/Q/E） | `src/lerobot/experiments/frs/keyboard_listener.py` |
| 方向 → IK → 参照チャンク | `src/lerobot/experiments/frs/reference_builder.py` |
| SO-101 URDF（IK用、メッシュ除去済み） | `experiments/frs/assets/so101.urdf` |

### Python API

```python
from lerobot.policies.common.frs import build_reference_chunk, normalize_action_chunk

# 正規化済み参照アクション (B, action_dim) から政策フォーマットのチャンクへ
ref = build_reference_chunk(normalized_ref, chunk_size=50, max_action_dim=7, mode="tile")

# ステアリング付き推論（kwargsは predict_action_chunk からそのまま透過）
chunk = policy.predict_action_chunk(batch, reference_actions=ref, num_reverse_steps=10)

# アクション→ノイズ（DSBC学習用）
noise = policy.model.reverse_actions(images, img_masks, lang_tokens, lang_masks, state, ref)
```

## 前提

1. **訓練済みフロー・マッチング政策**（pi0 / pi05 / smolvla）。
   未訓練の場合は先に `lerobot-train` で SO-101 データセットから訓練してください
   （AGENT_GUIDE.md §4/§6 参照。pi0 は `uv sync --locked --extra pi`、
   smolvla は `--extra smolvla`）。
2. **依存のインストール**:

```bash
uv pip install -e ".[kinematics,smolvla]"   # placo(IK) + num2words(smolvla推論)
uv pip install -e . --no-deps               # FRSスクリプトのエントリポイント登録
```

## 使い方

### 1. ゼロショット・ステアリング（SO-101 実機）

```bash
lerobot-frs-steering \
  --policy_path outputs/train/pi0_so101/pretrained_model \
  --task "pick up the object" \
  --mode steer \
  --robot_port /dev/ttyACM1 \
  --camera_serial 740112072058 \
  --kinematics.urdf_path experiments/frs/assets/so101.urdf
```

キー操作（このターミナルにフォーカスしてください）:

| キー | 動作 |
|---|---|
| W / S | EE +x / −x 方向（URDFのベース座標系） |
| A / D | EE −y / +y 方向 |
| Q / E | EE −z / +z 方向 |
| G | グリッパー開閉トグル |
| Space | この計画ステップは政策に委譲（defer） |
| Esc / x | 終了 |

押下ごとに `step_m`（既定1cm）の並進ステップをIKで関節ターゲットに変換し、
`action_repeats`（既定5）回繰り返した参照チャンクをFRSで射影します。
`execute_steps`（既定5）アクション実行ごとに再計画します（論文の
`action_repeats=5` 相当）。安全のため参照の関節変化は
`kinematics.max_joint_delta_deg`（既定10°）にクランプされます。

### 2. DSBC（ノイズ政策の学習 → 実行）

```bash
# 学習（デモデータセットの各フレームのアクションチャンクを逆流させ、
# チャンク軸平均のノイズを state→noise のMLPで模倣）
lerobot-frs-dsbc-train \
  --policy_path outputs/train/pi0_so101/pretrained_model \
  --dataset_repo_id <user>/so101_task \
  --num_reverse_steps 10 \
  --output_dir outputs/frs_dsbc
# → outputs/frs_dsbc/<user>_so101_task/noise_policy.pt

# 実行（モード dsbc）
lerobot-frs-steering \
  --policy_path outputs/train/pi0_so101/pretrained_model \
  --mode dsbc \
  --noise_policy_path outputs/frs_dsbc/<user>_so101_task/noise_policy.pt \
  --noise_sigma 1.0 \
  --robot_port /dev/ttyACM1 --camera_serial 740112072058
```

論文と同様、学習は軽量です（RTX 4090で数分・<1GB VRAM）。
`noise_sigma`（論文 Appendix C.3: σ=1〜2でむしろ性能向上）で探索ノイズを調整できます。

### 3. LIBERO等のシミュ評価に組み込む場合

`predict_action_chunk(batch, reference_actions=..., num_reverse_steps=...)` を
既存の評価ループから呼ぶだけです。粗い参照（スクリプト済み方向ベクトルのIK結果など）を
正規化 (`normalize_action_chunk`) → チャンク化 (`build_reference_chunk`) して渡してください。

## 主要パラメータ

| パラメータ | 既定 | 説明 |
|---|---|---|
| `num_reverse_steps` | 10 | 逆積分ステップ数（論文: 10, h=0.1）。多いほど復元精度↑・ノイズがOOD化 |
| `execute_steps` | 5 | 1チャンクのうち実行するアクション数（再計画間隔） |
| `action_repeats` | 5 | 参照チャンク長（IKターゲットの繰り返し数） |
| `kinematics.step_m` | 0.01 | キー1回のEE並進量 [m] |
| `kinematics.max_joint_delta_deg` | 10 | 参照関節変化の安全クランプ [deg] |
| `noise_sigma` | 1.0 | DSBC推論時の探索ノイズσ |
| `chunk_noise_mode` | tile (steering) / mean (dsbc) | 参照/ノイズのチャンク軸展開（論文 Appendix C.4） |

## テスト

```bash
# CPU: 逆積分の数学 + FRSヘルパー
pytest tests/policies/common/ -q

# GPU: pi0 / pi05 / smolvla でステアリング効果を検証
# (pi0/pi05 は gated な PaliGemma のため HF トークン + ライセンス承諾が必要。
#  最初の1実行で PaliGemma 設定がキャッシュされ、以降は HF_HUB_OFFLINE=1 でも実行可)
hf auth login

# 【重要】必ず1テストずつ実行すること。各テストはフルVLM（fp32で約16GiB VRAM）
# をGPUに載せるため、失敗テストのVRAMが解放されず、一括実行はOOM連鎖する
# （RTX 4090/3090の24GBで実証: 一括だと6失敗、1本ずつなら全部合格）
for t in $(pytest --collect-only -q tests/policies/pi0_pi05/test_frs.py | sed -n 's/^.*:://p'); do
  pytest "tests/policies/pi0_pi05/test_frs.py::$t" -v
done
```

## 注意事項

- **URDFとキャリブレーション**: URDFのキャリブレーションと実機のずれ
  （huggingface/lerobot#2413）がある場合、`so101_old_calib.urdf` 系を使うか
  再キャリブレーションしてください。IKの関節名・EEフレーム名は起動時にログ出力
  されるので確認してください。
- FRSはフロー・マッチング政策専用です（pi0 / pi05 / smolvla）。
  ACT等の決定論的政策や pi0_fast のトークン化政策には適用できません。
- **pi0.5 の正規化**: pi0.5 は既定で QUANTILES 正規化（q01/q99 統計必須）。
  データセット統計に q01/q99 がないと ValueError になるため、
  `augment_dataset_quantile_stats.py` で統計を追加するか
  `normalization_mapping` を変更してください。
- **メモリが厳しい場合**: `dtype="bfloat16"` で VRAM 約半減
  （RTX 3090実測: fp32 15.6GiB → bf16 8.8GiB。ステアリング効果・
  逆積分の数値は fp32 と同等、アクション出力は fp32 で返る）。
- 実行前に `lerobot-teleoperate` でアーム・カメラ・キャリブレーションが正常か確認してください。
