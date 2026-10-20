# FRS テストを RTX 3090 機で再現する手順書

作成日: 2026-10-20
対象: `lerobot` の Flow Reversal Steering (FRS) 統合テスト
`tests/policies/pi0_pi05/test_frs.py`（全8テスト）

## 目的

RTX 4090 機（WSL2）では最終実行時点で **8テスト中6つが失敗**していた。
同じテストを RTX 3090 機で実行し、結果を比較して失敗がハード/環境依存か
コード本来のバグかを切り分ける。

4090機での最終結果（再現の基準）:

| テスト | 結果 |
|---|---|
| test_pi0_frs_steering_pulls_actions_toward_reference | ❌ 失敗 |
| test_pi0_frs_tracks_its_own_reference | ❌ 失敗 |
| test_pi0_frs_reverse_actions_matches_reference_kwarg_path | ❌ 失敗 |
| test_pi0_frs_validation_errors | ❌ 失敗 |
| test_pi05_frs_steering_pulls_actions_toward_reference | ❌ 失敗 |
| test_pi05_frs_reverse_actions_matches_reference_kwarg_path | ❌ 失敗 |
| test_smolvla_frs_steering_pulls_actions_toward_reference | ✅ 合格 |
| test_smolvla_frs_reverse_actions_round_trip | ✅ 合格 |

判定の目安:

- **同じ6つが失敗** → コード側（FRS実装 or テスト）の問題。GPU非依存の可能性大
- **全テスト合格** → 4090機の環境（ドライバ/WSL/CUDA）が原因の可能性
- **CUDA OOM で落ちる** → 後述のメモリ注意を参照

## 前提条件

- RTX 3090（VRAM 24GB — 4090と同量なので十分）
- Linux または WSL2。NVIDIA ドライバは CUDA 12.8 が動くバージョン（R570系以降推奨）
- [uv](https://docs.astral.sh/uv/)（無ければ `curl -LsSf https://astral.sh/uv/install.sh | sh`）
- Hugging Face アカウント + アクセストークン（**PaliGemma は gated model のため、
  事前に https://huggingface.co/google/paligemma-3b-mix-224 で利用承認を受けておく**）
  - トークン作成: https://huggingface.co/settings/tokens （read 権限で可）
- ディスク空き: 数GB（PaliGemma 設定ファイルのダウンロード含む）

## 手順

### 1. リポジトリを取得

FRS実装はコミット済み（`5f967cf3`）。main をクローンすれば良い:

```bash
git clone https://github.com/kevineen/lerobot.git
cd lerobot
git log --oneline -3   # 5f967cf3 Add Flow Reversal Steering (FRS) ... が含まれることを確認
```

### 2. テストの差分パッチを適用

4090機にはコミット前の修正（pi0/pi05 の `n_action_steps=CHUNK_SIZE` 追加）が
残っているため、同様にする。このフォルダの `frs_test.patch` を使う:

```bash
git apply --check /path/to/frs_test.patch   # 先に当たるか確認
git apply /path/to/frs_test.patch
```

※ TurboVLA 関連の未コミット変更（pyproject/wandb 等）は FRS テストに不要なため
パッチには含めていない。

### 3. 依存関係のセットアップ

Python 3.12 以上。torch は pyproject 内の uv インデックス設定
（pytorch-cu128）から自動で入る:

```bash
# pytest は test extra、transformers/accelerate は smolvla extra 経由で入る。
# --all-extras は非推奨: 新しいCMake(4.x)環境で libero extra の egl-probe
# ビルドが失敗する（FRSテストに不要）。
# Python は 3.12 を明示（新しいディストロのデフォルト 3.14 だと
# pyrealsense2 等 cp313 までの wheel が無く解決失敗する）
uv sync --python 3.12 --extra test --extra smolvla
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期待値: 2.11.0+cu128 True
```

### 4. HF トークンを設定

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

（常設するなら ~/.bashrc に追記。`hf auth login` でも可）

### 5. 実行

GPUが空いていることを確認してから:

```bash
nvidia-smi
uv run pytest tests/policies/pi0_pi05/test_frs.py -v 2>&1 | tee frs_3090.log
```

初回は PaliGemma の設定ダウンロードが走る。各テストはフルVLM
（pi0/pi05 は約3Bパラメータ）を乱数初期化するため、**1テスト数分程度、全体で
20〜40分ほどかかる**。遅くても途中で切らないこと。

## メモリに関する注意（重要）

4090機ではこのテストの実行中に WSL2 がフリーズした。原因は:

- WSL設定がメモリ18GB上限・swap 0GB だった
- 各テストは GPU 転送前に CPU 上（fp32）でポリシーを構築するため、
  ピーク時 **1テストあたり約13GB** のRAMを消費する
- swap がないとメモリ枯渇時にプロセス停止（OOM kill）ではなくシステム全体が固まる

3090機の物理RAMが32GB以上なら問題ない。**WSL2 で実行する場合は**
`C:\Users\<user>\.wslconfig` で以下を推奨（`wsl --shutdown` で反映）:

```ini
[wsl2]
memory=24GB
swap=16GB
```

ネイティブLinuxなら、仮にメモリが足りなくてもテストプロセスが kill されるだけで
システムは固まらない。

## トラブルシュート

| 症状 | 対処 |
|---|---|
| `OSError: Not found / 401 / 403`（HFダウンロード時） | HF_TOKEN 未設定、または PaliGemma の利用承認が未完了 |
| `torch.cuda.is_available()` が False | ドライバ更新、WSLなら `nvidia-smi` が WSL 内で動くか確認 |
| `ImportError: TurboVLA...` | 出ないはず（パッチは test のみ）。出たらクローン元の main に余分な変更がないか確認 |
| CUDA out of memory | 他プロセスがVRAM使用中でないか `nvidia-smi` で確認。24GBあれば通常不要 |
| 動作が極端に遅い/固まる | 上記メモリ注意を参照。RAM不足の兆候 |

## 結果の報告方法

`frs_3090.log`（手順5で保存）を 4090機に持ち帰り、上の基準表と見比べる。
`uv run pytest ... -v` の出力には各テストの fail理由（assert文と期待値/実測値）が
出るので、失敗時はその部分を添えてもらえると次の調査が早い。
