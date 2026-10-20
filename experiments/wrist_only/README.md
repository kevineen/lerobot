# wrist_only 実験（2026-10-20）

SO-101 片腕 pickup（cobra データセット curated 版）を **wrist カメラのみ**で学習した比較実験。
データセット: `local/so101_singlearm_pickup_cobra_20261018_205119_curated`
（`~/.cache/huggingface/lerobot/local/...`。curate 前の元データから `so101-lab/tools/dataset-curator` で生成）

## 実行方法

```bash
cd ~/robot/lerobot
.venv/bin/lerobot-train --config experiments/wrist_only/configs/act_wrist_only.yaml
```

（MLflow に曲線を残す場合は so101-lab の `scripts/train_mlflow.sh` 経由。docs/17 の規約参照）

## 結果サマリー（RTX 4090 / WSL2）

| ポリシー | 設定 | steps | 最終 loss | 所要時間 | VRAM | 出力 |
|---|---|---|---|---|---|---|
| ACT | `configs/act_wrist_only.yaml` | 8,040 | **0.197**（l1: 0.165, kld: 0.003） | 約23分 | 6.7 GB | `outputs/train/act_so101_cobra_curated_wrist_only_20261020/` |
| Diffusion | `configs/diffusion_wrist_only.yaml` | 12,158（32K smpl 相当） | **0.010** | 約25分 | 6.2 GB | `outputs/train/diffusion_so101_cobra_curated_wrist_only_20261020/` |

- wandb 無効（ローカル実行）
- 生ログは `logs/`（git 管理外）。最終行のメトリクスのみ上表に残した
- チェックポイントは各 `outputs/train/<run>/checkpoints/` にあり
