# 実験ドキュメント（SO-101 trainswap ほか）

このディレクトリは、**二台分担**（手元 PC の WSL で知覚・計画、Raspberry Pi で SO-101 制御）を前提にした手順と Runbook を置きます。

## SO-101 プラレール trainswap（SayCan + CLIP + ArUco）

| ドキュメント | 用途 |
|--------------|------|
| [runbook_so101_trainswap.md](runbook_so101_trainswap.md) | **最初に読む**: Phase 0〜4 のチェックリスト、コマンドの時系列、成果物のパス、`scp` 例 |
| [wsl_perception.md](wsl_perception.md) | WSL での RealSense / USB カメラ、LM Studio、CLIP、`lerobot-saycan-trainswap` |
| [pi_robot.md](pi_robot.md) | Pi での SO-101 接続、姿勢保存、`lerobot-so101-primitive-server` |

リポジトリ直下の短いまとめ（ブックマーク用）は [../trainswap_so101.md](../trainswap_so101.md) を参照してください。

## 実験用フォルダ（設定・成果物の置き場）

コマンド例で使う相対パスの基準は [experiments/so101_trainswap/README.md](../../experiments/so101_trainswap/README.md) です。キャリブ JSON やログは `artifacts/` に集約し、Git には含めません。
