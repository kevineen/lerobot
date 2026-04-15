# SO-101 trainswap 実験サンドボックス

WSL（知覚・SayCan）と Raspberry Pi（primitive サーバ）で **同じ相対パス**を使えるよう、JSON やログをここに集約します。

## ドキュメント

- [docs/experiments/runbook_so101_trainswap.md](../../docs/experiments/runbook_so101_trainswap.md) — **最初に読む** Runbook
- [docs/experiments/wsl_perception.md](../../docs/experiments/wsl_perception.md)
- [docs/experiments/pi_robot.md](../../docs/experiments/pi_robot.md)
- [docs/trainswap_so101.md](../../docs/trainswap_so101.md) — クイックリファレンス

## ディレクトリ

| パス | 用途 |
|------|------|
| `env/wsl.env.example` | WSL 用変数テンプレート → `wsl.env` にコピー（Git 無視） |
| `env/pi.env.example` | Pi 用のメモ用テンプレート |
| `configs/saycan_wsl.example.sh` | `uv run lerobot-saycan-trainswap` の例 |
| `artifacts/` | `so101_poses.json`、`so101_tag_calibration.json`、ログ（**コミットしない**） |

LeRobot ルートから作業する:

```bash
cd /path/to/lerobot
```

以降の CLI 例は `experiments/so101_trainswap/artifacts/` を `--poses_path` / `--out_path` に指定します。
