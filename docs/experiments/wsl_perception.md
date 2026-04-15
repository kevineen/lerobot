# WSL（手元 PC）: 知覚・SayCan・CLIP

**役割**: Intel RealSense と（任意の）USB カメラで画像・深度・ArUco を取得し、`lerobot-so101-calibrate-tag` と `lerobot-saycan-trainswap` を実行する。LLM（LM Studio など OpenAI 互換 API）と CLIP はここで動かすのが一般的です。

## リポジトリと依存関係

LeRobot ルート（`pyproject.toml` があるディレクトリ）で:

```bash
uv sync --extra saycan
```

GPU がある場合は PyTorch の CUDA ビルドを環境に合わせて入れてください。CPU のみなら `--clip.device=cpu` で実行できます。

## RealSense を WSL2 で使う

WSL2 から USB バスに直接アクセスできないことが多いです。**Windows 側で RealSense を WSL にバインドする**手順（`usbipd-win` など）に従ってください。

- Intel RealSense の公式ドキュメント: [Librealsense installation on Windows 10 / 11 with WSL2](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_windows.md)（WSL2 の節を参照）
- デバイスが見えたら、`rs-enumerate-devices` や LeRobot スクリプトの `--serial_number_or_name` / `--realsense_serial` に **シリアル番号**を指定します。

同一 LAN 上の別マシンに RealSense を接続して **network device** として使う構成もあります。その場合は `RS_NETWORK_DEVICE_IP` 等、librealsense のネットワークデバイス手順に従ってください（詳細は Runbook のトラブルシュート参照）。

## USB 俯瞰カメラ（Phase 4）

`lerobot-saycan-trainswap` の `--use_track_vision=true --usb_track_camera_index=N` は **OpenCV のデバイス index** です。WSL で `/dev/video0` が見えているか、`v4l2-ctl --list-devices` 等で確認し、`N` を合わせます。

## 環境変数テンプレート

[experiments/so101_trainswap/env/wsl.env.example](../../experiments/so101_trainswap/env/wsl.env.example) を `experiments/so101_trainswap/env/wsl.env` にコピーして値を埋め、シェルで `source` してから [configs/saycan_wsl.example.sh](../../experiments/so101_trainswap/configs/saycan_wsl.example.sh) を実行するか、同等の `uv run lerobot-saycan-trainswap ...` を手で実行します。

**注意**: `wsl.env` は `.gitignore` 対象です。秘密（API キー等）はコミートしないでください。

## 主なコマンド

### ArUco 基準のキャリブレーション

RealSense が WSL で認識できている状態で、タグが安定して映る位置から:

```bash
cd /path/to/lerobot
uv run lerobot-so101-calibrate-tag \
  --serial_number_or_name="${REALSENSE_SERIAL}" \
  --tag_id=0 \
  --marker_length_m=0.04 \
  --out_path=experiments/so101_trainswap/artifacts/so101_tag_calibration.json \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --reference_pose_name=home
```

生成された `so101_tag_calibration.json` の `joint_gains_per_tag_tvec_m` を編集し、タグのカメラ座標変化に対する関節補正（**1 m あたりの関節単位**）を調整します（元手順は [../trainswap_so101.md](../trainswap_so101.md)）。

### SayCan + CLIP ループ

Pi 上で `lerobot-so101-primitive-server` が起動していること。

```bash
uv run lerobot-saycan-trainswap \
  --pi_base_url="${PI_BASE_URL}" \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --calibration_path=experiments/so101_trainswap/artifacts/so101_tag_calibration.json \
  --realsense_serial="${REALSENSE_SERIAL}" \
  --llm.base_url="${LM_BASE_URL}" \
  --llm.model="${LM_MODEL}" \
  --clip.device="${CLIP_DEVICE:-cpu}"
```

`LLMBackendConfig` のデフォルトは `http://127.0.0.1:1234/v1`（LM Studio のローカル例）です。LM Studio を **Windows ホスト**で動かし WSL から叩く場合は、WSL から到達可能なアドレス（例: Windows ホスト IP）に `LM_BASE_URL` を設定します。

## LM Studio

- サーバを起動し、OpenAI 互換 API を有効にする。
- `--llm.base_url` は **`.../v1` で終わる** URL に合わせる。
- `--llm.model` は LM Studio 側で読み込んだモデル名に合わせる。

## よくある切り分け

| 現象 | 確認 |
|------|------|
| RealSense が開けない | usbipd / デバイス独占 / USB3 ポート |
| Pi に HTTP が届かない | Pi の IP、`primitive-server` の `--host=0.0.0.0`、Pi および Windows ファイアウォール |
| CLIP が遅い | `--clip.device=cuda`、バッチサイズ（設定は `CLIPConfig`） |
