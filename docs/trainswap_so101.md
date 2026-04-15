# SO-101 プラレール trainswap（SayCan + CLIP + ArUco）

`lerobot` 内の **最小統合**（**WSL / PC で計画・知覚**、**Raspberry Pi で SO-101 制御**）の入口です。

## 詳細ドキュメント（こちらを主に更新します）

| 内容 | リンク |
|------|--------|
| 実験 Runbook（Phase 0〜4、チェックリスト、`scp`） | [docs/experiments/runbook_so101_trainswap.md](experiments/runbook_so101_trainswap.md) |
| WSL: RealSense・USB・SayCan・CLIP・LM Studio | [docs/experiments/wsl_perception.md](experiments/wsl_perception.md) |
| Pi: SO-101・primitive サーバ・姿勢 JSON | [docs/experiments/pi_robot.md](experiments/pi_robot.md) |
| 実験フォルダ（`env` テンプレ・`artifacts`） | [experiments/so101_trainswap/README.md](../experiments/so101_trainswap/README.md) |
| 実験ドキュメント索引 | [docs/experiments/README.md](experiments/README.md) |

---

以下は **クイックリファレンス**（詳細・切り分けは Runbook 側に集約）。

## 前提（短縮）

- **Pi**: `lerobot-so101-primitive-server`、SO-101 follower（USB）。
- **WSL / PC**: RealSense +（任意）USB 俯瞰、`lerobot-saycan-trainswap`。
- **共有ファイル**: `so101_poses.json`（Pi にもコピー）。推奨キー: `home`, `pre_pick`, `pick`, `lift`, `pre_place`, `place`, `post_place`、（任意）`power_pre`, `power_press`。

## 1) 姿勢の保存（Pi）

```bash
lerobot-so101-save-pose \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --pose_name=home \
  --out_path=experiments/so101_trainswap/artifacts/so101_poses.json
```

## 2) ArUco キャリブ（WSL / PC、RealSense 直結推奨）

```bash
lerobot-so101-calibrate-tag \
  --serial_number_or_name=YOUR_REALSENSE_SERIAL \
  --tag_id=0 \
  --marker_length_m=0.04 \
  --out_path=experiments/so101_trainswap/artifacts/so101_tag_calibration.json \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --reference_pose_name=home
```

## 3) Pi: primitive サーバ

```bash
lerobot-so101-primitive-server \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --host=0.0.0.0 \
  --port=8765
```

## 4) WSL / PC: SayCan + CLIP（LM Studio）

```bash
lerobot-saycan-trainswap \
  --pi_base_url=http://PI_IP:8765 \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --calibration_path=experiments/so101_trainswap/artifacts/so101_tag_calibration.json \
  --realsense_serial=YOUR_REALSENSE_SERIAL \
  --llm.base_url=http://HOST:1234/v1 \
  --llm.model=local-model \
  --clip.device=cuda
```

## Phase 4（USB 俯瞰レール推定）

`--use_track_vision=true --usb_track_camera_index=0 --track_shoulder_pan_deg_per_px=...`（WSL 上の OpenCV カメラ index）。

## 依存関係

- `lerobot[saycan]`、`lerobot[feetech]`、`pyrealsense2`、OpenCV の ArUco（`cv2.aruco`）。
