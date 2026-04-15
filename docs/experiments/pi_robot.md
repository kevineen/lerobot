# Raspberry Pi: SO-101 制御・primitive サーバ

**役割**: SO-101 follower を USB で駆動し、手元 WSL から送られた **関節プリミティブ**を HTTP で実行する。RealSense や CLIP はここでは不要（`lerobot[feetech]` が中心）。

## リポジトリと依存関係

Pi 上の LeRobot ルートで:

```bash
uv sync --extra feetech
```

SayCan ループ自体は WSL で動かす前提なら、Pi に `saycan` extra は必須ではありません。

## シリアルポート

SO-101 のシリアルデバイスは通常 `/dev/ttyACM0` または `/dev/ttyACM1` です。

```bash
ls -l /dev/ttyACM*
```

固定したい場合は **udev ルール**で `SYMLINK` を張るか、`--robot.port=` に実際のパスを毎回指定します。

## 姿勢ライブラリの保存

テレオペ等でアームを安全な姿勢に動かしたうえで、名前付き姿勢を JSON に追記します。出力は **Git に含めない** `experiments/so101_trainswap/artifacts/` を推奨します。

```bash
cd /path/to/lerobot
uv run lerobot-so101-save-pose \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --pose_name=home \
  --out_path=experiments/so101_trainswap/artifacts/so101_poses.json
```

`pre_pick`, `pick`, `lift`, `pre_place`, `place`, `post_place` など、Runbook の `pose_names` に必要なキーをすべて揃えます。使わないスキルがある場合は、PC 側のスキルマップと整合させるか、同じ関節角を別名で登録するなどして **欠けない**ようにします。

## primitive HTTP サーバ

WSL から Pi の IP で到達できるよう、`0.0.0.0` で待ち受けます。

```bash
uv run lerobot-so101-primitive-server \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --poses_path=experiments/so101_trainswap/artifacts/so101_poses.json \
  --host=0.0.0.0 \
  --port=8765
```

デフォルトのパスは `http://<PiのIP>:8765/primitive`（実装はスクリプトのログに表示）です。`lerobot-saycan-trainswap` の `--pi_base_url` には **`http://IP:8765` の形式**（スクリプト側で primitive パスを結合）で渡します。

## 環境変数テンプレート

[experiments/so101_trainswap/env/pi.env.example](../../experiments/so101_trainswap/env/pi.env.example) を参考に、ポートや `ROBOT_PORT` をローカル用に設定します（任意）。

## WSL へファイルを戻す

キャリブは WSL で行うことが多い一方、`so101_poses.json` は Pi で更新した内容を WSL の `artifacts/` にコピーして差分管理すると便利です。

```bash
# Pi 上から（例）
scp experiments/so101_trainswap/artifacts/so101_poses.json user@windows-wsl-host:/path/to/lerobot/experiments/so101_trainswap/artifacts/
```

（ネットワークの向きは環境に合わせて読み替えてください。詳細は [runbook_so101_trainswap.md](runbook_so101_trainswap.md)。）

## よくある切り分け

| 現象 | 確認 |
|------|------|
| Permission denied on ttyACM | `dialout` グループ、`udev` |
| WSL から接続できない | Pi と PCが同一セグメントか、`host=0.0.0.0`、ポート開放 |
| 動作が不安定 | 電源（USB 給電不足）、ケーブル、`lerobot-calibrate` の再実行 |
