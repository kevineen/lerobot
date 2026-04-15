# find port
lerobot-find-port

# follower robot
/dev/ttyACM1

# leader teleop
/dev/ttyACM0

# moter setup follower
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1

# moter setup leader
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0

# calibration follower
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower

# calibration leader
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader

# 操作
lerobot-teleoperate \
--robot.type=so101_follower \
--robot.port=/dev/ttyACM0 \
--robot.id=my_follower \
--teleop.type=so101_leader \
--teleop.port=/dev/ttyACM1 \
--teleop.id=my_leader

# camera デバイス取得
lerobot-find-cameras realsense

# Camera付き
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --robot.cameras='{ front: {type: intelrealsense, serial_number_or_name: "740112072058", width: 1920, height: 1080, fps: 30} }' \
  --display_data=true

# Rerun Viewer
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_leader \
  --robot.cameras='{ front: {type: intelrealsense, serial_number_or_name: "740112072058", width: 1920, height: 1080, fps: 30} }' \
  --display_data=true \
  --display_ip=192.168.2.12 \
  --display_port=9876

# 接続
rerun --bind 0.0.0.0 --port 9876
rerun --connect rerun+http://127.0.0.1:9876/proxy

---

## Say-Can + CLIP（Phase A: まずはスコアリング実験）

### セットアップ（Python依存）
- LeRobot 側で追加した extra を入れる場合は `lerobot[saycan]` をインストール。
  - 含まれるもの: `transformers`（CLIPModel/Processor）, `httpx`（vLLM/LM Studio接続）, `Pillow`（画像ロード）

### 1) RealSenseのフレーム + 固定候補を CLIP でスコア

```bash
lerobot-saycan-clip \
  --image_source.mode=front_cam \
  --image_source.front_cam.serial_number_or_name="740112072058" \
  --image_source.front_cam.width=1920 \
  --image_source.front_cam.height=1080 \
  --image_source.front_cam.fps=30 \
  --saycan.instruction="pick up the object" \
  --saycan.top_k=5 \
  --clip.pretrained_name="openai/clip-vit-base-patch32" \
  --clip.device=cpu \
  --num_steps=1
```

### 2) LLM（vLLM / LM Studio の OpenAI互換API）で候補を生成して CLIP でスコア

```bash
lerobot-saycan-clip \
  --image_source.mode=front_cam \
  --image_source.front_cam.serial_number_or_name="740112072058" \
  --saycan.llm_generate_candidates=true \
  --saycan.llm_num_candidates=10 \
  --llm.base_url="http://192.168.2.12:1234/v1" \
  --llm.model="local-model" \
  --clip.device=cpu \
  --num_steps=1
```

### 3) 保存フレーム（オフライン）で実験

```bash
lerobot-saycan-clip \
  --image_source.mode=path_glob \
  --image_source.paths='["outputs/captured_images/*.png"]' \
  --saycan.top_k=5 \
  --clip.device=cpu \
  --num_steps=3
```

### 4) Rerunへランキングも送る（Viewerは別PCで待ち受け）
- Windows/PC 側: `rerun --bind 0.0.0.0 --port 9876`

```bash
lerobot-saycan-clip \
  --image_source.mode=front_cam \
  --image_source.front_cam.serial_number_or_name="740112072058" \
  --rerun=true \
  --rerun_ip=192.168.2.12 \
  --rerun_port=9876 \
  --num_steps=1
```

## SO-101 trainswap（SayCan + CLIP + ArUco / Pi primitive サーバ）

詳細は [docs/trainswap_so101.md](docs/trainswap_so101.md)。

- 姿勢保存: `lerobot-so101-save-pose`
- タグ基準キャリブ: `lerobot-so101-calibrate-tag`
- Pi 側 primitive HTTP: `lerobot-so101-primitive-server`
- PC 側ループ: `lerobot-saycan-trainswap`

# cameraでデータ収集
python3 lerobot/scripts/record.py \
  --robot-path lerobot/configs/robot/so100.yaml \
  --robot-overrides \
    '{"follower": {"port": "/dev/ttyACM0"}, "leader": {"port": "/dev/ttyACM1"}}' \
  --camera-overrides \
    '{"front": {"_target_": "lerobot.cameras.realsense.camera_realsense.RealSenseCamera", "config": {"serial_number_or_name": "740112072058"}}}' \
  --fps 30 \
  --repo-id <YOUR_NAME>/so101_test_dataset \
  --num-episodes 10 \
  --push-to-hub 0


USBケーブルをなくして無線化することは**技術的に可能**ですが、いくつかのアプローチがあり、それぞれにメリットとデメリットがあります。

LeRobot（特にRaspberry Pi 4）を使用している場合、以下の3つのパターンが考えられます。

---

### 1. 「シリアル無線化アダプタ」を使う（一番シンプル）
USBケーブルの代わりに、BluetoothやWi-Fiでシリアル信号を飛ばすモジュールを間に挟む方法です。

*   **構成:**
    *   [ラズパイ] --- (Bluetooth/Wi-Fi) --- [ESP32などの無線マイコン] --- [モーター]
*   **方法:**
    *   **ESP32** などを使い、「無線で受け取ったデータをそのままシリアル（UART）でモーターに流す」という「透過プロキシ」を作ります。
    *   PC/ラズパイ側からは「仮想シリアルポート」として認識させます。
*   **注意点:**
    *   **遅延（レイテンシ）:** 無線化するとデータの往復に数ミリ〜数十ミリ秒の遅延が発生します。LeRobotのようなリアルタイム性が重要なシステムでは、動きがカクついたり、制御が不安定（発振）になるリスクがあります。

---

### 2. リーダーとフォロワーを別のラズパイにする（ネットワーク経由）
「リーダー側のラズパイ」と「フォロワー側のラズパイ」をWi-Fiで通信させる方法です。

*   **構成:**
    *   [リーダーの腕] --- [ラズパイA]  〜〜 (Wi-Fi / Socket通信) 〜〜 [ラズパイB] --- [フォロワーの腕]
*   **方法:**
    *   現在 LeRobot がサポートしている標準機能は「1台のPCに両方の腕が繋がっていること」を前提としています。
    *   これを無線化するには、片方で読み取った角度データをネットワーク（UDPやZMQなど）で飛ばし、もう片方で受け取って動かすという**「通信プログラム」の自作**が必要になります。

---

### 3. ロボット側にラズパイとバッテリーを載せる（推奨）
「通信を無線にする」のではなく、**「ロボットを自律（コードレス）にする」**方法です。これが最も一般的で、LeRobotの本来の使い道に近いです。

*   **構成:**
    *   ロボットにラズパイ4とモバイルバッテリー（またはLiPo電池）を搭載します。
    *   **操作（テレオペ）時:** 手元のPCから **Wi-Fi（SSH）** でラズパイにログインして `lerobot-teleoperate` を実行します。
    *   **推論（自動実行）時:** ラズパイ単体で動き回ります。
*   **メリット:**
    *   制御の遅延がゼロ（USB直結なので）。
    *   PCとロボットの間にケーブルがないので、ロボットが自由に動けます。

---

### 無線化する際の最大の壁：遅延と通信断
ロボット（特にサーボモーター）の制御において、無線化には以下のリスクが伴います：

1.  **制御周期（FPS）の低下:**
    LeRobotは秒間30〜60回以上のやり取りを期待しますが、Wi-Fiの混雑などでデータが遅れると、ロボットの動きがギクシャクします。
2.  **安全性の問題:**
    通信が切れた瞬間にロボットが「最後の命令（全速力など）」を維持してしまうと危険です。無線化する場合は「通信が切れたら止まる」というコードを仕込む必要があります。

### 結論としてのおすすめ
まずは **「パターン3（ロボットにラズパイとバッテリーを載せて、Wi-Fi越しにコマンドを送る）」** から始めるのが一番確実でトラブルが少ないです。

もし「リーダー（操作側）の腕だけを完全に独立させて、離れた場所から操作したい」ということであれば、**ESP32を使用したUDP通信**による自作システムを検討することになります。

具体的に「どのようなシーンで無線にしたい（例：リーダーを手に持って歩き回りたい、など）」という希望はありますか？それによって最適な機材が変わります。