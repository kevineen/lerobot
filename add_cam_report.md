SO-101とLeRobotプロジェクトに、ネットワーク経由のRealSenseデータを用いたSay-Canアルゴリズムを統合するための実装計画書を提案します。

この計画は、**「計測（ベンチマーク）」→「安定化」→「統合」**の順で、リスクを抑えながら進める構成にしています。

---

# SO-101 + LeRobot + Say-Can 統合実装計画書

## 1. 目的
Raspberry Pi 4（以下RPi4）に接続されたRealSenseの映像を、同一LAN内のPCで実行されるLeRobotに統合し、Say-Canアルゴリズムを用いた言語指示によるSO-101の操作を実現する。

---

## 2. ステップ1：カメラ配信基盤の構築 (RPi4側)
まずは、RealSenseのデータを低遅延で配信するサーバーを構築します。

### 実施内容
1.  **librealsenseのビルド:** * RPi4上でネットワークデバイスサポートを有効にしてビルドする。
    * `cmake ../ -DBUILD_NETWORK_DEVICE=ON -DCMAKE_BUILD_TYPE=Release`
2.  **rs-serverの起動:**
    * ビルドしたバイナリから `./rs-server` を実行。
3.  **ネットワーク最適化:**
    * 有線LAN（Cat6以上）の使用を推奨。
    * MTU値の調整など、パケットロス対策の検討。

---

## 3. ステップ2：性能判定と遅延計測 (実験フェーズ)
システム全体を統合する前に、現在のネットワーク構成がロボット制御に耐えうるか判定します。

### 実験項目
* **レイテンシ計測:** * RPi4側でストップウォッチを映し、PC側の `realsense-viewer` で表示される時刻との差分を測定。
    * **目標値:** 50ms以下。100msを超える場合は解像度を下げる。
* **FPS安定性テスト:**
    * `rs-enumerate-devices` を使用し、指定したFPS（例：15fpsや30fps）が一定時間維持されるかログを解析。
* **解像度・圧縮率の決定:**
    * `424x240` (低負荷) vs `640x480` (高精度) で、LeRobotのポリシー学習に最適なバランスを特定する。



---

## 4. ステップ3：LeRobotへの統合 (データ収集・推論)
既存のLeRobot環境がネットワーク経由のカメラを「ローカルカメラ」として認識するように設定します。

### 実装タスク
1.  **環境変数の設定:**
    * PC側で `export RS_NETWORK_DEVICE_IP=<RPi4のIP>` を設定。
2.  **LeRobot Configの修正:**
    * `robot_config.yaml` 内のカメラデバイス指定を、ネットワークデバイスのシリアル番号に更新。
3.  **データ収集テスト:**
    * `python lerobot/scripts/record.py` を実行し、ネットワーク越しに画像が欠落なく保存されるか確認。
4.  **ポリシー実行テスト:**
    * 学習済みの単純なスキル（例：手を上げる）を動かし、視覚フィードバックの遅れによる振動（ハンチング）が発生しないか確認。

---

## 5. ステップ4：Say-Canレイヤーの実装
プランニング（LLM）と動作（LeRobot）を接続します。

### 構成要素
1.  **Skill Library（スキルライブラリ）:**
    * LeRobotで学習させた各タスク（例：`pick_apple`, `place_in_box`）を関数として定義。
2.  **Affordance Functions（適格性評価）:**
    * VLM（GPT-4o-miniやCLIPなど）を用い、現在の画像から「どのスキルが実行可能か」を0.0〜1.0でスコアリングする。
3.  **Planning Engine:**
    * LLMに対して、現在の指示とAffordanceスコアを入力し、最適なステップを選択させる。

### 実装コード構造（イメージ）
```python
while not task_completed:
    img = get_network_frame()  # ステップ3で作成
    affordance_scores = eval_vlm(img, skills) # 「何ができるか」判定
    chosen_skill = llm_planner(user_prompt, affordance_scores) # 行動決定
    execute_lerobot_policy(chosen_skill) # SO-101を動かす
```

---

## 6. ステップ5：最終評価とチューニング
1.  **成功率の測定:** * 「赤いボールを箱に入れて」などの指示に対し、Say-Canが正しいスキルを選択し、SO-101が完遂できるか。
2.  **リカバリ実験:**
    * 動作中に物体を動かした場合、Say-Canがプランを再構築して追従できるか（クローズドループの評価）。

---

## 7. 必要なリソース・環境
* **ハードウェア:** Raspberry Pi 4 (4GB以上推奨), Intel RealSense (D435等), SO-101, 推論用PC (GPU搭載)。
* **ソフトウェア:** `librealsense` (network-enabled), `lerobot` library, `openai` or `anthropic` API (for LLM/VLM)。



---

## 8. リスク管理
* **RPi4の熱暴走:** `rs-server` は負荷が高いため、アクティブクーリングを必須とする。
* **帯域制限:** 同一LAN内に他の高負荷通信がある場合、専用のルーターを用意するか、優先制御（QoS）を検討する。