# SO-101 + LeRobot + Say-Can 統合に向けたネットワーク RealSense 実験レポート

**著者 (Authors):** プロジェクト記録
**日付 (Date):** 2026-04-13

**キーワード (Keywords):** SO-101, LeRobot, RealSense, Say-Can, ネットワークカメラ
---

## 要旨 (Abstract)

Raspberry Pi 4 に接続した Intel RealSense の映像を同一 LAN 上の PC で動作する LeRobot に取り込み、
Say-Can による言語指示から SO-101 を操作するまでの経路について、配信基盤・遅延計測・統合・評価を段階的に整理する。
計画の進め方は「計測（ベンチマーク）→安定化→統合」とし、リスクを抑えつつ検証する。


## はじめに (Introduction)

本レポートは実装計画書に基づき、カメラ配信基盤（RPi4 側）、性能判定、LeRobot 統合、Say-Can 層、最終評価、
必要リソースおよびリスク管理を論文形式の章立てで記録するためのサンプル入力である。


## システム構成・材料 (System setup and materials)

### ハードウェア (Hardware)
- Raspberry Pi 4（4GB 以上推奨）
- Intel RealSense（D435 等）
- SO-101
- 推論用 PC（GPU 搭載）

### ソフトウェア (Software)
- librealsense（network device 有効ビルド）
- LeRobot
- OpenAI または Anthropic API（LLM / VLM）

### ネットワーク (Network)
有線 LAN（Cat6 以上）を推奨。MTU 調整などパケットロス対策を検討する。
PC 側では `export RS_NETWORK_DEVICE_IP=<RPi4のIP>` によりネットワークデバイスとして RealSense を扱う。


## 方法・実験プロトコル (Methods and protocol)

**ステップ1 — カメラ配信基盤（RPi4）**
- librealsense を `cmake ../ -DBUILD_NETWORK_DEVICE=ON -DCMAKE_BUILD_TYPE=Release` でビルド。
- `./rs-server` を起動して低遅延配信を行う。

**ステップ2 — 性能判定と遅延計測**
- レイテンシ: RPi4 側のストップウォッチと PC 側 `realsense-viewer` の表示時刻差を測定（目標 50ms 以下、100ms 超過時は解像度見直し）。
- FPS: `rs-enumerate-devices` で指定 FPS（例 15/30）の安定性をログ解析。
- 解像度比較: 424x240（低負荷）と 640x480（高精度）のバランスをポリシー学習観点で評価。

**ステップ3 — LeRobot 統合**
- `robot_config.yaml` のカメラをネットワークデバイスのシリアルに合わせる。
- `lerobot-record` 等で欠落なく画像が保存されるか確認。
- 単純スキル実行時に視覚フィードバック遅延によるハンチングがないか確認。


### 実験設計の詳細 (Experimental design)
統合前にネットワーク構成が制御に耐えるかを判定するフェーズ。

1. レイテンシ計測（目標 ≤50ms）
2. FPS 安定性テスト
3. 解像度・負荷のトレードオフ評価

## 実験結果 (Results)

### 計測目標（プレースホルダー — 実測値に差し替え）

| 指標 (Metric) | 値 (Value) | 単位 (Unit) | 備考 (Notes) |
| --- | --- | --- | --- |
| エンドツーエンド遅延 | <50 | ms | 目標。100ms 超なら解像度低下を検討 |
| 目標 FPS（例） | 15 または 30 | fps | 一定時間維持できるかログで確認 |
| 解像度候補 | 424x240 / 640x480 | px | 学習精度と帯域のバランス |



## 統合評価 (Integration and evaluation)

**Say-Can 層**
- スキルライブラリ: LeRobot で学習したタスクを関数化（例 `pick_apple`, `place_in_box`）。
- アフォーダンス: VLM でスキル実行可能性を 0.0–1.0 でスコアリング。
- プランナ: 指示とスコアから LLM がステップを選択。

**最終評価**
- 言語指示に対するスキル選択の成功率とタスク完遂率。
- 動作中に物体を動かした際のプラン再構築（リカバリ）の評価。


## 考察・リスク (Discussion and risks)

- RPi4 の熱: `rs-server` 負荷が高いためアクティブクーリングを推奨。
- 帯域: 同一 LAN に高負荷トラフィックがある場合は専用ルータや QoS の検討。


## 結論 (Conclusion)

ネットワーク RealSense 経路はレイテンシと FPS の事前計測により安定化し、その後 LeRobot 収集・推論および Say-Can 統合へ進む段階的アプローチが妥当である。本 YAML は実測値を追記しながらレポートを更新する。


## 参考文献 (References)
- LeRobot: https://github.com/huggingface/lerobot
- Intel RealSense SDK: https://github.com/IntelRealSense/librealsense
