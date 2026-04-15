#!/bin/bash
# LeRobot Raspberry Pi 4 再構築用スクリプト

# 1. システムライブラリのインストール（lzmaエラー防止）
sudo apt update
sudo apt install -y liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev tk-dev libatomic1 libopenblas-dev

# 2. Python 3.10.13 の再ビルド（pyenvを使っている場合）
# pyenv install 3.10.13  # 必要に応じて

# 3. 依存関係のインストール（順番とバージョンが命）
pip install "numpy<2"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install "opencv-python<4.10"
pip install draccus datasets huggingface-hub safetensors click tqdm termcolor pillow pynput pyserial pyyaml scipy

# 4. LeRobot 本体のインストール
cd ~/robot/lerobot
pip install -e . --no-deps

# 5. 環境変数の設定（.bashrcに追記）
grep -q "OPENBLAS_CORETYPE" ~/.bashrc || echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc
grep -q "libatomic.so.1" ~/.bashrc || echo 'export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libatomic.so.1' >> ~/.bashrc

echo "Setup Complete! Please restart your terminal or run 'source ~/.bashrc'"
