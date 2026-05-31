#!/bin/bash
#SBATCH --job-name=server-lecture   # ジョブ名
#SBATCH --nodes=1                   # 使用ノード数
#SBATCH --gres=gpu:1                # GPU数
#SBATCH --time=1-00:00:00           # 使用時間

uv run src/train.py -m