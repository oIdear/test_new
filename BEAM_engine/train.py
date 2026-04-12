#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
from BEAM_model import BEAM
from shutil import get_terminal_size
import click
import os
import torch  # 新增：加载checkpoint需要

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.caida_as_rel.fetch_data import get as prepare_edge_file

@click.command()
@click.option("--serial", "-s", type=click.Choice(["1", "2"]), default="1", help="serial 1 or 2")
@click.option("--time", "-t", type=int, required=True, help="timestamp, e.g., 20200901")
@click.option("--Q", "Q", type=int, default=10, help="hyperparameter Q, e.g., 10")
@click.option("--dimension", type=int, default=128, help="hyperparameter dimension size, e.g., 128")
@click.option("--epoches", type=int, default=1000, help="epoches to train, e.g., 1000")
@click.option("--device", type=int, default=0, help="device to train on")
@click.option("--num-workers", type=int, default=10, help="number of workers")
@click.option("--resume", type=str, default="", help="恢复训练的checkpoint路径")  # 新增：resume参数
def main(serial, time, device, resume,** model_params):  # 新增：resume单独传参
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"

    edge_file = prepare_edge_file(serial, time)
    assert edge_file.exists(), f"fail to prepare {edge_file}"

    model_params["edge_file"] = edge_file

    # 打印参数（方便调试）
    for k, v in model_params.items():
        print(f"{k}: {v}")
    print(f"resume: {resume}")
    print("*"*get_terminal_size().columns)

    # 构建训练目录
    train_dir = Path(__file__).resolve().parent/"models"/ \
        f"{edge_file.stem}.{model_params['epoches']}.{model_params['Q']}.{model_params['dimension']}"
    train_dir.mkdir(parents=True, exist_ok=True)
    model_params["train_dir"] = train_dir
    total_epoches = model_params.pop("epoches")  # 总训练轮数

    # 1. 初始化模型
    model = BEAM(**model_params)

    # 2. 核心：加载checkpoint（恢复训练进度）
    start_epoch = 1  # 默认从第1轮开始
    if resume and os.path.exists(resume):
        try:
            # 加载checkpoint文件
            checkpoint = torch.load(resume, map_location=f"cuda:{device}")
            # 恢复模型参数
            model.load_state_dict(checkpoint.get("model_state_dict", {}))
            # 恢复上次训练的轮数（如果checkpoint里有epoch记录）
            start_epoch = checkpoint.get("epoch", 1)
            print(f"✅ 成功加载checkpoint: {resume}")
            print(f"✅ 从第 {start_epoch} 轮继续训练（总轮数：{total_epoches}）")
        except Exception as e:
            print(f"⚠️ 加载checkpoint失败: {e}，将从头开始训练")
            start_epoch = 1

    # 3. 开始训练（从start_epoch开始，而不是从1开始）
    model.train(epoches=total_epoches, start_epoch=start_epoch)
    model.save_embeddings(path=str(train_dir))

if __name__ == "__main__":
    main()