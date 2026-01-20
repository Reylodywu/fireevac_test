import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import sys

# 引入你的工具库以便复用 moving_average 等函数
# 假设脚本在项目根目录，直接 import
import rl_utils


def moving_average(a, window_size):
    # 复用 rl_utils 里的逻辑，或者直接复制过来
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def replot(checkpoint_path, save_dir=".", manual_time_str=None):
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path} ...")
    # 加载数据，不加载到 GPU 以免报错
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)

    loss_history = checkpoint.get('loss_history', None)
    return_list = checkpoint.get('return_list', None)
    episode = checkpoint.get('episode', 0)

    if loss_history is None or return_list is None:
        print("Error: Checkpoint does not contain loss_history or return_list.")
        return

    print(f"Loaded data up to episode {episode}.")

    # 自动推断 agent 数量和名字
    # loss_history['critic_loss'] 是一个列表，长度等于 agent 数量
    n_agents = len(loss_history['critic_loss'])
    agent_names = [f"agent_{i}" for i in range(n_agents)]

    # ================= 绘图逻辑 (复用 utils.py) =================
    # 1. Loss Curves
    episodes = loss_history['episodes']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    title_text = f"Training Losses (Episodes 0-{episode})"
    if manual_time_str:
        title_text += f" | Total Duration: {manual_time_str}"
    fig.suptitle(title_text, fontsize=16, y=0.98)

    metrics = [
        ('critic_loss', "Critic Loss"),
        ('actor_loss', "Actor Loss"),
        ('alpha_loss', "Alpha Loss"),
        ('alpha_value', "Alpha Value")
    ]

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for i, agent_name in enumerate(agent_names):
            data = loss_history[key][i]
            # 过滤 None
            valid_data = [(ep, val) for ep, val in zip(episodes, data) if val is not None]
            if valid_data:
                eps, vals = zip(*valid_data)
                # 平滑处理
                if len(vals) > 10:
                    try:
                        smoothed = moving_average(list(vals), 9)
                        ax.plot(eps, smoothed, label=agent_name)
                    except:
                        ax.plot(eps, vals, label=agent_name, alpha=0.5)
                else:
                    ax.plot(eps, vals, label=agent_name)

        ax.set_title(title)
        ax.set_xlabel("Episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "replot_losses.png")
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

    # 2. Return Curves
    # 假设 eval_freq = 100 (或者你可以作为参数传入)
    # 我们可以通过数据点数量反推： episode / len(return_list)
    return_array = np.array(return_list)
    if len(return_list) > 0:
        est_freq = episode / len(return_list)
    else:
        est_freq = 1

    for i, agent_name in enumerate(agent_names):
        plt.figure()
        x_axis = np.arange(len(return_array)) * est_freq

        # 平滑
        if len(return_array[:, i]) > 10:
            y_data = moving_average(return_array[:, i], 9)
        else:
            y_data = return_array[:, i]

        plt.plot(x_axis, y_data)

        t_str = f"{agent_name} Returns"
        if manual_time_str:
            t_str += f"\n(Duration: {manual_time_str})"
        plt.title(t_str)
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"replot_{agent_name}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    # 用法：python replot_from_checkpoint.py
    # 你可以在这里手动填入你记得的训练时间，比如 "5h 30m"
    manual_time = "Unknown"

    # 指向你的 checkpoint 文件
    ckpt_path = "checkpoints/latest_checkpoint.pth"

    replot(ckpt_path, manual_time_str=manual_time)