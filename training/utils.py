import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import rl_utils  # 保持引用你原有的工具库


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(episode, masac, replay_buffer, return_list, loss_history, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pth')

    checkpoint = {
        'episode': episode,
        'model_state_dict': masac.get_state_dict(),
        'replay_buffer_state': replay_buffer.get_state(),
        'return_list': return_list,
        'loss_history': loss_history,
    }
    torch.save(checkpoint, path)

    # 保存 latest
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    print(f"✓ Checkpoint saved: {path}")


def load_checkpoint(path, masac, replay_buffer, device):
    if not os.path.exists(path):
        print(f"⚠️ Checkpoint not found: {path}")
        return 0, [], _init_empty_loss_history(len(masac.agents))

    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    masac.load_state_dict(checkpoint['model_state_dict'])
    if 'replay_buffer_state' in checkpoint:
        replay_buffer.load_state(checkpoint['replay_buffer_state'])

    return checkpoint['episode'], checkpoint['return_list'], checkpoint['loss_history']


def _init_empty_loss_history(n_agents):
    return {
        'critic_loss': [[] for _ in range(n_agents)],
        'actor_loss': [[] for _ in range(n_agents)],
        'alpha_loss': [[] for _ in range(n_agents)],
        'alpha_value': [[] for _ in range(n_agents)],
        'episodes': []
    }


def plot_training_curves(loss_history, return_list, agent_names, eval_freq, save_dir="."):
    """封装原本巨大的绘图逻辑"""
    episodes = loss_history['episodes']
    if not episodes: return

    # 1. 绘制 Loss 曲线 (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
            # 过滤 None 并平滑
            valid_data = [(ep, val) for ep, val in zip(episodes, data) if val is not None]
            if valid_data:
                eps, vals = zip(*valid_data)
                smoothed = rl_utils.moving_average(list(vals), 9)
                ax.plot(eps, smoothed, label=agent_name)

        ax.set_title(title)
        ax.set_xlabel("Episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "masac_losses.png"))
    plt.close()
    print("✓ Loss curves saved.")

    # 2. 绘制 Return 曲线
    return_array = np.array(return_list)
    for i, agent_name in enumerate(agent_names):
        if return_array.shape[0] == 0: continue
        plt.figure()
        x_axis = np.arange(len(return_array)) * eval_freq
        plt.plot(x_axis, rl_utils.moving_average(return_array[:, i], 9))
        plt.title(f"{agent_name} Returns")
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        plt.savefig(os.path.join(save_dir, f"{agent_name}_masac.png"))
        plt.close()