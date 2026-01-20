import argparse
import torch


def get_config():
    parser = argparse.ArgumentParser(description='Fire Evacuation MASAC Training')

    # ==================== 基础模式 ====================
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'both'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_cuda', type=bool, default=True)

    # ==================== 环境参数 ====================
    parser.add_argument('--n_agents', type=int, default=1)
    parser.add_argument('--grid_size_x', type=int, default=80)
    parser.add_argument('--grid_size_y', type=int, default=40)
    parser.add_argument('--action_bound', type=float, default=1.0)
    parser.add_argument('--render_mode', type=str, default='human')
    parser.add_argument('--mask_fire_obs', type=bool, default=False,
                        help='If True, mask fire perception (danger sensor and fire rays) from observation.')

    # ==================== 算法参数 ====================
    parser.add_argument('--algo', type=str, default='masac', choices=['masac', 'maddpg'])
    parser.add_argument('--num_ensembles', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--alpha_lr', type=float, default=5e-5)
    parser.add_argument('--target_entropy', type=float, default=-2.0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=5e-3)

    # ==================== 训练参数 ====================
    parser.add_argument('--num_episodes', type=int, default=4000)
    parser.add_argument('--episode_length', type=int, default=150)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--update_interval', type=int, default=3)
    parser.add_argument('--minimal_size', type=int, default=2000)

    # PER (优先经验回放) 参数
    parser.add_argument('--per_alpha', type=float, default=0.6)
    parser.add_argument('--per_beta_start', type=float, default=0.4)
    parser.add_argument('--per_beta_frames', type=int, default=100000)

    # ==================== 评估与保存 ====================
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)
    parser.add_argument('--load_path', type=str, default='checkpoints/latest_checkpoint.pth')
    parser.add_argument('--resume_training', type=bool, default=True)
    parser.add_argument('--plot_results', type=bool, default=True)

    args = parser.parse_args()

    # 自动设置设备
    args.device = torch.device('cuda' if args.enable_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {args.device}")

    return args