import time
import numpy as np
import torch
import os
from MASAC_PER import MASAC
from MADDPG_PER import MADDPG
from envs.fire_evac_env import FireEvacuationParallelEnv


class MATester:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.env = FireEvacuationParallelEnv(
            n_agents=args.n_agents,
            grid_size_x=args.grid_size_x,
            grid_size_y=args.grid_size_y,
            max_steps=args.episode_length,
            action_bound=args.action_bound,
            render_mode="human",  # 强制 human 模式用于观看
            mask_fire_obs=args.mask_fire_obs
        )

    def test(self):
        # 1. 准备环境参数
        self.env.reset()
        state_dims = []
        action_dims = []
        for agent in self.env.agents:
            state_dims.append(self.env.observation_space(agent).shape[0])
            action_dims.append(self.env.action_space(agent).shape[0])

        critic_input_dim = sum(state_dims) + sum(action_dims)

        # 2. 初始化 Agent
        agent = None
        if self.args.algo == 'masac':
            print("🚀 Initializing MASAC Agent for Testing...")
            agent = MASAC(
                env=self.env, device=self.device, actor_lr=0, critic_lr=0,
                hidden_dim=self.args.hidden_dim, state_dims=state_dims, action_dims=action_dims,
                critic_input_dim=critic_input_dim, gamma=self.args.gamma, tau=self.args.tau,
                action_bound=self.args.action_bound, num_ensembles=self.args.num_ensembles, alpha_lr=0
            )
        elif self.args.algo == 'maddpg':
            print("🚀 Initializing MADDPG Agent for Testing...")
            agent = MADDPG(
                env=self.env, device=self.device, actor_lr=0, critic_lr=0,
                hidden_dim=self.args.hidden_dim, state_dims=state_dims, action_dims=action_dims,
                critic_input_dim=critic_input_dim, gamma=self.args.gamma, tau=self.args.tau,
                action_bound=self.args.action_bound, num_ensembles=self.args.num_ensembles, exploration_noise=0.0
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.args.algo}")

        # 3. 加载 Checkpoint
        path = self.args.load_path
        if not os.path.exists(path):
            print(f"❌ Checkpoint not found: {path}")
            return

        print(f"Loading model from: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        agent.load_state_dict(checkpoint['model_state_dict'])

        # 4. 开始测试循环
        print(f"Starting evaluation for {self.args.eval_episodes} episodes...")
        success_count = 0

        # [新增] 全局统计容器
        global_metrics = {
            'safety_score': [],
            'total_exposure': [],
            'max_danger': [],
            'path_length': []  # 假设你在Env的evaluate里也加了这个，如果没有可忽略
        }

        for ep in range(self.args.eval_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0
            steps = 0
            current_ep_dir = os.path.join("results", f"ep_{ep}")
            # 渲染初始状态
            self.env.render()
            time.sleep(0.5)

            termination_reason = "Max Steps"

            for _ in range(self.args.episode_length):
                actions = agent.take_action(obs, explore=False)
                obs, reward, term, trunc, info = self.env.step(actions)

                rewards_sum = sum(reward.values())
                ep_reward += rewards_sum
                steps += 1

                self.env.render()
                time.sleep(0.05)

                if steps in [30, 100, 200]:
                    # 直接通过 env.visualizer 调用
                    self.env.visualizer.save_paper_assets(step=steps, output_dir=current_ep_dir)

                if all(term.values()):
                    termination_reason = "Success"
                    success_count += 1
                    break
                if all(trunc.values()):
                    termination_reason = "Timeout"
                    break

            # === 回合结束，调用风险评估 ===
            # 获取本回合所有智能体的风险数据
            # 格式: {agent_id: {'total_exposure': ..., 'max_danger': ...}}
            risk_stats = self.env.evaluate_trajectory_risk()

            # 打印本回合报告
            print(f"\n📊 Episode {ep + 1} Report [{termination_reason}]")
            print(f"   Steps: {steps} | Reward: {ep_reward:.2f}")

            for agent_id, stats in risk_stats.items():
                # 直接读取 Env 算好的数据
                s_score = stats['safety_score']
                max_d = stats['max_danger']
                exp = stats['total_exposure']
                pl = stats['path_length']

                # 存入全局列表 (用于最后求平均)
                global_metrics['safety_score'].append(s_score)
                global_metrics['total_exposure'].append(exp)
                global_metrics['max_danger'].append(max_d)

                # 简单的危险图标逻辑
                icon = "🟢"
                if max_d > 0.8:
                    icon = "🔴"
                elif max_d > 0.4:
                    icon = "🟠"

                print(f"   - {agent_id}: Score={s_score:.1f} | Length={pl:.1f}m | MaxDanger={max_d:.2f} {icon}")

            time.sleep(0.5)

        # 5. 最终总结
        print("\n" + "=" * 50)
        print("🚀 FINAL EVALUATION REPORT")
        print("=" * 50)
        print(
            f"Success Rate:     {success_count}/{self.args.eval_episodes} ({success_count / self.args.eval_episodes * 100:.1f}%)")
        print(f"Avg Reward:      {ep_reward:.2f} (Last Ep)")  # 这里可以优化为存所有ep reward求平均

        if global_metrics['safety_score']:
            print(
                f"Avg Safety Score: {np.mean(global_metrics['safety_score']):.2f} ± {np.std(global_metrics['safety_score']):.2f}")
            print(f"Avg Exposure:     {np.mean(global_metrics['total_exposure']):.2f}")
            print(f"Avg Max Danger:   {np.mean(global_metrics['max_danger']):.2f}")
            print(f"Worst Case Danger:{np.max(global_metrics['max_danger']):.2f}")
        else:
            print("No risk metrics collected.")

        print("=" * 50)
        self.env.close()