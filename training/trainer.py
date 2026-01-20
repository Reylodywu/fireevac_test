import time
import numpy as np
import torch
import rl_utils
import datetime
from MASAC_PER import MASAC
from MADDPG_PER import MADDPG
from envs.fire_evac_env import FireEvacuationParallelEnv
from .utils import save_checkpoint, load_checkpoint, plot_training_curves, _init_empty_loss_history


class MATrainer:
    """
    通用多智能体训练器，支持 MASAC 和 MADDPG
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device

        # 1. 初始化环境
        self.env = FireEvacuationParallelEnv(
            n_agents=args.n_agents,
            grid_size_x=args.grid_size_x,
            grid_size_y=args.grid_size_y,
            action_bound=args.action_bound,
            max_steps=args.episode_length,
            render_mode=args.render_mode if args.mode == 'train' else None,
            mask_fire_obs=args.mask_fire_obs
        )
        self.env.reset()

        # 2. 初始化 Buffer (PER)
        self.replay_buffer = rl_utils.PrioritizedReplayBufferWithSave(
            capacity=args.buffer_size,
            alpha=args.per_alpha,
            beta_start=args.per_beta_start,
            beta_frames=args.per_beta_frames
        )

        # 3. 初始化 Agent (根据 args.algo 选择)
        self._init_agent()

        # 4. 状态记录
        self.loss_history = _init_empty_loss_history(len(self.env.agents))
        self.return_list = []
        self.start_episode = 0

    def _init_agent(self):
        """根据配置初始化 MASAC 或 MADDPG"""
        state_dims = []
        action_dims = []
        for agent in self.env.agents:
            obs_space = self.env.observation_space(agent)
            act_space = self.env.action_space(agent)
            state_dims.append(obs_space.shape[0])
            action_dims.append(act_space.shape[0])

        critic_input_dim = sum(state_dims) + sum(action_dims)

        if self.args.algo == 'masac':
            print("🚀 Initializing MASAC Agent...")
            self.agent = MASAC(
                env=self.env,
                device=self.device,
                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                hidden_dim=self.args.hidden_dim,
                state_dims=state_dims,
                action_dims=action_dims,
                critic_input_dim=critic_input_dim,
                gamma=self.args.gamma,
                tau=self.args.tau,
                action_bound=self.args.action_bound,
                num_ensembles=self.args.num_ensembles,
                alpha_lr=self.args.alpha_lr,
                target_entropy=self.args.target_entropy
            )
        elif self.args.algo == 'maddpg':
            print("🚀 Initializing MADDPG Agent...")
            # 确保 args 中包含 exploration_noise，如果没有则给默认值
            noise = getattr(self.args, 'exploration_noise', 0.1)
            self.agent = MADDPG(
                env=self.env,
                device=self.device,
                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                hidden_dim=self.args.hidden_dim,
                state_dims=state_dims,
                action_dims=action_dims,
                critic_input_dim=critic_input_dim,
                gamma=self.args.gamma,
                tau=self.args.tau,
                action_bound=self.args.action_bound,
                num_ensembles=self.args.num_ensembles,  # 实验中为了公平对比，DDPG也可以用Ensemble
                exploration_noise=noise
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.args.algo}")

    def train(self):
        # 尝试恢复训练
        if self.args.resume_training:
            self.start_episode, self.return_list, self.loss_history = load_checkpoint(
                self.args.load_path, self.agent, self.replay_buffer, self.device
            )

        print(f"🚀 Start Training ({self.args.algo}) from Episode {self.start_episode}...")
        training_start_time = time.time()
        total_step = 0

        for i_episode in range(self.start_episode, self.args.num_episodes):
            state, _ = self.env.reset()

            # 本轮的临时Loss记录
            ep_losses = {k: [[] for _ in range(self.args.n_agents)] for k in ['critic', 'actor', 'alpha', 'val']}

            for _ in range(self.args.episode_length):
                # 1. 动作与交互 (MASAC和MADDPG接口一致)
                actions = self.agent.take_action(state, explore=True)
                next_state, reward, done, truncated, _ = self.env.step(actions)
                self.replay_buffer.add(state, actions, reward, next_state, done)
                state = next_state
                total_step += 1

                # 2. 训练更新 (PER逻辑)
                if self.replay_buffer.size() >= self.args.minimal_size and total_step % self.args.update_interval == 0:
                    self._update_agent(ep_losses)

            # 3. 记录日志
            self._record_episode_losses(i_episode, ep_losses)

            # --- 3. 打印训练统计 (每100轮) ---
            if (i_episode + 1) % self.args.eval_freq == 0:
                self._log_training_stats(i_episode, ep_losses, training_start_time)
                # self._log_training_stats(i_episode, ep_losses)

            # 4. 定期评估
            if (i_episode + 1) % self.args.eval_freq == 0:
                self._evaluate_and_log(i_episode)

            # 5. 保存模型
            if (i_episode + 1) % self.args.checkpoint_freq == 0:
                save_checkpoint(i_episode + 1, self.agent, self.replay_buffer,
                                self.return_list, self.loss_history)

        # 结束处理
        self.env.close()
        if self.args.plot_results:
            plot_training_curves(
                self.loss_history,
                self.return_list,
                list(self.env.agents),
                self.args.eval_freq,
            )

    def _update_agent(self, ep_losses):
        *sample, indices, weights = self.replay_buffer.sample(self.args.batch_size)
        sample_processed = [self._stack_array(x) for x in sample]
        weights = torch.FloatTensor(weights).to(self.device)

        all_td_errors = []
        for a_i in range(len(self.env.agents)):
            # update 接口已对齐：
            # MASAC 返回: (c_loss, a_loss, alpha, alpha_loss, td_error)
            # MADDPG 返回: (c_loss, a_loss, 0.0,   0.0,        td_error)
            c_loss, a_loss, alpha_val, alpha_loss, td_error = self.agent.update(sample_processed, a_i, weights)
            all_td_errors.append(td_error)

            # 记录
            ep_losses['critic'][a_i].append(c_loss)
            ep_losses['actor'][a_i].append(a_loss)
            # 只有 SAC 记录 meaningful 的 alpha
            if self.args.algo == 'masac':
                ep_losses['alpha'][a_i].append(alpha_loss)
                ep_losses['val'][a_i].append(alpha_val)

        # 更新 PER 优先级
        avg_td_errors = np.mean(all_td_errors, axis=0)
        self.replay_buffer.update_priorities(indices, avg_td_errors)
        self.agent.update_all_targets()

    def _log_training_stats(self, i_episode, ep_losses,start_time=None):
        """打印训练统计信息"""
        time_str = ""
        if start_time:
            elapsed_seconds = int(time.time() - start_time)
            # 格式化为 HH:MM:SS
            time_str = f" | Time: {datetime.timedelta(seconds=elapsed_seconds)}"

        # [修改] 在标题中加入时间信息
        print(f"\nEpisode {i_episode + 1} [{self.args.algo}] Debug Info{time_str}:")
        # 1. 打印 PER Buffer 统计
        rb = self.replay_buffer
        if hasattr(rb, 'tree'):
            print(f"  [PER] Buffer: {rb.size()}/{rb.capacity} | "
                  f"Max Prio: {rb.max_priority:.4f} | "
                  f"Total Prio: {rb.tree.total():.2f} | "
                  f"Beta: {rb.beta_by_frame(rb.frame):.4f}")
        else:
            print(f"  [Buffer] Size: {rb.size()}/{rb.capacity}")

        # 2. 打印 Alpha 值 (仅限 MASAC)
        if self.args.algo == 'masac':
            for a_i in range(len(self.env.agents)):
                alphas = ep_losses['val'][a_i]
                valid_alphas = [a for a in alphas if a is not None]
                if valid_alphas:
                    avg_alpha = np.mean(valid_alphas)
                    print(f"  [Agent {a_i}] Avg Alpha: {avg_alpha:.4f}")

    def _evaluate_and_log(self, i_episode):
        n_episodes = self.args.eval_episodes
        returns = self._evaluate_rollout(n_episode=n_episodes)
        self.return_list.append(returns)
        print(f"Episode: {i_episode + 1}, Returns: {returns}")

    def _evaluate_rollout(self, n_episode):
        returns = np.zeros(self.args.n_agents)
        eval_steps = self.args.episode_length

        for _ in range(n_episode):
            s, _ = self.env.reset()
            ep_r = np.zeros(self.args.n_agents)

            for _ in range(eval_steps):
                # MADDPG 和 MASAC 的 take_action 接口一致
                a = self.agent.take_action(s, explore=False)
                s, r, d, t, _ = self.env.step(a)

                r_array = np.array([r.get(ag, 0.0) for ag in self.env.agents])
                ep_r += r_array

                if all(d.values()) or all(t.values()):
                    break
            returns += ep_r

        return (returns / n_episode).tolist()

    def _record_episode_losses(self, episode, ep_losses):
        self.loss_history['episodes'].append(episode + 1)
        for i in range(len(self.env.agents)):
            # 基础 Loss
            for k, target_k in zip(['critic', 'actor'], ['critic_loss', 'actor_loss']):
                if ep_losses[k][i]:
                    self.loss_history[target_k][i].append(np.mean(ep_losses[k][i]))
                else:
                    self.loss_history[target_k][i].append(None)

            # Alpha 相关 (仅 SAC 记录数值，DDPG 记为 None)
            if self.args.algo == 'masac':
                for k, target_k in zip(['alpha', 'val'], ['alpha_loss', 'alpha_value']):
                    if ep_losses[k][i]:
                        self.loss_history[target_k][i].append(np.mean(ep_losses[k][i]))
                    else:
                        self.loss_history[target_k][i].append(None)
            else:
                self.loss_history['alpha_loss'][i].append(None)
                self.loss_history['alpha_value'][i].append(None)

    def _stack_array(self, x):
        """处理 PER 采样的数据结构"""
        if not x: return []
        first = x[0]
        keys = list(first[0].keys()) if isinstance(first, tuple) else list(first.keys())
        rearranged = []
        for k in keys:
            data = [d[0][k] if isinstance(d, tuple) else d[k] for d in x]
            rearranged.append(data)
        return [torch.FloatTensor(np.vstack(a)).to(self.device) for a in rearranged]