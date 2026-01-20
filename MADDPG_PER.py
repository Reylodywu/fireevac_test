import torch
import torch.nn.functional as F
import numpy as np


class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient"""

    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau, action_bound,
                 num_ensembles=1,  # DDPG通常为1，但为了对比可以设为与SAC一致
                 exploration_noise=0.1):  # DDPG特有的探索噪声标准差

        self.agents = []
        self.agent_names = list(env.agents) if hasattr(env, 'agents') else [f'agent_{i}' for i in
                                                                            range(len(state_dims))]

        action_bounds = [action_bound] * len(env.agents)
        self.exploration_noise = exploration_noise

        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device, action_bounds[i],
                     num_ensembles=num_ensembles,
                     tau=tau))

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def get_state_dict(self):
        """获取所有智能体的完整状态字典"""
        state_dict = {}
        for i, agent in enumerate(self.agents):
            state_dict[f'agent_{i}'] = {
                'actor': agent.actor.state_dict(),
                'target_actor': agent.target_actor.state_dict(),  # ✅ DDPG需要保存Target Actor
                'critic': agent.critic.state_dict(),
                'target_critic': agent.target_critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                # ❌ DDPG没有 alpha 相关参数
            }
        return state_dict

    def load_state_dict(self, state_dict):
        """加载所有智能体的完整状态字典"""
        for i, agent in enumerate(self.agents):
            agent_data = state_dict[f'agent_{i}']

            # 加载网络参数
            agent.actor.load_state_dict(agent_data['actor'])
            agent.target_actor.load_state_dict(agent_data['target_actor'])  # ✅ 加载Target Actor
            agent.critic.load_state_dict(agent_data['critic'])
            agent.target_critic.load_state_dict(agent_data['target_critic'])

            # 加载优化器
            agent.actor_optimizer.load_state_dict(agent_data['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(agent_data['critic_optimizer'])

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        """获取所有智能体的目标策略网络"""
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore=True):
        """所有智能体采取动作"""
        if isinstance(states, tuple):
            observations = states[0]
        else:
            observations = states

        states_list = [observations[agent_name] for agent_name in self.agent_names]

        states_tensor = [
            torch.from_numpy(states_list[i]).float().unsqueeze(0).to(self.device)
            for i in range(len(self.agents))
        ]

        # 传递 exploration_noise 参数
        actions_list = [
            agent.take_action(state.cpu().numpy()[0], explore, self.exploration_noise)
            for agent, state in zip(self.agents, states_tensor)
        ]

        actions_dict = {
            self.agent_names[i]: actions_list[i]
            for i in range(len(self.agent_names))
        }

        return actions_dict

    def update(self, sample, i_agent, importance_weights=None):
        """
        ✅ MADDPG的更新逻辑
        """
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        # 处理PER权重
        if importance_weights is None:
            importance_weights = torch.ones(obs[0].shape[0], 1).to(self.device)
        else:
            if not isinstance(importance_weights, torch.Tensor):
                importance_weights = torch.FloatTensor(importance_weights).to(self.device)
            if importance_weights.dim() == 1:
                importance_weights = importance_weights.unsqueeze(1)

        # ========== 1. 更新Critic ==========
        cur_agent.critic_optimizer.zero_grad()

        with torch.no_grad():
            # ✅ DDPG关键差异：使用 Target Actor 计算下一时刻动作
            all_target_next_actions = []
            for i, (target_pi, _next_obs) in enumerate(zip(self.target_policies, next_obs)):
                # DDPG Update中计算目标值时不加噪声
                next_action = target_pi(_next_obs)
                all_target_next_actions.append(next_action)

            # 构造Target Critic输入
            target_critic_input = torch.cat((*next_obs, *all_target_next_actions), dim=1)

            # 获取目标Q值
            # 注意：如果 num_ensembles > 1，这里可以取 mean 或 min。
            # 传统的 MADDPG 只用 1 个 Q，但为了 robustness 我们取 min (类似 TD3) 或 mean。
            target_q_values = cur_agent.target_critic(target_critic_input)
            target_q = torch.min(target_q_values, dim=1, keepdim=True)[0]  # Min-Q Trick

            # ✅ DDPG 目标 Q 值：r + γ * (1-d) * Q_target(s', a'_target)
            # (没有熵项)
            target_q_value = rew[i_agent].view(-1, 1) + \
                             self.gamma * (1 - done[i_agent].view(-1, 1)) * target_q

        # 当前Critic的Q值估计
        critic_input = torch.cat((*obs, *act), dim=1)
        current_q_values = cur_agent.critic(critic_input)

        # 计算TD误差
        current_q_mean = current_q_values.mean(dim=1, keepdim=True)
        td_errors = (target_q_value - current_q_mean).detach().cpu().numpy().squeeze()

        # 计算Critic损失
        target_q_expanded = target_q_value.repeat(1, current_q_values.shape[1])
        elementwise_loss = F.mse_loss(current_q_values, target_q_expanded, reduction='none')
        elementwise_loss = elementwise_loss.mean(dim=1, keepdim=True)
        weighted_loss = elementwise_loss * importance_weights
        critic_loss = weighted_loss.mean()

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(cur_agent.critic.parameters(), max_norm=1.0)
        cur_agent.critic_optimizer.step()

        # ========== 2. 更新Actor ==========
        cur_agent.actor_optimizer.zero_grad()

        # 生成当前智能体的新动作（使用当前Actor）
        new_action = cur_agent.actor(obs[i_agent])

        # 构造输入：其他智能体动作不变(detached)，当前智能体使用新动作
        all_actions = []
        for i, _act in enumerate(act):
            if i == i_agent:
                all_actions.append(new_action)
            else:
                all_actions.append(_act.detach())

        # 构造Critic输入
        actor_critic_input = torch.cat((*obs, *all_actions), dim=1)

        # ✅ Actor损失：最大化 Q(s, a) -> 最小化 -Q(s, a)
        q_values = cur_agent.critic(actor_critic_input)
        q_value = q_values.mean(dim=1, keepdim=True)  # 使用Ensemble的均值

        actor_loss = -q_value.mean()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), max_norm=1.0)
        cur_agent.actor_optimizer.step()

        # DDPG 没有 Alpha 更新

        return (
            critic_loss.item(),
            actor_loss.item(),
            0.0,  # alpha value placeholder
            0.0,  # alpha loss placeholder
            td_errors
        )

    def update_all_targets(self):
        """✅ DDPG软更新：同时更新 Critic 和 Actor 的目标网络"""
        for agt in self.agents:
            agt.soft_update(agt.critic, agt.target_critic, self.tau)
            agt.soft_update(agt.actor, agt.target_actor, self.tau)


# ========== EnsembleCritic (保持不变) ==========
class EnsembleCritic(torch.nn.Module):
    """集成Critic网络"""

    def __init__(self, num_in, hidden_dim, num_ensembles=1):
        super(EnsembleCritic, self).__init__()
        self.num_ensembles = num_ensembles

        self.encoder = torch.nn.Linear(num_in, hidden_dim)

        self.q_networks = torch.nn.ModuleList()
        for i in range(num_ensembles):
            q_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            )
            self.q_networks.append(q_net)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        q_values = [q_net(x) for q_net in self.q_networks]
        q_values = torch.cat(q_values, dim=1)
        return q_values


# ========== DeterministicActor (替换 StochasticActor) ==========
class DeterministicActor(torch.nn.Module):
    """✅ DDPG的确定性策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super(DeterministicActor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)  # 直接输出动作维度
        self.action_bound = action_bound

    def forward(self, state):
        """前向传播，输出确定性动作"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_bound


# ========== DDPG智能体类 (替换 SAC) ==========
class DDPG:
    """DDPG算法（适配多智能体场景）"""

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, action_bound,
                 num_ensembles=1,
                 tau=0.005):

        # ✅ 确定性策略 Actor
        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim, action_bound).to(device)
        self.target_actor = DeterministicActor(state_dim, action_dim, hidden_dim, action_bound).to(device)
        # 初始化 Target Actor
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Critic (可以选 num_ensembles=1 变为标准 DDPG)
        self.critic = EnsembleCritic(critic_input_dim, hidden_dim, num_ensembles).to(device)
        self.target_critic = EnsembleCritic(critic_input_dim, hidden_dim, num_ensembles).to(device)
        # 初始化 Target Critic
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_bound = action_bound
        self.action_dim = action_dim
        self.device = device
        self.tau = tau

    def take_action(self, state, explore=True, noise_std=0.1):
        """采取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 获取确定性动作
        action = self.actor(state_tensor).detach().cpu().numpy()[0]

        if explore:
            # ✅ DDPG 探索：添加高斯噪声
            noise = np.random.normal(0, noise_std * self.action_bound, size=self.action_dim)
            action = np.clip(action + noise, -self.action_bound, self.action_bound)

        return action

    def soft_update(self, net, target_net, tau):
        """软更新目标网络"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)