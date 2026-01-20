import torch
import torch.nn.functional as F
import numpy as np


class MASAC:
    """Multi-Agent Soft Actor-Critic"""

    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau, action_bound,
                 num_ensembles=4,  # 集成Q网络数量
                 alpha_lr=3e-4,  # 温度参数学习率
                 target_entropy=None):  # 目标熵（默认为-action_dim）

        self.agents = []
        self.agent_names = list(env.agents) if hasattr(env, 'agents') else [f'agent_{i}' for i in
                                                                            range(len(state_dims))]

        action_bounds = [action_bound] * len(env.agents)

        # 如果没有指定目标熵，为每个智能体设置默认值
        if target_entropy is None:
            # target_entropies = [-np.log(action_dims[i]) for i in range(len(env.agents))]
            target_entropies = [-float(action_dims[i]) for i in range(len(env.agents))]
        else:
            target_entropies = [target_entropy] * len(env.agents)

        for i in range(len(env.agents)):
            self.agents.append(
                SAC(state_dims[i], action_dims[i], critic_input_dim,
                    hidden_dim, actor_lr, critic_lr, device, action_bounds[i],
                    num_ensembles=num_ensembles,
                    alpha_lr=alpha_lr,
                    target_entropy=target_entropies[i],
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
                'critic': agent.critic.state_dict(),
                'target_critic': agent.target_critic.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'alpha_log': agent.alpha_log.detach().cpu(),  # ✅ detach并转到CPU
                'alpha_optimizer': agent.alpha_optimizer.state_dict(),
            }
        return state_dict

    def load_state_dict(self, state_dict):
        """加载所有智能体的完整状态字典"""
        for i, agent in enumerate(self.agents):
            agent_data = state_dict[f'agent_{i}']

            # 加载网络参数
            agent.actor.load_state_dict(agent_data['actor'])
            agent.critic.load_state_dict(agent_data['critic'])
            agent.target_critic.load_state_dict(agent_data['target_critic'])

            # ✅ 正确加载 alpha_log（保持requires_grad=True）
            with torch.no_grad():
                agent.alpha_log.copy_(agent_data['alpha_log'].to(agent.alpha_log.device))

            # ✅ 先加载优化器状态，再更新参数引用
            agent.actor_optimizer.load_state_dict(agent_data['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(agent_data['critic_optimizer'])
            agent.alpha_optimizer.load_state_dict(agent_data['alpha_optimizer'])

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

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

        actions_list = [
            agent.take_action(state.cpu().numpy()[0], explore)
            for agent, state in zip(self.agents, states_tensor)
        ]

        actions_dict = {
            self.agent_names[i]: actions_list[i]
            for i in range(len(self.agent_names))
        }

        return actions_dict

    def update(self, sample, i_agent, importance_weights=None):
        """
        ✅ MASAC的更新逻辑（支持PER）

        Args:
            sample: (obs, act, rew, next_obs, done)
            i_agent: 当前更新的智能体索引
            importance_weights: 重要性采样权重 [batch_size, 1]，用于修正PER的偏差

        Returns:
            tuple: (critic_loss, actor_loss, alpha_value, alpha_loss, td_errors)
                   td_errors 用于更新PER的优先级
        """
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        # ✅ 如果没有提供权重，使用均匀权重
        if importance_weights is None:
            importance_weights = torch.ones(obs[0].shape[0], 1).to(self.device)
        else:
            # 确保权重是正确的形状和设备
            if not isinstance(importance_weights, torch.Tensor):
                importance_weights = torch.FloatTensor(importance_weights).to(self.device)
            if importance_weights.dim() == 1:
                importance_weights = importance_weights.unsqueeze(1)

        # ========== 1. 更新Critic ==========
        cur_agent.critic_optimizer.zero_grad()

        with torch.no_grad():
            # 获取所有智能体在下一状态的动作
            all_next_actions = []
            current_agent_next_logprob = None

            for i, (pi, _next_obs) in enumerate(zip(self.policies, next_obs)):
                if i == i_agent:
                    # 当前智能体：需要对数概率用于熵项
                    next_action, next_logprob = cur_agent.actor.get_action_logprob(_next_obs)
                    all_next_actions.append(next_action)
                    current_agent_next_logprob = next_logprob
                else:
                    # 其他智能体：只需要动作
                    next_action = self.agents[i].actor.get_action(_next_obs)
                    all_next_actions.append(next_action)

            # 构造Critic输入：(所有obs, 所有actions)
            target_critic_input = torch.cat((*next_obs, *all_next_actions), dim=1)

            # 获取目标Q值（取所有Q网络的最小值）
            target_q_values = cur_agent.target_critic(target_critic_input)
            target_q = torch.min(target_q_values, dim=1, keepdim=True)[0]

            # 当前alpha值
            alpha = cur_agent.alpha_log.exp()

            # ✅ SAC的目标Q值：r + γ * (Q(s',a') - α * log π(a'|s'))
            target_q_value = rew[i_agent].view(-1, 1) + \
                             self.gamma * (1 - done[i_agent].view(-1, 1)) * \
                             (target_q - alpha * current_agent_next_logprob)

        # 当前Critic的Q值估计
        critic_input = torch.cat((*obs, *act), dim=1)
        current_q_values = cur_agent.critic(critic_input)

        # ✅ 计算TD误差（用于更新PER优先级）
        # 使用Q网络的平均值计算TD误差
        current_q_mean = current_q_values.mean(dim=1, keepdim=True)
        td_errors = (target_q_value - current_q_mean).detach().cpu().numpy().squeeze()

        # ✅ 计算加权的Critic损失
        target_q_expanded = target_q_value.repeat(1, current_q_values.shape[1])

        # 对每个Q网络计算MSE损失
        elementwise_loss = F.mse_loss(current_q_values, target_q_expanded, reduction='none')

        # 对ensemble维度求平均
        elementwise_loss = elementwise_loss.mean(dim=1, keepdim=True)

        # ✅ 应用重要性采样权重
        weighted_loss = elementwise_loss * importance_weights
        critic_loss = weighted_loss.mean()

        # 反向传播更新Critic
        critic_loss.backward()

        # # ✅ 可选：梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(cur_agent.critic.parameters(), max_norm=1.0)

        cur_agent.critic_optimizer.step()

        # ========== 2. 更新Actor ==========
        cur_agent.actor_optimizer.zero_grad()

        # 采样当前智能体的新动作
        new_action, new_logprob = cur_agent.actor.get_action_logprob(obs[i_agent])

        # 构造所有智能体的动作（其他智能体使用当前动作）
        all_actions = []
        for i, _act in enumerate(act):
            if i == i_agent:
                all_actions.append(new_action)
            else:
                all_actions.append(_act.detach())

        # 构造Critic输入
        actor_critic_input = torch.cat((*obs, *all_actions), dim=1)

        # 获取Q值（使用集成Q网络的均值）
        q_values = cur_agent.critic(actor_critic_input)
        q_value = q_values.mean(dim=1, keepdim=True)

        alpha = cur_agent.alpha_log.exp().detach()

        # ✅ Actor损失：最大化 Q(s,a) - α * log π(a|s)
        # 注意：这里不需要应用importance_weights，因为Actor是on-policy更新
        actor_loss = (alpha * new_logprob - q_value).mean()
        # 反向传播更新Actor
        actor_loss.backward()

        # # ✅ 可选：梯度裁剪
        torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), max_norm=1.0)

        cur_agent.actor_optimizer.step()

        # ========== 3. 更新温度参数 α ==========
        cur_agent.alpha_optimizer.zero_grad()

        # ✅ Alpha损失（标准SAC实现）
        alpha_loss = -(cur_agent.alpha_log * (new_logprob.detach() + cur_agent.target_entropy)).mean()

        alpha_loss.backward()
        cur_agent.alpha_optimizer.step()

        # ✅ 返回损失值和TD误差
        return (
            critic_loss.item(),
            actor_loss.item(),
            alpha.item(),
            alpha_loss.item(),
            td_errors  # 用于更新PER优先级
        )

    def update_all_targets(self):
        """✅ SAC每次都软更新目标网络（没有延迟）"""
        for agt in self.agents:
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


# ========== EnsembleCritic（精简版）==========
class EnsembleCritic(torch.nn.Module):
    """集成Critic网络"""

    def __init__(self, num_in, hidden_dim, num_ensembles=4):
        super(EnsembleCritic, self).__init__()
        self.num_ensembles = num_ensembles

        # 共享编码器
        self.encoder = torch.nn.Linear(num_in, hidden_dim)

        # 多个独立的Q网络
        self.q_networks = torch.nn.ModuleList()
        for i in range(num_ensembles):
            q_net = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            )
            self.q_networks.append(q_net)

    def forward(self, x):
        """
        输入：完整的critic输入(所有obs, 所有actions)
        输出：shape=[batch_size, num_ensembles] 的Q值
        """
        x = F.relu(self.encoder(x))
        q_values = [q_net(x) for q_net in self.q_networks]
        q_values = torch.cat(q_values, dim=1)
        return q_values


# ========== StochasticActor（高斯策略）==========
class StochasticActor(torch.nn.Module):
    """SAC的随机策略网络，输出高斯分布参数"""

    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super(StochasticActor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # 输出均值和log标准差，所以是 action_dim * 2
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim * 2)
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.ActionDist = torch.distributions.Normal

    def forward(self, state):
        """返回确定性动作（用于评估）"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # 只取均值部分
        a_avg = output[:, :self.action_dim]
        return torch.tanh(a_avg) * self.action_bound

    def get_action(self, state):
        """采样随机动作（用于探索）"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # 分离均值和log标准差
        a_avg, a_std_log = output.chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        # 构建高斯分布并采样
        dist = self.ActionDist(a_avg, a_std)
        action = dist.rsample()  # 重参数化采样
        action_tanh = torch.tanh(action) * self.action_bound

        return action_tanh

    def get_action_logprob(self, state):
        """返回动作及其对数概率（用于训练）"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        a_avg, a_std_log = output.chunk(2, dim=1)
        a_std = a_std_log.clamp(-16, 2).exp()

        # 采样动作
        dist = self.ActionDist(a_avg, a_std)
        action = dist.rsample()

        # 计算tanh后的动作
        action_tanh = torch.tanh(action)

        # 计算对数概率
        logprob = dist.log_prob(action)

        # ✅ tanh变换的雅可比修正
        logprob -= torch.log((1 - action_tanh.pow(2)) + 1e-6)
        logprob = logprob.sum(1, keepdim=True)

        action_tanh = action_tanh * self.action_bound

        return action_tanh, logprob


# ========== SAC智能体类 ==========
class SAC:
    """SAC算法（适配多智能体场景）"""

    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device, action_bound,
                 num_ensembles=4,
                 alpha_lr=3e-4,
                 target_entropy=None,
                 tau=0.005):

        # 随机策略Actor
        self.actor = StochasticActor(state_dim, action_dim, hidden_dim, action_bound).to(device)

        # 集成Critic（输入是全局观测+全局动作）
        self.critic = EnsembleCritic(critic_input_dim, hidden_dim, num_ensembles).to(device)
        self.target_critic = EnsembleCritic(critic_input_dim, hidden_dim, num_ensembles).to(device)

        # 初始化目标网络
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 自动调节的温度参数 α
        self.alpha_log = torch.tensor(np.log(0.1), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.alpha_log], lr=alpha_lr)

        # 目标熵
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy
        self.action_bound = action_bound
        self.device = device
        self.tau = tau

    def take_action(self, state, explore=True):
        """采取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if explore:
            action = self.actor.get_action(state_tensor)
        else:
            action = self.actor(state_tensor)

        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        """软更新目标网络"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)