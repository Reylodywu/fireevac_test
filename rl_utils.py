from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from collections import deque
import numpy as np
from collections import deque


class SumTree:
    """
    SumTree数据结构用于高效的优先级采样
    - 叶子节点存储优先级
    - 父节点存储子节点优先级之和
    - O(log n) 时间复杂度进行采样和更新
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 存储优先级的完全二叉树
        self.data = np.zeros(capacity, dtype=object)  # 存储经验数据
        self.write = 0  # 当前写入位置
        self.n_entries = 0  # 当前存储的经验数量

    def _propagate(self, idx, change):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """从根节点开始检索，找到累积优先级为s的叶子节点"""
        left = 2 * idx + 1
        right = left + 1

        # 到达叶子节点
        if left >= len(self.tree):
            return idx

        # 在左子树中查找
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # 在右子树中查找
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """返回根节点的优先级总和"""
        return self.tree[0]

    def add(self, priority, data):
        """添加新的经验"""
        idx = self.write + self.capacity - 1  # 叶子节点索引

        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        """更新叶子节点的优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """根据累积优先级值s获取经验"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """
    基于SumTree的优先经验回放缓冲区
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=uniform, 1=full prioritization)
            beta_start: 重要性采样初始值
            beta_frames: beta退火的总帧数
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        # 用于数值稳定性
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def _get_priority(self, td_error):
        """将TD误差转换为优先级"""
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def beta_by_frame(self, frame_idx):
        """Beta线性退火"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, state, action, reward, next_state, done):
        """添加经验，使用最大优先级"""
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        """优先级采样"""
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        # 计算当前beta值
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # 分段采样
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, priority, data = self.tree.get(s)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probabilities) ** (-beta)
        weights /= weights.max()  # 归一化

        # 解包经验
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        """返回当前缓冲区大小"""
        return self.tree.n_entries

class PrioritizedReplayBufferWithSave(PrioritizedReplayBuffer):
    """
    支持保存和加载的PER缓冲区
    """

    def get_state(self):
        """获取缓冲区状态用于保存"""
        return {
            'tree_data': self.tree.data[:self.tree.n_entries].copy(),
            'tree_priorities': self.tree.tree.copy(),
            'write_pos': self.tree.write,
            'n_entries': self.tree.n_entries,
            'max_priority': self.max_priority,
            'frame': self.frame,
        }

    def load_state(self, state):
        """从保存的状态恢复缓冲区"""
        self.tree.data[:state['n_entries']] = state['tree_data']
        self.tree.tree = state['tree_priorities']
        self.tree.write = state['write_pos']
        self.tree.n_entries = state['n_entries']
        self.max_priority = state['max_priority']
        self.frame = state['frame']
        print(f"✓ PER缓冲区已恢复 (size: {self.tree.n_entries}, frame: {self.frame})")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        # 在这里添加调试代码
        # print("--- Debugging Replay Buffer Sample ---")
        # print(state[0])  # 打印前两个状态样本
        return state, action, reward, next_state, done

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                