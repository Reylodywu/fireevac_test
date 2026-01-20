import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing import Tuple, List, Optional

# 导入组件
from .components.building import BuildingManager
from .components.fire_system import FireSystem
from .components.perception import PerceptionSystem
from .components.navigation import NavigationSystem
from .components.reward import RewardCalculator
from .components.visualizer import Visualizer


class FireEvacuationParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "fire_evacuation_v1"}

    def __init__(self,
                 n_agents: int = 3,
                 grid_size_x: int = 20,
                 grid_size_y: int = 15,
                 fire_source: Tuple[int, int] = (55, 20),
                 agent_dt: float = 0.5,
                 action_bound: float = None,
                 max_speed: float = 1.2,
                 max_steps: int = 200,
                 target_reward: float = 200.0,
                 raycast_num_directions: int = 60,
                 max_ray_distance: float = 25.0,
                 render_mode: Optional[str] = None,
                 fds_data_path: str = 'fire_env.npz',
                 fds_time_step: int = 330,
                 fixed_fire: bool = True,
                 fds_dt: float = 1.0,
                 mask_fire_obs: bool = False):
        super().__init__()

        # 参数
        self.n_agents = n_agents
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.agent_dt = agent_dt
        self.action_bound = action_bound
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.raycast_num_directions = raycast_num_directions
        self.max_ray_distance = max_ray_distance
        self.mask_fire_obs = mask_fire_obs

        # 1. 初始化组件
        self.building = BuildingManager(grid_size_x, grid_size_y)

        self.fire_system = FireSystem(fds_data_path, grid_size_x, grid_size_y, fire_source, fds_dt, fds_time_step,fixed_fire)
        # FDS 更新逻辑
        # 每 2 个 agent step，更新 1 次 fire step
        self.steps_per_fds = int(fds_dt / agent_dt) if self.fire_system.fds_data else 0

        self.navigation = NavigationSystem(grid_size_x, grid_size_y, self.building.exits, self.building)

        self.perception = PerceptionSystem(raycast_num_directions, self.max_ray_distance, grid_size_x, grid_size_y,mask_fire_obs = self.mask_fire_obs)

        self.reward_calculator = RewardCalculator(
            reward_weights={'exit': 1.0, 'environment': 1.0, 'crowding': 1.0, 'time': 1.0,
                            'collision': 1.0, 'smoothness': 1.0, 'movement': 5.0, 'open': 1.0},
            target_reward=target_reward,
            max_speed=max_speed,
            max_ray_distance=max_ray_distance,
            exits=self.building.exits,
            raycast_num_directions=raycast_num_directions
        )

        self.visualizer = Visualizer(grid_size_x, grid_size_y, render_mode)

        # PettingZoo 属性
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]

        self._init_spaces()

        # 状态容器
        self.agent_positions = {}
        self.agent_velocities = {}
        self.agent_paths = {}
        self.agent_dangers = {}
        self.previous_pathfinding_distance = {}
        self.agent_path_length = {}
        self.agent_rewards = {}
        self.current_step = 0

    def _init_spaces(self):
        self._action_spaces = {
            agent: spaces.Box(low=-self.action_bound, high=self.action_bound, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        N = self.raycast_num_directions

        if self.mask_fire_obs:
            # 修改: 如果屏蔽火灾感知
            # Obs: [abs_pos(2), vel(2), goal(2), wall(N)] -> 6 + N
            # 移除了 danger(1) 和 fire(N)
            obs_dim = 6 + N
            print(f"⚠️ Warning: Fire Perception Masked! Obs dim: {obs_dim}")
        else:
            # 原有逻辑: 包含火灾信息
            # Obs: [abs_pos(2), vel(2), danger(1), goal(2), wall(N), fire(N)] -> 7 + 2N
            obs_dim = 7 + 2 * N

        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.fire_system.reset()
        self.perception.reset_cache()

        # 初始化位置
        zone_config1 = [
            {'x_range': (1, self.grid_size_x/2), 'y_range': (1, self.grid_size_y - 1), 'probability': 1.0,'name': 'zone1'}
        ]
        zone_config2 = [
            {'x_range': (40, 50), 'y_range': (10,30), 'probability': 1.0, 'name': 'zone2'}
        ]
        self.agent_positions = self.building.initialize_positions_configurable(self.agents, zone_config2,
                                                                               self.fire_system)

        # 初始化状态
        self.agent_velocities = {agent: np.zeros(2, dtype=np.float32) for agent in self.agents}
        self.agent_paths = {agent: [self.agent_positions[agent].copy()] for agent in self.agents}
        self.agent_dangers = {agent: [1.0-float(self.fire_system.get_env_factor_at_position(self.agent_positions[agent]))] for agent in self.agents}
        self.agent_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.previous_pathfinding_distance = {
            agent: self.navigation.get_pathfinding_distance(self.agent_positions[agent])
            for agent in self.agents
        }
        self.agent_path_length = {agent: 0.0 for agent in self.agents}

        return self._get_observations(), self.infos

    def step(self, actions):
        rewards = {}
        self.perception.reset_cache()  # 每步清空感知缓存

        for agent in self.agents:
            if agent not in actions or self.terminations[agent]:
                rewards[agent] = 0.0
                continue

            # 1. 移动逻辑
            normalized_action = actions[agent]
            self.agent_velocities[agent] = normalized_action
            movement = normalized_action * self.max_speed
            old_pos = self.agent_positions[agent].copy()
            target_pos = old_pos + movement

            old_path_dist = self.previous_pathfinding_distance[agent]

            # 2. 碰撞检测
            # A. 静态点检测（防止终点正好落在障碍物内部）
            is_static_wall = not self.building.is_valid_position(target_pos)
            # B. 射线检测（防止穿墙）
            # 调用我们在 BuildingManager 中新加的函数
            is_ray_hit = self.building.check_segment_collision(old_pos, target_pos)
            # 综合判断：如果是墙（终点无效 或 路径碰撞），则回退
            is_wall = is_static_wall or is_ray_hit
            # C. 火灾检测
            env_factor = self.fire_system.get_env_factor_at_position(target_pos)
            is_fire = env_factor < 0.3  # Forbidden threshold

            collision_occurred = False
            entered_forbidden = False

            if is_wall:
                self.agent_positions[agent] = old_pos
                collision_occurred = True
                current_path_dist = old_path_dist
            elif is_fire:
                self.agent_positions[agent] = old_pos
                entered_forbidden = True
                current_path_dist = old_path_dist
                self.infos[agent]['reason'] = 'scorched'
            else:
                self.agent_positions[agent] = target_pos
                actual_dist = np.linalg.norm(target_pos - old_pos)
                self.agent_path_length[agent] += actual_dist
                current_path_dist = self.navigation.get_pathfinding_distance(target_pos)

            self.previous_pathfinding_distance[agent] = current_path_dist
            distance_change = old_path_dist - current_path_dist
            self.agent_paths[agent].append(self.agent_positions[agent].copy())

            # 存储每一步risk以评估路线安全性
            env_factor = self.fire_system.get_env_factor_at_position(self.agent_positions[agent])
            current_danger = 1.0 - float(env_factor)
            self.agent_dangers[agent].append(current_danger)

            # 3. 感知与奖励计算 (需要先获取 Observation 填充缓存)
            # 这里稍微有些Trick: 为了算Reward我们需要Perception的结果
            # 所以我们显式调用 perception 获取数据，放入缓存
            _, wall_dists, fire_dists = self.perception.get_observations(
                agent, self.agent_positions[agent], self.agent_velocities[agent],
                self.building.exits, self.fire_system, self.building
            )
            self.perception.update_cache(agent, wall_dists, fire_dists)

            reward = self.reward_calculator.calculate_reward_smooth(
                agent, self.agent_positions[agent], distance_change, collision_occurred, entered_forbidden,
                fire_dists, wall_dists, self.agent_velocities[agent],
                self.fire_system, self.building
            )

            # 加上平滑惩罚
            smoothness = self.reward_calculator.calculate_smoothness_penalty(self.agent_paths[agent],self.agent_positions[agent])
            reward += smoothness

            rewards[agent] = reward
            self.agent_rewards[agent] += reward

            # 4. 终止条件
            if self.building.is_at_exit(self.agent_positions[agent]):
                self.terminations[agent] = True
                self.infos[agent]['reason'] = 'success'

        self.current_step += 1
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.truncations[agent] = True

        # FDS 更新
        if self.steps_per_fds > 0 and self.current_step % self.steps_per_fds == 0:
            # step_increment 设为 1，代表 FDS 数据向前走一帧（即走过 1.0s）
            self.fire_system.update(step_increment=1)

        return self._get_observations(), rewards, self.terminations, self.truncations, self.infos

    def _get_observations(self):
        obs = {}
        for agent in self.agents:
            if not self.terminations[agent]:
                # 这一步会利用缓存 (如果step里已经算过)
                o, _, _ = self.perception.get_observations(
                    agent, self.agent_positions[agent], self.agent_velocities[agent],
                    self.building.exits, self.fire_system, self.building
                )
                obs[agent] = o
            else:
                # 结束后的 dummy observation
                obs[agent] = np.zeros(self.observation_space(agent).shape, dtype=np.float32)
        return obs

    def render(self):
        return self.visualizer.render(
            self.agent_positions, self.agent_paths, self.building.exits,
            self.fire_system, self.building, self.current_step
        )

    def close(self):
        self.visualizer.close()

    def evaluate_trajectory_risk(self):
        """
        评估路径风险并计算综合评分
        直接在 Env 内部完成计算，减轻外部负担
        """
        stats = {}
        for agent in self.agents:
            # 1. 获取基础数据
            dangers = np.array(self.agent_dangers[agent])
            paths = np.array(self.agent_paths[agent])

            # 防止空数据报错
            if len(dangers) == 0:
                stats[agent] = {"safety_score": 0.0, "total_exposure": 0.0, "max_danger": 0.0, "path_length": 0.0}
                continue

            # 2. 计算核心指标
            total_exposure = np.sum(dangers)
            max_danger = np.max(dangers)
            avg_danger = np.mean(dangers)

            # 计算路径长度 (欧几里得距离累加)
            # paths shape: (N, 2) -> diff shape: (N-1, 2)
            if len(paths) > 1:
                step_distances = np.linalg.norm(paths[1:] - paths[:-1], axis=1)
                path_length = np.sum(step_distances)
            else:
                path_length = 0.0

            # 3. 计算综合安全评分 (Safety Score) 0~100分
            # 公式设计逻辑：
            # - 基准分 100
            # - 扣分项 1: 累积伤害 (total_exposure)。权重 1.0。比如在火里待了 20步(danger=1.0)，扣20分。
            # - 扣分项 2: 瞬间峰值 (max_danger)。权重 40.0。只要瞬间摸到 danger=1.0，直接扣 40分 (不及格)。
            base_score = 100.0
            penalty_exposure = total_exposure * 1.0
            penalty_peak = max_danger * 40.0

            score = base_score - penalty_exposure - penalty_peak
            score = max(0.0, score)  # 兜底，不出现负分

            # 4. 打包返回
            stats[agent] = {
                "safety_score": float(score),
                "total_exposure": float(total_exposure),
                "max_danger": float(max_danger),
                "avg_danger": float(avg_danger),
                "path_length": float(path_length),
                "step_count": len(dangers)
            }
        return stats