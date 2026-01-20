import numpy as np


class PerceptionSystem:
    def __init__(self, raycast_num_directions, max_ray_distance, grid_size_x, grid_size_y, query_resolution=0.1,mask_fire_obs=False):
        self.raycast_num_directions = raycast_num_directions
        self.max_ray_distance = max_ray_distance
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.query_resolution = query_resolution
        self.FIRE_THRESHOLD = 0.4
        self.mask_fire_obs = mask_fire_obs

        # 缓存
        self.cached_wall_dists = {}
        self.cached_fire_dists = {}

    def reset_cache(self):
        self.cached_wall_dists = {}
        self.cached_fire_dists = {}

    def get_observations(self, agent, agent_pos, agent_vel, exits, fire_system, building_manager):
        """核心感知函数 (最终优化版：双向遮挡 + 读缓存)"""
        max_distance = np.sqrt(self.grid_size_x ** 2 + self.grid_size_y ** 2)

        # ==========================================
        # 1. 尝试读取缓存 (Hit Cache)
        # ==========================================
        # 如果当前步已经计算过，直接拿处理好的"可见距离"数据，省去射线计算
        if agent in self.cached_wall_dists and agent in self.cached_fire_dists:
            visible_wall_dists = self.cached_wall_dists[agent]
            visible_fire_dists = self.cached_fire_dists[agent]
        else:
            # ==========================================
            # 2. 缓存未命中：计算原始数据 (Miss Cache)
            # ==========================================
            # 获取【原始】数据
            raw_wall_dists = self._get_obstacle_distances_by_rays_analytical(
                agent_pos, self.raycast_num_directions, self.max_ray_distance, building_manager
            )

            raw_fire_dists = self._get_fire_distances_by_rays_vectorized(
                agent_pos, self.raycast_num_directions, self.max_ray_distance, fire_system
            )

            # ==========================================
            # 3. 处理双向遮挡 (Occlusion Logic)
            # ==========================================
            # A. 墙挡火: 墙比火近 -> 火不可见 (设为1.0)
            visible_fire_dists = np.where(raw_wall_dists < raw_fire_dists, 1.0, raw_fire_dists)

            # B. 火挡墙: 火比墙近 -> 墙不可见 (设为1.0) 【你的实验逻辑】
            # 这会让着火的墙在Observation中"消失"，导致没有墙壁斥力，主要靠火灾斥力起作用
            visible_wall_dists = np.where(raw_fire_dists < raw_wall_dists, 1.0, raw_wall_dists)

        # ==========================================
        # 4. 组装 Observation
        # ==========================================
        # 计算通用状态
        nearest_exit = min(exits, key=lambda exit: np.linalg.norm(agent_pos - exit))
        goal_relative_pos = (nearest_exit - agent_pos) / max_distance

        abs_pos = np.array([
            agent_pos[0] / self.grid_size_x,
            agent_pos[1] / self.grid_size_y
        ], dtype=np.float32)

        # 根据配置决定是否屏蔽火灾感知 (mask_fire_obs)
        if self.mask_fire_obs:
            # 【屏蔽模式】
            # 依然使用 visible_wall_dists，意味着虽然屏蔽了 danger_sensor，
            # 但智能体会发现某些墙"消失"了（被火遮挡），这是符合逻辑的副作用
            obs = np.concatenate([
                abs_pos,  # 2
                agent_vel,  # 2
                goal_relative_pos,  # 2
                visible_wall_dists  # N
            ]).astype(np.float32)
        else:
            # 【完整模式】
            current_temp = fire_system.get_env_factor_at_position(agent_pos)
            danger_sensor = np.array([1.0 - current_temp], dtype=np.float32)  # 1

            obs = np.concatenate([
                abs_pos,  # 2
                agent_vel,  # 2
                danger_sensor,  # 1
                goal_relative_pos,  # 2
                visible_wall_dists,  # N (包含被火挡墙的效果)
                visible_fire_dists  # N (包含被墙挡火的效果)
            ]).astype(np.float32)

        # 注意：这里我们不需要在函数内调用 update_cache
        # 因为 fire_evac_env.py 的 step 函数会拿到返回值后手动调用 update_cache
        # 从而确保下一帧(或 _get_observations 调用时)能命中上面的缓存逻辑
        return obs, visible_wall_dists, visible_fire_dists

    def update_cache(self, agent, wall_dists, fire_dists):
        self.cached_wall_dists[agent] = wall_dists
        self.cached_fire_dists[agent] = fire_dists

    def _get_fire_distances_by_rays_vectorized(self, agent_pos, num_directions, max_ray_length, fire_system):
        step_size = 0.2
        num_steps = int(max_ray_length / step_size)
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        steps = np.linspace(step_size, max_ray_length, num_steps)
        cos_sin = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        offsets = cos_sin[:, np.newaxis, :] * steps[np.newaxis, :, np.newaxis]
        check_coords = agent_pos + offsets

        map_indices = (check_coords / self.query_resolution).astype(np.int32)
        H, W = fire_system.high_res_env_factor_map.shape
        y_indices = np.clip(map_indices[..., 1], 0, H - 1)
        x_indices = np.clip(map_indices[..., 0], 0, W - 1)

        sampled_values = fire_system.high_res_env_factor_map[y_indices, x_indices]

        hit_mask = sampled_values < self.FIRE_THRESHOLD
        x_coords, y_coords = check_coords[..., 0], check_coords[..., 1]
        out_of_bounds = (x_coords < 0) | (x_coords > self.grid_size_x) | \
                        (y_coords < 0) | (y_coords > self.grid_size_y)
        final_hit_mask = hit_mask | out_of_bounds

        first_hit_indices = np.argmax(final_hit_mask, axis=1)
        row_indices = np.arange(num_directions)
        is_actually_hit = final_hit_mask[row_indices, first_hit_indices]

        normalized_distances = np.ones(num_directions, dtype=np.float32)
        hit_dist_values = steps[first_hit_indices] / max_ray_length
        normalized_distances[is_actually_hit] = hit_dist_values[is_actually_hit]

        return normalized_distances

    def _get_obstacle_distances_by_rays_analytical(self, agent_pos, num_directions, max_ray_length, building_manager):
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        min_distances = np.full(num_directions, max_ray_length, dtype=np.float32)

        if building_manager.static_lines is not None:
            static_dists = self._ray_lines_intersection_fast(
                agent_pos, directions,
                building_manager.static_line_starts,
                building_manager.static_line_vecs
            )
            min_distances = np.minimum(min_distances, static_dists)

        return np.minimum(min_distances / max_ray_length, 1.0)

    def _ray_lines_intersection_fast(self, ray_start, ray_dirs, line_starts, line_vecs):
        denominator = (ray_dirs[:, 0:1] * line_vecs[np.newaxis, :, 1] -
                       ray_dirs[:, 1:2] * line_vecs[np.newaxis, :, 0])
        valid_denom = np.abs(denominator) > 1e-10
        diff = line_starts[np.newaxis, :, :] - ray_start[np.newaxis, np.newaxis, :]

        t_numer = (diff[:, :, 0] * line_vecs[np.newaxis, :, 1] -
                   diff[:, :, 1] * line_vecs[np.newaxis, :, 0])
        s_numer = (diff[:, :, 0] * ray_dirs[:, np.newaxis, 1] -
                   diff[:, :, 1] * ray_dirs[:, np.newaxis, 0])

        t = np.divide(t_numer, denominator, where=valid_denom, out=np.full_like(denominator, np.inf))
        s = np.divide(s_numer, denominator, where=valid_denom, out=np.full_like(denominator, -1.0))

        is_valid = (t >= 0) & (s >= 0) & (s <= 1)
        final_t = np.where(is_valid, t, np.inf)
        return np.min(final_t, axis=1)