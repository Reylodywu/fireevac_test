import numpy as np


class RewardCalculator:
    def __init__(self, reward_weights, target_reward, max_speed, max_ray_distance, exits,
                 raycast_num_directions):
        self.reward_weights = reward_weights
        self.target_reward = target_reward
        self.max_speed = max_speed
        self.max_ray_distance = max_ray_distance
        self.exits = exits
        self.raycast_num_directions = raycast_num_directions

        self.AGENT_RADIUS = 0.3  # 需与Building一致

    def calculate_reward_smooth(self, agent, pos, distance_change, collision, entered_forbidden,
                                fire_dists, wall_dists, velocity_vector, fire_system, building_manager):
        """主奖励函数 (方案二：指数势能场版)"""
        # 1. 终点奖励
        exit_reward = self.target_reward if building_manager.is_at_exit(pos) else 0.0
        # 2. 状态判定
        min_fire_dist = self.max_ray_distance
        if fire_dists is not None:
            min_fire_dist = np.min(fire_dists) * self.max_ray_distance

        nearest_exit = min(self.exits, key=lambda e: np.linalg.norm(pos - e))
        dist_to_exit = np.linalg.norm(pos - nearest_exit)

        # === 指数级势能场 (Exponential Potential Field) ===

        # [参数设定]
        EXIT_OVERRIDE_DIST = 8.0
        PROXIMITY_WEIGHT = 0.2

        # sigma 控制衰减速度 ("坡度"的缓急)
        # 建议设为 HAZARD_RADIUS / 2.0。例如 HAZARD=4.0, 则 sigma=2.0
        # 效果：在 2米处强度显著，在 4米处衰减到 13%，在 6米处几乎为 0
        sigma = 4

        # A. 计算高斯危险强度 (范围 0.0 ~ 1.0)
        # 这是一个光滑曲线，解决了"断崖"问题
        danger_intensity = np.exp(-(min_fire_dist ** 2) / (2 * sigma ** 2))

        # B. 计算势能场惩罚
        # 离火越近，惩罚越接近 -2.0；离火越远，惩罚平滑趋近于 0
        fire_proximity_penalty = -PROXIMITY_WEIGHT * danger_intensity

        # C. 动态调整 Trust Factor (平滑过渡)
        # 定义：极度危险(intensity=1) -> trust=0.2 (保留一点动力防止死锁)
        #      完全安全(intensity=0) -> trust=1.0
        # 公式：y = 1.0 - 0.8 * x
        trust_factor = 1.0 - (0.7 * danger_intensity)

        # D. Open Space 权重与危险程度成正比
        open_space_weight = danger_intensity

        # # E. 冲刺逻辑 (Override)
        # # 如果离出口很近(或者满足之前的视线确认逻辑)，强制满信任度
        # if dist_to_exit < EXIT_OVERRIDE_DIST:
        #     trust_factor = 1.0
        #     open_space_weight = 0.0  # 冲刺时不需要空旷引导，只需盯着出口
        #
        #     # [可选优化] 冲刺时是否减免火灾心理压力？
        #     # 建议保留 fire_proximity_penalty (物理热量还在)，但因为 trust=1.0，
        #     # 巨大的 movement_reward 会压倒这个惩罚

        # === 修改结束 ===

        # 3. 移动奖励
        raw_movement_reward = (distance_change / (self.max_speed * np.sqrt(2))) * self.reward_weights['movement']
        movement_reward = raw_movement_reward * trust_factor

        # 4. fire_proximity_penalty

        # 5. 空旷引导
        open_space_reward = 0.0
        if open_space_weight > 0.01:
            # print(f'open_space_weight: {open_space_weight}')
            raw_open_space = self._calculate_guided_open_space_reward_origin(pos, velocity_vector, fire_dists, wall_dists)
            open_space_reward = raw_open_space * self.reward_weights['open'] * open_space_weight

        # 6. 其他
        time_penalty = -0.5 * self.reward_weights['time']

        env_factor = fire_system.get_env_factor_at_position(pos)
        temp_danger = np.log(1.0 / (env_factor + 1e-6))
        env_penalty = -temp_danger * self.reward_weights['environment']

        # wall_penalty = self._calculate_wall_soft_penalty(wall_dists)
        wall_penalty = 0  # 闸机处较窄，暂不考虑临近惩罚避免导致震荡
        hard_collision = -10.0 if collision else 0.0
        collision_penalty = wall_penalty + hard_collision
        forbidden_penalty = -20.0 if entered_forbidden else 0.0

        total_reward = (exit_reward + movement_reward + time_penalty +
                        collision_penalty + forbidden_penalty + fire_proximity_penalty +
                        env_penalty + open_space_reward)

        # print(f'min_fire_dist:{min_fire_dist}')
        # print(f'dist_to_exit:{dist_to_exit:.2f} | dist_to_fire:{min_fire_dist} | trust_factor:{trust_factor:.2f} | open_space_weight:{open_space_weight:.2f}')
        # print(f'exit_reward: {exit_reward} | movement_reward: {movement_reward:.2f} | wall_penalty:{wall_penalty}'
        #       f' open_space_reward: {open_space_reward:.2f}|fire_proximity_penalty: {fire_proximity_penalty} |total_reward: {total_reward:.2f}')
        if total_reward > 0:
            scaled_reward = np.log(1.0 + total_reward)
        else:
            scaled_reward = -np.log(1.0 + abs(total_reward))
        return scaled_reward

    def calculate_smoothness_penalty(self, path_history, current_pos):
        if len(path_history) < 2: return 0.0
        p_prev_2 = path_history[-2]
        p_prev_1 = path_history[-1]
        p_curr = current_pos
        v_last = p_prev_1 - p_prev_2
        v_curr = p_curr - p_prev_1
        diff_norm = np.linalg.norm(v_curr - v_last)
        return -diff_norm * 1.0 * self.reward_weights['smoothness']

    def _calculate_wall_soft_penalty(self, wall_dists):
        if wall_dists is None: return 0.0
        min_dist = np.min(wall_dists) * self.max_ray_distance
        surface_dist = max(0.0, min_dist - self.AGENT_RADIUS)
        # d_m为警戒半径
        d_m = 1.0
        r_obstacle = -1.0
        if surface_dist < d_m:
            return r_obstacle * (d_m - surface_dist)
        return 0.0

    def _calculate_guided_open_space_reward_origin(self, current_pos, velocity_vec, fire_dists, wall_dists):
        speed = np.linalg.norm(velocity_vec)
        if speed <= 1e-4: return 0.0
        current_move_dir = velocity_vec / speed

        final_dists = np.minimum(fire_dists, wall_dists)

        angles = np.linspace(0, 2 * np.pi, self.raycast_num_directions, endpoint=False)
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        nearest_exit = min(self.exits, key=lambda e: np.linalg.norm(current_pos - e))
        to_exit_vec = nearest_exit - current_pos
        dist_to_exit = np.linalg.norm(to_exit_vec)
        if dist_to_exit < 1e-5: return 0.0
        to_exit_dir = to_exit_vec / dist_to_exit

        alignments = np.dot(ray_dirs, to_exit_dir)
        normalized_align = (alignments + 1.0) / 2.0
        # lambda_weight越大，代表背向出口的方向得分越低
        lambda_weight = 0.75
        dir_factor = (1.0 - lambda_weight) + lambda_weight * normalized_align

        scores = final_dists * dir_factor
        best_idx = np.argmax(scores)
        # print(final_dists)
        # print(best_idx)
        best_dir = ray_dirs[best_idx]
        # # 打印出最佳方向向量
        # print(best_dir)
        match_score = np.dot(current_move_dir, best_dir)
        return max(0.0, match_score)

    def _calculate_guided_open_space_reward(self, current_pos, velocity_vec, fire_dists, wall_dists):
        # --- 0. 基础计算 ---
        speed = np.linalg.norm(velocity_vec)
        if speed <= 1e-4: return 0.0
        current_move_dir = velocity_vec / speed

        # 获取真实距离用于判断
        min_fire_real = np.min(fire_dists) * self.max_ray_distance
        min_wall_real = np.min(wall_dists) * self.max_ray_distance

        # --- 1. 威胁判定 (Is Fire Threatening?) ---
        # 逻辑：同时满足 "进入警戒圈" 和 "火比墙近"
        is_fire_mode = min_fire_real < min_wall_real

        # --- 2. 动态选择分数基准 & 衰减系数 ---
        if is_fire_mode:
            # 【避火模式】
            # 关注点：离火越远越好
            # 系数：1.0 (全额奖励，鼓励立刻逃离)
            base_scores = fire_dists
            scale_factor = 1.0
            # print("Mode: Fire Escape")
        else:
            # 【导航模式】 (闸机、走廊、远离火源)
            # 关注点：离墙越远越好 (保持通道中心)
            # 系数：使用比值 decay。
            # 逻辑：在闸机处，wall(0.5) < fire(5.0)，ratio = 0.1。奖励会被压缩得很小，防止刷分。
            #       在大厅处，wall(5.0) ~ fire(5.0)，ratio ≈ 1.0。奖励恢复正常。
            base_scores = wall_dists

            # 加上一个小 epsilon 防止除零
            ratio = min_wall_real / (min_fire_real + 1e-5)
            scale_factor = np.clip(ratio, 0.0, 1.0)
            # print(f"Mode: Navigation | Decay Ratio: {scale_factor:.2f}")

        # --- 3. 方向引导 (保留原来的逻辑，防止倒车) ---
        angles = np.linspace(0, 2 * np.pi, self.raycast_num_directions, endpoint=False)
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        nearest_exit = min(self.exits, key=lambda e: np.linalg.norm(current_pos - e))
        to_exit_vec = nearest_exit - current_pos
        dist_to_exit = np.linalg.norm(to_exit_vec)
        if dist_to_exit < 1e-5: return 0.0
        to_exit_dir = to_exit_vec / dist_to_exit

        alignments = np.dot(ray_dirs, to_exit_dir)
        normalized_align = (alignments + 1.0) / 2.0

        # 权重保持 0.75，兼顾方向和空间
        lambda_weight = 0.75
        dir_factor = (1.0 - lambda_weight) + lambda_weight * normalized_align

        # --- 4. 最终计算 ---
        # 并没有物理掩码，而是靠 scores 本身的数值大小来引导
        # 如果某方向墙很近，base_scores (wall_dists) 就会很小，自然得分低
        final_scores = base_scores * dir_factor

        best_idx = np.argmax(final_scores)
        best_dir = ray_dirs[best_idx]

        match_score = np.dot(current_move_dir, best_dir)

        # 最后乘上衰减系数
        return max(0.0, match_score) * scale_factor

