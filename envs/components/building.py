import numpy as np

class BuildingManager:
    def __init__(self, grid_size_x, grid_size_y,agent_radius=0.3):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.AGENT_RADIUS = agent_radius
        self.EXIT_RADIUS = 0.5

        self.walls = []
        self.solid_obstacles = []
        self.static_lines = None
        self.static_line_starts = None
        self.static_line_vecs = None
        self.walls_numpy = []

        # 初始化结构
        self._generate_building_structure_subway()
        self._build_static_geometry()

    def _generate_building_structure_enhance(self):
        """生成建筑结构 - 简化版机场航站楼"""
        self.walls = []
        self.solid_obstacles = []

        # ==================== 外墙边界 ====================
        self.walls.append(((0, 0), (self.grid_size_x, 0)))
        self.walls.append(((0, self.grid_size_y), (self.grid_size_x, self.grid_size_y)))
        self.walls.append(((0, 0), (0, self.grid_size_y)))
        self.walls.append(((self.grid_size_x, 0), (self.grid_size_x, self.grid_size_y)))

        # ==================== 内部矩形障碍物 ====================
        # 左侧：安检区
        self.solid_obstacles.append({'x_min': 5, 'y_min': 8, 'x_max': 11, 'y_max': 16})
        self.solid_obstacles.append({'x_min': 5, 'y_min': 20, 'x_max': 11, 'y_max': 28})
        # 中央：主候机区
        self.solid_obstacles.append({'x_min': 18, 'y_min': 22, 'x_max': 29, 'y_max': 31})
        self.solid_obstacles.append({'x_min': 18, 'y_min': 5, 'x_max': 29, 'y_max': 14})
        self.solid_obstacles.append({'x_min': 33, 'y_min': 22, 'x_max': 44, 'y_max': 31})
        self.solid_obstacles.append({'x_min': 33, 'y_min': 5, 'x_max': 44, 'y_max': 14})
        # 右侧：商业服务区
        self.solid_obstacles.append({'x_min': 46, 'y_min': 8, 'x_max': 49, 'y_max': 15})
        self.solid_obstacles.append({'x_min': 46, 'y_min': 21, 'x_max': 49, 'y_max': 28})

        # ==================== 中央通道分隔墙 ====================
        self.walls.append(((12, 17), (31, 17)))
        self.walls.append(((35, 17), (45, 17)))

        # 在出口位置开门
        self._create_exits()

        # 预转换为numpy数组
        self.walls_numpy = []
        for wall_start, wall_end in self.walls:
            self.walls_numpy.append((
                np.array(wall_start, dtype=np.float64),
                np.array(wall_end, dtype=np.float64)
            ))


    def _generate_building_structure_subway(self):
        """
        生成建筑结构 - 进阶地铁站厅层 (逻辑优化版)
        修改逻辑：
        1. 采用 "墙体 - 孔洞" (Wall minus Holes) 的生成逻辑。
        2. 先定义完整的四面付费区围墙。
        3. 计算闸机生成的实际范围，作为"孔洞"。
        4. 自动切割墙体，并在孔洞处填充闸机障碍物。
        """
        self.walls = []
        self.solid_obstacles = []

        # ==================== 1. 基础参数定义 ====================
        grid_x = self.grid_size_x
        grid_y = self.grid_size_y
        center_x, center_y = grid_x / 2, grid_y / 2

        # 付费区尺寸
        paid_zone_width = grid_x * 0.7
        paid_zone_height = grid_y * 0.6

        # 付费区四条边界 (围墙的基准线)
        paid_x_min = center_x - paid_zone_width / 2
        paid_x_max = center_x + paid_zone_width / 2
        paid_y_min = center_y - paid_zone_height / 2
        paid_y_max = center_y + paid_zone_height / 2

        # 辅助函数：添加矩形障碍物
        def add_obstacle(x1, y1, x2, y2):
            self.solid_obstacles.append({
                'x_min': min(x1, x2), 'y_min': min(y1, y2),
                'x_max': max(x1, x2), 'y_max': max(y1, y2)
            })

        # ==================== 2. 外墙与出口 ====================
        self.walls.extend([
            ((0, 0), (grid_x, 0)), ((0, grid_y), (grid_x, grid_y)),
            ((0, 0), (0, grid_y)), ((grid_x, 0), (grid_x, grid_y))
        ])
        margin = 3
        self.exits = [(grid_x - margin, grid_y - margin)]

        # ==================== 3. 内部设施 (楼梯/客服中心) ====================

        # --- B. 楼梯 (Stairs) ---
        stair_width = 6
        stair_height = 3
        stair_pad_x = 5
        stair_pad_y = 0

        stair_x_positions = [
            paid_x_min + stair_pad_x,
            center_x - stair_width / 2,
            paid_x_max - stair_pad_x - stair_width
        ]
        stair_y_positions = [
            paid_y_min + stair_pad_y,
            paid_y_max - stair_pad_y - stair_height
        ]

        for x in stair_x_positions:
            for y in stair_y_positions:
                add_obstacle(x, y, x + stair_width, y + stair_height)

        # --- C. 客服中心 ---
        service_size = 4
        service_margin = 2
        add_obstacle(center_x - service_size / 2, paid_y_min - service_margin - service_size,
                     center_x + service_size / 2, paid_y_min - service_margin)
        add_obstacle(center_x - service_size / 2, paid_y_max + service_margin,
                     center_x + service_size / 2, paid_y_max + service_margin + service_size)

        # ==================== 4. 闸机生成与墙体开孔 (核心修改) ====================

        gate_len = 2.0
        gate_thick = 0.5

        # 用字典存储四面墙上需要"挖掉"的区域
        # key: 'left', 'right', 'top', 'bottom'
        # value: list of tuples [(start, end), ...]
        wall_holes = {'left': [], 'right': [], 'top': [], 'bottom': []}

        def add_gate_array_and_record_hole(side, fixed_axis, start_pos, end_pos, num_gates):
            """
            生成闸机障碍物，并计算精确的开孔位置记录到 wall_holes 中。
            side: 'left', 'right' (Vertical) 或 'top', 'bottom' (Horizontal)
            """
            is_vertical = side in ['left', 'right']
            total_span = end_pos - start_pos
            unit_size = total_span / num_gates

            # 1. 生成闸机障碍物
            for i in range(num_gates):
                base = start_pos + i * unit_size
                obs_start = base + (unit_size * 0.2)
                obs_end = obs_start + gate_thick

                if is_vertical:
                    add_obstacle(fixed_axis - gate_len / 2, obs_start, fixed_axis + gate_len / 2, obs_end)
                else:
                    add_obstacle(obs_start, fixed_axis - gate_len / 2, obs_end, fixed_axis + gate_len / 2)

            # 2. 计算墙体需要挖掉的精确范围 (从第一台闸机起点 到 最后一台闸机终点)
            hole_start = start_pos + (unit_size * 0.2)
            hole_end = start_pos + (num_gates - 1) * unit_size + (unit_size * 0.2) + gate_thick

            wall_holes[side].append((hole_start, hole_end))

        # --- 配置闸机位置 ---
        v_gate_margin = (paid_y_max - paid_y_min) * 0.2
        gap = 2

        # 水平闸机位置 (避开楼梯)
        h_gate_left_start = paid_x_min + stair_pad_x + stair_width + gap
        h_gate_left_end = center_x - stair_width / 2 - gap
        h_gate_right_start = center_x + stair_width / 2 + gap
        h_gate_right_end = paid_x_max - stair_pad_x - stair_width - gap

        # 生成闸机 (障碍物) 并 记录孔洞
        # 1. 垂直方向
        add_gate_array_and_record_hole('left', paid_x_min, paid_y_min + v_gate_margin, paid_y_max - v_gate_margin, 5)
        add_gate_array_and_record_hole('right', paid_x_max, paid_y_min + v_gate_margin, paid_y_max - v_gate_margin, 5)

        # # 2. 水平方向
        # # 下边 (分左右两段)
        # add_gate_array_and_record_hole('bottom', paid_y_min, h_gate_left_start, h_gate_left_end, 4)
        # add_gate_array_and_record_hole('bottom', paid_y_min, h_gate_right_start, h_gate_right_end, 4)
        # # 上边 (分左右两段)
        # add_gate_array_and_record_hole('top', paid_y_max, h_gate_left_start, h_gate_left_end, 4)
        # add_gate_array_and_record_hole('top', paid_y_max, h_gate_right_start, h_gate_right_end, 4)

        # ==================== 5. 自动生成剩余墙体 (Wall Generator) ====================

        def generate_walls_with_holes(fixed_axis, start_limit, end_limit, holes, is_vertical):
            """
            根据给定的全长范围和孔洞列表，生成剩余的墙体线段。
            """
            # 对孔洞按位置排序
            holes.sort(key=lambda x: x[0])

            current_pos = start_limit

            for hole_start, hole_end in holes:
                # 如果当前位置在孔洞开始之前，说明中间有墙
                if current_pos < hole_start:
                    if is_vertical:
                        self.walls.append(((fixed_axis, current_pos), (fixed_axis, hole_start)))
                    else:
                        self.walls.append(((current_pos, fixed_axis), (hole_start, fixed_axis)))

                # 跳过孔洞，更新当前位置到孔洞结束处
                current_pos = max(current_pos, hole_end)

            # 处理最后一个孔洞到终点的墙
            if current_pos < end_limit:
                if is_vertical:
                    self.walls.append(((fixed_axis, current_pos), (fixed_axis, end_limit)))
                else:
                    self.walls.append(((current_pos, fixed_axis), (end_limit, fixed_axis)))

        # 分别生成四面墙
        generate_walls_with_holes(paid_x_min, paid_y_min, paid_y_max, wall_holes['left'], is_vertical=True)  # 左墙
        generate_walls_with_holes(paid_x_max, paid_y_min, paid_y_max, wall_holes['right'], is_vertical=True)  # 右墙
        generate_walls_with_holes(paid_y_min, paid_x_min, paid_x_max, wall_holes['bottom'], is_vertical=False)  # 下墙
        generate_walls_with_holes(paid_y_max, paid_x_min, paid_x_max, wall_holes['top'], is_vertical=False)  # 上墙

        # ==================== 6. [全局] 结构柱 ====================
        column_size = 1.5
        column_offset_y = 3.0
        num_total_columns = 10
        map_start_x = grid_x * 0.05
        map_end_x = grid_x * 0.95
        column_x_positions = np.linspace(map_start_x, map_end_x, num_total_columns)

        def is_position_clear(cx, cy, size):
            margin = 0
            c_min_x, c_max_x = cx - size / 2 - margin, cx + size / 2 + margin
            c_min_y, c_max_y = cy - size / 2 - margin, cy + size / 2 + margin
            for obs in self.solid_obstacles:
                if not (c_max_x < obs['x_min'] or c_min_x > obs['x_max'] or
                        c_max_y < obs['y_min'] or c_min_y > obs['y_max']):
                    return False
            return True

        for col_x in column_x_positions:
            col_y_top = center_y + column_offset_y
            if is_position_clear(col_x, col_y_top, column_size):
                if abs(col_x - paid_x_min) > 1.5 and abs(col_x - paid_x_max) > 1.5:
                    add_obstacle(col_x - column_size / 2, col_y_top,
                                 col_x + column_size / 2, col_y_top + column_size)
            col_y_bottom = center_y - column_offset_y - column_size
            if is_position_clear(col_x, col_y_bottom + column_size / 2, column_size):
                if abs(col_x - paid_x_min) > 1.5 and abs(col_x - paid_x_max) > 1.5:
                    add_obstacle(col_x - column_size / 2, col_y_bottom,
                                 col_x + column_size / 2, col_y_bottom + column_size)

        # ==================== 7. 数据转换 ====================
        self._create_exits()
        self.walls_numpy = [(np.array(s, dtype=np.float64), np.array(e, dtype=np.float64)) for s, e in self.walls]

    def _create_exits(self):
        """在墙壁上创建出口"""
        for exit_pos in self.exits:
            exit_x, exit_y = int(exit_pos[0]), int(exit_pos[1])
            new_walls = []
            for wall_start, wall_end in self.walls:
                if self._point_on_wall_segment((exit_x, exit_y), wall_start, wall_end):
                    gate_width = 1
                    if wall_start[0] == wall_end[0]:  # 垂直墙
                        if exit_y - gate_width // 2 > wall_start[1]:
                            new_walls.append((wall_start, (wall_start[0], exit_y - gate_width // 2)))
                        if exit_y + gate_width // 2 < wall_end[1]:
                            new_walls.append(((wall_end[0], exit_y + gate_width // 2), wall_end))
                    else:  # 水平墙
                        if exit_x - gate_width // 2 > wall_start[0]:
                            new_walls.append((wall_start, (exit_x - gate_width // 2, wall_start[1])))
                        if exit_x + gate_width // 2 < wall_end[0]:
                            new_walls.append(((exit_x + gate_width // 2, wall_end[1]), wall_end))
                else:
                    new_walls.append((wall_start, wall_end))
            self.walls = new_walls

    def _point_on_wall_segment(self, point, wall_start, wall_end):
        px, py = point
        x1, y1 = wall_start
        x2, y2 = wall_end
        if not (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)):
            return False
        if x1 == x2:
            return px == x1
        elif y1 == y2:
            return py == y1
        else:
            return abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)) < 1e-6

    def _build_static_geometry(self):
        """
        【统一预处理】
        1. 生成用于"物理碰撞"的矩形列表 (Vectorized)
        2. 生成用于"射线检测"的所有线段 (Vectorized)
        3. 生成用于"Dijkstra寻路"的静态栅格地图 (Rasterized)
        """
        W, H = self.grid_size_x, self.grid_size_y

        # 初始化栅格地图 (H, W)
        self.static_grid_map = np.zeros((H, W), dtype=bool)

        # ==========================================
        # 1. 处理矩形障碍物 (Vectorized Rasterization)
        # ==========================================
        # 预先生成物理数据
        if self.solid_obstacles:
            self.obs_rects_array = np.array([
                [obs['x_min'], obs['y_min'], obs['x_max'], obs['y_max']]
                for obs in self.solid_obstacles
            ], dtype=np.float32)

            # --- 优化点：批量计算整数索引 ---
            # 1. 左上角向下取整(floor)，右下角向上取整(ceil) -> 保证覆盖所有接触到的格子
            # 2. clip 限制在地图范围内
            # 3. astype(int) 转为整数
            x_starts = np.floor(self.obs_rects_array[:, 0]).astype(int).clip(0, W)
            y_starts = np.floor(self.obs_rects_array[:, 1]).astype(int).clip(0, H)
            x_ends = np.ceil(self.obs_rects_array[:, 2]).astype(int).clip(0, W)
            y_ends = np.ceil(self.obs_rects_array[:, 3]).astype(int).clip(0, H)

            # --- 赋值循环 (现在非常干净) ---
            # NumPy 的切片赋值极其高效，这里虽然有 Python 循环，但只循环"障碍物个数"次，开销极小
            for i in range(len(self.solid_obstacles)):
                self.static_grid_map[y_starts[i]:y_ends[i], x_starts[i]:x_ends[i]] = True
        else:
            self.obs_rects_array = np.zeros((0, 4), dtype=np.float32)

        # ==========================================
        # 2. 处理独立墙壁 (Line Rasterization)
        # ==========================================
        if self.walls:
            walls_arr = np.array(self.walls, dtype=np.float32)
            self.wall_starts_array = walls_arr[:, 0, :]
            self.wall_vecs_array = walls_arr[:, 1, :] - walls_arr[:, 0, :]
            self.wall_lens_sq_array = np.sum(self.wall_vecs_array ** 2, axis=1)
            self.wall_lens_sq_array[self.wall_lens_sq_array < 1e-10] = 1e-10

            # --- 优化点：简化线段光栅化 ---
            # 这里如果不引入 skimage/cv2，linspace 是最稳健的 NumPy 方法
            # 我们将其封装得更紧凑
            for (start, end) in self.walls:
                p0, p1 = np.array(start), np.array(end)
                dist = np.linalg.norm(p1 - p0)
                if dist < 1e-6: continue

                # 采样点数：长度 * 2 保证不漏点
                num = int(np.ceil(dist * 2)) + 1

                # 生成坐标 (N, 2)
                # 使用 stack 和 linspace 一次性生成 x 和 y
                pts = np.linspace(p0, p1, num)  # Shape: (num, 2)

                # 批量转整数并裁剪
                indices = pts.astype(int)
                xs = indices[:, 0].clip(0, W - 1)
                ys = indices[:, 1].clip(0, H - 1)

                # 批量赋值
                self.static_grid_map[ys, xs] = True
        else:
            self.wall_starts_array = np.zeros((0, 2), dtype=np.float32)
            self.wall_vecs_array = np.zeros((0, 2), dtype=np.float32)
            self.wall_lens_sq_array = np.zeros((0,), dtype=np.float32)

        # ==========================================
        # 3. 准备射线检测专用数据 (Raycasting Data)
        # ==========================================
        # 这部分代码本身已经很紧凑了，保持原样即可
        all_edges = []
        if len(self.walls) > 0:
            all_edges.append(np.array(self.walls, dtype=np.float32))

        if self.solid_obstacles:  # 只有当有障碍物时才计算
            # 利用广播机制一次性生成所有障碍物的4条边
            # rects: (N, 4) -> x1, y1, x2, y2
            rects = self.obs_rects_array
            x1, y1, x2, y2 = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]

            # 构造 4 条边: (N, 4, 2, 2) -> (N*4, 2, 2)
            # 这是一个高级 NumPy 技巧，如果觉得难懂，保留你原来的写法也没问题
            obs_edges = np.stack([
                np.stack([x1, y1, x2, y1], axis=-1),  # Bottom
                np.stack([x2, y1, x2, y2], axis=-1),  # Right
                np.stack([x2, y2, x1, y2], axis=-1),  # Top
                np.stack([x1, y2, x1, y1], axis=-1)  # Left
            ], axis=1).reshape(-1, 2, 2)

            all_edges.append(obs_edges)

        if all_edges:
            self.static_lines = np.vstack(all_edges)
            self.static_line_starts = self.static_lines[:, 0, :]
            self.static_line_vecs = self.static_lines[:, 1, :] - self.static_lines[:, 0, :]
        else:
            self.static_lines = None
            self.static_line_starts = None
            self.static_line_vecs = None

    def is_valid_position(self, pos):
        """检查位置是否有效（包含墙壁和边界）"""
        x, y = pos
        margin = 0.1
        if (x < margin or x >= self.grid_size_x - margin or
                y < margin or y >= self.grid_size_y - margin):
            return False
        if self._is_wall_collision(pos):
            return False
        return True

    def check_segment_collision(self, p1, p2):
        """
        利用向量叉乘检测移动路径 p1->p2 是否与任何墙壁相交。
        逻辑源自 PerceptionSystem._ray_lines_intersection_fast，但简化为单射线模式。
        """
        # 如果没有墙壁数据，直接安全
        if self.static_lines is None:
            return False

        # 1. 准备向量
        # 智能体的移动向量 "Ray": r = p2 - p1
        ray_start = np.array(p1, dtype=np.float32)
        ray_vec = np.array(p2, dtype=np.float32) - ray_start

        # 墙壁向量 "Line": s = wall_end - wall_start
        # 已经在 _build_static_geometry 中预计算好了
        line_starts = self.static_line_starts  # shape: (N, 2)
        line_vecs = self.static_line_vecs  # shape: (N, 2)

        # 2. 计算分母 (Ray x Line)
        # 2D 叉乘公式: a x b = a.x * b.y - a.y * b.x
        # 这里 ray_vec 是 (2,)，line_vecs 是 (N, 2)，利用广播直接计算
        denominator = (ray_vec[0] * line_vecs[:, 1] -
                       ray_vec[1] * line_vecs[:, 0])

        # 3. 过滤平行线 (分母接近 0)
        valid_mask = np.abs(denominator) > 1e-10
        if not np.any(valid_mask):
            return False  # 所有墙壁都平行（概率极低，但也可能意味着没有有效交叉）

        # 只取出非平行的墙壁进行计算，减少计算量
        denom_valid = denominator[valid_mask]
        line_starts_valid = line_starts[valid_mask]
        line_vecs_valid = line_vecs[valid_mask]

        # 4. 计算分子
        # diff = q - p (墙起点 - 移动起点)
        diff = line_starts_valid - ray_start

        # t_numer = (q - p) x s  (用于计算射线上的比例 t)
        t_numer = diff[:, 0] * line_vecs_valid[:, 1] - diff[:, 1] * line_vecs_valid[:, 0]

        # u_numer = (q - p) x r  (用于计算墙壁上的比例 u，也就是你代码里的 s)
        u_numer = diff[:, 0] * ray_vec[1] - diff[:, 1] * ray_vec[0]

        # 5. 计算比例 t 和 u
        # t: 交点在移动向量上的比例。0 <= t <= 1 代表交点在 p1 和 p2 之间
        t = t_numer / denom_valid
        # u: 交点在墙壁线段上的比例。0 <= u <= 1 代表交点在墙壁端点之间
        u = u_numer / denom_valid

        # 6. 判定碰撞
        # 必须同时满足 t 和 u 都在 [0, 1] 范围内
        collision_mask = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

        # 只要有一个墙壁满足碰撞条件，就返回 True
        return np.any(collision_mask)

    def _is_wall_collision(self, pos, use_safety_margin=True):
        """
        【向量化版本】检查点是否与任何墙壁或障碍物碰撞
        """
        pos = np.array(pos, dtype=np.float32)  # 确保是numpy array
        x, y = pos
        safety_margin = self.AGENT_RADIUS if use_safety_margin else 0

        # ---------------------------------------------------------
        # 1. 矩形障碍物检查 (Vectorized Area Check)
        # ---------------------------------------------------------
        if self.obs_rects_array.shape[0] > 0:
            # 利用广播机制一次性比较：
            # 判断点是否在 [x_min-margin, x_max+margin] 和 [y_min-margin, y_max+margin] 范围内

            # 提取坐标列 (N,)
            x_mins = self.obs_rects_array[:, 0]
            y_mins = self.obs_rects_array[:, 1]
            x_maxs = self.obs_rects_array[:, 2]
            y_maxs = self.obs_rects_array[:, 3]

            # 并行比较
            in_x = (x >= x_mins - safety_margin) & (x <= x_maxs + safety_margin)
            in_y = (y >= y_mins - safety_margin) & (y <= y_maxs + safety_margin)

            # 如果同时满足 x 和 y 范围，说明碰撞
            if np.any(in_x & in_y):
                return True

        # ---------------------------------------------------------
        # 2. 墙壁线段检查 (Vectorized Point-to-Segment Distance)
        # ---------------------------------------------------------
        if self.wall_starts_array.shape[0] > 0:
            # 计算点 P 到所有线段起点的向量 AP = P - A
            # pos (2,) - starts (M, 2) -> ap (M, 2)
            ap = pos - self.wall_starts_array

            # 计算投影比例 t = dot(AP, AB) / dot(AB, AB)
            t_numer = np.sum(ap * self.wall_vecs_array, axis=1)

            # 安全除法：如果墙长为0（退化为点），t设为0
            t = np.divide(t_numer, self.wall_lens_sq_array,
                          out=np.zeros_like(t_numer),
                          where=self.wall_lens_sq_array > 1e-10)

            # 将 t 限制在线段范围内 [0, 1]
            t = np.clip(t, 0.0, 1.0)

            # 计算线段上最近点 C = A + t * AB
            # t[:, np.newaxis] 将 (M,) 变为 (M, 1) 以支持广播
            closest_points = self.wall_starts_array + t[:, np.newaxis] * self.wall_vecs_array

            # 计算距离平方 ||P - C||^2
            dists_sq = np.sum((pos - closest_points) ** 2, axis=1)

            # 检查是否有任何距离小于 safety_margin^2
            # 使用平方比较避免开根号，性能更好
            if np.any(dists_sq < safety_margin ** 2):
                return True

        return False

    def is_at_exit(self, pos):
        for exit_pos in self.exits:
            to_exit = np.linalg.norm(pos - exit_pos)
            if to_exit <= self.EXIT_RADIUS:
                return True
        return False

    def initialize_positions_configurable(self, agents, zone_config, fire_system,
                                          strict_safe_threshold=0.9, safety_buffer_radius=1.5):
        """
        初始化智能体位置，依赖 FireSystem 进行安全检查
        """
        agent_positions = {}
        used_positions = []
        probabilities = [z['probability'] for z in zone_config]

        for agent in agents:
            if len(zone_config) == 1 and zone_config[0]['probability'] == 1.0:
                zone = zone_config[0]
            else:
                zone = np.random.choice(zone_config, p=probabilities)

            attempts = 0
            max_attempts = 2000

            while attempts < max_attempts:
                x = np.random.uniform(zone['x_range'][0], zone['x_range'][1])
                y = np.random.uniform(zone['y_range'][0], zone['y_range'][1])
                center_pos = np.array([x, y], dtype=np.float32)

                if self.is_valid_position(center_pos):
                    is_neighborhood_safe = True
                    # 检查中心点
                    if fire_system.get_env_factor_at_position(center_pos) < strict_safe_threshold:
                        is_neighborhood_safe = False

                    # 检查周围卫星点
                    if is_neighborhood_safe:
                        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                        for angle in angles:
                            check_pos = center_pos + np.array([
                                np.cos(angle) * safety_buffer_radius,
                                np.sin(angle) * safety_buffer_radius
                            ])
                            factor = fire_system.get_env_factor_at_position(check_pos)
                            if factor < strict_safe_threshold:
                                is_neighborhood_safe = False
                                break

                    if is_neighborhood_safe:
                        too_close = False
                        for used_pos in used_positions:
                            if np.linalg.norm(center_pos - used_pos) < 1.0:
                                too_close = True
                                break
                        if not too_close:
                            agent_positions[agent] = center_pos
                            used_positions.append(center_pos)
                            break
                attempts += 1

            if attempts >= max_attempts:
                print(f"⚠️ Warning: Could not find STRICT safe position for {agent}.")

        return agent_positions