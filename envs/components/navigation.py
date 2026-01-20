import numpy as np
from heapq import heappush, heappop
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


class NavigationSystem:
    def __init__(self, grid_size_x, grid_size_y, exits, building_manager):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.exits = exits
        self.building = building_manager

        self.distance_fields = {}
        self.merged_interpolator = None
        self.max_pathfinding_distance = 0

        # 预计算
        self._precompute_distance_field()
        # # 可视化检查距离场（调试用）
        # self.visualize_distance_field()

    def _precompute_distance_field(self):
        """
        计算从每个出口出发的全局距离场 (Dijkstra 算法)。
        利用 BuildingManager 预先生成的静态网格地图 (static_grid_map) 进行极速查表。
        """
        # 确保引入堆操作库
        from heapq import heappush, heappop

        W, H = self.grid_size_x, self.grid_size_y

        # 8方向移动向量
        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # 移动代价：直线=1.0, 对角线=1.414
        BASE_COST = np.array([1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414])

        self.distance_fields = {}
        self.max_pathfinding_distance = 0

        # =========================================================
        # 1. 获取静态障碍物掩码 (核心优化)
        # 直接引用 BuildingManager 中计算好的栅格地图
        # 形状: (H, W), 索引: [y, x], 值: True=障碍, False=可通行
        # =========================================================
        if not hasattr(self.building, 'static_grid_map'):
            raise RuntimeError("BuildingManager 尚未生成 static_grid_map，请确保先调用 _build_static_geometry")
        self.wall_mask = self.building.static_grid_map

        # =========================================================
        # 2. 对每个出口运行 Dijkstra
        # =========================================================
        for exit_pos in sorted(self.exits, key=tuple):
            ex, ey = int(exit_pos[0]), int(exit_pos[1])
            # 初始化距离矩阵为无穷大
            distances = np.full((H, W), np.inf, dtype=np.float64)
            distances[ey, ex] = 0.0
            # 优先队列: (distance, counter, x, y)
            # counter 用于在 distance 相同时打破平局，虽然 python 元组比较通常不需要
            queue = [(0.0, 0, ex, ey)]
            counter = 0
            while queue:
                dist, _, x, y = heappop(queue)
                # 如果当前路径比已知路径长，跳过
                if dist > distances[y, x]:
                    continue
                for i, (dx, dy) in enumerate(DIRS):
                    nx, ny = x + dx, y + dy

                    # A. 越界检查
                    if not (0 <= nx < W and 0 <= ny < H):
                        continue
                    # B. 障碍物检查 (查表 O(1)，取代原来的 is_wall_collision)
                    if self.wall_mask[ny, nx]:
                        continue
                    # C. 对角线移动时的“穿墙缝”检查 (避免穿过两个角接触的障碍物)
                    # 如果 dx和dy都不为0，说明是斜向移动
                    if abs(dx) + abs(dy) == 2:
                        # 检查水平和垂直分量对应的格子是否是墙
                        # 如果两边都是墙，说明在钻缝，禁止通行
                        if self.wall_mask[y, x + dx] or self.wall_mask[y + dy, x]:
                            continue
                    # D. 更新距离
                    new_dist = dist + BASE_COST[i]
                    if new_dist < distances[ny, nx]:
                        distances[ny, nx] = new_dist
                        counter += 1
                        heappush(queue, (new_dist, counter, nx, ny))

            # 存储该出口的距离场
            self.distance_fields[tuple(exit_pos)] = distances

            # 更新全局最大距离 (用于后续观测值的归一化)
            valid_mask = np.isfinite(distances)
            if valid_mask.any():
                local_max = float(distances[valid_mask].max())
                self.max_pathfinding_distance = max(self.max_pathfinding_distance, local_max)

        # 3. 计算合并距离场 (即每个点到最近出口的距离)
        self._precompute_merged_distance_field()

    def _precompute_merged_distance_field(self):
        """
        计算合并距离场，并对墙壁区域进行梯度外推填充。
        """
        # 1. 聚合所有出口的场，取最小值
        valid_fields = [self.distance_fields[tuple(e)] for e in self.exits]

        # 原始场 (墙内是 inf)
        # Shape: (H, W)
        raw_field = np.min(valid_fields, axis=0)

        # 2. 识别墙壁区域
        inf_mask = np.isinf(raw_field)

        # --- [核心逻辑] 使用 EDT 算法填充墙壁区域 ---
        # 如果全图都是路(无inf)或者全图无路(全inf)，就不需要/无法填充
        if inf_mask.any() and not inf_mask.all():
            # distance_transform_edt 计算二值图中 True(墙) 到最近 False(路) 的距离
            # dists: 墙内点到最近路点的欧几里得距离
            # indices: 最近路点的坐标索引 [y_indices, x_indices]
            dists, indices = distance_transform_edt(
                inf_mask,
                return_distances=True,
                return_indices=True
            )

            # A. 取出最近路点的原始距离值 (Base Value)
            base_values = raw_field[indices[0], indices[1]]

            # B. 计算填充值 = Base Value + 物理距离
            # 这样在墙内构建了一个线性增长的斜坡，梯度指向墙外
            filled_field = base_values + dists
        else:
            # 如果没有 inf 需要填充，或者全是 inf 没法填充，就保持原样
            filled_field = raw_field

        # 用于调试查看 (可选)
        self.merged_field_raw = filled_field

        # 3. 创建插值器
        x_coords = np.arange(self.grid_size_x)
        y_coords = np.arange(self.grid_size_y)

        # method='linear': 因为我们已经把墙壁的值平滑化了，线性插值能提供最佳梯度
        # bounds_error=False, fill_value=None: 允许轻微的越界外推
        self.merged_interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            filled_field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )


    def _is_diagonal_blocked(self, x, y, dx, dy):
        """
        检查对角线移动是否被阻挡（防止穿墙缝）。
        [优化] 直接查 wall_mask 表，不再调用耗时的 building._is_wall_collision
        """
        # 必须是对角线移动 (dx, dy 绝对值之和为 2)
        if abs(dx) + abs(dy) != 2:
            return False

        # 获取两个相邻的“直角”邻居坐标
        # 例如：向右下(1,1)移动，需检查 右(1,0) 和 下(0,1)
        n1_x, n1_y = x + dx, y
        n2_x, n2_y = x, y + dy

        # 检查边界
        W, H = self.grid_size_x, self.grid_size_y
        mask = self.wall_mask  # 确保你的类里有这个成员变量 (在 _precompute 之前生成的)

        # 检查邻居1
        blocked1 = False
        if 0 <= n1_x < W and 0 <= n1_y < H:
            if mask[n1_y, n1_x]: blocked1 = True

        # 检查邻居2
        blocked2 = False
        if 0 <= n2_x < W and 0 <= n2_y < H:
            if mask[n2_y, n2_x]: blocked2 = True

        # 策略：如果两个直角邻居都是墙，肯定堵住了（钻不进去）
        return blocked1 and blocked2

    def get_pathfinding_distance(self, pos):
        """
        获取任意浮点坐标的导航距离
        """
        x, y = pos[0], pos[1]

        # 宽容的边界检查：如果在地图外，稍微外推一点也没关系，
        # 但如果太远（超过 grid_size），返回 inf 引导它回来
        if not (0 <= x <= self.grid_size_x and 0 <= y <= self.grid_size_y):
            return np.inf

        # RegularGridInterpolator 要求的输入顺序是 (y, x)
        # 定义时用的 (y_coords, x_coords)
        return float(self.merged_interpolator([y, x])[0])


    # [新增] 可视化方法
    def visualize_distance_field(self):
        """
        可视化合并后的距离场。
        优化点：墙壁不参与热力图数值计算，避免拉伸色阶。
        """
        # 1. 获取原始数据
        data = self.merged_field_raw.copy()

        # -----------------------------------------------------------
        # 关键修改：不要把 inf 替换成大数值！而是创建一个掩码数组 (Masked Array)
        # -----------------------------------------------------------

        # np.ma.masked_invalid 会自动把 np.inf 和 np.nan 标记为无效
        # 这样 matplotlib 在计算颜色最大值(vmax)时，会自动忽略这些点
        data_masked = np.ma.masked_invalid(data)

        # 2. 设置绘图
        plt.figure(figsize=(10, 8))

        # 3. 配置颜色映射 (Colormap)
        current_cmap = plt.cm.get_cmap("viridis").copy()

        # set_bad 用来设置 "无效值" (即被 mask 的墙壁) 的颜色
        # 建议使用灰色或黑色，以便与热力图区分开
        current_cmap.set_bad(color='lightgray')

        # 4. 绘制热力图
        # Matplotlib 会自动根据 masked_data 中的 *有效数据* 的最大/最小值来归一化颜色
        # 这样，即使墙壁是 inf，也不会影响有效路径的颜色显示
        plt.imshow(data_masked,
                   cmap=current_cmap,
                   origin='lower',
                   interpolation='nearest')  # nearest 防止边缘模糊

        plt.colorbar(label='Distance to Nearest Exit')

        # 5. 绘制出口位置 (保持不变)
        exits_arr = np.array(self.exits)
        if len(exits_arr) > 0:
            plt.scatter(exits_arr[:, 0], exits_arr[:, 1],
                        c='red', s=100, marker='X', edgecolors='white',
                        label='Exits', zorder=10)  # zorder确保出口显示在最上层

        plt.title(f"Distance Field Visualization (Grid: {self.grid_size_x}x{self.grid_size_y})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(False)  # 通常热力图不需要网格线
        plt.show()