import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import os


class Visualizer:
    def __init__(self, grid_size_x, grid_size_y, render_mode):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.colorbar = None

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']

    def render(self, agent_positions, agent_paths, exits, fire_system, building_manager, current_step):
        if self.render_mode is None: return

        if self.fig is None:
            aspect_ratio = self.grid_size_y / self.grid_size_x
            fig_width = 12
            fig_height = fig_width * aspect_ratio
            self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))

        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        # 移除完组件后，再清空画布
        self.ax.clear()

        # 1. 绘制建筑结构
        self._draw_building_structure(building_manager)

        # 2. 绘制环境因子热图
        self._draw_env_factor_heatmap_relative1(fire_system)

        # # 3. 绘制智能体
        # colors = ['green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']
        # agents = list(agent_positions.keys())  # 假设顺序一致
        # for i, agent in enumerate(agents):
        #     color = colors[i % len(colors)]
        #
        #     # Path
        #     if agent in agent_paths and len(agent_paths[agent]) > 1:
        #         path_array = np.array(agent_paths[agent])
        #         self.ax.plot(path_array[:, 0], path_array[:, 1], color=color, linewidth=2, alpha=0.7)
        #
        #     # Position
        #     pos = agent_positions[agent]
        #     self.ax.scatter(pos[0], pos[1], c=color, s=50, marker='o', edgecolors='black', label=agent, zorder=20)

        # 4. 绘制出口
        for i, exit_pos in enumerate(exits):
            self.ax.scatter(exit_pos[0], exit_pos[1], c='blue', s=200, marker='s', edgecolors='white', zorder=15)

        # Settings
        self.ax.set_xlim(0, self.grid_size_x)
        self.ax.set_ylim(0, self.grid_size_y)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Env Factor Evacuation - Step: {current_step}')
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if self.render_mode == "human":
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return buf

    def _draw_env_factor_heatmap(self, fire_system):
        # 假设 data 是速度折减系数 f (0 到 1)
        f = fire_system.high_res_env_factor_map

        if f is None: return

        # 1. 计算风险系数 xi = 1/f
        xi = 1.0 / f

        # 2. 归一化处理以匹配文献的 0-1 Colorbar
        # 文献提到将风险系数归一化，使得安全区域为 0，最危险区域为 1
        xi_threshold = 1.0 / 0.3  # 设置阈值以利于可视化
        xi_clipped = np.clip(xi, 1.0, xi_threshold)

        # 然后再进行归一化
        risk_map = (xi_clipped - 1.0) / (xi_threshold - 1.0)

        # 3. 绘图使用计算后的 risk_map
        # # 自定义色条
        # colors = ['darkred', 'red', 'orange', 'yellow', 'white']
        # cmap = LinearSegmentedColormap.from_list('fire', colors, N=256)
        cmap = plt.cm.jet  # 使用MATLAB 风格颜色映射
        im = self.ax.imshow(risk_map, origin='lower', cmap=cmap,
                            extent=[0, self.grid_size_x, 0, self.grid_size_y],
                            aspect='equal', vmin=0, vmax=np.max(risk_map), alpha=0.8, zorder=0)

        if self.colorbar is None:
            self.colorbar = plt.colorbar(im, ax=self.ax, shrink=0.6)
            self.colorbar.set_label('Risk', rotation=270, labelpad=15)

    def _draw_env_factor_heatmap_relative1(self, fire_system):
        """
        可视化方案：使用 Risk = 1 - f，并配合 Min-Max 归一化
        优点：线性、稳定、直观。
        """
        f = fire_system.high_res_env_factor_map
        if f is None: return

        # 1. 计算线性风险 (Linear Risk)
        # 物理含义: 速度损失率 (Speed Loss Ratio)
        # f=1.0 -> Risk=0.0; f=0.0 -> Risk=1.0
        risk_map = 1.0 - f

        # 2. 获取当前统计值
        curr_min = np.min(risk_map)
        curr_max = np.max(risk_map)

        # 3. 计算动态分母 (Contrast Stretching)
        # 这一步是为了让早期微小的风险也能显示出红色梯度
        denominator = curr_max - curr_min
        if denominator < 1e-6:
            denominator = 1.0

        # 4. 执行 Min-Max 归一化
        # 公式: (x - min) / (max - min)
        # 结果: 当前最安全的点 -> 0.0 (蓝); 当前最危险的点 -> 1.0 (红)
        risk_norm = (risk_map - curr_min) / denominator

        # 5. 绘图
        # 【注意】这里换回了标准的 'jet'
        # 因为现在数值越大越危险 (Blue -> Red)
        cmap = plt.cm.jet

        im = self.ax.imshow(risk_norm, origin='lower', cmap=cmap,
                            extent=[0, self.grid_size_x, 0, self.grid_size_y],
                            aspect='equal',
                            vmin=0.0, vmax=1.0,
                            alpha=0.8, zorder=0)

        # 6. 设置 Colorbar
        if self.colorbar is not None:
            self.colorbar.remove()

        self.colorbar = plt.colorbar(im, ax=self.ax, shrink=0.6)

        # 标签说明：显示的是速度损失率
        # 早期可能是: 0.00 (Blue) -> 0.02 (Red)
        # 晚期可能是: 0.00 (Blue) -> 1.00 (Red)
        label_str = f'Risk (1-f): {curr_min:.3f} (Blue) -> {curr_max:.3f} (Red)'
        self.colorbar.set_label(label_str, rotation=270, labelpad=15)

        # 7. 左上角显示真实的峰值风险
        self.ax.text(0.95, 0.95, f"Peak Risk: {curr_max:.3f}",
                     transform=self.ax.transAxes, color='white',
                     ha='right', va='top', fontweight='bold',
                     bbox=dict(boxstyle="round", fc="black", ec="none", alpha=0.5))

    def _draw_env_factor_heatmap_relative(self, fire_system):
        """
        基于 Env Factor 的 Min-Max 归一化可视化方案
        优点：在火灾早期也能看清梯度，晚期也不会过饱和。
        """
        f = fire_system.high_res_env_factor_map
        if f is None: return

        # 1. 获取当前帧的极值
        curr_min = np.min(f)
        curr_max = np.max(f)

        # 2. 计算分母 (Range)
        # 如果全场数值一样 (比如刚开始都是 1.0)，防止除以 0
        denominator = curr_max - curr_min
        if denominator < 1e-6:
            denominator = 1.0

        # 3. 执行 Min-Max 归一化 -> 映射到 [0, 1]
        # 公式: (f - min) / (max - min)
        # 结果: 最危险的点(min) -> 0.0; 最安全的点(max) -> 1.0
        f_norm = (f - curr_min) / denominator

        # 4. 绘图
        # 【关键】使用 'jet_r' (reversed jet)
        # jet_r 的定义: 0.0 是深红 (Red), 1.0 是深蓝 (Blue)
        # 刚好对应: 0.0 是最危险 (Min f), 1.0 是最安全 (Max f)
        cmap = plt.cm.jet_r

        im = self.ax.imshow(f_norm, origin='lower', cmap=cmap,
                            extent=[0, self.grid_size_x, 0, self.grid_size_y],
                            aspect='equal',
                            vmin=0.0, vmax=1.0,  # 归一化后固定为 0-1
                            alpha=0.8, zorder=0)

        # 5. 设置 Colorbar
        if self.colorbar is not None:
            self.colorbar.remove()

        self.colorbar = plt.colorbar(im, ax=self.ax, shrink=0.6)

        # 标签清楚地说明：红色代表当前的最小值，蓝色代表当前的最大值
        label_str = f'Env Factor: {curr_min:.3f} (Red) -> {curr_max:.3f} (Blue)'
        self.colorbar.set_label(label_str, rotation=270, labelpad=15)

        # 6. 左上角辅助信息 (可选，方便看绝对数值)
        self.ax.text(0.95, 0.95, f"Min F: {curr_min:.3f}",
                     transform=self.ax.transAxes, color='white',
                     ha='right', va='top', fontweight='bold',
                     bbox=dict(boxstyle="round", fc="black", ec="none", alpha=0.5))

    def _draw_building_structure(self, building_manager):
        for obs in building_manager.solid_obstacles:
            rect = Rectangle((obs['x_min'], obs['y_min']),
                             obs['x_max'] - obs['x_min'],
                             obs['y_max'] - obs['y_min'],
                             facecolor='#4A4A4A', edgecolor='black', linewidth=2, alpha=0.85, zorder=2)
            self.ax.add_patch(rect)

        for wall_start, wall_end in building_manager.walls:
            self.ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]],
                         color='#2C3E50', linewidth=6, solid_capstyle='round', alpha=0.95, zorder=3)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

    def save_paper_assets(self, step, output_dir="paper_assets"):
        """
        保存两张图：
        1. Context View: 右半场 (30-80m)，展示整体路径
        2. Zoom View: 局部特写 (45-65m)，展示避火细节
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 保存当前的视图状态，以便画完后恢复
        original_xlim = self.ax.get_xlim()
        original_ylim = self.ax.get_ylim()
        original_title = self.ax.get_title()

        # --- 导出素材 A: 主视图 (Context View) ---
        # 调整视野范围：只看右半场 (假设火和出口都在右边)
        self.ax.set_xlim(30, 80)
        self.ax.set_ylim(0, 40)
        self.ax.set_title("")

        # 保存高清图 (dpi=300)
        file_main = os.path.join(output_dir, f"step_{step}_context.png")
        self.fig.savefig(file_main, dpi=300, bbox_inches='tight')
        print(f"📸 Saved Context View: {file_main}")

        # --- 导出素材 B: 局部特写 (Zoom View) ---
        # 调整视野范围：聚焦火源附近 (假设火源在 55, 20)
        self.ax.set_xlim(45, 65)
        self.ax.set_ylim(10, 30)

        # 特写图通常不需要坐标轴刻度，为了干净
        self.ax.axis('off')

        file_zoom = os.path.join(output_dir, f"step_{step}_zoom.png")
        self.fig.savefig(file_zoom, dpi=300, bbox_inches='tight')
        print(f"📸 Saved Zoom View: {file_zoom}")

        # --- 恢复状态 ---
        # 必须恢复，否则屏幕上的动态演示会卡在局部视图里
        self.ax.axis('on')  # 重新开启坐标轴
        self.ax.set_xlim(original_xlim)
        self.ax.set_ylim(original_ylim)
        self.ax.set_title(original_title)

        # 如果是 human 模式，重绘一下以免界面闪烁
        if self.render_mode == "human":
            self.fig.canvas.draw()