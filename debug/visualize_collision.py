import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # 如果没有 tqdm 可以去掉，只是为了看进度条
from envs.components.building import BuildingManager


def main():
    # ================= 1. 设置参数 =================
    GRID_W, GRID_H = 80, 40
    RESOLUTION = 0.1  # 分辨率：每隔多少米检测一次（越小越精细，但越慢）

    print(f"Initializing BuildingManager ({GRID_W}x{GRID_H})...")
    bm = BuildingManager(grid_size_x=GRID_W, grid_size_y=GRID_H)

    # 获取 Agent 半径（用于显示标题）
    agent_radius = bm.AGENT_RADIUS
    print(f"Agent Radius: {agent_radius}m")

    # ================= 2. 生成检测网格 =================
    print("Generating collision map (this might take a moment)...")

    x_range = np.arange(0, GRID_W, RESOLUTION)
    y_range = np.arange(0, GRID_H, RESOLUTION)

    # 创建网格 (Height, Width) 注意矩阵索引通常是 row(y), col(x)
    grid_map = np.zeros((len(y_range), len(x_range)))

    # ================= 3. 计算碰撞 (逐点扫描) =================
    # 为了速度，这里简单粗暴地双重循环。
    # 也可以改写 BuildingManager 支持 batch 输入，但用来画图这样够了。

    for i, y in enumerate(tqdm(y_range, desc="Scanning Rows")):
        for j, x in enumerate(x_range):
            pos = np.array([x, y])

            # 核心检测：检查是否撞墙 (考虑半径)
            # use_safety_margin=True 代表考虑 agent_radius
            if bm._is_wall_collision(pos, use_safety_margin=True):
                grid_map[i, j] = 1.0  # 1 代表障碍/不可通行区域
            else:
                grid_map[i, j] = 0.0  # 0 代表自由区域

    # ================= 4. 绘图 =================
    plot_collision_map(bm, grid_map, GRID_W, GRID_H, RESOLUTION)


def plot_collision_map(bm, grid_map, w, h, res):
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- A. 画碰撞热力图 (背景) ---
    # extent=[xmin, xmax, ymin, ymax] 确保坐标对齐
    # origin='lower' 确保 (0,0) 在左下角
    # cmap='Reds' 让障碍物显示为红色，空地为白色
    im = ax.imshow(grid_map, origin='lower', cmap='Reds',
                   extent=[0, w, 0, h], vmin=0, vmax=1.5, alpha=0.6)

    # --- B. 画原始矢量障碍物 (参考线) ---
    # 这样你可以看到“膨胀”了多少

    # 1. 矩形
    for obs in bm.solid_obstacles:
        rect = plt.Rectangle(
            (obs['x_min'], obs['y_min']),
            obs['x_max'] - obs['x_min'],
            obs['y_max'] - obs['y_min'],
            edgecolor='blue', facecolor='none', linewidth=1, linestyle='--', label='Original Obstacle'
        )
        ax.add_patch(rect)

    # 2. 墙壁
    for wall_start, wall_end in bm.walls:
        ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]],
                color='black', linewidth=1.5, label='Original Wall')

    # --- C. 装饰 ---
    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax.set_title(f"Collision Map (Resolution: {res}m, Agent Radius: {bm.AGENT_RADIUS}m)\n"
                 f"Red Areas = Where center of agent cannot go")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(False)  # 关掉网格线以免干扰

    print("Displaying plot...")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()