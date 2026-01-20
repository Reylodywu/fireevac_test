import numpy as np
import matplotlib.pyplot as plt
from envs.components.building import BuildingManager


def main():
    # ================= 1. 初始化地图 =================
    # 使用与你训练配置相同的尺寸
    grid_w, grid_h = 80, 40

    print(f"Initializing BuildingManager ({grid_w}x{grid_h})...")
    bm = BuildingManager(grid_size_x=grid_w, grid_size_y=grid_h)

    # ================= 2. 定义你想测试的点 =================
    # 格式: (x, y)
    test_points = [
        (68.0, 24.0),  # 假设是空地
        (10.0, 10.0),  # 随便一个点
        (6.0, 9.0),  # 可能是安检区障碍物内部？
        (79.0, 39.0),  # 靠近出口
    ]

    print("\n--- Collision Check Results ---")
    for pt in test_points:
        # 注意：pos 需要是 numpy array 或 list/tuple
        pos = np.array(pt)

        # 调用核心碰撞检测函数
        # use_safety_margin=True 会考虑 agent_radius (默认0.3)
        is_hit = bm._is_wall_collision(pos, use_safety_margin=True)

        # 也可以测试 is_valid_position (它包含边界检查 + 碰撞检查)
        is_valid = bm.is_valid_position(pos)

        status = "❌ COLLISION" if is_hit else "✅ FREE"
        print(f"Point {pt}: {status} (Valid Position: {is_valid})")

    # ================= 3. 可视化验证 (强烈推荐) =================
    plot_verification(bm, test_points, grid_w, grid_h)


def plot_verification(bm, test_points, w, h):
    """画出障碍物和测试点"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. 画矩形障碍物
    for obs in bm.solid_obstacles:
        rect = plt.Rectangle(
            (obs['x_min'], obs['y_min']),
            obs['x_max'] - obs['x_min'],
            obs['y_max'] - obs['y_min'],
            color='gray', alpha=0.5
        )
        ax.add_patch(rect)

    # 2. 画墙壁线段
    for wall_start, wall_end in bm.walls:
        ax.plot([wall_start[0], wall_end[0]], [wall_start[1], wall_end[1]],
                color='black', linewidth=2)

    # 3. 画测试点
    for pt in test_points:
        pos = np.array(pt)
        is_hit = bm._is_wall_collision(pos, use_safety_margin=True)
        color = 'red' if is_hit else 'green'
        marker = 'x' if is_hit else 'o'

        ax.scatter(pt[0], pt[1], c=color, s=100, marker=marker, label='Tested Point', zorder=10)
        # 标注坐标
        ax.text(pt[0] + 0.5, pt[1] + 0.5, f"{pt}", fontsize=9, color=color)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.set_title("Collision Detection Verification")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()