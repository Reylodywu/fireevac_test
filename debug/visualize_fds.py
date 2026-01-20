import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def visualize_fds_results(
        npz_file='fire_env.npz',
        bounds=(0.0, 80.0, 0.0, 40.0),
        save_anim=False,
        snapshot_times=None,  # ✨ 新增：指定需要截图的时间列表，例如 [60, 300, 600]
        output_dir='output_images'  # ✨ 新增：截图保存的文件夹
):
    """
    可视化 FDS 数据，支持生成动画和特定时刻的快照。

    :param snapshot_times: list, 需要保存快照的时间点(秒), 例如 [100, 300, 600]
    """

    # --- 1. 数据加载与预处理 ---
    print(f"📂 正在加载数据: {npz_file} ...")
    if not os.path.exists(npz_file):
        print("❌ 找不到文件，请先运行提取脚本。")
        return

    data = np.load(npz_file)
    times = data['times']
    temp = data['temperature'] if 'temperature' in data else None
    co = data['co'] if 'co' in data else None
    vis = data['visibility'] if 'visibility' in data else None

    # 物理范围
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]

    # --- 2. 初始化绘图画布 ---
    # 创建文件夹
    if snapshot_times and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # 存储绘图对象句柄
    plots = {}

    # (A) 温度 Temperature
    if temp is not None:
        vmin, vmax = 20, np.percentile(temp, 99)
        im_temp = axes[0].imshow(temp[0], origin='lower', extent=extent,
                                 cmap='inferno', vmin=vmin, vmax=vmax)
        axes[0].set_title("Temperature ($^\circ$C)")
        fig.colorbar(im_temp, ax=axes[0], label='T ($^\circ$C)')
        plots['temp'] = im_temp
    else:
        axes[0].text(0.5, 0.5, 'No Data', ha='center', transform=axes[0].transAxes)

    # (B) CO Concentration (转化为 ppm)
    if co is not None:
        co_ppm = co * 1e6
        vmin, vmax = 0, np.max(co_ppm) * 0.8  # 稍微压低上限以突显细节
        im_co = axes[1].imshow(co_ppm[0], origin='lower', extent=extent,
                               cmap='Oranges', vmin=vmin, vmax=vmax)
        axes[1].set_title("CO Concentration (ppm)")
        fig.colorbar(im_co, ax=axes[1], label='CO (ppm)')
        plots['co'] = (im_co, co_ppm)
    else:
        axes[1].text(0.5, 0.5, 'No Data', ha='center', transform=axes[1].transAxes)

    # (C) 能见度 Visibility
    if vis is not None:
        im_vis = axes[2].imshow(vis[0], origin='lower', extent=extent,
                                cmap='gray', vmin=0, vmax=30)
        axes[2].set_title("Visibility (m)")
        fig.colorbar(im_vis, ax=axes[2], label='Vis (m)')
        plots['vis'] = im_vis
    else:
        axes[2].text(0.5, 0.5, 'No Data', ha='center', transform=axes[2].transAxes)

    # 标签与时间文本
    axes[2].set_xlabel("Length X (m)")
    for ax in axes: ax.set_ylabel("Width Y (m)")
    time_text = axes[0].text(0.02, 1.05, '', transform=axes[0].transAxes,
                             fontsize=14, fontweight='bold', color='blue')

    # --- 3. 核心更新函数 (供动画和快照共用) ---
    def update_frame(frame_idx):
        """更新某一帧的所有子图数据"""
        current_time = times[frame_idx]
        time_text.set_text(f"Time: {current_time:.1f} s")

        # 更新温度
        if 'temp' in plots:
            plots['temp'].set_data(temp[frame_idx])

        # 更新 CO
        if 'co' in plots:
            img_obj, data_arr = plots['co']
            img_obj.set_data(data_arr[frame_idx])

        # 更新能见度
        if 'vis' in plots:
            plots['vis'].set_data(vis[frame_idx])

        return [plots.get('temp'), plots.get('co', (None,))[0], plots.get('vis'), time_text]

    # --- 4. ✨ 执行快照保存 (Snapshot Mode) ---
    if snapshot_times:
        print(f"📸 开始处理快照: {snapshot_times}")
        for target_t in snapshot_times:
            # 1. 找到最接近的时间点索引
            # abs(times - target) 找到差值最小的那个位置
            idx = (np.abs(times - target_t)).argmin()
            actual_t = times[idx]

            # 2. 更新画面
            update_frame(idx)

            # 3. 保存图片
            # 文件名包含目标时间和实际时间，防止混淆
            filename = os.path.join(output_dir, f"snapshot_t{int(target_t)}s.png")
            plt.savefig(filename, dpi=600, bbox_inches='tight')
            print(f"   ✅ 已保存: {filename} (实际时间: {actual_t:.2f}s)")

    # --- 5. 执行动画 (Animation Mode) ---
    # 如果只是为了截图，不需要弹出窗口，可以注释掉 plt.show()
    # 如果需要保存视频，则运行以下逻辑

    if save_anim or (snapshot_times is None):
        print(f"🎬 开始准备显示/保存动画...")
        ani = FuncAnimation(fig, update_frame, frames=len(times), interval=50, blit=False)

        if save_anim:
            ani.save("fds_simulation.mp4", writer='ffmpeg', fps=20, dpi=150)
            print("✅ 视频已保存")
        else:
            plt.show()  # 如果只跑了快照，不想看动画，可以把这个放在 else 里
    else:
        print("🚀 快照保存完毕。如需观看动画请设置 snapshot_times=None 或 save_anim=True")


if __name__ == "__main__":
    visualize_fds_results(
        npz_file='fire_env.npz',
        bounds=(0.0, 80.0, 0.0, 40.0),

        # ✨ 示例用法：
        # 1. 仅保存 60s, 120s, 300s 的高清截图
        snapshot_times=[200,250,300,350],
        save_anim=False

        # 2. 如果想看动画，把 snapshot_times 设为 None
        # snapshot_times=None,
        # save_anim=False
    )