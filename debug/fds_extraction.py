import numpy as np
import fdsreader
from visualize_fds import visualize_fds_results

def extract_fds_data_strict(
        case_path='.',
        target_quantities=None,
        target_z=1.6,
        target_bounds=(0.0, 50.0, 0.0, 35.0),
        resolution=0.2,  # ✨ 新增：显式指定分辨率
        output_file='fire_env.npz'
):
    # 1. 默认配置
    if target_quantities is None:
        target_quantities = {
            'TEMPERATURE': 'temperature',
            'VISIBILITY': 'visibility',
            'CARBON MONOXIDE': 'co',
            'SOOT DENSITY': 'soot'
        }

    # 2. 计算严格的期望形状
    xmin, xmax, ymin, ymax = target_bounds

    # 期望的网格数 = 总长度 / 分辨率 (强制取整)
    expected_nx = int(round((xmax - xmin) / resolution))  # 50/0.2 = 250
    expected_ny = int(round((ymax - ymin) / resolution))  # 35/0.2 = 175

    print(f"🚀 开始处理: {case_path}")
    print(f"🎯 目标高度: Z={target_z}m")
    print(f"📏 强制分辨率: {resolution}m")
    print(f"📐 期望输出形状 (Y, X): ({expected_ny}, {expected_nx})")

    try:
        sim = fdsreader.Simulation(case_path)
    except Exception as e:
        print(f"❌ 读取FDS失败: {e}")
        return

    result_data = {'times': None}

    for slc in sim.slices:
        # --- 过滤逻辑 (保持不变) ---
        if abs(slc.extent.z_start - slc.extent.z_end) > 0.001: continue
        if abs(slc.extent.z_start - target_z) > 0.1: continue

        quantity_name = slc.quantity.name
        save_key = None
        for key, val in target_quantities.items():
            if key in quantity_name:
                save_key = val
                break

        if not save_key: continue

        print(f"Processing: {quantity_name} -> {save_key}")

        try:
            # 获取数据和坐标
            out = slc.to_global(return_coordinates=True)
            if len(out) == 4:
                data, x_coords, y_coords, z_coords = out
            elif len(out) == 2:
                data, coords = out
                x_coords, y_coords = coords['x'], coords['y']
            else:
                raise ValueError("未知返回格式")

            if data.ndim == 4: data = np.squeeze(data, axis=3)

            # 保存时间
            if result_data['times'] is None:
                result_data['times'] = slc.times

            # --- ✨ 核心修改：基于起始点 + 固定步长截取 ---

            # 1. 寻找起始索引 (找到第一个 >= xmin 的点)
            # 使用 argmax 可以在布尔数组中快速找到第一个 True
            x_start_mask = x_coords >= (xmin - 0.01)
            y_start_mask = y_coords >= (ymin - 0.01)

            if not np.any(x_start_mask) or not np.any(y_start_mask):
                print(f"⚠️ 数据不在范围内，跳过")
                continue

            ix_start = np.argmax(x_start_mask)
            iy_start = np.argmax(y_start_mask)

            # 2. 强制计算结束索引
            ix_end = ix_start + expected_nx
            iy_end = iy_start + expected_ny

            # 3. 边界检查 (防止索引越界)
            current_nx = data.shape[1]
            current_ny = data.shape[2]

            if ix_end > current_nx:
                print(f"⚠️ X方向数据不足! 需要索引到 {ix_end}, 但只有 {current_nx}。将截断数据。")
                ix_end = current_nx

            if iy_end > current_ny:
                print(f"⚠️ Y方向数据不足! 需要索引到 {iy_end}, 但只有 {current_ny}。将截断数据。")
                iy_end = current_ny

            # 4. 裁剪
            cropped_data = data[:, ix_start:ix_end, iy_start:iy_end]

            # 5. 转置 (Time, Y, X)
            cropped_data = np.transpose(cropped_data, (0, 2, 1))

            # 6. 二次验证形状
            actual_ny, actual_nx = cropped_data.shape[1], cropped_data.shape[2]
            if actual_ny != expected_ny or actual_nx != expected_nx:
                print(f"❌ 形状警告: 实际 {actual_ny}x{actual_nx} != 期望 {expected_ny}x{expected_nx}")
                # 可选：如果差1-2个像素，可以在这里做 padding 或者 resize，但通常最好检查原始数据

            result_data[save_key] = cropped_data
            print(f"   Shape: {cropped_data.shape} (Time, Y, X) ✅")

        except Exception as e:
            print(f"⚠️ 出错: {e}")
            import traceback
            traceback.print_exc()

    # 保存
    if len(result_data) > 1:
        np.savez_compressed(output_file, **result_data)
        print(f"\n✅ 数据已保存至: {output_file}")
        if 'temperature' in result_data and result_data['temperature'] is not None:
            print(f"📊 最终验证: {result_data['temperature'].shape}")
    else:
        print("\n⚠️ 未提取到数据")


if __name__ == "__main__":
    extract_fds_data_strict(
        case_path='./subway_station',
        target_z=1.7,
        target_bounds=(0.0, 80.0, 0.0, 40.0),
        resolution=0.25  # ✨ 这里设置 FDS 的网格分辨率
    )
