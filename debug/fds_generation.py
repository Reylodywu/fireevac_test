import numpy as np
import math


def generate_fds_file(filename="subway_station.fds"):
    fds_content = []
    obstacles_bbox = []

    # ================= 1. 基础参数 =================
    grid_x, grid_y = 80.0, 40.0
    floor_z, ceiling_z = 0.0, 3.0

    # ---根据特征火源尺寸确定网格尺寸 ---
    # D* = 1.83m.
    # 设 res = 0.25m, Ratio = 7.32 (满足 4-16 范围，且能被 80 和 40 整除)
    res = 0.25

    # 扩展计算域
    buffer = 1.0
    mesh_xb = [-buffer, grid_x + buffer, -buffer, grid_y + buffer, -0.5, 4.0]

    # 辅助函数
    def add_obst(xb_coords, color='GRAY', desc='Obstacle', is_solid=True):
        line = f"&OBST XB={','.join(map(str, xb_coords))}, SURF_ID='INERT', COLOR='{color}' / {desc}"
        fds_content.append(line)
        if is_solid:
            obstacles_bbox.append({
                'x_min': min(xb_coords[0], xb_coords[1]),
                'x_max': max(xb_coords[0], xb_coords[1]),
                'y_min': min(xb_coords[2], xb_coords[3]),
                'y_max': max(xb_coords[2], xb_coords[3])
            })

    # ================= 2. 头部与网格 =================
    fds_content.append("&HEAD CHID='subway_station_refined', TITLE='Subway Station Refined Grid' /")
    fds_content.append("&TIME T_END=400.0 /")

    ijk = [int((mesh_xb[1] - mesh_xb[0]) / res), int((mesh_xb[3] - mesh_xb[2]) / res),
           int((mesh_xb[5] - mesh_xb[4]) / res)]
    fds_content.append(f"&MESH IJK={ijk[0]},{ijk[1]},{ijk[2]}, XB={','.join(map(str, mesh_xb))} /")

    for face in ['XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMAX']:
        fds_content.append(f"&VENT MB='{face}', SURF_ID='OPEN' /")

    # ================= 3. 火源与反应 (使用您提供的代码) =================
    fds_content.append("\n! ========== FIRE & REACTION ==========")

    # [核心修复] 使用 FORMULA 定义燃料组分，解决缺碳报错
    fds_content.append("&SPEC ID='SFPE POLYURETHANE_GM27_fuel', FORMULA='C1.0H1.7O0.3N0.08' /")

    fds_content.append("&REAC ID='SFPE POLYURETHANE_GM27',")
    fds_content.append("      FYI='SFPE Handbook, 5th Edition, Tables A.38 and A.39',")
    fds_content.append("      FUEL='SFPE POLYURETHANE_GM27_fuel',")
    fds_content.append("      CO_YIELD=0.042,")
    fds_content.append("      SOOT_YIELD=0.198,")
    fds_content.append("      HEAT_OF_COMBUSTION=1.64E+4,")
    fds_content.append("      RADIATIVE_FRACTION=0.537 /")

    # 火源参数
    Q_max = 5000.0
    alpha = 0.04689
    tau_q = -round(math.sqrt(Q_max / alpha), 2)

    # 定义火源表面
    fds_content.append(f"&SURF ID='FIRE', COLOR='RED', HRRPUA=5000.0, TAU_Q={tau_q} /")

    # 使用 SURF_ID6 定义行李箱
    fire_x, fire_y = 55.0, 20.0
    luggage_xb = [fire_x - 0.5, fire_x + 0.5, fire_y - 0.5, fire_y + 0.5, 0.0, 1.0]

    # SURF_ID6 顺序: xmin, xmax, ymin, ymax, zmin, zmax(顶面)
    fds_content.append(f"&OBST XB={','.join(map(str, luggage_xb))}, "
                       f"SURF_ID6='FIRE','FIRE','FIRE','FIRE','INERT','FIRE' / Luggage with Fire Top")

    # ================= 4. 建筑结构 =================
    fds_content.append("\n! ========== GEOMETRY ==========")
    add_obst([0, grid_x, 0, grid_y, -0.2, 0.0], 'GRAY', 'Floor', False)
    add_obst([0, grid_x, 0, grid_y, ceiling_z, ceiling_z + 0.2], 'GRAY', 'Ceiling', False)

    w_t = 0.2
    add_obst([0, grid_x, 0, w_t, 0, ceiling_z], 'IVORY', 'Wall Bot', True)
    add_obst([0, grid_x, grid_y - w_t, grid_y, 0, ceiling_z], 'IVORY', 'Wall Top', True)
    add_obst([0, w_t, 0, grid_y, 0, ceiling_z], 'IVORY', 'Wall Left', True)
    add_obst([grid_x - w_t, grid_x, 0, grid_y, 0, ceiling_z], 'IVORY', 'Wall Right', True)

    margin, d_w, d_h = 2.0, 4.0, 2.5
    fds_content.append(f"&HOLE XB={margin},{margin + d_w}, -0.1,{w_t + 0.1}, 0.0,{d_h} / Exit BL")
    fds_content.append(f"&HOLE XB={grid_x - margin - d_w},{grid_x - margin}, -0.1,{w_t + 0.1}, 0.0,{d_h} / Exit BR")
    fds_content.append(f"&HOLE XB={margin},{margin + d_w}, {grid_y - w_t - 0.1},{grid_y + 0.1}, 0.0,{d_h} / Exit TL")
    fds_content.append(
        f"&HOLE XB={grid_x - margin - d_w},{grid_x - margin}, {grid_y - w_t - 0.1},{grid_y + 0.1}, 0.0,{d_h} / Exit TR")

    # 楼梯 & 客服中心
    center_x, center_y = grid_x / 2, grid_y / 2
    paid_w, paid_h = grid_x * 0.7, grid_y * 0.6
    paid_x_min, paid_x_max = center_x - paid_w / 2, center_x + paid_w / 2
    paid_y_min, paid_y_max = center_y - paid_h / 2, center_y + paid_h / 2

    stair_w, stair_h = 6.0, 3.0
    stair_pad_x = 5.0

    # --- [修改点 2] 楼梯高度修正 ---
    # 不再延伸到 ceiling_z，而是设为一个合理的高度 (例如 1.2m)
    # 这样既能阻挡人员/形成障碍，又允许烟气在上方蔓延
    obst_stair_z_height = 1.2

    stair_xs = [paid_x_min + stair_pad_x, center_x - stair_w / 2, paid_x_max - stair_pad_x - stair_w]
    stair_ys = [paid_y_min, paid_y_max - stair_h]
    for x in stair_xs:
        for y in stair_ys:
            # 修改这里的 z2 参数
            add_obst([x, x + stair_w, y, y + stair_h, 0, obst_stair_z_height], 'GRAY', 'Stairs', True)

    svc_size = 4.0
    add_obst([center_x - svc_size / 2, center_x + svc_size / 2, paid_y_min - 2.0 - svc_size, paid_y_min - 2.0, 0, 1.2],
             'BLUE', 'Svc Bot', True)
    add_obst([center_x - svc_size / 2, center_x + svc_size / 2, paid_y_max + 2.0, paid_y_max + 2.0 + svc_size, 0, 1.2],
             'BLUE', 'Svc Top', True)

    # ================= 5. 闸机 (Gate) =================
    gate_len, gate_thick, gate_h = 2.0, 0.5, 1.2

    def create_gate_segment(is_vertical, fixed_pos, start_pos, end_pos, num_gates, wall_limits):
        total_span = end_pos - start_pos
        unit_size = total_span / num_gates
        for i in range(num_gates):
            base = start_pos + i * unit_size
            obs_start = base + (unit_size * 0.2)
            obs_end = obs_start + gate_thick
            p1, p2 = min(obs_start, obs_end), max(obs_start, obs_end)
            if is_vertical:
                add_obst([fixed_pos - gate_len / 2, fixed_pos + gate_len / 2, p1, p2, 0, gate_h], 'ORANGE', 'Gate',
                         True)
            else:
                add_obst([p1, p2, fixed_pos - gate_len / 2, fixed_pos + gate_len / 2, 0, gate_h], 'ORANGE', 'Gate',
                         True)

        first_gate_start = start_pos + (unit_size * 0.2)
        last_gate_end = start_pos + (num_gates - 1) * unit_size + (unit_size * 0.2) + gate_thick
        wall_t_gate = 0.2

        if is_vertical:
            add_obst([fixed_pos - wall_t_gate, fixed_pos + wall_t_gate, wall_limits[0], first_gate_start, 0, gate_h],
                     'ORANGE', 'Seal Wall', True)
            add_obst([fixed_pos - wall_t_gate, fixed_pos + wall_t_gate, last_gate_end, wall_limits[1], 0, gate_h],
                     'ORANGE', 'Seal Wall', True)
        else:
            add_obst([wall_limits[0], first_gate_start, fixed_pos - wall_t_gate, fixed_pos + wall_t_gate, 0, gate_h],
                     'ORANGE', 'Seal Wall', True)
            add_obst([last_gate_end, wall_limits[1], fixed_pos - wall_t_gate, fixed_pos + wall_t_gate, 0, gate_h],
                     'ORANGE', 'Seal Wall', True)

    v_gate_margin = (paid_y_max - paid_y_min) * 0.2
    gap = 2.0
    stair_L_right = stair_xs[0] + stair_w
    stair_C_left = stair_xs[1]
    stair_C_right = stair_xs[1] + stair_w
    stair_R_left = stair_xs[2]

    h_gate_left_start, h_gate_left_end = stair_L_right + gap, stair_C_left - gap
    h_gate_right_start, h_gate_right_end = stair_C_right + gap, stair_R_left - gap

    create_gate_segment(True, paid_x_min, paid_y_min + v_gate_margin, paid_y_max - v_gate_margin, 5,
                        (paid_y_min, paid_y_max))
    create_gate_segment(True, paid_x_max, paid_y_min + v_gate_margin, paid_y_max - v_gate_margin, 5,
                        (paid_y_min, paid_y_max))
    create_gate_segment(False, paid_y_min, h_gate_left_start, h_gate_left_end, 4, (paid_x_min, stair_C_left))
    create_gate_segment(False, paid_y_min, h_gate_right_start, h_gate_right_end, 4, (stair_C_right, paid_x_max))
    create_gate_segment(False, paid_y_max, h_gate_left_start, h_gate_left_end, 4, (paid_x_min, stair_C_left))
    create_gate_segment(False, paid_y_max, h_gate_right_start, h_gate_right_end, 4, (stair_C_right, paid_x_max))

    add_obst([stair_C_left, stair_C_right, paid_y_min - 0.2, paid_y_min + 0.2, 0, gate_h], 'ORANGE', 'Center Wall Bot',
             True)
    add_obst([stair_C_left, stair_C_right, paid_y_max - 0.2, paid_y_max + 0.2, 0, gate_h], 'ORANGE', 'Center Wall Top',
             True)

    # ================= 6. 全局结构柱 =================
    col_size = 1.0
    col_offset_y = 3.0
    col_x_positions = np.linspace(grid_x * 0.05, grid_x * 0.95, 10)

    def is_clear(cx, cy, s):
        margin = 0.1
        c_x1, c_x2 = cx - s / 2 - margin, cx + s / 2 + margin
        c_y1, c_y2 = cy - s / 2 - margin, cy + s / 2 + margin
        for obs in obstacles_bbox:
            if not (c_x2 < obs['x_min'] or c_x1 > obs['x_max'] or c_y2 < obs['y_min'] or c_y1 > obs['y_max']):
                return False
        return True

    for col_x in col_x_positions:
        y_top = center_y + col_offset_y
        y_bot = center_y - col_offset_y - col_size

        if is_clear(col_x, y_top, col_size):
            if abs(col_x - paid_x_min) > 1.5 and abs(col_x - paid_x_max) > 1.5:
                add_obst([col_x - col_size / 2, col_x + col_size / 2, y_top - col_size / 2, y_top + col_size / 2, 0,
                          ceiling_z], 'BLACK', 'Column', True)

        if is_clear(col_x, y_bot + col_size / 2, col_size):
            if abs(col_x - paid_x_min) > 1.5 and abs(col_x - paid_x_max) > 1.5:
                add_obst([col_x - col_size / 2, col_x + col_size / 2, y_bot, y_bot + col_size, 0, ceiling_z], 'BLACK',
                         'Column', True)

    # ================= 7. 输出 =================
    fds_content.append("\n! ========== OUTPUT ==========")
    fds_content.append("&DUMP DT_SLCF=1.0 /")
    fds_content.append("&SLCF PBZ=1.7, QUANTITY='TEMPERATURE' /")
    fds_content.append("&SLCF PBZ=1.7, QUANTITY='VISIBILITY' /")
    fds_content.append("&SLCF PBZ=1.7, QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON MONOXIDE' /")
    fds_content.append("&TAIL /")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fds_content))

    print(f"Generated {filename}")
    print(f"Mesh Res: {res}m (D*/dx = {1.83 / res:.2f})")


if __name__ == "__main__":
    generate_fds_file()