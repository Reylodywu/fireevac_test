import numpy as np
import os
from scipy.ndimage import gaussian_filter, zoom


class FireSystem:
    def __init__(self, fds_data_path, grid_size_x, grid_size_y, fire_source, fds_dt=1.0, fds_time_step=None, fixed_fire=False):
        self.fds_data_path = fds_data_path
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.fire_source = np.array(fire_source, dtype=np.float32)
        self.fds_dt = fds_dt
        self.initial_fds_step = fds_time_step

        # 设定起始步数（如果是 None 则从 0 开始）
        self.initial_fds_step = fds_time_step if fds_time_step is not None else 0
        self.is_fixed_mode = fixed_fire
        self.current_fds_step = 0

        # 分辨率设置
        self.fds_resolution = 0.25
        self.query_resolution = 0.1

        # 数据容器
        self.fds_data = self._load_fds_data(fds_data_path) if fds_data_path else None
        self.original_env_factor_map = None
        self.high_res_env_factor_map = None
        self.temperature_map = None
        self.toxic_gas_map = None
        self.visibility_map = None

        # 统计值
        self.temp_min = 25.0
        self.temp_max = 25.0

    def reset(self):
        # 重置回初始设定步数
        self.current_fds_step = self.initial_fds_step
        self.update(force=True)

    def update(self, step_increment=0, force=False):
        """更新火灾状态"""
        # 只有在显式开启 fixed_mode 且非强制更新时，才跳过
        if self.is_fixed_mode and not force:
            return

        if not force and step_increment == 0:
            return

        # 执行更新
        self.current_fds_step += step_increment
        self._generate_fire_data()
        self._update_high_res_env_factor_map()

    def _load_fds_data(self, npz_path):
        if not os.path.exists(npz_path):
            print(f"✗ FDS数据文件不存在: {npz_path}")
            return None
        try:
            data = np.load(npz_path)
            return {k: data[k] for k in ['times', 'temperature', 'visibility', 'co']}
        except Exception as e:
            print(f"✗ FDS数据加载失败: {e}")
            return None

    def _generate_fire_data(self):
        if self.fds_data is not None:
            t = min(self.current_fds_step, len(self.fds_data['times']) - 1)
            self.temperature_map = np.nan_to_num(self.fds_data['temperature'][t], nan=25.0)
            self.visibility_map = np.nan_to_num(self.fds_data['visibility'][t], nan=30.0)
            self.toxic_gas_map = np.nan_to_num(self.fds_data['co'][t], nan=0.0)
        else:
            # 模拟数据生成
            fire_x, fire_y = self.fire_source
            y, x = np.ogrid[:self.grid_size_y, :self.grid_size_x]
            distance = np.sqrt((x - fire_x) ** 2 + (y - fire_y) ** 2)
            self.temperature_map = 25.0 + 575.0 * np.exp(-distance * 0.4)
            self.toxic_gas_map = 0.1 * np.exp(-distance * 0.5)
            self.visibility_map = np.maximum(1.0, 30.0 - 25.0 * np.exp(-distance * 0.4))

        self.temp_min = self.temperature_map.min()
        self.temp_max = self.temperature_map.max()

    def _update_high_res_env_factor_map(self):
        env_factor_low = self._compute_env_factor_low()
        # 此时 env_factor_low 分辨率为 fds_resolution (160*320)
        env_factor_smoothed = self._smooth_env_factor_map_gaussian(env_factor_low, sigma=1.5)
        self.original_env_factor_map = env_factor_smoothed

        scale_factor = self.fds_resolution / self.query_resolution
        self.high_res_env_factor_map = zoom(env_factor_smoothed, zoom=scale_factor, order=1, mode='nearest')
        self.high_res_env_factor_map = np.clip(self.high_res_env_factor_map, 0.001, 1.0)

    def _compute_env_factor_low(self):
        # --- 1. CO 因子 (基于公式9: 单位%, RefTime=15min) ---
        co_pct = self.toxic_gas_map * 100.0
        co_f = np.ones_like(co_pct)
        exposure_time = 5.0  # 分钟
        # 中等浓度 (0.1%~0.25%): 应用衰减公式
        mask_co = (co_pct >= 0.1) & (co_pct < 0.25)
        co_f[mask_co] = 1.0 - (0.2125 + 1.788 * co_pct[mask_co]) * co_pct[mask_co] * exposure_time

        # 高浓度 (>=0.25%): 直接致死
        co_f[co_pct >= 0.25] = 0.0

        # --- 2. 温度因子 (简化后的分段函数) ---
        T = self.temperature_map
        temp_f = np.ones_like(T)

        # 30~60度: 初始下降
        mask_warm = (T >= 30) & (T < 60)
        temp_f[mask_warm] = (3.8 * ((T[mask_warm] - 30) / 30.0) ** 2 + 1) / 1.2

        # 60~120度: 急剧下降
        mask_hot = (T >= 60) & (T < 120)
        temp_f[mask_hot] = 4.167 * (1 - ((T[mask_hot] - 60) / 60.0) ** 2)  # 4.167 = 5.0/1.2

        # >120度: 致死
        temp_f[T >= 120] = 0.0

        # --- 3. 可见度因子 (初始化为0.2保底) ---
        V = self.visibility_map
        vis_f = np.full_like(V, 0.2)

        # >=3m: 清晰
        vis_f[V >= 3.0] = 1.0

        # 0.65m~3m: 线性衰减
        mask_vis = (V >= 0.65) & (V < 3.0)
        vis_f[mask_vis] = 1.0 - 0.34 * (3.0 - V[mask_vis])

        # --- 4. 融合与截断 ---
        # 必须 clip(0, 1) 防止 CO 计算结果出现负数
        env_factor = co_f * temp_f
        return np.clip(env_factor, 0.001, 1.0)

    def _smooth_env_factor_map_gaussian(self, env_factor_map, sigma=1.5):
        if env_factor_map.max() - env_factor_map.min() < 1e-6:
            return env_factor_map
        return gaussian_filter(env_factor_map, sigma=sigma, mode='nearest')

    def get_env_factor_at_position(self, pos):
        map_x = int(pos[0] / self.query_resolution)
        map_y = int(pos[1] / self.query_resolution)
        if self.high_res_env_factor_map is None: return 1.0
        H, W = self.high_res_env_factor_map.shape
        map_x = np.clip(map_x, 0, W - 1)
        map_y = np.clip(map_y, 0, H - 1)
        return float(self.high_res_env_factor_map[map_y, map_x])