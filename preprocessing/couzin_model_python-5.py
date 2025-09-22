import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SwarmParameters:
    """群体模拟参数配置"""
    number_of_fish: int = 100
    repulsion_radius: float = 1.0
    orientation_width: float = 10.0
    attraction_width: float = 14.0
    angle_of_perception: float = math.pi * 270 / 180  # 弧度
    turning_rate: float = math.pi * 40 / 180  # 弧度/时间单位
    speed: float = 3.0
    noise_sigma: float = 0.05
    dt: float = 0.1
    box_lengths: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    reflect_at_boundary: Tuple[bool, bool, bool] = (True, True, True)  # 改为默认反射
    
    def __post_init__(self):
        """参数验证"""
        if self.number_of_fish <= 0:
            raise ValueError("鱼群数量必须大于0")
        if self.repulsion_radius <= 0:
            raise ValueError("排斥半径必须大于0")
        if self.orientation_width <= 0:
            raise ValueError("定向宽度必须大于0")
        if self.attraction_width <= 0:
            raise ValueError("吸引宽度必须大于0")
        if not 0 < self.angle_of_perception <= 2 * math.pi:
            raise ValueError("感知角度必须在0到2π之间")
        if self.speed <= 0:
            raise ValueError("游泳速度必须大于0")
        if self.dt <= 0:
            raise ValueError("时间步长必须大于0")
        if self.repulsion_radius >= self.orientation_width:
            raise ValueError("排斥半径应该小于定向宽度")
        if self.orientation_width >= self.attraction_width:
            raise ValueError("定向宽度应该小于吸引宽度")

class Fish:
    """单个鱼类个体"""
    
    def __init__(self, x: float, y: float, z: float):
        self.position = np.array([x, y, z], dtype=float)
        # 随机初始方向
        self.velocity = np.random.uniform(-1, 1, 3)
        self.velocity = self.velocity / np.linalg.norm(self.velocity)
        self.acceleration = np.zeros(3)
    
    def update(self, fishes: List['Fish'], params: SwarmParameters) -> None:
        """更新鱼的位置和方向"""
        # 初始化加速度
        self.acceleration = np.zeros(3)
        
        # 应用三个核心规则，获取期望方向
        repulsion_direction = self._separate(fishes, params.repulsion_radius)
        alignment_direction = self._align(fishes, params.orientation_width, params.angle_of_perception)
        cohesion_direction = self._cohesion(fishes, params.attraction_width, params.angle_of_perception)
        
        # 当前速度归一化
        current_vel_magnitude = np.linalg.norm(self.velocity)
        if current_vel_magnitude > 0:
            current_direction = self.velocity / current_vel_magnitude
        else:
            current_direction = np.array([1.0, 0.0, 0.0])
        
        # 计算期望的新方向（加权平均）
        desired_direction = np.zeros(3)
        total_weight = 0.0
        
        # 排斥力权重最高
        if np.linalg.norm(repulsion_direction) > 0:
            desired_direction += repulsion_direction * 3.0
            total_weight += 3.0
        
        # 对齐力权重中等
        if np.linalg.norm(alignment_direction) > 0:
            desired_direction += alignment_direction * 1.5
            total_weight += 1.5
        
        # 聚集力权重最低
        if np.linalg.norm(cohesion_direction) > 0:
            desired_direction += cohesion_direction * 1.0
            total_weight += 1.0
        
        # 如果没有任何邻居影响，保持当前方向
        if total_weight > 0:
            desired_direction = desired_direction / total_weight
            desired_magnitude = np.linalg.norm(desired_direction)
            if desired_magnitude > 0:
                desired_direction = desired_direction / desired_magnitude
            else:
                desired_direction = current_direction
        else:
            desired_direction = current_direction
        
        # 计算转向力（期望方向与当前方向的差异）
        steering_force = desired_direction - current_direction
        
        # 添加随机噪声
        if params.noise_sigma > 0:
            noise = np.random.normal(0, params.noise_sigma, 3)
            steering_force += noise
        
        # 限制转向速率
        steering_magnitude = np.linalg.norm(steering_force)
        if steering_magnitude > params.turning_rate:
            steering_force = steering_force / steering_magnitude * params.turning_rate
        
        # 更新速度方向
        new_velocity = self.velocity + steering_force * params.dt
        
        # 归一化速度并应用恒定速度大小
        new_vel_magnitude = np.linalg.norm(new_velocity)
        if new_vel_magnitude > 0:
            self.velocity = new_velocity / new_vel_magnitude * params.speed
        else:
            # 如果速度为0，保持原来的方向或给随机方向
            if current_vel_magnitude > 0:
                self.velocity = current_direction * params.speed
            else:
                random_dir = np.random.uniform(-1, 1, 3)
                random_dir = random_dir / np.linalg.norm(random_dir)
                self.velocity = random_dir * params.speed
        
        # 更新位置
        self.position += self.velocity * params.dt
        
        # 边界处理
        self._handle_boundaries(params.box_lengths, params.reflect_at_boundary)
    
    def _separate(self, fishes: List['Fish'], radius: float) -> np.ndarray:
        """排斥行为：避免与邻居过近"""
        repulsion_force = np.zeros(3)
        count = 0
        
        for other in fishes:
            if other is self:
                continue
                
            diff = self.position - other.position
            distance = np.linalg.norm(diff)
            
            if 0 < distance < radius:
                # 排斥力方向：从邻居指向自己
                repulsion_direction = diff / distance
                # 距离越近，排斥力越强 (1/distance)
                repulsion_strength = 1.0 / distance
                repulsion_force += repulsion_direction * repulsion_strength
                count += 1
        
        if count > 0:
            # 归一化返回单位方向
            repulsion_magnitude = np.linalg.norm(repulsion_force)
            if repulsion_magnitude > 0:
                return repulsion_force / repulsion_magnitude
        
        return np.zeros(3)
    
    def _align(self, fishes: List['Fish'], radius: float, angle_of_perception: float) -> np.ndarray:
        """对齐行为：与邻居保持相同方向"""
        average_velocity = np.zeros(3)
        count = 0
        
        # 当前速度归一化
        current_vel_magnitude = np.linalg.norm(self.velocity)
        if current_vel_magnitude == 0:
            return np.zeros(3)
        current_vel_norm = self.velocity / current_vel_magnitude
        
        for other in fishes:
            if other is self:
                continue
                
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < radius:
                # 检查是否在感知角度内
                to_other = other.position - self.position
                to_other_norm = to_other / distance
                
                # 计算角度 
                dot_product = np.dot(current_vel_norm, to_other_norm)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = math.acos(dot_product)
                
                if angle <= angle_of_perception / 2:
                    average_velocity += other.velocity
                    count += 1
        
        if count > 0:
            average_velocity = average_velocity / count
            # 归一化平均速度方向
            avg_vel_magnitude = np.linalg.norm(average_velocity)
            if avg_vel_magnitude > 0:
                return average_velocity / avg_vel_magnitude
        
        return np.zeros(3)
    
    def _cohesion(self, fishes: List['Fish'], radius: float, angle_of_perception: float) -> np.ndarray:
        """聚集行为：向邻居的中心移动"""
        center = np.zeros(3)
        count = 0
        
        # 当前速度归一化
        current_vel_magnitude = np.linalg.norm(self.velocity)
        if current_vel_magnitude == 0:
            return np.zeros(3)
        current_vel_norm = self.velocity / current_vel_magnitude
        
        for other in fishes:
            if other is self:
                continue
                
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < radius:
                # 检查是否在感知角度内
                to_other = other.position - self.position
                to_other_norm = to_other / distance
                
                # 计算角度
                dot_product = np.dot(current_vel_norm, to_other_norm)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = math.acos(dot_product)
                
                if angle <= angle_of_perception / 2:
                    center += other.position
                    count += 1
        
        if count > 0:
            center = center / count
            return self._seek(center)
        
        return np.zeros(3)
    
    def _seek(self, target: np.ndarray) -> np.ndarray:
        """寻找目标位置"""
        desired = target - self.position
        desired_magnitude = np.linalg.norm(desired)
        if desired_magnitude > 0:
            desired = desired / desired_magnitude
            # 返回归一化的期望方向
            return desired
        return np.zeros(3)
    
    def _handle_boundaries(self, box_lengths: Tuple[float, float, float], 
                         reflect_at_boundary: Tuple[bool, bool, bool]) -> None:
        """处理边界条件"""
        width, height, depth = box_lengths
        
        # 检查是否超出边界并处理
        if self.position[0] < -width/2:
            if reflect_at_boundary[0]:
                self.position[0] = -width/2
                self.velocity[0] = -self.velocity[0]
            else:
                self.position[0] = width/2
                
        elif self.position[0] > width/2:
            if reflect_at_boundary[0]:
                self.position[0] = width/2
                self.velocity[0] = -self.velocity[0]
            else:
                self.position[0] = -width/2
                
        if self.position[1] < -height/2:
            if reflect_at_boundary[1]:
                self.position[1] = -height/2
                self.velocity[1] = -self.velocity[1]
            else:
                self.position[1] = height/2
                
        elif self.position[1] > height/2:
            if reflect_at_boundary[1]:
                self.position[1] = height/2
                self.velocity[1] = -self.velocity[1]
            else:
                self.position[1] = -height/2
                
        if self.position[2] < -depth/2:
            if reflect_at_boundary[2]:
                self.position[2] = -depth/2
                self.velocity[2] = -self.velocity[2]
            else:
                self.position[2] = depth/2
                
        elif self.position[2] > depth/2:
            if reflect_at_boundary[2]:
                self.position[2] = depth/2
                self.velocity[2] = -self.velocity[2]
            else:
                self.position[2] = -depth/2

class CouzinSwarmSimulation:
    """Couzin群体模拟器"""
    
    def __init__(self, params: SwarmParameters):
        self.params = params
        self.fishes = []
        self.time_step = 0
        self.simulation_data = []
        
        # 初始化鱼群
        self._initialize_fishes()
    
    def _initialize_fishes(self) -> None:
        """初始化鱼群位置和方向"""
        self.fishes = []
        width, height, depth = self.params.box_lengths
        
        for i in range(self.params.number_of_fish):
            # 随机位置（在边界内）
            x = np.random.uniform(-width/2 * 0.8, width/2 * 0.8)
            y = np.random.uniform(-height/2 * 0.8, height/2 * 0.8)
            z = np.random.uniform(-depth/2 * 0.8, depth/2 * 0.8)
            
            fish = Fish(x, y, z)
            self.fishes.append(fish)
    
    def simulate(self, n_steps: int) -> pd.DataFrame:
        """
        运行模拟并记录所有步骤的数据
        
        Parameters:
        -----------
        n_steps : int
            模拟步数
            
        Returns:
        --------
        pd.DataFrame
            包含所有时间步的位置和方向数据
        """
        print(f"开始模拟 {n_steps} 步，包含 {self.params.number_of_fish} 只鸟...")
        
        # 记录初始状态
        self._record_current_state()
        
        # 运行模拟
        for step in range(n_steps):
            if step % 500 == 0:
                print(f"进度: {step}/{n_steps} ({step/n_steps*100:.1f}%)")
            
            # 更新所有鱼的状态
            for fish in self.fishes:
                fish.update(self.fishes, self.params)
            
            self.time_step += 1
            
            # 记录当前状态
            self._record_current_state()
        
        print("模拟完成！")
        
        # 转换为DataFrame
        df = pd.DataFrame(self.simulation_data)
        return df
    
    # def _record_current_state(self) -> None:
    #     """记录当前时间步的所有鱼的状态"""
    #     for fish_id, fish in enumerate(self.fishes):
    #         record = {
    #             'time_step': self.time_step,
    #             'fish_id': fish_id,
    #             'position_x': fish.position[0],
    #             'position_y': fish.position[1],
    #             'position_z': fish.position[2],
    #             'velocity_x': fish.velocity[0],
    #             'velocity_y': fish.velocity[1],
    #             'velocity_z': fish.velocity[2],
    #             'speed': np.linalg.norm(fish.velocity)
    #         }
    #         self.simulation_data.append(record)
    
    def _record_current_state(self) -> None:
        """
        记录当前时间步的状态。
        """
        record = {'time_step': self.time_step}

        # 2. 遍历每一只鸟 (fish) 并展开其状态
        for i, fish in enumerate(self.fishes):
            # 为第 i 只鸟的位置 (x, y, z) 创建键值对
            record[f'pos_bird{i}_x'] = fish.position[0]
            record[f'pos_bird{i}_y'] = fish.position[1]
            record[f'pos_bird{i}_z'] = fish.position[2]
            
            # 为第 i 只鸟的速度 (vx, vy, vz) 创建键值对
            record[f'vel_bird{i}_x'] = fish.velocity[0]
            record[f'vel_bird{i}_y'] = fish.velocity[1]
            record[f'vel_bird{i}_z'] = fish.velocity[2]

        # 3. 将这个完全扁平化的字典追加到总的模拟数据列表中
        self.simulation_data.append(record)

    def calculate_polarization(self) -> float:
        """计算群体极化值"""
        if not self.fishes:
            return 0.0
        
        sum_velocity = np.sum([fish.velocity for fish in self.fishes], axis=0)
        polarization = np.linalg.norm(sum_velocity) / len(self.fishes)
        return polarization
    
    def calculate_statistics(self) -> dict:
        """计算群体统计信息"""
        if not self.fishes:
            return {}
        
        positions = np.array([fish.position for fish in self.fishes])
        velocities = np.array([fish.velocity for fish in self.fishes])
        
        # 计算中心位置
        center_of_mass = np.mean(positions, axis=0)
        
        # 计算分散度
        distances_from_center = np.linalg.norm(positions - center_of_mass, axis=1)
        dispersion = np.mean(distances_from_center)
        
        # 计算平均速度
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        
        # 计算极化值
        polarization = self.calculate_polarization()
        
        return {
            'time_step': self.time_step,
            'center_of_mass': center_of_mass,
            'dispersion': dispersion,
            'average_speed': avg_speed,
            'polarization': polarization,
            'number_of_fish': len(self.fishes)
        }
    
    def save_results(self, filename: str, include_statistics: bool = False) -> None:
        """
        保存模拟结果到CSV文件
        
        Parameters:
        -----------
        filename : str
            输出文件名
        include_statistics : bool
            是否同时保存统计信息
        """
        if not self.simulation_data:
            print("警告：没有模拟数据可以保存")
            return
        
        # 确保目录存在
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存详细数据
        df = pd.DataFrame(self.simulation_data)
        df.to_csv(filename, index=False)
        print(f"详细数据已保存到: {filename}")
        print(f"数据维度: {df.shape}")
        print(f"时间步范围: {df['time_step'].min()} - {df['time_step'].max()}")
        #print(f"鱼类ID范围: {df['fish_id'].min()} - {df['fish_id'].max()}")
        
        # 保存每个时间步的统计信息
        if include_statistics:
            stats_filename = filename.replace('.csv', '_statistics.csv')
            self._save_time_series_statistics(stats_filename)
    
    # def _save_time_series_statistics(self, filename: str) -> None:
    #     """保存每个时间步的统计信息"""
    #     df = pd.DataFrame(self.simulation_data)
        
    #     # 按时间步分组计算统计量
    #     time_stats = []
    #     for time_step in df['time_step'].unique():
    #         step_data = df[df['time_step'] == time_step]
    #         # # 计算当前时间步的统计量
    #         # positions = step_data[['position_x', 'position_y', 'position_z']].values
    #         # velocities = step_data[['velocity_x', 'velocity_y', 'velocity_z']].values
    #         positions = step_data['positions_vector'][time_step].reshape(-1, 3)
    #         velocities = step_data['velocities_vector'][time_step].reshape(-1, 3)
            
    #         # 中心位置
    #         center_of_mass = np.mean(positions, axis=0)
            
    #         # 分散度
    #         distances_from_center = np.linalg.norm(positions - center_of_mass, axis=1)
    #         dispersion = np.mean(distances_from_center)
            
    #         # # 平均速度
    #         # avg_speed = np.mean(step_data['speed'])
            
    #         # 极化值
    #         sum_velocity = np.sum(velocities, axis=0)
    #         polarization = np.linalg.norm(sum_velocity) / len(velocities)
            
    #         time_stats.append({
    #             'time_step': time_step,
    #             'center_of_mass_x': center_of_mass[0],
    #             'center_of_mass_y': center_of_mass[1],
    #             'center_of_mass_z': center_of_mass[2],
    #             'dispersion': dispersion,
    #             'polarization': polarization,
    #             'number_of_fish': len(step_data)
    #         })
        
    #     stats_df = pd.DataFrame(time_stats)
    #     stats_df.to_csv(filename, index=False)
    #     print(f"时间序列统计信息已保存到: {filename}")
    #     print(f"最终极化值: {stats_df['polarization'].iloc[-1]:.3f}")
    #     print(f"平均极化值: {stats_df['polarization'].mean():.3f}")
    #     print(f"最大极化值: {stats_df['polarization'].max():.3f}")
    
    
    def reset(self) -> None:
        """重置模拟"""
        self.time_step = 0
        self.simulation_data = []
        self._initialize_fishes()

def main():
    """主函数示例"""
    global n_birds
    # 设置随机种子以确保可重现结果
    np.random.seed(42)
    n_birds = 2
    #swarm
    params = SwarmParameters(
        number_of_fish=n_birds,
        repulsion_radius=1.0,
        orientation_width=1.1,
        attraction_width=15.1,
        angle_of_perception=math.pi * 270 / 180,
        turning_rate=math.pi * 40 / 180,
        speed=2.0,
        noise_sigma=0.05,
        dt=0.1,
        box_lengths=(50.0, 50.0, 50.0)
    )
    
    print("=== Couzin群体模拟参数 ===")
    print(f"鱼群数量: {params.number_of_fish}")
    print(f"排斥半径: {params.repulsion_radius}")
    print(f"定向宽度: {params.orientation_width}")
    print(f"吸引宽度: {params.attraction_width}")
    print(f"感知角度: {params.angle_of_perception * 180 / math.pi:.1f}°")
    print(f"转向速率: {params.turning_rate * 180 / math.pi:.1f}°/s")
    print(f"游泳速度: {params.speed}")
    print(f"噪声强度: {params.noise_sigma}")
    print(f"时间步长: {params.dt}")
    print(f"边界大小: {params.box_lengths}")
    print()
    
    # 创建模拟器
    simulation = CouzinSwarmSimulation(params)
    
    # 运行模拟
    n_steps = 100  
    results_df = simulation.simulate(n_steps)

    
    # 保存结果
    output_filename = "dataset/Couzin/couzin_simulation.csv"
    simulation.save_results(output_filename)
    
    # 打印一些统计信息
    print(f"\n=== 模拟总结 ===")
    print(f"总时间步数: {n_steps + 1}")
    print(f"鱼群数量: {params.number_of_fish}")
    print(f"数据记录条数: {len(results_df)}")
    
    # 计算最终统计信息
    final_stats = simulation.calculate_statistics()
    print(f"最终极化值: {final_stats['polarization']:.3f}")
    print(f"最终分散度: {final_stats['dispersion']:.3f}")
    print(f"最终平均速度: {final_stats['average_speed']:.3f}")
    print(f"最终质心位置: ({final_stats['center_of_mass'][0]:.2f}, {final_stats['center_of_mass'][1]:.2f}, {final_stats['center_of_mass'][2]:.2f})")
    
    # 显示数据样本
    print(f"\n=== 数据样本 ===")
    print(results_df.head(10))
    print(f"\n=== 数据描述 ===")
    print(results_df.describe())
    
    return results_df

if __name__ == "__main__":
    results = main()
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='3d')
    marker_interval = 100  # 每 20 步做一个标记
    #colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    cmap = plt.get_cmap('plasma') 
    colors = [cmap(i) for i in np.linspace(0, 1, n_birds)]
    for bird in range(n_birds):
        start_col = bird * 6
        pos_cols = list(range(start_col, start_col + 3))  # 这只鸟的位置x,y,z
        # 假定DataFrame无列名可直接iloc，否则你可以用对应列名
        x = results.iloc[:, pos_cols[0]+1].values
        y = results.iloc[:, pos_cols[1]+1].values
        z = results.iloc[:, pos_cols[2]+1].values
        ax.plot3D(x, y, z, label=f'Bird {bird + 1}', color=colors[bird], alpha=min(1,2/n_birds), linewidth=2)
        # 标记起点
        ax.scatter(x[0], y[0], z[0], color=colors[bird], marker='o', s=60, edgecolors='k')
        # 标记终点
        ax.scatter(x[-1], y[-1], z[-1], color=colors[bird], marker='X', s=80, edgecolors='k')

        # marker_indices = range(marker_interval, len(x), marker_interval)
        # ax.scatter(x[marker_indices], y[marker_indices], z[marker_indices], 
        #            color=colors[bird % len(colors)], 
        #            marker='^',  # 使用三角形作为间隔标记
        #            s=40,        # 标记大小
        #            edgecolors='k', # 加个黑边更清晰
        #            alpha=0.8)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Birds 3D Trajectories')
    ax.legend()
    plt.tight_layout()
    plt.savefig('preprocessing/data_plot.png', dpi=300)  # 保存为高分辨率PNG
    plt.close()  # 关闭窗口，防止重复显示
