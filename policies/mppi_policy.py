from typing import List

import numpy as np
import scipy.signal
import torch
from scipy.spatial.transform import Rotation

import mppi
#========================================================
#【STORM】 这里导入我们在 pick_dyn_obstacles.py 里写好的函数
#========================================================
from mppi.pick_dyn_obstacles import (
    rollout_storm_no_loop,
    getCollisions
)
#========================================================
#【STORM】 这里导入我们在 pick_dyn_lifted_obstacles.py 里写好的函数
#========================================================
from mppi.pick_dyn_lifted_obstacles import rollout_storm_no_loop_lifted
from mppi.pick_dyn_door_obstacles import rollout_storm_no_loop_door

from env_ext.fetch import MPCControlGoalEnv
from mppi.pick_dyn_door_obstacles import getCollisions
from policies.policy import Policy


# TODO get this working for better performance than scipy
def savgol_filter_2d(tensor, window_length, polyorder, axis=0, deriv=0, delta=1.0):
    pass


class MPPIPolicy(Policy):
    Vector = Policy.Vector
    InfoVector = Policy.InfoVector
    LastPosition = np.zeros((3,))
    grip_action = -0.8

    def __init__(self, args):
        (
            self.K,
            self.T,
            self.Δt,
            self.α,
            self.F, # 单步 dynamics
            self.q, # 如果不再用旧的 state_cost，可为 None
            self.ϕ, # 如果不再用旧的 terminal_cost，可为 None
            self.Σ,
            self.λ,
            self.convert_to_target,
            self.dtype,
            self.device,
            self.dynamics_params  # 【STORM】包含 chain/robot_base_pos/向量化代价函数 等
        ) = mppi.get_mppi_parameters(args)


        # noise_mean_distribution = torch.zeros(3, dtype=self.dtype, device=self.device)

        # 分布初始化 (7维，对应关节速度增量)
        noise_mean_distribution = torch.zeros(7, dtype=self.dtype, device=self.device)
        self.noise_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=noise_mean_distribution, covariance_matrix=self.Σ)

        self.γ = self.λ * (1 - self.α)
        self.Σ_inv = torch.inverse(self.Σ)

        # MPPI 中的控制序列: (T,7)
        self.u = torch.zeros((self.T, 7), dtype=self.dtype, device=self.device)
        self.u_init = torch.zeros(7, dtype=self.dtype, device=self.device)

        self.prev_u = []

        # 用于可视化或调试
        self.collisions = []
        self.trajectory = None  # 外部可能会注入：每个时间步的“子目标”

        self.trajectory_rollouts = None
        self.obstacle_positions = None
        self.interim_goals = None
        self.target = None         # 用于记录下一步期望末端位置(可视化)


        #==================================================
        # (单层退火修改) 新增一个退火因子: horizon_diffuse_factor
        #==================================================
        # 这一因子决定了在第 i 个时间步时，我们对采样噪声乘以多少倍
        # 默认为 1.05，也可由外部 args 中带进来
        #==================================================
        self.horizon_diffuse_factor = getattr(args, 'horizon_diffuse_factor', 1.035) # 1.05
        # 【STORM + 单层退火(内层)】pick_dyn_front_v_static_groove_obstacles.py： #1.3 #1.35 #1.035
        # 【STORM + 单层退火(内层)】pick_dyn_front_u_static_groove_obstacles.py  #1.38 #1.35 #1.035
        # 【STORM + 单层退火(内层)】pick_dyn_rear_v_static_groove_obstacles.py #1.38 #1.35 #1.035

    def initial_info(self, obs: Vector) -> InfoVector:
        pass

    def reset(self):
        self.u = torch.zeros((self.T, 7), dtype=self.dtype, device=self.device)
        self.LastPosition = np.zeros((3,))

    def predict(self, obs: Vector) -> (Vector, InfoVector):
        return self.predict_with_goal(obs, obs[0]['desired_goal'])

    def set_envs(self, envs: List[MPCControlGoalEnv]):
        super().set_envs(envs)
        for env in envs:
            env.disable_action_limit()

    def predict_with_goal(self, obs: Vector, goal) -> (Vector, InfoVector):
        """
        跟原先一样，先解析观测，得到关节状态，再调用 update_control() 更新 MPPI 控制序列 self.u。
        取 self.u[0] 用 convert_to_target() 得到下一步末端位姿，然后与当前末端位姿做差作为 action。
        """
        # x_init, obstacle_positions = self.parse_observation(obs[0], goal)
        x_init, joint_init, obstacle_positions = self.parse_observation(obs[0], goal)
        goal_tensor = torch.tensor(goal, device=self.device)

        # shift the control inputs
        # 1) 将控制序列向前滚动 (往前 shift 一格)
        self.u = torch.roll(self.u, -1, dims=0)
        # 2) 把最后一帧初始化为 self.u_init (通常是0)
        self.u[-1] = self.u_init

        #self.update_control(joint_init, goal, obstacle_positions)
        # 3) 更新整条控制序列 (STORM式: 无循环rollout)
        self.update_control(joint_init, goal_tensor, obstacle_positions)

        # 4) 用更新后的 self.u[0] 作为当前时刻控制, 计算期望末端位姿
        #【GPU】(target - x_init) 在 GPU 上，需要先 .cpu() 再 .numpy()
        target = self.convert_to_target(joint_init, self.u[0])
        action = (target - x_init).cpu().numpy()  # 末端姿 - 当前末端姿 = 关节动作？
        #action_tensor = (self.convert_to_target(joint_init, self.u[0]) - x_init)  # GPU张量
        #action = action_tensor.cpu().numpy()  #【GPU】将结果拷回CPU并转成numpy

        #target = self.convert_to_target(joint_init, self.u[0])

        #action = (target - x_init).numpy()  # Action is the difference between current Position and Target Position

        # Calculate Real Target is Used for Vizualizing the desired Position in the next Timestep for Visualization
        # 5) 记录下一步期望末端位姿(用于可视化)
        self.target = self.calculate_real_target(joint_init, self.u[0])

        return [np.append(action, self.grip_action)], [{'direction': 'forward'}]

    def calculate_real_target(self, x, u):
        import os
        import pytorch_kinematics

        MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')

        xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
        dtype = torch.double
        chain = pytorch_kinematics.build_serial_chain_from_urdf(xml, end_link_name="panda_link8",
                                                                root_link_name="panda_link0")
        chain = chain.to(dtype=dtype)

        # Translational offset of Robot into World Coordinates
        robot_base_pos = torch.tensor([0.8, 0.75, 0.44])

        joint_pos = x[0:7]
        joint_vel = x[7:14]
        new_vel = joint_vel + u  # / (1 - torch.exp(torch.tensor(-0.01 / 0.100)))  # 0.175

        joint_pos = joint_pos + new_vel * self.Δt  # Calculate new Target Joint Positions
        # Calculate World Coordinate Target Position
        ret = chain.forward_kinematics(joint_pos, end_only=True)
        eef_matrix = ret.get_matrix()
        eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # Calculate World Coordinate Target
        eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])

        return torch.cat((eef_pos, eef_rot), dim=1)

    def parse_observation(self, obs, goal):
        """
        从env的 observation 中解析：
          - x_init(末端姿, 7维: pos+quat)
          - joint_init(14维: q+v)
          - obstacle_positions(含T个时刻的障碍物信息)
        """

        ob = obs['observation']
        eef_pos = ob[0:3]
        eef_rot = ob[3:7]
        q_pos = ob[7:14]  # Exclusive Finger Joints as they dont matter for MPPI
        q_vel = ob[16:23]  # Exclusive Finger Joints as they dont matter for MPPI

        if self.LastPosition[0] == 0:
            vel = np.zeros(3, )  # ob[20:23]
        else:
            vel = (eef_pos - self.LastPosition) / self.Δt
        self.LastPosition = eef_pos

        # 末端姿 => 7维
        x_init = torch.tensor([eef_pos[0], eef_pos[1], eef_pos[2], eef_rot[0], eef_rot[1], eef_rot[2], eef_rot[3]],
                              dtype=self.dtype, device=self.device)
        # 关节状态 => 14维
        joint_init = torch.tensor(np.concatenate([q_pos, q_vel]),
                                  dtype=self.dtype, device=self.device)

        # 这里提取 (T,...) 的障碍物信息
        parameters = self.envs[0].extract_parameters_3d(self.T, self.Δt, goal)
        obstacle_positions = parameters[:, 6:]  # shape (T, nObsDim)

        # 【STORM】确保 obstacles 是 torch.Tensor
        obstacle_positions = torch.tensor(obstacle_positions, dtype=self.dtype, device=self.device)

        return x_init, joint_init, obstacle_positions


    #========================================================
    #【STORM】核心：在此进行“无循环”的时域展开 + MPPI更新
    #========================================================
    def update_control(self, state, goal, obstacle_positions):
        """
        state: (14,) => (q(7), v(7))
        goal:  (d,)  => 这里通常是(3,) or (7,)
        obstacle_positions: (T, nObsDim)
        """
        # 1) 扩展初始状态到 K 条并行
        x_init = state.unsqueeze(0).expand(self.K, -1)  # => (K,14)

        # 2) 采样噪声: (K,T,7)
        ε = self.noise_distribution.rsample((self.K, self.T))

        #========================================================
        # (单层退火修改) 对每个时间步乘以 (horizon_diffuse_factor**i)
        # 这样就相当于让第 i 步的采样分布协方差变为 factor^2 * Σ
        #========================================================
        for i_hor in range(self.T):
            factor = (self.horizon_diffuse_factor ** i_hor)
            ε[:, i_hor, :] *= factor

        # 3) 叠加基准控制 self.u (只有 (1 - α)*K 的部分会加上)
        #v = torch.clone(ε)
        #mask = torch.arange(self.K) < int((1 - self.α) * self.K)
        #v[mask] += self.u

        # 【不加 mask 的，原汁原味的】
        # 3) 叠加基准控制 self.u (给所有 rollout 全部加上 self.u)
        v = torch.clone(ε)
        v += self.u

        # 4) 构造多步目标
        #    若 self.trajectory 不为空，则第 i 步用 self.trajectory[i]
        #    否则全程都用 goal
        #    => 形状 (T,d)
        T_list = []
        for i in range(self.T):
            if self.trajectory is not None:
                # 确保 self.trajectory[i] 是 Tensor
                traj_i = self.trajectory[i]
                if not isinstance(traj_i, torch.Tensor):
                    traj_i = torch.tensor(traj_i, dtype=self.dtype, device=self.device)
                T_list.append(traj_i)
            else:
                # 这里 goal 应该是 Tensor(因为你在上面用 goal_tensor = torch.tensor(goal, device=...) 了)
                T_list.append(goal)

        # 现在每个元素都是Tensor => stack不会报错
        interim_goals = torch.stack(T_list, dim=0).to(self.device)

        rollout_fn = self.dynamics_params["rollout_fn"]  # 这个在 get_parameters() 时已经放进来了

        #========================================================
        # (单层退火修改) 还需要在 cost 中正确使用 time-varying Σ_inv(t)
        #   如果 Σ(t) = factor^2(t)* Σ(0)，则 Σ_inv(t) = 1/factor^2(t)* Σ_inv(0)。
        #   我们通过一个 factor_inv 数组把它传进去，以便在 rollout_fn 中做 MPPI cost。
        #========================================================
        factor_inv = []
        for i_hor in range(self.T):
            factor_inv.append(1.0 / (self.horizon_diffuse_factor ** i_hor))
        factor_inv = torch.tensor(factor_inv, dtype=self.dtype, device=self.device)  # (T,)


        # 5) 【STORM】一次性展开：得到全时域状态 + cost
        states_rollout, cost = rollout_fn(
            x_init,
            v,           # (K,T,7)
            self.u,      # (T,7)
            self.Δt,
            self.dynamics_params,
            obstacle_positions,
            interim_goals,
            Σ_inv=self.Σ_inv,
            λ=self.λ,
            device=self.device,
            # ======== (单层退火修改) 额外带入 factor_inv ========
            per_step_factor_inv=factor_inv
        )

        # 6) MPPI加权更新
        β = torch.min(cost)
        ω = torch.exp((β - cost) / self.λ)
        η = torch.sum(ω)
        ω = ω / η

        # δu = sum_k [ ω_k * ε_k ] => shape (T,7)
        δu = torch.sum(ω.view(-1, 1, 1) * ε, dim=0)
        self.u += δu

        # 7) 可选：平滑 self.u
        self.u = torch.tensor(
            scipy.signal.savgol_filter(self.u.cpu().numpy(), window_length=self.T//2*2-1, polyorder=1, axis=0),
            dtype=self.dtype, device=self.device
        )

        # 8) 把 rollouts, collision, etc. 存下来以便可视化
        self.collisions = [getCollisions()]  # 这只是个示例; 你也可收集更多
        self.trajectory_rollouts = states_rollout.detach().cpu().numpy()  # (K,T,14)
        self.obstacle_positions = obstacle_positions
        self.interim_goals = interim_goals.detach().cpu().numpy()

    def is_goal_reachable(self, goal, obstacles):
        num_obstacles = len(obstacles) // 10
        for i in range(num_obstacles):
            obstacle_start = i * 10
            obstacle_end = obstacle_start + 10
            obstacle = obstacles[obstacle_start:obstacle_end]
            obstacle_position = obstacle[0:3]
            obstacle_rotation = obstacle[3:7]
            obstacle_dimensions = obstacle[7:10]
            endeffector_dimensions = np.array([0.04, 0.04, 0.03])

            obstacle_min = torch.tensor(obstacle_position - (obstacle_dimensions + endeffector_dimensions),
                                        dtype=self.dtype, device=self.device)
            obstacle_max = torch.tensor(obstacle_position + (obstacle_dimensions + endeffector_dimensions),
                                        dtype=self.dtype, device=self.device)

            # Calculate Transformed Goal to get the correct result
            r = Rotation.from_quat([np.roll(obstacle_rotation, -1)])
            goal_transformed = np.matmul(r.as_matrix(), goal.numpy())
            goal_transformed = torch.tensor(goal_transformed)

            if torch.all(torch.logical_and(goal_transformed >= obstacle_min, goal_transformed <= obstacle_max)):
                return False
