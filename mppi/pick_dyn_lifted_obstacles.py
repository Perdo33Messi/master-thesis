import os

import pytorch_kinematics
import torch

# ========================
# 在这里定义全局数量
# ========================
N_SAMPLES = 500

#【GPU】在初始化时就将 collision 放到 GPU(或 device) 上
collision = torch.zeros([N_SAMPLES, ], dtype=torch.bool, device='cuda') #【GPU】
#collision = torch.zeros([500, ], dtype=torch.bool, device='cuda') #【GPU】

#====================================================
#【STORM】方法A: 不再使用 "collision = ..." 做全局标记
#   不再维护 getCollisions() 全局函数(若不需要可删除)。
#   如果你仍需要 getCollisions() 做调试，也可以保留，但不会再被使用。
#====================================================
def getCollisions():
    """
    如果你不再使用任何 global collision，就可以让它返回一个空值或留空。
    这里示例仅保留以兼容老代码，但不再做实际用途。
    """
    return collision


#====================================================
#【STORM】 无循环rollout: 用下三角矩阵做一次性前缀和 (lifted版本)
#====================================================
def rollout_storm_no_loop_lifted(
    x_init,            # (K,14) => 初始状态(包含q+v)
    delta_v,           # (K,T,7) => 每时间步的“增量控制(噪声+基准)”
    u_base,            # (T,7)   => 基准u, 用于MPPI额外项
    dt,
    dynamics_params,   # 包含: chain, robot_base_pos, state_cost_vectorized_lifted, terminal_cost_vectorized_lifted, ...
    obstacle_positions,# (T, ...) => 若每时刻障碍物不同，需要扩展到(K,T,...) 形状
    goals,             # (T,d) or (1,d) => 每时间步的目标
    Σ_inv, λ, device,
    # ==============================================
    # (单层退火修改) 新增: per_step_factor_inv
    #   shape (T,)，表示 1/factor(t)
    #   若为 None，表示沿用原先不变的 Σ_inv
    # ==============================================
    per_step_factor_inv=None
):
    """
    返回:
      states_all: (K,T,14) => 全时域(q,v)
      cost: (K,) => 每条并行轨迹的总cost

    如果需要在多步上做单层退火，
    则外部会将每个时间步的退火系数 factor(t) 以 1/factor(t) 的形式传入 per_step_factor_inv。
    """
    #-------------------------------------------
    # 取出字典中的函数与参数
    #-------------------------------------------
    chain                       = dynamics_params["chain"]
    robot_base_pos              = dynamics_params["robot_base_pos"]
    state_cost_vector_lifted    = dynamics_params["state_cost_lifted"]
    terminal_cost_vector_lifted = dynamics_params["terminal_cost_lifted"]
    args                        = dynamics_params["args"]

    #-------------------------------------------
    # 1) 用下三角矩阵做前缀和
    #-------------------------------------------
    K = x_init.shape[0]
    T = delta_v.shape[1]

    q_init = x_init[:, 0:7]  # (K,7)
    v_init = x_init[:, 7:14] # (K,7)

    # 下三角矩阵 S_l (T,T)，再扩展到 (K,T,T)
    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)  # (K,T,T)

    # vAll = v_init + sum_{k=0..t-1} delta_v(k)
    vAll = torch.bmm(S_l_batched, delta_v)  # => (K,T,7)
    vAll = vAll + v_init.unsqueeze(1)

    # qAll = q_init + dt * sum_{k=0..t-1} v(t)
    qAll = torch.bmm(S_l_batched, vAll) * dt
    qAll = qAll + q_init.unsqueeze(1)

    # 拼成 (K,T,14)
    states_all = torch.cat([qAll, vAll], dim=2)  # (K,T,14)

    #-------------------------------------------
    # 2) 一次性做 Forward Kinematics
    #-------------------------------------------
    q_flat = qAll.reshape(K*T, 7)
    ret = chain.forward_kinematics(q_flat, end_only=True)
    eef_matrix = ret.get_matrix()  # (K*T,4,4)
    eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # (K*T,3)

    #-------------------------------------------
    # 3) 一次性向量化 计算 cost
    #-------------------------------------------

    # (3.1) 单步代价
    # flatten: x_flat=(K*T,14), eef_flat=(K*T,3)
    x_flat   = states_all.reshape(K*T, 14)
    eef_flat = eef_pos.reshape(K*T, 3)

    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1)  # =>(K,T,nObsDim)
    obs_flat = obs_batched.reshape(K*T, -1)                          # =>(K*T,nObsDim)

    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)             # =>(K,T,d)
    goals_flat = goals_batched.reshape(K*T, -1)                      # =>(K*T,d)


    step_cost_all = state_cost_vector_lifted(
        x_flat, eef_flat, goals_flat, obs_flat,
        args=args, device=device, robot_base_pos=robot_base_pos
    )  # =>(K*T,)
    step_cost_matrix = step_cost_all.view(K, T)
    cost = step_cost_matrix.sum(dim=1)  # =>(K,)

    # (3.2) 终端代价
    x_terminal = states_all[:, -1, :]       # (K,14)
    eef_all3   = eef_pos.view(K, T, 3)
    eef_terminal = eef_all3[:, -1, :]       # (K,3)
    goals_all3 = goals_batched.view(K, T, -1)
    goal_terminal = goals_all3[:, -1, :]    # (K,d)

    term_cost = terminal_cost_vector_lifted(
        x_terminal, eef_terminal, goal_terminal,
        args=args, device=device
    )  # =>(K,)

    cost += term_cost


    # (3.3) MPPI 额外项: \sum_{t=0..T-1} [ \delta_v(t)^\top Σ_inv(t) u_base(t) ]
    #
    # 说明:
    #   - 若 per_step_factor_inv is None，表示保持原有不变 => Σ_inv(t) = Σ_inv(0).
    #   - 若不为 None，则 per_step_factor_inv[t] = 1 / factor(t)，
    #     => Σ_inv(t) = 1/factor(t)^2 * Σ_inv(0)
    #     => 只需额外乘 1/factor(t) 即可.
    #-------------------------------------------
    if per_step_factor_inv is None:
        #【保持原逻辑】不做退火
        Ub = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T,7) =>(K,T)
        cost += mppi_term.sum(dim=1)  # =>(K,)
    else:
        #===========================
        # (单层退火修改) 正确缩放
        #===========================
        # 先算 Ub0 = Σ_inv(0) @ u_base(t)
        Ub0 = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)

        # 对每个时间步 t，再乘以 per_step_factor_inv[t].
        # => (T,7)
        Ub = Ub0 * per_step_factor_inv.unsqueeze(1)

        # delta_v 已经包含 "factor(t) * ε(t)" 的放大
        # => 现在再与 Ub(t) 做逐样本点乘
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)  # =>(K,)

    return states_all, cost

#====================================================
#【STORM】向量化 单步代价 (去掉 global collision)
#====================================================
def state_cost_vectorized_lifted(
    x,        # (N,14) => [q(7), v(7)]
    eef_pos,  # (N,3)
    goal,     # (N,3)
    obstacles,# (N,...) => 每个样本对应的障碍物
    args, device,
    robot_base_pos
):
    """
    返回 shape=(N,)
    不再使用 global collision,
    改用局部 collision_mask=(N,) 累加到 cost。
    """
    N = x.shape[0]

    # 1) 初始化 cost
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    # 2) 距离代价
    dist_robot_base = torch.norm(eef_pos - robot_base_pos, dim=1)
    goal_dist       = torch.norm(eef_pos - goal, dim=1)
    cost += 1000.0 * (goal_dist**2)

    #---------------------------------------
    # 3) 局部碰撞掩码 collision_mask
    #---------------------------------------
    collision_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # 假设 obstacles 包含多块(比如 3个物体?)
    # 只示例:  dist1EEF, dist2EEF, dist3EEF => 你可扩展
    # 这里与之前 pick_dyn_lifted_obstacles 的逻辑对齐:

    # dist1EEF => 与 obstacle1 [0:3] 的距离
    dist1EEF = torch.abs(eef_pos - obstacles[:, 0:3])
    in_box1 = torch.all(
        torch.le(dist1EEF, obstacles[:, 7:10] + torch.tensor([0.055,0.055,0.03], device=device)),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box1)

    # dist2EEF => obstacle2
    dist2EEF = torch.abs(eef_pos - obstacles[:, 10:13])
    in_box2 = torch.all(
        torch.le(dist2EEF, obstacles[:, 17:20] + torch.tensor([0.055,0.055,0.03], device=device)),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box2)

    # dist3EEF => obstacle3
    dist3EEF = torch.abs(eef_pos - obstacles[:, 20:23])
    in_box3 = torch.all(
        torch.le(dist3EEF, obstacles[:, 27:30] + torch.tensor([0.055,0.055,0.03], device=device)),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box3)

    # 手部碰撞
    hand = eef_pos.clone()
    hand[:, 2] += 0.11
    dist4 = torch.abs(hand - obstacles[:, 0:3])
    in_box4 = torch.all(
        torch.le(dist4, obstacles[:, 7:10] + torch.tensor([0.030, 0.08, 0.05], device=device)),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box4)

    dist5 = torch.abs(hand - obstacles[:, 10:13])
    in_box5 = torch.all(
        torch.le(dist5, obstacles[:, 17:20] + torch.tensor([0.030, 0.11, 0.05], device=device)),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box5)

    dist6 = torch.abs(hand - obstacles[:, 20:23])
    in_box6 = torch.all(
        torch.le(dist6, obstacles[:, 27:30] + torch.tensor([0.030, 0.08, 0.05], device=device)),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box6)

    # 额外: 撞桌子 / 超出工作空间
    table_collision    = torch.le(eef_pos[:,2], 0.38)
    workspace_violation= torch.ge(dist_robot_base, 0.8)

    # 4) 累加 cost
    cost += args.ω2 * collision_mask.float()
    #cost += args.ω3 * table_collision.float()
    cost += args.ω4 * workspace_violation.float()

    return cost


#====================================================
#【STORM】向量化 终端代价 (去掉 global collision)
#====================================================
def terminal_cost_vectorized_lifted(
    x,        # (N,14)
    eef_pos,  # (N,3)
    goal,     # (N,3)
    args, device
):
    """
    返回 shape=(N,)
    不再使用 global collision.
    """
    N = x.shape[0]
    cost = 10.0 * torch.norm(eef_pos - goal, dim=1)**2
    # 如果还想加 cost += args.ω_Φ * ...，请自行加

    # 如果你在终端时还想检测碰撞，也可以像单步代价那样加 collision_mask
    # cost += ...

    return cost



#====================================================
#【STORM】 get_parameters: 返回一切所需
#====================================================
def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 60  # 40  # 1.62e1
        args.σ = 0.20  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
        args.χ = 0.0  # 2.00e-2
        args.ω1 = 1.0003
        args.ω2 = 9.16e3#25160 #9160#9.16e3
        args.ω3 = 9.16e3
        args.ω4 = 9.16e3
        args.ω_Φ = 5.41
        args.d_goal = 0.15

    K = N_SAMPLES  # 直接复用全局变量 N_SAMPLES
    #K = 500
    T = 10
    Δt = 0.01
    T_system = 0.011

    dtype = torch.double
    #device = 'cpu'  # 'cuda'
    #【GPU】这里改为 'cuda'（如果要在 GPU 上跑）；如果想保留CPU，就改回 'cpu'
    device = 'cuda'  #【GPU】

    α = args.α
    λ = args.λ
    Σ = args.σ * torch.tensor([
        [2.5, args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, 1.0, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
    ], dtype=dtype, device=device)

    #-------------------------------------------
    # 构造 kinematics chain
    #-------------------------------------------
    # Ensure we get the path separator correct on windows
    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')

    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    dtype_kinematics = torch.double
    chain = pytorch_kinematics.build_serial_chain_from_urdf(xml, end_link_name="panda_link8",
                                                            root_link_name="panda_link0")
    chain = chain.to(dtype=dtype_kinematics, device=device) #【GPU】

    # Translational offset of Robot into World Coordinates
    robot_base_pos = torch.tensor([0.8, 0.75, 0.44],
                                  device=device, dtype=dtype_kinematics)

    #-------------------------------------------
    # 仍可定义单步 dynamics
    #-------------------------------------------
    def dynamics(x, u):

        new_vel = x[:, 7:14] + u
        new_pos = x[:, 0:7] + new_vel * Δt

        return torch.cat((new_pos, new_vel), dim=1)


    #-------------------------------------------
    # 同理: terminal_cost / state_cost
    # (若你想兼容老API,可以保留原函数名 & 不用)
    #-------------------------------------------
    def convert_to_target(x, u):
        """
        转到下一时刻关节位置 -> end effector 目标位置 (GPU 上).
        """

        joint_pos = x[0:7]
        joint_vel = x[7:14]
        new_vel = joint_vel + u / (
                1 - torch.exp(torch.tensor(-Δt / T_system)) * (
                1 + (Δt / T_system)))  # / (1 - torch.exp(torch.tensor(-0.01 / 0.150)))  # 0.175

        new_joint_pos = joint_pos + new_vel * Δt  # Calculate new Target Joint Positions

        # Calculate World Coordinate Target Position
        ret = chain.forward_kinematics(new_joint_pos, end_only=True)
        eef_matrix = ret.get_matrix()
        eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # Calculate World Coordinate Target
        eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])

        return torch.cat((eef_pos, eef_rot), dim=1)

    #-------------------------------------------
    # 把向量化cost函数 & rollout函数等封装进 dict
    #-------------------------------------------
    dynamics_params = {
        "chain": chain,
        "robot_base_pos": robot_base_pos,
        "args": args,
        #【STORM】注入我们定义的向量化 cost
        "state_cost_lifted": state_cost_vectorized_lifted,
        "terminal_cost_lifted": terminal_cost_vectorized_lifted,
        "rollout_fn": rollout_storm_no_loop_lifted
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,  # 兼容老API => state_cost=None
        None,  # 兼容老API => terminal_cost=None
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params  #【STORM】关键: 把新函数打包带回
    )
