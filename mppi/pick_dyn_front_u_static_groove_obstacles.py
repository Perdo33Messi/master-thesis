# 【STORM + 单层退火（内层）】pick_dyn_front_u_static_groove_obstacles.py

import os
import pytorch_kinematics
import torch

# ========================
# 在这里定义全局数量
# ========================
N_SAMPLES = 3000  # 原先 500, 这里设置成 3000 供并行采样

# ========================
# 全局 碰撞标记向量 (放在 GPU 上)
# ========================
collision = torch.zeros([N_SAMPLES, ], dtype=torch.bool, device='cuda')


def getCollisions():
    """
    用于外部获取碰撞标记的函数。
    """
    return collision


# ====================================================
# 【STORM】 无循环rollout: 用下三角矩阵做一次性前缀和
# ====================================================
def rollout_storm_no_loop_front_u(
        x_init,  # (K,14) => 初始状态(包含q和v)
        delta_v,  # (K,T,7) => 每时间步的“增量控制(噪声+基准)”
        u_base,   # (T,7)   => 基准u, 用于MPPI额外项
        dt,
        dynamics_params,  # 包含: chain, robot_base_pos, state_cost_vectorized, terminal_cost_vectorized, ...
        obstacle_positions,  # (T, ...)，若每时刻障碍物不同，需要再扩展到(K,T,...)形状
        goals,  # (T, d) or (1,d) => 每个时间步的目标(可相同或不同)
        Σ_inv, λ, device,
        #=====================================================
        # (单层退火修改) 新增: per_step_factor_inv
        #   若不为 None，则 shape=(T,) => 每个时间步 1/factor(t)
        #=====================================================
        per_step_factor_inv=None
):
    """
    返回:
      states_all: (K,T,14) => 全时域(q,v)
      cost: (K,) => 每条轨迹累积代价

    若想在外部进行单层退火（对噪声做 factor(t) 放大），
    则这里可传入 per_step_factor_inv 用于缩放 MPPI 额外项中的 Σ_inv(t)。
    """
    chain = dynamics_params["chain"]
    robot_base_pos = dynamics_params["robot_base_pos"]
    state_cost_vector = dynamics_params["state_cost"]
    terminal_cost_vector = dynamics_params["terminal_cost"]
    args = dynamics_params["args"]

    K = x_init.shape[0]
    T = delta_v.shape[1]

    q_init = x_init[:, 0:7]   # (K,7)
    v_init = x_init[:, 7:14]  # (K,7)

    # ===========================================
    # 1) 下三角矩阵 S_l 做速度和位置的前缀和
    # ===========================================
    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))  # (T,T)
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)                       # (K,T,T)

    # 速度: v(t)
    vAll = torch.bmm(S_l_batched, delta_v)            # =>(K,T,7)
    vAll = vAll + v_init.unsqueeze(1)                 # =>(K,T,7)

    # 位置: q(t)
    qAll = torch.bmm(S_l_batched, vAll)               # =>(K,T,7)
    qAll = qAll * dt
    qAll = qAll + q_init.unsqueeze(1)                 # =>(K,T,7)

    # 拼合成 (K,T,14)
    states_all = torch.cat([qAll, vAll], dim=2)       # =>(K,T,14)

    # ===========================================
    # 2) Forward Kinematics (一次性计算)
    # ===========================================
    q_flat = qAll.reshape(K * T, 7)   # =>(K*T,7)
    ret = chain.forward_kinematics(q_flat, end_only=True)
    eef_matrix = ret.get_matrix()     # =>(K*T,4,4)
    eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # =>(K*T,3)

    # ===========================================
    # 3) 一次性向量化计算 step cost
    # ===========================================
    x_flat = states_all.view(K * T, 14)     # =>(K*T,14)
    eef_flat = eef_pos.view(K * T, 3)       # =>(K*T,3)

    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1)  # =>(K,T,nObsDim)
    obs_flat = obs_batched.reshape(K * T, -1)                        # =>(K*T,nObsDim)

    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)  # =>(K,T,d)
    goals_flat = goals_batched.reshape(K * T, -1)         # =>(K*T,d)

    step_cost_all = state_cost_vector(
        x_flat, eef_flat, goals_flat, obs_flat,
        args=args, device=device, robot_base_pos=robot_base_pos
    )  # =>(K*T,)
    step_cost_matrix = step_cost_all.view(K, T)  # =>(K,T)
    cost = torch.sum(step_cost_matrix, dim=1)    # =>(K,)

    # ===========================================
    # 4) 终端代价
    # ===========================================
    x_terminal = states_all[:, -1, :]              # =>(K,14)
    eef_pos_T  = eef_pos.view(K, T, 3)[:, -1, :]   # =>(K,3)
    goal_T     = goals_batched.view(K, T, -1)[:, -1, :]  # =>(K,d)

    term_cost = terminal_cost_vector(
        x_terminal, eef_pos_T, goal_T,
        args=args, device=device
    )  # =>(K,)
    cost += term_cost

    # ===========================================
    # 5) MPPI额外项: delta_v dot (Σ_inv(t) @ u_base)
    # ===========================================
    if per_step_factor_inv is None:
        #【原逻辑: 固定 Σ_inv】
        Ub = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)               # =>(K,)
    else:
        #=========================================================
        # (单层退火修改) Σ_inv(t) = (1 / factor(t)^2) * Σ_inv(0)
        # => 相当于对 Ub(t) 再乘以 (1/factor(t))
        #=========================================================
        Ub0 = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        Ub_per_step = Ub0 * per_step_factor_inv.unsqueeze(1)  # =>(T,7)
        mppi_term = (delta_v * Ub_per_step.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)

    return states_all, cost


# ====================================================
# 【STORM】向量化 单步代价 (替代原 state_cost)
# ====================================================
def state_cost_vectorized(
    x,            # (N,14) => [q(7), v(7)]
    eef_pos,      # (N,3)
    goal,         # (N,d)
    obstacles,    # (N,20) => 传入时每行包含2个障碍物(蓝色方块 & U形)的信息
    args,
    device,
    robot_base_pos
):
    """
    返回: cost.shape = (N,)

    本函数针对两种障碍物:
      1) 动态的小蓝方块(obstacles的前10维)
      2) 静态的U型槽(后10维, 其中再拆分3块子几何)
    以及 桌面 & 工作空间越界 惩罚
    """
    global collision
    N = x.shape[0]
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    dist_robot_base = torch.norm(eef_pos - robot_base_pos, dim=1)  # (N,)
    goal_dist = torch.norm(eef_pos - goal[:, 0:3], dim=1)          # 只看 xyz
    cost += 1000.0 * (goal_dist ** 2)

    collision_mask = torch.zeros(N, dtype=torch.bool, device=device)

    #--- 1) 小蓝方块(动态)
    dist1_eef = torch.abs(eef_pos - obstacles[:, 0:3])
    offset_box1 = torch.tensor([0.08, 0.08, 0.03], device=device)
    in_box1_eef = torch.all(
        torch.le(dist1_eef, obstacles[:, 7:10] + offset_box1),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box1_eef)

    # 手部
    hand = eef_pos.clone()
    hand[:, 2] += 0.11
    hand_dimension = torch.tensor([0.030, 0.08, 0.05], device=device)
    dist1_hand = torch.abs(hand - obstacles[:, 0:3])
    in_box1_hand = torch.all(
        torch.le(dist1_hand, obstacles[:, 7:10] + hand_dimension),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box1_hand)

    #--- 2) U 型槽(静态)
    center_u = obstacles[:, 10:13]  # (N,3)
    # rotation_u = obstacles[:, 13:17]  # (N,4) - 如果您需要对 U 进行旋转再处理
    # size_u = obstacles[:, 17:20]     # (N,3) - 这里只做简单 AABB 检测

    # 左竖板
    offset_left = torch.tensor([-0.10, 0.0, 0.0], device=device)
    size_left   = torch.tensor([0.01, 0.10, 0.03], device=device)
    center_left = center_u + offset_left

    # 右竖板
    offset_right = torch.tensor([0.10, 0.0, 0.0], device=device)
    size_right   = torch.tensor([0.01, 0.10, 0.03], device=device)
    center_right = center_u + offset_right

    # 底部横板
    offset_bottom = torch.tensor([0.0, -0.10, 0.0], device=device)
    size_bottom   = torch.tensor([0.10, 0.01, 0.03], device=device)
    center_bottom = center_u + offset_bottom

    margin_u_eef = torch.tensor([0.03, 0.03, 0.03], device=device)

    # EEF 与 U
    dist2_eef_left   = torch.abs(eef_pos - center_left)
    dist2_eef_right  = torch.abs(eef_pos - center_right)
    dist2_eef_bottom = torch.abs(eef_pos - center_bottom)

    collision_2_eef_left = torch.all(
        torch.le(dist2_eef_left, size_left + margin_u_eef), dim=1
    )
    collision_2_eef_right = torch.all(
        torch.le(dist2_eef_right, size_right + margin_u_eef), dim=1
    )
    collision_2_eef_bottom = torch.all(
        torch.le(dist2_eef_bottom, size_bottom + margin_u_eef), dim=1
    )
    collision_u_eef = collision_2_eef_left | collision_2_eef_right | collision_2_eef_bottom

    # Hand 与 U
    dist2_hand_left   = torch.abs(hand - center_left)
    dist2_hand_right  = torch.abs(hand - center_right)
    dist2_hand_bottom = torch.abs(hand - center_bottom)

    margin_u_hand = torch.tensor([0.03, 0.03, 0.03], device=device)

    collision_2_hand_left = torch.all(
        torch.le(dist2_hand_left, size_left + margin_u_hand), dim=1
    )
    collision_2_hand_right = torch.all(
        torch.le(dist2_hand_right, size_right + margin_u_hand), dim=1
    )
    collision_2_hand_bottom = torch.all(
        torch.le(dist2_hand_bottom, size_bottom + margin_u_hand), dim=1
    )
    collision_u_hand = collision_2_hand_left | collision_2_hand_right | collision_2_hand_bottom

    collision_u_total = collision_u_eef | collision_u_hand
    collision_mask = torch.logical_or(collision_mask, collision_u_total)

    #--- 桌面 & 工作空间
    table_collision = torch.le(eef_pos[:, 2], 0.40)
    workspace_violation = torch.ge(dist_robot_base, 0.8)

    cost += args.ω2 * collision_mask.float()
    cost += args.ω3 * table_collision.float()
    cost += args.ω4 * workspace_violation.float()

    collision = collision_mask  # 记录

    return cost


# ====================================================
# 【STORM】向量化 终端代价 (替代原 terminal_cost)
# ====================================================
def terminal_cost_vectorized(
    x,         # (K,14)
    eef_pos,   # (K,3)
    goal,      # (K,d)
    args,
    device
):
    """
    只在终端做一个距离的惩罚 (也可在这里加碰撞等其它需求)
    """
    cost = 10.0 * torch.norm(eef_pos - goal[:, 0:3], dim=1) ** 2

    return cost


def get_parameters(args):
    """
    【STORM】风格的 get_parameters, 返回给 mppi_policy.py 用的各种变量:

    Returns:
      K, T, Δt, α,
      dynamics(若用单步即可, 也可 None),
      q(可为 None),
      ϕ(可为 None),
      Σ, λ,
      convert_to_target,
      dtype, device,
      dynamics_params
    """
    if args.tune_mppi <= 0:
        args.α = 0
        args.λ = 60
        args.σ = 0.20
        args.χ = 0.0
        args.ω1 = 1.0003
        args.ω2 = 9.16e3
        args.ω3 = 9.16e3
        args.ω4 = 9.16e3
        args.ω_Φ = 5.41
        args.d_goal = 0.15

    K = N_SAMPLES
    T = 10
    Δt = 0.01
    T_system = 0.011

    dtype = torch.double
    device = 'cuda'

    α = args.α
    λ = args.λ
    Σ = args.σ * torch.tensor([
        [1.5,  args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, 1.0,  args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
    ], dtype=dtype, device=device)

    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    chain = pytorch_kinematics.build_serial_chain_from_urdf(
        xml, end_link_name="panda_link8", root_link_name="panda_link0"
    ).to(dtype=dtype, device=device)

    robot_base_pos = torch.tensor([0.8, 0.75, 0.44], dtype=dtype, device=device)

    def dynamics(x, u):
        new_vel = x[:, 7:14] + u
        new_pos = x[:, 0:7] + new_vel * Δt

        return torch.cat((new_pos, new_vel), dim=1)

    def convert_to_target(x, u):
        joint_pos = x[0:7]
        joint_vel = x[7:14]
        new_vel = joint_vel + u / (
            1 - torch.exp(torch.tensor(-Δt / T_system, dtype=dtype, device=device)) *
            (1 + (Δt / T_system))
        )
        new_joint_pos = joint_pos + new_vel * Δt

        ret = chain.forward_kinematics(new_joint_pos.unsqueeze(0), end_only=True)
        eef_matrix = ret.get_matrix()
        eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
        eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])

        return torch.cat((eef_pos, eef_rot), dim=1)

    dynamics_params = {
        "chain": chain,
        "robot_base_pos": robot_base_pos,
        "args": args,
        "state_cost": state_cost_vectorized,
        "terminal_cost": terminal_cost_vectorized,
        # (单层退火修改) rollout_fn 增加对 per_step_factor_inv 的支持
        "rollout_fn": rollout_storm_no_loop_front_u
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,   # q (旧版的 state_cost)
        None,   # ϕ (旧版的 terminal_cost)
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params
    )

