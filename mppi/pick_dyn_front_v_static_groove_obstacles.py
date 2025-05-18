# 【STORM + 单层退火（内层）版本】pick_dyn_front_v_static_groove_obstacles.py

# ============================================
# 说明：
#  1) 针对动态蓝方块 + 静态 V 型凹槽场景的“无循环并行 STORM-MPPI”示例
#  2) 在 state_cost_vectorized() 中，保留并使用 robot_base_pos 参数
#  3) 在 rollout_storm_no_loop_front_v() 中调用 state_cost_vectorized() 时，也会传入 robot_base_pos
#  4) 加入单层退火思路：允许每个时间步使用不同协方差逆 Σ_inv(t)
# ============================================

import os
import math
import torch
import pytorch_kinematics

##############################################################################
# ========================
# 在这里定义全局数量 (【STORM】)
# ========================
##############################################################################
N_SAMPLES = 500  # 【STORM修改】并行样本数

##############################################################################
# 全局碰撞标志 (用于和原先结构保持一致；放到 cuda 上)
##############################################################################
collision = torch.zeros([N_SAMPLES, ], dtype=torch.bool, device='cuda')

def getCollisions():
    return collision


##############################################################################
# !!!【STORM新增】 通用的欧拉角->旋转矩阵函数
##############################################################################
def euler_to_matrix(euler_xyz, device, dtype):
    """
    这里示范以 Z-Y-X 的顺序将 (rx, ry, rz) 转成旋转矩阵 R = Rz(rz)*Ry(ry)*Rx(rx)
    若您在 MuJoCo XML 里设置了 euler="...", 并想准确匹配，请确保旋转顺序对应正确。
    """
    rx, ry, rz = euler_xyz
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rz = torch.tensor([
        [cz, -sz, 0.],
        [sz,  cz, 0.],
        [0.,   0., 1.]
    ], dtype=dtype, device=device)

    Ry = torch.tensor([
        [cy,  0., sy],
        [0.,  1., 0.],
        [-sy, 0., cy]
    ], dtype=dtype, device=device)

    Rx = torch.tensor([
        [1., 0.,  0.],
        [0., cx, -sx],
        [0., sx,  cx]
    ], dtype=dtype, device=device)

    # 先 Rz，再 Ry，再 Rx => 最终 R
    R = torch.mm(Rz, torch.mm(Ry, Rx))
    return R


##############################################################################
# !!!【STORM修改】 支持批量 OBB 碰撞检测
##############################################################################
def check_collision_obb_batch(
    pts_world,    # (N,3)  这里 N 一般是 K*T，即“并行采样数×时间步数”。
    center_world, # (N,3)
    R_world,      # (N,3,3)
    half_size,    # (N,3)
    margin,       # (3,) or (N,3)
    device, dtype
):
    """
    批量 OBB 碰撞检测:
      local_pt_i = R_world[i]^T * (pts_world[i] - center_world[i])
      如果 |local_pt_i| <= (half_size[i] + margin) 则判定碰撞
    """
    delta = pts_world - center_world            # (N,3)
    delta = delta.unsqueeze(1)                  # (N,1,3)
    R_T   = R_world.transpose(1, 2)             # (N,3,3), each is R^T
    local_pts = torch.bmm(delta, R_T).squeeze(1)# (N,3)

    dist_local = torch.abs(local_pts)

    if len(margin) == 3:
        margin_t = torch.tensor(margin, dtype=dtype, device=device).unsqueeze(0)
        margin_t = margin_t.expand_as(half_size)  # (N,3)
    else:
        margin_t = margin  # 若本身是(N,3)

    lim = half_size + margin_t
    collision_mask = torch.all(dist_local <= lim, dim=1)
    return collision_mask


##############################################################################
# !!!【STORM修改】 无循环并行 rollout
##############################################################################
def rollout_storm_no_loop_front_v(
    x_init,              # (K,14) => 初始状态(包含q和v)
    delta_v,             # (K,T,7) => 每时间步的“增量控制(噪声+基准)”
    u_base,              # (T,7)   => 基准u, 用于MPPI额外项
    dt,
    dynamics_params,     # 包含: chain, robot_base_pos, state_cost_vectorized, terminal_cost_vectorized, ...
    obstacle_positions,  # (T, nObsDim)
    goals,               # (T, d)
    Σ_inv,
    λ,
    device,
    #=====================================================
    # (单层退火修改) 新增: per_step_factor_inv
    #   若不为 None，则 shape=(T,) => 每个时间步 1/factor(t)
    #=====================================================
    per_step_factor_inv=None
):
    """
    一次性展开 T 步轨迹，并计算整条轨迹的累积 cost(含 step cost+terminal cost+mppi额外项).
    如果要做单层退火，则外部在噪声中乘以 factor(t)，在这里再把 Σ_inv(t) 做相应缩放。
    """
    chain               = dynamics_params["chain"]
    robot_base_pos      = dynamics_params["robot_base_pos"]  # 【STORM保留】后续要传给 cost 函数
    state_cost_fn       = dynamics_params["state_cost"]
    terminal_cost_fn    = dynamics_params["terminal_cost"]
    args                = dynamics_params["args"]

    K = x_init.shape[0]
    T = delta_v.shape[1]

    q_init = x_init[:, 0:7]  # (K,7)
    v_init = x_init[:, 7:14] # (K,7)

    # (1) 前缀和方式计算 q(t), v(t)
    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))  # (T,T)
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)                       # (K,T,T)

    vAll = torch.bmm(S_l_batched, delta_v) + v_init.unsqueeze(1)  # (K,T,7)
    qAll = torch.bmm(S_l_batched, vAll) * dt + q_init.unsqueeze(1)# (K,T,7)

    states_all = torch.cat([qAll, vAll], dim=2)  # (K,T,14)

    # (2) 正向运动学
    q_flat = qAll.reshape(K*T, 7)
    ret = chain.forward_kinematics(q_flat, end_only=True)
    eef_matrix = ret.get_matrix()                             # (K*T,4,4)
    eef_pos = eef_matrix[:, :3, 3] + robot_base_pos           # (K*T,3)

    # (3) 计算 cost
    #  3.1 单步代价
    x_flat = states_all.view(K*T, 14)
    eef_flat = eef_pos.view(K*T, 3)

    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1) # (K,T,nObsDim)
    obs_flat = obs_batched.contiguous().view(K*T, -1)               # (K*T,nObsDim)

    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)            # (K,T,d)
    goals_flat = goals_batched.contiguous().view(K*T, -1)           # (K*T,d)

    step_cost_all = state_cost_fn(
        x_flat,          # (K*T,14)
        eef_flat,        # (K*T,3)
        goals_flat,      # (K*T,d)
        obs_flat,        # (K*T,nObsDim)
        args,
        device,
        robot_base_pos   # 【STORM保留】将其直接传给 cost 函数
    )
    step_cost_matrix = step_cost_all.view(K, T)
    cost = torch.sum(step_cost_matrix, dim=1)  # =>(K,)

    #  3.2 终端代价
    x_terminal    = states_all[:, -1, :]                 # (K,14)
    eef_all3      = eef_pos.view(K, T, 3)
    eef_terminal  = eef_all3[:, -1, :]                   # (K,3)
    goals_all3    = goals_batched.view(K, T, -1)
    goal_terminal = goals_all3[:, -1, :]                 # (K,d)

    term_cost = terminal_cost_fn(
        x_terminal,
        eef_terminal,
        goal_terminal,
        args,
        device,
        robot_base_pos  # 【STORM保留】以防需要它
    )
    cost += term_cost

    #  3.3 MPPI额外项
    #     如果 per_step_factor_inv 不为 None，则对每个时间步做 Σ_inv(t) = 1/factor(t)^2 * Σ_inv(0)
    #     => 只需额外乘 1/factor(t)。
    if per_step_factor_inv is None:
        #【原逻辑，不做退火】
        Σ_inv_u_base = torch.matmul(Σ_inv, u_base.T).T  # => (T,7)
        mppi_term = (delta_v * Σ_inv_u_base.unsqueeze(0)).sum(dim=2)  # => (K,T)
        cost += mppi_term.sum(dim=1)
    else:
        #==========================================
        # (单层退火修改) time-varying Σ_inv(t)
        #==========================================
        Σ_inv_u_base0 = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        # => per-step multiply
        Σ_inv_u_base_var = Σ_inv_u_base0 * per_step_factor_inv.unsqueeze(1)  # =>(T,7)
        mppi_term = (delta_v * Σ_inv_u_base_var.unsqueeze(0)).sum(dim=2) # =>(K,T)
        cost += mppi_term.sum(dim=1)

    return states_all, cost


##############################################################################
# !!!【STORM修改】向量化单步代价 (保留 robot_base_pos)
##############################################################################
def state_cost_vectorized(
    x,           # (N,14) => 包含 [q(7), v(7)]
    eef_pos,     # (N,3)  => 末端执行器位置(已经做完FK + base offset)
    goal,        # (N,d)  => 每个样本对应的目标(常见 d=3)
    obstacles,   # (N,...) => 每个样本对应的障碍物信息(0:3,3:7,7:10,10:13,13:17,17:20)
    args,
    device,
    robot_base_pos   # 【STORM保留】 保留此参数
):
    global collision
    N = x.shape[0]
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    #--------------------------------------------------
    # 1) 目标距离代价
    #--------------------------------------------------
    goal_dist = torch.norm(eef_pos - goal, dim=1)
    cost += 1000.0 * (goal_dist ** 2)

    #--------------------------------------------------
    # 2) 动态蓝方块的碰撞
    #--------------------------------------------------
    obs_box_pos  = obstacles[:,  0: 3]
    obs_box_quat = obstacles[:,  3: 7]
    obs_box_size = obstacles[:,  7:10]

    obs_box_R = pytorch_kinematics.quaternion_to_matrix(obs_box_quat)  # (N,3,3)
    margin_box0 = [0.08, 0.08, 0.03]

    col_box0 = check_collision_obb_batch(
        pts_world    = eef_pos,
        center_world = obs_box_pos,
        R_world      = obs_box_R,
        half_size    = obs_box_size,
        margin       = margin_box0,
        device       = device,
        dtype        = x.dtype
    )

    #--------------------------------------------------
    # 3) V 型凹槽(静态): 取 obstacles[0,10:...] 作为 body
    #--------------------------------------------------
    vbody_pos_0  = obstacles[0, 10:13]
    vbody_quat_0 = obstacles[0, 13:17]
    Rv_body_0    = pytorch_kinematics.quaternion_to_matrix(vbody_quat_0.unsqueeze(0))[0] # (3,3)

    # 由于 V 型凹槽在场景中是静止的，这里只在 obstacles[0] 里取其信息，然后对所有 N 样本扩展即可:
    Rv_body_B    = Rv_body_0.unsqueeze(0).expand(N, -1, -1)
    vbody_pos_B  = vbody_pos_0.unsqueeze(0).expand(N, -1)

    # v1
    v1_local_pos = torch.tensor([-0.06, 0.0, 0.0], dtype=x.dtype, device=device)
    v1_euler     = [0.0, 0.0, 2.35619]
    Rv1_local    = euler_to_matrix(v1_euler, device=device, dtype=x.dtype)

    center_v1_0  = vbody_pos_0 + torch.matmul(Rv_body_0, v1_local_pos)
    center_v1_B  = center_v1_0.unsqueeze(0).expand(N, -1)
    R_world_v1_0 = torch.matmul(Rv_body_0, Rv1_local)
    R_world_v1_B = R_world_v1_0.unsqueeze(0).expand(N, -1, -1)
    margin_v1    = [0.065, 0.07, 0.03]
    v1_half_size = torch.tensor([0.10, 0.03, 0.03], dtype=x.dtype, device=device).unsqueeze(0).expand(N, -1)

    col_v1 = check_collision_obb_batch(
        pts_world    = eef_pos,
        center_world = center_v1_B,
        R_world      = R_world_v1_B,
        half_size    = v1_half_size,
        margin       = margin_v1,
        device       = device,
        dtype        = x.dtype
    )

    # v2
    v2_local_pos = torch.tensor([0.06, 0.0, 0.0], dtype=x.dtype, device=device)
    v2_euler     = [0.0, 0.0, 0.785398]
    Rv2_local    = euler_to_matrix(v2_euler, device=device, dtype=x.dtype)

    center_v2_0  = vbody_pos_0 + torch.matmul(Rv_body_0, v2_local_pos)
    center_v2_B  = center_v2_0.unsqueeze(0).expand(N, -1)
    R_world_v2_0 = torch.matmul(Rv_body_0, Rv2_local)
    R_world_v2_B = R_world_v2_0.unsqueeze(0).expand(N, -1, -1)
    margin_v2    = [0.065, 0.07, 0.03]
    v2_half_size = torch.tensor([0.10, 0.03, 0.03], dtype=x.dtype, device=device).unsqueeze(0).expand(N, -1)

    col_v2 = check_collision_obb_batch(
        pts_world    = eef_pos,
        center_world = center_v2_B,
        R_world      = R_world_v2_B,
        half_size    = v2_half_size,
        margin       = margin_v2,
        device       = device,
        dtype        = x.dtype
    )

    #--------------------------------------------------
    # 4) 手掌碰撞
    #--------------------------------------------------
    hand_pos = eef_pos.clone()
    hand_pos[:, 2] += 0.11
    hand_dim = [0.03, 0.08, 0.05]

    col_box0_hand = check_collision_obb_batch(
        pts_world    = hand_pos,
        center_world = obs_box_pos,
        R_world      = obs_box_R,
        half_size    = obs_box_size,
        margin       = hand_dim,
        device       = device,
        dtype        = x.dtype
    )

    col_v1_hand = check_collision_obb_batch(
        pts_world    = hand_pos,
        center_world = center_v1_B,
        R_world      = R_world_v1_B,
        half_size    = v1_half_size,
        margin       = hand_dim,
        device       = device,
        dtype        = x.dtype
    )

    col_v2_hand = check_collision_obb_batch(
        pts_world    = hand_pos,
        center_world = center_v2_B,
        R_world      = R_world_v2_B,
        half_size    = v2_half_size,
        margin       = hand_dim,
        device       = device,
        dtype        = x.dtype
    )

    collision_mask = (col_box0 | col_v1 | col_v2 | col_box0_hand | col_v1_hand | col_v2_hand)

    #--------------------------------------------------
    # 5) 工作空间约束 (使用 robot_base_pos)
    #--------------------------------------------------
    table_collision = (eef_pos[:, 2] <= 0.40)
    dist_robot_base = torch.norm(eef_pos - robot_base_pos, dim=1)
    workspace_violation = (dist_robot_base >= 0.8)

    cost += args.ω2 * collision_mask
    cost += args.ω3 * table_collision
    cost += args.ω4 * workspace_violation

    collision = collision_mask  # 记录最新一次调用的碰撞(可选)
    return cost


##############################################################################
# !!!【STORM修改】 终端代价(向量化), 也保留 robot_base_pos
##############################################################################
def terminal_cost_vectorized(
    x,         # (K,14)
    eef_pos,   # (K,3)
    goal,      # (K,d)
    args,
    device,
    robot_base_pos  # 【STORM保留】如需在终端时还要用 base_pos
):
    global collision
    cost = 10.0 * torch.norm(eef_pos - goal, dim=1) ** 2
    return cost


##############################################################################
# 主函数：get_parameters(args)
##############################################################################
def get_parameters(args):
    """
    返回：
      (K, T, Δt, α,
       dynamics,
       None,
       None,
       Σ, λ,
       convert_to_target,
       dtype, device,
       dynamics_params
      )
    """
    if args.tune_mppi <= 0:
        args.α   = 0.0
        args.λ   = 60.0
        args.σ   = 0.20
        args.χ   = 0.0
        args.ω1  = 1.0003
        args.ω2  = 9.16e3
        args.ω3  = 9.16e3
        args.ω4  = 9.16e3
        args.ω_Φ = 5.41
        args.d_goal = 0.15

    K = N_SAMPLES
    T = 10
    Δt = 0.01
    T_system = 0.011

    dtype = torch.double
    device = 'cuda'  # GPU

    α = args.α
    λ = args.λ

    Σ = args.σ * torch.tensor([
        [1.5,   args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, 0.75,  args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, 1.0,   args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, 1.25,  args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, 1.50,  args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00,  args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
    ], dtype=dtype, device=device)

    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    chain = pytorch_kinematics.build_serial_chain_from_urdf(
        xml, end_link_name="panda_link8", root_link_name="panda_link0"
    ).to(dtype=dtype, device=device)

    # 单步 dynamics
    def dynamics(x, u):
        new_vel = x[:, 7:14] + u
        new_pos = x[:, 0:7] + new_vel * Δt
        return torch.cat((new_pos, new_vel), dim=1)

    # 计算下一时刻末端位姿 => 用于 mppi_policy.py 的 predict()
    def convert_to_target(x, u):
        joint_pos = x[0:7]
        joint_vel = x[7:14]

        new_vel = joint_vel + u / (
            1.0 - torch.exp(torch.tensor(-Δt / T_system, dtype=dtype, device=device)) * (
                1.0 + (Δt / T_system))
        )
        new_joint_pos = joint_pos + new_vel * Δt

        ret = chain.forward_kinematics(new_joint_pos, end_only=True)
        eef_matrix = ret.get_matrix()
        robot_base_pos = torch.tensor([0.8, 0.75, 0.44], dtype=dtype, device=device)
        eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
        eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
        return torch.cat((eef_pos, eef_rot), dim=1)

    # 机器人基座位置(保留)
    robot_base_pos = torch.tensor([0.8, 0.75, 0.44], dtype=dtype, device=device)

    dynamics_params = {
        "chain": chain,
        "robot_base_pos": robot_base_pos,  # 【STORM保留】
        "args": args,
        #=== (单层退火修改) rollout_fn 允许接收 per_step_factor_inv
        "rollout_fn": rollout_storm_no_loop_front_v,
        "state_cost": state_cost_vectorized,
        "terminal_cost": terminal_cost_vectorized,
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,
        None,
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params
    )
