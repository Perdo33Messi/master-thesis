# 【STORM + 单层退火（内层）】pick_dyn_rear_v_static_groove_obstacles.py
# 用了欧拉角，没有去用四元数

# ============================================
# 说明：
#  1) 场景：动态小方块(用 AABB) + 静态后置 V 型凹槽(分 v1, v2 两块 OBB)
#  2) 采用 STORM 无循环并行计算思路 => 一次性展开并在张量上计算代价
#  3) 使用 GPU: device='cuda'，并行数量: N_SAMPLES=2500
#  4) 单层退火：允许在 MPPI 采样时每个时间步使用 factor(t) 放大噪声，并在此对 Σ_inv(t) 做相应缩放。
# ============================================

import os
import math

import torch
import pytorch_kinematics

##############################################################################
# ========================
# 在这里定义全局数量 (【STORM修改】)
# ========================
##############################################################################
N_SAMPLES = 3000

##############################################################################
# 全局碰撞标志 (放到 cuda 上；和原先结构保持一致)
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
    若与 MuJoCo XML <geom euler="..."> 完全匹配，请根据实际需要调整旋转顺序。
    """
    rx, ry, rz = euler_xyz
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)

    Rz = torch.tensor([
        [cz, -sz, 0.],
        [sz,  cz, 0.],
        [0.,  0., 1.]
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

    R = torch.mm(Rz, torch.mm(Ry, Rx))
    return R

##############################################################################
# !!!【STORM新增】 check_collision_aabb_batch: 批量 AABB 碰撞
##############################################################################
def check_collision_aabb_batch(
    pts_world,    # (N,3)
    center_world, # (N,3)
    half_size,    # (N,3)
    margin,       # (3,) or (N,3)
    device, dtype
):
    """
    对 N 个点和相应的 N 个 AABB 做碰撞检测:
      - lower_bound_i = center_world[i] - (half_size[i] + margin)
      - upper_bound_i = center_world[i] + (half_size[i] + margin)
      - 如果点的坐标均在 [lower_bound_i, upper_bound_i] 内 => 碰撞
    返回 collision_mask: (N,)
    """
    if len(margin) == 3:
        margin_t = torch.tensor(margin, dtype=dtype, device=device).unsqueeze(0).expand_as(half_size)
    else:
        margin_t = margin  # 若已是(N,3)

    lower_bound = center_world - (half_size + margin_t)
    upper_bound = center_world + (half_size + margin_t)

    in_x = (pts_world[:, 0] >= lower_bound[:, 0]) & (pts_world[:, 0] <= upper_bound[:, 0])
    in_y = (pts_world[:, 1] >= lower_bound[:, 1]) & (pts_world[:, 1] <= upper_bound[:, 1])
    in_z = (pts_world[:, 2] >= lower_bound[:, 2]) & (pts_world[:, 2] <= upper_bound[:, 2])

    collision_mask = in_x & in_y & in_z
    return collision_mask

##############################################################################
# !!!【STORM新增】 check_collision_obb_batch: 批量 OBB 碰撞
##############################################################################
def check_collision_obb_batch(
    pts_world,    # (N,3)
    center_world, # (N,3)
    R_world,      # (N,3,3)
    half_size,    # (N,3)
    margin,       # (3,) or (N,3)
    device, dtype
):
    """
    对 N 个点与 N 个 OBB 做批量碰撞检测:
      local_pt_i = R_world[i]^T * (pts_world[i] - center_world[i])
      若各维绝对值 <= half_size[i] + margin => 碰撞
    返回 collision_mask: (N,)
    """
    if len(margin) == 3:
        margin_t = torch.tensor(margin, dtype=dtype, device=device).unsqueeze(0).expand_as(half_size)
    else:
        margin_t = margin

    delta = pts_world - center_world
    delta = delta.unsqueeze(1)  # (N,1,3)
    R_T   = R_world.transpose(1, 2)  # (N,3,3)
    local_pts = torch.bmm(delta, R_T).squeeze(1)  # (N,3)

    dist_local = torch.abs(local_pts)
    lim = half_size + margin_t
    collision_mask = torch.all(dist_local <= lim, dim=1)
    return collision_mask

##############################################################################
# !!!【STORM新增】 rollout_storm_no_loop_rear_v: 一次性并行展开 T 步
##############################################################################
def rollout_storm_no_loop_rear_v(
    x_init,              # (K,14)
    delta_v,             # (K,T,7)
    u_base,              # (T,7)
    dt,
    dynamics_params,     # dict: 包含 chain, robot_base_pos, state_cost_vectorized, terminal_cost_vectorized, ...
    obstacle_positions,  # (T, nObsDim) => [动态小方块 + 静态后V形槽]
    goals,               # (T,d)
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
    返回:
      states_all: (K,T,14)
      cost:       (K,)

    如果要让 Σ 随时间步退火，需要外部对噪声 (delta_v) 做 factor(t) 乘法，
    这里再通过 per_step_factor_inv 调整 MPPI 额外项中的 Σ_inv(t)。
    """
    chain            = dynamics_params["chain"]
    robot_base_pos   = dynamics_params["robot_base_pos"]
    state_cost_fn    = dynamics_params["state_cost"]
    terminal_cost_fn = dynamics_params["terminal_cost"]
    args             = dynamics_params["args"]

    K = x_init.shape[0]
    T = delta_v.shape[1]

    # 1) 前缀和 => v(t), q(t)
    q_init = x_init[:, 0:7]
    v_init = x_init[:, 7:14]

    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))   # (T,T)
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)                        # (K,T,T)

    vAll = torch.bmm(S_l_batched, delta_v) + v_init.unsqueeze(1)  # (K,T,7)
    qAll = torch.bmm(S_l_batched, vAll) * dt + q_init.unsqueeze(1)# (K,T,7)

    states_all = torch.cat([qAll, vAll], dim=2)  # (K,T,14)

    # 2) 正向运动学 => eef_pos
    q_flat = qAll.reshape(K*T, 7)
    ret = chain.forward_kinematics(q_flat, end_only=True)
    eef_matrix = ret.get_matrix()                   # (K*T,4,4)
    eef_pos = eef_matrix[:, :3, 3] + robot_base_pos # (K*T,3)

    # 3) 计算 cost
    #   3.1 单步代价
    x_flat = states_all.view(K*T, 14)
    eef_flat = eef_pos.view(K*T, 3)

    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1)  # (K,T,nObsDim)
    obs_flat    = obs_batched.reshape(K*T, -1)                       # (K*T,nObsDim)

    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)  # (K,T,d)
    goals_flat    = goals_batched.reshape(K*T, -1)        # (K*T,d)

    step_cost_all = state_cost_fn(
        x_flat,
        eef_flat,
        goals_flat,
        obs_flat,
        args=args,
        device=device,
        robot_base_pos=robot_base_pos
    )  # => (K*T,)

    step_cost_matrix = step_cost_all.view(K, T)
    cost = torch.sum(step_cost_matrix, dim=1)  # (K,)

    #   3.2 终端代价
    x_terminal = states_all[:, -1, :]          # (K,14)
    eef_all3   = eef_pos.view(K, T, 3)
    eef_terminal = eef_all3[:, -1, :]          # (K,3)
    goals_all3   = goals_batched.view(K, T, -1)
    goal_terminal= goals_all3[:, -1, :]

    term_cost = terminal_cost_fn(
        x_terminal,
        eef_terminal,
        goal_terminal,
        args=args,
        device=device,
        robot_base_pos=robot_base_pos
    )
    cost += term_cost

    #   3.3 MPPI 额外项: ∑ δv(t)ᵀ Σ_inv(t) u_base(t)
    if per_step_factor_inv is None:
        #【原逻辑：固定 Σ_inv】
        Σ_inv_u_base = torch.matmul(Σ_inv, u_base.T).T   # (T,7)
        mppi_term = (delta_v * Σ_inv_u_base.unsqueeze(0)).sum(dim=2)  # (K,T)
        cost += mppi_term.sum(dim=1)
    else:
        #=====================================================
        # (单层退火修改) Σ_inv(t) = 1/factor(t)^2 * Σ_inv(0)
        # => 在代码里相当于再乘以 per_step_factor_inv[t]
        #=====================================================
        Σ_inv_u_base0 = torch.matmul(Σ_inv, u_base.T).T  # (T,7)
        Σ_inv_u_base_var = Σ_inv_u_base0 * per_step_factor_inv.unsqueeze(1)  # (T,7)
        mppi_term = (delta_v * Σ_inv_u_base_var.unsqueeze(0)).sum(dim=2)
        cost += mppi_term.sum(dim=1)

    return states_all, cost

##############################################################################
# !!!【STORM新增】 state_cost_vectorized: 动态小方块(AABB) + 静态后 V 槽(OBB)
##############################################################################
def state_cost_vectorized(
    x,           # (N,14)
    eef_pos,     # (N,3)
    goal,        # (N,d)
    obstacles,   # (N,nObsDim)
    args,
    device,
    robot_base_pos
):
    """
    - obstacles[...,0:3]  => 动态小方块 pos
    - obstacles[...,7:10] => 动态小方块 half-size
    - obstacles[...,10:13]=> v_body pos(静态 => 每行都相同)
    - obstacles[...,13:17]=> v_body quat
    - ...
    """
    global collision
    N = x.shape[0]
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    # 1) 目标距离
    goal_dist = torch.norm(eef_pos - goal, dim=1)
    cost += 1000.0 * (goal_dist ** 2)

    # 2) 动态小方块 => AABB
    box_pos  = obstacles[:, 0:3]
    box_size = obstacles[:, 7:10]
    margin_box0 = [0.055, 0.055, 0.03]

    col_box = check_collision_aabb_batch(
        pts_world    = eef_pos,
        center_world = box_pos,
        half_size    = box_size,
        margin       = margin_box0,
        device       = device,
        dtype        = x.dtype
    )

    # 3) 后 V 形槽(静态): v1, v2 => OBB
    v_body_pos_0  = obstacles[0, 10:13]
    v_body_quat_0 = obstacles[0, 13:17]
    v_body_quat_t = v_body_quat_0.unsqueeze(0)
    Rv_body_0 = pytorch_kinematics.quaternion_to_matrix(v_body_quat_t)[0]  # (3,3)

    Rv_body_B   = Rv_body_0.unsqueeze(0).expand(N, -1, -1)
    v_body_pos_B= v_body_pos_0.unsqueeze(0).expand(N, -1)

    # v1
    v1_local_pos = torch.tensor([-0.06, 0.0, 0.0], dtype=x.dtype, device=device)
    v1_euler     = [0.0, 0.0, 2.35619]
    Rv1_local    = euler_to_matrix(v1_euler, device=device, dtype=x.dtype)
    center_v1_0  = v_body_pos_0 + torch.matmul(Rv_body_0, v1_local_pos)
    center_v1_B  = center_v1_0.unsqueeze(0).expand(N, -1)
    R_world_v1_0 = torch.matmul(Rv_body_0, Rv1_local)
    R_world_v1_B = R_world_v1_0.unsqueeze(0).expand(N, -1, -1)
    margin_v1    = [0.065, 0.07, 0.03]
    v1_size      = torch.tensor([0.10, 0.03, 0.03], dtype=x.dtype, device=device).unsqueeze(0).expand(N, -1)

    col_v1 = check_collision_obb_batch(
        pts_world    = eef_pos,
        center_world = center_v1_B,
        R_world      = R_world_v1_B,
        half_size    = v1_size,
        margin       = margin_v1,
        device       = device,
        dtype        = x.dtype
    )

    # v2
    v2_local_pos = torch.tensor([0.06, 0.0, 0.0], dtype=x.dtype, device=device)
    v2_euler     = [0.0, 0.0, 0.785398]
    Rv2_local    = euler_to_matrix(v2_euler, device=device, dtype=x.dtype)
    center_v2_0  = v_body_pos_0 + torch.matmul(Rv_body_0, v2_local_pos)
    center_v2_B  = center_v2_0.unsqueeze(0).expand(N, -1)
    R_world_v2_0 = torch.matmul(Rv_body_0, Rv2_local)
    R_world_v2_B = R_world_v2_0.unsqueeze(0).expand(N, -1, -1)
    margin_v2    = [0.065, 0.07, 0.03]
    v2_size      = torch.tensor([0.10, 0.03, 0.03], dtype=x.dtype, device=device).unsqueeze(0).expand(N, -1)

    col_v2 = check_collision_obb_batch(
        pts_world    = eef_pos,
        center_world = center_v2_B,
        R_world      = R_world_v2_B,
        half_size    = v2_size,
        margin       = margin_v2,
        device       = device,
        dtype        = x.dtype
    )

    # 4) 手爪 => 同理
    hand_pos = eef_pos.clone()
    hand_pos[:, 2] += 0.11
    # 小方块
    margin_box0_hand = [0.03 + margin_box0[0], 0.08 + margin_box0[1], 0.05 + margin_box0[2]]
    col_box_hand = check_collision_aabb_batch(
        pts_world    = hand_pos,
        center_world = box_pos,
        half_size    = box_size,
        margin       = margin_box0_hand,
        device       = device,
        dtype        = x.dtype
    )
    # v1
    margin_v1_hand = [0.03 + margin_v1[0], 0.08 + margin_v1[1], 0.05 + margin_v1[2]]
    col_v1_hand = check_collision_obb_batch(
        pts_world    = hand_pos,
        center_world = center_v1_B,
        R_world      = R_world_v1_B,
        half_size    = v1_size,
        margin       = margin_v1_hand,
        device       = device,
        dtype        = x.dtype
    )
    # v2
    margin_v2_hand = [0.03 + margin_v2[0], 0.08 + margin_v2[1], 0.05 + margin_v2[2]]
    col_v2_hand = check_collision_obb_batch(
        pts_world    = hand_pos,
        center_world = center_v2_B,
        R_world      = R_world_v2_B,
        half_size    = v2_size,
        margin       = margin_v2_hand,
        device       = device,
        dtype        = x.dtype
    )

    collision_mask = (col_box | col_v1 | col_v2 | col_box_hand | col_v1_hand | col_v2_hand)

    # 5) 工作空间/桌面约束
    table_collision = (eef_pos[:, 2] <= 0.40)
    dist_robot_base = torch.norm(eef_pos - robot_base_pos, dim=1)
    workspace_violation = (dist_robot_base >= 0.8)

    cost += args.ω2 * collision_mask
    cost += args.ω3 * table_collision
    cost += args.ω4 * workspace_violation

    collision = collision_mask
    return cost

##############################################################################
# !!!【STORM新增】 terminal_cost_vectorized
##############################################################################
def terminal_cost_vectorized(
    x,         # (K,14)
    eef_pos,   # (K,3)
    goal,      # (K,d)
    args,
    device,
    robot_base_pos
):
    """
    简单示例：只加终端到目标的距离
    """
    global collision
    cost = 10.0 * torch.norm(eef_pos - goal, dim=1) ** 2
    return cost

##############################################################################
# get_parameters(args)
##############################################################################
def get_parameters(args):
    """
    返回 (K, T, Δt, α,
          dynamics,
          None, None,
          Σ, λ,
          convert_to_target,
          dtype, device,
          dynamics_params )
    以兼容【STORM】mppi_policy.py
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
    device = 'cuda'  # 【STORM修改】GPU

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

    # 用于 mppi_policy.py => predict()
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

    robot_base_pos = torch.tensor([0.8, 0.75, 0.44], dtype=dtype, device=device)

    # (单层退火修改) => rollout_fn 新增 per_step_factor_inv
    dynamics_params = {
        "chain": chain,
        "robot_base_pos": robot_base_pos,
        "args": args,
        "rollout_fn": rollout_storm_no_loop_rear_v,
        "state_cost": state_cost_vectorized,
        "terminal_cost": terminal_cost_vectorized
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,  # 旧API兼容
        None,  # 旧API兼容
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params
    )




# # 用了欧拉角，没有去用四元数
#
# import os
# import pytorch_kinematics
# import torch
# import math
#
# ##############################################################################
# # 全局碰撞标志 (原样保留)
# ##############################################################################
# collision = torch.zeros([500, ], dtype=torch.bool)
#
# def getCollisions():
#     return collision
#
#
# ##############################################################################
# # ！！！这是新增函数：将欧拉角 (rx, ry, rz) 转成 3x3 旋转矩阵
# #   若与 MuJoCo 中 <geom euler="..."> 的定义完全匹配，需要根据实际情况
# #   调整旋转顺序。这里示例按 Z-Y-X 顺序( extrinsic X->Y->Z ) 给出。
# ##############################################################################
# def euler_to_matrix(euler_xyz, device, dtype):
#     """
#     以 Z-Y-X 的顺序将 euler_xyz = (rx, ry, rz) 转成 3x3 矩阵:
#       R = Rz(rz) * Ry(ry) * Rx(rx)
#     若需要和 MuJoCo 中 euler="..." 更严格匹配，请结合 XML 文件中注释进行调整。
#     """
#     rx, ry, rz = euler_xyz
#     sx, cx = math.sin(rx), math.cos(rx)
#     sy, cy = math.sin(ry), math.cos(ry)
#     sz, cz = math.sin(rz), math.cos(rz)
#
#     Rz = torch.tensor([[cz, -sz,  0],
#                        [sz,  cz,  0],
#                        [ 0,   0,  1]], dtype=dtype, device=device)
#     Ry = torch.tensor([[ cy, 0, sy],
#                        [  0, 1,  0],
#                        [-sy, 0, cy]], dtype=dtype, device=device)
#     Rx = torch.tensor([[1,  0,   0],
#                        [0, cx, -sx],
#                        [0, sx,  cx]], dtype=dtype, device=device)
#
#     # 最终 R = Rz * Ry * Rx
#     R = torch.mm(Rz, torch.mm(Ry, Rx))
#     return R
#
#
# ##############################################################################
# # ！！！这是原本的函数：OBB（定向包围盒）碰撞检测
# ##############################################################################
# def check_collision_obb(pts_world, center_world, R_world, half_size, margin, device, dtype):
#     """
#     判断每个点与给定 OBB 是否碰撞:
#       1) 将点转换到障碍物局部系 local_pt = R^T * (world_pt - center)
#       2) 若 local_pt 的绝对值在 (half_size + margin) 范围内，则视为碰撞
#     返回 bool 张量 [N,]，表示每个点是否碰撞。
#     """
#     center = torch.tensor(center_world, dtype=dtype, device=device)
#     R = torch.tensor(R_world, dtype=dtype, device=device)
#     R_t = R.transpose(0, 1)  # R 的转置，用于世界 -> 局部
#     local_pts = torch.matmul(pts_world - center, R_t)
#     dist_local = torch.abs(local_pts)
#
#     lim = torch.tensor(half_size, dtype=dtype, device=device) \
#         + torch.tensor(margin, dtype=dtype, device=device)
#     collision_mask = torch.all(dist_local <= lim, dim=1)
#     return collision_mask
#
#
# ##############################################################################
# # ！！！这是新增函数：AABB（轴对齐包围盒）碰撞检测
# ##############################################################################
# def check_collision_aabb(pts_world, center_world, half_size, margin, device, dtype):
#     """
#     对“轴对齐”包围盒(AABB)进行碰撞检测。
#       - pts_world: [N,3] 批量点（末端等）的世界坐标
#       - center_world: [3,] 障碍物中心(世界坐标)
#       - half_size: [3,] 障碍物的 X/Y/Z 半尺寸
#       - margin: [3,] 安全裕度
#     算法步骤:
#       1) 计算包围盒的 lower_bound = center_world - (half_size + margin)
#          以及 upper_bound = center_world + (half_size + margin)
#       2) 若世界坐标下的某个点 p 满足:
#          lower_bound[i] <= p[i] <= upper_bound[i] (对 i=0,1,2),
#          则说明该点与该 AABB 相交(碰撞)。
#     返回 bool 张量 [N,]，表示每个点是否碰撞。
#     """
#     center = torch.tensor(center_world, dtype=dtype, device=device)
#     half_sz = torch.tensor(half_size, dtype=dtype, device=device)
#     marg = torch.tensor(margin, dtype=dtype, device=device)
#
#     lower_bound = center - (half_sz + marg)
#     upper_bound = center + (half_sz + marg)
#
#     # 分别检查 x, y, z 维度是否都在 AABB 范围内
#     in_x = (pts_world[:, 0] >= lower_bound[0]) & (pts_world[:, 0] <= upper_bound[0])
#     in_y = (pts_world[:, 1] >= lower_bound[1]) & (pts_world[:, 1] <= upper_bound[1])
#     in_z = (pts_world[:, 2] >= lower_bound[2]) & (pts_world[:, 2] <= upper_bound[2])
#
#     collision_mask = in_x & in_y & in_z
#     return collision_mask
#
#
# def get_parameters(args):
#     # 原始参数设定，保持不变
#     if args.tune_mppi <= 0:
#         args.α = 0
#         args.λ = 60
#         args.σ = 0.20
#         args.χ = 0.0
#         args.ω1 = 1.0003
#         args.ω2 = 9.16e3
#         args.ω3 = 9.16e3
#         args.ω4 = 9.16e3
#         args.ω_Φ = 5.41
#         # args.d_goal = 0.3
#         args.d_goal = 0.15
#
#     K = 500
#     T = 10
#     Δt = 0.01
#     T_system = 0.011
#
#     dtype = torch.double
#     device = 'cpu'  # 或 'cuda'
#
#     α = args.α
#     λ = args.λ
#     Σ = args.σ * torch.tensor([
#         [1.5,  args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
#         [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, 1.0,  args.χ, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
#     ], dtype=dtype, device=device)
#
#     # 读取 franka_panda_arm.urdf，保持不变
#     MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
#     xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
#     dtype_kinematics = torch.double
#     chain = pytorch_kinematics.build_serial_chain_from_urdf(
#         xml, end_link_name="panda_link8", root_link_name="panda_link0"
#     )
#     chain = chain.to(dtype=dtype_kinematics, device=device)
#
#     # 机械臂基座在 Mujoco 场景中的平移偏移
#     robot_base_pos = torch.tensor([0.8, 0.75, 0.44], device=device, dtype=dtype_kinematics)
#
#     def dynamics(x, u):
#         new_vel = x[:, 7:14] + u
#         new_pos = x[:, 0:7] + new_vel * Δt
#         return torch.cat((new_pos, new_vel), dim=1)
#
#     ###########################################################################
#     # ！！！这是修改后最终版本的 state_cost，参考 pick_dyn_front... 的写法。
#     #     注意到:
#     #       1) 第一个障碍物(小蓝方块)用 AABB
#     #       2) 第二个障碍物(后 V 形槽)拆分为 v1、v2 两块 OBB
#     #       3) 统一对 numpy -> torch 的转换, 然后 .unsqueeze(0) 再使用 quaternion_to_matrix
#     ###########################################################################
#     def state_cost(x, goal, obstacles):
#         global collision
#         batch_size = x.shape[0]
#
#         # 1) 正向运动学，得到 panda_link8(末端) 在世界坐标的位置
#         joint_values = x[:, 0:7]
#         ret = chain.forward_kinematics(joint_values, end_only=False)
#         link8_matrix = ret['panda_link8'].get_matrix()  # [batch_size, 4, 4]
#         link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # [batch_size, 3]
#
#         # 2) 计算距离目标点的代价
#         goal_dist = torch.norm(link8_pos - goal, dim=1)
#         cost = 1000.0 * (goal_dist ** 2)
#
#         # 3) 计算工作空间约束(例如超出一定范围则加大cost)
#         dist_robot_base = torch.norm(link8_pos - robot_base_pos, dim=1)
#
#         # -----------------------------
#         # 第一个障碍物(小蓝方块) => 用 AABB
#         # obstacles[0:3] => (x, y, z)
#         # obstacles[3:7] => [w, x, y, z]
#         # obstacles[7:10] => (sx, sy, sz)
#         # -----------------------------
#         obs0_pos_np = obstacles[0:3]
#         obs0_size_np = obstacles[7:10]
#
#         # AABB margin
#         margin_box0 = [0.055, 0.055, 0.03]
#
#         col0 = check_collision_aabb(
#             pts_world=link8_pos,
#             center_world=obs0_pos_np,
#             half_size=obs0_size_np,
#             margin=margin_box0,
#             device=device,
#             dtype=dtype
#         )
#
#         # -----------------------------
#         # 第二个障碍物(后 V 形槽) => OBB + 拆成两个斜板
#         #
#         # obstacles[10:13] => body pos
#         # obstacles[13:17] => (w, x, y, z) body quat
#         # obstacles[17:20] => half-size (如 0.12,0.03,0.03), 但我们后面拆v1,v2
#         # -----------------------------
#         v_body_pos_np = obstacles[10:13]
#         v_body_quat_np = obstacles[13:17]
#
#         # 转成 torch
#         v_body_pos_t = torch.tensor(v_body_pos_np, dtype=dtype, device=device)
#         v_body_quat_t = torch.tensor(v_body_quat_np, dtype=dtype, device=device).unsqueeze(0)
#
#         # body旋转矩阵
#         Rv_body = pytorch_kinematics.quaternion_to_matrix(v_body_quat_t)[0]
#
#         # -------- v1 左侧斜板 --------
#         v1_local_pos = [-0.06, 0.0, 0.0]
#         v1_euler = [0.0, 0.0, 2.35619]
#         v1_size = [0.10, 0.03, 0.03]
#
#         Rv1_local = euler_to_matrix(v1_euler, device=device, dtype=dtype)
#         v1_lpos_t = torch.tensor(v1_local_pos, dtype=dtype, device=device)
#
#         world_pos_v1 = v_body_pos_t + torch.matmul(Rv_body, v1_lpos_t)
#         R_world_v1 = torch.matmul(Rv_body, Rv1_local)
#         margin_v1 = [0.065, 0.07, 0.03]
#
#         col_v1 = check_collision_obb(
#             pts_world=link8_pos,
#             center_world=world_pos_v1,
#             R_world=R_world_v1,
#             half_size=v1_size,
#             margin=margin_v1,
#             device=device,
#             dtype=dtype
#         )
#
#         # -------- v2 右侧斜板 --------
#         v2_local_pos = [0.06, 0.0, 0.0]
#         v2_euler = [0.0, 0.0, 0.785398]
#         v2_size = [0.10, 0.03, 0.03]
#
#         Rv2_local = euler_to_matrix(v2_euler, device=device, dtype=dtype)
#         v2_lpos_t = torch.tensor(v2_local_pos, dtype=dtype, device=device)
#         world_pos_v2 = v_body_pos_t + torch.matmul(Rv_body, v2_lpos_t)
#         R_world_v2 = torch.matmul(Rv_body, Rv2_local)
#         margin_v2 = [0.065, 0.07, 0.03]
#
#         col_v2 = check_collision_obb(
#             pts_world=link8_pos,
#             center_world=world_pos_v2,
#             R_world=R_world_v2,
#             half_size=v2_size,
#             margin=margin_v2,
#             device=device,
#             dtype=dtype
#         )
#
#         # -----------------------------
#         # 手爪与障碍物的碰撞：小蓝块(AABB) + v槽(OBB)
#         # -----------------------------
#         hand_pos = link8_pos.clone()
#         hand_pos[:, 2] += 0.11  # 简易近似: 手爪顶端
#
#         # 小蓝方块(AABB)
#         margin_box0_hand_with_obstacle = [
#             0.03 + margin_box0[0],
#             0.08 + margin_box0[1],
#             0.05 + margin_box0[2]
#         ]
#         col0_hand = check_collision_aabb(
#             pts_world=hand_pos,
#             center_world=obs0_pos_np,
#             half_size=obs0_size_np,
#             margin=margin_box0_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # v1
#         margin_v1_hand_with_obstacle = [
#             0.03 + margin_v1[0],
#             0.08 + margin_v1[1],
#             0.05 + margin_v1[2]
#         ]
#         col_v1_hand = check_collision_obb(
#             pts_world=hand_pos,
#             center_world=world_pos_v1,
#             R_world=R_world_v1,
#             half_size=v1_size,
#             margin=margin_v1_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # v2
#         margin_v2_hand_with_obstacle = [
#             0.03 + margin_v2[0],
#             0.08 + margin_v2[1],
#             0.05 + margin_v2[2]
#         ]
#         col_v2_hand = check_collision_obb(
#             pts_world=hand_pos,
#             center_world=world_pos_v2,
#             R_world=R_world_v2,
#             half_size=v2_size,
#             margin=margin_v2_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # 组合所有碰撞结果
#         collision_mask = col0 | col_v1 | col_v2 | col0_hand | col_v1_hand | col_v2_hand
#
#         # 桌面碰撞 (末端 z 低于某个值)
#         table_collision = (link8_pos[:, 2] <= 0.40)
#         # 工作空间限制
#         workspace_costs = (dist_robot_base >= 0.8)
#
#         # 累加进 cost
#         cost += args.ω2 * collision_mask
#         cost += args.ω3 * table_collision
#         cost += args.ω4 * workspace_costs
#
#         # 更新全局 collision (可每个 step 清空，也可累积)
#         collision = collision_mask
#         collision = torch.zeros([batch_size], dtype=torch.bool, device=device)
#
#         return cost
#
#     # GoalEnv methods
#     # ---------------
#     def terminal_cost(x, goal):
#         global collision
#         joint_values = x[:, 0:7]
#         ret = chain.forward_kinematics(joint_values, end_only=True)
#
#         eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
#         cost = 10 * torch.norm((eef_pos - goal), dim=1) ** 2
#         collision = torch.zeros([500, ], dtype=torch.bool)
#         return cost
#
#     def convert_to_target(x, u):
#         joint_pos = x[0:7]
#         joint_vel = x[7:14]
#         # 原逻辑
#         new_vel = joint_vel + u / (
#             1 - torch.exp(torch.tensor(-Δt / T_system)) * (1 + (Δt / T_system))
#         )
#         new_joint_pos = joint_pos + new_vel * Δt
#
#         # 计算末端在世界坐标下的位置(仅作一些可视化或记录)
#         ret = chain.forward_kinematics(new_joint_pos, end_only=True)
#         eef_matrix = ret.get_matrix()
#         eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
#         eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
#
#         return torch.cat((eef_pos, eef_rot), dim=1)
#
#     return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device



# # 【这个版本修改的地方】 修正障碍物四元数的解析，确保与环境中 dyn_obstacles 存储的一致性（[w, x, y, z] 格式）；
#
# import os
# import pytorch_kinematics
# import torch
# import math
#
# ##############################################################################
# # 全局碰撞标志 (原样保留)
# ##############################################################################
# collision = torch.zeros([500, ], dtype=torch.bool)
#
# def getCollisions():
#     return collision
#
# ##############################################################################
# # ！！！这是新增函数：将欧拉角 (rx, ry, rz) 转成 3x3 旋转矩阵
# #   若与 MuJoCo 中 <geom euler="..."> 的定义完全匹配，需要根据实际情况
# #   调整旋转顺序。这里示例按 Z-Y-X 顺序( extrinsic X->Y->Z ) 给出。
# ##############################################################################
# def euler_to_matrix(euler_xyz, device, dtype):
#     """
#     以 Z-Y-X 的顺序将 euler_xyz = (rx, ry, rz) 转成 3x3 矩阵:
#       R = Rz(rz) * Ry(ry) * Rx(rx)
#     若需要和 MuJoCo 中 euler="..." 更严格匹配，请结合 XML 文件中注释进行调整。
#     """
#     rx, ry, rz = euler_xyz
#     sx, cx = math.sin(rx), math.cos(rx)
#     sy, cy = math.sin(ry), math.cos(ry)
#     sz, cz = math.sin(rz), math.cos(rz)
#
#     Rz = torch.tensor([[cz, -sz,  0],
#                        [sz,  cz,  0],
#                        [ 0,   0,  1]], dtype=dtype, device=device)
#     Ry = torch.tensor([[ cy, 0, sy],
#                        [  0, 1,  0],
#                        [-sy, 0, cy]], dtype=dtype, device=device)
#     Rx = torch.tensor([[1,  0,   0],
#                        [0, cx, -sx],
#                        [0, sx,  cx]], dtype=dtype, device=device)
#
#     # 最终 R = Rz * Ry * Rx
#     R = torch.mm(Rz, torch.mm(Ry, Rx))
#     return R
#
# ##############################################################################
# # ！！！这是原本的函数：OBB（定向包围盒）碰撞检测
# ##############################################################################
# def check_collision_obb(pts_world, center_world, R_world, half_size, margin, device, dtype):
#     """
#     判断每个点与给定 OBB 是否碰撞:
#       1) 将点转换到障碍物局部系 local_pt = R^T * (world_pt - center)
#       2) 若 local_pt 的绝对值在 (half_size + margin) 范围内，则视为碰撞
#     返回 bool 张量 [N,]，表示每个点是否碰撞。
#     """
#     center = torch.tensor(center_world, dtype=dtype, device=device)
#     R = torch.tensor(R_world, dtype=dtype, device=device)
#     R_t = R.transpose(0, 1)  # R 的转置，用于世界 -> 局部
#     local_pts = torch.matmul(pts_world - center, R_t)
#     dist_local = torch.abs(local_pts)
#
#     lim = torch.tensor(half_size, dtype=dtype, device=device) \
#         + torch.tensor(margin, dtype=dtype, device=device)
#     collision_mask = torch.all(dist_local <= lim, dim=1)
#     return collision_mask
#
# ##############################################################################
# # ！！！这是新增函数：AABB（轴对齐包围盒）碰撞检测
# ##############################################################################
# def check_collision_aabb(pts_world, center_world, half_size, margin, device, dtype):
#     """
#     对“轴对齐”包围盒(AABB)进行碰撞检测。
#       - pts_world: [N,3] 批量点（末端等）的世界坐标
#       - center_world: [3,] 障碍物中心(世界坐标)
#       - half_size: [3,] 障碍物的 X/Y/Z 半尺寸
#       - margin: [3,] 安全裕度
#     算法步骤:
#       1) 计算包围盒的 lower_bound = center_world - (half_size + margin)
#          以及 upper_bound = center_world + (half_size + margin)
#       2) 若世界坐标下的某个点 p 满足:
#          lower_bound[i] <= p[i] <= upper_bound[i] (对 i=0,1,2),
#          则说明该点与该 AABB 相交(碰撞)。
#     返回 bool 张量 [N,]，表示每个点是否碰撞。
#     """
#     center = torch.tensor(center_world, dtype=dtype, device=device)
#     half_sz = torch.tensor(half_size, dtype=dtype, device=device)
#     marg = torch.tensor(margin, dtype=dtype, device=device)
#
#     lower_bound = center - (half_sz + marg)
#     upper_bound = center + (half_sz + marg)
#
#     # 分别检查 x, y, z 维度是否都在 AABB 范围内
#     in_x = (pts_world[:, 0] >= lower_bound[0]) & (pts_world[:, 0] <= upper_bound[0])
#     in_y = (pts_world[:, 1] >= lower_bound[1]) & (pts_world[:, 1] <= upper_bound[1])
#     in_z = (pts_world[:, 2] >= lower_bound[2]) & (pts_world[:, 2] <= upper_bound[2])
#
#     collision_mask = in_x & in_y & in_z
#     return collision_mask
#
# def get_parameters(args):
#     # 原始参数设定，保持不变
#     if args.tune_mppi <= 0:
#         args.α = 0
#         args.λ = 60
#         args.σ = 0.20
#         args.χ = 0.0
#         args.ω1 = 1.0003
#         args.ω2 = 9.16e3
#         args.ω3 = 9.16e3
#         args.ω4 = 9.16e3
#         args.ω_Φ = 5.41
#         args.d_goal = 0.15
#
#     K = 500
#     T = 10
#     Δt = 0.01
#     T_system = 0.011
#
#     dtype = torch.double
#     device = 'cpu'  # 或 'cuda'
#
#     α = args.α
#     λ = args.λ
#     Σ = args.σ * torch.tensor([
#         [1.5,  args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
#         [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, 1.0,  args.χ, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
#     ], dtype=dtype, device=device)
#
#     # 读取 franka_panda_arm.urdf，保持不变
#     MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
#     xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
#     dtype_kinematics = torch.double
#     chain = pytorch_kinematics.build_serial_chain_from_urdf(
#         xml, end_link_name="panda_link8", root_link_name="panda_link0"
#     )
#     chain = chain.to(dtype=dtype_kinematics, device=device)
#
#     # 机械臂基座在 Mujoco 场景中的平移偏移
#     robot_base_pos = torch.tensor([0.8, 0.75, 0.44], device=device, dtype=dtype_kinematics)
#
#     def dynamics(x, u):
#         new_vel = x[:, 7:14] + u
#         new_pos = x[:, 0:7] + new_vel * Δt
#         return torch.cat((new_pos, new_vel), dim=1)
#
#     ###########################################################################
#     # ！！！这是修改的地方：修正对障碍物四元数的解释，确保与环境一致，
#     # 并将 numpy -> torch 后再 .unsqueeze(0)。
#     #
#     # 注：因为 obstacles 是 numpy array，必须先转成 torch tensor，
#     # 才能使用 unsqueeze()
#     ###########################################################################
#     def state_cost(x, goal, obstacles):
#         global collision
#         batch_size = x.shape[0]
#
#         # 1) 正向运动学，得到 panda_link8(末端) 在世界坐标的位置
#         joint_values = x[:, 0:7]
#         ret = chain.forward_kinematics(joint_values, end_only=False)
#         link8_matrix = ret['panda_link8'].get_matrix()  # [batch_size, 4, 4]
#         link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # [batch_size, 3]
#
#         # 2) 计算距离目标点的代价
#         goal_dist = torch.norm(link8_pos - goal, dim=1)
#         cost = 1000.0 * (goal_dist ** 2)
#
#         # 3) 计算工作空间约束(例如超出一定范围则加大cost)
#         dist_robot_base = torch.norm(link8_pos - robot_base_pos, dim=1)
#
#         # -----------------------------
#         # 第一个 obstacle(小蓝方块) => 用 AABB
#         # 注意：obstacles[0:3] => (x, y, z)
#         #       obstacles[3:7] => [w, x, y, z]
#         #       obstacles[7:10] => (sx, sy, sz)
#         # -----------------------------
#         obs0_pos_np = obstacles[0:3]
#         # obs0_quat_np = obstacles[3:7]  # 对 AABB 不需要用 quat
#         obs0_size_np = obstacles[7:10]
#         #margin_box0 = [0.03, 0.03, 0.03]
#         margin_box0 = [0.055, 0.055, 0.03]
#
#         col0 = check_collision_aabb(
#             pts_world=link8_pos,
#             center_world=obs0_pos_np,
#             half_size=obs0_size_np,
#             margin=margin_box0,
#             device=device,
#             dtype=dtype
#         )
#
#         # -----------------------------
#         # 第二个 obstacle(后 V 形槽) => 继续用 OBB
#         #   obstacles[10:13] => (x, y, z) 位置
#         #   obstacles[13:17] => (w, x, y, z) 四元数
#         #   obstacles[17:20] => (sx, sy, sz) half-size
#         # -----------------------------
#         v_body_pos_np = obstacles[10:13]
#         v_body_quat_np = obstacles[13:17]  # 这里是 numpy.ndarray
#         # v_body_size_np = obstacles[17:20] # 本例仅拆分v1,v2使用
#
#         # ！！将它们转换为 torch.tensor
#         v_body_pos_t = torch.tensor(v_body_pos_np, dtype=dtype, device=device)
#         v_body_quat_t = torch.tensor(v_body_quat_np, dtype=dtype, device=device).unsqueeze(0)
#         # ↑↑↑ 修正：先转成 tensor，再用 .unsqueeze(0)
#
#         # 然后再计算旋转矩阵
#         Rv_body = pytorch_kinematics.quaternion_to_matrix(v_body_quat_t)[0]  # (3,3)
#
#         # -------- v1 对应 左侧斜板 --------
#         v1_local_pos = [-0.06, 0.0, 0.0]
#         v1_euler = [0.0, 0.0, 2.35619]   # z轴旋转
#         v1_size = [0.10, 0.03, 0.03]
#
#         Rv1_local = euler_to_matrix(v1_euler, device=device, dtype=dtype)
#         v1_lpos_t = torch.tensor(v1_local_pos, dtype=dtype, device=device)
#
#         world_pos_v1 = v_body_pos_t + torch.matmul(Rv_body, v1_lpos_t)
#         R_world_v1 = torch.matmul(Rv_body, Rv1_local)
#         margin_v1 = [0.065, 0.07, 0.03]
#
#         col_v1 = check_collision_obb(
#             pts_world=link8_pos,
#             center_world=world_pos_v1,
#             R_world=R_world_v1,
#             half_size=v1_size,
#             margin=margin_v1,
#             device=device,
#             dtype=dtype
#         )
#
#         # -------- v2 对应 右侧斜板 --------
#         v2_local_pos = [0.06, 0.0, 0.0]
#         v2_euler = [0.0, 0.0, 0.785398]
#         v2_size = [0.10, 0.03, 0.03]
#
#         Rv2_local = euler_to_matrix(v2_euler, device=device, dtype=dtype)
#         v2_lpos_t = torch.tensor(v2_local_pos, dtype=dtype, device=device)
#
#         world_pos_v2 = v_body_pos_t + torch.matmul(Rv_body, v2_lpos_t)
#         R_world_v2 = torch.matmul(Rv_body, Rv2_local)
#         margin_v2 = [0.065, 0.07, 0.03]
#
#         col_v2 = check_collision_obb(
#             pts_world=link8_pos,
#             center_world=world_pos_v2,
#             R_world=R_world_v2,
#             half_size=v2_size,
#             margin=margin_v2,
#             device=device,
#             dtype=dtype
#         )
#
#         # -----------------------------
#         # 手爪与障碍物的碰撞：小蓝块(AABB) + v槽(OBB)
#         # -----------------------------
#         # 手爪碰撞(末端位置 + 偏移 + 简易 box)
#         hand_pos = link8_pos.clone()
#         hand_pos[:, 2] += 0.11  # 原作者的简化做法
#
#         # ------ 手爪 + 小蓝块(AABB) ------
#         margin_box0_hand_with_obstacle = [
#             0.03 + margin_box0[0],
#             0.08 + margin_box0[1],
#             0.05 + margin_box0[2]
#         ]
#         col0_hand = check_collision_aabb(
#             pts_world=hand_pos,
#             center_world=obs0_pos_np,
#             half_size=obs0_size_np,
#             margin=margin_box0_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # ------ 手爪 + v1(OBB) ------
#         margin_v1_hand_with_obstacle = [
#             0.03 + margin_v1[0],
#             0.08 + margin_v1[1],
#             0.05 + margin_v1[2]
#         ]
#         col_v1_hand = check_collision_obb(
#             pts_world=hand_pos,
#             center_world=world_pos_v1,
#             R_world=R_world_v1,
#             half_size=v1_size,
#             margin=margin_v1_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # ------ 手爪 + v2(OBB) ------
#         margin_v2_hand_with_obstacle = [
#             0.03 + margin_v2[0],
#             0.08 + margin_v2[1],
#             0.05 + margin_v2[2]
#         ]
#         col_v2_hand = check_collision_obb(
#             pts_world=hand_pos,
#             center_world=world_pos_v2,
#             R_world=R_world_v2,
#             half_size=v2_size,
#             margin=margin_v2_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # 组合所有碰撞结果
#         collision_mask = col0 | col_v1 | col_v2 | col0_hand | col_v1_hand | col_v2_hand
#
#         # 桌面碰撞 (末端 z 低于某个值)
#         table_collision = (link8_pos[:, 2] <= 0.40)
#         # 工作空间限制
#         workspace_costs = (dist_robot_base >= 0.8)
#
#         # 累加进 cost
#         cost += args.ω2 * collision_mask
#         cost += args.ω3 * table_collision
#         cost += args.ω4 * workspace_costs
#
#         # 更新全局 collision (本例中按 step 清空也可)
#         collision = collision_mask
#         collision = torch.zeros([batch_size], dtype=torch.bool, device=device)
#
#         return cost
#
#     # 原始的终端代价函数，保持大体不变
#     def terminal_cost(x, goal):
#         global collision
#         joint_values = x[:, 0:7]
#         ret = chain.forward_kinematics(joint_values, end_only=True)
#
#         eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
#         cost = 10 * torch.norm((eef_pos - goal), dim=1) ** 2
#         collision = torch.zeros([500, ], dtype=torch.bool)
#         return cost
#
#     def convert_to_target(x, u):
#         joint_pos = x[0:7]
#         joint_vel = x[7:14]
#         # 原逻辑
#         new_vel = joint_vel + u / (
#             1 - torch.exp(torch.tensor(-Δt / T_system)) * (1 + (Δt / T_system))
#         )
#         new_joint_pos = joint_pos + new_vel * Δt
#
#         # 计算末端在世界坐标下的位置(仅作一些可视化或记录)
#         ret = chain.forward_kinematics(new_joint_pos, end_only=True)
#         eef_matrix = ret.get_matrix()
#         eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
#         eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
#
#         return torch.cat((eef_pos, eef_rot), dim=1)
#
#     return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device


# ################################################################################################################################

# import os
# import pytorch_kinematics
# import torch
# import math
#
# ##############################################################################
# # 全局碰撞标志 (原样保留)
# ##############################################################################
# collision = torch.zeros([500, ], dtype=torch.bool)
#
# def getCollisions():
#     return collision
#
# ##############################################################################
# # ！！！这是新增函数：将欧拉角 (rx, ry, rz) 转成 3x3 旋转矩阵
# #   若与 MuJoCo 中 <geom euler="..."> 的定义完全匹配，需要根据实际情况
# #   调整旋转顺序。这里示例按 Z-Y-X 顺序( extrinsic X->Y->Z ) 给出。
# ##############################################################################
# def euler_to_matrix(euler_xyz, device, dtype):
#     """
#     以 Z-Y-X 的顺序将 euler_xyz = (rx, ry, rz) 转成 3x3 矩阵:
#       R = Rz(rz) * Ry(ry) * Rx(rx)
#     若需要和 MuJoCo 中 euler="..." 更严格匹配，请结合 XML 文件中注释进行调整。
#     """
#     rx, ry, rz = euler_xyz
#     sx, cx = math.sin(rx), math.cos(rx)
#     sy, cy = math.sin(ry), math.cos(ry)
#     sz, cz = math.sin(rz), math.cos(rz)
#
#     Rz = torch.tensor([[cz, -sz,  0],
#                        [sz,  cz,  0],
#                        [ 0,   0,  1]], dtype=dtype, device=device)
#     Ry = torch.tensor([[ cy, 0, sy],
#                        [  0, 1,  0],
#                        [-sy, 0, cy]], dtype=dtype, device=device)
#     Rx = torch.tensor([[1,  0,   0],
#                        [0, cx, -sx],
#                        [0, sx,  cx]], dtype=dtype, device=device)
#
#     # 最终 R = Rz * Ry * Rx
#     R = torch.mm(Rz, torch.mm(Ry, Rx))
#     return R
#
# ##############################################################################
# # ！！！这是原本的函数：OBB（定向包围盒）碰撞检测
# ##############################################################################
# def check_collision_obb(pts_world, center_world, R_world, half_size, margin, device, dtype):
#     """
#     判断每个点与给定 OBB 是否碰撞:
#       1) 将点转换到障碍物局部系 local_pt = R^T * (world_pt - center)
#       2) 若 local_pt 的绝对值在 (half_size + margin) 范围内，则视为碰撞
#     返回 bool 张量 [N,]，表示每个点是否碰撞。
#     """
#     center = torch.tensor(center_world, dtype=dtype, device=device)
#     R = torch.tensor(R_world, dtype=dtype, device=device)
#     R_t = R.transpose(0, 1)  # R 的转置，用于世界 -> 局部
#     local_pts = torch.matmul(pts_world - center, R_t)
#     dist_local = torch.abs(local_pts)
#
#     lim = torch.tensor(half_size, dtype=dtype, device=device) \
#         + torch.tensor(margin, dtype=dtype, device=device)
#     collision_mask = torch.all(dist_local <= lim, dim=1)
#     return collision_mask
#
# ##############################################################################
# # ！！！这是新增函数：AABB（轴对齐包围盒）碰撞检测
# ##############################################################################
# def check_collision_aabb(pts_world, center_world, half_size, margin, device, dtype):
#     """
#     对“轴对齐”包围盒(AABB)进行碰撞检测。
#       - pts_world: [N,3] 批量点（末端等）的世界坐标
#       - center_world: [3,] 障碍物中心(世界坐标)
#       - half_size: [3,] 障碍物的 X/Y/Z 半尺寸
#       - margin: [3,] 安全裕度
#     算法步骤:
#       1) 计算包围盒的 lower_bound = center_world - (half_size + margin)
#          以及 upper_bound = center_world + (half_size + margin)
#       2) 若世界坐标下的某个点 p 满足:
#          lower_bound[i] <= p[i] <= upper_bound[i] (对 i=0,1,2),
#          则说明该点与该 AABB 相交(碰撞)。
#     返回 bool 张量 [N,]，表示每个点是否碰撞。
#     """
#     center = torch.tensor(center_world, dtype=dtype, device=device)
#     half_sz = torch.tensor(half_size, dtype=dtype, device=device)
#     marg = torch.tensor(margin, dtype=dtype, device=device)
#
#     lower_bound = center - (half_sz + marg)
#     upper_bound = center + (half_sz + marg)
#
#     # 分别检查 x, y, z 维度是否都在 AABB 范围内
#     in_x = (pts_world[:, 0] >= lower_bound[0]) & (pts_world[:, 0] <= upper_bound[0])
#     in_y = (pts_world[:, 1] >= lower_bound[1]) & (pts_world[:, 1] <= upper_bound[1])
#     in_z = (pts_world[:, 2] >= lower_bound[2]) & (pts_world[:, 2] <= upper_bound[2])
#
#     collision_mask = in_x & in_y & in_z
#     return collision_mask
#
# def get_parameters(args):
#     # 原始参数设定，保持不变
#     if args.tune_mppi <= 0:
#         args.α = 0
#         args.λ = 60
#         args.σ = 0.20
#         args.χ = 0.0
#         args.ω1 = 1.0003
#         args.ω2 = 9.16e3
#         args.ω3 = 9.16e3
#         args.ω4 = 9.16e3
#         args.ω_Φ = 5.41
#         args.d_goal = 0.15
#
#     K = 500
#     T = 10
#     Δt = 0.01
#     T_system = 0.011
#
#     dtype = torch.double
#     device = 'cpu'  # 或 'cuda'
#
#     α = args.α
#     λ = args.λ
#     Σ = args.σ * torch.tensor([
#         [1.5,  args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
#         [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, 1.0,  args.χ, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
#         [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
#     ], dtype=dtype, device=device)
#
#     # 读取 franka_panda_arm.urdf，保持不变
#     MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
#     xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
#     dtype_kinematics = torch.double
#     chain = pytorch_kinematics.build_serial_chain_from_urdf(
#         xml, end_link_name="panda_link8", root_link_name="panda_link0"
#     )
#     chain = chain.to(dtype=dtype_kinematics, device=device)
#
#     # 机械臂基座在 Mujoco 场景中的平移偏移
#     robot_base_pos = torch.tensor([0.8, 0.75, 0.44], device=device, dtype=dtype_kinematics)
#
#     def dynamics(x, u):
#         new_vel = x[:, 7:14] + u
#         new_pos = x[:, 0:7] + new_vel * Δt
#         return torch.cat((new_pos, new_vel), dim=1)
#
#     ###########################################################################
#     # ！！！这是修改的地方：蓝色方块改为 AABB，其它障碍物保持 OBB
#     ###########################################################################
#     def state_cost(x, goal, obstacles):
#         global collision
#         batch_size = x.shape[0]
#
#         # 1) 正向运动学，得到 panda_link8(末端) 在世界坐标的位置
#         joint_values = x[:, 0:7]
#         ret = chain.forward_kinematics(joint_values, end_only=False)
#         link8_matrix = ret['panda_link8'].get_matrix()  # [batch_size, 4, 4]
#         link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # [batch_size, 3]
#
#         # 2) 计算距离目标点的代价
#         goal_dist = torch.norm(link8_pos - goal, dim=1)
#         # cost = 1000.0 * (goal_dist ** 2)
#         cost = 500.0 * (goal_dist ** 2)
#
#         # 3) 计算工作空间约束(例如超出一定范围则加大cost)
#         dist_robot_base = torch.norm(link8_pos - robot_base_pos, dim=1)
#
#         # -----------------------------
#         # 第一个 obstacle(小蓝方块) => 用 AABB
#         # -----------------------------
#         # obstacles 的结构假设为: [x, y, z, qx, qy, qz, qw, sx, sy, sz] ...
#         obs0_pos = obstacles[0:3]    # (x, y, z)
#         # 我们虽然拿到 quaternion，但对于 AABB 不再使用
#         # obs0_quat = obstacles[3:7]   # (qx, qy, qz, qw)  <-- 可忽略
#         obs0_size = obstacles[7:10]  # (sx, sy, sz)
#         # 给第一个障碍物一些安全裕度
#         # margin_box0 = [0.08, 0.08, 0.03]
#         margin_box0 = [0.03, 0.03, 0.03]
#
#         # 计算末端与小蓝方块的碰撞 (AABB)
#         col0 = check_collision_aabb(
#             pts_world=link8_pos,
#             center_world=obs0_pos,
#             half_size=obs0_size,
#             margin=margin_box0,
#             device=device,
#             dtype=dtype
#         )
#
#         # -----------------------------
#         # 第二个 obstacle(后 V 形槽) => 继续用 OBB
#         # -----------------------------
#         #   obstacles[10:13] => pos
#         #   obstacles[13:17] => quat
#         #   obstacles[17:20] => half-size(可视为这个 V 槽 body 的最大外包盒，但要拆成v1和v2)
#         v_body_pos = obstacles[10:13]
#         v_body_quat = obstacles[13:17]
#
#         # V 槽 body 的旋转矩阵 (供后面v1/v2用)
#         vq_t = torch.tensor(v_body_quat, dtype=dtype, device=device).unsqueeze(0)
#         Rv_body = pytorch_kinematics.quaternion_to_matrix(vq_t)[0]  # shape (3,3)
#
#         # -------- v1 对应 左侧斜板 --------
#         v1_local_pos = [-0.06, 0.0, 0.0]
#         v1_euler = [0.0, 0.0, 2.35619]   # z轴旋转
#         v1_size = [0.10, 0.03, 0.03]
#
#         Rv1_local = euler_to_matrix(v1_euler, device=device, dtype=dtype)
#         v1_lpos_t = torch.tensor(v1_local_pos, dtype=dtype, device=device)
#         v_body_pos_t = torch.tensor(v_body_pos, dtype=dtype, device=device)
#
#         # V 槽 body -> world
#         world_pos_v1 = v_body_pos_t + torch.matmul(Rv_body, v1_lpos_t)
#         R_world_v1 = torch.matmul(Rv_body, Rv1_local)
#         margin_v1 = [0.065, 0.07, 0.03]  # 适当留点安全余量
#         # margin_v1 = [0.03, 0.03, 0.03]  # 适当留点安全余量
#
#         col_v1 = check_collision_obb(
#             pts_world=link8_pos,
#             center_world=world_pos_v1,
#             R_world=R_world_v1,
#             half_size=v1_size,
#             margin=margin_v1,
#             device=device,
#             dtype=dtype
#         )
#
#         # -------- v2 对应 右侧斜板 --------
#         v2_local_pos = [0.06, 0.0, 0.0]
#         v2_euler = [0.0, 0.0, 0.785398]
#         v2_size = [0.10, 0.03, 0.03]
#
#         Rv2_local = euler_to_matrix(v2_euler, device=device, dtype=dtype)
#         v2_lpos_t = torch.tensor(v2_local_pos, dtype=dtype, device=device)
#         world_pos_v2 = v_body_pos_t + torch.matmul(Rv_body, v2_lpos_t)
#         R_world_v2 = torch.matmul(Rv_body, Rv2_local)
#         margin_v2 = [0.065, 0.07, 0.03]  # 适当留点安全余量
#         # margin_v2 = [0.03, 0.03, 0.03]  # 适当留点安全余量
#
#         col_v2 = check_collision_obb(
#             pts_world=link8_pos,
#             center_world=world_pos_v2,
#             R_world=R_world_v2,
#             half_size=v2_size,
#             margin=margin_v2,
#             device=device,
#             dtype=dtype
#         )
#
#         # -----------------------------
#         # 手爪与障碍物的碰撞：小蓝块(AABB) + v槽(OBB)
#         # -----------------------------
#         # 手爪碰撞(末端位置 + 偏移 + 简易 box)
#         hand_pos = link8_pos.clone()
#         hand_pos[:, 2] += 0.11  # 原作者的简化做法
#
#         # ------ 手爪 + 小蓝块(AABB) ------
#         margin_box0_hand_with_obstacle = [
#             0.03 + margin_box0[0],
#             0.08 + margin_box0[1],
#             0.05 + margin_box0[2]
#         ]
#         col0_hand = check_collision_aabb(
#             pts_world=hand_pos,
#             center_world=obs0_pos,
#             half_size=obs0_size,
#             margin=margin_box0_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # ------ 手爪 + v1(OBB) ------
#         margin_v1_hand_with_obstacle = [
#             0.03 + margin_v1[0],
#             0.08 + margin_v1[1],
#             0.05 + margin_v1[2]
#         ]
#         col_v1_hand = check_collision_obb(
#             pts_world=hand_pos,
#             center_world=world_pos_v1,
#             R_world=R_world_v1,
#             half_size=v1_size,
#             margin=margin_v1_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # ------ 手爪 + v2(OBB) ------
#         margin_v2_hand_with_obstacle = [
#             0.03 + margin_v2[0],
#             0.08 + margin_v2[1],
#             0.05 + margin_v2[2]
#         ]
#         col_v2_hand = check_collision_obb(
#             pts_world=hand_pos,
#             center_world=world_pos_v2,
#             R_world=R_world_v2,
#             half_size=v2_size,
#             margin=margin_v2_hand_with_obstacle,
#             device=device,
#             dtype=dtype
#         )
#
#         # 组合所有碰撞结果
#         collision_mask = col0 | col_v1 | col_v2 | col0_hand | col_v1_hand | col_v2_hand
#
#         # 桌面碰撞 (末端 z 低于某个值)
#         table_collision = (link8_pos[:, 2] <= 0.40)
#         # 工作空间限制
#         workspace_costs = (dist_robot_base >= 0.8)
#
#         # 累加进 cost
#         cost += args.ω2 * collision_mask
#         cost += args.ω3 * table_collision
#         cost += args.ω4 * workspace_costs
#
#         # 更新全局 collision (本例中按 step 清空也可)
#         collision = collision_mask
#         collision = torch.zeros([batch_size], dtype=torch.bool, device=device)
#
#         return cost
#
#     # 原始的终端代价函数，保持大体不变
#     def terminal_cost(x, goal):
#         global collision
#         joint_values = x[:, 0:7]
#         ret = chain.forward_kinematics(joint_values, end_only=True)
#
#         eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
#         cost = 10 * torch.norm((eef_pos - goal), dim=1) ** 2
#         collision = torch.zeros([500, ], dtype=torch.bool)
#         return cost
#
#     def convert_to_target(x, u):
#         joint_pos = x[0:7]
#         joint_vel = x[7:14]
#         # 原逻辑
#         new_vel = joint_vel + u / (
#             1 - torch.exp(torch.tensor(-Δt / T_system)) * (1 + (Δt / T_system))
#         )
#         new_joint_pos = joint_pos + new_vel * Δt
#
#         # 计算末端在世界坐标下的位置(仅作一些可视化或记录)
#         ret = chain.forward_kinematics(new_joint_pos, end_only=True)
#         eef_matrix = ret.get_matrix()
#         eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
#         eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
#
#         return torch.cat((eef_pos, eef_rot), dim=1)
#
#     return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
#
#
#
#
# # import os
# #
# # import pytorch_kinematics
# # import torch
# # import math
# #
# # ##############################################################################
# # # 全局碰撞标志 (原样保留)
# # ##############################################################################
# # collision = torch.zeros([500, ], dtype=torch.bool)
# #
# # def getCollisions():
# #     return collision
# #
# #
# # ##############################################################################
# # # ！！！这是新增函数：将欧拉角 (rx, ry, rz) 转成 3x3 旋转矩阵
# # #   若与 MuJoCo 中 <geom euler="..."> 的定义完全匹配，需要根据实际情况
# # #   调整旋转顺序。这里示例按 Z-Y-X 顺序( extrinsic X->Y->Z ) 给出。
# # ##############################################################################
# # def euler_to_matrix(euler_xyz, device, dtype):
# #     """
# #     以 Z-Y-X 的顺序将 euler_xyz = (rx, ry, rz) 转成 3x3 矩阵:
# #       R = Rz(rz) * Ry(ry) * Rx(rx)
# #     若需要和 MuJoCo 中 euler="..." 更严格匹配，请结合 XML 文件中注释进行调整。
# #     """
# #     rx, ry, rz = euler_xyz
# #     sx, cx = math.sin(rx), math.cos(rx)
# #     sy, cy = math.sin(ry), math.cos(ry)
# #     sz, cz = math.sin(rz), math.cos(rz)
# #
# #     Rz = torch.tensor([[cz, -sz,  0],
# #                        [sz,  cz,  0],
# #                        [ 0,   0,  1]], dtype=dtype, device=device)
# #     Ry = torch.tensor([[ cy, 0, sy],
# #                        [  0, 1,  0],
# #                        [-sy, 0, cy]], dtype=dtype, device=device)
# #     Rx = torch.tensor([[1,  0,   0],
# #                        [0, cx, -sx],
# #                        [0, sx,  cx]], dtype=dtype, device=device)
# #
# #     # 最终 R = Rz * Ry * Rx
# #     R = torch.mm(Rz, torch.mm(Ry, Rx))
# #     return R
# #
# #
# # ##############################################################################
# # # ！！！这是新增函数：OBB（定向包围盒）碰撞检测
# # #   - pts_world: [N,3]，批量点的世界坐标
# # #   - center_world: [3,]，OBB中心在世界坐标下的位置
# # #   - R_world: [3,3]，OBB在世界下的旋转矩阵(把局部坐标系变换到世界)
# # #   - half_size: [3,]，OBB的 X/Y/Z 半尺寸
# # #   - margin: [3,]，安全裕度
# # ##############################################################################
# # def check_collision_obb(pts_world, center_world, R_world, half_size, margin, device, dtype):
# #     # OBB（Oriented Bounding Box，定向包围盒） 指的是一个任意朝向的盒子（矩形平行六面体）。
# #     # 与 AABB（Axis Aligned Bounding Box）不同，OBB 并不与世界坐标轴对齐；
# #     # 它可以用一个 3×3 旋转矩阵来描述其方向。
# #
# #     """
# #     判断每个点与给定 OBB 是否碰撞:
# #       1) 将点转换到障碍物局部系 local_pt = R^T * (world_pt - center)
# #       2) 若 local_pt 的绝对值在 (half_size + margin) 范围内，则视为碰撞
# #     返回 bool 张量 [N,]，表示每个点是否碰撞。
# #     """
# #     center = torch.tensor(center_world, dtype=dtype, device=device)
# #     R = torch.tensor(R_world, dtype=dtype, device=device)
# #     R_t = R.transpose(0, 1)  # R 的转置，用于世界 -> 局部
# #     local_pts = torch.matmul(pts_world - center, R_t)
# #     dist_local = torch.abs(local_pts)
# #
# #     lim = torch.tensor(half_size, dtype=dtype, device=device) \
# #         + torch.tensor(margin, dtype=dtype, device=device)
# #     collision_mask = torch.all(dist_local <= lim, dim=1)
# #     return collision_mask
# #
# #
# # def get_parameters(args):
# #     # 原始参数设定，保持不变
# #     if args.tune_mppi <= 0:
# #         args.α = 0
# #         args.λ = 60
# #         args.σ = 0.20
# #         args.χ = 0.0
# #         args.ω1 = 1.0003
# #         args.ω2 = 9.16e5
# #         args.ω3 = 9.16e3
# #         args.ω4 = 9.16e3
# #         args.ω_Φ = 5.41
# #         args.d_goal = 0.15
# #
# #     K = 500
# #     T = 10
# #     Δt = 0.01
# #     T_system = 0.011
# #
# #     dtype = torch.double
# #     device = 'cpu'  # 或 'cuda'
# #
# #     α = args.α
# #     λ = args.λ
# #     Σ = args.σ * torch.tensor([
# #         [1.5,  args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, 1.0,  args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
# #     ], dtype=dtype, device=device)
# #
# #     # 这里是 franka_panda_arm.urdf 的读取，保持不变
# #     MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
# #     xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
# #     dtype_kinematics = torch.double
# #     chain = pytorch_kinematics.build_serial_chain_from_urdf(
# #         xml, end_link_name="panda_link8", root_link_name="panda_link0"
# #     )
# #     chain = chain.to(dtype=dtype_kinematics, device=device)
# #
# #     # 机械臂基座在 Mujoco 场景中的平移偏移
# #     robot_base_pos = torch.tensor([0.8, 0.75, 0.44], device=device, dtype=dtype_kinematics)
# #
# #     def dynamics(x, u):
# #         new_vel = x[:, 7:14] + u
# #         new_pos = x[:, 0:7] + new_vel * Δt
# #         return torch.cat((new_pos, new_vel), dim=1)
# #
# #
# #     ###########################################################################
# #     # ！！！这是修改的地方：使用 OBB 碰撞检测
# #     ###########################################################################
# #     def state_cost(x, goal, obstacles):
# #         global collision
# #         batch_size = x.shape[0]
# #
# #         # 1) 正向运动学，得到 panda_link8(末端) 在世界坐标的位置
# #         joint_values = x[:, 0:7]
# #         ret = chain.forward_kinematics(joint_values, end_only=False)
# #         link8_matrix = ret['panda_link8'].get_matrix()  # [batch_size, 4, 4]
# #         link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # [batch_size, 3]
# #
# #         # 2) 计算距离目标点的代价
# #         goal_dist = torch.norm(link8_pos - goal, dim=1)
# #         cost = 1000.0 * (goal_dist ** 2)
# #
# #         # 3) 计算工作空间约束(例如超出一定范围则加大cost)
# #         dist_robot_base = torch.norm(link8_pos - robot_base_pos, dim=1)
# #
# #         # obstacles 长度为 2个动态障碍物，每个是 [x, y, z, qx, qy, qz, qw, sx, sy, sz]
# #         #
# #         # 第一个 obstacle(小蓝方块):
# #         #   obstacles[0:3]  => pos
# #         #   obstacles[3:7]  => quat
# #         #   obstacles[7:10] => half-size
# #         obs0_pos = obstacles[0:3]    # (x, y, z)
# #         obs0_quat = obstacles[3:7]   # (qx, qy, qz, qw)
# #         obs0_size = obstacles[7:10]  # (sx, sy, sz)
# #
# #         # 转成 tensor
# #         obs0_quat_t = torch.tensor(obs0_quat, dtype=dtype, device=device).unsqueeze(0)
# #         R0 = pytorch_kinematics.quaternion_to_matrix(obs0_quat_t)[0]  # (3,3)
# #         # 给第一个障碍物一些安全裕度
# #         margin_box0 = [0.08, 0.08, 0.03]
# #
# #         # 计算末端与小蓝方块的碰撞
# #         col0 = check_collision_obb(
# #             pts_world=link8_pos,
# #             center_world=obs0_pos,
# #             R_world=R0,
# #             half_size=obs0_size,
# #             margin=margin_box0,
# #             device=device,
# #             dtype=dtype
# #         )
# #
# #         # 第二个 obstacle(后 V 形槽):
# #         #   obstacles[10:13] => pos
# #         #   obstacles[13:17] => quat
# #         #   obstacles[17:20] => half-size(可视为这个 V 槽 body 的最大外包盒，但我们要拆成v1和v2)
# #         v_body_pos = obstacles[10:13]
# #         v_body_quat = obstacles[13:17]
# #         # v_body_size = obstacles[17:20]  # 这里不直接用，因为要拆成两块板
# #
# #         # 先得到 V 槽 body 的旋转矩阵
# #         vq_t = torch.tensor(v_body_quat, dtype=dtype, device=device).unsqueeze(0)
# #         Rv_body = pytorch_kinematics.quaternion_to_matrix(vq_t)[0]  # shape (3,3)
# #
# #         # -----------------------------
# #         # v1 对应 左侧斜板
# #         #   <geom euler="0 0 2.35619" pos="-0.06 0 0" size="0.10 0.03 0.03" />
# #         # -----------------------------
# #         v1_local_pos = [-0.06, 0.0, 0.0]
# #         v1_euler = [0.0, 0.0, 2.35619]   # z轴旋转
# #         v1_size = [0.10, 0.03, 0.03]
# #
# #         Rv1_local = euler_to_matrix(v1_euler, device=device, dtype=dtype)
# #         v1_lpos_t = torch.tensor(v1_local_pos, dtype=dtype, device=device)
# #         v_body_pos_t = torch.tensor(v_body_pos, dtype=dtype, device=device)
# #
# #         # V 槽 body -> world
# #         world_pos_v1 = v_body_pos_t + torch.matmul(Rv_body, v1_lpos_t)
# #         R_world_v1 = torch.matmul(Rv_body, Rv1_local)
# #         margin_v1 = [0.065, 0.07, 0.03]  # 适当留点安全余量
# #         # margin_v1 = [0.08, 0.08, 0.03]  # 适当留点安全余量
# #
# #         col_v1 = check_collision_obb(
# #             pts_world=link8_pos,
# #             center_world=world_pos_v1,
# #             R_world=R_world_v1,
# #             half_size=v1_size,
# #             margin=margin_v1,
# #             device=device,
# #             dtype=dtype
# #         )
# #
# #         # -----------------------------
# #         # v2 对应 右侧斜板
# #         #   <geom euler="0 0 0.785398" pos="0.06 0 0" size="0.10 0.03 0.03" />
# #         # -----------------------------
# #         v2_local_pos = [0.06, 0.0, 0.0]
# #         v2_euler = [0.0, 0.0, 0.785398]
# #         v2_size = [0.10, 0.03, 0.03]
# #
# #         Rv2_local = euler_to_matrix(v2_euler, device=device, dtype=dtype)
# #         v2_lpos_t = torch.tensor(v2_local_pos, dtype=dtype, device=device)
# #         world_pos_v2 = v_body_pos_t + torch.matmul(Rv_body, v2_lpos_t)
# #         R_world_v2 = torch.matmul(Rv_body, Rv2_local)
# #         margin_v2 = [0.065, 0.07, 0.03]  # 适当留点安全余量
# #         # margin_v2 = [0.08, 0.08, 0.03]  # 适当留点安全余量
# #
# #         col_v2 = check_collision_obb(
# #             pts_world=link8_pos,
# #             center_world=world_pos_v2,
# #             R_world=R_world_v2,
# #             half_size=v2_size,
# #             margin=margin_v2,
# #             device=device,
# #             dtype=dtype
# #         )
# #
# #         # 手爪碰撞(末端位置 + 偏移 + 简易 box)
# #         hand_pos = link8_pos.clone()
# #         hand_pos[:, 2] += 0.11  # 原作者的简化做法
# #         hand_dim = [0.03, 0.08, 0.05]  # 原逻辑: 当作 margin 用
# #         # margin_box0 = [0.08, 0.08, 0.03]
# #         margin_box0_hand_with_obstacle = [
# #             0.03 + margin_box0[0],
# #             0.08 + margin_box0[1],
# #             0.05 + margin_box0[2]
# #         ]
# #
# #         # 对小蓝块 + 手爪
# #         col0_hand = check_collision_obb(
# #             pts_world=hand_pos,
# #             center_world=obs0_pos,
# #             R_world=R0,
# #             half_size=obs0_size,
# #             margin=margin_box0_hand_with_obstacle,
# #             device=device,
# #             dtype=dtype
# #         )
# #
# #         # margin_v1_hand_with_obstacle = [0.03, 0.08, 0.05] + margin_v1  #
# #         # hand_dim = [0.03, 0.08, 0.05]
# #         margin_v1_hand_with_obstacle = [
# #             0.03 + margin_v1[0],
# #             0.08 + margin_v1[1],
# #             0.05 + margin_v1[2]
# #         ]
# #         # 对 v1 + 手爪
# #         col_v1_hand = check_collision_obb(
# #             pts_world=hand_pos,
# #             center_world=world_pos_v1,
# #             R_world=R_world_v1,
# #             half_size=v1_size,
# #             margin=margin_v1_hand_with_obstacle,
# #             device=device,
# #             dtype=dtype
# #         )
# #
# #         # margin_v2_hand_with_obstacle = [0.03, 0.08, 0.05] + margin_v2  #
# #         # hand_dim = [0.03, 0.08, 0.05]
# #         margin_v2_hand_with_obstacle = [
# #             0.03 + margin_v2[0],
# #             0.08 + margin_v2[1],
# #             0.05 + margin_v2[2]
# #         ]
# #         # 对 v2 + 手爪
# #         col_v2_hand = check_collision_obb(
# #             pts_world=hand_pos,
# #             center_world=world_pos_v2,
# #             R_world=R_world_v2,
# #             half_size=v2_size,
# #             margin=margin_v2_hand_with_obstacle,
# #             device=device,
# #             dtype=dtype
# #         )
# #
# #         # 组合所有碰撞结果
# #         collision_mask = col0 | col_v1 | col_v2 | col0_hand | col_v1_hand | col_v2_hand
# #
# #         # 桌面碰撞 (末端 z 低于某个值)
# #         table_collision = (link8_pos[:, 2] <= 0.40)
# #         # 工作空间限制
# #         workspace_costs = (dist_robot_base >= 0.8)
# #
# #         # 累加进 cost
# #         cost += args.ω2 * collision_mask
# #         cost += args.ω3 * table_collision
# #         cost += args.ω4 * workspace_costs
# #
# #         # 更新全局 collision (本例中按 step 清空也可)
# #         collision = collision_mask
# #         collision = torch.zeros([batch_size], dtype=torch.bool, device=device)
# #
# #         return cost
# #
# #
# #     # 原始的终端代价函数，保持大体不变
# #     def terminal_cost(x, goal):
# #         global collision
# #         joint_values = x[:, 0:7]
# #         ret = chain.forward_kinematics(joint_values, end_only=True)
# #
# #         eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
# #         cost = 10 * torch.norm((eef_pos - goal), dim=1) ** 2
# #         # cost += args.ω_Φ * torch.norm(x[:, 3:6], dim=1) ** 2
# #         collision = torch.zeros([500, ], dtype=torch.bool)
# #         return cost
# #
# #     def convert_to_target(x, u):
# #         joint_pos = x[0:7]
# #         joint_vel = x[7:14]
# #         # 原逻辑
# #         new_vel = joint_vel + u / (1 - torch.exp(torch.tensor(-Δt / T_system)) \
# #                        * (1 + (Δt / T_system)))
# #         new_joint_pos = joint_pos + new_vel * Δt
# #
# #         # 计算末端在世界坐标下的位置(仅作一些可视化或记录)
# #         ret = chain.forward_kinematics(new_joint_pos, end_only=True)
# #         eef_matrix = ret.get_matrix()
# #         eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
# #         eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
# #
# #         return torch.cat((eef_pos, eef_rot), dim=1)
# #
# #     return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
#
#
#
# # import os
# # import pytorch_kinematics
# # import torch
# #
# # collision = torch.zeros([500, ], dtype=torch.bool)
# #
# # def getCollisions():
# #     return collision
# #
# # def get_parameters(args):
# #     if args.tune_mppi <= 0:
# #         args.α = 0  # 5.94e-1
# #         args.λ = 60  # 40  # 1.62e1
# #         args.σ = 0.20  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
# #         args.χ = 0.0  # 2.00e-2
# #         args.ω1 = 1.0003
# #         args.ω2 = 9.16e3
# #         args.ω3 = 9.16e3
# #         args.ω4 = 9.16e3
# #         args.ω_Φ = 5.41
# #         args.d_goal = 0.15
# #
# #     K = 500
# #     T = 10
# #     Δt = 0.01
# #     T_system = 0.011
# #
# #     dtype = torch.double
# #     device = 'cpu'  # or 'cuda' if available
# #
# #     α = args.α
# #     λ = args.λ
# #     Σ = args.σ * torch.tensor([
# #         [1.5, args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, 1.0, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
# #     ], dtype=dtype, device=device)
# #
# #     MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
# #     xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
# #     chain = pytorch_kinematics.build_serial_chain_from_urdf(
# #         xml, end_link_name="panda_link8", root_link_name="panda_link0"
# #     ).to(dtype=dtype, device=device)
# #
# #     # 机器人基座的世界坐标平移
# #     robot_base_pos = torch.tensor([0.8, 0.75, 0.44], device=device, dtype=dtype)
# #
# #     # ---------------------
# #     # 一些辅助函数
# #     # ---------------------
# #     def quaternion_to_matrix(q):
# #         """
# #         q: (N, 4) [w, x, y, z]
# #         返回: (N, 3, 3) 的旋转矩阵
# #         """
# #         q = q / torch.norm(q, dim=1, keepdim=True)  # 保证单位四元数
# #         w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
# #
# #         one = torch.ones_like(w)
# #         two = 2.0 * one
# #         R = torch.stack([
# #             one - two*(y*y + z*z), two*(x*y - w*z),       two*(x*z + w*y),
# #             two*(x*y + w*z),       one - two*(x*x + z*z), two*(y*z - w*x),
# #             two*(x*z - w*y),       two*(y*z + w*x),       one - two*(x*x + y*y)
# #         ], dim=1).reshape(-1, 3, 3)
# #         return R
# #
# #     def euler_to_quaternion(euler_xyz):
# #         """
# #         euler_xyz: (3,) -> (roll, pitch, yaw)
# #         返回: 四元数 (w, x, y, z)
# #         """
# #         rx, ry, rz = euler_xyz[0], euler_xyz[1], euler_xyz[2]
# #         cx, sx = torch.cos(rx/2), torch.sin(rx/2)
# #         cy, sy = torch.cos(ry/2), torch.sin(ry/2)
# #         cz, sz = torch.cos(rz/2), torch.sin(rz/2)
# #         # 旋转顺序: Rz * Ry * Rx
# #         qw = cz*cy*cx + sz*sy*sx
# #         qx = cz*cy*sx - sz*sy*cx
# #         qy = cz*sy*cx + sz*cy*sx
# #         qz = sz*cy*cx - cz*sy*sx
# #         return torch.tensor([qw, qx, qy, qz], dtype=dtype, device=device)
# #
# #     def quaternion_mul(q1, q2):
# #         """
# #         q1, q2: (4,) [w,x,y,z]
# #         返回 q1 * q2: (4,)
# #         """
# #         w1, x1, y1, z1 = q1
# #         w2, x2, y2, z2 = q2
# #         w = w1*w2 - x1*x2 - y1*y2 - z1*z2
# #         x = w1*x2 + x1*w2 + y1*z2 - z1*y2
# #         y = w1*y2 + y1*w2 + z1*x2 - x1*z2
# #         z = w1*z2 + z1*w2 + x1*y2 - y1*x2
# #         return torch.tensor([w, x, y, z], dtype=dtype, device=device)
# #
# #     def obb_check(eef_pos_batch, box_center, box_quat, half_size, margin):
# #         """
# #         在世界坐标系下做 OBB 碰撞检测:
# #             p_local = R^T * (eef_pos - box_center)
# #             if |p_local| < half_size + margin => inside
# #         """
# #         N = eef_pos_batch.shape[0]
# #         box_quat_batch = box_quat.unsqueeze(0).repeat(N, 1)  # (N,4)
# #         R = quaternion_to_matrix(box_quat_batch)  # (N,3,3)
# #
# #         diff = eef_pos_batch - box_center.unsqueeze(0)  # (N,3)
# #         diff = diff.unsqueeze(-1)  # (N,3,1)
# #         p_local = torch.bmm(R.transpose(-1,-2), diff).squeeze(-1)  # (N,3)
# #
# #         threshold = half_size + margin
# #         inside = torch.all(torch.le(torch.abs(p_local), threshold.unsqueeze(0)), dim=1)
# #         return inside
# #
# #     # ---------------------
# #     # 动力学/代价函数
# #     # ---------------------
# #     def dynamics(x, u):
# #         new_vel = x[:, 7:14] + u
# #         new_pos = x[:, 0:7] + new_vel * Δt
# #         return torch.cat((new_pos, new_vel), dim=1)
# #
# #     def state_cost(x, goal, obstacles):
# #         """
# #         obstacles: 20维
# #           [0:3]   -> block_center
# #           [3:7]   -> block_quat (w, x, y, z)
# #           [7:10]  -> block_half_size
# #           [10:13] -> v_center
# #           [13:17] -> v_quat (w, x, y, z)
# #           [17:20] -> v_half_size
# #         """
# #         global collision
# #
# #         joint_values = x[:, 0:7]
# #         ret = chain.forward_kinematics(joint_values, end_only=False)
# #         link8_matrix = ret['panda_link8'].get_matrix()  # (N,4,4)
# #         link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # (N,3) in world frame
# #
# #         dist_robot_base = torch.norm(link8_pos - robot_base_pos, dim=1)
# #         goal_dist = torch.norm(link8_pos - goal, dim=1)
# #         cost = 1000 * goal_dist**2
# #
# #         # ------------------------------------------------
# #         # ！！！修改：小蓝方块回退到 "老模式" (AABB in world)
# #         # ------------------------------------------------
# #         block_center    = torch.tensor(obstacles[0:3], dtype=dtype, device=device)  # (3,)
# #         # 这里的四元数 block_quat 不再使用
# #         # block_quat      = torch.tensor(obstacles[3:7], dtype=dtype, device=device)
# #         block_half_size = torch.tensor(obstacles[7:10], dtype=dtype, device=device) # (3,)
# #         # block_margin    = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, device=device)
# #         block_margin = torch.tensor([0.02, 0.02, 0.02], dtype=dtype, device=device)
# #
# #         # EEF vs 小蓝方块
# #         dist_block = torch.abs(link8_pos - block_center)
# #         block_threshold = block_half_size + block_margin
# #         in_block = torch.all(dist_block <= block_threshold, dim=1)
# #
# #         # ------------------------------------------------
# #         # V形槽 (obstacle2) - 保留 OBB
# #         # ------------------------------------------------
# #         v_center    = torch.tensor(obstacles[10:13], dtype=dtype, device=device)
# #         v_quat      = torch.tensor(obstacles[13:17], dtype=dtype, device=device)
# #         v_half_size = torch.tensor(obstacles[17:20], dtype=dtype, device=device)
# #
# #         # 这里依然按照 OBB 的思路拆分为两块
# #         # local 1: offset=-0.06; euler_left=135deg
# #         offset_left = torch.tensor([-0.06, 0.0, 0.0], dtype=dtype, device=device)
# #         euler_left  = torch.tensor([0.0, 0.0, 2.35619], dtype=dtype, device=device)
# #         q_local_left = euler_to_quaternion(euler_left)
# #         q_left = quaternion_mul(v_quat, q_local_left)
# #         R_body = quaternion_to_matrix(v_quat.unsqueeze(0))[0]
# #         left_center_global = v_center + R_body @ offset_left
# #         left_half_size = torch.tensor([0.10, 0.03, 0.03], dtype=dtype, device=device)
# #
# #         # 给斜板留一些 margin
# #         v_margin_single = torch.tensor([0.03, 0.03, 0.0], dtype=dtype, device=device)
# #
# #         in_v_left = obb_check(link8_pos,
# #                               left_center_global,
# #                               q_left,
# #                               left_half_size,
# #                               v_margin_single)
# #
# #         # local 2: offset=+0.06; euler_right=45deg
# #         offset_right = torch.tensor([0.06, 0.0, 0.0], dtype=dtype, device=device)
# #         euler_right  = torch.tensor([0.0, 0.0, 0.785398], dtype=dtype, device=device)
# #         q_local_right = euler_to_quaternion(euler_right)
# #         q_right = quaternion_mul(v_quat, q_local_right)
# #         right_center_global = v_center + R_body @ offset_right
# #         right_half_size = torch.tensor([0.10, 0.03, 0.03], dtype=dtype, device=device)
# #
# #         in_v_right = obb_check(link8_pos,
# #                                right_center_global,
# #                                q_right,
# #                                right_half_size,
# #                                v_margin_single)
# #
# #         in_v_slot = torch.logical_or(in_v_left, in_v_right)
# #
# #         # 合并(小方块 + V槽)
# #         in_obstacle = torch.logical_or(in_block, in_v_slot)
# #
# #         # ------------------------------------------------
# #         # 手爪碰撞 (简化: 末端 +0.11m)
# #         # ------------------------------------------------
# #         hand_pos = link8_pos.clone()
# #         hand_pos[:, 2] += 0.11
# #         # 此处手爪 dimension 也可酌情加大
# #         hand_dimension = torch.tensor([0.05, 0.1, 0.05], dtype=dtype, device=device)
# #
# #         # ！！！修改：小蓝方块回退到 "老模式" (AABB in world)
# #         dist_block_hand = torch.abs(hand_pos - block_center)
# #         block_threshold_hand = block_half_size + hand_dimension
# #         in_block_hand = torch.all(dist_block_hand <= block_threshold_hand, dim=1)
# #
# #         # V槽: 依然 OBB
# #         in_v_left_hand  = obb_check(hand_pos, left_center_global,  q_left,
# #                                     left_half_size,  hand_dimension)
# #         in_v_right_hand = obb_check(hand_pos, right_center_global, q_right,
# #                                     right_half_size, hand_dimension)
# #         in_v_slot_hand  = torch.logical_or(in_v_left_hand, in_v_right_hand)
# #
# #         # 合并(小方块 + V槽)
# #         in_obstacle = torch.logical_or(in_obstacle,
# #                                        torch.logical_or(in_block_hand, in_v_slot_hand))
# #
# #         collision = torch.logical_or(collision, in_obstacle)
# #
# #         # table check
# #         table_collision = torch.le(link8_pos[:, 2], 0.40)
# #         workspace_costs = torch.ge(dist_robot_base, 0.8)
# #
# #         cost += args.ω2 * collision
# #         cost += args.ω3 * table_collision
# #         cost += args.ω4 * workspace_costs
# #
# #         collision = torch.zeros([500, ], dtype=torch.bool)
# #         return cost
# #
# #     def terminal_cost(x, goal):
# #         global collision
# #         joint_values = x[:, 0:7]
# #         ret = chain.forward_kinematics(joint_values, end_only=True)
# #         eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
# #
# #         cost = 10 * torch.norm((eef_pos - goal), dim=1)**2
# #         collision = torch.zeros([500, ], dtype=torch.bool)
# #         return cost
# #
# #     def convert_to_target(x, u):
# #         joint_pos = x[0:7]
# #         joint_vel = x[7:14]
# #
# #         # 滞后 / 低通
# #         c = 1 - torch.exp(torch.tensor(-Δt / T_system, dtype=dtype)) * (1 + (Δt / T_system))
# #         new_vel = joint_vel + u / c
# #         new_joint_pos = joint_pos + new_vel * Δt
# #
# #         ret = chain.forward_kinematics(new_joint_pos, end_only=True)
# #         eef_matrix = ret.get_matrix()
# #         eef_pos = eef_matrix[:, :3, 3] + robot_base_pos
# #         eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
# #         return torch.cat((eef_pos, eef_rot), dim=1)
# #
# #     return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
#
#
# # import os
# #
# # import pytorch_kinematics
# # import torch
# #
# # collision = torch.zeros([500, ], dtype=torch.bool)
# #
# # def getCollisions():
# #     return collision
# #
# #
# # def get_parameters(args):
# #     if args.tune_mppi <= 0:
# #         args.α = 0  # 5.94e-1
# #         args.λ = 60  # 40  # 1.62e1
# #         args.σ = 0.20  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
# #         args.χ = 0.0  # 2.00e-2
# #         args.ω1 = 1.0003
# #         args.ω2 = 9.16e3
# #         args.ω3 = 9.16e3
# #         args.ω4 = 9.16e3
# #         args.ω_Φ = 5.41
# #         args.d_goal = 0.15
# #
# #     K = 500
# #     T = 10
# #     Δt = 0.01
# #     T_system = 0.011
# #
# #     dtype = torch.double
# #     device = 'cpu'  # or 'cuda' if available
# #
# #     α = args.α
# #     λ = args.λ
# #     Σ = args.σ * torch.tensor([
# #         [1.5, args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, 1.0, args.χ, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
# #         [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
# #     ], dtype=dtype, device=device)
# #
# #     # Ensure we get the path separator correct on windows
# #     MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')
# #
# #     xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
# #     dtype_kinematics = torch.double
# #     chain = pytorch_kinematics.build_serial_chain_from_urdf(
# #         xml, end_link_name="panda_link8", root_link_name="panda_link0"
# #     )
# #     chain = chain.to(dtype=dtype, device=device)
# #
# #     # Translational offset of Robot into World Coordinates
# #     robot_base_pos = torch.tensor([0.8, 0.75, 0.44], device=device, dtype=dtype)
# #
# #     # ！！！修改处：增加一些辅助函数，用于处理四元数与欧拉角
# #     def quaternion_to_matrix(q):
# #         """
# #         q: (N, 4) [w, x, y, z]
# #         返回: (N, 3, 3) 的旋转矩阵
# #         """
# #         # 保证 q 是单位四元数
# #         q = q / torch.norm(q, dim=1, keepdim=True)
# #         w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
# #
# #         # 按照公式构造旋转矩阵
# #         # 参考: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
# #         one = torch.ones_like(w)
# #         two = 2.0 * one
# #         R = torch.stack([
# #             one - two*(y*y + z*z), two*(x*y - w*z),       two*(x*z + w*y),
# #             two*(x*y + w*z),       one - two*(x*x + z*z), two*(y*z - w*x),
# #             two*(x*z - w*y),       two*(y*z + w*x),       one - two*(x*x + y*y)
# #         ], dim=1).reshape(-1, 3, 3)
# #         return R
# #
# #     def euler_to_quaternion(euler_xyz):
# #         """
# #         euler_xyz: (3,)的张量, 表示 (roll, pitch, yaw) = (rx, ry, rz)
# #         返回: 四元数 (w, x, y, z)
# #         注意: 这里假定旋转顺序是 extrinsic xyz, 或 intrinsic某一序列,
# #              视实际情况而定(仅演示,只需和Mujoco euler匹配).
# #         """
# #         # 这里只用到 z 轴, 也保留 x,y 以防扩展
# #         rx, ry, rz = euler_xyz[0], euler_xyz[1], euler_xyz[2]
# #         # 一半角
# #         cx, sx = torch.cos(rx/2), torch.sin(rx/2)
# #         cy, sy = torch.cos(ry/2), torch.sin(ry/2)
# #         cz, sz = torch.cos(rz/2), torch.sin(rz/2)
# #
# #         # 旋转顺序: Rz * Ry * Rx
# #         # w, x, y, z
# #         qw = cz*cy*cx + sz*sy*sx
# #         qx = cz*cy*sx - sz*sy*cx
# #         qy = cz*sy*cx + sz*cy*sx
# #         qz = sz*cy*cx - cz*sy*sx
# #         return torch.tensor([qw, qx, qy, qz], dtype=dtype, device=device)
# #
# #     def quaternion_mul(q1, q2):
# #         """
# #         q1, q2: shape=(4,)的张量, [w,x,y,z]
# #         返回 q1 * q2, shape=(4,)
# #         """
# #         w1, x1, y1, z1 = q1
# #         w2, x2, y2, z2 = q2
# #         w = w1*w2 - x1*x2 - y1*y2 - z1*z2
# #         x = w1*x2 + x1*w2 + y1*z2 - z1*y2
# #         y = w1*y2 + y1*w2 + z1*x2 - x1*z2
# #         z = w1*z2 + z1*w2 + x1*y2 - y1*x2
# #         return torch.tensor([w, x, y, z], dtype=dtype, device=device)
# #
# #     def obb_check(eef_pos_batch, box_center, box_quat, half_size, margin):
# #         """
# #         OBB碰撞检测:
# #         - eef_pos_batch: (N,3) -> world frame
# #         - box_center: (3,) -> world frame
# #         - box_quat: (4,)   -> world frame, [w,x,y,z]
# #         - half_size: (3,)
# #         - margin: (3,)
# #         返回: (N,) 的 bool 张量, 表示每条轨迹是否在此OBB内部
# #         """
# #         N = eef_pos_batch.shape[0]
# #         # 先把四元数变成 (N, 4)，以便向量化
# #         box_quat_batch = box_quat.unsqueeze(0).repeat(N, 1)  # (N,4)
# #         R = quaternion_to_matrix(box_quat_batch)  # (N,3,3)
# #
# #         # p_local = R^T * (eef_pos - box_center)
# #         # 其中 eef_pos - box_center 是( N,3 ), R^T是( N,3,3 )
# #         diff = eef_pos_batch - box_center.unsqueeze(0)  # (N,3)
# #         diff = diff.unsqueeze(-1)  # (N,3,1)
# #         # R^T = R.transpose(-1,-2) => shape (N,3,3)
# #         p_local = torch.bmm(R.transpose(-1,-2), diff).squeeze(-1)  # (N,3)
# #
# #         threshold = half_size + margin  # (3,)
# #
# #         inside = torch.all(torch.le(torch.abs(p_local), threshold.unsqueeze(0)), dim=1)
# #         return inside
# #
# #     def dynamics(x, u):
# #         new_vel = x[:, 7:14] + u
# #         new_pos = x[:, 0:7] + new_vel * Δt
# #         return torch.cat((new_pos, new_vel), dim=1)
# #
# #     def state_cost(x, goal, obstacles):
# #         """
# #         obstacles 为 20 维：
# #           [0:3]   -> block_center
# #           [3:7]   -> block_quat (w, x, y, z)
# #           [7:10]  -> block_half_size
# #           [10:13] -> v_center
# #           [13:17] -> v_quat (w, x, y, z)
# #           [17:20] -> v_half_size
# #         """
# #         global collision
# #
# #         joint_values = x[:, 0:7]
# #         ret = chain.forward_kinematics(joint_values, end_only=False)
# #
# #         link8_matrix = ret['panda_link8'].get_matrix()  # Nx4x4
# #         link8_pos = link8_matrix[:, :3, 3] + robot_base_pos  # (N,3) in world frame
# #
# #         dist_robot_base = torch.norm(link8_pos - robot_base_pos, dim=1)
# #         goal_dist = torch.norm((link8_pos - goal), dim=1)
# #         cost = 1000 * goal_dist**2
# #
# #         # ！！！修改处：改成 OBB 碰撞检测
# #
# #         # -------- 小蓝方块 (obstacle1) --------
# #         block_center    = torch.tensor(obstacles[0:3], dtype=dtype, device=device)        # (3,)
# #         block_quat      = torch.tensor(obstacles[3:7], dtype=dtype, device=device)        # (4,)
# #         block_half_size = torch.tensor(obstacles[7:10], dtype=dtype, device=device)       # (3,)
# #         block_margin    = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, device=device)     # 可调
# #         # 检测 eef vs block
# #         in_block = obb_check(link8_pos, block_center, block_quat,
# #                              block_half_size, block_margin)
# #
# #         # -------- V形槽 (obstacle2) --------
# #         # 这是 body 的整体位置、旋转
# #         v_center    = torch.tensor(obstacles[10:13], dtype=dtype, device=device)  # (3,)
# #         v_quat      = torch.tensor(obstacles[13:17], dtype=dtype, device=device)  # (4,)
# #         v_half_size = torch.tensor(obstacles[17:20], dtype=dtype, device=device)  # (3,) ~ [0.12, 0.03, 0.03] 只是 body-level
# #         # 但实际我们有 2 块板 geom_v1 / geom_v2，各自有局部 offset + 局部旋转
# #         # local 1: offset = [-0.06, 0, 0], euler z=135°
# #         # local 2: offset = [0.06, 0, 0],  euler z=45°
# #         # nominal half_size_single = [0.10, 0.03, 0.03] (from the xml)
# #         # margin:
# #         #v_margin_single = torch.tensor([0.01, 0.01, 0.0], dtype=dtype, device=device)
# #         v_margin_single = torch.tensor([0.03, 0.03, 0.0], dtype=dtype, device=device)
# #
# #         # --- 左斜板 ---
# #         offset_left = torch.tensor([-0.06, 0.0, 0.0], dtype=dtype, device=device)
# #         euler_left  = torch.tensor([0.0, 0.0, 2.35619], dtype=dtype, device=device)  # 135 deg
# #         q_local_left = euler_to_quaternion(euler_left)  # (4,)
# #
# #         # 合并得到左板全局 quaternion
# #         q_left = quaternion_mul(v_quat, q_local_left)  # q_body * q_local
# #
# #         # 左板中心 = v_center + R_body * offset_left
# #         R_body = quaternion_to_matrix(v_quat.unsqueeze(0))[0]  # (3,3)
# #         left_center_global = v_center + R_body @ offset_left
# #
# #         left_half_size = torch.tensor([0.10, 0.03, 0.03], dtype=dtype, device=device)
# #         in_v_left = obb_check(link8_pos,
# #                               left_center_global,
# #                               q_left,
# #                               left_half_size,
# #                               v_margin_single)
# #
# #         # --- 右斜板 ---
# #         offset_right = torch.tensor([0.06, 0.0, 0.0], dtype=dtype, device=device)
# #         euler_right  = torch.tensor([0.0, 0.0, 0.785398], dtype=dtype, device=device)  # 45 deg
# #         q_local_right = euler_to_quaternion(euler_right)
# #         q_right = quaternion_mul(v_quat, q_local_right)
# #
# #         right_center_global = v_center + R_body @ offset_right
# #         right_half_size = torch.tensor([0.10, 0.03, 0.03], dtype=dtype, device=device)
# #
# #         in_v_right = obb_check(link8_pos,
# #                                right_center_global,
# #                                q_right,
# #                                right_half_size,
# #                                v_margin_single)
# #
# #         in_v_slot = torch.logical_or(in_v_left, in_v_right)
# #
# #         # 总碰撞
# #         in_obstacle = torch.logical_or(in_block, in_v_slot)
# #
# #         # ---------------------------
# #         # 手爪的碰撞 (简化: 把末端 +0.11m 当作手爪中心)
# #         # ---------------------------
# #         hand_pos = link8_pos.clone()
# #         hand_pos[:, 2] += 0.11
# #         #hand_dimension = torch.tensor([0.030, 0.08, 0.05], dtype=dtype, device=device)  # 近似手爪
# #         hand_dimension = torch.tensor([0.050, 0.1, 0.05], dtype=dtype, device=device)  # 近似手爪
# #
# #         # 对手爪 vs 小蓝方块
# #         in_block_hand = obb_check(hand_pos, block_center, block_quat,
# #                                   block_half_size, hand_dimension)
# #
# #         # 对手爪 vs 左板
# #         in_v_left_hand = obb_check(hand_pos, left_center_global, q_left,
# #                                    left_half_size, hand_dimension)
# #         # 对手爪 vs 右板
# #         in_v_right_hand = obb_check(hand_pos, right_center_global, q_right,
# #                                     right_half_size, hand_dimension)
# #
# #         in_v_slot_hand = torch.logical_or(in_v_left_hand, in_v_right_hand)
# #
# #         # 叠加手爪碰撞
# #         in_obstacle = torch.logical_or(in_obstacle,
# #                                        torch.logical_or(in_block_hand, in_v_slot_hand))
# #
# #         collision = torch.logical_or(collision, in_obstacle)
# #
# #         # 如果某条轨迹任何时刻都没碰，则 collision[i] = False, 若碰了则 True
# #         if torch.any(collision):
# #             # print("some collision among the 500 samples")
# #             pass
# #
# #         table_collision = torch.le(link8_pos[:, 2], 0.40)
# #         workspace_costs = torch.ge(dist_robot_base, 0.8)
# #
# #         cost += args.ω2 * collision
# #         cost += args.ω3 * table_collision
# #         cost += args.ω4 * workspace_costs
# #
# #         # 每次返回前，把 collision 清空（不累加到下一帧）
# #         collision = torch.zeros([500, ], dtype=torch.bool)
# #         return cost
# #
# #     def terminal_cost(x, goal):
# #         global collision
# #         joint_values = x[:, 0:7]
# #         ret = chain.forward_kinematics(joint_values, end_only=True)
# #
# #         eef_pos = ret.get_matrix()[:, :3, 3] + robot_base_pos
# #         cost = 10 * torch.norm((eef_pos - goal), dim=1) ** 2
# #         # cost += args.ω_Φ * torch.norm(x[:, 3:6], dim=1) ** 2
# #
# #         collision = torch.zeros([500, ], dtype=torch.bool)
# #         return cost
# #
# #     def convert_to_target(x, u):
# #         joint_pos = x[0:7]
# #         joint_vel = x[7:14]
# #         # 考虑关节速度的实际滞后 / 低通效应
# #         new_vel = joint_vel + u / (
# #             1 - torch.exp(torch.tensor(-Δt / T_system, dtype=dtype)) * (1 + (Δt / T_system))
# #         )
# #
# #         new_joint_pos = joint_pos + new_vel * Δt  # Calculate new Target Joint Positions
# #
# #         ret = chain.forward_kinematics(new_joint_pos, end_only=True)
# #         eef_matrix = ret.get_matrix()
# #         eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # World Coord
# #         eef_rot = pytorch_kinematics.matrix_to_quaternion(eef_matrix[:, :3, :3])
# #
# #         return torch.cat((eef_pos, eef_rot), dim=1)
# #
# #     return K, T, Δt, α, dynamics, state_cost, terminal_cost, Σ, λ, convert_to_target, dtype, device
