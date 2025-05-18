import os

import numpy as np
import pytorch_kinematics
import torch
from scipy.spatial.transform import Rotation

# ========================
# 在这里定义全局数量
# ========================
N_SAMPLES = 3000 #500
#【GPU】全局 collision 放到 GPU
collision = torch.zeros([N_SAMPLES, ], dtype=torch.bool, device='cuda') #【GPU】
#collision = torch.zeros([500, ], dtype=torch.bool, device='cuda') #【GPU】

#====================================================
#【STORM】如果你还想保留 getCollisions() 以兼容老代码，就定义一个空返回
#         不再真正使用全局 collision
#====================================================
def getCollisions():
    return collision


#====================================================
#【STORM】无循环: 一次性rollout整条时域
#====================================================
def rollout_storm_no_loop_door(
    x_init,           # (K,14) => 初始状态(包含q和v)
    delta_v,          # (K,T,7) => 每时间步的“增量控制(噪声+基准)”
    u_base,           # (T,7)   => 基准u, 用于MPPI额外项
    dt,
    dynamics_params,  # 里头包含 chain, robot_base_pos, state_cost_door, terminal_cost_door, ...
    obstacle_positions, # (T,...) => 若每时刻障碍物不同，需要扩展到(K,T,...)形状
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
      states_all: (K,T,14) => 整条时域 (q,v)
      cost: (K,)          => 每条并行轨迹的总cost

    若外部施加单层退火, 在噪声 delta_v 时乘了 factor(t)，则
    Σ(t) = factor(t)^2 * Σ(0) => Σ_inv(t) = 1/factor(t)^2 * Σ_inv(0)，
    即再多乘 (1/factor(t)) 即可得到对的 MPPI 额外项。
    """
    chain             = dynamics_params["chain"]
    robot_base_pos    = dynamics_params["robot_base_pos"]
    state_cost_door   = dynamics_params["state_cost_door"]    #【STORM】新的单步cost
    terminal_cost_door= dynamics_params["terminal_cost_door"] #【STORM】新的终端cost
    args              = dynamics_params["args"]

    K = x_init.shape[0]
    T = delta_v.shape[1]

    q_init = x_init[:, 0:7]
    v_init = x_init[:, 7:14]

    #-------------------------------------------
    # 构造下三角矩阵 S_l (T,T) 并扩展到 (K,T,T)
    #-------------------------------------------
    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)  # (K,T,T)

    # vAll = v_init + sum_{k=0..t-1} delta_v(k)
    vAll = torch.bmm(S_l_batched, delta_v)  # =>(K,T,7)
    vAll = vAll + v_init.unsqueeze(1)

    # qAll = q_init + dt * sum_{k=0..t-1} v(t)
    qAll = torch.bmm(S_l_batched, vAll) * dt
    qAll = qAll + q_init.unsqueeze(1)

    states_all = torch.cat([qAll, vAll], dim=2)  # (K,T,14)

    #-------------------------------------------
    # 一次性做 Forward Kinematics
    #-------------------------------------------
    q_flat = qAll.reshape(K*T, 7)
    ret = chain.forward_kinematics(q_flat, end_only=True)
    eef_matrix = ret.get_matrix()  # (K*T,4,4)
    eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # (K*T,3)

    #-------------------------------------------
    # 计算 step cost
    #-------------------------------------------
    x_flat   = states_all.reshape(K*T, 14)
    eef_flat = eef_pos.reshape(K*T, 3)

    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1)  # =>(K,T,nObsDim)
    obs_flat    = obs_batched.reshape(K*T, -1)                       # =>(K*T,nObsDim)

    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)             # =>(K,T,d)
    goals_flat    = goals_batched.reshape(K*T, -1)                   # =>(K*T,d)

    step_cost_all = state_cost_door(
        x_flat, eef_flat, goals_flat, obs_flat,
        args=args, device=device, robot_base_pos=robot_base_pos
    )  # => (K*T,)

    step_cost_matrix = step_cost_all.view(K, T)
    cost = step_cost_matrix.sum(dim=1)  # =>(K,)

    #-------------------------------------------
    # 终端代价
    #-------------------------------------------
    x_terminal = states_all[:, -1, :]  # (K,14)
    eef_all3 = eef_pos.view(K, T, 3)
    eef_terminal = eef_all3[:, -1, :]  # (K,3)
    goals_all3 = goals_batched.view(K, T, -1)
    goal_terminal = goals_all3[:, -1, :]

    term_cost = terminal_cost_door(
        x_terminal, eef_terminal, goal_terminal,
        args=args, device=device
    )  # =>(K,)

    cost += term_cost

    #-------------------------------------------
    # MPPI额外项: delta_v dot (Σ_inv@u_base)
    #-------------------------------------------
    # 原先只用固定 Σ_inv(0). 现在若 per_step_factor_inv 不为空,
    # 则要对每个时间步 t 额外乘以 (1/factor(t)).
    if per_step_factor_inv is None:
        # 保持原逻辑
        Ub = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2) # =>(K,T)
        cost += mppi_term.sum(dim=1)
    else:
        #=============================================
        # (单层退火修改) 时间步级别的 Σ_inv(t) = 1/factor(t)^2 * Σ_inv(0).
        # => 只需乘以 1/factor(t) 即可
        #=============================================
        Ub0 = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        # Ub(t) = Ub0(t) * per_step_factor_inv[t]
        Ub = Ub0 * per_step_factor_inv.unsqueeze(1)  # =>(T,7)
        # delta_v 已包含 factor(t)*epsilon(t)
        # => 这里再与 Ub(t) 做点乘
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)

    return states_all, cost


#====================================================
#【STORM】向量化 单步代价: state_cost_vectorized_door
#====================================================
def state_cost_vectorized_door(
    x,        # (N,14)
    eef_pos,  # (N,3)
    goal,     # (N,3)
    obstacles,# (N,...) => 每条样本对应的门障碍物信息
    args, device,
    robot_base_pos
):
    """
    返回 (N,) 的cost.
    去掉 global collision, 改用本地 collision_mask=(N,).
    """
    N = x.shape[0]
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    dist_robot_base = torch.norm(eef_pos - robot_base_pos, dim=1)
    goal_dist = torch.norm(eef_pos - goal, dim=1)

    # 【STORM】示例: 你原先 cost = args.ω1 * goal_dist^2
    cost += args.ω1 * (goal_dist**2)
    # cost += 1000.0 * (goal_dist ** 2)

    #---------------------------------------
    # 碰撞检测: 局部 collision_mask
    #---------------------------------------
    collision_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # 1) 把门的quaternion(3:7)转成旋转矩阵
    #   => 注意需要对每条并行(N)都做?
    #   如果 obstacles[:,3:7] 不同于 “单个值” => 需向量化 Rotation
    #   这里给出一个简化示例(只处理同一个门?).
    #
    #【如果你真的对每个N都不一样，就要 vectorized 旋转(稍麻烦).
    # 也可在外部先处理. 这里仅做简化演示.
    #---------------------------------------
    # For demonstration, assume obstacles的 3:7 是同一个门的 quat
    # => 你可能只能取 obstacles[0,3:7] + batch broadcast
    # => or 干脆不做旋转?
    #
    # 这里先写个非常简化的, 跟你原先的 => 只做 single obstacle transform
    #
    quat_np = obstacles[0, 3:7].detach().cpu().numpy()  # shape=(4,)
    quat_np = np.roll(quat_np, -1) # 你原先roll
    r = Rotation.from_quat(quat_np)
    r_mat = r.inv().as_matrix()
    r_mat_torch = torch.tensor(r_mat, device=device, dtype=x.dtype)

    # x_translated => (N,3)
    x_translated = eef_pos - obstacles[0, 0:3]
    # => (3,N)
    x_transposed = x_translated.transpose(0,1)
    # => (3,N)
    x_rotated = torch.matmul(r_mat_torch, x_transposed)
    # =>(N,3)
    state_transformed_obs1 = x_rotated.transpose(0,1)

    dist1 = torch.abs(state_transformed_obs1)
    # box half-extent => obstacles[0,7:10], plus offset e.g. [0.055,0.055,0.03]
    #  => shape=(3,)
    box_extent = obstacles[0,7:10] + torch.tensor([0.055,0.055,0.03], device=device, dtype=x.dtype)
    in_box = torch.all(torch.le(dist1, box_extent), dim=1) # (N,)
    collision_mask = torch.logical_or(collision_mask, in_box)

    # 2) eef_pos 与 obstacles[0,10:13], [0,20:23] => 你原先写到 dist2EEF, dist3EEF
    dist2EEF = torch.abs(eef_pos - obstacles[0,10:13])
    dist3EEF = torch.abs(eef_pos - obstacles[0,20:23])
    in_box2 = torch.all(
        torch.le(
            dist2EEF,
            obstacles[0,17:20] + torch.tensor([0.055,0.055,0.03], device=device, dtype=x.dtype)
        ),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box2)

    in_box3 = torch.all(
        torch.le(
            dist3EEF,
            obstacles[0,27:30] + torch.tensor([0.055,0.055,0.03], device=device, dtype=x.dtype)
        ),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box3)

    # 3) 撞桌子 + 超出workspace
    table_collision = torch.le(eef_pos[:,2], 0.40)
    workspace_violation = torch.ge(dist_robot_base, 0.8)

    # 4) 累加到 cost
    cost += args.ω2 * collision_mask.float()
    cost += args.ω3 * table_collision.float()
    cost += args.ω4 * workspace_violation.float()

    return cost

#====================================================
#【STORM】向量化 终端代价: terminal_cost_vectorized_door
#====================================================
def terminal_cost_vectorized_door(
    x,
    eef_pos,
    goal,
    args, device
):
    """
    返回 (N,) 的 cost.
    如果你还要碰撞, 同理在这里再写 collision_mask,
    否则仅对距离做加权即可.
    """
    N = x.shape[0]
    cost = 10.0 * torch.norm(eef_pos - goal, dim=1)**2
    # cost += args.ω_Φ * torch.norm(x[:,3:6], dim=1)**2
    return cost


#====================================================
#【STORM】 get_parameters: 返回一切所需
#====================================================
def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 60  # 40  # 1.62e1
        args.σ = 0.201  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
        args.χ = 0.0  # 2.00e-2

        # args.ω1 = 1000  # 1.0003
        # args.ω2 = 100  # 9.16e3
        # args.ω3 = 9  # 9.16e3
        # args.ω4 = 9  # 9.16e3

        # args.ω1 = 2000 #1000 #1.0003
        # args.ω2 = 9.16e3
        # args.ω3 = 9.16e3
        # args.ω4 = 9.16e3

        # args.ω1 = 1000 #1.0003
        # args.ω2 = 10000 #9.16e3
        # args.ω3 = 9280 #9.16e3
        # args.ω4 = 9280 #9.16e3

        # args.ω1 = 1000 #1.0003
        # args.ω2 = 100 #9.16e3
        # args.ω3 = 9 #9.16e3
        # args.ω4 = 9 #9.16e3

        args.ω1 = 1000 #1.0003
        args.ω2 = 600 #9.16e3
        args.ω3 = 13 #9.16e3
        args.ω4 = 13 #9.16e3

        args.ω_Φ = 5.41
        args.d_goal = 0.15

    K = N_SAMPLES  # 直接复用全局变量 N_SAMPLES
    #K = 500
    T = 10
    Δt = 0.01
    T_system = 0.011 #0.011

    dtype = torch.double
    #device = 'cpu'  # 'cuda'

    #【GPU】把 device 从 'cpu' 改成 'cuda'
    device = 'cuda'  #【GPU】

    α = args.α
    λ = args.λ
    Σ = args.σ * torch.tensor([
        [1.5, args.χ, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, 0.75, args.χ, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, 1.0, args.χ, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, 1.25, args.χ, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, 1.50, args.χ, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, 2.00, args.χ],
        [args.χ, args.χ, args.χ, args.χ, args.χ, args.χ, 2.00]
    ], dtype=dtype, device=device)

    # Ensure we get the path separator correct on windows
    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')

    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    dtype_kinematics = torch.double
    chain = pytorch_kinematics.build_serial_chain_from_urdf(xml, end_link_name="panda_link8",
                                                            root_link_name="panda_link0")
    chain = chain.to(dtype=dtype_kinematics, device=device)

    # Translational offset of Robot into World Coordinates
    robot_base_pos = torch.tensor([0.8, 0.75, 0.44],
                                  device=device, dtype=dtype_kinematics)

    #-------------------------------------------
    #【STORM】单步动力学
    #-------------------------------------------
    def dynamics(x, u):

        new_vel = x[:, 7:14] + u
        new_pos = x[:, 0:7] + new_vel * Δt

        return torch.cat((new_pos, new_vel), dim=1)


    #-------------------------------------------
    # convert_to_target
    #-------------------------------------------
    def convert_to_target(x, u):
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
        #【STORM】注入我们定义的向量化cost
        "state_cost_door": state_cost_vectorized_door,
        "terminal_cost_door": terminal_cost_vectorized_door,
        "rollout_fn": rollout_storm_no_loop_door
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,  # 不再用旧 state_cost
        None,  # 不再用旧 terminal_cost
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params
    )
