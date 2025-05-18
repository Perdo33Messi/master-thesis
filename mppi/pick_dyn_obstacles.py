import os

import pytorch_kinematics
import torch

# ========================
# 在这里定义全局数量
# ========================
N_SAMPLES = 500

#【GPU】在初始化时就将 collision 放到 GPU 上
collision = torch.zeros([N_SAMPLES, ], dtype=torch.bool, device='cuda') #【GPU】
#collision = torch.zeros([500, ], dtype=torch.bool, device='cuda') #【GPU】


def getCollisions():
    return collision


# ====================================================
# 【STORM】 无循环rollout: 用下三角矩阵做一次性前缀和
# ====================================================
def rollout_storm_no_loop(
        x_init,  # (K,14) => 初始状态(包含q和v)
        delta_v,  # (K,T,7) => 每时间步的“增量控制(噪声+基准)”
        u_base,  # (T,7)   => 基准u, 用于MPPI额外项
        dt,
        dynamics_params,  # 包含: chain, robot_base_pos, state_cost_vectorized, terminal_cost_vectorized, ...
        obstacle_positions,  # (T, ...)，若每时刻障碍物不同，需要再扩展到(K,T,...)形状
        goals,  # (T, d) or (1,d) => 每个时间步的目标(可相同或不同)
        Σ_inv, λ, device,
        # ==============================================
        # (单层退火修改) 新增: per_step_factor_inv
        #   shape (T,)，表示 1/factor(t)
        # ==============================================
        per_step_factor_inv=None
):
    """
    返回:
      states_all: (K,T,14) => 全时域(q,v)
      cost: (K,) => 每条轨迹累积代价

    注意: 这里的 delta_v(k,t,:) 已经在外部被乘以 factor(t) 了,
      我们只需要在 MPPI cost 中，对 Σ_inv 做相应缩放。
    """
    # -------------------------------------------
    # 取出字典中的函数与参数
    # -------------------------------------------
    chain = dynamics_params["chain"]
    robot_base_pos = dynamics_params["robot_base_pos"]
    state_cost_vector = dynamics_params["state_cost"]
    terminal_cost_vector = dynamics_params["terminal_cost"]
    args = dynamics_params["args"]  # 可能需要在cost里用到 args.ω2 等

    # ===========================================
    # 1) 用下三角矩阵做前缀和：一次性计算所有时刻的q,v
    # ===========================================
    K = x_init.shape[0]
    T = delta_v.shape[1]

    # 拆分初始状态
    q_init = x_init[:, 0:7]  # (K,7)
    v_init = x_init[:, 7:14]  # (K,7)

    # 构造下三角矩阵 S_l (T,T)，然后扩展到(K,T,T)
    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)  # (K,T,T)

    # v(t) = v_init + sum_{k=0..t-1} delta_v(k)
    vAll = torch.bmm(S_l_batched, delta_v)  # (K,T,7)
    vAll = vAll + v_init.unsqueeze(1)  # 加上初始速度

    # q(t) = q_init + dt * sum_{k=0..t-1} v(t)
    qAll = torch.bmm(S_l_batched, vAll)  # (K,T,7)
    qAll = qAll * dt
    qAll = qAll + q_init.unsqueeze(1)

    # 拼成 (K,T,14)
    states_all = torch.cat([qAll, vAll], dim=2)  # (K,T,14)

    # ===========================================
    # 2) 一次性做 Forward Kinematics
    # ===========================================
    # 先把 (K,T,7) => flatten 成 (K*T,7)
    q_flat = qAll.reshape(K * T, 7)
    ret = chain.forward_kinematics(q_flat, end_only=True)
    eef_matrix = ret.get_matrix()  # (K*T,4,4)
    eef_pos = eef_matrix[:, :3, 3] + robot_base_pos  # (K*T,3)

    # ===========================================
    # 3) 一次性向量化计算 cost
    # ===========================================
    #   3.1) step cost for each time step
    # -------------------------------------------
    # flatten 后: x_flat=(K*T,14), eef_pos=(K*T,3)
    x_flat = states_all.view(K * T, 14)
    eef_flat = eef_pos.view(K * T, 3)

    # obstacle_positions => (T,...) => 先扩展到(K,T,...) => 再flatten => (K*T,...)
    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1)  # =>(K,T,nObsDim)
    obs_flat = obs_batched.reshape(K * T, -1)  # =>(K*T,nObsDim)

    # goals => (T,d) => 扩展到(K,T,d) => flatten => (K*T,d)
    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)
    goals_flat = goals_batched.reshape(K * T, -1)

    # 调用 vectorized single-step cost
    step_cost_all = state_cost_vector(
        x_flat, eef_flat, goals_flat, obs_flat,
        args=args, device=device, robot_base_pos=robot_base_pos
    )  # (K*T,)
    step_cost_matrix = step_cost_all.view(K, T)  # =>(K,T)
    cost = torch.sum(step_cost_matrix, dim=1)  # =>(K,)

    # -------------------------------------------
    #   3.2) terminal cost
    # -------------------------------------------
    # 取最后时刻 idx=T-1 => flatten里对应 row = (T-1) + each_k*T
    # 也可直接 gather
    x_terminal = states_all[:, -1, :]  # shape (K,14)
    # eef_pos 里也取最后时刻 => (K,3)
    #   Flatten index = offset + (T-1), offset=0..(K-1)*T
    #   可以简单 reshape back => eef_pos.view(K,T,3) 再 [:, -1, :]
    eef_all3 = eef_pos.view(K, T, 3)
    eef_terminal = eef_all3[:, -1, :]  # (K,3)

    # goals 里也取最后时刻 => (T,d) 中 index=(T-1)
    #   先 reshape =>(K,T,d) =>  [:, -1, :]
    goals_all3 = goals_batched.view(K, T, -1)
    goal_terminal = goals_all3[:, -1, :]  # (K,d)

    term_cost = terminal_cost_vector(
        x_terminal, eef_terminal, goal_terminal,
        args=args, device=device
    )  # =>(K,)

    cost += term_cost  # =>(K,)

    # -------------------------------------------
    #   3.3) MPPI额外项: \sum_{t=0..T-1} [ \delta_v(t)^\top Σ_inv(t) u_base(t) ]
    # -------------------------------------------
    #   若 Σ(t) = factor(t)^2 Σ(0)，则 Σ_inv(t) = (1 / factor(t)^2) Σ_inv(0)。
    #   我们外部将 \delta_v(t) = factor(t) * \epsilon(t)。所以最终需乘以 1/factor(t)。
    #
    #   实现方式：定义 Ub(t) = per_step_factor_inv[t] * (Σ_inv(0) @ u_base(t))，
    #   然后与 \delta_v(t) 点乘即可。
    # -------------------------------------------
    if per_step_factor_inv is None:
        # 若外面没传，就默认不变
        # (仍与原先相同：Ub = Σ_inv @ u_base)
        Ub = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)
    else:
        # (单层退火修改) 正确缩放
        Ub0 = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        # per_step_factor_inv: (T,)
        # 做广播乘 => (T,7)
        Ub = Ub0 * per_step_factor_inv.unsqueeze(1)

        # 现在 mppi_term = sum_over_dim=2 of [ delta_v(k,t,:) * Ub(t,:) ]
        # delta_v: shape (K,T,7)
        # Ub      shape (T,7) => unsqueeze(0) => (1,T,7)
        # => (K,T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)  # =>(K,)

    return states_all, cost


# ====================================================
# 【STORM】向量化 单步代价 (替代原 state_cost)
# ====================================================
def state_cost_vectorized(
    x,           # (N,14) => 包含 [q(7), v(7)]
    eef_pos,     # (N,3)  => 末端执行器位置(已经做完FK + base offset)
    goal,        # (N,3)  => 每个样本对应的目标
    obstacles,   # (N,...) => 每个样本对应的障碍物信息(比如 0:3, 7:10 等)
    args,
    device,
    robot_base_pos
):
    """
    返回: cost.shape = (N,)

    - 不再使用 global collision, 改为本地 collision_mask.
    - 每个样本 i 是否碰撞，保存在 collision_mask[i] 里。
    - 碰撞项乘以 args.ω2 累加到 cost 中。
    - 其他(比如与桌子碰撞、workspace范围)也直接加到 cost。
    """

    # N = x.shape[0], 例如 K*T
    N = x.shape[0]

    #----------------------------------------
    # 1) 初始化 cost (N,)
    #----------------------------------------
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    #----------------------------------------
    # 2) 距离/目标等常规代价
    #----------------------------------------
    dist_robot_base = torch.norm(eef_pos - robot_base_pos, dim=1)  # (N,)
    goal_dist       = torch.norm(eef_pos - goal, dim=1)            # (N,)

    cost += 1000.0 * (goal_dist ** 2)

    #----------------------------------------
    # 3) 局部碰撞掩码 collision_mask (N,)
    #----------------------------------------
    collision_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # === 示例：检测第一个障碍物 ===
    dist1 = torch.abs(eef_pos - obstacles[:, 0:3])       # (N,3)
    # 这里假设 obstacles[:, 7:10] 是第一个障碍物的半径(或一半边长)
    offset_box1 = torch.tensor([0.08, 0.08, 0.03], device=device)
    in_box1 = torch.all(  # (N,) bool
        torch.le(dist1, obstacles[:, 7:10] + offset_box1),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box1)

    # === 示例：检测第二个障碍物 ===
    dist2 = torch.abs(eef_pos - obstacles[:, 10:13])
    offset_box2 = torch.tensor([0.055, 0.055, 0.03], device=device)
    in_box2 = torch.all(
        torch.le(dist2, obstacles[:, 17:20] + offset_box2),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box2)

    # === 示例：手部检测 ===
    hand = eef_pos.clone()
    hand[:, 2] += 0.11
    hand_dimension = torch.tensor([0.030, 0.08, 0.05], device=device)

    dist3 = torch.abs(hand - obstacles[:, 0:3])
    in_box3 = torch.all(
        torch.le(dist3, obstacles[:, 7:10] + hand_dimension),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box3)

    dist4 = torch.abs(hand - obstacles[:, 10:13])
    in_box4 = torch.all(
        torch.le(dist4, obstacles[:, 17:20] + hand_dimension),
        dim=1
    )
    collision_mask = torch.logical_or(collision_mask, in_box4)

    #----------------------------------------
    # 4) 加到cost
    #----------------------------------------
    cost += args.ω2 * collision_mask.float()

    #----------------------------------------
    # 5) 与桌子碰撞、越界代价等
    #----------------------------------------
    table_collision    = torch.le(eef_pos[:,2], 0.40)  # (N,) bool
    workspace_violation = torch.ge(dist_robot_base, 0.8) # (N,) bool

    # 如果你想把这两项也加到 cost:
    # cost += args.ω3 * table_collision.float()
    cost += args.ω4 * workspace_violation.float()

    return cost


# ====================================================
# 【STORM】向量化 终端代价 (替代原 terminal_cost)
# ====================================================
def terminal_cost_vectorized(
        x,  # (N,14)
        eef_pos,  # (N,3)
        goal,  # (N,3)
        args, device
):
    # 不再使用
    global collision

    N = x.shape[0]

    # 1) 基础终端误差项
    cost = 10.0 * torch.norm(eef_pos - goal, dim=1) ** 2
    # 如果你还想加 cost += args.ω_Φ * torch.norm(x[:,3:6], dim=1)**2
    # cost += args.ω_Φ * torch.norm(x[:,3:6], dim=1)**2

    #collision = torch.zeros([500, ], dtype=torch.bool)
    # 2) 如果你还有某些姿态误差等，就在这里加:
    # cost += args.ω_Φ * torch.norm(x[:,3:6], dim=1)**2

    # 3) 若你还想做终端时的碰撞检测，也可和单步代价同理：构造 collision_mask=(N,)
    #    collision_mask = ...
    #    cost += args.ω2 * collision_mask.float()

    return cost



#====================================================
#【STORM】 get_parameters: 返回一切所需
#====================================================
#====================================================
# 接下来给一个“打包返回”函数以兼容你原先的结构
#====================================================
def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 60  # 40  # 1.62e1
        args.σ = 0.20  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
        args.χ = 0.0  # 2.00e-2
        args.ω1 = 1.0003
        args.ω2 = 9.16e3
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
    device = 'cuda'  # 【GPU】改成cuda，若你只想用CPU则改回'cpu'

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

    #-------------------------------------------
    # 构造 kinematics chain
    #-------------------------------------------
    # Ensure we get the path separator correct on windows
    MODEL_URDF_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_panda_arm.urdf')

    xml = bytes(bytearray(open(MODEL_URDF_PATH).read(), encoding='utf-8'))
    dtype_kinematics = torch.double
    chain = pytorch_kinematics.build_serial_chain_from_urdf(xml, end_link_name="panda_link8",
                                                            root_link_name="panda_link0")
    chain = chain.to(dtype=dtype_kinematics, device=device) # 放到GPU

    # Translational offset of Robot into World Coordinates
    robot_base_pos = torch.tensor([0.8, 0.75, 0.44],
                                  device=device, dtype=dtype_kinematics)

    #【修改】在此集中创建所有偏移常量，放在 GPU 上
    offset_box1 = torch.tensor([0.08, 0.08, 0.03], dtype=dtype, device=device)
    offset_box2 = torch.tensor([0.055, 0.055, 0.03], dtype=dtype, device=device)
    hand_dimension = torch.tensor([0.030, 0.08, 0.05], dtype=dtype, device=device)


    #-------------------------------------------
    # 仍然可提供传统的“单步”dynamics
    #-------------------------------------------
    def dynamics(x, u):
        new_vel = x[:, 7:14] + u
        new_pos = x[:, 0:7] + new_vel * Δt
        return torch.cat((new_pos, new_vel), dim=1)

    def convert_to_target(x, u):

        #把 x, u 转成下一时刻的关节位置 -> 计算末端位姿 (GPU 上).


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
    # 将向量化的cost函数和一些常量打包
    #-------------------------------------------
    dynamics_params = {
        "chain": chain,
        "robot_base_pos": robot_base_pos,
        "args": args,
        # 这里把“向量化单步”和“向量化终端”函数注入
        "state_cost": state_cost_vectorized,
        "terminal_cost": terminal_cost_vectorized,
        "rollout_fn": rollout_storm_no_loop
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,   # 这里如果你想兼容老API,可以把 state_cost=None
        None,   # 同理 terminal_cost=None
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params  #【STORM】关键: 把向量化所需的都装进来
    )
