import math
import os

import pytorch_kinematics
import torch
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

# ========================
# 在这里定义全局数量
# ========================
N_SAMPLES = 500
#【GPU】初始化 collision 放到 GPU
collision = torch.zeros([N_SAMPLES, ], dtype=torch.bool, device='cuda') # 【GPU】
#collision = torch.zeros([500, ], dtype=torch.bool, device='cuda') #【GPU】

#====================================================
#【STORM】如果你想兼容老代码中的 getCollisions()
#         可以在这里定义一个空函数或简单返回None
#====================================================
def getCollisions():
    return None
    #return collision


#====================================================
#【STORM】 无循环rollout: 用下三角矩阵做一次性前缀和
#====================================================
def rollout_storm_no_loop_3d(
    x_init,             # (K,14) => 初始状态(包含q+v)
    delta_v,            # (K,T,7) => 每时间步的“增量控制(噪声+基准)”
    u_base,             # (T,7)   => 基准u
    dt,
    dynamics_params,    # 包含: chain, robot_base_pos, state_cost_vectorized_3d, terminal_cost_vectorized_3d
    obstacle_positions, # (T,...) => 若每时刻障碍物不同，需要再扩展到(K,T,...)形状
    goals,              # (T,d) => 每时间步的目标
    Σ_inv, λ, device,
    # ====================================================
    # (单层退火修改) 新增参数: per_step_factor_inv
    #   若非 None，则 shape=(T,)，表示每步 1/factor(t)
    # ====================================================
    per_step_factor_inv=None
):
    """
    返回:
      states_all: (K,T,14) => 整条时域(q,v)
      cost:       (K,)     => 每条并行轨迹的总cost

    如果要做单层退火，外部会传一个 per_step_factor_inv[t] = 1.0 / factor(t)。
    """
    chain                   = dynamics_params["chain"]
    robot_base_pos          = dynamics_params["robot_base_pos"]
    link_verticies          = dynamics_params["link_verticies"]  # 供碰撞检测
    joint_collision_calc    = dynamics_params["joint_collision_calc"]
    state_cost_3d           = dynamics_params["state_cost_3d"]    #【STORM】新的向量化单步cost
    terminal_cost_3d        = dynamics_params["terminal_cost_3d"] #【STORM】新的向量化终端cost
    args                    = dynamics_params["args"]

    K = x_init.shape[0]
    T = delta_v.shape[1]

    q_init = x_init[:, 0:7]
    v_init = x_init[:, 7:14]

    # 构造下三角矩阵 S_l
    S_l = torch.tril(torch.ones(T, T, dtype=x_init.dtype, device=device))
    S_l_batched = S_l.unsqueeze(0).expand(K, -1, -1)  # (K,T,T)

    # v(t)
    vAll = torch.bmm(S_l_batched, delta_v)  # (K,T,7)  # bmm： 批量矩阵乘法（batch matrix multiplication）
    vAll = vAll + v_init.unsqueeze(1)

    # q(t)
    qAll = torch.bmm(S_l_batched, vAll) * dt
    qAll = qAll + q_init.unsqueeze(1)

    states_all = torch.cat([qAll, vAll], dim=2)  # (K,T,14)

    # 一次性 Forward Kinematics
    q_flat = qAll.reshape(K*T, 7)
    ret = chain.forward_kinematics(q_flat, end_only=False)
    # 由于要做 link-level 碰撞，需要 end_only=False，也就是所有关节都需要计算出来

    # 取 eef (panda_link8) 的位置
    link8_matrix = ret['panda_link8'].get_matrix()  # (K*T,4,4)
    eef_pos = link8_matrix[:, :3, 3] + robot_base_pos  # (K*T,3)

    # Flatten states
    x_flat   = states_all.view(K*T, 14) # x_flat 是一个形状为 (K*T, 14) 的张量
    eef_flat = eef_pos.view(K*T, 3)

    obs_batched = obstacle_positions.unsqueeze(0).expand(K, -1, -1)  # =>(K,T,...)
    obs_flat    = obs_batched.reshape(K*T, -1)                       # =>(K*T,...)
    # 障碍物信息扩展
    # obstacle_positions 的原形状大概是 (T, something)，表示在每个时间步（T）对应的障碍物状态（位置、旋转等）。
    # unsqueeze(0) 在第 0 维插入一个新维度，变成 (1, T, ...)。
    # .expand(K, -1, -1) 将第 0 维复制到 K，得到 (K,T,...)，这样就与 states_all 相对应：对每条轨迹都使用相同的障碍物信息（如果不同轨迹有不同障碍物，也可以另外组织数据）。
    # 再度展平
    # obs_batched.reshape(K*T, -1) 得到 (K*T, ...)，与 x_flat、eef_flat 在批量维度对齐，从而能在后续的单步代价函数里“一对一”地取到对应时刻的障碍物信息。

    goals_batched = goals.unsqueeze(0).expand(K, -1, -1)             # =>(K,T,d)
    goals_flat    = goals_batched.reshape(K*T, -1)                   # =>(K*T,d)
    # 目标信息的扩展
    # goals 可能是 (T,d)，表示 T 个时间步、每步一个 d 维目标（如末端位置 (x, y, z)）。
    # 同障碍物一样，插入维度得到 (1,T,d)，再 .expand(K, -1, -1) 变成 (K,T,d)，为每条轨迹都提供相同目标（如果每条轨迹的目标不同，需要自行修改数据结构）。
    # 展平
    # 变形为 (K*T,d)，便于和 x_flat 等做一一配对，用向量化的代价函数进行计算。

    #-------------------------------------------
    # 单步代价 (K*T,) => reshape =>(K,T) => sum(dim=1)
    #-------------------------------------------

    step_cost_all = state_cost_3d(
        x_flat, eef_flat, goals_flat, obs_flat,
        ret,                #【STORM】把整条FK记录(字典)传过去，以便做 link-level 碰撞
        link_verticies,     # 同理
        joint_collision_calc,
        args=args,
        device=device,
        robot_base_pos = robot_base_pos,  # [CHANGED] 把 robot_base_pos 也传进去
    )
    step_cost_matrix = step_cost_all.view(K, T)
    cost = step_cost_matrix.sum(dim=1)  # =>(K,)

    #-------------------------------------------
    # 终端代价
    #-------------------------------------------
    x_terminal = states_all[:, -1, :]
    eef_all3   = eef_pos.view(K, T, 3)
    eef_terminal = eef_all3[:, -1, :]
    goals_all3   = goals_batched.view(K, T, -1)
    goal_terminal= goals_all3[:, -1, :]

    term_cost = terminal_cost_3d(
        x_terminal, eef_terminal, goal_terminal,
        args=args, device=device, chain=chain, robot_base_pos=robot_base_pos
    )
    cost += term_cost

    #-------------------------------------------
    # MPPI 额外项: delta_v dot (Σ_inv @ u_base)
    #-------------------------------------------
    if per_step_factor_inv is None:
        #【不做退火】保持原先逻辑
        Ub = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)
    else:
        #===========================================
        # (单层退火修改) 对每个时间步 t:
        #   Σ_inv(t) = 1/factor(t)^2 * Σ_inv(0)
        # => 只需对 Ub(t) 额外乘以 1/factor(t)
        #===========================================
        Ub0 = torch.matmul(Σ_inv, u_base.T).T  # =>(T,7)
        Ub  = Ub0 * per_step_factor_inv.unsqueeze(1)  # =>(T,7)
        mppi_term = (delta_v * Ub.unsqueeze(0)).sum(dim=2)  # =>(K,T)
        cost += mppi_term.sum(dim=1)

    return states_all, cost

#====================================================
#【STORM】向量化 单步代价: state_cost_vectorized_3d
#====================================================
def state_cost_vectorized_3d(
    x,           # (N,14)  # x 就是 x_flat，它是一个形状为 (K*T, 14) 的张量
    eef_pos,     # (N,3)
    goal,        # (N,3)
    obstacles,   # (N,...) => 每个样本对应的障碍物
    fk_ret,      # forward_kinematics的结果(字典), shape(多link)
    link_verticies,
    joint_collision_calc,
    args, device,
    robot_base_pos    # [CHANGED] 新增一个参数
):
    """
    返回 shape=(N,) 的 cost
    - 不再使用 global collision
    - 每个样本 i:
        1) 末端与goal距离
        2) link-level 碰撞(循环 link_key)
        3) table/workspace
    """
    N = x.shape[0] # N = K*T
    cost = torch.zeros(N, dtype=x.dtype, device=device)

    # 1) goal距离代价
    #dist_robot_base = torch.norm(eef_pos - fk_ret['panda_link0'].get_matrix()[:, :3, 3], dim=1) \
                      #if 'panda_link0' in fk_ret else torch.norm(eef_pos, dim=1)

    if 'panda_link0' in fk_ret:
        link0_mat = fk_ret['panda_link0'].get_matrix()
        base_pos  = link0_mat[:, :3, 3]  # (N,3)
        dist_robot_base = torch.norm(eef_pos - base_pos, dim=1)
    else:
        dist_robot_base = torch.norm(eef_pos, dim=1)

    goal_dist = torch.norm(eef_pos - goal, dim=1)
    cost += 1000.0 * (goal_dist**2)

    # 2) link-level 碰撞
    collision_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # 这里 obstacles[i, ...] 是每个样本的障碍物 => 只展示“单个障碍物”
    # 如果你有多障碍物，需要再扩展
    #-------------------------------------------
    # 以 link_key in link_verticies 做遍历
    #-------------------------------------------
    for link_key, link_vertices_value in link_verticies.items():
        # link_transform => (N,) batch of transforms
        #    fk_ret[link_key] 是 pytorch_kinematics.TransformSet,
        #    其中 get_matrix() => (N,4,4)
        link_matrices = fk_ret[link_key].get_matrix()  # (N,4,4)
        # 交给 joint_collision_calc(...) 做每个样本的点云 transform + dist
        # 你可以用 small for loop in python,
        # or 试图 vectorize => 需要 transform_points batch 方式

        # 这里只演示:
        # 【我的思考】这里只是某一个 link[i] 去做碰撞检测
        link_collisions = joint_collision_calc(
            link_matrices,
            link_vertices_value,
            obstacles,
            device=device,
            robot_base_pos=robot_base_pos
        )
        # link_matrices: (N,4,4), N 为 K*T。
        # N 个变换矩阵 T
        # link_vertices_value，这里应该是只有一套 link[i]的散点
        # obstacles 也已经被扩展了。
        #     # 障碍物信息扩展
        #     # obstacle_positions 的原形状大概是 (T, something)，表示在每个时间步（T）对应的障碍物状态（位置、旋转等）。
        #     # unsqueeze(0) 在第 0 维插入一个新维度，变成 (1, T, ...)。
        #     # .expand(K, -1, -1) 将第 0 维复制到 K，得到 (K,T,...)，这样就与 states_all 相对应：对每条轨迹都使用相同的障碍物信息（如果不同轨迹有不同障碍物，也可以另外组织数据）。
        #     # 再度展平
        #     # obs_batched.reshape(K*T, -1) 得到 (K*T, ...)，与 x_flat、eef_flat 在批量维度对齐，从而能在后续的单步代价函数里“一对一”地取到对应时刻的障碍物信息。

        collision_mask = torch.logical_or(collision_mask, link_collisions)

    # 3) 其他(比如 撞桌子 / workspace)
    table_collision    = torch.le(eef_pos[:,2], 0.45)
    workspace_violation= torch.ge(dist_robot_base, 0.8)

    cost += args.ω2 * collision_mask.float()
    cost += args.ω3 * table_collision.float()
    cost += args.ω4 * workspace_violation.float()

    return cost

#====================================================
#【STORM】向量化 终端代价: terminal_cost_vectorized_3d
#====================================================
def terminal_cost_vectorized_3d(
    x,
    eef_pos,
    goal,
    args, device,
    chain,
    robot_base_pos
):
    """
    返回 shape=(N,) 的 cost
    若需要终端碰撞, 可类似 single-step.
    """
    N = x.shape[0]
    cost = 10.0 * torch.norm(eef_pos - goal, dim=1)**2
    return cost


#====================================================
#【STORM-向量化】完全向量化: batched_link_collision_calc
#====================================================
def batched_link_collision_calc(
    batch_matrices,  # (N,4,4)， N 为 K*T
    link_vertices_value,   # (M,3)
    obstacles,       # (N,10) => ， N 为 K*T， 10 中包含了 pos, rot, dim
    device,
    robot_base_pos         # [CHANGED] 新增一个参数
):
    """
    一次性处理 (N*M) 点而不再 python for i in range(N).

    1) 把 link_vertices 变成 (1,M,3), 再 expand => (N,M,3)
    2) 把 batch_matrices => (N,4,4), 做 (N,M,4,4)? 其实不需要，只需对 (N,M,4) * (N,4,4) => (N,M,4)
    3) obstacles => pos(0:3), rot(3:7), dim(7:10)
    """
    N = batch_matrices.shape[0]
    M = link_vertices_value.shape[0]

    # 1) 扩展 link_points => shape (N,M,3)
    link_pts = link_vertices_value.unsqueeze(0).expand(N, -1, -1)  # (M,3) → (1,M,3) → (N,M,3)

    # 2) 先做齐次坐标 => (N,M,4)
    ones = torch.ones(N, M, 1, dtype=torch.float64, device=device)
    link_pts_h = torch.cat([link_pts.to(torch.float64), ones], dim=2)  # (N,M,4)

    # 3) 世界坐标 => (N,M,4) x (N,4,4) => (N,M,4)
    #   需注意: PyTorch不自带 batch*(M,4) x (4,4) => 只能 swap
    #   Trick: (N,M,4) => (N,M,4,1)? or flatten
    #   见 https://discuss.pytorch.org/t/batched-matrix-multiplication-of-matrices-with-different-shapes/78154
    #   这里使用 bmm => 需要 shape (N,M,4) => (N,4,M)? => 不是很直接
    #   => 我们可以 permute to (N,4,M), do a bmm with (N,4,4)
    #   => or simpler: 先 permute (N,M,4)->(N,4,M), then for batch i multiply with (4,4).
    #   => 见下 simplification
    batch_matrices_3d = batch_matrices.to(torch.float64)  # (N,4,4)

    # reshape => (N,1,4,4) * (N,M,4,1)? => or we do a manual approach
    #【STORM-向量化】更简单:  manual approach with @ if we do .unsqueeze(2)
    # trick:
    link_pts_h_trans = link_pts_h.permute(0,2,1)  # =>(N,4,M)
    # expand batch_matrices => (N,4,4) x =>(N,4,4)
    # do => out = batch_matrices_3d @ link_pts_h_trans => shape(N,4,M)
    out_world = torch.bmm(batch_matrices_3d, link_pts_h_trans)  # =>(N,4,M)

    out_world = out_world.permute(0,2,1)  # =>(N,M,4)
    # => discard w
    out_world_xyz = out_world[..., :3]  # (N,M,3)

    # [CHANGED] 这里加上 robot_base_pos，使其真正位于世界系
    out_world_xyz = out_world_xyz + robot_base_pos  # (N,M,3)

    # 4) obstacles => pos(0:3), rot(3:7), dim(7:10)
    #   shape (N,10)
    #   => expand => pos =>(N,1,3), subtract =>(N,M,3)
    pos_i = obstacles[:, 0:3].unsqueeze(1)  # =>(N,1,3)
    link_translated = out_world_xyz - pos_i  # =>(N,M,3)

    # 5) rotation => obstacles[:,3:7] => shape(N,4)
    #   => convert to matrix =>(N,3,3)
    #   => invert =>(N,3,3)
    rot_i = obstacles[:, 3:7]  # (N,4)
    # quaternion to matrix
    # pytorch3d的 quaternion_to_matrix 不支持 batch。**我们自己分batch**:
    # Actually it DOES from v0.7.0 but let's show manual approach:

    # *Method1: a small loop => still python-level
    # *Method2: we do from_quat(rot_i) one by one => slow
    # *Method3: we can write a custom quaternion->mat batch code
    #【STORM-向量化】这里直接写个 batch code

    # quick batch code:
    # reorder to w,x,y,z
    rot_i = torch.roll(rot_i, shifts=-1, dims=1)  # =>(N,4) => [w,x,y,z]
    w, x, y, z = rot_i[:,0], rot_i[:,1], rot_i[:,2], rot_i[:,3]
    #
    # 3x3 => see https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # r00=1-2y^2-2z^2, r01=2xy-2zw, ...
    # We'll do them with broadcast
    rx = 1 - 2*(y**2 + z**2)
    ry = 2*(x*y - z*w)
    rz = 2*(x*z + y*w)
    r10= 2*(x*y + z*w)
    r11= 1 - 2*(x**2 + z**2)
    r12= 2*(y*z - x*w)
    r20= 2*(x*z - y*w)
    r21= 2*(y*z + x*w)
    r22= 1 - 2*(x**2 + y**2)

    R = torch.zeros(N,3,3, dtype=torch.float64, device=device)
    R[:,0,0] = rx;  R[:,0,1] = ry;  R[:,0,2] = rz
    R[:,1,0] = r10; R[:,1,1] = r11; R[:,1,2] = r12
    R[:,2,0] = r20; R[:,2,1] = r21; R[:,2,2] = r22

    # invert => R^T (since rotation is orthonormal)
    R_inv = R.permute(0,2,1)  # =>(N,3,3)

    # 6) do (N,M,3) x (N,3,3) =>(N,M,3)
    #   => we can do bmm if we reshape =>(N,M,3)->(N,3,M)
    link_translated_t = link_translated.permute(0,2,1)  # =>(N,3,M)
    link_in_obs_t = torch.bmm(R_inv, link_translated_t) # =>(N,3,M)
    link_in_obs = link_in_obs_t.permute(0,2,1) # =>(N,M,3)

    # 7) check box => obstacles[:,7:10] =>(N,3)
    dim_i = obstacles[:, 7:10].unsqueeze(1) # =>(N,1,3)

    """

    ########
    # 修改
    ########

    # clamp => (N,M,3)
    #   clamp(link_in_obs, -dim_i, dim_i)
    #   Python里可用 torch.clamp(input, min, max)
    #   但 min, max 若是张量 => 需要手动 broadcast, PyTorch 1.11+ 已支持
    #   这里做个简单写法:
    closest_point = torch.where(link_in_obs < -dim_i, -dim_i, link_in_obs)
    # torch.where(condition, x, y) 会逐元素检查 condition，如果为 True，则选 x；否则选 y。
    # 这里的 condition = (link_in_obs < -dim_i) 会产生一个布尔张量：每个元素对比 -dim_i

    closest_point = torch.where(closest_point >  dim_i,  dim_i, closest_point)
    # 这里再对 closest_point 进行一次“若大于 dim_i 则置为 dim_i，否则保留原值”的操作。
    # 这样就把坐标大于 +dim_i 的部分强行截断到 +dim_i。

    # 上述相当于 clamp，但支持 batch 维度(先比较 < -dim_i 再比较 > dim_i)

    # 在机器人/图形学的碰撞检测里，这一步常用来找到“点到 AABB（轴对齐的长方体）最近的那一点”。
    # 若点在盒子外，就会被投影到盒子的表面上；若点已在盒子内，就原封不动。

    # dist => (N,M)
    dist = torch.norm(link_in_obs - closest_point, dim=2)

    # CPU 版里用 0.12 作为碰撞阈值
    inside = torch.le(dist, 0.12)  # =>(N,M)
    collision_mask = torch.any(inside, dim=1) # =>(N,)

    """
    margin = 0.12
    box_extent = dim_i + margin  # =>(N,1,3)
    # => compare abs
    inside = torch.all(torch.le(torch.abs(link_in_obs), box_extent), dim=2)  # =>(N,M)
    collision_mask = torch.any(inside, dim=1) # =>(N,)
    

    return collision_mask


#====================================================
# 几个工具函数 (保留并略作改动去掉 global collision)
#====================================================

def calculate_link_verticies(links):
    link_verticies = {}
    for link_key in links:
        link = links[link_key]
        length = torch.norm(link)
        points_distance = 0.03
        points_count = math.ceil(length / points_distance)
        points = torch.zeros((points_count, 3), device=link.device, dtype=link.dtype)
        for i in range(points_count):
            factor = i / points_count
            points[i, :] = link * factor
        link_verticies[link_key] = points
    return link_verticies


#====================================================
#【STORM】 get_parameters: 返回一切所需
#====================================================
def get_parameters(args):
    if args.tune_mppi <= 0:
        args.α = 0  # 5.94e-1
        args.λ = 60  # 40  # 1.62e1
        args.σ = 0.201  # 0.01  # 08  # 0.25  # 4.0505  # 10.52e1
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
    T_system = 0.01

    dtype = torch.double
    #device = 'cpu'  # 'cuda'

    #【GPU】改成 'cuda'，若要在CPU保留原先则写'cpu'
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
    # link_dimensions + link_verticies
    #-------------------------------------------
    link_dimensions = {
        'panda_link0': torch.tensor([0.0, 0.0, 0.333], dtype=dtype_kinematics, device=device), #【GPU】指定 device
        # 'panda_link1': torch.tensor([0.0, 0.0, 0.000], dtype=dtype_kinematics),
        # Delete from Calculation for Computational Speed
        'panda_link2': torch.tensor([0.0, -0.316, 0.0], dtype=dtype_kinematics, device=device),
        'panda_link3': torch.tensor([0.0825, 0.0, 0.0], dtype=dtype_kinematics, device=device),
        'panda_link4': torch.tensor([-0.0825, 0.384, 0.0], dtype=dtype_kinematics, device=device),
        # 'panda_link5': torch.tensor([0.0, 0.0, 0.0], dtype=dtype_kinematics),
        # Delete from Calculation for Computational Speed
        'panda_link6': torch.tensor([0.088, 0.0, 0.0], dtype=dtype_kinematics, device=device),
        'panda_link7': torch.tensor([0.0, 0.0, 0.245], dtype=dtype_kinematics, device=device)
        # 'panda_link8': torch.tensor([0.0, 0.0, 0.0], dtype=dtype_kinematics)
        # Delete from Calculation for Computational Speed
    }

    link_verticies = calculate_link_verticies(link_dimensions)


    #-------------------------------------------
    #【STORM】单步动力学
    #-------------------------------------------
    def dynamics(x, u):

        new_vel = x[:, 7:14] + u
        new_pos = x[:, 0:7] + new_vel * Δt

        return torch.cat((new_pos, new_vel), dim=1)



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
        # return u

    #-------------------------------------------
    # 把向量化cost函数 & rollout函数等封装进 dict
    #-------------------------------------------
    dynamics_params = {
        "chain": chain,
        "robot_base_pos": robot_base_pos,
        "link_verticies": link_verticies,
        "joint_collision_calc": batched_link_collision_calc,
        "args": args,
        "state_cost_3d": state_cost_vectorized_3d,
        "terminal_cost_3d": terminal_cost_vectorized_3d,
        "rollout_fn": rollout_storm_no_loop_3d
    }

    return (
        K, T, Δt, α,
        dynamics,
        None,  # 不再用旧的 state_cost
        None,  # 不再用旧的 terminal_cost
        Σ, λ,
        convert_to_target,
        dtype, device,
        dynamics_params
    )

