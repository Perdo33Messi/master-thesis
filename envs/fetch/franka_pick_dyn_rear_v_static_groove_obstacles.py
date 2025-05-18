import copy
import math
import os
# Ensure we get the path separator correct on windows
from typing import List

import gym
import numpy as np
from gym_robotics.envs import rotations, robot_env, utils

MODEL_XML_PATH = os.path.join(os.getcwd(), 'envs', 'assets', 'fetch', 'franka_pick_dyn_rear_v_static_groove_obstacles.xml')


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


#class FrankaFetchPickDynObstaclesEnv(robot_env.RobotEnv, gym.utils.EzPickle):
class FrankaFetchPickDynRearVStaticGrooveEnv(robot_env.RobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type='sparse', n_substeps=20):

        """Initializes a new Fetch environment.
        """
        initial_qpos = {
            'robot0_joint1': -2.24,
            'robot0_joint2': -0.038,
            'robot0_joint3': 2.55,
            'robot0_joint4': -2.68,
            'robot0_joint5': 0.0,
            'robot0_joint6': 0.984,
            'robot0_joint7': 0.0327,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        model_path = MODEL_XML_PATH
        self.further = False
        self.gripper_extra_height = [0, 0, 0.035]
        self.block_gripper = True
        self.has_object = True
        self.block_object_in_gripper = True
        self.block_z = True
        self.target_in_the_air = False
        self.target_offset = 0.0
        self.obj_range = 0.06  # originally 0.15
        self.target_range = 0.05
        self.target_range_x = 0.2  # entire table: 0.125
        self.target_range_y = 0.02  # entire table: 0.175
        self.distance_threshold = 0.05
        self.reward_type = reward_type
        self.limit_action = 0.05  # limit maximum change in position

        # the [1.3, 0.75, 0.6, 0.25, 0.35, 0.2] could be used as field definition
        self.field = [1.3, 0.75, 0.6, 0.25, 0.35, 0.2]

        # ===========================
        # 原先这里只有 'obstacle2:geom'
        # 现在拆成了两个geom: 'obstacle2:geom_v1' 和 'obstacle2:geom_v2'
        # ===========================
        self.dyn_obstacles_geom_names = [
            'obstacle:geom',         # 小蓝方块
            'obstacle2:geom_v1',     # V形槽的左斜板
            'obstacle2:geom_v2'      # V形槽的右斜板
        ]

        self.stat_obstacles_geom_names = []
        self.stat_obstacles = []

        # ---------------------------
        # dyn_obstacles 里存的是障碍物中心、姿态、以及 size（包络框的 x/y/z 半尺寸）
        # ---------------------------
        self.dyn_obstacles = [
            # 第一个是小蓝方块 (obstacle)，在 y=0.80 | front
            # 1.0,0.0,0.0,0.0: [w,   x,   y,   z] 顺序
            [1.3, 0.80, 0.435, 1.0, 0.0, 0.0, 0.0, 0.03, 0.03, 0.03],  # obstacle
            # 第二个是 V 形槽 (obstacle2)，在 y=0.60 | rear
            # 1.0,0.0,0.0,0.0: [w,   x,   y,   z] 顺序
            [1.3, 0.60, 0.435, 1.0, 0.0, 0.0, 0.0, 0.12, 0.03, 0.03] # obstacle2
        ]

        # 这里
        # dyn_obstacles
        # 前7个是位置和四元数(pos + quat)，后3个是size(包围盒大小)。
        # 其中“V形槽”用两个geom（obstacle2: geom_v1和obstacle2:geom_v2）组合而成，但在这里抽象为一个障碍物位置和朝向。

        self.obstacles = self.dyn_obstacles + self.stat_obstacles
        self.obstacles_geom_names = self.dyn_obstacles_geom_names + self.stat_obstacles_geom_names
        self.block_max_z = 0.53

        # super(FrankaFetchPickDynObstaclesEnv, self).__init__(
        #     model_path=model_path, n_substeps=n_substeps, n_actions=8,
        #     initial_qpos=initial_qpos)

        super(FrankaFetchPickDynRearVStaticGrooveEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=8,
            initial_qpos=initial_qpos)

        # 这里动作维度8：(dx, dy, dz) + (四元数rot 4) + (gripper_control)，总计 3+4+1=8，后面会在 _set_action 中看到如何分解这8个量。



        gym.utils.EzPickle.__init__(self)
        self._setup_dyn_obstacles()

    def _setup_dyn_obstacles(self):
        # setup velocity limits
        self.vel_lims = np.array([0.6, 0.9])  # ([0.3, 0.45])
        self.n_moving_obstacles = len(self.dyn_obstacles)
        self.n_obstacles = len(self.dyn_obstacles) + len(self.stat_obstacles)
        self.current_obstacle_vels = []

        self._setup_dyn_limits()

        # joint indices
        self.obstacle_slider_idxs = []
        # obstacle
        self.obstacle_slider_idxs.append(self.sim.model.joint_names.index('obstacle:joint'))
        # obstacle2
        self.obstacle_slider_idxs.append(self.sim.model.joint_names.index('obstacle2:joint'))

        self.geom_id_object = self.sim.model.geom_name2id('object0')

        self.geom_ids_obstacles = []
        for name in self.obstacles_geom_names:
            self.geom_ids_obstacles.append(self.sim.model.geom_name2id(name))

    def _setup_dyn_limits(self):
        self.obstacle_upper_limits = []
        self.obstacle_lower_limits = []
        self.pos_difs = []

        # assume all obstacles are moving horizontally along X axis
        for obst in self.obstacles:
            up = self.field[0] + self.field[3] - obst[7]
            lw = self.field[0] - self.field[3] + obst[7]
            self.obstacle_upper_limits.append(up)
            self.obstacle_lower_limits.append(lw)
            self.pos_difs.append((up - lw) / 2.)

    def _set_obstacle_slide_pos(self, positions):
        qpos = self.sim.data.qpos.flat[:]
        for i in range(self.n_moving_obstacles):
            # move obstacles along the slide joint
            pos = positions[i]
            qpos[self.obstacle_slider_idxs[i]] = pos
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qpos=qpos)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def _set_obstacle_slide_vel(self, velocities):
        qvel = self.sim.data.qvel.flat[:]
        for i, vel in enumerate(velocities):
            qvel[self.obstacle_slider_idxs[i]] = vel
        to_mod = copy.deepcopy(self.sim.get_state())
        to_mod = to_mod._replace(qvel=qvel)
        self.sim.set_state(to_mod)
        self.sim.forward()

    def _compute_obstacle_rel_x_positions(self, time) -> np.ndarray:
        n = self.n_moving_obstacles
        new_positions = np.zeros(n)
        t = time

        # triangle wave for moving obstacles
        for i in range(self.n_moving_obstacles):
            max_q = self.pos_difs[i]
            s_q = max_q * 4
            v = self.current_obstacle_vels[i]
            a = max_q  # amplitude
            p = s_q / v  # period
            s = self.current_obstacle_shifts[i] * 2 * math.pi  # time shift
            # triangle wave
            new_pos_x = 2 * a / math.pi * math.asin(math.sin(s + 2 * math.pi / p * t))
            new_positions[i] = new_pos_x

        return new_positions

    def get_obstacles(self, time) -> List[List[float]]:
        t = time
        n = self.n_moving_obstacles
        new_positions_x = self._compute_obstacle_rel_x_positions(time=t)
        updated_dyn_obstacles = []

        for i in range(self.n_moving_obstacles):
            obstacle = self.dyn_obstacles[i].copy()
            obstacle[0] = obstacle[0] + new_positions_x[i]
            updated_dyn_obstacles.append(obstacle)

            # 这里 self.dyn_obstacles[i] 通常是一个形如 [x, y, z, qw, qx, qy, qz, sx, sy, sz] 的列表，其中 obstacle[0] 是障碍物的 X 坐标。
            # 我们先 copy() 一份，以免直接修改原对象。
            # 再把其 x 坐标增加 new_positions_x[i]，表示基准位置 + 动态相对位移，从而得到障碍物在时刻 t 的绝对坐标。
            # 存入 updated_dyn_obstacles。

        return updated_dyn_obstacles + self.stat_obstacles

    def _move_obstacles(self, t):
        old_positions_x = self._compute_obstacle_rel_x_positions(time=t - self.dt)
        new_positions_x = self._compute_obstacle_rel_x_positions(time=t)
        vel_x = (new_positions_x - old_positions_x) / self.dt
        self._set_obstacle_slide_pos(new_positions_x)
        self._set_obstacle_slide_vel(vel_x)

    def step(self, action):
        t = self.sim.get_state().time + self.dt
        self._move_obstacles(t)

        #return super(FrankaFetchPickDynObstaclesEnv, self).step(action)

        return super(FrankaFetchPickDynRearVStaticGrooveEnv, self).step(action)

    # GoalEnv methods
    # ----------------------------
    def compute_reward(self, achieved_goal, goal, info):  # leave unchanged
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _set_action(self, action):
        assert action.shape == (8,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[7]

        if self.block_gripper:
            gripper_ctrl = -0.8

        # limit maximum change in position
        pos_ctrl *= self.limit_action
        # 固定末端旋转 (例子：四元数 [0, 1, 0, 0])
        rot_ctrl = [0, 1., 0., 0.]

        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_z:
            pos_ctrl[2] = 0.
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_body_xpos('eef')
        grip_rot = self.sim.data.get_body_xquat('eef')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('grip_site') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # object observations
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            object_rel_pos = object_pos - grip_pos
            # 【重要】object_rel_pos = object_pos - grip_pos：物体相对于手爪的位置差，用于在强化学习中让智能体更好地知道“物体在自己末端基坐标中的偏移”。
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        # 在一些 Goal-based 环境中，需要一个 achieved_goal 用于计算奖励等。
        # 如果没有物体，则直接用末端执行器位置当作“当前实现的目标”；
        # 如果有物体，则用物体的位置作为“当前实现的目标”。
        # 这常见于“抓取并移动目标物体到目标点”的环境：我们希望最终目标是“物体到达某个位置”，所以 achieved_goal = “物体当前位置”。



        # =========================================
        # 障碍物信息：获取小蓝块 (obstacle) 与 V形 (obstacle2) 的位姿
        # =========================================
        body_id = self.sim.model.body_name2id('obstacle')
        pos1 = np.array(self.sim.data.body_xpos[body_id].copy())
        rot1 = np.array(self.sim.data.body_xquat[body_id].copy())
        dims1 = self.dyn_obstacles[0][7:10]
        ob1 = np.concatenate((pos1, rot1, dims1.copy()))

        body_id = self.sim.model.body_name2id('obstacle2')
        pos2 = np.array(self.sim.data.body_xpos[body_id].copy())
        rot2 = np.array(self.sim.data.body_xquat[body_id].copy())
        dims2 = self.dyn_obstacles[1][7:10]
        ob2 = np.concatenate((pos2, rot2, dims2.copy()))

        dyn_obstacles = np.array([ob1, ob2])

        # 这里要获取场景中两个动态障碍物(obstacle 和 obstacle2)的当前位姿及尺寸：
        # 通过 body_name2id 找到对应 body 的索引 body_id；
        # 读取 self.sim.data.body_xpos[body_id] 得到障碍物中心的位置；
        # 读取 self.sim.data.body_xquat[body_id] 得到障碍物的姿态(四元数)；
        # dims1 = self.dyn_obstacles[0][7:10] 这从环境的“障碍物描述”里拿到它的长宽高（半尺寸）信息。
        # 将 [pos, rot, dims] 拼接为一个 10 维左右的数组表示“障碍物完整状态”。
        # 最终 dyn_obstacles 里面存放了 2 个障碍物的 [pos, quat, half_sizes]。

        obs = np.concatenate([
            grip_pos, grip_rot, robot_qpos, robot_qvel,
            object_pos.ravel(), object_rel_pos.ravel(), gripper_state,
            object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel
        ])

        # obs = np.concatenate([
        #     grip_pos,            # [3]    机械臂末端位置
        #     grip_rot,            # [4]    机械臂末端四元数
        #     robot_qpos,          # [...]  机械臂所有关节位置
        #     robot_qvel,          # [...]  机械臂所有关节速度
        #     object_pos.ravel(),  # [3]    物体位置
        #     object_rel_pos.ravel(), # [3]  物体相对末端的位置
        #     gripper_state,       # [2]    爪子张合
        #     object_rot.ravel(),  # [3]    物体的欧拉角
        #     object_velp.ravel(), # [3]    物体线速度 (相对末端)
        #     object_velr.ravel(), # [3]    物体角速度
        #     grip_velp,           # [3]    末端线速度 * dt
        #     gripper_vel          # [2]    爪子速度 * dt
        # ])

        # 这段把机械臂、物体、手爪的相关信息都拼成一个一维数组 obs，以便后续给到智能体进行决策。
        # 总的维度取决于机械臂关节数、是否有物体等，一般几十维不等。
        # obs 在 Gym 中常用作 'observation' 字段。

        obj_dist = np.linalg.norm(object_rel_pos.ravel())

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'real_obstacle_info': dyn_obstacles,
            'object_dis': obj_dist
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('panda0_gripper')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 130.
        self.viewer.cam.elevation = -24.
        self.viewer._run_speed = 0.8

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

        # place safe area around the rect (如果需要可在此做其他可视化)
        sites_offset_3 = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()[3]
        body_id = self.sim.model.body_name2id('obstacle2')
        pos2 = np.array(self.sim.data.body_xpos[body_id].copy())
        site_id = self.sim.model.site_name2id('safe2:site')
        self.sim.model.site_pos[site_id] = pos2 - sites_offset_3

        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            if not self.block_object_in_gripper:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        if self.block_object_in_gripper:
            # open the gripper to place an object, next applied action will close it
            self.sim.data.set_joint_qpos('robot0_finger_joint1', 0.025)
            self.sim.data.set_joint_qpos('robot0_finger_joint2', 0.025)

        # randomize obstacles
        n_obst = len(self.obstacles)
        n_dyn = self.n_moving_obstacles
        directions = self.np_random.choice([-1, 1], size=n_dyn)

        self.current_obstacle_shifts = self.np_random.uniform(-1.0, 1.0, size=n_obst)
        self.current_obstacle_vels = directions * self.np_random.uniform(self.vel_lims[0], self.vel_lims[1], size=n_dyn)

        # 为了让第二个障碍 V 形槽通常不怎么动，可手动把其速度降为极小值
        self.current_obstacle_vels[1] = 0.00001

        print("Directions")
        print(directions)
        print("Obstacle Shifts")
        print(self.current_obstacle_shifts)
        print("Obstacle Vels")
        print(self.current_obstacle_vels)
        self._move_obstacles(t=self.sim.get_state().time)

        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.target_center.copy()

        goal[1] += self.np_random.uniform(-self.target_range_y, self.target_range_y)
        goal[0] += self.np_random.uniform(-self.target_range_x, self.target_range_x)

        print("Goal:")
        print(goal)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # initial markers
        self.target_center = self.sim.data.get_site_xpos('target_center')
        self.init_center = self.sim.data.get_site_xpos('init_center')
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()[6]

        # Move end effector into position.
        gripper_target = self.init_center + self.gripper_extra_height
        gripper_rotation = np.array([0, 1., 0., 0.])
        self.sim.data.set_mocap_pos('panda0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('panda0:mocap', gripper_rotation)

        pre_sub_steps = 200
        pre_steps = int(pre_sub_steps / self.sim.nsubsteps)

        for _ in range(pre_steps):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('grip_site').copy()
        object_xpos = self.initial_gripper_xpos
        object_xpos[2] = 0.4  # table height

        # 如果选择在仿真启动时直接把方块放在手爪里
        if self.block_object_in_gripper:
            # place object in the gripper
            object_xpos2 = self.initial_gripper_xpos[:2]
            object_qpos2 = self.sim.data.get_joint_qpos('object0:joint')
            object_qpos2[:2] = object_xpos2
            object_qpos2[2] += 0.015

            ### CHANGED: 给方块赋予和手爪相同的四元数，以免方块“歪”
            eef_quat = self.sim.data.get_body_xquat('eef')
            object_qpos2[3:7] = eef_quat  # 关键：使用手爪的当前四元数
            ### CHANGED END

            self.sim.data.set_joint_qpos('object0:joint', object_qpos2)

        # show init area
        site_id = self.sim.model.site_name2id('init_1')
        self.sim.model.site_pos[site_id] = object_xpos + [self.obj_range, self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_2')
        self.sim.model.site_pos[site_id] = object_xpos + [self.obj_range, -self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_3')
        self.sim.model.site_pos[site_id] = object_xpos + [-self.obj_range, self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_4')
        self.sim.model.site_pos[site_id] = object_xpos + [-self.obj_range, -self.obj_range, 0.0] - sites_offset

        # show target area range
        site_id = self.sim.model.site_name2id('mark1')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, self.target_range_y, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark2')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, self.target_range_y, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark3')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, -self.target_range_y, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark4')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, -self.target_range_y, 0.0] - sites_offset

        site_id = self.sim.model.site_name2id('mark5')
        self.sim.model.site_pos[site_id] = object_xpos - sites_offset

        self.sim.forward()

        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=1080, height=1080):
        #return super(FrankaFetchPickDynObstaclesEnv, self).render(mode, width, height)

        return super(FrankaFetchPickDynRearVStaticGrooveEnv, self).render(mode, width, height)




