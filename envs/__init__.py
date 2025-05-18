import gym

from .fetch.franka_pick_3d_target import FrankaFetchPick3DTarget
from .fetch.franka_pick_3d_target_obstacle import FrankaFetchPick3DTargetObstacle
from .fetch.franka_pick_dyn_door_obstacles import FrankaFetchPickDynDoorObstaclesEnv
from .fetch.franka_pick_dyn_labyrinth import FrankaFetchPickDynLabyrinthEnv
from .fetch.franka_pick_dyn_lifted_obstacles import FrankaFetchPickDynLiftedObstaclesEnv
from .fetch.franka_pick_dyn_obstacles import FrankaFetchPickDynObstaclesEnv
from .fetch.franka_pick_dyn_sqr_obstacle import FrankaFetchPickDynSqrObstacleEnv
from .fetch.pick_dyn_door_obstacles import FetchPickDynDoorObstaclesEnv
from .fetch.pick_dyn_labyrinth import FetchPickDynLabyrinthEnv
from .fetch.pick_dyn_lifted_obstacles import FetchPickDynLiftedObstaclesEnv
from .fetch.pick_dyn_obstacles import FetchPickDynObstaclesEnv
from .fetch.pick_dyn_obstacles2 import FetchPickDynObstaclesEnv2
from .fetch.pick_dyn_obstacles_max import FetchPickDynObstaclesMaxEnv
from .fetch.pick_dyn_sqr_obstacle import FetchPickDynSqrObstacleEnv

#【新场景 02 - 先静态的 V 形槽，后面是动态的方块】
from .fetch.franka_pick_dyn_front_v_static_groove_obstacles import FrankaFetchPickDynFrontVStaticGrooveEnv

# 【新场景 03 - 先静态的 U 形槽，后面是动态的方块】
from .fetch.franka_pick_dyn_front_u_static_groove_obstacles import FrankaFetchPickDynFrontUStaticGrooveEnv

#【新场景 04 - 前面是动态的方块，后静态的 V 形槽】
from .fetch.franka_pick_dyn_rear_v_static_groove_obstacles import FrankaFetchPickDynRearVStaticGrooveEnv




def register_custom_envs():
    gym.envs.register(
        id='FetchPickStaticSqrObstacle',
        entry_point='envs:FetchPickStaticSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynSqrObstacle-v1',
        entry_point='envs:FetchPickDynSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLabyrinthEnv-v1',
        entry_point='envs:FetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesEnv-v1',
        entry_point='envs:FetchPickDynObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesEnv-v2',
        entry_point='envs:FetchPickDynObstaclesEnv2',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLiftedObstaclesEnv-v1',
        entry_point='envs:FetchPickDynLiftedObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynObstaclesMaxEnv-v1',
        entry_point='envs:FetchPickDynObstaclesMaxEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynLabyrinth-v1',
        entry_point='envs:FetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FetchPickDynDoorObstaclesEnv-v1',
        entry_point='envs:FetchPickDynDoorObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynLabyrinth-v1',
        entry_point='envs:FrankaFetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynSqrObstacle-v1',
        entry_point='envs:FrankaFetchPickDynSqrObstacleEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynDoorObstaclesEnv-v1',
        entry_point='envs:FrankaFetchPickDynDoorObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynObstaclesEnv-v1',
        entry_point='envs:FrankaFetchPickDynObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPickDynLiftedObstaclesEnv-v1',
        entry_point='envs:FrankaFetchPickDynLiftedObstaclesEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPick3DTarget-v1',
        entry_point='envs:FrankaFetchPick3DTarget',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )
    gym.envs.register(
        id='FrankaFetchPick3DTargetObstacle-v1',
        entry_point='envs:FrankaFetchPick3DTargetObstacle',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )

    # 【新场景 01 - 迷宫】
    gym.envs.register(
        id='FrankaFetchPickDynLabyrinthEnv-v1',
        entry_point='envs:FrankaFetchPickDynLabyrinthEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )

    # 【新场景 02 - 先静态的 V 形槽，后面是动态的方块】
    gym.envs.register(
        id='FrankaFetchPickDynFrontVStaticGrooveEnv-v1',
        entry_point='envs:FrankaFetchPickDynFrontVStaticGrooveEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )


    # 【新场景 03 - 先静态的 U 形槽，后面是动态的方块】
    gym.envs.register(
        id='FrankaFetchPickDynFrontUStaticGrooveEnv-v1',
        entry_point='envs:FrankaFetchPickDynFrontUStaticGrooveEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )

    # 【新场景 04 - 前面是动态的方块，后静态的 V 形槽】
    gym.envs.register(
        id='FrankaFetchPickDynRearVStaticGrooveEnv-v1',
        entry_point='envs:FrankaFetchPickDynRearVStaticGrooveEnv',
        max_episode_steps=100,
        kwargs={'reward_type': 'sparse', 'n_substeps': 20},
    )


