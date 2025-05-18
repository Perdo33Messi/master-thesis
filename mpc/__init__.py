from .mpc_common import extract_parameters, make_obs, get_args
from .pick_3d_target_obstacle import generate_pathplanner
from .pick_dyn_door_obstacles import generate_pathplanner
from .pick_dyn_lifted_obstacles import generate_pathplanner
from .pick_dyn_obstacles import generate_pathplanner
from .pick_dyn_obstacles_max import generate_pathplanner
from .pick_dyn_sqr_obstacles import generate_pathplanner
from .plot import MPCDebugPlot


def make_mpc(args):
    a = {
        'FetchPickDynSqrObstacle-v1': pick_dyn_sqr_obstacles,
        'FetchPickDynObstaclesEnv-v1': pick_dyn_obstacles,
        'FetchPickDynObstaclesEnv-v2': pick_dyn_obstacles,
        'FetchPickDynLiftedObstaclesEnv-v1': pick_dyn_lifted_obstacles,
        # 'FetchPickDynObstaclesMaxEnv-v1': pick_dyn_obstacles_max,
        'FetchPickDynObstaclesMaxEnv-v1': pick_dyn_obstacles,

        'FrankaFetchPickDynSqrObstacle-v1': pick_dyn_sqr_obstacles,
        'FetchPickDynDoorObstaclesEnv-v1': pick_dyn_door_obstacles,
        'FrankaFetchPickDynObstaclesEnv-v1': pick_dyn_obstacles,
        'FrankaFetchPickDynLiftedObstaclesEnv-v1': pick_dyn_lifted_obstacles,
        'FrankaFetchPickDynDoorObstaclesEnv-v1': pick_dyn_door_obstacles,
        'FrankaFetchPick3DTargetObstacle-v1': pick_3d_target_obstacle,

        # 【新场景 02 - 先静态的 V 形槽，后面是动态的方块】
        'FrankaFetchPickDynFrontVStaticGrooveEnv-v1': pick_dyn_obstacles, ###### 【待修改】

        # 【新场景 03 - 先静态的 U 形槽，后面是动态的方块】
        'FrankaFetchPickDynFrontUStaticGrooveEnv-v1': pick_dyn_obstacles,  ###### 【待修改】

        # 【新场景 04 - 前面是动态的方块，后静态的 V 形槽】
        'FrankaFetchPickDynRearVStaticGrooveEnv-v1': pick_dyn_obstacles ###### 【待修改】

    }

    return a[args.env].generate_pathplanner(create=args.mpc_gen, path=args.mpc_path)
