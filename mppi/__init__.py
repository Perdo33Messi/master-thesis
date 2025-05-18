import mppi.pick_3d_target_obstacle
import mppi.pick_dyn_door_obstacles
import mppi.pick_dyn_lifted_obstacles
import mppi.pick_dyn_obstacles
import mppi.pick_dyn_sqr_obstacles
import mppi.pick_static_sqr_obstacles
# import mppi.pick_dyn_labyrinth

#【新场景 02 - 先静态的 V 形槽，后面是动态的方块】
import mppi.pick_dyn_front_v_static_groove_obstacles

#【新场景 03 - 先静态的 V 形槽，后面是动态的方块】
import  mppi.pick_dyn_front_u_static_groove_obstacles

# 【新场景 04 - 前面是动态的方块，后静态的 V 形槽】
import mppi.pick_dyn_rear_v_static_groove_obstacles


def get_mppi_parameters(args):
    if args.env == 'FetchPickStaticSqrObstacle':
        return pick_static_sqr_obstacles.get_parameters(args)
    elif args.env == 'FetchPickDynSqrObstacle-v1':
        return pick_dyn_sqr_obstacles.get_parameters(args)
    elif args.env == 'FetchPickDynObstaclesEnv-v1':
        # return pick_dyn_sqr_obstacles.get_parameters(args)
        return pick_dyn_obstacles.get_parameters(args)
    elif args.env == 'FetchPickDynObstaclesEnv-v2':
        return pick_dyn_obstacles.get_parameters(args)
    elif args.env == 'FetchPickDynLiftedObstaclesEnv-v1':
        return pick_dyn_lifted_obstacles.get_parameters(args)
    elif args.env == 'FetchPickDynObstaclesMaxEnv-v1':
        pass
    elif args.env == 'FrankaFetchPickDynSqrObstacle-v1':
        pass
    elif args.env == 'FetchPickDynDoorObstaclesEnv-v1':
        return pick_dyn_door_obstacles.get_parameters(args)
    elif args.env == 'FrankaFetchPickDynDoorObstaclesEnv-v1':
        return pick_dyn_door_obstacles.get_parameters(args)
    elif args.env == 'FrankaFetchPickDynLiftedObstaclesEnv-v1':
        return pick_dyn_lifted_obstacles.get_parameters(args)
    elif args.env == 'FrankaFetchPickDynObstaclesEnv-v1':
        return pick_dyn_obstacles.get_parameters(args)
    elif args.env == 'FrankaFetchPick3DTargetObstacle-v1':
        return pick_3d_target_obstacle.get_parameters(args)

    # elif args.env == 'FrankaFetchPickDynLabyrinthEnv-v1':
    #     return pick_dyn_labyrinth.get_parameters(args)

    # 【新场景 02 - 先静态的 V 形槽，后面是动态的方块】
    elif args.env == 'FrankaFetchPickDynFrontVStaticGrooveEnv-v1':
        return pick_dyn_front_v_static_groove_obstacles.get_parameters(args)

    # 【新场景 03 - 先静态的 V 形槽，后面是动态的方块】
    elif args.env == 'FrankaFetchPickDynFrontUStaticGrooveEnv-v1':
        return pick_dyn_front_u_static_groove_obstacles.get_parameters(args)

    # 【新场景 04 - 前面是动态的方块，后静态的 V 形槽】
    elif args.env == 'FrankaFetchPickDynRearVStaticGrooveEnv-v1':
        return pick_dyn_rear_v_static_groove_obstacles.get_parameters(args)

    else:
        # TODO throw some form of error
        pass
