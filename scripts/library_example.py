from giskardpy.data_types.data_types import PrefixName
from giskardpy.model.collision_avoidance_config import DisableCollisionAvoidanceConfig
from giskardpy.model.trajectory import Trajectory
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.user_interface import GiskardWrapper
import giskardpy.casadi_wrapper as cas


def execute_cart_goal(giskard: GiskardWrapper) -> Trajectory:
    init = 'init'
    g1 = 'g1'
    g2 = 'g2'
    init_goal1 = cas.TransMatrix(reference_frame=PrefixName('map'))
    init_goal1.x = -0.5

    base_goal1 = cas.TransMatrix(reference_frame=PrefixName('map'))
    base_goal1.x = 1.0

    base_goal2 = cas.TransMatrix(reference_frame=PrefixName('map'))
    base_goal2.x = -1.0

    giskard.monitors.add_set_seed_odometry(base_pose=init_goal1, name=init)
    giskard.motion_goals.add_cartesian_pose(goal_pose=base_goal1, name=g1,
                                            root_link='map',
                                            tip_link='base_footprint',
                                            start_condition=init,
                                            end_condition=g1)
    giskard.motion_goals.add_cartesian_pose(goal_pose=base_goal2, name=g2,
                                            root_link='map',
                                            tip_link='base_footprint',
                                            start_condition=g1)
    giskard.monitors.add_end_motion(start_condition=g2)
    return giskard.execute(sim_time=20)


def execute_joint_goal(giskard: GiskardWrapper) -> Trajectory:
    init = 'init'
    g1 = 'g1'
    g2 = 'g2'
    giskard.monitors.add_set_seed_configuration(seed_configuration={'joint_2': 2},
                                                    name=init)
    giskard.motion_goals.add_joint_position({'joint_2': -1}, name=g1,
                                                start_condition=init,
                                                end_condition=g1)
    giskard.motion_goals.add_joint_position({'joint_2': 1}, name=g2,
                                                start_condition=g1)
    giskard.monitors.add_end_motion(start_condition=g2)
    return giskard.execute()


if __name__ == '__main__':
    urdf = open('../test/urdfs/simple_7_dof_arm.urdf', 'r').read()
    giskard = GiskardWrapper(world_config=WorldWithOmniDriveRobot(urdf=urdf),
                             collision_avoidance_config=DisableCollisionAvoidanceConfig(),
                             qp_controller_config=QPControllerConfig())
    print(execute_cart_goal(giskard).to_dict())
    print(execute_joint_goal(giskard).to_dict())
