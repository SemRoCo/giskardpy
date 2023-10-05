from typing import Dict, Optional, List

from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Vector3Stamped

from giskardpy.goals.cartesian_goals import CartesianPose
from giskardpy.goals.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.my_types import goal_parameter
from giskardpy.python_interface.low_level_python_interface import LowLevelGiskardWrapper


class GiskardWrapper(LowLevelGiskardWrapper):
    # %% predefined goals
    def set_joint_goal(self,
                       goal_state: Dict[str, float],
                       group_name: Optional[str] = None,
                       weight: Optional[float] = None,
                       max_velocity: Optional[float] = None,
                       **kwargs: goal_parameter):
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints
        """
        monitor_name = 'joint goal reached'
        self.add_joint_pose_reached_monitor(name=monitor_name,
                                            goal_state=goal_state,
                                            crucial=True)
        self.add_motion_goal(goal_type='JointPositionList',
                             goal_state=goal_state,
                             group_name=group_name,
                             weight=weight,
                             max_velocity=max_velocity,
                             to_end=[monitor_name],
                             **kwargs)

    def set_cart_goal(self,
                      goal_pose: PoseStamped,
                      tip_link: str,
                      root_link: str,
                      tip_group: Optional[str] = None,
                      root_group: Optional[str] = None,
                      max_linear_velocity: Optional[float] = None,
                      max_angular_velocity: Optional[float] = None,
                      reference_linear_velocity: Optional[float] = None,
                      reference_angular_velocity: Optional[float] = None,
                      weight: Optional[float] = None,
                      **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        monitor_name = 'pose reached'
        self.add_cartesian_pose_reached_monitor(name=monitor_name,
                                                root_link=root_link,
                                                root_group=root_group,
                                                tip_link=tip_link,
                                                tip_group=tip_group,
                                                goal_pose=goal_pose)
        self.add_motion_goal(goal_type=CartesianPose.__name__,
                             to_end=[monitor_name],
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             root_link=root_link,
                             root_group=root_group,
                             tip_group=tip_group,
                             max_linear_velocity=max_linear_velocity,
                             max_angular_velocity=max_angular_velocity,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             **kwargs)

    def set_straight_cart_goal(self,
                               goal_pose: PoseStamped,
                               tip_link: str,
                               root_link: str,
                               tip_group: Optional[str] = None,
                               root_group: Optional[str] = None,
                               max_linear_velocity: Optional[float] = None,
                               max_angular_velocity: Optional[float] = None,
                               reference_linear_velocity: Optional[float] = None,
                               reference_angular_velocity: Optional[float] = None,
                               weight: Optional[float] = None,
                               **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between root and tip link to move tip link into the goal pose.
        The max velocities enforce a strict limit, but require a lot of additional constraints, thus making the
        system noticeably slower.
        The reference velocities don't enforce a strict limit, but also don't require any additional constraints.
        In contrast to set_cart_goal, this tries to move the tip_link in a straight line to the goal_point.
        :param root_link: name of the root link of the kin chain
        :param tip_link: name of the tip link of the kin chain
        :param goal_pose: the goal pose
        :param tip_group: a group name, where to search for tip_link, only required to avoid name conflicts
        :param root_group: a group name, where to search for root_link, only required to avoid name conflicts
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        self.add_motion_goal(goal_type='CartesianPoseStraight',
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             root_link=root_link,
                             root_group=root_group,
                             weight=weight,
                             max_linear_velocity=max_linear_velocity,
                             max_angular_velocity=max_angular_velocity,
                             **kwargs)

    def set_translation_goal(self,
                             goal_point: PointStamped,
                             tip_link: str,
                             root_link: str,
                             tip_group: Optional[str] = None,
                             root_group: Optional[str] = None,
                             max_velocity: Optional[float] = None,
                             reference_velocity: Optional[float] = 0.2,
                             weight: float = WEIGHT_ABOVE_CA,
                             **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point:
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param max_velocity: m/s
        :param reference_velocity: m/s
        :param weight:
        """
        self.add_motion_goal(goal_type='CartesianPosition',
                             goal_point=goal_point,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             max_velocity=max_velocity,
                             reference_velocity=reference_velocity,
                             weight=weight,
                             **kwargs)

    def set_seed_configuration(self, seed_configuration):
        self.add_motion_goal(goal_type='SetSeedConfiguration',
                             seed_configuration=seed_configuration)

    def set_straight_translation_goal(self,
                                      goal_pose: PoseStamped,
                                      tip_link: str,
                                      root_link: str,
                                      tip_group: Optional[str] = None,
                                      root_group: Optional[str] = None,
                                      reference_velocity: float = None,
                                      max_velocity: float = 0.2,
                                      weight: float = WEIGHT_ABOVE_CA,
                                      **kwargs: goal_parameter):
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        self.add_motion_goal(goal_type='CartesianPositionStraight',
                             goal_pose=goal_pose,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             max_velocity=max_velocity,
                             weight=weight,
                             **kwargs)

    def set_rotation_goal(self,
                          goal_orientation: QuaternionStamped,
                          tip_link: str,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          reference_velocity: Optional[float] = None,
                          max_velocity: Optional[float] = None,
                          weight=WEIGHT_ABOVE_CA,
                          **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_orientation.
        :param goal_orientation:
        :param tip_link: tip link of kinematic chain
        :param root_link: root link of kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: rad/s, approx limit
        :param max_velocity: rad/s, strict limit, but will slow the system down
        :param weight:
        """
        self.add_motion_goal(goal_type='CartesianOrientation',
                             goal_orientation=goal_orientation,
                             tip_link=tip_link,
                             root_link=root_link,
                             tip_group=tip_group,
                             root_group=root_group,
                             reference_velocity=reference_velocity,
                             max_velocity=max_velocity,
                             weight=weight,
                             **kwargs)

    def set_align_planes_goal(self,
                              goal_normal: Vector3Stamped,
                              tip_link: str,
                              tip_normal: Vector3Stamped,
                              root_link: str,
                              tip_group: str = None,
                              root_group: str = None,
                              max_angular_velocity: Optional[float] = None,
                              weight: Optional[float] = None,
                              **kwargs: goal_parameter):
        """
        This goal will use the kinematic chain between tip and root to align tip_normal with goal_normal.
        :param goal_normal:
        :param tip_link: tip link of the kinematic chain
        :param tip_normal:
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip_link is not unique, search in this group for matches.
        :param root_group: if root_link is not unique, search in this group for matches.
        :param max_angular_velocity: rad/s
        :param weight:
        """
        self.add_motion_goal(goal_type='AlignPlanes',
                             tip_link=tip_link,
                             tip_group=tip_group,
                             tip_normal=tip_normal,
                             root_link=root_link,
                             root_group=root_group,
                             goal_normal=goal_normal,
                             max_angular_velocity=max_angular_velocity,
                             weight=weight,
                             **kwargs)

    def set_prediction_horizon(self, prediction_horizon: int, **kwargs: goal_parameter):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        self.add_motion_goal(goal_type='SetPredictionHorizon',
                             prediction_horizon=prediction_horizon,
                             **kwargs)

    def set_max_traj_length(self, new_length: float, **kwargs: goal_parameter):
        """
        Overwrites Giskard trajectory length limit for planning.
        If the trajectory is longer than new_length, Giskard will prempt the goal.
        :param new_length: in seconds
        """
        self.add_motion_goal(goal_type='SetMaxTrajLength',
                             new_length=new_length,
                             **kwargs)

    def set_limit_cartesian_velocity_goal(self,
                                          tip_link: str,
                                          root_link: str,
                                          tip_group: Optional[str] = None,
                                          root_group: Optional[str] = None,
                                          max_linear_velocity: float = 0.1,
                                          max_angular_velocity: float = 0.5,
                                          weight: float = WEIGHT_ABOVE_CA,
                                          hard: bool = False,
                                          **kwargs: goal_parameter):
        """
        This goal will use put a strict limit on the Cartesian velocity. This will require a lot of constraints, thus
        slowing down the system noticeably.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param root_group: if the root_link is not unique, use this to say to which group the link belongs
        :param tip_group: if the tip_link is not unique, use this to say to which group the link belongs
        :param max_linear_velocity: m/s
        :param max_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        :param hard: Turn this into a hard constraint. This make create unsolvable optimization problems
        """
        self.add_motion_goal(goal_type='CartesianVelocityLimit',
                             root_link=root_link,
                             root_group=root_group,
                             tip_link=tip_link,
                             tip_group=tip_group,
                             weight=weight,
                             max_linear_velocity=max_linear_velocity,
                             max_angular_velocity=max_angular_velocity,
                             hard=hard,
                             **kwargs)

    def set_grasp_bar_goal(self,
                           bar_center: PointStamped,
                           bar_axis: Vector3Stamped,
                           bar_length: float,
                           tip_link: str,
                           tip_grasp_axis: Vector3Stamped,
                           root_link: str,
                           tip_group: Optional[str] = None,
                           root_group: Optional[str] = None,
                           reference_linear_velocity: Optional[float] = None,
                           reference_angular_velocity: Optional[float] = None,
                           weight: float = WEIGHT_ABOVE_CA,
                           **kwargs: goal_parameter):
        """
        Like a CartesianPose but with more freedom.
        tip_link is allowed to be at any point along bar_axis, that is without bar_center +/- bar_length.
        It will align tip_grasp_axis with bar_axis, but allows rotation around it.
        :param root_link: root link of the kinematic chain
        :param tip_link: tip link of the kinematic chain
        :param tip_grasp_axis: axis of tip_link that will be aligned with bar_axis
        :param bar_center: center of the bar to be grasped
        :param bar_axis: alignment of the bar to be grasped
        :param bar_length: length of the bar to be grasped
        :param root_group: if root_link is not unique, search in this group for matches
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight:
        """
        self.add_motion_goal(goal_type='GraspBar',
                             root_link=root_link,
                             tip_link=tip_link,
                             tip_grasp_axis=tip_grasp_axis,
                             bar_center=bar_center,
                             bar_axis=bar_axis,
                             bar_length=bar_length,
                             root_group=root_group,
                             tip_group=tip_group,
                             reference_linear_velocity=reference_linear_velocity,
                             reference_angular_velocity=reference_angular_velocity,
                             weight=weight,
                             **kwargs)

    def set_open_container_goal(self,
                                tip_link: str,
                                environment_link: str,
                                tip_group: Optional[str] = None,
                                environment_group: Optional[str] = None,
                                goal_joint_state: Optional[float] = None,
                                weight=WEIGHT_ABOVE_CA, ):
        """
        Open a container in an environment.
        Only works with the environment was added as urdf.
        Assumes that a handle has already been grasped.
        Can only handle containers with 1 dof, e.g. drawers or doors.
        :param tip_link: end effector that is grasping the handle
        :param environment_link: name of the handle that was grasped
        :param tip_group: if tip_link is not unique, search in this group for matches
        :param environment_group: if environment_link is not unique, search in this group for matches
        :param goal_joint_state: goal state for the container. default is maximum joint state.
        :param weight:
        """
        self.add_motion_goal(goal_type='Open',
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight)

    def set_close_container_goal(self,
                                 tip_link: str,
                                 environment_link: str,
                                 tip_group: Optional[str] = None,
                                 environment_group: Optional[str] = None,
                                 goal_joint_state: Optional[float] = None,
                                 weight=WEIGHT_ABOVE_CA, ):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.add_motion_goal(goal_type='Close',
                             tip_link=tip_link,
                             environment_link=environment_link,
                             tip_group=tip_group,
                             environment_group=environment_group,
                             goal_joint_state=goal_joint_state,
                             weight=weight)

    def set_pointing_goal(self,
                          goal_point: PointStamped,
                          tip_link: str,
                          pointing_axis: Vector3Stamped,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          max_velocity: float = 0.3,
                          weight: float = WEIGHT_BELOW_CA,
                          **kwargs: goal_parameter):
        """
        Will orient pointing_axis at goal_point.
        :param tip_link: tip link of the kinematic chain.
        :param goal_point: where to point pointing_axis at.
        :param root_link: root link of the kinematic chain.
        :param tip_group: if tip_link is not unique, search this group for matches.
        :param root_group: if root_link is not unique, search this group for matches.
        :param pointing_axis: the axis of tip_link that will be used for pointing
        :param max_velocity: rad/s
        :param weight:
        """
        self.add_motion_goal(goal_type='Pointing',
                             tip_link=tip_link,
                             tip_group=tip_group,
                             goal_point=goal_point,
                             root_link=root_link,
                             root_group=root_group,
                             pointing_axis=pointing_axis,
                             max_velocity=max_velocity,
                             weight=weight,
                             **kwargs)

    def set_avoid_joint_limits_goal(self,
                                    percentage: int = 15,
                                    joint_list: Optional[List[str]] = None,
                                    weight: Optional[float] = None):
        """
        This goal will push joints away from their position limits. For example if percentage is 15 and the joint
        limits are 0-100, it will push it into the 15-85 range.
        """
        self.add_motion_goal(goal_type='AvoidJointLimits',
                             percentage=percentage,
                             weight=weight,
                             joint_list=joint_list)
