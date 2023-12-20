from typing import Dict, Optional, List

from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped, Vector3Stamped

from giskard_msgs.msg import MoveResult, CollisionEntry
from giskardpy.goals.tasks.task import WEIGHT_ABOVE_CA, WEIGHT_BELOW_CA
from giskardpy.my_types import goal_parameter
from giskardpy.python_interface.low_level_python_interface import LowLevelGiskardWrapper


class GiskardWrapper(LowLevelGiskardWrapper):
    def execute(self, wait: bool = True) -> MoveResult:
        local_min_reached_monitor_name = self.monitors.add_local_minimum_reached()
        for goal in self.motion_goals._goals:
            goal.end_monitors.append(local_min_reached_monitor_name)
        return super().execute(wait)

    def projection(self, wait: bool = True) -> MoveResult:
        local_min_reached_monitor_name = self.monitors.add_local_minimum_reached()
        for goal in self.motion_goals._goals:
            goal.end_monitors.append(local_min_reached_monitor_name)
        return super().projection(wait)

    # %% predefined goals
    def set_joint_goal(self,
                       goal_state: Dict[str, float],
                       group_name: Optional[str] = None,
                       weight: Optional[float] = None,
                       max_velocity: Optional[float] = None,
                       add_monitor: bool = True,
                       **kwargs: goal_parameter):
        """
        Sets joint position goals for all pairs in goal_state
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints
        """
        if add_monitor:
            monitor_name = self.monitors.add_joint_position(goal_state=goal_state,
                                                            crucial=True)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_joint_position(goal_state=goal_state,
                                         group_name=group_name,
                                         weight=weight,
                                         max_velocity=max_velocity,
                                         end_monitors=end_monitors_monitors,
                                         **kwargs)

    def add_cart_goal(self,
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
                      add_monitor: bool = True,
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
        if add_monitor:
            monitor_name = self.monitors.add_cartesian_pose(root_link=root_link,
                                                            root_group=root_group,
                                                            tip_link=tip_link,
                                                            tip_group=tip_group,
                                                            goal_pose=goal_pose)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_cartesian_pose(goal_pose=goal_pose,
                                             tip_link=tip_link,
                                             root_link=root_link,
                                             root_group=root_group,
                                             tip_group=tip_group,
                                             max_linear_velocity=max_linear_velocity,
                                             max_angular_velocity=max_angular_velocity,
                                             reference_linear_velocity=reference_linear_velocity,
                                             reference_angular_velocity=reference_angular_velocity,
                                             weight=weight,
                                             end_monitors=end_monitors_monitors,
                                             **kwargs)

    def set_diff_drive_base_goal(self,
                                 goal_pose: PoseStamped,
                                 tip_link: str,
                                 root_link: str,
                                 tip_group: Optional[str] = None,
                                 root_group: Optional[str] = None,
                                 reference_linear_velocity: Optional[float] = None,
                                 reference_angular_velocity: Optional[float] = None,
                                 weight: Optional[float] = None,
                                 add_monitor: bool = True,
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
        :param reference_linear_velocity: m/s
        :param reference_angular_velocity: rad/s
        :param weight: default WEIGHT_ABOVE_CA
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} pose reached'
            self.monitors.add_cartesian_pose(name=monitor_name,
                                             root_link=root_link,
                                             root_group=root_group,
                                             tip_link=tip_link,
                                             tip_group=tip_group,
                                             position_threshold=0.02,
                                             goal_pose=goal_pose)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_diff_drive_base(end_monitors=end_monitors_monitors,
                                              goal_pose=goal_pose,
                                              tip_link=tip_link,
                                              root_link=root_link,
                                              root_group=root_group,
                                              tip_group=tip_group,
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
                               reference_linear_velocity: Optional[float] = None,
                               reference_angular_velocity: Optional[float] = None,
                               weight: Optional[float] = None,
                               add_monitor: bool = True,
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
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} pose reached'
            self.monitors.add_cartesian_pose(name=monitor_name,
                                             root_link=root_link,
                                             root_group=root_group,
                                             tip_link=tip_link,
                                             tip_group=tip_group,
                                             goal_pose=goal_pose)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_cartesian_pose_straight(end_monitors=end_monitors_monitors,
                                                      goal_pose=goal_pose,
                                                      tip_link=tip_link,
                                                      tip_group=tip_group,
                                                      root_link=root_link,
                                                      root_group=root_group,
                                                      weight=weight,
                                                      reference_linear_velocity=reference_linear_velocity,
                                                      reference_angular_velocity=reference_angular_velocity,
                                                      **kwargs)

    def set_translation_goal(self,
                             goal_point: PointStamped,
                             tip_link: str,
                             root_link: str,
                             tip_group: Optional[str] = None,
                             root_group: Optional[str] = None,
                             reference_velocity: Optional[float] = 0.2,
                             weight: float = WEIGHT_ABOVE_CA,
                             add_monitor: bool = True,
                             **kwargs: goal_parameter):
        """
        Will use kinematic chain between root_link and tip_link to move tip_link to goal_point.
        :param goal_point:
        :param tip_link: tip link of the kinematic chain
        :param root_link: root link of the kinematic chain
        :param tip_group: if tip link is not unique, you can use this to tell Giskard in which group to search.
        :param root_group: if root link is not unique, you can use this to tell Giskard in which group to search.
        :param reference_velocity: m/s
        :param weight:
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} position reached'
            self.monitors.add_cartesian_position(name=monitor_name,
                                                 root_link=root_link,
                                                 root_group=root_group,
                                                 tip_link=tip_link,
                                                 tip_group=tip_group,
                                                 goal_point=goal_point)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.set_translation_goal(end_monitors=end_monitors_monitors,
                                               goal_point=goal_point,
                                               tip_link=tip_link,
                                               root_link=root_link,
                                               tip_group=tip_group,
                                               root_group=root_group,
                                               reference_velocity=reference_velocity,
                                               weight=weight,
                                               **kwargs)

    def set_seed_configuration(self, seed_configuration, group_name: Optional[str] = None):
        self.motion_goals.set_seed_configuration(seed_configuration=seed_configuration,
                                                 group_name=group_name)

    def set_straight_translation_goal(self,
                                      goal_point: PointStamped,
                                      tip_link: str,
                                      root_link: str,
                                      tip_group: Optional[str] = None,
                                      root_group: Optional[str] = None,
                                      reference_velocity: float = None,
                                      weight: float = WEIGHT_ABOVE_CA,
                                      add_monitor: bool = True,
                                      **kwargs: goal_parameter):
        """
        Same as set_translation_goal, but will try to move in a straight line.
        """
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} position reached'
            self.monitors.add_cartesian_position(name=monitor_name,
                                                 root_link=root_link,
                                                 root_group=root_group,
                                                 tip_link=tip_link,
                                                 tip_group=tip_group,
                                                 goal_point=goal_point)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.set_straight_translation_goal(end_monitors=end_monitors_monitors,
                                                        goal_point=goal_point,
                                                        tip_link=tip_link,
                                                        root_link=root_link,
                                                        tip_group=tip_group,
                                                        root_group=root_group,
                                                        reference_velocity=reference_velocity,
                                                        weight=weight,
                                                        **kwargs)

    def set_rotation_goal(self,
                          goal_orientation: QuaternionStamped,
                          tip_link: str,
                          root_link: str,
                          tip_group: Optional[str] = None,
                          root_group: Optional[str] = None,
                          reference_velocity: Optional[float] = None,
                          weight=WEIGHT_ABOVE_CA,
                          add_monitor: bool = True,
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
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} orientation reached'
            self.monitors.add_cartesian_orientation(name=monitor_name,
                                                    root_link=root_link,
                                                    root_group=root_group,
                                                    tip_link=tip_link,
                                                    tip_group=tip_group,
                                                    goal_orientation=goal_orientation)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_cartesian_orientation(end_monitors=end_monitors_monitors,
                                                    goal_orientation=goal_orientation,
                                                    tip_link=tip_link,
                                                    root_link=root_link,
                                                    tip_group=tip_group,
                                                    root_group=root_group,
                                                    reference_velocity=reference_velocity,
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
                              add_monitor: bool = True,
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
        if add_monitor:
            monitor_name = f'{root_link}/{tip_link} vectors aligned {len(self.monitors._monitors)}'
            self.monitors.add_vectors_aligned(name=monitor_name,
                                              root_link=root_link,
                                              tip_link=tip_link,
                                              goal_normal=goal_normal,
                                              tip_normal=tip_normal,
                                              root_group=root_group,
                                              tip_group=tip_group)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_align_planes(end_monitors=end_monitors_monitors,
                                           tip_link=tip_link,
                                           tip_group=tip_group,
                                           tip_normal=tip_normal,
                                           root_link=root_link,
                                           root_group=root_group,
                                           goal_normal=goal_normal,
                                           reference_angular_velocity=max_angular_velocity,
                                           weight=weight,
                                           **kwargs)

    def set_prediction_horizon(self, prediction_horizon: int, **kwargs: goal_parameter):
        """
        Will overwrite the prediction horizon for a single goal.
        Setting it to 1 will turn of acceleration and jerk limits.
        :param prediction_horizon: size of the prediction horizon, a number that should be 1 or above 5.
        """
        self.motion_goals.set_prediction_horizon(prediction_horizon=prediction_horizon,
                                                 **kwargs)

    def set_max_traj_length(self, new_length: float, **kwargs: goal_parameter):
        """
        Overwrites Giskard trajectory length limit for planning.
        If the trajectory is longer than new_length, Giskard will prempt the goal.
        :param new_length: in seconds
        """
        self.motion_goals.set_max_traj_length(new_length=new_length,
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
        self.motion_goals.add_limit_cartesian_velocity(root_link=root_link,
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
                           add_monitor: bool = True,
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
        end_monitors_monitors = []
        if add_monitor:
            monitor_name = self.monitors.add_distance_to_line(root_link=root_link,
                                                              tip_link=tip_link,
                                                              center_point=bar_center,
                                                              line_axis=bar_axis,
                                                              line_length=bar_length,
                                                              root_group=root_group,
                                                              tip_group=tip_group)
            end_monitors_monitors.append(monitor_name)
            monitor_name = self.monitors.add_vectors_aligned(root_link=root_link,
                                                             tip_link=tip_link,
                                                             goal_normal=bar_axis,
                                                             tip_normal=tip_grasp_axis,
                                                             root_group=root_group,
                                                             tip_group=tip_group)
            end_monitors_monitors.append(monitor_name)
        self.motion_goals.add_grasp_bar(end_monitors=end_monitors_monitors,
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
                                weight=WEIGHT_ABOVE_CA):
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
        self.motion_goals.add_open_container(tip_link=tip_link,
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
                                 weight: Optional[float] = None):
        """
        Same as Open, but will use minimum value as default for goal_joint_state
        """
        self.motion_goals.add_close_container(tip_link=tip_link,
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
                          add_monitor: bool = True,
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
        if add_monitor:
            monitor_name = self.monitors.add_pointing_at(goal_point=goal_point,
                                                         tip_link=tip_link,
                                                         pointing_axis=pointing_axis,
                                                         root_link=root_link,
                                                         tip_group=tip_group,
                                                         root_group=root_group)
            end_monitors_monitors = [monitor_name]
        else:
            end_monitors_monitors = []
        self.motion_goals.add_pointing(end_monitors=end_monitors_monitors,
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
        self.motion_goals.add_avoid_joint_limits(percentage=percentage,
                                                 weight=weight,
                                                 joint_list=joint_list)

    # %% collision avoidance
    def _add_collision_entries_as_goals(self):
        if not self.motion_goals._collision_entries:
            self.avoid_all_collisions()
        super()._add_collision_entries_as_goals()

    def allow_collision(self,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL,
                        start_monitors: Optional[List[str]] = None,
                        hold_monitors: Optional[List[str]] = None,
                        end_monitors: Optional[List[str]] = None):
        """
        Tell Giskard to allow collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param group1: name of the first group
        :param group2: name of the second group
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = str(group1)
        collision_entry.group2 = str(group2)
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_monitors=start_monitors,
                                      hold_monitors=hold_monitors,
                                      end_monitors=end_monitors)

    def avoid_collision(self,
                        min_distance: Optional[float] = None,
                        group1: str = CollisionEntry.ALL,
                        group2: str = CollisionEntry.ALL,
                        start_monitors: Optional[List[str]] = None,
                        hold_monitors: Optional[List[str]] = None,
                        end_monitors: Optional[List[str]] = None):
        """
        Tell Giskard to avoid collision between group1 and group2. Use CollisionEntry.ALL to allow collision with all
        groups.
        :param min_distance: set this to overwrite the default distances
        :param group1: name of the first group
        :param group2: name of the second group
        """
        if min_distance is None:
            min_distance = - 1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        collision_entry.group1 = group1
        collision_entry.group2 = group2
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_monitors=start_monitors,
                                      hold_monitors=hold_monitors,
                                      end_monitors=end_monitors)

    def allow_all_collisions(self,
                             start_monitors: Optional[List[str]] = None,
                             hold_monitors: Optional[List[str]] = None,
                             end_monitors: Optional[List[str]] = None):
        """
        Allows all collisions for next goal.
        """
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_monitors=start_monitors,
                                      hold_monitors=hold_monitors,
                                      end_monitors=end_monitors)

    def avoid_all_collisions(self,
                             min_distance: Optional[float] = None,
                             start_monitors: Optional[List[str]] = None,
                             hold_monitors: Optional[List[str]] = None,
                             end_monitors: Optional[List[str]] = None):
        """
        Avoids all collisions for next goal.
        If you don't want to override the distance, don't call this function. Avoid all is the default, if you don't
        add any collision entries.
        :param min_distance: set this to overwrite default distances
        """
        if min_distance is None:
            min_distance = -1
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.distance = min_distance
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_monitors=start_monitors,
                                      hold_monitors=hold_monitors,
                                      end_monitors=end_monitors)

    def allow_self_collision(self,
                             robot_name: Optional[str] = None,
                             start_monitors: Optional[List[str]] = None,
                             hold_monitors: Optional[List[str]] = None,
                             end_monitors: Optional[List[str]] = None):
        """
        Allows the collision of the robot with itself for the next goal.
        :param robot_name: if there are multiple robots, specify which one.
        """
        if robot_name is None:
            robot_name = self.robot_name
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.group1 = robot_name
        collision_entry.group2 = robot_name
        self._add_collision_avoidance(collisions=[collision_entry],
                                      start_monitors=start_monitors,
                                      hold_monitors=hold_monitors,
                                      end_monitors=end_monitors)
