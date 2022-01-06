from copy import deepcopy

import numpy
import yaml
import pybullet as p
from py_trees import Status

from giskardpy import identifier, RobotName
from giskardpy.data_types import Trajectory, Collisions
from giskardpy.tree.get_goal import GetGoal
from giskardpy.utils.tfwrapper import np_to_pose_stamped, transform_pose, msg_to_homogeneous_matrix
from giskardpy.utils.utils import convert_dictionary_to_ros_message


class GlobalPlannerNeeded(GetGoal):

    def __init__(self, name, as_name, solver):
        GetGoal.__init__(self, name, as_name)

        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.supported_cart_goals = ['CartesianPose', 'CartesianPosition', 'CartesianPathCarrot']

        self.pose_goal = None
        self.__goal_dict = None
        self.solver = solver

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type in self.supported_cart_goals)
        except StopIteration:
            return None

    def is_global_path_needed(self, env_group='kitchen'):
        env_link_names = self.world.groups[env_group].link_names_with_collisions
        ids = list()
        for l in env_link_names:
            if l in self.collision_scene.object_name_to_bullet_id:
                ids.append(self.collision_scene.object_name_to_bullet_id[l])
            else:
                raise Exception(u'Link {} with collision was not added in the collision scene.'.format(l))
        return self.__is_global_path_needed(ids)

    def __is_global_path_needed(self, coll_body_ids):
        """
        (disclaimer: for correct format please see the source code)

        Returns whether a global path is needed by checking if the shortest path to the goal
        is free of collisions.

        start        new_start (calculated with vector v)
        XXXXXXXXXXXX   ^
        XXcollXobjXX   |  v
        XXXXXXXXXXXX   |
                     goal

        The vector from start to goal in 2D collides with collision object, but the vector
        from new_start to goal does not collide with the environment, but...

        start        new_start
        XXXXXXXXXXXXXXXXXXXXXX
        XXcollisionXobjectXXXX
        XXXXXXXXXXXXXXXXXXXXXX
                     goal

        ... here the shortest path to goal is in collision. Therefore, a global path
        is needed.

        :rtype: boolean
        """
        pose_matrix = self.get_robot().get_fk(self.root_link, self.tip_link)
        curr_R_pose = np_to_pose_stamped(pose_matrix, self.root_link)
        curr_pos = transform_pose(self.map_frame, curr_R_pose).pose.position
        curr_arr = numpy.array([curr_pos.x, curr_pos.y, curr_pos.z])
        goal_pos = self.pose_goal.pose.position
        goal_arr = numpy.array([goal_pos.x, goal_pos.y, goal_pos.z])
        obj_id, _, _, _, normal = p.rayTest(curr_arr, goal_arr)[0]
        if obj_id in coll_body_ids:
            diff = numpy.sqrt(numpy.sum((curr_arr - goal_arr) ** 2))
            v = numpy.array(list(normal)) * diff
            new_curr_arr = goal_arr + v
            obj_id, _, _, _, _ = p.rayTest(new_curr_arr, goal_arr)[0]
            return obj_id in coll_body_ids
        else:
            return False

    def save_cart_goal(self, cart_c):

        self.__goal_dict = yaml.load(cart_c.parameter_value_pair)
        ros_pose = convert_dictionary_to_ros_message(self.__goal_dict[u'goal'])
        self.pose_goal = transform_pose(self.map_frame, ros_pose)

        self.root_link = self.__goal_dict[u'root_link']
        self.tip_link = self.__goal_dict[u'tip_link']
        link_names = self.get_robot().link_names

        if self.root_link not in link_names:
            raise Exception(u'Root_link {} is no known link of the robot.'.format(self.root_link))
        if self.tip_link not in link_names:
            raise Exception(u'Tip_link {} is no known link of the robot.'.format(self.tip_link))
        if not self.get_robot().are_linked(self.root_link, self.tip_link):
            raise Exception(u'Did not found link chain of the robot from'
                            u' root_link {} to tip_link {}.'.format(self.root_link, self.tip_link))

    def clear_trajectory(self):
        self.world.fast_all_fks = None
        self.collision_scene.reset_cache()
        self.get_god_map().set_data(identifier.closest_point, Collisions(self.world, 1))
        self.get_god_map().set_data(identifier.time, 1)
        current_js = deepcopy(self.get_god_map().get_data(identifier.joint_states))
        trajectory = Trajectory()
        trajectory.set(0, current_js)
        self.get_god_map().set_data(identifier.trajectory, trajectory)
        trajectory = Trajectory()
        self.get_god_map().set_data(identifier.debug_trajectory, trajectory)

    def reset_robot_state_and_pose(self):
        start_joint_states = deepcopy(self.god_map.get_data(identifier.old_joint_states))
        self.get_world().state.update(start_joint_states)
        start_map_T_base = deepcopy(self.god_map.get_data(identifier.old_map_T_base))
        self.world.update_joint_parent_T_child(self.world.groups[RobotName].attachment_joint_name, start_map_T_base)

    def reset(self):
        self.clear_trajectory()
        self.reset_robot_state_and_pose()

    def is_qp_solving_running(self):
        return self.solver.my_status == Status.RUNNING

    def update(self):

        # Check if giskard is solving currently
        if self.is_qp_solving_running():
            return Status.RUNNING

        # Check if someone set the variable to true
        if self.get_god_map().get_data(identifier.global_planner_needed):
            self.reset()
            return Status.RUNNING

        # Check if move_cmd exists
        move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
        if not move_cmd:
            self.get_god_map().set_data(identifier.global_planner_needed, False)
            return Status.RUNNING

        # Check if move_cmd contains a Cartesian Goal
        cart_c = self.get_cart_goal(move_cmd)
        if not cart_c:
            self.get_god_map().set_data(identifier.global_planner_needed, False)
            return Status.RUNNING

        if cart_c.type == 'CartesianPathCarrot':
            self.reset()
            self.get_god_map().set_data(identifier.global_planner_needed, True)
            return Status.RUNNING

        # Parse and save the Cartesian Goal Constraint
        self.save_cart_goal(cart_c)
        # fixme: two sync calls - what the fuck
        self.collision_scene.sync()
        self.collision_scene.sync()
        if self.is_global_path_needed():
            self.reset()
            self.get_god_map().set_data(identifier.global_planner_needed, True)
        else:
            self.get_god_map().set_data(identifier.global_planner_needed, False)

        return Status.RUNNING