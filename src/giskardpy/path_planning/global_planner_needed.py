import threading
from copy import deepcopy

import numpy
import rospy
import yaml
import pybullet as p
from py_trees import Status

from giskardpy import identifier
from giskard_msgs.srv import GlobalPathNeeded, GlobalPathNeededResponse
from giskardpy.model.collision_world_syncer import Collisions
from giskardpy.model.trajectory import Trajectory
from giskardpy.path_planning.motion_validator import SimpleRayMotionValidator, ObjectRayMotionValidator
from giskardpy.path_planning.state_validator import GiskardRobotBulletCollisionChecker
from giskardpy.tree.behaviors.get_goal import GetGoal
from giskardpy.utils.tfwrapper import np_to_pose_stamped, transform_pose, pose_stamped_to_np, np_to_pose
from giskardpy.utils.utils import convert_dictionary_to_ros_message


class GlobalPlannerNeeded(GetGoal):

    def __init__(self, name, as_name):
        GetGoal.__init__(self, name, as_name)

        self.map_frame = 'map'
        self.global_path_needed_lock = threading.Lock()
        self.supported_cart_goals = ['CartesianPathCarrot']
        # self.solver = None
        
    def setup(self, timeout=5.0):
        self.srv_path_needed = rospy.Service(u'~is_global_path_needed', GlobalPathNeeded, self.is_global_path_needed_cb)
        return super(GlobalPlannerNeeded, self).setup(timeout)

    def get_cart_goal(self, cmd):
        try:
            return next(c for c in cmd.constraints if c.type in self.supported_cart_goals)
        except StopIteration:
            return None

    def get_collision_ids(self, env_group='kitchen'):
        env_link_names = self.world.groups[env_group].link_names_with_collisions
        ids = list()
        for l in env_link_names:
            if l in self.collision_scene.object_name_to_bullet_id:
                ids.append(self.collision_scene.object_name_to_bullet_id[l])
            else:
                raise Exception(u'Link {} with collision was not added in the collision scene.'.format(l))
        return ids

    def is_global_path_needed_cb(self, req):
        resp = GlobalPathNeededResponse()
        if req.env_group != '':
            resp.needed = self.is_global_path_needed(req.root_link, req.tip_link, req.pose_goal, req.simple,
                                                     env_group=req.env_group)
        else:
            resp.needed = self.is_global_path_needed(req.root_link, req.tip_link, req.pose_goal, req.simple)
        return resp

    def is_global_path_needed(self, root_link, tip_link, pose_goal, simple, env_group='kitchen'):
        with self.global_path_needed_lock:
            # fixme: two sync calls - what the fuck
            self.collision_scene.sync()
            self.collision_scene.sync()
            #if simple:
            #    ids = self.get_collision_ids(env_group)
            #    return self.__is_global_path_needed(root_link, tip_link, pose_goal, ids)
            #else:
            links = self.world.groups[self.robot.name].link_names
            if tip_link not in links:
                raise Exception('wa')
            collision_checker = GiskardRobotBulletCollisionChecker(tip_link != 'base_footprint', root_link,
                                                                   tip_link, self.collision_scene, self.get_god_map())
            if simple:
                m = SimpleRayMotionValidator(self.collision_scene, tip_link, self.god_map,
                                             js=self.get_god_map().get_data(identifier.joint_states))
            else:
                m = ObjectRayMotionValidator(self.collision_scene, tip_link, self.robot, collision_checker, self.god_map,
                                             js=self.get_god_map().get_data(identifier.joint_states))
            start = np_to_pose(self.robot.get_fk(root_link, tip_link))
            result = not m.check_motion(start, pose_goal)
            collision_checker.clear()
            m.clear()
            return result

    # def parse_cart_goal(self, cart_c):
    #
    #     __goal_dict = yaml.load(cart_c.parameter_value_pair)
    #     ros_pose = convert_dictionary_to_ros_message(__goal_dict[u'goal'])
    #     pose_goals = list()
    #     if 'goals' in __goal_dict:
    #         pose_goals = list(map(convert_dictionary_to_ros_message, __goal_dict[u'goals']))
    #     pose_goal = transform_pose(self.map_frame, ros_pose).pose
    #
    #     root_link = __goal_dict[u'root_link']
    #     tip_link = __goal_dict[u'tip_link']
    #     link_names = self.robot.link_names
    #
    #     if root_link not in link_names:
    #         raise Exception(u'Root_link {} is no known link of the robot.'.format(root_link))
    #     if tip_link not in link_names:
    #         raise Exception(u'Tip_link {} is no known link of the robot.'.format(tip_link))
    #
    #     return root_link, tip_link, pose_goal, pose_goals

    # def clear_trajectory(self):
    #     self.world.fast_all_fks = None
    #     self.collision_scene.reset_cache()
    #     self.get_god_map().set_data(identifier.closest_point, Collisions(self.god_map, 1))
    #     self.get_god_map().set_data(identifier.time, 1)
    #     current_js = deepcopy(self.get_god_map().get_data(identifier.joint_states))
    #     trajectory = Trajectory()
    #     trajectory.set(0, current_js)
    #     self.get_god_map().set_data(identifier.trajectory, trajectory)
    #     trajectory = Trajectory()
    #     self.get_god_map().set_data(identifier.debug_trajectory, trajectory)

    # def reset_robot_state_and_pose(self):
    #     start_joint_states = deepcopy(self.god_map.get_data(identifier.old_joint_states))
    #     self.get_world().state.update(start_joint_states)
    #     start_map_T_base = deepcopy(self.god_map.get_data(identifier.old_map_T_base))
    #     self.world.update_joint_parent_T_child(self.world.groups['robot'].attachment_joint_name, start_map_T_base)

    # def reset(self):
    #     self.clear_trajectory()
    #     self.reset_robot_state_and_pose()

    # def is_qp_solving_running(self):
    #     return self.solver is None or self.solver.my_status == Status.RUNNING

    # def get_cartesian_move_constraint(self):
    #
    #     # Check if move_cmd exists
    #     move_cmd = self.get_god_map().get_data(identifier.next_move_goal)  # type: MoveCmd
    #     if not move_cmd:
    #         return None
    #
    #     # Check if move_cmd contains a Cartesian Goal
    #     return self.get_cart_goal(move_cmd)

    # def is_cartesian_constraint_nontrivial(self, cartesian_constraint):
    #     if cartesian_constraint.type == 'CartesianPathCarrot':
    #         return True
    #     else:
    #         r, t, p, gs = self.parse_cart_goal(cartesian_constraint)
    #         return self.is_global_path_needed(r, t, p, True)

    # def is_unplanned(self, cartesian_constraint):
    #     if cartesian_constraint.type == 'CartesianPathCarrot':
    #         r, t, p, gs = self.parse_cart_goal(cartesian_constraint)
    #         return len(gs) == 0
    #    else:
    #        raise Exception('no path constraint')

    def update(self):

        # # Check if giskard is solving currently
        # if self.is_qp_solving_running():
        #     return Status.RUNNING
        #
        # # Check if someone set the variable to true
        # if self.get_god_map().get_data(identifier.global_planner_needed):
        #     self.reset()
        #     return Status.RUNNING
        #
        # # Check if cartesian goals are defined
        # cart_c = self.get_cartesian_move_constraint()
        # if cart_c is None:
        #     self.get_god_map().set_data(identifier.global_planner_needed, False)
        #     return Status.RUNNING
        #
        # if self.is_unplanned(cart_c):
        #     self.reset()
        #     self.god_map.set_data(identifier.global_planner_needed, True)
        #
        # Else check if cartesian goal is nontrivial
        # if self.is_cartesian_constraint_nontrivial(cart_c):
        #     self.reset()
        #     self.get_god_map().set_data(identifier.global_planner_needed, True)
        # else:
        #     self.get_god_map().set_data(identifier.global_planner_needed, False)

        return Status.RUNNING