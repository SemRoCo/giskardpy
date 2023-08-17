import math

import numpy as np
from actionlib import SimpleActionClient

from giskardpy import identifier
from giskardpy.hand_model import Hand, Finger
from giskardpy import casadi_wrapper as w
from giskardpy.goals.goal import Goal, WEIGHT_BELOW_CA, WEIGHT_ABOVE_CA, WEIGHT_COLLISION_AVOIDANCE
from typing import Optional
from geometry_msgs.msg import Vector3Stamped, PointStamped, QuaternionStamped
from giskardpy.goals.cartesian_goals import CartesianOrientation, RotationVelocityLimit
from giskardpy.goals.align_planes import AlignPlanes
from copy import deepcopy

from giskardpy.model.links import BoxGeometry
from giskardpy.utils.tfwrapper import point_to_np


class PlotGoal(Goal):
    def __init__(self, tip_link, root_link):
        super(PlotGoal, self).__init__()
        self.tip = self.world.search_for_link_name(tip_link)
        self.root = self.world.search_for_link_name(root_link)

    def make_constraints(self):
        root_T_tip = self.get_fk(self.root, self.tip)
        self.add_debug_expr('position', root_T_tip.to_position()[:3])
        self.add_debug_expr('orientation', root_T_tip.to_rotation().to_quaternion())

    def __str__(self):
        s = super(PlotGoal, self).__str__()
        return '{}/{}/{}'.format(s, self.root, self.tip)


# TODO: re-implent the FreeGraspBoxFinger goal from my Thesis.
#       - improve on smoother transitions
#       - do I need to implement the heuristic decision? it is nice to have for online adaptation
#           - maybe possible to just copy it, then I should use it
#       - different parameters for the objects that are used to plan the approach movement and to check collisions
#       to fingers (plan_object, collision_object)

# class HSRGripper:
#     def __init__(self):
#         self._gripper_apply_force_client = SimpleActionClient('/hsrb/gripper_controller/grasp',
#                                                               GripperApplyEffortAction)


class GraspBoxMalte(Goal):
    def __init__(self, object_name, root_link: str = 'map', grasp_distance=0.02, grasp_radius=0.2,
                 max_linear_velocity=0.1, group='robot',
                 weight=WEIGHT_BELOW_CA, map_link='map', blocked_directions=None,
                 approach_hint: PointStamped = None, object_root=None):
        super().__init__()
        hand = Hand(hand_tool_frame='hsrb/hand_tool_frame',
                    palm_link='hsrb/hand_palm_link',
                    thumb=Finger(tip_tool_frame='hsrb/thumb_tool_frame',
                                 collision_links=['hsrb/hand_l_distal_link',
                                                  'hsrb/hand_l_proximal_link']),
                    fingers=[Finger(tip_tool_frame='hsrb/finger_tool_frame',
                                    collision_links=['hsrb/hand_l_distal_link',
                                                     'hsrb/hand_l_proximal_link'])
                             ],
                    finger_js={'hand_motor_joint': 0.7},
                    opening_width=0.06)
        self.object = object_name
        self.world_object = self.world.groups[object_name]
        self.object_root = object_root
        self.weight = weight
        self.max_velocity = max_linear_velocity
        self.root_link = root_link
        self.palm_link = hand['palm_link']
        self.grasp_distance = grasp_distance
        self.thumb = hand['thumb']
        self.fingers = hand['fingers']
        self.hand_frame = hand['hand_tool_frame']
        self.opening_width = hand['opening_width']
        self.map_link = map_link
        all_finger = deepcopy(self.fingers)
        all_finger.append(self.thumb)
        self.all_finger = all_finger
        self.blocked = self.approach_hint_to_blocked_directions(approach_hint)
        print(self.blocked)
        # if blocked_directions is not None and len(blocked_directions) == 6:
        #     self.blocked = blocked_directions
        # else:
        #     self.blocked = [0] * 6
        self.finger_js = hand['finger_js']
        self.grasp_radius = grasp_radius
        self.group = group

    def approach_hint_to_blocked_directions(self, approach_hint: PointStamped):
        map_P_hint = self.transform_msg(self.world.root_link_name, approach_hint)
        z = self.world.compute_fk_pose(self.world.root_link_name, self.world_object.root_link_name).pose.position.z
        map_P_hint.point.z = max(z, map_P_hint.point.z)
        object_P_hint = self.transform_msg(self.world_object.root_link_name, map_P_hint)
        object_P_hint = point_to_np(object_P_hint.point)[:3]
        geometry: BoxGeometry = self.world_object.root_link.collisions[0]
        directions = [(np.array([1, 0, 0]), geometry.x_size, [0, 1, 1, 1, 1, 1]),
                      (np.array([-1, 0, 0]), geometry.x_size, [1, 1, 1, 0, 1, 1]),
                      (np.array([0, 1, 0]), geometry.y_size, [1, 0, 1, 1, 1, 1]),
                      (np.array([0, -1, 0]), geometry.y_size, [1, 1, 1, 1, 0, 1]),
                      (np.array([0, 0, 1]), geometry.z_size, [1, 1, 0, 1, 1, 1]),
                      (np.array([0, 0, -1]), geometry.z_size, [1, 1, 1, 1, 1, 0])]
        possible_axis = sorted(directions, key=lambda x: x[1])[2:]
        return max(possible_axis, key=lambda x: np.dot(x[0], object_P_hint))[2]

    def get_actual_distance(self, link):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  (link,
                                                                   self.world.search_for_link_name(self.object)),
                                                                  'contact_distance'])

    def get_actual_distance_object(self, link, obj):
        return self.god_map.to_symbol(identifier.closest_point + ['get_external_collisions_long_key',
                                                                  (link, obj),
                                                                  'contact_distance'])

    def get_map_p_pa(self, coll_link):
        return self.god_map.to_expr(identifier.closest_point + ['get_external_collisions_long_key',
                                                                (coll_link,
                                                                 self.world.search_for_link_name(self.object)),
                                                                'map_P_pa'])

    def get_map_p_pa_object(self, coll_link, obj):
        return self.god_map.to_expr(identifier.closest_point + ['get_external_collisions_long_key',
                                                                (coll_link, obj),
                                                                'map_P_pa'])

    def get_map_p_pb_object(self, coll_link, obj):
        return self.god_map.to_expr(identifier.closest_point + ['get_external_collisions_long_key',
                                                                (coll_link, obj),
                                                                'map_P_pb'])

    def get_actual_distance_finger(self, thumb, tip):
        return self.god_map.to_symbol(identifier.closest_point + ['self_collisions', (
            thumb, tip), 0, 'contact_distance'])

    def get_map_v_n(self, coll_link):
        return self.god_map.to_expr(identifier.closest_point + ['get_external_collisions_long_key',
                                                                (coll_link,
                                                                 self.world.search_for_link_name(self.object)),
                                                                'map_V_n'])

    def get_map_v_n_object(self, coll_link, obj):
        return self.god_map.to_expr(identifier.closest_point + ['get_external_collisions_long_key',
                                                                (coll_link, obj),
                                                                'map_V_n'])

    def stop_non_grasp_object(self):
        return 0
        all_objects = self.world.group_names
        if len(all_objects) > 2:
            for name in all_objects:
                if name != self.group and name != self.object:
                    sum_finger = 0
                    for finger in self.fingers:
                        tip_coll = finger['collision_links'][0]
                        self.add_collision_check(tip_coll, name, 1)
                        distance = self.get_actual_distance_object(tip_coll, name)
                        sum_finger += w.if_less(distance, 0.005, 1, 0)  # change back to 0.01
                    thumb_tip_coll = self.thumb['collision_links'][0]
                    self.add_collision_check(thumb_tip_coll, name, 1)
                    distance = self.get_actual_distance_object(thumb_tip_coll, name)
                    condition_value = w.min(sum_finger, 1) + w.if_less(distance, 0.02, 1, 0)
                    return w.if_eq(condition_value, 2, 1, 0)
        return 0

    def add_points(self, point1: w.Point3, point2: w.Point3):
        s = [point1[0] + point2[0],
             point1[1] + point2[1],
             point1[2] + point2[2]]
        return w.Point3(s)

    def make_constraints(self):
        # --------------------------First Phase------------------------------------------------------------------
        # The Goal is to move to the closest pregrasp point in grasp distance, by testing all six possible positions
        root_T_hand = self.get_fk(self.root_link, self.hand_frame)
        root_P_hand = root_T_hand.to_position()
        directions = [[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]]
        grasp_orientations = []
        finger_planes = []
        penalty = []
        if self.object_root is None:
            object_collisions = self.world.groups[self.object].root_link.collisions[0]
        else:
            object_collisions = self.world.groups[self.object_root].links[self.object].collisions[0]
        object_dims = [object_collisions.depth, object_collisions.width, object_collisions.height]
        # Calculate the smallest width of the object in the 2D-Graspplane
        for direction in directions:
            p = [0, 0, 0]
            for i, elem in enumerate(direction):
                p[i] = object_dims[i] - abs(elem) * object_dims[i]
            max_elem = max(p)
            for i, elem in enumerate(p):
                if elem == max_elem:
                    p[i] = 0
                    break
            if max(p) >= self.opening_width:
                penalty.append(100)
            else:
                penalty.append(0)
            finger_planes.append(w.Vector3(object_dims) - w.Vector3(p))
            p = -w.Vector3(p) - w.Vector3(p)
            p /= w.norm(p)
            grasp_orientations.append(p)

        root_T_object = self.get_fk_evaluated(self.root_link, self.world.search_for_link_name(self.object))
        root_P_object = root_T_object.to_position()
        root_V_goal = w.dot(root_T_object, w.Vector3(directions[0]))
        root_V_goal.vis_frame = self.world_object.root_link_name
        self.add_debug_expr('root_V_goal', root_V_goal)
        root_P_goal = root_P_object + (self.grasp_radius + w.abs(w.Vector3(directions[0]).dot(w.Vector3(object_dims)))
                                       / 2) * root_V_goal
        self.add_debug_expr('v_goal0', root_P_goal)
        self.add_debug_expr('object', root_P_object)
        root_V_orientation = w.dot(root_T_object, grasp_orientations[0])
        object_finger_plane = finger_planes[0]
        distance_old = (root_P_goal - root_P_hand).norm() + penalty[0] + self.blocked[0] * 100
        for i, direction in enumerate(directions):
            # self.add_debug_expr('v_goal' + str(i), V_goal)
            if i == 0:
                continue

            V_goal = w.dot(root_T_object, w.Vector3(direction))
            V_goal.vis_frame = self.world_object.root_link_name
            goal = root_P_object + (
                    (self.grasp_radius + w.abs(w.Vector3(direction).dot(w.Vector3(object_dims))) / 2) * V_goal)
            distance_new = (goal - root_P_hand).norm() + penalty[i] + self.blocked[i] * 100

            new_root_P_goal = w.if_less(distance_new,
                                        distance_old,
                                        goal, root_P_goal)
            new_root_V_goal = w.if_less(distance_new,
                                        distance_old,
                                        V_goal, root_V_goal)
            new_root_V_orientation = w.if_less(distance_new,
                                               distance_old,
                                               w.dot(root_T_object, grasp_orientations[i]), root_V_orientation)
            new_object_finger_plane = w.if_less(distance_new,
                                                distance_old,
                                                finger_planes[i], object_finger_plane)
            distance_old = w.if_less(distance_new, distance_old, distance_new, distance_old)
            root_P_goal = new_root_P_goal
            root_V_goal = new_root_V_goal
            root_V_orientation = new_root_V_orientation
            object_finger_plane = new_object_finger_plane

        # calculate error and ecpression for the constraint to orient the line between the finger like the smallest
        # object dimension
        root_T_hand = self.get_fk(self.root_link, self.hand_frame)
        root_V_tips = w.dot(root_T_hand, w.Vector3([1, 0, 0]))

        root_V_goal_2 = root_V_orientation
        lower_error = -root_V_goal_2 - root_V_tips  # TODO reset the minus
        upper_error = root_V_goal_2 - root_V_tips
        angle = w.min(w.angle_between_vector(root_V_goal_2, root_V_tips),
                      w.angle_between_vector(-root_V_goal_2, root_V_tips))
        lower_error = w.if_less(w.angle_between_vector(root_V_goal_2, root_V_tips),
                                w.angle_between_vector(-root_V_goal_2, root_V_tips), -root_V_goal_2 - root_V_tips,
                                -root_V_goal_2 - root_V_tips)
        root_T_hand = self.get_fk(self.root_link, self.hand_frame)
        root_V_current = w.dot(root_T_hand, w.Vector3([0, 0, 1]))
        angle = w.max(angle, w.angle_between_vector(-root_V_goal, root_V_current))

        # clculate the weight to stop and start prepositioning of the hand.
        # define a cylinder. endpoint=root_P_goal, startpoint=root_P_object,
        # radius=max(object_dims-object_finger_plane)/2.
        # calculate distance of root_P_hand to the axis of the cylinder.
        # if distance < radius, then stop the prepositioning and start the contact
        radius = 0.08  # 0.02
        dist, nearest = w.distance_point_to_line_segment(root_P_hand, root_P_object, root_P_goal + root_V_goal * 0.5)

        weight_outside_expr1 = w.if_greater_eq(dist, radius, self.weight, 0)
        weight_outside_expr2 = w.if_greater_eq(angle, 0.2, self.weight, 0)
        weight_outside_cylinder = w.if_eq(weight_outside_expr1, weight_outside_expr2, weight_outside_expr1, self.weight)
        weight_inside_cylinder = w.if_eq(weight_outside_cylinder, 0, self.weight, 0)
        # self.add_debug_expr('weight_outside', weight_outside_cylinder/weight*1000)
        self.add_debug_expr('angle', angle)
        self.add_debug_expr('dist', dist)
        # self.add_debug_vector('root_V_orientation', root_V_orientation[:3])

        root_T_hand = self.get_fk(self.root_link, self.hand_frame)
        root_V_current = w.dot(root_T_hand, w.Vector3([0, 0, 1]))
        error_direction = -root_V_goal[:3] - root_V_current[:3]
        error = root_P_goal - root_T_hand.to_position()

        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity * 1] * 3,
                                            equality_bounds=error_direction,
                                            task_expression=root_V_current[:3],
                                            weights=[weight_outside_cylinder] * 3,
                                            names=['orthoAxisx1', 'orthoAxisy1', 'orthoAxisz1'])

        self.add_debug_expr('handGoal', root_P_goal)
        # self.add_debug_expr('V_goal', root_V_goal)
        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity / 1] * 3,
                                            equality_bounds=error[:3],
                                            task_expression=root_T_hand.to_position()[:3],
                                            weights=[weight_outside_cylinder, weight_outside_cylinder,
                                                     weight_outside_cylinder],
                                            names=['orthoPointx', 'orthoPointy', 'orthoPointz'])
        # self.add_inequality_constraint_vector(reference_velocities=[self.max_velocity / 1] * 3,
        #                                       lower_errors=error[:3] - 0.05,
        #                                       upper_errors=error[:3] + 0.05,
        #                                       task_expression=root_T_hand.to_position()[:3],
        #                                       weights=[weight_outside_cylinder,
        #                                                weight_outside_cylinder,
        #                                                weight_outside_cylinder],
        #                                       names=['orthoPointx', 'orthoPointy',
        #                                              'orthoPointz'])

        # orient the line between the fingers parallel to the line of the smallest width
        root_V_y = w.dot(root_T_hand, w.Vector3([0, 1, 0]))
        root_V_z = w.Vector3([0, 0, 1])
        hacked_error = root_V_z - root_V_y
        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity * 3] * 3,
                                            equality_bounds=hacked_error[:3],
                                            task_expression=root_V_y[:3],
                                            weights=[weight_outside_cylinder, weight_outside_cylinder,
                                                     weight_outside_cylinder],
                                            names=['orient_x', 'orient_y', 'orient_z'])
        # ---------------------------SecondPhase-------------------------------------------------------------------------
        self.add_collision_check(self.palm_link, self.world.search_for_link_name(self.object), 1)
        distance = self.get_actual_distance(self.palm_link)
        value_non_grasp_contact_stop = self.stop_non_grasp_object()
        # self.add_debug_expr('distance_palm', w.min(distance, 0.1))
        palm_contact_weight1 = w.if_less(distance, self.grasp_distance, 0, weight_inside_cylinder)
        palm_contact_weight2 = w.if_eq(value_non_grasp_contact_stop, 1, 0, weight_inside_cylinder)
        palm_contact_weight = w.if_eq(palm_contact_weight1, palm_contact_weight2, palm_contact_weight1, 0)
        # Todo: bessere Bedingung, die checkt, ob conatct ist oder nicht. Nur ein finger nah genug am Hinderniss ist
        #       nicht zwangslaeufig ausreichend. Aktuell Okay
        root_P_hand = self.get_fk(self.root_link, self.hand_frame).to_position()
        root_P_object = self.get_fk_evaluated(self.root_link,
                                              self.world.search_for_link_name(self.object)).to_position()
        error = root_P_object - root_P_hand
        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity / 1] * 3,
                                            equality_bounds=error[:3],
                                            task_expression=root_P_hand[:3],
                                            weights=[palm_contact_weight] * 3,
                                            names=['CoM_x', 'CoM_y', 'CoM_z'])
        # self.add_inequality_constraint_vector(reference_velocities=[self.max_velocity / 3] * 3,
        #                                       lower_errors=(error - w.Vector3([0, 0, 0.05]))[:3],
        #                                       upper_errors=(error + w.Vector3([0, 0, 0.05]))[:3],
        #                                       task_expression=root_P_hand[:3],
        #                                       weights=[palm_contact_weight] * 3,
        #                                       names=['CoM_x', 'CoM_y', 'CoM_z'])

        # keep orientation with the hand tool frame pointing on the object to pick up smaller objects.
        # continuation from above
        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity * 1] * 3,
                                            equality_bounds=error_direction,
                                            task_expression=root_V_current[:3],
                                            weights=[palm_contact_weight] * 3,
                                            names=['orthoAxisx2', 'orthoAxisy2', 'orthoAxisz2'])
        self.add_equality_constraint_vector(reference_velocities=[0.2, 0.2, 0.2],
                                            equality_bounds=hacked_error[:3],
                                            task_expression=root_V_y[:3],
                                            weights=[palm_contact_weight] * 3,
                                            names=['orient_x2', 'orient_y2', 'orient_z2'])
        # ---------------------Keep Hand open during Pahse 2------------------------------------------------------
        for key in self.finger_js.keys():
            joint_symbol = self.get_joint_position_symbol(self.world.search_for_joint_name(key))
            goal_position = self.finger_js[key]
            error = goal_position - joint_symbol
            self.add_equality_constraint(reference_velocity=self.max_velocity,
                                         equality_bound=error,
                                         task_expression=joint_symbol,
                                         weight=w.max(palm_contact_weight, weight_outside_cylinder),
                                         name='pose_{}'.format(key))

        # -------------------------------Phase 3---------------------------------------------------------
        # object_finger_plane beschreibt die Flaeche auf die die Finger greifen
        # ich muss testen welche Fingerspitzen in der Flaeche liegen
        # spaeter welche Finger durch die Flaeche gehen
        # was mache ich mit dem Finger der drin ist?
        #   - Weight aktiviieren, für schließen bis Kontakt und Position zur Mitelpunkt Berechnung fuer den
        #   daumen nehmen
        #   - gezielt besser über die Flaeche verteilen und ...
        object_P_virtual = w.Point3([0, 0, 0])
        sum = 0
        power_sum = 0
        object_T_root = self.get_fk(self.world.search_for_link_name(self.object), self.root_link)
        for finger in self.fingers:
            tip = finger['tip_tool_frame']
            coll = finger['collision_links'][0]
            medial = finger['collision_links'][1]
            finger_root = finger['collision_links'][-1]
            self.add_collision_check(coll, self.world.search_for_link_name(self.object), 1)
            local_weight = w.if_eq(palm_contact_weight, 0, weight_inside_cylinder, 0)
            object_P_tip = self.get_fk_evaluated(self.world.search_for_link_name(self.object), tip).to_position() * 100
            object_dims = object_finger_plane * 100 / 2

            # test fingertip position for fingertip grasp
            lesser = w.dot(w.TransMatrix(w.diag(object_P_tip)), object_dims)
            greater = w.dot(w.TransMatrix(w.diag(object_dims)), object_dims)
            local_weight = w.if_less_eq(w.abs(lesser[0]), greater[0], local_weight, 0)
            local_weight = w.if_less_eq(w.abs(lesser[1]), greater[1], local_weight, 0)
            local_weight = w.if_less_eq(w.abs(lesser[2]), greater[2], local_weight, 0)
            object_P_tip = self.get_fk_evaluated(self.world.search_for_link_name(self.object), tip).to_position() * 100
            object_P_virtual = self.add_points(object_P_virtual, object_P_tip * (local_weight / self.weight))
            sum = sum + 1 * (local_weight / self.weight)

            # test ... for power grasp
            # linie vom abd_link(root_link des finger) zumfinger tool frame im objekt KS
            # ignorieren einer dimension
            # testen ob diese 2D linie das rechteck doppelt schneidet
            # Das oben genannte waere ideal, zum testen gucke ich aber nur ob tip nicht in der flaeche und medial link in der flaeche
            power_weight = w.if_eq(palm_contact_weight, 0, weight_inside_cylinder, 0)
            power_weight = w.if_eq(local_weight, 0, power_weight, 0)
            object_P_medial = self.get_fk_evaluated(self.world.search_for_link_name(self.object),
                                                    medial).to_position() * 100
            lesser = w.dot(w.TransMatrix(w.diag(object_P_medial)), object_dims)
            power_weight = w.if_less_eq(w.abs(lesser[0]), greater[0] + 3, power_weight, 0)
            power_weight = w.if_less_eq(w.abs(lesser[1]), greater[1] + 3, power_weight, 0)
            power_weight = w.if_less_eq(w.abs(lesser[2]), greater[2] + 3, power_weight, 0)

            # close the finger, for fingertip grasp
            palm_T_tip = self.get_fk(self.hand_frame, tip)
            palm_P_tip = palm_T_tip.to_position()
            palm_P_goal = palm_P_tip + 0.01 * w.Vector3([1, 0, 0])

            distance = self.get_actual_distance(coll)
            local_weight = w.if_less(distance, 0.005, 0, local_weight)

            error = palm_P_goal - palm_P_tip
            self.add_equality_constraint_vector(reference_velocities=[self.max_velocity] * 3,
                                                equality_bounds=error[:3],
                                                task_expression=palm_P_tip[:3],
                                                weights=[local_weight, local_weight, local_weight],
                                                names=['{}_x1'.format(tip), '{}_y1'.format(tip),
                                                       '{}_z1'.format(tip)])

            # close the finger for power grasp
            palm_V_x = w.dot(palm_T_tip, w.Vector3([1, 0, 0]))
            palm_P_goal = palm_P_tip + 0.01 * palm_V_x
            power_weight = w.if_less(distance, 0.005, 0, power_weight)
            error = palm_P_goal - palm_P_tip
            self.add_equality_constraint_vector(reference_velocities=[self.max_velocity] * 3,
                                                equality_bounds=error[:3],
                                                task_expression=palm_P_tip[:3],
                                                weights=[power_weight, power_weight, power_weight],
                                                names=['power_{}_x'.format(tip), 'power_{}_y'.format(tip),
                                                       'power_{}_z'.format(tip)])
            power_sum += power_weight

        object_P_virtual /= (sum * 100)  # Virtual Finger
        object_P_virtual[3] = 1
        # invert the dimension, that is to be grasped. new point is goal for the thumb
        object_T_root = self.get_fk_evaluated(self.world.search_for_link_name(self.object), self.root_link)
        object_V_orientation = w.dot(object_T_root, root_V_orientation)
        inverter = w.diag(w.Expression(w.abs(object_V_orientation) * -2 + w.Point3([1, 1, 1])))  # .nz
        inverted_object_P_virtual = w.dot(w.TransMatrix(inverter), object_P_virtual)

        # close the thumb to the object
        palm_T_object = self.get_fk(self.palm_link, self.world.search_for_link_name(self.object))
        palm_T_thumb = self.get_fk(self.palm_link, self.thumb['tip_tool_frame'])
        palm_P_thumb = palm_T_thumb.to_position()
        self.add_collision_check(self.thumb['collision_links'][0], self.world.search_for_link_name(self.object), 1)
        distance = self.get_actual_distance(self.thumb['collision_links'][0])
        local_weight = w.if_eq(palm_contact_weight, 0, weight_inside_cylinder, 0)
        local_weight = w.if_less(power_sum, self.weight, local_weight, 0)  # movement delay
        local_weight = w.if_less(distance, 0.005, 0, local_weight)
        for finger in self.fingers:
            coll = finger['collision_links'][0]
            self.add_collision_check(self.thumb['collision_links'][0], coll, 1)
            distance = self.get_actual_distance_finger(self.thumb['collision_links'][0], coll)
            # self.add_debug_expr('distance_thumb_{}'.format(coll), distance)
            local_weight = w.if_less(distance, 0.005, 0, local_weight)

        precision_weight = w.if_eq(sum, 0, 0, local_weight)
        error = w.dot(palm_T_object, w.Point3(inverted_object_P_virtual)) - palm_P_thumb
        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity] * 3,
                                            equality_bounds=error[:3],
                                            task_expression=palm_P_thumb[:3],
                                            weights=[precision_weight, precision_weight, precision_weight],
                                            names=['thumb_x', 'thumb_y', 'thumb_z'])

        # close thumb for power grasp
        palm_V_x = w.dot(palm_T_thumb, w.Vector3([1, 0, 0]))
        palm_P_goal = palm_P_thumb + 0.005 * palm_V_x

        power_weight = w.if_eq(sum, 0, local_weight, 0)
        error = palm_P_goal - palm_P_thumb
        self.add_equality_constraint_vector(reference_velocities=[self.max_velocity] * 3,
                                            equality_bounds=error[:3],
                                            task_expression=palm_P_thumb[:3],
                                            weights=[power_weight, power_weight, power_weight],
                                            names=['thumb_power_x', 'thumb_power_y',
                                                   'thumb_power_z'])
        self.add_debug_expr('Phase1', weight_outside_cylinder)
        self.add_debug_expr('Phase2', palm_contact_weight)
        self.add_debug_expr('PhaseTip', precision_weight)
        self.add_debug_expr('toolFrame', root_P_hand[:3])

    def __str__(self):
        s = super().__str__()
        return '{}/{}/{}'.format(s, self.root_link, self.palm_link)
