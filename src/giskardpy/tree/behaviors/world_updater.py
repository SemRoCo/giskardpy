import traceback
from collections import defaultdict
from copy import deepcopy
from itertools import product
from queue import Queue
from xml.etree.ElementTree import ParseError

import rospy
from py_trees import Status
from py_trees.meta import running_is_success
from tf2_py import TransformException
from visualization_msgs.msg import MarkerArray, Marker

import giskardpy.casadi_wrapper as w
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest, GetGroupNamesResponse, \
    GetGroupNamesRequest, RegisterGroupRequest, RegisterGroupResponse, \
    GetGroupInfoResponse, GetGroupInfoRequest, DyeGroupResponse, GetGroupNames, GetGroupInfo, RegisterGroup, DyeGroup, \
    DyeGroupRequest
from giskardpy.data_types import JointStates
from giskardpy.exceptions import CorruptShapeException, UnknownGroupException, \
    UnsupportedOptionException, DuplicateNameException, UnknownLinkException
from giskardpy.god_map import god_map
from giskardpy.model.world import WorldBranch
from giskardpy.data_types import PrefixName
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.sync_joint_state import SyncJointState
from giskardpy.tree.behaviors.sync_tf_frames import SyncTfFrames
from giskardpy.utils import logging
from giskardpy.utils.decorators import record_time
from giskardpy.utils.tfwrapper import transform_pose, msg_to_homogeneous_matrix


def exception_to_response(e, req):
    def error_in_list(error, list_of_errors):
        result = False
        for x in list_of_errors:
            result |= isinstance(error, x)
        return result

    if error_in_list(e, [CorruptShapeException, ParseError]):
        traceback.print_exc()
        if req.body.type == req.body.MESH_BODY:
            return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_MESH_ERROR, str(e))
        elif req.body.type == req.body.URDF_BODY:
            return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_URDF_ERROR, str(e))
        return UpdateWorldResponse(UpdateWorldResponse.CORRUPT_SHAPE_ERROR, str(e))
    elif error_in_list(e, [UnknownGroupException]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.UNKNOWN_GROUP_ERROR, str(e))
    elif error_in_list(e, [UnknownLinkException]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.UNKNOWN_LINK_ERROR, str(e))
    elif error_in_list(e, [DuplicateNameException]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.DUPLICATE_GROUP_ERROR, str(e))
    elif error_in_list(e, [UnsupportedOptionException]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS, str(e))
    elif error_in_list(e, [TransformException]):
        return UpdateWorldResponse(UpdateWorldResponse.TF_ERROR, str(e))
    else:
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.ERROR,
                                   '{}: {}'.format(e.__class__.__name__,
                                                   str(e)))


class WorldUpdater(GiskardBehavior):
    READY = 0
    BUSY = 1
    STALL = 2

    @profile
    def __init__(self, name: str):
        super().__init__(name)
        self.original_link_names = god_map.world.link_names_as_set
        self.service_in_use = Queue(maxsize=1)
        self.work_permit = Queue(maxsize=1)
        self.update_ticked = Queue(maxsize=1)
        self.timer_state = self.READY

    @record_time
    @profile
    def setup(self, timeout: float = 5.0):
        self.marker_publisher = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1)
        self.srv_update_world = rospy.Service('~update_world', UpdateWorld, self.update_world_cb)
        self.get_group_names = rospy.Service('~get_group_names', GetGroupNames, self.get_group_names_cb)
        self.get_group_info = rospy.Service('~get_group_info', GetGroupInfo, self.get_group_info_cb)
        self.register_groups = rospy.Service('~register_groups', RegisterGroup, self.register_groups_cb)
        self.dye_group = rospy.Service('~dye_group', DyeGroup, self.dye_group)
        return super(WorldUpdater, self).setup(timeout)

    def dye_group(self, req: DyeGroupRequest):
        res = DyeGroupResponse()
        try:
            god_map.world.dye_group(req.group_name, req.color)
            res.error_codes = DyeGroupResponse.SUCCESS
            for link_name in god_map.world.groups[req.group_name].links:
                god_map.world.links[link_name].reset_cache()
            logging.loginfo(
                f'dyed group \'{req.group_name}\' to r:{req.color.r} g:{req.color.g} b:{req.color.b} a:{req.color.a}')
        except UnknownGroupException:
            res.error_codes = DyeGroupResponse.GROUP_NOT_FOUND_ERROR
        return res

    @profile
    def register_groups_cb(self, req: RegisterGroupRequest) -> RegisterGroupResponse:
        link_name = god_map.world.search_for_link_name(req.root_link_name, req.parent_group_name)
        god_map.world.register_group(req.group_name, link_name)
        res = RegisterGroupResponse()
        res.error_codes = res.SUCCESS
        return res

    @profile
    def get_group_names_cb(self, req: GetGroupNamesRequest) -> GetGroupNamesResponse:
        group_names = god_map.world.group_names
        res = GetGroupNamesResponse()
        res.group_names = list(group_names)
        return res

    @profile
    def get_group_info_cb(self, req: GetGroupInfoRequest) -> GetGroupInfoResponse:
        res = GetGroupInfoResponse()
        res.error_codes = GetGroupInfoResponse.SUCCESS
        try:
            group = god_map.world.groups[req.group_name]  # type: WorldBranch
            res.controlled_joints = [str(j.short_name) for j in group.controlled_joints]
            res.links = list(sorted(str(x.short_name) for x in group.link_names_as_set))
            res.child_groups = list(sorted(str(x) for x in group.groups.keys()))
            res.root_link_pose.pose = group.base_pose
            res.root_link_pose.header.frame_id = str(god_map.world.root_link_name)
            for key, value in group.state.items():
                res.joint_state.name.append(str(key))
                res.joint_state.position.append(value.position)
                res.joint_state.velocity.append(value.velocity)
        except KeyError as e:
            logging.logerr(f'no object with the name {req.group_name} was found')
            res.error_codes = GetGroupInfoResponse.GROUP_NOT_FOUND_ERROR

        return res

    @record_time
    @profile
    def update(self):
        try:
            if self.timer_state == self.STALL:
                self.timer_state = self.READY
                return Status.SUCCESS
            if self.service_in_use.empty():
                return Status.SUCCESS
            else:
                if self.timer_state == self.READY:
                    self.timer_state = self.BUSY
                    self.work_permit.put(1)
            return Status.RUNNING
        finally:
            if self.update_ticked.empty():
                self.update_ticked.put(1)

    @profile
    def update_world_cb(self, req: UpdateWorldRequest) -> UpdateWorldResponse:
        """
        Callback function of the ROS service to update the internal giskard world.
        :param req: Service request as received from the service client.
        :return: Service response, reporting back any runtime errors that occurred.
        """
        self.service_in_use.put('muh')
        try:
            # make sure update had a chance to add a work permit
            self.update_ticked.get()
            # calling this twice, because it may still have a tick from the prev update call
            self.update_ticked.get()
            self.work_permit.get(timeout=req.timeout)
            try:
                if req.operation == UpdateWorldRequest.ADD:
                    self.add_object(req)
                elif req.operation == UpdateWorldRequest.UPDATE_PARENT_LINK:
                    self.update_parent_link(req)
                elif req.operation == UpdateWorldRequest.UPDATE_POSE:
                    self.update_group_pose(req)
                elif req.operation == UpdateWorldRequest.REMOVE:
                    self.remove_object(req.group_name)
                elif req.operation == UpdateWorldRequest.REMOVE_ALL:
                    self.clear_world()
                else:
                    return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                               f'Received invalid operation code: {req.operation}')
                return UpdateWorldResponse()
            except Exception as e:
                return exception_to_response(e, req)
        except Exception as e:
            response = UpdateWorldResponse()
            response.error_codes = UpdateWorldResponse.BUSY
            logging.logwarn('Rejected world update because Giskard is busy.')
            return response
        finally:
            self.timer_state = self.STALL
            self.service_in_use.get_nowait()
            self.clear_markers()

    @profile
    def add_object(self, req: UpdateWorldRequest):
        req.parent_link = god_map.world.search_for_link_name(req.parent_link, req.parent_link_group)
        world_body = req.body
        if req.pose.header.frame_id == '':
            raise TransformException('Frame_id in pose is not set.')
        try:
            global_pose = transform_pose(target_frame=god_map.world.root_link_name, pose=req.pose, timeout=0.5)
        except:
            req.pose.header.frame_id = god_map.world.search_for_link_name(req.pose.header.frame_id)
            global_pose = god_map.world.transform_msg(god_map.world.root_link_name, req.pose)

        global_pose = god_map.world.transform_pose(req.parent_link, global_pose).pose
        god_map.world.add_world_body(group_name=req.group_name,
                                     msg=world_body,
                                     pose=global_pose,
                                     parent_link_name=req.parent_link)
        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        logging.loginfo(f'Attached object \'{req.group_name}\' at \'{req.parent_link}\'.')
        if world_body.joint_state_topic:
            god_map.tree.wait_for_goal.synchronization.sync_joint_state_topic(
                group_name=req.group_name,
                topic_name=world_body.joint_state_topic)
        # FIXME also keep track of base pose
        if world_body.tf_root_link_name:
            raise NotImplementedError('tf_root_link_name is not implemented')
        parent_group = god_map.world.get_parent_group_name(req.group_name)
        new_links = god_map.world.groups[req.group_name].link_names_with_collisions
        god_map.collision_scene.update_self_collision_matrix(parent_group, new_links)
        god_map.collision_scene.blacklist_inter_group_collisions()

    @profile
    def update_group_pose(self, req: UpdateWorldRequest):
        if req.group_name not in god_map.world.groups:
            raise UnknownGroupException(f'Can\'t update pose of unknown group: \'{req.group_name}\'')
        group = god_map.world.groups[req.group_name]
        joint_name = group.root_link.parent_joint_name
        pose = god_map.world.transform_pose(god_map.world.joints[joint_name].parent_link_name, req.pose).pose
        god_map.world.joints[joint_name].update_transform(pose)
        god_map.world.notify_state_change()
        god_map.collision_scene.remove_links_from_self_collision_matrix(set(group.link_names_with_collisions))
        god_map.collision_scene.update_collision_blacklist(
            link_combinations=set(product(group.link_names_with_collisions,
                                          god_map.world.link_names_with_collisions)))

    @profile
    def update_parent_link(self, req: UpdateWorldRequest):
        req.parent_link = god_map.world.search_for_link_name(link_name=req.parent_link,
                                                             group_name=req.parent_link_group)
        if req.group_name not in god_map.world.groups:
            raise UnknownGroupException(f'Can\'t attach to unknown group: \'{req.group_name}\'')
        group = god_map.world.groups[req.group_name]
        if group.root_link_name != req.parent_link:
            old_parent_link = group.parent_link_of_root
            god_map.world.move_group(req.group_name, req.parent_link)
            logging.loginfo(f'Reattached \'{req.group_name}\' from \'{old_parent_link}\' to \'{req.parent_link}\'.')
            parent_group = god_map.world.get_parent_group_name(req.group_name)
            new_links = god_map.world.groups[req.group_name].link_names_with_collisions
            god_map.collision_scene.remove_links_from_self_collision_matrix(new_links)
            god_map.collision_scene.update_self_collision_matrix(parent_group, new_links)
            god_map.collision_scene.blacklist_inter_group_collisions()
        else:
            logging.logwarn(f'Didn\'t update world. \'{req.group_name}\' is already attached to \'{req.parent_link}\'.')

    @profile
    def remove_object(self, name: str):
        if name not in god_map.world.groups:
            raise UnknownGroupException(f'Can not remove unknown group: {name}.')
        god_map.world.delete_group(name)
        god_map.world.cleanup_unused_free_variable()
        god_map.tree.wait_for_goal.synchronization.remove_group_behaviors(name)
        logging.loginfo(f'Deleted \'{name}\'.')

    @profile
    def clear_world(self):
        tmp_state = deepcopy(god_map.world.state)
        god_map.world.delete_all_but_robots()
        god_map.tree.wait_for_goal.synchronization.remove_added_behaviors()
        # copy only state of joints that didn't get deleted
        remaining_free_variables = list(god_map.world.free_variables.keys()) + list(
            god_map.world.virtual_free_variables.keys())
        god_map.world.state = JointStates({k: v for k, v in tmp_state.items() if k in remaining_free_variables})
        god_map.world.notify_state_change()
        god_map.collision_scene.sync()
        god_map.collision_avoidance_config.setup()
        self.clear_markers()
        logging.loginfo('Cleared world.')

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_publisher.publish(msg)
