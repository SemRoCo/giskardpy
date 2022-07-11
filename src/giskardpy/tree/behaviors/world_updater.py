import traceback
from itertools import product
from queue import Queue
from xml.etree.ElementTree import ParseError

import rospy
from py_trees import Status
from py_trees.meta import running_is_success
from tf2_py import TransformException
from visualization_msgs.msg import MarkerArray, Marker

import giskardpy.casadi_wrapper as w
import giskardpy.identifier as identifier
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest, GetGroupNamesResponse, \
    GetGroupNamesRequest, RegisterGroupRequest, RegisterGroupResponse, \
    GetGroupInfoResponse, GetGroupInfoRequest, DyeGroupResponse, GetGroupNames, GetGroupInfo, RegisterGroup, DyeGroup
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import CorruptShapeException, UnknownGroupException, \
    UnsupportedOptionException, DuplicateNameException, UnknownLinkException
from giskardpy.model.world import SubWorldTree
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.behaviors.sync_configuration import SyncConfiguration
from giskardpy.tree.behaviors.sync_localization import SyncLocalization
from giskardpy.utils import logging
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

    # TODO reject changes if plugin not active or something
    def __init__(self, name: str):
        self.added_plugin_names = []
        super(WorldUpdater, self).__init__(name)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.original_link_names = self.robot.link_names
        self.service_in_use = Queue(maxsize=1)
        self.work_permit = Queue(maxsize=1)
        self.update_ticked = Queue(maxsize=1)
        self.timer_state = self.READY

    @profile
    def setup(self, timeout: float = 5.0):
        self.marker_publisher = rospy.Publisher('~visualization_marker_array', MarkerArray, queue_size=1)
        self.srv_update_world = rospy.Service('~update_world', UpdateWorld, self.update_world_cb)
        self.get_group_names = rospy.Service('~get_group_names', GetGroupNames, self.get_group_names_cb)
        self.get_group_info = rospy.Service('~get_group_info', GetGroupInfo, self.get_group_info_cb)
        self.register_groups = rospy.Service('~register_groups', RegisterGroup, self.register_groups_cb)
        self.dye_group = rospy.Service('~dye_group', DyeGroup, self.dye_group)
        # self.dump_state_srv = rospy.Service('~dump_state', Trigger, self.dump_state_cb)
        return super(WorldUpdater, self).setup(timeout)

    def dye_group(self, req):
        group_name = req.group_name
        res = DyeGroupResponse()
        if group_name in self.world.groups:
            for _, link in self.world.groups[req.group_name].links.items():
                link.dye_collisions(req.color)
            res.error_codes = DyeGroupResponse.SUCCESS
        else:
            res.error_codes = DyeGroupResponse.GROUP_NOT_FOUND_ERROR
        return res

    def register_groups_cb(self, req: RegisterGroupRequest) -> RegisterGroupResponse:
        link_name = self.world.groups[req.parent_group_name].get_link_short_name_match(req.root_link_name)
        self.world.register_group(req.group_name, link_name)
        res = RegisterGroupResponse()
        res.error_codes = res.SUCCESS
        return res

    def get_group_names_cb(self, req: GetGroupNamesRequest) -> GetGroupNamesResponse:
        group_names = self.world.group_names
        res = GetGroupNamesResponse()
        res.group_names = group_names
        return res

    def get_group_info_cb(self, req: GetGroupInfoRequest) -> GetGroupInfoResponse:
        res = GetGroupInfoResponse()
        res.error_codes = GetGroupInfoResponse.SUCCESS
        try:
            group = self.world.groups[req.group_name]  # type: SubWorldTree
            res.controlled_joints = group.controlled_joints
            res.links = list(sorted(str(x) for x in group.link_names))
            res.child_groups = list(group.groups.keys())
            # tree = self.god_map.unsafe_get_data(identifier.tree_manager)  # type: TreeManager
            # node_name = str(PrefixName(req.group_name, 'js'))
            # if node_name in tree.tree_nodes:
            #     res.joint_state_topic = tree.tree_nodes[node_name].node.joint_state_topic
            res.root_link_pose.pose = group.base_pose
            res.root_link_pose.header.frame_id = self.get_god_map().get_data(identifier.map_frame)
            for key, value in group.state.items():
                res.joint_state.name.append(str(key))
                res.joint_state.position.append(value.position)
                res.joint_state.velocity.append(value.velocity)
        except KeyError as e:
            logging.logerr('no object with the name {} was found'.format(req.group_name))
            res.error_codes = GetGroupInfoResponse.GROUP_NOT_FOUND_ERROR

        return res

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

    def update_world_cb(self, req):
        """
        Callback function of the ROS service to update the internal giskard world.
        :param req: Service request as received from the service client.
        :type req: UpdateWorldRequest
        :return: Service response, reporting back any runtime errors that occurred.
        :rtype UpdateWorldResponse
        """
        self.service_in_use.put('muh')
        try:
            # make sure update had a chance to add a work permit
            self.update_ticked.get()
            # calling this twice, because it may still have a tick from the prev update call
            self.update_ticked.get()
            self.work_permit.get(timeout=req.timeout)
            with self.get_god_map():
                self.clear_markers()
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

    def handle_convention(self, req: UpdateWorldRequest):
        # default to world root if all is empty
        if req.parent_link_group == '' and req.parent_link == '':
            req.parent_link = self.world.root_link_name
        # default to robot group, if parent link name is not empty
        else:
            if req.parent_link_group == '':
                req.parent_link_group = self.god_map.unsafe_get_data(identifier.robot_group_name)
            elif req.parent_link == '':
                req.parent_link = self.world.groups[req.parent_link_group].root_link_name
            req.parent_link = self.world.groups[req.parent_link_group].get_link_short_name_match(req.parent_link)
        return req

    def add_object(self, req: UpdateWorldRequest):
        # assumes that parent has god map lock
        req = self.handle_convention(req)
        world_body = req.body
        global_pose = transform_pose(self.world.root_link_name, req.pose)
        global_pose = self.world.transform_pose(req.parent_link, global_pose).pose
        self.world.add_world_body(group_name=req.group_name,
                                  msg=world_body,
                                  pose=global_pose,
                                  parent_link_name=req.parent_link)
        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        # FIXME also keep track of base pose
        logging.loginfo(f'Added object \'{req.group_name}\' at \'{req.parent_link_group}/{req.parent_link}\'.')
        if world_body.joint_state_topic:
            plugin_name = str(PrefixName(req.group_name, 'js'))
            plugin = running_is_success(SyncConfiguration)(plugin_name,
                                                           group_name=req.group_name,
                                                           joint_state_topic=world_body.joint_state_topic)
            self.tree.insert_node(plugin, 'Synchronize', 1)
            self.added_plugin_names.append(plugin_name)
            logging.loginfo(f'Added configuration plugin for \'{req.group_name}\' to tree.')
        if world_body.tf_root_link_name:
            plugin_name = str(PrefixName(world_body.name, 'localization'))
            plugin = SyncLocalization(plugin_name,
                                      group_name=req.group_name,
                                      tf_root_link_name=world_body.tf_root_link_name)
            self.tree.insert_node(plugin, 'Synchronize', 1)
            self.added_plugin_names.append(plugin_name)
            logging.loginfo(f'Added localization plugin for \'{req.group_name}\' to tree.')
        parent_group = self.world.get_parent_group_name(req.group_name)
        self.collision_scene.update_group_blacklist(parent_group)
        self.collision_scene.blacklist_inter_group_collisions()

    def update_group_pose(self, req: UpdateWorldRequest):
        if req.group_name not in self.world.groups:
            raise UnknownGroupException(f'Can\'t update pose of unknown group: \'{req.group_name}\'')
        group = self.world.groups[req.group_name]
        joint_name = group.root_link.parent_joint_name
        pose = self.world.transform_pose(self.world.joints[joint_name].parent_link_name, req.pose).pose
        pose = w.Matrix(msg_to_homogeneous_matrix(pose))
        self.world.update_joint_parent_T_child(joint_name, pose)
        # self.collision_scene.remove_black_list_entries(set(group.link_names_with_collisions))
        # self.collision_scene.update_collision_blacklist(
        #     link_combinations=set(product(group.link_names_with_collisions,
        #                                   self.world.link_names_with_collisions)))

    def update_parent_link(self, req: UpdateWorldRequest):
        # assumes that parent has god map lock
        req = self.handle_convention(req)
        if req.group_name not in self.world.groups:
            raise UnknownGroupException(f'Can\'t attach to unknown group: \'{req.group_name}\'')
        group = self.world.groups[req.group_name]
        if group.root_link_name != req.parent_link:
            old_parent_link = group.parent_link_of_root
            self.world.move_group(req.group_name, req.parent_link)
            logging.loginfo(f'Reattached \'{req.group_name}\' from \'{old_parent_link}\' to \'{req.parent_link}\'.')
            self.collision_scene.remove_black_list_entries(set(group.link_names_with_collisions))
            self.collision_scene.update_collision_blacklist(
                link_combinations=set(product(group.link_names_with_collisions,
                                              self.world.link_names_with_collisions)))
        else:
            logging.logwarn(f'Didn\'t update world. \'{req.group_name}\' is already attached to \'{req.parent_link}\'.')

    def remove_object(self, name):
        # assumes that parent has god map lock
        if name not in self.world.groups:
            raise UnknownGroupException(f'Can not remove unknown group: {name}.')
        self.collision_scene.remove_black_list_entries(set(self.world.groups[name].link_names_with_collisions))
        self.world.delete_group(name)
        self._remove_plugin(str(PrefixName(name, 'js')))
        self._remove_plugin(str(PrefixName(name, 'localization')))
        logging.loginfo('Deleted \'{}\''.format(name))

    def _remove_plugin(self, name):
        tree = self.god_map.unsafe_get_data(identifier.tree_manager)  # type: TreeManager
        if name in tree.tree_nodes:
            tree.remove_node(name)
            self.added_plugin_names.remove(name)

    def clear_world(self):
        # assumes that parent has god map lock
        self.collision_scene.reset_collision_blacklist()
        self.world.delete_all_but_robot()
        for plugin_name in self.added_plugin_names:
            self.tree.remove_node(plugin_name)
        self.added_plugin_names = []
        logging.loginfo('Cleared world.')

    def clear_markers(self):
        msg = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        msg.markers.append(marker)
        self.marker_publisher.publish(msg)

    # def dump_state_cb(self, data):
    #     try:
    #         path = self.get_god_map().unsafe_get_data(identifier.data_folder)
    #         folder_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    #         folder_path = '{}{}'.format(path, folder_name)
    #         os.mkdir(folder_path)
    #         robot = self.unsafe_get_robot()
    #         world = self.unsafe_get_world()
    #         with open("{}/dump.txt".format(folder_path), 'w') as f:
    #             tree_manager = self.get_god_map().unsafe_get_data(identifier.tree_manager)  # type: TreeManager
    #             joint_state_message = tree_manager.get_node('js1').lock.get()
    #             f.write("initial_robot_joint_state_dict = ")
    #             write_dict(to_joint_state_position_dict(joint_state_message), f)
    #             f.write("try:\n" +
    #                     "   x_joint = initial_robot_joint_state_dict[\"odom_x_joint\"]\n" +
    #                     "   y_joint = initial_robot_joint_state_dict[\"odom_y_joint\"]\n" +
    #                     "   z_joint = initial_robot_joint_state_dict[\"odom_z_joint\"]\n" +
    #                     "   base_pose = PoseStamped()\n" +
    #                     "   base_pose.header.frame_id = \"map\"\n" +
    #                     "   base_pose.pose.position = Point(x_joint, y_joint, 0)\n" +
    #                     "   base_pose.pose.orientation = Quaternion(*quaternion_about_axis(z_joint, [0, 0, 1]))\n" +
    #                     "   zero_pose.teleport_base(base_pose)\n" +
    #                     "except:\n" +
    #                     "   logging.loginfo(\'no x,y and z joint\')\n\n")
    #             f.write("zero_pose.send_and_check_joint_goal(initial_robot_joint_state_dict)\n")
    #             robot_base_pose = PoseStamped()
    #             robot_base_pose.header.frame_id = 'map'
    #             robot_base_pose.pose = robot.base_pose
    #             f.write("map_odom_transform_dict = ")
    #             write_dict(convert_ros_message_to_dictionary(robot_base_pose), f)
    #             f.write(
    #                 "map_odom_pose_stamped = convert_dictionary_to_ros_message(\'geometry_msgs/PoseStamped\', map_odom_transform_dict)\n")
    #             f.write("map_odom_transform = Transform()\n" +
    #                     "map_odom_transform.rotation = map_odom_pose_stamped.pose.orientation\n" +
    #                     "map_odom_transform.translation = map_odom_pose_stamped.pose.position\n\n")
    #             f.write(
    #                 "set_odom_map_transform = rospy.ServiceProxy('/map_odom_transform_publisher/update_map_odom_transform', UpdateTransform)\n")
    #             f.write("set_odom_map_transform(map_odom_transform)\n")
    #
    #             original_robot = URDFObject(robot.original_urdf)
    #             link_names = robot.get_link_names()
    #             original_link_names = original_robot.get_link_names()
    #             attached_objects = list(set(link_names).difference(original_link_names))
    #             for object_name in attached_objects:
    #                 parent = robot.get_parent_link_of_joint(object_name)
    #                 pose = robot.compute_fk_pose(parent, object_name)
    #                 world_object = robot.get_sub_tree_at_joint(object_name)
    #                 f.write("#attach {}\n".format(object_name))
    #                 with open("{}/{}.urdf".format(folder_path, object_name), 'w') as f_urdf:
    #                     f_urdf.write(world_object.original_urdf)
    #
    #                 f.write('with open(\'{}/{}.urdf\', \"r\") as f:\n'.format(folder_path, object_name))
    #                 f.write("   {}_urdf = f.read()\n".format(object_name))
    #                 f.write("{0}_name = \"{0}\"\n".format(object_name))
    #                 f.write("{}_pose_stamped_dict = ".format(object_name))
    #                 write_dict(convert_ros_message_to_dictionary(pose), f)
    #                 f.write(
    #                     "{0}_pose_stamped = convert_dictionary_to_ros_message('geometry_msgs/PoseStamped', {0}_pose_stamped_dict)\n".format(
    #                         object_name))
    #                 f.write(
    #                     "zero_pose.add_urdf(name={0}_name, urdf={0}_urdf, pose={0}_pose_stamped)\n".format(object_name))
    #                 f.write(
    #                     "zero_pose.attach_existing(name={0}_name, frame_id=\'{1}\')\n".format(object_name, parent))
    #
    #             for object_name, world_object in world.get_objects().items():  # type: (str, WorldObject)
    #                 f.write("#add {}\n".format(object_name))
    #                 with open("{}/{}.urdf".format(folder_path, object_name), 'w') as f_urdf:
    #                     f_urdf.write(world_object.original_urdf)
    #
    #                 f.write('with open(\'{}/{}.urdf\', \"r\") as f:\n'.format(folder_path, object_name))
    #                 f.write("   {}_urdf = f.read()\n".format(object_name))
    #                 f.write("{0}_name = \"{0}\"\n".format(object_name))
    #                 f.write("{0}_js_topic = \"{0}_js_topic\"\n".format(object_name))
    #                 f.write("{}_pose_dict = ".format(object_name))
    #                 write_dict(convert_ros_message_to_dictionary(world_object.base_pose), f)
    #                 f.write(
    #                     "{0}_pose = convert_dictionary_to_ros_message('geometry_msgs/Pose', {0}_pose_dict)\n".format(
    #                         object_name))
    #                 f.write("{}_pose_stamped = PoseStamped()\n".format(object_name))
    #                 f.write("{0}_pose_stamped.pose = {0}_pose\n".format(object_name))
    #                 f.write("{0}_pose_stamped.header.frame_id = \"map\"\n".format(object_name))
    #                 f.write(
    #                     "zero_pose.add_urdf(name={0}_name, urdf={0}_urdf, pose={0}_pose_stamped, js_topic={0}_js_topic, set_js_topic=None)\n".format(
    #                         object_name))
    #                 f.write("{}_joint_state = ".format(object_name))
    #                 write_dict(to_joint_state_position_dict((dict_to_joint_states(world_object.joint_state))), f)
    #                 f.write("zero_pose.set_object_joint_state({0}_name, {0}_joint_state)\n\n".format(object_name))
    #
    #             last_goal = self.get_god_map().unsafe_get_data(identifier.next_move_goal)
    #             if last_goal:
    #                 f.write('last_goal_dict = ')
    #                 write_dict(convert_ros_message_to_dictionary(last_goal), f)
    #                 f.write(
    #                     "last_goal_msg = convert_dictionary_to_ros_message('giskard_msgs/MoveCmd', last_goal_dict)\n")
    #                 f.write("last_action_goal = MoveActionGoal()\n")
    #                 f.write("last_move_goal = MoveGoal()\n")
    #                 f.write("last_move_goal.cmd_seq = [last_goal_msg]\n")
    #                 f.write("last_move_goal.type = MoveGoal.PLAN_AND_EXECUTE\n")
    #                 f.write("last_action_goal.goal = last_move_goal\n")
    #                 f.write("zero_pose.send_and_check_goal(goal=last_action_goal)\n")
    #             else:
    #                 f.write('#no goal\n')
    #         logging.loginfo('saved dump to {}'.format(folder_path))
    #     except:
    #         logging.logerr('failed to dump state pls try again')
    #         res = TriggerResponse()
    #         res.message = 'failed to dump state pls try again'
    #         return TriggerResponse()
    #     res = TriggerResponse()
    #     res.success = True
    #     res.message = 'saved dump to {}'.format(folder_path)
    #     return res
