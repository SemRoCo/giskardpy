import traceback
from multiprocessing import Lock
from xml.etree.ElementTree import ParseError

try:
    # Python 2
    from Queue import Empty, Queue
except ImportError:
    # Python 3
    from queue import Queue, Empty

import rospy
from giskard_msgs.srv import UpdateWorld, UpdateWorldResponse, UpdateWorldRequest, GetObjectNames, \
    GetObjectNamesResponse, GetObjectInfo, GetObjectInfoResponse, GetAttachedObjects, GetAttachedObjectsResponse
from py_trees import Status
from tf2_py import InvalidArgumentException, ExtrapolationException, TransformException
from visualization_msgs.msg import MarkerArray, Marker

import giskardpy.identifier as identifier
from giskardpy import RobotPrefix, RobotName
from giskardpy.data_types import PrefixName
from giskardpy.exceptions import CorruptShapeException, UnknownBodyException, \
    UnsupportedOptionException, DuplicateNameException
from giskardpy.model.world import SubWorldTree
from giskardpy.tree.plugin import GiskardBehavior
from giskardpy.tree.plugin_configuration import ConfigurationPlugin
from giskardpy.tree.tree_manager import TreeManager
from giskardpy.utils import logging
from giskardpy.utils.tfwrapper import transform_pose


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
    elif error_in_list(e, [UnknownBodyException, KeyError]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.MISSING_BODY_ERROR, str(e))
    elif error_in_list(e, [DuplicateNameException]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.DUPLICATE_BODY_ERROR, str(e))
    elif error_in_list(e, [UnsupportedOptionException]):
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.UNSUPPORTED_OPTIONS, str(e))
    elif error_in_list(e, [TransformException]):
        return UpdateWorldResponse(UpdateWorldResponse.TF_ERROR, str(e))
    else:
        traceback.print_exc()
        return UpdateWorldResponse(UpdateWorldResponse.ERROR,
                                   u'{}: {}'.format(e.__class__.__name__,
                                                    str(e)))


class WorldUpdater(GiskardBehavior):
    READY = 0
    BUSY = 1
    STALL = 2

    # TODO reject changes if plugin not active or something
    def __init__(self, name):
        self.added_plugin_names = []
        super(WorldUpdater, self).__init__(name)
        self.map_frame = self.get_god_map().get_data(identifier.map_frame)
        self.original_link_names = self.robot.link_names
        self.tree_tick_rate = self.god_map.get_data(identifier.tree_tick_rate) / 2
        self.queue = Queue(maxsize=1)
        self.queue2 = Queue(maxsize=1)
        self.timer_state = self.READY

    def setup(self, timeout=5.0):
        # TODO make service name a parameter
        self.marker_publisher = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)
        self.srv_update_world = rospy.Service(u'~update_world', UpdateWorld, self.update_world_cb)
        self.get_object_names = rospy.Service(u'~get_object_names', GetObjectNames, self.get_object_names)
        self.get_object_info = rospy.Service(u'~get_object_info', GetObjectInfo, self.get_object_info)
        self.get_attached_objects = rospy.Service(u'~get_attached_objects', GetAttachedObjects,
                                                  self.get_attached_objects)
        # self.dump_state_srv = rospy.Service(u'~dump_state', Trigger, self.dump_state_cb)
        return super(WorldUpdater, self).setup(timeout)

    def get_object_names(self, req):
        object_names = self.world.group_names
        res = GetObjectNamesResponse()
        res.object_names = object_names
        return res

    def get_object_info(self, req):
        res = GetObjectInfoResponse()
        res.error_codes = GetObjectInfoResponse.SUCCESS
        try:
            object = self.world.groups[req.object_name]  # type: SubWorldTree
            res.joint_state_topic = ''
            tree = self.god_map.unsafe_get_data(identifier.tree_manager)  # type: TreeManager
            node_name = str(PrefixName(req.object_name, 'js'))
            if node_name in tree.tree_nodes:
                res.joint_state_topic = tree.tree_nodes[node_name].node.joint_state_topic
            res.pose.pose = object.base_pose
            res.pose.header.frame_id = self.get_god_map().get_data(identifier.map_frame)
            for key, value in object.state.items():
                res.joint_state.name.append(str(key))
                res.joint_state.position.append(value.position)
                res.joint_state.velocity.append(value.velocity)
        except KeyError as e:
            logging.logerr('no object with the name {} was found'.format(req.object_name))
            res.error_codes = GetObjectInfoResponse.NAME_NOT_FOUND_ERROR

        return res

    def get_attached_objects(self, req):
        link_names = self.robot.link_names
        attached_links = [str(s) for s in set(link_names).difference(self.original_link_names)]
        attachment_points = []
        res = GetAttachedObjectsResponse()
        res.object_names = attached_links
        res.attachment_points = attachment_points
        return res

    def update(self):
        if self.timer_state == self.STALL:
            self.timer_state = self.READY
            return Status.SUCCESS
        if self.queue.empty():
            return Status.SUCCESS
        else:
            if self.timer_state == self.READY:
                self.timer_state = self.BUSY
                self.queue2.put(1)
        return Status.RUNNING

    def update_world_cb(self, req):
        """
        Callback function of the ROS service to update the internal giskard world.
        :param req: Service request as received from the service client.
        :type req: UpdateWorldRequest
        :return: Service response, reporting back any runtime errors that occurred.
        :rtype UpdateWorldResponse
        """
        self.queue.put('busy')
        try:
            self.queue2.get(timeout=req.timeout)
            with self.get_god_map():
                self.clear_markers()
                try:
                    if req.operation == UpdateWorldRequest.ADD:
                        self.add_object(req)
                    elif req.operation == UpdateWorldRequest.ATTACH:
                        self.attach_object(req)
                    elif req.operation == UpdateWorldRequest.REMOVE:
                        self.remove_object(req.body.name)
                    elif req.operation == UpdateWorldRequest.REMOVE_ALL:
                        self.clear_world()
                    elif req.operation == UpdateWorldRequest.DETACH:
                        self.detach_object(req)
                    else:
                        return UpdateWorldResponse(UpdateWorldResponse.INVALID_OPERATION,
                                                   u'Received invalid operation code: {}'.format(req.operation))
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
            self.queue.get_nowait()

    def add_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        # assumes that parent has god map lock
        world_body = req.body
        global_pose = transform_pose(req.parent_link, req.pose).pose
        self.world.add_world_body(world_body, global_pose, req.parent_link)
        # SUB-CASE: If it is an articulated object, open up a joint state subscriber
        # FIXME also keep track of base pose
        logging.loginfo('Added object \'{}\' at \'{}\'.'.format(req.body.name, req.parent_link))
        if world_body.joint_state_topic:
            plugin_name = str(PrefixName(world_body.name, 'js'))
            plugin = ConfigurationPlugin(plugin_name, prefix=None, joint_state_topic=world_body.joint_state_topic)
            self.tree.insert_node(plugin, 'Synchronize', 1)
            self.added_plugin_names.append(plugin_name)
            logging.loginfo('Added configuration plugin for \'{}\' to tree.'.format(req.body.name))

    def detach_object(self, req):
        # assumes that parent has god map lock
        if req.body.name not in self.world.groups:
            raise UnknownBodyException('Can\'t detach \'{}\' because it doesn\'t exist.'.format(req.body.name))
        req.parent_link = self.world.root_link_name
        self.attach_object(req)

    def attach_object(self, req):
        """
        :type req: UpdateWorldRequest
        """
        # assumes that parent has god map lock
        if req.parent_link not in self.world.link_names:
            raise UnknownBodyException('There is no link named \'{}\'.'.format(req.parent_link))
        if req.body.name not in self.world.groups:
            self.add_object(req)
        elif self.world.groups[req.body.name].root_link_name != req.parent_link:
            old_parent_link = self.world.groups[req.body.name].attachment_link_name
            self.world.move_group(req.body.name, req.parent_link)
            logging.loginfo('Attached \'{}\' from \'{}\' to \'{}\'.'.format(req.body.name,
                                                                            old_parent_link,
                                                                            req.parent_link))
        else:
            logging.logwarn('Didn\'t update world because \'{}\' is already attached to \'{}\'.'.format(req.body.name,
                                                                                                        req.parent_link))

    def remove_object(self, name):
        # assumes that parent has god map lock
        self.world.delete_group(name)
        tree = self.god_map.unsafe_get_data(identifier.tree_manager)  # type: TreeManager
        name = str(PrefixName(name, 'js'))
        if name in tree.tree_nodes:
            tree.remove_node(name)
            self.added_plugin_names.remove(name)
        logging.loginfo('Deleted \'{}\''.format(name))

    def clear_world(self):
        # assumes that parent has god map lock
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
    #         with open("{}/dump.txt".format(folder_path), u'w') as f:
    #             tree_manager = self.get_god_map().unsafe_get_data(identifier.tree_manager)  # type: TreeManager
    #             joint_state_message = tree_manager.get_node(u'js1').lock.get()
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
    #                 with open("{}/{}.urdf".format(folder_path, object_name), u'w') as f_urdf:
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
    #                 with open("{}/{}.urdf".format(folder_path, object_name), u'w') as f_urdf:
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
    #                 f.write(u'last_goal_dict = ')
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
    #                 f.write(u'#no goal\n')
    #         logging.loginfo(u'saved dump to {}'.format(folder_path))
    #     except:
    #         logging.logerr('failed to dump state pls try again')
    #         res = TriggerResponse()
    #         res.message = 'failed to dump state pls try again'
    #         return TriggerResponse()
    #     res = TriggerResponse()
    #     res.success = True
    #     res.message = 'saved dump to {}'.format(folder_path)
    #     return res
