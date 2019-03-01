from giskardpy.utils import resolve_ros_iris_in_urdf
from giskardpy.world_object import WorldObject


class PyBulletWorldObj(WorldObject):
    """
    Keeps track of and offers convenience functions for an urdf object in bullet.
    """
    # TODO maybe merge symengine robot with this class?
    base_link_name = u'base'
    pass
    def __init__(self, urdf, controlled_joints=None, base_pose=None, calc_self_collision_matrix=False,
                 path_to_data_folder=''):
        """
        :type name: str
        :param urdf: Path to URDF file, or content of already loaded URDF file.
        :type urdf: str
        :type base_pose: Transform
        :type calc_self_collision_matrix: bool
        :param path_to_data_folder: where the self collision matrix is stored
        :type path_to_data_folder: str
        """
        super(PyBulletWorldObj, self).__init__(urdf)
        # self.path_to_data_folder = path_to_data_folder + u'collision_matrix/'
        # # self.name = name
        # # TODO i think resolve iris can be removed because I do this at the start
        # self.resolved_urdf = resolve_ros_iris_in_urdf(urdf)
        # self.id = load_urdf_string_into_bullet(self.resolved_urdf, base_pose)
        # self.__sync_with_bullet()
        # self.attached_objects = {}
        # self.controlled_joints = controlled_joints
        # if calc_self_collision_matrix:
        #     if not self.load_self_collision_matrix():
        #         self.init_self_collision_matrix()
        #         self.save_self_collision_matrix()
        # else:
        #     self.self_collision_matrix = set()
    #
    # def init_self_collision_matrix(self):
    #     self.self_collision_matrix = self.calc_collision_matrix(set(combinations(self.get_link_names(), 2)))
    #
    # def load_self_collision_matrix(self):
    #     """
    #     :rtype: bool
    #     """
    #     urdf_hash = hashlib.md5(self.resolved_urdf).hexdigest()
    #     path = self.path_to_data_folder + urdf_hash
    #     if os.path.isfile(path):
    #         with open(path) as f:
    #             self.self_collision_matrix = pickle.load(f)
    #             print(u'loaded self collision matrix {}'.format(urdf_hash))
    #             return True
    #     return False
    #
    # def save_self_collision_matrix(self):
    #     urdf_hash = hashlib.md5(self.resolved_urdf).hexdigest()
    #     path = self.path_to_data_folder + urdf_hash
    #     if not os.path.exists(os.path.dirname(path)):
    #         try:
    #             dir_name = os.path.dirname(path)
    #             if dir_name != u'':
    #                 os.makedirs(dir_name)
    #         except OSError as exc:  # Guard against race condition
    #             if exc.errno != errno.EEXIST:
    #                 raise
    #     with open(path, u'w') as file:
    #         print(u'saved self collision matrix {}'.format(path))
    #         pickle.dump(self.self_collision_matrix, file)
    #
    #
    # def set_joint_state(self, joint_state):
    #     """
    #
    #     :param joint_state:
    #     :type joint_state: dict
    #     :return:
    #     """
    #     for joint_name, singe_joint_state in joint_state.items():
    #         p.resetJointState(self.id, self.joint_name_to_info[joint_name].joint_index, singe_joint_state.position)
    #
    # def set_base_pose(self, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
    #     """
    #     Set base pose in bullet world frame.
    #     :param position:
    #     :type position: list
    #     :param orientation:
    #     :type orientation: list
    #     """
    #     p.resetBasePositionAndOrientation(self.id, position, orientation)
    #
    # # def as_marker_msg(self, ns=u'', id=1):
    # # FIXME
    # #     # TODO put this into util or objects
    # #     parsed_urdf = URDF.from_xml_string(self.get_urdf())
    # #     if len(parsed_urdf.joints) != 0 or len(parsed_urdf.links) != 1:
    # #         raise Exception(u'can\'t attach urdf with joints')
    # #     link = parsed_urdf.links[0]
    # #     m = Marker()
    # #     m.ns = u'{}/{}'.format(ns, self.name)
    # #     m.id = id
    # #     geometry = link.visual.geometry
    # #     if isinstance(geometry, Box):
    # #         m.type = Marker.CUBE
    # #         m.scale = Vector3(*geometry.size)
    # #     elif isinstance(geometry, Sphere):
    # #         m.type = Marker.SPHERE
    # #         m.scale = Vector3(geometry.radius,
    # #                           geometry.radius,
    # #                           geometry.radius)
    # #     elif isinstance(geometry, Cylinder):
    # #         m.type = Marker.CYLINDER
    # #         m.scale = Vector3(geometry.radius,
    # #                           geometry.radius,
    # #                           geometry.length)
    # #     else:
    # #         raise Exception(u'world body type {} can\'t be converted to marker'.format(geometry.__class__.__name__))
    # #     m.color = ColorRGBA(0, 1, 0, 0.8)
    # #     m.frame_locked = True
    # #     return m
    #
    # def get_base_pose(self):
    #     """
    #     Retrieves the current base pose of the robot in the PyBullet world.
    #     :return: Base pose of the robot in the world.
    #     :rtype: Transform
    #     """
    #     #FIXME
    #     [position, orientation] = p.getBasePositionAndOrientation(self.id)
    #     base_pose = PoseStamped()
    #     base_pose.header.frame_id = MAP
    #     base_pose.pose.position = Point(*position)
    #     base_pose.pose.orientation = Quaternion(*orientation)
    #     return base_pose
    #
    # def __sync_with_bullet(self):
    #     """
    #     Syncs joint and link infos with bullet
    #     """
    #     self.joint_id_map = {}
    #     self.link_name_to_id = {}
    #     self.link_id_to_name = {}
    #     self.joint_name_to_info = OrderedDict()
    #     self.joint_id_to_info = OrderedDict()
    #     self.joint_name_to_info[self.base_link_name] = JointInfo(*([-1, self.base_link_name] + [None] * 10 +
    #                                                                [self.base_link_name] + [None] * 4))
    #     self.joint_id_to_info[-1] = JointInfo(*([-1, self.base_link_name] + [None] * 10 +
    #                                             [self.base_link_name] + [None] * 4))
    #     self.link_id_to_name[-1] = self.base_link_name
    #     self.link_name_to_id[self.base_link_name] = -1
    #     for joint_index in range(p.getNumJoints(self.id)):
    #         joint_info = JointInfo(*p.getJointInfo(self.id, joint_index))
    #         self.joint_name_to_info[joint_info.joint_name] = joint_info
    #         self.joint_id_to_info[joint_info.joint_index] = joint_info
    #         self.joint_id_map[joint_index] = joint_info.joint_name
    #         self.joint_id_map[joint_info.joint_name] = joint_index
    #         self.link_name_to_id[joint_info.link_name] = joint_index
    #         self.link_id_to_name[joint_index] = joint_info.link_name
    #
    # def get_self_collision_matrix(self):
    #     """
    #     :return: (link1, link2) -> min allowed distance
    #     """
    #     return self.self_collision_matrix
    #
    # def get_joint_state(self):
    #     mjs = dict()
    #     for joint_info in self.joint_name_to_info.values():
    #         if joint_info.joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
    #             sjs = SingleJointState()
    #             sjs.name = joint_info.joint_name
    #             sjs.position = p.getJointState(self.id, joint_info.joint_index)[0]
    #             mjs[sjs.name] = sjs
    #     return mjs
    #
    # def calc_collision_matrix(self, link_combinations, d=0.05, d2=0.0, num_rnd_tries=1000):
    #     """
    #     :param link_combinations: set with link name tuples
    #     :type link_combinations: set
    #     :param d: distance threshold to detect links that are always in collision
    #     :type d: float
    #     :param d2: distance threshold to find links that are sometimes in collision
    #     :type d2: float
    #     :param num_rnd_tries:
    #     :type num_rnd_tries: int
    #     :return: set of link name tuples which are sometimes in collision.
    #     :rtype: set
    #     """
    #     # TODO computational expansive because of too many collision checks
    #     print(u'calculating self collision matrix')
    #     seed(1337)
    #     always = set()
    #
    #     # find meaningless self-collisions
    #     for link_name_a, link_name_b in link_combinations:
    #         if self.get_parent_link_name(link_name_a) == link_name_b or \
    #                 self.get_parent_link_name(link_name_b) == link_name_a:
    #             # if self.joint_id_to_info[link_name_a].parent_index == link_name_b or \
    #             #         self.joint_id_to_info[link_name_b].parent_index == link_name_a:
    #             always.add((link_name_a, link_name_b))
    #     rest = link_combinations.difference(always)
    #     self.set_joint_state(self.get_zero_joint_state())
    #     always = always.union(self._check_collisions(rest, d))
    #     rest = rest.difference(always)
    #
    #     # find meaningful self-collisions
    #     self.set_joint_state(self.get_min_joint_state())
    #     sometimes = self._check_collisions(rest, d2)
    #     rest = rest.difference(sometimes)
    #     self.set_joint_state(self.get_max_joint_state())
    #     sometimes2 = self._check_collisions(rest, d2)
    #     rest = rest.difference(sometimes2)
    #     sometimes = sometimes.union(sometimes2)
    #     for i in range(num_rnd_tries):
    #         self.set_joint_state(self.get_rnd_joint_state())
    #         sometimes2 = self._check_collisions(rest, d2)
    #         if len(sometimes2) > 0:
    #             rest = rest.difference(sometimes2)
    #             sometimes = sometimes.union(sometimes2)
    #     return sometimes
    #
    # # def get_parent_link_name(self, child_link_name):
    # #     return self.joint_id_to_info[self.__get_pybullet_link_id(child_link_name)].parent_index
    #
    # def get_possible_collisions(self, link):
    #     # TODO speed up by saving this
    #     possible_collisions = set()
    #     for link1, link2 in self.get_self_collision_matrix():
    #         if link == link1:
    #             possible_collisions.add(link2)
    #         elif link == link2:
    #             possible_collisions.add(link1)
    #     return possible_collisions
    #
    # def _check_collisions(self, link_combinations, d):
    #     in_collision = set()
    #     for link_name_1, link_name_2 in link_combinations:
    #         link_id_1 = self.__get_pybullet_link_id(link_name_1)
    #         link_id_2 = self.__get_pybullet_link_id(link_name_2)
    #         if len(p.getClosestPoints(self.id, self.id, d, link_id_1, link_id_2)) > 0:
    #             in_collision.add((link_name_1, link_name_2))
    #     return in_collision
    #
    # def attach_urdf_object(self, urdf_object, parent_link, pose):
    #     super(WorldObject, self).attach_urdf_object(urdf_object, parent_link, pose)
    #     # self._urdf_robot = up.URDF.from_xml_string(self.get_urdf())
    #
    #     # assemble and store URDF string of new link and fixed joint
    #     # self.extend_urdf(urdf, transform, parent_link)
    #
    #     # # remove last robot and load new robot from new URDF
    #     # self.sync_urdf_with_bullet()
    #     #
    #     # # update the collision matrix for the newly attached object
    #     # self.add_self_collision_entries(urdf_object.get_name())
    #     # print(u'object {} attached to {} in pybullet world'.format(urdf_object.get_name(), self.name))
    #
    #
    # def get_zero_joint_state(self):
    #     return self.generate_joint_state(lambda x: 0)
    #
    # def get_max_joint_state(self):
    #     return self.generate_joint_state(lambda x: x.joint_upper_limit)
    #
    # def get_min_joint_state(self):
    #     return self.generate_joint_state(lambda x: x.joint_lower_limit)
    #
    # def get_rnd_joint_state(self):
    #     def f(joint_info):
    #         lower_limit = joint_info.joint_lower_limit
    #         upper_limit = joint_info.joint_upper_limit
    #         if lower_limit is None:
    #             return np.random.random() * np.pi * 2
    #         lower_limit = max(lower_limit, -10)
    #         upper_limit = min(upper_limit, 10)
    #         return (np.random.random() * (upper_limit - lower_limit)) + lower_limit
    #
    #     return self.generate_joint_state(f)
    #
    # def generate_joint_state(self, f):
    #     """
    #     :param f: lambda joint_info: float
    #     :return:
    #     """
    #     js = {}
    #     for joint_name in self.controlled_joints:
    #         joint_info = self.joint_name_to_info[joint_name]
    #         if joint_info.joint_type in [JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_PLANAR, JOINT_SPHERICAL]:
    #             sjs = SingleJointState()
    #             sjs.name = joint_name
    #             sjs.position = f(joint_info)
    #             js[joint_name] = sjs
    #     return js
    #
    # def __get_pybullet_link_id(self, link_name):
    #     """
    #     :type link_name: str
    #     :rtype: int
    #     """
    #     return self.link_name_to_id[link_name]
    #
    #
    # # def has_attached_object(self, object_name):
    # #     """
    # #     Checks whether an object with this name has already been attached to the robot.
    # #     :type object_name: str
    # #     :rtype: bool
    # #     """
    # #     return object_name in self.attached_objects.keys()
    #
    # # def attach_object(self, object_, parent_link_name, transform):
    # #     """
    # #     Rigidly attach another object to the robot.
    # #     :param object_: Object that shall be attached to the robot.
    # #     :type object_: UrdfObject
    # #     :param parent_link_name: Name of the link to which the object shall be attached.
    # #     :type parent_link_name: str
    # #     :param transform: Hom. transform between the reference frames of the parent link and the object.
    # #     :type transform: Transform
    # #     """
    # #FIXME
    # #     # TODO should only be called through world because this class does not know which objects exist
    # #     if self.has_attached_object(object_.name):
    # #         # TODO: choose better exception type
    # #         raise DuplicateNameException(
    # #             u'An object \'{}\' has already been attached to the robot.'.format(object_.name))
    # #
    # #     # assemble and store URDF string of new link and fixed joint
    # #     self.extend_urdf(object_, transform, parent_link_name)
    # #
    # #     # remove last robot and load new robot from new URDF
    # #     self.sync_urdf_with_bullet()
    # #
    # #     # update the collision matrix for the newly attached object
    # #     self.add_self_collision_entries(object_.name)
    # #     print(u'object {} attached to {} in pybullet world'.format(object_.name, self.name))
    #
    # def add_self_collision_entries(self, object_name):
    #     link_pairs = {(object_name, link_name) for link_name in self.get_link_names()}
    #     link_pairs.remove((object_name, object_name))
    #     self_collision_with_object = self.calc_collision_matrix(link_pairs)
    #     self.self_collision_matrix.update(self_collision_with_object)
    #
    # # def remove_self_collision_entries(self, object_name):
    # #     self.self_collision_matrix = {(link1, link2) for link1, link2 in self.get_self_collision_matrix()
    # #                                   if link1 != object_name and link2 != object_name}
    #
    # # def extend_urdf(self, object_, transform, parent_link_name):
    # #     new_joint = FixedJoint(u'{}_joint'.format(object_.name), transform, parent_link_name,
    # #                            object_.name)
    # #     self.attached_objects[object_.name] = u'{}{}'.format(to_urdf_string(new_joint),
    # #                                                          remove_outer_tag(object_.get_urdf()))
    #
    # def sync_urdf_with_bullet(self):
    #     joint_state = self.get_joint_state()
    #     base_pose = self.get_base_pose()
    #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    #     p.removeBody(self.id)
    #     self.id = load_urdf_string_into_bullet(self.get_urdf(), base_pose)
    #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    #     self.__sync_with_bullet()
    #     self.set_joint_state(joint_state)
    #
    # # def get_urdf(self):
    # #     """
    # #     :rtype: str
    # #     """
    # #     # for each attached object, insert the corresponding URDF sub-string into the original URDF string
    # #     new_urdf_string = self.resolved_urdf
    # #     for sub_string in self.attached_objects.values():
    # #         new_urdf_string = new_urdf_string.replace(u'</robot>', u'{}</robot>'.format(sub_string))
    # #     return new_urdf_string
    #
    # # def as_urdf_object(self):
    # #     return urdfob
    # #
    # # def detach_object(self, object_name):
    # #     """
    # #     Detaches an attached object from the robot.
    # #     :param object_name: Name of the object that shall be detached from the robot.
    # #     :type object_name: str
    # #     """
    # #FIXME
    # #     if not self.has_attached_object(object_name):
    # #         # TODO: choose better exception type
    # #         raise RuntimeError(u"No object '{}' has been attached to the robot.".format(object_name))
    # #
    # #     del (self.attached_objects[object_name])
    # #
    # #     self.sync_urdf_with_bullet()
    # #     self.remove_self_collision_entries(object_name)
    # #     print(u'object {} detachted from {} in pybullet world'.format(object_name, self.name))
    #
    # # def detach_all_objects(self):
    # #     """
    # #     Detaches all object that have been attached to the robot.
    # #     """
    # #FIXME
    # #     if self.attached_objects:
    # #         for attached_object in self.attached_objects:
    # #             self.remove_self_collision_entries(attached_object)
    # #         self.attached_objects = {}
    # #         self.sync_urdf_with_bullet()
    #
    # # def __str__(self):
    # #     return u'{}/{}'.format(self.name, self.id)
