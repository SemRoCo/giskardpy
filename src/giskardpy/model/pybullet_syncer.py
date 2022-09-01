import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import MarkerArray, Marker

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy.data_types import BiDict, CollisionAABB
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, Collisions, Collision
from giskardpy.model.pybullet_wrapper import ContactInfo
from giskardpy.utils.tfwrapper import pose_to_list
from giskardpy.utils.utils import resolve_ros_iris, write_to_tmp


class PyBulletSyncer(CollisionWorldSynchronizer):
    hack_name = 'hack'

    def __init__(self, world, gui=False):
        super(PyBulletSyncer, self).__init__(world)
        # pbw.start_pybullet(self.god_map.get_data(identifier.gui))
        self.client_id = pbw.start_pybullet(gui)
        pbw.deactivate_rendering(client_id=self.client_id)
        self.object_name_to_bullet_id = BiDict()

    @profile
    def add_object(self, link):
        """
        :type link: giskardpy.model.world.Link
        """
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        self.object_name_to_bullet_id[link.name] = pbw.load_urdf_string_into_bullet(link.as_urdf(),
                                                                                    position=position,
                                                                                    orientation=orientation,
                                                                                    client_id=self.client_id)

    @profile
    def update_pose(self, link):
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[link.name], position, orientation,
                                            physicsClientId=self.client_id)

    def check_collisions2(self, link_combinations, distance):
        in_collision = set()
        self.sync()
        for link_a, link_b in link_combinations:
            if self.in_collision(link_a, link_b, distance):
                in_collision.add((link_a, link_b))
        return in_collision

    @profile
    def check_collisions(self, cut_off_distances, collision_list_size=15, buffer=0.05):
        """
        :param cut_off_distances: (robot_link, body_b, link_b) -> cut off distance. Contacts between objects not in this
                                    dict or further away than the cut off distance will be ignored.
        :type cut_off_distances: dict
        :param self_collision_d: distances grater than this value will be ignored
        :type self_collision_d: float
        :type enable_self_collision: bool
        :return: (robot_link, body_b, link_b) -> Collision
        :rtype: Collisions
        """
        collisions = Collisions(self.world.god_map, collision_list_size)
        for (link_a, link_b), distance in cut_off_distances.items():
            link_b_id = self.object_name_to_bullet_id[link_b]
            link_a_id = self.object_name_to_bullet_id[link_a]
            contacts = [ContactInfo(*x) for x in pbw.getClosestPoints(link_a_id, link_b_id,
                                                                      distance+buffer, physicsClientId=self.client_id)]
            if len(contacts) > 0:
                for contact in contacts:  # type: ContactInfo
                    map_P_pa = contact.position_on_a
                    map_P_pb = contact.position_on_b
                    map_V_n = contact.contact_normal_on_b
                    collision = Collision(link_a=link_a,
                                          link_b=link_b,
                                          map_P_pa=map_P_pa,
                                          map_P_pb=map_P_pb,
                                          map_V_n=map_V_n,
                                          contact_distance=contact.contact_distance)
                    if self.__should_flip_collision(map_P_pa, link_a):
                        collision = Collision(link_a=link_a,
                                              link_b=link_b,
                                              map_P_pa=map_P_pb,
                                              map_P_pb=map_P_pa,
                                              map_V_n=tuple([-x for x in map_V_n]),
                                              contact_distance=contact.contact_distance)
                    collisions.add(collision)
        return collisions

    def in_collision(self, link_a, link_b, distance):
        link_id_a = self.object_name_to_bullet_id[link_a]
        link_id_b = self.object_name_to_bullet_id[link_b]
        return len(pbw.getClosestPoints(link_id_a, link_id_b, distance, physicsClientId=self.client_id)) > 0

    @profile
    def sync(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        # logging.logwarn(self.world.version)
        # if self.world.model_version == 6:
        #     traceback.print_stack()
        if self.has_world_changed():
            self.object_name_to_bullet_id = BiDict()
            pbw.clear_pybullet(client_id=self.client_id)
            self.fks = self.world.compute_all_fks()
            for link_name, link in self.world.links.items():
                if link.has_collisions():
                    self.add_object(link)
            # logging.logwarn('synced world')
        else:
            # logging.logwarn('updated world')
            self.fks = self.world.compute_all_fks()
            for link_name, link in self.world.links.items():
                if link.has_collisions():
                    self.update_pose(link)

    #def get_pose(self, link_name):
    #    map_T_link = PoseStamped()
    #    position, orientation = pbw.getBasePositionAndOrientation(self.object_name_to_bullet_id[link_name],
    #                                                              physicsClientId=self.client_id)
    #    map_T_link.header.frame_id = self.world.root_link_name
    #    map_T_link.pose.position = Point(*position)
    #    map_T_link.pose.orientation = Quaternion(*orientation)
    #    return map_T_link

    def get_aabb_info(self, link_name):
        if self.world.has_link_collisions(link_name):
            link_id = self.object_name_to_bullet_id[link_name]
            aabbs = pbw.p.getAABB(link_id, physicsClientId=self.client_id)
            return CollisionAABB(link_name, aabbs[0], aabbs[1])

    def __add_pybullet_bug_fix_hack(self):
        if self.hack_name not in self.object_name_to_bullet_id:
            path = resolve_ros_iris('package://giskardpy/urdfs/tiny_ball.urdf')
            with open(path, 'r') as f:
                self.object_name_to_bullet_id[self.hack_name] = pbw.load_urdf_string_into_bullet(f.read(),
                                                                                                 client_id=self.client_id)

    def __move_hack(self, pose):
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[self.hack_name],
                                            position, orientation, physicsClientId=self.client_id)

    def __should_flip_collision(self, position_on_a_in_map, link_a):
        """
        :type collision: ContactInfo
        :rtype: bool
        """
        self.__add_pybullet_bug_fix_hack()
        new_p = Pose()
        new_p.position = Point(*position_on_a_in_map)
        new_p.orientation.w = 1

        self.__move_hack(new_p)
        hack_id = self.object_name_to_bullet_id[self.hack_name]
        body_a_id = self.object_name_to_bullet_id[link_a]
        try:
            contact_info3 = ContactInfo(
                *[x for x in pbw.getClosestPoints(hack_id,
                                                  body_a_id, 0.001,
                                                  physicsClientId=self.client_id) if
                  abs(x[8] + 0.005) < 0.0005][0])
            return not contact_info3.body_unique_id_b == body_a_id
        except Exception as e:
            return True


class PyBulletMotionValidationIDs():

    def __init__(self, collision_scene, environment_name='kitchen', environment_object_names=None, moving_links=None):
        self.collision_scene = collision_scene
        self.collision_matrix = self.collision_scene.update_collision_environment()
        self.client_id = collision_scene.client_id

        # Save objects for with collision should be checked from ...
        # ... the environment.
        self.environment_ids = list()
        self.environment_group = environment_name

        # ... other objects.
        self.environment_object_ids = list()
        if environment_object_names is None:
            self.environment_object_groups = list()
        else:
            self.environment_object_groups = environment_object_names

        # Ignore collisions with these objects
        self.ignore_object_ids = list()
        if moving_links is None:
            self.moving_links = list()
        else:
            self.moving_links = moving_links

        self.setup()

    def setup(self):
        self.environment_ids = list()
        for e_l_n in self.collision_scene.world.groups[self.environment_group].link_names_with_collisions:
            self.environment_ids.append(self.collision_scene.object_name_to_bullet_id[e_l_n])
        self.environment_object_ids = list()
        self.ignore_object_ids = list()
        for o_g in self.environment_object_groups:
            for o_l_n in self.collision_scene.world.groups[o_g].link_names_with_collisions:
                if o_l_n in self.moving_links:
                    continue
                ignore_object_link = True
                for r_link in self.moving_links:
                    if self.should_ignore_collision(o_l_n, r_link):
                        ignore_object_link = False
                        break
                o_l_id = self.collision_scene.object_name_to_bullet_id[o_l_n]
                if ignore_object_link:
                    self.ignore_object_ids.append(o_l_id)
                else:
                    self.environment_object_ids.append(o_l_id)

    def ignore_id(self, id):
        return id in self.ignore_object_ids or (
                id not in self.environment_object_ids and id not in self.environment_ids and id != -1)

    def should_ignore_collision(self, link1, link2):
        for (robot_link, link_b), _ in self.collision_matrix.items():
            if robot_link == link1 and link_b == link2:
                return False
            elif link_b == link1 and robot_link == link2:
                return False
        return True

    def close_pybullet(self):
        pass

    def clear(self):
        pass


class PyBulletRayTester():

    def __init__(self, pybulletenv):
        self.pybulletenv = pybulletenv
        self.once = False
        self.link_id_start = -1
        self.collision_free_id = -1
        self.collisionFilterGroup = 0x1
        self.noCollisionFilterGroup = 0x0

    def clear(self):
        self.pybulletenv.close_pybullet()

    def pre_ray_test(self):
        bodies_num = pbw.p.getNumBodies(physicsClientId=self.pybulletenv.client_id)
        if bodies_num > 1:
            for id in range(0, bodies_num):
                links_num = pbw.p.getNumJoints(id, physicsClientId=self.pybulletenv.client_id)
                for link_id in range(self.link_id_start, links_num):
                    if not self.pybulletenv.ignore_id(id):
                        pbw.p.setCollisionFilterGroupMask(id, link_id, self.collisionFilterGroup,
                                                          self.collisionFilterGroup,
                                                          physicsClientId=self.pybulletenv.client_id)
                    else:
                        pbw.p.setCollisionFilterGroupMask(id, link_id, self.noCollisionFilterGroup,
                                                          self.noCollisionFilterGroup,
                                                          physicsClientId=self.pybulletenv.client_id)

    def ray_test_batch(self, rayFromPositions, rayToPositions):
        bodies_num = pbw.p.getNumBodies(physicsClientId=self.pybulletenv.client_id)
        if bodies_num > 1:
            query_res = pbw.p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=0,
                                           physicsClientId=self.pybulletenv.client_id,
                                           collisionFilterMask=self.collisionFilterGroup)
        else:
            query_res = pbw.p.rayTestBatch(rayFromPositions, rayToPositions, numThreads=0,
                                           physicsClientId=self.pybulletenv.client_id)
        if any(v[0] for v in query_res if self.pybulletenv.ignore_id(v[0])):
            rospy.logerr('fak')
        coll_links = []
        dists = []
        for i in range(0, len(query_res)):
            obj_id = query_res[i][0]
            n = query_res[i][-1]
            if obj_id != self.collision_free_id:
                coll_links.append(pbw.p.getBodyInfo(obj_id)[0])
                d = np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
                dists.append(d)
        fractions = [query_res[i][2] for i in range(0, len(query_res))]
        return all([v[0] == self.collision_free_id for v in query_res]), coll_links, dists, fractions

    def post_ray_test(self):
        bodies_num = pbw.p.getNumBodies(physicsClientId=self.pybulletenv.client_id)
        if bodies_num > 1:
            for id in range(0, pbw.p.getNumBodies(physicsClientId=self.pybulletenv.client_id)):
                links_num = pbw.p.getNumJoints(id, physicsClientId=self.pybulletenv.client_id)
                for link_id in range(self.link_id_start, links_num):
                    pbw.p.setCollisionFilterGroupMask(id, link_id, self.collisionFilterGroup, self.collisionFilterGroup,
                                                      physicsClientId=self.pybulletenv.client_id)


class PyBulletBoxSpace():

    def __init__(self, world, robot, map_frame, pybullet_env, publish_collision_boxes=False):
        self.pybullet_env = pybullet_env
        self.world = world
        self.robot = robot
        self.map_frame = map_frame
        self.publish_collision_boxes = publish_collision_boxes

        if self.publish_collision_boxes:
            self.collision_box_name_prefix = 'collision_box_name_prefix'
            self.pub_collision_marker = rospy.Publisher(u'~visualization_marker_array', MarkerArray, queue_size=1)

    def _get_pitch(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([dx, dy, 0])
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_a)) ** 2))
        return np.arctan2(dz, a)

    def get_pitch(self, pos_b):
        l_a = self._get_pitch(pos_b)
        if pos_b[0] >= 0:
            return -l_a
        else:
            return np.pi - l_a

    def _get_yaw(self, pos_b):
        pos_a = np.zeros(3)
        dx = pos_b[0] - pos_a[0]
        dz = pos_b[2] - pos_a[2]
        pos_c = pos_a + np.array([0, 0, dz])
        pos_d = pos_c + np.array([dx, 0, 0])
        g = np.sqrt(np.sum((np.array(pos_b) - np.array(pos_d)) ** 2))
        a = np.sqrt(np.sum((np.array(pos_c) - np.array(pos_d)) ** 2))
        return np.arctan2(g, a)

    def get_yaw(self, pos_b):
        l_a = self._get_yaw(pos_b)
        if pos_b[0] >= 0 and pos_b[1] >= 0:
            return l_a
        elif pos_b[0] >= 0 and pos_b[1] <= 0:
            return -l_a
        elif pos_b[0] <= 0 and pos_b[1] >= 0:
            return np.pi - l_a
        else:
            return np.pi + l_a

    def compute_pose_of_box(self, pos_a, pos_b):
        b_to_a = np.array(pos_a) - np.array(pos_b)
        c = np.array(pos_b) + b_to_a / 2.
        # Compute the pitch and yaw based on the pybullet box and map coordinates
        # https://i.stack.imgur.com/f190Q.png, https://stackoverflow.com/questions/58469297/how-do-i-calculate-the-yaw-pitch-and-roll-of-a-point-in-3d/58469298#58469298
        q = quaternion_from_euler(0, self.get_pitch(pos_b - c), self.get_yaw(pos_b - c))
        # rospy.logerr(u'pitch: {}, yaw: {}'.format(self._get_pitch(pos_a, pos_b), self._get_yaw(pos_a, pos_b)))
        return Pose(Point(c[0], c[1], c[2]), Quaternion(q[0], q[1], q[2], q[3]))

    def is_colliding(self, min_sizes, start_positions, end_positions):
        if self.robot:
            for i, (pos_a, pos_b) in enumerate(zip(start_positions, end_positions)):
                if np.all(pos_a == pos_b):
                    continue
                contact_points = self._check_for_collisions(i, pos_a, pos_b, min_sizes[i])
                if self._is_colliding(contact_points):
                    return True
        return False

    def is_colliding_timed(self, min_sizes, start_positions, end_positions, s1, s2):
        ret = False
        fs = list()
        if self.robot:
            for i, (pos_a, pos_b) in enumerate(zip(start_positions, end_positions)):
                if np.all(pos_a == pos_b):
                    continue
                contact_points = self._check_for_collisions(i, pos_a, pos_b, min_sizes[i])
                r, f = self._is_colliding_timed(s1, s2, contact_points)
                ret = ret or r
                fs.append(f)
        return ret, min(fs) if fs else 0.99

    def _check_for_collisions(self, i, pos_a, pos_b, min_size, max_dist=0.1):
        contact_points = tuple()
        # create box
        pose = self.compute_pose_of_box(pos_a, pos_b)
        length = np.sqrt(np.sum((np.array(pos_a) - np.array(pos_b)) ** 2)) + min_size
        width = min_size
        height = min_size
        coll_id = pbw.create_collision_box(pose_to_list(pose), length, width, height,
                                           client_id=self.pybullet_env.client_id)
        if self.publish_collision_boxes:
            name = '{}_{}'.format(self.collision_box_name_prefix, str(i))
            self.pub_marker(length, width, height, pose, name)
        for environment_id in self.pybullet_env.environment_ids:
            contact_points += pbw.getClosestPoints(environment_id, coll_id, max_dist,
                                                   physicsClientId=self.pybullet_env.client_id)
        for obj_id in self.pybullet_env.environment_object_ids:
            contact_points += pbw.getClosestPoints(obj_id, coll_id, max_dist,
                                                   physicsClientId=self.pybullet_env.client_id)
        if self.publish_collision_boxes:
            self.del_marker(name)
        pbw.removeBody(coll_id, physicsClientId=self.pybullet_env.client_id)
        return contact_points

    def _is_colliding(self, contact_points):
        for c in contact_points:
            if c[8] < 0.0:
                return True

    def get_normal_v(self, p, a, b):
        ap = p - a
        ab = b - a
        ab = ab / np.linalg.norm(ab)
        ab = ab * (np.dot(ap.T, ab))
        normal = a + ab
        if np.all(normal < np.min([a, b], axis=0)):
            return np.array([0, 0, 0])
        elif np.all(normal >= np.max([a, b], axis=0)):
            return b - a
        else:
            return ab

    def _is_colliding_timed(self, s1, s2, contact_points):
        fs = list()
        dist = np.linalg.norm(s1-s2) + 0.001
        for c in contact_points:
            if c[8] < 0.0:
                a = c[5]
                b = c[6]
                f_a = np.linalg.norm(self.get_normal_v(a, s1, s2))/dist
                f_b = np.linalg.norm(self.get_normal_v(b, s1, s2))/dist
                fs.append(min([f_a, f_b]))
        return len(fs) != 0, min(fs) if fs else 0.99

    def pub_marker(self, d, w, h, p, name):
        ma = MarkerArray()
        marker = Marker()
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.pose = p
        marker.type = Marker.CUBE
        marker.scale.x = d
        marker.scale.y = w
        marker.scale.z = h
        marker.header.frame_id = self.map_frame
        marker.ns = u'world' + name
        ma.markers.append(marker)
        self.pub_collision_marker.publish(ma)

    def del_marker(self, name):
        ma = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETE
        marker.header.frame_id = self.map_frame
        marker.ns = u'world' + name
        ma.markers.append(marker)
        self.pub_collision_marker.publish(ma)
