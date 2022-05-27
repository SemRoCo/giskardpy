import traceback

import numpy as np
import rospy
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy import identifier, RobotName
from giskardpy.data_types import BiDict, Collisions, Collision, CollisionAABB
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.pybullet_wrapper import ContactInfo
from giskardpy.model.world import WorldTree
from giskardpy.utils import logging
from giskardpy.utils.utils import resolve_ros_iris


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
    def check_collisions(self, cut_off_distances, collision_list_size=15):
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
        collisions = Collisions(self.world, collision_list_size)
        for (robot_link, body_b, link_b), distance in cut_off_distances.items():
            link_b_id = self.object_name_to_bullet_id[link_b]
            robot_link_id = self.object_name_to_bullet_id[robot_link]
            contacts = [ContactInfo(*x) for x in pbw.getClosestPoints(robot_link_id, link_b_id,
                                                                      distance * 1.1, physicsClientId=self.client_id)]
            if len(contacts) > 0:
                for contact in contacts:  # type: ContactInfo
                    collision = Collision(link_a=robot_link,
                                          body_b=body_b,
                                          link_b=link_b,
                                          map_P_pa=contact.position_on_a,
                                          map_P_pb=contact.position_on_b,
                                          map_V_n=contact.contact_normal_on_b,
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
            self.world.fast_all_fks = None
            self.fks = self.world.compute_all_fks()
            for link_name, link in self.world.links.items():
                if link.has_collisions():
                    self.add_object(link)
            self.init_collision_matrix(RobotName)
            # logging.logwarn('synced world')
        else:
            # logging.logwarn('updated world')
            try:
                self.fks = self.world.compute_all_fks()
            except:
                self.world.fast_all_fks = None
                self.fks = self.world.compute_all_fks()
            for link_name, link in self.world.links.items():
                if link.has_collisions():
                    self.update_pose(link)

    def get_pose(self, link_name):
        map_T_link = PoseStamped()
        position, orientation = pbw.getBasePositionAndOrientation(self.object_name_to_bullet_id[link_name],
                                                                  physicsClientId=self.client_id)
        map_T_link.header.frame_id = self.world.root_link_name
        map_T_link.pose.position = Point(*position)
        map_T_link.pose.orientation = Quaternion(*orientation)
        return map_T_link

    def get_aabb_info(self, link_name):
        if self.world.has_link_collisions(link_name):
            link_id = self.object_name_to_bullet_id[link_name]
            aabbs = pbw.p.getAABB(link_id, physicsClientId=self.client_id)
            return CollisionAABB(link_name, aabbs[0], aabbs[1])

    def __add_pybullet_bug_fix_hack(self):
        if self.hack_name not in self.object_name_to_bullet_id:
            path = resolve_ros_iris(u'package://giskardpy/urdfs/tiny_ball.urdf')
            with open(path, 'r') as f:
                self.object_name_to_bullet_id[self.hack_name] = pbw.load_urdf_string_into_bullet(f.read(),
                                                                                                 client_id=self.client_id)

    def __move_hack(self, pose):
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[self.hack_name],
                                            position, orientation, client_id=self.client_id)

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


class PyBulletRayTesterEnv():

    def __init__(self, collision_scene, environment_name='kitchen', environment_object_names=None,
                 ignore_objects_ids=None):
        self.collision_scene = collision_scene
        self.client_id = collision_scene.client_id
        if ignore_objects_ids is None:
            self.ignore_object_ids = list()
        else:
            self.ignore_object_ids = ignore_objects_ids
        if environment_object_names is None:
            self.environment_object_groups = list()
        else:
            self.environment_object_groups = environment_object_names
        self.environment_group = environment_name
        self.setup()

    def setup(self):
        self.environment_ids = list()
        for l_n in self.collision_scene.world.groups[self.environment_group].link_names_with_collisions:
            self.environment_ids.append(self.collision_scene.object_name_to_bullet_id[l_n])
        self.environment_object_ids = list()
        for o_g in self.environment_object_groups:
            for l_n in self.collision_scene.world.groups[o_g].link_names_with_collisions:
                self.environment_object_ids.append(self.collision_scene.object_name_to_bullet_id[l_n])
        #self.collision_scene.world.notify_state_change()
        pbw.p.stepSimulation(physicsClientId=self.client_id)

    def ignore_id(self, id):
        return id in self.ignore_object_ids or (
                id not in self.environment_object_ids and id not in self.environment_ids and id != -1)


class PyBulletRayTester():

    def __init__(self, pybulletenv):
        self.pybulletenv = pybulletenv
        self.once = False
        self.link_id_start = -1
        self.collision_free_id = -1
        self.collisionFilterGroup = 0x1
        self.noCollisionFilterGroup = 0x0

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
