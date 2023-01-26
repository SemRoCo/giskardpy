from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy.data_types import BiDict
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer, Collisions, Collision
from giskardpy.model.pybullet_wrapper import ContactInfo
from giskardpy.utils.utils import resolve_ros_iris


class PyBulletSyncer(CollisionWorldSynchronizer):
    hack_name = 'hack'

    def __init__(self, world, gui=False):
        super(PyBulletSyncer, self).__init__(world)
        # pbw.start_pybullet(self.god_map.get_data(identifier.gui))
        pbw.start_pybullet(gui)
        pbw.deactivate_rendering()
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
                                                                                    orientation=orientation)

    @profile
    def update_pose(self, link):
        pose = self.fks[link.name]
        position = pose[:3]
        orientation = pose[4:]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[str(link.name)], position, orientation)

    def check_collisions2(self, link_combinations, distance):
        in_collision = set()
        self.sync()
        for link_a, link_b in link_combinations:
            if self.in_collision(link_a, link_b, distance):
                in_collision.add((link_a, link_b))
        return in_collision

    @profile
    def check_collisions(self, cut_off_distances, collision_list_size):
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
        collisions = Collisions(collision_list_size)
        for (link_a, link_b), distance in cut_off_distances.items():
            link_b_id = self.object_name_to_bullet_id[link_b]
            robot_link_id = self.object_name_to_bullet_id[link_a]
            contacts = [ContactInfo(*x) for x in pbw.getClosestPoints(robot_link_id, link_b_id,
                                                                      distance * 1.1)]
            if len(contacts) > 0:
                for contact in contacts:  # type: ContactInfo
                    collision = Collision(link_a=link_a,
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
        return len(pbw.getClosestPoints(link_id_a, link_id_b, distance)) > 0

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
            pbw.clear_pybullet()
            self.world.fast_all_fks = None
            self.fks = self.world.compute_all_fks()
            for link_name, link in self.world.links.items():
                if link.has_collisions():
                    self.add_object(link)
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
        position, orientation = pbw.getBasePositionAndOrientation(self.object_name_to_bullet_id[link_name])
        map_T_link.header.frame_id = self.world.root_link_name
        map_T_link.pose.position = Point(*position)
        map_T_link.pose.orientation = Quaternion(*orientation)
        return map_T_link

    def __add_pybullet_bug_fix_hack(self):
        if self.hack_name not in self.object_name_to_bullet_id:
            path = resolve_ros_iris('package://giskardpy/urdfs/tiny_ball.urdf')
            with open(path, 'r') as f:
                self.object_name_to_bullet_id[self.hack_name] = pbw.load_urdf_string_into_bullet(f.read())

    def __move_hack(self, pose):
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[self.hack_name],
                                            position, orientation)

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
                                                  body_a_id, 0.001) if
                  abs(x[8] + 0.005) < 0.0005][0])
            return not contact_info3.body_unique_id_b == body_a_id
        except Exception as e:
            return True
