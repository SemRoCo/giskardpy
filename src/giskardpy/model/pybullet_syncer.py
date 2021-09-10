from time import time

from tf.transformations import quaternion_matrix, quaternion_from_matrix

import giskardpy.model.pybullet_wrapper as pbw
from giskardpy.data_types import BiDict
from giskardpy.model.world import WorldTree
from giskardpy.utils.tfwrapper import np_to_pose


class PyBulletObject(object):
    def __init__(self):
        pass


class PyBulletSyncer(object):
    def __init__(self, world, gui=False):
        pbw.start_pybullet(gui)
        self.object_name_to_bullet_id = BiDict()
        self.world = world # type: WorldTree

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
        pbw.resetBasePositionAndOrientation(self.object_name_to_bullet_id[link.name], position, orientation)

    @profile
    def sync(self):
        """
        :type world: giskardpy.model.world.WorldTree
        """
        # pbw.clear_pybullet()
        t = time()
        self.fks = self.world.compute_all_fks()
        for link_name, link in self.world.links.items():
            if link.has_collisions():
                if link_name in self.object_name_to_bullet_id:
                    self.update_pose(link)
                else:
                    self.add_object(link)
        print('sync took {}'.format(time() - t))
