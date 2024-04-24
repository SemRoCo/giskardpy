from giskardpy.data_types.data_types import PrefixName
from giskardpy.model.joints import OmniDrive
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.world import WorldTree


class TestWorld:
    def test_moving_box(self):
        box_name = PrefixName('box')
        world = WorldTree()
        with world.modify_world():
            box = Link(box_name)
            box_geometry = BoxGeometry(1, 1, 1)
            box.collisions.append(box_geometry)
            box.visuals.append(box_geometry)
            world.add_link(box)

            joint_name = PrefixName('box_joint')
            joint = OmniDrive(name=joint_name, parent_link_name=world.root_link_name, child_link_name=box_name)
            world.add_joint(joint)
        pass
