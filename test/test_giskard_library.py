from giskardpy.data_types.data_types import PrefixName, Derivatives
from giskardpy.model.joints import OmniDrive, PrismaticJoint
from giskardpy.model.links import Link, BoxGeometry
from giskardpy.model.world import WorldTree


class TestWorld:
    def test_moving_box(self):
        box_name = PrefixName('box')
        root_link_name = PrefixName('map')
        joint_name = PrefixName('box_joint')

        world = WorldTree()
        with world.modify_world():
            root_link = Link(root_link_name)
            world.add_link(root_link)

            box = Link(box_name)
            box_geometry = BoxGeometry(1, 1, 1)
            box.collisions.append(box_geometry)
            box.visuals.append(box_geometry)
            world.add_link(box)

            joint = PrismaticJoint(name=joint_name,
                                   free_variable_name=joint_name,
                                   parent_link_name=root_link_name,
                                   child_link_name=box_name,
                                   axis=(1, 0, 0),
                                   lower_limits={Derivatives.position: -1,
                                                 Derivatives.velocity: -1,
                                                 Derivatives.acceleration: None,
                                                 Derivatives.jerk: -30},
                                   upper_limits={Derivatives.position: 1,
                                                 Derivatives.velocity: 1,
                                                 Derivatives.acceleration: None,
                                                 Derivatives.jerk: 30})
            world.add_joint(joint)
        assert joint_name in world.joints
        assert root_link_name in world.root_link_name
        assert box_name in world.links
        world.update_state()
        pass
