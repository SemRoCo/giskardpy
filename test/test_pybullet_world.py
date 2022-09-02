import shutil
from collections import defaultdict

import pytest
from geometry_msgs.msg import Pose
from giskard_msgs.msg import CollisionEntry

from giskardpy import RobotName
from giskardpy.exceptions import PhysicsWorldException, UnknownGroupException
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.model.pybullet_syncer import PyBulletSyncer
from giskardpy.model.utils import make_world_body_box
from giskardpy.utils import logging
from test_world import create_world_with_pr2, create_world_with_donbot, allow_all_entry, avoid_all_entry

folder_name = 'tmp_data/'


@pytest.fixture(scope='module')
def module_setup(request):
    logging.loginfo('starting pybullet')
    # pbw.start_pybullet(True)

    logging.loginfo('deleting tmp test folder')
    try:
        shutil.rmtree(folder_name)
    except:
        pass

    def kill_pybullet():
        logging.loginfo('shutdown pybullet')
        # pbw.stop_pybullet()

    request.addfinalizer(kill_pybullet)


@pytest.fixture()
def function_setup(request, module_setup):
    # pbw.clear_pybullet()

    def kill_pybullet():
        logging.loginfo('resetting pybullet')
        # pbw.clear_pybullet()

    request.addfinalizer(kill_pybullet)


@pytest.fixture()
def pr2_world(request, function_setup):
    """
    :rtype: World
    """

    world = create_world_with_pr2()
    pbs = CollisionWorldSynchronizer(world)
    pbs.sync()
    return pbs


@pytest.fixture()
def donbot_world(request, function_setup):
    """
    :rtype: World
    """

    world = create_world_with_donbot()
    pbs = CollisionWorldSynchronizer(world)
    pbs.sync()
    return pbs


@pytest.fixture()
def delete_test_folder(request):
    """
    :rtype: World
    """
    folder_name = 'tmp_data/'
    try:
        shutil.rmtree(folder_name)
    except:
        pass
    return folder_name


# def assert_num_pybullet_objects(num):
#     assert p.getNumBodies() == num, pbw.print_body_names()


class TestPyBulletSyncer(object):
    def test_load_pr2(self, pr2_world):
        pass
        # assert len(pbw.get_body_names()) == 46

    def test_set_pr2_js(self, pr2_world):
        pr2_world.world.state['torso_lift_link'] = 1
        pr2_world.sync()
        # assert len(pbw.get_body_names()) == 46

    def test_update_blacklist(self, pr2_world: CollisionWorldSynchronizer):
        pr2_world.update_collision_blacklist()
        pass

    def test_compute_collision_matrix_donbot(self, donbot_world):
        """
        :type pr2_world: PyBulletSyncer
        """
        donbot_world.init_collision_matrix(RobotName)
        collision_matrix = donbot_world.collision_matrices[RobotName]
        for entry in collision_matrix:
            assert entry[0] != entry[-1]

        assert len(collision_matrix) == 125

    def test_add_object(self, pr2_world):
        """
        :type pr2_world: PyBulletSyncer
        """
        o = make_world_body_box()
        p = Pose()
        p.orientation.w = 1
        pr2_world.world.add_world_body(group_name='box',
                                       msg=o,
                                       pose=p)
        pr2_world.sync()
        # assert len(pbw.get_body_names()) == 47

    def test_delete_object(self, pr2_world):
        """
        :type pr2_world: PyBulletSyncer
        """
        o_name = 'box'
        o = make_world_body_box()
        p = Pose()
        p.orientation.w = 1
        pr2_world.world.add_world_body(group_name=o_name, msg=o, pose=p)
        pr2_world.sync()
        pr2_world.world.delete_branch(o_name)
        pr2_world.sync()
        # assert len(pbw.get_body_names()) == 46

    def test_attach_object(self, pr2_world):
        """
        :type pr2_world: PyBulletSyncer
        """
        o_name = 'box'
        o = make_world_body_box()
        p = Pose()
        p.orientation.w = 1
        pr2_world.world.add_world_body(group_name=o_name, msg=o, pose=p)
        pr2_world.world.move_group(o_name, 'r_gripper_tool_frame')
        pr2_world.sync()
        # assert len(pbw.get_body_names()) == 47

    def test_compute_collision_matrix_attached(self, pr2_world):
        """
        :type pr2_world: PyBulletSyncer
        """
        pr2_world.init_collision_matrix(RobotName)
        pr2_world.world.register_group('r_hand', 'r_wrist_roll_link')
        old_collision_matrix = pr2_world.collision_matrices[RobotName]
        o = make_world_body_box()
        p = Pose()
        p.orientation.w = 1
        pr2_world.world.add_world_body(o, p)
        pr2_world.world.move_group(o.position_name, 'r_gripper_tool_frame')
        pr2_world.init_collision_matrix(RobotName)
        assert len(pr2_world.collision_matrices[RobotName]) > len(old_collision_matrix)
        contains_box = False
        for entry in pr2_world.collision_matrices[RobotName]:
            contains_box |= o.position_name in entry
            if o.position_name in entry:
                contains_box |= True
                if o.position_name == entry[0]:
                    assert entry[1] not in pr2_world.world.groups['r_hand'].links
                if o.position_name == entry[1]:
                    assert entry[0] not in pr2_world.world.groups['r_hand'].links
        assert contains_box
        pr2_world.world.delete_branch(o.position_name)
        pr2_world.init_collision_matrix(RobotName)
        assert pr2_world.collision_matrices[RobotName] == old_collision_matrix

    def test_verify_collision_entries_empty(self, donbot_world):
        ces = []
        new_ces = donbot_world.verify_collision_entries(ces)
        assert len(new_ces) == 1

    def test_verify_collision_entries_allow_all(self, donbot_world):
        ces = [allow_all_entry()]
        new_ces = donbot_world.verify_collision_entries(ces)
        assert len(new_ces) == 0

    def test_verify_collision_entries_allow_all_self(self, donbot_world):
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.group1 = CollisionEntry.ALL
        ce.group2 = CollisionEntry.ALL
        ces = [ce]
        new_ces = donbot_world.verify_collision_entries(ces)
        assert len(new_ces) == 0

    def test_verify_collision_entries_unknown_group(self, donbot_world):
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.group1 = 'muh'
        ce.distance = min_dist
        ces.append(ce)
        try:
            new_ces = donbot_world.verify_collision_entries(ces)
        except UnknownGroupException:
            assert True
        else:
            assert False, 'expected exception'

    def test_verify_collision_entries_unknown_group2(self, donbot_world):
        min_dist = 0.1
        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.group2 = 'asdf'
        ce.distance = min_dist
        ces.append(ce)
        try:
            new_ces = donbot_world.verify_collision_entries(ces)
        except UnknownGroupException:
            assert True
        else:
            assert False, 'expected exception'

    def test_verify_collision_entries_cut_off1(self, donbot_world):
        min_dist = 0.1
        ces = []
        ces.append(avoid_all_entry(min_dist))
        ces.append(allow_all_entry())
        new_ces = donbot_world.verify_collision_entries(ces)
        assert len(new_ces) == 0

    def test_collision_goals_to_collision_matrix1(self, donbot_world: CollisionWorldSynchronizer):
        """
        test with no collision entries which is equal to avoid all collisions
        collision matrix should be empty, because world has no collision checker
        :param test_folder:
        :return:
        """
        min_dist = defaultdict(lambda: 0.05)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix([], min_dist, {})
        assert len(collision_matrix) == 0

    def test_collision_goals_to_collision_matrix2(self, donbot_world):
        """
        avoid all with an added object should enlarge the collision matrix
        :type donbot_world: PyBulletSyncer
        """
        min_dist = defaultdict(lambda: 0.05)
        base_collision_matrix = donbot_world.collision_goals_to_collision_matrix([], min_dist)
        name = 'muh'
        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix([], min_dist)
        assert len(collision_matrix) == len(base_collision_matrix) + len(donbot_world.robot.link_names_with_collisions)
        robot_link_names = donbot_world.robot.link_names
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix3(self, donbot_world):
        """
        empty list should have the same effect than avoid all entry
        :type donbot_world: PyBulletSyncer
        """
        min_dist = defaultdict(lambda: 0.05)
        base_collision_matrix = donbot_world.collision_goals_to_collision_matrix([], min_dist)
        name = 'muh'
        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)
        ces = []
        ces.append(avoid_all_entry(min_dist))
        collision_matrix = donbot_world.collision_goals_to_collision_matrix(ces, min_dist)
        assert len(collision_matrix) == len(base_collision_matrix) + len(donbot_world.robot.link_names_with_collisions)
        robot_link_names = donbot_world.robot.link_names
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix4(self, donbot_world):
        """
        allow all should lead to an empty collision matrix
        :param test_folder:
        :return:
        """
        name = 'muh'
        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)

        ces = []
        ces.append(allow_all_entry())
        ces.append(allow_all_entry())
        min_dist = defaultdict(lambda: 0.05)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == 0

    def test_collision_goals_to_collision_matrix_avoid_only_box(self, donbot_world):
        name = 'muh'
        robot_link_names = list(donbot_world.robot().link_names_with_collisions)

        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)

        ces = []
        ces.append(allow_all_entry())
        ce = CollisionEntry()
        ce.type = CollisionEntry.AVOID_COLLISION
        ce.robot_links = [robot_link_names[0]]
        ce.body_b = name
        ce.min_dist = 0.1
        ces.append(ce)
        min_dist = defaultdict(lambda: 0.1)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix(ces, min_dist)

        assert len(collision_matrix) == 1
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == ce.min_dist
            assert body_b == name
            assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix6(self, donbot_world):
        """
        allow collision with a specific object
        """
        name = 'muh'
        robot_link_names = list(donbot_world.robot().link_names_with_collisions)
        min_dist = defaultdict(lambda: 0.1)

        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)

        allowed_link = robot_link_names[0]

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [allowed_link]
        ce.link_bs = [CollisionEntry.ALL]
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix(ces, min_dist)

        assert len([x for x in collision_matrix if x[0] == allowed_link]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix7(self, donbot_world):
        """
        allow collision with specific object
        """
        name = 'muh'
        name2 = 'muh2'
        robot_link_names = list(donbot_world.robot().link_names_with_collisions)
        min_dist = defaultdict(lambda: 0.05)

        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)
        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name2), p)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.body_b = name2
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix(ces, min_dist)

        assert len([x for x in collision_matrix if x[2] == name2]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
            if body_b == name:
                assert body_b_link == name
            assert robot_link in robot_link_names

    def test_collision_goals_to_collision_matrix8(self, donbot_world):
        """
        allow collision between specific object and link
        """
        name = 'muh'
        name2 = 'muh2'
        robot_link_names = list(donbot_world.robot().link_names_with_collisions)
        allowed_link = robot_link_names[0]
        min_dist = defaultdict(lambda: 0.05)

        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name), p)
        p = Pose()
        p.orientation.w = 1
        donbot_world.world.add_world_body(make_world_body_box(name2), p)

        ces = []
        ce = CollisionEntry()
        ce.type = CollisionEntry.ALLOW_COLLISION
        ce.robot_links = [allowed_link]
        ce.body_b = name2
        ce.min_dist = 0.1
        ces.append(ce)
        collision_matrix = donbot_world.collision_goals_to_collision_matrix(ces, min_dist)

        assert len([x for x in collision_matrix if x[0] == allowed_link and x[2] == name2]) == 0
        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert dist == min_dist[robot_link]
<<<<<<< HEAD
            if body_b != donbot_world.robot.position_name:
=======
            if body_b != donbot_world.robot().name:
>>>>>>> a018cd7d105dd186f3940076d6fa666a95610d18
                assert body_b_link == name or body_b_link == name2
            assert robot_link in robot_link_names
            if body_b == name2:
                assert robot_link != robot_link_names[0]

    def test_collision_goals_to_collision_matrix9(self, pr2_world):
        """
        allow self collision
        """
        min_dist = defaultdict(lambda: 0.05)
        ces = []
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.ALLOW_COLLISION
        collision_entry.robot_links = ['l_gripper_l_finger_tip_link', 'l_gripper_r_finger_tip_link',
                                       'l_gripper_l_finger_link', 'l_gripper_r_finger_link',
                                       'l_gripper_r_finger_link', 'l_gripper_palm_link']
<<<<<<< HEAD
        collision_entry.body_b = pr2_world.robot.position_name
=======
        collision_entry.body_b = pr2_world.robot().name
>>>>>>> a018cd7d105dd186f3940076d6fa666a95610d18
        collision_entry.link_bs = ['r_wrist_flex_link', 'r_wrist_roll_link', 'r_forearm_roll_link',
                                   'r_forearm_link', 'r_forearm_link']
        ces.append(collision_entry)

        collision_matrix = pr2_world.collision_goals_to_collision_matrix(ces, min_dist)

        for (robot_link, body_b, body_b_link), dist in collision_matrix.items():
            assert not (robot_link in collision_entry.robot_links and body_b_link in collision_entry.link_bs)
            assert not (body_b_link in collision_entry.robot_links and robot_link in collision_entry.link_bs)

    def test_collision_goals_to_collision_matrix10(self, pr2_world):
        """
        avoid self collision with only specific links
        :param test_folder:
        :return:
        """
        min_dist = defaultdict(lambda: 0.05)
        ces = [allow_all_entry()]
        collision_entry = CollisionEntry()
        collision_entry.type = CollisionEntry.AVOID_COLLISION
        collision_entry.robot_links = ['base_link']
        collision_entry.body_b = RobotName
        collision_entry.link_bs = ['r_wrist_flex_link']
        ces.append(collision_entry)

        collision_matrix = pr2_world.collision_goals_to_collision_matrix(ces, min_dist)

        assert {('base_link', RobotName, 'r_wrist_flex_link'): 0.05} == collision_matrix

# import pytest
# pytest.main(['-s', __file__ + '::TestPyBulletSyncer::test_compute_collision_matrix'])
