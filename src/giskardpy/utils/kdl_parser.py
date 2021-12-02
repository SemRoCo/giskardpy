import numpy as np

import PyKDL as kdl
import rospy
import urdf_parser_py.urdf as up

from giskardpy.model.utils import hacky_urdf_parser_fix


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r/2.0), np.sin(p/2.0), np.sin(y/2.0)
    cr, cp, cy = np.cos(r/2.0), np.cos(p/2.0), np.cos(y/2.0)
    return [sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy]

def urdf_pose_to_kdl_frame(pose):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)),
                     kdl.Vector(*pos))

def urdf_joint_to_kdl_joint(jnt):
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == 'fixed':
        return kdl.Joint(jnt.name)
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == 'revolute':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'continuous':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.RotAxis)
    if jnt.joint_type == 'prismatic':
        return kdl.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, kdl.Joint.TransAxis)
    print("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name)

def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(i.mass, origin.p,
                               kdl.RotationalInertia(i.inertia.ixx,
                                                     i.inertia.iyy,
                                                     i.inertia.izz,
                                                     i.inertia.ixy,
                                                     i.inertia.ixz,
                                                     i.inertia.iyz))
    return origin.M * rbi

def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = kdl.Tree(root)
    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joint_map[joint])
                kdl_origin = urdf_pose_to_kdl_frame(urdf.joint_map[joint].origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt,
                                      kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)
    add_children_to_tree(root)
    return tree


def kdl_joint_limits_from_urdf_model(urdf, joint_names, static_joints=None):
    min = kdl.JntArray(len(joint_names))
    max = kdl.JntArray(len(joint_names))
    for i, joint_name in enumerate(joint_names):
        if static_joints and joint_name in static_joints:
            min[i] = 0
            max[i] = 0
            continue
        joint = urdf.joint_map[joint_name]
        if joint.limit.lower is not None:
            min[i] = joint.limit.lower
        else:
            rospy.logerr(u'Joint {} has no lower limits.'.format(joint_name))
        if joint.limit.upper is not None:
            max[i] = joint.limit.upper
        else:
            rospy.logerr(u'Joint {} has no upper limits.'.format(joint_name))
    return min, max


def joint_names_from_kdl_chain(chain):
    joints = []
    for i in range(chain.getNrOfSegments()):
        joint = chain.getSegment(i).getJoint()
        if joint.getType() != 8:
            joints.append(str(joint.getName()))
    return joints


class KDL(object):
    class KDLRobot(object):
        def __init__(self, joints, chain, chain_min, chain_max):
            self.joints = joints
            self.chain = chain
            self.chain_min = chain_min
            self.chain_max = chain_max
            self.fksolver = kdl.ChainFkSolverPos_recursive(self.chain)
            self.iksolver_vel = kdl.ChainIkSolverVel_pinv(self.chain)
            self.iksolver = kdl.ChainIkSolverPos_NR_JL(self.chain, self.chain_min, self.chain_max,
                                                       self.fksolver, self.iksolver_vel)
            self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
            self.jacobian = kdl.Jacobian(self.chain.getNrOfJoints())

        def get_joints(self):
            return joint_names_from_kdl_chain(self.chain)

        def fk(self, js_dict):
            js = [js_dict[j] for j in self.joints]
            f = kdl.Frame()
            joint_array = kdl.JntArray(len(js))
            for i in range(len(js)):
                joint_array[i] = js[i]
            self.fksolver.JntToCart(joint_array, f)
            return f

        def fk_np(self, js_dict):
            f = self.fk(js_dict)
            r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
                 [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
                 [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
                 [0, 0, 0, 1], ]
            return np.array(r)

        def fk_np_inv(self, js_dict):
            f = self.fk(js_dict).Inverse()
            r = [[f.M[0, 0], f.M[0, 1], f.M[0, 2], f.p[0]],
                 [f.M[1, 0], f.M[1, 1], f.M[1, 2], f.p[1]],
                 [f.M[2, 0], f.M[2, 1], f.M[2, 2], f.p[2]],
                 [0, 0, 0, 1], ]
            return np.array(r)

        def ik(self, js_dict, frame):
            js = [js_dict[j] for j in self.joints]
            theta_out = kdl.JntArray(len(js))
            theta_init = kdl.JntArray(len(js))
            for i in range(len(js)):
                theta_init[i] = js[i]
            self.iksolver.CartToJnt(theta_init, frame, theta_out)
            return theta_out

    def __init__(self, urdf):
        if urdf.endswith(u'.urdfs'):
            with open(urdf, u'r') as file:
                urdf = file.read()
        self.urdf = up.URDF.from_xml_string(hacky_urdf_parser_fix(urdf))
        self.tree = kdl_tree_from_urdf_model(self.urdf)
        self.robots = {}

    def get_joints(self, chain):
        return joint_names_from_kdl_chain(chain)

    def get_robot(self, root, tip, static_joints=None):
        root = str(root)
        tip = str(tip)
        if (root, tip) not in self.robots:
            chain = self.tree.getChain(root, tip)
            joints = self.get_joints(chain)
            chain_min, chain_max = kdl_joint_limits_from_urdf_model(self.urdf, joints, static_joints=static_joints)
            self.robots[root, tip] = self.KDLRobot(joints, chain, chain_min, chain_max)
        return self.robots[root, tip]