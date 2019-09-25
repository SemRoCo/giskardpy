from __future__ import division
import PyKDL as kdl
import numpy as np
from iai_naive_kinematics_sim.srv import SetJointState, SetJointStateRequest
from transforms3d._gohlketransforms import quaternion_matrix
from transforms3d.derivations.angle_axes import angle_axis2mat

f = np.array([-0.5,0,0])

com = np.array([0,0,0])

def muh(f, com, r,):
    p = com + r
    goal_p = p + f
    rr1 = r / np.linalg.norm(r)
    wrench = np.zeros(6)
    wrench[:3] = rr1 * np.dot(rr1, f)
    wrench[3:] = np.cross(r, f - wrench[:3])
    print('wrench {}'.format(wrench))
    print('goal p {}'.format(goal_p))
    for i in range(5):
        w = wrench * ((i+1)/5)
        new_com = com + w[:3]
        print('new com {}'.format(new_com))
        # r2 = p - new_com
        # print('new r {}'.format(r2))
        # print(np.cos(np.dot(r, r2) / (np.linalg.norm(r) * np.linalg.norm(r2))))

        m = angle_axis2mat(w[-1], [0,0,1])
        m = np.array(m).astype(float)
        new_r = np.dot(m, r)
        new_p = new_com + new_r
        print('new r {}'.format(new_r))
        print('new p {}'.format(new_p))
        print('dist to old p {}'.format(np.linalg.norm(p-new_p)))

        r2 = new_com + r
        r3 = goal_p - new_com
        print('r2 {}'.format(r2))
        print('r3 {}'.format(r3))
        print('angle {}'.format(np.arccos(np.dot(r3, r2) / (np.linalg.norm(r3) * np.linalg.norm(r2)))))
        print('=================================================================')

muh(f,com,
    r=np.array([0.5,1,0]))
# muh(np.array([-1,0,0]),com,r, np.array([-0.5,1,0]))
