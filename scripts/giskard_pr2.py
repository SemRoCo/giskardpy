#!/usr/bin/env python
import rospy

from giskardpy.configs.pr2 import PR2_Mujoco, PR2_IAI, PR2StandAlone, PR2_Unreal
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    config = rospy.get_param('~config')
    check_dependencies()
    if config == 'PR2_Mujoco':
        giskard = PR2_Mujoco()
    elif config == 'PR2_IAI':
        giskard = PR2_IAI()
    elif config == 'PR2_Unreal':
        giskard = PR2_Unreal()
    else:
        giskard = PR2StandAlone()
    giskard.live()
