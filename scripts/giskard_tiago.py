#!/usr/bin/env python
import rospy

from giskardpy.configs.tiago import TiagoMujoco, IAI_Tiago, TiagoStandAlone
from giskardpy.utils.dependency_checking import check_dependencies

if __name__ == '__main__':
    rospy.init_node('giskard')
    check_dependencies()
    giskard = IAI_Tiago()
    # giskard = TiagoMujoco()
    # giskard = TiagoStandAlone()
    giskard.live()
