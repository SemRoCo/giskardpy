import rospy

from giskardpy.python_interface import GiskardWrapper

rospy.init_node('carry_my_bs')
giskard = GiskardWrapper()
giskard.set_json_goal('CarryMyBullshit')
giskard.set_json_goal('EndlessMode')
giskard.set_max_traj_length(new_length=10000)
giskard.allow_all_collisions()
giskard.plan_and_execute()
rospy.spin()