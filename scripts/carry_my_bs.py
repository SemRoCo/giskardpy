import rospy

from giskardpy.python_interface import GiskardWrapper

rospy.init_node('carry_my_bs')
giskard = GiskardWrapper()
giskard.set_json_goal('CarryMyBullshit',
                        patrick_topic_name='/robokudo2/human_position',
                      camera_link='head_rgbd_sensor_link',
                      laser_topic_name='/hsrb/base_scan')
giskard.set_json_goal('EndlessMode')
giskard.set_max_traj_length(new_length=10000)
giskard.allow_all_collisions()
giskard.plan_and_execute()
rospy.spin()