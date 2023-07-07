import rospy

from giskardpy.python_interface import GiskardWrapper

rospy.init_node('carry_my_bs')
giskard = GiskardWrapper()
giskard.set_json_goal('CarryMyBullshit',
                      patrick_topic_name='robokudosuturo3/human_position')
giskard.allow_all_collisions()
giskard.plan_and_execute(wait=False)
input('asdf')
# rospy.sleep(15)
giskard.cancel_all_goals()
print('goal canceled')
giskard.set_json_goal('CarryMyBullshit',
                      patrick_topic_name='robokudosuturo3/human_position',
                      drive_back=True)
giskard.allow_all_collisions()
giskard.plan_and_execute(wait=False)
input('asdf')
giskard.cancel_all_goals()
print('goal canceled')