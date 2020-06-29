import rospy
from giskard_msgs.msg import MoveCmd, JointConstraint
from giskard_msgs.msg import MoveAction
from giskard_msgs.msg import MoveGoal

# Brings in the SimpleActionClient
import actionlib
from giskard_msgs.msg import MoveResult


def execute_joint_goal():
    # Creates the SimpleActionClient, passing the type of the action
    # (MoveAction) to the constructor.
    client = actionlib.SimpleActionClient("/giskardpy/command", MoveAction)

    # Waits until the action server has started up and started
    # listening for goals.
    print('waiting for giskard')
    client.wait_for_server()
    print('connected to giskard')

    # Creates a goal to send to the action server.
    action_goal = MoveGoal()
    action_goal.type = MoveGoal.PLAN_AND_EXECUTE

    goal = MoveCmd()

    joint_goal = JointConstraint()

    joint_goal.type = JointConstraint.JOINT
    # this can be any subset of the robots joints
    # joint_goal.goal_state is a normal sensor_msgs/JointState
    joint_goal.goal_state.name = ["odom_x_joint", "odom_y_joint", "odom_z_joint",
                                  "fr_caster_rotation_joint", "fr_caster_l_wheel_joint", "fr_caster_r_wheel_joint",
                                  "bl_caster_rotation_joint", "bl_caster_l_wheel_joint", "bl_caster_r_wheel_joint",
                                  "br_caster_rotation_joint", "br_caster_l_wheel_joint", "br_caster_r_wheel_joint",
                                  "torso_lift_joint", "torso_lift_motor_screw_joint", "head_pan_joint",
                                  "head_tilt_joint", "laser_tilt_mount_joint", "r_shoulder_pan_joint",
                                  "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_forearm_roll_joint",
                                  "r_elbow_flex_joint", "r_wrist_flex_joint", "r_wrist_roll_joint",
                                  "r_gripper_motor_slider_joint", "r_gripper_motor_screw_joint",
                                  "r_gripper_l_finger_joint",
                                  "r_gripper_r_finger_joint", "r_gripper_l_finger_tip_joint",
                                  "r_gripper_r_finger_tip_joint",
                                  "r_gripper_joint", "l_shoulder_pan_joint", "l_shoulder_lift_joint",
                                  "l_upper_arm_roll_joint", "l_forearm_roll_joint", "l_elbow_flex_joint",
                                  "l_wrist_flex_joint", "l_wrist_roll_joint", "l_gripper_motor_slider_joint",
                                  "l_gripper_motor_screw_joint", "l_gripper_l_finger_joint", "l_gripper_r_finger_joint",
                                  "l_gripper_l_finger_tip_joint", "l_gripper_r_finger_tip_joint", "l_gripper_joint",
                                  "fl_caster_rotation_joint", "fl_caster_l_wheel_joint", "fl_caster_r_wheel_joint"
                                  ]
    joint_goal.goal_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1687062500000004,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.1400000000000001, -1.0499999999999994,
                                      0.0, 0.0, 0.0, 0.54, 0.54, 0.54, 0.54, 2.220446049250313e-16, 0.0, 0.0, 0.0, 0.0,
                                      -1.1399999999999988, -1.0499999999999994, 0.0, 0.0, 0.0, 0.54, 0.54, 0.54, 0.54,
                                      2.220446049250313e-16, 0.0, 0.0, 0.0]

    #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.220446049250313e-16,
                                      #0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.220446049250313e-16,
                                      #-2.220446049250313e-16, 0.0, 0.0, 0.0, 0.54, 0.54, 0.54, 0.54,
                                      #2.220446049250313e-16, 0.0, 0.0, 0.0, 0.0, -2.220446049250313e-16,
                                      #-2.220446049250313e-16, 0.0, 0.0, 0.0, 0.54, 0.54, 0.54, 0.54,
                                      #2.220446049250313e-16, 0.0, 0.0, 0.0]
    goal.joint_constraints.append(joint_goal)
    action_goal.cmd_seq.append(goal)

    # Sends the goal to the action server.
    client.send_goal(action_goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    result = client.get_result()  # type: MoveResult
    if result.error_code == MoveResult.SUCCESS:
        print('giskard returned success')
    else:
        print('something went wrong')


if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('joint_space_client')
        execute_joint_goal()
    except rospy.ROSInterruptException:
        print("program interrupted before completion")
