class World(object):
    def spawn_urdf_robot(self, urdf_string, robot_name, base_pose, init_joint_state):
        raise NotImplementedError('Please implement spawn_urdf_robot in World.')

    def set_joint_state(self, robot_name, joint_state):
        """
        Set the current joint state readings for a robot in the world.
        :param robot_name: name of the robot to update
        :type string
        :param joint_state: sensor readings for the entire robot
        :type dict{string, JointState}
        """
        raise NotImplementedError('Please implement set_joint_state in World.')

    def activate_viewer(self):
        raise NotImplementedError('Please implement activate_viewer in World.')

    def deactivate_viewer(self):
        raise NotImplementedError('Please implement deactivate_viewer in World.')