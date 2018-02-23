class Plugin(object):

    def start(self):
        raise NotImplementedError('Please implement the start method of this plugin.')

    def stop(self):
        raise NotImplementedError('Please implement the stop method of this plugin.')

    def update(self):
        raise NotImplementedError('Please implement the update method of this plugin.')


class PluginCollection(object):
    def __init__(self):
        pass

    def get_plugins(self):
        pass

class BulletRobot(PluginCollection):
    def __init__(self, joint_state_io):
        self.joint_state_io = joint_state_io
        self.base_pose_io = None
        self.collision_io = None
        super(BulletRobot, self).__init__()

    def get_plugins(self):
        return [self.joint_state_io,
                self.base_pose_io,
                self.collision_io]


class RLRobot(PluginCollection):
    def __init__(self):
        self.joint_state_io = None
        self.fk_io = None
        super(RLRobot, self).__init__()

    def get_plugins(self):
        return [self.joint_state_io,
                self.fk_io]

