import copy

from std_msgs.msg import String

import rospy
from py_trees import Status

from giskardpy import identifier
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.model.utils import make_world_body_box


class SyncPouringActions(GiskardBehavior):

    @profile
    def __init__(self, group_name: str, state_topic='/pouringActions'):
        """
        :type js_identifier: str
        """
        super().__init__(str(self))
        self.state_topic = state_topic
        if not self.state_topic.startswith('/'):
            self.ft_topic = '/' + self.state_topic
        super().__init__(str(self))
        self.commands = []
        self.command_base = identifier.pouring_actions
        self.map_key_command = {'w': 'forward',
                                's': 'backward',
                                'a': 'left',
                                'd': 'right',
                                'u': 'up',
                                'j': 'down',
                                'y': 'move_to',
                                'g': 'tilt',
                                'h': 'tilt_back',
                                'q': 'keep_upright'}
        self.all_commands = {'forward': 0,
                             'backward': 0,
                             'left': 0,
                             'right': 0,
                             'up': 0,
                             'down': 0,
                             'move_to': 0,
                             'tilt': 0,
                             'tilt_back': 0,
                             'keep_upright': 0}
        self.all_commands_empty = copy.deepcopy(self.all_commands)
        self.god_map.set_data(identifier.pouring_actions, self.all_commands)
        self.counter = 0

    @profile
    def setup(self, timeout=0.0):
        self.sub = rospy.Subscriber(self.state_topic, String, self.cb, queue_size=10)
        return super().setup(timeout)

    def cb(self, data: String):
        keys = data.data.split(';')
        self.commands = [self.map_key_command[key] for key in keys]
        self.counter = 0

    @profile
    def update(self):
        self.all_commands = copy.deepcopy(self.all_commands_empty)
        for command in self.commands:
            self.all_commands[command] = 1

        self.god_map.set_data(identifier.pouring_actions, self.all_commands)

        if self.counter > 10:
            self.commands = ''

        self.counter += 1
        return Status.RUNNING
