from collections import OrderedDict


class Controller(object):
    def __init__(self, robot):
        self.__robot = robot
        self.__state = OrderedDict()  # e.g. goal

        self.init()

    def get_state(self):
        return self.__state

    def get_robot(self):
        return self.__robot

    def update_observables(self, updates):
        """
        :param updates: dict{str->float} observable name to it value
        :return: dict{str->float} joint name to vel command
        """
        self.__state.update(updates)

    def init(self):
        raise (NotImplementedError)

    def get_next_command(self):
        raise(NotImplementedError)
