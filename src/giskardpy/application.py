import rospy

class Application(object):
    def __init__(self):
        self.timer = rospy.Timer(rospy.Duration(rospy.get_param("loop_period")), self.callback)
        # TODO: construct ProcessManager
        # TODO: register Plugins

    def callback(self, time_event):
        pass
        # TODO: call ProcessManager

    def run(self):
        # TODO: start Plugins
        # TODO: catch stop exception
        try:
            rospy.spin()
        except Exception as e:
            pass
            # TODO: stop all Plugins