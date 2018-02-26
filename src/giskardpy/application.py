import traceback

import rospy

class ROSApplication(object):
    def __init__(self, process_manager):
        self.process_manager = process_manager
        self.process_manager.start()
        self.timer = rospy.Timer(rospy.Duration(rospy.get_param("loop_period", 0.1)), self.callback)

    def callback(self, time_event):
        self.process_manager.update()

    def run(self):
        try:
            self.process_manager.start()
            rospy.spin()
        except Exception as e:
            traceback.print_exc()
        finally:
            self.process_manager.stop()