from py_trees import Status

from giskard_msgs.msg import MoveFeedback
from giskardpy.tree.behaviors.action_server import ActionServerBehavior
from giskardpy.utils.decorators import record_time


class PublishFeedback(ActionServerBehavior):
    @profile
    def __init__(self, name, as_name, feedback):
        super().__init__(name, as_name)
        self.feedback = feedback

    @record_time
    @profile
    def update(self):
        feedback_msg = MoveFeedback()
        feedback_msg.state = self.feedback
        self.as_handler.send_feedback(feedback_msg)
        return Status.SUCCESS