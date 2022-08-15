from giskardpy.tree.behaviors.action_server import ActionServerBehavior


class GetGoal(ActionServerBehavior):
    @profile
    def __init__(self, name, as_name):
        super().__init__(name, as_name)

    def pop_goal(self):
        return self.get_as().pop_goal()