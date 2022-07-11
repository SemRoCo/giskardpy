from __future__ import division

from giskardpy.goals.goal import Goal


class OverwriteWeights(Goal):

    def __init__(self, updates: dict, **kwargs):
        """
        Changes the weights for one goal.
        :param updates:  e.g.
                    {
                        1: {
                            'joint1': 0.001,
                        },
                        2: {
                            'joint1': 0.0,
                        },
                        3: {
                            'joint1': 0.001,
                        }
                    }
        """
        super().__init__(**kwargs)
        # ints get parsed as strings, when they arrive here...
        updates = {int(k): v for k, v in updates.items()}
        self.world.overwrite_joint_weights(updates)
