import py_trees.composites
from py_trees import common, Status


# copy paste from py_trees 2.x
class ParallelPolicy(object):
    """
    Configurable policies for :py:class:`~py_trees.composites.Parallel` behaviours.
    """
    class Base(object):
        """
        Base class for parallel policies. Should never be used directly.
        """
        def __init__(self, synchronise=False):
            """
            Default policy configuration.
            Args:
                synchronise (:obj:`bool`): stop ticking of children with status :py:data:`~py_trees.common.Status.SUCCESS` until the policy criteria is met
            """
            self.synchronise = synchronise

    class SuccessOnAll(Base):
        """
        Return :py:data:`~py_trees.common.Status.SUCCESS` only when each and every child returns
        :py:data:`~py_trees.common.Status.SUCCESS`. If synchronisation is requested, any children that
        tick with :data:`~py_trees.common.Status.SUCCESS` will be skipped on subsequent ticks until
        the policy criteria is met, or one of the children returns status :data:`~py_trees.common.Status.FAILURE`.
        """
        def __init__(self, synchronise=True):
            """
            Policy configuration.
            Args:
                synchronise (:obj:`bool`): stop ticking of children with status :py:data:`~py_trees.common.Status.SUCCESS` until the policy criteria is met
            """
            super().__init__(synchronise=synchronise)

    class SuccessOnOne(Base):
        """
        Return :py:data:`~py_trees.common.Status.SUCCESS` so long as at least one child has :py:data:`~py_trees.common.Status.SUCCESS`
        and the remainder are :py:data:`~py_trees.common.Status.RUNNING`
        """
        def __init__(self):
            """
            No configuration necessary for this policy.
            """
            super().__init__(synchronise=False)

    class SuccessOnSelected(Base):
        """
        Return :py:data:`~py_trees.common.Status.SUCCESS` so long as each child in a specified list returns
        :py:data:`~py_trees.common.Status.SUCCESS`. If synchronisation is requested, any children that
        tick with :data:`~py_trees.common.Status.SUCCESS` will be skipped on subsequent ticks until
        the policy criteria is met, or one of the children returns status :data:`~py_trees.common.Status.FAILURE`.
        """
        def __init__(self, children, synchronise=True):
            """
            Policy configuraiton.
            Args:
                children ([:class:`~py_trees.behaviour.Behaviour`]): list of children to succeed on
                synchronise (:obj:`bool`): stop ticking of children with status :py:data:`~py_trees.common.Status.SUCCESS` until the policy criteria is met
            """
            super().__init__(synchronise=synchronise)
            self.children = children


class Parallel(py_trees.composites.Parallel):
    def tick(self):
        """
        Tick over the children.

        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        if self.status != Status.RUNNING:
            # subclass (user) handling
            self.initialise()
        # self.logger.debug("%s.tick()" % self.__class__.__name__)
        # process them all first
        for child in self.children:
            if self.policy.synchronise and child.status == common.Status.SUCCESS:
                continue
            for node in child.tick():
                yield node
        # new_status = Status.SUCCESS if self.policy == common.ParallelPolicy.SUCCESS_ON_ALL else Status.RUNNING
        new_status = Status.RUNNING
        if any([c.status == Status.FAILURE for c in self.children]):
            new_status = Status.FAILURE
        else:
            if isinstance(self.policy, ParallelPolicy.SuccessOnAll):
                if all([c.status == Status.SUCCESS for c in self.children]):
                    new_status = Status.SUCCESS
            elif isinstance(self.policy, ParallelPolicy.SuccessOnOne):
                if any([c.status == Status.SUCCESS for c in self.children]):
                    new_status = Status.SUCCESS
        # special case composite - this parallel may have children that are still running
        # so if the parallel itself has reached a final status, then these running children
        # need to be made aware of it too
        if new_status != Status.RUNNING:
            for child in self.children:
                if child.status == Status.RUNNING:
                    # interrupt it (exactly as if it was interrupted by a higher priority)
                    child.stop(Status.INVALID)
            self.stop(new_status)
        self.status = new_status
        yield self