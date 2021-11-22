import py_trees.composites
from py_trees import Composite, common, Status


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


# class Parallel(Composite):
#     """
#     Parallels enable a kind of concurrency
#
#     .. graphviz:: dot/parallel.dot
#
#     Ticks every child every time the parallel is run (a poor man's form of parallelism).
#
#     * Parallels will return :data:`~py_trees.common.Status.FAILURE` if any
#       child returns :py:data:`~py_trees.common.Status.FAILURE`
#     * Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnAll`
#       only returns :py:data:`~py_trees.common.Status.SUCCESS` if **all** children
#       return :py:data:`~py_trees.common.Status.SUCCESS`
#     * Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnOne`
#       return :py:data:`~py_trees.common.Status.SUCCESS` if **at least one** child
#       returns :py:data:`~py_trees.common.Status.SUCCESS` and others are
#       :py:data:`~py_trees.common.Status.RUNNING`
#     * Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnSelected`
#       only returns :py:data:`~py_trees.common.Status.SUCCESS` if a **specified subset**
#       of children return :py:data:`~py_trees.common.Status.SUCCESS`
#
#     Policies :class:`~py_trees.common.ParallelPolicy.SuccessOnAll` and
#     :class:`~py_trees.common.ParallelPolicy.SuccessOnSelected` may be configured to be
#     *synchronised* in which case children that tick with
#     :data:`~py_trees.common.Status.SUCCESS` will be skipped on subsequent ticks until
#     the policy criteria is met, or one of the children returns
#     status :data:`~py_trees.common.Status.FAILURE`.
#
#     Parallels with policy :class:`~py_trees.common.ParallelPolicy.SuccessOnSelected` will
#     check in both the :meth:`~py_trees.behaviour.Behaviour.setup` and
#     :meth:`~py_trees.behaviour.Behaviour.tick` methods to to verify the
#     selected set of children is actually a subset of the children of this parallel.
#
#     .. seealso::
#        * :ref:`Context Switching Demo <py-trees-demo-context-switching-program>`
#     """
#     def __init__(self,
#                  name=common.Name.AUTO_GENERATED,
#                  policy=ParallelPolicy.SuccessOnAll(),
#                  children=None
#                  ):
#         """
#         Args:
#             name (:obj:`str`): the composite behaviour name
#             policy (:class:`~py_trees.common.ParallelPolicy`): policy to use for deciding success or otherwise
#             children ([:class:`~py_trees.behaviour.Behaviour`]): list of children to add
#         """
#         super(Parallel, self).__init__(name, children)
#         self.policy = policy
#         self.current_child = None
#
#     def setup(self, *args, **kwargs):
#         """
#         Detect before ticking whether the policy configuration is invalid.
#
#         Args:
#             **kwargs (:obj:`dict`): distribute arguments to this
#                behaviour and in turn, all of it's children
#
#         Raises:
#             RuntimeError: if the parallel's policy configuration is invalid
#             Exception: be ready to catch if any of the children raise an exception
#         """
#         self.logger.debug("%s.setup()" % (self.__class__.__name__))
#         self.validate_policy_configuration()
#
#     def tick(self):
#         """
#         Tick over the children.
#
#         Yields:
#             :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
#
#         Raises:
#             RuntimeError: if the policy configuration was invalid
#         """
#         self.logger.debug("%s.tick()" % self.__class__.__name__)
#         self.validate_policy_configuration()
#
#         # reset
#         if self.status != common.Status.RUNNING:
#             self.logger.debug("%s.tick(): re-initialising" % self.__class__.__name__)
#             for child in self.children:
#                 # reset the children, this ensures old SUCCESS/FAILURE status flags
#                 # don't break the synchronisation logic below
#                 if child.status != common.Status.INVALID:
#                     child.stop(common.Status.INVALID)
#             self.current_child = None
#             # subclass (user) handling
#             self.initialise()
#
#         # nothing to do
#         if not self.children:
#             self.current_child = None
#             self.stop(common.Status.SUCCESS)
#             yield self
#             return
#
#         # process them all first
#         for child in self.children:
#             if self.policy.synchronise and child.status == common.Status.SUCCESS:
#                 continue
#             for node in child.tick():
#                 yield node
#
#         # determine new status
#         new_status = common.Status.RUNNING
#         self.current_child = self.children[-1]
#         try:
#             failed_child = next(child for child in self.children if child.status == common.Status.FAILURE)
#             self.current_child = failed_child
#             new_status = common.Status.FAILURE
#         except StopIteration:
#             if type(self.policy) is ParallelPolicy.SuccessOnAll:
#                 if all([c.status == common.Status.SUCCESS for c in self.children]):
#                     new_status = common.Status.SUCCESS
#                     self.current_child = self.children[-1]
#             elif type(self.policy) is ParallelPolicy.SuccessOnOne:
#                 successful = [child for child in self.children if child.status == common.Status.SUCCESS]
#                 if successful:
#                     new_status = common.Status.SUCCESS
#                     self.current_child = successful[-1]
#             elif type(self.policy) is common.ParallelPolicy.SuccessOnSelected:
#                 if all([c.status == common.Status.SUCCESS for c in self.policy.children]):
#                     new_status = common.Status.SUCCESS
#                     self.current_child = self.policy.children[-1]
#             else:
#                 raise RuntimeError("this parallel has been configured with an unrecognised policy [{}]".format(type(self.policy)))
#         # this parallel may have children that are still running
#         # so if the parallel itself has reached a final status, then
#         # these running children need to be terminated so they don't dangle
#         if new_status != common.Status.RUNNING:
#             self.stop(new_status)
#         self.status = new_status
#         yield self
#
#     def stop(self, new_status=common.Status.INVALID):
#         """
#         For interrupts or any of the termination conditions, ensure that any
#         running children are stopped.
#
#         Args:
#             new_status : the composite is transitioning to this new status
#         """
#         # clean up dangling (running) children
#         for child in self.children:
#             if child.status == common.Status.RUNNING:
#                 # interrupt it exactly as if it was interrupted by a higher priority
#                 child.stop(common.Status.INVALID)
#         # only nec. thing here is to make sure the status gets set to INVALID if
#         # it was a higher priority interrupt (new_status == INVALID)
#         Composite.stop(self, new_status)
#
#     def validate_policy_configuration(self):
#         """
#         Policy configuration can be invalid if:
#
#         * Policy is SuccessOnSelected and no behaviours have been specified
#         * Policy is SuccessOnSelected and behaviours that are not children exist
#
#         Raises:
#             RuntimeError: if policy configuration was invalid
#         """
#         if type(self.policy) is ParallelPolicy.SuccessOnSelected:
#             if not self.policy.children:
#                 error_message = ("policy SuccessOnSelected requires a non-empty "
#                                  "selection of children [{}]".format(self.name))
#                 self.logger.error(error_message)
#                 raise RuntimeError(error_message)
#             missing_children_names = [child.name for child in self.policy.children if child not in self.children]
#
#             if missing_children_names:
#                 error_message = ("policy SuccessOnSelected has selected behaviours that are "
#                                  "not children of this parallel {}[{}]""".format(missing_children_names, self.name))
#                 self.logger.error(error_message)
#                 raise RuntimeError(error_message)

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
        self.logger.debug("%s.tick()" % self.__class__.__name__)
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