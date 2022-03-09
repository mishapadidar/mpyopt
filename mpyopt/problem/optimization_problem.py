import abc
from abc import ABC, abstractmethod

class OptimizationProblem(object):
    """Base class for optimization problems."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.dim = None
        self.lb = None
        self.ub = None
        self.minimum = None

    def __check_input__(self, x):
        if len(x) != self.dim:
            raise ValueError("Dimension mismatch")

    def __call__(self,xx):
      return self.eval(xx)

    @abstractmethod
    def eval(self,xx):  # pragma: no cover
        raise NotImplementedError()

    @abstractmethod
    def grad(self,xx): 
        raise NotImplementedError()

    @abstractmethod
    def hess(self,xx): 
        raise NotImplementedError()

