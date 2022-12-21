# Author: Francois Aguet
import numpy as np
import rpy2
from rpy2.robjects.packages import importr
from collections.abc import Iterable
from contextlib import contextmanager

# silence R warnings
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def p_adjust(p, method='BH'):
    """Wrapper for p.adjust"""
    rp = rpy2.robjects.vectors.FloatVector(p)
    p_adjust = rpy2.robjects.r['p.adjust']
    return np.array(p_adjust(rp, method=method))


def qvalue(p, lambda_qvalue=None):
    """Wrapper for qvalue::qvalue"""
    qvalue = importr("qvalue")
    rp = rpy2.robjects.vectors.FloatVector(p)
    if lambda_qvalue is None:
        q = qvalue.qvalue(rp)
    else:
        if not isinstance(lambda_qvalue, Iterable):
            lambda_qvalue = [lambda_qvalue]
        rlambda = rpy2.robjects.vectors.FloatVector(lambda_qvalue)
        q = qvalue.qvalue(rp, **{'lambda':rlambda})
    qval = np.array(q.rx2('qvalues'))
    pi0 = np.array(q.rx2('pi0'))[0]
    return qval, pi0


def pi0est(p, lambda_qvalue=None):
    """Wrapper for qvalue::pi0est"""
    qvalue = importr("qvalue")
    rp = rpy2.robjects.vectors.FloatVector(p)
    # with suppress_stdout():
    if lambda_qvalue is None:
        pi0res = qvalue.pi0est(rp)
    else:
        if not isinstance(lambda_qvalue, Iterable):
            lambda_qvalue = [lambda_qvalue]
        rlambda = rpy2.robjects.vectors.FloatVector(lambda_qvalue)
        pi0res = qvalue.pi0est(rp, rlambda)
    pi0 = np.array(pi0res.rx2('pi0'))[0]
    pi0_lambda = np.array(pi0res.rx2('pi0.lambda'))
    lambda_vec = np.array(pi0res.rx2('lambda'))
    pi0_smooth = np.array(pi0res.rx2('pi0.smooth'))
    return pi0, pi0_lambda, lambda_vec, pi0_smooth
