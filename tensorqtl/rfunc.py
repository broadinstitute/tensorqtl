# Author: Francois Aguet
import rpy2
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr


def p_adjust(p, method='BH'):
    """Wrapper for p.adjust"""
    rp = rpy2.robjects.vectors.FloatVector(p)
    r_padjust = robjects.r['p.adjust']
    return rpy2.robjects.numpy2ri.ri2py(r_padjust(rp, method=method))


def qvalue(p, lambda_qvalue=None):
    """Wrapper for qvalue::qvalue"""
    qvalue = importr("qvalue")
    rp = rpy2.robjects.vectors.FloatVector(p)
    if lambda_qvalue is None:
        q = qvalue.qvalue(rp)
    else:
        rlambda = rpy2.robjects.vectors.FloatVector(lambda_qvalue)
        q = qvalue.qvalue(rp, **{'lambda':rlambda})
    qval = rpy2.robjects.numpy2ri.ri2py(q.rx2('qvalues'))
    pi0 = rpy2.robjects.numpy2ri.ri2py(q.rx2('pi0'))[0]
    return qval, pi0


def pi0est(p, lambda_qvalue=None):
    """Wrapper for qvalue::pi0est"""
    qvalue = importr("qvalue")
    rp = rpy2.robjects.vectors.FloatVector(p)
    if lambda_qvalue is None:
        pi0res = qvalue.pi0est(rp)
    else:
        rlambda = rpy2.robjects.vectors.FloatVector(lambda_qvalue)
        pi0res = qvalue.pi0est(rp, rlambda)
    pi0 = rpy2.robjects.numpy2ri.ri2py(pi0res.rx2('pi0'))
    pi0_lambda = rpy2.robjects.numpy2ri.ri2py(pi0res.rx2('pi0.lambda'))
    lambda_vec = rpy2.robjects.numpy2ri.ri2py(pi0res.rx2('lambda'))
    pi0_smooth = rpy2.robjects.numpy2ri.ri2py(pi0res.rx2('pi0.smooth'))
    return pi0, pi0_lambda, lambda_vec, pi0_smooth
