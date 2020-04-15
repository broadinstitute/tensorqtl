# Author: Francois Aguet
import numpy as np
import pandas as pd
from collections import Iterable
from contextlib import contextmanager

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri, numpy2ri
numpy2ri.activate()
pandas2ri.activate()


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
    rp = ro.vectors.FloatVector(p)
    p_adjust = ro.r['p.adjust']
    return np.array(p_adjust(rp, method=method))


def qvalue(p, lambda_qvalue=None):
    """Wrapper for qvalue::qvalue"""
    qvalue = importr("qvalue")
    rp = ro.vectors.FloatVector(p)
    if lambda_qvalue is None:
        q = qvalue.qvalue(rp)
    else:
        if not isinstance(lambda_qvalue, Iterable):
            lambda_qvalue = [lambda_qvalue]
        rlambda = ro.vectors.FloatVector(lambda_qvalue)
        q = qvalue.qvalue(rp, **{'lambda':rlambda})
    qval = np.array(q.rx2('qvalues'))
    pi0 = np.array(q.rx2('pi0'))[0]
    return qval, pi0


def pi0est(p, lambda_qvalue=None):
    """Wrapper for qvalue::pi0est"""
    qvalue = importr("qvalue")
    rp = ro.vectors.FloatVector(p)
    # with suppress_stdout():
    if lambda_qvalue is None:
        pi0res = qvalue.pi0est(rp)
    else:
        if not isinstance(lambda_qvalue, Iterable):
            lambda_qvalue = [lambda_qvalue]
        rlambda = ro.vectors.FloatVector(lambda_qvalue)
        pi0res = qvalue.pi0est(rp, rlambda)
    pi0 = np.array(pi0res.rx2('pi0'))[0]
    pi0_lambda = np.array(pi0res.rx2('pi0.lambda'))
    lambda_vec = np.array(pi0res.rx2('lambda'))
    pi0_smooth = np.array(pi0res.rx2('pi0.smooth'))
    return pi0, pi0_lambda, lambda_vec, pi0_smooth


def impute_knn(df):
    """Wrapper for impute::impute.knn"""
    impute = importr("impute")
    impute_knn_ = ro.r['impute.knn']
    as_matrix = ro.r['as.matrix']

    with localconverter(ro.default_converter + pandas2ri.converter):
        rdf = ro.conversion.py2rpy(df)
    res = impute_knn_(as_matrix(rdf))

    imputed_df = pd.DataFrame(np.array(res[0]), index=df.index, columns=df.columns)
    seed = np.array(res[1])[0]
    return imputed_df
