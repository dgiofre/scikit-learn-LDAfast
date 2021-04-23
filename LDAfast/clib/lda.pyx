#cython: language_level=3, profile=True, boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True

# import cython
import numpy as np
# from cython.parallel import prange

cimport numpy as np


# np.import_array()

from libc.math cimport exp, fabs, log




#from sklearn.decomposition._online_lda_fast import (mean_change, _dirichlet_expectation_1d, _dirichlet_expectation_2d)

cdef double EPS = 2.220446049250313e-16
cdef double EULER = 0.57721566490153286060651209
cdef double MIN_FLOAT = 1.0e-7




# @cython.boundscheck(False)  # Deactivate bounds checking
# @cython.wraparound(False)
cpdef  _update_doc_distribution_cython(float[:,::1]  X, double[:,::1] exp_topic_word_distr,  float doc_topic_prior,
                              int max_iters,
                             float mean_change_tol,  bint cal_sstats,  object random_state):
    
    
    cdef size_t n_samples = X.shape[0]
    cdef size_t n_topics = exp_topic_word_distr.shape[0]
    cdef size_t n_features = exp_topic_word_distr.shape[1]
    

    cdef double[:,::1] suff_stats #cvarray(shape=(n_topics, n_features), itemsize=sizeof(double), format="d") 

    ### init for c-compiler
    cdef size_t idx_d, i, j
    cdef float[::1] cnts 
    cdef double[::1] doc_topic_d 
    cdef double[::1] exp_doc_topic_d 


    
    
    cdef double[::1] last_d
    cdef double[:, ::1] doc_topic_distr #= cvarray(shape=(n_samples, n_topics), itemsize=sizeof(double), format="d")
    cdef double[::1] norm_phi = np.empty((n_features,), dtype=np.double)
    cdef double[::1] division = np.empty((n_features,), dtype=np.double)
    cdef double[::1] dot_prod = np.empty((n_topics,), dtype=np.double)

    if cal_sstats:
        doc_topic_distr = random_state.gamma(100., 0.01, (n_samples, n_topics)) 
    else:
        doc_topic_distr = np.ones((n_samples, n_topics), dtype=np.double)
    

    # In the literature, this is `exp(E[log(theta)])`
    cdef double[:, ::1] doc_topic_dirichelt = np.empty_like(doc_topic_distr)
    cdef double[:, ::1] exp_doc_topic = np.empty_like(doc_topic_dirichelt)

    _dirichlet_expectation_2d_c(doc_topic_distr, doc_topic_dirichelt)
    exp_2d(doc_topic_dirichelt, exp_doc_topic)


    # diff on `component_` (only calculate it when `cal_diff` is True)
    if cal_sstats:
        suff_stats = np.zeros((n_topics, n_features), dtype=np.double)
    else:
        suff_stats = np.empty((n_topics, n_features), dtype=np.double)
    

    
    for idx_d in range(n_samples):#range(n_samples):#
        
        cnts = X[idx_d, :]
        doc_topic_d = doc_topic_distr[idx_d, :]
        # The next one is a copy, since the inner loop overwrites it.
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        # Iterate between `doc_topic_d` and `norm_phi` until convergence
        for _ in range(0, max_iters):
            last_d  = doc_topic_d.copy()

            # The optimal phi_{dwk} is proportional to
            # exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]).
            norm_phi=dotvm_c(exp_doc_topic_d, exp_topic_word_distr)    
            divide1d_c(cnts,norm_phi, division)
            dotvm_f(division, exp_topic_word_distr.T, exp_doc_topic_d, doc_topic_d)
            # for i in range(n_topics):#prange(n_topics,schedule="guided", nogil=True):
            #     doc_topic_d[i] = exp_doc_topic_d[i] * dot_prod[i]

            # Note: adds doc_topic_prior to doc_topic_d, in-place.
            _dirichlet_expectation_1d_c(doc_topic_d, doc_topic_prior,
                                      exp_doc_topic_d)

            if mean_change_c(last_d, doc_topic_d) < mean_change_tol:
                break
        # doc_topic_distr[idx_d, :] = doc_topic_d
        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_sstats:
            norm_phi=dotvm_c(exp_doc_topic_d, exp_topic_word_distr)  
            divide1d_c(cnts,norm_phi, division)  
            outervv_add_c(exp_doc_topic_d,division, suff_stats)


    return (np.asarray(doc_topic_distr), np.asarray(suff_stats))






cdef double mean_change_c(double[:] arr_1,
                double[:] arr_2) nogil:
    """Calculate the mean difference between two arrays.

    Equivalent to np.abs(arr_1 - arr2).mean().
    """

    cdef double total, diff
    cdef size_t i, size  #np.npy_intp

    size = arr_1.shape[0]
    total = 0.0
    for i in range(size):
        diff = fabs(arr_1[i] - arr_2[i])
        total += diff

    return total / size

# @cython.cdivision(True)
cdef void _dirichlet_expectation_1d_c(double[::1] doc_topic,
                              double doc_topic_prior,
                              double[::1] out) nogil:
    """Dirichlet expectation for a single sample:
        exp(E[log(theta)]) for theta ~ Dir(doc_topic)
    after adding doc_topic_prior to doc_topic, in-place.

    Equivalent to
        doc_topic += doc_topic_prior
        out[:] = np.exp(psi(doc_topic) - psi(np.sum(doc_topic)))
    """

    cdef np.float64_t dt, psi_total, total
    cdef size_t i, size

    size = doc_topic.shape[0]

    total = 0.0
    for i in range(size):
        dt = doc_topic[i] + doc_topic_prior
        doc_topic[i] = dt
        total += dt
    psi_total = psi(total)

    for i in range(size): ##prange(size, schedule='guided'):#range(size): ##
        out[i] = exp(psi(doc_topic[i]) - psi_total)

# @cython.cdivision(True)
cdef void  _dirichlet_expectation_2d_c(double[:,::1] arr, double[:,::1] d_exp) nogil:  #without inline
    """Dirichlet expectation for multiple samples:
    E[log(theta)] for theta ~ Dir(arr).

    Equivalent to psi(arr) - psi(np.sum(arr, axis=1))[:, np.newaxis].

    Note that unlike _dirichlet_expectation_1d, this function doesn't compute
    the exp and doesn't add in the prior.
    """
    cdef double row_total, psi_row_total
    cdef size_t i, j, n_rows, n_cols

    n_rows = arr.shape[0]
    n_cols = arr.shape[1]

    # cdef double[:,::1] d_exp = np.empty_like(arr)#cvarray(shape=(n_rows, n_cols), itemsize=sizeof(double), format="d")
    # with nogil:
    for i in range(n_rows):
        row_total = 0
        for j in range(n_cols):
            row_total += arr[i, j]
        psi_row_total = psi(row_total)

        for j in range(n_cols):
            d_exp[i, j] = psi(arr[i, j]) - psi_row_total

    # return d_exp


# @cython.cdivision(True)
cdef double[::1]   dotvm_c(double[::1] A, double[:, ::1] B):
    '''matrix multiply matrices A (n) and B (n x l)

    Parameters
    ----------
    A : memoryview (numpy array)
        n  left vector
    B : memoryview (numpy array)
        n x l right matrix
    out : memoryview (numpy array)
        l output vector
    '''
    cdef size_t i, j
    cdef double s
    cdef size_t n = A.shape[0]
    cdef size_t l = B.shape[1]

    cdef double[::1] out = np.zeros((l,), dtype=np.double)#cvarray(shape=(l,), itemsize=sizeof(double), format="d") 

    for i in range(n):
        if A[i]>MIN_FLOAT:
            for j in range(l):
                out[j] += A[i]*B[i, j]

    return out

cdef void dotvm_f(double[::1] A, double[::1, :] B, double[::1] C, double[::1] out) nogil:
    '''matrix multiply matrices A (n) and B (n x l)

    Parameters
    ----------
    A : memoryview (numpy array)
        n  left vector
    B : memoryview (numpy array)
        n x l right matrix
    C : memoryview (numpy array)
        l product vector
    out : memoryview (numpy array)
        l output vector
    '''
    cdef size_t i, j
    cdef double s
    cdef size_t n = A.shape[0]
    cdef size_t l = B.shape[1]

    # cdef double[::1] out = np.empty((l,), dtype=np.double)#cvarray(shape=(l,), itemsize=sizeof(double), format="d") 

    for j in range(l):
        s = 0
        if C[j]>MIN_FLOAT:
            for i in range(n):#range(n):
                s += A[i]*B[i, j]

            out[j] = s * C[j]
        else:
            out[j] = s

# @cython.cdivision(True)
cdef  void outervv_add_c(double[::1] A, double[::1] B, double[:,::1] C) nogil:
    '''vector outer product vector A (m) and vector B (n) and add to matrices C (m x n)

    Parameters
    ----------
    A : memoryview (numpy array)
        m  left vector
    B : memoryview (numpy array)
        n right vector
    C : memoryview (numpy array)
        m x n output matrices summed with the outer product of A and B
    '''
    cdef size_t i, j
    cdef size_t m = A.shape[0]
    cdef size_t n = B.shape[0]

    # cdef double[:,::1] C = cvarray(shape=(m, n), itemsize=sizeof(double), format="d") 
    for i in range(m):#prange(m, schedule="guided"):#range(m):#
        if A[i]> MIN_FLOAT:
            for j in range(n):
                C[i,j] += A[i]*B[j]

# @cython.cdivision(True)
cdef void divide1d_c(float[::1] A, double[::1] B, double[::1] out) nogil:
    '''array division array A and B same size n

    Parameters
    ----------
    A : memoryview array
        n dividend array
    B : memoryview array
        n divisor array
    out : memoryview array
        n output vector
    '''

    cdef Py_ssize_t i
    cdef Py_ssize_t n = A.shape[0]

    # cdef double[::1] out = np.empty((n,), dtype=np.double) #cvarray(shape=(n, ), itemsize=sizeof(double), format="d")
    for i in range(n):#range(n):
        out[i]= A[i]/(B[i]+EPS)

    # return out

# @cython.cdivision(True)
cdef void  exp_2d(double[:,::1] arr_2D, double[:,::1] exp_arr_2D) nogil:
    '''array 2D exponential via single element 
    Parameters
    ----------
    arr_2D : memoryview array (size1 x size2)
             input vector
    exp_arr_2D : memoryview array (size1 x size2)
             output vector exponential array (size1 x size2)
    '''
    cdef size_t i,j, size1, size2
    size1 = arr_2D.shape[0]
    size2 = arr_2D.shape[1]
    
    # cdef double[:,::1] exp_arr_2D = np.empty_like(arr_2D) #cvarray(shape=(size1, size2), itemsize=sizeof(double), format="d") 
    for i in range(size1):#prange(size1, schedule='dynamic'):#
        for j in range(size2):
            exp_arr_2D[i,j]=exp(arr_2D[i,j])

    # return exp_arr_2D;


# Psi function for positive arguments. Optimized for speed, not accuracy.
#
# After: J. Bernardo (1976). Algorithm AS 103: Psi (Digamma) Function.
# https://www.uv.es/~bernardo/1976AppStatist.pdf
# @cython.cdivision(True)
cdef double psi(double x) nogil:
    if x <= 1e-6:
        # psi(x) = -EULER - 1/x + O(x)
        return -EULER - 1. / x

    cdef double r, result = 0

    # psi(x + 1) = psi(x) + 1/x
    while x < 6:
        result -= 1. / x
        x += 1

    # psi(x) = log(x) - 1/(2x) - 1/(12x**2) + 1/(120x**4) - 1/(252x**6)
    #          + O(1/x**8)
    r = 1. / x
    result += log(x) - .5 * r
    r = r * r
    result -= r * ((1./12.) - r * ((1./120.) - r * (1./252.)))
    return result;


