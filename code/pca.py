#!/usr/bin/env python3
"""
This is a utility to compute the PCA modes of a given data set out of the
correlation matrix.

All inner-product heavy parts are multithreaded, see n_jobs in stages 2 and 3.
It is up to the user to implement the functions `state` and `inprod` which are
problem specific.
These implementations should *not* spawn their own threads.
See the rest of the documentation for details.

Data is passed around in its original type.
PCA modes are returned in the type of the data.
Right singular vectors are returned as NumPy vectors.
"""
from sys import exit

import numpy as np
import psutil
from joblib import Parallel, delayed
from scipy.linalg import svd
from tqdm import tqdm

# Use one fewer than the number of total cores available to multithread
def_n_jobs = len(psutil.Process().cpu_affinity())
if def_n_jobs > 1:
    def_n_jobs = def_n_jobs - 1

# If the inner product doesn't release GIL, do NOT use "threading", instead
# use "loky"
def_joblib_backend = "loky"
def_joblib_verbosity = 0


def stage1(numSnaps, state, inprod, doAvg=True, printMessages=False):
    """Computes the average of the state vectors if demanded, otherwise returns
    None for compatibility with the rest of the code.

    Parameters
    ----------
    numSnaps : int
        Number of state vectors.
    state : function
        Function that returns a state vector given an index.
        It should have the signature
            state(i, doAvg, avgState),
        see __main__ for an example.
    inprod : function
        Function that returns the inner product of two state vectors.
        It should have the calling signature
            inprod(stateI, stateJ),
        see __main__ for an example.
    doAvg : bool, optional
        Compute the average of states (True) or not (False).
    printMessages : bool, optional
        Whether to print informative / debugging messages.

    Returns
    -------
    ndarray
        If `doAvg`, the average of the state vectors. None otherwise.

    """

    # Getting started
    printIf(printMessages, "PCA stage 1.")

    # Allocate its array and work on it if so
    if doAvg:
        printIf(printMessages, "Calculating the average state.")
        avgState = state(0, doAvg=False, avgState=None)
        # Find out the average vector
        for i in range(1, numSnaps):
            # As the average is not calculated yet, be careful not to pass its flags
            avgState = avgState + state(i, doAvg=False, avgState=None)
        avgState = (1 / numSnaps) * avgState
        # Compute the norm squared of the average state
        avgNorm = inprod(avgState, avgState)
        printIf(printMessages, f"The average state has norm squared: {avgNorm:.4e}.")

    # For easy compatibility with the state and inprod functions
    else:
        avgState = None

    printIf(printMessages, "PCA stage 1 done.")

    return avgState


def stage2(
    numSnaps,
    state,
    inprod,
    avgState,
    doAvg=True,
    c_sigma=0.999,
    c_chi=None,
    eps=10 ** (-8),
    svdDebug=True,
    returnAll=False,
    n_jobs=def_n_jobs,
    joblib_backend=def_joblib_backend,
    joblib_verbosity=def_joblib_verbosity,
    printMessages=False,
    corrMatrix=None,
    keepCorrMatrix=False,
    nModesManual=None,
    primes=None,
):
    """Computes the correlation matrix out of the state vectors, takes it SVD,
    and returns the right singular vectors of the correlation matrix as columns
    of a matrix together with the singular values, both up to a cut off.

    Parameters
    ----------
    numSnaps : int
        Number of state vectors.
    state : function
        Function that returns a state vector given an index.
        It should have the signature
            state(i, doAvg, avgState),
        see __main__ for an example.
    inprod : function
        Function that returns the inner product of two state vectors.
        It should have the calling signature
            inprod(stateI, stateJ),
        see __main__ for an example.
    avgState : abstract state
        Average of the state vectors if the average is to be subtracted,
        None otherwise.
    doAvg : bool, optional
        Compute the average of states (True) or not (False).
    c_sigma : float, optional
        Singular values are cut off at the index where the sum of their squares
        exceed this value proportional to the total sum and when `c_chi`, see
        next, is also satisfied.
        Ignored if `nModesManual` is provided, see below.
    c_chi : float, optional
        If provided, the average of the ignored singular values needs to be
        smaller than the first singular value to this ratio.
        Ignored if `nModesManual` is provided, see below.
    eps : float, optional
        A near-zero value, for comparison against zero.
    svdDebug : bool, optional
        Check the SVD procedure with some tests (True) or not (False).
        Ignored if `nModesManual` is provided, see below.
    returnAll : bool
        Return all singular values and modes up to `eps` precision checking
        their orthonormality, bypassing cutoffs.
        Ignored if `nModesManual` is provided, see below.
    n_jobs : int
        Number of threads to use for the `inprod` heavy parts.
        Advised is never to go above the number of *physical* cores minus one.
    printMessages : bool, optional
        Whether to print informative / debugging messages.
    corrMatrix : ndarray, optional
        If provided, use it as the correlation matrix instead of computing it.
    keepCorrMatrix : bool, optional
        Whether to return the correlation matrix.
    nModesManual : int, optional
        If provided, return exactly `nModesManual` right singular vectors,
        bypassing checks on them.

    Returns
    -------
    tuple
        Tuple of (v, singValsSq) or (v, singValsSq, corrMatrix).
        The right singular vectors of the
        correlation matrix as columns and the singular values squared of the
        data matrix, former up to a cut off, the latter with the complete
        set.
        corrMatrix is returned if keepCorrMatrix==True.

    """
    printIf(printMessages, "PCA stage 2.")
    printIf(printMessages, f"Using {n_jobs} threads.")

    dataType = np.asarray(inprod(state(0), state(1))).dtype
    if corrMatrix is None:
        # Allocate the correlation matrix
        # LAPACK routines expect Fortran contigous arrays
        # Correlation matrix
        # Get the data type
        corrMatrix = np.zeros((numSnaps, numSnaps), dtype=dataType, order="F")
        printIf(printMessages, f"Calculating the correlation matrix.")

        if primes is None:
            symmetryModulus = 1
            # numPrimeSnaps = numSnaps
            # nTot = (numSnaps ** 2 + numSnaps) // 2
        else:
            symmetryModulus = len(primes)
            # numPrimeSnaps = numSnaps // symmetryModulus
            # nTot = ((2 * symmetryModulus - 1) * numPrimeSnaps ** 2 + numPrimeSnaps) // 2

        nInprods = (
            (1 + (numSnaps - 1) // symmetryModulus)
            * (2 * numSnaps - symmetryModulus * ((numSnaps - 1) // symmetryModulus))
            // 2
        )

        # Given i, for each (j>=i) create a job
        def fillCorrMatrix(i, j, corrMatrix, pbar):
            stateI = state(i, doAvg=doAvg, avgState=avgState)

            if j == i:
                stateJ = stateI
            else:
                stateJ = state(j, doAvg=doAvg, avgState=avgState)

            norm = inprod(stateI, stateJ)
            # Place the result into the correlation matrix
            # which is symmetric
            corrMatrix[i, j] = norm
            corrMatrix[j, i] = np.conj(norm)
            pbar.update()

        with tqdm(total=nInprods, disable=not printMessages) as pbar:
            Parallel(n_jobs=n_jobs, backend=joblib_backend, verbose=joblib_verbosity)(
                delayed(fillCorrMatrix)(i, j, corrMatrix, pbar)
                for i in range(0, numSnaps, symmetryModulus)
                for j in range(i, numSnaps)
            )

        # If using tricks for symmetry copies, compute the rest
        # The rest follow from invariance of the inner product
        if primes is not None:
            for i in range(0, numSnaps):
                iL = i % symmetryModulus
                if iL != 0:
                    for j in range(i, numSnaps):
                        iR = j % symmetryModulus
                        pL = primes[iL]
                        pR = primes[iR]

                        # Factorize
                        for p in primes[1:]:
                            if (not (pL == 1 or pR == 1)) and (
                                pL % p == 0 and pR % p == 0
                            ):
                                pL = pL // p
                                pR = pR // p
                        shift = primes.index(pL * pR)
                        norm = corrMatrix[i - iL, j - iR + shift]
                        corrMatrix[i, j] = norm
                        corrMatrix[j, i] = np.conj(norm)
    else:
        assert corrMatrix.shape == (numSnaps, numSnaps)
        assert corrMatrix.dtype == dataType

    printIf(printMessages, "Calculating the SVD of the correlation matrix.")

    # Take the SVD of the correlation matrix
    # GESDD is supposedly faster, but numerically less stable.
    v, singValsSq, vT = svd(corrMatrix, lapack_driver="gesvd")
    # Note that this gives v (and vT) in Fortran contigous form

    # Free memory
    if not keepCorrMatrix:
        del corrMatrix
    del vT

    # Cut at nModesManual if provided
    if nModesManual is not None:
        if nModesManual > numSnaps:
            exit("Number of PCA modes cannot be larger than the number of snapshots.")
        if nModesManual <= 0:
            exit(f"nModesManual={nModesManual} doesn't make sense")
        else:
            v = v[:, :nModesManual]
    else:
        # Will compute the cutting index
        pcaCutIndex = numSnaps
        if not returnAll:
            if c_chi is not None:
                printIf(
                    printMessages,
                    f"Finding the cut-off index. Set fractions are {c_sigma:.4f} and {c_chi:.4f}.",
                )
            else:
                printIf(
                    printMessages,
                    f"Finding the cut-off index. Set fraction is {c_sigma:.4f}.",
                )
            # Decide on where to cut the expansion
            sumEigAll = np.sum(singValsSq)
            sumEigCut = 0.0
            # Add up \sigma^2 until the thresholds are met
            for i in range(numSnaps):
                sumEigCut = sumEigCut + singValsSq[i]
                # Test the thresholds
                if c_chi is not None:
                    if (
                        sumEigCut / sumEigAll > c_sigma
                        and i + 1 != numSnaps
                        and np.average(singValsSq[i + 1 :] / singValsSq[0]) < c_chi
                    ):
                        pcaCutIndex = i + 1
                        break
                else:
                    if sumEigCut / sumEigAll > c_sigma and i + 1 != numSnaps:
                        pcaCutIndex = i + 1
                        break

        # Sanity checks
        if pcaCutIndex == 0:
            exit(f"pcaCutIndex = 0, quitting.")

        if not returnAll:
            printIf(
                printMessages, f"Last included (zero-based) index is {pcaCutIndex - 1}."
            )

        # Construct the cut right singular eigenvectors
        v = v[:, :pcaCutIndex]

        if svdDebug:
            printIf(
                printMessages, "Testing the right singular vectors for orthonormality."
            )
            # Construct the test matrix
            testMatrix = np.matmul(v.T.conj(), v, order="F")

            # The diagonals should all be 1
            for i in range(pcaCutIndex):
                if np.abs(testMatrix[i, i] - 1) > eps:
                    exit(
                        f"Diagonal test failed at {i+1} (s2 = {singValsSq[i]:.4e}) of {numSnaps} with error {np.abs(testMatrix[i, i] - 1)/eps:.4e}."
                    )
            # The rest should be zero
            for i in range(pcaCutIndex):
                for j in range(i + 1, pcaCutIndex):
                    if np.abs(testMatrix[i, j]) > eps:
                        exit(
                            f"Off-diagonal test failed at {i+1} (s2 = {singValsSq[i]:.4e}), {j} (s2 = {singValsSq[j]:.4e}) of {numSnaps} with error {np.abs(testMatrix[i, j])/eps:.4e}."
                        )

    printIf(printMessages, "PCA stage 2 done.")

    if not keepCorrMatrix:
        return v, singValsSq, None
    else:
        return v, singValsSq, corrMatrix


def stage3(
    numSnaps,
    state,
    inprod,
    avgState,
    vCut,
    singValsSq,
    doAvg=True,
    svdDebug=True,
    eps=10 ** (-8),
    n_jobs=def_n_jobs,
    joblib_backend=def_joblib_backend,
    joblib_verbosity=def_joblib_verbosity,
    printMessages=False,
    nModesManual=None,
):
    """Computes the PCA modes given the data matrix, the right singular vectors
    of the correlation matrix and the singular values squared of the data
    matrix, all up to a cut off.

    Parameters
    ----------
    numSnaps : int
        Number of state vectors.
    state : function
        Function that returns a state vector given an index.
        It should have the signature
            state(i, doAvg, avgState),
        see __main__ for an example.
    inprod : function
        Function that returns the inner product of two state vectors.
        It should have the calling signature
            inprod(stateI, stateJ),
        see __main__ for an example.
    avgState : abstract state
        Average of the state vectors if the average is to be subtracted,
        None otherwise.
    vCut : ndarray
        Right singular vectors of the correlation matrix in columns.
        Of shape (numSnaps, cutOff).
    singValsSq : ndarray
        Singular eigenvalues squared of the data matrix.
        Of shape (>= cutOff).
    doAvg : bool, optional
        Compute the average of states (True) or not (False).
    svdDebug : bool, optional
        Check the SVD procedure with some tests (True) or not (False).
        Ignored if `nModesManual` is provided, see below.
    eps: float, optional
        Epsilon for the various sanity checks.
    n_jobs : int
        Number of threads to use for the `inprod` heavy parts.
        Advised is never to go above the number of *physical* cores minus one.
    printMessages : bool, optional
        Whether to print informative / debugging messages.
    nModesManual : int, optional
        If provided, return exactly `nModesManual` PCA modes,
        bypassing checks on them.

    Returns
    -------
    dict
        PCA modes of the state vectors up to a cut off.

    """

    printIf(printMessages, "PCA stage 3.")
    printIf(printMessages, f"Using {n_jobs} threads.")

    printIf(printMessages, "Calculating the PCA modes.")

    pcaModes = {}

    pcaCutIndex = vCut.shape[-1]
    if nModesManual is not None:
        if nModesManual > pcaCutIndex:
            exit(f"nModesManual={nModesManual} > pcaCutIndex={pcaCutIndex}.")
        if nModesManual <= 0:
            exit(f"nModesManual={nModesManual} doesn't make sense")
        else:
            pcaCutIndex = nModesManual

    with tqdm(total=numSnaps, disable=not printMessages) as pbar:
        for i in range(numSnaps):
            # Load the i'th snapshot
            snap = state(i, doAvg=doAvg, avgState=avgState)
            for j in range(pcaCutIndex):
                # Add up its contributions to the final output
                if i == 0:
                    pcaModes[j] = vCut[i, j] * snap
                else:
                    pcaModes[j] += vCut[i, j] * snap
            pbar.update()

    ###########################################################################

    for i in range(pcaCutIndex):
        pcaModes[i] = (1 / np.sqrt(singValsSq[i])) * pcaModes[i]

    # Sanity checks
    if (nModesManual is None) and svdDebug:
        printIf(printMessages, "Checking the orthonormality of the PCA modes.")
        dataType = np.asarray(inprod(state(0), state(1))).dtype

        # Test the orthonormality of the PCA modes
        testMatrix = np.zeros((pcaCutIndex, pcaCutIndex), dtype=dataType, order="F")

        # For each (i, j>=i) create a job, they are all in memory
        def fillTestMatrix(i, j, pbar):
            norm = inprod(pcaModes[i], pcaModes[j])
            testMatrix[i, j] = norm
            testMatrix[j, i] = np.conj(norm)
            pbar.update()

        with tqdm(
            total=(pcaCutIndex**2 + pcaCutIndex) // 2, disable=not printMessages
        ) as pbar:
            Parallel(n_jobs=n_jobs, backend=joblib_backend, verbose=joblib_verbosity)(
                delayed(fillTestMatrix)(i, j, pbar)
                for i in range(pcaCutIndex)
                for j in range(i, pcaCutIndex)
            )

        printIf(printMessages, "Deciding where to cut-off for numerical accuracy.")

        numericalCutOff = pcaCutIndex
        # The diagonals should all be 1
        for i in range(pcaCutIndex):
            if np.abs(testMatrix[i, i] - 1) > eps:
                printIf(
                    printMessages,
                    f"Diagonal test failed at index {i+1} (s2 = {singValsSq[i]:.4e}) of {numSnaps} with error {np.abs(testMatrix[i, i] - 1)/eps:.4e}.",
                )

                # Numerical problems appear when including the last few singular
                # values.
                # Cutoff at the first index that fails the diagonal test
                numericalCutOff = i
                break

        # Cutoff index may have changed for the case of no manual cutoffs
        # In which case we trash after the cut off
        for i in range(numericalCutOff, len(pcaModes)):
            del pcaModes[i]
        printIf(
            printMessages,
            f"Cutting at index {numericalCutOff} (s2 = {singValsSq[numericalCutOff-1]:.4e}) of {numSnaps} as decided by the diagonal test.",
        )
        pcaCutIndex = numericalCutOff

        # The rest should be zero
        for i in range(pcaCutIndex):
            for j in range(i + 1, pcaCutIndex):
                if np.abs(testMatrix[i, j]) > eps:
                    printIf(
                        printMessages,
                        f"Off-diagonal test failed at indices {i+1} (s2 = {singValsSq[i]:.4e}), {j+1} (s2 = {singValsSq[j]:.4e}) of {numSnaps} with error {np.abs(testMatrix[i, j])/eps:.4e}.",
                    )

                    # The problematic index is the largest of the two
                    # indices that are tested. It replaces the previous
                    # cutoff if it's smaller than that.
                    numericalCutOff = min(max(i, j), numericalCutOff)

        # Free memory
        del testMatrix

        # Cutoff index may have changed, again, for the case of no manual cutoffs
        # In which case we trash after the cut off
        for i in range(numericalCutOff, len(pcaModes)):
            del pcaModes[i]
        printIf(
            printMessages,
            f"Cutting at index {numericalCutOff} (s2 = {singValsSq[numericalCutOff-1]:.4e}) of {numSnaps} as decided by the off-diagonal test.",
        )

        printIf(printMessages, "Checking the RMS property.")
        # Rms property
        testVector = np.zeros(numSnaps, dtype=dataType, order="F")

        def fillTestVector(i, pbar):
            stateR = state(i, doAvg=doAvg, avgState=avgState)
            norm = inprod(stateR, pcaModes[0])
            testVector[i] = norm
            pbar.update()

        with tqdm(total=numSnaps, disable=not printMessages) as pbar:
            Parallel(n_jobs=n_jobs, backend=joblib_backend, verbose=joblib_verbosity)(
                delayed(fillTestVector)(i, pbar) for i in range(numSnaps)
            )

        rms = np.sqrt(np.sum(testVector * testVector.conj()))
        if np.abs(rms - np.sqrt(singValsSq[0])) > eps:
            exit("Rms test failed.")

        # Free memory
        del testVector

    printIf(printMessages, "PCA stage 3 done.")

    return pcaModes


def printIf(printMessages, message):
    if printMessages:
        print(message, flush=True)
