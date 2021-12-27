#!/usr/bin/env python3
from sys import exit

import numpy as np
import psutil
from joblib import Parallel, delayed
from tqdm import tqdm

import pca

# Use one fewer than the number of total cores available to multithread
def_n_jobs = len(psutil.Process().cpu_affinity())
if def_n_jobs > 1:
    def_n_jobs = def_n_jobs - 1

# If the inner product doesn't release GIL, do NOT use "threading", instead
# use "loky"
def_joblib_backend = "threading"
def_joblib_verbosity = 0

# (Complex) DMD modes and basic arithmetic over them
class dmdMode:
    def __init__(self, x, y):
        self.real = x
        self.imag = y

    def __add__(self, other):
        return dmdMode(self.real + other.real, self.imag + other.imag)

    def __rmul__(self, other):
        # Multiplication with a scalar from the left
        return dmdMode(
            other.real * self.real - other.imag * self.imag,
            other.real * self.imag + other.imag * self.real,
        )


# Inner product over (complex) DMD modes inherited from the inner product
# over state vectors
def dmdInprod(mod1, mod2, inprod):

    # Returns inprod(mod1*, mod2)

    return (
        inprod(mod1.real, mod2.real)
        + inprod(mod1.imag, mod2.imag)
        + 1j * inprod(mod1.real, mod2.imag)
        - 1j * inprod(mod1.imag, mod2.real)
    )


def dmdPinv(
    modes,
    inprod,
    n_jobs=def_n_jobs,
    joblib_backend=def_joblib_backend,
    joblib_verbosity=def_joblib_verbosity,
    svdDebug=True,
    eps=10 ** (-8),
    testInverses=True,
    printMessages=False,
    nModesManual=None,
):
    """Find the "effective inverse" of the input DMD modes.

    Parameters
    ----------
    modes : dict
        DMD modes. Provide exactly the DMD modes that you want to use,
        as this is a "best fit".
    inprod : function
        Function that returns the inner product of two state vectors.
        It should have the calling signature
            inprod(stateI, stateJ).
    n_jobs : int
        Number of threads to use for the `inprod` heavy parts.
        Advised is never to go above the number of *physical* cores minus one.
    svdDebug : bool, optional
        Check the SVD procedure with some tests (True) or not (False).
        Ignored if `nModesManual` is provided, see below.
    eps : float, optional
        A near-zero value, for comparison to zero.
    testInverses : bool, optional
        Whether to check if DMD modes agree with their inverses.
        Ignored if `nModesManual` is provided, see below.
    printMessages : bool, optional
        Whether to print informative / debugging messages.
    nModesManual : int, optional
        If provided, return exactly `nModesManual` inverse DMD modes,
        bypassing checks on them.

    Returns
    -------
    tuple
        Tuple of (vCutInvSigma, pcaModes) containing the parts for the
        "inverse DMD modes" action
            coefficients[i] = (pcaModes[i],  state0),
            coefficients = vCutInvSigma @ coefficients.
    """

    def state_(i, doAvg=False, avgState=None):
        return modes[i]

    def inprod_(mod1, mod2):
        res = dmdInprod(mod1, mod2, inprod)
        return res

    numSnaps = len(modes)
    printIf(printMessages, "DMD: Computing the inverse of the modes.")
    vCut, singValsSq, _ = pca.stage2(
        numSnaps,
        state_,
        inprod_,
        None,
        doAvg=False,
        eps=eps,
        svdDebug=svdDebug,
        returnAll=True,
        n_jobs=n_jobs,
        joblib_backend=joblib_backend,
        joblib_verbosity=joblib_verbosity,
        keepCorrMatrix=False,
        printMessages=printMessages,
        nModesManual=nModesManual,
    )
    pcaModes = pca.stage3(
        numSnaps,
        state_,
        inprod_,
        None,
        vCut,
        singValsSq,
        doAvg=False,
        svdDebug=svdDebug,
        eps=eps,
        n_jobs=n_jobs,
        joblib_backend=joblib_backend,
        joblib_verbosity=joblib_verbosity,
        printMessages=printMessages,
        nModesManual=nModesManual,
    )
    numInvModes = len(pcaModes)
    vCutInvSigma = vCut[:, :numInvModes] @ np.diag(
        1 / np.sqrt(singValsSq[:numInvModes])
    )

    # Test the inverses
    if (nModesManual is None) and testInverses:
        testMatrix = np.zeros((numInvModes, numInvModes), dtype=np.complex128)

        # For each (i, j>=i) create a job, they are all in memory
        def fillTestMatrix(i, j, pbar):
            norm = inprod_(pcaModes[i], state_(j))
            testMatrix[i, j] = norm
            pbar.update()

        printIf(printMessages, "Testing inverse DMD modes.")
        with tqdm(total=numInvModes ** 2, disable=not printMessages) as pbar:
            Parallel(n_jobs=n_jobs, backend=joblib_backend, verbose=joblib_verbosity)(
                delayed(fillTestMatrix)(i, j, pbar)
                for i in range(numInvModes)
                for j in range(numInvModes)
            )
        testMatrix = vCutInvSigma @ testMatrix

        printIf(printMessages, "Deciding where to cut-off for numerical accuracy.")

        numericalCutOff = numInvModes
        # The diagonals should all be 1
        for i in range(numInvModes):
            if np.abs(testMatrix[i, i] - 1) > eps:
                printIf(
                    printMessages,
                    f"Diagonal test failed at index {i+1} of {numInvModes} with error {np.abs(testMatrix[i, i] - 1)/eps:.4e}.",
                )

                numericalCutOff = i
                break

        # Cutoff index may have changed for the case of no manual cutoffs
        # In which case we trash after the cut off
        for i in range(numericalCutOff, len(pcaModes)):
            del pcaModes[i]
        printIf(
            printMessages,
            f"Cutting at index {numericalCutOff} of {numInvModes} as decided by the diagonal test.",
        )
        numInvModes = numericalCutOff

        # The rest should be zero
        for i in range(numInvModes):
            for j in range(i + 1, numInvModes):
                if np.abs(testMatrix[i, j]) > eps:
                    printIf(
                        printMessages,
                        f"Off-diagonal test failed at indices {i+1}, {j+1} of {numInvModes} with error {np.abs(testMatrix[i, j])/eps:.4e}.",
                    )

                    # The problematic index is the largest of the two
                    # indices that are tested. It replaces the previous
                    # cutoff if it's smaller than that.
                    numericalCutOff = min(max(i, j), numericalCutOff)

        numInvModes = numericalCutOff
        vCutInvSigma = vCutInvSigma[:, :numInvModes]
        for i in range(numInvModes, len(pcaModes)):
            del pcaModes[i]

    return (
        vCutInvSigma,
        pcaModes,
    )


def dmdSpectrum(
    numPairs,
    state,
    inprod,
    c_sigma=0.999,
    c_chi=0.001,
    eps=10 ** (-8),
    eps_zero_eigenvalue=10 ** (-13),
    svdDebug=True,
    returnAll=False,
    debugExact=True,
    n_jobs=def_n_jobs,
    joblib_backend=def_joblib_backend,
    joblib_verbosity=def_joblib_verbosity,
    printMessages=False,
    corrMatrix=None,
    keepCorrMatrix=False,
    nModesManual=None,
    debugUnitNorm=False,
    forceUnitNorm=False,
):
    """Computes the DMD of snapshots where the intermediate PCA is done
    using the method of snapshots (Sirovich.)
    Intermediate singular values are cut at a threshold determined by the
    input parameters.
    Returned DMD modes are in the original memory layout of the snapshots,
    note that they are complex valued.

    Parameters
    ----------
    numPairs : int
        Number of [state(t), state(t + dt)] pairs.
    state : function
        Function that returns a state vector given an index.
        It should have the signature
            state(i, element, doAvg, avgState).
        Element 'initial' is for state at time t(i), and element 'forward'
        is for state at time t(i) + \delta t.
    inprod : function
        Function that returns the inner product of two state vectors.
        It should have the calling signature
            inprod(stateI, stateJ).
    c_sigma : float, optional
        Singular values are cut off at the index where the sum of their squares
        exceed this value proportional to the total sum and when `c_chi`, see
        next, is also satisfied.
    c_chi : float, optional
        The first excluded singular value squared needs to be smaller than the sum
        with respect to this value.
    eps : float, optional
        A near-zero value, for comparison to zero.
    eps_zero_eigenvalue : float, optional
        A near-zero value, for comparison to zero of DMD eigenvalues specifically.
    svdDebug : bool, optional
        Check the SVD procedure with some tests (True) or not (False).
    returnAll : bool
        Return all singular values and modes up to `eps` precision checking
        their orthonormality, bypassing cutoffs.
    debugExact : bool
        Check that DMD modes have non-zero magnitude. There's a distinction
        between exact and projected DMD modes, where the latter appears to be
        necessary when the corresponding eigenvalue is zero and the exact mode
        is also zero. I will for now only check that the magnitudes are non-zero
        and forget about projected DMD modes. / g 200416
    debugUnitNorm : bool
        Check that DMD modes have unit magnitude.
    forceUnitNorm : bool
        Normalize DMD modes to have unit magnitude.
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
        If provided, return exactly `nModesManual` DMD modes,
        bypassing checks on them.

    Returns
    -------
    tuple
        Tuple of (Lambda, phi) or (Lamba, phi, corrMatrix).
        The DMD eigenvalues and eigenvectors sorted in increasing real part and
        then increasing imaginary part.
        corrMatrix is returned if keepCorrMatrix==True.

    """

    # Compute the SVD of snapshots 0, 1, ..., m-1
    # Singular values and vectors of the correlation matrix
    # (Singular values squared and right singular values of the data matrix)
    vCut, singValsSq, corrMatrix = pca.stage2(
        numPairs,
        state,
        inprod,
        None,
        doAvg=False,
        c_sigma=c_sigma,
        c_chi=c_chi,
        eps=eps,
        svdDebug=svdDebug,
        returnAll=returnAll,
        n_jobs=n_jobs,
        joblib_backend=joblib_backend,
        joblib_verbosity=joblib_verbosity,
        corrMatrix=corrMatrix,
        keepCorrMatrix=keepCorrMatrix,
        printMessages=printMessages,
        nModesManual=nModesManual,
    )
    # Find the PCA modes
    pcaModes = pca.stage3(
        numPairs,
        state,
        inprod,
        None,
        vCut,
        singValsSq,
        doAvg=False,
        svdDebug=svdDebug,
        eps=eps,
        n_jobs=n_jobs,
        joblib_backend=joblib_backend,
        joblib_verbosity=joblib_verbosity,
        printMessages=printMessages,
        nModesManual=nModesManual,
    )

    # Compute Ξ' V

    pcaCutIndex = len(pcaModes)
    kivi = {}

    printIf(printMessages, "Calculating Ξ' V.")
    with tqdm(total=(numPairs), disable=not printMessages) as pbar:
        for i in range(numPairs):
            # Load time forward snapshots
            snap = state(i, element="forward")
            for j in range(pcaCutIndex):
                # Add up its contributions to the final output
                if i == 0:
                    kivi[j] = vCut[i, j] * snap
                else:
                    kivi[j] += vCut[i, j] * snap
            pbar.update()

    # Inverse of the singular values
    invSigma = 1 / np.sqrt(singValsSq[:pcaCutIndex])

    # Reshapes are temporary to do the matrix products and compute the eigenspace,
    # in the end arrays are put back into their original shapes

    # This is used in two places
    kiviInvSigma = [invSigma[i] * kivi[i] for i in range(pcaCutIndex)]

    # This is maybe the most expensive part:
    # Computing the inner products between the PCA modes and the primed data matrix
    tildeA = np.zeros((pcaCutIndex, pcaCutIndex), order="F")

    def fillTildeA(i, j, pbar):
        tildeA[i, j] = inprod(pcaModes[i], kiviInvSigma[j])
        pbar.update()

    printIf(printMessages, "Calculating tilde(A) = dagger(U) Ξ' V (1/s).")
    with tqdm(total=pcaCutIndex ** 2, disable=not printMessages) as pbar:
        Parallel(n_jobs=n_jobs, backend=joblib_backend, verbose=joblib_verbosity)(
            delayed(fillTildeA)(i, j, pbar)
            for i in range(pcaCutIndex)
            for j in range(pcaCutIndex)
        )

    Lambda, tildePhi = np.linalg.eig(tildeA)

    # Filter out zero eigenvalues
    normLambda = np.sqrt(np.conj(Lambda) * Lambda)
    nonzeros = np.nonzero(normLambda > eps_zero_eigenvalue)[0]
    n_nonzeros = len(nonzeros)
    if n_nonzeros == 0:
        exit("No non-zero DMD eigenvalues.")
    elif n_nonzeros != pcaCutIndex:
        print(f"Found {pcaCutIndex - n_nonzeros} zero eigenvalues.")

    Lambda = Lambda[nonzeros]
    tildePhi = tildePhi[:, nonzeros]

    # Compute the DMD modes of A
    # Exact DMD modes.
    phi = {}

    printIf(printMessages, "Calculating exact DMD modes, ϕ = Ξ' V (1/s) tilde(ϕ).")
    with tqdm(total=pcaCutIndex, disable=not printMessages) as pbar:
        for i in range(pcaCutIndex):
            for j in range(n_nonzeros):
                snap = kiviInvSigma[i]
                # Need to convert to our special dmdMode data type
                if i == 0:
                    phi[j] = dmdMode(
                        tildePhi[i, j].real * snap, tildePhi[i, j].imag * snap
                    )
                else:
                    phi[j] += dmdMode(
                        tildePhi[i, j].real * snap, tildePhi[i, j].imag * snap
                    )
            pbar.update()

    # Sort in order of decreasing real part and then increasing imaginary part
    # of the *eigenvalues*
    sorter = np.argsort(np.imag(np.log(Lambda)))
    Lambda = Lambda[sorter]
    phi = {newidx: phi[i] for newidx, i in enumerate(sorter)}
    sorter = np.argsort(np.real(np.log(Lambda)))[::-1]
    Lambda = Lambda[sorter]
    phi = {newidx: phi[i] for newidx, i in enumerate(sorter)}

    # Check that DMD modes have non-zero magnitude.
    # Otherwise we'll have to take into account projected DMD modes as well.
    if forceUnitNorm or debugExact or debugUnitNorm:
        printIf(printMessages, "Checking the magnitudes of DMD modes.")
        if forceUnitNorm:
            printIf(printMessages, "Normalizing DMD modes.")
        with tqdm(total=n_nonzeros, disable=not printMessages) as pbar:
            for i in range(n_nonzeros):
                # ! Taking the L2 norm here
                magnitude = dmdInprod(phi[i], phi[i], inprod)
                if (forceUnitNorm or debugExact) and np.abs(magnitude) < eps:
                    print(f"Eigenvalue corresponding to zero mode: {Lambda[i]}.")
                    print(f"Norm2 of the zero mode: {magnitude}.")
                    print(f"DMD mode {i} has near-zero magnitude.")
                elif debugUnitNorm and np.abs(magnitude - 1) > eps:
                    print(
                        f"Warning: DMD mode {i} has non-unit magnitude:",
                        np.abs(magnitude),
                    )

                if forceUnitNorm:
                    # Normalize
                    phi[i] = phi[i] / np.sqrt(np.abs(magnitude))
                pbar.update()

    printIf(printMessages, "Done.")
    return Lambda, phi, corrMatrix


def filterSpectrum(exponents, modes, muMax=0.1, eps=10 ** (-16)):
    """Filters and sorts a DMD spectrum.

    What this function does:
        1. Select
            abs(exponents.real) < muMax
        2. Throw away ALL (if there's any) real exponents EXCEPT the smallest one.
        3. Order the remaining exponents in increasing absolute value
           of imaginary parts. As a convention, for a complex conjugate
           pair, the positive frequency will come first.

    This routine will exit if
        - Complex exponents, within the
          muMax interval, do not come in complex conjugate pairs.
        - There are no exponents within the muMax threshold.


    Parameters
    ----------
    exponents : ndarray
        NumPy array of DMD exponents (or some other numbers associated to the
        modes to base the filtering and ordering on.)
    modes : list
        List of DMD modes.
    muMax : float
        The upper bound for the absolute value of the real part of the exponents.
    eps : float
        A small number for comparisons against zero.

    Returns
    -------
    tuple
        Tuple of 'exponentsFiltered', 'modesFiltered', 'picker',
        containing the filtered and sorted exponents and DMD modes,
        and the NumPy array that does the job
            exponentsFiltered = exponents[picker],
            modesFiltered = [modes[i] for i in picker],
        in case it turns out to be necessary.
        That is, 'picker' has the filtered and ordered indices corresponding
        to 'exponents' and 'modes'.
    """

    # First bound with muMax
    picker = np.nonzero(np.abs(exponents.real) < muMax)[0]
    if len(picker) == 0:
        exit(f"No exponents found within muMax = {muMax}.")
    exponentsFiltered = exponents[picker]

    # list to output
    pickerOut = []

    # Find the purely real ones
    realsPicker = np.nonzero(np.abs(exponentsFiltered.imag) < eps)[0]
    if len(realsPicker) != 0:
        reals = exponentsFiltered[realsPicker]
        # Let's sort in case there are multiple
        sorter = np.argsort(np.abs(reals))
        realsPicker = realsPicker[sorter]

        # Choose the smallest one
        pickerOut.append(picker[realsPicker[0]])

    # Now pick the complex ones
    complexsPicker = np.nonzero(np.abs(exponentsFiltered.imag) > eps)[0]
    complexs = exponentsFiltered[complexsPicker]
    # Sort them with the imaginary part
    # I wonder if it's ever possible to have the same imaginary part
    # but different real parts
    sorter = np.argsort(np.abs(complexs.imag))
    complexs = complexs[sorter]
    complexsPicker = complexsPicker[sorter]

    print(complexs)
    # Check that they come in pairs AND apply the convention of positive
    # frequency first
    # Dumb check
    if len(complexs) % 2 != 0:
        print("Number of complex exponents is not even.")
        complexs = complexs[0 : (len(complexs) - 1)]
        print("complexs=", complexs)
    for i in range(0, len(complexs), 2):
        f0 = complexs[i]
        f1 = complexs[i + 1]
        if np.abs(f0.real - f1.real) > eps:
            print(f"Non-conjugate pair {f0}, {f1}.")
        if np.abs(np.abs(f0.imag) - np.abs(f1.imag)) > eps:
            print(f"Non-conjugate pair {f0}, {f1}.")
        else:
            # Otherwise take the positive one first
            print(f"c.c. pair ok")
            if f0.imag > 0:
                pickerOut.append(picker[complexsPicker[i]])
                pickerOut.append(picker[complexsPicker[i + 1]])
            else:
                pickerOut.append(picker[complexsPicker[i + 1]])
                pickerOut.append(picker[complexsPicker[i]])

    picker = np.array(pickerOut)
    exponentsFiltered = exponents[picker]
    modesFiltered = [modes[i] for i in picker]

    return exponentsFiltered, modesFiltered, picker


def printIf(printMessages, message):
    if printMessages:
        print(message, flush=True)
