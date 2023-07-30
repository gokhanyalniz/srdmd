#!/usr/bin/env python3
"""
Make sure to do the following exports to prevent resource oversubscription:

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"""
import random
import sys
from collections import deque
from os import makedirs
from pathlib import Path
from sys import exit

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import inv
from psutil import cpu_count
from tqdm import tqdm

# See the documentation at GitHub for how to compile libpycf
# the path below should refer to the directory containing it
libpycf_path = Path('/nfs/scistore12/hofgrp/gyalniz/') / 'channelflow/build-gcc10.2/python-wrapper'
sys.path.append(str(libpycf_path.resolve()))
import libpycf

import dmd

###############################################################################
# Path to data directory (containing u*.nc)
runDir = Path("/nfs/scistore12/hofgrp/emarensi/runs/HKW-re400/optslice-data1/")

# Path to output directory
# Can set it to be the same as runDir if you want the output to be saved
# in a subfolder of where your data are
outDir = Path("./")

# fmt: on
bestfit = True  # Implement best fit method to find coefficient
bestfit_recomp = False  # recompute best-fit coeff after chosing modes

testing = False  # Can set some things differently when testing
if testing:
    testcounter = 1

# Control parameters ##########################################################

# Subsample available snapshots to samplingSteps
# (take every samplingSteps^th snapshot from the folder)
samplingSteps = 1

# Using [state(t) + state(t + nSeperation * \delta t)] pairs
# Set nSeparation = 1 for NO random picking
nSeparation = 10

# Number of snapshots that go in a window
# WARNING: As many states are going to be kept in memory!
# Consider this when choosing the window size.
lenWindow = 600
# How many snapshots to pick at random from (lenWindow - nSeparation) options
# If None will use all (NO random picking)
nRandomPairs = 100

if nRandomPairs is not None and nRandomPairs > lenWindow - nSeparation:
    exit("nRandomPairs > lenWindow - nSeparation.")

if nRandomPairs is None:
    nPairs = lenWindow - nSeparation
else:
    nPairs = nRandomPairs

# Number of snapshots
print("lenWindow = ", lenWindow)
print("nPairs = ", nPairs)

# Number of DMD modes to use.
# If None, gets determined dynamically for each window based on
#   eps, returnAll, c_sigma, c_chi, svdDebugModes, svdDebugInvModes, testInverses
# below.
# As setting it disables all orthonormality and some other checks, suggested
# is to use a few modes less than lenWindow - nSeparation.
nModesManual = None
if nModesManual is not None and nModesManual > nPairs:
    exit("nModesManual > nPairs.")

# Number of DMD modes to use to reconstruct the field
# (MUST BE EVEN, THEN IT WILL BE ADJUSTED DEPENDING ON
# NUMBER OF REAL MODES/CC PAIRS)
Nd = None

# Time step to shift the window
Dw = 5.0

# "Special time", if not None, look only at the window starting there
tSpecial = None

# Initial time to start sliding
tInitial = 1295.0

# Final time of states to go into windows
# If None, uses all states in the folder
tFinal = 1355.0

# PCA parameters ##########################################################

# If True, uses all PCA modes, ignoring `c_sigma` and `c_chi` below
# Ignored if nModesManual is set
returnAll = True
# "Kinetic energy" percentage to cut off the PCA modes at.
# Ignored if nModesManual is set
# Ignored if returnAll == True
c_sigma = 0.9999
# The average of the ignored PCA modes's "kinetic energy" should be less than
# the "kinetic energy" of the first PCA mode to this fraction
# Ignored if set to None
# Ignored if nModesManual is set
# Ignored if returnAll == True
c_chi = 0.001
# In above, I put "kinetic energy" in quotes, because it's really the
# sum of the squares of the singular values of the data matrix

# If True, checks for orthonormality of the PCA modes and return only
# those that are orthonormal.
# If returnAll == True, recommend to set this to True as well.
# Ignored if nModesManual is set
svdDebugModes = True

# construct initial guess for RPO/nearly-periodic processes
rpo = True
if rpo:
    nh = 2
    print("nh=", nh)

# DMD parameters ##########################################################

# Check that DMD modes have non-zero magnitude
debugExact = True
# Check the SVD part of the inverse modes computation
# Ignored if nModesManual is set
svdDebugInvModes = True
# Check the inverse modes
# Ignored if nModesManual is set
testInverses = True

# Epsilon used for checks in various equalities to zero
eps = 10 ** (-8)
eps0 = 10 ** (-12)
########################################################################
# Number of threads to use, leave at None to set automatically
n_jobs_manual = None


# What to save ##########################################################

# Reconstructed states
saveReconstruction = True
# Reconstruction errors
saveErrors = True
# DMD exponents
saveExponents = True
# DMD coefficients
saveCoeff = True
# DMD modes (WITHOUT coefficients embedded)
saveModes = True
# Snapshots of DMD modes (WITH coefficients embedded)
# Be careful this will output snapshots of ALL dmd modes
# at each time within the lenWindow
# I  suggest to set it to False when looping over a lot of time windows
saveSnap1 = False

# What to plot ##########################################################

# Reconstruction errors
plotErrors = True
# DMD exponents
plotExponents = True
# DMD Spectrum (Tu et al 2015)
plotSpec = True

# Processing parameters #######################################################

# Assume all the files with names "u*.nc" are Channelflow data files
# Get the full paths to them
stateFiles = [str(stateFile.resolve()) for stateFile in runDir.glob("u*.nc")]
# Extract times from file names
# u[time].nc
times = np.array([float(stateFile.name[1:-3]) for stateFile in runDir.glob("u*.nc")])
# Sort in time, *VERY IMPORTANT*
sorter = np.argsort(times)
times = times[sorter]
stateFiles = [stateFiles[i] for i in sorter]
# Subsample
times = times[::samplingSteps]
stateFiles = stateFiles[::samplingSteps]
# Time steps going in to the PCA (assuming fixed time steps)
dt_dmd = times[nSeparation] - times[0]
dt_samples = times[1] - times[0]
print("dt_dmd = ", dt_dmd)
print("dt_samples = ", dt_samples)
Tw = lenWindow * dt_samples
print("Tw=", Tw)
mDMD = int(Tw / dt_dmd)
print("mDMD=", mDMD)
# Number of snaphots to shift the window at each step
# Assumed lenWindowShift < lenWindow
lenWindowShift = int(Dw / dt_samples)
print("lenWindowShift=", lenWindowShift)
if lenWindowShift >= lenWindow:
    exit("lenWindowShift >= lenWindow.")

# Cut at the desired range
if tSpecial is None:
    if tFinal is None:
        tFinal = times[-1]
    intervalCut = np.logical_and(times >= tInitial, times <= tFinal)
    if len(intervalCut) == 0:
        exit("tInitial and / or tFinal don't fit the available snapshots.")
    times = times[intervalCut]
    stateFiles = [stateFiles[i] for i in np.nonzero(intervalCut)[0]]
else:
    stateFiles = [stateFiles[i] for i in np.nonzero(times >= tSpecial)[0]][:lenWindow]
    times = times[times >= tSpecial][:lenWindow]

# Number of threads to use
if n_jobs_manual is not None:
    n_jobs = n_jobs_manual
else:
    n_jobs = cpu_count(logical=False) - 1
print(
    f"Using {n_jobs} threads out of {cpu_count(logical=False)} physical ({cpu_count(logical=True)} logical) cores."
)
if n_jobs > cpu_count(logical=True):
    exit("ERROR: More threads set than available.")
if n_jobs > cpu_count(logical=False) - 1:
    print("WARNING: You are leaving no physical cores to the OS, you may get errors.")

# Name of subfolder to save the data at
# If tSpecial is not None, it gets appended to this name
if testing is True:
    dataFolderName = (
        "test" + f"{testcounter}" + f"-Tw{Tw}" + f"-ts{tInitial}" + f"-tf{tFinal}"
    )
else:
    if nModesManual is not None:
        dataFolderName = (
            "dmd"
            + f"-Tw{Tw}"
            + f"-ts{tInitial}"
            + f"-tf{tFinal}"
            + f"-Nc{nModesManual}"
        )
    elif returnAll is True:
        dataFolderName = (
            "dmd" + f"-Tw{Tw}" + f"-ts{tInitial}" + f"-tf{tFinal}" + f"-returnAll"
        )
    else:
        dataFolderName = (
            "dmd"
            + f"-Tw{Tw}"
            + f"-ts{tInitial}"
            + f"-tf{tFinal}"
            + f"-cE{c_sigma}"
            + f"-iE{c_chi}"
        )
if Nd is not None:
    dataFolderName = dataFolderName + f"-Nd{Nd}" + f"-mDMD{mDMD}"
if tSpecial is not None:
    dataFolderName += f"-{tSpecial}"
if bestfit:
    if bestfit_recomp:
        dataFolderName += f"-bf2"
    else:
        dataFolderName += f"-bf"
if rpo:
    dataFolderName += f"-nh{nh}"
if nRandomPairs is not None:
    dataFolderName = dataFolderName + f"-nRand{nRandomPairs}" + f"outof{lenWindow}"
print("Creating folder", dataFolderName)
# Create the data folder
dataFolder = outDir / dataFolderName
if not Path.is_dir(dataFolder):
    makedirs(dataFolder)
if saveSnap1 is True:
    makedirs(dataFolder / f"saveSnap1")

if plotErrors or plotExponents:
    import math

    import matplotlib.cm as cm
    from matplotlib import pyplot as plt

# Window loop #################################################################

# Total number of windows
print("Start time=", times[0])
print("Final time=", times[-1])
lenWindowsTot = int(((times[-1] - times[0]) - Tw) // Dw) + 1
print("Total number of windows=", lenWindowsTot)
# lenWindowsTot = len(times[: -(lenWindow + nSeparation)][::lenWindowShift])
strlen_lenWindowsTot = len(str(lenWindowsTot))

# Will keep the states of the current window in memory
states = deque(maxlen=lenWindow)

# Will keep here indices for random picks
indicesRandom = None

# Returns the i'th state
def state(i, element="initial", doAvg=False, avgState=None, use_all=False):
    if not use_all:
        if nRandomPairs is None:
            if element == "initial":
                return states[i]
            else:
                return states[i + nSeparation]
        else:
            if element == "initial":
                return states[indicesRandom[i]]
            else:
                return states[indicesRandom[i] + nSeparation]
    else:
        return states[i]


# Returns the inner product between two states in memory
def inprod(stateI, stateJ):
    res = libpycf.L2IP(stateI, stateJ)
    return res


# Load all the states in the first window
for i in range(lenWindow):
    states.append(libpycf.FlowField(stateFiles[i]))
nStatesPast = lenWindow

if nRandomPairs is not None:
    random.seed()

with tqdm(total=lenWindowsTot) as pbar:
    fmeanerr = open(dataFolder / f"meanerravg.txt", "ab")
    if rpo:
        fperiodicity = open(dataFolder / f"periodicity.txt", "ab")
    for iWindow in range(lenWindowsTot):
        if iWindow != 0:
            for i in range(lenWindowShift):
                states.append(libpycf.FlowField(stateFiles[nStatesPast + i]))
            nStatesPast += lenWindowShift

        if nRandomPairs is not None:
            indicesRandom = sorted(
                random.sample([i for i in range(lenWindow - nSeparation)], nRandomPairs)
            )

        # Find the DMD spectrum
        print(
            f"Window {iWindow} starting at ts=",
            (tInitial + iWindow * lenWindowShift * dt_samples),
        )
        Lambda, phi, _ = dmd.dmdSpectrum(
            nPairs,
            state,
            inprod,
            n_jobs=n_jobs,
            c_sigma=c_sigma,
            c_chi=c_chi,
            eps=eps,
            returnAll=returnAll,
            svdDebug=svdDebugModes,
            debugExact=debugExact,
            printMessages=False,
            nModesManual=nModesManual,
        )
        print(f"{len(phi)} DMD modes before their pseudo-inversion.")

        numModes = len(phi)

        # Compute the exponents
        exponents = np.log(Lambda) / dt_dmd

        striWindow = str(iWindow).zfill(strlen_lenWindowsTot)

        ################################################################################################
        if rpo:
            expF, phiF, picker = dmd.filterSpectrum(exponents, phi)
            for k in range(0, len(phiF)):
                phi[k] = phiF[k]

            numModes = int(1 + nh * 2)
            for i in range(0, numModes):
                magnitudeC = np.abs(dmd.dmdInprod(phi[i], phi[i], inprod))
            exponents = expF[0:numModes]
            Lambda = np.exp(dt_dmd * exponents)

            def get_fundFreq(
                lamh,
            ):  # takes as input imaginary part of harmonics ordered from smallest to largest
                om_f = 0.0
                nh_ = min(nh, np.size(lamh))
                for j in range(int(nh_)):
                    om_f = om_f + lamh[j]

                om_f = 2.0 / (nh * (nh + 1)) * om_f
                print("om_f=", om_f)
                return om_f

            def estimate_err(lamh, om_f):
                eps_om = 0.0
                nh_ = min(nh, np.size(lamh))
                for j in range(int(nh_)):
                    eps_om = eps_om + np.abs(lamh[j] - (j + 1) * om_f) ** 2
                eps_om = eps_om / (nh * np.abs(om_f) ** 2)
                print("eps_om=", eps_om)
                return eps_om

            frequencies = np.imag(exponents[1::2])
            print("frequencies=", frequencies)
            fundFreq = get_fundFreq(frequencies)
            error = estimate_err(frequencies, fundFreq)
            Tg = 2.0 * math.pi / fundFreq
            print("Guess period Tg=", Tg, "with error=", error)

        ##############################################################################################################
        if bestfit:
            print("#####################################")
            print("Elena: Find the best-fit coefficient")
            print("#####################################")
            q = np.zeros(numModes, dtype=np.complex128)
            P = np.zeros((numModes, numModes), dtype=np.complex128)

            # Assumes Hermiticity
            def fillP(i, j, P, pbar):
                P[i, j] = dmd.dmdInprod(phi[i], phi[j], inprod)  # <phi_i*, phi_j>
                C = 1.0  # complex(1.0,0.0)
                for m in range(1, mDMD):
                    C += ((np.conj(Lambda[i])) ** m) * (Lambda[j]) ** m
                P[i, j] = P[i, j] * C

                P[j, i] = np.conj(P[i, j])
                pbar.update()

            with tqdm(total=numModes * (numModes + 1) // 2) as pbar:
                Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(fillP)(i, j, P, pbar)
                    for i in range(0, numModes)
                    for j in range(i, numModes)
                )

            Pinv = inv(P)

            initialState = dmd.dmdMode(
                state(0, use_all=True), 0 * state(0, use_all=True)
            )

            def fillQ(i, q, pbar):
                q[i] = dmd.dmdInprod(phi[i], initialState, inprod)
                for m in range(1, mDMD):
                    m_ = m * nSeparation
                    xi_m = dmd.dmdMode(
                        state(m_, use_all=True), 0 * state(m_, use_all=True)
                    )  # xi_m = xi(m dt_dmd)
                    q[i] += ((np.conj(Lambda[i])) ** m) * dmd.dmdInprod(
                        phi[i], xi_m, inprod
                    )  # (Lambda_i*)^m * <phi_i*, xi_m>
                pbar.update()

            with tqdm(total=numModes) as pbar:
                Parallel(n_jobs=n_jobs, backend="threading")(
                    delayed(fillQ)(i, q, pbar) for i in range(0, numModes)
                )

            coeffs = np.dot(Pinv, q)

        times_window_ = times[iWindow * lenWindowShift :][:lenWindow]
        times_window = times_window_[::nSeparation]
        title = f"$w_i = {iWindow}$, $N = {numModes}$, $t\\in [{times_window[0]}, {times_window[-1]}]$"
        #        striWindow = str(iWindow).zfill(strlen_lenWindowsTot)

        ##############################################################################################################
        # Embed coefficients into the DMD modes, the result will be called as
        # the DMD modes from now on
        phi_ = {}
        for i in range(numModes):
            phi_[i] = phi[i]
            phi[i] = coeffs[i] * phi[i]

        # Compute the norm of DMD modes
        magnitude = np.zeros(numModes)
        magnitude_ = np.zeros(numModes)
        for i in range(numModes):
            # ! Taking the L2 norm here
            magnitude[i] = np.abs(dmd.dmdInprod(phi[i], phi[i], inprod))
            magnitude_[i] = np.abs(dmd.dmdInprod(phi_[i], phi_[i], inprod))

        scalmagn = magnitude * (np.abs(Lambda)) ** (mDMD)
        # Order the DMD modes from "most" to "least" dominant
        if not rpo:
            sortdom = np.argsort(scalmagn)[::-1]
            Lambda = Lambda[sortdom]
            exponents = exponents[sortdom]
            magnitude = magnitude[sortdom]
            coeffs = coeffs[sortdom]
            scalmagn = scalmagn[sortdom]
            phi = {newidx: phi[i] for newidx, i in enumerate(sortdom)}
            phi_ = {newidx: phi_[i] for newidx, i in enumerate(sortdom)}
            print("Scaled Magnitude sorted=", scalmagn)

            ####### Choose how many DMD modes to retain to reconstruct the state ##########
            numModes_ = numModes

            if Nd is not None:
                if Nd >= numModes:
                    print("Nd >= numModes, so keep Nd=numModes ", numModes)
                    Nd_ = None
                    realsPicker = np.nonzero(np.abs(Lambda.imag)[:Nd] < eps0)[0]
                    print("idx real=", realsPicker)
                    Nr = len(realsPicker)
                    print("Nr=", Nr)
                    reals = Lambda[realsPicker]
                    print("Lamda reals=", reals)
                else:
                    realsPicker = np.nonzero(np.abs(Lambda.imag)[:Nd] < eps0)[0]
                    print("idx real=", realsPicker)
                    Nr = len(realsPicker)
                    print("Nr=", Nr)
                    reals = Lambda[realsPicker]
                    print("Lamda reals=", reals)
                    if Nr % 2 != 0:  # odd number of real modes (assumes Nd even)
                        Nd_ = Nd - 1  # eg. 9 with 1 real number (1 real + 4 cc pairs)
                    else:
                        Nd_ = Nd  # e.g. 10 with 2 real numbers (2 real + 4 cc pairs)
                    numModes = Nd_

                # Set Nd_ = None #To avoid checks below, COMMENT if you want the checks!!
                ### checks ######
                if Nd_ is not None:
                    print("Check if Nd=", Nd_, "is suitable")
                    complexsPicker = np.nonzero(np.abs(Lambda.imag)[:Nd_] > eps0)[0]
                    complexs = Lambda[complexsPicker]
                    if len(complexs) % 2 != 0:
                        print("WARNING: Number of complexs should be even")
                    print("complexs=", complexs)
                    i = 0
                    counter = 0
                    while True:
                        f0 = complexs[i]
                        f1 = complexs[i + 1]
                        if (np.abs(f0.real - f1.real) > eps0) or (
                            np.abs(np.abs(f0.imag) - np.abs(f1.imag)) > eps0
                        ):
                            counter = counter + 1
                            if counter > 1:
                                numModes = (
                                    numModes_  # keep the original number of modes
                                )
                                print(
                                    "WARNING: There is another non c.c. pair, this should not happen!"
                                )
                                break
                            else:
                                print(f"Non-conjugate pair {f0}, {f1}.")
                                idiscard = i
                                Nd_ = Nd_ + 1
                                complexsPicker = np.nonzero(
                                    np.abs(Lambda.imag)[:Nd_] > eps0
                                )[0]
                                complexs = Lambda[complexsPicker]
                                i = i + 1
                        else:
                            print("c.c. pair ok")
                            i = i + 2
                        if i >= (Nd_ - Nr):
                            break

                    if counter == 1:
                        complexsPickerOK = np.append(
                            complexsPicker[:idiscard], complexsPicker[idiscard + 1 :]
                        )
                        Nd_ = Nr + len(complexsPickerOK)
                        print("Nd=", Nd_)
                        modesPickerOK = np.append(realsPicker, complexsPickerOK)
                        sorter = np.argsort(modesPickerOK)
                        modesPickerOK = modesPickerOK[sorter]
                        for i in range(0, Nd_):
                            exponents[i] = exponents[modesPickerOK[i]]
                            Lambda[i] = Lambda[modesPickerOK[i]]
                            magnitude[i] = magnitude[modesPickerOK[i]]
                            coeffs[i] = coeffs[modesPickerOK[i]]
                            scalmagn[i] = scalmagn[modesPickerOK[i]]
                            phi[i] = phi[modesPickerOK[i]]
                        for i in range(Nd_, (numModes_ - 1)):
                            exponents[i] = exponents[i + 1]
                            Lambda[i] = Lambda[i + 1]
                            magnitude[i] = magnitude[i + 1]
                            coeffs[i] = coeffs[i + 1]
                            scalmagn[i] = scalmagn[i + 1]
                            phi[i] = phi[i + 1]
                        # Append the non-c.c. pair at the end
                        exponents[-1] = exponents[complexsPicker[idiscard]]
                        Lambda[-1] = Lambda[complexsPicker[idiscard]]
                        magnitude[-1] = magnitude[complexsPicker[idiscard]]
                        coeffs[-1] = coeffs[complexsPicker[idiscard]]
                        scalmagn[-1] = scalmagn[complexsPicker[idiscard]]
                        phi[-1] = phi[complexsPicker[idiscard]]

                print("Original number of modes from PCA=", numModes_)
                print("Use first numModes=", numModes, "modes with exponents:")
                print(exponents[:numModes])

                ##############################################################################################################
                if bestfit_recomp:
                    print("##########################################################")
                    print("Elena: Recalculate best-fit coefficient with Nd=", numModes)
                    print("##########################################################")
                    q = np.zeros(numModes, dtype=np.complex128)
                    P = np.zeros((numModes, numModes), dtype=np.complex128)

                    # Assumes Hermiticity
                    def fillP_(i, j, P, pbar):
                        P[i, j] = dmd.dmdInprod(
                            phi_[i], phi_[j], inprod
                        )  # <phi_i*, phi_j>
                        C = 1.0  # complex(1.0,0.0)
                        for m in range(1, mDMD):
                            C += ((np.conj(Lambda[i])) ** m) * (Lambda[j]) ** m
                        P[i, j] = P[i, j] * C

                        P[j, i] = np.conj(P[i, j])
                        pbar.update()

                    with tqdm(total=numModes * (numModes + 1) // 2) as pbar:
                        Parallel(n_jobs=n_jobs, backend="threading")(
                            delayed(fillP_)(i, j, P, pbar)
                            for i in range(0, numModes)
                            for j in range(i, numModes)
                        )

                    Pinv = inv(P)

                    initialState = dmd.dmdMode(
                        state(0, use_all=True), 0 * state(0, use_all=True)
                    )

                    def fillQ_(i, q, pbar):
                        q[i] = dmd.dmdInprod(phi_[i], initialState, inprod)
                        for m in range(1, mDMD):
                            m_ = m * nSeparation
                            xi_m = dmd.dmdMode(
                                state(m_, use_all=True), 0 * state(m_, use_all=True)
                            )  # xi_m = xi(m dt_dmd)
                            q[i] += ((np.conj(Lambda[i])) ** m) * dmd.dmdInprod(
                                phi_[i], xi_m, inprod
                            )  # (Lambda_i*)^m * <phi_i*, xi_m>
                        pbar.update()

                    with tqdm(total=numModes) as pbar:
                        Parallel(n_jobs=n_jobs, backend="threading")(
                            delayed(fillQ_)(i, q, pbar) for i in range(0, numModes)
                        )

                    coeffs = np.dot(Pinv, q)
                    ###########################################################################

                    # Embed coefficients into the DMD modes, the result will be called as
                    # the DMD modes from now on
                    for i in range(numModes):
                        # phi_[i] = phi[i]
                        phi[i] = coeffs[i] * phi_[i]

                    # Compute the norm of DMD modes
                    for i in range(numModes):
                        magnitude[i] = np.abs(dmd.dmdInprod(phi[i], phi[i], inprod))
                    scalmagn = magnitude * (np.abs(Lambda)) ** (mDMD)

        # Save stuff
        if saveCoeff:
            np.savetxt(
                dataFolder / f"coeff_{striWindow}.gp", coeffs,
            )

        if saveExponents:
            np.savetxt(
                dataFolder / f"norm_{striWindow}.gp",  # magnitude,
                np.column_stack((magnitude, scalmagn)),
            )

        # Reconstruct states
        states_ = {}

        for j in range(numModes):
            for i in range(mDMD):
                if j == 0:
                    states_[i] = np.exp(exponents[j] * i * dt_dmd) * phi[j]
                else:
                    states_[i] += np.exp(exponents[j] * i * dt_dmd) * phi[j]

        if saveReconstruction:
            strlen_recon = len(str(mDMD))
            for idx in range(mDMD):
                rout = str(
                    (
                        dataFolder
                        / f"recon-{striWindow}-{str(idx).zfill(strlen_recon)}"
                    ).resolve()
                )
                states_[idx].real.save(rout)

        # Reconstruction errors (real part only)
        relErr = np.zeros(mDMD)

        def fillRelErrs(i):
            i_ = i * nSeparation
            stateI = state(i_, use_all=True)
            kinStateI = inprod(stateI, stateI)
            # Taking real parts here, imaginary parts are checked next
            deltaState = stateI - states_[i].real
            relErr[i] = inprod(deltaState, deltaState) / kinStateI

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(fillRelErrs)(i) for i in range(mDMD)
        )

        if saveExponents:
            np.savetxt(
                dataFolder / f"exp_{striWindow}.gp", exponents,
            )

        if saveErrors:
            np.savetxt(
                dataFolder / f"err_{striWindow}.gp",
                np.column_stack((times_window, relErr)),
            )

        relErrAvg = np.sum(np.sqrt(relErr)) / mDMD

        np.savetxt(
            fmeanerr, np.column_stack((times_window[0], relErrAvg, numModes)),
        )
        fmeanerr.flush()
        print("DMD residual (L2 norm!)=", relErrAvg, "with N=", numModes)
        if rpo:
            np.savetxt(
                fperiodicity, np.column_stack((times_window[0], error, Tg)),
            )
            fperiodicity.flush()
        if saveModes:
            strlen_modes = len(str(numModes))
            for idx in range(numModes):
                rout = str(
                    (
                        dataFolder
                        / f"dmd-{striWindow}-{str(idx).zfill(strlen_modes)}-r"
                    ).resolve()
                )
                iout = str(
                    (
                        dataFolder
                        / f"dmd-{striWindow}-{str(idx).zfill(strlen_modes)}-i"
                    ).resolve()
                )
                phi_[idx].real.save(rout)
                phi_[idx].imag.save(iout)

        if saveSnap1:
            strlen_recon = len(str(mDMD))
            strlen_modes = len(str(numModes))
            for idxt in range(mDMD):
                for idxm in range(numModes):
                    rout = str(
                        (
                            dataFolder
                            / f"saveSnap1"
                            / f"snapdmd-w{striWindow}-t{str(idxt).zfill(strlen_recon)}-m{str(idxm).zfill(strlen_modes)}-r"
                        ).resolve()
                    )
                    iout = str(
                        (
                            dataFolder
                            / f"saveSnap1"
                            / f"snapdmd-w{striWindow}-t{str(idxt).zfill(strlen_recon)}-m{str(idxm).zfill(strlen_modes)}-i"
                        ).resolve()
                    )
                    snap = np.exp(exponents[idxm] * idxt * dt_dmd) * phi[idxm]
                    snap.real.save(rout)
                    snap.imag.save(iout)

        if plotErrors:
            fig, ax = plt.subplots()
            ax.plot(times_window, relErr)
            ax.set_xlim(times_window[0], times_window[-1])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("$t$")
            ax.set_ylabel("Rel. err.")
            ax.set_title(title)
            fig.savefig(dataFolder / f"err_{striWindow}.png")
            plt.close(fig)

        if plotExponents:
            fig, ax = plt.subplots()
            ax.scatter(
                np.real(exponents), np.imag(exponents), marker="x", s=80, zorder=9,
            )
            ax.set_xlabel("$\\mu$")
            ax.set_ylabel("$\\omega$")
            ax.grid(axis="both")
            ax.set_title(title)
            fig.savefig(dataFolder / f"exp_{striWindow}.png")
            plt.close(fig)

        if plotSpec:
            fig, ax = plt.subplots()
            xs = np.imag(exponents) / (2.0 * math.pi)
            ys = magnitude * (np.abs(Lambda)) ** (mDMD)
            colors = cm.rainbow(np.linspace(0, 1, len(ys)))

            for x, y, c in zip(xs, ys, colors):
                ax.scatter(x, y, color=c, marker="x", s=80, zorder=9)

            ax.set_yscale("log")
            plt.axvline(x=0, color="gray", linewidth=0.5)

            ax.set_ylabel("$|\Lambda|^m \,||\phi||^2_2$")
            ax.set_xlabel("$f$")
            ax.set_xlim(-0.005, None)
            #            ax.set_xlim(0.0000001, None)

            ax.set_title(title)
            fig.savefig(dataFolder / f"spec_{striWindow}.png")
            plt.close(fig)

        pbar.update()
    fmeanerr.close()
    if rpo:
        fperiodicity.close()
