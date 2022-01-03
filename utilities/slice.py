#!/usr/bin/env python3
import argparse
import sys
from os import getenv
from pathlib import Path

import numpy as np
import psutil
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

cmap = "plasma"

# fmt: off
# Set the paths to channelflow-python
CHANNELFLOW_PYTHON = getenv("CHANNELFLOW_PYTHON")
if CHANNELFLOW_PYTHON is None:
    exit("Set the alias CHANNELFLOW_PYTHON to use this script.")
else:
    CHANNELFLOW_PYTHON = Path(CHANNELFLOW_PYTHON)
sys.path.append(str((CHANNELFLOW_PYTHON / "lib").resolve()))
import libpycf as cf

# fmt: on

n_jobs = len(psutil.Process().cpu_affinity())
if n_jobs > 1:
    n_jobs = n_jobs - 1
print(f"Leaving {n_jobs} threads to parallelization.", flush=True)


def main():

    parser = argparse.ArgumentParser(description="Slice states.")

    parser.add_argument(
        "rundir", type=str, help="path to the directory of input state files."
    )
    parser.add_argument(
        "savedir",
        type=str,
        help="path to the directory to read templates from and save results to.",
    )
    parser.add_argument("dt", type=float, help="dt of states.")
    parser.add_argument(
        "--t_i", type=float, default=-np.inf, dest="t_i", help="initial time.",
    )
    parser.add_argument(
        "--t_f", type=float, default=np.inf, dest="t_f", help="final time.",
    )
    parser.add_argument(
        "--savestates",
        action="store_true",
        dest="savestates",
        help="save sliced states to disk.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        dest="reprocess",
        help="reprocess existing data.",
    )
    parser.add_argument(
        "--f_subsamp",
        type=int,
        default=1,
        dest="f_subsamp",
        help="subsampling frequency of the found state files.",
    )

    args = vars(parser.parse_args())
    slice(**args)


def slice(
    rundir,
    savedir,
    dt,
    t_i=-np.inf,
    t_f=np.inf,
    savestates=False,
    reprocess=False,
    f_subsamp=1,
):
    rundir = Path(rundir).resolve()
    savedir = Path(savedir).resolve()

    # Assume all the files with names "u*.nc" are Channelflow data files
    # Get the full paths to them
    statefiles = [str(stateFile.resolve()) for stateFile in rundir.glob("u*.nc")]
    # Extract times from file names
    # u[time].nc
    times = np.array(
        [float(stateFile.name[1:-3]) for stateFile in rundir.glob("u*.nc")]
    )
    # Sort in time
    sorter = np.argsort(times)
    times = times[sorter]
    statefiles = [statefiles[i] for i in sorter]
    # Filter in time
    t_filter = np.nonzero(np.logical_and(times >= t_i, times < t_f))[0]
    times = times[t_filter]
    statefiles = [statefiles[i] for i in t_filter]
    # Subsample
    times = times[::f_subsamp]
    statefiles = statefiles[::f_subsamp]

    n_states = len(statefiles)

    print(f"Slicing from time {times[0]} to time {times[- 1]} with dt {dt}.")

    if not reprocess:

        tx = cf.FlowField(str((savedir / f"slicetempx.nc").resolve()))
        tz = cf.FlowField(str((savedir / f"slicetempz.nc").resolve()))
        tgx = apply_shifts(tx, [-0.25, 0])
        tgz = apply_shifts(tz, [0, -0.25])

        shifts = np.zeros((n_states, 2))
        projections_x = np.zeros((n_states), dtype=np.complex128)
        projections_z = np.zeros((n_states), dtype=np.complex128)

        def fill_shifts(i):
            state = cf.FlowField(statefiles[i])
            shiftx, shiftz, ptgx, ptx, ptgz, ptz = find_shifts(state, tgx, tx, tgz, tz)
            shifts[i, :] = np.array([shiftx, shiftz])
            projections_x[i] = ptx + 1j * ptgx
            projections_z[i] = ptz + 1j * ptgz
            state = apply_shifts(state, shifts[i, :])
            if savestates:
                statename = Path(statefiles[i]).name
                stateout = str((savedir / statename).resolve())
                state.save(stateout)

        ProgressParallel(n_jobs=n_jobs, backend="threading", total=n_states)(
            delayed(fill_shifts)(i) for i in range(n_states)
        )

        np.savetxt(savedir / "shifts.gp", shifts)
        np.savetxt(savedir / "projections_x.gp", projections_x.view(float))
        np.savetxt(savedir / "projections_z.gp", projections_z.view(float))

    else:
        shifts = np.loadtxt(savedir / "shifts.gp")
        projections_x = np.loadtxt(savedir / "projections_x.gp").view(complex)
        projections_z = np.loadtxt(savedir / "projections_z.gp").view(complex)

    phases_x = np.unwrap(shifts[:, 0] * 2 * np.pi)
    phases_z = np.unwrap(shifts[:, 1] * 2 * np.pi)

    # Plot phases
    fig, ax = plt.subplots()
    ax.plot(times, phases_x)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\phi_x$")
    ax.set_xlim(left=times[0], right=times[-1])
    fig.savefig(savedir / f"phases_x.png", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(times, phases_z)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\phi_z$")
    ax.set_xlim(left=times[0], right=times[-1])
    fig.savefig(savedir / f"phases_z.png", bbox_inches="tight")

    dphases_x = (phases_x[1:] - phases_x[:-1]) / dt
    dphases_z = (phases_z[1:] - phases_z[:-1]) / dt

    # Plot phase derivatives
    fig, ax = plt.subplots()
    ax.plot(times[:-1], dphases_x)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\dot{{\\phi_x}}$")
    ax.set_xlim(left=times[0], right=times[-2])
    fig.savefig(savedir / f"dphases_x.png", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(times[:-1], dphases_z)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\dot{{\\phi_z}}$")
    ax.set_xlim(left=times[0], right=times[-2])
    fig.savefig(savedir / f"dphases_z.png", bbox_inches="tight")

    # Plot projections
    fig, ax = plt.subplots()
    ax.plot(times, np.abs(projections_x))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$|p_x|$")
    ax.set_xlim(left=times[0], right=times[-1])
    fig.savefig(savedir / f"projections_x.png", bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.plot(times, np.abs(projections_z))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$|p_z|$")
    ax.set_xlim(left=times[0], right=times[-1])
    fig.savefig(savedir / f"projections_z.png", bbox_inches="tight")


def shift_center(shift):
    if np.abs(shift - 1) < shift:
        shift_ = shift - 1
    else:
        shift_ = shift

    return shift_


def find_shifts(state, tgx, tx, tgz, tz):

    ptgx = cf.L2IP(state, tgx)
    ptx = cf.L2IP(state, tx)
    ptgz = cf.L2IP(state, tgz)
    ptz = cf.L2IP(state, tz)

    shiftx = shift_center((np.arctan2(ptgx, ptx) / (2 * np.pi)) % 1)
    shiftz = shift_center((np.arctan2(ptgz, ptz) / (2 * np.pi)) % 1)

    return np.array([shiftx, shiftz, ptgx, ptx, ptgz, ptz])


def apply_shifts(state, shifts):
    shifter = cf.FieldSymmetry(1, 1, 1, shifts[0], shifts[1], 1)
    return shifter(state)


def colorbar(ax, im, label=None):

    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, label=label)

    return cbar


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


if __name__ == "__main__":
    main()
