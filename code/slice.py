#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import psutil
from joblib import delayed
from matplotlib import pyplot as plt

import cf

def_n_jobs = len(psutil.Process().cpu_affinity())
if def_n_jobs > 1:
    def_n_jobs = def_n_jobs - 1
print(f"Leaving {def_n_jobs} threads to parallelization.", flush=True)

def_joblib_backend = "loky"

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
        "--t_i",
        type=float,
        default=-np.inf,
        dest="t_i",
        help="initial time.",
    )
    parser.add_argument(
        "--t_f",
        type=float,
        default=np.inf,
        dest="t_f",
        help="final time.",
    )
    # parser.add_argument(
    #     "-n_jobs",
    #     default=def_n_jobs,
    #     type=int,
    #     dest="n_jobs",
    #     help="number of threads to use",
    # )
    parser.add_argument(
        "--savestates",
        action="store_true",
        dest="savestates",
        help="save sliced states to disk.",
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
    # n_jobs=def_n_jobs,
    n_jobs=1,
    t_i=-np.inf,
    t_f=np.inf,
    savestates=False,
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

    _, state0, _, ygrid, _, props = cf.read_state(statefiles[0])
    lx = props["lx"]
    ly = props["ly"]
    lz = props["lz"]
    a = props["a"]
    b = props["b"]
    _, tx, _, _, _, _ = cf.read_state(savedir / f"opt_slicetemp_x.nc")
    _, tz, _, _, _, _ = cf.read_state(savedir / f"opt_slicetemp_z.nc")
    tgx = cf.apply_shifts(tx, [0.25, 0], lx, lz)
    tgz = cf.apply_shifts(tz, [0, 0.25], lx, lz)

    projections_x = np.zeros((n_states), dtype=np.complex128)
    projections_z = np.zeros((n_states), dtype=np.complex128)

    shifts = np.zeros((n_states, 2))

    def fill_shifts(i, time):
        _, state, _, _, _, _ = cf.read_state(statefiles[i])
        shiftx, shiftz, ptgx, ptx, ptgz, ptz = cf.find_shifts(state, tgx, tx, tgz, tz, ygrid, ly=ly)
        projections_x[i] = ptx + 1j * ptgx
        projections_z[i] = ptz + 1j * ptgz
        shifts[i, :] = np.array([shiftx, shiftz])
        state = cf.apply_shifts(state, shifts[i, :], lx, lz)

        if savestates:
            statename = Path(statefiles[i]).name
            stateout = savedir / statename
            cf.write_state(stateout, state, lx, lz, a=a, b=b)

    cf.ProgressParallel(n_jobs=n_jobs, backend=def_joblib_backend, total=n_states)(
        delayed(fill_shifts)(i, times[i]) for i in range(n_states)
    )

    np.savetxt(savedir / "shifts.gp", shifts)
    np.savetxt(savedir / "projections_x.gp", projections_x.view(float))
    np.savetxt(savedir / "projections_z.gp", projections_z.view(float))

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


if __name__ == "__main__":
    main()
