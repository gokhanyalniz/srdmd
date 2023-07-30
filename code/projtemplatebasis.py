#!/usr/bin/env python3
import argparse
from pathlib import Path

import cf
import numpy as np
import psutil
from joblib import delayed

def_n_jobs = len(psutil.Process().cpu_affinity())
if def_n_jobs > 1:
    def_n_jobs = def_n_jobs - 1
print(f"Leaving {def_n_jobs} threads to parallelization.", flush=True)

def_joblib_backend = "loky"


def main():
    parser = argparse.ArgumentParser(
        description="Computes projections against unit templates and finds the optimal linear combination that maximizes the projection amplitudes."
    )

    parser.add_argument("rundir", type=str, help="path to the directory of the run.")
    parser.add_argument(
        "savedir", type=str, help="path to the directory to save results."
    )
    parser.add_argument(
        "n_basis", type=int, help="will check against a basis of size n_basis."
    )
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
    # parser.add_argument(
    #     "-n_jobs",
    #     default=def_n_jobs,
    #     type=int,
    #     dest="n_jobs",
    #     help="number of threads to use",
    # )

    args = vars(parser.parse_args())
    template_opt(**args)


def template_opt(
    rundir,
    savedir,
    n_basis,
    t_i=-np.inf,
    t_f=np.inf,
    reprocess=False,
    f_subsamp=1,
    # n_jobs=def_n_jobs,
    n_jobs=1,
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

    print(
        f"Projecting from time {times[0]} to time {times[-1]} every {f_subsamp}'th state.",
        flush=True,
    )

    _, state0, _, ygrid, _, props = cf.read_state(statefiles[0])
    lx = props["lx"]
    ly = props["ly"]
    lz = props["lz"]
    a = props["a"]
    b = props["b"]

    templates = []
    template_tags = []

    component_tags = ["x", "y", "z"]
    for i in range(n_basis):
        for component_tag in component_tags:
            tag = f"{component_tag}_{i}"

            _, tx, _, _, _, _ = cf.read_state(savedir / f"slicetemp_x_{tag}.nc")
            _, tz, _, _, _, _ = cf.read_state(savedir / f"slicetemp_z_{tag}.nc")
            norm_tx = np.sqrt(cf.inprod(tx, tx, ygrid, ly=ly))
            norm_tz = np.sqrt(cf.inprod(tz, tz, ygrid, ly=ly))
            tx = (1 / norm_tx) * tx
            tz = (1 / norm_tz) * tz
            tgx = cf.apply_shifts(tx, [0.25, 0], lx, lz)
            tgz = cf.apply_shifts(tz, [0, 0.25], lx, lz)

            templates.append([tgx, tx, tgz, tz])
            template_tags.append(tag)

    if not reprocess:
        projections_x = np.zeros((n_states, 3 * n_basis), dtype=np.complex128)
        projections_z = np.zeros((n_states, 3 * n_basis), dtype=np.complex128)
        norms = np.zeros((n_states))

        def filldata(i):
            _, state, _, _, _, _ = cf.read_state(statefiles[i])
            for j, template in enumerate(templates):
                tgx, tx, tgz, tz = template
                shiftx, shiftz, ptgx, ptx, ptgz, ptz = cf.find_shifts(
                    state, tgx, tx, tgz, tz, ygrid
                )
                projections_x[i, j] = ptx + 1j * ptgx
                projections_z[i, j] = ptz + 1j * ptgz
                norms[i] = np.sqrt(cf.inprod(state, state, ygrid, ly=ly))

        cf.ProgressParallel(n_jobs=n_jobs, backend=def_joblib_backend, total=n_states)(
            delayed(filldata)(i) for i in range(n_states)
        )

        np.savetxt(savedir / "projections_x.gp", projections_x.view(float))
        np.savetxt(savedir / "projections_z.gp", projections_z.view(float))
        np.savetxt(savedir / "norms.gp", norms)

    else:
        projections_x = np.loadtxt(savedir / "projections_x.gp").view(complex)
        projections_z = np.loadtxt(savedir / "projections_z.gp").view(complex)
        norms = np.loadtxt(savedir / "norms.gp")

    # exact solution
    for iproj, proj in [("x", projections_x), ("z", projections_z)]:

        # might need to swap the complex conjugates below
        u, _, _ = np.linalg.svd(
            np.einsum("ki,kj", np.conj(proj), proj).real, hermitian=True
        )
        coeffs = u[:, 0] / np.sqrt(np.sum(u[:, 0] * u[:, 0]))

        # save coefficients
        np.savetxt(savedir / f"coefficients_{iproj}.gp", coeffs)

        template = np.zeros(state0.shape)
        for icoeff in range(len(coeffs)):
                if iproj == "x":
                    template += coeffs[icoeff] * templates[icoeff][1]
                elif iproj == "z":
                    template += coeffs[icoeff] * templates[icoeff][3]

        norm = np.sqrt(cf.inprod(template, template, ygrid, ly=ly))
        template = template / norm

        cf.write_state(savedir / f"opt_slicetemp_{iproj}.nc", template, lx, lz, a=a, b=b)

if __name__ == "__main__":
    main()
