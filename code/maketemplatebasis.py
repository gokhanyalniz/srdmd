#!/usr/bin/env python3
import argparse
from pathlib import Path

import cf
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


def main():
    parser = argparse.ArgumentParser(
        description="Creates unit templates to project onto."
    )

    parser.add_argument(
        "outdir", type=str, help="path to the directory to save the unit templates to."
    )
    parser.add_argument(
        "n_basis", type=int, help="will construct a basis of size n_basis."
    )
    parser.add_argument(
        "lx",
        type=float,
    )
    parser.add_argument(
        "lz",
        type=float,
    )
    parser.add_argument(
        "nx",
        type=int,
    )
    parser.add_argument(
        "ny",
        type=int,
    )
    parser.add_argument(
        "nz",
        type=int,
    )
    parser.add_argument(
        "-a",
        type=float,
        default=-1,
        dest="a",
    )
    parser.add_argument(
        "-b",
        type=float,
        default=1,
        dest="b",
    )

    args = vars(parser.parse_args())
    maketemplatebasis(**args)


def maketemplatebasis(
    outdir,
    n_basis,
    lx,
    lz,
    nx,
    ny,
    nz,
    a=-1,
    b=1,
):
    write_unit_templates(outdir, n_basis, lx, lz, nx, ny, nz, a=a, b=b)

def write_unit_templates(outdir, n_basis, lx, lz, nx, ny, nz, a=-1, b=1):
    outdir = Path(outdir)

    for n in range(n_basis):
        coeffs = [0.0 for n_ in range(n_basis)]
        coeffs[n] = 1.0
        for component, tag in enumerate(["x", "y", "z"]):
            fy = Chebyshev(coeffs, domain=[a, b])

            tempx_name = f"slicetemp_x_{tag}_{n}.nc"
            tempz_name = f"slicetemp_z_{tag}_{n}.nc"

            tempx = np.zeros((3, nx, ny, nz))
            tempz = np.zeros((3, nx, ny, nz))

            xgrid = np.linspace(0, lx, nx, endpoint=False)
            ygrid = ((cf.ygrid_chebyshev_extrema(ny) + 1) / 2) * (b - a) + a
            zgrid = np.linspace(0, lz, nz, endpoint=False)

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        tempx[component, i, j, k] = np.float64(
                            fy(ygrid[j]) * np.cos(xgrid[i] * 2 * np.pi / lx)
                        )

                        tempz[component, i, j, k] = np.float64(
                            fy(ygrid[j]) * np.cos(zgrid[k] * 2 * np.pi / lz)
                        )

            cf.write_state(outdir / tempx_name, tempx, lx, lz, a=a, b=b)
            cf.write_state(outdir / tempz_name, tempz, lx, lz, a=a, b=b)

if __name__ == "__main__":
    main()

