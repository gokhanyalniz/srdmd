#!/usr/bin/env python3
import argparse
import sys
from os import getenv
from pathlib import Path

import numpy as np
import psutil
import pyvista as pv
from joblib import Parallel, delayed
from tqdm import tqdm

import vis

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

def_n_jobs = len(psutil.Process().cpu_affinity())
if def_n_jobs > 1:
    def_n_jobs = def_n_jobs - 1

def_joblib_verbosity = 0
def_joblib_backend = "threading"

# Example ffmpeg command to merge png files to an mp4:
# ffmpeg -framerate 24 -i %06d.png -c:v libx264 -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4

# This script removes the laminar part


def main():

    parser = argparse.ArgumentParser(
        description="Produce 3D visualizations of a state",
    )
    parser.add_argument("statesDir", type=str, help="path to the state.")
    parser.add_argument(
        "--xvfb", action="store_true", dest="xvfb", help="render to a virtual display."
    )
    parser.add_argument(
        "-cvel",
        default=0.5,
        type=float,
        dest="cvel",
        help="multiplier for velocity isosurfaces",
    )
    parser.add_argument(
        "-cvor",
        default=0.5,
        type=float,
        dest="cvor",
        help="multiplier for vorticity isosurfaces",
    )
    parser.add_argument(
        "--mirror_y",
        action="store_true",
        dest="mirror_y",
        help="display the fundamental domain of mirror_y.",
    )
    args = vars(parser.parse_args())

    visbatch(**args)


def visbatch(
    statesDir,
    xvfb=False,
    cvel=0.5,
    cvor=0.5,
    mirror_y=False,
    print_messages=True,
    n_jobs=def_n_jobs,
):

    statesDir = Path(statesDir)
    states = sorted(list(statesDir.glob("u*.nc")))
    min_vel, min_vor = np.inf, np.inf
    max_vel, max_vor = -np.inf, -np.inf

    print("Finding the maxima in the snapshots.")
    with tqdm(total=len(states), disable=not print_messages) as pbar:
        for i, state in enumerate(states):
            state = Path(state)
            velocity = cf.FlowField(str(state.resolve()))

            vorticity = cf.curl(velocity)
            vorticity.zeroPaddedModes()
            state_vorticity = state.parent / f"vor_{state.name}"
            vorticity.save(str(state_vorticity.resolve()))

            xgrid, ygrid, zgrid, velx, _, _ = vis.channelflow_to_numpy(state)
            _, _, _, vorx, _, _ = vis.channelflow_to_numpy(state_vorticity)
            nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)
            lx = velocity.Lx
            lz = velocity.Lz

            ny_display = ny
            if mirror_y:
                if ny % 2 != 0:
                    ny_display = ny // 2 + 1
                else:
                    ny_display = ny // 2

                ygrid = ygrid[-ny_display:]
                velx = velx[:, -ny_display:, :]
                vorx = vorx[:, -ny_display:, :]
            else:
                ny_display = ny

            min_vel, min_vor = min(min_vel, np.amin(velx)), min(min_vor, np.amin(vorx))
            max_vel, max_vor = max(max_vel, np.amax(velx)), max(max_vor, np.amax(vorx))
            pbar.update()

    print("Minima:")
    print(min_vel, min_vor)
    print("Maxima:")
    print(max_vel, max_vor)

    if xvfb:
        pv.start_xvfb()

    def render_state_i(i):
        state = Path(states[i])
        velocity = cf.FlowField(str(state.resolve()))

        vorticity = cf.curl(velocity)
        vorticity.zeroPaddedModes()
        state_vorticity = state.parent / f"vor_{state.name}"
        vorticity.save(str(state_vorticity.resolve()))

        xgrid, ygrid, zgrid, velx, _, _ = vis.channelflow_to_numpy(state)
        _, _, _, vorx, _, _ = vis.channelflow_to_numpy(state_vorticity)
        nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)
        lx = velocity.Lx
        lz = velocity.Lz

        ny_display = ny
        if mirror_y:
            if ny % 2 != 0:
                ny_display = ny // 2 + 1
            else:
                ny_display = ny // 2

            ygrid = ygrid[-ny_display:]
            velx = velx[:, -ny_display:, :]
            vorx = vorx[:, -ny_display:, :]
        else:
            ny_display = ny

        vel_levels = cvel * np.array([np.amin(velx), np.amax(velx)])
        vor_levels = cvor * np.array([np.amin(vorx), np.amax(vorx)])

        grid = pv.RectilinearGrid(xgrid, ygrid, zgrid)
        # sad to need order="F" here
        grid.point_data["velx"] = np.reshape(velx, (nx * ny_display * nz), order="F")
        grid.point_data["vorx"] = np.reshape(vorx, (nx * ny_display * nz), order="F")

        p = pv.Plotter(off_screen=True)
        p.add_mesh(grid.outline(), color="k")
        p.add_mesh(
            grid.contour(isosurfaces=vel_levels, scalars="velx"),
            smooth_shading=True,
            opacity=0.35,
            cmap=["blue", "red"],
            clim=vel_levels,
            show_scalar_bar=False,
        )
        p.add_mesh(
            grid.contour(isosurfaces=vor_levels, scalars="vorx"),
            smooth_shading=True,
            opacity=0.35,
            cmap=["purple", "green"],
            clim=vor_levels,
            show_scalar_bar=False,
        )
        p.show_axes()
        p.show_bounds(xlabel="x", ylabel="y", zlabel="z")

        p.camera.roll += 90
        p.camera.elevation -= 15
        p.camera.azimuth -= 45
        p.camera.roll += 30
        p.camera.azimuth -= 45
        p.camera.roll -= 10

        p.show(screenshot=f"{state.stem}_isosurf.png")

        # hope memory doesn't leak
        p.clear()
        p.deep_clean()

    Parallel(n_jobs=n_jobs, backend=def_joblib_backend, verbose=def_joblib_verbosity)(
        delayed(render_state_i)(i) for i in tqdm(range(len(states)))
    )


if __name__ == "__main__":
    main()
