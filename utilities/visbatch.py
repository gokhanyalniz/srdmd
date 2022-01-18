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
# ffmpeg -framerate 24 -i %03d.png -c:v libx264 -r 60 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4

# Two videos can be merged side-by-side with
# ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4

# This script removes the laminar part


def main():

    parser = argparse.ArgumentParser(
        description="Produce 3D visualizations of a state",
    )
    parser.add_argument("statesdir", type=str, help="path to the state.")
    parser.add_argument(
        "savedir", type=str, help="where to save the stills.",
    )
    parser.add_argument(
        "--t_i", type=float, default=-np.inf, dest="t_i", help="initial time.",
    )
    parser.add_argument(
        "--t_f", type=float, default=np.inf, dest="t_f", help="final time.",
    )
    parser.add_argument(
        "--userecon",
        action="store_true",
        dest="userecon",
        help="expect DMD-reconstructed data as input.",
    )
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
        "--manual",
        action="store_true",
        dest="manual",
        help="use manually provided (lvelmin/max, lvormin/max) isosurface levels.",
    )
    parser.add_argument(
        "--absolutelevels",
        action="store_true",
        dest="absolutelevels",
        help="use isosurface levels (computed with cvel/cvor) that remain constant over time.",
    )
    parser.add_argument(
        "-lvelmin",
        default=0,
        type=float,
        dest="lvelmin",
        help="lower velocity isosurface",
    )
    parser.add_argument(
        "-lvelmax",
        default=0,
        type=float,
        dest="lvelmax",
        help="upper velocity isosurface",
    )
    parser.add_argument(
        "-lvormin",
        default=0,
        type=float,
        dest="lvormin",
        help="lower vorticity isosurface",
    )
    parser.add_argument(
        "-lvormax",
        default=0,
        type=float,
        dest="lvormax",
        help="upper vorticity isosurface",
    )
    parser.add_argument(
        "--mirror_y",
        action="store_true",
        dest="mirror_y",
        help="display the fundamental domain of mirror_y.",
    )
    parser.add_argument(
        "--show_axes", action="store_true", dest="show_axes", help="display compass.",
    )
    parser.add_argument(
        "--show_bounds",
        action="store_true",
        dest="show_bounds",
        help="display ticks on the grid.",
    )
    args = vars(parser.parse_args())

    visbatch(**args)


def visbatch(
    statesdir,
    savedir,
    t_i=-np.inf,
    t_f=np.inf,
    userecon=False,
    xvfb=False,
    cvel=0.5,
    cvor=0.5,
    manual=False,
    absolutelevels=False,
    lvelmin=0,
    lvelmax=0,
    lvormin=0,
    lvormax=0,
    mirror_y=False,
    print_messages=True,
    n_jobs=def_n_jobs,
    show_axes=False,
    show_bounds=False,
):

    statesdir = Path(statesdir)
    savedir = Path(savedir)
    if not userecon:
        states = sorted(list(statesdir.glob("u*.nc")))
        times = np.array([float(stateFile.name[1:-3]) for stateFile in states])
        sorter = np.argsort(times)
        times = times[sorter]
        states = [states[i] for i in sorter]
        t_filter = np.nonzero(np.logical_and(times >= t_i, times < t_f))[0]
        times = times[t_filter]
        states = [states[i] for i in t_filter]
    else:
        states = sorted(list(statesdir.glob("recon*.nc")))

    if absolutelevels:
        min_vel, min_vor = np.inf, np.inf
        max_vel, max_vor = -np.inf, -np.inf

        print("Finding levels.")
        with tqdm(total=len(states), disable=not print_messages) as pbar:
            for i, state in enumerate(states):
                state = Path(state)
                velocity = cf.FlowField(str(state.resolve()))

                vorticity = cf.curl(velocity)
                vorticity.zeroPaddedModes()
                state_vorticity = savedir / f"vor_{state.name}"
                vorticity.save(str(state_vorticity.resolve()))

                xgrid, ygrid, zgrid, velx, _, _ = vis.channelflow_to_numpy(state)
                _, _, _, vorx, _, _ = vis.channelflow_to_numpy(state_vorticity)
                nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)

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

                min_vel, min_vor = (
                    min(min_vel, np.amin(velx)),
                    min(min_vor, np.amin(vorx)),
                )
                max_vel, max_vor = (
                    max(max_vel, np.amax(velx)),
                    max(max_vor, np.amax(vorx)),
                )
                pbar.update()

        lvelmin, lvelmax = cvel * min_vel, cvel * max_vel
        lvormin, lvormax = cvor * min_vor, cvor * max_vor

        print("vel_levels:", lvelmin, lvelmax)
        print("vor_levels:", lvormin, lvormax)

    if xvfb:
        pv.start_xvfb()
    pv.set_plot_theme("document")

    def render_state_i(i):
        state = states[i]
        if manual or absolutelevels:
            state_vorticity = savedir / f"vor_{state.name}"
        else:
            velocity = cf.FlowField(str(state.resolve()))

            vorticity = cf.curl(velocity)
            vorticity.zeroPaddedModes()
            state_vorticity = savedir / f"vor_{state.name}"
            vorticity.save(str(state_vorticity.resolve()))

        xgrid, ygrid, zgrid, velx, _, _ = vis.channelflow_to_numpy(state)
        _, _, _, vorx, _, _ = vis.channelflow_to_numpy(state_vorticity)
        nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)

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

        if manual or absolutelevels:
            vel_levels = np.array([lvelmin, lvelmax])
            vor_levels = np.array([lvormin, lvormax])
        else:
            vel_levels = cvel * np.array([np.amin(velx), np.amax(velx)])
            vor_levels = cvor * np.array([np.amin(vorx), np.amax(vorx)])

        state_vorticity.unlink()

        grid = pv.RectilinearGrid(xgrid, ygrid, zgrid)
        # sad to need order="F" here
        grid.point_data["velx"] = np.reshape(velx, (nx * ny_display * nz), order="F")
        grid.point_data["vorx"] = np.reshape(vorx, (nx * ny_display * nz), order="F")

        p = pv.Plotter(off_screen=True)
        p.set_background("white")
        p.add_mesh(grid.outline(), color="k")
        contour_vel = grid.contour(isosurfaces=vel_levels, scalars="velx")
        if contour_vel.n_points > 0:
            p.add_mesh(
                contour_vel,
                smooth_shading=True,
                opacity=0.5,
                cmap=["blue", "red"],
                clim=vel_levels,
                show_scalar_bar=False,
            )
        contour_vor = grid.contour(isosurfaces=vor_levels, scalars="vorx")
        if contour_vor.n_points > 0:
            p.add_mesh(
                contour_vor,
                smooth_shading=True,
                opacity=1.0,
                cmap=["purple", "green"],
                clim=vor_levels,
                show_scalar_bar=False,
            )
        if show_axes:
            p.show_axes()

        if show_bounds:
            if not show_axes:
                p.show_bounds(xlabel="x", ylabel="y", zlabel="z")
            else:
                p.show_bounds(xlabel="", ylabel="", zlabel="")

        cpos = [
            (-8.7087276075403, 5.0595647811549345, 5.758270814130649),
            (2.6920391113975386, -0.16116068344232165, 1.9920099095386092),
            (0.3727620195457797, 0.9169022252699194, -0.14261411598864254),
        ]
        p.show(screenshot=savedir / f"{state.name}_isosurf.png", cpos=cpos)

        # hope memory doesn't leak
        p.clear()
        p.deep_clean()

    print("Rendering frames.")
    Parallel(n_jobs=n_jobs, backend=def_joblib_backend, verbose=def_joblib_verbosity)(
        delayed(render_state_i)(i) for i in tqdm(range(len(states)))
    )


if __name__ == "__main__":
    main()
