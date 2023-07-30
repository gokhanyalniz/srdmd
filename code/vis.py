#!/usr/bin/env python3
import argparse
from pathlib import Path

import cf
import numpy as np
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(
        description="Produce 3D visualizations of a state",
    )
    parser.add_argument("state", type=str, help="path to the state.")
    parser.add_argument(
        "--noshow", action="store_true", dest="noshow", help="do not display the plots."
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
        "--show_axes",
        action="store_true",
        dest="show_axes",
        help="display compass.",
    )
    parser.add_argument(
        "--show_bounds",
        action="store_true",
        dest="show_bounds",
        help="display ticks on the grid.",
    )

    args = vars(parser.parse_args())

    vis(**args)


def vis(
    state,
    noshow=False,
    xvfb=False,
    mirror_y=False,
    cvel=0.5,
    cvor=0.5,
    manual=False,
    lvelmin=0,
    lvelmax=0,
    lvormin=0,
    lvormax=0,
    show_axes=False,
    show_bounds=False,
):
    if xvfb:
        noshow = True
        pv.start_xvfb()

    pv.set_plot_theme("document")
    state = Path(state)
    _, velocity, xgrid, ygrid, zgrid, props = cf.read_state(state)

    velx = velocity[0, :, :, :]
    vorx = cf.vorticity(velocity, ygrid, props["lx"], props["lz"])

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
        print("ny_display = ", ny_display)
    else:
        ny_display = ny

    if not manual:
        vel_levels = cvel * np.array([np.amin(velx), np.amax(velx)])
        vor_levels = cvor * np.array([np.amin(vorx), np.amax(vorx)])

        print("vel_levels:", vel_levels)
        print("vor_levels:", vor_levels)
    else:
        vel_levels = np.array([lvelmin, lvelmax])
        vor_levels = np.array([lvormin, lvormax])

    grid = pv.RectilinearGrid(xgrid, ygrid, zgrid)

    grid.point_data["velx"] = np.reshape(velx, (nx * ny_display * nz), order="F")
    grid.point_data["vorx"] = np.reshape(vorx, (nx * ny_display * nz), order="F")

    p = pv.Plotter(off_screen=noshow)
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

    cpos = p.show(
        f"{state.name}.png",
        return_cpos=True,
    )

    # print("cpos:", cpos)


if __name__ == "__main__":
    main()
