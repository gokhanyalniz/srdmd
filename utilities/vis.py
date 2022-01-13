#!/usr/bin/env python3
import argparse
import sys
from os import getenv
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import scipy.interpolate as spi

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
        default=0.25,
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
        "--show_axes", action="store_true", dest="show_axes", help="display compass.",
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
    cvor=0.25,
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
    velocity = cf.FlowField(str(state.resolve()))

    vorticity = cf.curl(velocity)
    vorticity.zeroPaddedModes()
    state_vorticity = state.parent / f"vor_{state.name}"
    vorticity.save(str(state_vorticity.resolve()))

    xgrid, ygrid, zgrid, velx, _, _ = channelflow_to_numpy(state)
    _, _, _, vorx, _, _ = channelflow_to_numpy(state_vorticity)
    nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)
    # lx = velocity.Lx
    # lz = velocity.Lz
    # print(f"nx, ny, nz = {nx}, {ny}, {nz}")
    # print(f"lx, lz = {lx}, {lz}")

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

    state_vorticity.unlink()

    grid = pv.RectilinearGrid(xgrid, ygrid, zgrid)
    # sad to need order="F" here
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
            opacity=0.35,
            cmap=["blue", "red"],
            clim=vel_levels,
            show_scalar_bar=False,
        )
    contour_vor = grid.contour(isosurfaces=vor_levels, scalars="vorx")
    if contour_vor.n_points > 0:
        p.add_mesh(
            contour_vor,
            smooth_shading=True,
            opacity=0.35,
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

    p.camera.roll += 90
    p.camera.elevation -= 15
    p.camera.azimuth -= 45
    p.camera.roll += 30
    p.camera.azimuth -= 45
    p.camera.roll -= 10

    # Old orientations from neubauten
    # p.camera.position = [-3.8165567986481825, 2.432563985013201, 1.623442321119651]
    # p.camera.focal_point = [1.5630592574663666, 0.15588548135152808, 0.4599843715554596]
    # p.camera.view_angle = 30.0
    # p.camera.view_up = [0.37308813472900676, 0.9240636797097904, -0.08313578992006013]
    # p.camera.clipping_range = [2.2454569218403426, 10.160445581565819]

    # Elena's numbers
    # scene.scene.camera.position = [-3.8165567986481825, 2.432563985013201, 1.623442321119651]
    # scene.scene.camera.focal_point = [1.5630592574663666, 0.15588548135152808, 0.4599843715554596]
    # scene.scene.camera.view_angle = 30.0
    # scene.scene.camera.view_up = [0.37308813472900676, 0.9240636797097904, -0.08313578992006013]
    # scene.scene.camera.clipping_range = [2.2454569218403426, 10.160445581565819]
    # scene.scene.camera.compute_view_plane_normal()

    p.show(screenshot=f"{state.name}_isosurf.png")


def channelflow_to_numpy(statefile, uniform=False):

    with h5py.File(statefile, mode="r") as file:

        # channelflow grid
        ygrid_c = np.array(file["Y"])[::-1]
        ny = len(ygrid_c)

        # uniform grid
        ygrid_u = np.linspace(ygrid_c[0], ygrid_c[-1], ny)

        if uniform:
            ygrid = ygrid_u
        else:
            ygrid = ygrid_c

        xgrid = np.array(file["X"])
        zgrid = np.array(file["Z"])
        nx = len(xgrid)
        nz = len(zgrid)

        def chebyshev_to_uniform(data):
            inarray = np.moveaxis(np.array(data), [0, 1, 2], [2, 1, 0])[:, ::-1, :]
            outarray = np.zeros((nx, ny, nz))
            if uniform:
                for i in range(nx):
                    for k in range(nz):
                        interpolator = spi.interp1d(
                            ygrid_c, inarray[i, :, k], kind="cubic"
                        )
                        outarray[i, :, k] = interpolator(ygrid_u)

                return outarray
            else:
                return inarray

        # Copy over the data
        velx = chebyshev_to_uniform(file["Velocity_X"])
        vely = chebyshev_to_uniform(file["Velocity_Y"])
        velz = chebyshev_to_uniform(file["Velocity_Z"])

        return xgrid, ygrid, zgrid, velx, vely, velz


if __name__ == "__main__":
    main()
