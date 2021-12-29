#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import scipy.interpolate as spi

import libpycf as cf


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
        "--addbase",
        action="store_true",
        dest="addbase",
        help="remove laminar part from states.",
    )
    parser.add_argument(
        "-cvel",
        default=0.75,
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

    args = vars(parser.parse_args())

    dnsvis(**args)


def dnsvis(
    state, noshow=False, xvfb=False, addbase=False, cvel=0.75, cvor=0.5,
):

    if xvfb:
        noshow = True
        pv.start_xvfb()

    pv.set_plot_theme("document")
    state = Path(state)
    velocity = cf.FlowField(str(state.resolve()))
    nx_ = velocity.Nx
    ny_ = velocity.Ny
    nz_ = velocity.Nz
    lx = velocity.Lx
    lz = velocity.Lz
    a = velocity.a
    b = velocity.b

    vorticity = cf.curl(velocity)
    vorticity.zeroPaddedModes()
    state_vorticity = state.parent / f"{state.stem}_vor.nc"
    vorticity.save(str(state_vorticity.resolve()))

    xgrid, ygrid, zgrid, velx, _, _ = channelflow_to_numpy(state)
    _, _, _, vorx, _, _ = channelflow_to_numpy(state_vorticity)
    dx, dz = xgrid[1] - xgrid[0], zgrid[1] - zgrid[0]
    nx, ny, nz = len(xgrid), len(ygrid), len(zgrid)
    # print(f"lx, ly, lz = {dx*nx}, {ygrid[-1]-ygrid[0]}, {dz*nz}")
    # print(f"nx, ny, nz = {nx}, {ny}, {nz}")

    u = pv.wrap(velx)
    om = pv.wrap(vorx)

    vel_levels = cvel * np.array([np.min(velx), np.max(velx)])
    vor_levels = cvor * np.array([np.min(vorx), np.max(vorx)])

    p = pv.Plotter(off_screen=noshow)

    p.add_mesh(u.outline(), color="k")
    p.add_mesh(
        u.contour(vel_levels),
        smooth_shading=True,
        opacity=0.35,
        cmap=["red", "blue"],
        clim=vel_levels,
        show_scalar_bar=False,
    )
    p.add_mesh(
        om.contour(vor_levels),
        smooth_shading=True,
        opacity=0.35,
        cmap=["green", "purple"],
        clim=vor_levels,
        show_scalar_bar=False,
    )
    p.show_axes()

    #
    p.camera.roll += 90
    p.camera.elevation -= 15
    p.camera.azimuth -= 45
    p.camera.roll += 30
    p.camera.azimuth -= 45
    p.camera.roll -= 10

    p.show(screenshot=f"{state.stem}_isosurf.png")


def channelflow_to_numpy(statefile, uniform=True):

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
