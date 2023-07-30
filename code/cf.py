#!/usr/bin/env python3
from sys import exit

import netCDF4 as nc
import numpy as np
from scipy.fft import irfftn, rfftn

from joblib import Parallel
from tqdm.auto import tqdm

def read_state(
    path,
    keep_spectral=False,
):
    with nc.Dataset(path, mode="r") as file:
        nx = file.dimensions["X"].size
        nz = file.dimensions["Z"].size
        ny = file.dimensions["Y"].size
        ny_cut = ny

        xgrid = np.array(file["X"][:])
        ygrid = np.array(file["Y"][:])[::-1]
        zgrid = np.array(file["Z"][:])

        a = file.a
        b = file.b
        ly = b-a

        # infer domain length
        dx = np.average(xgrid[1:] - xgrid[:-1])
        dz = np.average(zgrid[1:] - zgrid[:-1])
        lx = nx * dx
        lz = nz * dz

        physical = np.zeros((3, nx, ny_cut, nz))

        physical[0, :, :, :] = np.swapaxes(np.array(file["Velocity_X"][:]), 0, 2)[
            :, ::-1, :
        ]
        physical[1, :, :, :] = np.swapaxes(np.array(file["Velocity_Y"][:]), 0, 2)[
            :, ::-1, :
        ]
        physical[2, :, :, :] = np.swapaxes(np.array(file["Velocity_Z"][:]), 0, 2)[
            :, ::-1, :
        ]

        props = {
            "lx": lx,
            "ly": ly,
            "lz": lz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "a": a,
            "b": b,
        }

    if keep_spectral:
        spectral = phys_to_fourier(physical)
    else:
        spectral = None

    return spectral, physical, xgrid, ygrid, zgrid, props


def domain_wavelengths(lx, lz):
    alpha = 2 * np.pi / lx
    beta = 2 * np.pi / lz
    return alpha, beta


def ygrid_chebyshev_extrema(ny):
    ys = -np.cos(np.arange(ny) * np.pi / (ny - 1))
    return ys


def write_state(
    path,
    velocity,
    lx,
    lz,
    a=-1,
    b=1,
):
    nx, ny, nz = velocity.shape[1:]
    with nc.Dataset(path, "w", format="NETCDF4") as file:
        file.a = a
        file.b = b
        file.Lx = lx
        file.Lz = lz
        # assuming 3/2 expansion for x an z, no expansion for y
        file.Nx = 3 * (nx // 2)
        file.Ny = ny
        file.Nz = 3 * (nz // 2)
        file.createDimension("X", nx)
        file.createDimension("Y", ny)
        file.createDimension("Z", nz)

        file.createVariable("X", "f8", ("X"))
        file.createVariable("Y", "f8", ("Y"))
        file.createVariable("Z", "f8", ("Z"))

        file["X"][:] = np.linspace(0, lx, nx, endpoint=False)
        ygrid = ((ygrid_chebyshev_extrema(ny) + 1) / 2) * (b - a) + a
        file["Y"][:] = ygrid[::-1]
        file["Z"][:] = np.linspace(0, lz, nz, endpoint=False)

        # variables are left empty
        for component in ["X", "Y", "Z"]:
            file.createVariable(f"Velocity_{component}", "f8", ("Z", "Y", "X"))

        file["Velocity_X"][:,:,:] = np.swapaxes(velocity[0, :, ::-1, :], 0, 2)
        file["Velocity_Y"][:,:,:] = np.swapaxes(velocity[1, :, ::-1, :], 0, 2)
        file["Velocity_Z"][:,:,:] = np.swapaxes(velocity[2, :, ::-1, :], 0, 2)


def phys_to_fourier(physical):
    if len(physical.shape) == 4:
        # assume y is there
        spectral = rfftn(physical, norm="forward", axes=(1, 3))
    elif len(physical.shape) == 3:
        # assume this is ycut
        spectral = rfftn(physical, norm="forward", axes=(1, 2))

    return spectral


def fourier_to_phys(spectral):
    if len(spectral.shape) == 4:
        # assume y is there
        physical = irfftn(spectral, norm="forward", axes=(1, 3))
    elif len(spectral.shape) == 3:
        # assume this is a ycut
        physical = irfftn(spectral, norm="forward", axes=(1, 2))

    return physical


def phys_to_fourier_i(physical_i):
    if len(physical_i.shape) == 3:
        # assume y is there
        spectral = rfftn(physical_i, norm="forward", axes=(0, 2))
    elif len(physical_i.shape) == 2:
        # assume this is a ycut
        spectral = rfftn(physical_i, norm="forward", axes=(0, 1))

    return spectral


def fourier_to_phys_i(spectral_i):
    if len(spectral_i.shape) == 3:
        # assume y is there
        physical = irfftn(spectral_i, norm="forward", axes=(0, 2))
    elif len(spectral_i.shape) == 2:
        # assume this is a ycut
        physical = irfftn(spectral_i, norm="forward", axes=(0, 1))

    return physical


def derivative_spectral_i_inout_spec(spectral_i_ycut, axis, lx, lz):
    # computes the derivative of a given component
    # input assumed to be spectral in xz

    nx, nz = spectral_i_ycut.shape

    kx, kz = wavenumbers(lx, lz, nx, nz)

    out = np.zeros((spectral_i_ycut.shape), dtype=np.complex128)
    if axis == "x":
        for i in range(nx):
            out[i, :] = 1j * kx[i] * spectral_i_ycut[i, :]
    elif axis == "z":
        for k in range(nz // 2 + 1):
            out[:, k] = 1j * kz[k] * spectral_i_ycut[:, k]
    else:
        exit("wrong axis specified.")

    return out


def derivative_spectral_i_inout_phys(physical_i, axis, lx, lz):
    spectral_i = phys_to_fourier_i(physical_i)
    if len(physical_i.shape) == 3:
        # assume y is there
        derivative_spec = np.zeros(spectral_i.shape, dtype=np.complex128)
        for iy in range(physical_i.shape[1]):
            derivative_spec[:, iy, :] = derivative_spectral_i_inout_spec(
                spectral_i[:, iy, :], axis, lx, lz
            )
    elif len(physical_i.shape) == 2:
        # assume this is ycut
        derivative_spec = derivative_spectral_i_inout_spec(spectral_i, axis, lx, lz)

    derivative_phys = fourier_to_phys_i(derivative_spec)

    return derivative_phys


def vorticity(vfield, ygrid, lx, lz):
    omfield = np.zeros((vfield.shape[1:]))
    if lx is None or lz is None:
        exit("Need to provide Lx and Lz.")

    # om_x = d_y w - d_z v
    omfield = np.gradient(
        vfield[2, :, :, :], ygrid, axis=1
    ) - derivative_spectral_i_inout_phys(vfield[1, :, :, :], "z", lx, lz)

    return omfield


def inprod(state1, state2, ygrid, ly=2):
    # \int dx dy dz (u*u' + v*v' + w*w') / (lx*ly*lz)
    nx, ny, nz = state1.shape[1:]

    return np.trapz(np.sum(state1 * state2, axis=(0, 1, 3)), x=ygrid) / (nx * ly * nz)


def inprod_i(state1, state2, ygrid, ly=2):
    # \int dx dy dz u*u' / (lx*ly*lz)
    nx, ny, nz = state1.shape

    return np.trapz(np.sum(state1 * state2, axis=(0, 2)), x=ygrid) / (nx * ly * nz)


def Tx(dx, u, lx, lz):
    # Tx(dx) u(x) = u(x-dx)

    if len(u.shape) == 4:
        _, nx, ny, nz_half_p1 = u.shape
        nz = (nz_half_p1 - 1) * 2
        kkx, _ = wavenumbers(lx, lz, nx, nz)
        image = np.zeros((3, nx, ny, nz // 2 + 1), dtype=np.complex128)
        for ix, kx in enumerate(kkx):
            # Re part
            image[:, ix, :, :] = (
                np.cos(kx * dx) * u[:, ix, :, :].real
                + np.sin(kx * dx) * u[:, ix, :, :].imag
            )
            # Im part
            image[:, ix, :, :] += 1j * (
                -np.sin(kx * dx) * u[:, ix, :, :].real
                + np.cos(kx * dx) * u[:, ix, :, :].imag
            )
    elif len(u.shape) == 3:
        _, nx, nz_half_p1 = u.shape
        nz = (nz_half_p1 - 1) * 2
        kkx, _ = wavenumbers(lx, lz, nx, nz)
        image = np.zeros((3, nx, nz // 2 + 1), dtype=np.complex128)
        for ix, kx in enumerate(kkx):
            # Re part
            image[:, ix, :] = (
                np.cos(kx * dx) * u[:, ix, :].real + np.sin(kx * dx) * u[:, ix, :].imag
            )
            # Im part
            image[:, ix, :] += 1j * (
                -np.sin(kx * dx) * u[:, ix, :].real + np.cos(kx * dx) * u[:, ix, :].imag
            )
    elif len(u.shape) == 2:
        nx, nz_half_p1 = u.shape
        nz = (nz_half_p1 - 1) * 2
        kkx, _ = wavenumbers(lx, lz, nx, nz)
        image = np.zeros((nx, nz // 2 + 1), dtype=np.complex128)
        for ix, kx in enumerate(kkx):
            # Re part
            image[ix, :] = (
                np.cos(kx * dx) * u[ix, :].real + np.sin(kx * dx) * u[ix, :].imag
            )
            # Im part
            image[ix, :] += 1j * (
                -np.sin(kx * dx) * u[ix, :].real + np.cos(kx * dx) * u[ix, :].imag
            )
    else:
        exit("Tx got incompatible data.")

    return image


def Tz(dz, u, lx, lz):
    # Tz(dz) u(z) = u(z-dz)

    if len(u.shape) == 4:
        _, nx, ny, nz_half_p1 = u.shape
        nz = (nz_half_p1 - 1) * 2
        _, kkz = wavenumbers(lx, lz, nx, nz)
        image = np.zeros((3, nx, ny, nz // 2 + 1), dtype=np.complex128)
        for iz, kz in enumerate(kkz):
            # Re part
            image[:, :, :, iz] = (
                np.cos(kz * dz) * u[:, :, :, iz].real
                + np.sin(kz * dz) * u[:, :, :, iz].imag
            )
            # Im part
            image[:, :, :, iz] += 1j * (
                -np.sin(kz * dz) * u[:, :, :, iz].real
                + np.cos(kz * dz) * u[:, :, :, iz].imag
            )
    elif len(u.shape) == 3:
        _, nx, nz_half_p1 = u.shape
        nz = (nz_half_p1 - 1) * 2
        _, kkz = wavenumbers(lx, lz, nx, nz)
        image = np.zeros((3, nx, nz // 2 + 1), dtype=np.complex128)
        for iz, kz in enumerate(kkz):
            # Re part
            image[:, :, iz] = (
                np.cos(kz * dz) * u[:, :, iz].real + np.sin(kz * dz) * u[:, :, iz].imag
            )
            # Im part
            image[:, :, iz] += 1j * (
                -np.sin(kz * dz) * u[:, :, iz].real + np.cos(kz * dz) * u[:, :, iz].imag
            )
    elif len(u.shape) == 2:
        nx, nz_half_p1 = u.shape
        nz = (nz_half_p1 - 1) * 2
        _, kkz = wavenumbers(lx, lz, nx, nz)
        image = np.zeros((nx, nz // 2 + 1), dtype=np.complex128)
        for iz, kz in enumerate(kkz):
            # Re part
            image[:, iz] = (
                np.cos(kz * dz) * u[:, iz].real + np.sin(kz * dz) * u[:, iz].imag
            )
            # Im part
            image[:, iz] += 1j * (
                -np.sin(kz * dz) * u[:, iz].real + np.cos(kz * dz) * u[:, iz].imag
            )
    else:
        exit("Tz got incompatible data.")

    return image


def wavenumbers(lx, lz, nx, nz):
    kx = np.zeros((nx), dtype=np.float64)
    kz = np.zeros((nz // 2 + 1), dtype=np.float64)

    for i in range(nx):
        if i <= nx // 2:
            kx[i] = i * (2 * np.pi / lx)
        else:
            kx[i] = (i - nx) * (2 * np.pi / lx)

    for k in range(nz // 2 + 1):
        kz[k] = k * (2 * np.pi / lz)

    return kx, kz


def find_shifts(state, tgx, tx, tgz, tz, ygrid, ly=2):
    ptgx = inprod(state, tgx, ygrid, ly=ly)
    ptx = inprod(state, tx, ygrid, ly=ly)
    ptgz = inprod(state, tgz, ygrid, ly=ly)
    ptz = inprod(state, tz, ygrid, ly=ly)

    # returns relative shifts (in units of domain size)
    shiftx = -np.arctan2(ptgx, ptx) / (2 * np.pi)
    shiftz = -np.arctan2(ptgz, ptz) / (2 * np.pi)

    return np.array([shiftx, shiftz, ptgx, ptx, ptgz, ptz])


def apply_shifts(state, shifts, lx, lz):

    # takes relative shifts (in units of domain size)
    state_spec = phys_to_fourier(state)
    state_spec = Tx(lx*shifts[0], state_spec, lx, lz)
    state_spec = Tz(lz*shifts[1], state_spec, lx, lz)
    state_ = fourier_to_phys(state_spec)
    return state_


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
