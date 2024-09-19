import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from xbout import open_boutdataset
from pathlib import Path
from matplotlib import animation
from matplotlib.colors import LightSource


def read_boutdata(filepath: str, remove_xgc: bool = True) -> xr.Dataset:
    """Read bout dataset with xy geometry

    :param filepath: Filepath to bout data
    :param remove_xgc: Whether to remove the x guard cells from output, defaults to True
    :return: _description_
    """

    ds = open_boutdataset(
        chunks={"t": 4},
        datapath=Path(filepath) / "BOUT.dmp.*.nc",
    )
    # Use squeeze() to get rid of the y-dimension, which has length 1 as blob2d does not
    # simulate the parallel dimension.
    dx = ds["dx"].isel(y=0).values
    dy = ds["dy"].isel(x=0).values

    ds = ds.squeeze(drop=True)

    # Get rid of existing "x" coordinate, which is just the index values.
    x = "x"
    y = "y"
    ds = ds.drop("x")
    ds = ds.drop("y")
    # # # Create a new coordinate, which is length in units of rho_s
    ds = ds.assign_coords(x=np.arange(ds.sizes[x]) * dx)
    ds = ds.assign_coords(y=np.arange(ds.sizes["y"]) * dy)

    # ngc = int((len(ds["x"]) - len(ds["y"]))/2)
    if remove_xgc:
        ngc = 4
        ds = ds.isel(x=range(ngc, len(ds["x"]) - ngc))

    return ds


def contour_list(
    ds: xr.Dataset, vars: list[str] = ["P", "psi", "omega", "phi"], t: int = -1
):
    """Plot contour maps of several variables at a given timestamp

    :param ds: Bout dataset
    :param vars: List of variables, defaults to ["P", "psi", "omega", "phi"]
    :param t: Integer timestamp, defaults to -1
    """
    fig, ax = plt.subplots(ncols=len(vars))
    for i, v in enumerate(vars):
        if v == "P" or v == "psi":
            vmin = ds[v][0].values.min()
            vmax = ds[v][0].values.max()
        else:
            vmin = None
            vmax = None
        ax[i].contourf(
            ds["x"], ds["y"], ds[v][t].values.T, vmin=vmin, vmax=vmax, levels=20
        )
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("y")
        ax[i].set_title(v)
    fig.tight_layout()


def contour_overlay(ds: xr.Dataset, var: str = "P", timestamps: list[int] = [0, -1]):
    """Plot overlaid contours at several timestamps


    :param ds: Bout dataset
    :param var: Variable to plot, defaults to "P"
    :param timestamps: List of integer timestamps to overlay, defaults to [0, -1]
    """
    fig, ax = plt.subplots(1)
    vmin = ds[var][0].min()
    vmax = ds[var][0].max()
    levels = np.sort(list(np.linspace(vmin, vmax, 20)))
    if var == "psi":
        levels = np.sort(np.array(list(levels) + [0]))
    for t in timestamps:
        ax.contour(ds["x"], ds["x"], ds[var][t].values.T, linestyles="-", levels=levels)


def animate_contour_list(
    ds: xr.Dataset,
    vars: list[str] = ["P", "psi"],
    savepath: str | None = None,
    plot_every: int = 1,
):
    """Animate a list of variables using axes.contourf

    :param ds: Bout dataset
    :param vars: List of variables to plot, defaults to ["P", "psi"]
    :param savepath: _description_, defaults to None
    :return: Animation
    """
    # TODO: Add colorbar
    # Generate grid for plotting
    xmin = 0
    xmax = -1
    xvals = ds["x"][xmin:xmax]
    yvals = ds["y"][xmin:xmax]
    title = str(vars)

    fig, ax = plt.subplots(nrows=1, ncols=len(vars))
    if len(vars) == 1:
        ax = [ax]

    levels = {}
    var_arrays = {}
    for j, v in enumerate(vars):
        var_arrays[v] = ds[v].values[:, xmin:xmax, xmin:xmax]
        ax[j].set_xlim((0, xvals.values.max()))
        ax[j].set_ylim((0, yvals.values.max()))
        ax[j].set_aspect("equal")
        vmin = var_arrays[v][0].min()
        vmax = var_arrays[v][0].max()
        levels[v] = np.sort(list(np.linspace(vmin, vmax, 20)))
        # levels[v] = np.linspace(0.001,1.0,20)
        if v == "psi":
            levels[v] = np.sort(np.array(list(levels[v]) + [0]))

    def animate(i):
        for j, v in enumerate(vars):
            ax[j].clear()
            z = var_arrays[v][plot_every * i].T
            # cont = ax[j].contour(xvals, yvals, z, vmin=levels[v].min(), vmax=levels[v].max(), levels=levels[v])
            cont = ax[j].contourf(
                xvals,
                yvals,
                z,
                vmin=levels[v].min(),
                vmax=levels[v].max(),
                levels=levels[v],
                cmap="inferno",
            )
            ax[j].set_xlabel("x")
            ax[j].set_ylabel("y")
            ax[j].set_title(v + ", " + str(plot_every * i))

        return cont

    anim = animation.FuncAnimation(
        fig, animate, frames=int(var_arrays[vars[0]].shape[0] / plot_every)
    )
    if savepath is not None:
        anim.save("demo.gif", fps=24)

    return anim


def animate_vector(
    ds: xr.Dataset, vec_var: str, scalar: str = "P", savepath: str | None = None
):
    """Animate vector field

    :param ds: Bout dataset
    :param vec_var: Vector variable
    :param scalar: Scalar variable to plot underneath, defaults to "P"
    :param savepath: Filepath to save gif, defaults to None
    :return: Animation
    """

    # Generate grid for plotting
    X, Y = np.meshgrid(ds["x"], ds["y"])
    vec_x = ds[vec_var + "_x"]
    vec_y = ds[vec_var + "_y"]
    vec_mag = np.sqrt(vec_x**2 + vec_y**2)
    scalar = ds["P"]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, X.max()), ylim=(0, Y.max()))
    # ls = LightSource(azdeg=110, altdeg=10)

    # animation function
    plot_every = 1

    def animate(i):
        ax.clear()
        cont = ax.pcolormesh(X, Y, scalar.isel(t=i).values.T)
        # rgb = ls.shade(scalar.isel(t=i).values.T, vert_exag=50, cmap=plt.cm.magma, blend_mode="overlay")
        # cont = ax.imshow(rgb)
        lw = 2 * vec_mag.isel(t=i).values.T / vec_mag.isel(t=i).values.max()
        cont = ax.streamplot(
            X,
            Y,
            vec_x.isel(t=i).values.T,
            vec_y.isel(t=i).values.T,
            color="black",
            linewidth=lw,
            integration_direction="both",
            broken_streamlines=False,
            density=0.25,
        )
        # cont = ax.quiver(X, Y, vec_x.isel(t=i).values.T, vec_y.isel(t=i).values.T, color="black")
        if i == 0:
            ax.set_title(plot_every * i)
        else:
            ax.set_title(plot_every * i)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return cont

    anim = animation.FuncAnimation(fig, animate, frames=int(len(ds.t) / plot_every))
    if savepath is not None:
        anim.save(savepath, fps=24)

    return anim


def integrate_dxdy(ds: xr.Dataset, var: str):
    I = np.zeros(len(ds.t))
    for t in range(len(ds.t)):
        I[t] = np.sum(ds.dx * ds.dy * var[t])
    return I


def get_tot_pol_flux(ds):
    # [psi_0 a_mid^2]
    tot_pol_flux = integrate_dxdy(ds, ds["psi"])
    return tot_pol_flux


def get_tot_pressure(ds):
    # [P_0 a_mid^2]
    tot_pressure = integrate_dxdy(ds, ds["P"])
    return tot_pressure


def get_tot_energy(ds):
    # [P_0 a_mid^2]
    ds = ds
    beta_p = ds.metadata["beta_p"]
    epsilon = ds.metadata["epsilon"]
    R_0 = ds.metadata["R_0"]
    a_mid = ds.metadata["a_mid"]
    psi_0 = ds.metadata["psi_0"]
    P_0 = ds.metadata["P_0"]
    B_pmid = ds.metadata["B_pmid"]

    x_c = ds["x"] - 0.5 * np.sum(ds["dx"].isel(y=0))
    B_px = ds["B_x"]
    B_py = ds["B_y"]
    P = ds["P"]
    u_x = ds["u_x"]
    u_y = ds["u_y"]
    mu_0 = 1.256637e-6

    mag_energy = (P_0 / beta_p) * a_mid**2 * integrate_dxdy(ds, B_px**2 + B_py**2)
    kin_energy = 0.5 * a_mid**2 * P_0 * integrate_dxdy(ds, u_x**2 + u_y**2)
    pot_energy = -P_0 * epsilon * a_mid**2 * integrate_dxdy(ds, P * x_c)

    return (
        mag_energy / (P_0 * a_mid**2),
        kin_energy / (P_0 * a_mid**2),
        pot_energy / (P_0 * a_mid**2),
    )


def plot_conservation(ds: xr.Dataset):
    """Plot conserved quantities over time

    :param ds: Bout dataset
    """

    ds = ds.isel(t=range(0, 50))
    tot_pol_flux = get_tot_pol_flux(ds)
    tot_pressure = get_tot_pressure(ds)
    M, K, Pi = get_tot_energy(ds)

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(ds[r"t"], tot_pol_flux)
    ax[0].set_title(r"$\psi$")
    ax[0].set_ylabel(r"[$\psi_0 a_{mid}^2$]")

    ax[1].plot(ds[r"t"], tot_pressure)
    ax[1].set_title(r"$P$")
    ax[1].set_ylabel(r"[$P_0 a_{mid}^2$]")

    # ax[2].plot(ds_trimmed[r"t"], (M) + (K) + (Pi), "-", color=r"black")
    # ax[2].set_title(r"$E=M+K+\Pi$")
    # ax[2].set_ylabel(r"[$P_0 a_{mid}^2$]")

    relative = True
    if relative:
        ax[2].plot(ds[r"t"], M - M[0], "-", label=r"$M-M(t=0)$")
        ax[2].plot(ds[r"t"], K - K[0], "-", label=r"$K-K(t=0)$")
        ax[2].plot(ds[r"t"], Pi - Pi[0], "-", label=r"$\Pi-\Pi(t=0)$")
        ax[2].plot(
            ds[r"t"],
            (M - M[0]) + (K - K[0]) + (Pi - Pi[0]),
            "--",
            color=r"black",
            label=r"$E-E(t=0)$",
        )
        # ax[2].plot(ds[r"t"], (K-K[0]) + (Pi-Pi[0]), "--", color=r"black", label=r"$E-E(t=0)$")
    else:
        ax[2].plot(ds[r"t"], M, "-", label=r"$M-M(t=0)$")
        ax[2].plot(ds[r"t"], K, "-", label=r"$K-K(t=0)$")
        ax[2].plot(ds[r"t"], Pi, "-", label=r"$\Pi-\Pi(t=0)$")
        ax[2].plot(ds[r"t"], M + K + Pi, "--", color=r"black", label=r"$E-E(t=0)$")

    ax[2].set_ylabel(r"[$P_0 a_{mid}^2$]")
    ax[2].set_title(r"$E=M+K+\Pi$")

    [ax[i].legend(loc="lower left") for i in range(len(ax))]
    ax[-1].set_xlabel(r"$t\ [t_0]$")
    [a.grid() for a in ax]
    fig.tight_layout()
