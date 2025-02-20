from tempfile import tempdir
import time
from psutil import swap_memory
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from xbout import open_boutdataset
from pathlib import Path
from matplotlib import animation
from matplotlib.colors import LightSource
from matplotlib.widgets import Button, Slider
import os

mu_0 = 1.256637e-6
el_charge = 1.602e-19
m_e = 9.11e-31
m_i = 2 * 1.667e-27
mu_0 = 1.256637e-6
eps_0 = 8.854188e-12
boltzmann_k = 1.380649e-23


def read_boutdata(
    filepath: str, remove_xgc: bool = True, units: str = "a_mid"
) -> xr.Dataset:
    """Read bout dataset with xy geometry

    :param filepath: Filepath to bout data
    :param remove_xgc: Whether to remove the x guard cells from output, defaults to True
    :param units: Whether to convert axes units to "m", "cm" or keep in "a_mid"
    :return: xbout dataset
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
    # ds = ds.assign_coords(x=np.arange(ds.sizes[x]) * dx * ds.metadata["a_mid"] * 100)
    # ds = ds.assign_coords(y=np.arange(ds.sizes["y"]) * dy * ds.metadata["a_mid"] * 100)

    ds = ds.assign_coords(x=ds["x_c"].isel(y=0).values)
    ds = ds.assign_coords(y=ds["y_c"].isel(x=0).values)
    if units == "m":
        ds = ds.assign_coords(x=ds.x.values * ds.metadata["a_mid"])
        ds = ds.assign_coords(y=ds.y.values * ds.metadata["a_mid"])
    elif units == "cm":
        ds = ds.assign_coords(x=ds.x.values * ds.metadata["a_mid"] * 100)
        ds = ds.assign_coords(y=ds.y.values * ds.metadata["a_mid"] * 100)

    ds.metadata["grid_units"] = units

    # ngc = int((len(ds["x"]) - len(ds["y"]))/2)
    if remove_xgc:
        ngc = 2
        ds = ds.isel(x=range(ngc, len(ds["x"]) - ngc))

        # ds = ds.assign_coords(x=ds.x - ds.x[0])
        # ds = ds.assign_coords(y=ds.y - ds.y[0])

    # Calculate conductive, convective and total heat fluxes
    q_prefactor = ds.metadata["P_0"] * ds.metadata["C_s0"]
    ds["q_conv_x"] = (1.5 * ds["P"] + 0.5 * (ds["u_x"] ** 2 + ds["u_y"] ** 2)) * ds[
        "u_x"
    ]
    ds["q_conv_y"] = (1.5 * ds["P"] + 0.5 * (ds["u_x"] ** 2 + ds["u_y"] ** 2)) * ds[
        "u_y"
    ]
    if hasattr(ds, "q_perp_x") is False:
        ds["q_perp_x"] = 0.0
        ds["q_perp_y"] = 0.0

    ds["q_cond_x"] = ds["q_par_x"] + ds["q_perp_x"]
    ds["q_cond_y"] = ds["q_par_y"] + ds["q_perp_y"]
    ds["q_tot_x"] = ds["q_cond_x"] + ds["q_conv_x"]
    ds["q_tot_y"] = ds["q_cond_y"] + ds["q_conv_y"]
    ds["beta_p"] = (
        2
        * mu_0
        * ds.metadata["P_0"]
        * ds["P"]
        / (ds.metadata["B_pmid"] * (ds["B_x"] ** 2 + ds["B_y"] ** 2))
    )
    ds["rho"] = np.sqrt((ds["psi"] - 1.0) / (0.0 - 1.0))
    ds["u_mag"] = np.sqrt(ds["u_x"] ** 2 + ds["u_y"] ** 2)

    return ds


def contour_list(
    ds: xr.Dataset, vars: list[str] = ["P", "psi", "omega", "phi"], t: int = -1
):
    """Plot contour maps of several variables at a given timestep

    :param ds: Bout dataset
    :param vars: List of variables, defaults to ["P", "psi", "omega", "phi"]
    :param t: Integer timestep, defaults to -1
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

    return ax


def contour_overlay(
    ds: xr.Dataset,
    var: str = "P",
    timesteps: list[int] = [0, -1],
    colorbar: bool = False,
    fill: bool = False,
    levels: int | list[float] | np.ndarray = 50,
    plot_r_cz: bool = False,
    savepath: str | None = None,
    title: str | None = None,
    extend: str = "neither",
):
    """Plot overlaid contours at several timesteps


    :param ds: Bout dataset
    :param var: Variable to plot, defaults to "P"
    :param timesteps: List of integer timesteps to overlay, defaults to [0, -1]
    :param fill: Whether to plot using contourf or contour, defaults to False
    :param levels: Number of levels to use or list of specific levels to plot
    :param savepath: Where to save figure
    """

    if title is None:
        title = var
    # linestyles = ["-", "--", ".-"]
    timesteps = list(reversed(timesteps))
    if len(timesteps) > 1:
        alphas = np.linspace(0.25, 1.0, len(timesteps))
    else:
        alphas = [1.0]
    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")
    vmin = ds[var].isel(t=timesteps).min()
    vmax = ds[var].isel(t=timesteps).max()
    if isinstance(levels, int):
        plot_levels = np.sort(list(np.linspace(vmin, vmax, levels)))
        if var == "psi":
            plot_levels = np.sort(
                np.array(
                    list(
                        -np.linspace(0, (-vmin) ** (1 / 1.5), int(levels / 2)) ** (1.5)
                    )
                    + list(np.linspace(0, vmax ** (1 / 1.5), int(levels / 2)) ** (1.5))[
                        1:
                    ]
                )
            )
    else:
        plot_levels = levels
    for i, t in enumerate(timesteps):
        if fill:
            c = ax.contourf(
                ds["x"],
                ds["y"],
                ds[var][t].values.T,
                alpha=alphas[i],
                levels=plot_levels,
                cmap="inferno",
                extend=extend,
            )
        else:
            c = ax.contour(
                ds["x"],
                ds["y"],
                ds[var][t].values.T,
                alpha=alphas[i],
                levels=plot_levels,
                cmap="inferno",
                extend=extend,
            )
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_title(title)
    if colorbar:
        fig.colorbar(c)

    if plot_r_cz:
        # r_cz = predicted_r_cz(ds)
        r_cz = 10.0
        theta_cz = np.linspace(0, 2 * np.pi, 100)
        x_cz = np.median(ds.x) + r_cz * np.cos(theta_cz)
        y_cz = np.median(ds.y) + r_cz * np.sin(theta_cz)
        ax.plot(x_cz, y_cz, linestyle="--", color="red")

    if savepath is not None:
        fig.savefig(savepath)

    # ax.set_xlim((-10, 10))
    # ax.set_ylim((-15, 15))

    return ax


def animate_contour_list(
    ds: xr.Dataset,
    vars: list[str] = ["P", "psi"],
    savepath: str | None = None,
    plot_every: int = 1,
    fps: int | float = 24,
    plot_r_cz: bool = True,
    min_timestep: int | None = None,
    max_timestep: int | None = None,
    trim_cells: int | None = None,
    num_levels: int = 50,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Animate a list of variables using axes.contourf

    :param ds: Bout dataset
    :param vars: List of variables to plot, defaults to ["P", "psi"]
    :param savepath: Location to save gif, defaults to None
    :param plot_r_cz: Overlay the predicted convection zone, defaults to True
    :param min_timestep: Minimin timestep to plot
    :param max_timestep: Maximum timestep to plot
    :param trim_cells: Number of cells to trim from edges in plot
    :param num_levels: Number of levels in contour plots
    :return: Animation
    """

    if trim_cells is None:
        ds_plot = ds
    else:
        ds_plot = ds.isel(
            x=range(trim_cells, len(ds.x) - trim_cells),
            y=range(trim_cells, len(ds.y) - trim_cells),
        )
    # TODO: Add colorbar
    # Generate grid for plotting
    xmin = 0
    xmax = -1
    xvals = ds_plot["x"][xmin:xmax]
    yvals = ds_plot["y"][xmin:xmax]
    title = str(vars)

    fig, ax = plt.subplots(nrows=1, ncols=len(vars))
    if len(vars) == 1:
        ax = [ax]

    if max_timestep is None:
        max_timestep = len(ds_plot.t)
    if min_timestep is None:
        min_timestep = 0
    ds_plot = ds_plot.isel(t=range(min_timestep, max_timestep))

    # Get the predicted convection zone region
    r_cz = predicted_r_cz(ds_plot)
    theta_cz = np.linspace(0, 2 * np.pi, 100)
    x_cz = np.median(ds_plot.x) + r_cz * np.cos(theta_cz)
    y_cz = np.median(ds_plot.y) + r_cz * np.sin(theta_cz)
    print("r_cz = {:.2f}cm".format(r_cz * ds_plot.metadata["a_mid"] * 100))

    levels = {}
    var_arrays = {}
    for j, v in enumerate(vars):
        var_arrays[v] = ds_plot[v].values[:, xmin:xmax, xmin:xmax]
        ax[j].set_xlim((0, xvals.values.max()))
        ax[j].set_ylim((0, yvals.values.max()))
        ax[j].set_aspect("equal")
        if vmin is None:
            vmin = var_arrays[v][0].min() - 0.05 * var_arrays[v][0].max()
        if vmax is None:
            vmax = var_arrays[v][0].max() + 0.05 * var_arrays[v][0].max()
        # vmin = var_arrays[v][0].min()
        # vmax = var_arrays[v][0].max()
        if v == "psi":
            levels[v] = np.sort(
                np.array(
                    list(-np.linspace(0, -vmin, int(num_levels / 2)))
                    + list(np.linspace(0, vmax, int(num_levels / 2)))[1:]
                )
            )
        else:
            levels[v] = np.sort(list(np.linspace(vmin, vmax, 2 * num_levels)))

    def animate(i):
        for j, v in enumerate(vars):
            ax[j].clear()
            z = var_arrays[v][plot_every * i].T
            # cont = ax[j].contour(xvals, yvals, z, vmin=levels[v].min(), vmax=levels[v].max(), levels=levels[v])
            if v == "psi":
                cont = ax[j].contour(
                    xvals,
                    yvals,
                    z,
                    vmin=levels[v].min(),
                    vmax=levels[v].max(),
                    levels=levels[v],
                    cmap="inferno",
                    extend="both",
                )
            else:
                cont = ax[j].contourf(
                    xvals,
                    yvals,
                    z,
                    vmin=levels[v].min(),
                    vmax=levels[v].max(),
                    levels=levels[v],
                    cmap="inferno",
                    extend="both",
                )
            if plot_r_cz:
                ax[j].plot(x_cz, y_cz, linestyle="--", color="red")
            ax[j].set_xlabel("x")
            ax[j].set_ylabel("y")
            timestep = (
                ds_plot.t.values[plot_every * i] - ds.t.values[0]
            ) * ds.metadata["t_0"]
            ax[j].set_title(v + ", t={:.2f}ms".format(timestep * 1000))

        return cont

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=int(var_arrays[vars[0]].shape[0] / plot_every),
    )
    if savepath is not None:
        anim.save(savepath, fps=fps)

    return anim


def animate_vector(
    ds: xr.Dataset,
    vec_var: str,
    scalar: str = "P",
    savepath: str | None = None,
    lw_prefactor: float | None = None,
    density: float = 0.25,
    plot_every: int = 1,
    fps: int = 10,
    **mpl_kwargs,
):
    """Animate vector field

    :param ds: Bout dataset
    :param vec_var: Vector variable
    :param scalar: Scalar variable to plot underneath, defaults to "P"
    :param savepath: Filepath to save gif, defaults to None
    :param mpl_kwargs: Keyword arguments to matplotlib streamlines function
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

    xmin = np.min(ds.x)
    xmax = np.max(ds.x)
    ymin = np.min(ds.y)
    ymax = np.max(ds.y)

    # animation function
    def animate(i):
        ax.clear()
        cont = ax.pcolormesh(X, Y, scalar.isel(t=i).values.T, cmap="inferno")
        # rgb = ls.shade(scalar.isel(t=i).values.T, vert_exag=50, cmap=plt.cm.magma, blend_mode="overlay")
        # cont = ax.imshow(rgb)
        if lw_prefactor is None:
            lw = vec_mag.isel(t=i).values.T / vec_mag.isel(t=0).values.max()
        else:
            lw = lw_prefactor * vec_mag.isel(t=i).values.T
        cont = ax.streamplot(
            X,
            Y,
            vec_x.isel(t=i).values.T,
            vec_y.isel(t=i).values.T,
            color="red",
            linewidth=lw,
            integration_direction="both",
            density=density,
            **mpl_kwargs,
        )
        # cont = ax.quiver(X, Y, vec_x.isel(t=i).values.T, vec_y.isel(t=i).values.T, color="black")
        if i == 0:
            ax.set_title(plot_every * i)
        else:
            ax.set_title(plot_every * i)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        return cont

    anim = animation.FuncAnimation(fig, animate, frames=int(len(ds.t) / plot_every))
    if savepath is not None:
        anim.save(savepath, fps=fps)

    return anim


def plot_vector(
    ds: xr.Dataset,
    vec_var: str,
    scalar: str = "P",
    t: int = 1,
    lw_prefactor: float | None = None,
    savepath: str | None = None,
    use_seed_points: bool = False,
    **kwargs,
):
    """Plot vector field at a single timestep

    :param ds: Bout dataset
    :param vec_var: Vector variable
    :param scalar: Scalar variable to plot underneath, defaults to "P"
    :param t: Timestamp
    :param savepath: Where to save figure
    :param use_seed_points: Launch streamlines from top of domain (should be used with broken_streamlines=False, and will generally require some fine tuning)
    :return: Animation
    """

    # Generate grid for plotting
    X, Y = np.meshgrid(ds["x"], ds["y"])
    vec_x = ds[vec_var + "_x"]
    vec_y = ds[vec_var + "_y"]
    vec_mag = np.sqrt(vec_x**2 + vec_y**2)
    scalar = ds["P"]

    # Generate seed points
    num_seed_points = 500
    if use_seed_points:
        seed_points = np.array(
            [
                np.linspace(ds.x.min() + 1e-6, ds.x.max() - 1e-6, num_seed_points),
                [ds.y.values.max() - 1e-6 for i in range(num_seed_points)],
            ]
        ).T

    fig = plt.figure()
    ax = plt.axes()
    # ls = LightSource(azdeg=110, altdeg=10)

    cont = ax.contourf(X, Y, scalar.isel(t=t).values.T, cmap="inferno", levels=500)
    # rgb = ls.shade(scalar.isel(t=t).values.T, vert_exag=50, cmap=plt.cm.magma, blend_mode="overlay")
    # cont = ax.imshow(rgb)
    if lw_prefactor is None:
        # lw = 10 * vec_mag.isel(t=t).values.T / vec_mag.isel(t=t).values.max()
        lw = 3 * np.sqrt(vec_mag.isel(t=t).values.T / vec_mag.isel(t=t).values.max())
    else:
        lw = lw_prefactor * vec_mag.isel(t=t).values.T
        # lw = lw_prefactor * np.sqrt(vec_mag.isel(t=t).values.T)
    if use_seed_points:
        cont = ax.streamplot(
            X,
            Y,
            vec_x.isel(t=t).values.T,
            vec_y.isel(t=t).values.T,
            color="red",
            linewidth=lw,
            integration_direction="both",
            start_points=seed_points,
            **kwargs,
        )
    else:
        cont = ax.streamplot(
            X,
            Y,
            vec_x.isel(t=t).values.T,
            vec_y.isel(t=t).values.T,
            color="red",
            linewidth=lw,
            integration_direction="both",
            **kwargs,
        )

    xmin = np.min(ds.x)
    xmax = np.max(ds.x)
    ymin = np.min(ds.y)
    ymax = np.max(ds.y)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(vec_var)

    if savepath is not None:
        fig.savefig(savepath)


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

    # x_c = ds["x"] - 0.5 * np.sum(ds["dx"].isel(y=0))
    x_c = ds["x_c"]
    B_px = ds["B_x"]
    B_py = ds["B_y"]
    P = ds["P"]
    u_x = ds["u_x"]
    u_y = ds["u_y"]

    mag_energy = (P_0 / beta_p) * a_mid**2 * integrate_dxdy(ds, B_px**2 + B_py**2)
    kin_energy = 0.5 * a_mid**2 * P_0 * integrate_dxdy(ds, u_x**2 + u_y**2)
    pot_energy = -2 * P_0 * epsilon * a_mid**2 * integrate_dxdy(ds, P * x_c)

    return (
        mag_energy / (P_0 * a_mid**2),
        kin_energy / (P_0 * a_mid**2),
        pot_energy / (P_0 * a_mid**2),
    )


def plot_conservation(ds: xr.Dataset, relative: bool = True):
    """Plot conserved quantities over time

    :param ds: Bout dataset
    """

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

    if relative:
        ax[2].plot(ds[r"t"], M - M[0], "-", label=r"$M-M_0)$")
        ax[2].plot(ds[r"t"], K - K[0], "-", label=r"$K-K_0$")
        ax[2].plot(ds[r"t"], Pi - Pi[0], "-", label=r"$\Pi-\Pi_0$")
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

    return ax


def predicted_r_cz(ds):
    d_over_a = 0.71 * (
        ds.metadata["beta_p"] * ds.metadata["a_mid"] / ds.metadata["R_0"]
    ) ** (1 / 3)
    return d_over_a


def predicted_tau(ds):
    tau = (
        1.3
        * np.sqrt(predicted_r_cz(ds) * ds.metadata["a_mid"] * ds.metadata["R_0"])
        / np.sqrt(2 * 1.602e-19 * ds.metadata["T_sepx"] / ds.metadata["m_i"])
    )
    return tau


def l2_err_t0(ds: xr.Dataset, var: str = "P"):
    err = ds[var] - ds[var].isel(t=0)


def animate_q_targets(
    ds: xr.Dataset,
    plot_every: int = 1,
    normalise: bool = False,
    savepath: str | None = None,
):
    """Animate q_tot to each diverotr leg (assuming snowflake config)

    :param ds: Xarray dataset from BOUT++
    :param plot_every: Plot every x timesteps, defaults to 1
    :param normalise: Whether to normalise to Q_tot at t=0, defaults to False
    :param savepath: where to save animation, defaults to None
    :return: Animation
    """
    nx = len(ds.x)
    ny = len(ds.y)
    x0 = range(len(ds.x))
    y1 = range(ny)
    x2 = range(int(nx / 2), nx)
    x3 = range(int(nx / 2))
    y4 = range(ny)
    # y1 = range(int(ny / 4), int(3 * ny / 4))
    # x2 = range(int(nx / 2), nx)
    # x3 = range(int(nx / 2))
    # y4 = range(int(ny / 4), int(3 * ny / 4))
    qin, q1, q2, q3, q4 = get_q_legs(ds)
    Qin, Q1, Q2, Q3, Q4 = get_Q_legs(ds)

    if normalise:
        q1 = (q1 / Q1).values
        q2 = (q2 / Q2).values
        q3 = (q3 / Q3).values
        q4 = (q4 / Q4).values
    else:
        q1 = q1.values
        q2 = q2.values
        q3 = q3.values
        q4 = q4.values

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False)

    max_timestep = None
    min_timestep = None
    if max_timestep is None:
        max_timestep = len(ds.t)
    if min_timestep is None:
        min_timestep = 0
    ds = ds.isel(t=range(min_timestep, max_timestep))
    fig.subplots_adjust(hspace=0.0)
    # fig.tight_layout(h_pad=0)

    max_q1 = q1.max()
    max_q2 = q2.max()
    max_q3 = q3.max()
    max_q4 = q4.max()
    min_q1 = q1.min()
    min_q2 = q2.min()
    min_q3 = q3.min()
    min_q4 = q4.min()

    def animate(i):

        ax[0].clear()
        l = ax[0].plot(
            range(len(y1)),
            q1[plot_every * i, :],
            linestyle="-",
            color="black",
            label="Leg 1 (E)",
        )
        l = ax[0].plot(
            range(len(y1)),
            q1[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )

        ax[1].clear()
        l = ax[1].plot(
            range(len(x2)),
            q2[plot_every * i, :],
            linestyle="-",
            color="black",
            label="Leg 2 (SE)",
        )
        l = ax[1].plot(
            range(len(x2)),
            q2[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )

        ax[2].clear()
        l = ax[2].plot(
            range(len(x3)),
            q3[plot_every * i, :],
            linestyle="-",
            color="black",
            label="Leg 3 (SW)",
        )
        l = ax[2].plot(
            range(len(x3)),
            q3[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )

        ax[3].clear()
        l = ax[3].plot(
            ds.y.isel(y=range(len(y4))),
            q4[plot_every * i, :],
            linestyle="-",
            color="black",
            label="Leg 4 (W)",
        )
        l = ax[3].plot(
            ds.y.isel(y=range(len(y4))),
            q4[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )

        timestep = ds.t.values[plot_every * i] - ds.t.values[0]
        ax[0].set_title("t={:.2f}$t_0$".format(timestep))
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[2].legend(loc="upper right")
        ax[3].legend(loc="upper right")

        if normalise:
            ax[1].set_ylabel(r"$q_{\parallel} / \int q_{\parallel}ds$ [MWm$^{-2}$]")
        else:
            ax[1].set_ylabel(r"$q_{\parallel}$ [MWm$^{-2}$]")
        ax[-1].set_xlabel("x")
        ax[0].set_ylim(min_q1, max_q1)
        ax[1].set_ylim(min_q2, max_q2)
        ax[2].set_ylim(min_q3, max_q3)
        ax[3].set_ylim(min_q4, max_q4)

        return l

    anim = animation.FuncAnimation(fig, animate, frames=int(len(ds.t) / plot_every))
    if savepath is not None:
        anim.save(savepath, fps=24)

    return anim


def plot_q_targets(
    ds: xr.Dataset,
    show_t0: bool = True,
    normalise: bool = False,
    timestep: int = -1,
):
    """Plot q_tot to each diverotr leg (assuming snowflake config)

    :param ds: Xarray dataset from BOUT++
    :param normalise: Whether to normalise to Q_tot at t=0, defaults to False
    :return: Animation
    """
    nx = len(ds.x)
    ny = len(ds.y)
    x0 = range(len(ds.x))
    y1 = range(ny)
    x2 = range(int(nx / 2), nx)
    x3 = range(int(nx / 2))
    y4 = range(ny)
    # y1 = range(int(ny / 4), int(3 * ny / 4))
    # x2 = range(int(nx / 2), nx)
    # x3 = range(int(nx / 2))
    # y4 = range(int(ny / 4), int(3 * ny / 4))
    qin, q1, q2, q3, q4 = get_q_legs(ds)
    Qin, Q1, Q2, Q3, Q4 = get_Q_legs(ds)

    if normalise:
        q1 = (q1 / Q1).values
        q2 = (q2 / Q2).values
        q3 = (q3 / Q3).values
        q4 = (q4 / Q4).values
    else:
        q1 = q1.values
        q2 = q2.values
        q3 = q3.values
        q4 = q4.values

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=False)

    fig.subplots_adjust(hspace=0.0)

    ax[0].clear()
    l = ax[0].plot(
        range(len(y1)),
        q1[timestep, :],
        linestyle="-",
        color="black",
        label="Leg 1 (E)",
    )
    ax[1].clear()
    l = ax[1].plot(
        range(len(x2)),
        q2[timestep, :],
        linestyle="-",
        color="black",
        label="Leg 2 (SE)",
    )

    ax[2].clear()
    l = ax[2].plot(
        range(len(x3)),
        q3[timestep, :],
        linestyle="-",
        color="black",
        label="Leg 3 (SW)",
    )

    ax[3].clear()
    l = ax[3].plot(
        ds.y.isel(y=range(len(y4))),
        q4[timestep, :],
        linestyle="-",
        color="black",
        label="Leg 4 (W)",
    )

    if show_t0:
        l = ax[0].plot(
            range(len(y1)),
            q1[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )
        l = ax[1].plot(
            range(len(x2)),
            q2[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )
        l = ax[2].plot(
            range(len(x3)),
            q3[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )
        l = ax[3].plot(
            ds.y.isel(y=range(len(y4))),
            q4[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )

    timestep = ds.t.values[timestep] - ds.t.values[0]
    ax[0].set_title("t={:.2f}$t_0$".format(timestep))
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    ax[3].legend(loc="upper right")

    if normalise:
        ax[1].set_ylabel(r"$q_{\parallel} / \int q_{\parallel}ds$ [MWm$^{-2}$]")
    else:
        ax[1].set_ylabel(r"$q_{\parallel}$ [MWm$^{-2}$]")
    ax[-1].set_xlabel("x")


def Q_target_proportions(
    ds: xr.Dataset,
    normalise: bool = True,
    savepath: str | None = None,
    ylim: list | tuple | None = None,
):
    """Plot total heat flow to each divertor leg (assuming snowflake config)

    :param ds: Xarray dataset
    :param normalise: whether to normalise heat flow to Q_tot, defaults to True
    :param savepath: where to save figure
    """
    qin, q1, q2, q3, q4 = get_q_legs(ds)
    if ds.metadata["grid_units"] == "cm":
        Q_prefactor = 1 / 100.0
    elif ds.metadata["grid_units"] == "m":
        Q_prefactor = 1.0
    elif ds.metadata["grid_units"] == "a_mid":
        Q_prefactor = ds.metadata["a_mid"]
    qin = qin.integrate(coord="x") * Q_prefactor
    q1 = q1.integrate(coord="y") * Q_prefactor
    q2 = q2.integrate(coord="x") * Q_prefactor
    q3 = q3.integrate(coord="x") * Q_prefactor
    q4 = q4.integrate(coord="y") * Q_prefactor

    q_tot = q1 + q2 + q3 + q4

    fig, ax = plt.subplots(1)
    if normalise:
        ax.stackplot(
            (ds.t - ds.t[0]) * ds.metadata["t_0"] * 1000,
            [q1 / q_tot, q2 / q_tot, q3 / q_tot, q4 / q_tot],
            labels=["Leg 1 (E)", "Leg 2 (SE)", "Leg 3 (SW)", "Leg 4 (W)"],
        )
        ax.set_ylabel(r"$P_{l} / P_{tot}$")

    else:
        ax.stackplot(
            (ds.t - ds.t[0]) * ds.metadata["t_0"] * 1000,
            [q1, q2, q3, q4],
            labels=["Leg 1 (E)", "Leg 2 (SE)", "Leg 3 (SW)", "Leg 4 (W)"],
        )
        # ax.plot(ds.t - ds.t[0], -qin)
        ax.set_ylabel(r"$P_{l}$ [MWm$^{-1}$]")

    ax.legend(loc="upper left")
    ax.set_xlabel("$t$ [ms]")
    ax.grid()
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()

    print(
        "At first timestep, fractions are:\n Leg 1 = {:.2f}% | Leg 2 = {:.2f}% | Leg 3 = {:.2f}% | Leg 4 = {:.2f}%".format(
            100 * q1[0] / q_tot[0],
            100 * q2[0] / q_tot[0],
            100 * q3[0] / q_tot[0],
            100 * q4[0] / q_tot[0],
        )
    )
    print(
        "At last timestep, fractions are:\n Leg 1 = {:.2f}% | Leg 2 = {:.2f}% | Leg 3 = {:.2f}% | Leg 4 = {:.2f}%".format(
            100 * q1[-1] / q_tot[-1],
            100 * q2[-1] / q_tot[-1],
            100 * q3[-1] / q_tot[-1],
            100 * q4[-1] / q_tot[-1],
        )
    )

    if savepath is not None:
        fig.savefig(savepath)


def get_q_legs(ds: xr.Dataset) -> tuple[xr.DataArray]:
    """Get the heat flux into each divertor leg, assuming snowflake configuration. Note that numbering here is leg 1 = east-most leg, counting up anticlockwise

    :param ds: Dataset output from BOUT++ simulation
    :return: qin, q1, q2, q3, q4
    """
    q_prefactor = 1e-6 * (ds.metadata["P_0"] * ds.metadata["C_s0"])
    nx = len(ds.x)
    ny = len(ds.y)
    x0 = range(len(ds.x))
    y1 = range(ny)
    x2 = range(int(nx / 2), nx)
    x3 = range(int(nx / 2))
    y4 = range(ny)

    # qin = -q_prefactor * (ds["q_tot_y"] ).isel(y=-1, x=x0)
    # q1 = q_prefactor * (ds["q_tot_x"] ).isel(x=-1, y=y1)
    # q2 = -q_prefactor * (ds["q_tot_y"] ).isel(y=0, x=x2)
    # q3 = -q_prefactor * (ds["q_tot_y"] ).isel(y=0, x=x3)
    # q4 = -q_prefactor * (ds["q_tot_x"] ).isel(x=0, y=y4)

    qin = -q_prefactor * (ds["q_cond_y"]).isel(y=-1, x=x0)
    q1 = q_prefactor * (ds["q_cond_x"]).isel(x=-1, y=y1)
    q2 = -q_prefactor * (ds["q_cond_y"]).isel(y=0, x=x2)
    q3 = -q_prefactor * (ds["q_cond_y"]).isel(y=0, x=x3)
    q4 = -q_prefactor * (ds["q_cond_x"]).isel(x=0, y=y4)

    return qin, q1, q2, q3, q4


def get_Q_legs(ds: xr.Dataset):
    """Get line-integrated heat flux going into each divertor leg, assuming snowflake configuration

    :param ds: xarray dataset
    :return: Qin, Q1 Q2, Q3, Q4
    """
    qin, q1, q2, q3, q4 = get_q_legs(ds)
    if ds.metadata["grid_units"] == "cm":
        Q_prefactor = 1 / 100.0
    elif ds.metadata["grid_units"] == "m":
        Q_prefactor = 1.0
    elif ds.metadata["grid_units"] == "a_mid":
        Q_prefactor = ds.metadata["a_mid"]
    Qin = qin.integrate(coord="x") * Q_prefactor
    Q1 = q1.integrate(coord="y") * Q_prefactor
    Q2 = q2.integrate(coord="x") * Q_prefactor
    Q3 = q3.integrate(coord="x") * Q_prefactor
    Q4 = q4.integrate(coord="y") * Q_prefactor

    return Qin, Q1, Q2, Q3, Q4


def find_nearest(array, value):
    """Find index of nearest value in array

    :param array: input array
    :param value: value
    :return: integer index
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_null_coords(
    ds: xr.Dataset,
    num: int = 2,
    timestep: int = 0,
) -> list:
    """Use magnetic pressure minimums to identify location of null points

    :param ds: xarray dataset output from BOUT++
    :param num: number of nulls (1 or 2)
    :param timestep: which timestep in which to find nulls
    :return: x1, x2, y1, y2, psi1, psi2
    """
    if not hasattr(ds, "P_b"):
        ds["P_b"] = (
            10
            * ((ds["B_x"] ** 2 + ds["B_y"] ** 2) * ds.metadata["B_pmid"] ** 2)
            / (2 * mu_0)
        )
    dr = ds["P_b"].isel(t=timestep)
    firstmin = dr.argmin(dim=["x", "y"], keep_attrs=True)
    i1 = firstmin["x"].values
    j1 = firstmin["y"].values
    x1 = dr.x[i1]
    y1 = dr.y[j1]
    dr[i1, j1] = dr.values.max()
    if num == 2:
        for k in range(100):
            secondmin = dr.argmin(dim=["x", "y"], keep_attrs=True)
            i2 = secondmin["x"].values
            j2 = secondmin["y"].values

            if (abs(i2 - i1) > 1) or (abs(j2 - j1) > 1):
                x2 = dr.x[i2]
                y2 = dr.y[j2]
                break
            else:
                dr[i2, j2] = dr.values.max()
    else:
        x2 = x1
        y2 = y1

    psi1 = ds["psi"].isel(x=i1, y=j1, t=timestep).values
    psi2 = ds["psi"].isel(x=i2, y=j2, t=timestep).values

    # Ensure upper null is always primary
    if y2 > y1:
        tmp_x2 = x1
        tmp_y2 = y1
        tmp_psi2 = psi1

        x1 = x2
        y1 = y2
        psi1 = psi2

        x2 = tmp_x2
        y2 = tmp_y2
        psi2 = tmp_psi2

    return x1, x2, y1, y2, psi1, psi2


def find_primary_sepx(
    ds: xr.Dataset,
    max_min_num: int = 5,
    timestep: int = 0,
    target_psi: float = 0.0,
    sep_thresh: float = 1.0,
) -> list:
    """Use magnetic pressure minimums to identify location of null points

    :param ds: xarray dataset output from BOUT++
    :param num: number of nulls (1 or 2)
    :param timestep: which timestep in which to find nulls
    :return: x1, x2, y1, y2, psi1, psi2
    """
    if not hasattr(ds, "B_p"):
        ds["B_p"] = np.sqrt(ds["B_x"] ** 2 + ds["B_y"] ** 2)
    dr = ds["B_p"].isel(t=timestep)

    # Find the first minimum
    firstmin = dr.argmin(dim=["x", "y"], keep_attrs=True)
    i1 = firstmin["x"].values
    j1 = firstmin["y"].values
    x1 = dr.x[i1]
    y1 = dr.y[j1]
    maxval = dr.values.max()
    dr[i1, j1] = maxval
    i_mins = [i1]
    j_mins = [j1]

    # Find some extra minima spatially separated from all previous minima
    for min_num in range(1, max_min_num):
        next_minimum_found = False
        max_search_steps = 100
        cur_search_step = 0
        while next_minimum_found is False:
            nextmin = dr.argmin(dim=["x", "y"], keep_attrs=True)
            inext = nextmin["x"].values
            jnext = nextmin["y"].values
            dr[inext, jnext] = maxval

            # Check whether this minimum is spatially separated from others
            distinct_from_others = []
            for q in range(min_num):
                # dr[inext, jnext] = maxval
                dist = np.sqrt((inext - i_mins[q]) ** 2 + (jnext - j_mins[q]) ** 2)
                if dist > sep_thresh:
                    distinct_from_others.append(True)
                else:
                    distinct_from_others.append(False)
            if all(distinct_from_others):
                i_mins.append(inext)
                j_mins.append(jnext)
                next_minimum_found = True
            else:
                cur_search_step += 1
            if cur_search_step > max_search_steps:
                raise Exception(
                    "Max search steps exceeded; could not find a spatially separated minima from all others."
                )

    # Store min coordinates
    x_mins = [dr.x[i] for i in i_mins]
    y_mins = [dr.y[j] for j in j_mins]

    # Find psis
    psi_mins = np.array(
        [
            ds["psi"].isel(x=i_mins[i], y=j_mins[i], t=timestep).values
            for i in range(max_min_num)
        ]
    )

    # Find closest min to
    # minloc = np.argmin(abs(psi_mins-target_psi))
    # psi_0 = psi_mins[minloc]
    # x_0 = x_mins[minloc]
    # y_0 = y_mins[minloc]
    # return x_0, y_0, psi_0

    return x_mins, y_mins, sorted(psi_mins)


def find_primary_sepx_2(
    ds: xr.Dataset,
    timestep: int = 0,
    target_psi: float = 0.0,
    num_minima: int = 1000,
    w1: float = 0.5,
    w2: float = 0.5,
) -> list:
    if not hasattr(ds, "B_p"):
        ds["B_p"] = np.sqrt(ds["B_x"] ** 2 + ds["B_y"] ** 2)
    db = ds["B_p"].isel(t=timestep)
    dp = ds["psi"].isel(t=timestep)

    nx = len(db.x)
    ny = len(db.y)

    # Find location of minima in B_p
    vals = db.values.flatten()
    minlocs = np.argsort(vals)

    # Remove adjacent locs
    # How to do efficiently?
    # i = [int(minloc/ny) for minloc in minlocs]
    # j = [minloc%ny for minloc in minlocs]
    # for k in range(1,len(minlocs)):
    #     j = minlocs[k]%ny
    #     i = int(minlocs[k]/ny)
    #     j_prev = minlocs[k-1]%ny
    #     i_prev = int(minlocs[k-1]/ny)
    #     if (abs(j - j_prev) == 1) or (abs(i - i_prev) == 1):
    #         minlocs[k] = 0
    # minlocs = [minloc for minloc in minlocs if minloc != 0]

    # Find psi at B_p minima
    psi_vals = ds["psi"].isel(t=timestep).values.flatten()
    psi_at_minlocs = psi_vals[minlocs]

    # Find psi closest to target value
    psi_target_loc = abs(psi_at_minlocs - target_psi).argmin()
    psi_minloc = minlocs[psi_target_loc]

    # Get psi at primary sepx and x,y coords of primary null
    j_x = psi_minloc % ny
    i_x = int(psi_minloc / ny)
    x_0 = ds.x[i_x]
    y_0 = ds.y[j_x]
    psi_sepx = psi_vals[psi_minloc]

    # db = ds["B_p"].isel(t=timestep)
    # dp = abs(ds["psi"].isel(t=timestep))
    # score1 = ((db - db.min()) / (db.max() - db.min()))
    # score2 = ((dp - dp.min()) / (dp.max() - dp.min()))
    # score = (w1*score1 + w2*score2) / (w1+w2)
    # minloc = score.argmin(dim=["x", "y"], keep_attrs=True)
    # i_0 = minloc["x"].values
    # j_0 = minloc["y"].values
    # x_0 = ds.x[i_0]
    # y_0 = ds.y[j_0]
    # psi_sepx = ds["psi"].isel(t=timestep,x=i_0,y=j_0)

    return x_0, y_0, psi_sepx


def plot_nulls(
    ds: xr.Dataset,
    num: int = 2,
    timestep: int = 0,
    cmap: str = "inferno",
) -> list:
    """Use magnetic pressure minimums to identify location of null points and plot

    :param ds: xarray dataset output from BOUT++
    :param num: number of nulls (1 or 2)
    :param timestep: which timestep in which to find nulls
    :param cmap: colourmap to use for underlying pressure profile
    :param print_output: whether to print locations of nulls ofund for pasting into BOUT.inp file
    :return: [x1, x2, y1, y2]
    """
    ds["P_b"] = (
        10
        * ((ds["B_x"] ** 2 + ds["B_y"] ** 2) * ds.metadata["B_pmid"] ** 2)
        / (2 * mu_0)
    )
    x1, x2, y1, y2, psi1, psi2 = find_null_coords(ds, num=num, timestep=timestep)

    print(r"x_1 = {:.4f} # x-coordinate of first X-point  [a_mid]".format(x1))
    print(r"y_1 = {:.4f} # y-coordinate of first X-point  [a_mid]".format(y1))
    print(r"x_2 = {:.4f} # x-coordinate of second X-point  [a_mid]".format(x2))
    print(r"y_2 = {:.4f} # y-coordinate of second X-point  [a_mid]".format(y2))

    fig, ax = plt.subplots(1)

    ax.pcolormesh(ds.x, ds.y, ds["P"].isel(t=timestep).values.T, cmap=cmap)
    ax.contour(
        ds.x,
        ds.y,
        ds["psi"].isel(t=timestep).values.T,
        levels=np.sort([psi1, psi2]),
        colors="white",
        linestyles=["--", "-"],
    )
    ax.scatter([x1, x2], [y1, y2], color="red", marker="x", zorder=999)

    print("")
    print(r"psi_1 = {:.5f} ".format(psi1))
    print(r"psi_2 = {:.5f} ".format(psi2))

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    return


def animate_nulls(
    ds,
    plot_every: int = 1,
    fps: int = 10,
    savepath: str | None = None,
    refind_nulls_each_timestep: bool = True,
):
    """Animate hte position of both nulls over time

    :param ds: xarray dataset from BOUT++
    :param plot_every: plot every n timesteps, defaults to 1
    :param fps: fps of saved animation, defaults to 10
    :param savepath: savepath of output gif or video, defaults to None
    :return: animation
    """
    ds["P_b"] = (
        10
        * ((ds["B_x"] ** 2 + ds["B_y"] ** 2) * ds.metadata["B_pmid"] ** 2)
        / (2 * mu_0)
    )
    ds_plot = ds

    # TODO: Add colorbar
    # Generate grid for plotting
    xmin = 0
    xmax = -1
    xvals = ds_plot["x"][xmin:xmax]
    yvals = ds_plot["y"][xmin:xmax]
    vars = ["P"]
    fig, ax = plt.subplots(nrows=1, ncols=len(vars))
    if len(vars) == 1:
        ax = [ax]

    max_timestep = len(ds_plot.t)
    min_timestep = 0
    ds_plot = ds_plot.isel(t=range(min_timestep, max_timestep))

    levels = {}
    var_arrays = {}
    num_levels = 200
    for j, v in enumerate(vars):
        var_arrays[v] = ds_plot[v].values[:, xmin:xmax, xmin:xmax]
        ax[j].set_xlim((0, xvals.values.max()))
        ax[j].set_ylim((0, yvals.values.max()))
        ax[j].set_aspect("equal")
        vmin = var_arrays[v][0].min() - 0.05 * var_arrays[v][0].max()
        vmax = var_arrays[v][0].max() + 0.05 * var_arrays[v][0].max()
        levels[v] = np.sort(list(np.linspace(vmin, vmax, 2 * num_levels)))

    if not refind_nulls_each_timestep:
        x1, x2, y1, y2, psi1, psi2 = find_null_coords(ds, timestep=0)
    else:
        psi1 = None
        psi2 = None

    def animate(i, refind_nulls_each_timestep, psi1=None, psi2=None):
        if refind_nulls_each_timestep:
            x1, x2, y1, y2, psi1, psi2 = find_null_coords(ds, timestep=i * plot_every)

        for j, v in enumerate(vars):
            ax[j].clear()
            z = var_arrays[v][plot_every * i].T
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
            timestep = ds_plot.t.values[plot_every * i] - ds.t.values[0]
            ax[j].set_title(v + ", t={:.2f}$t_0$".format(timestep))

            ax[j].contour(
                ds_plot.x,
                ds_plot.y,
                ds_plot["psi"].isel(t=i * plot_every).values.T,
                levels=np.sort([psi1, psi2]),
                colors="white",
                linestyles=["--", "-"],
                zorder=998,
            )
            if refind_nulls_each_timestep:
                ax[j].scatter([x1, x2], [y1, y2], color="red", marker="x", zorder=999)

        return cont

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=int(var_arrays[vars[0]].shape[0] / plot_every),
        fargs=[refind_nulls_each_timestep, psi1, psi2],
    )
    if savepath is not None:
        anim.save(savepath, fps=fps)

    return anim


def plot_power_deposition_vs_d_sep(
    rundirs: list[str],
    rundir_labels: list[str],
    sepdirs: list[str],
    seps: list[float],
    timestep: list[int] = -1,
    avg_window: list[int] | int = 50,
    tgrid_rundir: str = None,
    xlim: list[float] | None = None,
    p1_ylim: list[float] | None = None,
    p2_ylim: list[float] | None = None,
    p3_ylim: list[float] | None = None,
    p4_ylim: list[float] | None = None,
    p1_ylog: bool = False,
    p2_ylog: bool = False,
    p3_ylog: bool = False,
    p4_ylog: bool = False,
    rundir_colours: list[str] | None = None,
    plot_p1: bool = True,
    plot_p2: bool = True,
    plot_p3: bool = True,
    plot_p4: bool = True,
    legend_loc: str = "upper right",
) -> None:
    """Plot power deposition to each divertor leg as a function of inter-null separation distance

    :param rundirs: List of directories containing xarray datasets from BOUT++ simulation
    :param rundir_labels: Labels to apply to plot for each rundir
    :param sepdirs: List of different internull separations
    :param seps: Values of internull separations
    :param timestep: Timestamp(s) at which to calculate P_l, defaults to -1
    :param avg_window: Averaging window to use, defaults to 50
    :param tgrid_rundir: Which rundir to use as a reference time grid to interpolate other runs to, defaults to None
    :param xlim: x-axis limits, defaults to None
    :param p1_ylim: P_1 y-axis limits, defaults to None
    :param p2_ylim: P_2 y-axis limits, defaults to None
    :param p3_ylim: P_3 y-axis limits, defaults to None
    :param p4_ylim: P_4 y-axis limits, defaults to None
    :param p1_ylog: Whether to plot P_1 on log scale, defaults to False
    :param p2_ylog: Whether to plot P_2 on log scale, defaults to False
    :param p3_ylog: Whether to plot P_3 on log scale, defaults to False
    :param p4_ylog: Whether to plot P_4 on log scale, defaults to False
    :param rundir_colours: Colours to use for each rundir, defaults to None
    :param plot_p1: Whether to plot P_1, defaults to True
    :param plot_p2: Whether to plot P_2, defaults to True
    :param plot_p3: Whether to plot P_3, defaults to True
    :param plot_p4: Whether to plot P_4, defaults to True
    :param legend_loc: Legend location, defaults to "upper right"
    """
    if tgrid_rundir is None:
        tgrid_rundir = os.path.join(rundirs[0], sepdirs[0])
    tgrid = read_boutdata(
        tgrid_rundir,
        remove_xgc=True,
    ).t

    if isinstance(avg_window, int):
        avg_window = [avg_window] * len(rundirs)

    if isinstance(timestep, int):
        timestep = [timestep] * len(rundirs)

    markers = ["x", "+", "*", "o", "v", "^"]

    P1s = np.zeros((len(rundirs), len(sepdirs)))
    P2s = np.zeros((len(rundirs), len(sepdirs)))
    P3s = np.zeros((len(rundirs), len(sepdirs)))
    P4s = np.zeros((len(rundirs), len(sepdirs)))
    for i, rundir in enumerate(rundirs):
        for j, sigma in enumerate(sepdirs):
            ds = read_boutdata(os.path.join(rundir, sigma), remove_xgc=True)
            ds.interp(t=tgrid)
            ds_avg = ds.rolling(t=avg_window[i], min_periods=1).mean()

            _, Q1, Q2, Q3, Q4 = get_Q_legs(ds_avg)
            Q_tot = Q1 + Q2 + Q3 + Q4
            P1s[i, j] = (Q1 / Q_tot).isel(t=timestep[i])
            P2s[i, j] = (Q2 / Q_tot).isel(t=timestep[i])
            P3s[i, j] = (Q3 / Q_tot).isel(t=timestep[i])
            P4s[i, j] = (Q4 / Q_tot).isel(t=timestep[i])

    if rundir_colours is None:
        alphas = np.linspace(1.0, 0.25, len(rundirs))
        rundir_colours = ["red"] * len(rundirs)
    else:
        alphas = np.ones(len(rundirs))

    num_ax = plot_p1 + plot_p2 + plot_p3 + plot_p4
    fig, ax = plt.subplots(num_ax, sharex=True)

    ax_j = 0
    if plot_p4:
        for i in range(len(rundirs)):
            ax[ax_j].plot(
                seps,
                P4s[i, :],
                color=rundir_colours[i],
                linestyle="--",
                marker=markers[i],
                alpha=alphas[i],
            )
        if p4_ylog:
            ax[ax_j].set_yscale("log")
        ax[ax_j].set_ylabel("$P_4$")
        ax[ax_j].grid()
        if p4_ylim is not None:
            ax[ax_j].set_ylim(p4_ylim)
        ax_j += 1

    if plot_p3:
        for i in range(len(rundirs)):
            ax[ax_j].plot(
                seps,
                P3s[i, :],
                color=rundir_colours[i],
                linestyle="--",
                marker=markers[i],
                alpha=alphas[i],
            )
        if p3_ylog:
            ax[ax_j].set_yscale("log")
        ax[ax_j].set_ylabel("$P_3$")
        ax[ax_j].grid()
        if p3_ylim is not None:
            ax[ax_j].set_ylim(p3_ylim)
        ax_j += 1

    if plot_p2:
        for i in range(len(rundirs)):
            ax[ax_j].plot(
                seps,
                P2s[i, :],
                color=rundir_colours[i],
                linestyle="--",
                marker=markers[i],
                alpha=alphas[i],
            )
        if p2_ylog:
            ax[ax_j].set_yscale("log")
        ax[ax_j].set_ylabel("$P_2$")
        ax[ax_j].grid()
        if p2_ylim is not None:
            ax[ax_j].set_ylim(p2_ylim)
        ax_j += 1

    if plot_p1:
        for i in range(len(rundirs)):
            ax[ax_j].plot(
                seps,
                P1s[i, :],
                color=rundir_colours[i],
                linestyle="--",
                marker=markers[i],
                alpha=alphas[i],
            )
        if p1_ylog:
            ax[ax_j].set_yscale("log")
        ax[ax_j].set_ylabel("$P_1$")
        ax[ax_j].grid()
        if p1_ylim is not None:
            ax[ax_j].set_ylim(p1_ylim)
        ax_j += 1

    for i in range(len(rundirs)):
        ax[0].plot(
            [],
            [],
            color=rundir_colours[i],
            marker=markers[i],
            linestyle="--",
            alpha=alphas[i],
            label=rundir_labels[i],
        )
    ax[0].legend(loc=legend_loc)

    ax[-1].set_xlabel("Inter-null distance $d_{xx}$ [cm]")
    if xlim is not None:
        ax[-1].set_xlim(xlim)


def grid_limited_d_sol(target_d_sol: float = 10.0, dx: float = 0.9) -> float:
    """Get the domain length [cm] in y-axis, measured from primary x point to the upstream/core boundary,
    required to give the provided target SOL width at nulls points, assuming only flux expansion contributes

    :param target_d_sol: target value of SOL width at null points, defaults to 10.0
    :param dx: x grid width, defaults to 0.9 [cm]
    :return : L_y, in same units as inputs
    """
    L_y = (target_d_sol / (dx ** (1 / 3))) ** (3 / 2)
    return L_y


def lineslice(
    ds: xr.Dataset,
    variable: str,
    line_coords: np.ndarray,
    timestep: int = -1,
):
    """Evaluate a simulation variable along a line given by line_coords

    :param ds: Xarray dataset from BOUT++
    :param variable: Name of variable to evaluate
    :param line_coords: 2D array of line coordinates [[x1, y1], ..., [xN, yN]]
    :param timestep: Integer timestep to evluate at, defaults to -1
    :return: 1D numpy array
    """
    result = np.zeros(len(line_coords))
    for i in range(len(line_coords)):
        x = line_coords[i][0]
        y = line_coords[i][1]
        result[i] = ds[variable].interp(x=x, y=y).isel(t=timestep).values
    return result


def plot_lineslice(
    ds: xr.Dataset,
    variable: str,
    line_coords: np.ndarray | None = None,
    timesteps: int | list[int] = -1,
):
    """Plot values of a given variable along a line given by the input coordinates

    :param ds: xarray dataset from BOUT++ simulation
    :param variable: Name of variable to plot
    :param line_coords: Line coordinates
    :param timesteps: Timestamp to plot, defaults to -1
    """
    if line_coords is None:
        x1, _, y1, _, _, _ = find_null_coords(ds, timestep=0)
        Lx = ds.x.max() - ds.x.min()
        p0 = [x1 - 0.5 * Lx, y1 + 0.5 * np.tan(30 * np.pi / 180) * Lx]
        p1 = [x1, y1]
        line_coords = np.array(
            [np.linspace(p0[0], p1[0], 100), np.linspace(p0[1], p1[1], 100)]
        ).transpose()

    # Get parallel coordinate
    s = np.zeros(len(line_coords))
    for i in range(1, len(line_coords)):
        s[i] = s[i - 1] + np.sqrt(
            (line_coords[i][0] - line_coords[i - 1][0]) ** 2
            + (line_coords[i][1] - line_coords[i - 1][1]) ** 2
        )

    if isinstance(timesteps, int):
        timesteps = [timesteps]

    vals = []
    for timestep in timesteps:
        vals.append(lineslice(ds, variable, line_coords, timestep))

    if variable == "P":
        vals = [(v - ds["P"].values[0, 0, 0]) * ds.metadata["P_0"] for v in vals]

    # Get 1/e drop off point
    P_max_over_e = []
    for i, timestep in enumerate(timesteps):
        P_max_over_e.append(np.max([v for v in vals[i] if not np.isnan(v)]) / np.e)

    fig, ax = plt.subplots(2)

    ax[0].contourf(
        ds.x,
        ds.y,
        ds.isel(t=timesteps[0])[variable].values.T,
        cmap="inferno",
        levels=200,
    )
    x = line_coords.T[0]
    y = line_coords.T[1]
    ax[0].plot(x, y, color="red", linestyle="-")
    for i, timestep in enumerate(timesteps):
        (l,) = ax[1].plot(
            s,
            vals[i],
            label="t = {:.2f}ms".format(
                1000 * (ds.t[timestep] - ds.t[0]) * ds.metadata["t_0"]
            ),
        )
        ax[1].axhline(P_max_over_e[i], color=l.get_color(), linestyle="--")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylabel(variable)
    fig.tight_layout()


def ddx(ds: xr.Dataset, f: str) -> xr.DataArray:
    """Calculate x partial deriviative of variable f in dataset ds

    :param ds: xarray dataset from BOUT++
    :param f: Name of variable in ds
    :return: df/dx
    """
    f = ds[f].values
    dx = ds["dx"].values
    dfdx = np.zeros(f.shape)
    for t in range(dfdx.shape[0]):
        for ix in range(1, dfdx.shape[1] - 1):
            dfdx[t, ix, :] = (f[t, ix + 1, :] - f[t, ix - 1, :]) / (2 * dx[ix, :])

    dfdx = xr.DataArray(data=dfdx, dims=["t", "x", "y"])
    return dfdx


def ddy(ds: xr.Dataset, f: str) -> xr.DataArray:
    """Calculate y partial deriviative of variable f in dataset ds

    :param ds: xarray dataset from BOUT++
    :param f: Name of variable in ds
    :return: df/dx
    """
    f = ds[f].values
    dy = ds["dy"].values
    dfdy = np.zeros(f.shape)
    for t in range(dfdy.shape[0]):
        for iy in range(1, dfdy.shape[2] - 1):
            dfdy[t, :, iy] = (f[t, :, iy + 1] - f[t, :, iy - 1]) / (2 * dy[:, iy])

    dfdy = xr.DataArray(data=dfdy, dims=["t", "x", "y"])
    return dfdy


def explore_nulls(ds: xr.Dataset, t: int = 0, colorbar: bool = False):
    """Interactive plot with sliders for two surfaces of constant psi

    :param ds: xarray dataset
    :param t: timestep to plot, defaults to 0
    :param colorbar: whether to add colorbar to plot
    :return: sliders to enable interactivity in notebook
    """
    fig, ax = plt.subplots(1)
    cont = ax.contourf(
        ds.x, ds.y, ds["P"].isel(t=t).transpose(), cmap="inferno", levels=200
    )
    if colorbar:
        fig.colorbar(cont, label="$P$")
    _, _, _, _, psi1, psi2 = find_null_coords(ds, timestep=t)
    surfs = [
        ax.contour(
            ds.x,
            ds.y,
            ds["psi"].isel(t=t).transpose(),
            levels=[psi1],
            colors="white",
            zorder=999,
            linestyles="-",
        ),
        ax.contour(
            ds.x,
            ds.y,
            ds["psi"].isel(t=t).transpose(),
            levels=[psi2],
            colors="white",
            zorder=999,
            linestyles="--",
        ),
    ]

    fig.subplots_adjust(left=0.15, bottom=0.3, right=0.85, top=0.92)

    axsurf1 = fig.add_axes([0.15, 0.15, 0.6, 0.03])
    axsurf2 = fig.add_axes([0.15, 0.05, 0.6, 0.03])
    psi_rng = ds["psi"].isel(t=t).max().values - ds["psi"].isel(t=t).min().values

    surf1_slider = Slider(
        ax=axsurf1,
        label=r"$\psi_1$",
        valmin=-0.02 * psi_rng,
        valmax=0.01 * psi_rng,
        valinit=psi1,
    )
    surf2_slider = Slider(
        ax=axsurf2,
        label=r"$\psi_2$",
        valmin=-0.02 * psi_rng,
        valmax=0.01 * psi_rng,
        valinit=psi2,
    )

    def update_surf1(val):
        surfs[0].remove()
        surfs[0] = ax.contour(
            ds.x,
            ds.y,
            ds["psi"].isel(t=t).transpose(),
            levels=[val],
            colors="white",
            zorder=999,
            linestyles="-",
        )
        return surfs

    def update_surf2(val):
        surfs[1].remove()
        surfs[1] = ax.contour(
            ds.x,
            ds.y,
            ds["psi"].isel(t=t).transpose(),
            levels=[val],
            colors="white",
            zorder=999,
            linestyles="--",
        )
        return surfs

    surf1_slider.on_changed(update_surf1)
    surf2_slider.on_changed(update_surf2)

    ax.plot([], [], color="white", linestyle="-", label=r"$\psi_1$")
    ax.plot([], [], color="white", linestyle="--", label=r"$\psi_2$")
    ax.legend(loc="upper left")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    # plt.show()
    return surf1_slider, surf2_slider


def extract_rundeck_data(runs: list[str], d_sol: float, avg_window: int = 50) -> dict:
    """Extract churning mode relevant data from a rundeck

    :param runs: List of churning mode run data directories
    :param d_sol: Width of SOL at null point [m]
    :return: Dict containing a series of arrays
    """

    # Create some arrays
    nr = len(runs)
    p_null = np.zeros(nr)
    P_out_wo = np.zeros(nr)
    P_out_w = np.zeros(nr)
    chi_perp_eff_wo = np.zeros(nr)
    chi_perp_eff_w = np.zeros(nr)
    n_sepx = np.zeros(nr)
    r_ch = np.zeros(nr)
    grad_p = np.zeros(nr)
    beta_p = np.zeros(nr)
    v_th_null = np.zeros(nr)
    tau_ch = np.zeros(nr)
    D_x_th = np.zeros(nr)
    max_t = np.zeros(nr)
    d_xx = np.zeros(nr)

    for i, r in enumerate(runs):
        # Read in dataset
        ds = read_boutdata(r, remove_xgc=True, units="m")
        ds = ds.rolling(t=avg_window, min_periods=1).mean()
        _, Q1, Q2, Q3, Q4 = get_Q_legs(ds)

        # Find the nulls
        x1, x2, y1, y2, psi1, psi2 = find_null_coords(ds, timestep=0)
        d_xx[i] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Find the line crossing the null region
        Lx = ds.x.max() - ds.x.min()
        p0 = [
            x1 - 0.5 * Lx,
            y1 + 0.5 * np.tan(30 * np.pi / 180) * Lx,
        ]
        p1 = [x1, y1]
        null_region_line = np.array(
            [np.linspace(p0[0], p1[0], 100), np.linspace(p0[1], p1[1], 100)]
        ).transpose()

        # Extract other data
        n_sepx[i] = ds.metadata["n_sepx"]
        p_null[i] = (
            lineslice(ds, "P", null_region_line, timestep=0).max() * ds.metadata["P_0"]
        )
        r_ch[i] = (
            0.81
            * ds.metadata["a_mid"]
            * (
                ((p_null[i] * 2 * mu_0) / ds.metadata["B_pmid"] ** 2)
                * ds.metadata["a_mid"]
                / ds.metadata["R_0"]
            )
            ** (1 / 3)
        )
        grad_p[i] = p_null[i] / d_sol
        max_t[i] = ds.t.values[-1] - ds.t.values[0]
        P_out_wo[i] = (
            Q1.isel(t=0) + Q2.isel(t=0) + Q3.isel(t=0) + Q4.isel(t=0)
        ).values * 1e6
        P_out_w[i] = (
            Q1.isel(t=-1) + Q2.isel(t=-1) + Q3.isel(t=-1) + Q4.isel(t=-1)
        ).values * 1e6
        chi_perp_eff_wo[i] = (P_out_wo[i] / ds.metadata["a_mid"]) / grad_p[i]
        chi_perp_eff_w[i] = (P_out_w[i] / ds.metadata["a_mid"]) / grad_p[i]
        beta_p[i] = 2.0 * mu_0 * p_null[i] / ds.metadata["B_pmid"] ** 2
        v_th_null[i] = np.sqrt(p_null[i] / (n_sepx[i] * m_i))
        tau_ch[i] = np.sqrt(ds.metadata["R_0"] * r_ch[i]) / v_th_null[i]
        D_x_th[i] = 0.5 * r_ch[i] ** 2 / tau_ch[i]

    # Store in a dict (TODO: use a dataarray here instead?)
    out = {}
    out["p_null"] = p_null
    out["P_out_wo"] = P_out_wo
    out["P_out_w"] = P_out_w
    out["chi_perp_eff_wo"] = chi_perp_eff_wo
    out["chi_perp_eff_w"] = chi_perp_eff_w
    out["n_sepx"] = n_sepx
    out["r_ch"] = r_ch
    out["grad_p"] = grad_p
    out["beta_p"] = beta_p
    out["v_th_null"] = p_null
    out["tau_ch"] = tau_ch
    out["D_x_th"] = D_x_th
    out["max_t"] = max_t

    return out
