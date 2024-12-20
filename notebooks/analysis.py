import time
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from xbout import open_boutdataset
from pathlib import Path
from matplotlib import animation
from matplotlib.colors import LightSource
import os

mu_0 = 1.256637e-6


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

    # ngc = int((len(ds["x"]) - len(ds["y"]))/2)
    if remove_xgc:
        ngc = 4
        ds = ds.isel(x=range(ngc, len(ds["x"]) - ngc))

        # ds = ds.assign_coords(x=ds.x - ds.x[0])
        # ds = ds.assign_coords(y=ds.y - ds.y[0])

    # Calculate conductive, convective and total heat fluxes
    ds["q_conv_x"] = (
        (
            2.5 * ds.metadata["P_0"] * ds["P"]
            + 0.5
            * ds.metadata["rho"]
            * ds.metadata["C_s0"] ** 2
            * (ds["u_x"] ** 2 + ds["u_y"] ** 2)
        )
        * ds.metadata["C_s0"]
        * ds["u_x"]
    )
    ds["q_conv_y"] = (
        (
            2.5 * ds.metadata["P_0"] * ds["P"]
            + 0.5
            * ds.metadata["rho"]
            * ds.metadata["C_s0"] ** 2
            * (ds["u_x"] ** 2 + ds["u_y"] ** 2)
        )
        * ds.metadata["C_s0"]
        * ds["u_y"]
    )
    if hasattr(ds, "q_perp_x") is False:
        ds["q_perp_x"] = 0.0
        ds["q_perp_y"] = 0.0
    ds["q_cond_x"] = (
        ds.metadata["n_sepx"]
        * ds.metadata["D_0"]
        * ds.metadata["e"]
        * ds.metadata["T_sepx"]
        / ds.metadata["a_mid"]
    ) * (ds["q_par_x"] + ds["q_perp_x"])
    ds["q_cond_y"] = (
        ds.metadata["n_sepx"]
        * ds.metadata["D_0"]
        * ds.metadata["e"]
        * ds.metadata["T_sepx"]
        / ds.metadata["a_mid"]
    ) * (ds["q_par_y"] + ds["q_perp_y"])
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

    return ax


def contour_overlay(
    ds: xr.Dataset,
    var: str = "P",
    timestamps: list[int] = [0, -1],
    colorbar: bool = False,
    fill: bool = False,
    levels: int | list[float] | np.ndarray = 50,
    savepath: str | None = None,
):
    """Plot overlaid contours at several timestamps


    :param ds: Bout dataset
    :param var: Variable to plot, defaults to "P"
    :param timestamps: List of integer timestamps to overlay, defaults to [0, -1]
    :param fill: Whether to plot using contourf or contour, defaults to False
    :param levels: Number of levels to use or list of specific levels to plot
    :param savepath: Where to save figure
    """
    # linestyles = ["-", "--", ".-"]
    timestamps = list(reversed(timestamps))
    if len(timestamps) > 1:
        alphas = np.linspace(0.25, 1.0, len(timestamps))
    else:
        alphas = [1.0]
    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")
    vmin = ds[var].min()
    vmax = ds[var].max()
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
    for i, t in enumerate(timestamps):
        if fill:
            c = ax.contourf(
                ds["x"],
                ds["y"],
                ds[var][t].values.T,
                alpha=alphas[i],
                levels=plot_levels,
                cmap="inferno",
            )
        else:
            c = ax.contour(
                ds["x"],
                ds["y"],
                ds[var][t].values.T,
                alpha=alphas[i],
                levels=plot_levels,
                cmap="inferno",
            )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(var)
    if colorbar:
        fig.colorbar(c)

    if savepath is not None:
        fig.savefig(savepath)

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
        vmin = var_arrays[v][0].min() - 0.05 * var_arrays[v][0].max()
        vmax = var_arrays[v][0].max() + 0.05 * var_arrays[v][0].max()
        if v == "psi":
            # levels[v] = np.sort(list(np.linspace(vmin, vmax, num_levels)))
            # levels[v] = np.sort(np.array(list(levels[v]) + [0]))
            levels[v] = np.sort(
                np.array(
                    list(-np.linspace(0, (-vmin) ** (1 / 2), 25) ** (2))
                    + list(np.linspace(0, vmax ** (1 / 2), 25) ** (2))[1:]
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
                )
            if plot_r_cz:
                ax[j].plot(x_cz, y_cz, linestyle="--", color="red")
            ax[j].set_xlabel("x")
            ax[j].set_ylabel("y")
            timestamp = ds_plot.t.values[plot_every * i] - ds.t.values[0]
            ax[j].set_title(v + ", t={:.2f}$t_0$".format(timestamp))

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
    density: float = 0.4,
    lw_prefactor: float | None = None,
    savepath: str | None = None,
    **kwargs,
):
    """Plot vector field at a single timestamp

    :param ds: Bout dataset
    :param vec_var: Vector variable
    :param scalar: Scalar variable to plot underneath, defaults to "P"
    :param t: Timestamp
    :param savepath: Where to save figure
    :return: Animation
    """

    # Generate grid for plotting
    X, Y = np.meshgrid(ds["x"], ds["y"])
    vec_x = ds[vec_var + "_x"]
    vec_y = ds[vec_var + "_y"]
    vec_mag = np.sqrt(vec_x**2 + vec_y**2)
    scalar = ds["P"]

    fig = plt.figure()
    ax = plt.axes()
    # ls = LightSource(azdeg=110, altdeg=10)

    cont = ax.pcolormesh(X, Y, scalar.isel(t=t).values.T, cmap="inferno")
    # rgb = ls.shade(scalar.isel(t=t).values.T, vert_exag=50, cmap=plt.cm.magma, blend_mode="overlay")
    # cont = ax.imshow(rgb)
    if lw_prefactor is None:
        # lw = 10 * vec_mag.isel(t=t).values.T / vec_mag.isel(t=t).values.max()
        lw = 3 * np.sqrt(vec_mag.isel(t=t).values.T / vec_mag.isel(t=t).values.max())
    else:
        lw = lw_prefactor * vec_mag.isel(t=t).values.T
        # lw = lw_prefactor * np.sqrt(vec_mag.isel(t=t).values.T)
    cont = ax.streamplot(
        X,
        Y,
        vec_x.isel(t=t).values.T,
        vec_y.isel(t=t).values.T,
        color="red",
        linewidth=lw,
        integration_direction="both",
        density=density,
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

    x_c = ds["x"] - 0.5 * np.sum(ds["dx"].isel(y=0))
    B_px = ds["B_x"]
    B_py = ds["B_y"]
    P = ds["P"]
    u_x = ds["u_x"]
    u_y = ds["u_y"]

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
    y1 = range(int(ny / 4), int(3 * ny / 4))
    x2 = range(int(nx / 2), nx)
    x3 = range(int(nx / 2))
    y4 = range(int(ny / 4), int(3 * ny / 4))
    qin, q1, q2, q3, q4 = get_q_legs(ds)

    Q1 = q1.integrate(coord="y")
    Q2 = q2.integrate(coord="x")
    Q3 = q3.integrate(coord="x")
    Q4 = q4.integrate(coord="y")

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
            range(len(y4)),
            q4[plot_every * i, :],
            linestyle="-",
            color="black",
            label="Leg 4 (W)",
        )
        l = ax[3].plot(
            range(len(y4)),
            q4[0, :],
            linestyle="--",
            color="gray",
            label="t=0",
        )

        timestamp = ds.t.values[plot_every * i] - ds.t.values[0]
        ax[0].set_title("t={:.2f}$t_0$".format(timestamp))
        ax[0].legend(loc="upper right")
        ax[1].legend(loc="upper right")
        ax[2].legend(loc="upper right")
        ax[3].legend(loc="upper right")

        if normalise:
            ax[1].set_ylabel(r"$q_{\parallel} / \int q_{\parallel}ds$ [MWm$^{-3}$]")
        else:
            ax[1].set_ylabel(r"$q_{\parallel}$ [MWm$^{-3}$]")
        ax[-1].set_xlabel("x")
        ax[0].set_ylim(0, max_q1)
        ax[1].set_ylim(0, max_q2)
        ax[2].set_ylim(0, max_q3)
        ax[3].set_ylim(0, max_q4)

        return l

    anim = animation.FuncAnimation(fig, animate, frames=int(len(ds.t) / plot_every))
    if savepath is not None:
        anim.save(savepath, fps=24)

    return anim


def q_target_proportions(
    ds: xr.Dataset, normalise: bool = True, savepath: str | None = None
):
    """Plot total heat flow to each divertor leg (assuming snowflake config)

    :param ds: Xarray dataset
    :param normalise: whether to normalise heat flow to Q_tot, defaults to True
    :param savepath: where to save figure
    """
    qin, q1, q2, q3, q4 = get_q_legs(ds)
    qin = qin.integrate(coord="x")
    q1 = q1.integrate(coord="y")
    q2 = q2.integrate(coord="x")
    q3 = q3.integrate(coord="x")
    q4 = q4.integrate(coord="y")

    q_tot = q1 + q2 + q3 + q4

    fig, ax = plt.subplots(1)
    if normalise:
        ax.stackplot(
            (ds.t - ds.t[0]) * ds.metadata["t_0"] * 1000,
            [q1 / q_tot, q2 / q_tot, q3 / q_tot, q4 / q_tot],
            labels=["1", "2", "3", "4"],
        )
        ax.set_ylabel(r"$P_{l} / P_{tot}$")

    else:
        ax.stackplot(
            (ds.t - ds.t[0]) * ds.metadata["t_0"] * 1000,
            [q1, q2, q3, q4],
            labels=["1", "2", "3", "4"],
        )
        # ax.plot(ds.t - ds.t[0], -qin)
        ax.set_ylabel(r"$P_{l}$ [MWm$^{-2}$]")

    ax.legend(loc="upper left")
    ax.set_xlabel("$t$ [ms]")
    ax.grid()
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
    prefactor = (
        1e-6
        * ds.metadata["n_sepx"]
        * ds.metadata["D_0"]
        * ds.metadata["e"]
        * ds.metadata["T_sepx"]
        / ds.metadata["a_mid"]
    )
    nx = len(ds.x)
    ny = len(ds.y)
    x0 = range(len(ds.x))
    y1 = range(int(ny / 4), int(3 * ny / 4))
    x2 = range(int(nx / 2), nx)
    x3 = range(int(nx / 2))
    y4 = range(int(ny / 4), int(3 * ny / 4))
    qin = -prefactor * np.sqrt(ds["q_tot_x"] ** 2 + ds["q_tot_y"] ** 2).isel(y=-1, x=x0)
    q1 = prefactor * np.sqrt(ds["q_tot_x"] ** 2 + ds["q_tot_y"] ** 2).isel(x=-1, y=y1)
    q2 = prefactor * np.sqrt(ds["q_tot_x"] ** 2 + ds["q_tot_y"] ** 2).isel(y=0, x=x2)
    q3 = prefactor * np.sqrt(ds["q_tot_x"] ** 2 + ds["q_tot_y"] ** 2).isel(y=0, x=x3)
    q4 = prefactor * np.sqrt(ds["q_tot_x"] ** 2 + ds["q_tot_y"] ** 2).isel(x=0, y=y4)

    # Calculate heat flux normal to boundaries
    B_mag = np.sqrt(ds["B_x"] ** 2 + ds["B_y"] ** 2)
    q1 = q1 * ds["B_x"].isel(x=-1, y=y1) / B_mag.isel(x=-1, y=y1)
    q2 = q2 * ds["B_y"].isel(y=0, x=x2) / B_mag.isel(y=0, x=x2)
    q3 = -q3 * ds["B_y"].isel(y=0, x=x3) / B_mag.isel(y=0, x=x3)
    q4 = q4 * ds["B_x"].isel(x=0, y=y4) / B_mag.isel(x=0, y=y4)

    return qin, q1, q2, q3, q4


def get_Q_legs(ds: xr.Dataset):
    qin, q1, q2, q3, q4 = get_q_legs(ds)
    Qin = qin.integrate(coord="x") * ds.metadata["a_mid"]
    Q1 = q1.integrate(coord="y") * ds.metadata["a_mid"]
    Q2 = q2.integrate(coord="x") * ds.metadata["a_mid"]
    Q3 = q3.integrate(coord="x") * ds.metadata["a_mid"]
    Q4 = q4.integrate(coord="y") * ds.metadata["a_mid"]

    return Qin, Q1, Q2, Q3, Q4


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nulls(
    ds: xr.Dataset, num: int = 2, timestamp: int = 0, cmap: str = "inferno"
) -> list:
    """Use magnetic pressure minimums to identify location of null points

    :param ds: xarray dataset output from BOUT++
    :param num: number of nulls (1 or 2)
    :param timestamp: which timestamp in which to find nulls
    :param cmap: colourmap to use for underlying pressure profile
    :return: [x1, x2, y1, y2]
    """
    ds["P_b"] = (
        10
        * ((ds["B_x"] ** 2 + ds["B_y"] ** 2) * ds.metadata["B_pmid"] ** 2)
        / (2 * mu_0)
    )
    dr = ds["P_b"].isel(t=timestamp)
    firstmin = dr.argmin(dim=["x", "y"], keep_attrs=True)
    i1 = firstmin["x"].values
    j1 = firstmin["y"].values
    x1 = dr.x[i1]
    y1 = dr.y[j1]
    # print("(x1,y1) = ({:.4f},{:.4f})".format(x1, y1))
    dr[i1, j1] = dr.values.max()
    if num == 2:
        for k in range(100):
            secondmin = dr.argmin(dim=["x", "y"], keep_attrs=True)
            i2 = secondmin["x"].values
            j2 = secondmin["y"].values

            if (abs(i2 - i1) > 1) or (abs(j2 - j1) > 1):
                x2 = dr.x[i2]
                y2 = dr.y[j2]
                # print("(x2,y2) = ({:.4f},{:.4f})".format(x2, y2))
                break
            else:
                dr[i2, j2] = dr.values.max()
    else:
        x2 = x1
        y2 = y1

    print(r"x_1 = {:.4f} # x-coordinate of first X-point  [a_mid]".format(x1))
    print(r"y_1 = {:.4f} # y-coordinate of first X-point  [a_mid]".format(y1))
    print(r"x_2 = {:.4f} # x-coordinate of second X-point  [a_mid]".format(x2))
    print(r"y_2 = {:.4f} # y-coordinate of second X-point  [a_mid]".format(y2))

    fig, ax = plt.subplots(1)
    # ax.pcolormesh(
    #     ds.x, ds.y, np.log(ds["P_b"].isel(t=timestamp)).values.T, cmap="inferno"
    # )

    psi1 = ds["psi"].isel(x=i1, y=j1, t=timestamp).values
    psi2 = ds["psi"].isel(x=i2, y=j2, t=timestamp).values
    ax.pcolormesh(ds.x, ds.y, ds["P"].isel(t=timestamp).values.T, cmap=cmap)
    ax.contour(
        ds.x,
        ds.y,
        ds["psi"].isel(t=timestamp).values.T,
        levels=np.sort([psi1, psi2]),
        colors="white",
        linestyles=["--", "-"],
    )
    ax.scatter([x1, x2], [y1, y2], color="red", marker="x", zorder=999)

    print("")
    print(r"psi_1 = {:.5f} ".format(psi1))
    print(r"psi_2 = {:.5f} ".format(psi2))

    return [x1, x2, y1, y2]


def plot_power_deposition_vs_d_sep(
    rundirs: list[str],
    rundir_labels: list[str],
    sepdirs: list[str],
    seps: list[float],
    timestamp: int = -1,
    avg_window: int = 50,
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
) -> None:
    if tgrid_rundir is None:
        tgrid_rundir = os.path.join(rundirs[0], sepdirs[0])
    tgrid = read_boutdata(
        tgrid_rundir,
        remove_xgc=True,
    ).t

    markers = ["x", "+", "*", "o", "v", "^"]

    P1s = np.zeros((len(rundirs), len(sepdirs)))
    P2s = np.zeros((len(rundirs), len(sepdirs)))
    P3s = np.zeros((len(rundirs), len(sepdirs)))
    P4s = np.zeros((len(rundirs), len(sepdirs)))
    for i, rundir in enumerate(rundirs):
        for j, sigma in enumerate(sepdirs):
            ds = read_boutdata(os.path.join(rundir, sigma), remove_xgc=True)
            ds.interp(t=tgrid)
            ds_avg = ds.rolling(t=avg_window, min_periods=1).mean()

            _, Q1, Q2, Q3, Q4 = get_Q_legs(ds_avg)
            Q_tot = Q1 + Q2 + Q3 + Q4
            P1s[i, j] = (Q1 / Q_tot).isel(t=timestamp)
            P2s[i, j] = (Q2 / Q_tot).isel(t=timestamp)
            P3s[i, j] = (Q3 / Q_tot).isel(t=timestamp)
            P4s[i, j] = (Q4 / Q_tot).isel(t=timestamp)

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
    ax[0].legend(loc="upper right")

    ax[-1].set_xlabel("Inter-null separation [cm]")
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
