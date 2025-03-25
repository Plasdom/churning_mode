#include "header.hxx"
#include "bout/field_factory.hxx"

int Churn::init(bool restarting) // TODO: Use the restart flag
{

    /******************Reading options *****************/

    auto &globalOptions = Options::root();
    auto &options = globalOptions["model"];

    // Load system parameters
    chi_diff = options["chi_diff"].doc("Thermal diffusivity").withDefault(0.0);
    chi_par = options["chi_par"].doc("Parallel thermal conductivity").withDefault(0.0);
    chi_perp = options["chi_perp"].doc("Perpendicular thermal conductivity").withDefault(0.0);
    D_m = options["D_m"].doc("Magnetic diffusivity").withDefault(0.0);
    mu = options["mu"].doc("Kinematic viscosity").withDefault(0.0);
    R_0 = options["R_0"].doc("Major radius [m]").withDefault(1.5);
    a_mid = options["a_mid"].doc("Minor radius at outer midplane [m]").withDefault(0.6);
    n_sepx = options["n_sepx"].doc("Electron density at separatrix [m^-3]").withDefault(1.0e19);
    T_sepx = options["T_sepx"].doc("Plasma temperature at separatrix [eV]").withDefault(100.0);
    B_t0 = options["B_t0"].doc("Toroidal field strength [T]").withDefault(2.0);
    B_pmid = options["B_pmid"].doc("Poloidal field strength at outer midplane [T]").withDefault(0.25);
    T_down = options["T_down"].doc("Downstream fixed temperature [eV]").withDefault(10.0);
    D_x = options["D_x"].doc("Peak of additional perpendicular diffusion coefficient [m^2/s]").withDefault(0.0);
    x_1 = options["x_1"].doc("x-coordinate of first X-point (centred on the middle of the simulation domain) [a_mid]").withDefault(0.0);
    x_2 = options["x_2"].doc("x-coordinate of second X-point (centred on the middle of the simulation domain) [a_mid]").withDefault(0.0);
    y_1 = options["y_1"].doc("y-coordinate of first X-point (centred on the middle of the simulation domain) [a_mid]").withDefault(0.0);
    y_2 = options["y_2"].doc("y-coordinate of second X-point (centred on the middle of the simulation domain) [a_mid]").withDefault(0.0);
    r_star = options["r_star"].doc("Radius of the additional mixing zone [a_mid]").withDefault(0.1);
    lambda_SOL_rho = options["lambda_SOL_rho"].doc("SOL width parameter in units of normalised flux coordinate").withDefault(3.0);
    P_core = options["P_core"].doc("Pressure at core boundary [P_0]").withDefault(1.0);
    psi_bndry_P_core_BC = options["psi_bndry_P_core_BC"].doc("Psi vlaue defining core boundary for fixed_P_core option").withDefault(0.0);
    Q_in = options["Q_in"].doc("Input power to top of domain [MW]").withDefault(1.0);
    alpha_fl = options["alpha_fl"].doc("Flux limiter").withDefault(0.2);
    alpha_rot = options["alpha_rot"].doc("Rotation angle of initial poloidal flux").withDefault(0.0);

    // Model option switches
    evolve_pressure = options["evolve_pressure"]
                          .doc("Evolve plasma pressure")
                          .withDefault(true);
    include_mag_restoring_term = options["include_mag_restoring_term"]
                                     .doc("Include the poloidal magnetic field restoring term in the vorticity equation (2 / beta_p * {psi, Del^2 psi})")
                                     .withDefault(true);
    include_churn_drive_term = options["include_churn_drive_term"]
                                   .doc("Include the churn driving term in the vorticity equation (2 * epsilon * dP/dy)")
                                   .withDefault(true);
    include_thermal_force_term = options["include_thermal_force_term"]
                                     .doc("Include the thermal force term in the psi equation")
                                     .withDefault(false);
    invert_laplace = options["invert_laplace"]
                         .doc("Use Laplace inversion routine to solve phi (if false, will instead solve via a constraint)")
                         .withDefault(true);
    include_advection = options["include_advection"]
                            .doc("Include advection terms or not")
                            .withDefault(true);
    fixed_T_down = options["fixed_T_down"]
                       .doc("Use a constant value for P on downstream boundaries")
                       .withDefault(false);
    fixed_P_core = options["fixed_P_core"]
                       .doc("Fix upstream boundary condition in the core (i.e. within the separatrix defined by psi)")
                       .withDefault(true);
    use_classic_div_q_par = options["use_classic_div_q_par"]
                                .doc("Use a classic stencil for the parallel div_q term")
                                .withDefault(false);
    use_classic_div_q_perp = options["use_classic_div_q_perp"]
                                 .doc("Use a classic stencil for the perpendicular div_q term")
                                 .withDefault(false);
    use_gunter_div_q_par = options["use_gunter_div_q_par"]
                               .doc("Use the Gunter stencil for the parallel div_q term")
                               .withDefault(false);
    use_gunter_div_q_perp = options["use_gunter_div_q_perp"]
                                .doc("Use the Gunter stencil for the perpendicular div_q term")
                                .withDefault(false);
    use_modified_stegmeir_div_q_par = options["use_modified_stegmeir_div_q_par"]
                                          .doc("Use a modified version of the Stegmeir stencil for the parallel div_q term")
                                          .withDefault(false);
    use_linetrace_div_q_par = options["use_linetrace_div_q_par"]
                                  .doc("Use a fielline tracing algorithm for the parallel div_q term")
                                  .withDefault(false);
    T_dependent_q_par = options["T_dependent_q_par"]
                            .doc("Use Spitzer-Harm parallel thermal conductivity (implemented for modified_stegmeir_div_q_par only)")
                            .withDefault(false);
    fixed_Q_in = options["fixed_Q_in"]
                     .doc("Fixed input energy at top of domain")
                     .withDefault(false);
    use_flux_limiter = options["use_flux_limiter"]
                           .doc("Use a flux limiter on parallel thermal conduction")
                           .withDefault(false);
    disable_qin_outside_core = options["disable_qin_outside_core"]
                                   .doc("Set input heat flux to zero outside psi=psi_bndry_P_core_BC")
                                   .withDefault(false);

    // Constants
    m_i = options["m_i"].withDefault(2 * 1.667e-27);
    e = 1.602e-19;
    m_e = 9.11e-31;
    mu_0 = 1.256637e-6;
    eps_0 = 8.854188e-12;
    pi = 3.14159;
    boltzmann_k = 1.380649e-23;

    // Get normalisation and derived parameters
    c = 1.0;
    rho = (m_i + m_e) * n_sepx;
    P_0 = e * n_sepx * T_sepx;
    C_s0 = sqrt(P_0 / rho);
    t_0 = a_mid / C_s0;
    D_0 = a_mid * C_s0;
    psi_0 = B_pmid * R_0 * a_mid;
    phi_0 = pow(C_s0, 2) * abs(B_t0) * t_0 / c;
    epsilon = a_mid / R_0;
    // beta_p = 1.0e-7 * 8.0 * pi * P_0 / pow(B_pmid, 2.0);
    beta_p = 2.0 * mu_0 * P_0 / pow(B_pmid, 2.0);
    P_grad_0 = P_0 / a_mid;
    Omega_i0 = abs(B_t0) * e / m_i;
    b0 = B_t0 / sqrt(pow(B_t0, 2.0));
    delta = (1.0 / (2.0 * t_0 * Omega_i0));

    // phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
    Options &phi_init_options = Options::root()["phi"];
    if (invert_laplace)
    {
        // TODO: Get LaplaceXY working as it is quicker
        //  phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
        phi = 0.0; // Starting guess for first solve (if iterative)
        phi_init_options.setConditionallyUsed();
        mm.A = 0.0;
        mm.D = 1.0;
        mm.ngcx_tot = ngcx_tot;
        mm.ngcy_tot = ngcy_tot;
        mm.nx_tot = nx_tot;
        mm.ny_tot = ny_tot;
        mm.nz_tot = nz_tot;
        mySolver.setOperatorFunction(mm);
        mySolver.setup();

        SOLVE_FOR(P, psi, omega);
        SAVE_REPEAT(u, phi, B);
    }
    else
    {
        phi.setBoundary("phi");

        SOLVE_FOR(P, psi, omega, phi);
        SAVE_REPEAT(u, B);
    }

    if (chi_par > 0.0)
    {
        if (T_dependent_q_par)
        {
            SAVE_REPEAT(kappa_par);
        }
        SAVE_REPEAT(q_par);
    }
    if ((chi_perp + D_x) > 0.0)
    {
        SAVE_ONCE(kappa_perp);
        SAVE_REPEAT(q_perp);
    }

    // if (fixed_Q_in)
    // {
    //     FieldFactory f(mesh);
    //     q_in_yup = f.create2D("model:q_in_yup");
    //     q_in_yup = q_in_yup / (P_0 * C_s0);
    //     SAVE_ONCE(q_in_yup);
    // }
    // else
    // {
    //     Options::root()["model:q_in_yup"].setConditionallyUsed();
    // }

    // SAVE_REPEAT(debugvar);
    // debugvar = 0.0;
    // SAVE_REPEAT(div_q);
    SAVE_REPEAT(thermal_force_term);
    thermal_force_term = 0.0;

    if (fixed_Q_in)
    {
        num_q_in_cells = round((1.0 / 4.0) * static_cast<BoutReal>(mesh->GlobalNxNoBoundaries));
        q_in = Q_in / (2.0 * pi * R_0 * num_q_in_cells * mesh->getCoordinates()->dx(0, 0) * a_mid);
        q_in = 1.0e6 * q_in / (C_s0 * P_0);
    }

    // Set downstream pressure boundaries
    // TODO: Find out why this is needed instead of just setting bndry_xin=dirichlet, etc
    if (fixed_T_down)
    {
        // X boundaries
        if (mesh->firstX())
        {
            for (int ix = 0; ix < ngcx_tot; ix++)
            {
                for (int iy = 0; iy < mesh->LocalNy; iy++)
                {
                    for (int iz = 0; iz < mesh->LocalNz; iz++)
                    {
                        P(ix, iy, iz) = T_down / T_sepx;
                        // P(ix, iy, iz) = 0.0;
                    }
                }
            }
        }
        if (mesh->lastX())
        {
            for (int ix = mesh->LocalNx - ngcx_tot; ix < mesh->LocalNx; ix++)
            {
                for (int iy = 0; iy < mesh->LocalNy; iy++)
                {
                    for (int iz = 0; iz < mesh->LocalNz; iz++)
                    {
                        P(ix, iy, iz) = T_down / T_sepx;
                        // P(ix, iy, iz) = 0.0;
                    }
                }
            }
        }
        // Y boundaries
        for (itl.first(); !itl.isDone(); itl++)
        {
            // it.ind contains the x index
            for (int iy = 0; iy < ngcy_tot; iy++)
            {
                for (int iz = 0; iz < mesh->LocalNz; iz++)
                {
                    P(itl.ind, iy, iz) = T_down / T_sepx;
                    // P(itl.ind, iy, iz) = 0.0;
                }
            }
        }
        // for (itu.first(); !itu.isDone(); itu++)
        // {
        //   for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        //   {
        //     for (int iz = 0; iz < mesh->LocalNz; iz++)
        //     {
        //       // P(itu.ind, iy, iz) = T_down / T_sepx;
        //       P(itu.ind, iy, iz) = 0.0;
        //     }
        //   }
        // }
    }
    if (fixed_P_core)
    {
        // TODO: use the BC function for this
        for (itu.first(); !itu.isDone(); itu++)
        {
            // for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
            for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
            {
                for (int iz = 0; iz < mesh->LocalNz; iz++)
                {
                    if (psi(itu.ind, iy, iz) >= 0.0)
                    {
                        P(itu.ind, iy, iz) = P_core;
                    }
                    else
                    {
                        P(itu.ind, iy, iz) = P_core * exp(-pow((sqrt((psi(itu.ind, iy, iz) - 1.0) / (-1.0)) - 1.0) / lambda_SOL_rho, 2.0));
                    }
                }
            }
        }
    }

    // Initialise unit vector in z direction
    e_z.x = 0.0;
    e_z.y = 0.0;
    e_z.z = 1.0;

    // Initialise heat flow
    q_par = 0.0;
    q_perp = 0.0;
    kappa_par = 0.0;
    kappa_perp = 0.0;
    div_q = 0.0;

    // Output constants, input options and derived parameters
    SAVE_ONCE(e, m_i, m_e, chi_diff, D_m, mu, epsilon, beta_p, rho, P_0);
    SAVE_ONCE(C_s0, t_0, D_0, psi_0, phi_0, R_0, a_mid, n_sepx);
    SAVE_ONCE(T_sepx, B_t0, B_pmid, evolve_pressure, include_churn_drive_term, include_mag_restoring_term, P_grad_0);
    SAVE_ONCE(ngcx, ngcx_tot, ngcy, ngcy_tot, chi_perp, chi_perp_eff, chi_par);
    SAVE_ONCE(x_c, y_c, delta, b0, Omega_i0);

    Coordinates *coord = mesh->getCoordinates();

    // generate coordinate system
    // coord->Bxy = 1.0; // TODO: Use B_t here?
    coord->g11 = 1.0;
    coord->g22 = 1.0;
    coord->g33 = 1.0;
    coord->g12 = 0.0;
    coord->g13 = 0.0;
    coord->g23 = 0.0;

    // BoutReal x_0, y_0;
    // x_0 = (mesh->GlobalNx / 2) * coord->dx(0, 0, 0);
    // y_0 = (mesh->GlobalNy / 2) * coord->dy(0, 0, 0);
    x_c.allocate();
    y_c.allocate();
    for (auto i : x_c)
    {
        x_c[i] = mesh->getGlobalXIndex(i.x()) * coord->dx[i] - (mesh->GlobalNx / 2) * coord->dx[i];
        y_c[i] = mesh->getGlobalYIndex(i.y()) * coord->dy[i] - (mesh->GlobalNy / 2) * coord->dy[i];
    }

    // chi_perp_eff
    chi_perp_eff = D_x * exp(-pow(((x_c - x_1) / r_star), 2.0) - pow(((y_c - y_1) / r_star), 2.0));
    chi_perp_eff += D_x * exp(-pow(((x_c - x_2) / r_star), 2.0) - pow(((y_c - y_2) / r_star), 2.0));

    // Initialise perpendicular conductivity
    kappa_perp = (chi_perp / D_0);
    if (D_x > 0.0)
    {
        kappa_perp += (chi_perp_eff / D_0);
    }

    if (T_dependent_q_par == false)
    {
        kappa_par = chi_par / D_0;
    }

    // Initialise B field
    B.x = 0.0;
    B.y = 0.0;
    B.z = (1.0 / (1.0 + x_c * epsilon)) * B_t0 / B_pmid;

    return 0;
}

BOUTMAIN(Churn)
