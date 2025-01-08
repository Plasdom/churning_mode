#include "header.hxx"

int Churn::rhs(BoutReal UNUSED(t))
{
    // Reset the upstream P boundary
    if (fixed_P_core)
    {
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

    // Solve phi
    ////////////////////////////////////////////////////////////////////////////
    if (invert_laplace)
    {
        mesh->communicate(P, psi, omega);
        phi = mySolver.invert(omega);
        try
        {
            for (int i = 0; i < nits_inv_extra; i++)
            {
                phi = mySolver.invert(omega, phi);
                mesh->communicate(phi);
            }
        }
        catch (BoutException &e)
        {
        };
    }
    else
    {
        mesh->communicate(P, psi, omega, phi);
        ddt(phi) = (D2DX2(phi) + D2DY2(phi)) - omega;
    }
    mesh->communicate(phi);

    // Calculate velocity
    u = -cross(e_z, Grad(phi));

    // mesh->communicate(u);

    // // Calculate B
    B.x = -(1.0 / (1.0 + x_c * epsilon)) * DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B.y = (1.0 / (1.0 + x_c * epsilon)) * DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B_mag = abs(B);

    // mesh->communicate(B);

    // Get T
    T = P; // Normalised T = normalised P when rho = const

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
        if (include_advection)
        {
            if (mesh->StaggerGrids)
            {
                ddt(P) = -V_dot_Grad(u, P);
            }
            else
            {
                ddt(P) = -(DDX(P) * u.x + u.y * DDY(P));
            }
        }
        else
        {
            ddt(P) = 0;
        }
        if (use_classic_div_q_par || use_gunter_div_q_par || use_modified_stegmeir_div_q_par || use_linetrace_div_q_par)
        {
            // Calculate parallel conductivity
            if (T_dependent_q_par)
            {
                kappa_par = spitzer_harm_conductivity(T);
            }
            else
            {
                kappa_par = chi_par / D_0;
            }

            // Calculate div_q_par term using specified method
            if (use_classic_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                ddt(P) += div_q_par_classic(T, kappa_par, B / B_mag);
                q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2);
            }
            else if (use_gunter_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                ddt(P) += div_q_par_gunter(T, kappa_par, B / B_mag);
                q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2); // TODO: This should be calculated using Gunter stencil really
            }
            else if (use_modified_stegmeir_div_q_par)
            {
                ddt(P) += div_q_par_modified_stegmeir(T, kappa_par, B / B_mag);
                q_par = 0.5 * (Q_plus(T, kappa_par, B / B_mag) + Q_minus(T, kappa_par, B / B_mag)) * (B / B_mag);
            }
            else if (use_linetrace_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                // ddt(P) += div_q_par_linetrace(T, kappa_par, B / B_mag);
                // div_q = div_q_par_linetrace2(T, kappa_par, B / B_mag);
                // ddt(P) += div_q;
            }

            // // Calculate q for output
            // // TODO: This should match the method used in div_q calculation
            // q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2);
        }
        if (use_classic_div_q_perp || use_gunter_div_q_perp)
        {
            // kappa_perp = chi_perp / D_0;
            if (use_classic_div_q_perp)
            {
                ddt(P) += div_q_perp_classic(T, chi_perp / D_0, B / B_mag);
            }
            else if (use_gunter_div_q_perp)
            {
                ddt(P) += div_q_perp_gunter(T, chi_perp / D_0, B / B_mag);
            }

            // Calculate q for output
            // TODO: This should match the method used in div_q calculation
            q_perp = -kappa_perp * (Grad(T) - B * (B * Grad(T)) / pow(B_mag, 2));
        }
        if (D_x > 0.0)
        {
            // TODO: I think this should be simple diffusive rather than perp diffusive?
            if (use_classic_div_q_perp)
            {
                ddt(P) += div_q_perp_classic(T, chi_perp_eff / D_0, B / B_mag);
            }
            else if (use_gunter_div_q_perp)
            {
                ddt(P) += div_q_perp_gunter(T, chi_perp_eff / D_0, B / B_mag);
            }

            // Calculate q for output
            q_perp += -(chi_perp_eff / D_0) * (Grad(T) - B * (B * Grad(T)) / pow(B_mag, 2));
        }
    }

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
        if (mesh->StaggerGrids)
        {
            ddt(psi) = -V_dot_Grad(u, psi);
        }
        else
        {
            ddt(psi) = -(DDX(psi) * u.x + u.y * DDY(psi));
        }
    }
    else
    {
        ddt(psi) = 0;
    }
    ddt(psi) += (D_m / D_0) * (D2DX2(psi) + D2DY2(psi));

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
        if (mesh->StaggerGrids)
        {
            ddt(omega) = -V_dot_Grad(u, omega);
        }
        else
        {
            ddt(omega) = -(DDX(omega) * u.x + u.y * DDY(omega));
        }
    }
    else
    {
        ddt(omega) = 0;
    }
    ddt(omega) += (mu / D_0) * (D2DX2(omega) + D2DY2(omega));
    if (include_churn_drive_term)
    {
        ddt(omega) += 2.0 * epsilon * DDY(P, CELL_CENTER, "DEFAULT", "RGN_ALL");
    }
    if (include_mag_restoring_term)
    {
        // Basic approach
        ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL"), CELL_CENTER, "DEFAULT", "RGN_ALL") - DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL"), CELL_CENTER, "DEFAULT", "RGN_ALL"));

        // Using 3rd derivative stencils
        // ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi) * (D3D2XDY(psi) + D3DY3(psi)) - DDY(psi) * (D3D2YDX(psi) + D3DX3(psi)));

        // Using a rotated Laplacian stencil
        // ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi) * DDY(rotated_laplacexy(psi)) - DDY(psi) * DDX(rotated_laplacexy(psi)));
    }

    // Apply ddt = 0 BCs
    // X boundaries
    if (mesh->firstX())
    {
        for (int ix = 0; ix < ngcx_tot; ix++)
        {
            for (int iy = 0; iy < mesh->LocalNy; iy++)
            {
                for (int iz = 0; iz < mesh->LocalNz; iz++)
                {
                    ddt(omega)(ix, iy, iz) = 0.0;
                    ddt(psi)(ix, iy, iz) = 0.0;
                    ddt(P)(ix, iy, iz) = 0.0;
                    if (invert_laplace == false)
                    {
                        ddt(phi)(ix, iy, iz) = 0.0;
                    }
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
                    ddt(omega)(ix, iy, iz) = 0.0;
                    ddt(psi)(ix, iy, iz) = 0.0;
                    ddt(P)(ix, iy, iz) = 0.0;
                    if (invert_laplace == false)
                    {
                        ddt(phi)(ix, iy, iz) = 0.0;
                    }
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
                ddt(omega)(itl.ind, iy, iz) = 0.0;
                ddt(psi)(itl.ind, iy, iz) = 0.0;
                ddt(P)(itl.ind, iy, iz) = 0.0;
                if (invert_laplace == false)
                {
                    ddt(phi)(itl.ind, iy, iz) = 0.0;
                }
            }
        }
    }
    for (itu.first(); !itu.isDone(); itu++)
    {
        // it.ind contains the x index
        for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        {
            for (int iz = 0; iz < mesh->LocalNz; iz++)
            {
                ddt(omega)(itu.ind, iy, iz) = 0.0;
                ddt(psi)(itu.ind, iy, iz) = 0.0;
                ddt(P)(itu.ind, iy, iz) = 0.0;
                if (invert_laplace == false)
                {
                    ddt(phi)(itu.ind, iy, iz) = 0.0;
                }
            }
        }
    }

    return 0;
}