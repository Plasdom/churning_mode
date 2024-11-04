#include "header.hxx"

int Churn::rhs(BoutReal UNUSED(t))
{

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
    B.x = -DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B.y = DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B_mag = abs(B);
    if (use_sk9_anis_diffop)
    {
        BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
        {
            theta_m[i] = atan2(sqrt(pow(B.x[i], 2.0) + pow(B.y[i], 2.0)), B.z[i]);
            phi_m[i] = atan2(B.y[i], B.x[i]);
        }
    }

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
        if (chi_diff > 0.0)
        {
            ddt(P) += (chi_diff / D_0) * (D2DX2(P) + D2DY2(P));
        }
        else if ((chi_par > 0.0) || (chi_perp > 0.0))
        {
            // TODO: Add T-dependent q_par
            if (use_symmetric_div_q_stencil)
            {
                ddt(P) += div_q_symmetric(T, chi_par / D_0, chi_perp / D_0, B / B_mag);
            }
            else if (use_sk9_anis_diffop)
            {
                ddt(P) += (chi_par / D_0) * Anis_Diff_SK9(P,
                                                          pow(sin(theta_m), 2.0) * pow(cos(phi_m), 2.0),
                                                          pow(sin(theta_m), 2.0) * pow(sin(phi_m), 2.0),
                                                          pow(sin(theta_m), 2.0) * cos(phi_m) * sin(phi_m));
                ddt(P) += (chi_perp / D_0) * Anis_Diff_SK9(P,
                                                           pow(cos(theta_m), 2.0) * pow(cos(phi_m), 2.0) + pow(sin(phi_m), 2.0),
                                                           pow(cos(theta_m), 2.0) * pow(sin(phi_m), 2.0) + pow(cos(phi_m), 2.0),
                                                           (-sin(phi_m) * cos(phi_m) * (pow(cos(theta_m), 2.0) + 1.0)));
            }
            else
            {
                // ddt(P) += (chi_par / D_0) * (DDX(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(B.y * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")) / pow(B_mag, 2);
                ddt(P) += (chi_perp / D_0) * (D2DX2(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(T, CELL_CENTER, "DEFAULT", "RGN_ALL") - (DDX(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")) / pow(B_mag, 2));
                // div_q = (chi_par / D_0) * (DDX(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(B.y * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")) / pow(B_mag, 2);
                div_q = stegmeir_div_q(T, chi_par / D_0, chi_perp / D_0, B / B_mag);
                ddt(P) += div_q;
                // div_q = Q_plus(T, chi_par / D_0, B / B_mag);
                // div_q = Q_minus(T, chi_par / D_0, B / B_mag);

                // div_q = Q_linetrace(T, chi_par / D_0, B / B_mag);
                // ddt(P) += div_q;
            }
            // Calculate q for output
            q_par = -(chi_par / D_0) * B * (B * Grad(T)) / pow(B_mag, 2);
            q_perp = -(chi_perp / D_0) * (Grad(T) - B * (B * Grad(T)) / pow(B_mag, 2));
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