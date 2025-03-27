#include "header.hxx"

int Churn::rhs(BoutReal UNUSED(t))
{
    mesh->communicate(P, psi);
    // // Calculate B
    B.x = -(1.0 / (1.0 + x_c * epsilon)) * DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B.y = (1.0 / (1.0 + x_c * epsilon)) * DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B_mag = abs(B);

    // Apply upstream P boundary condition
    if (fixed_P_core)
    {
        fixed_P_core_BC(P_core);
    }
    else if (fixed_Q_in)
    {
        dPdy0_BC();
    }
    // else if (par_extrap_P_up)
    // if (par_extrap_P_up)
    // {
    //     debugvar = test_par_extrap_P_up_BC();
    //     // par_extrap_P_up_BC();
    // }

    // Solve phi
    ////////////////////////////////////////////////////////////////////////////
    if (invert_laplace)
    {
        mesh->communicate(omega);
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
        mesh->communicate(omega, phi);
        ddt(phi) = (D2DX2(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(phi, CELL_CENTER, "DEFAULT", "RGN_ALL")) - omega;
    }
    mesh->communicate(phi);

    // Calculate velocity
    // phi.applyBoundary("neumann");
    u = b0 * cross(e_z, Grad(phi));
    u.applyBoundary("dirichlet");

    // Get T
    T = P; // Normalised T = normalised P when rho = const

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
        // TODO: we still have convection across the boundaries even though u should be zero - need to revisit the convection term or calculation of u. Then we can redo some energy analysis stuff
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

            // Calculate div_q_par term using specified method
            if (use_classic_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                ddt(P) += (2.0 / 3.0) * div_q_par_classic(T, kappa_par, B / B_mag);
                q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2);
                // q_par.x = div_q_par_classic(T, kappa_par, B / B_mag);
            }
            else if (use_gunter_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                ddt(P) += (2.0 / 3.0) * div_q_par_gunter(T, kappa_par, B / B_mag);
                q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2); // TODO: This should be calculated using Gunter stencil really
            }
            else if (use_modified_stegmeir_div_q_par)
            {
                Field3D q_par_plus = Q_plus(T, kappa_par, B / B_mag);
                Field3D q_par_minus = Q_minus(T, kappa_par, B / B_mag);
                ddt(P) += -(2.0 / 3.0) * 0.5 * (Q_plus_T(q_par_plus, B / B_mag) + Q_minus_T(q_par_minus, B / B_mag));
                q_par = -0.5 * (q_par_plus + q_par_minus) * (B / B_mag);
                // q_par.x = -0.5 * (Q_plus_T(q_par_plus, B / B_mag) + Q_minus_T(q_par_minus, B / B_mag));

                // ddt(P) += div_q_par_modified_stegmeir(T, kappa_par, B / B_mag);
                // q_par = -0.5 * (Q_plus(T, kappa_par, B / B_mag) + Q_minus(T, kappa_par, B / B_mag)) * (B / B_mag);
                // q_par = (-kappa_par * B * (B * Grad(T)) / pow(B_mag, 2));
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
            if (use_classic_div_q_perp)
            {
                ddt(P) += (2.0 / 3.0) * div_q_perp_classic(T, kappa_perp, B / B_mag);
                // ddt(P) += D2DX2_DIFF(P, kappa_perp) + D2DY2_DIFF(P, kappa_perp);
                q_perp = -kappa_perp * Grad(T);
            }
            else if (use_gunter_div_q_perp)
            {
                ddt(P) += (2.0 / 3.0) * div_q_perp_gunter(T, kappa_perp, B / B_mag);
                q_perp = -kappa_perp * (Grad(T) - (B * (B * Grad(T)) / pow(B_mag, 2)));
            }

            // Calculate q for output
            // TODO: This should match the method used in div_q calculation
            // q_perp = -kappa_perp * (Grad(T) - (B * (B * Grad(T)) / pow(B_mag, 2)));
            // q_perp.x += div_q_perp_classic(T, kappa_perp, B / B_mag);
            // q_perp.x = div_q_perp_classic(T, kappa_perp, B / B_mag);
        }

        // Boundary q_in
        if (fixed_Q_in)
        {
            fixed_Q_in_BC();
        }
        // TODO: Put this in BC module
        // for (itu.first(); !itu.isDone(); itu++)
        // {
        //     // for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        //     for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        //     {
        //         if ((mesh->getGlobalXIndex(itu.ind) == int(mesh->GlobalNx / 2)) || (mesh->getGlobalXIndex(itu.ind) == int(mesh->GlobalNx / 2) + 1))
        //         {
        //             ddt(P)(itu.ind, iy - 1, 0) += q_in / (2.0 * mesh->getCoordinates()->dy(itu.ind, iy, 0));
        //         }
        //     }
        // }
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
    if (include_thermal_force_term)
    {
        // thermal_force_term = (0.71 / B_t0) * grad_par_custom(T, B / B_mag);
        thermal_force_term = 1.71 * delta * (DDX(psi) * DDY(P) - DDY(psi) * DDX(P));
        ddt(psi) -= b0 * thermal_force_term;
    }

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
        ddt(omega) -= cos(alpha_rot) * b0 * 2.0 * epsilon * DDY(P);
        ddt(omega) += sin(alpha_rot) * b0 * 2.0 * epsilon * DDX(P);
    }
    if (include_mag_restoring_term)
    {
        // // Basic approach
        // ddt(omega) -= b0 * (2.0 / (beta_p)) * (DDX(psi) * DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) - DDY(psi) * DDX(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")));

        // Using 3rd derivative stencils
        // ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi) * (D3D2XDY(psi) + D3DY3(psi)) - DDY(psi) * (D3D2YDX(psi) + D3DX3(psi)));
        ddt(omega) += b0 * (2.0 / (beta_p)) * (DDX(psi) * (DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DY3(psi)) - DDY(psi) * (DDX(D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DX3(psi)));

        // Using a rotated Laplacian stencil
        // ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi) * DDY(rotated_laplacexy(psi)) - DDY(psi) * DDX(rotated_laplacexy(psi)));
    }
    if (include_thermal_force_term)
    {
        Field3D lap_P = D2DX2(P, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(P, CELL_CENTER, "DEFAULT", "RGN_ALL");
        ddt(omega) -= 0.5 * delta * b0 * (Laplace(DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL")));
        ddt(omega) -= 0.5 * delta * b0 * (DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL"));
        ddt(omega) -= 0.5 * delta * b0 * (DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(omega, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(omega, CELL_CENTER, "DEFAULT", "RGN_ALL"));
    }

    // Apply ddt = 0 BCs
    ddt0_BCs();

    return 0;
}