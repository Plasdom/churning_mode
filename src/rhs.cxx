#include "header.hxx"

int Churn::rhs(BoutReal t)
{    
    mesh->communicate(P, psi);
    if (evolve_density)
    {
        mesh->communicate(n);
    }
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
    T = P / n; // Normalised T = normalised P when rho = const
    
    // Apply the density source to the upper boundary cells
    if (evolve_density)
    {
        apply_P_core_density_source();
    }

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
                ddt(P) += (2.0 / 3.0) * div_q_par_classic(T, n * kappa_par, B / B_mag);
                q_par = -n * kappa_par * B * (B * Grad(T)) / pow(B_mag, 2);
            }
            else if (use_gunter_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                ddt(P) += (2.0 / 3.0) * div_q_par_gunter(T, n * kappa_par, B / B_mag);
                q_par = -n * kappa_par * B * (B * Grad(T)) / pow(B_mag, 2); // TODO: This should be calculated using Gunter stencil really
            }
            else if (use_modified_stegmeir_div_q_par)
            {
                ddt(P) += (2.0 / 3.0) * div_q_par_modified_stegmeir(T, n * kappa_par, B / B_mag);
                // q_par is calculated and set in in div_q_par_modified_stegmeir()
            }
            else if (use_linetrace_div_q_par)
            {
                // TODO: Implement spatially varying kappa_par
                ddt(P) += (2.0 / 3.0) * div_q_par_linetrace(T, n * kappa_par, B / B_mag);
                // q_par is calculated and set in in div_q_par_linetrace()
            }
        }
        if (use_classic_div_q_perp || use_gunter_div_q_perp)
        {
            if (use_classic_div_q_perp)
            {
                // ddt(P) += (2.0 / 3.0) * div_q_perp_classic(T, kappa_perp, B / B_mag);
                ddt(P) += D2DX2_DIFF(P, n * kappa_perp) + D2DY2_DIFF(P, n * kappa_perp);
                q_perp = - n * kappa_perp * Grad(T);
            }
            else if (use_gunter_div_q_perp)
            {
                ddt(P) += (2.0 / 3.0) * div_q_perp_gunter(T, n * kappa_perp, B / B_mag);
                q_perp = -n * kappa_perp * (Grad(T) - (B * (B * Grad(T)) / pow(B_mag, 2)));
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

        // Density source 
        if (evolve_density)
        {
            ddt(P) += T * density_source * (t_0 / n_sepx);
        }

    }

    // Density equation
    if (evolve_density)
    {
        ddt(n) = density_source * (t_0 / n_sepx);
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

    // Advection
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

    // Vorticity diffusion (kinematic viscosity)
    ddt(omega) += (mu / D_0) * (D2DX2(omega) + D2DY2(omega));
    
    // Curvature drive
    if (include_churn_drive_term)
    {
        if (evolve_density)
        {
            ddt(omega) -= (cos(alpha_rot) * b0 * 2.0 * epsilon * DDY(P)) / n;
            ddt(omega) += (sin(alpha_rot) * b0 * 2.0 * epsilon * DDX(P)) / n;
        }
        else 
        {
            ddt(omega) -= cos(alpha_rot) * b0 * 2.0 * epsilon * DDY(P);
            ddt(omega) += sin(alpha_rot) * b0 * 2.0 * epsilon * DDX(P);
        }
    }
    
    // Line bending
    if (include_mag_restoring_term)
    {
        // Using 3rd derivative stencils
        if (evolve_density)
        {
            ddt(omega) += (1.0/n) * (b0 * (2.0 / (beta_p)) * (DDX(psi) * (DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DY3(psi)) - DDY(psi) * (DDX(D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DX3(psi))));
        }
        else 
        {
            ddt(omega) += b0 * (2.0 / (beta_p)) * (DDX(psi) * (DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DY3(psi)) - DDY(psi) * (DDX(D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DX3(psi)));
        }
    }
    
    // Terms arising from extended Ohm's law (thermal force and parallel pressure gradient)
    if (include_thermal_force_term)
    {
        if (evolve_density)
        {
            Field3D lap_P = D2DX2(P, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(P, CELL_CENTER, "DEFAULT", "RGN_ALL");
            ddt(omega) -= (1.0/n) * (0.5 * delta * b0 * (Laplace(DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL"))));
            ddt(omega) -= (1.0/n) * (0.5 * delta * b0 * (DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL")));
            ddt(omega) -= (1.0/n) * (0.5 * delta * b0 * (DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(omega, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(omega, CELL_CENTER, "DEFAULT", "RGN_ALL")));
        }
        else 
        {
            Field3D lap_P = D2DX2(P, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(P, CELL_CENTER, "DEFAULT", "RGN_ALL");
            ddt(omega) -= 0.5 * delta * b0 * (Laplace(DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL")));
            ddt(omega) -= 0.5 * delta * b0 * (DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL"));
            ddt(omega) -= 0.5 * delta * b0 * (DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(omega, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(omega, CELL_CENTER, "DEFAULT", "RGN_ALL"));
        }
    }

    // Particle source injected with zero momentum
    if (evolve_density)
    {
        ddt(omega) -= density_source * (t_0 / n_sepx) * omega / n; 
    }

    // Apply ddt = 0 BCs
    ddt0_BCs();

    return 0;
}