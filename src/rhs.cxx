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
    //TODO: Check switch logic here, what happens if more than one of these enabled?
    if (fixed_P_core)
    {
        fixed_P_core_BC(P_core, B / B_mag);
    }

    // Solve phi
    ////////////////////////////////////////////////////////////////////////////
    if (phi_parallel_neumann_yup)
    {
        parallel_neumann_yup(phi, B/B_mag);
    }
    if (invert_laplace)
    {
        //TODO: Work out why we can't use mySolver.invert(omega - delta*lap_P) here if using extended model
        mesh->communicate(omega);
        if(electrostatic)
        {
            phi = 0.0;
        }
        phi = mySolver.invert(omega, phi);
        if (electrostatic)
        {
            try
            {
                for (int i = 0; i < 3; i++)
                {
                    phi = mySolver.invert(omega, phi);
                    mesh->communicate(phi);
                }
            }
            catch (BoutException &e)
            {
            };
        }

    }
    else
    {
        //TODO: If we can get it working in Laplace inversion, should add Laplace(P) correction to phi here
        mesh->communicate(omega, phi);
        ddt(phi) = (phi_constraint_lambda_1/D_0) * ((D2DX2(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(phi, CELL_CENTER, "DEFAULT", "RGN_ALL"))/phi_constraint_lambda_2 - omega);
        
        // // Avoid solving the vorticity equation
        // Field3D curv_drive = (-cos(alpha_rot) * b0 * 2.0 * epsilon * DDY(P)) + (sin(alpha_rot) * b0 * 2.0 * epsilon * DDX(P));
        // ddt(phi) = 1e0*(b0 * (2.0 / (beta_p)) * (b0 * div_q_par_modified_stegmeir(phi, 1/eta, B/B_mag) - b0 * 1.71 * delta * div_q_par_modified_stegmeir(P, 1/eta, B/B_mag)) - curv_drive);
        // // Field3D lap_phi = D2DX2(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(phi, CELL_CENTER, "DEFAULT", "RGN_ALL");
        // // ddt(phi) -= 1e0*(DDX(phi)*DDY(lap_phi) - DDX(lap_phi)*DDY(phi));
    }
    mesh->communicate(phi);

    // Calculate velocity
    u = b0 * cross(e_z, Grad(phi/phi_constraint_lambda_2));
    if (phi_BC_width == 0)
    {
        u.applyBoundary("dirichlet");
    }

    T = P; // Assume normalised n = 1 if density is not evolved

    // Calculate resistivity 
    if (use_spitzer_resistivity)
    {
        BoutReal lambda_ei = 10.0;
        BoutReal T_capped, T_max_ev, T_min_ev;
        T_min_ev = 1.0;
        T_max_ev = 1.0e4; // Required to relax the equations

        for (auto i: eta)
        {
            T_capped = std::min(std::max(T[i]*T_sepx,T_min_ev),T_max_ev);
            eta[i] = 1.0e-4*lambda_ei*pow(T_capped,-3.0/2.0);
        }
        eta = eta * eta_0;
    }

    if (electrostatic){
        J = -b0 * (DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL")) / eta;
        if (include_thermal_force_term)
        {
            J += 1.71 * delta * b0 * (DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(P, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL")) / eta;
        }
    }
    else 
    {
        if (use_rotated_laplace_cur){
            J = -rotated_laplacexy(psi);
        }
        else {
            J = -(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL"));
        }
    }

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
        // TODO: we still have convection across the boundaries even though u should be zero - need to revisit the convection term or calculation of u. Then we can redo some energy analysis stuff
        if (include_advection)
        {
            ddt(P) = -V_dot_Grad(u, P); 
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
                ddt(P) += (2.0 / 3.0) * div_q_par_classic(T, kappa_par, B / B_mag);
                q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2);

            }
            else if (use_gunter_div_q_par)
            {
                ddt(P) += (2.0 / 3.0) * div_q_par_gunter(T, kappa_par, B / B_mag);
                q_par = -kappa_par * B * (B * Grad(T)) / pow(B_mag, 2); // TODO: This should be calculated using Gunter stencil really
            }
            else if (use_modified_stegmeir_div_q_par)
            {
                ddt(P) += (2.0 / 3.0) * div_q_par_modified_stegmeir(T, kappa_par, B / B_mag); // q_par is calculated and set in in div_q_par_modified_stegmeir()
            }
            else if (use_linetrace_div_q_par)
            {
                ddt(P) += (2.0 / 3.0) * div_q_par_linetrace(T, kappa_par, B / B_mag); // q_par is calculated and set in in div_q_par_linetrace() 
            }

            // Find the heat flux across the boundaries
            q_out = calculate_q_out(T, kappa_par, kappa_perp, B/B_mag);
            q_out_conv = calculate_q_out_conv(P, u);
        }
        if (use_classic_div_q_perp || use_gunter_div_q_perp)
        {
            if (use_classic_div_q_perp)
            {
                ddt(P) += (2.0 / 3.0) * div_q_perp_classic(T, kappa_perp, B / B_mag);
                // ddt(P) += (2.0/3.0) * D2DX2_DIFF(P, kappa_perp) + D2DY2_DIFF(P, kappa_perp);
                // ddt(P) += (2.0 / 3.0) * kappa_perp * Laplace(T);
                q_perp = -kappa_perp * (Grad(T) - ((Grad(T) * (B/B_mag)) * (B/B_mag)));
                
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

        // Add resistive heating terms 
        if (include_resistive_heating)
        {
            ddt(P) += (2.0/3.0) * (2.0 * eta / beta_p) * pow(J,2.0);
            ddt(P) -= b0 * (2.0/3.0) * (4.0 * 0.71 * delta / beta_p) * J * (DDX(psi) * DDY(P) - DDY(psi) * DDX(P));
        }

        // // Add resistive convection term 
        // if (include_thermal_force_term)
        // {
        //     ddt(P) += 2.0 * eta * nu * (pow(DDX(P),2.0) + pow(DDY(P),2.0));
        // }

        // Boundary q_in
        if (fixed_Q_in)
        {
            fixed_Q_in_BC();
        }

    }

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    if (electrostatic)
    {
        ddt(psi) = 0.0;
    }
    else 
    {
        if (include_advection)
        {
            ddt(psi) = -V_dot_Grad(u, psi);
        }
        else
        {
            ddt(psi) = 0.0;
        }
        // Magnetic diffusivity / resistivity
        ddt(psi) -= eta * J;
        if (include_thermal_force_term)
        {
            // thermal_force_term = (0.71 / B_t0) * grad_par_custom(T, B / B_mag);
            thermal_force_term = -1.71 * delta * (DDX(psi) * DDY(P) - DDY(psi) * DDX(P));
            ddt(psi) += b0 * thermal_force_term;
        }

        // Magnetic hyperdiffusion / hyper-resistivity
        if (hyperres > 0.0)
        {
            ddt(psi) += (hyperres / (pow(a_mid,6)/t_0)) * (D4DX4(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D4DY4(D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")));
        }
    }
    
    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    // Advection
    if (include_advection)
    {
        ddt(omega) = -V_dot_Grad(u, omega);
    }
    else
    {
        ddt(omega) = 0;
    }

    // Vorticity diffusion (kinematic viscosity)
    ddt(omega) += (mu / D_0) * (D2DX2(omega) + D2DY2(omega));

    // Vorticity hyperdiffusion / hyper-viscosity 
    if (hypervisc > 0.0){
        ddt(omega) += (hypervisc / (pow(a_mid,6)/t_0)) * (D4DX4(D2DX2(omega, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D4DY4(D2DY2(omega, CELL_CENTER, "DEFAULT", "RGN_ALL")));
    }
    
    // Curvature drive
    if (include_churn_drive_term)
    {
        Field3D curv_drive = (-cos(alpha_rot) * b0 * 2.0 * epsilon * DDY(P)) + (sin(alpha_rot) * b0 * 2.0 * epsilon * DDX(P));
        ddt(omega) += curv_drive;
    }
    
    // Line bending
    if (include_mag_restoring_term)
    {
        // ddt(omega) += b0 * (2.0 / (beta_p)) * (DDX(psi) * (DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DY3(psi)) - DDY(psi) * (DDX(D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + D3DX3(psi)));
        if (electrostatic)
        {
            if (include_thermal_force_term)
            {
                ddt(omega) += -b0 * (2.0 / (beta_p)) * (b0 * div_q_par_modified_stegmeir(phi/phi_constraint_lambda_2 - 1.71 * delta * P, 1/eta, B/B_mag, false));
                // ddt(omega) += -b0 * (2.0 / (beta_p)) * (b0 * div_q_par_gunter(phi - 1.71 * delta * P, 1/eta, B/B_mag));
            }
            else 
            {
                ddt(omega) += -b0 * (2.0 / (beta_p)) * (b0 * div_q_par_modified_stegmeir(phi/phi_constraint_lambda_2, 1/eta, B/B_mag, false));
                // ddt(omega) += -b0 * (2.0 / (beta_p)) * (b0 * div_q_par_gunter(phi, 1/eta, B/B_mag));
            }
        }
        else 
        {
            ddt(omega) += -b0 * (2.0 / (beta_p)) * (DDX(psi) * DDY(J) - DDY(psi) * DDX(J));
        }
    }

    // // Diamagnetic convection of vorticity
    // lap_P = D2DX2(P, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(P, CELL_CENTER, "DEFAULT", "RGN_ALL");
    // ddt(omega) -= 0.5 * delta * b0 * (Laplace(DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL")));
    // ddt(omega) -= 0.5 * delta * b0 * (DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(lap_P, CELL_CENTER, "DEFAULT", "RGN_ALL"));
    // ddt(omega) -= 0.5 * delta * b0 * (DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(omega, CELL_CENTER, "DEFAULT", "RGN_ALL") - DDX(P, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(omega, CELL_CENTER, "DEFAULT", "RGN_ALL"));

    // phi = 1.71*delta*P;

    //     // // Resistive contribution to vorticity convection
    //     // lap_phi = D2DX2(phi) + D2DY2(phi);
        // ddt(omega) -= 2.0 * eta * nu * (lap_phi * lap_P + (DDX(lap_P * DDX(phi, CELL_CENTER, "DEFAULT", "RGN_ALL")) + DDY(lap_P * DDY(phi, CELL_CENTER, "DEFAULT", "RGN_ALL"))));
    // }
    return 0;
}