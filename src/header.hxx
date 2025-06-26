#ifndef INCLUDE_GUARD_H
#define INCLUDE_GUARD_H

#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/invertable_operator.hxx>
#include <bout/interpolation.hxx>

struct Point
{
    double x;
    double y;
};

struct ClosestPoint
{
    double x;
    double y;
    double distance;
};

struct InterpolationPoint
{
    double x;
    double y;
    double distance;
    double parallel_distance;
};

struct CellIntersect
{
    double x;
    double y;
    int face; // 0 = west, 1 = north, 2 = east, 3 = south;
};

struct TwoIntersects
{
    CellIntersect first;
    CellIntersect second;
};

struct customLaplaceInverter
{
    BoutReal D = 1.0, A = 0.0;
    int ngcx_tot, ngcy_tot, nx_tot, ny_tot, nz_tot;

    Field3D operator()(const Field3D &input);
};

/// Churning mode model
///
///
class Churn : public PhysicsModel
{
public:
    int ngcx = (mesh->GlobalNx - mesh->GlobalNxNoBoundaries) / 2;
    int ngcy = (mesh->GlobalNy - mesh->GlobalNyNoBoundaries) / 2;

    int ngc_extra = 0;
    int nx_tot = mesh->GlobalNx, ny_tot = mesh->GlobalNy, nz_tot = mesh->GlobalNz;
    int ngcx_tot = ngcx + ngc_extra, ngcy_tot = ngcy + ngc_extra;

private:
    // Evolving variables
    Field3D P, psi, omega; ///< Pressure, poloidal magnetic flux and vorticity
    Vector3D B;
    Field3D n; // Density

    // Auxilliary variables
    Field3D phi;
    Vector3D u;   // Advection velocity
    Vector3D e_z; // Unit vector in z direction

    // Heat flow variables
    Vector3D q_par, q_perp;
    Field3D kappa_par, kappa_perp;
    Field3D B_mag, T;
    // Field3D div_q, div_q2;
    Field3D T_init;
    Field3D chi_perp_eff;
    Field3D x_c, y_c;
    // Field2D q_in_yup;
    Field3D thermal_force_term;
    Field3D lap_P;
    Field3D lap_phi;
    Field3D eta;
    Field3D J_par;
    Field3D q_out;

    // Input Parameters
    BoutReal chi_diff;       ///< Isotropic thermal diffusivity [m^2 s^-1]
    BoutReal chi_par;        ///< Parallel thermal diffusivity [m^2 s^-1]
    BoutReal chi_perp;       ///< Perpendicular thermal diffusivity [m^2 s^-1]
    BoutReal D_m;            ///< Magnetic diffusivity [m^2 s^-1]
    BoutReal mu;             ///< Vorticity diffusivity [m^2 s^-1]
    BoutReal R_0;            ///< Major radius [m]
    BoutReal a_mid;          ///< Minor radius at midplane [m]
    BoutReal n_sepx;         ///< Electron density at separatrix [m^-3]
    BoutReal T_sepx;         ///< Plasma temperature at separatrix [eV]
    BoutReal B_t0;           ///< Toroidal field strength [T]
    BoutReal B_pmid;         ///< Poloidal field strength [T]
    BoutReal T_down;         ///< Fixed downstream temperature [eV]
    BoutReal D_x;            ///< Peak of additional perpendicular diffusion coefficient [m^2/s]
    BoutReal x_1;            ///< x-coordinate of first X-point (centred on the middle of the simulation domain) [a_mid]
    BoutReal x_2;            ///< x-coordinate of second X-point (centred on the middle of the simulation domain) [a_mid]
    BoutReal y_1;            ///< y-coordinate of first X-point (centred on the middle of the simulation domain) [a_mid]
    BoutReal y_2;            ///< y-coordinate of second X-point (centred on the middle of the simulation domain) [a_mid]
    BoutReal r_star;         ///< Radius of the additional mixing zone [a_mid]
    BoutReal P_core;         ///< Pressure at core boundary [P_0]
    BoutReal lambda_SOL_rho; ///< SOL width parameter in units of normalised flux coordinate
    BoutReal Q_in;           ///< Inpuw power to top of domain [MW]
    BoutReal psi_bndry_P_core_BC; ///< When fixed_P_core=true, set P_core to the specified value within field lines where psi is greater than this value
    BoutReal alpha_rot;           ///< Rotation angle of initial poloidal flux
    BoutReal density_source; ///< Volumetric particle source [m^3 s^-1] if performing a density ramp

    // Other parameters
    BoutReal mu_0;        ///< Vacuum permeability [N A^-2]
    BoutReal e;           ///< Electric charge [C]
    BoutReal m_e;         ///< Electron mass [kg]
    BoutReal eps_0;       ///< Vacuum permittivity [F m^-1]
    BoutReal m_i;         ///< Ion mass [kg]
    BoutReal pi;          ///< Pi
    BoutReal rho;         ///< Plasma mass density [kg m^-3]
    BoutReal P_0;         ///< Pressure normalisation [Pa]
    BoutReal C_s0;        ///< Plasma sound speed at P_0, rho [m s^-1]
    BoutReal t_0;         ///< Time normalisation [s]
    BoutReal D_0;         ///< Diffusivity normalisation [m^2 s^-1]
    BoutReal psi_0;       ///< Poloidal flux normalisation [T m^2]
    BoutReal phi_0;       ///< Phi normalisation
    BoutReal c;           ///< Reference fluid velocity [m s^-1]
    BoutReal epsilon;     ///< Inverse aspect ratio [-]
    BoutReal beta_p;      ///< Poloidal beta [-]
    BoutReal P_grad_0;    ///< Vertical pressure gradient normalisation
    BoutReal boltzmann_k; ///< Boltzmann's constant
    BoutReal q_in;        ///< Heat flux into domain if fixed_Q_in option is true
    BoutReal num_Q_in_cells; ///< Number of cells over which to distribute Q_in
    BoutReal alpha_fl;       ///< Flux limiter
    BoutReal Omega_i0;       ///< Ion cyclotron frequency
    BoutReal delta;          ///< Parameter related to strength of thermal force and grad p terms
    BoutReal nu;          ///< Constant in front of resisitive terms, nu = C_s0^2 / v_A^2
    BoutReal eta_0;          ///< Constant in front of resistivity, eta_0 = t_0 * c^2 / (4.0 * pi * a_mid^2)
    // BoutReal thermal_force_b0_factor; ///< b0 factor to apply to thermal force terms (analogous to UEDGE parameter bbb.b)
    double b0;               ///< 1 if toroidal field is in +z direction, -1 if in -z direction

    // Switches
    bool evolve_pressure;            ///< Evolve plasma pressure
    bool evolve_density;            ///< Evolve plasma density
    bool include_mag_restoring_term; ///< Include the poloidal magnetic field restoring term in the vorticity equation
    bool include_churn_drive_term;   ///< Include the churn driving term in the vorticity equation
    bool include_thermal_force_term; ///< Include the thermal force term in the psi equation
    bool invert_laplace;             ///< Use Laplace inversion routine to solve phi (if false, will instead solve via a constraint) (TODO: Implement this option)
    bool include_advection;          ///< Use advection terms
    bool fixed_T_down;               ///< Use a constant value for P on downstream boundaries
    bool fixed_P_core;               ///< Fix upstream boundary condition in the core (i.e. within the separatrix defined by psi)
    bool use_classic_div_q_par;
    bool use_gunter_div_q_par;
    bool use_modified_stegmeir_div_q_par;
    bool use_linetrace_div_q_par;
    bool use_classic_div_q_perp;
    bool use_gunter_div_q_perp;
    bool T_dependent_q_par;
    bool fixed_Q_in;
    bool use_flux_limiter;
    bool disable_qin_outside_core;
    bool electrostatic;             ///< Use electrostatic model
    bool use_spitzer_resistivity;   ///< Use Spitzer values for resistivity as opposed to spatially constant value. If false, D_m will be used.
    bool include_resistive_heating; ///< Include the (parallel) resistive heating term in the pressure equation

    // std::unique_ptr<LaplaceXY> phiSolver{nullptr};
    customLaplaceInverter mm;
    bout::inversion::InvertableOperator<Field3D> mySolver;
    const int nits_inv_extra = 0;

    // Methods related to difference heat conduction models
    Field3D div_q_par_classic(const Field3D &T, const Field3D &K_par, const Vector3D &b);
    Field3D div_q_perp_classic(const Field3D &T, const Field3D &K_perp, const Vector3D &b);
    Field3D div_q_par_gunter(const Field3D &T, const Field3D &K_par, const Vector3D &b);
    Field3D div_q_perp_gunter(const Field3D &T, const Field3D &K_perp, const Vector3D &b);
    TwoIntersects get_intersects(const double &xlo, const double &xhi, const double &ylo, const double &yhi, const CellIntersect &P, const double &bx, const double &by);
    CellIntersect get_next_intersect(const double &xlo, const double &xhi, const double &ylo, const double &yhi, const CellIntersect &prev_intersect, const double &bx, const double &by);
    Ind3D increment_cell(const Ind3D &i, const Ind3D &i_prev, const CellIntersect &P_next, const double &dx, const double &dy);
    Ind3D increment_cell_2(const Ind3D &i, const Ind3D &i_prev, const int &face);
    InterpolationPoint trace_field_lines(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_x_inc, const int &max_y_inc, const int &max_steps, const bool &plus);
    InterpolationPoint trace_field_lines_2(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_steps, const bool &plus);
    ClosestPoint get_closest_p(const CellIntersect &P, const Point &P0, const double &bx, const double &by);
    Field3D div_q_par_linetrace(const Field3D &u, const Field3D &K_par, const Vector3D &b);
    Field3D div_q_par_linetrace2(const Field3D &u, const Field3D &K_par, const Vector3D &b);
    Field3D Q_plus(const Field3D &u, const BoutReal &K_par, const Vector3D &b);
    Field3D Q_plus(const Field3D &u, const Field3D &K_par, const Vector3D &b);
    Field3D Q_plus_T(const Field3D &u, const Vector3D &b);
    Field3D Q_minus(const Field3D &u, const BoutReal &K_par, const Vector3D &b);
    Field3D Q_minus(const Field3D &u, const Field3D &K_par, const Vector3D &b);
    Field3D Q_minus_T(const Field3D &u, const Vector3D &b);
    Field3D div_q_par_modified_stegmeir(const Field3D &T, const Field3D &K_par, const Vector3D &b);
    Field3D div_q_par_modified_stegmeir_efficient(const Field3D &T, const Field3D &K_par, const Vector3D &b);
    Field3D spitzer_harm_conductivity(const Field3D &T, const BoutReal &Te_limit_ev_low = 10.0, const BoutReal &Te_limit_ev_high = 500.0);
    Field3D calculate_q_out(const Field3D &T, const Field3D &kappa_par, const Field3D &kappa_perp, const Vector3D &b);

    // Boundary conditions
    RangeIterator itl = mesh->iterateBndryLowerY();
    RangeIterator itu = mesh->iterateBndryUpperY();
    void fixed_P_core_BC(const BoutReal &P_core_set, const Vector3D &b);
    void fixed_Q_in_BC();
    void ddt0_BCs();
    void dPdy0_BC();
    void dPdy0_BC_outside_core();
    void apply_P_core_density_source();
    void parallel_neumann_yup(const Vector3D &b, const bool &apply_outside_core_only = false);
    // Field3D test_par_extrap_P_up_BC();
    // void par_extrap_P_up_BC();

    // Define some extra derivative functions
    Field3D D3DX3(const Field3D &f);
    Field3D D3DY3(const Field3D &f);
    Field3D D3D2YDX(const Field3D &f);
    Field3D D3D2XDY(const Field3D &f);
    Field3D rotated_laplacexy(const Field3D &f);
    Field3D D2DX2_DIFF(const Field3D &f, const Field3D &A);
    Field3D D2DY2_DIFF(const Field3D &f, const Field3D &A);
    Field3D grad_par_custom(const Field3D &u, const Vector3D &b);
    Field3D V_dot_grad_no_bndry_flow(const Vector3D &v, const Field3D &f);

protected:
    int init(bool restarting) override;
    int rhs(BoutReal) override;
};

#endif
