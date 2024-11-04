#ifndef INCLUDE_GUARD_H
#define INCLUDE_GUARD_H

#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/invertable_operator.hxx>
#include <bout/interpolation.hxx>

class Point
{
public:
    float x;
    float y;
};

class ClosestPoint : public Point
{
public:
    float distance;
};

class InterpolationPoint : public ClosestPoint
{
public:
    float parallel_distance;
};

class CellIntersect : public Point
{
public:
    int face; // 0 = west, 1 = north, 2 = east, 3 = south;
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

    // Auxilliary variables
    Field3D phi;
    Vector3D u;   // Advection velocity
    Vector3D e_z; // Unit vector in z direction

    // Heat flow variables
    Vector3D q_par, q_perp;
    Field3D kappa_par, kappa_perp;
    Field3D B_mag, T, lambda_ei, tau_e;
    Field3D div_q;

    // Input Parameters
    BoutReal chi_diff; ///< Isotropic thermal diffusivity [m^2 s^-1]
    BoutReal chi_par;  ///< Parallel thermal diffusivity [m^2 s^-1]
    BoutReal chi_perp; ///< Perpendicular thermal diffusivity [m^2 s^-1]
    BoutReal D_m;      ///< Magnetic diffusivity [m^2 s^-1]
    BoutReal mu;       ///< Vorticity diffusivity [m^2 s^-1]
    BoutReal R_0;      ///< Major radius [m]
    BoutReal a_mid;    ///< Minor radius at midplane [m]
    BoutReal n_sepx;   ///< Electron density at separatrix [m^-3]
    BoutReal T_sepx;   ///< Plasma temperature at separatrix [eV]
    BoutReal B_t0;     ///< Toroidal field strength [T]
    BoutReal B_pmid;   ///< Poloidal field strength [T]
    BoutReal T_down;   ///< Fixed downstream temperature [eV]

    // Other parameters
    BoutReal mu_0;     ///< Vacuum permeability [N A^-2]
    BoutReal e;        ///< Electric charge [C]
    BoutReal m_e;      ///< Electron mass [kg]
    BoutReal eps_0;    ///< Vacuum permittivity [F m^-1]
    BoutReal m_i;      ///< Ion mass [kg]
    BoutReal pi;       ///< Pi
    BoutReal rho;      ///< Plasma mass density [kg m^-3]
    BoutReal P_0;      ///< Pressure normalisation [Pa]
    BoutReal C_s0;     ///< Plasma sound speed at P_0, rho [m s^-1]
    BoutReal t_0;      ///< Time normalisation [s]
    BoutReal D_0;      ///< Diffusivity normalisation [m^2 s^-1]
    BoutReal psi_0;    ///< Poloidal flux normalisation [T m^2]
    BoutReal phi_0;    ///< Phi normalisation
    BoutReal c;        ///< Reference fluid velocity [m s^-1]
    BoutReal epsilon;  ///< Inverse aspect ratio [-]
    BoutReal beta_p;   ///< Poloidal beta [-]
    BoutReal P_grad_0; ///< Vertical pressure gradient normalisation

    // Switches
    bool evolve_pressure;            ///< Evolve plasma pressure
    bool include_mag_restoring_term; ///< Include the poloidal magnetic field restoring term in the vorticity equation
    bool include_churn_drive_term;   ///< Include the churn driving term in the vorticity equation
    bool invert_laplace;             ///< Use Laplace inversion routine to solve phi (if false, will instead solve via a constraint) (TODO: Implement this option)
    bool include_advection;          ///< Use advection terms
    bool fixed_T_down;               ///< Use a constant value for P on downstream boundaries
    bool use_classic_div_q_par;
    bool use_gunter_div_q_par;
    bool use_modified_stegmeir_div_q_par;
    bool use_linetrace_div_q_par;
    bool use_classic_div_q_perp;
    bool use_gunter_div_q_perp;

    // std::unique_ptr<LaplaceXY> phiSolver{nullptr};
    customLaplaceInverter mm;
    bout::inversion::InvertableOperator<Field3D> mySolver;
    const int nits_inv_extra = 0;

    // Methods related to difference heat conduction models
    Field3D div_q_par_classic(const Field3D &T, const BoutReal &K_par, const Vector3D &b);
    Field3D div_q_perp_classic(const Field3D &T, const BoutReal &K_perp, const Vector3D &b);
    Field3D div_q_par_gunter(const Field3D &T, const BoutReal &K_par, const Vector3D &b);
    Field3D div_q_perp_gunter(const Field3D &T, const BoutReal &K_perp, const Vector3D &b);
    std::vector<CellIntersect> get_intersects(const float &xlo, const float &xhi, const float &ylo, const float &yhi, const CellIntersect &P, const float &bx, const float &by);
    CellIntersect get_next_intersect(const float &xlo, const float &xhi, const float &ylo, const float &yhi, const CellIntersect &prev_intersect, const float &bx, const float &by);
    Ind3D increment_cell(const Ind3D &i, const Ind3D &i_prev, const CellIntersect &P_next, const float &dx, const float &dy);
    InterpolationPoint trace_field_lines(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_x_inc, const int &max_y_inc, const int &max_steps, const bool &plus);
    ClosestPoint get_closest_p(const CellIntersect &P, const Point &P0, const float &bx, const float &by);
    Field3D div_q_par_linetrace(const Field3D &u, const BoutReal &K_par, const Vector3D &b);
    Field3D Q_plus(const Field3D &u, const BoutReal &K_par, const Vector3D &b);
    Field3D Q_plus_T(const Field3D &u, const Vector3D &b);
    Field3D Q_minus(const Field3D &u, const BoutReal &K_par, const Vector3D &b);
    Field3D Q_minus_T(const Field3D &u, const Vector3D &b);
    Field3D div_q_par_modified_stegmeir(const Field3D &T, const BoutReal &K_par, const Vector3D &b);

    // Y boundary iterators
    // RangeIterator itl;
    // RangeIterator itu;
    RangeIterator itl = mesh->iterateBndryLowerY();
    RangeIterator itu = mesh->iterateBndryUpperY();

    // Define some extra derivative functions
    Field3D D3DX3(const Field3D &f);
    Field3D D3DY3(const Field3D &f);
    Field3D D3D2YDX(const Field3D &f);
    Field3D D3D2XDY(const Field3D &f);
    Field3D rotated_laplacexy(const Field3D &f);

protected:
    int init(bool restarting) override;
    int rhs(BoutReal) override;
};

#endif
