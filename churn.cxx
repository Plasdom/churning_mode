#include <bout/derivs.hxx>           // To use DDZ()
#include <bout/physicsmodel.hxx>     // Commonly used BOUT++ components
#include "bout/invert/laplacexy.hxx" // Laplacian inversion

/// Churning mode model
///
///
class Churn : public PhysicsModel
{
private:
  // Evolving variables
  Field3D P, psi, omega; ///< Pressure, poloidal magnetic flux and vorticity

  // Auxilliary variables
  Field3D phi, u_x, u_y; // TODO: Use Vector2D object for u
  // Field2D phi2, phi2d, omega2d;
  // Field3D u_x_stgd, u_y_stgd;
  // Field3D u_x_cntr, u_y_cntr;

  // Input Parameters
  BoutReal chi;    ///< Thermal diffusivity [m^2 s^-1]
  BoutReal D_m;    ///< Magnetic diffusivity [m^2 s^-1]
  BoutReal mu;     ///< Vorticity diffusivity [m^2 s^-1]
  BoutReal R_0;    ///< Major radius [m]
  BoutReal a_mid;  ///< Minor radius at midplane [m]
  BoutReal n_sepx; ///< Electron density at separatrix [m^-3]
  BoutReal T_sepx; ///< Plasma temperature at separatrix [eV]
  BoutReal B_t0;   ///< Toroidal field strength [T]
  BoutReal B_pmid; ///< Poloidal field strength [T]

  // Other parameters
  BoutReal mu_0;    ///< Vacuum permeability [N A^-2]
  BoutReal e;       ///< Electric charge [C]
  BoutReal m_e;     ///< Electron mass [kg]
  BoutReal m_i;     ///< Ion mass [kg]
  BoutReal pi;      ///< Pi
  BoutReal rho;     ///< Plasma mass density [kg m^-3]
  BoutReal P_0;     ///< Pressure normalisation [Pa]
  BoutReal C_s0;    ///< Plasma sound speed at P_0, rho [m s^-1]
  BoutReal t_0;     ///< Time normalisation [s]
  BoutReal D_0;     ///< Diffusivity normalisation [m^2 s^-1]
  BoutReal psi_0;   ///< Poloidal flux normalisation [T m^2]
  BoutReal phi_0;   ///< Phi normalisation
  BoutReal c;       ///< Reference fluid velocity [m s^-1]
  BoutReal epsilon; ///< Inverse aspect ratio [-]
  BoutReal beta_p;  ///< Poloidal beta [-]

  // Switches
  bool evolve_pressure;            ///< Evolve plasma pressure
  bool include_mag_restoring_term; ///< Include the poloidal magnetic field restoring term in the vorticity equation
  bool include_churn_drive_term;   ///< Include the churn driving term in the vorticity equation

  // std::unique_ptr<LaplaceXY> phiSolver{nullptr};

protected:
  int init(bool UNUSED(restarting)) // TODO: Use the restart flag
  {

    /******************Reading options *****************/

    auto &globalOptions = Options::root();
    auto &options = globalOptions["model"];

    // Load system parameters
    chi = options["chi"].doc("Thermal diffusivity").withDefault(0.0);
    D_m = options["D_m"].doc("Magnetic diffusivity").withDefault(0.0);
    mu = options["mu"].doc("Kinematic viscosity").withDefault(0.0);
    R_0 = options["R_0"].doc("Major radius [m]").withDefault(1.5);
    a_mid = options["a_mid"].doc("Minor radius at outer midplane [m]").withDefault(0.6);
    n_sepx = options["n_sepx"].doc("Electron density at separatrix [m^-3]").withDefault(1.0e19);
    T_sepx = options["T_sepx"].doc("Plasma temperature at separatrix [eV]").withDefault(100);
    B_t0 = options["B_t0"].doc("Toroidal field strength [T]").withDefault(2);
    B_pmid = options["B_pmid"].doc("Poloidal field strength at outer midplane [T]").withDefault(0.25);

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

    // Constants
    e = options["e"].withDefault(1.602e-19);
    m_i = options["m_i"].withDefault(2 * 1.667e-27);
    m_e = options["m_e"].withDefault(9.11e-31);
    mu_0 = options["mu_0"].withDefault(1.256637e-6);
    pi = 3.14159;

    // Get normalisation and derived parameters
    c = 1.0;
    rho = (m_i + m_e) * n_sepx;
    P_0 = e * n_sepx * T_sepx;
    C_s0 = sqrt(P_0 / rho);
    t_0 = a_mid / C_s0;
    D_0 = a_mid * C_s0;
    psi_0 = B_pmid * R_0 * a_mid;
    phi_0 = pow(C_s0, 2) * B_t0 * t_0 / c;
    epsilon = a_mid / R_0;
    beta_p = mu_0 * 2 * P_0 / pow(B_pmid, 2); // Maxim I think used this formula, although paper says beta_p = mu_0 * 8 * pi * P_0 / pow(B_pmid, 2)

    // phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
    // phiSolver->setCoefs(1.0, 0.0);
    phi = 0.0; // Starting guess for first solve (if iterative)
    // phi2 = 0.0;
    // u_x = 0.0;
    // u_y = 0.0;
    phi.setBoundary("phi"); // TODO: Remove this and line above?
    // phi2.setBoundary("phi2");
    // u_x.setLocation(CELL_XLOW);
    // u_y.setLocation(CELL_YLOW);
    // u_x_stgd.setLocation(CELL_XLOW);
    // u_y_stgd.setLocation(CELL_YLOW);
    // u_x_cntr.setLocation(CELL_CENTRE);
    // u_y_cntr.setLocation(CELL_CENTRE);

    /************ Tell BOUT++ what to solve ************/

    // omega.setLocation(CELL_YLOW);
    SOLVE_FOR(P, psi, omega, phi);

    // Output flow velocity
    SAVE_REPEAT(u_x, u_y);
    // SAVE_REPEAT(u_x, u_y, phi2);

    // Output constants, input options and derived parameters
    SAVE_ONCE(e, m_i, m_e, chi, D_m, mu, epsilon, beta_p, rho, P_0);
    SAVE_ONCE(C_s0, t_0, D_0, psi_0, phi_0, R_0, a_mid, n_sepx);
    SAVE_ONCE(T_sepx, B_t0, B_pmid, evolve_pressure, include_churn_drive_term, include_mag_restoring_term);

    Coordinates *coord = mesh->getCoordinates();

    // generate coordinate system
    // coord->Bxy = 1.0; // TODO: Use B_t here?
    coord->g11 = 1.0;
    coord->g22 = 1.0;
    coord->g33 = 1.0;
    coord->g12 = 0.0;
    coord->g13 = 0.0;
    coord->g23 = 0.0;

    return 0;
  }

  int rhs(BoutReal UNUSED(t))
  {

    // Run communications
    ////////////////////////////////////////////////////////////////////////////
    // mesh->communicate(P, psi, omega, phi, phi2);
    mesh->communicate(P, psi, omega, phi);

    // Invert div(n grad(phi)) = n Delp_perp^2(phi) = omega
    ////////////////////////////////////////////////////////////////////////////

    ddt(phi) = Laplace(phi) - omega;
    // omega2d = DC(omega);
    // phi2.applyBoundary();
    // phi2 = phiSolver->solve(omega2d, phi2);
    // Calculate u_x and u_y components
    // u = Grad(phi);
    u_x = DDY(phi);
    u_y = DDX(phi);
    // u_x = DDY(phi2);
    // u_y = DDX(phi2);
    // u_x_stgd = DDY(phi);
    // u_y_stgd = DDX(phi);
    // u_x_cntr = interp_to(u_x_stgd, CELL_CENTRE);
    // u_y_cntr = interp_to(u_y_stgd, CELL_CENTRE);

    mesh->communicate(u_x, u_y);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
      ddt(P) = -(DDX(P) * u_x - u_y * DDY(P));
      // ddt(P) = 0;
      // ddt(P) = -bracket(P, phi);
      ddt(P) += (chi / D_0) * Laplace(P);
    }

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(psi) = -(DDX(psi) * u_x - u_y * DDY(psi));
    // ddt(psi) = 0;
    // ddt(psi) = -bracket(psi, phi);
    ddt(psi) += (D_m / D_0) * Laplace(psi);

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(omega) = -(DDX(omega) * u_x - u_y * DDY(omega));
    // ddt(omega) = 0;
    // ddt(omega) = -bracket(omega, phi);
    ddt(omega) += (mu / D_0) * Laplace(omega);
    if (include_churn_drive_term)
    {
      ddt(omega) += 2 * epsilon * DDY(P);
    }
    if (include_mag_restoring_term)
    {
      // ddt(omega) += (2 / beta_p) * (DDX(psi) * DDY(D2DX2(psi) + D2DY2(psi)) - DDX(D2DX2(psi) + D2DY2(psi)) * DDY(psi));
      ddt(omega) += (2 / beta_p) * (DDX(psi) * DDY(Laplace(psi)) - DDX(Laplace(psi)) * DDY(psi));
      // ddt(omega) += (2 / beta_p) * bracket(psi, Laplace(psi));
    }

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
