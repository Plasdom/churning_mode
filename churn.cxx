#include <bout/derivs.hxx>         // To use DDZ()
#include <bout/invert_laplace.hxx> // Laplacian inversion
#include <bout/physicsmodel.hxx>   // Commonly used BOUT++ components

/// Churning mode model
///
///
class Churn : public PhysicsModel
{
private:
  // Evolving variables
  Field3D P, psi, omega; ///< Pressure, poloidal magnetic flux and vorticity

  // Auxilliary variables
  Field3D phi, u_x, u_z; // TODO: Use Vector2D object for u

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
  BoutReal mu_0;     ///< Vacuum permeability [N A^-2]
  BoutReal e;        ///< Electric charge [C]
  BoutReal m_e;      ///< Electron mass [kg]
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

  // std::unique_ptr<LaplaceXY> phiSolver{nullptr};
  std::unique_ptr<Laplacian> phiSolver{
      nullptr}; ///< Performs Laplacian inversions to calculate phi

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
    invert_laplace = options["invert_laplace"]
                         .doc("Use Laplace inversion routine to solve phi (if false, will instead solve via a constraint)")
                         .withDefault(true);
    include_advection = options["include_advection"]
                            .doc("Include advection terms or not")
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
    P_grad_0 = P_0 / a_mid;

    // phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
    Options &phi_init_options = Options::root()["phi"];
    Options &non_boussinesq_options = Options::root()["phiSolver"];
    if (invert_laplace)
    {
      phiSolver = Laplacian::create(&non_boussinesq_options);
      phi = 0.0; // Starting guess for first solve (if iterative)
      phi_init_options.setConditionallyUsed();

      SOLVE_FOR(P, psi, omega);
      SAVE_REPEAT(u_x, u_z, phi);
    }
    else
    {
      phi.setBoundary("phi");
      non_boussinesq_options.setConditionallyUsed();

      SOLVE_FOR(P, psi, omega, phi);
      SAVE_REPEAT(u_x, u_z);
    }

    // Output constants, input options and derived parameters
    SAVE_ONCE(e, m_i, m_e, chi, D_m, mu, epsilon, beta_p, rho, P_0);
    SAVE_ONCE(C_s0, t_0, D_0, psi_0, phi_0, R_0, a_mid, n_sepx);
    SAVE_ONCE(T_sepx, B_t0, B_pmid, evolve_pressure, include_churn_drive_term, include_mag_restoring_term, P_grad_0);

    Coordinates *coord = mesh->getCoordinates();

    // generate coordinate system
    coord->Bxy = 1.0; // TODO: Use B_t here?
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
    if (invert_laplace)
    {
      mesh->communicate(P, psi, omega);
      phi = phiSolver->solve(omega);

      mesh->communicate(phi);
    }
    else
    {
      mesh->communicate(P, psi, omega, phi);
      ddt(phi) = Delp2(phi) - omega;
    }

    u_x = DDZ(phi);
    u_z = DDX(phi);
    mesh->communicate(u_x, u_z);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
      if (include_advection)
      {
        ddt(P) = -bracket(phi, P);
      }
      else
      {
        ddt(P) = 0;
      }
      ddt(P) += (chi / D_0) * Delp2(P);
    }

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
      ddt(psi) = -bracket(phi, psi);
    }
    else
    {
      ddt(psi) = 0;
    }
    ddt(psi) += (D_m / D_0) * Delp2(psi);

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
      ddt(omega) = -bracket(phi, omega);
    }
    else
    {
      ddt(omega) = 0;
    }
    ddt(omega) += (mu / D_0) * Delp2(omega);
    if (include_churn_drive_term)
    {
      ddt(omega) += 2 * epsilon * DDZ(P);
    }
    if (include_mag_restoring_term)
    {
      ddt(omega) += (2 / beta_p) * bracket(psi, Delp2(psi));
    }

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
