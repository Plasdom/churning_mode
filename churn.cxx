#include <bout/derivs.hxx>         // To use DDZ()
#include <bout/invert_laplace.hxx> // Laplacian inversion
#include <bout/physicsmodel.hxx>   // Commonly used BOUT++ components
#include <bout/interpolation.hxx>

/// Churning mode model
///
///
class Churn : public PhysicsModel
{
private:
  // Evolving variables
  Field3D P, psi, omega; ///< Pressure, poloidal magnetic flux and vorticity

  // Auxilliary variables
  Field3D u_x, u_y; // TODO: Use Vector2D object for u
  // Field3D u_x, u_y; // TODO: Use Vector2D object for u
  Field3D phi_xlow, phi_ylow;

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
  // std::unique_ptr<Laplacian> phiSolver{
  //     nullptr}; ///< Performs Laplacian inversions to calculate phi

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
    beta_p = mu_0 * 8 * pi * P_0 / pow(B_pmid, 2);
    P_grad_0 = P_0 / a_mid;

    // phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
    Options &phi_init_options = Options::root()["phi"];
    Options &non_boussinesq_options = Options::root()["phiSolver"];
    if (invert_laplace)
    {
      // phiSolver = Laplacian::create(&non_boussinesq_options);
      // phi = 0.0; // Starting guess for first solve (if iterative)
      // phi_init_options.setConditionallyUsed();

      SOLVE_FOR(P, psi, omega);
      // SAVE_REPEAT(u_x, u_y, phi);
    }
    else
    {
      // phi.setBoundary("phi");
      phi_xlow.setBoundary("phi");
      phi_ylow.setBoundary("phi");
      // phi = 0.0;
      phi_xlow = 0.0;
      phi_ylow = 0.0;
      phi_xlow.setLocation(CELL_XLOW);
      phi_ylow.setLocation(CELL_YLOW);
      non_boussinesq_options.setConditionallyUsed();

      // SOLVE_FOR(P, psi, omega, phi, phi_xlow, phi_ylow);
      SOLVE_FOR(P, psi, omega, phi_xlow, phi_ylow);
      SAVE_REPEAT(u_x, u_y);
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

    // Solve phi
    ////////////////////////////////////////////////////////////////////////////
    if (invert_laplace)
    {
      // TODO: Use LaplaceXY when this option is switched on
      mesh->communicate(P, psi, omega);
      // phi = phiSolver->solve(omega);

      // mesh->communicate(phi);
    }
    else
    {
      mesh->communicate(P, psi, omega, phi_xlow, phi_ylow);
      // ddt(phi) = Laplace(phi) - omega;
      // ddt(phi) = (D2DX2(phi) + D2DY2(phi)) - omega;
      ddt(phi_xlow) = Laplace(phi_xlow, CELL_XLOW) - interp_to(omega, CELL_XLOW);
      ddt(phi_ylow) = Laplace(phi_ylow, CELL_YLOW) - interp_to(omega, CELL_YLOW);
    }

    u_x = DDY(phi_ylow, CELL_CENTER);
    u_y = DDX(phi_xlow, CELL_CENTER);
    mesh->communicate(u_x, u_y);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
      if (include_advection)
      {
        // ddt(P) = -bracket(phi, P);
        ddt(P) = -(DDX(P) * u_x - u_y * DDY(P));
      }
      else
      {
        ddt(P) = 0;
      }
      // ddt(P) += (chi / D_0) * Laplace(P);
      ddt(P) += (chi / D_0) * (D2DX2(P) + D2DY2(P));
    }

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
      // ddt(psi) = -bracket(phi, psi);
      ddt(psi) = -(DDX(psi) * u_x - u_y * DDY(psi));
    }
    else
    {
      ddt(psi) = 0;
    }
    // ddt(psi) += (D_m / D_0) * Laplace(psi);
    ddt(psi) += (D_m / D_0) * (D2DX2(psi) + D2DY2(psi));

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
      // ddt(omega) = -bracket(phi, omega);
      ddt(omega) = -(DDX(omega) * u_x - u_y * DDY(omega));
    }
    else
    {
      ddt(omega) = 0;
    }
    // ddt(omega) += (mu / D_0) * Laplace(omega);
    ddt(omega) += (mu / D_0) * (D2DX2(omega) + D2DY2(omega));
    if (include_churn_drive_term)
    {
      ddt(omega) += 2 * epsilon * DDY(P);
    }
    if (include_mag_restoring_term)
    {
      // ddt(omega) += -(2 / beta_p) * bracket(psi, Delp2(psi), BRACKET_ARAKAWA);
      // ddt(omega) += -(2 / beta_p) * (DDX(psi) * DDY(D2DX2(psi) + D2DY2(psi)) - DDY(psi) * DDX(D2DX2(psi) + D2DY2(psi)));
      ddt(omega) += -(2 / beta_p) * (DDX(psi) * DDY(Laplace(psi)) - DDY(psi) * DDX(Laplace(psi)));

      // ddt(omega) += DDY(D2DX2(psi));
      // ddt(omega) += DDY(D2DY2(psi));
      // ddt(omega) += DDX(D2DY2(psi));
      // ddt(omega) += DDX(D2DX2(psi));
    }

    // Apply additional BCs on first two cells all around domain to handle thrid derivatives
    RangeIterator xrdn = mesh->iterateBndryLowerY();
    for (xrdn.first(); !xrdn.isDone(); xrdn.next())
    {
      for (int jy = mesh->ystart + 1; jy >= 0; jy--)
      {
        for (int jz = 0; jz < mesh->LocalNz; jz++)
        {
          ddt(omega)(xrdn.ind, jy, jz) = 0;
          ddt(phi_xlow)(xrdn.ind, jy, jz) = 0;
          ddt(phi_ylow)(xrdn.ind, jy, jz) = ddt(phi_ylow)(xrdn.ind, jy + 1, jz);
          ddt(psi)(xrdn.ind, jy, jz) = 0;
          ddt(P)(xrdn.ind, jy, jz) = 0;
          u_x(xrdn.ind, jy, jz) = 0;
          u_y(xrdn.ind, jy, jz) = 0;
        }
      }
    }

    RangeIterator xrup = mesh->iterateBndryUpperY();
    for (xrup.first(); !xrup.isDone(); xrup.next())
    {
      for (int jy = mesh->yend - 1; jy < mesh->LocalNy; jy++)
      {
        for (int jz = 0; jz < mesh->LocalNz; jz++)
        {
          ddt(omega)(xrup.ind, jy, jz) = 0;
          ddt(phi_xlow)(xrup.ind, jy, jz) = 0;
          ddt(phi_ylow)(xrup.ind, jy, jz) = 0;
          ddt(psi)(xrup.ind, jy, jz) = 0;
          ddt(P)(xrup.ind, jy, jz) = 0;
          u_x(xrdn.ind, jy, jz) = 0;
          u_y(xrdn.ind, jy, jz) = 0;
        }
      }
    }

    // TODO: There is no mesh->iterateBndryInnerX object, so having to do this in a janky way which won't parallelise. Fix in future.
    for (xrdn.first(); !xrdn.isDone(); xrdn.next())
    {
      for (int jy = mesh->ystart + 1; jy >= 0; jy--)
      {
        for (int jz = 0; jz < mesh->LocalNz; jz++)
        {
          ddt(omega)(jy, xrdn.ind, jz) = 0;
          ddt(phi_xlow)(jy, xrdn.ind, jz) = ddt(phi_xlow)(jy + 1, xrdn.ind, jz);
          ddt(phi_ylow)(jy, xrdn.ind, jz) = 0;
          ddt(psi)(jy, xrdn.ind, jz) = 0;
          ddt(P)(jy, xrdn.ind, jz) = 0;
          u_x(jy, xrdn.ind, jz) = 0;
          u_y(jy, xrdn.ind, jz) = 0;
        }
      }
    }

    for (xrup.first(); !xrup.isDone(); xrup.next())
    {
      for (int jy = mesh->yend - 1; jy < mesh->LocalNy; jy++)
      {
        for (int jz = 0; jz < mesh->LocalNz; jz++)
        {
          ddt(omega)(jy, xrup.ind, jz) = 0;
          ddt(phi_xlow)(jy, xrup.ind, jz) = 0;
          ddt(phi_ylow)(jy, xrup.ind, jz) = 0;
          ddt(psi)(jy, xrup.ind, jz) = 0;
          ddt(P)(jy, xrup.ind, jz) = 0;
          u_x(jy, xrup.ind, jz) = 0;
          u_y(jy, xrup.ind, jz) = 0;
        }
      }
    }

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
