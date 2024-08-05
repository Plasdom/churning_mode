#include <bout/derivs.hxx> // To use DDZ()
// #include "bout/invert/laplacexy.hxx" // Laplacian inversion
#include "bout/invert_laplace.hxx" // Laplacian inversion
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
  // Field3D phi;      ///< Electrostatic potential
  Field3D u_x, u_y; ///< Flow velocity

  // 2D variables
  Field3D phi;

  // Parameters
  BoutReal chi;     ///< Thermal diffusivity
  BoutReal D_m;     ///< Magnetic diffusivity
  BoutReal mu;      ///< Vorticity diffusivity
  BoutReal epsilon; ///< Aspect ratio
  BoutReal beta_p;  ///< Poloidal beta

  std::unique_ptr<Laplacian> phiSolver{nullptr}; ///< Performs Laplacian inversions to calculate phi
  // std::unique_ptr<LaplaceXY> phiSolver{nullptr};

protected:
  int init(bool UNUSED(restarting))
  {

    /******************Reading options *****************/

    auto &globalOptions = Options::root();
    auto &options = globalOptions["model"];

    // Load system parameters
    chi = options["chi"].doc("Thermal diffusivity").withDefault(0.0);
    D_m = options["D_m"].doc("Magnetic diffusivity").withDefault(1.0e-06);
    mu = options["mu"].doc("Voriticity diffusivity").withDefault(1.0e-06);
    epsilon = options["epsilon"].doc("Aspect ratio").withDefault(1.0e-06);
    beta_p = options["beta_p"].doc("Poloidal beta").withDefault(1.0e-06);

    /************ Create a solver for potential ********/

    Options &boussinesq_options = Options::root()["phiBoussinesq"];

    // BOUT.inp section "phiBoussinesq"
    phiSolver = Laplacian::create(&boussinesq_options);
    // phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);

    // Set the coefficients once here
    // phiSolver->setCoefs(Field2D(1.0), Field2D(0.0));
    // phiSolver->setCoefs(1.0, 0.0);

    phi = 0.0; // Starting guess for first solve (if iterative)
    phi.setBoundary("phi");

    /************ Tell BOUT++ what to solve ************/

    SOLVE_FOR(P, psi, omega, phi);

    // Output phi
    SAVE_REPEAT(phi, u_x, u_y);
    // SAVE_REPEAT(phi);

    return 0;
  }

  int rhs(BoutReal UNUSED(t))
  {

    // Run communications
    ////////////////////////////////////////////////////////////////////////////
    mesh->communicate(P, psi, omega);

    // Invert div(n grad(phi)) = n Delp_perp^2(phi) = omega
    ////////////////////////////////////////////////////////////////////////////

    // Background density only (1 in normalised units)
    // phi = phiSolver->solve(omega);
    // Field2D omega2D = DC(omega);
    // phi = phiSolver->solve(omega, phi);

    ddt(phi) = Delp2(phi) - omega;
    phi.applyBoundary();
    u_x = DDY(phi);
    u_y = DDX(phi);

    mesh->communicate(phi, u_x, u_y);
    // mesh->communicate(phi);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    // P.applyBoundary();
    // ddt(P) = -bracket(P, phi, BRACKET_ARAKAWA);
    // ddt(P) += chi * Delp2(P);

    // ddt(P) = -(DDX(P) * DDY(phi) - DDX(phi) * DDY(P));
    // ddt(P) += chi * Laplace(P);

    // ddt(P) = -bracket(P, phi);
    // ddt(P) += chi * Laplace(P);

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(psi) = -bracket(psi, phi, BRACKET_ARAKAWA);
    // ddt(psi) += D_m * Delp2(psi);

    // ddt(psi) = -(DDX(psi) * DDY(phi) - DDX(phi) * DDY(psi));
    // ddt(psi) += D_m * Laplace(psi);

    // ddt(psi) = -bracket(psi, phi);
    // ddt(psi) += D_m * Laplace(psi);

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(omega) = -bracket(omega, phi, BRACKET_ARAKAWA);
    // ddt(omega) += mu * Delp2(omega);
    // ddt(omega) += 2 * epsilon * DDZ(P);
    ddt(omega) += (2 / beta_p) * bracket(psi, Delp2(psi), BRACKET_ARAKAWA);

    // ddt(omega) = -(DDX(omega) * DDY(phi) - DDX(phi) * DDY(omega));
    // ddt(omega) += mu * Laplace(omega);
    // ddt(omega) += 2 * epsilon * DDY(P);
    // ddt(omega) += (2 / beta_p) * (DDX(psi) * DDY(Laplace(psi)) - DDX(Laplace(psi)) * DDY(psi));

    // ddt(omega) = -bracket(omega, phi);
    // ddt(omega) += mu * Laplace(omega);
    // ddt(omega) += 2 * epsilon * DDY(P);
    // ddt(omega) += (2 / beta_p) * bracket(psi, Laplace(psi));

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
