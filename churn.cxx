#include <bout/derivs.hxx> // To use DDZ()
// #include "bout/invert/laplacexy.hxx" // Laplacian inversion
// #include "bout/invert_laplace.hxx" // Laplacian inversion
#include <bout/physicsmodel.hxx> // Commonly used BOUT++ components

/// Churning mode model
///
///
class Churn : public PhysicsModel
{
private:
  // Evolving variables
  Field3D P, psi, omega; ///< Pressure, poloidal magnetic flux and vorticity

  // Auxilliary variables
  Field3D phi, u_x, u_y;

  // Parameters
  // TODO: Normalisation on diffusion coeffs
  BoutReal chi;     ///< Thermal diffusivity
  BoutReal D_m;     ///< Magnetic diffusivity
  BoutReal mu;      ///< Vorticity diffusivity
  BoutReal epsilon; ///< Aspect ratio
  BoutReal beta_p;  ///< Poloidal beta

  // std::unique_ptr<Laplacian> phiSolver{nullptr}; ///< Performs Laplacian inversions to calculate phi
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
    // phiSolver = Laplacian::create(&boussinesq_options);
    // phi = 0.0; // Starting guess for first solve (if iterative)
    // phi.setBoundary("phi");

    /************ Tell BOUT++ what to solve ************/

    SOLVE_FOR(P, psi, omega, phi);

    // Output phi
    SAVE_REPEAT(u_x, u_y);

    Coordinates *coord = mesh->getCoordinates();

    // generate coordinate system
    coord->Bxy = 1;

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
    mesh->communicate(P, psi, omega, phi);

    // Invert div(n grad(phi)) = n Delp_perp^2(phi) = omega
    ////////////////////////////////////////////////////////////////////////////

    // Background density only (1 in normalised units)
    ddt(phi) = Laplace(phi) - omega;
    // phi = phiSolver->solve(omega);
    // phi.applyBoundary();

    // Calculate u_x and u_y components
    u_x = DDX(phi);
    u_y = DDY(phi);

    // mesh->communicate(phi);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    ddt(P) = -(DDX(P) * DDY(phi) - DDX(phi) * DDY(P));
    ddt(P) += chi * Laplace(P);

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(psi) = -(DDX(psi) * DDY(phi) - DDX(phi) * DDY(psi));
    ddt(psi) += D_m * Delp2(psi);

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(omega) = -(DDX(omega) * DDY(phi) - DDX(phi) * DDY(omega));
    ddt(omega) += mu * Laplace(omega);
    ddt(omega) += 2 * epsilon * DDY(P);
    ddt(omega) += (2 / beta_p) * (DDX(psi) * DDY(Laplace(psi)) - DDX(Laplace(psi)) * DDY(psi));

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
