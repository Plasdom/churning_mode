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
  Field3D phi, u_x, u_y;

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
    phi = 0.0; // Starting guess for first solve (if iterative)
    phi.setBoundary("phi");

    /************ Tell BOUT++ what to solve ************/

    SOLVE_FOR(P, psi, omega);

    // Output phi
    SAVE_REPEAT(phi, u_x, u_y);

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
    mesh->communicate(P, psi, omega);

    // Invert div(n grad(phi)) = n Delp_perp^2(phi) = omega
    ////////////////////////////////////////////////////////////////////////////

    // Background density only (1 in normalised units)
    phi = phiSolver->solve(omega);
    phi.applyBoundary();
    u_x = DDX(phi);
    u_y = DDZ(phi);

    mesh->communicate(phi);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    ddt(P) = -bracket(phi, P, BRACKET_ARAKAWA);
    // ddt(P) = (DDX(P) * DDZ(phi) - DDX(phi) * DDZ(P));
    ddt(P) += chi * Delp2(P);

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(psi) = -bracket(phi, psi, BRACKET_ARAKAWA);
    ddt(psi) += D_m * Delp2(psi);

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(omega) = -bracket(phi, omega, BRACKET_ARAKAWA);
    ddt(omega) += mu * Delp2(omega);
    ddt(omega) += 2 * epsilon * DDX(P);
    ddt(omega) += (2 / beta_p) * bracket(Delp2(psi), psi, BRACKET_ARAKAWA);

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
