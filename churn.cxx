#include <bout/derivs.hxx>           // To use DDZ()
#include "bout/invert/laplacexy.hxx" // Laplacian inversion
// #include "bout/invert_laplace.hxx" // Laplacian inversion
#include <bout/physicsmodel.hxx> // Commonly used BOUT++ components

/// Churning mode model
///
///
class Churn : public PhysicsModel
{
private:
  // Evolving variables
  Field2D P, psi, omega; ///< Pressure, poloidal magnetic flux and vorticity

  // Auxilliary variables
  Field2D phi;      ///< Electrostatic potential
  Field2D u_x, u_y; ///< Flow velocity

  // Parameters
  BoutReal chi;     ///< Thermal diffusivity
  BoutReal D_m;     ///< Magnetic diffusivity
  BoutReal mu;      ///< Vorticity diffusivity
  BoutReal epsilon; ///< Aspect ratio
  BoutReal beta_p;  ///< Poloidal beta

  // std::unique_ptr<Laplacian> phiSolver{nullptr}; ///< Performs Laplacian inversions to calculate phi
  std::unique_ptr<LaplaceXY> phiSolver{nullptr};

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
    // phiSolver = new LaplaceXY(mesh);
    phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);

    // Set the coefficients once here
    // phiSolver->setCoefs(Field2D(1.0), Field2D(0.0));
    phiSolver->setCoefs(1.0, 0.0);

    phi = 0.0; // Starting guess for first solve (if iterative)
    // phi.setBoundary(0);

    /************ Tell BOUT++ what to solve ************/

    SOLVE_FOR(P, psi, omega);

    // Output phi
    SAVE_REPEAT(phi, u_x, u_y);

    return 0;
  }

  int rhs(BoutReal UNUSED(t))
  {

    // Run communications
    ////////////////////////////////////////////////////////////////////////////
    mesh->communicate(P, psi);

    // Invert div(n grad(phi)) = n Delp_perp^2(phi) = omega
    ////////////////////////////////////////////////////////////////////////////

    // Background density only (1 in normalised units)
    // Field2D omega2D = Laplace(phi); // n=0 component
    // phi = phiSolver->solve(omega);
    phi = phiSolver->solve(omega, phi);
    u_x = -DDY(phi);
    u_y = -DDX(phi);

    mesh->communicate(phi, u_x, u_y);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    P.applyBoundary();
    // ddt(P) = -bracket(P, phi) + chi * Laplace(P);
    ddt(P) = -(DDX(P) * DDY(phi) - DDY(P) * DDX(phi)) + chi * (D2DX2(P) + D2DY2(P));

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    // ddt(psi) = -bracket(psi, phi) + D_m * Laplace(psi);
    ddt(psi) = -(DDX(psi) * DDY(phi) - DDY(psi) * DDX(phi)) + D_m * (D2DX2(P) + D2DY2(P));

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    // ddt(omega) = -bracket(omega, phi) + mu * Laplace(omega) + 2 * epsilon * DDY(P) + (1 / beta_p) * bracket(psi, Laplace(phi));
    ddt(omega) = -(DDX(omega) * DDY(phi) - DDY(omega) * DDX(phi)) + mu * (D2DX2(omega) + D2DY2(omega)) + 2 * epsilon * DDY(P) + (1 / beta_p) * (DDX(psi) * (D2DX2(phi) + D2DY2(phi)) - DDY(psi) * (D2DX2(phi) + D2DY2(phi)));

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
