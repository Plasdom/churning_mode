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
  Field3D P, psi; ///< Pressure and poloidal magnetic flux

  // Parameters
  BoutReal chi; ///< Thermal diffusivity
  BoutReal D_m; ///< Magnetic diffusivity

protected:
  int init(bool UNUSED(restarting))
  {

    /******************Reading options *****************/

    auto &globalOptions = Options::root();
    auto &options = globalOptions["model"];

    // Load system parameters
    chi = options["chi"].doc("Thermal diffusivity").withDefault(0.0);
    D_m = options["D_m"].doc("Magnetic diffusivity").withDefault(1.0e-06);

    /************ Tell BOUT++ what to solve ************/

    SOLVE_FOR(P, psi);

    return 0;
  }

  int rhs(BoutReal UNUSED(t))
  {

    // Run communications
    ////////////////////////////////////////////////////////////////////////////
    mesh->communicate(P, psi);

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(P) = chi * (D2DX2(P) + D2DZ2(P));

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    ddt(psi) = D_m * (D2DX2(psi) + D2DZ2(psi));

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
