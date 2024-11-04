#include <bout/derivs.hxx> // To use DDZ()
// #include "bout/invert/laplacexy.hxx" // Laplacian inversion
#include <bout/physicsmodel.hxx> // Commonly used BOUT++ components
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
  Field3D theta_m, phi_m;

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
  bool evolve_pressure;             ///< Evolve plasma pressure
  bool include_mag_restoring_term;  ///< Include the poloidal magnetic field restoring term in the vorticity equation
  bool include_churn_drive_term;    ///< Include the churn driving term in the vorticity equation
  bool invert_laplace;              ///< Use Laplace inversion routine to solve phi (if false, will instead solve via a constraint) (TODO: Implement this option)
  bool include_advection;           ///< Use advection terms
  bool use_sk9_anis_diffop;         ///< Use the SK9 stencil for anisotropic heat flow operator
  bool fixed_T_down;                ///< Use a constant value for P on downstream boundaries
  bool T_dependent_q_par;           ///< Use Spitzer-Harm form of parallel conductivity
  bool use_symmetric_div_q_stencil; ///< Use a symmetric stencil for the div_q term

  // std::unique_ptr<LaplaceXY> phiSolver{nullptr};
  struct myLaplacian
  {
    BoutReal D = 1.0, A = 0.0;
    int ngcx_tot, ngcy_tot, nx_tot, ny_tot, nz_tot;

    Field3D operator()(const Field3D &input)
    {
      Field3D result = A * input + D * (D2DX2(input) + D2DY2(input));

      // Ensure boundary points are set appropriately as given by the input field.
      // TODO: Check this is doing what is expected. Surely it's the input, not result, we should be applying boundaries to?
      Mesh *mesh = result.getMesh();
      // X boundaries
      if (mesh->firstX())
      {
        for (int ix = 0; ix < ngcx_tot; ix++)
        {
          for (int iy = 0; iy < mesh->LocalNy; iy++)
          {
            for (int iz = 0; iz < mesh->LocalNz; iz++)
            {
              result(ix, iy, iz) = 0.0;
            }
          }
        }
      }
      if (mesh->lastX())
      {
        for (int ix = mesh->LocalNx - ngcx_tot; ix < mesh->LocalNx; ix++)
        {
          for (int iy = 0; iy < mesh->LocalNy; iy++)
          {
            for (int iz = 0; iz < mesh->LocalNz; iz++)
            {
              result(ix, iy, iz) = 0.0;
            }
          }
        }
      }
      // Y boundaries
      RangeIterator itl = mesh->iterateBndryLowerY();
      for (itl.first(); !itl.isDone(); itl++)
      {
        // it.ind contains the x index
        for (int iy = 0; iy < ngcy_tot; iy++)
        {
          for (int iz = 0; iz < mesh->LocalNz; iz++)
          {
            result(itl.ind, iy, iz) = 0.0;
          }
        }
      }
      RangeIterator itu = mesh->iterateBndryUpperY();
      for (itu.first(); !itu.isDone(); itu++)
      {
        // it.ind contains the x index
        for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        {
          for (int iz = 0; iz < mesh->LocalNz; iz++)
          {
            result(itu.ind, iy, iz) = 0.0;
          }
        }
      }

      // result.setBoundaryTo(input);

      return result;
    };
  };
  myLaplacian mm;
  bout::inversion::InvertableOperator<Field3D> mySolver;
  const int nits_inv_extra = 0;

  // Y boundary iterators
  RangeIterator itl = mesh->iterateBndryLowerY();
  RangeIterator itu = mesh->iterateBndryUpperY();

  // Skewed 9-point stencil for anisotropic diffusion (A,B & C currently assumed to not vary with f, and b field direction assumed constant)
  Field3D Anis_Diff_SK9(const Field3D &f, const Field3D &A, const Field3D &B, const Field3D &C)
  {
    TRACE("Anis_Diff");

    Field3D result;
    BoutReal r;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
      r = coord->dx[i] / coord->dy[i];
      result[i] =
          ((B[i] * pow(r, 2.0) - C[i] * r) * f[i.yp()] + C[i] * r * f[i.yp().xp()] + (A[i] - C[i] * r) * f[i.xm()] + (-2.0 * A[i] - 2.0 * B[i] * pow(r, 2.0) + 2.0 * C[i] * r) * f[i] + (A[i] - C[i] * r) * f[i.xp()] + C[i] * r * f[i.xm().ym()] + (B[i] * pow(r, 2.0) - C[i] * r) * f[i.ym()]) / (coord->dy[i] * coord->dx[i]);
    }

    return result;
  }

  // Symettric stencil for div_q term (assumes K_par, K_perp constant in space)
  Field3D div_q_symmetric(const Field3D &T, const BoutReal &K_par, const BoutReal &K_perp, const Vector3D &b)
  {
    TRACE("div_q_symmetric");

    Field3D result;
    Field3D bx_corners, by_corners, DTDX_corners, DTDY_corners, q_parx_corners, q_pary_corners, q_perpx_corners, q_perpy_corners;

    Coordinates *coord = mesh->getCoordinates();

    // TODO: Check below is valid when dx!=dy
    bx_corners.allocate();
    by_corners.allocate();
    for (auto i : result)
    {
      bx_corners[i] = 0.25 * (b.x[i.xm()] + b.x[i.xm().ym()] + b.x[i.ym()] + b.x[i]);
      by_corners[i] = 0.25 * (b.y[i.xm()] + b.y[i.xm().ym()] + b.y[i.ym()] + b.y[i]);
    }

    // Find temperature gradients on cell corners
    DTDX_corners.allocate();
    DTDY_corners.allocate();
    for (auto i : DTDX_corners)
    {

      DTDX_corners[i] = (1.0 / (2.0 * coord->dx[i])) * ((T[i] + T[i.ym()]) - (T[i.xm()] + T[i.xm().ym()]));
      DTDY_corners[i] = (1.0 / (2.0 * coord->dy[i])) * ((T[i] + T[i.xm()]) - (T[i.ym()] + T[i.xm().ym()]));
    }

    q_parx_corners = K_par * bx_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners);
    q_pary_corners = K_par * by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners);
    q_perpx_corners = K_perp * (DTDX_corners - by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners));
    q_perpy_corners = K_perp * (DTDY_corners - by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners));

    result.allocate();
    for (auto i : result)
    {
      result[i] = (1.0 / (2.0 * coord->dx[i])) * (q_parx_corners[i.xp().yp()] + q_parx_corners[i.xp()] - q_parx_corners[i.yp()] - q_parx_corners[i]);
      result[i] += (1.0 / (2.0 * coord->dy[i])) * (q_pary_corners[i.xp().yp()] + q_pary_corners[i.yp()] - q_pary_corners[i.xp()] - q_pary_corners[i]);
      result[i] += (1.0 / (2.0 * coord->dx[i])) * (q_perpx_corners[i.xp().yp()] + q_perpx_corners[i.xp()] - q_perpx_corners[i.yp()] - q_perpx_corners[i]);
      result[i] += (1.0 / (2.0 * coord->dy[i])) * (q_perpy_corners[i.xp().yp()] + q_perpy_corners[i.yp()] - q_perpy_corners[i.xp()] - q_perpy_corners[i]);
    }

    return result;
  }

  std::vector<CellIntersect> get_intersects(const float &xlo, const float &xhi, const float &ylo, const float &yhi, const CellIntersect &P, const float &bx, const float &by)
  {
    // Find the intersection points between a line with gradient given by by/bx, where (Px,Py) is a point on the line, and the box bounded by xlo, xhi, ylo, yhi
    // TODO: Check edge cases when intersection is on a corner
    TRACE("intersects_plus");

    // std::vector<std::vector<float>> result = {{0.0, 1.0}, {2.0, 3.0}};
    std::vector<CellIntersect> result;
    CellIntersect intersect;
    float m, c, y_xlo, y_xhi, x_ylo, x_yhi;

    result.resize(2);

    // Find line equation for b
    if (bx == 0.0)
    {
      m = 1e12;
    }
    else
    {
      m = by / bx;
    }
    c = P.y - m * P.x;

    // Determine which of box faces are intersected
    y_xlo = m * xlo + c;
    y_xhi = m * xhi + c;
    x_ylo = (ylo - c) / m;
    x_yhi = (yhi - c) / m;

    if ((ylo <= y_xlo) && (y_xlo < yhi))
    {
      // Intersect lower x face
      intersect.face = 0;
      intersect.x = xlo;
      intersect.y = y_xlo;
      result[0] = intersect;
    }
    if ((ylo < y_xhi) && (y_xhi <= yhi))
    {
      // Intersect upper x face
      intersect.face = 2;
      intersect.x = xhi;
      intersect.y = y_xhi;
      result[1] = intersect;
    }
    if ((xlo < x_ylo) && (x_ylo <= xhi))
    {
      // Intersect lower y face
      intersect.face = 3;
      intersect.x = x_ylo;
      intersect.y = ylo;
      result[0] = intersect;
    }
    if ((xlo <= x_yhi) && (x_yhi < xhi))
    {
      // Intersect upper y face
      intersect.face = 1;
      intersect.x = x_yhi;
      intersect.y = yhi;
      result[1] = intersect;
    }

    return result;
  }

  CellIntersect get_next_intersect(const float &xlo, const float &xhi, const float &ylo, const float &yhi, const CellIntersect &prev_intersect, const float &bx, const float &by)
  {
    // Find the intersection points between a line with gradient given by by/bx, where (Px,Py) is a point on the line, and the box bounded by xlo, xhi, ylo, yhi
    TRACE("get_next_intersect");

    // std::vector<std::vector<float>> result = {{0.0, 1.0}, {2.0, 3.0}};
    CellIntersect result;
    float m, c, y_xlo, y_xhi, x_ylo, x_yhi;

    // Find line equation for b
    if (bx == 0.0)
    {
      m = 1e12;
    }
    else
    {
      m = by / bx;
    }
    c = prev_intersect.y - m * prev_intersect.x;

    // Determine which of box faces are intersected
    y_xlo = m * xlo + c;
    y_xhi = m * xhi + c;
    x_ylo = (ylo - c) / m;
    x_yhi = (yhi - c) / m;

    if ((ylo <= y_xlo) && (y_xlo < yhi) && (prev_intersect.face != 2))
    {
      // Intersect lower x face
      result.face = 0;
      result.x = xlo;
      result.y = y_xlo;
    }

    if ((ylo < y_xhi) && (y_xhi <= yhi) && (prev_intersect.face != 0))
    {
      // Intersect upper x face
      result.face = 2;
      result.x = xhi;
      result.y = y_xhi;
    }

    if ((xlo < x_ylo) && (x_ylo <= xhi) && (prev_intersect.face != 1))
    {
      // Intersect lower y face
      result.face = 3;
      result.x = x_ylo;
      result.y = ylo;
    }

    if ((xlo <= x_yhi) && (x_yhi < xhi) && (prev_intersect.face != 3))
    {
      // Intersect upper y face
      result.face = 1;
      result.x = x_yhi;
      result.y = yhi;
    }

    // TODO: Confirm one of above if statements is always entered

    return result;
  }

  Ind3D increment_cell(const Ind3D &i, const Ind3D &i_prev, const CellIntersect &P_next, const float &dx, const float &dy)
  {
    // Determine which cell to move to next
    TRACE("increment_cell");

    Ind3D i_next;
    float tol = 1.0e-6;
    int x_inc, y_inc;

    i_next = i_prev;
    x_inc = static_cast<float>(i_prev.x() - i.x());
    y_inc = static_cast<float>(i_prev.y() - i.y());

    if (abs(((P_next.x / dx) - 0.5) - x_inc) < tol)
    {
      // Move one cell to the right
      i_next = i_next.xp();
    }
    else if (abs(((P_next.x / dx) - 0.5) - (x_inc - 1.0)) < tol)
    {
      // Move one cell to the left
      i_next = i_next.xm();
    }
    if (abs(((P_next.y / dy) - 0.5) - y_inc) < tol)
    {
      // Move one cell upwards
      i_next = i_next.yp();
    }
    else if (abs(((P_next.y / dy) - 0.5) - (y_inc - 1.0)) < tol)
    {
      // Move one cell downwards
      i_next = i_next.ym();
    }
    // else
    // {
    // TODO: Check that this condition is never met
    //   result[i] = -999;
    // }

    return i_next;
  }

  InterpolationPoint trace_field_lines(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_x_inc, const int &max_y_inc, const int &max_steps, const bool &plus)
  {
    InterpolationPoint result;
    CellIntersect next_intersect, prev_intersect;
    std::vector<CellIntersect> intersects;
    float par_dist, par_dist_closest;
    Ind3D i_next, i_prev;
    Point cell_centre;
    ClosestPoint p, p_closest;
    int n_steps;
    bool continue_tracing = true;

    // Initialise the next intersection
    next_intersect.x = 0.0;
    next_intersect.y = 0.0;
    par_dist = 0.0;
    p_closest.x = 0.0;
    p_closest.y = 0.0;
    p_closest.distance = 1.0e10;
    p.x = 0.0;
    p.y = 0.0;
    p.distance = 0.0;

    // Get the first intersect from the centre of cell (i,j)
    intersects = get_intersects(-dx / 2.0, dx / 2.0, -dy / 2.0, dy / 2.0, next_intersect, b.x[i], b.y[i]);

    // Specify plus/minus direction (choice is arbitrary)
    // TODO: Check the logic of selcting next intersect here. What if intersects.size != 2?
    prev_intersect = next_intersect;
    if (plus == true)
    {
      if (intersects[1].x > 0.0)
      {
        next_intersect = intersects[1];
      }
      else
      {
        next_intersect = intersects[0];
      }
    }
    else
    {
      if (intersects[1].x < 0.0)
      {
        next_intersect = intersects[1];
      }
      else
      {
        next_intersect = intersects[0];
      }
    }
    // if (plus == true)
    // {
    //   next_intersect = intersects[0];
    // }
    // else
    // {
    //   next_intersect = intersects[1];
    // }

    // Find the parallel distance par_dist = distance from (x_plus_prev, y_plus_prev) to (x_plus, y_plus) (i.e. it's the cumulative distance just to the next intersection point)
    par_dist += sqrt(pow(next_intersect.x - 0.0, 2.0) + pow(next_intersect.y - 0.0, 2.0)) * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);

    // Determine which cell to move to next
    i_prev = i;
    i_next = increment_cell(i, i_prev, next_intersect, dx, dy);

    // Find the impact parameter in this adjacent cell
    cell_centre.x = (i_next.x() - i.x()) * dx;
    cell_centre.y = (i_next.y() - i.y()) * dy;
    p = get_closest_p(next_intersect, cell_centre, b.x[i_next], b.y[i_next]);

    // Update the closest point to a cell centre found so far
    p_closest = p;
    par_dist_closest = par_dist + sqrt(pow(p_closest.x - next_intersect.x, 2.0) + pow(p_closest.y - next_intersect.y, 2.0)) * sqrt(pow(b.z[i_next], 2.0) / (pow(b.x[i_next], 2.0) + pow(b.y[i_next], 2.0)) + 1.0);

    // Continue to trace field lines in the plus-direction
    n_steps = 1;
    while (continue_tracing == true)
    {

      // Find intercepts in the new cell
      prev_intersect = next_intersect;
      next_intersect = get_next_intersect((-dx / 2.0) + (i_next.x() - i_prev.x()) * dx,
                                          (dx / 2.0) + (i_next.x() - i_prev.x()) * dx,
                                          (-dy / 2.0) + (i_next.y() - i_prev.y()) * dy,
                                          (dy / 2.0) + (i_next.y() - i_prev.y()) * dy,
                                          prev_intersect,
                                          b.x[i_next],
                                          b.y[i_next]);

      // Find the parallel distance par_dist = distance from (x_plus_prev, y_plus_prev) to (x_plus, y_plus) (i.e. it's the cumulative distance just to the next intersection point)
      par_dist += sqrt(pow(next_intersect.x - prev_intersect.x, 2.0) + pow(next_intersect.y - prev_intersect.y, 2.0)) * sqrt(pow(b.z[i_next], 2.0) / (pow(b.x[i_next], 2.0) + pow(b.y[i_next], 2.0)) + 1.0);

      // Get subsequent intersects in plus direction
      i_prev = i_next;
      i_next = increment_cell(i, i_prev, next_intersect, dx, dy);

      // Find the impact parameter in this adjacent cell
      cell_centre.x = (i_next.x() - i.x()) * dx;
      cell_centre.y = (i_next.y() - i.y()) * dy;
      p = get_closest_p(next_intersect, cell_centre, b.x[i_next], b.y[i_next]);

      // Update the closest point to a cell centre found so far
      if (p.distance < p_closest.distance)
      {
        p_closest = p;
        par_dist_closest = par_dist + sqrt(pow(p_closest.x - next_intersect.x, 2.0) + pow(p_closest.y - next_intersect.y, 2.0)) * sqrt(pow(b.z[i_next], 2.0) / (pow(b.x[i_next], 2.0) + pow(b.y[i_next], 2.0)) + 1.0);
      }

      n_steps++;

      // Cease tracing
      if (abs(i_next.x() - i.x()) >= max_x_inc)
      {
        continue_tracing = false;
      }
      if (abs(i_next.y() - i.y()) >= max_y_inc)
      {
        continue_tracing = false;
      }
      if (n_steps >= max_steps)
      {
        continue_tracing = false;
      }
    }

    result.x = p_closest.x;
    result.y = p_closest.y;
    result.distance = p_closest.distance;
    result.parallel_distance = par_dist_closest;

    return result;
  }

  ClosestPoint get_closest_p(const CellIntersect &P, const Point &P0, const float &bx, const float &by)
  {
    // Find the closest point on the line following the magnetic field, extending from an cell face intersection point (Px,Py), to a point at (x0, y0), e.g. a cell centre. Output is a vector with three elements (x, y, distance)
    TRACE("get_closest_p");

    ClosestPoint result;
    float distance, x_closest, y_closest;
    float A, B, C;

    // Find line equation
    B = 1.0;
    if (bx == 0.0)
    {
      A = 1e12;
    }
    else
    {
      A = -by / bx;
    }
    C = -(P.y + A * P.x);

    // Find closest distance
    distance = abs(A * P0.x + B * P0.y + C) / sqrt(pow(A, 2.0) + pow(B, 2.0));

    // Find the coordinates of this closest point
    x_closest = (B * (B * P0.x - A * P0.y) - A * C) / (pow(A, 2.0) + pow(B, 2.0));
    y_closest = (A * (-B * P0.x + A * P0.y) - B * C) / (pow(A, 2.0) + pow(B, 2.0));

    result.x = x_closest;
    result.y = y_closest;
    result.distance = distance;
    return result;
  }

  Field3D Q_linetrace(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
  {
    TRACE("Q_linetrace");

    Field3D result;
    Field3D x_plus, y_plus, x_minus, y_minus, parallel_distances_plus, parallel_distances_minus, q_plus, q_minus;
    Coordinates *coord = mesh->getCoordinates();
    InterpolationPoint interp_p_plus, interp_p_minus;
    int n_steps, n_x, n_y;
    float f_x, f_y, u_plus, q_plus_T, u_minus, q_minus_T, div_q_plus, div_q_minus;
    int max_x_inc = 3;
    int max_y_inc = 3;
    int max_steps = 8;
    Ind3D i_offset;

    x_plus = 0.0;
    y_plus = 0.0;
    x_minus = 0.0;
    y_minus = 0.0;
    q_plus = 0.0;
    q_minus = 0.0;
    parallel_distances_plus = 0.0;
    parallel_distances_minus = 0.0;
    result = 0.0;
    // BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    for (auto i : result)
    {

      interp_p_plus = trace_field_lines(i, b, coord->dx[i], coord->dy[i], max_x_inc, max_y_inc, max_steps, true);
      x_plus[i] = interp_p_plus.x;
      y_plus[i] = interp_p_plus.y;
      parallel_distances_plus[i] = interp_p_plus.parallel_distance;

      interp_p_minus = trace_field_lines(i, b, coord->dx[i], coord->dy[i], max_x_inc, max_y_inc, max_steps, false);
      x_minus[i] = interp_p_minus.x;
      y_minus[i] = interp_p_minus.y;
      parallel_distances_minus[i] = interp_p_minus.parallel_distance;

      n_x = static_cast<int>(floor(x_plus[i] / coord->dx[i]));
      n_x = std::min(std::max(n_x, -ngcx), ngcx);
      n_y = static_cast<int>(floor(y_plus[i] / coord->dy[i]));
      n_y = std::min(std::max(n_y, -ngcy), ngcy);
      i_offset = i.offset(n_x, n_y, 0);

      f_x = (x_plus[i] - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_plus[i] - n_y * coord->dy[i]) / coord->dy[i];

      u_plus = (1.0 - f_y) * ((1 - f_x) * u[i_offset] + f_x * u[i_offset.xp()]) + f_y * ((1 - f_x) * u[i_offset.yp()] + f_x * u[i_offset.xp().yp()]);
      q_plus[i] = K_par * (u_plus - u[i]) / parallel_distances_plus[i];

      n_x = static_cast<int>(floor(x_minus[i] / coord->dx[i]));
      n_x = std::min(std::max(n_x, -ngcx), ngcx);
      n_y = static_cast<int>(floor(y_minus[i] / coord->dy[i]));
      n_y = std::min(std::max(n_y, -ngcy), ngcy);
      i_offset = i.offset(n_x, n_y, 0);

      f_x = (x_minus[i] - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_minus[i] - n_y * coord->dy[i]) / coord->dy[i];
      u_minus = (1.0 - f_y) * ((1 - f_x) * u[i_offset] + f_x * u[i_offset.xp()]) + f_y * ((1 - f_x) * u[i_offset.yp()] + f_x * u[i_offset.xp().yp()]);
      q_minus[i] = -K_par * (u_minus - u[i]) / parallel_distances_minus[i];
    }

    // result = y_plus / coord->dy;
    // // Naive method
    // result = 2.0 * (q_plus - q_minus) / (parallel_distances_plus + parallel_distances_minus);

    // Support operator method
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
      n_x = static_cast<int>(floor(x_plus[i] / coord->dx[i]));
      n_x = std::min(std::max(n_x, -ngcx), ngcx);
      n_y = static_cast<int>(floor(y_plus[i] / coord->dy[i]));
      n_y = std::min(std::max(n_y, -ngcy), ngcy);
      i_offset = i.offset(-n_x, -n_y, 0);

      f_x = (x_plus[i] - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_plus[i] - n_y * coord->dy[i]) / coord->dy[i];
      q_plus_T = (1.0 - f_y) * ((1 - f_x) * q_plus[i_offset] + f_x * q_plus[i_offset.xm()]) + f_y * ((1 - f_x) * q_plus[i_offset.ym()] + f_x * q_plus[i_offset.xm().ym()]);
      // div_q_plus = (parallel_distances_plus[i] / (0.5 * (parallel_distances_plus[i] + parallel_distances_minus[i]))) * ((q_plus_T - q_plus[i]) / parallel_distances_plus[i]);
      div_q_plus = ((q_plus_T - q_plus[i]) / parallel_distances_plus[i]);

      n_x = static_cast<int>(floor(x_minus[i] / coord->dx[i]));
      n_x = std::min(std::max(n_x, -ngcx), ngcx);
      n_y = static_cast<int>(floor(y_minus[i] / coord->dy[i]));
      n_y = std::min(std::max(n_y, -ngcy), ngcy);
      i_offset = i.offset(-n_x, -n_y, 0);

      f_x = (x_minus[i] - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_minus[i] - n_y * coord->dy[i]) / coord->dy[i];
      q_minus_T = (1.0 - f_y) * ((1 - f_x) * q_minus[i_offset] + f_x * q_minus[i_offset.xm()]) + f_y * ((1 - f_x) * q_minus[i_offset.ym()] + f_x * q_minus[i_offset.xm().ym()]);
      div_q_minus = -(parallel_distances_minus[i] / (0.5 * (parallel_distances_plus[i] + parallel_distances_minus[i]))) * ((q_minus_T - q_minus[i]) / parallel_distances_minus[i]);

      result[i] = -0.5 * (div_q_plus + div_q_minus);
      // result[i] = -div_q_plus;
    }

    return result;
  }

  Field3D Q_plus(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
  {
    TRACE("Q_plus");

    Field3D result;
    BoutReal f_x, f_y;
    float y_plus, x_plus, ds_p, ds, u_plus;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    // BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    for (auto i : result)
    {
      // // Modified Stegmeir a
      // ds_p = sqrt(pow(coord->dx[i], 2) + pow(coord->dy[i], 2));
      // x_plus = ds_p * cos(atan2(b.y[i], b.x[i]));
      // y_plus = x_plus * b.y[i] / b.x[i];
      // n_x = static_cast<int>(floor(x_plus / coord->dx[i]));
      // n_y = static_cast<int>(floor((y_plus) / coord->dy[i]));

      // Modified Stegmeir b
      x_plus = coord->dy[i] * abs(b.x[i] / b.y[i]);
      x_plus = std::min(static_cast<float>(coord->dx[i]), x_plus);
      y_plus = coord->dx[i] * abs(b.y[i] / b.x[i]);
      y_plus = std::min(static_cast<float>(coord->dy[i]), y_plus);
      if (b.x[i] >= 0)
      {
        n_x = 0;
      }
      else
      {
        n_x = -1;
        x_plus = -x_plus;
      }
      if (b.y[i] >= 0)
      {
        n_y = 0;
      }
      else
      {
        n_y = -1;
        y_plus = -y_plus;
      }
      ds_p = sqrt(pow(x_plus, 2) + pow(y_plus, 2));

      ds = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
      f_x = (x_plus - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_plus - n_y * coord->dy[i]) / coord->dy[i];
      u_plus = (1.0 - f_y) * ((1 - f_x) * u(i.x() + n_x, i.y() + n_y, i.z()) + f_x * u(i.x() + n_x + 1, i.y() + n_y, i.z())) + f_y * ((1 - f_x) * u(i.x() + n_x, i.y() + n_y + 1, i.z()) + f_x * u(i.x() + n_x + 1, i.y() + n_y + 1, i.z()));
      result[i] = K_par * (u_plus - u[i]) / ds;
    }

    return result;
  }

  Field3D
  Q_plus_T(const Field3D &u, const Vector3D &b)
  {
    TRACE("Q_plus_T");

    Field3D result;
    BoutReal f_x, f_y;
    float y_plus, x_plus, ds_p, ds, u_plus;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
      // // Modified Stegmeir a
      // ds_p = 0.5 * sqrt(pow(coord->dx[i], 2) + pow(coord->dy[i], 2));
      // x_plus = ds_p * cos(atan2(b.y[i], b.x[i]));
      // y_plus = x_plus * b.y[i] / b.x[i];
      // n_x = static_cast<int>(floor(x_plus / coord->dx[i]));
      // n_y = static_cast<int>(floor(y_plus / coord->dy[i]));

      // Modified Stegmeir b
      x_plus = coord->dy[i] * abs(b.x[i] / b.y[i]);
      x_plus = std::min(static_cast<float>(coord->dx[i]), x_plus);
      y_plus = coord->dx[i] * abs(b.y[i] / b.x[i]);
      y_plus = std::min(static_cast<float>(coord->dy[i]), y_plus);
      if (b.x[i] >= 0)
      {
        n_x = 0;
      }
      else
      {
        n_x = -1;
        x_plus = -x_plus;
      }
      if (b.y[i] >= 0)
      {
        n_y = 0;
      }
      else
      {
        n_y = -1;
        y_plus = -y_plus;
      }
      ds_p = sqrt(pow(x_plus, 2) + pow(y_plus, 2));

      ds = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
      f_x = (x_plus - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_plus - n_y * coord->dy[i]) / coord->dy[i];
      u_plus = (1.0 - f_y) * ((1 - f_x) * u(i.x() - n_x, i.y() - n_y, i.z()) + f_x * u(i.x() - n_x - 1, i.y() - n_y, i.z())) + f_y * ((1 - f_x) * u(i.x() - n_x, i.y() - n_y - 1, i.z()) + f_x * u(i.x() - n_x - 1, i.y() - n_y - 1, i.z()));
      result[i] = (u_plus - u[i]) / ds;
    }

    return result;
  }

  Field3D Q_minus(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
  {
    TRACE("Q_minus");

    Field3D result;
    BoutReal f_x, f_y;
    float y_minus, x_minus, ds_p, ds, u_minus;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    // BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    for (auto i : result)
    {

      // // Modified Stegmeir a
      // ds_p = 0.5 * sqrt(pow(coord->dx[i], 2) + pow(coord->dy[i], 2));
      // x_minus = ds_p * cos(atan2(b.y[i], b.x[i]));
      // y_minus = x_minus * b.y[i] / b.x[i];
      // n_x = static_cast<int>(floor(x_minus / coord->dx[i]));
      // n_y = static_cast<int>(floor(y_minus / coord->dy[i]));

      // Modified Stegmeir b
      x_minus = coord->dy[i] * abs(b.x[i] / b.y[i]);
      x_minus = std::min(static_cast<float>(coord->dx[i]), x_minus);
      y_minus = coord->dx[i] * abs(b.y[i] / b.x[i]);
      y_minus = std::min(static_cast<float>(coord->dy[i]), y_minus);
      if (b.x[i] >= 0)
      {
        n_x = 0;
      }
      else
      {
        n_x = -1;
        x_minus = -x_minus;
      }
      if (b.y[i] >= 0)
      {
        n_y = 0;
      }
      else
      {
        n_y = -1;
        y_minus = -y_minus;
      }
      ds_p = sqrt(pow(x_minus, 2) + pow(y_minus, 2));

      ds = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
      f_x = (x_minus - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_minus - n_y * coord->dy[i]) / coord->dy[i];
      u_minus = (1.0 - f_y) * ((1 - f_x) * u(i.x() - n_x, i.y() - n_y, i.z()) + f_x * u(i.x() - n_x - 1, i.y() - n_y, i.z())) + f_y * ((1 - f_x) * u(i.x() - n_x, i.y() - n_y - 1, i.z()) + f_x * u(i.x() - n_x - 1, i.y() - n_y - 1, i.z()));
      result[i] = -K_par * (u_minus - u[i]) / ds;
    }

    return result;
  }

  Field3D
  Q_minus_T(const Field3D &u, const Vector3D &b)
  {
    TRACE("Q_minus_T");

    Field3D result;
    BoutReal f_x, f_y;
    float y_minus, x_minus, ds_p, ds, u_minus;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
      // // Modified Stegmeir a
      // ds_p = 0.5 * sqrt(pow(coord->dx[i], 2) + pow(coord->dy[i], 2));
      // x_minus = ds_p * cos(atan2(b.y[i], b.x[i]));
      // y_minus = x_minus * b.y[i] / b.x[i];
      // n_x = static_cast<int>(floor(x_minus / coord->dx[i]));
      // n_y = static_cast<int>(floor(y_minus / coord->dy[i]));

      // Modified Stegmeir b
      x_minus = coord->dy[i] * abs(b.x[i] / b.y[i]);
      x_minus = std::min(static_cast<float>(coord->dx[i]), x_minus);
      y_minus = coord->dx[i] * abs(b.y[i] / b.x[i]);
      y_minus = std::min(static_cast<float>(coord->dy[i]), y_minus);
      if (b.x[i] >= 0)
      {
        n_x = 0;
      }
      else
      {
        n_x = -1;
        x_minus = -x_minus;
      }
      if (b.y[i] >= 0)
      {
        n_y = 0;
      }
      else
      {
        n_y = -1;
        y_minus = -y_minus;
      }
      ds_p = sqrt(pow(x_minus, 2) + pow(y_minus, 2));

      ds = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
      f_x = (x_minus - n_x * coord->dx[i]) / coord->dx[i];
      f_y = (y_minus - n_y * coord->dy[i]) / coord->dy[i];
      u_minus = (1.0 - f_y) * ((1 - f_x) * u(i.x() + n_x, i.y() + n_y, i.z()) + f_x * u(i.x() + n_x + 1, i.y() + n_y, i.z())) + f_y * ((1 - f_x) * u(i.x() + n_x, i.y() + n_y + 1, i.z()) + f_x * u(i.x() + n_x + 1, i.y() + n_y + 1, i.z()));
      result[i] = -(u_minus - u[i]) / ds;
    }

    return result;
  }

  Field3D stegmeir_div_q(const Field3D &T, const BoutReal &K_par, const BoutReal &K_perp, const Vector3D &b)
  {
    TRACE("stegmeir_div_q");

    Field3D result;
    Field3D ds;
    BoutReal dz;

    Coordinates *coord = mesh->getCoordinates();

    // // Naive
    // // ds = sqrt(pow(((coord->dx / coord->dy) * (b.y / b.x)) * coord->dy, 2.0) + pow(coord->dx, 2.0));
    // dz = (2.0 * pi * R_0 / a_mid) / 500.0;
    // ds.allocate();
    // for (auto i : ds)
    // {
    //   ds[i] = sqrt(pow(dz * tan(atan2(b.y[i], b.z[i])), 2.0) + pow(dz * tan(atan2(b.x[i], b.z[i])), 2.0) + pow(dz, 2.0));
    // }
    // result = (Q_plus(T, K_par, b) - Q_minus(T, K_par, b)) / ds;

    // Support method
    result = -0.5 * (Q_plus_T(Q_plus(T, K_par, b), b) + Q_minus_T(Q_minus(T, K_par, b), b));

    return result;
  }

  Field3D D3DX3(const Field3D &f)
  {
    TRACE("D3DX3");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
      // 2nd order
      // result[i] = (1.0 / (2.0 * pow(coord->dx[i], 3.0))) * (-f[i.xmm()] + 2.0 * f[i.xm()] - 2.0 * f[i.xp()] + f[i.xpp()]);

      // 4th order
      result[i] = (1.0 / (8.0 * pow(coord->dx[i], 3.0))) * (f[i.xm().xm().xm()] - 8.0 * f[i.xmm()] + 13.0 * f[i.xm()] - 13.0 * f[i.xp()] + 8.0 * f[i.xpp()] - f[i.xp().xp().xp()]);
    }

    return result;
  }

  Field3D D3DY3(const Field3D &f)
  {
    TRACE("D3DY3");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
      // 2nd order
      // result[i] = (1.0 / (2.0 * pow(coord->dy[i], 3.0))) * (-f[i.ymm()] + 2.0 * f[i.ym()] - 2.0 * f[i.yp()] + f[i.ypp()]);

      // 4th order
      result[i] = (1.0 / (8.0 * pow(coord->dy[i], 3.0))) * (f[i.ym().ym().ym()] - 8.0 * f[i.ymm()] + 13.0 * f[i.ym()] - 13.0 * f[i.yp()] + 8.0 * f[i.ypp()] - f[i.yp().yp().yp()]);
    }

    return result;
  }

  Field3D D3D2YDX(const Field3D &f)
  {
    TRACE("D3D2YDX");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
      // 2nd order
      // result[i] = (1.0 / (2.0 * pow(coord->dy[i], 2.0) * coord->dx[i])) * (f[i.xp().ym()] - f[i.xm().ym()] + f[i.xp().yp()] - f[i.xm().yp()] - 2.0 * f[i.xp()] + 2.0 * f[i.xm()]);

      // 4th order
      result[i] = (1.0 / (144.0 * pow(coord->dy[i], 2.0) * coord->dx[i])) * (-(-f[i.xpp().ypp()] + 8.0 * f[i.xp().ypp()] - 8.0 * f[i.xm().ypp()] + f[i.xmm().ypp()]) + 16.0 * (-f[i.xpp().yp()] + 8.0 * f[i.xp().yp()] - 8.0 * f[i.xm().yp()] + f[i.xmm().yp()]) - 30.0 * (-f[i.xpp()] + 8.0 * f[i.xp()] - 8.0 * f[i.xm()] + f[i.xmm()]) + 16.0 * (-f[i.xpp().ym()] + 8.0 * f[i.xp().ym()] - 8.0 * f[i.xm().ym()] + f[i.xmm().ym()]) - (-f[i.xpp().ymm()] + 8.0 * f[i.xp().ymm()] - 8.0 * f[i.xm().ymm()] + f[i.xmm().ymm()]));
    }

    return result;
  }

  Field3D D3D2XDY(const Field3D &f)
  {
    TRACE("D3D2XDY");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
      // 2nd order
      // result[i] = (1.0 / (2.0 * pow(coord->dx[i], 2.0) * coord->dy[i])) * (f[i.yp().xm()] - f[i.ym().xm()] + f[i.yp().xp()] - f[i.ym().xp()] - 2.0 * f[i.yp()] + 2.0 * f[i.ym()]);

      // 4th order
      result[i] = (1.0 / (144.0 * pow(coord->dx[i], 2.0) * coord->dy[i])) * (-(-f[i.ypp().xpp()] + 8.0 * f[i.yp().xpp()] - 8.0 * f[i.ym().xpp()] + f[i.ymm().xpp()]) + 16.0 * (-f[i.ypp().xp()] + 8.0 * f[i.yp().xp()] - 8.0 * f[i.ym().xp()] + f[i.ymm().xp()]) - 30.0 * (-f[i.ypp()] + 8.0 * f[i.yp()] - 8.0 * f[i.ym()] + f[i.ymm()]) + 16.0 * (-f[i.ypp().xm()] + 8.0 * f[i.yp().xm()] - 8.0 * f[i.ym().xm()] + f[i.ymm().xm()]) - (-f[i.ypp().xmm()] + 8.0 * f[i.yp().xmm()] - 8.0 * f[i.ym().xmm()] + f[i.ymm().xmm()]));
    }

    return result;
  }

  Field3D rotated_laplacexy(const Field3D &f)
  {
    TRACE("rotated_laplacexy");

    Field3D result, f_corners;

    Coordinates *coord = mesh->getCoordinates();

    f_corners.allocate();
    for (auto i : result)
    {
      f_corners[i] = 0.25 * (f[i.xm()] + f[i.xm().ym()] + f[i.ym()] + f[i]);
    }

    result.allocate();
    BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
    {
      result[i] = (2.0 / (12.0 * (pow(coord->dx[i], 2.0) + pow(coord->dy[i], 2.0)))) * (-f_corners[i.xpp().ypp()] + 16.0 * f_corners[i.xp().yp()] - 30.0 * f[i] + 16.0 * f_corners[i] - f_corners[i.xm().ym()]);
      result[i] += (2.0 / (12.0 * (pow(coord->dx[i], 2.0) + pow(coord->dy[i], 2.0)))) * (-f_corners[i.xm().ypp()] + 16.0 * f_corners[i.yp()] - 30.0 * f[i] + 16.0 * f_corners[i.xp()] - f_corners[i.xpp().ym()]);
    }

    return result;
  }

protected:
  int init(bool restarting) // TODO: Use the restart flag
  {

    /******************Reading options *****************/

    auto &globalOptions = Options::root();
    auto &options = globalOptions["model"];

    // Load system parameters
    chi_diff = options["chi_diff"].doc("Thermal diffusivity").withDefault(0.0);
    chi_par = options["chi_par"].doc("Parallel thermal conductivity").withDefault(0.0);
    chi_perp = options["chi_perp"].doc("Perpendicular thermal conductivity").withDefault(0.0);
    D_m = options["D_m"].doc("Magnetic diffusivity").withDefault(0.0);
    mu = options["mu"].doc("Kinematic viscosity").withDefault(0.0);
    R_0 = options["R_0"].doc("Major radius [m]").withDefault(1.5);
    a_mid = options["a_mid"].doc("Minor radius at outer midplane [m]").withDefault(0.6);
    n_sepx = options["n_sepx"].doc("Electron density at separatrix [m^-3]").withDefault(1.0e19);
    T_sepx = options["T_sepx"].doc("Plasma temperature at separatrix [eV]").withDefault(100.0);
    B_t0 = options["B_t0"].doc("Toroidal field strength [T]").withDefault(2.0);
    B_pmid = options["B_pmid"].doc("Poloidal field strength at outer midplane [T]").withDefault(0.25);
    T_down = options["T_down"].doc("Downstream fixed temperature [eV]").withDefault(10.0);

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
    use_sk9_anis_diffop = options["use_sk9_anis_diffop"]
                              .doc("Use SK9 stencil for the anisotropic heat flow operator")
                              .withDefault(false);
    fixed_T_down = options["fixed_T_down"]
                       .doc("Use a constant value for P on downstream boundaries")
                       .withDefault(false);
    T_dependent_q_par = options["T_dependent_q_par"]
                            .doc("Use Spitzer-Harm form of parallel conductivity")
                            .withDefault(false);
    use_symmetric_div_q_stencil = options["use_symmetric_div_q_stencil"]
                                      .doc("Use a symmetric stencil for the div_q term")
                                      .withDefault(true);

    // Constants
    m_i = options["m_i"].withDefault(2 * 1.667e-27);
    e = 1.602e-19;
    m_e = 9.11e-31;
    mu_0 = 1.256637e-6;
    eps_0 = 8.854188e-12;
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
    // beta_p = 1.0e-7 * 8.0 * pi * P_0 / pow(B_pmid, 2.0); // TODO: Double check units on this
    beta_p = 2.0 * mu_0 * P_0 / pow(B_pmid, 2.0); // TODO: Double check units on this
    P_grad_0 = P_0 / a_mid;

    // phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
    Options &phi_init_options = Options::root()["phi"];
    if (invert_laplace)
    {
      // TODO: Get LaplaceXY working as it is quicker
      //  phiSolver = bout::utils::make_unique<LaplaceXY>(mesh);
      phi = 0.0; // Starting guess for first solve (if iterative)
      phi_init_options.setConditionallyUsed();
      mm.A = 0.0;
      mm.D = 1.0;
      mm.ngcx_tot = ngcx_tot;
      mm.ngcy_tot = ngcy_tot;
      mm.nx_tot = nx_tot;
      mm.ny_tot = ny_tot;
      mm.nz_tot = nz_tot;
      mySolver.setOperatorFunction(mm);
      mySolver.setup();

      SOLVE_FOR(P, psi, omega);
      SAVE_REPEAT(u, phi, B);
    }
    else
    {
      phi.setBoundary("phi");

      SOLVE_FOR(P, psi, omega, phi);
      SAVE_REPEAT(u, B);
    }

    if (chi_par > 0.0)
    {
      if (T_dependent_q_par)
      {
        SAVE_REPEAT(q_par, tau_e, lambda_ei)
      }
      else
      {
        SAVE_REPEAT(q_par)
      }
    }
    if (chi_perp > 0.0)
    {
      SAVE_REPEAT(q_perp)
    }

    SAVE_REPEAT(div_q)

    // Set downstream pressure boundaries
    // TODO: Find out why this is needed instead of just setting bndry_xin=dirichlet, etc
    if (fixed_T_down)
    {
      // X boundaries
      if (mesh->firstX())
      {
        for (int ix = 0; ix < ngcx_tot; ix++)
        {
          for (int iy = 0; iy < mesh->LocalNy; iy++)
          {
            for (int iz = 0; iz < mesh->LocalNz; iz++)
            {
              P(ix, iy, iz) = T_down / T_sepx;
              // P(ix, iy, iz) = 0.0;
            }
          }
        }
      }
      if (mesh->lastX())
      {
        for (int ix = mesh->LocalNx - ngcx_tot; ix < mesh->LocalNx; ix++)
        {
          for (int iy = 0; iy < mesh->LocalNy; iy++)
          {
            for (int iz = 0; iz < mesh->LocalNz; iz++)
            {
              P(ix, iy, iz) = T_down / T_sepx;
              // P(ix, iy, iz) = 0.0;
            }
          }
        }
      }
      // Y boundaries
      for (itl.first(); !itl.isDone(); itl++)
      {
        // it.ind contains the x index
        for (int iy = 0; iy < ngcy_tot; iy++)
        {
          for (int iz = 0; iz < mesh->LocalNz; iz++)
          {
            P(itl.ind, iy, iz) = T_down / T_sepx;
            // P(itl.ind, iy, iz) = 0.0;
          }
        }
      }
      // for (itu.first(); !itu.isDone(); itu++)
      // {
      //   for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
      //   {
      //     for (int iz = 0; iz < mesh->LocalNz; iz++)
      //     {
      //       // P(itu.ind, iy, iz) = T_down / T_sepx;
      //       P(itl.ind, iy, iz) = 0.0;
      //     }
      //   }
      // }
    }

    // Initialise unit vector in z direction
    e_z.x = 0.0;
    e_z.y = 0.0;
    e_z.z = 1.0;

    // Initialise poloidal B field
    B.x = 0.0;
    B.y = 0.0;
    B.z = B_t0 / B_pmid;
    if (use_sk9_anis_diffop)
    {
      phi_m = 0.0;
      theta_m = 0.0;
    }

    // Initialise heat flow
    q_par = 0.0;
    q_perp = 0.0;
    kappa_par = 0.0;
    kappa_perp = 0.0;
    lambda_ei = 0.0;
    tau_e = 0.0;
    div_q = 0.0;

    // Output constants, input options and derived parameters
    SAVE_ONCE(e, m_i, m_e, chi_diff, D_m, mu, epsilon, beta_p, rho, P_0);
    SAVE_ONCE(C_s0, t_0, D_0, psi_0, phi_0, R_0, a_mid, n_sepx);
    SAVE_ONCE(T_sepx, B_t0, B_pmid, evolve_pressure, include_churn_drive_term, include_mag_restoring_term, P_grad_0);
    SAVE_ONCE(ngcx, ngcx_tot, ngcy, ngcy_tot, chi_perp, chi_par);

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

    // Solve phi
    ////////////////////////////////////////////////////////////////////////////
    if (invert_laplace)
    {
      mesh->communicate(P, psi, omega);
      phi = mySolver.invert(omega);
      try
      {
        for (int i = 0; i < nits_inv_extra; i++)
        {
          phi = mySolver.invert(omega, phi);
          mesh->communicate(phi);
        }
      }
      catch (BoutException &e)
      {
      };
    }
    else
    {
      mesh->communicate(P, psi, omega, phi);
      ddt(phi) = (D2DX2(phi) + D2DY2(phi)) - omega;
    }
    mesh->communicate(phi);

    // Calculate velocity
    u = -cross(e_z, Grad(phi));

    // mesh->communicate(u);

    // // Calculate B
    B.x = -DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B.y = DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL");
    B_mag = abs(B);
    if (use_sk9_anis_diffop)
    {
      BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
      {
        theta_m[i] = atan2(sqrt(pow(B.x[i], 2.0) + pow(B.y[i], 2.0)), B.z[i]);
        phi_m[i] = atan2(B.y[i], B.x[i]);
      }
    }

    // mesh->communicate(B);

    // Get T
    T = P; // Normalised T = normalised P when rho = const

    // Pressure Evolution
    /////////////////////////////////////////////////////////////////////////////
    if (evolve_pressure)
    {
      if (include_advection)
      {
        if (mesh->StaggerGrids)
        {
          ddt(P) = -V_dot_Grad(u, P);
        }
        else
        {
          ddt(P) = -(DDX(P) * u.x + u.y * DDY(P));
        }
      }
      else
      {
        ddt(P) = 0;
      }
      if (chi_diff > 0.0)
      {
        ddt(P) += (chi_diff / D_0) * (D2DX2(P) + D2DY2(P));
      }
      else if ((chi_par > 0.0) || (chi_perp > 0.0))
      {
        // TODO: Add T-dependent q_par
        if (use_symmetric_div_q_stencil)
        {
          ddt(P) += div_q_symmetric(T, chi_par / D_0, chi_perp / D_0, B / B_mag);
        }
        else if (use_sk9_anis_diffop)
        {
          ddt(P) += (chi_par / D_0) * Anis_Diff_SK9(P,
                                                    pow(sin(theta_m), 2.0) * pow(cos(phi_m), 2.0),
                                                    pow(sin(theta_m), 2.0) * pow(sin(phi_m), 2.0),
                                                    pow(sin(theta_m), 2.0) * cos(phi_m) * sin(phi_m));
          ddt(P) += (chi_perp / D_0) * Anis_Diff_SK9(P,
                                                     pow(cos(theta_m), 2.0) * pow(cos(phi_m), 2.0) + pow(sin(phi_m), 2.0),
                                                     pow(cos(theta_m), 2.0) * pow(sin(phi_m), 2.0) + pow(cos(phi_m), 2.0),
                                                     (-sin(phi_m) * cos(phi_m) * (pow(cos(theta_m), 2.0) + 1.0)));
        }
        else
        {
          // ddt(P) += (chi_par / D_0) * (DDX(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(B.y * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")) / pow(B_mag, 2);
          // ddt(P) += (chi_perp / D_0) * (D2DX2(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(T, CELL_CENTER, "DEFAULT", "RGN_ALL") - (DDX(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")) / pow(B_mag, 2));
          // div_q = (chi_par / D_0) * (DDX(B.x * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(B.y * (B.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + B.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")) / pow(B_mag, 2);
          // ddt(P) += stegmeir_div_q(T, chi_par / D_0, chi_perp / D_0, B / B_mag);
          // div_q = stegmeir_div_q(T, chi_par / D_0, chi_perp / D_0, B / B_mag);
          // div_q = Q_plus(T, chi_par / D_0, B / B_mag);
          // div_q = Q_minus(T, chi_par / D_0, B / B_mag);

          div_q = Q_linetrace(T, chi_par / D_0, B / B_mag);
          ddt(P) += div_q;
          // div_q = Q_linetrace(T, chi_par / D_0, B / B_mag);
        }
        // Calculate q for output
        q_par = -(chi_par / D_0) * B * (B * Grad(T)) / pow(B_mag, 2);
        q_perp = -(chi_perp / D_0) * (Grad(T) - B * (B * Grad(T)) / pow(B_mag, 2));
      }
    }

    // Psi evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
      if (mesh->StaggerGrids)
      {
        ddt(psi) = -V_dot_Grad(u, psi);
      }
      else
      {
        ddt(psi) = -(DDX(psi) * u.x + u.y * DDY(psi));
      }
    }
    else
    {
      ddt(psi) = 0;
    }
    ddt(psi) += (D_m / D_0) * (D2DX2(psi) + D2DY2(psi));

    // Vorticity evolution
    /////////////////////////////////////////////////////////////////////////////

    if (include_advection)
    {
      if (mesh->StaggerGrids)
      {
        ddt(omega) = -V_dot_Grad(u, omega);
      }
      else
      {
        ddt(omega) = -(DDX(omega) * u.x + u.y * DDY(omega));
      }
    }
    else
    {
      ddt(omega) = 0;
    }
    ddt(omega) += (mu / D_0) * (D2DX2(omega) + D2DY2(omega));
    if (include_churn_drive_term)
    {
      ddt(omega) += 2.0 * epsilon * DDY(P, CELL_CENTER, "DEFAULT", "RGN_ALL");
    }
    if (include_mag_restoring_term)
    {
      // Basic approach
      ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDY(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL"), CELL_CENTER, "DEFAULT", "RGN_ALL") - DDY(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") * DDX(D2DX2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(psi, CELL_CENTER, "DEFAULT", "RGN_ALL"), CELL_CENTER, "DEFAULT", "RGN_ALL"));

      // Using 3rd derivative stencils
      // ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi) * (D3D2XDY(psi) + D3DY3(psi)) - DDY(psi) * (D3D2YDX(psi) + D3DX3(psi)));

      // Using a rotated Laplacian stencil
      // ddt(omega) += -(2.0 / (beta_p)) * (DDX(psi) * DDY(rotated_laplacexy(psi)) - DDY(psi) * DDX(rotated_laplacexy(psi)));
    }

    // Apply ddt = 0 BCs
    // X boundaries
    if (mesh->firstX())
    {
      for (int ix = 0; ix < ngcx_tot; ix++)
      {
        for (int iy = 0; iy < mesh->LocalNy; iy++)
        {
          for (int iz = 0; iz < mesh->LocalNz; iz++)
          {
            ddt(omega)(ix, iy, iz) = 0.0;
            ddt(psi)(ix, iy, iz) = 0.0;
            ddt(P)(ix, iy, iz) = 0.0;
            if (invert_laplace == false)
            {
              ddt(phi)(ix, iy, iz) = 0.0;
            }
          }
        }
      }
    }
    if (mesh->lastX())
    {
      for (int ix = mesh->LocalNx - ngcx_tot; ix < mesh->LocalNx; ix++)
      {
        for (int iy = 0; iy < mesh->LocalNy; iy++)
        {
          for (int iz = 0; iz < mesh->LocalNz; iz++)
          {
            ddt(omega)(ix, iy, iz) = 0.0;
            ddt(psi)(ix, iy, iz) = 0.0;
            ddt(P)(ix, iy, iz) = 0.0;
            if (invert_laplace == false)
            {
              ddt(phi)(ix, iy, iz) = 0.0;
            }
          }
        }
      }
    }
    // Y boundaries
    for (itl.first(); !itl.isDone(); itl++)
    {
      // it.ind contains the x index
      for (int iy = 0; iy < ngcy_tot; iy++)
      {
        for (int iz = 0; iz < mesh->LocalNz; iz++)
        {
          ddt(omega)(itl.ind, iy, iz) = 0.0;
          ddt(psi)(itl.ind, iy, iz) = 0.0;
          ddt(P)(itl.ind, iy, iz) = 0.0;
          if (invert_laplace == false)
          {
            ddt(phi)(itl.ind, iy, iz) = 0.0;
          }
        }
      }
    }
    for (itu.first(); !itu.isDone(); itu++)
    {
      // it.ind contains the x index
      for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
      {
        for (int iz = 0; iz < mesh->LocalNz; iz++)
        {
          ddt(omega)(itu.ind, iy, iz) = 0.0;
          ddt(psi)(itu.ind, iy, iz) = 0.0;
          ddt(P)(itu.ind, iy, iz) = 0.0;
          if (invert_laplace == false)
          {
            ddt(phi)(itu.ind, iy, iz) = 0.0;
          }
        }
      }
    }

    return 0;
  }
};

// Define a standard main() function
BOUTMAIN(Churn);
