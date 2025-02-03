#include "header.hxx"

Field3D Churn::div_q_par_classic(const Field3D &T, const Field3D &K_par, const Vector3D &b)
{
    // Classic stencil for parallel heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_par_classic");

    // Field3D result;
    // result = D2DX2_DIFF(T, K_par * pow(b.x, 2.0)) + D2DY2_DIFF(T, K_par * pow(b.y, 2.0)) + DDX(K_par * b.x * b.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")) + DDY(K_par * b.x * b.y * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL"));
    // return result;

    Field3D result;
    BoutReal A_plus_half, A_minus_half, ddy_plus, ddy_minus, ddx_plus, ddx_minus;
    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    // D2DX2 term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        A_plus_half = (0.5 * (K_par[i] * (pow(b.x[i], 2.0)) + K_par[i.xp()] * (pow(b.x[i.xp()], 2.0))));
        A_minus_half = (0.5 * (K_par[i] * (pow(b.x[i], 2.0)) + K_par[i.xm()] * (pow(b.x[i.xm()], 2.0))));
        result[i] += (1.0 / (pow(coord->dx[i], 2.0))) * (A_plus_half * (T[i.xp()] - T[i]) - A_minus_half * (T[i] - T[i.xm()]));
    }

    // D2DY2 term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        A_plus_half = (0.5 * (K_par[i] * (pow(b.y[i], 2.0)) + K_par[i.yp()] * (pow(b.y[i.yp()], 2.0))));
        A_minus_half = (0.5 * (K_par[i] * (pow(b.y[i], 2.0)) + K_par[i.ym()] * (pow(b.y[i.ym()], 2.0))));

        // Apply grad_perp P = 0 BC if using fixed Q_in
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
            {
                if (fixed_Q_in)
                {
                    A_plus_half = 0.0;
                }
            }
        }

        result[i] += (1.0 / (pow(coord->dy[i], 2.0))) * (A_plus_half * (T[i.yp()] - T[i]) - A_minus_half * (T[i] - T[i.ym()]));
    }

    // DXDY term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        ddy_plus = (0.5 / (coord->dy[i])) * (T[i.xp().yp()] - T[i.xp().ym()]);
        ddy_minus = (0.5 / (coord->dy[i])) * (T[i.xm().yp()] - T[i.xm().ym()]);

        // Apply grad_perp P = 0 BC if using fixed Q_in
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
            {
                if (fixed_Q_in)
                {
                    ddy_plus = 0.0;
                    ddy_minus = 0.0;
                }
            }
        }

        result[i] += (0.5 / (coord->dx[i])) * (K_par[i.xp()] * b.x[i.xp()] * b.y[i.xp()] * ddy_plus - K_par[i.xm()] * b.x[i.xm()] * b.y[i.xm()] * ddy_minus);
    }

    // DYDX term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        ddx_plus = (0.5 / (coord->dx[i])) * (T[i.yp().xp()] - T[i.yp().xm()]);
        ddx_minus = (0.5 / (coord->dx[i])) * (T[i.ym().xp()] - T[i.ym().xm()]);

        // Apply grad_perp P = 0 BC if using fixed Q_in
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
            {
                if (fixed_Q_in)
                {
                    ddx_plus = 0.0;
                }
            }
        }

        result[i] += (0.5 / (coord->dy[i])) * (K_par[i.yp()] * b.x[i.yp()] * b.y[i.yp()] * ddx_plus - K_par[i.ym()] * b.x[i.ym()] * b.y[i.ym()] * ddx_minus);
    }

    return result;
}

Field3D Churn::div_q_perp_classic(const Field3D &T, const Field3D &K_perp, const Vector3D &b)
{
    // Classic stencil for perpendicular heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_perp_classic");

    // Field3D result;
    // result = D2DX2_DIFF(T, K_perp * (1.0 - pow(b.x, 2.0))) + D2DY2_DIFF(T, K_perp * (1.0 - pow(b.y, 2.0))) - DDX(K_perp * b.x * b.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")) - DDY(K_perp * b.x * b.y * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL"));
    // return result;

    Field3D result;
    BoutReal A_plus_half, A_minus_half, ddy_plus, ddy_minus, ddx_plus, ddx_minus;
    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;

    // TODO: Could handle the BCs here by just setting dP/dy=0 in upper boundary cells and using the diffops functions instead

    // D2DX2 term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        A_plus_half = (0.5 * (K_perp[i] * (1.0 - pow(b.x[i], 2.0)) + K_perp[i.xp()] * (1.0 - pow(b.x[i.xp()], 2.0))));
        A_minus_half = (0.5 * (K_perp[i] * (1.0 - pow(b.x[i], 2.0)) + K_perp[i.xm()] * (1.0 - pow(b.x[i.xm()], 2.0))));
        result[i] += (1.0 / (pow(coord->dx[i], 2.0))) * (A_plus_half * (T[i.xp()] - T[i]) - A_minus_half * (T[i] - T[i.xm()]));
    }

    // D2DY2 term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        A_plus_half = (0.5 * (K_perp[i] * (1.0 - pow(b.y[i], 2.0)) + K_perp[i.yp()] * (1.0 - pow(b.y[i.yp()], 2.0))));
        A_minus_half = (0.5 * (K_perp[i] * (1.0 - pow(b.y[i], 2.0)) + K_perp[i.ym()] * (1.0 - pow(b.y[i.ym()], 2.0))));

        // Apply grad_perp P = 0 BC if using fixed Q_in
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
            {
                if (fixed_Q_in)
                {
                    A_plus_half = 0.0;
                }
            }
        }

        result[i] += (1.0 / (pow(coord->dy[i], 2.0))) * (A_plus_half * (T[i.yp()] - T[i]) - A_minus_half * (T[i] - T[i.ym()]));
    }

    // DXDY term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        ddy_plus = (0.5 / (coord->dy[i])) * (T[i.xp().yp()] - T[i.xp().ym()]);
        ddy_minus = (0.5 / (coord->dy[i])) * (T[i.xm().yp()] - T[i.xm().ym()]);

        // Apply grad_perp P = 0 BC if using fixed Q_in
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
            {
                if (fixed_Q_in)
                {
                    ddy_plus = 0.0;
                    ddy_minus = 0.0;
                }
            }
        }

        result[i] -= (0.5 / (coord->dx[i])) * (K_perp[i.xp()] * b.x[i.xp()] * b.y[i.xp()] * ddy_plus - K_perp[i.xm()] * b.x[i.xm()] * b.y[i.xm()] * ddy_minus);
    }

    // DYDX term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        ddx_plus = (0.5 / (coord->dx[i])) * (T[i.yp().xp()] - T[i.yp().xm()]);
        ddx_minus = (0.5 / (coord->dx[i])) * (T[i.ym().xp()] - T[i.ym().xm()]);

        // Apply grad_perp P = 0 BC if using fixed Q_in
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
            {
                if (fixed_Q_in)
                {
                    ddx_plus = 0.0;
                }
            }
        }

        result[i] -= (0.5 / (coord->dy[i])) * (K_perp[i.yp()] * b.x[i.yp()] * b.y[i.yp()] * ddx_plus - K_perp[i.ym()] * b.x[i.ym()] * b.y[i.ym()] * ddx_minus);
    }

    return result;
}

Field3D Churn::div_q_par_gunter(const Field3D &T, const Field3D &K_par, const Vector3D &b)
{
    // Gunter stencil for parallel heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_par_gunter");

    Field3D result;
    Field3D bx_corners, by_corners, K_par_corners, DTDX_corners, DTDY_corners, q_parx_corners, q_pary_corners, q_perpx_corners, q_perpy_corners;

    Coordinates *coord = mesh->getCoordinates();

    // TODO: Check below is valid when dx!=dy
    bx_corners.allocate();
    by_corners.allocate();
    K_par_corners.allocate();
    for (auto i : result)
    {
        bx_corners[i] = 0.25 * (b.x[i.xm()] + b.x[i.xm().ym()] + b.x[i.ym()] + b.x[i]);
        by_corners[i] = 0.25 * (b.y[i.xm()] + b.y[i.xm().ym()] + b.y[i.ym()] + b.y[i]);
        K_par_corners[i] = 0.25 * (K_par[i.xm()] + K_par[i.xm().ym()] + K_par[i.ym()] + K_par[i]);
    }

    // Find temperature gradients on cell corners
    DTDX_corners.allocate();
    DTDY_corners.allocate();
    for (auto i : DTDX_corners)
    {

        DTDX_corners[i] = (1.0 / (2.0 * coord->dx[i])) * ((T[i] + T[i.ym()]) - (T[i.xm()] + T[i.xm().ym()]));
        DTDY_corners[i] = (1.0 / (2.0 * coord->dy[i])) * ((T[i] + T[i.xm()]) - (T[i.ym()] + T[i.xm().ym()]));
    }

    q_parx_corners = K_par_corners * bx_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners);
    q_pary_corners = K_par_corners * by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners);

    result.allocate();
    // for (auto i : result)
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        result[i] = (1.0 / (2.0 * coord->dx[i])) * (q_parx_corners[i.xp().yp()] + q_parx_corners[i.xp()] - q_parx_corners[i.yp()] - q_parx_corners[i]);
        result[i] += (1.0 / (2.0 * coord->dy[i])) * (q_pary_corners[i.xp().yp()] + q_pary_corners[i.yp()] - q_pary_corners[i.xp()] - q_pary_corners[i]);
    }

    return result;
}

Field3D Churn::div_q_perp_gunter(const Field3D &T, const Field3D &K_perp, const Vector3D &b)
{
    // Gunter stencil for perpendicular heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_perp_gunter");

    Field3D result;
    Field3D bx_corners, by_corners, K_perp_corners, DTDX_corners, DTDY_corners, q_perpx_corners, q_perpy_corners;

    Coordinates *coord = mesh->getCoordinates();

    // TODO: Check below is valid when dx!=dy
    bx_corners.allocate();
    by_corners.allocate();
    K_perp_corners.allocate();
    for (auto i : result)
    {
        bx_corners[i] = 0.25 * (b.x[i.xm()] + b.x[i.xm().ym()] + b.x[i.ym()] + b.x[i]);
        by_corners[i] = 0.25 * (b.y[i.xm()] + b.y[i.xm().ym()] + b.y[i.ym()] + b.y[i]);
        K_perp_corners[i] = 0.25 * (K_perp[i.xm()] + K_perp[i.xm().ym()] + K_perp[i.ym()] + K_perp[i]);
    }

    // Find temperature gradients on cell corners
    DTDX_corners.allocate();
    DTDY_corners.allocate();
    for (auto i : DTDX_corners)
    {

        DTDX_corners[i] = (1.0 / (2.0 * coord->dx[i])) * ((T[i] + T[i.ym()]) - (T[i.xm()] + T[i.xm().ym()]));
        DTDY_corners[i] = (1.0 / (2.0 * coord->dy[i])) * ((T[i] + T[i.xm()]) - (T[i.ym()] + T[i.xm().ym()]));
    }
    q_perpx_corners = K_perp_corners * (DTDX_corners - bx_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners));
    q_perpy_corners = K_perp_corners * (DTDY_corners - by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners));

    // Apply q_perp=0 BC on upper y boundary
    if (fixed_Q_in)
    {
        int k = 0;
        for (int i = 0; i < mesh->LocalNx; i++)
        {
            if (mesh->lastY(i))
            {
                for (int j = mesh->LocalNy - ngcy_tot; j < mesh->LocalNy; j++)
                {
                    q_perpy_corners(i, j, k) = 0.0;
                }
            }
        }
    }

    result.allocate();
    // for (auto i : result)
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        result[i] = (1.0 / (2.0 * coord->dx[i])) * (q_perpx_corners[i.xp().yp()] + q_perpx_corners[i.xp()] - q_perpx_corners[i.yp()] - q_perpx_corners[i]);
        result[i] += (1.0 / (2.0 * coord->dy[i])) * (q_perpy_corners[i.xp().yp()] + q_perpy_corners[i.yp()] - q_perpy_corners[i.xp()] - q_perpy_corners[i]);
    }

    return result;
}

TwoIntersects Churn::get_intersects(const double &xlo, const double &xhi, const double &ylo, const double &yhi, const CellIntersect &P, const double &bx, const double &by)
{
    // Find the intersection points between a line with gradient given by by/bx, where (Px,Py) is a point on the line, and the box bounded by xlo, xhi, ylo, yhi
    // TODO: Check edge cases when intersection is on a corner
    TRACE("intersects_plus");

    // std::vector<std::vector<double>> result = {{0.0, 1.0}, {2.0, 3.0}};
    TwoIntersects result;
    CellIntersect intersect;
    double m, c, y_xlo, y_xhi, x_ylo, x_yhi;

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
        result.first = intersect;
    }
    if ((ylo < y_xhi) && (y_xhi <= yhi))
    {
        // Intersect upper x face
        intersect.face = 2;
        intersect.x = xhi;
        intersect.y = y_xhi;
        result.second = intersect;
    }
    if ((xlo < x_ylo) && (x_ylo <= xhi))
    {
        // Intersect lower y face
        intersect.face = 3;
        intersect.x = x_ylo;
        intersect.y = ylo;
        result.first = intersect;
    }
    if ((xlo <= x_yhi) && (x_yhi < xhi))
    {
        // Intersect upper y face
        intersect.face = 1;
        intersect.x = x_yhi;
        intersect.y = yhi;
        result.second = intersect;
    }

    return result;
}

CellIntersect Churn::get_next_intersect(const double &xlo, const double &xhi, const double &ylo, const double &yhi, const CellIntersect &prev_intersect, const double &bx, const double &by)
{
    // Find the intersection points between a line with gradient given by by/bx, where (Px,Py) is a point on the line, and the box bounded by xlo, xhi, ylo, yhi
    TRACE("get_next_intersect");

    // std::vector<std::vector<double>> result = {{0.0, 1.0}, {2.0, 3.0}};
    CellIntersect result;
    double m, c, y_xlo, y_xhi, x_ylo, x_yhi;

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

Ind3D Churn::increment_cell(const Ind3D &i, const Ind3D &i_prev, const CellIntersect &P_next, const double &dx, const double &dy)
{
    // Determine which cell to move to next
    TRACE("increment_cell");

    Ind3D i_next;
    double tol = 1.0e-6;
    int x_inc, y_inc;

    i_next = i_prev;
    x_inc = static_cast<double>(i_prev.x() - i.x());
    y_inc = static_cast<double>(i_prev.y() - i.y());

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

InterpolationPoint Churn::trace_field_lines(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_x_inc, const int &max_y_inc, const int &max_steps, const bool &plus)
{
    InterpolationPoint result;
    CellIntersect next_intersect, prev_intersect;
    TwoIntersects intersects;
    double par_dist, par_dist_closest;
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
        if (intersects.second.x > 0.0)
        {
            next_intersect = intersects.second;
        }
        else
        {
            next_intersect = intersects.first;
        }
    }
    else
    {
        if (intersects.second.x < 0.0)
        {
            next_intersect = intersects.second;
        }
        else
        {
            next_intersect = intersects.first;
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
    n_steps = 0;
    while ((abs(i_next.x() - i.x()) < max_x_inc - 1) && (abs(i_next.y() - i.y()) < max_y_inc - 1) && (n_steps < max_steps))
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
    }

    result.x = p_closest.x;
    result.y = p_closest.y;
    result.distance = p_closest.distance;
    result.parallel_distance = par_dist_closest;

    return result;
}

ClosestPoint Churn::get_closest_p(const CellIntersect &P, const Point &P0, const double &bx, const double &by)
{
    // Find the closest point on the line following the magnetic field, extending from an cell face intersection point (Px,Py), to a point at (x0, y0), e.g. a cell centre. Output is a vector with three elements (x, y, distance)
    TRACE("get_closest_p");

    ClosestPoint result;
    double distance, x_closest, y_closest;
    double A, B, C;

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

Field3D Churn::div_q_par_linetrace(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
{
    TRACE("div_q_par_linetrace");

    Field3D result;
    Field3D x_plus, y_plus, x_minus, y_minus, parallel_distances_plus, parallel_distances_minus, q_plus, q_minus;
    Coordinates *coord = mesh->getCoordinates();
    InterpolationPoint interp_p_plus, interp_p_minus;
    int n_steps, n_x, n_y;
    double f_x, f_y, u_plus, q_plus_T, u_minus, q_minus_T, div_q_plus, div_q_minus;
    int max_x_inc = 3;
    int max_y_inc = 3;
    int max_steps = 12;
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
        div_q_plus = (parallel_distances_plus[i] / (0.5 * (parallel_distances_plus[i] + parallel_distances_minus[i]))) * ((q_plus_T - q_plus[i]) / parallel_distances_plus[i]);

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

Field3D Churn::div_q_par_linetrace2(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
{
    TRACE("div_q_par_linetrace2");

    Ind3D i_offset;
    int n_steps, n_x, n_y;
    double f_x, f_y, u_plus, q_plus_T, u_minus, q_minus_T, div_q_plus, div_q_minus;
    Coordinates *coord = mesh->getCoordinates();
    Field3D result;
    bool continue_tracing;
    Ind3D i_prev, i_plus, i_minus;
    TwoIntersects intersects;
    CellIntersect intersect, intersect_plus, intersect_minus, prev_intersect;
    double m, c, y_xlo, y_xhi, x_ylo, x_yhi, xlo, xhi, ylo, yhi, tracebox_width, tracebox_height;
    double tol = 1.0e-6;
    Field3D x_plus, y_plus, x_minus, y_minus, parallel_distances_plus, parallel_distances_minus, q_plus, q_minus;
    double prev_x, prev_y;
    int prev_face;

    prev_intersect.x = 0.0;
    prev_intersect.y = 0.0;
    prev_intersect.face = 0;
    intersect_plus.x = 0.0;
    intersect_plus.y = 0.0;
    intersect_plus.face = 0;
    intersect_minus.x = 0.0;
    intersect_minus.y = 0.0;
    intersect_minus.face = 0;

    // Find the intersects of field lines with the trace box
    result = 0.0;
    x_plus = 0.0;
    y_plus = 0.0;
    x_minus = 0.0;
    y_minus = 0.0;
    parallel_distances_plus = 1.0e-10;
    parallel_distances_minus = 1.0e-10;
    BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
    {

        tracebox_width = coord->dx[i];
        tracebox_height = coord->dy[i];

        // Get intersect with current cell face
        xlo = -coord->dx[i] / 2.0;
        xhi = coord->dx[i] / 2.0;
        ylo = -coord->dy[i] / 2.0;
        yhi = coord->dy[i] / 2.0;
        // Find line equation for b
        if (b.x[i] == 0.0)
        {
            m = 1e12;
        }
        else
        {
            m = b.y[i] / b.x[i];
        }
        // c = P.y - m * P.x;
        c = 0.0;

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
            intersects.first = intersect;
        }
        if ((ylo < y_xhi) && (y_xhi <= yhi))
        {
            // Intersect upper x face
            intersect.face = 2;
            intersect.x = xhi;
            intersect.y = y_xhi;
            intersects.second = intersect;
        }
        if ((xlo < x_ylo) && (x_ylo <= xhi))
        {
            // Intersect lower y face
            intersect.face = 3;
            intersect.x = x_ylo;
            intersect.y = ylo;
            intersects.first = intersect;
        }
        if ((xlo <= x_yhi) && (x_yhi < xhi))
        {
            // Intersect upper y face
            intersect.face = 1;
            intersect.x = x_yhi;
            intersect.y = yhi;
            intersects.second = intersect;
        }

        ///////// PLUS ////////////
        // Initialise first intersect and cell increment
        intersect_plus = intersects.second;
        i_plus = i;
        parallel_distances_plus[i] = sqrt(pow(intersect_plus.x, 2.0) + pow(intersect_plus.y, 2.0)) * sqrt(pow(b.z[i_plus], 2) / (pow(b.x[i_plus], 2) + pow(b.y[i_plus], 2)) + 1.0);
        // result[i] = intersect_plus.face;

        // Trace each intersect the the edge of the box extending + / -dx, + / -dy from current cell
        continue_tracing = true;
        n_steps = 1;
        while (continue_tracing)
        {
            // Determine which cell we're moving into
            i_prev = i_plus;
            i_plus = increment_cell(i, i_prev, intersect_plus, coord->dx[i], coord->dy[i]);

            xlo = (i_plus.x() - i_prev.x()) * coord->dx[i] - coord->dx[i] / 2.0;
            xlo = std::max(static_cast<double>(xlo), static_cast<double>(-coord->dx[i]));
            xhi = (i_plus.x() - i_prev.x()) * coord->dx[i] + coord->dx[i] / 2.0;
            xhi = std::min(static_cast<double>(xhi), static_cast<double>(coord->dx[i]));
            ylo = (i_plus.y() - i_prev.y()) * coord->dy[i] - coord->dy[i] / 2.0;
            ylo = std::max(static_cast<double>(ylo), static_cast<double>(-coord->dy[i]));
            yhi = (i_plus.y() - i_prev.y()) * coord->dy[i] + coord->dy[i] / 2.0;
            yhi = std::min(static_cast<double>(yhi), static_cast<double>(coord->dy[i]));

            // Find the next intersect
            prev_intersect = intersect_plus;
            // intersect_plus = get_next_intersect(xlo,
            //                                     xhi,
            //                                     ylo,
            //                                     yhi,
            //                                     prev_intersect,
            //                                     b.x[i_plus],
            //                                     b.y[i_plus]);
            /*******************/
            // Find line equation for b
            if (b.x[i_plus] == 0.0)
            {
                m = 1e12;
            }
            else
            {
                m = b.y[i_plus] / b.x[i_plus];
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
                intersect_plus.face = 0;
                intersect_plus.x = xlo;
                intersect_plus.y = y_xlo;
            }

            if ((ylo < y_xhi) && (y_xhi <= yhi) && (prev_intersect.face != 0))
            {
                // Intersect upper x face
                intersect_plus.face = 2;
                intersect_plus.x = xhi;
                intersect_plus.y = y_xhi;
            }

            if ((xlo < x_ylo) && (x_ylo <= xhi) && (prev_intersect.face != 1))
            {
                // Intersect lower y face
                intersect_plus.face = 3;
                intersect_plus.x = x_ylo;
                intersect_plus.y = ylo;
            }

            if ((xlo <= x_yhi) && (x_yhi < xhi) && (prev_intersect.face != 3))
            {
                // Intersect upper y face
                intersect_plus.face = 1;
                intersect_plus.x = x_yhi;
                intersect_plus.y = yhi;
            }
            /*******************/

            // Increment the parallel distance
            parallel_distances_plus[i] += sqrt(pow(intersect_plus.x - prev_intersect.x, 2.0) + pow(intersect_plus.y - prev_intersect.y, 2.0)) * sqrt(pow(b.z[i_plus], 2) / (pow(b.x[i_plus], 2) + pow(b.y[i_plus], 2)) + 1.0);

            // Check if we've reached the edge of the intersection region
            if (abs(abs(intersect_plus.x) - tracebox_width) < tol)
            {
                continue_tracing = false;
            }
            else if (abs(abs(intersect_plus.y) - tracebox_height) < tol)
            {
                continue_tracing = false;
            }
            if (n_steps >= 1)
            {
                continue_tracing = false;
            }
            n_steps++;
            // continue_tracing = false;
        }
        // Store information from the final intersection point
        x_plus[i] = intersect_plus.x;
        y_plus[i] = intersect_plus.y;

        ///////// MINUS ////////////
        // Initialise first intersect and cell increment
        intersect_minus = intersects.first;
        i_minus = i;
        parallel_distances_minus[i] = sqrt(pow(intersect_minus.x, 2.0) + pow(intersect_minus.y, 2.0)) * sqrt(pow(b.z[i_minus], 2) / (pow(b.x[i_minus], 2) + pow(b.y[i_minus], 2)) + 1.0);

        // Trace each intersect the the edge of the box extending + / -dx, + / -dy from current cell
        continue_tracing = true;
        n_steps = 1;
        while (continue_tracing)
        {
            // Determine which cell we're moving into
            i_prev = i_minus;
            i_minus = increment_cell(i, i_prev, intersect_minus, coord->dx[i], coord->dy[i]);

            xlo = (i_minus.x() - i_prev.x()) * coord->dx[i] - coord->dx[i] / 2.0;
            xlo = std::max(static_cast<double>(xlo), static_cast<double>(-coord->dx[i]));
            xhi = (i_minus.x() - i_prev.x()) * coord->dx[i] + coord->dx[i] / 2.0;
            xhi = std::min(static_cast<double>(xhi), static_cast<double>(coord->dx[i]));
            ylo = (i_minus.y() - i_prev.y()) * coord->dy[i] - coord->dy[i] / 2.0;
            ylo = std::max(static_cast<double>(ylo), static_cast<double>(-coord->dy[i]));
            yhi = (i_minus.y() - i_prev.y()) * coord->dy[i] + coord->dy[i] / 2.0;
            yhi = std::min(static_cast<double>(yhi), static_cast<double>(coord->dy[i]));

            // Find the next intersect
            prev_intersect = intersect_minus;
            /*******************/
            // Find line equation for b
            if (b.x[i_minus] == 0.0)
            {
                m = 1e12;
            }
            else
            {
                m = b.y[i_minus] / b.x[i_minus];
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
                intersect_minus.face = 0;
                intersect_minus.x = xlo;
                intersect_minus.y = y_xlo;
            }

            if ((ylo < y_xhi) && (y_xhi <= yhi) && (prev_intersect.face != 0))
            {
                // Intersect upper x face
                intersect_minus.face = 2;
                intersect_minus.x = xhi;
                intersect_minus.y = y_xhi;
            }

            if ((xlo < x_ylo) && (x_ylo <= xhi) && (prev_intersect.face != 1))
            {
                // Intersect lower y face
                intersect_minus.face = 3;
                intersect_minus.x = x_ylo;
                intersect_minus.y = ylo;
            }

            if ((xlo <= x_yhi) && (x_yhi < xhi) && (prev_intersect.face != 3))
            {
                // Intersect upper y face
                intersect_minus.face = 1;
                intersect_minus.x = x_yhi;
                intersect_minus.y = yhi;
            }
            /*******************/

            // Increment the parallel distance
            parallel_distances_minus[i] += sqrt(pow(intersect_minus.x - prev_intersect.x, 2.0) + pow(intersect_minus.y - prev_intersect.y, 2.0)) * sqrt(pow(b.z[i_minus], 2) / (pow(b.x[i_minus], 2) + pow(b.y[i_minus], 2)) + 1.0);

            // Check if we've reached the edge of the intersection region
            if (abs(abs(intersect_minus.x) - tracebox_width) < tol)
            {
                continue_tracing = false;
            }
            else if (abs(abs(intersect_minus.y) - tracebox_height) < tol)
            {
                continue_tracing = false;
            }
            if (n_steps >= 1)
            {
                continue_tracing = false;
            }
            n_steps++;
            // continue_tracing = false;
        }
        // Store information from the final intersection point
        x_minus[i] = intersect_minus.x;
        y_minus[i] = intersect_minus.y;
    }

    // Find q_plus and q_minus
    q_plus = 0.0;
    q_minus = 0.0;
    BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
    {
        n_x = static_cast<int>(floor(x_plus[i] / coord->dx[i]));
        n_y = static_cast<int>(floor(y_plus[i] / coord->dy[i]));
        i_offset = i.offset(n_x, n_y, 0);
        f_x = (x_plus[i] - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_plus[i] - n_y * coord->dy[i]) / coord->dy[i];

        u_plus = (1.0 - f_y) * ((1 - f_x) * u[i_offset] + f_x * u[i_offset.xp()]) + f_y * ((1 - f_x) * u[i_offset.yp()] + f_x * u[i_offset.xp().yp()]);
        q_plus[i] = K_par * (u_plus - u[i]) / parallel_distances_plus[i];

        n_x = static_cast<int>(floor(x_minus[i] / coord->dx[i]));
        n_y = static_cast<int>(floor(y_minus[i] / coord->dy[i]));
        i_offset = i.offset(n_x, n_y, 0);

        f_x = (x_minus[i] - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_minus[i] - n_y * coord->dy[i]) / coord->dy[i];
        u_minus = (1.0 - f_y) * ((1 - f_x) * u[i_offset] + f_x * u[i_offset.xp()]) + f_y * ((1 - f_x) * u[i_offset.yp()] + f_x * u[i_offset.xp().yp()]);
        q_minus[i] = -K_par * (u_minus - u[i]) / parallel_distances_minus[i];
    }

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
        div_q_plus = (parallel_distances_plus[i] / (0.5 * (parallel_distances_plus[i] + parallel_distances_minus[i]))) * ((q_plus_T - q_plus[i]) / parallel_distances_plus[i]);

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
    }
    // result = parallel_distances_plus;

    return result;
}

Field3D Churn::Q_plus(const Field3D &u, const Field3D &K_par, const Vector3D &b)
{
    TRACE("Q_plus");

    Field3D result, q_fs;
    BoutReal f_x, f_y;
    double y_plus, x_plus, ds_p, ds, u_plus; //, K_par_plus;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    for (auto i : result)
    {

        x_plus = coord->dy[i] * abs(b.x[i] / b.y[i]);
        x_plus = std::min(static_cast<double>(coord->dx[i]), x_plus);
        y_plus = coord->dx[i] * abs(b.y[i] / b.x[i]);
        y_plus = std::min(static_cast<double>(coord->dy[i]), y_plus);
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

        // Check if extrapolating across boundary
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y() + n_y + 1) > mesh->GlobalNy - ngcy_tot - 1)
            {
                u_plus = u[i];
            }
        }

        result[i] = K_par[i] * (u_plus - u[i]) / ds;
    }

    if (use_flux_limiter)
    {
        q_fs = 0.4 * sqrt(m_i / m_e) * pow(abs(u), 1.5);
        result = result / (1.0 + abs(result / (alpha_fl * (q_fs + 1e-10))));
    }

    return result;
}

Field3D Churn::Q_plus_T(const Field3D &u, const Vector3D &b)
{
    TRACE("Q_plus_T");

    Field3D result;
    BoutReal f_x, f_y;
    double y_plus, x_plus, ds_p, ds, u_plus;
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
        x_plus = std::min(static_cast<double>(coord->dx[i]), x_plus);
        y_plus = coord->dx[i] * abs(b.y[i] / b.x[i]);
        y_plus = std::min(static_cast<double>(coord->dy[i]), y_plus);
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

Field3D Churn::Q_minus(const Field3D &u, const Field3D &K_par, const Vector3D &b)
{
    TRACE("Q_minus");

    Field3D result, q_fs;
    BoutReal f_x, f_y;
    double y_minus, x_minus, ds_p, ds, u_minus; // K_par_minus;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    for (auto i : result)
    {

        x_minus = coord->dy[i] * abs(b.x[i] / b.y[i]);
        x_minus = std::min(static_cast<double>(coord->dx[i]), x_minus);
        y_minus = coord->dx[i] * abs(b.y[i] / b.x[i]);
        y_minus = std::min(static_cast<double>(coord->dy[i]), y_minus);
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

        // Check if extrapolating across boundary
        if (fixed_Q_in)
        {
            if (mesh->getGlobalYIndex(i.y() - n_y) > mesh->GlobalNy - ngcy_tot - 1)
            {
                u_minus = u[i];
            }
        }

        result[i] = -K_par[i] * (u_minus - u[i]) / ds;
    }

    if (use_flux_limiter)
    {
        q_fs = 0.4 * sqrt(m_i / m_e) * pow(abs(u), 1.5);
        result = result / (1.0 + abs(result / (alpha_fl * (q_fs + 1e-10))));
    }

    return result;
}

Field3D Churn::Q_minus_T(const Field3D &u, const Vector3D &b)
{
    TRACE("Q_minus_T");

    Field3D result;
    BoutReal f_x, f_y;
    double y_minus, x_minus, ds_p, ds, u_minus;
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
        x_minus = std::min(static_cast<double>(coord->dx[i]), x_minus);
        y_minus = coord->dx[i] * abs(b.y[i] / b.x[i]);
        y_minus = std::min(static_cast<double>(coord->dy[i]), y_minus);
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

Field3D Churn::div_q_par_modified_stegmeir(const Field3D &T, const Field3D &K_par, const Vector3D &b)
{
    // Modified Stegmeir stencil for parallel heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_par_modified_stegmeir");

    Field3D result;
    Field3D ds;
    BoutReal dz;

    // Support method
    result = -0.5 * (Q_plus_T(Q_plus(T, K_par, b), b) + Q_minus_T(Q_minus(T, K_par, b), b));

    return result;
}

Field3D Churn::div_q_par_modified_stegmeir_efficient(const Field3D &T, const Field3D &K_par, const Vector3D &b)
{
    // Modified Stegmeir stencil for parallel heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_par_modified_stegmeir");

    Field3D result, n_x_minus, n_y_minus, n_x_plus, n_y_plus, ds_minus, ds_plus, f_x_plus, f_x_minus, f_y_plus, f_y_minus, Q_plus, Q_minus;
    BoutReal f_x, f_y;
    double y_minus, x_minus, y_plus, x_plus, ds_p, T_minus, T_plus, Q_plus_T, Q_minus_T;
    int n_x, n_y;

    Coordinates *coord = mesh->getCoordinates();

    // Get field line geometry variables in minus direction
    ds_minus.allocate();
    f_x_minus.allocate();
    f_y_minus.allocate();
    n_x_minus.allocate();
    n_y_minus.allocate();
    for (auto i : result)
    {
        x_minus = coord->dy[i] * abs(b.x[i] / b.y[i]);
        x_minus = std::min(static_cast<double>(coord->dx[i]), x_minus);
        y_minus = coord->dx[i] * abs(b.y[i] / b.x[i]);
        y_minus = std::min(static_cast<double>(coord->dy[i]), y_minus);
        if (b.x[i] >= 0)
        {
            n_x_minus[i] = 0;
        }
        else
        {
            n_x_minus[i] = -1;
            x_minus = -x_minus;
        }
        if (b.y[i] >= 0)
        {
            n_y_minus[i] = 0;
        }
        else
        {
            n_y_minus[i] = -1;
            y_minus = -y_minus;
        }
        ds_p = sqrt(pow(x_minus, 2) + pow(y_minus, 2));

        ds_minus[i] = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
        f_x_minus[i] = (x_minus - n_x_minus[i] * coord->dx[i]) / coord->dx[i];
        f_y_minus[i] = (y_minus - n_y_minus[i] * coord->dy[i]) / coord->dy[i];
    }

    // Get field line geometry variables in plus direction
    ds_plus.allocate();
    f_x_plus.allocate();
    f_y_plus.allocate();
    n_x_plus.allocate();
    n_y_plus.allocate();
    for (auto i : result)
    {
        x_plus = coord->dy[i] * abs(b.x[i] / b.y[i]);
        x_plus = std::min(static_cast<double>(coord->dx[i]), x_plus);
        y_plus = coord->dx[i] * abs(b.y[i] / b.x[i]);
        y_plus = std::min(static_cast<double>(coord->dy[i]), y_plus);
        if (b.x[i] >= 0)
        {
            n_x_plus[i] = 0;
        }
        else
        {
            n_x_plus[i] = -1;
            x_plus = -x_plus;
        }
        if (b.y[i] >= 0)
        {
            n_y_plus[i] = 0;
        }
        else
        {
            n_y_plus[i] = -1;
            y_plus = -y_plus;
        }
        ds_p = sqrt(pow(x_plus, 2) + pow(y_plus, 2));

        ds_plus[i] = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
        f_x_plus[i] = (x_plus - n_x_plus[i] * coord->dx[i]) / coord->dx[i];
        f_y_plus[i] = (y_plus - n_y_plus[i] * coord->dy[i]) / coord->dy[i];
    }

    // Find Q_plus/minus
    Q_plus.allocate();
    Q_minus.allocate();
    for (auto i : result)
    {
        T_plus = (1.0 - f_y_plus[i]) * ((1 - f_x_plus[i]) * T(i.x() + n_x_plus[i], i.y() + n_y_plus[i], i.z()) + f_x_plus[i] * T(i.x() + n_x_plus[i] + 1, i.y() + n_y_plus[i], i.z())) + f_y_plus[i] * ((1 - f_x_plus[i]) * T(i.x() + n_x_plus[i], i.y() + n_y_plus[i] + 1, i.z()) + f_x_plus[i] * T(i.x() + n_x_plus[i] + 1, i.y() + n_y_plus[i] + 1, i.z()));
        Q_plus[i] = K_par[i] * (T_plus - T[i]) / ds_plus[i];

        T_minus = (1.0 - f_y_minus[i]) * ((1 - f_x_minus[i]) * T(i.x() - n_x_minus[i], i.y() - n_y_minus[i], i.z()) + f_x_minus[i] * T(i.x() - n_x_minus[i] - 1, i.y() - n_y_minus[i], i.z())) + f_y_minus[i] * ((1 - f_x_minus[i]) * T(i.x() - n_x_minus[i], i.y() - n_y_minus[i] - 1, i.z()) + f_x_minus[i] * T(i.x() - n_x_minus[i] - 1, i.y() - n_y_minus[i] - 1, i.z()));
        Q_minus[i] = -K_par[i] * (T_minus - T[i]) / ds_minus[i];
    }

    // Find div q
    result.allocate();
    for (auto i : result)
    {
        Q_plus_T = (1.0 - f_y_minus[i]) * ((1 - f_x_minus[i]) * Q_plus(i.x() - n_x_minus[i], i.y() - n_y_minus[i], i.z()) + f_x_minus[i] * Q_plus(i.x() - n_x_minus[i] - 1, i.y() - n_y_minus[i], i.z())) + f_y_minus[i] * ((1 - f_x_minus[i]) * Q_plus(i.x() - n_x_minus[i], i.y() - n_y_minus[i] - 1, i.z()) + f_x_minus[i] * Q_plus(i.x() - n_x_minus[i] - 1, i.y() - n_y_minus[i] - 1, i.z()));
        Q_minus_T = (1.0 - f_y_plus[i]) * ((1 - f_x_plus[i]) * Q_minus(i.x() + n_x_plus[i], i.y() + n_y_plus[i], i.z()) + f_x_plus[i] * Q_minus(i.x() + n_x_plus[i] + 1, i.y() + n_y_plus[i], i.z())) + f_y_plus[i] * ((1 - f_x_plus[i]) * Q_minus(i.x() + n_x_plus[i], i.y() + n_y_plus[i] + 1, i.z()) + f_x_plus[i] * Q_minus(i.x() + n_x_plus[i] + 1, i.y() + n_y_plus[i] + 1, i.z()));

        result[i] = -0.5 * (Q_plus_T - Q_plus[i]) / ds_minus[i];
        result[i] += 0.5 * (Q_minus_T - Q_minus[i]) / ds_plus[i];
    }

    return result;
}

Field3D Churn::spitzer_harm_conductivity(const Field3D &T, const BoutReal &Te_limit_ev)
{
    // Calculate the Spitzer-Harm thermal conductivity for electrons on the input temperature field
    TRACE("spitzer_harm_conductivity");

    Field3D result;
    BoutReal T_capped, tau_e, lambda_ei;

    result.allocate();
    BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
    {
        // Limit T used in parallel conduction to T_limit
        if (T[i] * T_sepx / 2.0 >= Te_limit_ev)
        {
            T_capped = T[i];
        }
        else
        {
            T_capped = Te_limit_ev * 2.0 / T_sepx;
        }

        // Find collision Coulomb log and collision time
        if (T_sepx * T_capped / 2.0 > 10.0)
        {
            lambda_ei = 24 - log(sqrt(rho / (m_e + m_i)) * pow(T_sepx * T_capped / 2.0, -1.0));
        }
        else
        {
            lambda_ei = 23 - log(sqrt(rho / (m_e + m_i)) * pow(T_sepx * T_capped / 2.0, -1.5));
        }
        tau_e = 3.0 * sqrt(m_e) * pow((e * T_sepx * T_capped / 2.0), 1.5) * pow((4.0 * pi * eps_0), 2.0) / (4.0 * sqrt(2.0 * pi) * (rho / (m_e + m_i)) * lambda_ei * pow(e, 4.0));

        // Calculate Spitzer-Harm parallel thermal conductivity
        result[i] = std::min((3.2 * n_sepx * boltzmann_k * e * T_sepx * T_capped * tau_e / m_e) / D_0, 1.0e7 / D_0);
    }

    return result;
}