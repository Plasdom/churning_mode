#include "header.hxx"

Field3D Churn::div_q_par_classic(const Field3D &T, const BoutReal &K_par, const Vector3D &b)
{
    TRACE("div_q_par_classic");

    Field3D result;

    result = K_par * (DDX(b.x * (b.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + b.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(b.y * (b.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + b.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL"));

    return result;
}

Field3D Churn::div_q_perp_classic(const Field3D &T, const BoutReal &K_perp, const Vector3D &b)
{
    TRACE("div_q_par_classic");

    Field3D result;

    result = K_perp * (D2DX2(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + D2DY2(T, CELL_CENTER, "DEFAULT", "RGN_ALL") - (DDX(b.x * (b.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + b.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL") + DDY(b.x * (b.x * DDX(T, CELL_CENTER, "DEFAULT", "RGN_ALL") + b.y * DDY(T, CELL_CENTER, "DEFAULT", "RGN_ALL")), CELL_CENTER, "DEFAULT", "RGN_ALL")));

    return result;
}

Field3D Churn::div_q_par_gunter(const Field3D &T, const BoutReal &K_par, const Vector3D &b)
{
    TRACE("div_q_par_gunter");

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

    result.allocate();
    for (auto i : result)
    {
        result[i] = (1.0 / (2.0 * coord->dx[i])) * (q_parx_corners[i.xp().yp()] + q_parx_corners[i.xp()] - q_parx_corners[i.yp()] - q_parx_corners[i]);
        result[i] += (1.0 / (2.0 * coord->dy[i])) * (q_pary_corners[i.xp().yp()] + q_pary_corners[i.yp()] - q_pary_corners[i.xp()] - q_pary_corners[i]);
    }

    return result;
}

Field3D Churn::div_q_perp_gunter(const Field3D &T, const BoutReal &K_perp, const Vector3D &b)
{
    TRACE("div_q_perp_gunter");

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

    q_perpx_corners = K_perp * (DTDX_corners - by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners));
    q_perpy_corners = K_perp * (DTDY_corners - by_corners * (bx_corners * DTDX_corners + by_corners * DTDY_corners));

    result.allocate();
    for (auto i : result)
    {
        result[i] += (1.0 / (2.0 * coord->dx[i])) * (q_perpx_corners[i.xp().yp()] + q_perpx_corners[i.xp()] - q_perpx_corners[i.yp()] - q_perpx_corners[i]);
        result[i] += (1.0 / (2.0 * coord->dy[i])) * (q_perpy_corners[i.xp().yp()] + q_perpy_corners[i.yp()] - q_perpy_corners[i.xp()] - q_perpy_corners[i]);
    }

    return result;
}

TwoIntersects Churn::get_intersects(const float &xlo, const float &xhi, const float &ylo, const float &yhi, const CellIntersect &P, const float &bx, const float &by)
{
    // Find the intersection points between a line with gradient given by by/bx, where (Px,Py) is a point on the line, and the box bounded by xlo, xhi, ylo, yhi
    // TODO: Check edge cases when intersection is on a corner
    TRACE("intersects_plus");

    // std::vector<std::vector<float>> result = {{0.0, 1.0}, {2.0, 3.0}};
    TwoIntersects result;
    CellIntersect intersect;
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

CellIntersect Churn::get_next_intersect(const float &xlo, const float &xhi, const float &ylo, const float &yhi, const CellIntersect &prev_intersect, const float &bx, const float &by)
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

Ind3D Churn::increment_cell(const Ind3D &i, const Ind3D &i_prev, const CellIntersect &P_next, const float &dx, const float &dy)
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

InterpolationPoint Churn::trace_field_lines(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_x_inc, const int &max_y_inc, const int &max_steps, const bool &plus)
{
    InterpolationPoint result;
    CellIntersect next_intersect, prev_intersect;
    TwoIntersects intersects;
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

ClosestPoint Churn::get_closest_p(const CellIntersect &P, const Point &P0, const float &bx, const float &by)
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

Field3D Churn::div_q_par_linetrace(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
{
    TRACE("div_q_par_linetrace");

    Field3D result;
    Field3D x_plus, y_plus, x_minus, y_minus, parallel_distances_plus, parallel_distances_minus, q_plus, q_minus;
    Coordinates *coord = mesh->getCoordinates();
    InterpolationPoint interp_p_plus, interp_p_minus;
    int n_steps, n_x, n_y;
    float f_x, f_y, u_plus, q_plus_T, u_minus, q_minus_T, div_q_plus, div_q_minus;
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
    float f_x, f_y, u_plus, q_plus_T, u_minus, q_minus_T, div_q_plus, div_q_minus;
    Coordinates *coord = mesh->getCoordinates();
    Field3D result;
    bool continue_tracing;
    Ind3D i_prev, i_plus, i_minus;
    TwoIntersects intersects;
    CellIntersect intersect, intersect_plus, intersect_minus, prev_intersect;
    float m, c, y_xlo, y_xhi, x_ylo, x_yhi, xlo, xhi, ylo, yhi, tracebox_width, tracebox_height;
    float tol = 1.0e-6;
    Field3D x_plus, y_plus, x_minus, y_minus, parallel_distances_plus, parallel_distances_minus, q_plus, q_minus;
    float prev_x, prev_y;
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
            xlo = std::max(static_cast<float>(xlo), static_cast<float>(-coord->dx[i]));
            xhi = (i_plus.x() - i_prev.x()) * coord->dx[i] + coord->dx[i] / 2.0;
            xhi = std::min(static_cast<float>(xhi), static_cast<float>(coord->dx[i]));
            ylo = (i_plus.y() - i_prev.y()) * coord->dy[i] - coord->dy[i] / 2.0;
            ylo = std::max(static_cast<float>(ylo), static_cast<float>(-coord->dy[i]));
            yhi = (i_plus.y() - i_prev.y()) * coord->dy[i] + coord->dy[i] / 2.0;
            yhi = std::min(static_cast<float>(yhi), static_cast<float>(coord->dy[i]));

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
            xlo = std::max(static_cast<float>(xlo), static_cast<float>(-coord->dx[i]));
            xhi = (i_minus.x() - i_prev.x()) * coord->dx[i] + coord->dx[i] / 2.0;
            xhi = std::min(static_cast<float>(xhi), static_cast<float>(coord->dx[i]));
            ylo = (i_minus.y() - i_prev.y()) * coord->dy[i] - coord->dy[i] / 2.0;
            ylo = std::max(static_cast<float>(ylo), static_cast<float>(-coord->dy[i]));
            yhi = (i_minus.y() - i_prev.y()) * coord->dy[i] + coord->dy[i] / 2.0;
            yhi = std::min(static_cast<float>(yhi), static_cast<float>(coord->dy[i]));

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

Field3D Churn::Q_plus(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
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

Field3D Churn::Q_plus_T(const Field3D &u, const Vector3D &b)
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

Field3D Churn::Q_minus(const Field3D &u, const BoutReal &K_par, const Vector3D &b)
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

Field3D Churn::Q_minus_T(const Field3D &u, const Vector3D &b)
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

Field3D Churn::div_q_par_modified_stegmeir(const Field3D &T, const BoutReal &K_par, const Vector3D &b)
{
    TRACE("div_q_par_modified_stegmeir");

    Field3D result;
    Field3D ds;
    BoutReal dz;

    Coordinates *coord = mesh->getCoordinates();

    // TODO: Size of the box should probably be changed from dx,dy to dx/2, dy/2 in order to not rely on slowly changing b field

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
