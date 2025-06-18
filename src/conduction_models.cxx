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

        // // Apply grad_perp P = 0 BC if using fixed Q_in
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             A_plus_half = 0.0;
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if ((mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1) && psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
        //         {
        //             A_plus_half = 0.0;
        //         }
        //     }
        // }

        result[i] += (1.0 / (pow(coord->dy[i], 2.0))) * (A_plus_half * (T[i.yp()] - T[i]) - A_minus_half * (T[i] - T[i.ym()]));
    }

    // DXDY term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        ddy_plus = (0.5 / (coord->dy[i])) * (T[i.xp().yp()] - T[i.xp().ym()]);
        ddy_minus = (0.5 / (coord->dy[i])) * (T[i.xm().yp()] - T[i.xm().ym()]);

        // // Apply grad_perp P = 0 BC if using fixed Q_in
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             ddy_plus = (1.0 / (coord->dy[i])) * (T[i.xp()] - T[i.xp().ym()]);
        //             ddy_minus = (1.0 / (coord->dy[i])) * (T[i.xm()] - T[i.xm().ym()]);
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if ((mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1) && psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
        //         {
        //             ddy_plus = (1.0 / (coord->dy[i])) * (T[i.xp()] - T[i.xp().ym()]);
        //             ddy_minus = (1.0 / (coord->dy[i])) * (T[i.xm()] - T[i.xm().ym()]);
        //         }
        //     }
        // }

        result[i] += (0.5 / (coord->dx[i])) * (K_par[i.xp()] * b.x[i.xp()] * b.y[i.xp()] * ddy_plus - K_par[i.xm()] * b.x[i.xm()] * b.y[i.xm()] * ddy_minus);
    }

    // DYDX term
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        // 2nd order
        ddx_plus = (0.5 / (coord->dx[i])) * (T[i.yp().xp()] - T[i.yp().xm()]);
        ddx_minus = (0.5 / (coord->dx[i])) * (T[i.ym().xp()] - T[i.ym().xm()]);

        // // Apply grad_perp P = 0 BC if using fixed Q_in
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             ddx_plus = 0.0;
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if ((mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1) && psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
        //         {
        //             ddx_plus = 0.0;
        //         }
        //     }
        // }

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
        if (mesh->lastY(i.x()))
        {
            if (fixed_Q_in)
            {
                if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
                {
                    A_plus_half = 0.0;
                }
            }
            else if (disable_qin_outside_core || fixed_P_core)
            {
                if ((mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1) && psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
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
        if (mesh->lastY(i.x()))
        {
            if (fixed_Q_in)
            {
                if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
                {
                    ddy_plus = (1.0 / (coord->dy[i])) * (T[i.xp()] - T[i.xp().ym()]);
                    ddy_minus = (1.0 / (coord->dy[i])) * (T[i.xm()] - T[i.xm().ym()]);
                }
            }
            else if (disable_qin_outside_core || fixed_P_core)
            {
                if ((mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1) && psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
                {
                    ddy_plus = (1.0 / (coord->dy[i])) * (T[i.xp()] - T[i.xp().ym()]);
                    ddy_minus = (1.0 / (coord->dy[i])) * (T[i.xm()] - T[i.xm().ym()]);
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
        if (mesh->lastY(i.x()))
        {
            if (fixed_Q_in)
            {
                if (mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1)
                {
                    ddx_plus = 0.0;
                }
            }
            else if (disable_qin_outside_core || fixed_P_core)
            {
                if ((mesh->getGlobalYIndex(i.y()) >= mesh->GlobalNy - ngcy_tot - 1) && psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
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

    // if (fixed_Q_in)
    // {
    //     for (int i = 0; i < mesh->LocalNx; i++)
    //     {
    //         if (mesh->lastY(i))
    //         {
    //             q_pary_corners(i, mesh->LocalNy - ngcy_tot, 0) = 0.0;
    //         }
    //     }
    // }
    // else if (disable_qin_outside_core)
    // {
    //     for (int i = 0; i < mesh->LocalNx; i++)
    //     {
    //         if (mesh->lastY(i) && psi(i, mesh->LocalNy - ngcy_tot, 0) < psi_bndry_P_core_BC)
    //         {
    //             q_pary_corners(i, mesh->LocalNy - ngcy_tot, 0) = 0.0;
    //         }
    //     }
    // }

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
    else if (disable_qin_outside_core  || fixed_P_core)
    {
        int k = 0;
        for (int i = 0; i < mesh->LocalNx; i++)
        {
            if (mesh->lastY(i) && psi(i, mesh->LocalNy - ngcy_tot, 0) < psi_bndry_P_core_BC)
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
    double tol = 1e-10;
    double m, c, y1, y2, x1, x2;

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
    y1 = m * xlo + c;
    x1 = (ylo - c) / m;
    y2 = m * xhi + c;
    x2 = (yhi - c) / m;

    // Initialise intersect faces to -1 for debugging
    result.first.face = -1;
    result.second.face = -1;

    if ((ylo < y1) && (y1 < yhi))
    {
        // Intersect lower x face
        intersect.face = 0;
        intersect.x = xlo;
        intersect.y = y1;
        result.first = intersect;
    }
    else if ((xlo < x1) && (x1 < xhi))
    {
        // Intersect lower y face
        intersect.face = 1;
        intersect.x = x1;
        intersect.y = ylo;
        result.first = intersect;
    }
    else if ((abs(y1-ylo) < tol) && (abs(x1-xlo) < tol))
    {
        // Intersect lower left corner
        intersect.face = 4;
        intersect.x = xlo;
        intersect.y = ylo;
        result.first = intersect;
    }
    else if ((abs(y1-yhi)<tol) && (abs(x2-xlo) < tol))
    {
        // Intersect upper left corner
        intersect.face = 5;
        intersect.x = xlo;
        intersect.y = yhi;
        result.first = intersect;
    }

    // Second intersect is just the other side of the box
    intersect.face = result.first.face+2;
    intersect.x = -result.first.x;
    intersect.y = -result.first.y;
    result.second = intersect;


    return result;
}

CellIntersect Churn::get_next_intersect(const double &xlo, const double &xhi, const double &ylo, const double &yhi, const CellIntersect &prev_intersect, const double &bx, const double &by)
{
    // Find the intersection points between a line with gradient given by by/bx, where (Px,Py) is a point on the line, and the box bounded by xlo, xhi, ylo, yhi
    TRACE("get_next_intersect");

    // std::vector<std::vector<double>> result = {{0.0, 1.0}, {2.0, 3.0}};
    CellIntersect result;
    double m, c, y1, y2, x1, x2;
    double tol = 1e-10;

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
    y1 = m * xlo + c;
    y2 = m * xhi + c;
    x1 = (ylo - c) / m;
    x2 = (yhi - c) / m;

    // Initialise intersect face to -1 for debugging
    result.face = -1;

    if ((ylo <= y1) && (y1 < yhi) && (prev_intersect.face != 2))
    {
        // Intersect lower x face
        result.face = 0;
        result.x = xlo;
        result.y = y1;
    }
    else if ((ylo < y2) && (y2 <= yhi) && (prev_intersect.face != 0))
    {
        // Intersect upper x face
        result.face = 2;
        result.x = xhi;
        result.y = y2;
    }
    else if ((xlo < x1) && (x1 <= xhi) && (prev_intersect.face != 3))
    {
        // Intersect lower y face
        result.face = 1;
        result.x = x1;
        result.y = ylo;
    }
    else if ((xlo <= x2) && (x2 < xhi) && (prev_intersect.face != 1))
    {
        // Intersect upper y face
        result.face = 3;
        result.x = x2;
        result.y = yhi;
    }
    else if ((abs(y1-ylo) < tol) && (abs(x1-xlo) < tol) && (prev_intersect.face != 6))
    {
        // Intersect lower left corner
        result.face = 4;
        result.x = xlo;
        result.y = ylo;
    }
    else if ((abs(y1-yhi)<tol) && (abs(x2-xlo) < tol) && (prev_intersect.face != 7))
    {
        // Intersect upper left corner
        result.face = 5;
        result.x = xlo;
        result.y = yhi;
    }
    else if ((abs(x2-xhi)<tol) && (abs(y2-yhi) < tol) && (prev_intersect.face != 4))
    {
        // Intersect upper right corner
        result.face = 6;
        result.x = xhi;
        result.y = yhi;
    }
    else if ((abs(x1-xhi)<tol) && (abs(y2-ylo) < tol) && (prev_intersect.face != 5))
    {
        // Intersect lower right corner
        result.face = 7;
        result.x = xhi;
        result.y = ylo;
    }

    // TODO: Confirm one of above if statements is always entered
    if (result.face ==-1)
    {
        std::cout << "WARNING: Field line tracing has not worked as expected.";
    }

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
Ind3D Churn::increment_cell_2(const Ind3D &i, const Ind3D &i_prev, const int &face)
{
    // Determine which cell to move to next
    TRACE("increment_cell_2");

    Ind3D i_next;
    double tol = 1.0e-6;
    int x_inc, y_inc;

    i_next = i_prev;
    
    if (face == 0)
    {
        i_next = i_next.xm();
    }
    else if (face == 3)
    {
        i_next = i_next.yp();
    }
    else if (face == 2)
    {
        i_next = i_next.xp();
    }
    else if (face == 1)
    {
        i_next = i_next.ym();
    }
    else if (face == 4)
    {
        i_next = i_next.xm().ym();
    }
    else if (face == 5)
    {
        i_next = i_next.xm().yp();
    }
    else if (face == 6)
    {
        i_next = i_next.xp().yp();
    }
    else if (face == 7)
    {
        i_next = i_next.xp().ym();
    }

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

InterpolationPoint Churn::trace_field_lines_2(const Ind3D &i, const Vector3D &b, const BoutReal &dx, const BoutReal &dy, const int &max_steps, const bool &plus)
{
    InterpolationPoint result;
    CellIntersect next_intersect, prev_intersect;
    TwoIntersects intersects;
    double par_dist, xlo, xhi, ylo, yhi, theta_g, theta_b;
    Ind3D i_next, i_prev;
    Point cell_centre;
    ClosestPoint p;
    int n_steps;
    bool continue_tracing = true;
    double tol = 1e-10;

    // Initialise the next intersection
    next_intersect.x = 0.0;
    next_intersect.y = 0.0;
    par_dist = 0.0;
    p.x = 0.0;
    p.y = 0.0;
    p.distance = 0.0;
    theta_g = atan2(dy,dx);

    // Get the first intersect from the centre of cell (i,j)
    intersects = get_intersects(-dx / 2.0, dx / 2.0, -dy / 2.0, dy / 2.0, next_intersect, b.x[i], b.y[i]);

    // Specify plus/minus direction (choice is arbitrary)
    // TODO: Check the logic of selecting next intersect here. What if intersects.size != 2?
    prev_intersect = next_intersect;
    theta_b = atan2(b.y[i],b.x[i]);
    if (plus == true)
    {
        if ((theta_b + tol >= -theta_g) && (theta_b < pi - theta_g))
        {
            next_intersect = intersects.second;
        }
        else
        {
            next_intersect = intersects.first;
        }
    }
    else{
        if ((theta_b + tol >= -theta_g) && (theta_b < pi - theta_g))
        {
            next_intersect = intersects.first;
        }
        else
        {
            next_intersect = intersects.second;
        }
    }


    // Find the parallel distance par_dist = distance from (x_plus_prev, y_plus_prev) to (x_plus, y_plus) (i.e. it's the cumulative distance just to the next intersection point)
    par_dist += sqrt(pow(next_intersect.x - 0.0, 2.0) + pow(next_intersect.y - 0.0, 2.0)) * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);

    // Determine which cell to move to next
    i_prev = i;
    i_next = increment_cell_2(i, i_prev, next_intersect.face);

    // Continue to trace field lines in the given direction (exclude last ghost cells in each direction as would involve tracing outside the simulation domain)
    if ((i_next.y() >= 1) && (i_next.y() < mesh->LocalNy-1) && (i_next.x() >= 1) && (i_next.x() < mesh->LocalNx-1))
    {
        n_steps = 0;
        while (continue_tracing == true)
        {

            // Find intercepts in the new cell
            prev_intersect = next_intersect;
            
            xlo = std::max(-dx + (i_next.x() - i_prev.x()) * dx, -dx);
            xhi = std::min(dx + (i_next.x() - i_prev.x()) * dx, dx);
            ylo = std::max(-dy + (i_next.y() - i_prev.y()) * dy, -dy);
            yhi = std::min(dy + (i_next.y() - i_prev.y()) * dy, dy);
            next_intersect = get_next_intersect(xlo,
                                                xhi,
                                                ylo,
                                                yhi,
                                                prev_intersect,
                                                b.x[i_next],
                                                b.y[i_next]);

            // Find the parallel distance par_dist = distance from (x_plus_prev, y_plus_prev) to (x_plus, y_plus) (i.e. it's the cumulative distance just to the next intersection point)
            par_dist += sqrt(pow(next_intersect.x - prev_intersect.x, 2.0) + pow(next_intersect.y - prev_intersect.y, 2.0)) * sqrt(pow(b.z[i_next], 2.0) / (pow(b.x[i_next], 2.0) + pow(b.y[i_next], 2.0)) + 1.0);
            
            // Get subsequent intersects in plus direction
            i_prev = i_next;
            i_next = increment_cell_2(i, i_prev, next_intersect.face);

            n_steps++;

            // Check whether tracing can stop
            if ((abs(next_intersect.x) >= dx)){
                continue_tracing = false;
            }
            else if ((abs(next_intersect.y) >= dy)){
                continue_tracing = false;
            }
            else if (n_steps > max_steps){
            // else if (n_steps > 0){
                continue_tracing = false;
            }
        }
    }
    
    result.x = next_intersect.x;
    result.y = next_intersect.y;
    result.parallel_distance = par_dist;

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

Field3D Churn::div_q_par_linetrace(const Field3D &u, const Field3D &K_par, const Vector3D &b)
{
    TRACE("div_q_par_linetrace");

    Field3D result;
    Field3D x_plus, y_plus, x_minus, y_minus, parallel_distances_plus, parallel_distances_minus, q_plus, q_minus;
    Coordinates *coord = mesh->getCoordinates();
    InterpolationPoint interp_p_plus, interp_p_minus;
    int n_steps, n_x, n_y;
    double f_x, f_y, u_plus, q_plus_T, u_minus, q_minus_T, div_q_plus, div_q_minus;
    int max_x_inc = 1;
    int max_y_inc = 1;
    int max_steps = 3;
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
    BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
    // for (auto i : result)
    {
        // q_plus
        interp_p_plus = trace_field_lines_2(i, b, coord->dx[i], coord->dy[i], max_steps, true);
        x_plus[i] = interp_p_plus.x;
        y_plus[i] = interp_p_plus.y;
        parallel_distances_plus[i] = interp_p_plus.parallel_distance;

        if (x_plus[i]>=0)
        {
            n_x = 0;
        }
        else 
        {
            n_x = -1;
        }
        if (y_plus[i]>=0)
        {
            n_y = 0;
        }
        else 
        {
            n_y = -1;
        }
        i_offset = i.offset(n_x, n_y, 0);


        f_x = (x_plus[i] - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_plus[i] - n_y * coord->dy[i]) / coord->dy[i];

        u_plus = (1.0 - f_y) * ((1 - f_x) * u[i_offset] + f_x * u[i_offset.xp()]) + f_y * ((1 - f_x) * u[i_offset.yp()] + f_x * u[i_offset.xp().yp()]);
        q_plus[i] = K_par[i] * (u_plus - u[i]) / parallel_distances_plus[i];

        // // Check if extrapolating across boundary
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i_offset.y() + 1) > mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             q_plus[i] = 0.0;;
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if ((mesh->getGlobalYIndex(i_offset.y() + 1) > mesh->GlobalNy - ngcy_tot - 1) && (psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC))
        //         {
        //             q_plus[i] = 0.0;
        //         }
        //     }
        // }

        // q_minus
        interp_p_minus = trace_field_lines_2(i, b, coord->dx[i], coord->dy[i], max_steps, false);
        x_minus[i] = interp_p_minus.x;
        y_minus[i] = interp_p_minus.y;
        parallel_distances_minus[i] = interp_p_minus.parallel_distance;

        if (x_minus[i]>=0)
        {
            n_x = 0;
        }
        else 
        {
            n_x = -1;
        }
        if (y_minus[i]>=0)
        {
            n_y = 0;
        }
        else 
        {
            n_y = -1;
        }
        i_offset = i.offset(n_x, n_y, 0);

        f_x = (x_minus[i] - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_minus[i] - n_y * coord->dy[i]) / coord->dy[i];
        u_minus = (1.0 - f_y) * ((1 - f_x) * u[i_offset] + f_x * u[i_offset.xp()]) + f_y * ((1 - f_x) * u[i_offset.yp()] + f_x * u[i_offset.xp().yp()]);
        q_minus[i] = -K_par[i] * (u_minus - u[i]) / parallel_distances_minus[i];

        // // Check if extrapolating across boundary
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i_offset.y() + 1) > mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             q_minus[i] = 0.0;
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if ((mesh->getGlobalYIndex(i_offset.y() + 1) > mesh->GlobalNy - ngcy_tot - 1) && (psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC))
        //         {
        //             q_minus[i] = 0.0;
        //         }
        //     }
        // }
    }

    // // Naive method
    // result = 2.0 * (q_plus - q_minus) / (parallel_distances_plus + parallel_distances_minus);

    // Support operator method
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
        if (x_plus[i]>=0)
        {
            n_x = 0;
        }
        else 
        {
            n_x = -1;
        }
        if (y_plus[i]>=0)
        {
            n_y = 0;
        }
        else 
        {
            n_y = -1;
        }
        i_offset = i.offset(-n_x, -n_y, 0);

        f_x = (x_plus[i] - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_plus[i] - n_y * coord->dy[i]) / coord->dy[i];
        q_plus_T = (1.0 - f_y) * ((1 - f_x) * q_plus[i_offset] + f_x * q_plus[i_offset.xm()]) + f_y * ((1 - f_x) * q_plus[i_offset.ym()] + f_x * q_plus[i_offset.xm().ym()]);
        div_q_plus = (parallel_distances_plus[i] / (0.5 * (parallel_distances_plus[i] + parallel_distances_minus[i]))) * ((q_plus_T - q_plus[i]) / parallel_distances_plus[i]);

        if (x_minus[i]>=0)
        {
            n_x = 0;
        }
        else 
        {
            n_x = -1;
        }
        if (y_minus[i]>=0)
        {
            n_y = 0;
        }
        else 
        {
            n_y = -1;
        }
        i_offset = i.offset(-n_x, -n_y, 0);

        f_x = (x_minus[i] - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_minus[i] - n_y * coord->dy[i]) / coord->dy[i];
        q_minus_T = (1.0 - f_y) * ((1 - f_x) * q_minus[i_offset] + f_x * q_minus[i_offset.xm()]) + f_y * ((1 - f_x) * q_minus[i_offset.ym()] + f_x * q_minus[i_offset.xm().ym()]);
        div_q_minus = -(parallel_distances_minus[i] / (0.5 * (parallel_distances_plus[i] + parallel_distances_minus[i]))) * ((q_minus_T - q_minus[i]) / parallel_distances_minus[i]);

        result[i] = -0.5 * (div_q_plus + div_q_minus);
        
    }

    // Fill q_par output
    q_par = -0.5 * (q_minus + q_plus) * (B / B_mag);

    return result;
}

Field3D Churn::Q_plus(const Field3D &u, const Field3D &K_par, const Vector3D &b)
{
    TRACE("Q_plus");

    Field3D result, q_fs;
    BoutReal f_x, f_y;
    double y_plus, x_plus, ds_p, ds, u_plus, K_par_plus;
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
        K_par_plus = (1.0 - f_y) * ((1 - f_x) * K_par(i.x() + n_x, i.y() + n_y, i.z()) + f_x * K_par(i.x() + n_x + 1, i.y() + n_y, i.z())) + f_y * ((1 - f_x) * K_par(i.x() + n_x, i.y() + n_y + 1, i.z()) + f_x * K_par(i.x() + n_x + 1, i.y() + n_y + 1, i.z()));

        // // Check if extrapolating across boundary
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i.y() + n_y + 1) > mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             u_plus = u[i];
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if (mesh->getGlobalYIndex(i.y() + n_y + 1) > mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             if (psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
        //             {
        //                 u_plus = u[i];
        //             }
        //         }
        //     }
        // }

        // result[i] = K_par[i] * (u_plus - u[i]) / ds;
        result[i] = 0.5 * (K_par[i] + K_par_plus) * (u_plus - u[i]) / ds;
    }

    if (use_flux_limiter)
    {
        q_fs = 0.4 * sqrt(m_i / m_e) * pow(abs(u), 1.5);
        result = result / (1.0 + abs(result / (alpha_fl * (q_fs + 1e-15))));
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
    double y_minus, x_minus, ds_p, ds, u_minus, K_par_minus;
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
        K_par_minus = (1.0 - f_y) * ((1 - f_x) * K_par(i.x() - n_x, i.y() - n_y, i.z()) + f_x * K_par(i.x() - n_x - 1, i.y() - n_y, i.z())) + f_y * ((1 - f_x) * K_par(i.x() - n_x, i.y() - n_y - 1, i.z()) + f_x * K_par(i.x() - n_x - 1, i.y() - n_y - 1, i.z()));

        // // Check if extrapolating across boundary
        // if (mesh->lastY(i.x()))
        // {
        //     if (fixed_Q_in)
        //     {
        //         if (mesh->getGlobalYIndex(i.y() - n_y) > mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             u_minus = u[i];
        //         }
        //     }
        //     else if (disable_qin_outside_core)
        //     {
        //         if (mesh->getGlobalYIndex(i.y() - n_y) > mesh->GlobalNy - ngcy_tot - 1)
        //         {
        //             if (psi(i.x(), mesh->LocalNy - ngcy_tot, i.z()) < psi_bndry_P_core_BC)
        //             {
        //                 u_minus = u[i];
        //             }
        //         }   
        //     }
        // }

        // result[i] = -K_par[i] * (u_minus - u[i]) / ds;
        result[i] = -0.5 * (K_par[i] + K_par_minus) * (u_minus - u[i]) / ds;
    }

    if (use_flux_limiter)
    {
        q_fs = 0.4 * sqrt(m_i / m_e) * pow(abs(u), 1.5);
        result = result / (1.0 + abs(result / (alpha_fl * (q_fs + 1e-15))));
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

    Field3D q_par_plus, q_par_minus, result;
    Field3D ds;
    BoutReal dz;

    q_par_plus = Q_plus(T, K_par, b);
    q_par_minus = Q_minus(T, K_par, b);
    q_par = -0.5 * (q_par_plus + q_par_minus) * b;
    result = -0.5 * (Q_plus_T(q_par_plus, b) + Q_minus_T(q_par_minus, b));

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

Field3D Churn::spitzer_harm_conductivity(const Field3D &T, const BoutReal &Te_limit_ev_low, const BoutReal &Te_limit_ev_high)
{
    // Calculate the Spitzer-Harm thermal conductivity for electrons on the input temperature field
    TRACE("spitzer_harm_conductivity");

    //TODO: Modify this to allow for varying density

    Field3D result;
    BoutReal T_capped, tau_e, tau_e0, kappa_0, lambda_ei;
    // lambda_ei = 10.0;

    result.allocate();
    lambda_ei = 12.0;
    tau_e0 = 3.0 * sqrt(m_e) * pow((e * T_sepx / 2.0), 1.5) * pow((4.0 * pi * eps_0), 2.0) / (4.0 * sqrt(2.0 * pi) * (rho / (m_e + m_i)) * lambda_ei * pow(e, 4.0));
    kappa_0 = (3.2 * n_sepx * boltzmann_k * (e * T_sepx / 2.0) * tau_e0 / m_e) / D_0;
    BOUT_FOR(i, mesh->getRegion3D("RGN_ALL"))
    {
        // Limit T used in parallel conduction to T_limit
        T_capped = std::min(std::max(T[i], Te_limit_ev_low * 2.0 / T_sepx), Te_limit_ev_high * 2.0 / T_sepx);

        // tau_e = 3.0 * sqrt(m_e) * pow((e * T_sepx * T_capped / 2.0), 1.5) * pow((4.0 * pi * eps_0), 2.0) / (4.0 * sqrt(2.0 * pi) * (rho / (m_e + m_i)) * lambda_ei * pow(e, 4.0));

        // Calculate Spitzer-Harm parallel thermal conductivity
        // result[i] = std::min((3.2 * n_sepx * boltzmann_k * (e * T_sepx * T_capped / 2.0) * tau_e / m_e) / D_0, 1.0e7 / D_0);
        result[i] = kappa_0 * pow(T_capped,2.5);
    }

    return result;
}