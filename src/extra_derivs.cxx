#include "header.hxx"

Field3D Churn::D2DX2_DIFF(const Field3D &f, const Field3D &A)
{
    TRACE("D2DX2_DIFF");

    Field3D result;
    BoutReal A_plus_half, A_minus_half;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
        // 2nd order
        A_plus_half = (0.5 * (A[i] + A[i.xp()]));
        A_minus_half = (0.5 * (A[i] + A[i.xm()]));
        result[i] = (1.0 / (pow(coord->dx[i], 2.0))) * (A_plus_half * (f[i.xp()] - f[i]) - A_minus_half * (f[i] - f[i.xm()]));
    }

    return result;
}

Field3D Churn::D2DY2_DIFF(const Field3D &f, const Field3D &A)
{
    TRACE("D2DY2_DIFF");

    Field3D result;
    BoutReal A_plus_half, A_minus_half;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
        // 2nd order
        A_plus_half = (0.5 * (A[i] + A[i.yp()]));
        A_minus_half = (0.5 * (A[i] + A[i.ym()]));
        result[i] = (1.0 / (pow(coord->dy[i], 2.0))) * (A_plus_half * (f[i.yp()] - f[i]) - A_minus_half * (f[i] - f[i.ym()]));
    }

    return result;
}

Field3D Churn::D3DX3(const Field3D &f)
{
    TRACE("D3DX3");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
        // 2nd order
        result[i] = (1.0 / (2.0 * pow(coord->dx[i], 3.0))) * (-f[i.xmm()] + 2.0 * f[i.xm()] - 2.0 * f[i.xp()] + f[i.xpp()]);

        // 4th order
        // result[i] = (1.0 / (8.0 * pow(coord->dx[i], 3.0))) * (f[i.xm().xm().xm()] - 8.0 * f[i.xmm()] + 13.0 * f[i.xm()] - 13.0 * f[i.xp()] + 8.0 * f[i.xpp()] - f[i.xp().xp().xp()]);
    }

    return result;
}

Field3D Churn::D3DY3(const Field3D &f)
{
    TRACE("D3DY3");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
        // 2nd order
        result[i] = (1.0 / (2.0 * pow(coord->dy[i], 3.0))) * (-f[i.ymm()] + 2.0 * f[i.ym()] - 2.0 * f[i.yp()] + f[i.ypp()]);

        // 4th order
        // result[i] = (1.0 / (8.0 * pow(coord->dy[i], 3.0))) * (f[i.ym().ym().ym()] - 8.0 * f[i.ymm()] + 13.0 * f[i.ym()] - 13.0 * f[i.yp()] + 8.0 * f[i.ypp()] - f[i.yp().yp().yp()]);
    }

    return result;
}

Field3D Churn::D3D2YDX(const Field3D &f)
{
    TRACE("D3D2YDX");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
        // 2nd order
        result[i] = (1.0 / (2.0 * pow(coord->dy[i], 2.0) * coord->dx[i])) * (f[i.xp().ym()] - f[i.xm().ym()] + f[i.xp().yp()] - f[i.xm().yp()] - 2.0 * f[i.xp()] + 2.0 * f[i.xm()]);

        // 4th order
        // result[i] = (1.0 / (144.0 * pow(coord->dy[i], 2.0) * coord->dx[i])) * (-(-f[i.xpp().ypp()] + 8.0 * f[i.xp().ypp()] - 8.0 * f[i.xm().ypp()] + f[i.xmm().ypp()]) + 16.0 * (-f[i.xpp().yp()] + 8.0 * f[i.xp().yp()] - 8.0 * f[i.xm().yp()] + f[i.xmm().yp()]) - 30.0 * (-f[i.xpp()] + 8.0 * f[i.xp()] - 8.0 * f[i.xm()] + f[i.xmm()]) + 16.0 * (-f[i.xpp().ym()] + 8.0 * f[i.xp().ym()] - 8.0 * f[i.xm().ym()] + f[i.xmm().ym()]) - (-f[i.xpp().ymm()] + 8.0 * f[i.xp().ymm()] - 8.0 * f[i.xm().ymm()] + f[i.xmm().ymm()]));
    }

    return result;
}

Field3D Churn::D3D2XDY(const Field3D &f)
{
    TRACE("D3D2XDY");

    Field3D result;

    Coordinates *coord = mesh->getCoordinates();

    result.allocate();
    for (auto i : result)
    {
        // 2nd order
        result[i] = (1.0 / (2.0 * pow(coord->dx[i], 2.0) * coord->dy[i])) * (f[i.yp().xm()] - f[i.ym().xm()] + f[i.yp().xp()] - f[i.ym().xp()] - 2.0 * f[i.yp()] + 2.0 * f[i.ym()]);

        // 4th order
        // result[i] = (1.0 / (144.0 * pow(coord->dx[i], 2.0) * coord->dy[i])) * (-(-f[i.ypp().xpp()] + 8.0 * f[i.yp().xpp()] - 8.0 * f[i.ym().xpp()] + f[i.ymm().xpp()]) + 16.0 * (-f[i.ypp().xp()] + 8.0 * f[i.yp().xp()] - 8.0 * f[i.ym().xp()] + f[i.ymm().xp()]) - 30.0 * (-f[i.ypp()] + 8.0 * f[i.yp()] - 8.0 * f[i.ym()] + f[i.ymm()]) + 16.0 * (-f[i.ypp().xm()] + 8.0 * f[i.yp().xm()] - 8.0 * f[i.ym().xm()] + f[i.ymm().xm()]) - (-f[i.ypp().xmm()] + 8.0 * f[i.yp().xmm()] - 8.0 * f[i.ym().xmm()] + f[i.ymm().xmm()]));
    }

    return result;
}

Field3D Churn::rotated_laplacexy(const Field3D &f)
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

Field3D Churn::grad_par_custom(const Field3D &u, const Vector3D &b)
{
    TRACE("grad_par");

    Field3D result;
    BoutReal f_x, f_y;
    double y_plus, x_plus, ds_p, ds, u_plus, u_minus;
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
        ds_p = 2.0 * sqrt(pow(x_plus, 2) + pow(y_plus, 2));
        ds = ds_p * sqrt(pow(b.z[i], 2) / (pow(b.x[i], 2) + pow(b.y[i], 2)) + 1.0);
        f_x = (x_plus - n_x * coord->dx[i]) / coord->dx[i];
        f_y = (y_plus - n_y * coord->dy[i]) / coord->dy[i];
        u_plus = (1.0 - f_y) * ((1 - f_x) * u(i.x() + n_x, i.y() + n_y, i.z()) + f_x * u(i.x() + n_x + 1, i.y() + n_y, i.z())) + f_y * ((1 - f_x) * u(i.x() + n_x, i.y() + n_y + 1, i.z()) + f_x * u(i.x() + n_x + 1, i.y() + n_y + 1, i.z()));
        u_minus = (1.0 - f_y) * ((1 - f_x) * u(i.x() - n_x, i.y() - n_y, i.z()) + f_x * u(i.x() - n_x - 1, i.y() - n_y, i.z())) + f_y * ((1 - f_x) * u(i.x() - n_x, i.y() - n_y - 1, i.z()) + f_x * u(i.x() - n_x - 1, i.y() - n_y - 1, i.z()));

        result[i] = 0.5 * (u_plus - u_minus) / ds;
    }

    return result;
}