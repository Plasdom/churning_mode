#include "header.hxx"

Field3D Churn::D3DX3(const Field3D &f)
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

Field3D Churn::D3DY3(const Field3D &f)
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

Field3D Churn::D3D2YDX(const Field3D &f)
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

Field3D Churn::D3D2XDY(const Field3D &f)
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