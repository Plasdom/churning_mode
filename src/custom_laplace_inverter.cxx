
#include "header.hxx"

Field3D customLaplaceInverter::operator()(const Field3D &input)
{

    Field3D result = A * input + D * (D2DX2(input) + D2DY2(input));
    // Field3D result = A + D * (D2DX2(input) + D2DY2(input));

    // result.setBoundaryTo(input);

    // Apply the boundary condition
    //TODO: Find out why setting anything other than BC_width=0 results in memory leak
    if (BC_width == 0)
    {
        result.applyBoundary("dirichlet(0)");
    }
    else if (BC_width == 3)
    {
        result.applyBoundary("width(dirichlet(0),3)");
    }else if (BC_width == 4)
    {
        result.applyBoundary("width(dirichlet(0),4)");
    }
    else if (BC_width == 5)
    {
        result.applyBoundary("width(dirichlet(0),5)");
    }
    else if (BC_width == 6)
    {
        result.applyBoundary("width(dirichlet(0),6)");
    }
    

    return result;
};


Field3D customParLaplaceInverter::operator()(const Field3D &input)
{
    result = div_q_par_modified_stegmeir_2(input, b);
    result.applyBoundary("dirichlet(0)");
    return result;
};

Field3D Q_plus_2(const Field3D &u, const Vector3D &b)
{
    TRACE("Q_plus");

    Field3D result, q_fs;
    BoutReal f_x, f_y, psi_plus;
    double y_plus, x_plus, ds_p, ds, u_plus, K_par_plus;
    int n_x, n_y;

    Mesh* mesh = result.getMesh();
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

        result[i] = (u_plus - u[i]) / ds;

    }

    return result;
}

Field3D Q_plus_T_2(const Field3D &u, const Vector3D &b)
{
    TRACE("Q_plus_T");

    Field3D result;
    BoutReal f_x, f_y;
    double y_plus, x_plus, ds_p, ds, u_plus;
    int n_x, n_y;

    Mesh* mesh = result.getMesh();
    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
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

Field3D Q_minus_2(const Field3D &u, const Vector3D &b)
{
    TRACE("Q_minus");

    Field3D result, q_fs;
    BoutReal f_x, f_y, psi_minus;
    double y_minus, x_minus, ds_p, ds, u_minus, K_par_minus;
    int n_x, n_y;

    Mesh* mesh = result.getMesh();
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
        K_par_minus = (1.0 - f_y) * ((1 - f_x) + f_x) + f_y * ((1 - f_x) + f_x);

        result[i] = -(u_minus - u[i]) / ds;

    }

    return result;
}

Field3D Q_minus_T_2(const Field3D &u, const Vector3D &b)
{
    TRACE("Q_minus_T");

    Field3D result;
    BoutReal f_x, f_y;
    double y_minus, x_minus, ds_p, ds, u_minus;
    int n_x, n_y;

    Mesh* mesh = result.getMesh();
    Coordinates *coord = mesh->getCoordinates();

    result = 0.0;
    BOUT_FOR(i, mesh->getRegion3D("RGN_NOBNDRY"))
    {
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

Field3D div_q_par_modified_stegmeir_2(const Field3D &T, const Vector3D &b)
{
    // Modified Stegmeir stencil for parallel heat flux divergence term (spatially varying conductivity)
    TRACE("div_q_par_modified_stegmeir");

    Field3D q_par_plus, q_par_minus, result;
    q_par_plus = Q_plus_2(T, b);
    q_par_minus = Q_minus_2(T, b);

    result = -0.5 * (Q_plus_T_2(q_par_plus, b) + Q_minus_T_2(q_par_minus, b));

    return result;
}