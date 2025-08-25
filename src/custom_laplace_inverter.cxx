
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