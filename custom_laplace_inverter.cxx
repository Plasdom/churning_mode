
#include "header.hxx"

Field3D customLaplaceInverter::operator()(const Field3D &input)
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