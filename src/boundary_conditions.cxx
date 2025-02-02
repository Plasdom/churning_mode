#include "header.hxx"

void Churn::fixed_P_core_BC(const BoutReal &P_core_set)
{
    // TODO: Call this in init() rather than rhs() and, if not calling, set P to zero in boundary
    TRACE("fixed_P_core_BC");

    for (itu.first(); !itu.isDone(); itu++)
    {
        // for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        {
            for (int iz = 0; iz < mesh->LocalNz; iz++)
            {
                if (psi(itu.ind, iy, iz) >= 0.0)
                {
                    P(itu.ind, iy, iz) = P_core_set;
                }
                else
                {
                    P(itu.ind, iy, iz) = P_core_set * exp(-pow((sqrt((psi(itu.ind, iy, iz) - 1.0) / (-1.0)) - 1.0) / lambda_SOL_rho, 2.0));
                }
            }
        }
    }

    return;
}

void Churn::dPdy0_BC()
{
    // TODO: Use BOUT++ dirichlet BC here instead of setting it manually (may need to unset ddt(P)=0)
    TRACE("dPdy0_BC");

    int k = 0;
    for (int i = 0; i < mesh->LocalNx; i++)
    {
        if (mesh->lastY(i))
        {
            for (int j = mesh->LocalNy - ngcy_tot; j < mesh->LocalNy; j++)
            {
                P(i, j, k) = P(i, mesh->LocalNy - ngcy_tot - 1, k);
            }
        }
    }

    return;
}

void Churn::fixed_Q_in_BC()
{
    TRACE("fixed_Q_in_BC");

    for (itu.first(); !itu.isDone(); itu++)
    {
        for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
        {
            if ((mesh->getGlobalXIndex(itu.ind) > int(mesh->GlobalNx / 2) - int(num_q_in_cells / 2.0)) && (mesh->getGlobalXIndex(itu.ind) <= int(mesh->GlobalNx / 2) + int(num_q_in_cells / 2.0)))
            {
                ddt(P)(itu.ind, iy - 1, 0) += q_in / (2.0 * mesh->getCoordinates()->dy(itu.ind, iy, 0));
            }
        }
    }

    return;
}

void Churn::ddt0_BCs()
{
    TRACE("ddt0_BCs");

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
    return;
}

// struct ordering
// {
//     bool operator()(std::pair<BoutReal, int> const &a,
//                     std::pair<BoutReal, int> const &b)
//     {
//         return a.second < b.second;
//     }
// };

// void pairsort(BoutReal a[], BoutReal b[], int n)
// {
//     // Function to sort array b by values in array a
//     TRACE("pairsort");
//     std::pair<BoutReal, BoutReal> pairt[n];

//     // Store the array elements in pairs
//     for (int i = 0; i < n; i++)
//     {
//         pairt[i].first = a[i];
//         pairt[i].second = b[i];
//     }

//     // Sort the pair array
//     std::sort(pairt, pairt + n);

//     // Update original arrays
//     for (int i = 0; i < n; i++)
//     {
//         a[i] = pairt[i].first;
//         b[i] = pairt[i].second;
//     }
// }

// Field3D Churn::test_par_extrap_P_up_BC()
// {
//     TRACE("par_extrap_P_up_BC");

//     int nx = static_cast<int>(mesh->LocalNx);
//     BoutReal P_up[nx], psi_up[nx]; // Arrays to store P and psi values in the last row of cells before y_up boundary
//     double d0, d1;
//     Field3D result;
//     result = 0.0;

//     // Get P and psi in row of cells just below y_up boundary
//     // int k = 0;
//     // for (itu.first(); !itu.isDone(); itu++)
//     // {
//     //     int iy = mesh->LocalNy - ngcy_tot - 1;
//     //     int iz = 0;
//     //     // P_up[k] = P(itu.ind, iy, iz);
//     //     P_up[k] = k;
//     //     psi_up[k] = psi(itu.ind, iy, iz);
//     //     k++;
//     // }
//     // Check if in upper-most y region first
//     if (mesh->lastY())
//     {
//         int k = 0;
//         // Get P and psi in row of cells just below y_up boundary
//         // for (int i = 0; i < mesh->LocalNx; i++)
//         // {
//         //     int j = mesh->LocalNy - ngcy_tot - 1;
//         //     P_up[i] = P(i, j, k);
//         //     // P_up[i] = i;
//         //     psi_up[i] = psi(i, j, k);
//         // }

//         // Sort both arrays by values in psi_up
//         // pairsort(psi_up, P_up, nx);

//         // Interpolate P in boundary based on value of psi
//         for (int i = 0; i < mesh->LocalNx; i++)
//         {
//             for (int j = mesh->LocalNy - ngcy_tot - 2; j < mesh->LocalNy; j++)
//             {
//                 BoutReal psi_ij = psi(i, j, k);
//                 int nearest_psi_idx;
//                 for (int ii = 0; ii < nx; ii++)
//                 {
//                     if (psi_up[ii] > psi_ij)
//                     {
//                         nearest_psi_idx = ii;
//                         break;
//                     }
//                 }
//                 nearest_psi_idx = std::max(nearest_psi_idx, 1);
//                 d0 = psi_ij - psi_up[nearest_psi_idx - 1];
//                 d1 = psi_up[nearest_psi_idx] - psi_ij;
//                 result(i, j - 4, k) = (P_up[nearest_psi_idx] * d1 + P_up[nearest_psi_idx + 1] * d0) / (d0 + d1);
//             }
//         }
//     }
//     // for (int i = 0; i < mesh->LocalNx; i++)
//     // {
//     //     int j = mesh->LocalNy - ngcy_tot - 1;
//     //     if (mesh->getGlobalYIndex(j) == mesh->GlobalNy - 1)
//     //     {
//     //         P_up[i] = i;
//     //         psi_up[i] = psi(i, j, 0);
//     //     }
//     // }

//     // // Sort both arrays by values in psi_up
//     // pairsort(psi_up, P_up, nx);

//     // // Interpolate P in boundary based on value of psi
//     // for (int i = 0; i < mesh->LocalNx; i++)
//     // {
//     //     // for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
//     //     for (int iy = mesh->LocalNy - ngcy_tot - 2; iy < mesh->LocalNy; iy++)
//     //     {
//     //         for (int iz = 0; iz < mesh->LocalNz; iz++)
//     //         {
//     //             BoutReal psi_ij = psi(itu.ind, iy, iz);
//     //             int nearest_psi_idx;
//     //             for (int ii = 0; ii < nx; ii++)
//     //             {
//     //                 if (psi_up[ii] > psi_ij)
//     //                 {
//     //                     nearest_psi_idx = ii;
//     //                     break;
//     //                 }
//     //             }
//     //             nearest_psi_idx = std::max(nearest_psi_idx, 1);
//     //             d0 = psi_ij - psi_up[nearest_psi_idx - 1];
//     //             d1 = psi_up[nearest_psi_idx] - psi_ij;
//     //             // result(itu.ind, iy - 2, iz) = (P_up[nearest_psi_idx] * d1 + P_up[nearest_psi_idx + 1] * d0) / (d0 + d1);
//     //             // result(itu.ind, iy - 2, iz) = nearest_psi_idx;
//     //             // result(itu.ind, iy, iz) = P(itu.ind, iy - 1, iz);
//     //             result(itu.ind, iy, iz) = P_up[itu.ind];
//     //         }
//     //     }
//     // }
//     // for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
//     // for (int iy = mesh->LocalNy - ngcy_tot - 20; iy < mesh->LocalNy; iy++)
//     // {

//     //     // Interpolate P in boundary based on value of psi
//     //     int k = 0;
//     //     for (itu.first(); !itu.isDone(); itu++)
//     //     {
//     //         int iz = 0;
//     //         P_up[k] = P(itu.ind, iy, iz);
//     //         psi_up[k] = psi(itu.ind, iy, iz);
//     //         k++;
//     //     }
//     //     // pairsort(psi_up, P_up, nx);

//     //     for (itu.first(); !itu.isDone(); itu++)
//     //     {
//     //         int iz = 0;
//     //         // BoutReal psi_ij = psi(itu.ind, iy, iz);
//     //         // int nearest_psi_idx;
//     //         // for (int ii = 0; ii < nx; ii++)
//     //         // {
//     //         //     if (psi_up[ii] > psi_ij)
//     //         //     {
//     //         //         nearest_psi_idx = ii;
//     //         //         break;
//     //         //     }
//     //         // }
//     //         // nearest_psi_idx = std::max(nearest_psi_idx, 0);
//     //         // d0 = psi_ij - psi_up[nearest_psi_idx - 1];
//     //         // d1 = psi_up[nearest_psi_idx] - psi_ij;
//     //         // result(itu.ind, iy, iz) = (P_up[nearest_psi_idx] * d1 + P_up[nearest_psi_idx + 1] * d0) / (d0 + d1);
//     //         // result(itu.ind, iy, iz) = P_up[itu.ind];
//     //         result(itu.ind, iy, iz) = P_up[itu.ind];
//     //     }
//     // }

//     return result;
// }

// void Churn::par_extrap_P_up_BC()
// {
//     TRACE("par_extrap_P_up_BC");

//     int nx = static_cast<int>(mesh->LocalNx);
//     BoutReal P_up[nx], psi_up[nx]; // Arrays to store P and psi values in the last row of cells before y_up boundary
//     double d0, d1;
//     Field3D result;
//     result = 0.0;

//     // Get P and psi in row of cells just below y_up boundary
//     int k = 0;
//     for (itu.first(); !itu.isDone(); itu++)
//     {
//         int iy = mesh->LocalNy - ngcy_tot - 1;
//         int iz = 0;
//         P_up[k] = P(itu.ind, iy, iz);
//         psi_up[k] = psi(itu.ind, iy, iz);
//         k++;
//     }

//     // Sort both arrays by values in psi_up
//     pairsort(psi_up, P_up, nx);

//     // Interpolate P in boundary based on value of psi
//     for (itu.first(); !itu.isDone(); itu++)
//     {
//         // for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
//         for (int iy = mesh->LocalNy - ngcy_tot; iy < mesh->LocalNy; iy++)
//         {
//             for (int iz = 0; iz < mesh->LocalNz; iz++)
//             {
//                 BoutReal psi_ij = psi(itu.ind, iy, iz);
//                 int nearest_psi_idx;
//                 for (int ii = 0; ii < nx; ii++)
//                 {
//                     if (psi_up[ii] > psi_ij)
//                     {
//                         nearest_psi_idx = ii;
//                         break;
//                     }
//                 }
//                 nearest_psi_idx = std::max(nearest_psi_idx, 1);
//                 d0 = psi_ij - psi_up[nearest_psi_idx - 1];
//                 d1 = psi_up[nearest_psi_idx] - psi_ij;
//                 P(itu.ind, iy, iz) = (P_up[nearest_psi_idx] * d1 + P_up[nearest_psi_idx + 1] * d0) / (d0 + d1);
//             }
//         }
//     }

//     return;
// }
