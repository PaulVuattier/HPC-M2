// pic_unit: field-solver unit tests on analytic sine fields (no particles).
//
//   ./pic_unit ampere  1|2 NX OUT.csv
//   ./pic_unit faraday 1|2 NX OUT.csv
//
// Periodic box of length L = 1 (NX cells per side, k = 2 pi / L). Each
// component is set at ITS OWN Yee location (offset_x_*/offset_y_* in
// pic_grid.h), the operator is applied once, and every output component is
// written next to the analytic value at the location it is stored at:
//   ampere : mu0 * j = curl B, reported as curl B (mu0 factored out)
//   faraday: B_new = B_old - dt curl E, dt = 1e-3, B_old = (0, 2, -1) (+ Bx in 2D)
// CSV columns: comp,i,j,x,y,num,exact  (j = y = 0 in 1D).
// Halving the cell size should divide the max error by 4 (second order).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pic_grid.h"

namespace {

constexpr double L = 1.0;
constexpr double DT = 1.0e-3;
const double K = 2.0 * M_PI / L;

void row(FILE* f, const char* comp, int i, int j, double x, double y,
         double num, double exact) {
    std::fprintf(f, "%s,%d,%d,%.15e,%.15e,%.15e,%.15e\n", comp, i, j, x, y,
                 num, exact);
}

// 1D: By = cos kx, Bz = sin kx at mid-cells -> curl B at nodes:
//   (curl B)_y = -dBz/dx = -k cos kx,  (curl B)_z = dBy/dx = -k sin kx.
void ampere_1d(int nx, FILE* f) {
    pic::Grid1D g(nx, L / nx);
    for (int i = 0; i < nx; ++i) {
        const double xm = (i + 0.5) * g.dx;
        g.B[1][(size_t)i] = std::cos(K * xm);
        g.B[2][(size_t)i] = std::sin(K * xm);
    }
    std::vector<double> jx(nx), jy(nx), jz(nx);
    g.ampere(jx.data(), jy.data(), jz.data());
    for (int i = 0; i < nx; ++i) {
        const double x = i * g.dx;
        row(f, "jy", i, 0, x, 0, jy[i] * pic::MU0, -K * std::cos(K * x));
        row(f, "jz", i, 0, x, 0, jz[i] * pic::MU0, -K * std::sin(K * x));
    }
}

// 1D: Ey = cos kx, Ez = sin kx at nodes -> at mid-cells after one step:
//   By += dt dEz/dx = dt k cos kx,  Bz -= dt dEy/dx = +dt k sin kx.
void faraday_1d(int nx, FILE* f) {
    pic::Grid1D g(nx, L / nx);
    for (int i = 0; i < nx; ++i) {
        const double x = i * g.dx;
        g.E[1][(size_t)i] = std::cos(K * x);
        g.E[2][(size_t)i] = std::sin(K * x);
        g.B[1][(size_t)i] = 2.0;
        g.B[2][(size_t)i] = -1.0;
    }
    g.faraday(DT);
    for (int i = 0; i < nx; ++i) {
        const double xm = (i + 0.5) * g.dx;
        row(f, "By", i, 0, xm, 0, g.B[1][(size_t)i],
            2.0 + DT * K * std::cos(K * xm));
        row(f, "Bz", i, 0, xm, 0, g.B[2][(size_t)i],
            -1.0 + DT * K * std::sin(K * xm));
    }
}

// Position of component c of E (is_e) or B on the 2D Yee grid.
void pos2d(const pic::Grid2D& g, bool is_e, int c, int i, int j, double& x,
           double& y) {
    x = (i + pic::offset_x_2d(is_e, c)) * g.dx;
    y = (j + pic::offset_y_2d(is_e, c)) * g.dy;
}

// 2D: Bx = cos ky, By = cos kx, Bz = sin kx cos ky (each at its own place)
//   (curl B)_x =  dBz/dy          = -k sin kx sin ky
//   (curl B)_y = -dBz/dx          = -k cos kx cos ky
//   (curl B)_z =  dBy/dx - dBx/dy = -k sin kx + k sin ky      (all at nodes)
void ampere_2d(int n, FILE* f) {
    pic::Grid2D g(n, n, L / n, L / n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double x, y;
            pos2d(g, false, 0, i, j, x, y);
            g.B[0][g.id(i, j)] = std::cos(K * y);
            pos2d(g, false, 1, i, j, x, y);
            g.B[1][g.id(i, j)] = std::cos(K * x);
            pos2d(g, false, 2, i, j, x, y);
            g.B[2][g.id(i, j)] = std::sin(K * x) * std::cos(K * y);
        }
    const size_t N = (size_t)n * n;
    std::vector<double> jx(N), jy(N), jz(N);
    g.ampere(jx.data(), jy.data(), jz.data());
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const size_t c = g.id(i, j);
            const double x = i * g.dx, y = j * g.dy;
            row(f, "jx", i, j, x, y, jx[c] * pic::MU0,
                -K * std::sin(K * x) * std::sin(K * y));
            row(f, "jy", i, j, x, y, jy[c] * pic::MU0,
                -K * std::cos(K * x) * std::cos(K * y));
            row(f, "jz", i, j, x, y, jz[c] * pic::MU0,
                -K * std::sin(K * x) + K * std::sin(K * y));
        }
}

// 2D: Ex = cos ky, Ey = cos kx, Ez = sin kx sin ky; B_old = (1, 2, -1).
//   dBx/dt = -dEz/dy          = -k sin kx cos ky
//   dBy/dt = +dEz/dx          =  k cos kx sin ky
//   dBz/dt = -(dEy/dx - dEx/dy) = k sin kx - k sin ky
// Each is compared at the place where that B component is stored.
void faraday_2d(int n, FILE* f) {
    pic::Grid2D g(n, n, L / n, L / n);
    const double b0[3] = {1.0, 2.0, -1.0};
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double x, y;
            const size_t c = g.id(i, j);
            pos2d(g, true, 0, i, j, x, y);
            g.E[0][c] = std::cos(K * y);
            pos2d(g, true, 1, i, j, x, y);
            g.E[1][c] = std::cos(K * x);
            pos2d(g, true, 2, i, j, x, y);
            g.E[2][c] = std::sin(K * x) * std::sin(K * y);
            for (int k = 0; k < 3; ++k) g.B[k][c] = b0[k];
        }
    g.faraday(DT);
    const char* name[3] = {"Bx", "By", "Bz"};
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                double x, y;
                pos2d(g, false, k, i, j, x, y);
                const double sx = std::sin(K * x), cx = std::cos(K * x);
                const double sy = std::sin(K * y), cy = std::cos(K * y);
                const double rate = (k == 0)   ? -K * sx * cy
                                    : (k == 1) ? K * cx * sy
                                               : K * sx - K * sy;
                row(f, name[k], i, j, x, y, g.B[k][g.id(i, j)],
                    b0[k] + DT * rate);
            }
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s ampere|faraday 1|2 NX OUT.csv\n",
                     argv[0]);
        return 1;
    }
    const std::string op = argv[1];
    const int dim = std::atoi(argv[2]), nx = std::atoi(argv[3]);
    if ((op != "ampere" && op != "faraday") || (dim != 1 && dim != 2) ||
        nx < 4) {
        std::fprintf(stderr, "error: bad arguments\n");
        return 1;
    }
    FILE* f = std::fopen(argv[4], "w");
    if (!f) {
        std::fprintf(stderr, "error: cannot open '%s'\n", argv[4]);
        return 1;
    }
    std::fprintf(f, "comp,i,j,x,y,num,exact\n");
    if (op == "ampere")
        dim == 1 ? ampere_1d(nx, f) : ampere_2d(nx, f);
    else
        dim == 1 ? faraday_1d(nx, f) : faraday_2d(nx, f);
    std::fclose(f);
    return 0;
}
