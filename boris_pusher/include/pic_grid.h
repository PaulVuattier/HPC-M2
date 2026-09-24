#pragma once
// PIC field grids: 1D and 2D Yee layouts with periodic BC.
//
// Mathematical layout (course PDF section 5):
//   1D (variation along x, cell size dx):
//     nodes i        : Ey, Ez, Bx   (mathematical x_i)
//     mid-cells      : Ex, By, Bz   (mathematical x_{i+1/2})
//   2D (cell dx x dy): "B on faces, E on edges" (PDF 5.1)
//     nodes (i,j)             : Ez
//     x-edges (i+1/2,j)       : Ex, By
//     y-edges (i,j+1/2)       : Ey, Bx
//     cell centres (i+1/2,j+1/2) : Bz
//   The PDF's 2D figure labels a node "Ez, Bz", but its Faraday stencil table
//   puts Bz between Ey(i,j+1/2), Ey(i+1,j+1/2), Ex(i+1/2,j), Ex(i+1/2,j+1),
//   i.e. at the cell centre -- the only place where that stencil is centred.
//   Storing Bz on the node instead left the Ex/Ey/Bz loop unstaggered and
//   the coupled 2D run went unstable.
//
// Storage: every component is one flat array indexed [i] / [i+j*nx].
// The half-offset is a PROPERTY of the component, not of the index
// (PDF section 5.4). All gather/scatter shifts and all curl stencils go
// through offset_x()/offset_y() below, so the bookkeeping lives in one
// place. Field updates use periodic wrapping.

#include <cmath>
#include <vector>

#include "boris.h"

namespace pic {

constexpr double MU0 = 1.25663706212e-6;   // N/A^2
constexpr double ELEM = 1.602176634e-19;   // C

// Which edge of the Yee cell a component lives on (in units of cells).
inline double offset_x_1d(int comp) {
    // comp: 0=x, 1=y, 2=z. E and B share the 1D layout: y,z node-based,
    // x mid-cell-based (see header comment).
    return (comp == 0) ? 0.5 : 0.0;
}

// 2D offsets in (x, y) cell units, per PDF section 5.1.
inline double offset_x_2d(bool is_e, int comp) {
    if (is_e) return (comp == 0) ? 0.5 : (comp == 1) ? 0.0 : 0.0;
    // Bx y-edge, By x-edge, Bz cell centre
    return (comp == 0) ? 0.0 : (comp == 1) ? 0.5 : 0.5;
}

inline double offset_y_2d(bool is_e, int comp) {
    if (is_e) return (comp == 0) ? 0.0 : (comp == 1) ? 0.5 : 0.0;
    return (comp == 0) ? 0.5 : (comp == 1) ? 0.0 : 0.5;
}

struct Grid1D {
    int nx = 0;
    double dx = 1.0, x0 = 0.0;
    // E[3], B[3] component-major: Ex[c*nx+i], c=0,1,2.
    std::vector<double> E[3], B[3];

    Grid1D() = default;
    Grid1D(int nx_, double dx_, double x0_ = 0.0) { init(nx_, dx_, x0_); }

    void init(int nx_, double dx_, double x0_ = 0.0) {
        nx = nx_;
        dx = dx_;
        x0 = x0_;
        for (int c = 0; c < 3; ++c) {
            E[c].assign((size_t)nx, 0.0);
            B[c].assign((size_t)nx, 0.0);
        }
    }

    int wrap(int i) const {
        i %= nx;
        return i < 0 ? i + nx : i;
    }

    void set_uniform(const Vec3& e, const Vec3& b) {
        const double ev[3] = {e.x, e.y, e.z}, bv[3] = {b.x, b.y, b.z};
        for (int c = 0; c < 3; ++c)
            for (int i = 0; i < nx; ++i) {
                E[c][(size_t)i] = ev[c];
                B[c][(size_t)i] = bv[c];
            }
    }

    // Sine field used by the solver tests: Ez(x) = a*sin(k*x) at nodes,
    // everything else zero. Analytic curl: dEz/dx = a*k*cos(k*x).
    void set_sine_ez(double a, double k) {
        for (int i = 0; i < nx; ++i) {
            const double x = x0 + i * dx;
            E[0][(size_t)i] = 0.0;
            E[1][(size_t)i] = 0.0;
            E[2][(size_t)i] = a * std::sin(k * x);
            B[0][(size_t)i] = B[1][(size_t)i] = B[2][(size_t)i] = 0.0;
        }
    }

    // Sine Ex(x) = a*sin(k*x) at mid-cells (gather-convergence tests: the
    // 1D mover drifts along x, so only Ex does work on it).
    void set_sine_ex(double a, double k) {
        for (int i = 0; i < nx; ++i) {
            const double xm = x0 + (i + 0.5) * dx;
            E[0][(size_t)i] = a * std::sin(k * xm);
            E[1][(size_t)i] = 0.0;
            E[2][(size_t)i] = 0.0;
            B[0][(size_t)i] = B[1][(size_t)i] = B[2][(size_t)i] = 0.0;
        }
    }

    // Faraday update: dB/dt = -curl(E), Yee-symmetric two-point stencils.
    // With 1D variation along x: dBx/dt = 0, dBy/dt = +dEz/dx (Ez at nodes,
    // By at mid-cells -> centred at i+1/2), dBz/dt = -dEy/dx.
    void faraday(double dt) {
        std::vector<double> nBy((size_t)nx), nBz((size_t)nx);
        for (int i = 0; i < nx; ++i) {
            const int ip = wrap(i + 1);
            const double dEz = (E[2][(size_t)ip] - E[2][(size_t)i]) / dx;
            const double dEy = (E[1][(size_t)ip] - E[1][(size_t)i]) / dx;
            nBy[(size_t)i] = B[1][(size_t)i] + dt * dEz;
            nBz[(size_t)i] = B[2][(size_t)i] - dt * dEy;
        }
        B[1] = nBy;
        B[2] = nBz;  // Bx static in 1D (no y/z derivatives).
    }

    // Ampere: mu0*j = curl(B), centred back at the nodes.
    // jx = 0 in 1D; jy = -dBz/dx / mu0; jz = +dBy/dx / mu0.
    // By/Bz live at mid-cells, so (B[i]-B[i-1])/dx is centred at node i.
    void ampere(double* jx, double* jy, double* jz) const {
        for (int i = 0; i < nx; ++i) {
            const int im = (i - 1 + nx) % nx;
            jx[i] = 0.0;
            jy[i] = -(B[2][(size_t)i] - B[2][(size_t)im]) / (dx * MU0);
            jz[i] = (B[1][(size_t)i] - B[1][(size_t)im]) / (dx * MU0);
        }
    }
};

struct Grid2D {
    int nx = 0, ny = 0;
    double dx = 1.0, dy = 1.0, x0 = 0.0, y0 = 0.0;
    std::vector<double> E[3], B[3];  // idx = i + j*nx

    Grid2D() = default;
    Grid2D(int nx_, int ny_, double dx_, double dy_) { init(nx_, ny_, dx_, dy_); }

    void init(int nx_, int ny_, double dx_, double dy_, double x0_ = 0.0,
              double y0_ = 0.0) {
        nx = nx_;
        ny = ny_;
        dx = dx_;
        dy = dy_;
        x0 = x0_;
        y0 = y0_;
        for (int c = 0; c < 3; ++c) {
            E[c].assign((size_t)nx * (size_t)ny, 0.0);
            B[c].assign((size_t)nx * (size_t)ny, 0.0);
        }
    }

    size_t id(int i, int j) const {
        i %= nx;
        j %= ny;
        if (i < 0) i += nx;
        if (j < 0) j += ny;
        return (size_t)i + (size_t)j * (size_t)nx;
    }

    void set_uniform(const Vec3& e, const Vec3& b) {
        const double ev[3] = {e.x, e.y, e.z}, bv[3] = {b.x, b.y, b.z};
        for (int c = 0; c < 3; ++c)
            for (size_t k = 0; k < E[c].size(); ++k) {
                E[c][k] = ev[c];
                B[c][k] = bv[c];
            }
    }

    void set_sine_ez(double a, double kx, double ky) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = x0 + i * dx, y = y0 + j * dy;
                E[0][id(i, j)] = E[1][id(i, j)] = 0.0;
                E[2][id(i, j)] = a * std::sin(kx * x) * std::sin(ky * y);
                B[0][id(i, j)] = B[1][id(i, j)] = B[2][id(i, j)] = 0.0;
            }
    }

    // Sine Ex(x,y) = a*sin(kx*x) at x-edges (2D gather-convergence tests).
    void set_sine_ex(double a, double kx) {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double xm = x0 + (i + 0.5) * dx;
                E[0][id(i, j)] = a * std::sin(kx * xm);
                E[1][id(i, j)] = 0.0;
                E[2][id(i, j)] = 0.0;
                B[0][id(i, j)] = B[1][id(i, j)] = B[2][id(i, j)] = 0.0;
            }
    }

    // Faraday: Bx[i,j+1/2] between Ez[i,j],Ez[i,j+1]; By[i+1/2,j] between
    // Ez[i,j],Ez[i+1,j]; Bz[i+1/2,j+1/2] between Ey[i,j+1/2],Ey[i+1,j+1/2]
    // and Ex[i+1/2,j],Ex[i+1/2,j+1] (PDF table). Every derivative is a
    // one-cell difference centred on the B component it updates.
    // Sign: dBz/dt = -(curl E)_z = -(dEy/dx - dEx/dy).
    void faraday(double dt) {
        std::vector<double> nBx(E[0].size()), nBy(E[0].size()),
            nBz(E[0].size());
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t c = id(i, j);
                const double dEz_dy =
                    (E[2][id(i, j + 1)] - E[2][c]) / dy;
                const double dEz_dx =
                    (E[2][id(i + 1, j)] - E[2][c]) / dx;
                const double dEy_dx =
                    (E[1][id(i + 1, j)] - E[1][c]) / dx;
                const double dEx_dy =
                    (E[0][id(i, j + 1)] - E[0][c]) / dy;
                nBx[c] = B[0][c] - dt * dEz_dy;
                nBy[c] = B[1][c] + dt * dEz_dx;
                nBz[c] = B[2][c] - dt * (dEy_dx - dEx_dy);
            }
        B[0] = nBx;
        B[1] = nBy;
        B[2] = nBz;
    }

    // Ampere at the nodes (centred differences of the staggered B).
    // Bx, By sit on edges, so a one-cell difference is node-centred. Bz sits
    // at the four surrounding cell centres (i+-1/2, j+-1/2): its one-cell
    // differences are centred on the two adjacent edges and are averaged
    // onto the node.
    void ampere(double* jx, double* jy, double* jz) const {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t c = id(i, j);
                const double dBz_dy =
                    ((B[2][c] - B[2][id(i, j - 1)]) +
                     (B[2][id(i - 1, j)] - B[2][id(i - 1, j - 1)])) /
                    (2 * dy);
                const double dBz_dx =
                    ((B[2][c] - B[2][id(i - 1, j)]) +
                     (B[2][id(i, j - 1)] - B[2][id(i - 1, j - 1)])) /
                    (2 * dx);
                const double dBx_dy =
                    (B[0][c] - B[0][id(i, j - 1)]) / dy;
                const double dBy_dx =
                    (B[1][c] - B[1][id(i - 1, j)]) / dx;
                jx[c] = (dBz_dy) / MU0;            // (curl B)_x, 2D: dBz/dy
                jy[c] = (-dBz_dx) / MU0;           // (curl B)_y: -dBz/dx
                jz[c] = (dBy_dx - dBx_dy) / MU0;   // (curl B)_z
            }
    }

    // Discrete div(B) at cell centres (i+1/2,j+1/2), for the conservation
    // test (PDF section 5.3): (Bx[i+1,j]-Bx[i,j])/dx + (By[i,j+1]-By[i,j])/dy.
    // Forward differences are REQUIRED here, not a style choice: with the
    // Faraday updates above, every Ez node value then enters div(B^{n+1}) -
    // div(B^n) twice with opposite signs and cancels term by term, so div B
    // is conserved to roundoff. A backward difference mixes different cell
    // centres and destroys the cancellation.
    double max_div_b() const {
        double m = 0.0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t c = id(i, j);
                const double dBx =
                    (B[0][id(i + 1, j)] - B[0][c]) / dx;
                const double dBy =
                    (B[1][id(i, j + 1)] - B[1][c]) / dy;
                m = std::max(m, std::fabs(dBx + dBy));
            }
        return m;
    }
};

}  // namespace pic
