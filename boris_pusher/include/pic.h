#pragma once
// PIC layer on top of include/boris.h (header-only, C++17).
//
// One PIC step (course PDF section 3):
//   1. gather E,B at each particle's mid-step position r^{n+1/2}
//      (Q(r) = sum Qij S(rp - rij), same S both directions);
//   2. move particles with the Boris kick-rotate-kick (PDF section 4.4);
//   3. deposit moments nij, vij (eq. 8-9) with linear CIC weights (eq. 10);
//   4. field solve: Faraday (dB/dt = -curl E), Ampere (mu0 j = curl B),
//      Ohm (E = -ve x B - grad(pe)/en, ve = vi - j/en).
//   5. t += dt.
//
// The velocity update is arithmetically identical to boris_push() in
// boris.h; only the position update is split into two half-drifts around
// the gather (PDF eq. 15), and the E,B inputs come from the mesh instead
// of being uniform constants.

#include <algorithm>
#include <cmath>
#include <vector>

#include "boris.h"
#include "pic_grid.h"

namespace pic {

// Particle with a macro weight w (number of real ions it stands for).
struct MacroParticle {
    Particle p;     // q, m, x, v (same meaning as boris.h)
    double w = 1.0; // statistical weight
};

// --- linear CIC (order-1 B-spline, PDF eq. 10) -----------------------------
// Node index + local coordinate for one dimension.
inline void cic_1d(double x, double x0, double dx, int n, int& i0,
                   double& fx) {
    double u = (x - x0) / dx;
    // Periodic wrap into [0, n).
    u -= std::floor(u / n) * n;
    int i = (int)std::floor(u);
    fx = u - i;
    i0 = ((i % n) + n) % n;
}

// --- gather ----------------------------------------------------------------
// Interpolate one Yee component at (x,y) with CIC, honouring its half-cell
// offset (ox, oy in cell units).
inline double gather_comp_2d(const std::vector<double>& f, int nx, int ny,
                             double x0, double y0, double dx, double dy,
                             double x, double y, double ox, double oy,
                             int& gx0, int& gy0, double& gwx, double& gwy) {
    int ix, iy;
    double fx, fy;
    cic_1d(x - ox * dx, x0, dx, nx, ix, fx);
    cic_1d(y - oy * dy, y0, dy, ny, iy, fy);
    auto at = [&](int i, int j) -> double {
        i %= nx;
        j %= ny;
        if (i < 0) i += nx;
        if (j < 0) j += ny;
        return f[(size_t)i + (size_t)j * (size_t)nx];
    };
    gx0 = ix;
    gy0 = iy;
    gwx = fx;
    gwy = fy;
    return at(ix, iy) * (1 - fx) * (1 - fy) + at(ix + 1, iy) * fx * (1 - fy) +
           at(ix, iy + 1) * (1 - fx) * fy + at(ix + 1, iy + 1) * fx * fy;
}

inline void gather_1d(const Grid1D& g, double x, Vec3& E, Vec3& B) {
    double ev[3] = {0, 0, 0}, bv[3] = {0, 0, 0};
    for (int c = 0; c < 3; ++c) {
        // E and B share the 1D Yee offsets (x-components at mid-cells).
        int i0;
        double fx;
        cic_1d(x - offset_x_1d(c) * g.dx, g.x0, g.dx, g.nx, i0, fx);
        const int i1 = (i0 + 1) % g.nx;
        ev[c] = g.E[c][(size_t)i0] * (1 - fx) + g.E[c][(size_t)i1] * fx;
        bv[c] = g.B[c][(size_t)i0] * (1 - fx) + g.B[c][(size_t)i1] * fx;
    }
    E = {ev[0], ev[1], ev[2]};
    B = {bv[0], bv[1], bv[2]};
}

inline void gather_2d(const Grid2D& g, double x, double y, Vec3& E, Vec3& B) {
    double ev[3] = {0, 0, 0}, bv[3] = {0, 0, 0};
    for (int c = 0; c < 3; ++c) {
        int ix, iy;
        double fx, fy;
        ev[c] = gather_comp_2d(g.E[c], g.nx, g.ny, g.x0, g.y0, g.dx, g.dy, x,
                               y, offset_x_2d(true, c), offset_y_2d(true, c),
                               ix, iy, fx, fy);
        bv[c] = gather_comp_2d(g.B[c], g.nx, g.ny, g.x0, g.y0, g.dx, g.dy, x,
                               y, offset_x_2d(false, c),
                               offset_y_2d(false, c), ix, iy, fx, fy);
    }
    E = {ev[0], ev[1], ev[2]};
    B = {bv[0], bv[1], bv[2]};
}

// --- Boris velocity update (no position change) ----------------------------
// Same arithmetic as boris_push() in boris.h, factored out so the PIC
// stepper can split the drift around the gather.
inline void boris_kick_rotate(Particle& p, const Vec3& E, const Vec3& B,
                              double dt) {
    const double h = (p.q / p.m) * dt * 0.5;
    Vec3 v_minus = p.v + E * h;
    Vec3 t = B * h;
    Vec3 v_prime = v_minus + cross(v_minus, t);
    Vec3 s = t * (2.0 / (1.0 + norm2(t)));
    Vec3 v_plus = v_minus + cross(v_prime, s);
    p.v = v_plus + E * h;
}

// --- deposit ---------------------------------------------------------------
// 1D: n[i] (number density), vi (bulk velocity) at nodes.
inline void deposit_1d(const std::vector<MacroParticle>& ps, const Grid1D& g,
                       std::vector<double>& n, std::vector<double>& vx,
                       std::vector<double>& vy, std::vector<double>& vz) {
    const double dV = g.dx;
    n.assign((size_t)g.nx, 0.0);
    vx.assign((size_t)g.nx, 0.0);
    vy.assign((size_t)g.nx, 0.0);
    vz.assign((size_t)g.nx, 0.0);
    for (const auto& mp : ps) {
        int i0;
        double fx;
        cic_1d(mp.p.x.x, g.x0, g.dx, g.nx, i0, fx);
        const int i1 = (i0 + 1) % g.nx;
        const double w0 = mp.w * (1 - fx), w1 = mp.w * fx;
        n[(size_t)i0] += w0 / dV;
        n[(size_t)i1] += w1 / dV;
        vx[(size_t)i0] += mp.p.v.x * w0 / dV;
        vx[(size_t)i1] += mp.p.v.x * w1 / dV;
        vy[(size_t)i0] += mp.p.v.y * w0 / dV;
        vy[(size_t)i1] += mp.p.v.y * w1 / dV;
        vz[(size_t)i0] += mp.p.v.z * w0 / dV;
        vz[(size_t)i1] += mp.p.v.z * w1 / dV;
    }
    for (int i = 0; i < g.nx; ++i) {
        if (n[(size_t)i] > 0) {
            vx[(size_t)i] /= n[(size_t)i];
            vy[(size_t)i] /= n[(size_t)i];
            vz[(size_t)i] /= n[(size_t)i];
        }
    }
}

// 2D: node-centred moments (deposit uses the node-based CIC weights; the
// Yee offsets only enter the gather and the curl stencils).
inline void deposit_2d(const std::vector<MacroParticle>& ps, const Grid2D& g,
                       std::vector<double>& n, std::vector<double>& vx,
                       std::vector<double>& vy, std::vector<double>& vz) {
    const double dV = g.dx * g.dy;
    const size_t N = (size_t)g.nx * (size_t)g.ny;
    n.assign(N, 0.0);
    vx.assign(N, 0.0);
    vy.assign(N, 0.0);
    vz.assign(N, 0.0);
    for (const auto& mp : ps) {
        int ix, iy;
        double fx, fy;
        cic_1d(mp.p.x.x, g.x0, g.dx, g.nx, ix, fx);
        cic_1d(mp.p.x.y, g.y0, g.dy, g.ny, iy, fy);
        const double w00 = mp.w * (1 - fx) * (1 - fy);
        const double w10 = mp.w * fx * (1 - fy);
        const double w01 = mp.w * (1 - fx) * fy;
        const double w11 = mp.w * fx * fy;
        const size_t c[4] = {g.id(ix, iy), g.id(ix + 1, iy), g.id(ix, iy + 1),
                             g.id(ix + 1, iy + 1)};
        const double w[4] = {w00, w10, w01, w11};
        for (int k = 0; k < 4; ++k) {
            n[c[k]] += w[k] / dV;
            vx[c[k]] += mp.p.v.x * w[k] / dV;
            vy[c[k]] += mp.p.v.y * w[k] / dV;
            vz[c[k]] += mp.p.v.z * w[k] / dV;
        }
    }
    for (size_t k = 0; k < N; ++k) {
        if (n[k] > 0) {
            vx[k] /= n[k];
            vy[k] /= n[k];
            vz[k] /= n[k];
        }
    }
}

// --- generalized Ohm law (hybrid closure, isothermal electrons) ------------
// E = -ve x B - grad(pe)/(e n), ve = vi - j/(e n), pe = n kTe.
// grad(pe) is a centred difference; n is floored to keep empty cells sane.
// The me*dve/dt inertia term is dropped (documented v1 simplification).
inline void ohm_1d(Grid1D& g, const std::vector<double>& n,
                   const std::vector<double>& vx,
                   const std::vector<double>& vy,
                   const std::vector<double>& vz, double kTe) {
    std::vector<double> jx((size_t)g.nx), jy((size_t)g.nx), jz((size_t)g.nx);
    g.ampere(jx.data(), jy.data(), jz.data());
    // Vacuum treatment: ve = vi - j/(e n) divides by the density, so
    // near-empty cells would produce astronomic E and blow up in a few
    // steps (the classic hybrid-PIC vacuum problem). Floor n at 2% of the
    // global mean density — a standard cheap remedy; a resistive or
    // masked-vacuum Ohm law is the usual next step.
    double mean = 0.0;
    for (double v : n) mean += v;
    mean /= g.nx;
    const double nfloor = std::max(0.02 * mean, 1e-30 / g.dx);
    for (int i = 0; i < g.nx; ++i) {
        const size_t k = (size_t)i;
        const double nn = std::max(n[k], nfloor);
        const double vex = vx[k] - jx[k] / (ELEM * nn);
        const double vey = vy[k] - jy[k] / (ELEM * nn);
        const double vez = vz[k] - jz[k] / (ELEM * nn);
        // B at the node for the cross product: average the two neighbouring
        // mid-cell values back to the node for x-components.
        const int im = (i - 1 + g.nx) % g.nx;
        const double bxn = g.B[0][k];
        const double byn = 0.5 * (g.B[1][k] + g.B[1][(size_t)im]);
        const double bzn = 0.5 * (g.B[2][k] + g.B[2][(size_t)im]);
        // -ve x B
        double ex = -(vey * bzn - vez * byn);
        double ey = -(vez * bxn - vex * bzn);
        double ez = -(vex * byn - vey * bxn);
        // -grad(pe)/(e n), pe = n kTe, centred at the node.
        const int ip = (i + 1) % g.nx;
        const double dndx =
            (std::max(n[(size_t)ip], 0.0) - std::max(n[(size_t)im], 0.0)) /
            (2 * g.dx);
        ex += -(kTe / ELEM) * dndx / nn;
        g.E[0][k] = ex;
        g.E[1][k] = ey;
        g.E[2][k] = ez;
    }
    // Mid-cell Ex must track the nodes for the gather: copy the average of
    // the two surrounding nodes (keeps uniform fields exact). Read from a
    // copy: in place, i = nx-1 would wrap onto the already-averaged E[0][0].
    const std::vector<double> exn = g.E[0];
    for (int i = 0; i < g.nx; ++i) {
        const int ip = (i + 1) % g.nx;
        g.E[0][(size_t)i] = 0.5 * (exn[(size_t)i] + exn[(size_t)ip]);
    }
}

inline void ohm_2d(Grid2D& g, const std::vector<double>& n,
                   const std::vector<double>& vx,
                   const std::vector<double>& vy,
                   const std::vector<double>& vz, double kTe) {
    const size_t N = (size_t)g.nx * (size_t)g.ny;
    std::vector<double> jx(N), jy(N), jz(N);
    g.ampere(jx.data(), jy.data(), jz.data());
    // Vacuum treatment, see ohm_1d: floor at 2% of mean density.
    double mean = 0.0;
    for (double v : n) mean += v;
    mean /= N;
    const double nfloor = std::max(0.02 * mean, 1e-30 / (g.dx * g.dy));
    std::vector<double> exn(N), eyn(N);  // node-centred Ex, Ey
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const size_t c = g.id(i, j);
            const double nn = std::max(n[c], nfloor);
            const double vex = vx[c] - jx[c] / (ELEM * nn);
            const double vey = vy[c] - jy[c] / (ELEM * nn);
            const double vez = vz[c] - jz[c] / (ELEM * nn);
            // B averaged to the node: edge components from their two
            // neighbours, cell-centred Bz from the four surrounding cells.
            const double bxn =
                0.5 * (g.B[0][c] + g.B[0][g.id(i, j - 1)]);
            const double byn =
                0.5 * (g.B[1][c] + g.B[1][g.id(i - 1, j)]);
            const double bzn =
                0.25 * (g.B[2][c] + g.B[2][g.id(i - 1, j)] +
                        g.B[2][g.id(i, j - 1)] + g.B[2][g.id(i - 1, j - 1)]);
            double ex = -(vey * bzn - vez * byn);
            double ey = -(vez * bxn - vex * bzn);
            double ez = -(vex * byn - vey * bxn);
            const double dndx = (std::max(n[g.id(i + 1, j)], 0.0) -
                                 std::max(n[g.id(i - 1, j)], 0.0)) /
                                (2 * g.dx);
            const double dndy = (std::max(n[g.id(i, j + 1)], 0.0) -
                                 std::max(n[g.id(i, j - 1)], 0.0)) /
                                (2 * g.dy);
            ex += -(kTe / ELEM) * dndx / nn;
            ey += -(kTe / ELEM) * dndy / nn;
            exn[c] = ex;
            eyn[c] = ey;
            g.E[2][c] = ez;  // Ez lives at the nodes: no shift needed.
        }
    // Ex lives on x-edges (i+1/2,j), Ey on y-edges (i,j+1/2): average the
    // node values onto them, as ohm_1d does for Ex (uniform fields exact).
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const size_t c = g.id(i, j);
            g.E[0][c] = 0.5 * (exn[c] + exn[g.id(i + 1, j)]);
            g.E[1][c] = 0.5 * (eyn[c] + eyn[g.id(i, j + 1)]);
        }
}

// Periodic particle wrap into the domain.
inline void wrap_1d(MacroParticle& mp, const Grid1D& g) {
    const double L = g.nx * g.dx;
    double u = mp.p.x.x - g.x0;
    u -= std::floor(u / L) * L;
    mp.p.x.x = g.x0 + u;
}

inline void wrap_2d(MacroParticle& mp, const Grid2D& g) {
    const double Lx = g.nx * g.dx, Ly = g.ny * g.dy;
    double ux = mp.p.x.x - g.x0, uy = mp.p.x.y - g.y0;
    ux -= std::floor(ux / Lx) * Lx;
    uy -= std::floor(uy / Ly) * Ly;
    mp.p.x.x = g.x0 + ux;
    mp.p.x.y = g.y0 + uy;
}

// --- full PIC steps ---------------------------------------------------------
struct PicOpts {
    double kTe = 0.0;      // electron temperature (J); 0 = cold, no pres. term
    bool freeze_E = false; // skip Ohm update (prescribed-field tests)
    bool freeze_B = false; // skip Faraday update (prescribed-field tests)
};

// 1D PIC step on (x, vx,vy,vz): half-drift, gather, Boris, half-drift,
// deposit, field solve. Returns max |E| for diagnostics.
inline double pic_step_1d(std::vector<MacroParticle>& ps, Grid1D& g, double dt,
                          const PicOpts& opt, std::vector<double>& n,
                          std::vector<double>& vx, std::vector<double>& vy,
                          std::vector<double>& vz) {
    for (auto& mp : ps) {
        mp.p.x.x += mp.p.v.x * dt * 0.5; // r^{n+1/2}
        wrap_1d(mp, g);
        Vec3 E, B;
        gather_1d(g, mp.p.x.x, E, B);
        boris_kick_rotate(mp.p, E, B, dt);
        mp.p.x.x += mp.p.v.x * dt * 0.5; // r^{n+1}
        wrap_1d(mp, g);
    }
    deposit_1d(ps, g, n, vx, vy, vz);
    if (!opt.freeze_E) ohm_1d(g, n, vx, vy, vz, opt.kTe);
    if (!opt.freeze_B) g.faraday(dt);
    double m = 0.0;
    for (int c = 0; c < 3; ++c)
        for (double v : g.E[c]) m = std::max(m, std::fabs(v));
    return m;
}

inline double pic_step_2d(std::vector<MacroParticle>& ps, Grid2D& g, double dt,
                          const PicOpts& opt, std::vector<double>& n,
                          std::vector<double>& vx, std::vector<double>& vy,
                          std::vector<double>& vz) {
    for (auto& mp : ps) {
        mp.p.x.x += mp.p.v.x * dt * 0.5; // r^{n+1/2}
        mp.p.x.y += mp.p.v.y * dt * 0.5;
        wrap_2d(mp, g);
        Vec3 E, B;
        gather_2d(g, mp.p.x.x, mp.p.x.y, E, B);
        boris_kick_rotate(mp.p, E, B, dt);
        mp.p.x.x += mp.p.v.x * dt * 0.5; // r^{n+1}
        mp.p.x.y += mp.p.v.y * dt * 0.5;
        wrap_2d(mp, g);
    }
    deposit_2d(ps, g, n, vx, vy, vz);
    if (!opt.freeze_E) ohm_2d(g, n, vx, vy, vz, opt.kTe);
    if (!opt.freeze_B) g.faraday(dt);
    double m = 0.0;
    for (int c = 0; c < 3; ++c)
        for (double v : g.E[c]) m = std::max(m, std::fabs(v));
    return m;
}

}  // namespace pic
