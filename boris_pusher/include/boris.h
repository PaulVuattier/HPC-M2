#pragma once

#include <vector>

// Non-relativistic Boris pusher (header-only).
//
// Solves m dv/dt = q (E + v x B) with a leapfrog stagger:
//   given x_n and v_{n-1/2}, one call advances to x_{n+1}, v_{n+1/2}.
// The magnetic rotation is time-reversible and preserves |v| exactly
// when E = 0 (pure rotation, no secular energy drift).

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

inline Vec3 operator*(double s, const Vec3& a) {
    return a * s;
}

inline Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline double norm2(const Vec3& a) {
    return dot(a, a);
}

struct Particle {
    Vec3 x;       // position
    Vec3 v;       // velocity (staggered by half step in leapfrog sense)
    double q = 1.0;
    double m = 1.0;
};

// Advance one particle by dt through uniform E, B over the step.
inline void boris_push(Particle& p, const Vec3& E, const Vec3& B, double dt) {
    const double qm = p.q / p.m;
    const double h = qm * dt * 0.5;

    // 1) First half electric kick.
    Vec3 v_minus = p.v + E * h;

    // 2) Magnetic rotation.
    Vec3 t = B * h;  // t = (q/m) B dt/2 ; safe when B = 0 (t = 0).
    Vec3 v_prime = v_minus + cross(v_minus, t);
    Vec3 s = t * (2.0 / (1.0 + norm2(t)));
    Vec3 v_plus = v_minus + cross(v_prime, s);

    // 3) Second half electric kick.
    p.v = v_plus + E * h;

    // 4) Drift: update position with the new velocity.
    p.x += p.v * dt;
}

// Advance a whole particle array by dt (serial loop; each particle is
// independent, so this loop is the future OpenMP/GPU parallelization point).
//
// That independence is the contract: pushing the array must be identical to
// pushing each particle on its own. `boris_pusher --selftest-array` asserts it
// bitwise, and is the regression guard to run after any rewrite of this loop.
inline void boris_push_array(std::vector<Particle>& ps, const Vec3& E,
                             const Vec3& B, double dt) {
    for (auto& p : ps) boris_push(p, E, B, dt);
}
