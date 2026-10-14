"""Session 3 (Level 3): simulate a car on a curve with a friction circle, and validate it.

Run:  python start.py 3
"""
# %% [markdown]
# # Session 3: simulate a car on a curve and validate it
# **A. Real-world problem.** Session 2 gave formulas for the limiting speed. But what really happens
# when a driver arrives too fast, or brakes in the middle of the bend, or the road turns wet? A
# time-stepping simulation can show the slide itself. We trust it only after it reproduces the
# formulas of Session 2 and obeys energy and momentum rules.
#
# **Model (simplified, educational, not a road design tool).** A car is a point mass on a road
# surface that is tilted inward by `theta` (a cone, so a flat road is theta = 0). The tyres give a
# force in the road surface. The driver asks for a force along the heading (braking or
# accelerating) and a sideways force to follow the circle of radius R. The **friction circle**: the
# total tyre force cannot exceed mu * N. If the request is larger, both parts are scaled down
# together (so braking costs cornering grip) and the car drifts wide.

# %%
import math
import pathlib

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path(".")) / "out"
G = 9.81            # m/s^2
KP, KD = 1.0, 1.4   # the driver's steering feedback gains (1/s^2 and 1/s)


# %% [markdown]
# ## Step 1: the equations the computer steps through (per unit mass)
# State: horizontal position (x, y) and velocity (vx, vy). With radius rho = |(x, y)| and tangential
# speed v_phi, the road normal force is the one that keeps the car on the surface:
#
#     N = g cos(theta) + (v_phi**2 / rho) sin(theta)
#
# The tyre force F = F_long (along the heading) + F_side (sideways, in the road surface). The driver
# wants F_side to give the inward acceleration v_phi**2 / R plus a correction KP (rho - R) + KD v_rho
# (more inward push when the car is outside the lane or moving outward) that steers back to the circle. Friction circle: if |F| > mu N, scale F down to mu N.
# We advance the state in time with the classical 4th-order Runge-Kutta method.

# %%
def _accel(x, y, vx, vy, R, mu, theta, a_long):
    """Horizontal acceleration (ax, ay) and the demand ratio |F| / (mu N) (per unit mass)."""
    rho = math.hypot(x, y)
    rx, ry = x / rho, y / rho                       # outward unit vector
    vrho = vx * rx + vy * ry
    vphi = -vx * ry + vy * rx
    st, ct = math.sin(theta), math.cos(theta)
    vz = math.tan(theta) * vrho
    s = math.sqrt(vx * vx + vy * vy + vz * vz)
    N = G * ct + vphi * vphi / rho * st
    ex, ey, ez = vx / s, vy / s, vz / s             # heading in the road surface
    nx, ny, nz = -st * rx, -st * ry, ct             # road normal
    lx, ly, lz = ny * ez - nz * ey, nz * ex - nx * ez, nx * ey - ny * ex   # sideways = n x e (inward)
    inward = -(lx * rx + ly * ry)                   # how much of "sideways" points to the centre
    inward = max(inward, 0.3)
    a_in = vphi * vphi / R + KP * (rho - R) + KD * vrho
    f_side = (a_in - N * st) / inward
    f_long = a_long
    cap = mu * N
    mag = math.hypot(f_side, f_long)
    scale = 1.0 if mag <= cap else (cap / mag if mag > 0 else 0.0)
    f_side *= scale
    f_long *= scale
    ax = N * nx + f_long * ex + f_side * lx
    ay = N * ny + f_long * ey + f_side * ly
    return ax, ay, (mag / cap if cap > 0 else (math.inf if mag > 0 else 0.0))


def simulate(v0, R, mu, theta=0.0, a_long=0.0, T=20.0, dt=0.01, stop_drift=None, rho0=None):
    """Drive a car that starts at (rho0, 0) heading counter-clockwise at v0 (m/s, horizontal speed).
    Returns a dict of arrays: t, x, y, v (road speed), rho, ratio (demand / capacity), energy."""
    theta = float(theta)
    rho0 = R if rho0 is None else rho0
    st = np.array([rho0, 0.0, 0.0, v0])
    n = int(round(T / dt))
    keys = ("t", "x", "y", "v", "rho", "ratio", "energy")
    rec = {k: [] for k in keys}

    def f(s):
        ax, ay, _ = _accel(s[0], s[1], s[2], s[3], R, mu, theta, a_long)
        return np.array([s[2], s[3], ax, ay])

    for i in range(n + 1):
        x, y, vx, vy = st
        rho = math.hypot(x, y)
        vz = math.tan(theta) * (x * vx + y * vy) / rho
        _, _, ratio = _accel(x, y, vx, vy, R, mu, theta, a_long)
        for k, val in zip(keys, (i * dt, x, y, math.sqrt(vx * vx + vy * vy + vz * vz), rho, ratio,
                                 0.5 * (vx * vx + vy * vy + vz * vz) + G * math.tan(theta) * rho)):
            rec[k].append(val)
        if stop_drift is not None and abs(rho - R) > stop_drift:
            break
        if i == n:
            break
        k1 = f(st)
        k2 = f(st + 0.5 * dt * k1)
        k3 = f(st + 0.5 * dt * k2)
        k4 = f(st + dt * k3)
        st = st + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {k: np.array(v) for k, v in rec.items()}


# %% [markdown]
# ## Step 2: when does the car "hold the road"?
# We call it a **slide** if the car drifts more than 1.0 m from the lane radius within 20 s.
# `sim_limit` finds the highest (or lowest) speed that still holds, by bisection.

# %%
def holds(v0, R, mu, theta=0.0, T=20.0, dt=0.01, tol=1.0):
    out = simulate(v0, R, mu, theta, 0.0, T, dt, stop_drift=tol)
    return bool(np.abs(out["rho"] - R).max() <= tol)


def sim_limit(R, mu, theta=0.0, which="max", T=20.0, dt=0.01, iters=11):
    """Bisection on the start speed. which = 'max' or 'min'. Returns a speed in m/s."""
    t = math.tan(theta)
    v_ideal = math.sqrt(G * R * t) if t > 0 else 0.0
    if which == "max":
        lo, hi = v_ideal, 3.0 * math.sqrt(G * R * (t + max(mu, 0.05)) / max(1 - mu * t, 0.1)) + 5
        if holds(hi, R, mu, theta, T, dt):
            return math.inf
    else:
        lo, hi = 1.0, v_ideal
        if holds(lo, R, mu, theta, T, dt):
            return 0.0
        lo, hi = hi, lo          # lo = a speed that holds, hi = one that slides
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if holds(mid, R, mu, theta, T, dt):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# %% [markdown]
# ## Step 3: analytic references to compare with
# * flat or banked, with friction: the limits of Session 2.
# * with braking or accelerating by `a` (m/s^2) on a flat road, the friction circle leaves lateral
#   acceleration sqrt((mu g)**2 - a**2), so the slide speed drops to
#   `v = sqrt(R sqrt((mu g)**2 - a**2))`.
# * once sliding without braking, the car keeps its speed and its path is a circle of radius
#   `v**2 / (mu g)` (flat road): larger than the lane, so it runs wide (understeer).

# %%
def analytic_limits(R, mu, theta=0.0):
    t = math.tan(theta)
    vmax2 = G * R * (t + mu) / (1 - mu * t) if mu * t < 1 else math.inf
    vmin2 = G * R * (t - mu) / (1 + mu * t)
    return math.sqrt(max(vmin2, 0.0)), math.sqrt(vmax2)


def combined_slide_speed(R, mu, a_long):
    """Flat road: speed at which the friction circle is full when also braking/accelerating at a_long."""
    rest = (mu * G) ** 2 - a_long ** 2
    return math.sqrt(R * math.sqrt(rest)) if rest > 0 else 0.0


def skid_radius(v, mu):
    """Flat road, saturated friction, no braking: radius of the path of a sliding car."""
    return v * v / (mu * G)


def skid_position(v, mu, R, t):
    """Flat road, saturated friction, no braking, entering at (R, 0) heading +y at speed v.
    The car moves on a circle of radius v**2 / (mu g) whose centre is at (R - rho_s, 0)."""
    rs = skid_radius(v, mu)
    ang = v * t / rs
    return np.array([R - rs + rs * math.cos(ang), rs * math.sin(ang)])


def grip_loss_speed(R, mu, a_long, v_start=3.0, T=40.0, dt=0.01):
    """Accelerate (a_long > 0) from v_start on the lane circle until the demand first exceeds the
    capacity; return the speed at that moment."""
    out = simulate(v_start, R, mu, 0.0, a_long, T, dt)
    idx = np.nonzero(out["ratio"] > 1.0 + 1e-9)[0]
    return float(out["v"][idx[0]]) if len(idx) else math.inf


# %% [markdown]
# ## Step 4: a roller-coaster loop (vertical circle)
# At the top of a loop of radius r the track pushes down with N and gravity also points down:
# `N + m g = m v**2 / r`. The cart stays on the track while N >= 0, so the minimum speed at the top
# is `v = sqrt(g r)`. Energy conservation (the bottom is 2 r lower) then gives the minimum speed at
# the bottom: `v_bottom = sqrt(5 g r)`. We integrate the motion and look for the first moment N < 0.

# %%
def loop_ride(v_bottom, r, dt=1e-3):
    """Frictionless cart entering a vertical loop at the bottom with speed v_bottom (m/s).
    Returns (reached_top, min_N_over_m, v_top)."""
    phi, w = 0.0, v_bottom / r            # angle from the bottom, angular speed
    min_n = math.inf
    for _ in range(200000):
        n_over_m = r * w * w + G * math.cos(phi)    # normal force per unit mass
        min_n = min(min_n, n_over_m)
        if n_over_m < 0 or w <= 0:
            return False, min_n, math.nan
        if phi >= math.pi:
            return True, min_n, r * w
        def acc(p):
            return -(G / r) * math.sin(p)
        k1p, k1w = w, acc(phi)
        k2p, k2w = w + 0.5 * dt * k1w, acc(phi + 0.5 * dt * k1p)
        k3p, k3w = w + 0.5 * dt * k2w, acc(phi + 0.5 * dt * k2p)
        k4p, k4w = w + dt * k3w, acc(phi + dt * k3p)
        phi += dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        w += dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return False, min_n, math.nan


def loop_min_bottom_speed(r, dt=1e-3, iters=30):
    lo, hi = 0.5 * math.sqrt(G * r), 3.0 * math.sqrt(G * r)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if loop_ride(mid, r, dt)[0]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# %% [markdown]
# ## Step 5: mini-project helper - a speed limit with a safety margin
# Rule used here: take the top speed on a WET road from Session 2, multiply by a safety margin
# (0.75 means "stay 25 percent under the limit"), and round DOWN to a multiple of 5 km/h.

# %%
def posted_speed_limit(R, theta_deg, mu_wet, margin=0.75):
    """Speed limit sign (km/h) for a curve. Educational rule, not a standard."""
    vmax = analytic_limits(R, mu_wet, math.radians(theta_deg))[1]
    return int(math.floor(margin * vmax * 3.6 / 5.0)) * 5


# %%
if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    R = 100.0

    # (a) validation of the limits (flat/banked, with and without friction)
    print("limit speeds, R = 100 m: simulation versus analytic (km/h)")
    cases = [("flat dry mu 0.8", 0.8, 0.0), ("flat wet mu 0.4", 0.4, 0.0), ("flat ice mu 0.1", 0.1, 0.0),
             ("bank 10 deg, mu 0.4", 0.4, math.radians(10)), ("bank 30 deg, mu 0.3", 0.3, math.radians(30))]
    rows = []
    for name, mu, th in cases:
        lo_a, hi_a = analytic_limits(R, mu, th)
        hi_s = sim_limit(R, mu, th, "max")
        lo_s = sim_limit(R, mu, th, "min")
        rows.append((name, hi_a, hi_s))
        print(f"  {name:22s} top: analytic {hi_a * 3.6:6.1f}  sim {hi_s * 3.6:6.1f}   lowest: analytic {lo_a * 3.6:5.1f}  sim {lo_s * 3.6:5.1f}")

    # (b) frictionless banked curve: only the ideal speed holds; energy is conserved
    th = math.radians(20)
    vi = math.sqrt(G * R * math.tan(th))
    for factor in (1.0, 0.9, 1.1):
        out = simulate(factor * vi, R, 0.0, th, 0.0, 20.0, 0.005)
        drift = np.abs(out["rho"] - R).max()
        e = out["energy"]
        print(f"frictionless 20 deg bank, speed {factor:.1f} x ideal ({factor * vi * 3.6:.1f} km/h): max drift {drift:.2f} m, "
              f"energy change {abs(e.max() - e.min()) / e[0]:.1e}")

    # (c) braking or accelerating on a flat wet curve
    mu = 0.4
    vflat = math.sqrt(mu * G * R)
    for a in (1.0, 2.0, 3.0):
        vs = grip_loss_speed(R, mu, a)
        print(f"accelerating at {a:.0f} m/s^2: grip lost at {vs * 3.6:.1f} km/h (formula {combined_slide_speed(R, mu, a) * 3.6:.1f})")
    print("braking inside the curve at 3 m/s^2, mu 0.4 (flat top speed 71.3 km/h); ratio = demand / capacity:")
    for fr in (0.7, 0.8, 0.9):
        v0 = fr * vflat
        coast = simulate(v0, R, mu, 0.0, 0.0, 4.0, 0.01)
        brake = simulate(v0, R, mu, 0.0, -3.0, 4.0, 0.01)
        print(f"  enter at {fr:.1f} x top speed ({v0 * 3.6:.1f} km/h): coasting max ratio {coast['ratio'].max():.2f}, drift {np.abs(coast['rho'] - R).max():.2f} m;"
              f" braking max ratio {brake['ratio'].max():.2f}, drift {np.abs(brake['rho'] - R).max():.2f} m")
    print(f"  formula: with 3 m/s^2 of braking the slide speed drops from {vflat * 3.6:.1f} to {combined_slide_speed(R, mu, -3.0) * 3.6:.1f} km/h")

    # (d) failure case: entering too fast
    v_in = 1.3 * vflat
    out = simulate(v_in, R, mu, 0.0, 0.0, 6.0, 0.005)
    print(f"failure case: enter at 1.3 x top speed ({v_in * 3.6:.1f} km/h): after 6 s the car is {out['rho'][-1]:.1f} m from the centre, "
          f"speed {out['v'][-1] * 3.6:.1f} km/h; skid radius {skid_radius(v_in, mu):.1f} m")
    sp = skid_position(v_in, mu, R, 6.0)
    print(f"  analytic skid circle puts it at ({sp[0]:.1f}, {sp[1]:.1f}) m, simulation ({out['x'][-1]:.1f}, {out['y'][-1]:.1f}) m")

    # (e) loop
    for r in (5.0, 10.0, 20.0):
        print(f"loop r = {r:.0f} m: minimum bottom speed simulated {loop_min_bottom_speed(r):.2f} m/s, sqrt(5 g r) = {math.sqrt(5 * G * r):.2f} m/s; "
              f"top speed needed sqrt(g r) = {math.sqrt(G * r):.2f} m/s")

    # (f) mini-project
    sign = posted_speed_limit(80.0, 4.0, 0.35)
    print("posted limit for R = 80 m, bank 4 deg, wet mu 0.35:", sign, "km/h; wet top speed",
          round(analytic_limits(80.0, 0.35, math.radians(4))[1] * 3.6, 1), "km/h")
    for name, mu_ in (("wet 0.35", 0.35), ("ice 0.1", 0.1)):
        print(f"  simulated at the sign speed on {name}:", "holds" if holds(sign / 3.6, 80.0, mu_, math.radians(4)) else "SLIDES")
    print(f"(run time so far {time.time() - t0:.1f} s)")

    # Figures
    fig, ax = plt.subplots(figsize=(6, 6))
    th_ = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R * np.cos(th_), R * np.sin(th_), "k--", lw=1, label="lane, R = 100 m")
    for fr, col in ((0.9, "tab:green"), (1.3, "tab:red")):
        o = simulate(fr * vflat, R, mu, 0.0, 0.0, 8.0, 0.01)
        ax.plot(o["x"], o["y"], color=col, label=f"enter at {fr:.1f} x top speed ({fr * vflat * 3.6:.0f} km/h)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Wet road (mu = 0.4): holding the lane, and the understeer slide")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "slide_paths.png", dpi=120)

    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    radii = np.linspace(20, 250, 60)
    for name, m_, col in (("dry mu 0.8", 0.8, "tab:green"), ("wet mu 0.4", 0.4, "tab:blue"), ("ice mu 0.1", 0.1, "tab:gray")):
        ax2.plot(radii, [math.sqrt(m_ * G * rr) * 3.6 for rr in radii], color=col, label=f"{name} (analytic)")
    for rr, m_, col in ((50.0, 0.8, "tab:green"), (100.0, 0.4, "tab:blue"), (200.0, 0.1, "tab:gray"), (100.0, 0.1, "tab:gray"), (50.0, 0.4, "tab:blue")):
        ax2.plot(rr, sim_limit(rr, m_, 0.0, "max") * 3.6, "o", color=col, mec="k", ms=6)
    ax2.plot([], [], "o", color="w", mec="k", label="simulation")
    ax2.set_xlabel("curve radius R (m)")
    ax2.set_ylabel("highest speed that holds the lane (km/h)")
    ax2.set_title("Sensitivity to friction and radius, flat road")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "limit_vs_radius.png", dpi=120)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    vb = np.linspace(0.6, 1.4, 40) * math.sqrt(5 * G * 10.0)
    ax3.plot(vb, [loop_ride(v, 10.0, 2e-3)[1] if loop_ride(v, 10.0, 2e-3)[1] != math.inf else 0 for v in vb], lw=2)
    ax3.axhline(0, color="k", lw=0.8)
    ax3.axvline(math.sqrt(5 * G * 10.0), color="tab:red", ls="--", label="sqrt(5 g r) = 22.1 m/s")
    ax3.set_xlabel("speed at the bottom (m/s)")
    ax3.set_ylabel("smallest track force per kg on the way (N/kg)")
    ax3.set_title("Loop r = 10 m: the track force must stay above zero")
    ax3.set_ylim(-12, 40)
    ax3.grid(alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(OUT / "loop_minimum.png", dpi=120)
    print("saved figures in", OUT)

# %% [markdown]
# ## Where this stops
# A point mass has no weight transfer, no tyre slip angle, no rollover and no suspension. The friction
# circle with one constant mu is the simplest tyre model that still shows understeer. Use the
# roadmap to add the missing pieces.
