import numpy as np
from scipy.integrate import solve_ivp
# %matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import optimize

R_M = 1737.4  # km
mu = 4.904E3  # km**3/s**2

r_LEO = R_M + 10  # km
v_LEO = np.sqrt(mu / r_LEO)  # km/s

r_2 = R_M  # km, GEO

r_0 = np.array((r_LEO, 0, 0))  # km
v_0 = np.array((0, v_LEO, 0))  # km/s
m_0 = np.array((20000))  # kg
Y_0 = np.hstack((r_0, v_0, m_0))

max_T = 4855  # kN
I_sp = 363  # s
g_0 = 9.8E-3  # km/s**2
t_end = 10000  # s
t_eval = np.linspace(0, t_end, int(1E4))


def nonimpulsive_maneuver(t, Y, mu, T, I_sp, g_0, r_2):
    """Residual function for non-impulsive maneuvers.

    t: Current simulation time
    Y: State vector [x y z xdot ydot zdot m], km, km/s, kg
    mu: Gravitational parameter, km**3/s**2
    T: Thrust, kN
    I_sp: Specific impulse, s
    g_0: Standard sea-level gravity, km/s**2
    """
    r = np.sqrt(np.dot(Y[0:3], Y[0:3]))
    v = np.sqrt(np.dot(Y[3:6], Y[3:6]))
    #print(t)
    #print(v)
    #print(r)
    #print(r - r_2 > .01 and v < .1)
    m = Y[-1]
    dY_dt = np.zeros(len(Y))
    dY_dt[0:3] = Y[3:6]
    dY_dt[3:6] = -mu * Y[0:3] / r ** 3 - (T * Y[3:6]) / (m * v)
    dY_dt[-1] = - T / (I_sp * g_0)
    return dY_dt


def reached_destination(t, Y, mu, T, I_sp, g_0, r_2):
    """Determine if the spacecraft reaches the destination radius.

    Returns the difference between the current orbital radius and the
    destination radius.
    """
    r_vec = Y[0:3]
    r = np.sqrt(np.dot(r_vec, r_vec))
    if r - r_2 < .0001:
        return 0
    return 1

reached_destination.terminal = True

def nolanding(t, Y, mu, T, I_sp, g_0, r_2):
    r = np.sqrt(np.dot(Y[0:3], Y[0:3]))
    v =  np.sqrt(np.dot(Y[3:6], Y[3:6]))
    if r - r_2 > .001 and v < .001:
        return 0
    return 1

nolanding.terminal = True

def toofast(t, Y, mu, T, I_sp, g_0, r_2):
    r_vec = Y[0:3]
    v_vec = Y[3:6]
    r = np.sqrt(np.dot(r_vec, r_vec))
    v = np.linalg.norm(v_vec)
    if v > v_LEO:
        return 0
    return 1

toofast.terminal = True

def mass(t, Y, mu, T, I_sp, g_0, r_2):
    """Return the current mass of the spacecraft.

    The mass is stored in the last element of the solution vector.
    If this becomes zero, the integration should terminate.
    """
    return Y[-1]


mass.terminal = True

def sim_orbit_opt(T):
    print(f"attempting {T}")
    if T == 0 :
        return v_LEO
    sol = solve_ivp(
        nonimpulsive_maneuver,
        t_span=(0, t_end),
        y0=Y_0,
        t_eval=t_eval,
        events=(reached_destination, mass, nolanding, toofast),
        rtol=1E-3,
        atol=1E-6,
        method="DOP853",
        args=(mu, T, I_sp, g_0, r_2),
        min_step=.1
    )
    #print(sol.status)

    r_vec = sol.y[0:3].T
    r = np.sqrt(r_vec[:, 0]**2 + r_vec[:, 1]**2 + r_vec[:, 2]**2)
    v_vec = sol.y[3:6].T
    v = np.sqrt(v_vec[:, 0]**2 + v_vec[:, 1]**2 + v_vec[:, 2]**2)
    m = sol.y[-1].T

    final_r = np.linalg.norm(r_vec[-1])
    final_V = np.linalg.norm(v_vec[-1])
    print(f"Thrust:{T} Kn")
    print(f"Impact V: {final_V*1000} m/s")
    print(f"Final Mass: {m[-1]}")
    print(f"end Radius: {final_r}")
    print(sol.message)
    print("")
    return final_V + abs(final_r - R_M)

res = optimize.minimize_scalar(sim_orbit_opt, bounds=(0, max_T), method='bounded', options={'xatol':.001})

T = res.x
sol = solve_ivp(
    nonimpulsive_maneuver,
    t_span=(0, t_end),
    y0=Y_0,
    t_eval=t_eval,
    events=(reached_destination, mass, nolanding, toofast),
    rtol=1E-12,
    atol=1E-15,
    method="DOP853",
    args=(mu, T, I_sp, g_0, r_2)
)
print(sol.status)

r_vec = sol.y[0:3].T
r = np.sqrt(r_vec[:, 0]**2 + r_vec[:, 1]**2 + r_vec[:, 2]**2)
v_vec = sol.y[3:6].T
v = np.sqrt(v_vec[:, 0]**2 + v_vec[:, 1]**2 + v_vec[:, 2]**2)
m = sol.y[-1]






plt.rc("font", size=20)
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
ax.axis("off")
ax.add_patch(Circle((0, 0), R_M, ec="none", fc="C0"))
ax.annotate("Moon", xy=(0, 0), ha="center", va="center")
ax.add_patch(Circle((0, 0), r_2, ec="C1", fc="none", lw=2, ls="--"))
ax.plot(r_vec[:, 0], r_vec[:, 1], color="C2")
#orbit_crossings = sol.y_events[2][:, 0]
#ax.plot(orbit_crossings, np.zeros(orbit_crossings.shape), 'ko', fillstyle='none')
#print(sol.y.shape)
#location = []
#for pos in sol.y:
    #location.append((pos[0], pos[1]))
#ax.plot(location)
#print(np.linalg.norm(sol.y[-1][3:6]))
#print(np.linalg.norm(sol.y[-1][0:3]))
#print(sol.y[-1])
print(f"Final Velocity: {np.linalg.norm(v_vec[-1])* 1000} m/s")
print(f"Final Location: {(np.linalg.norm(r_vec[-1]) - R_M) * 1000} meters off of surface")
plt.show()

pass