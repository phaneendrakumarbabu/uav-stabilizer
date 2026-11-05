"""
Aircraft physical and aerodynamic parameters for the UAV stability simulation.

All angles are stored internally in RADIANS. All derivatives are per-radian
unless noted otherwise. Body axes: x=forward, y=right, z=down (standard
aerospace body frame, as used throughout dynamics_6dof.py).
"""
import math

# ---------------------------------------------------------------------------
# Mass / inertia
# ---------------------------------------------------------------------------
mass = 2.5          # kg
g = 9.81            # m/s^2
Jx = 0.150          # kg*m^2
Jy = 0.300          # kg*m^2
Jz = 0.400          # kg*m^2
Jxz = 0.010         # kg*m^2  (product of inertia)

# Payload model (replace with measured mass/position for hardware claims).
# payload_level is the fraction of this physical payload mass, not a mass
# itself.  The longitudinal arm is selected from the stated nominal maximum
# CG shift; the small intrinsic inertia terms represent a compact payload box.
payload_mass_max = 1.5       # kg (60% of empty mass at payload_level=1)
payload_x_m = -0.104         # m, aft of nominal empty-airframe CG (x forward)
payload_y_m = 0.0
payload_z_m = 0.0
payload_Jx_per_kg = 0.010    # kg*m^2/kg
payload_Jy_per_kg = 0.020
payload_Jz_per_kg = 0.030

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
S = 0.55            # m^2   wing reference area
b = 2.10            # m     wingspan
c = 0.26            # m     mean aerodynamic chord
AR = b ** 2 / S     # aspect ratio
rho = 1.225         # kg/m^3  air density (sea-level standard, used throughout)

# ---------------------------------------------------------------------------
# Longitudinal aerodynamic derivatives
# ---------------------------------------------------------------------------
CL0 = 0.28
CL_alpha = 4.9          # per rad
CL_q = 6.9
CL_de = 0.30            # per rad

CD0 = 0.03
CD_alpha = 0.30
k_induced = 1.0 / (math.pi * 0.9 * AR)
CD_de_k = 0.02          # per rad^2
CD_tail_induced = 0.08  # incremental tail induced drag coefficient factor

Cm0 = 0.02
Cm_alpha = -0.60        # per rad
Cm_q = -8.5
Cm_de = -1.10           # per rad

# ---------------------------------------------------------------------------
# Tail geometry / morphing-tail derivative
# ---------------------------------------------------------------------------
S_tail = 0.11           # m^2
l_tail = 0.55           # m   (distance from CG to tail aero center)
CL_alpha_tail = 3.5     # per rad
tail_efficiency = 0.9

V_H = (S_tail * l_tail) / (S * c)                       # tail volume ratio
Cm_dtail = -V_H * CL_alpha_tail * tail_efficiency        # per rad (incidence)

# ---------------------------------------------------------------------------
# Lateral-directional aerodynamic derivatives
# ---------------------------------------------------------------------------
CY_beta = -0.35
CY_dr = 0.14
# other CY terms (CY_p, CY_r, CY_da) are zero per spec

Cl_beta = -0.06
Cl_p = -0.50
Cl_r = 0.09
Cl_da = 0.17
Cl_dr = 0.006

Cn_beta = 0.070
Cn_p = -0.03
Cn_r = -0.045
Cn_da = -0.010
Cn_dr = -0.061

# ---------------------------------------------------------------------------
# Actuator limits (stored in radians internally)
# ---------------------------------------------------------------------------
ELEVATOR_LIMIT_DEG = 25.0
AILERON_LIMIT_DEG = 20.0
RUDDER_LIMIT_DEG = 20.0

ELEVATOR_LIMIT = math.radians(ELEVATOR_LIMIT_DEG)
AILERON_LIMIT = math.radians(AILERON_LIMIT_DEG)
RUDDER_LIMIT = math.radians(RUDDER_LIMIT_DEG)

# ---------------------------------------------------------------------------
# Trim condition used as the reference operating point
# ---------------------------------------------------------------------------
Va_trim = 15.0          # m/s
alt_trim = 100.0        # m
Va_min_valid = 10.0     # m/s; below this linear aero model is not validated
Va_train_min = 12.0     # m/s; estimator training envelope lower bound
Va_train_max = 20.0     # m/s; estimator training envelope upper bound
Main_results_payload_max = 0.74  # positive-static-margin reporting limit
