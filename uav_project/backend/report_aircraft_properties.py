"""Print the assumed mass/CG/inertia and longitudinal static-margin table."""
import numpy as np
import aircraft_params as P
from dynamics_6dof import payload_effects, payload_cg_position, payload_inertias, static_margin

print("payload  mass_kg  cg_x_m  cg_shift_chord  static_margin  Jx Jy Jz Jxz")
for pl in np.linspace(0.0, 1.0, 11):
    _, _, _, mass = payload_effects(pl)
    jx, jy, jz, jxz = payload_inertias(pl)
    print(f"{pl:5.2f} {mass:8.3f} {payload_cg_position(pl):7.4f} "
          f"{(payload_cg_position(pl) / P.c):15.4f} {static_margin(pl):13.4f} "
          f"{jx:.4f} {jy:.4f} {jz:.4f} {jxz:.4f}")
