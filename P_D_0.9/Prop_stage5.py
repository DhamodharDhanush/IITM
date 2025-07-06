import numpy as np
import pandas as pd
import matplotlib
# Use non-GUI backend for rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from numba import njit, prange
import math
from CODES.Test_codes.Propeller_5_stages import Kt_chart_array_generator_rotation, RT_func, cavitation_Keller, Kt_base, Kq_base, delta_Kq, delta_Kt

pitch_array = np.linspace(0.2, 2.0, 1801)
J_array = np.linspace(0.1, 1.6, 1600)
Open_water_efficiency = 0.71
Hull_efficiency = 1.1503
Rotative_efficiency = 1.03


def generate_speed_power_curve(
    D_P,
    AE_AO,
    P_i,
    z,
    w,
    t,
    rho,
    eta_R,
    eta_H,
    eta_T,
    V_array):
    
    """
    Stage 5: Generate speed-power and engine speed curves for a range of ship speeds.
    
    Inputs:
    - D_P, AE_AO, P_i, z: optimized propeller geometry
    - RT_func: function RT_func(V) -> resistance [N]
    - w, t, rho, eta_R, eta_H, eta_T: hull/environment efficiencies
    - V_array: array of speeds [m/s] to generate curve over
    
    Returns:
    - pandas DataFrame with columns:
        ['V', 'EHP', 'n_P', 'P_E', 'DHP', 'BHP', 'eta_O']
    """
    results = []
    velocities = []
    eta = []
    pitch_ratio = P_i
    # Predefine J_array for open-water sampling
    J_array = np.linspace(0.1, 1.6, 1600)

    for i in range(len(V_array)):
        V = V_array[i]
        V_A = V * (1 - w)
        R_T = RT_func(V)
        T_required = R_T / (1 - t)
        
        # Thrust coefficient quadratic c2
        c2 = T_required / (rho * D_P**2 * V_A**2)
        # open-water data for geometry
        Kt_array, J_array_1 = Kt_chart_array_generator_rotation(J_array, V_A, z, AE_AO, pitch_ratio, 1, D_P)

        T_prop = Kt_array - c2 * J_array_1**2

        b = 'False'

        # find root in J_array as before
        sign_changes = np.where(np.sign(T_prop[:-1]) * np.sign(T_prop[1:]) < 0)[0]
        for idx in sign_changes:
            J_lo, J_hi = J_array[idx], J_array[idx+1]
            f_lo, f_hi = T_prop[idx], T_prop[idx+1]
            # simple bisection
            for _ in range(100):
                J_mid = 0.5 * (J_lo + J_hi)
                # interpolate Kt at J_mid
                Kt0 = Kt_base(J_mid, pitch_ratio, AE_AO, z)
                dKt = 0
                
                n_P = V_A / (J_mid * D_P)               
                chord = 2.057 * (D_P / z) * AE_AO
                vel_075 = n_P * D_P * ((J_mid**2 + (0.75 * 22 / 7)**2)**0.5)
                Re = vel_075 * chord / (1.0038e-6)
                
                if Re > 2_000_000:
                    dKt = delta_Kt(J_mid, pitch_ratio, AE_AO, z, Re)

                Kt_root = Kt0 + dKt
                f_mid = Kt_root - c2 * J_mid**2
                if abs(f_mid) < 1e-10:
                    b = 'true'
                    # print("Tolerance is reached, J_root is acheived in stage 4")
                    break

                if f_lo * f_mid < 0:
                    J_hi, f_hi = J_mid, f_mid
                else:
                    J_lo, f_lo = J_mid, f_mid
                continue
        
        J_root = J_mid
        Kq0 = Kq_base(J_mid, pitch_ratio, AE_AO, z)
        if Re > 2_000_000:
            dKq = delta_Kq(J_mid, pitch_ratio, AE_AO, z, Re)

        Kq_root = Kq0 + dKq

        # compute n_P, P_E
        n_P = V_A / (J_root * D_P)
        P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root

        T_P = rho * n_P**2 * D_P**4 * Kt_root

        if (abs(T_P - T_required) / T_required) <= 0.05:
            AE_AO_min_keller = cavitation_Keller(16.2, 1, T_P, D_P, z)
            
            
            # AE_AO_min_Burill = cavitation_Burill(n_P, 
            # (R_T * V / (Open_water_efficiency *  Hull_efficiency * Rotative_efficiency)),
            # Rotative_efficiency, 
            # V_A, 
            # 5.30, 
            # Open_water_efficiency, 
            # pitch_ratio, 
            # J_root)

            if AE_AO > AE_AO_min_keller:
                eta_O = (J_root / (2 * 22 / 7)) * (Kt_root / Kq_root)

                print(f"at velocity = {round(V, 2)}, T_required= {T_required}, T_prop = {T_P}, AE_AO_Keller = {AE_AO_min_keller}, AE_AO_Prop = {AE_AO}, efficiency = {eta_O} J_tolerance = {b} in iter= {_}")
                
                # EHP, DHP, BHP
                EHP = R_T * V
                # overall propulsive efficiency eta_D = eta_O * eta_H * eta_R
                eta_D = eta_O * eta_H * eta_R
                if eta_D <= 0:
                    DHP = 0
                    BHP = 0
                else:
                    DHP = EHP / eta_D
                    BHP = DHP / eta_T
                results.append({
                    'V': V,
                    'EHP': EHP,
                    'n_P': n_P,
                    'P_E': P_E,
                    'DHP': DHP,
                    'BHP': BHP,
                    'eta_O': eta_O
                })

                velocities.append(V)
                eta.append(eta_O)
        
        continue

    df = pd.DataFrame(results)
    return df, velocities, eta

Open_water_efficiency = 0.71
Hull_efficiency = 1.1503
Rotative_efficiency = 1.03
transmission_efficiency = 0.990

z = 4
w = 0.3054
t = 0.2010

next_round_velocities = np.linspace(2, 22, 201)

stage_3_parameters = {'AE_AO_opt': np.float64(0.577), 
'D_P_opt': np.float64(10.34), 
'pitch_ratio_opt': np.float64(7.46 / 10.34), 
'P_i_opt': np.float64(7.46), 
'V_max': np.float64(16 * 0.5144), 
'eta_O_at_V_max': 0.013365090871370151, 
'J_ratio': 0.4750000000000001, 
'soln_attained': True}

stage_4_parameters = {'n_P': np.float64(1.0933737311079572), 
'P_E': np.float64(29382672.34114404), 
'BHP': np.float64(28586898.83818547), 
'consistent': np.True_, 
'rel_diff_n': np.float64(0.013467478220278835)}

# D_P,
#     AE_AO,
#     P_i,
#     z,
#     w,
#     t,
#     rho,
#     eta_R,
#     eta_H,
#     eta_T,
#     V_array)

df, velocities, eta = generate_speed_power_curve(stage_3_parameters['D_P_opt'],
                                stage_3_parameters['AE_AO_opt'],
                                stage_3_parameters['pitch_ratio_opt'],
                                z, w, t, 1025, Rotative_efficiency, Hull_efficiency, transmission_efficiency, next_round_velocities)

# print(velocities)
# print()
# print(eta)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(velocities, eta, label=f"K_t (P/D={stage_3_parameters['pitch_ratio_opt']}, AE/A0={stage_3_parameters['AE_AO_opt']}, z={z})")

ax.set_xlabel("Velocity")
ax.set_ylabel("Coefficient / Efficiency")
# Major grid
ax.grid(which='major', linestyle='-', linewidth=0.05, axis="x")
ax.grid(which='major', linestyle='-', linewidth=0.01, axis="y")
# Minor ticks and grid
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))
ax.grid(which='minor', linestyle=':', linewidth=0.5)
ax.legend()
fig.tight_layout()
# Save and close

sub_folder = os.path.join(
        "CODES",
        f"Stage5"
    )

os.makedirs(sub_folder, exist_ok=True)

image_path = os.path.join(sub_folder, f"test_side_AE_AO{stage_3_parameters['AE_AO_opt']}.png")

fig.savefig(image_path)
plt.close(fig)