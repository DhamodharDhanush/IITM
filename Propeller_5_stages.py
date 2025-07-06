import numpy as np
import pandas as pd
import math
import sys
import gc
from functools import lru_cache
from numba import njit, prange
import matplotlib
# Use non-GUI backend for rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
import time
from datetime import timedelta
import shutil

# =============================================================================
# Propeller Design Automation: 5-Stage Process 
# 
# This code provides a template framework to implement the five-stage propeller
# design procedure described in the document. It requires the user to supply:
#  - Resistance curve function: RT_func(V) -> resistance at speed V [N]
#  - Open-water data function: open_water_data(AE_AO, pitch_ratio, z, J_array)
#       -> returns Kt_array, Kq_array, eta_O_array corresponding to J_array
#  - Cavitation criterion function: cavitation_min_AE_AO(D_P, T) -> minimum AE/AO
#  - Parent ship parameters or direct initial diameter if available
#  - Hull and environment parameters: wake fraction w, thrust deduction t, water density rho,
#    efficiencies eta_R, eta_T, ambient pressure p0, vapor pressure pv, etc.
#
# Each stage is implemented as a function. The main script shows example function calls
# (commented) indicating how to wire up user-supplied functions and data.
# =============================================================================

# =============================================================================

Cq = np.array([+0.00379368,+0.00886523,-0.032241    ,+0.00344778,-0.0408811   ,-0.108009    ,-0.0885381,+0.188561    ,-0.00370871,+0.00513696,+0.0209449   ,+0.00474319  ,-0.00723408 ,
+0.00438388,-0.0269403 ,+0.0558082   ,+0.0161886 ,+0.00318086  ,+0.015896    ,+0.0471729,+0.0196283   ,-0.0502782 ,-0.030055  ,+0.0417122   ,-0.0397722   ,-0.00350024 ,
-0.0106854 ,+0.00110903,-0.000313912 ,+0.0035985 ,-0.00142121  ,-0.00383637  ,+0.0126803,-0.00318278  ,+0.00334268,-0.00183491,+0.000112451 ,-0.0000297228,+0.000269551,
+0.00083265,+0.00155334,+0.000302683 ,-0.0001843 ,-0.000425399 ,+0.0000869243,-0.0004659,+0.0000554194], dtype=np.float64)
sq = np.array([0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0, 3, 0, 1, 0, 1, 3, 0, 3, 2, 0, 0, 3, 3, 0, 3, 0, 1, 0, 2, 0, 1, 3, 3, 1, 2, 0, 0, 0, 0, 3, 0, 1], dtype=np.float64)
tq = np.array([0, 0, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 2, 0, 3, 3, 0, 0, 0, 1, 1, 2, 3, 6, 0, 3, 6, 0, 6, 0, 2, 3, 6, 1, 2, 6, 0, 0, 2, 6, 0, 3, 3, 6, 6], dtype=np.float64)
uq = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.float64)
vq = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float64)    

# =============================================================================

Ct = np.array([+0.00880496,-0.204554    ,+0.166351    ,+0.158114    ,-0.147581    ,-0.481497    ,+0.415437    ,+0.0144043   ,-0.0530054   ,+0.0143481   ,+0.0606826   ,-0.0125894   ,
 +0.0109689 ,-0.133698    ,+0.00638407  ,-0.00132718  ,+0.168496    ,-0.0507214   ,+0.0854559   ,-0.0504475   ,+0.010465    ,-0.00648272  ,-0.00841728  ,+0.0168424   ,
 -0.00102296,-0.0317791   ,+0.018604    ,-0.00410798  ,-0.000606848 ,-0.0049819   ,+0.0025983   ,-0.000560528 ,-0.00163652  ,-0.000328787 ,+0.000116502 ,+0.000690904 ,
 +0.00421749,+0.0000565229,-0.00146564  ], dtype=np.float64)
st = np.array([0, 1, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 2, 3, 0, 2, 3, 1, 2, 0, 1, 3, 0, 1, 0, 0, 1, 2, 3, 1, 1, 2, 0, 0, 3, 0], dtype=np.float64)
tt = np.array([0, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 0, 0, 3, 6, 6, 0, 0, 0, 0, 6, 6, 3, 3, 3, 3, 0, 2, 0, 0, 0, 0, 2, 6, 6, 0, 3, 6, 3], dtype=np.float64)
ut = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2], dtype=np.float64)
vt = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float64)         

# =============================================================================


J_array = np.linspace(0.1, 1.6, 160)

pitch_array = np.linspace(0.5, 1.4, 91)

@njit(fastmath=True)
def make_new_map(Kt_map, Kq_map, eta_map, Kt_map_previous, Kq_map_previous, eta_map_previous):
    a, b, c = Kt_map.size
    d, e, f = Kt_map_previous.size
    
    super_kt = np.linspace((a + b+ e, c), dtype= np.float64)
    super_kq = np.linspace((a + b+ e, c), dtype= np.float64)
    super_eta = np.linspace((a + b+ e, c), dtype= np.float64)    

    a, b, c = Kt_map.size
    d, e, f = Kt_map_previous.size
    
    for i in prange(a):
        ai = 0
        bi = 0
        for j in range(b + e):
            if j % 2 == 0:
                # even slot: take from A
                super_kt[i, j, :]  = Kt_map_previous[i, ai, :]
                super_kq[i, j, :]  = Kq_map_previous[i, ai, :]
                super_eta[i, j, :] = eta_map_previous[i, ai, :]

                ai += 1
            else:
                # odd slot: take from B
                super_kt[i, j, :]  = Kt_map[i, ai, :]
                super_kq[i, j, :]  = Kq_map[i, ai, :]
                super_eta[i, j, :] = eta_map[i, ai, :]
                bi += 1

    return super_kt, super_kq, super_eta


@njit(fastmath=True)
def find_best_J_eta(P_E, n_P, rho, V, AE_AO, pitch_ratio,
                    J_array, Kq_array):
    """
    Given precomputed Kt, Kq, eta_O over J_array, find the J_root
    via your Q_prop bisection and then select the max-eta point.
    Returns (best_J, best_eta).
    """
    c4 = (P_E * n_P**2) / ((2 * np.pi) * rho * (V**5))
    tol_root = 1e-5

    best_eta = -1.0
    best_J   = 0.0

    # Precompute Q_prop
    Q_prop = np.empty(J_array.size)

    Q_prop = Kq_array - c4 * (J_array**5)

    # Find all sign‐change intervals
    # then do bisection in each, and afterwards pick the J_mid with highest eta
    for i in range(J_array.size - 1):
        if Q_prop[i] * Q_prop[i+1] < 0:
            J_lo, J_hi = J_array[i], J_array[i+1]
            f_lo, f_hi = Q_prop[i], Q_prop[i+1]

            # simple bisection
            for _ in range(100):
                J_mid = 0.5 * (J_lo + J_hi)
                # linear‐interpolate Kq_mid
                # find interval
                # (we assume J_array is sorted and evenly spaced)
                dKq = 0
                
                Kq = Kq_base(J_mid, pitch_ratio, AE_AO, z)

                D_P = V / (J_mid * n_P)               
                chord = 2.057 * (D_P / z) * AE_AO
                vel_075 = n_P * D_P * ((J_mid**2 + (0.75 * 22 / 7)**2)**0.5)
                Re = vel_075 * chord / (1.0038e-6)

                if Re > 2000000:
                    dKq = delta_Kq(J_mid, pitch_ratio, AE_AO, z, Re)

                Kq_mid = Kq + dKq

                f_mid = Kq_mid - c4 * (J_mid**5)
                
                if abs(f_mid) < tol_root:
                    break
                
                if f_lo * f_mid < 0:
                    J_hi, f_hi = J_mid, f_mid
                else:
                    J_lo, f_lo = J_mid, f_mid

            # now J_mid is a root; get its eta by linear interp on eta_O_array
            Kt = Kt_base(J_mid, pitch_ratio, AE_AO, z)
            dKt = 0
            if Re > 2000000:
                dKt = delta_Kt(J_mid, pitch_ratio, AE_AO, z, Re)
            Kt_mid = Kt + dKt

            eta_mid = (J_mid / (2 * 22 / 7)) * (Kt_mid / Kq_mid)

            if eta_mid > best_eta:
                best_eta = eta_mid
                best_J   = J_mid
                best_Kt = Kt_mid
                best_Kq = Kq_mid

    # print(best_J,  (V / (D_P * n_P)))
    return best_J, best_eta, best_Kt, best_Kq, D_P

# 2) Then wrap your pitch‐loop in another njit’d function
@njit(fastmath=True)
def optimize_for_one_AE_V2(P_E, n_P, rho, V_A, AE_AO,
                          J_array,
                          # and whatever else you need to compute K arrays:
                          z, area_index, velocity_index, Kt_map, Kq_map, eta_map):
    
    best_overall_eta = -1.0
    best_overall_J   = 0.0
    best_overall_pitch = 0.0
    best_Kt = 0.0
    best_Kq = 0.0

    # pre-allocate your open-water arrays
    Kt_arr = np.empty(J_array.size)
    Kq_arr = np.empty(J_array.size)
    eta_arr = np.empty(J_array.size)

    for pi in prange(pitch_array.size):
        pitch = pitch_array[pi]
        # call your pure‐Python implementation of open_water_data_diameter,
        # or better yet, precompute those K arrays outside this njit function
        # and pass them in here as arguments.

        # For illustration, assume a function fill_open_water(...)
        Kq_arr, J_array_best = prop_map_diameter(V_A, z, AE_AO, pitch, 1, n_P)

        J_root, eta_root, Kt, Kq, D_P = find_best_J_eta(P_E, n_P, rho, V_A, AE_AO, pitch,                                                                                                      
                                                    J_array_best, Kq_arr)

        Kt_map[area_index, velocity_index, pi]  = Kt
        Kq_map[area_index, velocity_index, pi]  = Kq
        eta_map[area_index, velocity_index, pi] = eta_root

        if eta_root > best_overall_eta:
            best_overall_eta   = eta_root
            best_overall_J     = J_root
            best_overall_pitch = pitch
            best_Kt            = Kt
            best_Kq            = Kq
            best_V_A           = V_A
            best_D_P           = D_P

    return best_overall_J, best_overall_eta, best_overall_pitch, best_Kt, best_Kq, Kt_map, Kq_map, eta_map

@njit(fastmath=True)
def prop_map_diameter(V, z, AE, pitch, d, n_P):
    #J_array, V_A, z, AE_AO, pitch_ratio, 1, n_P
    J_array_prop = np.linspace(0.1, 1.6, 160)
    return open_water_data_diameter(J_array_prop, V, z, AE, pitch, d, n_P)

@njit(fastmath=True)
def RT_func(V_S):
    Frictional_resistance = 1903000 * (V_S**2) / ((16 * 0.5144)**2)
    Appendage_resistance = 21960 * (V_S**2) / ((16 * 0.5144)**2)
    Wave_making_resistance = 26300
    Model_ship_correlation_resistance = 226800 * (V_S**2) / ((16 * 0.5144)**2)
    R_T = Frictional_resistance + Appendage_resistance + Wave_making_resistance + Model_ship_correlation_resistance
    return R_T

@njit(fastmath=True)
def cavitation_Keller(shaft_depth,
                              number_screws,
                              Thrust,
                              Diameter_prop,
                              num_blades
                              ):
    
    if number_screws == 1:
        Keller_criteria = 0.2 + ((1.3 + 0.3 * num_blades) * Thrust) / ((Diameter_prop**2) * (994.7 + 1025 * 9.81 * shaft_depth))
    if number_screws == 2:
        Keller_criteria = 0.1 + ((1.3 + 0.3 * num_blades) * Thrust) / ((Diameter_prop**2) * (994.7 + 1025 * 9.81 * shaft_depth))

    return Keller_criteria

@njit(fastmath=True)
def cavitation_Burill(n_P, 
                      DHP, rotative_efficiency, 
                      V_A, shaft_centre_height, 
                      open_water_eff, 
                      Pitch_ratio, advance_coefficient):
    # (n_P, 
    # (R_T * V / (Open_water_efficiency *  Hull_efficiency * Rotative_efficiency)),
    # Rotative_efficiency, 
    # V_A, 
    # 5.30, 
    # Open_water_efficiency, 
    # pitch_ratio, 
    # J_val)



    DHP_british = DHP / (745.7 * 0.9863) # converst BHP into british horse power units
            

    B_P = (n_P * 60) * ((DHP_british * rotative_efficiency)**0.5) / ((V_A / 0.5144)**2.5)                      
    F = (rotative_efficiency * (B_P**2) * ((V_A / 0.5144)**1.25)) / (278.4 * (10.18 + shaft_centre_height)**0.625)
    Burill_criteria = ( F * (open_water_eff / ((1 / advance_coefficient)**2))) / (((1 + 4.826 * ((1 / advance_coefficient)**2))**0.375)*(1.067 - 0.229 * Pitch_ratio))

    return Burill_criteria

@njit(fastmath=True)
def delta_Kt(J, P_D, AE_A0, z, Re_n):
    """
    Compute ΔK_T from Table 2.
    Inputs:
      J: scalar or array
      P_D: scalar
      AE_A0: scalar
      z: scalar (int)
      logRe_n: scalar = log10(Re_n)
    Returns:
      ΔK_T value(s), same shape as J
    """

    logRe_n = math.log10(Re_n)
    # Example structure based on your screenshot (fill actual numeric terms):
    x = logRe_n - 0.301  # as per your table
    # Start with zero (or the constant term):
    val = 0.0
    # Example terms (replace with actual from your Table 2):
    val += 0.000353485
    val += -0.00333758 * (AE_A0) * (J**2)
    val += -0.00478125 * (AE_A0) * (P_D) * (J)
    val += +0.000257792 * (x**2) * (AE_A0) * (J**2)
    val += +0.0000643192 * (x) * (P_D**6) * (J**2)
    val += -0.0000110636 * (x**2) * (P_D**6) * (J**2)
    val += -0.0000276305 * (x**2) * (z) * (AE_A0) * (J**2)
    val += +0.0000954 * (x) * (z) * (AE_A0) * (P_D) * (J)
    val += +0.0000032049 * (x)  * (z**2) * (AE_A0) * (P_D**3) * (J)
    # ... and so on for each polynomial term in Table 2 for ΔK_T ...
    # Make sure operations broadcast if J is array.
    return val

@njit(fastmath=True)
def delta_Kq(J, P_D, AE_A0, z, Re_n):
    """
    Compute ΔK_Q from Table 2.
    """
    logRe_n = math.log10(Re_n)
    x = logRe_n - 0.301
    val = 0.0
    # Example terms (replace with actual):
    val += -0.000591412
    val += +0.00696898 * (P_D)
    val += -0.0000666654 * (z) * (P_D**6)
    val += +0.0160818 * (AE_A0**2)
    val += -0.000938091 * (x) * (P_D)
    val += -0.00059593 * (x) * (P_D**2)
    val += +0.0000782099 * (x**2) * (P_D**2)
    val += +0.0000052199 * (x) * (z) * (AE_A0) * (J**2)
    val += -0.00000088528 * (x**2) * (z) * (AE_A0) * (P_D) * (J)
    val += +0.0000230171 * (x) * (z) * (P_D**6)
    val += -0.00000184341 * (x**2) * (z) * (P_D**6)
    val += -0.00400252 * (x) * (AE_A0**2)
    val += +0.000220915 * (x**2) * (AE_A0)

    # ... etc ...
    return val


# === 3. Define functions for base Kt, Kq ===
@njit(fastmath=True)
def Kt_base(J, P_D, AE_A0, z):   
    """
    Compute base (nominal) K_T from Table 1 polynomial.
    J: scalar or array
    P_D, AE_A0, z: scalars
    """
    val = 0.0

    for d in range(len(st)):
        # contribution: C * J^s * (P/D)^t * (AE/A0)^u * z^v
        val += Ct[d] * (J**st[d]) * (P_D**tt[d]) * (AE_A0**ut[d]) * (z**vt[d])
        
    return val

@njit(fastmath=True)
def Kq_base(J, P_D, AE_A0, z): 
            
    """
    Compute base (nominal) K_Q from Table 1 polynomial.
    """
    val = 0.0

    for d in range(len(sq)):
        # contribution: C * J^s * (P/D)^t * (AE/A0)^u * z^v
        val += Cq[d] * (J**sq[d]) * (P_D**tq[d]) * (AE_A0**uq[d]) * (z**vq[d])
    return val


@njit(fastmath=True)
def Kt_chart_array_generator_rotation(J, velocity, blades, area_ratio, pitch_ratio, b, diameter=0.24):
    """
    Compute Kt, Kq, efficiency for one configuration,
    """
    # Prepare folder
    # sub_folder = os.path.join(
    #     "Kt_Kq_Generation",
    #     f"Velocity_{velocity}_m_s",
    #     f"{blades}_blades",
    #     f"AE_AO_{area_ratio}",
    #     f"P_D_{pitch_ratio}"
    # )
    # os.makedirs(sub_folder, exist_ok=True)

    maxlen = J.size
    J_values = np.empty(maxlen, dtype= np.float64)
    Kt_vals  = np.empty(maxlen, dtype= np.float64)

    count = 0
    # Loop over J values
    for idx in range(maxlen):
        J_val = J[idx]
        if J_val == 0:
            RPM = 0
        else:
            RPM = velocity / (J_val * diameter)
        # chord length based on blade count
        if blades == 3:
            chord = 2.1475 * (diameter / blades) * area_ratio
        else:
            chord = 2.057 * (diameter / blades) * area_ratio

        vel_075 = RPM * diameter * ((J_val**2 + (0.75 * 22 / 7)**2)**0.5)
        Re = vel_075 * chord / (1.0038e-6)

        # Base coefficients
        Kt0 = Kt_base(J_val, pitch_ratio, area_ratio, blades)

        dKt = dKq = 0
        if Re > 2_000_000:
            dKt = delta_Kt(J_val, pitch_ratio, area_ratio, blades, Re)

        Kt = Kt0 + dKt

        if Kt < 0:
            Kt_vals[count]  = Kt
            J_values[count] = J_val
            count += 1
            # print(f"{b} Re = {Re} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
            
            # print(f"{b} advance coefficient = {J_val} Pitch ratio = {pitch_ratio} Expanded area Ratio = {area_ratio} blades = {blades} Velocity = {velocity}")
            # print(psutil.virtual_memory())

            break

        # print(f"{b} Re = {Reynolds} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
        # print(f"{b} advance coefficient = {J_val} Pitch ratio = {pitch_ratio} Expanded area Ratio = {area_ratio} blades = {blades} Velocity = {velocity}")

        Kt_vals[count]  = Kt
        J_values[count] = J_val
        count += 1
    
    return (Kt_vals[:count], J_values[:count])

@njit(fastmath=True)
def open_water_data_diameter(J, velocity, blades, area_ratio, pitch_ratio, b, rotation):
    """
    Compute Kt, Kq, efficiency for one configuration
    """
    # Prepare folder
    # sub_folder = os.path.join(
    #     "Kt_Kq_Generation",
    #     f"Velocity_{velocity}_m_s",
    #     f"{blades}_blades",
    #     f"AE_AO_{area_ratio}",
    #     f"P_D_{pitch_ratio}"
    # )
    # os.makedirs(sub_folder, exist_ok=True)
    RPM = rotation

    maxlen = J.size
    J_values = np.empty(maxlen, dtype= np.float64)
    Kq_vals  = np.empty(maxlen, dtype= np.float64)


    # Loop over J values
    count = 0

    for idx in range(maxlen):
        J_val = J[idx]
        
        if J_val == 0:
            continue
        # chord length based on blade count

        diameter = velocity / (J_val * RPM)

        if blades == 3:
            chord = 2.1475 * (diameter / blades) * area_ratio
        else:
            chord = 2.057 * (diameter / blades) * area_ratio

        vel_075 = RPM * diameter * ((J_val**2 + (0.75 * 22 / 7)**2)**0.5)
        Re = vel_075 * chord / (1.0038e-6)

        # Base coefficients

        Kq0 = Kq_base(J_val, pitch_ratio, area_ratio, blades)

        dKq = 0
        if Re > 2_000_000:

            dKq = delta_Kq(J_val, pitch_ratio, area_ratio, blades, Re)

        Kq = Kq0 + dKq



        if Kq < 0:
            Kq_vals[count]  = Kq
            J_values[count] = J_val
            count += 1
            # print(f"{b} Re = {Re} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
            
            # print(f"Stage 2 {b} advance coefficient = {J_val} Pitch ratio = {pitch_ratio} Expanded area Ratio = {area_ratio} blades = {blades} Velocity = {velocity}")
            # print(psutil.virtual_memory())

            break

        # print(f"{b} Re = {Reynolds} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
        # print(f"Stage 2 {b} advance coefficient = {J_val} Pitch ratio = {pitch_ratio} Expanded area Ratio = {area_ratio} blades = {blades} Velocity = {velocity}")

        Kq_vals[count]  = Kq
        J_values[count] = J_val
        count += 1
    
    return (Kq_vals[:count],
            J_values[:count])



# --------------------------- STAGE 1 ------------------------------------------

def estimate_initial_diameter(parent_D=None, parent_MCR=None, parent_nMCR=None, c1=1.0):
    """
    Estimate initial propeller diameter D_P.
    - If parent_D is provided (propeller diameter of parent ship), return it.
    - Else, if parent_MCR (power in W) and parent_nMCR (rps) are provided, use empirical formula:
        D_P = 15.4 * c1 * (parent_MCR / parent_nMCR**3)**(-0.2)
      Note: Ensure units are consistent (e.g., MCR in kW or W? The formula constant 15.4
      must match units convention; adjust as needed).
    """
    if parent_D is not None:
        return parent_D
    if parent_MCR is not None and parent_nMCR is not None:
        # Empirical formula; units must be consistent with the constant 15.4
        return 15.4 * c1 * (parent_MCR / parent_nMCR**3)**(-0.2)
    raise ValueError("Provide either parent_D or (parent_MCR and parent_nMCR).")


# --------------------------- STAGE 2 ------------------------------------------
def optimize_pitch(
    D_P,
    AE_AO,
    z,
    V_service,
    w,
    t,
    rho,
    eta_R,
    eta_T,
    J_array_t,
    tol_root=1e-3
):
    """
    Stage 2: For fixed diameter D_P, expanded area ratio AE/AO, blade count z,
    and service speed V_service, find optimal pitch ratio, engine speed n_P, delivered power P_E,
    and BHP at service condition by maximizing open-water efficiency eta_O.
    
    Inputs:
    - D_P: propeller diameter [m]
    - AE_AO: expanded area ratio (0-1)
    - z: number of blades
    - V_service: ship service speed [m/s]
    - RT_func: function RT_func(V) -> resistance [N]
    - w: wake fraction (e.g., 0.2)
    - t: thrust deduction factor (e.g., 0.2)
    - rho: water density [kg/m^3]
    - eta_R: hull-propeller relative rotative efficiency (e.g., 0.98)
    - eta_T: transmission efficiency (e.g., 0.99)
    - J_array: array of J values to sample (optional; if None, defaults np.linspace(0.1,1.2,200))
    - pitch_ratio_array: array of pitch ratios P/D to try (optional; if None, defaults np.linspace(0.5,2.0,30))
    - tol_root: tolerance for root-finding in J
    
    Returns:
    - dict with keys:
        'pitch_ratio_opt', 'P_i', 'n_P', 'P_E', 'BHP', 'eta_O_opt'
    """
    
    # Effective inflow speed
    V_A = V_service * (1 - w)
    # Required thrust
    R_T = RT_func(V_service)

    T_required = R_T / (1 - t)

    # Coefficient for thrust quadratic: Kt = c2 * J^2
    c2 = T_required / (rho * (D_P**2) * (V_A**2))
    
    best = {'eta_O': -np.inf}
    
    for pitch_ratio in pitch_array:
        # User-supplied function: given AE/AO, pitch_ratio, z, and J_array,
        # return open-water Kt_array, Kq_array, eta_O_array.
        Kt_array, J_array = Kt_chart_array_generator_rotation(J_array_t, V_A, z, AE_AO, pitch_ratio, 1, D_P)
        
        # Define function f(J) = Kt(J) - c2 * J^2. We find roots in J_array.
        T_prop = Kt_array - c2 * J_array**2
        
        # Search for sign changes to locate intervals for root-finding (bisection-like)
        sign_changes = np.where(np.sign(T_prop[:-1]) * np.sign(T_prop[1:]) < 0)[0]
        for idx in sign_changes:
            J_lo, J_hi = J_array[idx], J_array[idx+1]
            f_lo, f_hi = T_prop[idx], T_prop[idx+1]
            # simple bisection
            for _ in range(50):
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

                Kt = Kt0 + dKt
                Kt_mid = Kt
                f_mid = Kt_mid - c2 * J_mid**2
                if abs(f_mid) < tol_root:
                    # print("Tolerance is reached, J_root is acheived in stage 2")
                    break
                if f_lo * f_mid < 0:
                    J_hi, f_hi = J_mid, f_mid
                else:
                    J_lo, f_lo = J_mid, f_mid
                continue
            
            J_root = J_mid
            Kt_root = Kt

            # Interpolate Kq and eta at J_root
            Kq0 = Kq_base(J_mid, pitch_ratio, AE_AO, z)
            dKq = delta_Kq(J_mid, pitch_ratio, AE_AO, z, Re)
            Kq_root = Kq0 + dKq

            eta_root = (J_root / (2 * 22 / 7)) * (Kt_root / Kq_root)
            # Compute engine speed
            n_P = V_A / (J_root * D_P)  # [1/s]

            # Delivered power P_E from torque equation: P_E = 2*pi * rho * n_P^3 * D_P^5 * Kq
            P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root

            # Brake horsepower: account for hull-propeller and transmission efficiencies
            BHP = (P_E / eta_R) / eta_T

            
            # Check if this is better (max eta_O)
            if (abs((n_P - 1.1) / 1.1)) <= 0.05:
                if eta_root > best['eta_O']:
                    best = {
                        'pitch_ratio_opt': pitch_ratio,
                        'J_opt': J_root,
                        'eta_O': eta_root,
                        'n_P': n_P,
                        'P_E': P_E,
                        'BHP': BHP
                    }
    
    if best['eta_O'] < 0:
        raise RuntimeError("No feasible root found for any pitch ratio in Stage 2. Check inputs or expand search ranges.")
    
    best['P_i'] = best['pitch_ratio_opt'] * D_P  # pitch in same length units as D_P
    return best


# --------------------------- STAGE 3 ------------------------------------------
def optimize_dimensions(V_service,
    P_E,
    n_P,
    z,
    w,
    t,
    rho,
    J_array,
    AE_AO_array,
    V_max_search_array,
    Kt_map,
    Kq_map,
    eta_map,
    tol_thr=0.05
):
    
    """
    Stage 3: Given delivered power P_E and engine speed n_P (from Stage 2),
    find optimal dimensions D_P, pitch P_i, AE/AO, and maximum speed V_max
    by scanning AE/AO and candidate speed ranges.
    
    Inputs:
    - P_E: delivered power at NCR [W]
    - n_P: engine/propeller speed at NCR [1/s]
    - z: number of blades
    - RT_func: function RT_func(V) -> resistance [N]
    - w: wake fraction
    - t: thrust deduction factor
    - rho: water density
    - cavitation_min_AE_AO_func: function cavitation_min_AE_AO_func(D_P, T) -> minimum AE/AO
    - AE_AO_array: array of AE/AO values to scan (e.g., np.linspace(0.4, 0.7, 7))
    - pitch_ratio_array: array of pitch ratios to consider (e.g., np.linspace(0.5, 2.0, 30))
    - V_max_search_array: array of candidate speeds [m/s] to scan for maximum speed (e.g., np.linspace(V_service, V_upper, 20))
    - tol_thr: relative tolerance for thrust matching, e.g. 0.05 means ±5%
    
    Returns:
    - dict with keys:
        'AE_AO_opt', 'D_P_opt', 'pitch_ratio_opt', 'P_i_opt', 'V_max_opt', 'eta_O_at_V_max'
    """

    best = {'eta_O_at_V_max': -np.inf, 'V_max': -np.inf}
    
    d = 1

    AE_AO_limits = np.array([0.0, 1.2])

    area_ratio_array = np.array([])
    area_ratio_array = AE_AO_array

    V_A_array = V_max_search_array * (1 - w)
    R_T_array = 31765.225653587 * (V_max_search_array**2) + 26300
    T_array = R_T_array / (1 - t)

    total_iterations = len(AE_AO_array) * len(V_max_search_array)
    q = 1

    start_time = time.time()

    for i in range(len(AE_AO_array)):
        
        AE_AO = area_ratio_array[i]

        # if (i % 2) == 0:
        #     AE_AO = area_ratio_array[i//2]
        #     area_ratio_array = np.delete(area_ratio_array, 0)
        # else:
        #     AE_AO = area_ratio_array[-((i//2)+1)]
        #     area_ratio_array = np.delete(area_ratio_array, -1)

#--------------------------------------------------------------------------------------------------------

        for j in range(len(V_max_search_array)):
            V = V_max_search_array[j]
            V_A = V_A_array[j]
            R_T = R_T_array[j]
            T_required = T_array[j]
            # coefficient for Kq intersection: from torque eq:
            # we look for J such that P_E = 2*pi * rho * n_P^3 * D_P^5 * Kq(J).
            # But since D_P unknown, we instead for each J and pitch_ratio compute D_P, then T_P.

            J_root, eta_root, pitch_ratio, Kt_root, Kq_root, Kt_map, Kq_map, eta_map = optimize_for_one_AE_V2(P_E, n_P, rho, V_A, AE_AO,
                          J_array, z, i, j, Kt_map, Kq_map, eta_map)
            
            
            D_P = V_A / (n_P * J_root)
            T_P = rho * n_P**2 * D_P**4 * Kt_root

            # print("T_P, T_Requie4ed:", T_P, T_required)

            if (abs(T_P - T_required) / T_required <= tol_thr):
                
                # print(f"{d} tolerance: {abs(T_P - T_required) / T_required}")
                AE_AO_min_keller = cavitation_Keller(16.2, 1, T_P, D_P, z)
                
                
                # AE_AO_min_Burill = cavitation_Burill(n_P, 
                # (R_T * V / (eta_root *  Hull_efficiency * Rotative_efficiency)),
                # Rotative_efficiency, 
                # V_A, 
                # 5.30, 
                # eta_root, 
                # pitch_ratio, 
                # J_root)

                # print(f"keller: {AE_AO_min_keller} Burill: {AE_AO_min_Burill} at AE_AO = {AE_AO}, VELOCITY= {V} eta= {V_A}")
                # print(f"keller: {AE_AO_min_keller}  at AE_AO = {AE_AO}, VELOCITY= {V} eta= {eta_root}")
                d += 1
                
                # Burill criteria is turned off as we are not able to give right inputs to the function
                # if AE_AO < AE_AO_min_keller and AE_AO < AE_AO_min_Burill:
                if AE_AO < AE_AO_min_keller and AE_AO:

                    # print("yes I'm skipping")
                    current_time = time.time()
                    elapsed = current_time - start_time
                    avg_time = elapsed / (q)
                    estimated_total = avg_time * total_iterations
                    remaining = estimated_total - elapsed

                    # Formatting
                    elapsed_str = str(timedelta(seconds=int(elapsed)))
                    remaining_str = str(timedelta(seconds=int(remaining)))
                    total_str = str(timedelta(seconds=int(estimated_total)))

                    # cavitation risk: skip
                    gc.collect()
                    msg = (f"Completed velocity {round(V, 3)} in area {round(AE_AO, 3)}, "
                            f"iteration = {q}/{total_iterations}, "
                            f"completed = {round(((q / total_iterations) * 100), 3)}%, "
                            f"| Elapsed: {elapsed_str} "
                            f"| Remaining: {remaining_str} "
                            f"| Estimated Total: {total_str} "
                            )
                    q += 1

                    sys.stdout.write("\r" + msg)   # write carriage-return + message
                    sys.stdout.flush()             # force it onto the screen
                    # sys.stdout.write("\033[F\033[F\033[F")
                    gc.collect()
                    continue

                if V > best['V_max']:
                    best = {
                        'AE_AO_opt': AE_AO,
                        'D_P_opt': D_P,
                        'pitch_ratio_opt': pitch_ratio,
                        'P_i_opt': pitch_ratio * D_P,
                        'V_max': V,
                        'eta_O_at_V_max': eta_root,
                        'J_ratio': J_root,
                        'soln_attained': True
                    }
                    if (i % 2) == 0:
                        AE_AO_limits[0] = AE_AO
                    else:
                        AE_AO_limits[1] = AE_AO

                    print(AE_AO_limits)
                    print(f"v-max: {best['V_max']}, eta_O_at_V_max: {best['eta_O_at_V_max']}, n_P: {n_P}\n")
            
            # print("yes I'm skipping")
            current_time = time.time()
            elapsed = current_time - start_time
            avg_time = elapsed / (q)
            estimated_total = avg_time * total_iterations
            remaining = estimated_total - elapsed

            # Formatting
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remaining_str = str(timedelta(seconds=int(remaining)))
            total_str = str(timedelta(seconds=int(estimated_total)))

            msg = (f"Completed velocity {round(V, 3)} in area {round(AE_AO, 3)}, "
                            f"iteration = {q}/{total_iterations}, "
                            f"completed = {round(((q / total_iterations) * 100), 3)}%, "
                            f"| Elapsed: {elapsed_str} "
                            f"| Remaining: {remaining_str} "
                            f"| Estimated Total: {total_str} ")
            q += 1
            sys.stdout.write("\r" + msg + "\n")   # write carriage-return + message
            sys.stdout.flush()             # force it onto the screen
            sys.stdout.write("\033[F")

            gc.collect()
            continue

    print("\nCompleted!")
    final_time = time.time() - start_time
    print("Total Time:", str(timedelta(seconds=int(final_time))))
#--------------------------------------------------------------------------------------------------------

    if best['V_max'] < 0:
        print(f"v-max: {best['V_max']}")
        print(f"No feasible geometry found in Stage 3. Check scanning ranges or inputs." \
        f"\n with this AE ratios, new ae ratios limits are found {AE_AO_limits}")
        NEW_AE_AO_values = {
                        'AE_AO_values': np.linspace(AE_AO_limits[0], AE_AO_limits[1], int(((AE_AO_limits[1] - AE_AO_limits[0]) / 0.0004) + 1)),
                        'soln_attained': False 
                    }

        return NEW_AE_AO_values, Kt_map, Kq_map, eta_map

    return best, Kt_map, Kq_map, eta_map


# --------------------------- STAGE 4 ------------------------------------------
def check_consistency_stage4(
    D_P,
    P_i,
    AE_AO,
    z,
    V_design,
    w,
    t,
    rho,
    eta_R,
    eta_T,
    n_P_initial,
    P_E_initial,
    J_array,
    tol_n=0.05
):
    """
    Stage 4: Given the optimized propeller geometry and design speed V_design, compute
    engine speed n_P and delivered power P_E, then check consistency with initial n_P_initial.
    
    Inputs:
    - D_P, P_i, AE_AO, z: propeller geometry
    - V_design: design speed [m/s] (e.g., V_max from Stage 3 or service speed)
    - RT_func: function RT_func(V) -> resistance [N]
    - w, t, rho, eta_R, eta_T: hull and environment parameters
    - n_P_initial: engine speed from previous stage [1/s]
    - tol_n: allowable relative difference in n_P (e.g. 0.05 => 5%)
    
    Returns:
    - dict with keys:
        'n_P', 'P_E', 'BHP', 'consistent': bool, 'rel_diff_n': relative difference
    """
    # Effective inflow
    V_A = V_design * (1 - w)
    # Required thrust
    R_T = RT_func(V_design)
    T_required = R_T / (1 - t)
    # Thrust coefficient quadratic coefficient: find J from intersection: Kt(J) = c2 * J^2
    # c2 = T_required / (rho * D_P^2 * V_A^2)
    c2 = T_required / (rho * D_P**2 * V_A**2)
    
    # We need open-water Kt vs J for given geometry:
    pitch_ratio = P_i
    J_array = np.linspace(0.1, 1.5, 300)
    Kt_array, J_array = Kt_chart_array_generator_rotation(J_array, V_A, z, AE_AO, pitch_ratio, 1, D_P)
    T_prop = Kt_array - c2 * J_array**2

    # find root in J_array as before
    sign_changes = np.where(np.sign(T_prop[:-1]) * np.sign(T_prop[1:]) < 0)[0]
    for idx in sign_changes:
        J_lo, J_hi = J_array[idx], J_array[idx+1]
        f_lo, f_hi = T_prop[idx], T_prop[idx+1]
        # simple bisection
        for _ in range(50):
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

            Kt = Kt0 + dKt
            Kt_mid = Kt
            f_mid = Kt_mid - c2 * J_mid**2
            if abs(f_mid) < 1e-3:
                # print("Tolerance is reached, J_root is acheived in stage 4")
                break
            if f_lo * f_mid < 0:
                J_hi, f_hi = J_mid, f_mid
            else:
                J_lo, f_lo = J_mid, f_mid
            continue
        


    # Interpolate Kq and eta at J_root
    Kq0 = Kq_base(J_mid, pitch_ratio, AE_AO, z)
    dKq = delta_Kq(J_mid, pitch_ratio, AE_AO, z, Re)
    Kq_root = Kq0 + dKq

    P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root
    BHP = (P_E / eta_R) / eta_T
    rel_diff_n = abs(P_E - P_E_initial) / P_E_initial
    return {
        'n_P': n_P,
        'P_E': P_E,
        'BHP': BHP,
        'consistent': rel_diff_n <= tol_n,
        'rel_diff_n': rel_diff_n
    }


# --------------------------- STAGE 5 ------------------------------------------
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
    V_array, super_Kt_map, super_Kq_map, super_eta_map
            ):
    
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
    pitch_ratio = P_i / D_P
    # Predefine J_array for open-water sampling
    velocities = []
    eta = []
    for V in V_array:
        V_A = V * (1 - w)
        R_T = RT_func(V)
        T_required = R_T / (1 - t)
        # Thrust coefficient quadratic c2
        c2 = T_required / (rho * D_P**2 * V_A**2)

        f0 = (AE_AO         - area_limits[0]) / (area_limits[-1] - area_limits[0])
        f1 = (V             - V_array[0])      / (V_array[-1]      - V_array[0])
        f2 = (pitch_ratio   - pitch_array[0])  / (pitch_array[-1]  - pitch_array[0])

        # convert to integer index in [0, A-1], [0, V-1], [0, P-1]
        i0 = int(round(f0 * (area_limits.size - 1)))
        i1 = int(round(f1 * (V_array.size      - 1)))
        i2 = int(round(f2 * (pitch_array.size  - 1)))

        # now safely index
        # open-water data for geometry
        Kq_root = super_Kq_map[i0, i1, i2]
        Kt_root = super_Kt_map[i0, i1, i2]
        eta_O = super_eta_map[i0, i1, i2]

        J_root = eta_O * Kq_root / Kt_root
        # compute n_P, P_E
        n_P = V_A / (J_root * D_P)
        P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root
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
    df = pd.DataFrame(results)
    return df, velocities, eta


# =============================================================================
# Example usage (to be adapted by user with real input functions and data):
# 
# # User defines or provides:
# def RT_func(V): ...
# def open_water_data(AE_AO, pitch_ratio, z, J_array): ...
# def cavitation_min_AE_AO_func(D_P, T): ...
# 
# # Stage 1:
# D_P0 = estimate_initial_diameter(parent_D=..., parent_MCR=..., parent_nMCR=..., c1=1.0)
# 
# # Stage 2:
# stage2_res = optimize_pitch(
#     D_P=D_P0, AE_AO=0.55, z=4, V_service=..., RT_func=RT_func,
#     w=0.2, t=0.2, rho=1025, eta_R=0.98, eta_T=0.99
# )
# # Extract P_E0, n_P0:
# P_E0 = stage2_res['P_E']
# n_P0 = stage2_res['n_P']
# 
# # Stage 3:
# AE_AO_array = np.linspace(0.45, 0.65, 5)
# V_max_search_array = np.linspace(V_service, V_service*1.1, 10)
# stage3_res = optimize_dimensions(
#     P_E=P_E0, n_P=n_P0, z=4, RT_func=RT_func,
#     w=0.2, t=0.2, rho=1025,
#     cavitation_min_AE_AO_func=cavitation_min_AE_AO_func,
#     AE_AO_array=AE_AO_array,
#     pitch_ratio_array=np.linspace(0.6, 1.8, 25),
#     V_max_search_array=V_max_search_array
# )
# # Extract optimized geometry and V_max:
# D_P_opt = stage3_res['D_P_opt']
# AE_AO_opt = stage3_res['AE_AO_opt']
# P_i_opt = stage3_res['P_i_opt']
# V_max_opt = stage3_res['V_max']
# pitch_ratio_opt = stage3_res['pitch_ratio_opt']
# 
# # Stage 4: consistency check
# stage4_res = check_consistency_stage4(
#     D_P=D_P_opt, P_i=P_i_opt, AE_AO=AE_AO_opt, z=4,
#     V_design=V_max_opt, RT_func=RT_func,
#     w=0.2, t=0.2, rho=1025,
#     eta_R=0.98, eta_T=0.99,
#     n_P_initial=n_P0
# )
# if not stage4_res['consistent']:
#     print("Inconsistency detected; consider iterating Stage 3 with updated n_P:", stage4_res)
# else:
#     print("Geometry and engine match consistent:", stage4_res)
# 
# # Stage 5: Speed-power curve
# V_array = np.linspace(0.8 * V_service, V_max_opt * 1.05, 15)
# df_curve = generate_speed_power_curve(
#     D_P=D_P_opt, AE_AO=AE_AO_opt, P_i=P_i_opt, z=4,
#     RT_func=RT_func, w=0.2, t=0.2, rho=1025,
#     eta_R=0.98, eta_H=0.98, eta_T=0.99,
#     V_array=V_array
# )
# # Display results
# import ace_tools as tools; tools.display_dataframe_to_user("Speed-Power Curve", df_curve)
#
# =============================================================================
# End of template code.
# =============================================================================

if __name__ == "__main__":
    
    Open_water_efficiency = 0.71
    Hull_efficiency = 1.1503
    Rotative_efficiency = 1.03
    transmission_efficiency = 0.9979

    V_service = 16 * 0.5144

    w = 0.3054
    t = 0.2010
    D_parent = 9.93

    AE_AO = 0.4870
    z = 4

    stage_1_D_P = estimate_initial_diameter(D_parent)

    stage_2_parameters = optimize_pitch(stage_1_D_P, AE_AO, z, V_service, w, t, 1025, 
                                        Rotative_efficiency, transmission_efficiency, J_array)

    print(f'1st run stage 2 parameters are: \n {stage_2_parameters}')

    area_limits = np.linspace(0.3, 1.1, 801)
    velocity_limits = np.linspace(5, 10, 501)


    # 3) Precompute lookup tables
    shape = (area_limits.size, velocity_limits.size, pitch_array.size)
    Kt_map  = np.empty(shape, dtype=np.float64)
    Kq_map  = np.empty(shape, dtype=np.float64)
    eta_map = np.empty(shape, dtype=np.float64)
    
    for i in range(1, 8):
        area_limits = np.linspace(0.3, 1.1, 401)
        next_round_velocities = np.linspace(5, 10, 501)

        
        for j in range(1, 11):
            stage_3_parameters, Kt_map, Kq_map, eta_map = optimize_dimensions(V_service, stage_2_parameters['P_E'],
                                                stage_2_parameters['n_P'],
                                                z, w, t, 1025, J_array, area_limits, next_round_velocities,Kt_map, 
                                                                                                            Kq_map, 
                                                                                                            eta_map)
            if j == 1:
                super_Kt_map, super_Kq_map, super_eta_map = Kt_map, Kq_map, eta_map
            else:
                super_Kt_map, super_Kq_map, super_eta_map = make_new_map(Kt_map, Kq_map, eta_map, super_Kt_map, super_Kq_map, super_eta_map)

            if stage_3_parameters['soln_attained'] == False:

                print(f"the solution geomtery is not atained in the stage 3, so now we have changed limits to {area_limits[0]}, {area_limits[-1]}")

                if np.array_equal(area_limits, stage_3_parameters['AE_AO_values']):
                    print(f"The area limit have resulted agian to the same values in this __{(i - 1)}__{j} iteration, so now we increase the velocity range and try agian, \n wish me luck")
                    print(f"Yes  am increasing the velocity terms to this value {(10 * j * i)} ")

                    next_round_velocities = np.linspace((2 + velocity_limits[1]) / 2, (22 + (velocity_limits[-2])) / 2, velocity_limits.size - 1)

                    total_velocities = next_round_velocities.size + velocity_limits.size
                    shape = (area_limits.size, total_velocities, pitch_array.size)

                    Kt_map  = np.empty(shape, dtype=np.float64)
                    Kq_map  = np.empty(shape, dtype=np.float64)

                    velocity_limits = np.linspace(2, 22, total_velocities)

                area_limits = stage_3_parameters['AE_AO_values']
            else:
                break

        if i % 10 == 1:                         
            print(f'{i}st stage 3 parameters are: \n {stage_3_parameters}')
        elif i % 10 == 2:                         
            print(f'{i}nd stage 3 parameters are: \n {stage_3_parameters}')
        elif i % 10 == 3:                         
            print(f'{i}rd stage 3 parameters are: \n {stage_3_parameters}')
        else:                         
            print(f'{i}th stage 3 parameters are: \n {stage_3_parameters}')

        # best = {
        #                             'AE_AO_opt': AE_AO,
        #                             'D_P_opt': D_P_candidate,
        #                             'pitch_ratio_opt': pitch_ratio,
        #                             'P_i_opt': pitch_ratio * D_P_candidate,
        #                             'V_max': V,
        #                             'eta_O_at_V_max': eta_val
        #                         }

        stage_4_parameters = check_consistency_stage4(stage_3_parameters['D_P_opt'],
                                                    stage_3_parameters['pitch_ratio_opt'],
                                                    stage_3_parameters['AE_AO_opt'],
                                                    z, stage_3_parameters['V_max'],
                                                    w, t, 1025, Rotative_efficiency,
                                                    transmission_efficiency,
                                                    stage_2_parameters['n_P'],
                                                    stage_2_parameters['P_E'],
                                                    J_array)
        


        # stage_4_parameters are:    
        # 'n_P': n_P,
        #     'P_E': P_E,
        #     'BHP': BHP,
        #     'consistent': rel_diff_n <= tol_n,
        #     'rel_diff_n': rel_diff_n

        if i % 10 == 1:                         
            print(f'{i}st stage 4 parameters are: \n {stage_4_parameters}')
        elif i % 10 == 2:                        
            print(f'{i}nd stage 4 parameters are: \n {stage_4_parameters}')
        elif i % 10 == 3:                        
            print(f'{i}rd stage 4 parameters are: \n {stage_4_parameters}')
        else:                        
            print(f'{i}th stage 4 parameters are: \n {stage_4_parameters}')

        if stage_4_parameters['rel_diff_n'] < 0.05:
            break
        else:
            stage_2_parameters['P_E'] = stage_4_parameters['P_E']
            stage_2_parameters['n_P'] = stage_4_parameters['n_P']
            continue

    if stage_4_parameters['rel_diff_n'] <= 0.05:
        print(f'tolerance is acheived at the {i} run')


    # def generate_speed_power_curve(
    # D_P,
    # AE_AO,
    # P_i,
    # z,
    # RT_func,
    # w,
    # t,
    # rho,
    # eta_R,
    # eta_H,
    # eta_T,
    # V_array
    #         )

    df, velocities, eta = generate_speed_power_curve(stage_3_parameters['D_P_opt'],
                                stage_3_parameters['AE_AO_opt'],
                                stage_3_parameters['pitch_ratio_opt'],
                                z, w, t, 1025, Rotative_efficiency, Hull_efficiency, transmission_efficiency, next_round_velocities, super_Kt_map, super_Kq_map, super_eta_map)

    print(velocities)
    print()
    print(eta)

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
            f"Stage5",
            f"{z}_blades",
            f"AE_AO_{stage_3_parameters['AE_AO_opt']}",
            f"P_D_{stage_3_parameters['pitch_ratio_opt']}",
            f"Rotation{stage_4_parameters['n_P']}"
        )

    os.makedirs(sub_folder, exist_ok=True)

    image_path = os.path.join(sub_folder, f"J_Kt_Kq_n0_{stage_3_parameters['pitch_ratio_opt']}_{stage_3_parameters['AE_AO_opt']}_{z}_{stage_4_parameters['n_P']}.png")

    fig.savefig(image_path)
    plt.close(fig)
