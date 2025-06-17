import numpy as np
import pandas as pd
import os
import math
import psutil

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
def Kt_base(KT_terms, J, P_D, AE_A0, z):
    """
    Compute base (nominal) K_T from Table 1 polynomial.
    J: scalar or array
    P_D, AE_A0, z: scalars
    """
    val = 0.0
    s = KT_terms["s"]
    t = KT_terms["t"]
    u = KT_terms["u"]
    v = KT_terms["v"]
    C = KT_terms["C"]

    for d in range(len(s)):
        # contribution: C * J^s * (P/D)^t * (AE/A0)^u * z^v
        val += C[d] * (J**s[d]) * (P_D**t[d]) * (AE_A0**u[d]) * (z**v[d])
        
    return val

def Kq_base(KQ_terms, J, P_D, AE_A0, z):
    """
    Compute base (nominal) K_Q from Table 1 polynomial.
    """
    val = 0.0
    s = KQ_terms["s"]
    t = KQ_terms["t"]
    u = KQ_terms["u"]
    v = KQ_terms["v"]
    C = KQ_terms["C"]

    for d in range(len(s)):
        # contribution: C * J^s * (P/D)^t * (AE/A0)^u * z^v
        val += C[d] * (J**s[d]) * (P_D**t[d]) * (AE_A0**u[d]) * (z**v[d])
    return val


def open_water_data(J, velocity, blades, area_ratio, pitch_ratio, b, diameter=0.24):

    Kt_Terms = {
            "C" : [0.00880496, -0.204554, 0.166351, 0.158114, -0.147581, -0.481497, 0.415437, 0.0144043, -0.0530054, 0.0143481, 0.0606826, -0.0125894, 0.0109689, -0.133698, 
                 0.00638407, -0.00132718, 0.168496, -0.0507214, 0.0854559, -0.0504475, 0.010465, -0.00648272, -0.00841728, 0.0168424, -0.00102296, -0.0317791, 0.018604, 
                 -0.00410798, -0.000606848, -0.0049819, 0.0025983, -0.000560528, -0.00163652, -0.000328787, 0.000116502, 0.000690904, 0.00421749, 5.65229e-05, -0.00146564], 
            "s" : [0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 
                 3.0, 1.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
            "t" : [0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 3.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 
                   0.0, 2.0, 6.0, 6.0, 0.0, 3.0, 6.0, 3.0],
            "u" : [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 
                   0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
            "v" : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 
                   2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]           
            }


    Kq_Terms = {
            "C" : [0.00379368, 0.00886523, -0.032241, 0.00344778, -0.0408811, -0.108009, -0.0885381, 0.188561, -0.00370871, 0.00513696, 0.0209449, 0.00474319, -0.00723408, 
                   0.00438388, -0.0269403, 0.0558082, 0.0161886, 0.00318086, 0.015896, 0.0471729, 0.0196283, -0.0502782, -0.030055, 0.0417122, -0.0397722, -0.00350024, 
                   -0.0106854, 0.00110903, -0.000313912, 0.0035985, -0.00142121, -0.00383637, 0.0126803, -0.00318278, 0.00334268, -0.00183491, 0.000112451, -2.97228e-05, 
                   0.000269551, 0.00082365, 0.00155334, 0.000302683, -0.0001843, -0.000425399, 8.69243e-05, -0.0004659, 5.54194e-05], 
            "s" : [0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0, 3, 0, 1, 0, 1, 3, 0, 3, 2, 0, 0, 3, 3, 0, 3, 0, 1, 0, 2, 0, 1, 3, 3, 1, 2, 0, 0, 0, 0, 3, 0, 1],
            "t" : [0, 0, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 2, 0, 3, 3, 0, 0, 0, 1, 1, 2, 3, 6, 0, 3, 6, 0, 6, 0, 2, 3, 6, 1, 2, 6, 0, 0, 2, 6, 0, 3, 3, 6, 6],
            "u" : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "v" : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]           
            }

    """
    Compute Kt, Kq, efficiency for one configuration,
    save Excel and PNG with proper closing and sub-grids.
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

    Kt_vals, Kq_vals, eff_vals, J_values = [], [], [], []

    # Loop over J values
    for J_val in J:
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
        Kt0 = Kt_base(Kt_Terms, J_val, pitch_ratio, area_ratio, blades)
        Kq0 = Kq_base(Kq_Terms, J_val, pitch_ratio, area_ratio, blades)

        dKt = dKq = 0
        if Re > 2_000_000:
            dKt = delta_Kt(J_val, pitch_ratio, area_ratio, blades, Re)
            dKq = delta_Kq(J_val, pitch_ratio, area_ratio, blades, Re)

        Kt, Kq = Kt0 + dKt, Kq0 + dKq
        eta0 = (J_val / (2 * 22 / 7)) * (Kt / Kq)

        try:
            eta0 = (J_val / (2 * 22 / 7)) * (Kt / Kq)
        except:
            eta0 = 0

        if eta0 < 0:
            b += 1

            Kt_vals.append(Kt)
            Kq_vals.append(Kq)
            eff_vals.append(eta0)
            J_values.append(J_val)
            # print(f"{b} Re = {Re} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
            
            print(f"{b} advance coefficient = {J_val} Pitch ratio = {pitch_ratio} Expanded area Ratio = {area_ratio} blades = {blades} Velocity = {velocity}")
            print(psutil.virtual_memory())

            break

        # print(f"{b} Re = {Reynolds} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
        print(f"{b} advance coefficient = {J_val} Pitch ratio = {pitch_ratio} Expanded area Ratio = {area_ratio} blades = {blades} Velocity = {velocity}")

        Kt_vals.append(Kt)
        Kq_vals.append(Kq)
        eff_vals.append(eta0)
        J_values.append(J_val)

        pass
    
    return Kt_vals, Kq_vals, eff_vals, J_values 



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
    RT_func,
    w,
    t,
    rho,
    eta_R,
    eta_T,
    J_array=None,
    pitch_ratio_array=None,
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
    if J_array is None:
        J_array = np.linspace(0.1, 1.2, 400)
    if pitch_ratio_array is None:
        pitch_ratio_array = np.linspace(0.5, 2.0, 30)
    
    # Effective inflow speed
    V_A = V_service * (1 - w)
    # Required thrust
    R_T = RT_func(V_service)

    T_required = R_T / (1 - t)
    # Coefficient for thrust quadratic: Kt = c2 * J^2
    c2 = T_required / (rho * D_P**2 * V_A**2)
    
    best = {'eta_O': -np.inf}
    
    for pitch_ratio in pitch_ratio_array:
        # User-supplied function: given AE/AO, pitch_ratio, z, and J_array,
        # return open-water Kt_array, Kq_array, eta_O_array.
        Kt_array, Kq_array, eta_O_array, J_array = open_water_data(J_array, V_A, z, AE_AO, pitch_ratio, AE_AO, 1, D_P)
        
        # Define function f(J) = Kt(J) - c2 * J^2. We find roots in J_array.
        f = Kt_array - c2 * J_array**2
        
        # Search for sign changes to locate intervals for root-finding (bisection-like)
        sign_changes = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        for idx in sign_changes:
            J_lo, J_hi = J_array[idx], J_array[idx+1]
            f_lo, f_hi = f[idx], f[idx+1]
            # simple bisection
            for _ in range(50):
                J_mid = 0.5 * (J_lo + J_hi)
                # interpolate Kt at J_mid
                Kt_mid = np.interp(J_mid, J_array, Kt_array)
                f_mid = Kt_mid - c2 * J_mid**2
                if abs(f_mid) < tol_root:
                    break
                if f_lo * f_mid < 0:
                    J_hi, f_hi = J_mid, f_mid
                else:
                    J_lo, f_lo = J_mid, f_mid
            J_root = J_mid
            # Interpolate Kq and eta at J_root
            Kq_root = np.interp(J_root, J_array, Kq_array)
            eta_root = np.interp(J_root, J_array, eta_O_array)
            # Compute engine speed
            n_P = V_A / (J_root * D_P)  # [1/s]
            # Delivered power P_E from torque equation: P_E = 2*pi * rho * n_P^3 * D_P^5 * Kq
            P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root
            # Brake horsepower: account for hull-propeller and transmission efficiencies
            BHP = (P_E / eta_R) / eta_T
            # Check if this is better (max eta_O)
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
def optimize_dimensions(
    P_E,
    n_P,
    z,
    RT_func,
    w,
    t,
    rho,
    cavitation_min_AE_AO_func,
    AE_AO_array,
    pitch_ratio_array,
    V_max_search_array,
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
    best = {'V_max': -np.inf}
    
    for AE_AO in AE_AO_array:
        for V in V_max_search_array:
            V_A = V * (1 - w)
            R_T = RT_func(V)
            T_required = R_T / (1 - t)
            # coefficient for Kq intersection: from torque eq:
            # we look for J such that P_E = 2*pi * rho * n_P^3 * D_P^5 * Kq(J).
            # But since D_P unknown, we instead for each J and pitch_ratio compute D_P, then T_P.
            for pitch_ratio in pitch_ratio_array:
                # get open-water Kq and Kt arrays over J_array
                # Use a J_array that covers typical range, e.g. np.linspace(0.1, 1.2, 200)
                J_array = np.linspace(0.1, 1.2, 300)
                Kt_array, Kq_array, eta_O_array = open_water_data(AE_AO, pitch_ratio, z, J_array)
                # For each J in J_array, compute implied D_P from torque eq:
                # D_P = (P_E / (2*pi * rho * n_P^3 * Kq))**(1/5)
                # Exclude Kq <= 0 to avoid invalid values
                valid = Kq_array > 0
                if not np.any(valid):
                    continue
                D_P_array = (P_E / (2 * np.pi * rho * n_P**3 * Kq_array[valid]))**(1/5)
                # For each candidate D_P and J, compute thrust and compare with required thrust
                J_valid = J_array[valid]
                Kt_valid = Kt_array[valid]
                for D_P_candidate, J_val, Kt_val, eta_val in zip(D_P_array, J_valid, Kt_valid, eta_O_array[valid]):
                    # compute thrust from propeller
                    T_P = rho * n_P**2 * D_P_candidate**4 * Kt_val
                    # required thrust at this V:
                    # T_required already computed
                    # check if within tolerance
                    if abs(T_P - T_required) / T_required <= tol_thr:
                        # check cavitation: AE/AO must be >= min from criterion
                        AE_AO_min = cavitation_min_AE_AO_func(D_P_candidate, T_P)
                        if AE_AO < AE_AO_min:
                            # cavitation risk: skip
                            continue
                        # record feasible solution: geometry yields speed V
                        if V > best['V_max']:
                            best = {
                                'AE_AO_opt': AE_AO,
                                'D_P_opt': D_P_candidate,
                                'pitch_ratio_opt': pitch_ratio,
                                'P_i_opt': pitch_ratio * D_P_candidate,
                                'V_max': V,
                                'eta_O_at_V_max': eta_val
                            }
    if best['V_max'] < 0:
        raise RuntimeError("No feasible geometry found in Stage 3. Check scanning ranges or inputs.")
    return best


# --------------------------- STAGE 4 ------------------------------------------
def check_consistency_stage4(
    D_P,
    P_i,
    AE_AO,
    z,
    V_design,
    RT_func,
    w,
    t,
    rho,
    eta_R,
    eta_T,
    n_P_initial,
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
    pitch_ratio = P_i / D_P
    J_array = np.linspace(0.1, 1.2, 300)
    Kt_array, Kq_array, eta_O_array = open_water_data(AE_AO, pitch_ratio, z, J_array)
    f = Kt_array - c2 * J_array**2
    # find root in J_array as before
    sign_changes = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError("No intersection found in Stage 4 for thrust equation.")
    # take first intersection (or choose the one with reasonable n_P)
    idx = sign_changes[0]
    J_lo, J_hi = J_array[idx], J_array[idx+1]
    f_lo, f_hi = f[idx], f[idx+1]
    # bisection
    for _ in range(50):
        J_mid = 0.5 * (J_lo + J_hi)
        Kt_mid = np.interp(J_mid, J_array, Kt_array)
        f_mid = Kt_mid - c2 * J_mid**2
        if abs(f_mid) < 1e-4:
            break
        if f_lo * f_mid < 0:
            J_hi, f_hi = J_mid, f_mid
        else:
            J_lo, f_lo = J_mid, f_mid
    J_root = J_mid
    # compute n_P and P_E
    n_P = V_A / (J_root * D_P)
    Kq_root = np.interp(J_root, J_array, Kq_array)
    P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root
    BHP = (P_E / eta_R) / eta_T
    rel_diff_n = abs(n_P - n_P_initial) / n_P_initial
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
    RT_func,
    w,
    t,
    rho,
    eta_R,
    eta_H,
    eta_T,
    V_array
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
    J_array = np.linspace(0.1, 1.2, 300)
    
    for V in V_array:
        V_A = V * (1 - w)
        R_T = RT_func(V)
        T_required = R_T / (1 - t)
        # Thrust coefficient quadratic c2
        c2 = T_required / (rho * D_P**2 * V_A**2)
        # open-water data for geometry
        Kt_array, Kq_array, eta_O_array = open_water_data(AE_AO, pitch_ratio, z, J_array)
        # find J intersection as before
        f = Kt_array - c2 * J_array**2
        sign_changes = np.where(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
        if len(sign_changes) == 0:
            # No intersection: skip or record NaNs
            results.append({
                'V': V, 'EHP': np.nan, 'n_P': np.nan, 'P_E': np.nan,
                'DHP': np.nan, 'BHP': np.nan, 'eta_O': np.nan
            })
            continue
        idx = sign_changes[0]
        J_lo, J_hi = J_array[idx], J_array[idx+1]
        f_lo, f_hi = f[idx], f[idx+1]
        # bisection
        for _ in range(50):
            J_mid = 0.5 * (J_lo + J_hi)
            Kt_mid = np.interp(J_mid, J_array, Kt_array)
            f_mid = Kt_mid - c2 * J_mid**2
            if abs(f_mid) < 1e-4:
                break
            if f_lo * f_mid < 0:
                J_hi, f_hi = J_mid, f_mid
            else:
                J_lo, f_lo = J_mid, f_mid
        J_root = J_mid
        # compute n_P, P_E
        n_P = V_A / (J_root * D_P)
        Kq_root = np.interp(J_root, J_array, Kq_array)
        eta_O = np.interp(J_root, J_array, eta_O_array)
        P_E = 2 * np.pi * rho * n_P**3 * D_P**5 * Kq_root
        # EHP, DHP, BHP
        EHP = R_T * V
        # overall propulsive efficiency eta_D = eta_O * eta_H * eta_R
        eta_D = eta_O * eta_H * eta_R
        if eta_D <= 0:
            DHP = np.nan
            BHP = np.nan
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
    df = pd.DataFrame(results)
    return df


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
