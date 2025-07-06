import numpy as np
from numba import njit, prange

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

J_array = np.linspace(0.1, 1.6, 1600)
pitch_array = np.linspace(0.2, 2.0, 1801)
area_limits = np.linspace(0.3, 1.1, 801)
velocity_limits = np.linspace(2, 22, 2001)



@njit(fastmath=True)
def find_best_J_eta(P_E, n_P, rho, V, AE_AO, pitch_ratio,
                    J_array, Kq_array):
    """
    Given precomputed Kt, Kq, eta_O over J_array, find the J_root
    via your Q_prop bisection and then select the max-eta point.
    Returns (best_J, best_eta).
    """
    c4 = (P_E * n_P**2) / ((2 * np.pi) * rho * (V**5))
    tol_root = 1e-3

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
            for _ in range(50):
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
            dKt = delta_Kt(J_mid, pitch_ratio, AE_AO, z, Re)
            Kt_mid = Kt + dKt

            eta_mid = (J_mid / (2 * 22 / 7)) * (Kt_mid / Kq_mid)

            if eta_mid > best_eta:
                best_eta = eta_mid
                best_J   = J_mid
                best_Kt = Kt_mid
                best_Kq = Kq_mid

    return best_J, best_eta, best_Kt, best_Kq

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

        J_root, eta_root, Kt, Kq = find_best_J_eta(P_E, n_P, rho, V_A, AE_AO, pitch,                                                                                                      
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

    return best_overall_J, best_overall_eta, best_overall_pitch, best_Kt, best_Kq, Kt_map, Kq_map, eta_map

@njit(fastmath=True)
def prop_map_diameter(V, z, AE, pitch, d, n_P):
    #J_array, V_A, z, AE_AO, pitch_ratio, 1, n_P
    J_array_prop = np.linspace(0.0, 1.6, 161)
    return open_water_data_diameter(J_array_prop, V, z, AE, pitch, d, n_P)

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
