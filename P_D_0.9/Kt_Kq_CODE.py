import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import os
import pandas
import psutil

# === 1. Define base polynomial coefficients from Table 1 ===
# For example, you might create lists of dicts:
# Each term: {'s': int, 't': int, 'u': int, 'v': int, 'C_T': float, 'C_Q': float}
# You need to fill these lists from Table 1.

# === 2. Define Reynolds correction coefficients from Table 2 ===
# You can code delta_Kt and delta_Kq functions directly once you transcribe terms.
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

# === 4. Total Kt, Kq, efficiency ===
def compute_Kt_Kq_eta(Kt_Terms, Kq_Terms, J, P_D, AE_A0, z, V):
    """
    Returns Kt, Kq, eta arrays for given parameters.
    J: array
    P_D, AE_A0, z: scalars
    Re_n: scalar Reynolds number
    """
    diameter = 0.24

    b = 1
    for l in range(len(V)):
        velocity = V[l]
        # m = 5
        # if l != m:
        #     continue
        sub_folder = f"Kt_Kq_Generation\Velocity_{velocity}_m_s"
        os.makedirs(sub_folder, exist_ok= True)

        for k in range(len(z)):
            number_of_blades = z[k]
            sub_folder = f"Kt_Kq_Generation\Velocity_{velocity}_m_s\{number_of_blades}_blades"
            os.makedirs(sub_folder, exist_ok= True)
            

            for i in range(len(AE_A0)):
                Expanded_Area_Ratio = AE_A0[i]
                # y = 10
                # if i != y:
                #     continue
                sub_folder = f"Kt_Kq_Generation\Velocity_{velocity}_m_s\{number_of_blades}_blades\AE_AO_{Expanded_Area_Ratio}"
                os.makedirs(sub_folder, exist_ok= True)
                for j in range(len(P_D)):
                    pitch_ratio = P_D[j]
                    
                    sub_folder = f"Kt_Kq_Generation\Velocity_{velocity}_m_s\{number_of_blades}_blades\AE_AO_{Expanded_Area_Ratio}\P_D_{pitch_ratio}"
                    os.makedirs(sub_folder, exist_ok= True)
                    # n = 10
                    # if j != n:
                    #     continue
                    Kt_values = []
                    Kq_values = []
                    efficiency = []
                    J_Values =[]

                    # if b != 1: 
                    #     return 0, 0, 0    

                    for a in range(len(J)):
                        # x = 15
                        # if x != a:
                        #     continue
                        advance_coefficient = J[a]
                        
                        RPM = velocity / (advance_coefficient * diameter)
                        if z == 3:
                            chord_length = 2.1475 * (diameter / number_of_blades) * Expanded_Area_Ratio
                        else:
                            chord_length = 2.057 * (diameter / number_of_blades) * Expanded_Area_Ratio

                        velocity_0_75 = RPM  * diameter * (((advance_coefficient**2) + ((0.75 * 22 / 7)**2))**0.5)

                        Reynolds = velocity_0_75 * chord_length / (1.0038 * (10**-6))

                        Kt0 = Kt_base(Kt_Terms, advance_coefficient, pitch_ratio, Expanded_Area_Ratio, number_of_blades)
                        Kq0 = Kq_base(Kq_Terms, advance_coefficient, pitch_ratio, Expanded_Area_Ratio, number_of_blades)
                        
                        dKt = 0
                        dKq = 0
                        
                        if Reynolds > 2000000:
                            dKt = delta_Kt(advance_coefficient, pitch_ratio, Expanded_Area_Ratio, number_of_blades, Reynolds)
                            dKq = delta_Kq(advance_coefficient, pitch_ratio, Expanded_Area_Ratio, number_of_blades, Reynolds)
                                                
                        Kt = Kt0 + dKt
                        Kq = Kq0 + dKq

                        try:
                            OWE = (advance_coefficient / (2 * 22 / 7)) * (Kt / Kq)
                        except:
                            OWE = 0

                        if OWE < 0:
                            b += 1

                            Kt_values.append(Kt)
                            Kq_values.append(Kq)
                            efficiency.append(OWE)
                            J_Values.append(advance_coefficient)
                            # print(f"{b} Re = {Reynolds} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
                            print(f"{b} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
                            
                            print(psutil.virtual_memory())

                            break

                        # print(f"{b} Re = {Reynolds} Kt = {Kt} Kq = {Kq} efficiency = {OWE} chord length = {chord_length} Rpm = {RPM} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
                        print(f"{b} advance coefficient = {advance_coefficient} Pitch ratio = {pitch_ratio} Expanded area Ratio = {Expanded_Area_Ratio} blades = {number_of_blades} Velocity = {velocity}")
                            

                        print(psutil.virtual_memory())
                        b += 1

                        Kt_values.append(Kt)
                        Kq_values.append(Kq)
                        efficiency.append(OWE)
                        J_Values.append(advance_coefficient)

                    # print(Kt_values[14: 16])    
                    # print(Kq_values[14: 16])
                    # print(efficiency[14: 16])
                    
                    data = {'J' : J_Values, 'Kt' : Kt_values, 'Kq' : Kq_values, 'n_0' : efficiency}
                    df = pandas.DataFrame(data)
                    excel_path = os.path.join(sub_folder, f"J_Kt_Kq_n0_{pitch_ratio}_{Expanded_Area_Ratio}_{number_of_blades}_{velocity}.xlsx")
                    df.to_excel(excel_path, index= False)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.set_xlim(left= 0, right= 1.6)
                    ax.set_ylim(bottom= 0, top= 1.5)

                    ax.plot(J_Values, Kt_values, label=f"K_t (P/D={pitch_ratio}, AE/A0={Expanded_Area_Ratio}, z={number_of_blades})")
                    ax.plot(J_Values, Kq_values, label=f"K_q (P/D={pitch_ratio}, AE/A0={Expanded_Area_Ratio}, z={number_of_blades})")
                    ax.plot(J_Values, efficiency, label=f"η₀ (P/D={pitch_ratio}, AE/A0={Expanded_Area_Ratio}, z={number_of_blades})")
                    ax.set_xlabel("Advance Coefficient J")
                    
                    ax.set_ylabel("Thrust coefficient K_t")
                    ax.set_ylabel("Torque coefficient K_q")
                    ax.set_ylabel("Open-water efficiency η₀")

                    ax.legend()
                    ax.grid(True)

                    ax.minorticks_on()

                    # # Major grid
                    # ax.grid(which='major', linestyle='-', linewidth=0.75, color='black')

                    # Minor (sub-) grid
                    ax.grid(which='minor', linestyle=':', linewidth=0.01, color='gray', axis="x")
                    ax.grid(which='minor', linestyle=':', linewidth=0.05, color='gray', axis="y")

                    image_path = os.path.join(sub_folder, f"J_Kt_Kq_n0_{pitch_ratio}_{Expanded_Area_Ratio}_{number_of_blades}_{velocity}.png")

                    fig.savefig(image_path)
                    plt.close(fig)                  

                    pass
                pass
            pass
        pass
    return 0, 0, 0
                           
      
    # Avoid division by zero; typically J>0

KT_terms = {
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


KQ_terms = {
            "C" : [0.00379368, 0.00886523, -0.032241, 0.00344778, -0.0408811, -0.108009, -0.0885381, 0.188561, -0.00370871, 0.00513696, 0.0209449, 0.00474319, -0.00723408, 
                   0.00438388, -0.0269403, 0.0558082, 0.0161886, 0.00318086, 0.015896, 0.0471729, 0.0196283, -0.0502782, -0.030055, 0.0417122, -0.0397722, -0.00350024, 
                   -0.0106854, 0.00110903, -0.000313912, 0.0035985, -0.00142121, -0.00383637, 0.0126803, -0.00318278, 0.00334268, -0.00183491, 0.000112451, -2.97228e-05, 
                   0.000269551, 0.00082365, 0.00155334, 0.000302683, -0.0001843, -0.000425399, 8.69243e-05, -0.0004659, 5.54194e-05], 
            "s" : [0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 2, 1, 0, 3, 0, 1, 0, 1, 3, 0, 3, 2, 0, 0, 3, 3, 0, 3, 0, 1, 0, 2, 0, 1, 3, 3, 1, 2, 0, 0, 0, 0, 3, 0, 1],
            "t" : [0, 0, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 2, 0, 3, 3, 0, 0, 0, 1, 1, 2, 3, 6, 0, 3, 6, 0, 6, 0, 2, 3, 6, 1, 2, 6, 0, 0, 2, 6, 0, 3, 3, 6, 6],
            "u" : [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "v" : [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]           
            }

z = [3, 4, 5, 6, 7]
Expanded_Area_Ratios = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
Pitch_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
velocities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
J_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55]

folder_path = "Kt_Kq_Generation"
os.makedirs(folder_path, exist_ok= True)


# Compute
Kt_vals, Kq_vals, eta_vals = compute_Kt_Kq_eta(KT_terms, KQ_terms, J_vals, Pitch_ratios, Expanded_Area_Ratios, z, velocities)

