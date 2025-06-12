import os
import gc
from itertools import product
import pandas as pd
import matplotlib
# Use non-GUI backend for rendering
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import math
import psutil



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



def process_config(Kt_Terms, Kq_Terms, J, velocity, blades, area_ratio, pitch_ratio, b, diameter=0.24):
    """
    Compute Kt, Kq, efficiency for one configuration,
    save Excel and PNG with proper closing and sub-grids.
    """
    # Prepare folder
    sub_folder = os.path.join(
        "Kt_Kq_Generation",
        f"Velocity_{velocity}_m_s",
        f"{blades}_blades",
        f"AE_AO_{area_ratio}",
        f"P_D_{pitch_ratio}"
    )
    os.makedirs(sub_folder, exist_ok=True)

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

        b+= 1

    # Save to Excel
    df = pd.DataFrame({'J': J_values, 'Kt': Kt_vals, 'Kq': Kq_vals, 'n_0': eff_vals})
    excel_path = os.path.join(sub_folder, f"J_Kt_Kq_n0_{pitch_ratio}_{area_ratio}_{blades}_{velocity}.xlsx")
    df.to_excel(excel_path, index=False)

    # Plot with sub-grids
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(J_values, Kt_vals, label=f"K_t (P/D={pitch_ratio}, AE/A0={area_ratio}, z={blades})")
    ax.plot(J_values, Kq_vals, label=f"K_q (P/D={pitch_ratio}, AE/A0={area_ratio}, z={blades})")
    ax.plot(J_values, eff_vals, label=f"η₀ (P/D={pitch_ratio}, AE/A0={area_ratio}, z={blades})")

    # Labels
    ax.set_xlabel("Advance coefficient J")
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
    image_path = os.path.join(sub_folder, f"J_Kt_Kq_n0_{pitch_ratio}_{area_ratio}_{blades}_{velocity}.png")
    fig.savefig(image_path)
    plt.close(fig)

    # Force garbage collection
    gc.collect()

    return b


def compute_Kt_Kq_eta_batched(Kt_Terms, Kq_Terms, J, P_D, AE_A0, z, V, b, batch_size=1000):
    """
    Batch processing of configurations to limit memory usage.
    """
    # All configurations as tuples
    configs = list(product(V, z, AE_A0, P_D))
    total = len(configs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = configs[start:end]
        print(f"Processing batch {start+1}-{end} of {total}")

        for velocity, blades, area_ratio, pitch_ratio in batch:
            b = process_config(Kt_Terms, Kq_Terms, J, velocity, blades, area_ratio, pitch_ratio, b)

        # Optional: checkpoint
        with open("last_completed_batch.txt", "w") as ck:
            ck.write(str(end))

    print("All batches completed.")
    return 0, 0, 0



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

b = 1
z = [3, 4, 5, 6, 7]
Expanded_Area_Ratios = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
Pitch_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
velocities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
J_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55]

folder_path = "Kt_Kq_Generation"
os.makedirs(folder_path, exist_ok= True)


Kt_vals, Kq_vals, eta_vals = compute_Kt_Kq_eta_batched(KT_terms, KQ_terms, J_vals, Pitch_ratios, Expanded_Area_Ratios, z, velocities, b)
