from functools import lru_cache
import numpy as np
from CODES.Test_codes.Propeller_5_stages import open_water_data_diameter, Kq_base, Kt_base, delta_Kq, delta_Kt
import time
import sys



@lru_cache(maxsize=None)
def prop_map(AE, V, pitch):
    J_array = np.linspace(0.1, 1.6, 501)
    return open_water_data_diameter(J_array, V, 4, AE, pitch, 1, 3)

j = 1
AE_AO_array = np.linspace(0.3, 1.1, 201)
V_max_search_array = np.linspace(0, 20, 201)
pitch_ratio_array = np.linspace(0.2, 2.0, 501)
total = len(AE_AO_array) * len(V_max_search_array) * len(pitch_ratio_array)

# In your nested loops:
for i in range(1, 3):
    start_time = time.perf_counter()
    for AE in AE_AO_array:
        for V in V_max_search_array:
            for pitch in pitch_ratio_array:
                Kt_array, Kq_array, eta_array, J_array = prop_map(AE, V, pitch)
                msg = f"Completed iteration {j}/{total * i}"
                sys.stdout.write("\r" + msg)   # write carriage-return + message
                sys.stdout.flush()             # force it onto the screen
                j += 1
    name = "time"
    end_time = time.perf_counter()
    label = f" [{name}]" if name else ""
    print(f"Elapsed{label}: {end_time - start_time:.4f}s")
    