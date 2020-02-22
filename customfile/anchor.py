import numpy as np
scales_per_octave = 3
octave_base_scale = 4
octave_scales = np.array(
    [2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
anchor_scales = octave_scales * octave_base_scale
print(anchor_scales)