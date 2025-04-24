from opacus.accountants.analysis import rdp as privacy_analysis
import numpy as np

rdp = privacy_analysis.compute_rdp(
        q=1.0,
        noise_multiplier=3.35,
        steps=20,
        orders=[1.0 + 0.1 * t for t in range(1, 1000)],
    )
eps, opt_alpha = privacy_analysis.get_privacy_spent(
        orders=[1.0 + 0.1 * t for t in range(1, 1000)], rdp=rdp, delta=0.45
    )
print(eps)