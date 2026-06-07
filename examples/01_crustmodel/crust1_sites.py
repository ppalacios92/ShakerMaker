# 01 - CRUST1.0 tool: crustal profile at given lat/lon and a CrustModel snippet.
# Data: Laske et al., CRUST1.0 (attribution printed on use).
# 2026-06-06

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "supplementary", "crust1"))
from crust1 import Crust1

# sites: (lat, lon) in degrees
sites = [(-33.42, -70.61), (37.77, -122.51)]

crust1 = Crust1()                       # prints CRUST1 attribution + link

# per-site response: profile summary + ready-to-paste CrustModel snippet
for lat, lon in sites:
    p = crust1.profile_at(lat, lon)
    print(f"\nsite ({lat}, {lon})  avg_Vs={p['avg_vs']:.3f} km/s  moho={p['moho_depth_km']:.1f} km")
    crust1.print_shakermaker((lat, lon))

print("PASS")
