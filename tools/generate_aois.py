#!/usr/bin/env python3
"""
Generate simple AOI GeoJSON boxes for India demo.
Creates files: aoi/india_demo_box1.geojson, aoi/india_demo_box2.geojson, aoi/india_demo_box3.geojson
Coordinates chosen in Kerala / Western Ghats region (Wayānad, Agumbe etc) — adjust if you want.
"""
import json, os
from pathlib import Path

OUT = Path("aoi")
OUT.mkdir(parents=True, exist_ok=True)

# three example boxes (lon,lat) WGS84
boxes = {
    "india_wayanad_forest": [ [76.0,11.6], [76.6,11.6], [76.6,12.1], [76.0,12.1] ],
    "india_western_ghats_agumbe": [ [75.0,13.6], [75.6,13.6], [75.6,14.1], [75.0,14.1] ],
    "india_amboli_forest": [ [73.5,15.8], [74.1,15.8], [74.1,16.3], [73.5,16.3] ],
}

for name, coords in boxes.items():
    poly = {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[ [c[0], c[1]] for c in coords ] + [coords[0]] ]
        }
    }
    outp = OUT / f"{name}.geojson"
    with open(outp, "w") as f:
        json.dump(poly, f)
    print("Wrote AOI:", outp)
print("Done. AOI files in:", OUT)