"""
Convert a qBraid result JSON into a counts file.
Usage: python save_qbraid_counts.py <name> <result.json>
"""
import json
import sys

name, path = sys.argv[1], sys.argv[2]
d = json.load(open(path))
counts = (d.get("resultData", {}).get("measurementCounts") or d.get("measurementCounts")
          or d.get("counts") or d)
if not isinstance(counts, dict) or not all(isinstance(v, int) for v in counts.values()):
    sys.exit("Could not find a {bitstring: count} block in that file.")
out = dict(device=f"{name} (qBraid)", counts=counts, cost=d.get("cost"), job_status=d.get("status"))
json.dump(out, open(f"counts_{name}.json", "w"))
print(f"saved counts_{name}.json with {sum(counts.values())} shots")
