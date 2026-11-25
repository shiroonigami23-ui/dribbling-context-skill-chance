"""Download the public Wyscout data (Pappalardo et al., 2019) used in Studies 5-6.

The data are mirrored in github.com/koenvo/wyscout-soccer-match-event-dataset (original release:
figshare, doi:10.6084/m9.figshare.c.4415000).  This clones the mirror and unzips raw_data/*.zip into
data/wyscout_raw/ (override with the WYSCOUT_DIR environment variable).  About 1.5 GB unzipped.
"""
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

REPO = "https://github.com/koenvo/wyscout-soccer-match-event-dataset"
OUT = Path(os.environ.get("WYSCOUT_DIR", "data/wyscout_raw"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "teams.json").exists() and (OUT / "players.json").exists() and any(OUT.glob("events_*.json")):
        print(f"Wyscout data already in {OUT}"); return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", REPO, tmp], check=True)
        for j in Path(tmp, "raw_data").glob("*.json"):      # teams.json, players.json
            (OUT / j.name).write_bytes(j.read_bytes())
        for z in sorted(Path(tmp, "raw_data").glob("*.zip")):
            print("unzipping", z.name)
            with zipfile.ZipFile(z) as f:
                for m in f.namelist():
                    if m.endswith(".json") and not m.startswith("__MACOSX"):
                        (OUT / Path(m).name).write_bytes(f.read(m))
    print("files:", sorted(p.name for p in OUT.glob("*.json")))


if __name__ == "__main__":
    main()
