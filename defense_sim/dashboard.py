"""Build the offline episode replay from recorded evidence."""
import gzip
import json
from pathlib import Path


def main():
    root=Path("results")
    with gzip.open(root/"study.json.gz","rt",encoding="utf8") as f:
        rows=json.load(f)["dashboard"]
    data=json.dumps(rows,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    layout=Path("dashboard/template.html").read_text()
    (root/"defense-replay.html").write_text(layout.replace("__EPISODE_DATA__",data),encoding="utf8")


if __name__=="__main__":
    main()
