"""Preuzima DeepPCB u data/DeepPCB i provjerava da je kompletan."""
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/tangsanli5201/DeepPCB"
DEST = Path("data/DeepPCB")


def main():
    if DEST.exists():
        print(f"{DEST} već postoji, preskačem kloniranje.")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, str(DEST)], check=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.deeppcb import read_split

    total = 0
    for split, expected in [("trainval", 1000), ("test", 500)]:
        items = read_split(split, DEST)
        missing = [i for i in items if not (i["test"].exists() and i["temp"].exists() and i["ann"].exists())]
        print(f"{split}: {len(items)} slika (očekivano {expected}), nedostaje {len(missing)}")
        total += len(missing)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
