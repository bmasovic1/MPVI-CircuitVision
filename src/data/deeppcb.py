"""Učitavanje DeepPCB dataseta: parovi slika (test/template) i anotacije."""
from pathlib import Path

NAMES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}
DEFAULT_ROOT = Path("data/DeepPCB")


def read_split(split, root=DEFAULT_ROOT):
    """Vraća listu zapisa za 'trainval' ili 'test'.

    Svaki zapis: dict(test, temp, ann, group) gdje su test/temp/ann putanje (Path),
    a group naziv ploče-grupe (npr. 'group20085').
    Split fajlovi pokazuju na NNN.jpg, a stvarni fajlovi su NNN_test.jpg i NNN_temp.jpg.
    """
    data = Path(root) / "PCBData"
    items = []
    for line in (data / f"{split}.txt").read_text().splitlines():
        if not line.strip():
            continue
        img, ann = line.split()
        img = Path(img)
        folder = data / img.parent
        items.append({
            "test": folder / f"{img.stem}_test.jpg",
            "temp": folder / f"{img.stem}_temp.jpg",
            "ann": data / ann,
            "group": img.parts[0],
        })
    return items


def read_ann(path):
    """Čita anotacije: lista (x1, y1, x2, y2, tip). Format je odvojen razmakom, ne zarezom."""
    return [tuple(map(int, ln.split())) for ln in Path(path).read_text().splitlines() if ln.strip()]


def split_by_group(items, val_fraction=0.2, seed=42):
    """Dijeli zapise na trening/validaciju PO GRUPAMA (ista ploča se ne smije naći u oba skupa)."""
    import random
    groups = sorted({i["group"] for i in items})
    rng = random.Random(seed)
    rng.shuffle(groups)
    target = val_fraction * len(items)
    val_groups, count = set(), 0
    for g in groups:
        if count >= target:
            break
        n = sum(1 for i in items if i["group"] == g)
        val_groups.add(g)
        count += n
    train = [i for i in items if i["group"] not in val_groups]
    val = [i for i in items if i["group"] in val_groups]
    return train, val


def draw_boxes(item, ax=None):
    """Crta test sliku s boxovima i nazivima klasa (za provjeru parsera)."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    img = Image.open(item["test"]).convert("RGB")
    d = ImageDraw.Draw(img)
    for x1, y1, x2, y2, t in read_ann(item["ann"]):
        d.rectangle([x1, y1, x2, y2], outline="red", width=2)
        d.text((x1, max(0, y1 - 12)), NAMES[t], fill="yellow")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    return ax
