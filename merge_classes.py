"""
VisDrone 라벨 통합 스크립트

통합 규칙:
  pedestrian(0) + people(1) -> person(0)
  bicycle(2)                  -> bicycle(1)
  car(3) + van(4)             -> car(2)
  truck(5)                    -> truck(3)
  tricycle(6) + awning-tricycle(7) -> tricycle(4)
  bus(8)                      -> bus(5)
  motor(9)                    -> motor(6)

최종 7개 클래스:
  0 person
  1 bicycle
  2 car
  3 truck
  4 tricycle
  5 bus
  6 motor

사용법:
    python merge_classes.py datasets/VisDrone
    python merge_classes.py datasets/VisDrone --restore
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path


# 원본 10개 클래스
OLD_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# 통합 후 7개 클래스
NEW_NAMES = [
    "person",
    "bicycle",
    "car",
    "truck",
    "tricycle",
    "bus",
    "motor",
]


# 원본 class id -> 새로운 class id
CLASS_MAP = {
    0: 0,  # pedestrian      -> person
    1: 0,  # people          -> person

    2: 1,  # bicycle         -> bicycle

    3: 2,  # car             -> car
    4: 2,  # van             -> car

    5: 3,  # truck           -> truck

    6: 4,  # tricycle        -> tricycle
    7: 4,  # awning-tricycle -> tricycle

    8: 5,  # bus             -> bus
    9: 6,  # motor           -> motor
}


def remap(cls_id: int) -> int:
    return CLASS_MAP[cls_id]


def find_label_dirs(root: Path):
    dirs = [d for d in root.rglob("labels") if d.is_dir()]

    if not dirs:
        print(f"[에러] {root} 아래에서 labels 폴더를 찾지 못했습니다.")
        sys.exit(1)

    return sorted(dirs)


def backup_dir(d: Path) -> Path:
    return d.with_name(d.name + "_orig")


def do_backup(label_dirs):
    for d in label_dirs:
        bak = backup_dir(d)

        if bak.exists():
            print(f"  백업 이미 존재, 건너뜀: {bak}")
        else:
            shutil.copytree(d, bak)
            print(f"  백업 생성: {bak}")


def do_restore(label_dirs):
    for d in label_dirs:
        bak = backup_dir(d)

        if not bak.exists():
            print(f"  백업 없음, 건너뜀: {d}")
            continue

        shutil.rmtree(d)
        shutil.copytree(bak, d)

        print(f"  원복 완료: {d}")


def convert(label_dirs):
    before = Counter()
    after = Counter()

    n_files = 0

    for d in label_dirs:
        txts = sorted(d.rglob("*.txt"))

        print(f"  {d}  ({len(txts)}개 파일)")

        for txt in txts:
            out_lines = []

            for line in txt.read_text(encoding="utf-8").splitlines():

                parts = line.split()

                if len(parts) < 5:
                    continue

                old = int(float(parts[0]))

                if not 0 <= old < len(OLD_NAMES):
                    print(
                        f"    [경고] 범위 밖 클래스 {old} "
                        f"in {txt.name}"
                    )
                    continue

                before[old] += 1

                new = remap(old)

                after[new] += 1

                out_lines.append(
                    " ".join([str(new)] + parts[1:])
                )

            txt.write_text(
                "\n".join(out_lines) + "\n",
                encoding="utf-8",
            )

            n_files += 1

    print(f"\n총 {n_files}개 라벨 파일 변환 완료\n")

    print("[변환 전]")

    for i, name in enumerate(OLD_NAMES):
        print(
            f"  {i} {name:18s} "
            f"{before[i]:>7,}"
        )

    print("\n[변환 후]")

    for i, name in enumerate(NEW_NAMES):
        print(
            f"  {i} {name:18s} "
            f"{after[i]:>7,}"
        )


def detect_splits(root: Path):
    """
    train / val / test 이미지 경로를
    폴더 구조에서 자동 감지.
    """

    splits = {}

    img_root = root / "images"

    if img_root.is_dir():

        # 구조 A:
        # VisDrone/images/train
        # VisDrone/labels/train

        candidates = [
            d for d in img_root.iterdir()
            if d.is_dir()
        ]

        base = "images"

    else:

        # 구조 B:
        # VisDrone/
        #   VisDrone2019-DET-train/images
        #   VisDrone2019-DET-val/images

        candidates = [
            d / "images"
            for d in root.iterdir()
            if d.is_dir()
            and (d / "images").is_dir()
        ]

        base = None

    for d in candidates:

        key = (
            d.name
            if base
            else d.parent.name
        )

        low = key.lower()

        rel = (
            f"{base}/{d.name}"
            if base
            else f"{d.parent.name}/images"
        )

        if "train" in low:
            splits["train"] = rel

        elif "val" in low:
            splits["val"] = rel

        elif "test" in low:
            splits["test"] = rel

    return splits


def write_yaml(root: Path):

    out = root.parent / "VisDrone_merged.yaml"

    splits = detect_splits(root)

    if "train" not in splits or "val" not in splits:

        print(
            "\n[경고] train/val 이미지 폴더를 "
            "자동 감지하지 못했습니다."
        )

        splits.setdefault(
            "train",
            "images/train",
        )

        splits.setdefault(
            "val",
            "images/val",
        )

    lines = [
        f"path: {root}"
    ]

    for k in ("train", "val", "test"):

        if k in splits:
            lines.append(
                f"{k}: {splits[k]}"
            )

    names = "\n".join(
        f"  {i}: {n}"
        for i, n in enumerate(NEW_NAMES)
    )

    out.write_text(
        "\n".join(lines)
        + f"\n\nnames:\n{names}\n",
        encoding="utf-8",
    )

    print(f"\nyaml 생성: {out}")

    for k in ("train", "val", "test"):

        if k in splits:
            print(
                f"  {k}: {splits[k]}"
            )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "root",
        type=Path,
        help="VisDrone 데이터셋 루트 폴더",
    )

    ap.add_argument(
        "--restore",
        action="store_true",
        help="백업에서 원복",
    )

    args = ap.parse_args()

    root = args.root.resolve()

    if not root.exists():

        print(
            f"[에러] 경로 없음: {root}"
        )

        sys.exit(1)

    label_dirs = find_label_dirs(root)

    print(
        f"대상 labels 폴더 "
        f"{len(label_dirs)}개\n"
    )

    if args.restore:

        do_restore(label_dirs)

        return

    print("1) 백업")

    do_backup(label_dirs)

    print("\n2) 변환")

    convert(label_dirs)

    write_yaml(root)


if __name__ == "__main__":
    main()