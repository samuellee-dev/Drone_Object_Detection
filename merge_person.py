"""
VisDrone 라벨 통합 스크립트
  pedestrian(0) + people(1) -> person(0)
  이후 클래스 인덱스를 1씩 당김 (2..9 -> 1..8)

사용법:
    python merge_person.py datasets/VisDrone
    python merge_person.py datasets/VisDrone --restore   # 원복
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

# 원본 10개 -> 통합 후 9개
OLD_NAMES = ["pedestrian", "people", "bicycle", "car", "van",
             "truck", "tricycle", "awning-tricycle", "bus", "motor"]
NEW_NAMES = ["person", "bicycle", "car", "van",
             "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def remap(cls_id: int) -> int:
    if cls_id in (0, 1):
        return 0
    return cls_id - 1


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
    before, after = Counter(), Counter()
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
                    print(f"    [경고] 범위 밖 클래스 {old} in {txt.name}")
                    continue
                before[old] += 1
                new = remap(old)
                after[new] += 1
                out_lines.append(" ".join([str(new)] + parts[1:]))
            txt.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            n_files += 1

    print(f"\n총 {n_files}개 라벨 파일 변환 완료\n")
    print("[변환 전]")
    for i, name in enumerate(OLD_NAMES):
        print(f"  {i} {name:18s} {before[i]:>7,}")
    print("\n[변환 후]")
    for i, name in enumerate(NEW_NAMES):
        print(f"  {i} {name:18s} {after[i]:>7,}")


def detect_splits(root: Path):
    """train/val/test 이미지 경로를 폴더 구조에서 자동 감지."""
    splits = {}
    img_root = root / "images"

    if img_root.is_dir():
        # 구조 A:  VisDrone/images/train,  VisDrone/labels/train
        candidates = [d for d in img_root.iterdir() if d.is_dir()]
        base = "images"
    else:
        # 구조 B:  VisDrone/VisDrone2019-DET-train/images
        candidates = [d / "images" for d in root.iterdir()
                      if d.is_dir() and (d / "images").is_dir()]
        base = None

    for d in candidates:
        key = d.name if base else d.parent.name
        low = key.lower()
        rel = f"{base}/{d.name}" if base else f"{d.parent.name}/images"
        if "train" in low:
            splits["train"] = rel
        elif "val" in low:
            splits["val"] = rel
        elif "test" in low:
            splits["test"] = rel
    return splits


def write_yaml(root: Path):
    out = root.parent / "VisDrone_person.yaml"
    splits = detect_splits(root)
    if "train" not in splits or "val" not in splits:
        print("\n[경고] train/val 이미지 폴더를 자동 감지하지 못했습니다.")
        splits.setdefault("train", "images/train")
        splits.setdefault("val", "images/val")

    lines = [f"path: {root}"]
    for k in ("train", "val", "test"):
        if k in splits:
            lines.append(f"{k}: {splits[k]}")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NEW_NAMES))
    out.write_text("\n".join(lines) + f"\n\nnames:\n{names}\n", encoding="utf-8")

    print(f"\nyaml 생성: {out}")
    for k in ("train", "val", "test"):
        if k in splits:
            print(f"  {k}: {splits[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="VisDrone 데이터셋 루트 폴더")
    ap.add_argument("--restore", action="store_true", help="백업에서 원복")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"[에러] 경로 없음: {root}")
        sys.exit(1)

    label_dirs = find_label_dirs(root)
    print(f"대상 labels 폴더 {len(label_dirs)}개\n")

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