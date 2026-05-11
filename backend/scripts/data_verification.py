import os
import PIL.Image

def verify_and_count(data_path):
    print(f"Checking: {data_path}")
    if not os.path.exists(data_path):
        print("Error: Path does not exist.")
        return

    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Found {len(classes)} classes.\n")

    summary = []
    for cls in sorted(classes):
        cls_path = os.path.join(data_path, cls)
        files = os.listdir(cls_path)
        img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        corrupted = 0
        for f in img_files:
            try:
                with PIL.Image.open(os.path.join(cls_path, f)) as img:
                    img.verify()
            except Exception:
                corrupted += 1

        summary.append({
            "Class": cls,
            "Total": len(img_files),
            "Corrupted": corrupted
        })

    print(f"{'Class':<15} | {'Total':<10} | {'Corrupted':<10}")
    print("-" * 40)
    for row in summary:
        print(f"{row['Class']:<15} | {row['Total']:<10} | {row['Corrupted']:<10}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/digits"
    verify_and_count(path)
