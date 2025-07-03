import os

yolo_label_dir = r"E:\dev\pi_vision\manatee\yolo11_dataset\labels"

for root, dirs, files in os.walk(yolo_label_dir):
    for file in files:
        if file.endswith(".txt"):
            label_path = os.path.join(root, file)
            lines = open(label_path).readlines()
            with open(label_path, 'w') as out:
                for line in lines:
                    parts = line.strip().split()
                    parts[0] = "60"  # Replace class_id with 60
                    out.write(" ".join(parts) + "\n")
