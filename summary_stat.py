import os

base_path = r"dataset_cropped_split"
splits = ["train", "test"]

print("\n===== SUMMARY STATISTICS =====\n")

for split in splits:
    split_path = os.path.join( base_path, split )
    if not os.path.isdir( split_path ):
        print(f"Skipping missing folder: {split_path}")
        continue

    print(f"\n--- {split.upper()} ---")

    for class_name in os.listdir( split_path ):
        class_path = os.path.join( split_path, class_name )
        if os.path.isdir( class_path ):
            count = sum(
                1 for f in os.listdir( class_path )
                if f.lower().endswith( (".jpg", ".png", ".jpeg") )
            )
            print(f"{class_name:<12} : {count} images")