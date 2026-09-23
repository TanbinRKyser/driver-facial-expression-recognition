import os
import shutil
import random

SOURCE_DIR = "KMU-FED_Cropped"


DEST_DIR = "dataset_cropped_split"

# Emotion label mappings
LABEL_MAP = {
    "AN": "anger",
    "DI": "disgust",
    "FE": "fear",
    "HA": "happiness",
    "SA": "sadness",
    "SU": "surprise",
}

TRAIN_RATIO = 0.8

# Create folder structure
for split in ["train", "test"]:
    for label in LABEL_MAP.values():
        os.makedirs( os.path.join( DEST_DIR, split, label ), exist_ok = True )

# Read all image files
files = [f for f in os.listdir( SOURCE_DIR ) if f.lower().endswith(".jpg")]

# Group by class
class_groups = {k: [] for k in LABEL_MAP.keys()}

for f in files:
    parts = f.split("_")
    if len( parts ) < 2:
        continue
    
    class_code = parts[ 1 ] 
    
    if class_code in LABEL_MAP:
        class_groups[ class_code ].append( f )

# Shuffle and split
for code, file_list in class_groups.items():
    random.shuffle( file_list )
    split_idx = int( len( file_list ) * TRAIN_RATIO )

    train_files = file_list[ : split_idx ]
    test_files = file_list[ split_idx : ]

    for f in train_files:
        src = os.path.join( SOURCE_DIR, f )
        dst = os.path.join( DEST_DIR, "train", LABEL_MAP[ code ], f )
        shutil.copy( src, dst )


    for f in test_files:
        src = os.path.join( SOURCE_DIR, f )
        dst = os.path.join( DEST_DIR, "test", LABEL_MAP[ code ], f )
        shutil.copy( src, dst )

print("dataset split completed")

