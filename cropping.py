import os
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# Configuration
SOURCE_DIR = 'KMU-FED'      
DEST_DIR = 'KMU-FED_Cropped' 
IMG_SIZE = 224

# Emotion Mapping
EMOTION_MAP = {'AN': 0, 'DI': 1, 'FE': 2, 'HA': 3, 'SA': 4, 'SU': 5}


def crop_dataset():
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f'Running on device: {device}')

    mtcnn = MTCNN( image_size = IMG_SIZE, 
                    margin = 0, 
                    device = device, 
                    post_process = False )
    
    if not os.path.exists( DEST_DIR ):
        os.makedirs( DEST_DIR )

    count = 0

    # Iterate through files
    for filename in tqdm( os.listdir( SOURCE_DIR ) ):
        if not filename.lower().endswith( ('.jpg', '.jpeg', '.png') ):
            continue
            

        try:
            parts = filename.split('_')
            if parts[1] not in EMOTION_MAP: continue
        except: continue

        # Load and Crop
        img_path = os.path.join( SOURCE_DIR, filename )
        save_path = os.path.join( DEST_DIR, filename )
        
        try:
            img = Image.open( img_path ).convert('RGB')
            
            mtcnn( img, save_path = save_path ) 
            
            if os.path.exists( save_path ):
                count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Done. Processed {count} images.")

if __name__ == "__main__":
    crop_dataset()