import torch
from torch.utils.data import DataLoader
import os

# Import dataset and transforms from data_transformation.py
try:
    # from data_transformation import get_dataset, train_transform, test_transform
    from data_transformation_2 import get_dataset, train_transform, test_transform
except ImportError:
    print("Error: Make sure 'data_transformation.py' is in the same directory.")
    exit()

# --- Configuration ---

DATA_DIR = 'dataset_cropped_split' 
BATCH_SIZE = 32 
NUM_WORKERS = 4 

# --- End Configuration ---

def main():
    """
    Loads datasets and creates PyTorch DataLoaders.
    """
    
    train_dir = os.path.join( DATA_DIR, 'train' )
    test_dir = os.path.join( DATA_DIR, 'test' )

    # Check if paths exist
    if not os.path.isdir( train_dir ) or not os.path.isdir( test_dir ):
        print(f"Error: Train ({train_dir}) or Test ({test_dir}) directory not found.")
        print("Please run 'dataset_preprocessing.py' first.")
        return None, None

    # --- 1. Load Datasets ---
    print("Loading datasets...")

    # Load the training dataset using the augmented transform
    train_dataset = get_dataset(
        data_dir = train_dir,
        transform = train_transform
    )

    # Load the testing dataset using the non-augmented test transform
    test_dataset = get_dataset(
        data_dir = test_dir,
        transform = test_transform
    )

    if train_dataset is None or test_dataset is None:
        print("Failed to load datasets. Exiting.")
        return None, None
        
    if len( train_dataset ) == 0 or len( test_dataset ) == 0:
        print("Error: One of the datasets is empty. Check your data directories.")
        return None, None

    print(f"Total training images: {len( train_dataset )}")
    print(f"Total testing images: {len( test_dataset )}")
    print(f"Classes: {train_dataset.classes}")

    # --- 2. Create DataLoaders ---
    print(f"Creating DataLoaders with batch size {BATCH_SIZE}...")

    # Training DataLoader
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True, # Shuffle for training
        num_workers = NUM_WORKERS,
        pin_memory = True # Speeds up CPU to GPU data transfer
    )

    # Testing DataLoader
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False, 
        num_workers = NUM_WORKERS,
        pin_memory = True
    )

    print("DataLoaders created successfully.")
    return train_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    
    # Load the data
    train_loader, test_loader, class_names = main()
    
    # --- 3. Example: Check one batch ---
    if train_loader and test_loader:
        try:
            print("\nChecking the first batch from train_loader:")
            
            # Get one batch of training data
            images, labels = next( iter( train_loader ) )
            
            print(f"  Images batch shape: {images.shape}") # [Batch, Channels, H, W]
            print(f"  Labels batch shape: {labels.shape}") # [Batch]
            
            # Show the class name for the first label in the batch
            first_label_index = labels[0].item()
            print(f"  Label for first image in batch: {labels[ 0 ]} \
                        (Class: {class_names[ first_label_index ]})")

        except Exception as e:
            print(f"\nCould not iterate over DataLoader. Error: {e}")
            print("Dataset is empty or paths are wrong.")