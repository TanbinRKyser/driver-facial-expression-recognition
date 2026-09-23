import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os 



# --- ImageNet Statistics ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



# --- Data Transformations ---
# For Training: Includes augmentation
train_transform = transforms.Compose([
    transforms.Resize( (224, 224) ),
    transforms.Grayscale( num_output_channels = 3 ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation( 40 ),
    transforms.RandomAffine( degrees = 40, scale = (.3, 1.1), shear = 0.15 ),
    transforms.ColorJitter( brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1 ),
    transforms.GaussianBlur( kernel_size = 5 ),
    transforms.ToTensor(),
    transforms.Normalize( mean = IMAGENET_MEAN, std = IMAGENET_STD ),
    # transforms.RandomErasing(p=0.4, scale=(0.02, 0.2))
])

# For Testing/Validation: No augmentation, just resize and normalize
test_transform = transforms.Compose([
    transforms.Resize( (224, 224) ),
    transforms.ToTensor(),
    transforms.Normalize( mean = IMAGENET_MEAN, std = IMAGENET_STD )
])

# For Visualization: Just resize and convert to tensor
base_transform = transforms.Compose([
    transforms.Resize( (224, 224) ),
    transforms.ToTensor()
])



# --- Dataset Loading Function ---
def get_dataset( data_dir, transform = None ):
    """
    Loads an ImageFolder dataset from the specified directory.
    
    Args:
        data_dir (str): Path to the dataset directory (e.g., .../train)
        transform (callable, optional): Transformations to apply. 
                                        Defaults to train_transform if None.
    
    Returns:
        torchvision.datasets.ImageFolder: The loaded dataset.
    """
    if not os.path.isdir( data_dir ):
        print(f"Error: Dataset directory not found: {data_dir}")
        return None

    if transform is None:
        transform = train_transform

    dataset = datasets.ImageFolder( root = data_dir, transform = transform )

    return dataset



# --- Utility Function for Visualization ---
def denormalize_image( tensor ):
    """
    Reverses the Normalization transform for plotting.
    """
    mean = np.array( IMAGENET_MEAN )
    std = np.array( IMAGENET_STD )
    
    img = tensor.clone().detach().cpu().numpy()
    
    # Permute from (C, H, W) -> (H, W, C)
    img = img.transpose( (1, 2, 0) )
    
    # Denormalize: X * Std + Mean
    img = img * std + mean
    
    # Clip values to the valid [0, 1] range
    img = np.clip( img, 0, 1 )
    

    return img



# --- Main block for testing/visualization ---
if __name__ == "__main__":
    
    print("Running data_transformation.py for visualization...")

    # Path to the dataset directory
    data_dir = '/mnt/c/Taskerr/Practice/Python/ShuffleVit/dataset_split'
    train_dir = os.path.join(data_dir, 'train')

    if not os.path.isdir( train_dir ):
        print(f"Test directory not found: {train_dir}")
        print("Please run dataset_preprocessing.py or check the 'data_dir' path.")
    else:
        # Load augmented dataset
        train_dataset_augmented = get_dataset(
            data_dir = train_dir,
            transform = train_transform
        )
        
        # Load base dataset (for comparison)
        train_dataset_base = datasets.ImageFolder(
            root = train_dir, 
            transform = base_transform
        )

        if len( train_dataset_augmented ) == 0:
            print("No images found in the training directory.")
        else:
            print(f'Total training samples: {len( train_dataset_augmented )}')
            print(f'Classes: {train_dataset_augmented.classes}')

            # Visualize the first 3 images
            NUM_SAMPLES = 3
            fig, axes = plt.subplots( NUM_SAMPLES, 2, figsize = ( 8, 4 * NUM_SAMPLES ) )
            fig.suptitle('Image Augmentation Comparison (Before vs. After)', fontsize = 16 )

            for i in range( NUM_SAMPLES ):
                if i >= len( train_dataset_base ):
                    break # Stop if dataset has fewer than NUM_SAMPLES

                # Get base image
                base_tensor, base_label = train_dataset_base[ i ]
                base_image = base_tensor.permute( 1, 2, 0 ).numpy() # (H, W, C)
                
                # Get augmented image
                augmented_tensor, aug_label = train_dataset_augmented[ i ]
                augmented_image = denormalize_image( augmented_tensor )

                class_name = train_dataset_augmented.classes[ aug_label ]

                # Plot base Image
                axes[i, 0].imshow( base_image )
                axes[i, 0].set_title(f'Original (Class: {class_name})', color = 'blue' )
                axes[i, 0].axis('off')

                # Plot augmented Image
                axes[i, 1].imshow( augmented_image )
                axes[i, 1].set_title(f'Augmented (Class: {class_name})', color='red')
                axes[i, 1].axis('off')

            plt.tight_layout( rect = [0, 0.03, 1, 0.95] )
            print("Showing augmentation plot...")
            plt.show()