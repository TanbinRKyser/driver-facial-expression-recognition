import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def walk_throught_dir( base_path ):
    image_paths = []

    for root, dirs, files in os.walk( base_path ):
        for file in files:
            if file.lower().endswith( (".jpg", ".png", ".jpeg") ):
                full_path = os.path.join( root, file )
                image_paths.append( full_path )

    return image_paths

# if __name__ == "__main__":
#     base_path = r"/mnt/c/Taskerr/Practice/Python/ShuffleVit/dataset_split"
#     all_image_paths = walk_throught_dir( base_path )

#     print("\n===== EXPLORATION RESULTS =====\n")
#     print(f"Total images found: {len( all_image_paths )}\n")
#     print("Sample image paths:")
#     for img_path in all_image_paths[:10]:
#         print(img_path)

## setup test train split
def setup_train_test_split( base_path, train_ratio=0.8 ):
    image_paths = walk_throught_dir( base_path )
    total_images = len( image_paths )
    train_size = int( total_images * train_ratio )

    train_images = image_paths[:train_size]
    test_images = image_paths[train_size:]

    return train_images, test_images

# if __name__ == "__main__":
#     base_path = r"/mnt/c/Taskerr/Practice/Python/ShuffleVit/dataset_split"
#     train_images, test_images = setup_train_test_split( base_path )

#     print("\n===== TRAIN-TEST SPLIT RESULTS =====\n")
#     print(f"Total images found: {len( train_images ) + len( test_images )}")
#     print(f"Training images: {len( train_images )}")
#     print(f"Testing images: {len( test_images )}\n")

#     print("Sample training image paths:")
#     for img_path in train_images[:5]:
#         print(img_path)

#     print("\nSample testing image paths:")
#     for img_path in test_images[:5]:
#         print(img_path)

random.seed( 42 )

image_path = Path( r"dataset_cropped_split" )


image_path_list = list ( image_path.rglob( "*.jpg" ) )
# image_path_list = list( walk_throught_dir( r"/mnt/c/Taskerr/Practice/Python/ShuffleVit/dataset_split" ) )
# print( f"Total images found: {len( image_path_list )}" )
random_image_path = random.choice( image_path_list )


image_class = random_image_path.parent.stem


img = Image.open( random_image_path )

print( f"Random image path: {random_image_path}" )
print( f"Image size: {img.size}, Image mode: {img.mode}" )
print( f"Image format: {img.format}" )
# img.show()

### visualizing image with matplotlib
image_array = np.asarray( img )

print( f"Image array shape: {image_array.shape}, Image array dtype: {image_array.dtype}" )

plt.figure( figsize=(10, 6) )
plt.imshow( image_array )
plt.axis( 'off' )
plt.title( f"Image shape: {image_array.shape}, Image class: {image_class} " )
plt.axis( 'off' )
# plt.show()


## image statistics
# mean_per_channel = np.mean( image_array, axis=(0, 1) )
# std_per_channel = np.std( image_array, axis=(0, 1) )
# print( f"Mean per channel (R, G, B): {mean_per_channel}" )
# print( f"Stddev per channel (R, G, B): {std_per_channel}" )

transformed_image = transforms.Resize( (224, 224) )( img )

transformed_image_tensor = transforms.ToTensor()( transformed_image )

print( f"Transformed image tensor shape: {transformed_image_tensor.shape}, dtype: {transformed_image_tensor.dtype}" )   

# Calculate mean and stddev for the transformed image tensor
red_std = torch.std( transformed_image_tensor[ 0, :, : ] )
green_std = torch.std( transformed_image_tensor[ 1, :, : ] )    
blue_std = torch.std( transformed_image_tensor[ 2, :, : ] ) 

print( f"Stddev per channel (R, G, B): {red_std}, {green_std}, {blue_std}" )

# calculate mean for each channel
red_mean = torch.mean( transformed_image_tensor[ 0, :, : ] )
green_mean = torch.mean( transformed_image_tensor[ 1, :, : ] )
blue_mean = torch.mean( transformed_image_tensor[ 2, :, : ] )

print( f"Mean per channel (R, G, B): {red_mean}, {green_mean}, {blue_mean}" )