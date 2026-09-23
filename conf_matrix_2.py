import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
#%%

# Import your model
try:
    from engine import ShuffleEfficientViT
except ImportError:
    print("Error: Make sure 'engine.py' is in the same directory.")
    exit()
#%%

# --- CONFIGURATION ---
DATA_DIR = 'dataset_cropped_split/test' 
# MODEL_PATH = 'best_model_211225.pth'
MODEL_PATH = 'best_model_080126.pth'
# MODEL_PATH = 'best_model.pth'
BATCH_SIZE = 32
NUM_CLASSES = 6
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%

# Same transform as training
test_transform = transforms.Compose([
    transforms.Resize( (224, 224) ),
    transforms.ToTensor(),
    transforms.Normalize( mean =[ 0.485, 0.456, 0.406 ], 
                            std = [0.229, 0.224, 0.225 ] )
])
#%%

def plot_confusion_matrix( model, 
                            dataloader, 
                            class_names, 
                            device ):
    """ Plots the confusion matrix for the given model and dataloader. 
    Args:
        model: Trained model.
        dataloader: DataLoader for the test dataset.
        class_names: List of class names.
        device: Computation device (CPU/GPU).
    """

    # Set model to evaluation mode
    model.eval()


    all_preds = []
    all_labels = []

    print("Evaluating model on test set...")
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, labels in dataloader:

            # Move data to device
            inputs = inputs.to( device )
    
            # Forward pass
            outputs = model( inputs )
            # Get predictions
            _, preds = torch.max( outputs, 1 )
            
            # Store predictions and labels
            all_preds.extend( preds.cpu().numpy() )
            all_labels.extend( labels.numpy() )

    # 1. Confusion Matrix
    cm = confusion_matrix( all_labels, all_preds, normalize='true' )
    

    # 2. Classification Report
    print("\n--- Classification Report ---")
    print( classification_report( all_labels, all_preds, target_names = class_names ) )

    # 3. Plot Heatmap
    plt.figure( figsize = ( 10, 8 ) )
    sns.heatmap( cm, 
                annot = True, 
                fmt = '.2f', 
                cmap = 'Blues',
                xticklabels = class_names, 
                yticklabels = class_names,
                vmin = 0.0, 
                vmax = 1.0 )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
#%%
if __name__ == "__main__":
    # Load Dataset
    if not os.path.exists( DATA_DIR ):
        print(f"Error: Directory {DATA_DIR} not found.")
        exit()

    test_dataset = datasets.ImageFolder( root = DATA_DIR, transform = test_transform )
    test_loader = DataLoader( test_dataset, batch_size = BATCH_SIZE, shuffle = False )
    class_names = test_dataset.classes
    
    print(f"Classes found: {class_names}")

    # Load Model
    model = ShuffleEfficientViT( num_classes = NUM_CLASSES, 
                                    head_type ='conv' )
    model.load_state_dict(torch.load( MODEL_PATH, 
                                        map_location = DEVICE ) )
    model.to( DEVICE )

    # Run Analysis
    plot_confusion_matrix( model, 
                            test_loader, 
                            class_names, 
                            DEVICE )