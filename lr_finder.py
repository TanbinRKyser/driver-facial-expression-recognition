import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

#%%
# Import your model and data loader
try:
    from engine import ShuffleEfficientViT
    from create_dataloaders_2 import main as get_dataloaders
except ImportError:
    print("Error: Ensure 'engine.py' and 'create_dataloaders_2.py' are in the directory.")
    exit()
#%%

# --- CONFIGURATION ---
START_LR = 1e-7 # Start very low
END_LR = 10.0 # End high to find max LR
NUM_ITERATIONS = 100 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%

class LRFinder:

    def __init__( self, 
                    model, 
                    optimizer, 
                    criterion, 
                    device ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lrs = []
        self.losses = []

    def range_test( self, 
                    dataloader, 
                    start_lr, 
                    end_lr, 
                    num_iter):
        """Performs the learning rate range test.
        Args:
            dataloader: DataLoader for training data.
            start_lr (float): Starting learning rate.
            end_lr (float): Ending learning rate.
            num_iter (int): Number of iterations over which to test.
        """

        # Set model to training mode    
        self.model.train()
        
        # Calculate the multiplier to increase LR exponentially
        gamma = ( end_lr / start_lr ) ** ( 1 / num_iter )
        optimizer = self.optimizer
        
        # Initialize LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
        
        current_lr = start_lr
        best_loss = float('inf')
        
        iterator = iter( dataloader ) 
        
        print(f"Running LR Range Test from {start_lr} to {end_lr}...")
        
        for i in tqdm( range( num_iter ) ):
            try:
                inputs, labels = next( iterator )
            except StopIteration:
                iterator = iter( dataloader )
                inputs, labels = next( iterator )

            inputs, labels = inputs.to( self.device ), labels.to( self.device )
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model( inputs )
            loss = self.criterion( outputs, labels )
            
            # Record best loss to know when to stop if it explodes
            if loss.item() < best_loss:
                best_loss = loss.item()
            
            # Stop if loss explodes. 4x times the best and loss explodes.
            if loss.item() > 4 * best_loss and i > 10:
                print(f"Loss exploded at step {i}. Stopping early.")
                break
                
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record Data
            self.lrs.append( current_lr )
            self.losses.append( loss.item() )
            
            # Update LR for next step
            current_lr *= gamma
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    def plot( self, 
                filename = 'lr_finder_plot.png' ):
        """Plots the Loss vs Learning Rate
            param filename: File name to save the plot."""

        plt.figure( figsize = ( 10, 5 ) )
        plt.plot( self.lrs, self.losses )
        plt.xscale('log') 
        plt.xlabel('Learning Rate (Log Scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid( True, which = "both", ls = "-" )
        
        # Save and Show
        plt.savefig( filename )
        print(f"Plot saved to {filename}")
        plt.show()
#%%


def main():
    # 1. Setup Data & Model
    train_loader, _, _ = get_dataloaders()
    model = ShuffleEfficientViT( num_classes = 6, head_type = 'conv' ) 
    model.to( DEVICE )
    
    # 2. Setup Optimizer & Criterion
    # We test the HEAD first only, so all params are trainable
    optimizer = optim.AdamW( model.parameters(), lr = START_LR )
    criterion = nn.CrossEntropyLoss()
    
    # 3. Run Finder
    lr_finder = LRFinder( model, optimizer, criterion, DEVICE )
    lr_finder.range_test( train_loader, START_LR, END_LR, NUM_ITERATIONS )
    
    # 4. Generate Plot
    lr_finder.plot()
#%%
if __name__ == "__main__":
    main()
