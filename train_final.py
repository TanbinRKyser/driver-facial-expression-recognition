import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import Counter
#%%

# --- IMPORTS FROM YOUR FILES ---
try:
    from engine import ShuffleEfficientViT
    # from create_dataloaders import main as get_dataloaders
    from create_dataloaders_2 import main as get_dataloaders
except ImportError:
    print("Error: Make sure 'engine.py' and 'create_dataloaders.py' are in the same directory.")
    exit()

#%%
# --- CONFIGURATION ---
## based on LR finder results 5e-7 to 1e-2
PHASE1_LR = 1e-3         # Initial fast learning for head
PHASE2_BACKBONE_LR = 5e-5 # Slow fine-tuning for backbones
PHASE2_HEAD_LR = 5e-4     # Medium fine-tuning for head
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 50 
NUM_CLASSES = 6
DEVICE = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )

#%%

# --- UTILS ---
def set_seed( seed = 42 ):
    """Locks all random seeds for reproducible results."""
    random.seed( seed )
    np.random.seed( seed )
    torch.manual_seed( seed )
    torch.cuda.manual_seed_all( seed )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str( seed )
    print(f"Random seed set to {seed}")

#%%
# --- WEIGHT CALCULATION ---
def calculate_class_weights( dataset, device ):
    """Calculates inverse class weights to handle imbalance."""
    print("Calculating class weights...")

    all_labels = [ label for _, label in dataset ]
    count_dict = Counter( all_labels )
    total_samples = len( all_labels )
    weights = []

    for i in range( NUM_CLASSES ):
        count = count_dict[ i ]
        weight = total_samples / ( NUM_CLASSES * count )
        weights.append( weight )
    
    return torch.tensor( weights, dtype = torch.float ).to( device )
#%%

# --- PLOTTING ---
def plot_training_history( train_losses, val_losses, val_accs ):
    """Generates and saves the loss/accuracy plot."""


    epochs = range( 1, len( train_losses ) + 1 )

    plt.figure( figsize = ( 12, 5 ) )

    # Plot Loss
    # plt.subplot( 1, 2, 1 )
    plt.plot( epochs, train_losses, 'b-', label = 'Training Loss' )
    plt.plot( epochs, val_losses, 'orange', label = 'Validation Loss' )
    
    plt.axvline( x = PHASE1_EPOCHS, color = 'r', linestyle = '--', alpha = 0.5, label = 'Phase 2 Start' )

    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid( True )
    plt.legend()
    

    # Plot Accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot( epochs, val_accs, 'g-', label = 'Validation Accuracy (%)' )

    # plt.axvline( x = PHASE1_EPOCHS, color = 'r', linestyle = '--', alpha = 0.5, label = 'Phase 2 Start')
    
    # plt.title('Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy (%)')
    # plt.grid( True )
    # plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\nSaved as 'training_results.png'")
    plt.show()

#%%

# --- LOOPS ---
def train_one_epoch( model, 
                        dataloader, 
                        criterion, 
                        optimizer, 
                        device, 
                        epoch_idx ):
    
    # training mode
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm( dataloader, 
                    desc = f"Epoch {epoch_idx} (Train)", 
                    leave = False )
    
    # Training Loop
    for inputs, labels in pbar:

        # Move to device
        inputs, labels = inputs.to( device ), labels.to( device )
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model( inputs )

        # calculate loss
        loss = criterion( outputs, labels )
        
        # Backward pass and optimization
        loss.backward()
        # Gradient Clipping to prevent instability
        torch.nn.utils.clip_grad_norm_( model.parameters(), max_norm = 1.0 )
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size( 0 )
        _, predicted = torch.max( outputs.data, 1 )
        total += labels.size( 0 )
        correct += ( predicted == labels ).sum().item()
        
        pbar.set_postfix( Loss = running_loss / total )
        

    return running_loss / total, correct / total
#%%

# evaluate function
def evaluate_model( model, 
                    dataloader, 
                    criterion, 
                    device ):
    # evalution mode
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move to device
            inputs, labels = inputs.to( device ), labels.to( device )

            # Forward pass
            outputs = model( inputs )

            # 1. Forward Pass
            # outputs1 = model( inputs )
            
            # 2. Forward Pass with Horizontal
            # Flip along width axis (dim 3)
            # inputs_flipped = torch.flip( inputs, dims = [ 3 ] ) 
            # outputs2 = model( inputs_flipped )
            
            # 3. Average Predictions
            # outputs = ( outputs1 + outputs2 ) / 2.0

            # calculate loss 
            loss = criterion( outputs, labels )
            running_loss += loss.item() * inputs.size( 0 )
            
            # Get predictions
            _, predicted = torch.max( outputs.data, 1 )

            total += labels.size( 0 )
            correct += ( predicted == labels ).sum().item()
            

    return ( running_loss / total ), ( correct / total )

#%%
# --- MAIN ---
def main():

    # Set random seed
    set_seed( 42 )
    
    # 1. Load Data
    train_loader, test_loader, class_names = get_dataloaders()

    if train_loader is None: return


    # 2. Setup Weights & Model
    class_weights = calculate_class_weights( train_loader.dataset, DEVICE )

    model = ShuffleEfficientViT( num_classes = NUM_CLASSES, 
                                    freeze_features = True, 
                                    head_type = 'conv' )

    # Move model to device                                    
    model.to( DEVICE )
    
    # 3. Loss Function
    criterion = nn.CrossEntropyLoss( weight = class_weights, 
                                        label_smoothing=0.1 
                                        )
    
    # History storage for plotting
    history_loss = []
    history_val_loss = []
    history_acc = []


    # --- PHASE 1: HEAD TRAINING ---
    print(f"\n[PHASE 1] Training Head ({PHASE1_EPOCHS} Epochs)...")
    optimizer = optim.AdamW( filter( lambda p: p.requires_grad, model.parameters() ), 
                                lr = PHASE1_LR )
    
    for epoch in range( 1, PHASE1_EPOCHS + 1 ):
        # Train & Evaluate
        t_loss, t_acc = train_one_epoch( model, train_loader, criterion, optimizer, DEVICE, epoch)
        v_loss, v_acc = evaluate_model( model, test_loader, criterion, DEVICE)
        
        history_loss.append( t_loss )
        history_val_loss.append( v_loss ) 
        history_acc.append( v_acc * 100 )
        
        print(f"Ep {epoch}: Train Loss {t_loss : .3f} | Train Acc {t_acc * 100 : .2f}% |\
                Val Loss { v_loss : .3f} | Val Acc { v_acc * 100 : .2f}%")


    # --- PHASE 2: DIFFERENTIAL FINE-TUNING ---
    print(f"\n[PHASE 2] Differential Fine-Tuning ({PHASE2_EPOCHS} Epochs)...")
    
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
        
    # Differential Learning Rates
    params = [
        {'params': model.shufflenet_branch.parameters(), 'lr': PHASE2_BACKBONE_LR},
        {'params': model.efficientvit_branch.parameters(), 'lr': PHASE2_BACKBONE_LR},
        {'params': model.classification_head.parameters(), 'lr': PHASE2_HEAD_LR}
    ]

    # Optimizer    
    optimizer = optim.AdamW( params )
    
    # Scheduler for learning rate decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min', 
        factor = 0.1, 
        patience = 3
    )
    

    best_val_acc = 0.0
    

    for epoch in range( PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1 ):
        t_loss, t_acc = train_one_epoch( model, train_loader, criterion, optimizer, DEVICE, epoch )
        v_loss, v_acc = evaluate_model( model, test_loader, criterion, DEVICE )
        
        history_loss.append( t_loss )
        history_val_loss.append( v_loss )
        history_acc.append( v_acc * 100 )
        
        print(f"Ep {epoch}: Train Loss {t_loss : .3f} | Train Acc {t_acc * 100 : .2f}% |\
                Val Loss { v_loss : .3f} | Val Acc { v_acc * 100 : .2f}%")
        
        scheduler.step( v_loss )
        
        # Save Best Model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save( model.state_dict(), 'results/best_model.pth' )
            print(f"  >>> Best Model Saved: {best_val_acc * 100 : .2f}%")


    # --- FINAL PLOTTING ---
    print("\nTraining Finished. Generating Plots...")
    plot_training_history( history_loss, history_val_loss, history_acc )


#%%
if __name__ == "__main__":
    main()