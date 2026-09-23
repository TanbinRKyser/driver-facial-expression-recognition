import torch
import torch.nn as nn
import torchvision.models as models
import timm 

class ShuffleEfficientViT( nn.Module ):
    """
    A dual-branch model that fuses features from ShuffleNet V2 and EfficientViT-M2.
    
    1. ShuffleNet Branch -> Extracts convolutional features (1024 dims)
    2. EfficientViT Branch -> Extracts self-attention features (224 dims)
    3. Fusion -> Concatenates features (1248 dims)
    4. Classifier -> MLP head for final classification
    """
    def __init__( self, num_classes = 6, freeze_features = True ):
        
        super( ShuffleEfficientViT, self ).__init__()
        
        # --- 1. Feature Extraction Branches ---
        # Branch 1: ShuffleNet( Output: 1024 dims )
        self.shufflenet_branch = models.shufflenet_v2_x1_0(
            weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        )
        
        shufflenet_features = self.shufflenet_branch.fc.in_features # 1024
        self.shufflenet_branch.fc = nn.Identity()
        
        # Calculate total parameters in ShuffleNet branch
        shuff_params = sum( p.numel() for p in self.shufflenet_branch.parameters() )
        print(f"ShuffleNet branch parameters: {shuff_params: ,}")



        # Branch 2: EfficientViT-M2 (Output: 224 dims)
        try:
            self.efficientvit_branch = timm.create_model(
                'efficientvit_m2',
                pretrained = True,
                num_classes = 0 
            )

            # The EfficientViT-M2 architecture has a feature embedding of 224
            efficientvit_features = self.efficientvit_branch.num_features 
            
            effcientvit_params = sum( p.numel() for p in self.efficientvit_branch.parameters() )
            print(f"EfficientViT-M2 branch parameters: {effcientvit_params:,}")
        except Exception as e:
            print(f"Error loading EfficientViT model from timm: {e}")
            print("Please make sure 'timm' is installed (pip install timm) and you have an internet connection.")
            raise e

        # Freeze the feature backbones if specified
        if freeze_features:
            print("Freezing feature extraction branches...")
            for param in self.shufflenet_branch.parameters():
                param.requires_grad = False
            for param in self.efficientvit_branch.parameters():
                param.requires_grad = False
            print("Feature branches frozen.")


        # --- 2. Feature Fusion ---
        combined_features = shufflenet_features + efficientvit_features # 1024 + 224 = 1248

        
        # --- 3. Classification Head ---
        self.classification_head = nn.Sequential(
            # Layer 1: nn.Linear(1248, 512) -> nn.BatchNorm1d(512) -> nn.ReLU() -> nn.Dropout(p=0.5)
            nn.Linear( combined_features, 512 ),
            nn.BatchNorm1d( 512 ),
            nn.ReLU(),
            nn.Dropout( p = 0.5 ),

            # Layer 2: nn.Linear(512, 256) -> nn.BatchNorm1d(256) -> nn.ReLU() -> nn.Dropout(p=0.5)
            nn.Linear( 512, 256 ),
            nn.BatchNorm1d( 256 ),
            nn.ReLU(),
            nn.Dropout( p = 0.5 ),

            # Output Layer: nn.Linear(256, num_classes)
            nn.Linear( 256, num_classes )
        )

    def forward( self, x ):
        """
        The forward pass of the model.
        """

        # 1. Get features from both branches
        features_shufflenet = self.shufflenet_branch( x ) # Shape: [batch_size, 1024]
        features_efficientvit = self.efficientvit_branch( x ) # Shape: [batch_size, 224]

        # 2. Concatenate features along the channel dimension (dim = 1)
        combined_features = torch.cat(
            ( features_shufflenet, features_efficientvit ), dim = 1
        ) # Shape: [batch_size, 1248]

        # 3. Pass fused features through the classification head
        output = self.classification_head( combined_features ) # Shape: [batch_size, num_classes]
        
        return output

# --- Main block to test the model ---
if __name__ == "__main__":
    
    print("Creating and testing the ShuffleEfficientViT model...")

    num_classes = 6 
    
    try:
        model = ShuffleEfficientViT( num_classes = num_classes, freeze_features = True )
        dummy_input = torch.randn( 4, 3, 224, 224 ) # Batch of 4 images, 3 channels, 224x224

        print("Testing model with a dummy input batch...")
        output = model( dummy_input )
        
        print("\n--- Model Test Successful ---")
        print(f"Input shape:   {dummy_input.shape}")
        print(f"Output shape:  {output.shape}")
        
        assert output.shape == ( 4, num_classes )
        print("Output shape is correct!")

    except Exception as e:
        print(f"\n--- Model Test Failed ---")
        print(f"An error occurred: {e}")
        print("Please check your 'timm' installation and model definitions.")