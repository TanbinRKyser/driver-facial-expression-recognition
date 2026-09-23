import torch
import torch.nn as nn
import torchvision.models as models
import timm 
#%%
class ShuffleEfficientViT( nn.Module ):
    def __init__( self, 
                    num_classes = 6, 
                    freeze_features = True, 
                    head_type = 'fc' ):
        """
        Args:
            num_classes (int): Number of emotion classes.
            freeze_features (bool): Freeze backbone weights.
            head_type (str): Type of classifier head. Options: ['mlp', 'conv', 'fc']
        """

        super( ShuffleEfficientViT, self ).__init__()


        # --- 1. ShuffleNet Branch ---
        self.shufflenet_branch = models.shufflenet_v2_x1_0(
            # weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
        )

        shufflenet_features = self.shufflenet_branch.fc.in_features # 1024 features
        
        self.shufflenet_branch.fc = nn.Identity() # removing the last FC layer

        shufflenet_params = list( self.shufflenet_branch.parameters() )


        # --- 2. EfficientViT Branch ---
        try:
            self.efficientvit_branch = timm.create_model(
                'efficientvit_m2', 
                pretrained = True,
                num_classes = 0 # 224 features, removing the classification head
            )

            efficient_vit_features = self.efficientvit_branch.num_features # 224 features

            efficientvit_params = list( self.efficientvit_branch.parameters() ) 

        except Exception as e:
            print(f"Error loading EfficientViT: {e}")
            raise e

        # Freeze Backbones if specified, differential fine-tuning will unfreeze later
        if freeze_features:
            for param in self.shufflenet_branch.parameters():
                param.requires_grad = False
            for param in self.efficientvit_branch.parameters():
                param.requires_grad = False


        # --- 3. Feature Fusion & Classifier Head ---
        # 1024 (Shuffle) + 224 (ViT) = 1248 features
        # in_features = 1248
        in_features = shufflenet_features + efficient_vit_features  # 1248 features
        
        self.head_type = head_type

        # Single Fully Connected Layer from the original code
        if head_type == 'fc':
            # single Fully Connected Layer
            self.classification_head = nn.Linear( in_features, num_classes )

            classification_params = list( self.classification_head.parameters() )

        # Replacing Fully Connected Head with 2 layer MLP
        elif head_type == 'mlp':
            # two-Layer MLP
            self.classification_head = nn.Sequential(
                nn.Linear( in_features, 512 ),
                nn.BatchNorm1d( 512 ),
                nn.ReLU(),
                nn.Dropout( p = 0.5 ),
                nn.Linear( 512, num_classes )
            )

            classification_params = list( self.classification_head.parameters() )

        # Replacing FCN with Global Average Pooling Head
        elif head_type == 'conv':
            # 1x1 convolution + Global Average Pooling + Flatten
            # Input: (Batch, 1248, 1, 1) -> Conv -> (Batch, Classes, 1, 1) -> Flatten
            self.classification_head = nn.Sequential(
                nn.Unflatten( 1, ( in_features, 1, 1 ) ), # Reshape 1D -> 3D
                nn.Conv2d( in_features, num_classes, kernel_size = 1 ),
                nn.AdaptiveAvgPool2d( 1 ), # Global Average Pooling
                nn.Flatten()
            )

            classification_params = list( self.classification_head.parameters() )        

        else:
            raise ValueError(f"Invalid head_type '{head_type}'. Choose 'mlp', 'conv', or 'fc'.")


    def forward( self, x ):
        """Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, 3, 224, 224].
        Returns:
            torch.Tensor: Output logits of shape [Batch, num_classes].
        """
        
        # Extract Features
        out1 = self.shufflenet_branch( x ) # [Batch, 1024]
        out2 = self.efficientvit_branch( x ) # [Batch, 224]

        # Fuse
        combined = torch.cat( ( out1, out2 ), dim = 1 ) # [Batch, 1248]

        # Classify
        return self.classification_head( combined )

#%%

if __name__ == "__main__":
    # Test all 3 heads
    print("Testing Head Architectures...")
    dummy_input = torch.randn( 2, 3, 224, 224 ) # Batch of 2 images, 3 channels, 224x224
    
    for h_type in ['mlp', 'conv', 'fc']:
        model = ShuffleEfficientViT(  num_classes = 6, 
                                        head_type = h_type )
        output = model( dummy_input )
        print(f"Head: {h_type.ljust( 5 )} | Output Shape: {output.shape}") # Should be [2, 6]