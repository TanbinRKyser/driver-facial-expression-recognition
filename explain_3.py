import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps
import numpy as np

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

#%%
# Import your model
try:
    from engine import ShuffleEfficientViT
except ImportError:
    print("Error: 'engine.py' not found.")
    exit()
#%%

# --- CONFIGURATION ---
# MODEL_PATH = 'best_model_221225.pth'
# MODEL_PATH = 'best_model_080126.pth'
MODEL_PATH = 'best_model.pth'
# IMAGE_PATH = 'test_image_1.jpg' 
IMAGE_PATH = 'test_image_9.jpg' 
# IMAGE_PATH = 'test_image_4.jpg'
NUM_CLASSES = 6
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
#%%

# --- 1. Define Transform ---
def get_transform():
    """ Returns the preprocessing transform for the input image. """

    return transforms.Compose([
        transforms.Resize( (224, 224) ),
        transforms.Grayscale( num_output_channels = 3 ),
        transforms.ToTensor(),
        transforms.Normalize( mean = [ 0.485, 0.456, 0.406 ], 
                                std = [ 0.229, 0.224, 0.225 ] )
    ])

#%%
# --- 2. Main Function ---
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print("Loading model...")
    
    # Grad-CAM needs to compute gradients to generate the heatmap.
    model = ShuffleEfficientViT( num_classes = NUM_CLASSES, 
                                    freeze_features = False, 
                                    head_type ='conv' )
    
    # Load trained weights
    model.load_state_dict( torch.load( MODEL_PATH, 
                                        map_location = device ) )

    # Move model to device
    model.to( device )

    # Set to eval mode, but gradients are still tracked.
    model.eval() 

    # 2. Load and Preprocess Image
    img_pil = Image.open( IMAGE_PATH ).convert( 'RGB' )

    # --- STEP A: Face Detection (Using MTCNN instead of OpenCV) ---
    # Initialize MTCNN (ensure you have: pip install facenet-pytorch)
    mtcnn = MTCNN(keep_all=True, device=device)
    # Detect faces
    boxes, _ = mtcnn.detect(img_pil)

    if boxes is not None:
        print(f"Detected {len(boxes)} face(s). Cropping the largest one...")
        
        # Select the largest face (Width * Height)
        # box format is [x1, y1, x2, y2]
        largest_box = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[0]
        x1, y1, x2, y2 = largest_box
        
        # Add a small margin (10%)
        w = x2 - x1
        h = y2 - y1
        margin = int(w * 0.1)
        
        # Ensure coordinates are within image bounds
        x_start = max(0, int(x1 - margin))
        y_start = max(0, int(y1 - margin))
        x_end = min(img_pil.width, int(x2 + margin))
        y_end = min(img_pil.height, int(y2 + margin))
        
        # Crop using standard PIL
        img_pil = img_pil.crop((x_start, y_start, x_end, y_end))
    else:
        print("Warning: No face detected. Using full image (Prediction might be low confidence).")

    # --- STEP B: Domain Alignment (Crucial!) ---
    # Convert to Grayscale, then back to RGB to match training input (3 channels, same values)
    img_pil = ImageOps.grayscale(img_pil).convert('RGB')

    
    # # Convert PIL to OpenCV format (numpy array)
    # img_cv = np.array(img_pil)

    # # Convert RGB to Gray for detection
    # gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # # Load the standard Haar Cascade classifier
    # # (Make sure to import cv2 at the top)
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # # Detect faces
    # faces = face_cascade.detectMultiScale( gray, scaleFactor = 1.1, minNeighbors = 4 )

    # if len(faces) > 0:
    #     print(f"Detected {len(faces)} face(s). Cropping the largest one...")
    #     # Pick the largest face (w * h)
    #     (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        
    #     # Optional: Add a small margin so the crop isn't too tight
    #     margin = int(w * 0.1) # 10% margin
    #     x_start = max(0, x - margin)
    #     y_start = max(0, y - margin)
    #     x_end = min(img_pil.width, x + w + margin)
    #     y_end = min(img_pil.height, y + h + margin)
        
    #     # Crop the original PIL image
    #     img_pil = img_pil.crop((x_start, y_start, x_end, y_end))
    # else:
    #     print("Warning: No face detected. Using full image.")

    # transform and prepare tensor
    transform = get_transform()
    input_tensor = transform( img_pil ).unsqueeze( 0 ).to( device )

    # Convert to numpy for visualization
    rgb_img = np.array( img_pil.resize( ( 224, 224 ) ) )
    rgb_img = np.float32( rgb_img ) / 255

    # 3. Define Target Layers
    # ShuffleNet: 'conv5' is the final feature map
    target_layers_cnn = [ model.shufflenet_branch.conv5 ]
    
    # EfficientViT: Use the last stage. 
    target_layers_vit = [ model.efficientvit_branch.stages[ -1 ] ]
    # target_layers_vit = [ model.efficientvit_branch.stages[ -1 ].blocks[ -1 ].ffn1.m.pw2.bn ]
    # target_layers_vit = [ model.efficientvit_branch.stages[ -1 ].blocks[ -1 ] ]

    # 4. Initialize Grad-CAM
    # We create the CAM objects. The model is already unfrozen so this will work.
    cam_cnn = GradCAM( model = model, target_layers = target_layers_cnn )
    cam_vit = GradCAM( model = model, target_layers = target_layers_vit )

    # 5. Generate Heatmaps
    targets = None # Auto-select highest scoring class

    print("Generating CNN Heatmap...")
    grayscale_cam_cnn = cam_cnn( input_tensor = input_tensor, targets = targets )
    visualization_cnn = show_cam_on_image( rgb_img, grayscale_cam_cnn[ 0, : ], use_rgb = True  )

    print("Generating ViT Heatmap...")
    grayscale_cam_vit = cam_vit( input_tensor = input_tensor, targets = targets )
    visualization_vit = show_cam_on_image( rgb_img, grayscale_cam_vit[ 0, : ], use_rgb = True )

    # 6. Get Model Prediction for Title
    with torch.no_grad():
        # forwared pass
        output = model( input_tensor )
        # get predicted class
        probs = torch.nn.functional.softmax( output, dim = 1 )
        # probs to logits
        conf, idx = torch.max( probs, 1 )
        pred_label = CLASSES[ idx.item() ]
        conf_score = conf.item() * 100

    # 7. Plot and Save Results
    fig, axes = plt.subplots( 1, 3, figsize = ( 15, 7 ) )
    
    axes[0].imshow( rgb_img )
    axes[0].set_title(f"Pred: {pred_label}")
    axes[0].axis('off')

    axes[1].imshow( visualization_cnn )
    axes[1].set_title("ShuffleNet Focus (Local)")
    axes[1].axis('off')

    axes[2].imshow( visualization_vit )
    axes[2].set_title("EfficientViT Focus (Global)")
    axes[2].axis('off')

    output_filename = 'explainability_result_anger.png'
    plt.tight_layout()
    plt.savefig( output_filename )
    print(f"Saved explanation to '{output_filename}'")

if __name__ == "__main__":
    main()