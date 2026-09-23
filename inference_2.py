import torch
from torchvision import transforms
from PIL import Image, ImageOps
import os
from facenet_pytorch import MTCNN

import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt

# Import your model
try:
    from engine import ShuffleEfficientViT
except ImportError:
    print("Error: Make sure 'engine.py' is in the same directory.")
    exit()

# --- Configuration ---
MODEL_PATH = 'best_model.pth'  
# MODEL_PATH = 'best_model_080126.pth'  
IMAGE_PATH = 'test_image_9.jpg'  
NUM_CLASSES = 6
CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# --- 1. Define the Preprocessing ---
inference_transform = transforms.Compose([
    transforms.Resize( (224, 224) ),
    transforms.ToTensor(),
    transforms.Normalize( mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225] )
])

# --- 2. Prediction Function ---
def predict_image( image_path, model, device ):
    
    # 1. Load Image
    if not os.path.exists( image_path ):
        print(f"Error: Image not found at {image_path}")
        return

    img_pil = Image.open( image_path ).convert( 'RGB' )
    
    # 2. Face Detection & Cropping (MTCNN)
    mtcnn = MTCNN( keep_all = False, 
                    select_largest = True, 
                    device = device ) 
    
    boxes, _ = mtcnn.detect( img_pil )

    if boxes is not None:
        box = boxes[ 0 ]
        margin = 0.15
        w = box[ 2 ] - box[ 0 ]
        h = box[ 3 ] - box[ 1 ]
        x1 = max( 0, int( box[ 0 ] - w * margin ) )
        y1 = max( 0, int( box[ 1 ] - h * margin ) )
        x2 = min( img_pil.width, int( box[ 2 ] + w * margin ) )
        y2 = min( img_pil.height, int( box[ 3 ] + h * margin ) )

        face_crop = img_pil.crop( (x1, y1, x2, y2) )
        print("Face detected and cropped successfully.")
    else:
        print("WARNING: No face detected. Using full image.")
        face_crop = img_pil

    # 3. Grayscale Domain Alignment
    face_input = ImageOps.grayscale( face_crop ).convert('RGB')

    # 4. Prepare Tensor
    img_tensor = inference_transform( face_input ) 
    img_tensor = img_tensor.unsqueeze( 0 ).to( device )

    # 5. Predict
    model.eval()
    with torch.no_grad():
        outputs = model( img_tensor )
        probs = torch.nn.functional.softmax( outputs, dim = 1 )
        confidence, predicted_class_idx = torch.max( probs, 1 )
        predicted_label = CLASSES[ predicted_class_idx.item() ]
        confidence_score = confidence.item() * 100

    # 6. Visualize
    plt.figure(figsize=(6, 6))
    
    # FIX: We display 'face_crop' because 'original_image' is undefined
    plt.imshow(face_crop) 
    
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence_score:.2f}%", 
            color='green', fontsize = 14)
    
    output_filename = 'prediction_result.png'
    plt.savefig(output_filename)
    plt.close() # Good practice to close memory
    
    print(f"\n--- Prediction Result ---")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence:      {confidence_score:.2f}%")
    print(f"Saved visualization to: {output_filename}")

    print("\nClass Probabilities:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name.ljust(10)}: {probs[0][i].item() * 100:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = ShuffleEfficientViT(num_classes=NUM_CLASSES, head_type='conv')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Weights loaded successfully.")
    else:
        print(f"Error: '{MODEL_PATH}' not found.")
        exit()
    
    model.to(device)
    predict_image(IMAGE_PATH, model, device)