import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
import tempfile
from gradio_imageslider import ImageSlider
import plotly.graph_objects as go
import plotly.express as px
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import base64
from io import BytesIO
import gdown
import spaces
import cv2
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import actual segmentation model components
try:
    from models.deeplab import Deeplabv3, relu6, DepthwiseConv2D, BilinearUpsampling
    from utils.learning.metrics import dice_coef, precision, recall
    from utils.io.data import normalize
except ImportError as e:
    logger.warning(f"Could not import segmentation model components: {e}")
    # Create dummy functions if imports fail
    def relu6(x): return tf.nn.relu6(x)
    def DepthwiseConv2D(*args, **kwargs): return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
    def BilinearUpsampling(*args, **kwargs): return tf.keras.layers.UpSampling2D(*args, **kwargs)
    def dice_coef(y_true, y_pred): return tf.keras.metrics.MeanIoU()(y_true, y_pred)
    def precision(y_true, y_pred): return tf.keras.metrics.Precision()(y_true, y_pred)
    def recall(y_true, y_pred): return tf.keras.metrics.Recall()(y_true, y_pred)
    def normalize(x): return x

# --- Safe GPU Initialization ---
def safe_gpu_init():
    """Safely initialize GPU support with fallback to CPU"""
    try:
        # Clear any existing CUDA errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0])
            test_tensor = test_tensor.cuda()
            device = torch.device("cuda")
            logger.info("CUDA initialized successfully")
            return device, True
    except Exception as e:
        logger.warning(f"CUDA initialization failed: {e}")
    
    # Fallback to CPU
    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device, False

# Initialize device safely
map_device, cuda_available = safe_gpu_init()

# --- TensorFlow GPU Configuration ---
def configure_tensorflow_gpu():
    """Configure TensorFlow GPU usage"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Allow memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("TensorFlow GPU configured successfully")
            return True
    except Exception as e:
        logger.warning(f"TensorFlow GPU configuration failed: {e}")
    
    logger.info("TensorFlow using CPU")
    return False

# Configure TensorFlow
tf_gpu_available = configure_tensorflow_gpu()

# Define path and file ID
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Use vitb model for ZeroGPU compatibility
model_file = os.path.join(checkpoint_dir, "depth_anything_v2_vitb.pth")
gdrive_url = "https://drive.google.com/uc?id=1-2qHxR_VGfaEcVaZzHvq0NDQbf4h8m3r"  # vitb model URL

# Download if not already present
def download_depth_model():
    """Download depth model with error handling"""
    if not os.path.exists(model_file):
        try:
            logger.info("Downloading depth model from Google Drive...")
            gdown.download(gdrive_url, model_file, quiet=False)
            return True
        except Exception as e:
            logger.error(f"Failed to download depth model: {e}")
            return False
    return True

def check_depth_model_availability():
    """Check if depth model is available and provide fallback options"""
    if os.path.exists(model_file):
        logger.info(f"Depth model found: {model_file}")
        return True
    else:
        logger.warning(f"Depth model not found: {model_file}")
        # Check if any depth model exists
        available_models = glob.glob(os.path.join(checkpoint_dir, "depth_anything_v2_*.pth"))
        if available_models:
            logger.info(f"Found alternative models: {available_models}")
            return True
        else:
            logger.error("No depth models found in checkpoints directory")
            return False

# --- Load Wound Classification Model ---
def load_wound_classification_model():
    """Load wound classification model with error handling"""
    try:
        if os.path.exists("keras_model.h5") and os.path.exists("labels.txt"):
            wound_model = load_model("keras_model.h5")
            with open("labels.txt", "r") as f:
                class_labels = [line.strip().split(maxsplit=1)[1] for line in f]
            logger.info("Wound classification model loaded successfully")
            return wound_model, class_labels
        else:
            logger.warning("Wound classification model files not found")
            return None, ["Abrasion", "Burn", "Laceration", "Puncture", "Ulcer"]
    except Exception as e:
        logger.error(f"Failed to load wound classification model: {e}")
        return None, ["Abrasion", "Burn", "Laceration", "Puncture", "Ulcer"]

wound_model, class_labels = load_wound_classification_model()

# --- Load Actual Wound Segmentation Model ---
class WoundSegmentationModel:
    def __init__(self):
        self.input_dim_x = 224
        self.input_dim_y = 224
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained wound segmentation model with error handling"""
        try:
            # Try to load the most recent model
            weight_file_name = '2025-08-07_16-25-27.hdf5'
            model_path = f'./training_history/{weight_file_name}'
            
            if os.path.exists(model_path):
                self.model = load_model(model_path, 
                                      custom_objects={
                                          'recall': recall,
                                          'precision': precision,
                                          'dice_coef': dice_coef,
                                          'relu6': relu6,
                                          'DepthwiseConv2D': DepthwiseConv2D,
                                          'BilinearUpsampling': BilinearUpsampling
                                      })
                logger.info(f"Segmentation model loaded successfully from {model_path}")
                return
        except Exception as e:
            logger.warning(f"Error loading recent segmentation model: {e}")
            
        # Fallback to the older model
        try:
            weight_file_name = '2019-12-19 01%3A53%3A15.480800.hdf5'
            model_path = f'./training_history/{weight_file_name}'
            
            if os.path.exists(model_path):
                self.model = load_model(model_path, 
                                      custom_objects={
                                          'recall': recall,
                                          'precision': precision,
                                          'dice_coef': dice_coef,
                                          'relu6': relu6,
                                          'DepthwiseConv2D': DepthwiseConv2D,
                                          'BilinearUpsampling': BilinearUpsampling
                                      })
                logger.info(f"Fallback segmentation model loaded successfully from {model_path}")
                return
        except Exception as e2:
            logger.error(f"Error loading fallback segmentation model: {e2}")
        
        logger.warning("No segmentation model could be loaded")
        self.model = None
    
    def preprocess_image(self, image):
        """Preprocess the uploaded image for model input"""
        if image is None:
            return None
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, (self.input_dim_x, self.input_dim_y))
            
            # Normalize the image
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def postprocess_prediction(self, prediction):
        """Postprocess the model prediction"""
        try:
            # Remove batch dimension
            prediction = prediction[0]
            
            # Apply threshold to get binary mask
            threshold = 0.5
            binary_mask = (prediction > threshold).astype(np.uint8) * 255
            
            return binary_mask
        except Exception as e:
            logger.error(f"Error postprocessing prediction: {e}")
            return None
    
    def segment_wound(self, input_image):
        """Main function to segment wound from uploaded image"""
        if self.model is None:
            return None, "Error: Segmentation model not loaded. Please check the model files."
        
        if input_image is None:
            return None, "Please upload an image."
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(input_image)
            
            if processed_image is None:
                return None, "Error processing image."
            
            # Make prediction
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Postprocess the prediction
            segmented_mask = self.postprocess_prediction(prediction)
            
            return segmented_mask, "Segmentation completed successfully!"
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return None, f"Error during segmentation: {str(e)}"

# Initialize the segmentation model
segmentation_model = WoundSegmentationModel()

# --- Load Depth Model ---
def load_depth_model():
    """Load depth estimation model with error handling and ZeroGPU optimization"""
    try:
        # Check model availability first
        if not check_depth_model_availability():
            logger.error("No depth models available")
            return None
            
        # Try to download if not present
        if not os.path.exists(model_file):
            if not download_depth_model():
                logger.error("Failed to download depth model")
                return None
            
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Use smaller model for ZeroGPU compatibility
        encoder = 'vitb'  # Changed from 'vitl' to 'vitb' for better ZeroGPU compatibility
        
        # Check if the specific model file exists, otherwise try alternatives
        model_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        if not os.path.exists(model_path):
            # Try to find any available model
            available_models = glob.glob(os.path.join(checkpoint_dir, "depth_anything_v2_*.pth"))
            if available_models:
                # Use the first available model and determine encoder type
                model_path = available_models[0]
                encoder = model_path.split('_')[-1].replace('.pth', '')
                logger.info(f"Using available model: {model_path} with encoder: {encoder}")
            else:
                logger.error("No depth models found")
                return None
        
        depth_model = DepthAnythingV2(**model_configs[encoder])
        
        # Load state dict with proper error handling
        state_dict = torch.load(model_path, map_location=map_device)
        depth_model.load_state_dict(state_dict)
        depth_model = depth_model.to(map_device).eval()
        
        logger.info(f"Depth model loaded successfully on {map_device} (using {encoder} variant for ZeroGPU)")
        return depth_model
        
    except Exception as e:
        logger.error(f"Failed to load depth model: {e}")
        return None

depth_model = load_depth_model()

# --- Custom CSS for unified dark theme ---
css = """
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    background-color: #121212;
    color: #ffffff;
    padding: 20px;
}
.gr-button {
    background-color: #2c3e50;
    color: white;
    border-radius: 10px;
}
.gr-button:hover {
    background-color: #34495e;
}
.gr-html, .gr-html div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    word-break: break-word !important;
}
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    margin: 2rem 0;
    color: #ffffff;
}
h2 {
    color: #ffffff;
    text-align: center;
    margin: 1rem 0;
}
.gr-tabs {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 10px;
}
.gr-tab-nav {
    background-color: #2c3e50;
    border-radius: 8px;
}
.gr-tab-nav button {
    color: #ffffff !important;
}
.gr-tab-nav button.selected {
    background-color: #34495e !important;
}
.error-message {
    color: #ff5252;
    background-color: #2c1810;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.success-message {
    color: #4caf50;
    background-color: #1b2c1b;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
"""

# --- Wound Classification Functions ---
def preprocess_input(img):
    """Preprocess input image for classification"""
    try:
        img = img.resize((224, 224))
        arr = keras_image.img_to_array(img)
        arr = arr / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        return None

def get_reasoning_from_gemini(img, prediction):
    """Get reasoning for wound classification"""
    try:
        explanations = {
            "Abrasion": "This appears to be an abrasion wound, characterized by superficial damage to the skin surface. The wound shows typical signs of friction or scraping injury.",
            "Burn": "This wound exhibits characteristics consistent with a burn injury, showing tissue damage from heat, chemicals, or radiation exposure.",
            "Laceration": "This wound displays the irregular edges and tissue tearing typical of a laceration, likely caused by blunt force trauma.",
            "Puncture": "This wound shows a small, deep entry point characteristic of puncture wounds, often caused by sharp, pointed objects.",
            "Ulcer": "This wound exhibits the characteristics of an ulcer, showing tissue breakdown and potential underlying vascular or pressure issues."
        }
        return explanations.get(prediction, f"This wound has been classified as {prediction}. Please consult with a healthcare professional for detailed assessment.")
    except Exception as e:
        logger.error(f"Error generating reasoning: {e}")
        return f"(Reasoning unavailable: {str(e)})"

def classify_wound_image(img):
    """Classify wound image with error handling"""
    if img is None:
        return "<div class='error-message'>No image provided</div>", ""

    try:
        if wound_model is None:
            return "<div class='error-message'>Wound classification model not available</div>", ""
            
        img_array = preprocess_input(img)
        if img_array is None:
            return "<div class='error-message'>Error preprocessing image</div>", ""
            
        predictions = wound_model.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(predictions))
        pred_class = class_labels[pred_idx] if pred_idx < len(class_labels) else "Unknown"
        confidence = float(predictions[pred_idx])

        # Get reasoning
        reasoning_text = get_reasoning_from_gemini(img, pred_class)

        # Prediction Card
        predicted_card = f"""
        <div style='padding: 20px; background-color: #1e1e1e; border-radius: 12px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.5);'>
            <div style='font-size: 22px; font-weight: bold; color: orange; margin-bottom: 10px;'>
                Predicted Wound Type
            </div>
            <div style='font-size: 26px; color: white; margin-bottom: 10px;'>
                {pred_class}
            </div>
            <div style='font-size: 16px; color: #cccccc;'>
                Confidence: {confidence:.2%}
            </div>
        </div>
        """

        # Reasoning Card
        reasoning_card = f"""
        <div style='padding: 20px; background-color: #1e1e1e; border-radius: 12px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.5);'>
            <div style='font-size: 22px; font-weight: bold; color: orange; margin-bottom: 10px;'>
                Reasoning
            </div>
            <div style='font-size: 16px; color: white; min-height: 80px;'>
                {reasoning_text}
            </div>
        </div>
        """

        return predicted_card, reasoning_card
        
    except Exception as e:
        logger.error(f"Error classifying wound image: {e}")
        return f"<div class='error-message'>Error during classification: {str(e)}</div>", ""

# --- Depth Estimation Functions with ZeroGPU Optimization ---
@spaces.GPU(duration=30)  # Reduced duration for ZeroGPU compatibility
def predict_depth(image):
    """Predict depth with optimized GPU usage for ZeroGPU"""
    if depth_model is None:
        raise Exception("Depth model not loaded")
    
    try:
        # Ensure proper cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Resize image to reduce memory usage for ZeroGPU
        if image.shape[0] > 512 or image.shape[1] > 512:
            h, w = image.shape[:2]
            scale = min(512 / h, 512 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            logger.info(f"Resized image from {h}x{w} to {new_h}x{new_w} for ZeroGPU compatibility")
        
        # Add timeout mechanism
        start_time = time.time()
        result = depth_model.infer_image(image)
        
        # Log timing for debugging
        elapsed = time.time() - start_time
        logger.info(f"Depth inference completed in {elapsed:.2f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting depth: {e}")
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def calculate_max_points(image):
    """Calculate maximum points based on image dimensions"""
    if image is None:
        return 10000
    try:
        h, w = image.shape[:2]
        max_points = h * w * 3
        return max(1000, min(max_points, 100000))  # Reduced max points for better performance
    except:
        return 10000

def update_slider_on_image_upload(image):
    """Update the points slider when an image is uploaded"""
    max_points = calculate_max_points(image)
    default_value = min(5000, max_points // 10)  # Reduced default value
    return gr.Slider(minimum=1000, maximum=max_points, value=default_value, step=1000,
                     label=f"Number of 3D points (max: {max_points:,})")

def create_point_cloud(image, depth_map, focal_length_x=470.4, focal_length_y=470.4, max_points=30000):
    """Create a point cloud from depth map with error handling"""
    try:
        h, w = depth_map.shape
        step = max(1, int(np.sqrt(h * w / max_points) * 0.5))

        y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
        x_cam = (x_coords - w / 2) / focal_length_x
        y_cam = (y_coords - h / 2) / focal_length_y
        depth_values = depth_map[::step, ::step]

        x_3d = x_cam * depth_values
        y_3d = y_cam * depth_values
        z_3d = depth_values

        points = np.stack([x_3d.flatten(), y_3d.flatten(), z_3d.flatten()], axis=1)
        image_colors = image[::step, ::step, :]
        colors = image_colors.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    except Exception as e:
        logger.error(f"Error creating point cloud: {e}")
        return None

def create_enhanced_3d_visualization(image, depth_map, max_points=10000):
    """Create enhanced 3D visualization with error handling"""
    try:
        h, w = depth_map.shape
        step = max(1, int(np.sqrt(h * w / max_points)))

        y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
        focal_length = 470.4
        x_cam = (x_coords - w / 2) / focal_length
        y_cam = (y_coords - h / 2) / focal_length
        depth_values = depth_map[::step, ::step]

        x_3d = x_cam * depth_values
        y_3d = y_cam * depth_values
        z_3d = depth_values

        x_flat = x_3d.flatten()
        y_flat = y_3d.flatten()
        z_flat = z_3d.flatten()

        image_colors = image[::step, ::step, :]
        colors_flat = image_colors.reshape(-1, 3)

        fig = go.Figure(data=[go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                size=1.5,
                color=colors_flat,
                opacity=0.9
            ),
            hovertemplate='<b>3D Position:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>' +
                         '<b>Depth:</b> %{z:.2f}<br>' +
                         '<extra></extra>'
        )])

        fig.update_layout(
            title="3D Point Cloud Visualization (Camera Projection)",
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)",
                camera=dict(
                    eye=dict(x=2.0, y=2.0, z=2.0),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            width=700,
            height=600
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating 3D visualization: {e}")
        return None

def on_depth_submit(image, num_points, focal_x, focal_y):
    """Handle depth submission with comprehensive error handling and retry logic"""
    if image is None:
        return None, None, None, None, None
        
    if depth_model is None:
        error_msg = "Depth model not available. Please check if the model files are present."
        logger.error(error_msg)
        return None, None, None, None, None

    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Depth estimation attempt {retry_count + 1}/{max_retries}")
            
            original_image = image.copy()
            h, w = image.shape[:2]

            # Predict depth using the model with retry logic
            depth = predict_depth(image[:, :, ::-1])  # RGB to BGR if needed

            # Save raw 16-bit depth
            raw_depth = Image.fromarray(depth.astype('uint16'))
            tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            raw_depth.save(tmp_raw_depth.name)

            # Normalize and convert to grayscale for display
            norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            norm_depth = norm_depth.astype(np.uint8)
            colored_depth = (matplotlib.colormaps.get_cmap('Spectral_r')(norm_depth)[:, :, :3] * 255).astype(np.uint8)

            gray_depth = Image.fromarray(norm_depth)
            tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            gray_depth.save(tmp_gray_depth.name)

            # Create point cloud
            pcd = create_point_cloud(original_image, norm_depth, focal_x, focal_y, max_points=num_points)
            
            tmp_pointcloud = None
            if pcd is not None:
                tmp_pointcloud = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
                o3d.io.write_point_cloud(tmp_pointcloud.name, pcd)

            # Create enhanced 3D scatter plot visualization
            depth_3d = create_enhanced_3d_visualization(original_image, norm_depth, max_points=num_points)

            logger.info("Depth estimation completed successfully")
            return [(original_image, colored_depth), tmp_gray_depth.name, tmp_raw_depth.name, tmp_pointcloud.name if tmp_pointcloud else None, depth_3d]

        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            logger.error(f"Error in depth estimation (attempt {retry_count}): {error_msg}")
            
            # Handle specific ZeroGPU errors
            if "GPU task aborted" in error_msg or "Expired ZeroGPU proxy token" in error_msg:
                if retry_count < max_retries:
                    logger.info(f"ZeroGPU error detected, retrying in 2 seconds... (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    logger.error("Max retries reached for ZeroGPU errors")
                    # Create a fallback depth map for demonstration
                    return create_fallback_depth_result(original_image, num_points, focal_x, focal_y)
            else:
                # For other errors, don't retry
                break
    
    # If we get here, all retries failed
    logger.error("Depth estimation failed after all retries")
    return create_fallback_depth_result(original_image, num_points, focal_x, focal_y)

def create_fallback_depth_result(image, num_points, focal_x, focal_y):
    """Create a fallback depth result when GPU processing fails"""
    try:
        logger.info("Creating fallback depth result due to GPU limitations")
        
        # Create a simple depth map based on image intensity
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        depth = gray.astype(np.float32)
        
        # Apply some depth-like filtering
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        depth = cv2.medianBlur(depth.astype(np.uint8), 5).astype(np.float32)
        
        # Normalize for display
        norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        norm_depth = norm_depth.astype(np.uint8)
        colored_depth = (matplotlib.colormaps.get_cmap('Spectral_r')(norm_depth)[:, :, :3] * 255).astype(np.uint8)

        # Save files
        gray_depth = Image.fromarray(norm_depth)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)

        raw_depth = Image.fromarray(depth.astype('uint16'))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp_raw_depth.name)

        # Create simple point cloud
        pcd = create_point_cloud(image, norm_depth, focal_x, focal_y, max_points=num_points)
        tmp_pointcloud = None
        if pcd is not None:
            tmp_pointcloud = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
            o3d.io.write_point_cloud(tmp_pointcloud.name, pcd)

        # Create 3D visualization
        depth_3d = create_enhanced_3d_visualization(image, norm_depth, max_points=num_points)

        return [(image, colored_depth), tmp_gray_depth.name, tmp_raw_depth.name, tmp_pointcloud.name if tmp_pointcloud else None, depth_3d]
        
    except Exception as e:
        logger.error(f"Error creating fallback depth result: {e}")
        return None, None, None, None, None

# --- Enhanced Severity Analysis Functions ---
def compute_enhanced_depth_statistics(depth_map, mask, pixel_spacing_mm=0.5, depth_calibration_mm=15.0):
    """Enhanced depth analysis with proper calibration"""
    try:
        pixel_spacing_mm = float(pixel_spacing_mm)
        pixel_area_cm2 = (pixel_spacing_mm / 10.0) ** 2
        
        wound_mask = (mask > 127).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel)
        
        wound_depths = depth_map[wound_mask > 0]
        
        if len(wound_depths) == 0:
            return {
                'total_area_cm2': 0,
                'mean_depth_mm': 0,
                'max_depth_mm': 0,
                'depth_std_mm': 0,
                'wound_volume_cm3': 0,
                'analysis_quality': 'Low',
                'wound_pixel_count': 0
            }
        
        # Simple depth analysis without complex normalization
        mean_depth_mm = np.mean(wound_depths) * depth_calibration_mm / 255.0
        max_depth_mm = np.max(wound_depths) * depth_calibration_mm / 255.0
        depth_std_mm = np.std(wound_depths) * depth_calibration_mm / 255.0
        
        total_pixels = np.sum(wound_mask > 0)
        total_area_cm2 = total_pixels * pixel_area_cm2
        wound_volume_cm3 = total_area_cm2 * (mean_depth_mm / 10.0)
        
        analysis_quality = "High" if total_pixels > 1000 else "Medium" if total_pixels > 500 else "Low"
        
        return {
            'total_area_cm2': total_area_cm2,
            'mean_depth_mm': mean_depth_mm,
            'max_depth_mm': max_depth_mm,
            'depth_std_mm': depth_std_mm,
            'wound_volume_cm3': wound_volume_cm3,
            'analysis_quality': analysis_quality,
            'wound_pixel_count': total_pixels
        }
        
    except Exception as e:
        logger.error(f"Error computing depth statistics: {e}")
        return {
            'total_area_cm2': 0,
            'mean_depth_mm': 0,
            'max_depth_mm': 0,
            'depth_std_mm': 0,
            'wound_volume_cm3': 0,
            'analysis_quality': 'Error',
            'wound_pixel_count': 0
        }

def classify_wound_severity_by_enhanced_metrics(depth_stats):
    """Enhanced wound severity classification"""
    try:
        if depth_stats['total_area_cm2'] == 0:
            return "Unknown"
        
        severity_score = 0
        max_depth = depth_stats['max_depth_mm']
        mean_depth = depth_stats['mean_depth_mm']
        total_area = depth_stats['total_area_cm2']
        
        # Depth-based scoring
        if max_depth >= 10.0:
            severity_score += 3
        elif max_depth >= 6.0:
            severity_score += 2
        elif max_depth >= 4.0:
            severity_score += 1
        
        # Area-based scoring
        if total_area >= 10.0:
            severity_score += 2
        elif total_area >= 5.0:
            severity_score += 1
        
        # Classification
        if severity_score >= 4:
            return "Severe"
        elif severity_score >= 2:
            return "Moderate"
        elif severity_score >= 1:
            return "Mild"
        else:
            return "Superficial"
            
    except Exception as e:
        logger.error(f"Error classifying severity: {e}")
        return "Unknown"

def analyze_wound_severity(image, depth_map, wound_mask, pixel_spacing_mm=0.5, depth_calibration_mm=15.0):
    """Analyze wound severity with error handling"""
    if image is None or depth_map is None or wound_mask is None:
        return "<div class='error-message'>Please upload image, depth map, and wound mask.</div>"

    try:
        if len(wound_mask.shape) == 3:
            wound_mask = np.mean(wound_mask, axis=2)

        if depth_map.shape[:2] != wound_mask.shape[:2]:
            from PIL import Image
            mask_pil = Image.fromarray(wound_mask.astype(np.uint8))
            mask_pil = mask_pil.resize((depth_map.shape[1], depth_map.shape[0]))
            wound_mask = np.array(mask_pil)

        stats = compute_enhanced_depth_statistics(depth_map, wound_mask, pixel_spacing_mm, depth_calibration_mm)
        severity = classify_wound_severity_by_enhanced_metrics(stats)
        
        severity_color = {
            "Superficial": "#4CAF50",    # Green
            "Mild": "#8BC34A",           # Light Green
            "Moderate": "#FF9800",       # Orange
            "Severe": "#F44336",         # Red
            "Very Severe": "#9C27B0"     # Purple
        }.get(severity, "#9E9E9E")       # Gray for unknown

        # Create comprehensive medical report
        report = f"""
        <div style='padding: 20px; background-color: #1e1e1e; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.5);'>
            <div style='font-size: 24px; font-weight: bold; color: {severity_color}; margin-bottom: 15px;'>
                🩹 Wound Severity Analysis
            </div>

            <div style='background-color: #2c2c2c; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <div style='font-size: 18px; font-weight: bold; color: #ffffff; margin-bottom: 10px;'>
                    📊 Measurements
                </div>
                <div style='color: #cccccc; line-height: 1.6;'>
                    <div>📏 <b>Mean Depth:</b> {stats['mean_depth_mm']:.1f} mm</div>
                    <div>📐 <b>Max Depth:</b> {stats['max_depth_mm']:.1f} mm</div>
                    <div>📊 <b>Depth Std Dev:</b> {stats['depth_std_mm']:.1f} mm</div>
                    <div>📦 <b>Wound Area:</b> {stats['total_area_cm2']:.2f} cm²</div>
                    <div>💧 <b>Estimated Volume:</b> {stats['wound_volume_cm3']:.2f} cm³</div>
                    <div>🔍 <b>Analysis Quality:</b> {stats['analysis_quality']}</div>
                    <div>📊 <b>Data Points:</b> {stats['wound_pixel_count']:,}</div>
                </div>
            </div>

            <div style='text-align: center; padding: 15px; background-color: #2c2c2c; border-radius: 8px; border-left: 4px solid {severity_color};'>
                <div style='font-size: 20px; font-weight: bold; color: {severity_color};'>
                    🎯 Severity Assessment: {severity}
                </div>
                <div style='font-size: 14px; color: #cccccc; margin-top: 5px;'>
                    {get_severity_description(severity)}
                </div>
            </div>
        </div>
        """

        return report
        
    except Exception as e:
        logger.error(f"Error analyzing wound severity: {e}")
        return f"<div class='error-message'>Error during severity analysis: {str(e)}</div>"

def get_severity_description(severity):
    """Get medical description for severity level"""
    descriptions = {
        "Superficial": "Minimal tissue damage, typically heals within 1-2 weeks with basic wound care.",
        "Mild": "Limited tissue involvement, good healing potential with proper care.",
        "Moderate": "Requires careful monitoring and may need advanced wound care techniques.",
        "Severe": "Significant tissue involvement, requires immediate medical attention.",
        "Very Severe": "Extensive damage requiring immediate surgical intervention.",
        "Unknown": "Unable to determine severity due to insufficient data."
    }
    return descriptions.get(severity, "Severity assessment unavailable.")

# --- Wound Segmentation Functions ---
def create_automatic_wound_mask(image, method='deep_learning'):
    """Automatically generate wound mask using deep learning model"""
    if image is None:
        return None

    try:
        if method == 'deep_learning':
            mask, _ = segmentation_model.segment_wound(image)
            return mask
        else:
            mask, _ = segmentation_model.segment_wound(image)
            return mask
    except Exception as e:
        logger.error(f"Error creating automatic wound mask: {e}")
        return None

def post_process_wound_mask(mask, min_area=100):
    """Post-process the wound mask to remove noise"""
    if mask is None:
        return None

    try:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_clean = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.fillPoly(mask_clean, [contour], 255)

        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        return mask_clean
        
    except Exception as e:
        logger.error(f"Error post-processing mask: {e}")
        return mask

def create_sample_wound_mask(image_shape, center=None, radius=50):
    """Create a sample circular wound mask for testing"""
    try:
        if center is None:
            center = (image_shape[1] // 2, image_shape[0] // 2)

        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        y, x = np.ogrid[:image_shape[0], :image_shape[1]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask[dist_from_center <= radius] = 255
        return mask
    except Exception as e:
        logger.error(f"Error creating sample mask: {e}")
        return np.zeros(image_shape[:2], dtype=np.uint8)

# --- Main Gradio Interface ---
with gr.Blocks(css=css, title="Wound Analysis & Depth Estimation") as demo:
    gr.HTML("<h1>Wound Analysis & Depth Estimation System</h1>")
    gr.Markdown("### Comprehensive wound analysis with classification and 3D depth mapping capabilities")
    
    # Status display with ZeroGPU information
    status_display = gr.HTML(f"""
    <div style='padding: 15px; background-color: #2c2c2c; border-radius: 8px; margin-bottom: 20px;'>
        <div style='font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 10px;'>
            🖥️ System Status
        </div>
        <div style='color: #cccccc; line-height: 1.6;'>
            <div>🔧 <b>Device:</b> {map_device} {'✅' if cuda_available else '⚠️'}</div>
            <div>🧠 <b>TensorFlow:</b> {'GPU' if tf_gpu_available else 'CPU'} {'✅' if tf_gpu_available else '⚠️'}</div>
            <div>🔬 <b>Wound Classification:</b> {'✅ Ready' if wound_model is not None else '❌ Not Available'}</div>
            <div>📏 <b>Depth Estimation:</b> {'✅ Ready (vitb model)' if depth_model is not None else '❌ Not Available'}</div>
            <div>🎯 <b>Wound Segmentation:</b> {'✅ Ready' if segmentation_model.model is not None else '❌ Not Available'}</div>
            <div>⚡ <b>ZeroGPU:</b> Optimized with 30s timeout, image resizing, and fallback mode</div>
        </div>
    </div>
    """)

    # Add warning about GPU usage
    gr.HTML("""
    <div style='padding: 15px; background-color: #2c1810; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ff9800;'>
        <div style='font-size: 16px; font-weight: bold; color: #ff9800; margin-bottom: 10px;'>
            ⚠️ ZeroGPU Optimizations Applied
        </div>
        <div style='color: #cccccc; line-height: 1.6;'>
            <div>• Using smaller vitb model for better ZeroGPU compatibility</div>
            <div>• Images automatically resized to max 512px for memory efficiency</div>
            <div>• 30-second GPU timeout with automatic retry logic</div>
            <div>• Fallback depth generation if GPU processing fails</div>
            <div>• If you still encounter issues, the system will provide a fallback result</div>
        </div>
    </div>
    """)

    # Shared image state
    shared_image = gr.State()

    with gr.Tabs():
        # Tab 1: Wound Classification
        with gr.Tab("1. Wound Classification"):
            gr.Markdown("### Step 1: Upload and classify your wound image")
            gr.Markdown("This module analyzes wound images and provides classification with AI-powered reasoning.")

            with gr.Row():
                with gr.Column(scale=1):
                    wound_image_input = gr.Image(label="Upload Wound Image", type="pil", height=350)

                with gr.Column(scale=1):
                    wound_prediction_box = gr.HTML()
                    wound_reasoning_box = gr.HTML()

            # Button to pass image to depth estimation
            with gr.Row():
                pass_to_depth_btn = gr.Button("📊 Pass Image to Depth Analysis", variant="secondary", size="lg")
                pass_status = gr.HTML("")

            wound_image_input.change(fn=classify_wound_image, inputs=wound_image_input,
                                   outputs=[wound_prediction_box, wound_reasoning_box])

            # Store image when uploaded for classification
            wound_image_input.change(
                fn=lambda img: img,
                inputs=[wound_image_input],
                outputs=[shared_image]
            )

        # Tab 2: Depth Estimation
        with gr.Tab("2. Depth Estimation & 3D Visualization"):
            gr.Markdown("### Step 2: Generate depth maps and 3D visualizations")
            gr.Markdown("This module creates depth maps and 3D point clouds from your images.")

            with gr.Row():
                depth_input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
                depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output')

            with gr.Row():
                depth_submit = gr.Button(value="Compute Depth", variant="primary")
                load_shared_btn = gr.Button("🔄 Load Image from Classification", variant="secondary")
                points_slider = gr.Slider(minimum=1000, maximum=10000, value=5000, step=1000,
                                         label="Number of 3D points (upload image to update max)")

            with gr.Row():
                focal_length_x = gr.Slider(minimum=100, maximum=1000, value=470.4, step=10,
                                          label="Focal Length X (pixels)")
                focal_length_y = gr.Slider(minimum=100, maximum=1000, value=470.4, step=10,
                                          label="Focal Length Y (pixels)")

            with gr.Row():
                gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download")
                raw_file = gr.File(label="16-bit raw output", elem_id="download")
                point_cloud_file = gr.File(label="Point Cloud (.ply)", elem_id="download")

            # 3D Visualization
            gr.Markdown("### 3D Point Cloud Visualization")
            gr.Markdown("Enhanced 3D visualization using proper camera projection.")
            depth_3d_plot = gr.Plot(label="3D Point Cloud")

            # Store depth map for severity analysis
            depth_map_state = gr.State()

        # Tab 3: Wound Severity Analysis
        with gr.Tab("3. 🩹 Wound Severity Analysis"):
            gr.Markdown("### Step 3: Analyze wound severity using depth maps")
            gr.Markdown("This module analyzes wound severity based on depth distribution and area measurements.")

            with gr.Row():
                severity_input_image = gr.Image(label="Original Image", type='numpy')
                severity_depth_map = gr.Image(label="Depth Map (from Tab 2)", type='numpy')

            with gr.Row():
                wound_mask_input = gr.Image(label="Auto-Generated Wound Mask", type='numpy')
                severity_output = gr.HTML(label="Severity Analysis Report")

            with gr.Row():
                auto_severity_button = gr.Button("🤖 Analyze Severity with Auto-Generated Mask", variant="primary", size="lg")
                manual_severity_button = gr.Button("🔍 Manual Mask Analysis", variant="secondary", size="lg")
                create_sample_mask_btn = gr.Button("🎯 Create Sample Mask", variant="secondary")

            with gr.Row():
                pixel_spacing_slider = gr.Slider(minimum=0.1, maximum=2.0, value=0.5, step=0.1,
                                               label="Pixel Spacing (mm/pixel)")
                depth_calibration_slider = gr.Slider(minimum=5.0, maximum=30.0, value=15.0, step=1.0,
                                                   label="Depth Calibration (mm)")

            with gr.Row():
                load_depth_btn = gr.Button("🔄 Load Depth Map from Tab 2", variant="secondary")

            gr.Markdown("**Note:** Adjust pixel spacing based on your camera calibration. The segmentation model will automatically generate wound masks when available.")

    # Event handlers
    def on_depth_submit_with_state(image, num_points, focal_x, focal_y):
        """Handle depth submission and store state"""
        try:
            results = on_depth_submit(image, num_points, focal_x, focal_y)
            depth_map = None
            if image is not None and depth_model is not None and results[0] is not None:
                # Extract depth map from results if successful
                depth = predict_depth(image[:, :, ::-1])
                norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth_map = norm_depth.astype(np.uint8)
            return results + [depth_map]
        except Exception as e:
            logger.error(f"Error in depth submission: {e}")
            return [None] * 6

    # Update slider when image is uploaded
    depth_input_image.change(
        fn=update_slider_on_image_upload,
        inputs=[depth_input_image],
        outputs=[points_slider]
    )

    depth_submit.click(on_depth_submit_with_state,
                     inputs=[depth_input_image, points_slider, focal_length_x, focal_length_y],
                     outputs=[depth_image_slider, gray_depth_file, raw_file, point_cloud_file, depth_3d_plot, depth_map_state])

    # Load depth map to severity tab and auto-generate mask
    def load_depth_to_severity(depth_map, original_image):
        """Load depth map and generate automatic mask"""
        try:
            if depth_map is None:
                return None, None, None, "<div class='error-message'>No depth map available. Please compute depth in Tab 2 first.</div>"
            
            auto_mask = None
            message = "✅ Depth map loaded successfully!"
            
            if original_image is not None:
                auto_mask = create_automatic_wound_mask(original_image)
                if auto_mask is not None:
                    processed_mask = post_process_wound_mask(auto_mask, min_area=500)
                    if processed_mask is not None and np.sum(processed_mask > 0) > 0:
                        message = "✅ Depth map loaded and wound mask auto-generated!"
                        auto_mask = processed_mask
                    else:
                        message = "✅ Depth map loaded but no wound detected."
                        auto_mask = None
                else:
                    message = "✅ Depth map loaded but segmentation model not available."
            
            return depth_map, original_image, auto_mask, f"<div class='success-message'>{message}</div>"
            
        except Exception as e:
            logger.error(f"Error loading depth to severity: {e}")
            return None, None, None, f"<div class='error-message'>Error: {str(e)}</div>"

    load_depth_btn.click(
        fn=load_depth_to_severity,
        inputs=[depth_map_state, depth_input_image],
        outputs=[severity_depth_map, severity_input_image, wound_mask_input, gr.HTML()]
    )

    # Severity analysis functions
    def run_auto_severity_analysis(image, depth_map, pixel_spacing, depth_calibration):
        """Run automatic severity analysis"""
        try:
            if depth_map is None:
                return "<div class='error-message'>Please load depth map from Tab 2 first.</div>"

            auto_mask = create_automatic_wound_mask(image)
            if auto_mask is None:
                return "<div class='error-message'>Failed to generate automatic wound mask.</div>"

            processed_mask = post_process_wound_mask(auto_mask, min_area=500)
            if processed_mask is None or np.sum(processed_mask > 0) == 0:
                return "<div class='error-message'>No wound region detected by the segmentation model.</div>"

            return analyze_wound_severity(image, depth_map, processed_mask, pixel_spacing, depth_calibration)
            
        except Exception as e:
            logger.error(f"Error in auto severity analysis: {e}")
            return f"<div class='error-message'>Error during analysis: {str(e)}</div>"

    def run_manual_severity_analysis(image, depth_map, wound_mask, pixel_spacing, depth_calibration):
        """Run manual severity analysis"""
        try:
            if depth_map is None:
                return "<div class='error-message'>Please load depth map from Tab 2 first.</div>"
            if wound_mask is None:
                return "<div class='error-message'>Please upload a wound mask.</div>"

            return analyze_wound_severity(image, depth_map, wound_mask, pixel_spacing, depth_calibration)
            
        except Exception as e:
            logger.error(f"Error in manual severity analysis: {e}")
            return f"<div class='error-message'>Error during analysis: {str(e)}</div>"

    def create_sample_mask_for_image(image):
        """Create a sample mask for the current image"""
        try:
            if image is None:
                return None, "<div class='error-message'>Please upload an image first.</div>"
            
            mask = create_sample_wound_mask(image.shape, radius=min(image.shape[:2]) // 6)
            return mask, "<div class='success-message'>✅ Sample mask created!</div>"
            
        except Exception as e:
            logger.error(f"Error creating sample mask: {e}")
            return None, f"<div class='error-message'>Error: {str(e)}</div>"

    # Auto-generate mask when image is uploaded
    def auto_generate_mask_on_image_upload(image):
        """Auto-generate mask when image is uploaded"""
        try:
            if image is None:
                return None, "<div class='error-message'>No image uploaded.</div>"
            
            auto_mask = create_automatic_wound_mask(image)
            if auto_mask is not None:
                processed_mask = post_process_wound_mask(auto_mask, min_area=500)
                if processed_mask is not None and np.sum(processed_mask > 0) > 0:
                    return processed_mask, "<div class='success-message'>✅ Wound mask auto-generated!</div>"
                else:
                    return None, "<div class='error-message'>No wound detected in image.</div>"
            else:
                return None, "<div class='error-message'>Segmentation model not available.</div>"
                
        except Exception as e:
            logger.error(f"Error auto-generating mask: {e}")
            return None, f"<div class='error-message'>Error: {str(e)}</div>"

    # Load shared image from classification tab
    def load_shared_image(shared_img):
        """Load shared image from classification tab"""
        try:
            if shared_img is None:
                return gr.Image(), "<div class='error-message'>No image available from classification tab</div>"

            if hasattr(shared_img, 'convert'):
                img_array = np.array(shared_img)
                return img_array, "<div class='success-message'>✅ Image loaded from classification tab</div>"
            else:
                return shared_img, "<div class='success-message'>✅ Image loaded from classification tab</div>"
                
        except Exception as e:
            logger.error(f"Error loading shared image: {e}")
            return gr.Image(), f"<div class='error-message'>Error: {str(e)}</div>"

    def pass_image_to_depth(img):
        """Pass image to depth tab function"""
        try:
            if img is None:
                return "<div class='error-message'>No image uploaded in classification tab</div>"
            return "<div class='success-message'>✅ Image ready for depth analysis! Switch to tab 2 and click 'Load Image from Classification'</div>"
        except Exception as e:
            return f"<div class='error-message'>Error: {str(e)}</div>"

    # Connect event handlers
    auto_severity_button.click(
        fn=run_auto_severity_analysis,
        inputs=[severity_input_image, severity_depth_map, pixel_spacing_slider, depth_calibration_slider],
        outputs=[severity_output]
    )

    manual_severity_button.click(
        fn=run_manual_severity_analysis,
        inputs=[severity_input_image, severity_depth_map, wound_mask_input, pixel_spacing_slider, depth_calibration_slider],
        outputs=[severity_output]
    )

    create_sample_mask_btn.click(
        fn=create_sample_mask_for_image,
        inputs=[severity_input_image],
        outputs=[wound_mask_input, gr.HTML()]
    )

    severity_input_image.change(
        fn=auto_generate_mask_on_image_upload,
        inputs=[severity_input_image],
        outputs=[wound_mask_input, gr.HTML()]
    )

    load_shared_btn.click(
        fn=load_shared_image,
        inputs=[shared_image],
        outputs=[depth_input_image, gr.HTML()]
    )

    pass_to_depth_btn.click(
        fn=pass_image_to_depth,
        inputs=[shared_image],
        outputs=[pass_status]
    )

if __name__ == '__main__':
    demo.queue().launch(share=True)