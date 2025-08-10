import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

# Import custom modules
from models.deeplab import Deeplabv3, relu6, DepthwiseConv2D, BilinearUpsampling
from utils.learning.metrics import dice_coef, precision, recall
from utils.io.data import normalize

class WoundSegmentationApp:
    def __init__(self):
        self.input_dim_x = 224
        self.input_dim_y = 224
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained wound segmentation model"""
        try:
            # Load the model with custom objects
            weight_file_name = '2025-08-07_12-30-43.hdf5'  # Use the most recent model
            model_path = f'./training_history/{weight_file_name}'
            
            self.model = load_model(model_path, 
                                  custom_objects={
                                      'recall': recall,
                                      'precision': precision,
                                      'dice_coef': dice_coef,
                                      'relu6': relu6,
                                      'DepthwiseConv2D': DepthwiseConv2D,
                                      'BilinearUpsampling': BilinearUpsampling
                                  })
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to the older model if the newer one fails
            try:
                weight_file_name = '2019-12-19 01%3A53%3A15.480800.hdf5'
                model_path = f'./training_history/{weight_file_name}'
                
                self.model = load_model(model_path, 
                                      custom_objects={
                                          'recall': recall,
                                          'precision': precision,
                                          'dice_coef': dice_coef,
                                          'relu6': relu6,
                                          'DepthwiseConv2D': DepthwiseConv2D,
                                          'BilinearUpsampling': BilinearUpsampling
                                      })
                print(f"Model loaded successfully from {model_path}")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                self.model = None
    
    def preprocess_image(self, image):
        """Preprocess the uploaded image for model input"""
        if image is None:
            return None
        
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
    
    def postprocess_prediction(self, prediction):
        """Postprocess the model prediction"""
        # Remove batch dimension
        prediction = prediction[0]
        
        # Apply threshold to get binary mask
        threshold = 0.5
        binary_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # Convert to 3-channel image for visualization
        mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
        
        return mask_rgb
    
    def segment_wound(self, input_image):
        """Main function to segment wound from uploaded image"""
        if self.model is None:
            return None, "Error: Model not loaded. Please check the model files."
        
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
            
            # Create overlay image (original image with segmentation overlay)
            original_resized = cv2.resize(input_image, (self.input_dim_x, self.input_dim_y))
            if len(original_resized.shape) == 3:
                original_resized = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
            
            # Create overlay with red segmentation
            overlay = original_resized.copy()
            mask_red = np.zeros_like(original_resized)
            mask_red[:, :, 2] = segmented_mask[:, :, 0]  # Red channel
            
            # Blend overlay with original image
            alpha = 0.6
            overlay = cv2.addWeighted(overlay, 1-alpha, mask_red, alpha, 0)
            
            return segmented_mask, overlay
            
        except Exception as e:
            return None, f"Error during segmentation: {str(e)}"

def create_gradio_interface():
    """Create and return the Gradio interface"""
    
    # Initialize the app
    app = WoundSegmentationApp()
    
    # Define the interface
    with gr.Blocks(title="Wound Segmentation Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🩹 Wound Segmentation Tool
            
            Upload an image of a wound to get an automated segmentation mask.
            The model will identify and highlight the wound area in the image.
            
            **Instructions:**
            1. Upload an image of a wound
            2. Click "Segment Wound" to process the image
            3. View the segmentation mask and overlay results
            """
        )
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Wound Image",
                    type="numpy",
                    height=400
                )
                
                segment_btn = gr.Button(
                    "🔍 Segment Wound",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column():
                mask_output = gr.Image(
                    label="Segmentation Mask",
                    height=400
                )
                
                overlay_output = gr.Image(
                    label="Overlay Result",
                    height=400
                )
        
        # Status message
        status_msg = gr.Textbox(
            label="Status",
            interactive=False,
            placeholder="Ready to process images..."
        )
        
        # Example images
        gr.Markdown("### 📸 Example Images")
        gr.Markdown("You can test the tool with wound images from the dataset.")
        
        # Connect the button to the segmentation function
        def process_image(image):
            mask, overlay = app.segment_wound(image)
            if mask is None:
                return None, None, overlay  # overlay contains error message
            return mask, overlay, "Segmentation completed successfully!"
        
        segment_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[mask_output, overlay_output, status_msg]
        )
        
        # Auto-process when image is uploaded
        input_image.change(
            fn=process_image,
            inputs=[input_image],
            outputs=[mask_output, overlay_output, status_msg]
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 