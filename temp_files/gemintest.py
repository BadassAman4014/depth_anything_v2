import google.generativeai as genai
import gradio as gr
from PIL import Image

genai.configure(api_key="AIzaSyAnPnT6aS1Bq7hzvYxU7aDq7vuRNCfP8K8")
model = genai.GenerativeModel("gemini-2.5-pro")

def analyze_image(image):
    if image is None:
        return "Please upload an image."
    img = Image.open(image).convert("RGB")
    prompt = (
        "You are assisting in a research and image annotation task for medical education. "
        "Describe only the visible features in the image such as shapes, colors, textures, and relative positions. "
        "Do not provide medical advice or diagnosis. "
        "Keep the description objective and neutral."
    )
    response = model.generate_content([prompt, img])
    return response.text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🖼 Gemini AI Image Analyzer")
    gr.Markdown("Upload an image and let Google's Gemini AI analyze it.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Upload Image")
            analyze_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column(scale=2):
            output_card = gr.HTML(label="Gemini AI Analysis", value="""
                <div style="
                    background-color: #f8f9fa;
                    border-radius: 12px;
                    padding: 16px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    font-family: Arial, sans-serif;
                    min-height: 200px;  /* Keeps card height fixed while loading */
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #888;
                ">
                    Waiting for analysis...
                </div>
            """)

    def display_as_card(image):
        analysis = analyze_image(image)
        return f"""
        <div style="
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-family: Arial, sans-serif;
            min-height: 200px;
        ">
            <h3 style="color: #333;">Gemini AI Analysis</h3>
            <p style="color: #555;">{analysis}</p>
        </div>
        """

    analyze_btn.click(display_as_card, inputs=image_input, outputs=output_card)

if __name__ == "__main__":
    demo.launch()
