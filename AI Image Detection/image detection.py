import os
import gradio as gr
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import tempfile

# Load models
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Functions
def detect_objects(image_path):
    results = model.predict(source=image_path, conf=0.25, save=False)
    img_with_boxes = results[0].plot()
    return Image.fromarray(img_with_boxes)

def get_caption_and_audio(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = caption_processor(image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)

    tts = gTTS(caption, lang='en')
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)

    return caption, temp_audio.name

# Gradio Interface
with gr.Blocks(css="""
    .gradio-container {
        background-color: #FFFDE7 !important;
        color: #000;
        font-family: 'Segoe UI', sans-serif;
    }
    .upload-box .gr-image {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
    }
    .upload-box {
        width: 500px;
        height: 400px;
        margin: auto;
        border: 2px solid #ccc;
        border-radius: 8px;
        overflow: hidden;
    }
    .header {
        background-color: #F87F2F;
        color: white;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .footer {
        display: flex;
        justify-content: space-between;
        background-color: #F87F2F;
        color: white;
        padding: 16px;
        margin-top: 20px;
        border-radius: 8px;
        font-size: 14px;
        flex-wrap: wrap;
    }
    .footer div {
        flex: 1;
        text-align: center;
    }
    .footer-icons a {
        margin: 0 10px;
        text-decoration: none;
        color: white;
        font-size: 24px;
    }
""") as demo:

    gr.HTML("<div class='header'>Real world image captioning using AI Model</div>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image", elem_classes="upload-box")
            detect_btn = gr.Button("Detect & Caption", variant="primary")
        
        with gr.Column():
            detection_output = gr.Image(label="Detected Objects (YOLOv8)")
            detected_names = gr.Textbox(label="Detected Object Names", interactive=False)
            caption_text = gr.Textbox(label="Image Caption", lines=3, interactive=False)
            audio_output = gr.Audio(label="Caption Audio", interactive=True)

    def process_all(image):
        detected_image = detect_objects(image)

        results = model.predict(source=image, conf=0.25, save=False)
        if results and len(results) > 0 and hasattr(results[0], 'boxes'):
            detected_names_list = [model.names[int(c)] for c in results[0].boxes.cls]
            names_str = ", ".join(detected_names_list)
        else:
            names_str = "No objects detected"

        caption, audio_file = get_caption_and_audio(image)
        return detected_image, names_str, caption, audio_file

    detect_btn.click(fn=process_all, inputs=image_input,
                     outputs=[detection_output, detected_names, caption_text, audio_output])

    gr.HTML("""
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <div class='footer'>
            <div>
                <strong>IBADAT INTERNATIONAL UNIVERSITY</strong><br>
                ISLAMABAD, JAPAN ROAD
            </div>
            <div>
                <strong>IMAGE CAPTIONING | WEBCAM | REAL WORLD DETECTION</strong>
            </div>
            <div>
                <strong>CONTACT US</strong><br>
                <div class='footer-icons'>
                    <a href="https://github.com/" target="_blank"><i class="fab fa-github"></i></a>
                    <a href="https://linkedin.com/" target="_blank"><i class="fab fa-linkedin"></i></a>
                    <a href="https://wa.me/923001234567" target="_blank"><i class="fab fa-whatsapp"></i></a>
                </div>
            </div>
        </div>
    """)

demo.launch(share=True)
