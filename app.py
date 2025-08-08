from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from flask_cors import CORS

import torch

app = Flask(__name__)
CORS(app)  
# Load BLIP model and processor (image -> description)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
blip_model.eval()


@app.route('/')
def home():
    return "Blip Model API is running."

@app.route('/description', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    # Step 1: Load image
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    # Step 2: Generate description using BLIP
    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    description = blip_processor.decode(output[0], skip_special_tokens=True)

    return jsonify({
        'description': description
    })


if __name__ == '__main__':
    app.run(debug=True)
