from flask import Flask, render_template, request
from PIL import Image
import base64
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image from the request
        image = request.files['image']
        
        # Process the uploaded image
        raw_image = Image.open(image).convert('RGB')

        # Generate captions
        captions = generate_captions(raw_image)

        # Convert the image to base64 for display in HTML
        buffered = BytesIO()
        raw_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return render_template('index.html', image=img_str, captions=captions)
    else:
        return render_template('index.html')

def generate_captions(raw_image, num_captions=5):
    captions = []

    # Conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    # Set sampling options
    output_options = {
        "do_sample": True,
        "max_length": 20,
        "top_k": 50,
        "temperature": 0.7,
        "num_return_sequences": num_captions
    }

    out = model.generate(**inputs, **output_options)
    for sequence in out:
        caption = processor.decode(sequence, skip_special_tokens=True)
        captions.append(caption)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs, **output_options)
    for sequence in out:
        caption = processor.decode(sequence, skip_special_tokens=True)
        captions.append(caption)

    return captions

if __name__ == '__main__':
    app.run()
