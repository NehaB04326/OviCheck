import torch
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load your trained PyTorch model
model = torch.load('model.pth', map_location=torch.device('cpu'))  # Load model to CPU
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of your model
    transforms.ToTensor(),           # Convert image to tensor
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': 'No image provided.'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()  # Assuming binary classification
        result = 'Detected' if prediction > 0.5 else 'Not Detected'
        confidence = prediction  # Confidence score
        tips = "Consult a healthcare provider for further advice."  # Example tips

    return jsonify({'result': result, 'confidence': confidence, 'tips': tips})

if __name__ == '__main__':
    app.run(debug=True)
