import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import io

app = Flask(__name__)

# Load your trained PyTorch model
model = torch.load('model.pth')
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of your model
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': 'No image provided.'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        # Assuming output is a single value for binary classification
        prediction = torch.sigmoid(output).item()
        confidence = prediction  # For binary classification
        result = 'Detected' if prediction > 0.5 else 'Not Detected'
        tips = "Consult a healthcare provider for further advice."  # Example tips

    return jsonify({'result': result, 'confidence': confidence, 'tips': tips})

if __name__ == '__main__':
    app.run(debug=True)
