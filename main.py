import os
from PIL import Image
import requests
from io import BytesIO
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torch
from torch import nn
from flask import Flask, request

def load_picture(link):
    image_url = "https://" + link
    response = requests.get(image_url)

    try:
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data)).convert("RGB")
            return image
        else:
            raise Exception
    except Exception as e:
        raise f"ошибка: {e}"


def make_prediction(link, model, device):

    LABELS = {
        0: "clean_photo",
        1: "infographics"
    }

    image = load_picture(link)
    transform = VGG16_BN_Weights.DEFAULT.transforms()
    image = transform(image).to(device).unsqueeze(0)

    model.eval()
    logits = model(image)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    return LABELS[predicted_class]

def get_model(device):
    model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 100),
        nn.Softmax(),
        nn.Linear(100, 2)
    )

    model.load_state_dict(torch.load('./models/model_gr_or_ph.pth', map_location=torch.device(device)))
    model = model.to(device)
    return model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello!"

@app.route('/predict/')
def model_prediction():

    link = request.args.get('link')
    try:
        label = make_prediction(link, model, device)
        return label
    except Exception as e:
        return f"Неверная ссылка, ошибка: \n {e}"


if __name__ == '__main__':
    print("START SERVER")
    app.run(host='0.0.0.0', port=2345)

