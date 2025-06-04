# app/model_loader.py
import torch
import torchvision.transforms as transforms
from PIL import Image

def get_season(img_path: str) -> int:
    model = torch.load("best_model_resnet_ALL.pt", map_location="cpu")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    return output.argmax().item()
