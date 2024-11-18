import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Загрузка сохраненной модели
model_path = 'improved_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Создайте модель с той же архитектурой, что и обученная модель
model = models.resnet50(pretrained=False)  # Используем ResNet50, если это была ваша основа
model.fc = torch.nn.Linear(model.fc.in_features, 37)  # 37 классов для Oxford Pets Dataset
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Определите трансформации для обработки изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Список классов (породы животных)
classes = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Egyptian_Mau", 
    "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx",
    # Добавьте оставшиеся классы...
]

# Интерфейс Streamlit
st.title("Pet Breed Classifier")
st.write("Upload an image of a pet, and the model will classify its breed along with probabilities.")

# Загрузка изображения
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Обрабатываем изображение для модели
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Отображаем результаты
    top5_prob, top5_classes = torch.topk(probabilities, 5)
    st.write("### Predictions:")
    for i in range(5):
        st.write(f"{classes[top5_classes[i]]}: {top5_prob[i].item() * 100:.2f}%")
