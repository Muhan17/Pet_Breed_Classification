import streamlit as st
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms

# Создайте архитектуру модели
model = models.resnet18(pretrained=False)  # Создаем базовую архитектуру ResNet18
model.fc = torch.nn.Linear(model.fc.in_features, 23)  # Укажите число классов (например, 23)

# Загрузите сохранённые веса
model.load_state_dict(torch.load('improved_model.pth', map_location=torch.device('cpu')))
model.eval()  # Перевод модели в режим оценки

# Определите трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Streamlit интерфейс
st.title("Pet Breed Classifier")
st.write("Upload an image of a pet, and the model will classify its breed!")

# Загрузка изображения
uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открыть изображение
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Преобразовать изображение для модели
    img_tensor = transform(image).unsqueeze(0)  # Добавить batch размер

    # Выполнить предсказание
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Вероятности классов
        top_probs, top_indices = torch.topk(probabilities, 3)  # Топ-3 вероятности

    # Список классов
    breeds = [
        'abyssinian', 'american shorthair', 'beagle', 'boxer', 'bulldog',
        'chihuahua', 'corgi', 'dachshund', 'german shepherd', 'golden retriever',
        'husky', 'labrador', 'maine coon', 'mumbai cat', 'persian cat',
        'pomeranian', 'pug', 'ragdoll cat', 'rottwiler', 'shiba inu',
        'siamese cat', 'sphynx', 'yorkshire terrier'
    ]

    # Отобразить результаты
    st.subheader("Top-3 Predictions:")
    for i in range(3):
        st.write(f"{breeds[top_indices[i]]}: {top_probs[i].item() * 100:.2f}%")
