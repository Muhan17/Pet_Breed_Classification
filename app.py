import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Загрузка сохраненной полной модели
model_path = 'improved_model_full.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = torch.load(model_path, map_location=device)
model.eval()  # Переводим модель в режим оценки

# Определите трансформации для обработки изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Список классов (породы животных)
classes = [
    'abyssinian', 'american shorthair', 'beagle', 'boxer', 'bulldog',
    'chihuahua', 'corgi', 'dachshund', 'german shepherd', 'golden retriever',
    'husky', 'labrador', 'maine coon', 'mumbai cat', 'persian cat',
    'pomeranian', 'pug', 'ragdoll cat', 'rottwiler', 'shiba inu',
    'siamese cat', 'sphynx', 'yorkshire terrier'
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

