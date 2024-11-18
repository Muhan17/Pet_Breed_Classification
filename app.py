import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Load trained model
model = torch.load('improved_model.pth', map_location=torch.device('cpu'))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List of breeds
breeds = [
    'abyssinian', 'american shorthair', 'beagle', 'boxer', 'bulldog',
    'chihuahua', 'corgi', 'dachshund', 'german shepherd', 'golden retriever',
    'husky', 'labrador', 'maine coon', 'mumbai cat', 'persian cat',
    'pomeranian', 'pug', 'ragdoll cat', 'rottwiler', 'shiba inu',
    'siamese cat', 'sphynx', 'yorkshire terrier'
]

# Streamlit app title and description
st.title("Pet Breed Classifier")
st.write(
    "Upload an image of a dog or cat, and the model will predict the breed with probabilities!"
)

# Upload image
uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top-3 predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_breeds = [breeds[idx] for idx in top_indices.tolist()]
    top_probs = top_probs.tolist()

    # Display top-3 results
    st.subheader("Top-3 Predictions:")
    for breed, prob in zip(top_breeds, top_probs):
        st.write(f"**{breed}**: {prob * 100:.2f}%")
