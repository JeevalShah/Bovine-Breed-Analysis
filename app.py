import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------------------------------
# Config
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnet_finetuned.pth"
CLASSES = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss',
           'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian',
           'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh',
           'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori',
           'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane',
           'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # match val_test_transforms
        transforms.ToTensor(),          # same as training
        # NO normalization! Training didn't normalize
    ])
    return transform(image).unsqueeze(0)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🐮 Bovine Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probs, k=3)

    st.write("### Top 3 Predictions:")
    for i in range(3):
        st.write(f"{CLASSES[top3_idx[0][i].item()]}: {top3_prob[0][i].item():.4f}")
