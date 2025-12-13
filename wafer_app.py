import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class WaferFault(nn.Module):
    def __init__(self, num_classes=9):
        super(WaferFault, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


@st.cache_resource
def load_model():
    model=WaferFault(num_classes=9)
    model.load_state_dict(torch.load("wafer_fault_Detect/wafer_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model=load_model()
class_names = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none']

st.title("Wafer Fault Detection")
st.write("Upload a wafer map image to detect defects.")

file=st.file_uploader("Choose an image:", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Wafer Map', width=300)
    
    img_tensor=transform(image).unsqueeze(0) 
    
    if st.button("Analyze Defect"):
        with torch.no_grad():
            outputs=model(img_tensor)
            _,predicted=torch.max(outputs,1)
            confidence=torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

        result=class_names[predicted.item()]
 
        if result=="none":
            color="green"
        else:
            color="red"
        
        st.markdown(f"### Predicted Defect: <span style='color:{color}'>{result}</span>",unsafe_allow_html=True)
        st.write(f"Confidence: {confidence*100:.2f}%")