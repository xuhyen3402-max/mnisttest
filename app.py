import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# ===== 1. 모델 정의 (학습할 때와 동일한 구조) =====
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ===== 2. 모델 로드 (@st.cache_resource로 1회만 실행) =====
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
    model.eval()
    return model

# ===== 3. 이미지 전처리 (MNIST 형식에 맞게 변환) =====
def preprocess_image(image):
    """업로드된 이미지를 MNIST 입력 형식(1, 1, 28, 28)으로 변환"""
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 흑백 변환
        transforms.Resize((28, 28)),                   # 28x28 크기 조정
        transforms.ToTensor(),                         # 텐서 변환 (0~1)
    ])
    image = TF.invert(image)               # 색상 반전 (흰 바탕→검은 바탕)
    return preprocess(image).unsqueeze(0)   # 배치 차원 추가 (1, 1, 28, 28)

# ===== 4. Streamlit UI =====
st.set_page_config(page_title="MNIST 숫자 인식", page_icon="✍️")
st.title("✍️ MNIST 숫자 인식")
st.write("숫자가 적힌 이미지를 업로드하면 AI가 판독합니다.")

model = load_model()

uploaded = st.file_uploader(
    "이미지를 선택하세요",
    type=["png", "jpg", "jpeg"],
    help="0~9 숫자가 적힌 이미지를 업로드하세요"
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("업로드 이미지")
        st.image(image, width=200)

    tensor = preprocess_image(image)
    with col2:
        st.subheader("전처리 결과")
        st.image(tensor.squeeze().numpy(), width=200, caption="28x28 흑백 변환")

    with torch.inference_mode():
        output = model(tensor)
        probabilities = torch.exp(output)   # log_softmax → 확률 변환
        predicted = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted].item()

    st.divider()
    st.success(f"### 판독 결과: **{predicted}**  (확신도: {confidence:.1%})")

    st.subheader("숫자별 확률")
    prob_data = probabilities[0].detach().numpy()
    st.bar_chart({str(i): float(prob_data[i]) for i in range(10)})
else:
    st.info("👆 위 버튼을 클릭하여 숫자 이미지를 업로드하세요.")
    st.markdown("""
    **참고:**
    - 흰 바탕에 검은 숫자로 쓴 이미지가 가장 잘 인식됩니다
    - 손글씨, 인쇄체 모두 가능합니다
    - 이 모델은 MNIST 데이터셋(28x28 흑백)으로 학습되었습니다
    """)