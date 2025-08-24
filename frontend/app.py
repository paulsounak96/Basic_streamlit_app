import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import torch
from backend.model import FeedForwardNet

@st.cache_resource
def load_model(path='../backend/model.pt'):
    model = FeedForwardNet(20, 64, 2)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

st.title('🔮 Feedforward Net Demo')
model = load_model()

st.write('## Input Features')
features = [st.slider(f'Feature {i}', -5.0, 5.0, 0.0) for i in range(20)]
x = torch.tensor([features], dtype=torch.float32)

if st.button('Predict'):
    with st.spinner('Computing…'):
        preds = model(x).softmax(dim=1)[0]
    st.write('### Class 0 probability:', float(preds[0]))
    st.write('### Class 1 probability:', float(preds[1]))
