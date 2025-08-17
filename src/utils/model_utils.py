import streamlit as st
import torch
from models.src.predict_model import LatentOHLCVRNN, LatentOHLCVLSTM
from models.src.VAE_trainer import TransformerVAE_TDist
from models.src.dataloader_setup import LatentOHLCVPredictor


@st.cache_resource
def load_vae_model(vae_path, feature_dim, latent_dim, embed_dim, df, device):
    model = TransformerVAE_TDist(feature_dim=feature_dim, latent_dim=latent_dim, embed_dim=embed_dim, df=df).to(device)
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


# --- 【关键修改 A】: load_predictor_model 现在不再依赖于动态的 seq_length ---
# 我们把它从@st.cache_resource中移除，因为它现在依赖于固定的超参数，或者确保它的参数在一次运行中是恒定的
# 为了避免缓存冲突，最简单的方法是确保传入的seq_length是恒定的
@st.cache_resource
def load_predictor_model(model_name, predictor_path, device):
    # 使用固定的seq_length来初始化模型
    new_model = None
    if model_name == "rnn":
        new_model = LatentOHLCVRNN().to(device)
    elif model_name == "lstm":
        new_model = LatentOHLCVLSTM().to(device)
    else:
        new_model = LatentOHLCVPredictor().to(device)

    model = new_model
    model.load_state_dict(torch.load(predictor_path, map_location=device))
    model.eval()
    return model