import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tempfile
import requests
from io import BytesIO

st.set_page_config(
    page_title="MAE Image Reconstruction",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; }
    .main-header p { font-size: 1.2rem; opacity: 0.9; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1.5rem; border-radius: 10px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border: none;
        width: 100%; border-radius: 5px; padding: 0.5rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

class Patchify(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch_size
        imgs = imgs.reshape(B, C, H//p, p, W//p, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1).contiguous()
        return imgs.reshape(B, (H//p)*(W//p), p*p*C)

def get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    assert embed_dim % 2 == 0
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].flatten())
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].flatten())
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(pos_embed).float()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class RandomMasking(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore, ids_keep

class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 dropout=0.1, num_patches=196, patch_size=16):
        super().__init__()
        self.num_patches = num_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        grid_size = int(num_patches ** 0.5)
        dec_pos = get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.register_buffer('pos_embed', dec_pos.unsqueeze(0))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def forward(self, x, ids_restore):
        B, len_keep, _ = x.shape
        mask_tokens = self.mask_token.repeat(B, self.num_patches - len_keep, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x))

class MaskedAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
                 mask_ratio=0.75, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.patchify = Patchify(patch_size)
        self.patch_embed = nn.Linear(patch_size * patch_size * in_chans, encoder_embed_dim)
        pos = get_2d_sincos_pos_embed(encoder_embed_dim, img_size // patch_size)
        self.register_buffer('pos_embed', pos.unsqueeze(0))
        self.masking = RandomMasking(mask_ratio)
        self.encoder = MAEEncoder(encoder_embed_dim, encoder_depth, encoder_num_heads, mlp_ratio, dropout)
        self.enc_to_dec = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.decoder = MAEDecoder(decoder_embed_dim, decoder_depth, decoder_num_heads,
                                   mlp_ratio, dropout, self.num_patches, patch_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.constant_(self.patch_embed.bias, 0)
        nn.init.xavier_uniform_(self.enc_to_dec.weight)
        nn.init.constant_(self.enc_to_dec.bias, 0)

    def forward(self, imgs):
        x = self.patch_embed(self.patchify(imgs))
        x = x + self.pos_embed
        x_visible, mask, ids_restore, _ = self.masking(x)
        latent = self.encoder(x_visible)
        latent = self.enc_to_dec(latent)
        pred = self.decoder(latent, ids_restore)
        return pred, mask, ids_restore

    def reconstruct_image(self, pred):
        B, N, _ = pred.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        pred = pred.reshape(B, h, w, p, p, 3)
        pred = pred.permute(0, 5, 1, 3, 2, 4).contiguous()
        return pred.reshape(B, 3, h*p, w*p)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedAutoencoder(
        img_size=224, patch_size=16,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
        mask_ratio=0.75
    ).to(device)

    github_url = "https://github.com/Mustehsan-Nisar-Rao/MAE/releases/download/v1/best_model.2.pth"

    try:
        with st.spinner("📥 Downloading model weights..."):
            response = requests.get(github_url, stream=True, timeout=30)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            state_dict = torch.load(tmp_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            os.unlink(tmp_path)
            st.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.warning(f"⚠️ Could not load weights: {str(e)}")
        st.info("Using untrained model for demo.")

    return model, device

def tensor_to_image(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return np.clip(img, 0, 1)

def process_image(image, model, device, mask_ratio):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    model.masking.mask_ratio = mask_ratio / 100.0
    with torch.no_grad():
        pred, mask, _ = model(img_tensor)
    patchify_layer = Patchify(16)
    patches = patchify_layer(img_tensor)
    masked_patches = patches.clone()
    masked_patches[mask.bool()] = 0
    masked_img = model.reconstruct_image(masked_patches)
    recon_img = model.reconstruct_image(pred)
    mask_percentage = (mask.sum().item() / mask.numel()) * 100
    return masked_img, recon_img, img_tensor, mask_percentage

# UI
st.markdown("""
<div class="main-header">
    <h1>🎨 MAE Image Reconstruction</h1>
    <p>Masked Autoencoder for image inpainting and reconstruction</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### 🎭 Mask Settings")
    mask_ratio = st.slider("Mask Ratio (%)", 0, 100, 75, 5,
        help="Percentage of patches to mask")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Encoder**: ViT-Base (12 layers)
    - **Decoder**: ViT-Small (12 layers)
    - **Patch Size**: 16x16
    - **Parameters**: ~107M
    """)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Masked Autoencoder (MAE) for image reconstruction.")

model, device = load_model()

col1, col2 = st.columns([2, 1])
with col2:
    st.markdown("### 📤 Upload Image")
    uploaded_image = st.file_uploader("Choose an image...", type=['jpg','jpeg','png','bmp'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    with col1:
        st.markdown("### 🖼️ Original Image")
        st.image(image, use_container_width=True)

    if st.button("🚀 Run Reconstruction", use_container_width=True):
        with st.spinner("🔄 Processing..."):
            masked_img, recon_img, original_img, mask_percentage = process_image(
                image, model, device, mask_ratio)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mask_ratio}%</div><div class="metric-label">Mask Ratio</div></div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mask_percentage:.1f}%</div><div class="metric-label">Actual Masked</div></div>', unsafe_allow_html=True)
        with col_m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">224x224</div><div class="metric-label">Image Size</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📊 Results")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown("### 🎯 Original")
            st.image(image, use_container_width=True)
        with col_r2:
            st.markdown(f"### 🎭 Masked ({mask_ratio}%)")
            st.image(tensor_to_image(masked_img[0]), use_container_width=True)
        with col_r3:
            st.markdown("### 🔄 Reconstructed")
            st.image(tensor_to_image(recon_img[0]), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Download Results")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            buf = BytesIO()
            image.save(buf, format='PNG')
            st.download_button("📥 Original", buf.getvalue(), "original.png", "image/png", use_container_width=True)
        with col_d2:
            buf = BytesIO()
            Image.fromarray((tensor_to_image(masked_img[0]) * 255).astype(np.uint8)).save(buf, format='PNG')
            st.download_button("📥 Masked", buf.getvalue(), f"masked_{mask_ratio}.png", "image/png", use_container_width=True)
        with col_d3:
            buf = BytesIO()
            Image.fromarray((tensor_to_image(recon_img[0]) * 255).astype(np.uint8)).save(buf, format='PNG')
            st.download_button("📥 Reconstructed", buf.getvalue(), "reconstructed.png", "image/png", use_container_width=True)
else:
    st.markdown(f"""
    <div class="info-box">
        <h3>👆 Upload an image to get started</h3>
        <p>The MAE model will mask {mask_ratio}% of patches and reconstruct them.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>Built with Streamlit • Masked Autoencoder • Facebook AI Research</div>", unsafe_allow_html=True)
