#!/bin/bash

# Crie pastas necessárias
mkdir -p /workspace/ComfyUI
cd /workspace

# Clone do ComfyUI (se ainda não existir)
if [ ! -d "ComfyUI" ]; then
    echo "📦 Clonando ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git  /workspace/ComfyUI
fi

# Ative o ambiente virtual do Jupyter
echo "🔌 Ativando ambiente Conda..."
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Instale dependências básicas
echo "🧰 Instalando PyTorch e dependências principais..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 
pip install -r /workspace/ComfyUI/requirements.txt

# --- Baixe modelos para o Sonic Lip Sync ---
SONIC_MODEL_DIR="/workspace/ComfyUI/models/sonic"
mkdir -p "$SONIC_MODEL_DIR"

echo "🎵 Baixando modelos do Sonic..."

wget -nc -O "$SONIC_MODEL_DIR/unet.pth" \
  "https://huggingface.co/smthemex/Sonic/raw/main/unet.pth" 

wget -nc -O "$SONIC_MODEL_DIR/audio2bucket.pth" \
  "https://huggingface.co/smthemex/Sonic/raw/main/audio2bucket.pth" 

wget -nc -O "$SONIC_MODEL_DIR/audio2token.pth" \
  "https://huggingface.co/smthemex/Sonic/raw/main/audio2token.pth" 

# RIFE (flownet.pkl) - usado para interpolação de frames
mkdir -p "/workspace/ComfyUI/models/sonic/RIFE"
RIFE_PATH="/workspace/ComfyUI/models/sonic/RIFE/flownet.pkl"

if [ ! -f "$RIFE_PATH" ]; then
    echo "🚀 Baixando flownet.pkl (RIFE)..."
    wget -O "$RIFE_PATH" "https://huggingface.co/LeonJoe13/Sonic/resolve/main/RIFE/flownet.pkl" 
fi

# Whisper-tiny (para processamento de áudio)
WHISPER_DIR="/workspace/ComfyUI/models/sonic/whisper-tiny"
mkdir -p "$WHISPER_DIR"

echo "🎤 Baixando modelos Whisper-tiny..."
wget -nc -O "$WHISPER_DIR/model.safetensors" \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors" 

wget -nc -O "$WHISPER_DIR/preprocessor_config.json" \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json" 

wget -nc -O "$WHISPER_DIR/config.json" \
  "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json" 


# --- Instale o custom node: VideoHelperSuite (VHS) ---
VHS_DIR="/workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"
if [ ! -d "$VHS_DIR" ]; then
    echo "📼 Instalando ComfyUI-VideoHelperSuite..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 
    cd ComfyUI-VideoHelperSuite
    pip install -r requirements.txt
else
    echo "⏩ ComfyUI-VideoHelperSuite já instalado."
fi


# --- Instale o custom node: Sonic Lip Sync ---
SONIC_DIR="/workspace/ComfyUI/custom_nodes/ComfyUI_Sonic"
if [ ! -d "$SONIC_DIR" ]; then
    echo "🔊 Instalando ComfyUI_Sonic..."
    cd /workspace/ComfyUI/custom_nodes
    git clone https://github.com/smthemex/ComfyUI_Sonic.git 
    cd ComfyUI_Sonic
    pip install -r requirements.txt
else
    echo "⏩ ComfyUI_Sonic já instalado."
fi


# --- Baixe modelos Stable Video Diffusion (opcional) ---
CKPT_DIR="/workspace/ComfyUI/models/checkpoints"
mkdir -p "$CKPT_DIR"

echo "🎬 Baixando modelos SVD..."
wget -nc -O "$CKPT_DIR/svd_xt_1_1.safetensors" \
  "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors" 

wget -nc -O "$CKPT_DIR/svd_xt.safetensors" \
  "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/svd_xt.safetensors" 

wget -nc -O "$CKPT_DIR/svd_xt_image_decoder.svd_xt_image_decoder.safetensors" \
  "https://huggingface.co/FrancisRing/StableAnimator/resolve/main/stable-video-diffusion-img2vid-xt/svd_xt_image_decoder.safetensors" 


# Reinicie o ComfyUI
echo "🔁 Iniciando ComfyUI..."
cd /workspace/ComfyUI
python main.py --port 8188 --host 0.0.0.0
