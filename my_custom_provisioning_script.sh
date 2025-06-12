#!/bin/bash

# Define o diretório persistente para Vast.ai
PERSISTENT_DIR="/workspace"
cd "$PERSISTENT_DIR"

# Causa o script a sair em caso de falha de qualquer comando.
set -eo pipefail

echo "Iniciando provisionamento personalizado para Vast.ai (com organização de diretórios)..."

# --- Ativação do Ambiente Python (Prioriza venv, depois Conda) ---
echo "Tentando ativar ambiente Python..."
if [ -f "/venv/main/bin/activate" ]; then
    . /venv/main/bin/activate
    echo "Ambiente venv '/venv/main' ativado."
elif CONDA_BASE_PATH=$(conda info --base 2>/dev/null); then
    source "$CONDA_BASE_PATH"/etc/profile.d/conda.sh
    echo "Conda base path: $CONDA_BASE_PATH"
    if conda activate comfy; then
        echo "Ambiente 'comfy' ativado."
    elif conda activate base; then
        echo "Ambiente 'base' ativado."
    else
        echo "Nenhum ambiente Conda detectado ou ativado. Usando ambiente de sistema para pip."
    fi
else
    echo "Nenhum ambiente venv ou Conda detectado. Usando ambiente de sistema para pip."
fi

# --- Clonar/Atualizar ComfyUI ---
COMFYUI_DIR="$PERSISTENT_DIR/ComfyUI"
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "ComfyUI não encontrado em $COMFYUI_DIR. Clonando..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    git config --global --add safe.directory "$COMFYUI_DIR"
    echo "ComfyUI clonado."
fi

echo "Forçando atualização do ComfyUI via git pull e pip install..."
cd "$COMFYUI_DIR"
git config --global --add safe.directory "$(pwd)"
git config pull.rebase false
git pull origin master

# Instalação das dependências PyTorch com CUDA (cu124 - ajustado para o log)
echo "Instalando PyTorch com CUDA (cu124 - ajustado para o log)..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124 || \
echo "Aviso: Falha na instalação de PyTorch com cu124. Verifique a compatibilidade CUDA."

echo "Instalando requisitos base do ComfyUI e pacotes adicionais..."
pip install -r requirements.txt --no-cache-dir --upgrade --force-reinstall
pip install bitsandbytes>=0.43.0 gguf --upgrade

echo "Limpeza de dependências: 'pip autoremove' não disponível, pulando."



# --- 3. Instalação e Configuração do Sonic (ComfyUI_Sonic) e seus Modelos ---
echo "Instalando ComfyUI_Sonic e baixando modelos necessários..."
SONIC_NODE_DIR="$PERSISTENT_DIR/ComfyUI_Sonic" # Diretório do Custom Node
if [ ! -d "$SONIC_NODE_DIR" ]; then
    git clone --recursive https://github.com/smthemex/ComfyUI_Sonic.git "$SONIC_NODE_DIR" # --recursive para submodules
fi
git config --global --add safe.directory "$SONIC_NODE_DIR"
cd "$SONIC_NODE_DIR"
pip install -r requirements.txt --no-cache-dir

echo "Instalando gdown para download de arquivos do Google Drive..."
pip install gdown

# --- Baixar Modelos do Sonic para ComfyUI/models/sonic/ ---
# Esta é a pasta de organização solicitada para os modelos do Sonic
SONIC_MODELS_DIR="$COMFYUI_DIR/models/sonic"
mkdir -p "$SONIC_MODELS_DIR"
echo "Baixando arquivos para $SONIC_MODELS_DIR (unet.pth, audio2token.pth, audio2bucket.pth, yoloface_v5m.pt)..."
gdown --id 1mjIqU-c5q3qMI74XZd3UrkZek0IDTUUh -O "$SONIC_MODELS_DIR/unet.pth" || echo "unet.pth já existe ou falhou ao baixar."
gdown --id 1vUY-b5NMvDA2XsxRZcB3nF3u1trOtK53h -O "$SONIC_MODELS_DIR/audio2token.pth" || echo "audio2token.pth já existe ou falhou ao baixar."
gdown --id 1RHWasbgUWZg-mFaQhDJtF1KhpUSecC5d -O "$SONIC_MODELS_DIR/audio2bucket.pth" || echo "audio2bucket.pth já existe ou falhou ao baixar."
gdown --id 13Hpfi-cBvlmNvTv6W4Oa7agWyzmvmofB4 -O "$SONIC_MODELS_DIR/yoloface_v5m.pt" || echo "yoloface_v5m.pt já existe ou falhou ao baixar."

# --- RIFE (flownet.pkl) - **NOVO LOCAL: ComfyUI/models/RIFE** ---
# Este é o local padrão onde os Custom Nodes de RIFE geralmente procuram.
RIFE_DIR="$COMFYUI_DIR/models/sonic/RIFE"
mkdir -p "$RIFE_DIR"
echo "Baixando flownet.pkl para $RIFE_DIR..."
gdown --id 1UnSd-s5DhPRZu4C23I4uOmmahH0J3Dkwl -O "$RIFE_DIR/flownet.pkl" || echo "flownet.pkl já existe ou falhou ao baixar."

# --- Baixar Modelos Whisper (para transcrição) ---
# Os modelos Whisper (tiny.en, base.en, etc.) são gerenciados pela biblioteca openai-whisper
# e baixados para ~/.cache/whisper ou $HF_HOME.
# O ComfyUI_Sonic pode precisar de configuração ou links simbólicos para encontrar esses modelos
# se eles não estiverem no local que ele espera.
WHISPER_CACHE_DIR="$PERSISTENT_DIR/.cache/whisper"
echo "Limpando cache de modelos Whisper em $WHISPER_CACHE_DIR..."
rm -rf "$WHISPER_CACHE_DIR"
mkdir -p "$WHISPER_CACHE_DIR"

export HF_HOME="$PERSISTENT_DIR/.cache/huggingface" # Garante que Hugging Face baixe para diretório persistente
mkdir -p "$HF_HOME"

echo "Baixando modelo OpenAI Whisper 'tiny.en' (forçando download via biblioteca)..."
pip install git+https://github.com/openai/whisper.git # Instala a biblioteca Whisper do GitHub
python -c "import whisper; print('Downloading Whisper tiny.en model...'); whisper.load_model('tiny.en'); print('Whisper tiny.en model downloaded successfully.');" 2>&1 | tee "$PERSISTENT_DIR/whisper_download_log.txt"

WHISPER_MODEL_PATH="$WHISPER_CACHE_DIR/tiny.en.pt"
if [ -f "$WHISPER_MODEL_PATH" ] && [ -s "$WHISPER_MODEL_PATH" ]; then
    echo "Verificação: Modelo Whisper tiny.en encontrado e não está vazio em $WHISPER_MODEL_PATH."
    file "$WHISPER_MODEL_PATH"
else
    echo "ATENÇÃO: Modelo Whisper tiny.en NÃO encontrado ou está vazio em $WHISPER_MODEL_PATH. O download pode ter falhado."
    echo "Verifique o log de download em $PERSISTENT_DIR/whisper_download_log.txt para mais detalhes."
fi

# --- Baixar Modelo Whisper-tiny (do Hugging Face) ---
# Este modelo será baixado para $HF_HOME (que está em $PERSISTENT_DIR/.cache/huggingface)
echo "Baixando modelo Whisper-tiny (openai/whisper-tiny) do Hugging Face via biblioteca transformers..."
WHISPER_TINY_HF_MODEL_NAME="openai/whisper-tiny"
pip install transformers # Garante que transformers esteja instalado para este download
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    processor = WhisperProcessor.from_pretrained('$WHISPER_TINY_HF_MODEL_NAME'); \
    model = WhisperForConditionalGeneration.from_pretrained('$WHISPER_TINY_HF_MODEL_NAME'); \
    print(f'Modelo $WHISPER_TINY_HF_MODEL_NAME baixado e carregado com sucesso.')" 2>&1 | tee "$PERSISTENT_DIR/whisper_tiny_hf_download_log.txt"

echo "ComfyUI_Sonic e modelos necessários instalados."



# --- Instalação do VideoHelperSuite (VHS com vídeo combine) ---
# Este é um Custom Node, seu lugar é em custom_nodes.
echo "Configurando VideoHelperSuite..."
VIDEO_HELPER_SUITE_DIR="$COMFYUI_DIR/VideoHelperSuite"
mkdir -p "$VIDEO_HELPER_SUITE_DIR"
if [ ! -d "$VIDEO_HELPER_SUITE_DIR" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$VIDEO_HELPER_SUITE_DIR"
fi
git config --global --add safe.directory "$VIDEO_HELPER_SUITE_DIR" # Adiciona à lista segura do Git
cd "$VIDEO_HELPER_SUITE_DIR"
pip install -r requirements.txt --no-cache-dir
echo "VideoHelperSuite configurado."

# --- 4. Baixar Modelos Essenciais (SDXL, SVD, Wan-AI/Wan2.1-T2V-14B) ---
echo "Baixando modelos essenciais para ComfyUI..."
COMFYUI_CHECKPOINTS_DIR="$COMFYUI_DIR/models/checkpoints" # LOCAL PADRÃO
COMFYUI_SVD_DIR="$COMFYUI_DIR/models/svd" # LOCAL PADRÃO
mkdir -p "$COMFYUI_CHECKPOINTS_DIR"
mkdir -p "$COMFYUI_SVD_DIR"

# Stable Diffusion XL Base (para Text-to-Video e Image-to-Image/Video)
echo "Baixando Stable Diffusion XL Base..."
wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/sd_xl_base_1.0.safetensors" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" || echo "SDXL Base model already exists ou falhou ao baixar."

# SVD (Stable Video Diffusion) - Image-to-Video
echo "Baixando Stable Video Diffusion SVD_XT_1_1..."
if [ -z "$TOKEN_HF" ]; then
    echo "AVISO: Variável de ambiente TOKEN_HF não definida. Não será possível baixar modelos Hugging Face privados ou com Gated Access."
else
    wget -nc --header="Authorization: Bearer $TOKEN_HF" -O "$COMFYUI_SVD_DIR/svd_xt_1_1.safetensors" \
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors" || \
    echo "SVD_XT_1_1 já existe ou download falhou (verifique token/rede)."

    wget -nc --header="Authorization: Bearer $TOKEN_HF" -O "$COMFYUI_SVD_DIR/svd_xt_image_decoder.safetensors" \
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors" || \
    echo "SVD_XT_Image_Decoder já existe ou download falhou (verifique token/rede)."
fi


echo "Provisionamento personalizado concluído."
