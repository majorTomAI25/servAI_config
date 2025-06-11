#!/bin/bash

# Define o diretório persistente para Vast.ai
PERSISTENT_DIR="/workspace"
cd "$PERSISTENT_DIR"

# Causa o script a sair em caso de falha de qualquer comando
set -eo pipefail

echo "Iniciando provisionamento personalizado para Vast.ai (adaptado do Kaggle)..."

# --- 1. Verificação e Ativação do Ambiente Python (Crucial!) ---
# Tenta detectar e ativar o ambiente Conda.
CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE_PATH" ]; then
    source "$CONDA_BASE_PATH"/etc/profile.d/conda.sh
    echo "Conda base path: $CONDA_BASE_PATH"
    if conda activate comfy; then
        echo "Ambiente 'comfy' ativado."
    elif conda activate base; then
        echo "Ambiente 'base' ativado (ambiente 'comfy' não encontrado)."
    else
        echo "Nenhum ambiente Conda detectado ou ativado. Verifique a instalação do Conda."
    fi
else
    echo "Conda não encontrado no sistema. Assumindo ambiente de sistema para pip."
fi

# Verifica se o ComfyUI existe. Se não, clona.
COMFYUI_DIR="$PERSISTENT_DIR/ComfyUI"
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "ComfyUI não encontrado em $COMFYUI_DIR. Clonando..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    echo "ComfyUI clonado."
fi

# **FORÇAR ATUALIZAÇÃO DO COMFYUI AQUI**
echo "Forçando atualização do ComfyUI via git pull e pip install..."
cd "$COMFYUI_DIR"
git config pull.rebase false
git pull origin master # Garante que puxa da branch master, que é a mais atual

# Instalação das dependências PyTorch com CUDA (VERIFIQUE SUA VERSÃO CUDA: cu124 ou cu128)
echo "Instalando PyTorch com CUDA (cu128 - ajuste se sua GPU for cu124)..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 || \
echo "Aviso: Falha na instalação de PyTorch com cu128. Verifique a compatibilidade CUDA."

echo "Instalando requisitos base do ComfyUI e pacotes adicionais..."
pip install -r requirements.txt --no-cache-dir --upgrade --force-reinstall
pip install bitsandbytes>=0.43.0 gguf --upgrade # bitsandbytes e gguf

# Remove dependências Python não utilizadas
echo "Removendo dependências Python não utilizadas..."
pip autoremove -y
echo "Limpeza de dependências concluída."

# --- 2. Instalação de CUSTOM NODES (do Template Kaggle) ---
echo "Instalando Custom Nodes para ComfyUI..."
COMFYUI_CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
mkdir -p "$COMFYUI_CUSTOM_NODES_DIR"
cd "$COMFYUI_CUSTOM_NODES_DIR"

echo "Clonando ComfyUI-Manager..."
if [ ! -d "ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ComfyUI-Manager
    pip install -r requirements.txt --no-cache-dir
    cd ..
else
    echo "ComfyUI-Manager já existe, atualizando..."
    cd ComfyUI-Manager
    git pull
    pip install -r requirements.txt --no-cache-dir
    cd ..
fi

echo "Clonando cg-use-everywhere..."
if [ ! -d "cg-use-everywhere" ]; then
    git clone https://github.com/chrisgoringe/cg-use-everywhere.git
fi

echo "Clonando ComfyUI-Custom-Scripts..."
if [ ! -d "ComfyUI-Custom-Scripts" ]; then
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
fi

echo "Clonando was-node-suite-comfyui..."
if [ ! -d "was-node-suite-comfyui" ]; then
    git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
fi

echo "Clonando rgthree-comfy..."
if [ ! -d "rgthree-comfy" ]; then
    git clone https://github.com/rgthree/rgthree-comfy.git
fi

echo "Clonando ComfyUI-GGUF..."
if [ ! -d "ComfyUI-GGUF" ]; then
    git clone https://github.com/city96/ComfyUI-GGUF
fi

echo "Clonando ComfyUI-Crystools..."
if [ ! -d "ComfyUI-Crystools" ]; then
    git clone https://github.com/crystian/ComfyUI-Crystools.git
    cd ComfyUI-Crystools
    pip install -r requirements.txt --no-cache-dir
    cd ..
else
    echo "ComfyUI-Crystools já existe, atualizando..."
    cd ComfyUI-Crystools
    git pull
    pip install -r requirements.txt --no-cache-dir
    cd ..
fi

echo "Clonando ComfyUI-KJNodes..."
if [ ! -d "ComfyUI-KJNodes" ]; then
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
fi

echo "Clonando comfyui-workspace-manager..."
if [ ! -d "comfyui-workspace-manager" ]; then
    git clone https://github.com/11cafe/comfyui-workspace-manager.git
fi

# Instalar requisitos de custom_nodes que possam ter um requirements.txt em custom_nodes (se existir)
# Atenção: Esta linha foi movida para o final da seção de custom nodes, como no seu template original Kaggle,
# mas `pip install -r` só funciona se o requirements.txt estiver no diretório atual.
# Muitos custom nodes têm seus próprios requirements.txt dentro de seus subdiretórios, que são instalados ao cloná-los e cd para eles.
# Esta linha é um fallback se houver um requirements.txt genérico em $COMFYUI_CUSTOM_NODES_DIR
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --no-cache-dir
fi
echo "Custom Nodes instalados e atualizados."

# --- 3. Instalação e Configuração do Sonic (ComfyUI_Sonic) ---
echo "Instalando ComfyUI_Sonic e baixando modelos necessários..."
SONIC_DIR="$PERSISTENT_DIR/ComfyUI_Sonic"
if [ ! -d "$SONIC_DIR" ]; then
    git clone --recursive https://github.com/smthemex/ComfyUI_Sonic.git "$SONIC_DIR" # --recursive para submodules
fi
cd "$SONIC_DIR"
pip install -r requirements.txt --no-cache-dir

# Instalação do FFmpeg (se não estiver na imagem base)
# Assumimos que a imagem Vast.ai/comfy já tem ffmpeg. Se não, adicione:
# echo "Verificando e instalando FFmpeg..."
# sudo apt-get update && sudo apt-get install -y ffmpeg

# --- Baixar Modelo UNET do Google Drive para Sonic ---
echo "Instalando gdown para download de arquivos do Google Drive..."
pip install gdown
mkdir -p "$COMFYUI_DIR/models/sonic" # Cria a pasta models/sonic dentro do ComfyUI
echo "Baixando unet.pth para ComfyUI/models/sonic..."
gdown --id 1mjIqU-c5q3qMI74XZd3UrkZek0IDTUUh -O "$COMFYUI_DIR/models/sonic/unet.pth" || echo "unet.pth já existe ou falhou ao baixar."

# --- Baixar Outros Arquivos do Google Drive para Sonic ---
# Certifique-se de que esses IDs e nomes de arquivo estão corretos e que as pastas existem
# Exemplo: audio2token.pth, audio2bucket.pth, yoloface_v5m.pt
echo "Baixando arquivos adicionais para ComfyUI/models/sonic..."
gdown --id 1vUY-b5NMvDA2XsxRZcB3nF3u1trOtK53h -O "$COMFYUI_DIR/models/sonic/audio2token.pth" || echo "audio2token.pth já existe ou falhou ao baixar."
gdown --id 1RHWasbgUWZg-mFaQhDJtF1KhpUSecC5d  -O "$COMFYUI_DIR/models/sonic/audio2bucket.pth" || echo "audio2bucket.pth já existe ou falhou ao baixar."
gdown --id 13Hpfi-cBvlmNvTv6W4Oa7agWyzmvmofB4 -O "$COMFYUI_DIR/models/sonic/yoloface_v5m.pt" || echo "yoloface_v5m.pt já existe ou falhou ao baixar."

# --- Baixar Modelos Whisper (para transcrição) ---
WHISPER_CACHE_DIR="$PERSISTENT_DIR/.cache/whisper"
echo "Limpando cache de modelos Whisper em $WHISPER_CACHE_DIR..."
rm -rf "$WHISPER_CACHE_DIR" # Remove o diretório inteiro
mkdir -p "$WHISPER_CACHE_DIR" # Recria o diretório

export HF_HOME="$PERSISTENT_DIR/.cache/huggingface" # Garante que Hugging Face baixe para diretório persistente
mkdir -p "$HF_HOME"

echo "Baixando modelo OpenAI Whisper 'tiny.en' (forçando download)..."
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

# --- RIFE (flownet.pkl) ---
RIFE_DIR="$COMFYUI_DIR/models/RIFE"
mkdir -p "$RIFE_DIR"
echo "Baixando flownet.pkl para ComfyUI/models/RIFE..."
gdown --id 1UnSd-s5DhPRZu4C23I4uOmmahH0J3Dkwl -O "$RIFE_DIR/flownet.pkl" || echo "flownet.pkl já existe ou falhou ao baixar."

# --- Baixar Modelos Whisper (para transcrição) ---
# Limpa qualquer cache de modelos Whisper existente antes de tentar baixar novamente.
# O diretório de cache padrão do Whisper é ~/.cache/whisper
WHISPER_CACHE_DIR="$PERSISTENT_DIR/.cache/whisper" # Assumindo ~/.cache é mapeado para PERSISTENT_DIR/.cache
echo "Limpando cache de modelos Whisper em $WHISPER_CACHE_DIR..."
rm -rf "$WHISPER_CACHE_DIR" # Remove o diretório inteiro
mkdir -p "$WHISPER_CACHE_DIR" # Recria o diretório

# Define a variável de ambiente HF_HOME para garantir que os modelos sejam baixados para o diretório persistente
export HF_HOME="$PERSISTENT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "Baixando modelo OpenAI Whisper 'tiny.en' (forçando download via biblioteca)..."
pip install git+https://github.com/openai/whisper.git # Instala a biblioteca Whisper do GitHub
python -c "import whisper; print('Downloading Whisper tiny.en model...'); whisper.load_model('tiny.en')" 2>&1 | tee "$PERSISTENT_DIR/whisper_download_log.txt"

WHISPER_MODEL_PATH="$WHISPER_CACHE_DIR/tiny.en.pt"
if [ -f "$WHISPER_MODEL_PATH" ] && [ -s "$WHISPER_MODEL_PATH" ]; then
    echo "Verificação: Modelo Whisper tiny.en encontrado e não está vazio em $WHISPER_MODEL_PATH."
    file "$WHISPER_MODEL_PATH"
else
    echo "ATENÇÃO: Modelo Whisper tiny.en NÃO encontrado ou está vazio em $WHISPER_MODEL_PATH. O download pode ter falhado."
    echo "Verifique o log de download em $PERSISTENT_DIR/whisper_download_log.txt para mais detalhes."
fi

# --- **NOVA SEÇÃO: Whisper-tiny (do Hugging Face) ---
# Esta seção foi atualizada para puxar o modelo diretamente via a biblioteca transformers,
# que é a forma recomendada para modelos do Hugging Face.
echo "Baixando modelo Whisper-tiny (openai/whisper-tiny) do Hugging Face..."
WHISPER_TINY_HF_MODEL_NAME="openai/whisper-tiny"
# A biblioteca 'transformers' (que será instalada pelas dependências do ComfyUI_Sonic ou já existe)
# gerencia o download. Basta instanciar o modelo.
python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    processor = WhisperProcessor.from_pretrained('$WHISPER_TINY_HF_MODEL_NAME'); \
    model = WhisperForConditionalGeneration.from_pretrained('$WHISPER_TINY_HF_MODEL_NAME'); \
    print(f'Modelo $WHISPER_TINY_HF_MODEL_NAME baixado e carregado com sucesso.')" 2>&1 | tee "$PERSISTENT_DIR/whisper_tiny_hf_download_log.txt"


echo "ComfyUI_Sonic e modelos necessários instalados."

# --- 4. Instalação do VideoHelperSuite (VHS com vídeo combine) ---
echo "Configurando VideoHelperSuite..."
VIDEO_HELPER_SUITE_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite" # É um Custom Node
mkdir -p "$VIDEO_HELPER_SUITE_DIR"
if [ ! -d "$VIDEO_HELPER_SUITE_DIR" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$VIDEO_HELPER_SUITE_DIR"
fi
cd "$VIDEO_HELPER_SUITE_DIR"
pip install -r requirements.txt --no-cache-dir
echo "VideoHelperSuite configurado."

# --- 5. Baixar Modelos Essenciais (SDXL, SVD, Wan-AI/Wan2.1-T2V-14B) ---
echo "Baixando modelos essenciais para ComfyUI..."
COMFYUI_CHECKPOINTS_DIR="$COMFYUI_DIR/models/checkpoints"
COMFYUI_SVD_DIR="$COMFYUI_DIR/models/svd"
mkdir -p "$COMFYUI_CHECKPOINTS_DIR"
mkdir -p "$COMFYUI_SVD_DIR"

# Stable Diffusion XL Base (para Text-to-Video e Image-to-Image/Video)
echo "Baixando Stable Diffusion XL Base..."
wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/sd_xl_base_1.0.safetensors" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" || echo "SDXL Base model already exists or failed to download."

# SVD (Stable Video Diffusion) - Image-to-Video
echo "Baixando Stable Video Diffusion SVD_XT_1_1..."
# O TOKEN_HF precisa ser passado como uma variável de ambiente secreta no Vast.ai.
# Exemplo no comando vastai create instance: -e TOKEN_HF="hf_SEU_TOKEN_DE_LEITURA"
if [ -z "$TOKEN_HF" ]; then
    echo "AVISO: Variável de ambiente TOKEN_HF não definida. Não será possível baixar modelos Hugging Face privados ou com Gated Access."
else
    # SVD_XT_1_1
    wget -nc --header="Authorization: Bearer $TOKEN_HF" -O "$COMFYUI_SVD_DIR/svd_xt_1_1.safetensors" \
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors" || \
    echo "SVD_XT_1_1 já existe ou download falhou (verifique token/rede)."

    # SVD_XT_Image_Decoder
    wget -nc --header="Authorization: Bearer $TOKEN_HF" -O "$COMFYUI_SVD_DIR/svd_xt_image_decoder.safetensors" \
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors" || \
    echo "SVD_XT_Image_Decoder já existe ou download falhou (verifique token/rede)."
fi

# Wan-AI/Wan2.1-T2V-14B (Text-to-Video específico)
echo "Baixando modelo Wan-AI/Wan2.1-T2V-14B..."
WAN_AI_MODEL_URL="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B.safetensors?download=true"
WAN_AI_MODEL_NAME="Wan2.1-T2V-14B.safetensors"
WAN_AI_MODEL_PATH="$COMFYUI_DIR/models/unet/$WAN_AI_MODEL_NAME"

mkdir -p "$(dirname "$WAN_AI_MODEL_PATH")"
wget -nc -O "$WAN_AI_MODEL_PATH" "$WAN_AI_MODEL_URL" || echo "Modelo Wan-AI/Wan2.1-T2V-14B já existe ou falhou ao baixar."

echo "Modelos essenciais instalados."

echo "Provisionamento personalizado concluído."
