#!/bin/bash

# Define o diretório persistente para Vast.ai
PERSISTENT_DIR="/workspace"
cd "$PERSISTENT_DIR"

# Causa o script a sair em caso de falha de qualquer comando.
# Adicionado -u para sair se uma variável não definida for usada.
set -euo pipefail

echo "Iniciando provisionamento personalizado para Vast.ai (versão FINAL)..."

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
    # Adiciona o diretório do ComfyUI à lista de diretórios seguros do Git IMEDIATAMENTE APÓS O CLONE
    git config --global --add safe.directory "$COMFYUI_DIR"
    echo "ComfyUI clonado."
fi

echo "Forçando atualização do ComfyUI via git pull e pip install..."
cd "$COMFYUI_DIR"
# Adiciona o diretório atual (ComfyUI) à lista de diretórios seguros do Git antes do pull
git config --global --add safe.directory "$(pwd)"
git config pull.rebase false
git pull origin master

# Instalação das dependências PyTorch com CUDA (usar cu124 conforme log)
echo "Instalando PyTorch com CUDA (cu124 - ajustado para o log)..."
# Usando if/else para melhor feedback em caso de falha
if pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124; then
    echo "PyTorch com cu124 instalado com sucesso."
else
    echo "ERRO: Falha na instalação de PyTorch com cu124. Verifique a compatibilidade CUDA ou tente uma versão diferente."
    exit 1 # Sai do script se o PyTorch, que é crítico, não instalar.
fi

echo "Instalando requisitos base do ComfyUI e pacotes adicionais..."
# Adicionado --break-system-packages para sistemas que bloqueiam instalação direta
pip install -r requirements.txt --no-cache-dir --upgrade --force-reinstall || pip install -r requirements.txt --no-cache-dir --upgrade --force-reinstall --break-system-packages
pip install bitsandbytes>=0.43.0 gguf --upgrade || pip install bitsandbytes>=0.43.0 gguf --upgrade --break-system-packages # bitsandbytes e gguf

echo "Limpeza de dependências: 'pip autoremove' não disponível, pulando."

# --- 3. Instalação e Configuração do Sonic (ComfyUI_Sonic) ---
echo "Instalando ComfyUI_Sonic e baixando modelos necessários..."
SONIC_DIR="$PERSISTENT_DIR/ComfyUI_Sonic"
if [ ! -d "$SONIC_DIR" ]; then
    git clone --recursive https://github.com/smthemex/ComfyUI_Sonic.git "$SONIC_DIR" # --recursive para submodules
fi
# Adiciona o diretório do Sonic à lista de diretórios seguros do Git
git config --global --add safe.directory "$SONIC_DIR"
cd "$SONIC_DIR"
pip install -r requirements.txt --no-cache-dir || pip install -r requirements.txt --no-cache-dir --break-system-packages
echo "ComfyUI_Sonic configurado."

echo "Instalando gdown para download de arquivos do Google Drive..."
pip install gdown || pip install gdown --break-system-packages

# --- Baixar Modelo UNET e Outros Arquivos do Google Drive para Sonic ---
mkdir -p "$COMFYUI_DIR/models/sonic" # Cria a pasta models/sonic dentro do ComfyUI
echo "Baixando arquivos para ComfyUI/models/sonic (unet.pth, audio2token.pth, audio2bucket.pth, yoloface_v5m.pt)..."
gdown --id 1mjIqU-c5q3qMI74XZd3UrkZek0IDTUUh -O "$COMFYUI_DIR/models/sonic/unet.pth" || echo "Aviso: unet.pth já existe ou falhou ao baixar."
gdown --id 1vUY-b5NMvDA2XsxRZcB3nF3u1trOtK53h -O "$COMFYUI_DIR/models/sonic/audio2token.pth" || echo "Aviso: audio2token.pth já existe ou falhou ao baixar."
gdown --id 1RHWasbgUWZg-mFaQhDJtF1KhpUSecC5d -O "$COMFYUI_DIR/models/sonic/audio2bucket.pth" || echo "Aviso: audio2bucket.pth já existe ou falhou ao baixar."
gdown --id 13Hpfi-cBvlmNvTv6W4Oa7agWyzmvmofB4 -O "$COMFYUI_DIR/models/sonic/yoloface_v5m.pt" || echo "Aviso: yoloface_v5m.pt já existe ou falhou ao baixar."

# --- Baixar Modelos Whisper (para transcrição) ---
# Usando uma variável diferente para o cache de Hugging Face para evitar conflitos com outros caches
export HF_HOME_CUSTOM="$PERSISTENT_DIR/.cache/huggingface_models"
mkdir -p "$HF_HOME_CUSTOM"
# Aponta a variável de ambiente para o cache de Hugging Face
export HF_HOME="$HF_HOME_CUSTOM"

echo "Baixando modelo OpenAI Whisper 'tiny.en' (forçando download via biblioteca)..."
pip install git+https://github.com/openai/whisper.git || pip install git+https://github.com/openai/whisper.git --break-system-packages # Instala a biblioteca Whisper do GitHub
# Adicionado timeout para o download do modelo Whisper e melhor tratamento de erro
if timeout 300 python -c "import whisper; print('Downloading Whisper tiny.en model...'); whisper.load_model('tiny.en'); print('Whisper tiny.en model downloaded successfully.');" 2>&1 | tee "$PERSISTENT_DIR/whisper_download_log.txt"; then
    echo "Download do modelo Whisper tiny.en concluído."
else
    echo "ERRO: Download do modelo Whisper tiny.en falhou ou excedeu o tempo limite. Verifique o log."
    # Não sair aqui, pois o resto do script ainda pode ser útil.
fi

WHISPER_MODEL_PATH="$HF_HOME_CUSTOM/models--openai--whisper-tiny.en/snapshots" # Caminho comum após download do HF
# Tenta encontrar o arquivo .pt dentro do snapshot mais recente
WHISPER_TINY_PT=$(find "$WHISPER_MODEL_PATH" -name "tiny.en.pt" -print -quit)

if [ -f "$WHISPER_TINY_PT" ] && [ -s "$WHISPER_TINY_PT" ]; then
    echo "Verificação: Modelo Whisper tiny.en encontrado e não está vazio em $WHISPER_TINY_PT."
    file "$WHISPER_TINY_PT"
else
    echo "ATENÇÃO: Modelo Whisper tiny.en NÃO encontrado ou está vazio. O download pode ter falhado."
    echo "Verifique o log de download em $PERSISTENT_DIR/whisper_download_log.txt para mais detalhes."
fi

# --- Baixar Modelo Whisper-tiny (do Hugging Face) ---
echo "Baixando modelo Whisper-tiny (openai/whisper-tiny) do Hugging Face via biblioteca transformers..."
WHISPER_TINY_HF_MODEL_NAME="openai/whisper-tiny"
pip install transformers || pip install transformers --break-system-packages # Garante que transformers esteja instalado para este download
# Adicionado timeout para o download do modelo Hugging Face
if timeout 300 python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; \
    processor = WhisperProcessor.from_pretrained('$WHISPER_TINY_HF_MODEL_NAME'); \
    model = WhisperForConditionalGeneration.from_pretrained('$WHISPER_TINY_HF_MODEL_NAME'); \
    print(f'Modelo $WHISPER_TINY_HF_MODEL_NAME baixado e carregado com sucesso.')" 2>&1 | tee "$PERSISTENT_DIR/whisper_tiny_hf_download_log.txt"; then
    echo "Download do modelo Whisper-tiny HF concluído."
else
    echo "ERRO: Download do modelo Whisper-tiny HF falhou ou excedeu o tempo limite. Verifique o log."
fi

echo "ComfyUI_Sonic e modelos necessários instalados."

# --- RIFE (flownet.pkl) ---
RIFE_DIR="$COMFYUI_DIR/models/RIFE"
mkdir -p "$RIFE_DIR"
echo "Baixando flownet.pkl para ComfyUI/models/RIFE..."
gdown --id 1UnSd-s5DhPRZu4C23I4uOmmahH0J3Dkwl -O "$RIFE_DIR/flownet.pkl" || echo "Aviso: flownet.pkl já existe ou falhou ao baixar."

# --- Instalação do VideoHelperSuite (VHS com vídeo combine) ---
echo "Configurando VideoHelperSuite..."
VIDEO_HELPER_SUITE_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-VideoHelperSuite" # É um Custom Node
mkdir -p "$VIDEO_HELPER_SUITE_DIR"
if [ ! -d "$VIDEO_HELPER_SUITE_DIR" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$VIDEO_HELPER_SUITE_DIR"
fi
# Adiciona o diretório do VideoHelperSuite à lista de diretórios seguros do Git
git config --global --add safe.directory "$VIDEO_HELPER_SUITE_DIR"
cd "$VIDEO_HELPER_SUITE_DIR"
pip install -r requirements.txt --no-cache-dir || pip install -r requirements.txt --no-cache-dir --break-system-packages
echo "VideoHelperSuite configurado."

# --- 4. Baixar Modelos Essenciais (SDXL, SVD, Wan-AI/Wan2.1-T2V-14B) ---
echo "Baixando modelos essenciais para ComfyUI..."
COMFYUI_CHECKPOINTS_DIR="$COMFYUI_DIR/models/checkpoints"
COMFYUI_SVD_DIR="$COMFYUI_DIR/models/svd"
mkdir -p "$COMFYUI_CHECKPOINTS_DIR"
mkdir -p "$COMFYUI_SVD_DIR"

# Stable Diffusion XL Base (para Text-to-Video e Image-to-Image/Video)
echo "Baixando Stable Diffusion XL Base..."
# Adicionado timeout e verificação de sucesso
if timeout 600 wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/sd_xl_base_1.0.safetensors" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"; then
    echo "SDXL Base model baixado com sucesso."
else
    echo "Aviso: SDXL Base model já existe ou falhou ao baixar."
fi

# SVD (Stable Video Diffusion) - Image-to-Video
echo "Baixando Stable Video Diffusion SVD_XT_1_1 e Image_Decoder..."
if [ -z "$TOKEN_HF" ]; then
    echo "AVISO: Variável de ambiente TOKEN_HF não definida. Não será possível baixar modelos Hugging Face privados ou com Gated Access."
else
    # SVD_XT_1_1
    if timeout 600 wget -nc --header="Authorization: Bearer $TOKEN_HF" -O "$COMFYUI_SVD_DIR/svd_xt_1_1.safetensors" \
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/svd_xt_1_1.safetensors"; then
        echo "SVD_XT_1_1 baixado com sucesso."
    else
        echo "Aviso: SVD_XT_1_1 já existe ou download falhou (verifique token/rede/timeout)."
    fi

    # SVD_XT_Image_Decoder
    if timeout 600 wget -nc --header="Authorization: Bearer $TOKEN_HF" -O "$COMFYUI_SVD_DIR/svd_xt_image_decoder.safetensors" \
    "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors"; then
        echo "SVD_XT_Image_Decoder baixado com sucesso."
    else
        echo "Aviso: SVD_XT_Image_Decoder já existe ou download falhou (verifique token/rede/timeout)."
    fi
fi

echo "Modelos essenciais instalados."
cd "$PERSISTENT_DIR" # Volta para o diretório persistente principal

# --- Finalização do provisionamento ---
# Criar o arquivo .provisioning_completed para sinalizar a conclusão
touch "$PERSISTENT_DIR/.provisioning_completed"
echo "Provisionamento personalizado concluído."

# Comando para iniciar o ComfyUI (exemplo, ajuste conforme sua necessidade)
# Este comando será executado após o provisionamento.
echo "Iniciando ComfyUI..."
python "$COMFYUI_DIR/main.py" --listen --port 8188 --enable-cors-header "*" --max-batch-size 1 --checkpoints-path "$COMFYUI_CHECKPOINTS_DIR" --output-directory "$PERSISTENT_DIR/ComfyUI/output"
