#!/bin/bash

echo "Iniciando script de provisionamento personalizado para Vast.ai adptacao do Kaggle..."

# Diretório persistente no Vast.ai: /workspace
PERSISTENT_DIR="/workspace"
cd "$PERSISTENT_DIR"

# --- 1. Verificação e Ativação do Ambiente Python (Crucial!) ---
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
    echo "Conda não encontrado. Assumindo ambiente de sistema para pip."
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

# Instalação das dependências PyTorch com CUDA 12.4 (se a imagem Vast.ai tiver CUDA 12.4)
# Adapte o cu version (cu124) para o seu CUDA (ex: cu118 para CUDA 11.8)
echo "Instalando PyTorch com CUDA (cu124)..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

echo "Instalando requisitos base do ComfyUI e pacotes adicionais..."
pip install -r requirements.txt --no-cache-dir --upgrade --force-reinstall
pip install bitsandbytes>=0.43.0 gguf --upgrade # bitsandbytes e gguf

# Remove dependências Python não utilizadas
echo "Removendo dependências Python não utilizadas..."
pip autoremove -y
echo "Limpeza de dependências concluída."

# --- 2. Instalação de CUSTOM NODES ---
echo "Instalando Custom Nodes para ComfyUI..."
COMFYUI_CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
mkdir -p "$COMFYUI_CUSTOM_NODES_DIR"
cd "$COMFYUI_CUSTOM_NODES_DIR"

echo "Clonando ComfyUI-Manager..."
if [ ! -d "ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ComfyUI-Manager
    pip install -r requirements.txt --no-cache-dir
    git pull # Atualiza o Manager
    cd .. # Volta para custom_nodes
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

# Voltar para o diretório raiz do ComfyUI e instalar requirements.txt de custom_nodes (se existir)
cd "$COMFYUI_CUSTOM_NODES_DIR"
if [ -f "requirements.txt" ]; then # Alguns custom nodes podem ter um requirements.txt em custom_nodes
    pip install -r requirements.txt --no-cache-dir
fi
echo "Custom Nodes instalados e atualizados."

# --- 3. Instalação do Wav2Lip (para Lip Sync) ---
echo "Instalando Wav2Lip..."
WAV2LIP_DIR="$PERSISTENT_DIR/Wav2Lip"
if [ ! -d "$WAV2LIP_DIR" ]; then
    git clone https://github.com/Rudrabha/Wav2Lip.git "$WAV2LIP_DIR"
fi
cd "$WAV2LIP_DIR"
pip install -r requirements.txt --no-cache-dir
mkdir -p checkpoints
wget -nc -O checkpoints/wav2lip.pth "https://iiitaph.github.io/Wav2Lip/checkpoints/wav2lip.pth" || echo "Wav2Lip model already exists or failed to download."
echo "Wav2Lip instalado."

# --- 4. Instalação e Configuração do Sonic (ComfyUI_Sonic) ---
echo "Instalando ComfyUI_Sonic e baixando modelos necessários..."
SONIC_DIR="$PERSISTENT_DIR/ComfyUI_Sonic"
if [ ! -d "$SONIC_DIR" ]; then
    git clone https://github.com/smthemex/ComfyUI_Sonic.git "$SONIC_DIR"
fi
cd "$SONIC_DIR"
pip install -r requirements.txt --no-cache-dir

WHISPER_CACHE_DIR="$PERSISTENT_DIR/.cache/whisper"
echo "Limpando cache de modelos Whisper em $WHISPER_CACHE_DIR..."
rm -rf "$WHISPER_CACHE_DIR"
mkdir -p "$WHISPER_CACHE_DIR"

export HF_HOME="$PERSISTENT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "Baixando modelo OpenAI Whisper 'tiny.en' (forçando download)..."
pip install openai-whisper
python -c "import whisper; print('Downloading Whisper tiny.en model...'); whisper.load_model('tiny.en'); print('Whisper tiny.en model downloaded successfully.');" 2>&1 | tee "$PERSISTENT_DIR/whisper_download_log.txt"

WHISPER_MODEL_PATH="$WHISPER_CACHE_DIR/tiny.en.pt"
if [ -f "$WHISPER_MODEL_PATH" ] && [ -s "$WHISPER_MODEL_PATH" ]; then
    echo "Verificação: Modelo Whisper tiny.en encontrado e não está vazio em $WHISPER_MODEL_PATH."
    file "$WHISPER_MODEL_PATH"
else
    echo "ATENÇÃO: Modelo Whisper tiny.en NÃO encontrado ou está vazio em $WHISPER_MODEL_PATH. O download pode ter falhado."
    echo "Verifique o log de download em $PERSISTENT_DIR/whisper_download_log.txt para mais detalhes."
fi
echo "ComfyUI_Sonic e modelos necessários instalados."

# --- 5. VideoHelpSuite (Ferramentas de Vídeo - FFmpeg scripts) ---
echo "Configurando VideoHelpSuite..."
VIDEO_TOOLS_DIR="$PERSISTENT_DIR/video_help_suite"
mkdir -p "$VIDEO_TOOLS_DIR"
# Aqui você pode copiar seus scripts personalizados para dentro de $VIDEO_TOOLS_DIR.
# Exemplo: wget -O "$VIDEO_TOOLS_DIR/vhs_effect.sh" "https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/vhs_effect.sh"
# chmod +x "$VIDEO_TOOLS_DIR/vhs_effect.sh"
echo "VideoHelpSuite configurado."

# --- 6. Baixar Modelo Wan-AI/Wan2.1-T2V-14B ---
echo "Baixando modelo Wan-AI/Wan2.1-T2V-14B..."
WAN_AI_MODEL_URL="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B.safetensors?download=true"
WAN_AI_MODEL_NAME="Wan2.1-T2V-14B.safetensors"
WAN_AI_MODEL_PATH="$COMFYUI_DIR/models/unet/$WAN_AI_MODEL_NAME" # Caminho original do Kaggle

mkdir -p "$(dirname "$WAN_AI_MODEL_PATH")" # Garante que a pasta 'unet' exista
wget -nc -O "$WAN_AI_MODEL_PATH" "$WAN_AI_MODEL_URL" || echo "Modelo Wan-AI/Wan2.1-T2V-14B já existe ou falhou ao baixar."
echo "Modelo Wan-AI/Wan2.1-T2V-14B instalado com sucesso!"

echo "Provisionamento personalizado concluído."
