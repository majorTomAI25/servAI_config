#!/bin/bash

echo "Iniciando script de provisionamento personalizado..."

cd /workspace

# --- 1. Verificação e Ativação do Ambiente Python (Crucial!) ---
# A imagem Vast.ai/comfy geralmente usa Conda.
# tentar detectar o ambiente ComfyUI corretamente.

CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE_PATH" ]; then
    source "$CONDA_BASE_PATH"/etc/profile.d/conda.sh # Importa as funções do conda
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
COMFYUI_DIR="/workspace/ComfyUI"
if [ ! -d "$COMFYUI_DIR" ]; then
    echo "ComfyUI não encontrado em $COMFYUI_DIR. Clonando..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    echo "ComfyUI clonado."
fi

# **FORÇAR ATUALIZAÇÃO DO COMFYUI AQUI**
echo "Forçando atualização do ComfyUI via git pull e pip install..."
cd "$COMFYUI_DIR"
git config pull.rebase false # Evita problemas de rebase em caso de conflitos
git pull origin master # Garante que puxa da branch master, que é a mais atual
# Força a reinstalação de todas as dependências e atualiza as existentes
pip install -r requirements.txt --no-cache-dir --upgrade --force-reinstall
pip install xformers --upgrade # Garante que xformers esteja atualizado
echo "ComfyUI backend e dependências forçadas."

# **LIMPEZA DE DEPENDÊNCIAS ANTIGAS**
echo "Removendo dependências Python não utilizadas..."
pip autoremove -y # Remove pacotes que não são mais necessários por nenhum pacote instalado
echo "Limpeza de dependências concluída."


# --- 2. Instalação de Ferramentas Adicionais  ---

# 2.1. Wav2Lip (para Lip Sync)
echo "Instalando Wav2Lip..."
WAV2LIP_DIR="/workspace/Wav2Lip"
if [ ! -d "$WAV2LIP_DIR" ]; then
    git clone https://github.com/justinjohn0306/Wav2Lip-GFPGAN.git "$WAV2LIP_DIR"
fi
cd "$WAV2LIP_DIR"
pip install -r requirements.txt --no-cache-dir
# Baixar o modelo pré-treinado do Wav2Lip
mkdir -p models
wget -nc -O models/wav2lip_gan.pth "https://github.com/justinjohn0306/Wav2Lip-GFPGAN/releases/download/v1.0/wav2lip_gan.pth" || echo "Wav2Lip model already exists or failed to download."
echo "Wav2Lip instalado."

# 2.2. Sonic (assumindo bibliotecas Python para áudio)
echo "Instalando bibliotecas Sonic (processamento de áudio)..."
pip install pydub librosa SpeechRecognition # Adicione outras que precisar
echo "Bibliotecas Sonic instaladas."

# 2.3. VideoHelpSuite (Ferramentas de Vídeo - FFmpeg scripts)
echo "Configurando VideoHelpSuite..."
VIDEO_TOOLS_DIR="/workspace/video_help_suite"
mkdir -p "$VIDEO_TOOLS_DIR"
# copiar seus scripts personalizados para dentro de $VIDEO_TOOLS_DIR.
# Exemplo: wget -O "$VIDEO_TOOLS_DIR/vhs_effect.sh" "https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/vhs_effect.sh"
# chmod +x "$VIDEO_TOOLS_DIR/vhs_effect.sh"
echo "VideoHelpSuite configurado."

# --- 3. Instalação de ComfyUI Manager  ---
echo "Instalando ComfyUI Manager..."
COMFYUI_CUSTOM_NODES_DIR="$COMFYUI_DIR/custom_nodes"
MANAGER_DIR="$COMFYUI_CUSTOM_NODES_DIR/ComfyUI-Manager"
mkdir -p "$COMFYUI_CUSTOM_NODES_DIR"
if [ ! -d "$MANAGER_DIR" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git "$MANAGER_DIR"
fi
cd "$MANAGER_DIR"
pip install -r requirements.txt --no-cache-dir
echo "ComfyUI Manager instalado. Reinicie o ComfyUI para vê-lo."

# --- 4. Baixar Modelos Essenciais (SDXL ou outros de sua escolha) ---
echo "Baixando modelos essenciais..."
COMFYUI_CHECKPOINTS_DIR="$COMFYUI_DIR/models/checkpoints"
mkdir -p "$COMFYUI_CHECKPOINTS_DIR"
# Exemplo: SDXL Base (se você for usar SDXL)
# wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/sd_xl_base_1.0.safetensors" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" || echo "SDXL Base model already exists or failed to download."
# Exemplo: Waifu Diffusion 2.1 (se for um modelo checkpoint)
# wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/waifudiffusion_v2_1.safetensors" "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-v1-4-anime_vae.safetensors" || echo "Waifu Diffusion model already exists or failed to download."

echo "personalizado concluído."
