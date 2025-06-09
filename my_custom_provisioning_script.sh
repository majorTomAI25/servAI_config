#!/bin/bash

echo "Iniciando meu script de provisionamento personalizado..."

# Vast.ai usa /workspace como diretório persistente. Vamos para lá.
cd /workspace

# --- 1. Verificação e Ativação do Ambiente Python (Crucial!) ---
# A imagem Vast.ai/comfy geralmente usa Conda. Precisamos ativar o ambiente correto.
# Tente ativar o ambiente padrão do ComfyUI. Se não funcionar, ajuste o nome.
source /opt/conda/bin/activate comfy

# Verifica se o ComfyUI existe e o atualiza
COMFYUI_DIR="/workspace/ComfyUI" # Vast.ai geralmente instala aqui
if [ -d "$COMFYUI_DIR" ]; then
    echo "ComfyUI encontrado em $COMFYUI_DIR. Atualizando..."
    cd "$COMFYUI_DIR"
    git pull
    pip install -r requirements.txt --no-cache-dir --upgrade
    echo "ComfyUI atualizado."
else
    echo "ComfyUI não encontrado em $COMFYUI_DIR. Clonando..."
    # Se por algum motivo o template base não tiver o ComfyUI, clone.
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    cd "$COMFYUI_DIR"
    pip install -r requirements.txt --no-cache-dir
    echo "ComfyUI clonado e dependências instaladas."
fi

# --- 2. Instalação de Ferramentas Adicionais ---

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
# Aqui você pode copiar seus scripts personalizados para dentro de $VIDEO_TOOLS_DIR.
# Exemplo: wget -O "$VIDEO_TOOLS_DIR/vhs_effect.sh" "https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/vhs_effect.sh"
# chmod +x "$VIDEO_TOOLS_DIR/vhs_effect.sh"
echo "VideoHelpSuite configurado."

# --- 3. Instalação de ComfyUI Manager (altamente recomendado!) ---
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
# Você pode usar o ComfyUI Manager para baixar a maioria dos modelos.
# Mas para modelos iniciais, ou se quiser pré-carregar um específico:
echo "Baixando modelos essenciais..."
COMFYUI_CHECKPOINTS_DIR="$COMFYUI_DIR/models/checkpoints"
mkdir -p "$COMFYUI_CHECKPOINTS_DIR"
# Exemplo: SDXL Base (se você for usar SDXL)
# wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/sd_xl_base_1.0.safetensors" "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" || echo "SDXL Base model already exists or failed to download."
# Exemplo: Waifu Diffusion 2.1 (se for um modelo checkpoint)
# wget -nc -O "$COMFYUI_CHECKPOINTS_DIR/waifudiffusion_v2_1.safetensors" "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-v1-4-anime_vae.safetensors" || echo "Waifu Diffusion model already exists or failed to download."
# Nota: "-nc" evita baixar se o arquivo já existir.

echo "Provisionamento personalizado concluído."
