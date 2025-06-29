#!/bin/bash

# Crie pastas necessárias
mkdir -p /workspace/ComfyUI
cd /workspace

# Função para verificar e atualizar o ComfyUI
install_or_update_comfyui() {
    if [ -d "ComfyUI" ] && [ -d "ComfyUI/.git" ]; then
        echo " ComfyUI já está instalado. Atualizando para a versão mais recente..."
        cd ComfyUI
        git fetch origin
        git pull origin main
        cd ..
    else
        echo " Clonando ComfyUI..."
        git clone https://github.com/comfyanonymous/ComfyUI.git
    fi
}

# Instala ou atualiza o ComfyUI
install_or_update_comfyui

# Ative o ambiente virtual do Conda
echo " Ativando ambiente Conda..."
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# Instale dependências básicas
echo " Instalando PyTorch e dependências principais..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r /workspace/ComfyUI/requirements.txt

echo " Iniciando ComfyUI..."
cd /workspace/ComfyUI
python main.py --port 8188 --host 0.0.0.0
