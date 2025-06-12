#!/bin/bash

# --- Configurações Base ---
PERSISTENT_DIR="/workspace"
COMFYUI_DIR="$PERSISTENT_DIR/ComfyUI"
SONIC_NODE_DIR="$PERSISTENT_DIR/ComfyUI_Sonic"
SONIC_MODELS_DIR="$COMFYUI_DIR/models/sonic"
RIFE_DIR="$COMFYUI_DIR/models/sonic/RIFE"  # Corrigido para estrutura do Sonic

# --- 1. Verificação de Conexão ---
echo "Verificando conectividade com GitHub..."
if ! curl -sI https://github.com | grep -q "HTTP/2 200"; then
    echo "ERRO: Falha na conexão com GitHub. Verifique:"
    echo "1. Se há bloqueio de firewall"
    echo "2. Se o DNS está funcionando (try 'nslookup github.com')"
    echo "3. Se há problemas com certificados TLS (try 'openssl s_client -connect github.com:443')"
    exit 1
fi

# --- 2. Instalação do Sonic com Fallbacks ---
echo "Instalando ComfyUI_Sonic (com fallback para mirror GitLab)..."
if [ ! -d "$SONIC_NODE_DIR" ]; then
    # Tentativa principal com GitHub
    if ! git clone --recursive https://github.com/smthemex/ComfyUI_Sonic.git "$SONIC_NODE_DIR"; then
        echo "Falha no GitHub, tentando mirror GitLab..."
        git clone --recursive https://gitlab.com/smthemex/ComfyUI_Sonic.git "$SONIC_NODE_DIR" || {
            echo "ERRO CRÍTICO: Falha em todos os mirrors. Soluções:"
            echo "1. Use VPN para contornar bloqueios"
            echo "2. Faça upload manual via SFTP"
            exit 1
        }
    fi
fi

# --- 3. Dependências com Verificação de Versões ---
cd "$SONIC_NODE_DIR"
echo "Instalando dependências com versões específicas..."
pip install --upgrade "numpy<2.0"  # Evita conflitos com versão 2.x
pip install -r requirements.txt --no-cache-dir || {
    echo "Falha nas dependências. Tentando instalação manual..."
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    pip install opencv-python-headless==4.7.0.72 librosa==0.9.2
}

# --- 4. Download de Modelos com Verificação ---
mkdir -p "$SONIC_MODELS_DIR" "$RIFE_DIR"

declare -A MODEL_URLS=(
    ["unet.pth"]="https://drive.google.com/uc?id=1mjIqU-c5q3qMI74XZd3UrkZek0IDTUUh"
    ["audio2token.pth"]="https://drive.google.com/uc?id=1vUY-b5NMvDA2XsxRZcB3nF3u1trOtK53h"
    ["flownet.pkl"]="https://drive.google.com/uc?id=1UnSd-s5DhPRZu4C23I4uOmmahH0J3Dkwl"
)

for model in "${!MODEL_URLS[@]}"; do
    if [ ! -f "$SONIC_MODELS_DIR/$model" ] && [ ! -f "$RIFE_DIR/$model" ]; then
        echo "Baixando $model..."
        if ! gdown --id "$(echo "${MODEL_URLS[$model]}" | grep -o 'id=[^&]*' | cut -d= -f2)" -O "$SONIC_MODELS_DIR/$model"; then
            echo "Falha no gdown, tentando wget..."
            wget -O "$SONIC_MODELS_DIR/$model" "${MODEL_URLS[$model]}" || echo "AVISO: $model não baixado"
        fi
    else
        echo "$model já existe."
    fi
done

# --- 5. Verificação Final ---
required_files=(
    "$SONIC_NODE_DIR/__init__.py"
    "$SONIC_MODELS_DIR/unet.pth"
    "$RIFE_DIR/flownet.pkl"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERRO: Arquivo essencial faltando: $file"
        echo "Soluções:"
        echo "1. Reexecute o script com VPN ativa"
        echo "2. Faça upload manual via:"
        echo "   scp -P PORT arquivo root@IP_DA_INSTANCIA:$file"
        exit 1
    fi
done

echo "✅ ComfyUI_Sonic instalado com sucesso!"
