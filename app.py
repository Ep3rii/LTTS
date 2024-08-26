import cv2
import pytesseract
from gtts import gTTS
import os
import streamlit as st
import numpy as np
from PIL import Image

# Configuração do caminho do Tesseract no Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Função para processar a imagem e aplicar OCR
def process_frame(image):
    # Converter para tons de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ruído
    gray = cv2.medianBlur(gray, 3)

    # Aplicar binarização para converter em preto e branco
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Configurações do Tesseract
    config = '--oem 3 --psm 11'

    # Aplicar OCR no frame binário
    text = pytesseract.image_to_string(binary, config=config)
    
    return text

# Interface do Streamlit
st.title("OCR em Tempo Real com Tesseract e Streamlit")
st.write("Carregue uma imagem ou use o upload para realizar OCR e ouvir o texto reconhecido.")

# Opção para escolher entre upload de imagem
option = st.selectbox("Escolha uma opção", ("Carregar Imagem"))

if option == "Carregar Imagem":
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Converte a imagem para BGR (OpenCV usa BGR por padrão)
        if image.ndim == 3 and image.shape[2] == 4:  # Se a imagem tiver um canal alfa (transparência)
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        text = process_frame(image)
        
        if text:
            st.write("Texto reconhecido:")
            st.text(text)
            
            # Converter o texto em áudio
            tts = gTTS(text, lang='pt')
            audio_file = "output.mp3"
            tts.save(audio_file)
            st.audio(audio_file, format="audio/mp3")
        
        # Exibir a imagem carregada
        st.image(image, channels="BGR")

# Instruções para encerrar a aplicação
st.write("Feche a aba do navegador para sair.")
