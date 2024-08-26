import cv2
import pytesseract
from gtts import gTTS
import os
import numpy as np
import streamlit as st
from PIL import Image

# Configuração do caminho do Tesseract no Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
st.title("OCR com Tesseract e Streamlit")
st.write("Carregue uma imagem para realizar OCR e ouvir o texto reconhecido.")

# Opção para carregar imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    text = process_frame(image)
    
    if text:
        st.write("Texto reconhecido:")
        st.text(text)
        tts = gTTS(text, lang='pt')
        tts.save("output.mp3")
        st.audio("output.mp3", format="audio/mp3")
    
    # Exibir a imagem carregada
    st.image(image, channels="RGB")

# Nota para usuários
st.write("A funcionalidade de captura de vídeo em tempo real não está disponível no Streamlit online.")
