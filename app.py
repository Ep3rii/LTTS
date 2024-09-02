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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    bright = cv2.convertScaleAbs(sharp, alpha=1.2, beta=30)
    _, binary = cv2.threshold(bright, 127, 255, cv2.THRESH_BINARY)
    config = '--oem 3 --psm 11'
    text = pytesseract.image_to_string(binary, config=config)
    
    return text, binary

# Função para identificar o tipo de texto
def identify_text_type(text):
    words = text.split()
    if len(words) == 1:
        return "Palavra Única"
    elif len(words) > 1 and len(words) <= 10:
        return "Frase"
    else:
        return "Parágrafo"

# Configurações de estilo e layout do Streamlit
st.set_page_config(page_title="LTTR", page_icon="📖", layout="wide")

# Exibir cabeçalho fixo no canto superior esquerdo
st.header("LTTR")
st.markdown("Leitura de Texto em Tempo Real")

# Menu lateral com opções de captura de imagem ou upload
st.sidebar.title("Menu")
option = st.sidebar.radio("Escolha uma opção:", ("Captura de Imagem", "Carregar Imagem"))

# CSS para centralizar a imagem e definir tamanho original
st.markdown("""
<style>
    .stImage {
        display: flex;
        justify-content: center;
    }
    .stImage > img {
        max-width: none;
        width: auto;
        height: auto;
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

# Interface principal
if option == "Captura de Imagem":
    st.title("Captura de Imagem")
    cap = cv2.VideoCapture(0)

    if st.button("Capturar"):
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        text, processed_image = process_frame(frame)
        
        if text:
            st.write("Texto reconhecido:")
            st.text(text)
            
            text_type = identify_text_type(text)
            st.write(f"Tipo de Texto: {text_type}")
            
            tts = gTTS(text, lang='pt')
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Imagem Capturada Original:")
            st.image(frame, channels="BGR", use_column_width=True, clamp=True)
        
        with col2:
            st.write("Imagem Pós-Processada (Preto e Branco):")
            st.image(processed_image, channels="GRAY", use_column_width=True, clamp=True)

    cap.release()

elif option == "Carregar Imagem":
    st.title("Upload de Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        text, processed_image = process_frame(image)
        
        if text:
            st.write("Texto reconhecido:")
            st.text(text)
            
            text_type = identify_text_type(text)
            st.write(f"Tipo de Texto: {text_type}")
            
            tts = gTTS(text, lang='pt')
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Imagem Carregada Original:")
            st.image(image, channels="RGB", use_column_width=True, clamp=True)
        
        with col2:
            st.write("Imagem Pós-Processada (Preto e Branco):")
            st.image(processed_image, channels="GRAY", use_column_width=True, clamp=True)
