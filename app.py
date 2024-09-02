import cv2
import pytesseract
from gtts import gTTS
import streamlit as st
import numpy as np
from PIL import Image

# Configuração do caminho do Tesseract no Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Função para processar a imagem e aplicar OCR
def process_frame(image, alpha, beta):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)
    bright = cv2.convertScaleAbs(sharp, alpha=alpha, beta=beta)
    _, binary = cv2.threshold(bright, 100, 255, cv2.THRESH_BINARY)
    
    # Operações morfológicas para melhorar a imagem
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    config = '--oem 3 --psm 1'
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

# Configuração das barras deslizantes para os parâmetros de imagem
alpha = st.sidebar.slider("Valor de Alpha", 0.5, 3.0, 1.2)
beta = st.sidebar.slider("Valor de Beta", -100, 100, -60)

# Armazenar imagem capturada no estado da sessão
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'captured_text' not in st.session_state:
    st.session_state.captured_text = ""
if 'text_type' not in st.session_state:
    st.session_state.text_type = ""

# Função para atualizar a imagem processada
def update_processed_image():
    if st.session_state.captured_image is not None:
        st.session_state.captured_text, st.session_state.processed_image = process_frame(st.session_state.captured_image, alpha, beta)
        st.session_state.text_type = identify_text_type(st.session_state.captured_text)

# Interface principal
if option == "Captura de Imagem":
    st.title("Captura de Imagem")
    cap = cv2.VideoCapture(0)
    
    # Configura a resolução da câmera (exemplo: 1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if st.button("Capturar"):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            st.session_state.captured_image = frame
            # Inicializar imagens processadas
            st.session_state.processed_image = None
            st.session_state.captured_text = ""
            st.session_state.text_type = ""
            # Atualizar a imagem processada automaticamente após captura
            update_processed_image()
        else:
            st.error("Falha ao capturar imagem. Verifique a câmera.")
    cap.release()

    if st.session_state.captured_image is not None:
        # Exibir imagens lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Imagem Capturada Original:")
            st.image(st.session_state.captured_image, channels="BGR", use_column_width=True, clamp=True)
        
        with col2:
            if st.session_state.processed_image is not None:
                st.write("Imagem Pós-Processada (Preto e Branco):")
                st.image(st.session_state.processed_image, channels="GRAY", use_column_width=True, clamp=True)
            else:
                st.write("Imagem Pós-Processada será exibida aqui após atualização.")
        
        if st.session_state.captured_text:
            st.write("Texto reconhecido:")
            st.text(st.session_state.captured_text)
            
            st.write(f"Tipo de Texto: {st.session_state.text_type}")
            
            if st.session_state.captured_text.strip():  # Verifica se o texto não está vazio
                tts = gTTS(st.session_state.captured_text, lang='pt')
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3")
            else:
                st.write("Texto não reconhecido. Não há áudio para reproduzir.")

elif option == "Carregar Imagem":
    st.title("Upload de Imagem")
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        text, processed_image = process_frame(image, alpha, beta)
        
        st.write("Texto reconhecido:")
        st.text(text)
        
        st.write(f"Tipo de Texto: {identify_text_type(text)}")
        
        if text.strip():  # Verifica se o texto não está vazio
            tts = gTTS(text, lang='pt')
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
        else:
            st.write("Texto não reconhecido. Não há áudio para reproduzir.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Imagem Carregada Original:")
            st.image(image, channels="RGB", use_column_width=True, clamp=True)
        
        with col2:
            st.write("Imagem Pós-Processada (Preto e Branco):")
            st.image(processed_image, channels="GRAY", use_column_width=True, clamp=True)
