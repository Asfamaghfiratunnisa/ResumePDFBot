import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import PyPDF2
import torch
import base64
import io

# Sidebar contents
with st.sidebar:
    st.title('Selamat datang di Aplikasi PDF Summarizer!')
    st.markdown('''
                
    ## About
    Fitur utama aplikasi ini meliputi:
    - Kemampuan mengunggah berkas PDF Anda dengan mudah.
    - Proses merangkum yang cepat dan akurat menggunakan LLM.
    - Tampilan PDF hasil ringkasan untuk kemudahan Anda dalam membaca.

                
    ## Credits
    - Annisa Kusumawati (202010370311138)
    - Asfa Maghfiratunnisa (202010370311142)
    &nbsp;  
    &nbsp;
    &nbsp;
    &nbsp;
    &nbsp;
    ''')

# # #  F U N C T I O N # # # 

# Model and tokenizer loading
@st.cache_data()
def load_model():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)
    return tokenizer, base_model

# File preprocessing
@st.cache_data
def file_preprocessing(uploaded_file):
    file_stream = uploaded_file.read()
    reader = PyPDF2.PdfFileReader(io.BytesIO(file_stream))
    final_texts = ""
    
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        final_texts += page.extractText() + "\n"
    
    return final_texts

# LLM pipeline
@st.cache_data()
def llm_pipeline(input_text, _tokenizer, _base_model):
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)
    
    pipe_sum = pipeline('summarization', model=base_model, tokenizer=tokenizer, max_length=500, min_length=50)
    result = pipe_sum(input_text)
    return result[0]['summary_text']

# Function to display PDF
def display_pdf(uploaded_file):
    base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# # #  M A I N # # # 

# Streamlit UI code
def main():
    st.title("PDF Summarizer")
    uploaded_file = st.file_uploader("Upload your PDF here", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            st.write("Summarizing...")

            tokenizer, base_model = load_model()
            processed_text = file_preprocessing(uploaded_file)
            summary = llm_pipeline(processed_text, tokenizer, base_model)

            st.info("Summarization Complete, Thank you")
            st.success(summary)

if __name__ == "__main__":
    main()
