import os
import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import google.generativeai as genai   # ✅ Gemini Import

# ✅ Load environment variables from .env if available
load_dotenv()

# ✅ Get API keys (Groq + Gemini)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY and not GEMINI_API_KEY:
    st.error("❌ No API keys found. Please set GROQ_API_KEY and/or GEMINI_API_KEY in your .env file.")
    st.stop()

# ✅ Initialize Groq client
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# ✅ Initialize Gemini client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ----------- PDF Text Extractor -----------
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text.strip()

# ----------- Groq Helper -----------
def ask_groq(prompt, model_choice):
    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=st.session_state["chat_history"] + [{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        output = response.choices[0].message.content
        st.session_state["chat_history"].append({"role": "assistant", "content": output})
        return output
    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"

# ----------- Gemini Helper -----------
def ask_gemini(prompt, model_choice="gemini-1.5-flash"):
    try:
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        output = response.text
        st.session_state["chat_history"].append({"role": "assistant", "content": output})
        return output
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

# ----------- FIXED PDF Export Function -----------
def export_to_pdf(summary, qa_list, history, full_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    def add_block(title, content):
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        for line in content.split("\n"):
            if line.strip():
                elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    if summary:
        add_block("📌 SUMMARY", summary)
    if qa_list:
        for q, a in qa_list:
            add_block(f"❓ Q: {q}", f"🤖 A: {a}")
    if history:
        hist_text = ""
        for msg in history:
            role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
            hist_text += f"{role}: {msg['content']}<br/>"
        add_block("📝 CONVERSATION HISTORY", hist_text)
    if full_text:
        add_block("📖 FULL TEXT", full_text)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----------- Streamlit UI -----------
st.set_page_config(page_title="📑 Automated Research Assistant", layout="wide")

st.title("📑 Automated Research Assistant")
st.markdown("Upload multiple PDFs and get **Summaries, Q&A, and Full Text** using **Groq or Gemini LLMs** ⚡")

# Initialize session memory
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "system", "content": "You are a helpful research assistant."}]

if "qa_list" not in st.session_state:
    st.session_state["qa_list"] = []

# Model selector
model_choice = st.selectbox(
    "🤖 Choose a Model",
    [
        "Groq - llama-3.3-70b-versatile",
        "Groq - llama-3.1-8b-instant",
        "Gemini - gemini-1.5-flash",
        "Gemini - gemini-1.5-pro"
    ],
    index=0
)

# File uploader
uploaded_files = st.file_uploader("Upload your PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_text = extract_text_from_pdfs(uploaded_files)
    st.info(f"📊 PDF contains **{len(pdf_text.split())} words** and **{len(pdf_text)} characters**.")

    tab1, tab2, tab3, tab4 = st.tabs(["📌 Summary", "❓ Q&A", "📖 Full Text", "📝 History"])

    def run_model(prompt):
        if model_choice.startswith("Groq"):
            model_clean = model_choice.split(" - ")[1]
            return ask_groq(prompt, model_clean)
        elif model_choice.startswith("Gemini"):
            model_clean = model_choice.split(" - ")[1]
            return ask_gemini(prompt, model_clean)

    with tab1:
        st.subheader("📌 Summary of PDFs")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = run_model(f"Summarize the following text in bullet points:\n\n{pdf_text}")
                st.session_state["summary"] = summary
            st.success("✅ Summary generated!")
            st.write(st.session_state["summary"])
            st.download_button("⬇️ Download Summary", st.session_state["summary"], file_name="summary.txt")

    with tab2:
        st.subheader("❓ Ask Questions about PDFs")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                answer = run_model(f"Answer the following question based on the text:\n\nText: {pdf_text}\n\nQuestion: {query}")
                st.session_state["qa_list"].append((query, answer))
            st.success("✅ Answer generated!")
            st.write(answer)
            st.download_button("⬇️ Download Answer", answer, file_name="answer.txt")

    with tab3:
        st.subheader("📖 Full Extracted Text from PDFs")
        st.text_area("Full Text", pdf_text, height=400)

    with tab4:
        st.subheader("📝 Conversation History")
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**🤖 Assistant:** {msg['content']}")

    if st.button("📥 Download Everything as PDF"):
        summary_text = st.session_state.get("summary", "")
        qa_list = st.session_state.get("qa_list", [])
        history = st.session_state["chat_history"]
        pdf_buffer = export_to_pdf(summary_text, qa_list, history, pdf_text)
        st.download_button("⬇️ Save PDF Report", pdf_buffer, file_name="research_report.pdf", mime="application/pdf")

else:
    st.info("👆 Please upload one or more PDFs to get started.")
