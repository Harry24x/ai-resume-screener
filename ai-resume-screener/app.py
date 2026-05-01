import streamlit as st
import pdfplumber
import os
import json
import re
from groq import Groq

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }
  .stApp { background: #f7f6f2; }
  .hero {
    background: linear-gradient(135deg, #01696f 0%, #0c4e54 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    color: #fff;
    text-align: center;
  }
  .hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.5rem; letter-spacing: -0.02em; }
  .hero p  { font-size: 1rem; opacity: 0.85; margin: 0; }
  .section-card {
    background: #fff;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 1.5rem;
  }
  .section-card h3 {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a7974;
    margin: 0 0 1rem;
  }
  .score-badge {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #fff;
    margin-bottom: 1.5rem;
  }
  .score-badge .score-num { font-size: 3.5rem; font-weight: 700; line-height: 1; }
  .score-badge .score-lbl { font-size: 0.9rem; opacity: 0.9; margin-top: 0.25rem; }
  .score-green  { background: linear-gradient(135deg, #437a22, #2e5c10); }
  .score-yellow { background: linear-gradient(135deg, #d19900, #b07a00); }
  .score-red    { background: linear-gradient(135deg, #a12c7b, #7d1e5e); }
  .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
  .chip { padding: 0.3rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 500; }
  .chip-match   { background: #d4dfcc; color: #1e3f0a; }
  .chip-missing { background: #e0ced7; color: #561740; }
  .rec-box {
    background: #f3f0ec;
    border-left: 3px solid #01696f;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #28251d;
  }
  .footer { text-align: center; padding: 2rem 0 1rem; font-size: 0.8rem; color: #7a7974; }
  .footer a { color: #01696f; text-decoration: none; font-weight: 500; }
  .footer a:hover { text-decoration: underline; }
  [data-testid="stFileUploader"] {
    background: #f9f8f5;
    border-radius: 10px;
    border: 2px dashed #d4d1ca;
    padding: 0.5rem;
  }
  .stButton > button {
    background: #01696f !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    width: 100%;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #0c4e54 !important; }
  .stTextArea textarea {
    border-radius: 8px !important;
    border: 1px solid #d4d1ca !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
  }
</style>
""", unsafe_allow_html=True)


def get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("GROQ_API_KEY", "")


def extract_pdf_text(uploaded_file) -> str:
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def analyse_resume(cv_text: str, job_desc: str, api_key: str) -> dict:
    client = Groq(api_key=api_key)
    system_prompt = """You are an expert technical recruiter and career coach.
Analyse the provided CV against the job description and return a JSON object
with EXACTLY these keys (no extra keys, no markdown fences):

{
  "match_score": <integer 0-100>,
  "matching_skills": [<string>, ...],
  "missing_skills": [<string>, ...],
  "recommendation": "<two to three sentence paragraph>"
}

Be specific. Skills should be concrete technologies, tools, or competencies."""

    user_prompt = f"""## CV
{cv_text[:6000]}

## Job Description
{job_desc[:3000]}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def score_class(score: int) -> str:
    if score >= 70:
        return "score-green"
    elif score >= 40:
        return "score-yellow"
    return "score-red"


def score_label(score: int) -> str:
    if score >= 70:
        return "Strong Match ✅"
    elif score >= 40:
        return "Partial Match ⚠️"
    return "Weak Match ❌"


st.markdown("""
<div class="hero">
  <h1>🎯 AI Resume Screener</h1>
  <p>Upload a CV and paste a job description — get an instant AI-powered match analysis.</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-card"><h3>📄 Upload CV (PDF)</h3>', unsafe_allow_html=True)
    uploaded_cv = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>💼 Job Description</h3>', unsafe_allow_html=True)
    job_description = st.text_area(
        "",
        placeholder="Paste the full job description here…",
        height=260,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    analyse_btn = st.button("⚡ Analyse Match", use_container_width=True)

with col_right:
    if analyse_btn:
        if not uploaded_cv:
            st.error("Please upload a CV PDF first.")
            st.stop()
        if not job_description.strip():
            st.error("Please paste a job description.")
            st.stop()

        api_key = get_api_key()
        if not api_key:
            st.error("🔑 GROQ_API_KEY not found. Add it to your .env file.")
            st.stop()

        with st.spinner("🤖 AI is analysing your CV…"):
            try:
                cv_text = extract_pdf_text(uploaded_cv)
                if not cv_text.strip():
                    st.error("Could not extract text from the PDF. Make sure it is not a scanned image.")
                    st.stop()
                result = analyse_resume(cv_text, job_description, api_key)
            except json.JSONDecodeError:
                st.error("The AI returned an unexpected format. Please try again.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        score    = result.get("match_score", 0)
        matching = result.get("matching_skills", [])
        missing  = result.get("missing_skills", [])
        rec      = result.get("recommendation", "")

        cls = score_class(score)
        lbl = score_label(score)
        st.markdown(f"""
        <div class="score-badge {cls}">
          <div class="score-num">{score}%</div>
          <div class="score-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3>✅ Matching Skills</h3>', unsafe_allow_html=True)
        if matching:
            chips_html = "".join(f'<span class="chip chip-match">{s}</span>' for s in matching)
            st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#7a7974;font-size:0.9rem'>No strong matches found.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3>⚠️ Gap / Missing Skills</h3>', unsafe_allow_html=True)
        if missing:
            chips_html = "".join(f'<span class="chip chip-missing">{s}</span>' for s in missing)
            st.markdown(f'<div class="chips">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#7a7974;font-size:0.9rem'>No significant gaps detected.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card"><h3>💡 AI Recommendation</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box">{rec}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="height:100%;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;gap:1rem;color:#7a7974;text-align:center;padding:4rem 2rem;">
          <div style="font-size:3rem;">🎯</div>
          <div style="font-size:1rem;font-weight:500;">Results will appear here</div>
          <div style="font-size:0.875rem;max-width:28ch;line-height:1.6;">
            Upload a PDF CV and paste a job description, then click Analyse Match.
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  Built by <a href="https://www.linkedin.com/in/harmanbhangu" target="_blank">Harman Bhangu</a>
  &nbsp;·&nbsp; Powered by Groq + Llama 3 · Streamlit
</div>
""", unsafe_allow_html=True)