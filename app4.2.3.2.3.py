"""
AI Job Hunt OS — Test Build v4.2.3.2.3 (Supabase Edition)
Streamlit app that tracks job applications and uses AI (Groq + AI Singapore)
to tailor resumes and generate cover letters.

Now with:
- Quick Import from Job Link inside the existing "New Application" view using Jina Reader + SLM
- "Edit & Export" view with a rich text (Markdown) editor (streamlit-lexical-extended)
- PDF / DOCX export, downloaded to the user's local drive (no .txt, no Storage bucket)
- Filenames written back to the existing `applications` table columns
  (resume_filename / cover_letter_filename / doc_number)
- LLM Monitoring (tokens, cost, latency, failures) including cached input tokens
- Human-in-the-Loop document generation choices
- Document naming format: PWA00286_Resume for [Role]_[Company]_DDMMYY_CV

Run:
    streamlit run app.py
"""

import hashlib
import io
import os
import re
import time
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from groq import Groq
from openai import OpenAI

# Optional UI components (the app degrades gracefully if they are missing)
try:
    from streamlit_lexical_extended import streamlit_lexical_extended
except ImportError:
    streamlit_lexical_extended = None

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Job Hunt OS",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ============================================================
# 🚀 API KEY & DATABASE CONFIGURATION
# ============================================================

def _get_secret(key: str, default=None):
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except ImportError:
        pass

    val = os.environ.get(key)
    if val:
        return val

    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass

    return default


GROQ_API_KEY = _get_secret("GROQ_API_KEY")
SEALION_API_KEY = _get_secret("SEALION_API_KEY")
HF_TOKEN = _get_secret("HF_TOKEN")
SUPABASE_DB_URL = _get_secret("SUPABASE_DB_URL")

if not GROQ_API_KEY and not SEALION_API_KEY:
    st.error(
        "❌ No API keys found. Set at least one of GROQ_API_KEY or "
        "SEALION_API_KEY in your `.env` file or Streamlit secrets."
    )
    st.stop()

if not SUPABASE_DB_URL:
    st.error(
        "❌ No SUPABASE_DB_URL found. Set your Supabase PostgreSQL connection string "
        "in your `.env` file or Streamlit secrets."
    )
    st.stop()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

AI_SINGAPORE_BASE_URL = "https://api.sea-lion.ai/v1"
ai_singapore_client = (
    OpenAI(api_key=SEALION_API_KEY, base_url=AI_SINGAPORE_BASE_URL)
    if SEALION_API_KEY else None
)

# Constants -----------------------------------------------------------------

STAGES = [
    {"id": "Applied",      "name": "Applied",      "color": "#3498db"},
    {"id": "Acknowledged", "name": "Acknowledged", "color": "#f39c12"},
    {"id": "Screening",    "name": "Screening",    "color": "#9b59b6"},
    {"id": "Interviewing", "name": "Interviewing", "color": "#e67e22"},
    {"id": "Offer",        "name": "Offer",        "color": "#27ae60"},
    {"id": "Rejected",     "name": "Rejected",     "color": "#e74c3c"},
]

# ============================================================
# 🔌 DATABASE URL & DRIVER SETUP
# ============================================================

if SUPABASE_DB_URL.startswith("postgres://"):
    DATABASE_URL = SUPABASE_DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif SUPABASE_DB_URL.startswith("postgresql://"):
    DATABASE_URL = SUPABASE_DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    DATABASE_URL = SUPABASE_DB_URL

# ============================================================
# 🧠 MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {
    "Sea-Lion v4 27B": {
        "provider": "ai_singapore",
        "model_id": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
        "description": "Gemma 3 27B fine-tuned for SEA languages by AI Singapore.",
    },
    "Sea-Lion v4 32B (Qwen)": {
        "provider": "ai_singapore",
        "model_id": "aisingapore/Qwen-SEA-LION-v4-32B-IT",
        "description": "Qwen3-32B fine-tuned for SEA tasks.",
    },
    "Groq: GPT-OSS 120B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "description": "OpenAI's open-weight MoE, 120B params, reasoning-capable.",
    },
    "Groq: GPT-OSS 20B": {
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "description": "Smaller, faster OpenAI open-weight model.",
    },
}

DEFAULT_MODEL_KEY = "Sea-Lion v4 27B"

# ============================================================
# 💰 MODEL PRICING (per 1M tokens, USD)
# ============================================================

MODEL_PRICING = {
    "Qwen/Qwen3.5-9B": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
        "context_window": 131072,
        "max_output": 32768,
    },
    "openai/gpt-oss-120b": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
        "context_window": 131072,
        "max_output": 65536,
    },
    "openai/gpt-oss-20b": {
        "input": 0.075,
        "cached_input": 0.037,
        "output": 0.30,
        "context_window": 131072,
        "max_output": 65536,
    },
    "aisingapore/Gemma-SEA-LION-v4-27B-IT": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
        "context_window": 131072,
        "max_output": 65536,
    },
    "aisingapore/Qwen-SEA-LION-v4-32B-IT": {
        "input": 0.0,
        "cached_input": 0.0,
        "output": 0.0,
        "context_window": 131072,
        "max_output": 65536,
    },
}


def get_pricing(model_id: str) -> dict:
    return MODEL_PRICING.get(
        model_id,
        {"input": 0.0, "output": 0.0, "cached_input": 0.0,
         "context_window": 0, "max_output": 0},
    )


def compute_cost_usd(
    model_id: str,
    uncached_input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    p = get_pricing(model_id)
    return (
        (uncached_input_tokens / 1_000_000) * p["input"]
        + (cached_input_tokens / 1_000_000) * p["cached_input"]
        + (output_tokens / 1_000_000) * p["output"]
    )


# ---------------------------------------------------------------------------
# 2. DATABASE LAYER
# ---------------------------------------------------------------------------

Base = declarative_base()


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True)
    doc_number = Column(Integer, default=0)
    company = Column(String, nullable=False)
    role = Column(String, nullable=False)
    job_url = Column(String, default="")
    status = Column(String, default="Applied")
    job_description = Column(Text, default="")
    resume_text = Column(Text, default="")
    cover_letter = Column(Text, default="")
    resume_filename = Column(String, default="")
    cover_letter_filename = Column(String, default="")
    match_score = Column(String, default="")
    applied_date = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id = Column(Integer, primary_key=True)
    called_at = Column(DateTime, default=datetime.now, index=True)
    provider = Column(String)
    model_key = Column(String)
    model_id = Column(String)
    operation = Column(String)
    input_tokens = Column(Integer, default=0)
    cached_input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    latency_seconds = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    success = Column(Integer, default=1)
    error_message = Column(Text, default="")

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


@st.cache_resource
def get_engine():
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=120,
        pool_size=5,
        max_overflow=5
    )
    return engine


@st.cache_resource
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def db_session():
    return get_session_factory()()


# ---------------------------------------------------------------------------
# 3. AI & SCRAPING LAYER
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Embedding model unavailable: {e}")
        return None


def fetch_job_details_from_url(url: str) -> dict:
    """
    Robust end-to-end job fetcher: Uses Jina Reader API with custom User-Agent headers
    to bypass JS rendering and anti-bot walls on portals like LinkedIn, JobStreet, and MyCareersFuture.
    """
    result = {"company": "", "role": "", "job_description": "", "job_url": url}
    if not url or not url.startswith("http"):
        return result

    markdown_content = ""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "application/json",
            "X-With-Generated-Alt": "true",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        response = requests.get(jina_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            markdown_content = response.text
        else:
            fallback_resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=10)
            if fallback_resp.status_code == 200:
                soup = BeautifulSoup(fallback_resp.text, "html.parser")
                for s in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
                    s.extract()
                markdown_content = soup.get_text(separator="\n", strip=True)

    except Exception as e:
        print(f"Error fetching URL content: {e}")
        return result

    if not markdown_content or len(markdown_content.strip()) < 50:
        return result

    system_prompt = (
        "You are a precise data extraction micro-service. "
        "Your task is to extract exactly 4 fields from the provided job posting text: "
        "'company', 'role', 'job_url', and 'job_description'. "
        "Output ONLY valid raw JSON. Do not include markdown code blocks, backticks, or conversational text."
    )
    
    user_prompt = f"""Job URL: {url}

Job Posting Content:
{markdown_content[:10000]}

Extract into this exact JSON structure:
{{
  "company": "",
  "role": "",
  "job_url": "{url}",
  "job_description": ""
}}
"""

    try:
        extraction_prompt_combined = f"{system_prompt}\n\n{user_prompt}"
        ai_response = call_llm(extraction_prompt_combined, operation="robust_url_extraction")
        
        cleaned = re.sub(r"^```(?:json)?\s*", "", ai_response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        parsed = json.loads(cleaned)

        result["company"] = parsed.get("company", "").strip()
        result["role"] = parsed.get("role", "").strip()
        result["job_url"] = parsed.get("job_url", "").strip() or url
        result["job_description"] = parsed.get("job_description", "").strip()

    except Exception as llm_err:
        print(f"LLM structured extraction failed: {llm_err}")
        lines = [line.strip() for line in markdown_content.splitlines() if line.strip()]
        result["role"] = lines[0][:100] if lines else ""
        result["job_description"] = markdown_content[:2000]

    return result


def get_active_model() -> tuple[str, str, str]:
    key = st.session_state.get("selected_model_key", DEFAULT_MODEL_KEY)
    entry = MODEL_REGISTRY.get(key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    return entry["provider"], entry["model_id"], key


def load_llm_for(provider: str, model_id: str):
    if provider == "groq":
        if not GROQ_API_KEY:
            return None, "Groq API key not set"
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=model_id, temperature=0.7, api_key=GROQ_API_KEY)
            return llm, f"Groq ({model_id})"
        except Exception as e:
            return None, f"Groq init failed: {e}"

    if provider == "ai_singapore":
        api_key_to_use = SEALION_API_KEY or HF_TOKEN
        base_url_to_use = AI_SINGAPORE_BASE_URL if SEALION_API_KEY else "https://api-inference.huggingface.co/v1/"

        if not api_key_to_use:
            return None, "API key or HF_TOKEN not set for Hugging Face / Sea-Lion endpoint"
      
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model_id,
                temperature=0.7,
                api_key=api_key_to_use,
                base_url=base_url_to_use,
            )
            return llm, f"Provider ({model_id})"
        except Exception as e:
            return None, f"Provider init failed: {e}"

    return None, f"Unknown provider: {provider}"


def _extract_usage_from_langchain(result) -> tuple[int, int, int]:
    um = getattr(result, "usage_metadata", None)
    if um:
        input_tokens = int(um.get("input_tokens", 0) or 0)
        output_tokens = int(um.get("output_tokens", 0) or 0)
        details = um.get("input_token_details") or {}
        cached = int(
            details.get("cache_read", 0)
            or details.get("cached_tokens", 0)
            or 0
        )
        return input_tokens, cached, output_tokens

    rm = getattr(result, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage") or {}
    input_tokens = int(tu.get("prompt_tokens", 0) or 0)
    output_tokens = int(tu.get("completion_tokens", 0) or 0)
    details = tu.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)
    return input_tokens, cached, output_tokens


def _extract_cached_tokens_from_usage(usage) -> int:
    if not usage:
        return 0

    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details") or {}

    if isinstance(details, dict):
        return int(details.get("cached_tokens", 0) or 0)

    return int(getattr(details, "cached_tokens", 0) or 0)


def svc_log_llm_usage(
    *, provider, model_key, model_id, operation,
    input_tokens, output_tokens, latency_seconds,
    cached_input_tokens=0, success=True, error_message="",
):
    total_prompt = int(input_tokens or 0)
    cached = int(cached_input_tokens or 0)
    uncached = max(total_prompt - cached, 0)
    output = int(output_tokens or 0)

    cost = (
        compute_cost_usd(model_id, uncached, output, cached)
        if success else 0.0
    )

    with db_session() as db:
        row = LLMUsage(
            provider=provider,
            model_key=model_key,
            model_id=model_id,
            operation=operation,
            input_tokens=uncached,
            cached_input_tokens=cached,
            output_tokens=output,
            total_tokens=uncached + cached + output,
            latency_seconds=float(latency_seconds or 0.0),
            cost_usd=cost,
            success=1 if success else 0,
            error_message=error_message or "",
        )
        db.add(row)
        db.commit()


def calculate_match_score(resume_text: str, job_description: str) -> float:
    if not resume_text or not job_description:
        return 0.0
    model = load_embedding_model()
    if model is None:
        return 0.0
    from sentence_transformers import util
    emb = model.encode([job_description, resume_text], convert_to_tensor=True)
    score = util.cos_sim(emb[0], emb[1])
    return float(score[0][0])


def call_llm(prompt: str, operation: str = "generic") -> str:
    provider, model_id, label = get_active_model()
    llm, status = load_llm_for(provider, model_id)

    if llm is None:
        svc_log_llm_usage(
            provider=provider,
            model_key=label,
            model_id=model_id,
            operation=operation,
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            latency_seconds=0.0,
            success=False,
            error_message=str(status),
        )
        return f"⚠️ {status}"

    start = time.time()
    try:
        result = llm.invoke(prompt)
        latency = time.time() - start
        in_tok, cached_tok, out_tok = _extract_usage_from_langchain(result)

        svc_log_llm_usage(
            provider=provider,
            model_key=label,
            model_id=model_id,
            operation=operation,
            input_tokens=in_tok,
            cached_input_tokens=cached_tok,
            output_tokens=out_tok,
            latency_seconds=latency,
            success=True,
        )
        return getattr(result, "content", str(result))
    except Exception as e:
        latency = time.time() - start
        svc_log_llm_usage(
            provider=provider,
            model_key=label,
            model_id=model_id,
            operation=operation,
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            latency_seconds=latency,
            success=False,
            error_message=str(e),
        )
        return f"⚠️ LLM call failed ({status}): {e}"


def tailor_resume(resume_text: str, job_description: str) -> str:
    prompt = f"""You are an objective professional resume writer. Your task is to draft a resume based ONLY on the provided job description.
Strict Rules:
Use quantifiable achievements and keywords from the job posting.
Do not invent companies, job titles, dates, or metrics (e.g., percentages, dollar amounts) that are missing from the raw text.
If a metric is not explicitly stated in the source text, describe the qualitative impact instead of guessing a number.
If information is missing, use placeholders like [Insert Metric] rather than fabricating data.
Keep the format clean and professional. Do NOT invent experience.

JOB DESCRIPTION:
{job_description}

CURRENT RESUME:
{resume_text}

TAILORED RESUME:"""
    return call_llm(prompt, operation="tailor_resume")


def generate_cover_letter(resume_text: str, job_description: str, company: str, role: str) -> str:
    prompt = f"""Write a professional, concise cover letter for the {role} role at {company}.
Ground it in the resume and job description below. Do NOT use placeholders like [Your Name].

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

COVER LETTER:"""
    return call_llm(prompt, operation="cover_letter")


def extract_pdf_text(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        return "".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


# ---------------------------------------------------------------------------
# 3b. DOCUMENT NAMING HELPERS
# ---------------------------------------------------------------------------

DOC_NUMBER_START = 291


def sanitize_filename_part(value: str, max_len: int = 60) -> str:
    value = (value or "").strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")[:max_len].strip(" .")


def clean_text(s) -> str:
    return re.sub(r"\*+", "", str(s or "")).strip()


def _today_ddmmyy() -> str:
    try:
        now = datetime.now(ZoneInfo("Asia/Singapore"))
    except Exception:
        now = datetime.now()
    return now.strftime("%d%m%y")


def get_next_doc_number() -> int:
    from sqlalchemy import func

    with db_session() as db:
        max_doc = db.query(func.max(Application.doc_number)).scalar()

    if max_doc:
        return max(int(max_doc) + 1, DOC_NUMBER_START)
    return DOC_NUMBER_START


def build_document_filename(
    doc_number: int,
    role: str,
    company: str,
    doc_type: str,
) -> str:
    date_str = _today_ddmmyy()
    role_part = sanitize_filename_part(role)
    company_part = sanitize_filename_part(company)
    prefix = f"PWA{int(doc_number):05d}"

    if doc_type.upper() == "CV":
        return f"{prefix}_Resume for {role_part}_{company_part}_{date_str}_CV"

    return f"{prefix}_Cover Letter for {role_part}_{company_part}_{date_str}_CL"


# ---------------------------------------------------------------------------
# 3c. EXPORT HELPERS (Markdown -> PDF / DOCX)
# ---------------------------------------------------------------------------

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _unescape_md(text: str) -> str:
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", text)


def _parse_markdown_blocks(md: str):
    blocks = []
    in_skills_or_competencies = False
    
    for raw in (md or "").replace("\r\n", "\n").split("\n"):
        line = re.sub(r"(<br\s*/?>|\\)\s*$", "", raw.rstrip()).rstrip()
        if not line.strip():
            in_skills_or_competencies = False
            blocks.append(("blank", ""))
            continue
            
        if re.match(r"^\s*([-*_])\1{2,}\s*$", line):
            blocks.append(("blank", ""))
            continue
            
        m_head = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m_head:
            heading_text = m_head.group(2).replace("**", "").strip().upper()
            if "CORE COMPETENCIES" in heading_text or "TECHNICAL SKILLSETS" in heading_text or "EDUCATION" in heading_text:
                in_skills_or_competencies = True
            else:
                in_skills_or_competencies = False
            blocks.append((f"h{len(m_head.group(1))}", _unescape_md(m_head.group(2).replace("**", "").strip())))
            continue
        
        m_bullet = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if m_bullet:
            blocks.append(("bullet", m_bullet.group(1).strip()))
            continue
            
        if in_skills_or_competencies:
            is_job_header = bool(re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\b20\d{2}\b)", line)) or line.endswith(", Singapore")
            if not is_job_header:
                blocks.append(("bullet", line.strip()))
                continue

        blocks.append(("para", line.strip()))
    return blocks


def _inline_tokens(text: str):
    for part in re.split(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            yield _unescape_md(part[2:-2]), True, False
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            yield _unescape_md(part[1:-1]), False, True
        else:
            yield _unescape_md(part), False, False


def markdown_to_docx_bytes(md: str) -> bytes:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    for kind, text in _parse_markdown_blocks(md):
        if kind == "blank":
            continue
        if kind in ("h1", "h2", "h3"):
            doc.add_heading(text, level=int(kind[1]))
            continue
        para = (
            doc.add_paragraph(style="List Bullet")
            if kind == "bullet" else doc.add_paragraph()
        )
        for t, bold, italic in _inline_tokens(text):
            run = para.add_run(t)
            run.bold = bold
            run.italic = italic

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def markdown_to_pdf_bytes(md: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    if md:
        md = (
            md.replace("\u00a0", " ")
              .replace("\u202f", " ")
              .replace("\u2002", " ")
              .replace("\u2003", " ")
              .replace("\u2011", "-")
              .replace("\u2013", "-")
              .replace("\u2014", "-")
              .replace("\u201c", '"')
              .replace("\u201d", '"')
              .replace("\u2018", "'")
              .replace("\u2019", "'")
        )

    buf = io.BytesIO()
    pdf = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    base = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=base["BodyText"],
                          fontName="Helvetica", fontSize=10.5, leading=14)
    heads = {
        "h1": ParagraphStyle("H1", parent=body, fontName="Helvetica-Bold",
                             fontSize=16, leading=20, spaceAfter=6),
        "h2": ParagraphStyle("H2", parent=body, fontName="Helvetica-Bold",
                             fontSize=13, leading=17, spaceBefore=6, spaceAfter=4),
        "h3": ParagraphStyle("H3", parent=body, fontName="Helvetica-Bold",
                             fontSize=11.5, leading=15, spaceBefore=4, spaceAfter=2),
    }
    bullet_style = ParagraphStyle(
        "Bullet", 
        parent=body, 
        leftIndent=18, 
        bulletIndent=6, 
        spaceAfter=3
    )

    def to_markup(text: str) -> str:
        cleaned_text = re.sub(r"^\s*[-*•]\s+", "", text)
        out = []
        for t, bold, italic in _inline_tokens(cleaned_text):
            s = xml_escape(t)
            if bold:
                s = f"<b>{s}</b>"
            if italic:
                s = f"<i>{s}</i>"
            out.append(s)
        return "".join(out)

    story = []
    for kind, text in _parse_markdown_blocks(md):
        if kind == "blank":
            story.append(Spacer(1, 6))
        elif kind in heads:
            story.append(Paragraph(xml_escape(text), heads[kind]))
        elif kind == "bullet":
            story.append(Paragraph(to_markup(text), bullet_style, bulletText="•"))
        else:
            story.append(Paragraph(to_markup(text), body))
    if not story:
        story.append(Paragraph(" ", body))

    pdf.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 4. SERVICE LAYER
# ---------------------------------------------------------------------------

def svc_list_applications():
    with db_session() as db:
        return [a.to_dict() for a in db.query(Application).order_by(Application.last_updated.desc()).all()]


def svc_get_application(app_id: int):
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        return app.to_dict() if app else None


def svc_create_application(**kwargs):
    with db_session() as db:
        app = Application(**kwargs)
        db.add(app)
        db.commit()
        db.refresh(app)
        return app.id


def svc_update_application(app_id: int, **fields) -> bool:
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        if not app:
            return False
        for k, v in fields.items():
            setattr(app, k, v)
        app.last_updated = datetime.now()
        db.commit()
        return True


# ---------------------------------------------------------------------------
# 5. UI VIEW IMPLEMENTATION: NEW APPLICATION
# ---------------------------------------------------------------------------

def view_new_application():
    st.title("➕ New Application")
    st.caption(
        "Paste the job description and upload your resume, or paste a job link "
        "below to pull details automatically."
    )

    for key in ["app_company", "app_role", "app_url", "app_desc"]:
        if key not in st.session_state:
            st.session_state[key] = ""

    with st.container(border=True):
        st.markdown("**🔗 Quick Import from Job Link**")
        col_url, col_btn = st.columns([4, 1])
        with col_url:
            quick_url = st.text_input(
                "Job Link",
                placeholder="https://...",
                label_visibility="collapsed",
                key="quick_url_input"
            )
        with col_btn:
            if st.button("Fetch Details", use_container_width=True):
                if quick_url:
                    with st.spinner("Pulling job information via AI..."):
                        scraped = fetch_job_details_from_url(quick_url)
                        st.session_state["app_company"] = scraped.get("company", "")
                        st.session_state["app_role"] = scraped.get("role", "")
                        st.session_state["app_url"] = scraped.get("job_url", quick_url)
                        st.session_state["app_desc"] = scraped.get("job_description", "")
                    
                    if st.session_state["app_company"] or st.session_state["app_role"]:
                        st.success("Details pulled successfully! Review them below.")
                    else:
                        st.warning("⚠️ Scraper could not extract fields (site may use strict bot protection). Please fill them in manually.")
                    st.rerun()
                else:
                    st.warning("Please paste a URL first.")

    _, _, model_label = get_active_model()
    st.info(f"Using model: **{model_label}**")

    with st.form("new_app_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        company = c1.text_input("Company Name *", value=st.session_state["app_company"])
        role = c2.text_input("Role Title *", value=st.session_state["app_role"])
        job_url = st.text_input("Job URL (optional)", value=st.session_state["app_url"])
        job_description = st.text_area("Job Description", value=st.session_state["app_desc"], height=180)

        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        st.markdown("**Human-in-the-loop generation**")
        generation_choice = st.radio(
            "Choose what AI should generate",
            [
                "Resume only",
                "Cover letter only",
                "Resume and cover letter",
                "None",
            ],
            index=2,
            horizontal=True,
        )

        submitted = st.form_submit_button("🔍 Analyze & Add", type="primary")

    if submitted:
        for key in ["app_company", "app_role", "app_url", "app_desc"]:
            st.session_state[key] = ""

        company, role = clean_text(company), clean_text(role)
        if not company or not role:
            st.error("Company and Role are required.")
            return

        resume_text = extract_pdf_text(resume_file) if resume_file else ""
        match_str = ""

        tailored = ""
        cover = ""
        resume_filename = ""
        cover_letter_filename = ""
        doc_number = 0

        if resume_text and job_description:
            with st.spinner("Scoring resume against job..."):
                score = calculate_match_score(resume_text, job_description)
                match_str = f"{score:.0%}"
                st.metric("Profile Match Score", match_str)

            generate_resume = generation_choice in [
                "Resume only",
                "Resume and cover letter",
            ]
            generate_cover = generation_choice in [
                "Cover letter only",
                "Resume and cover letter",
            ]

            if generate_resume:
                with st.spinner(f"Tailoring resume with {model_label}..."):
                    tailored = tailor_resume(resume_text, job_description)

            if generate_cover:
                base_resume = tailored if tailored else resume_text
                with st.spinner(f"Drafting cover letter with {model_label}..."):
                    cover = generate_cover_letter(
                        base_resume,
                        job_description,
                        company,
                        role,
                    )

            if tailored or cover:
                doc_number = get_next_doc_number()

                if tailored:
                    resume_filename = build_document_filename(
                        doc_number, role, company, "CV"
                    )

                if cover:
                    cover_letter_filename = build_document_filename(
                        doc_number, role, company, "CL"
                    )

            st.success("Document generation complete.")

            if tailored:
                with st.expander("📄 Tailored Resume (preview)", expanded=False):
                    st.text_area(
                        "Tailored Resume",
                        tailored,
                        height=280,
                        label_visibility="collapsed",
                        disabled=True,
                    )

            if cover:
                with st.expander("✉️ Cover Letter (preview)", expanded=False):
                    st.text_area(
                        "Cover Letter",
                        cover,
                        height=280,
                        label_visibility="collapsed",
                        disabled=True,
                    )

            app_id = svc_create_application(
                company=company,
                role=role,
                job_url=job_url,
                job_description=job_description,
                resume_text=tailored if tailored else resume_text,
                cover_letter=cover,
                match_score=match_str,
                status="Applied",
                doc_number=doc_number,
                resume_filename=resume_filename,
                cover_letter_filename=cover_letter_filename,
            )
            st.session_state["edit_app_id"] = app_id
            st.success(f"✅ Application #{app_id} saved to Tracker.")
            if tailored or cover:
                st.info(
                    "Open **✏️ Edit & Export** in the sidebar to polish the "
                    "documents and download them as PDF or DOCX."
                )

        else:
            app_id = svc_create_application(
                company=company,
                role=role,
                job_url=job_url,
                job_description=job_description,
                status="Applied",
            )
            st.info(
                f"Application #{app_id} saved without AI analysis. "
                "Upload a PDF + paste a job description to enable AI features."
            )


# ---------------------------------------------------------------------------
# 6. MAIN APP ROUTER
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("AI Job Hunt OS")
    
    # Model Selector in Sidebar
    model_keys = list(MODEL_REGISTRY.keys())
    selected_model = st.sidebar.selectbox("Active AI Model", model_keys, index=0)
    st.session_state["selected_model_key"] = selected_model

    menu_option = st.sidebar.radio("Navigation", ["Tracker", "New Application", "Settings"])

    if menu_option == "New Application":
        view_new_application()
    elif menu_option == "Tracker":
        st.title("📊 Application Tracker")
        apps = svc_list_applications()
        if apps:
            df = pd.DataFrame(apps)
            st.dataframe(df[["id", "company", "role", "status", "match_score", "applied_date"]])
        else:
            st.info("No job applications tracked yet. Use 'New Application' to add one!")
    else:
        st.title("⚙️ Settings")
        st.write("Database URL connected securely via Supabase.")


if __name__ == "__main__":
    main()
