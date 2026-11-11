"""
AI Job Hunt OS — Test Build v4.2.3.3.0 (Supabase + MCP Edition)
Streamlit app that tracks job applications and uses AI (Groq + AI Singapore)
to tailor resumes, generate cover letters, and connect via Model Context Protocol (MCP).

Run:
    streamlit run app.py
"""

import hashlib
import io
import os
import re
import time
import asyncio
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

# Optional MCP imports (graceful degradation if mcp SDK is not installed)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Optional UI components (the app degrades gracefully if they are missing)
try:
    from streamlit_lexical_extended import streamlit_lexical_extended
except ImportError:
    streamlit_lexical_extended = None

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Job Hunt OS (MCP Enabled)",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

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

# Optional MCP Server configuration via environment or secrets
MCP_SERVER_COMMAND = _get_secret("MCP_SERVER_COMMAND", "npx")
MCP_SERVER_ARGS = _get_secret("MCP_SERVER_ARGS", "@modelcontextprotocol/server-filesystem,/tmp")

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
# 🌐 MODEL REGISTRY & MCP CLIENT HELPER
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

# --- MCP Client Connection Wrapper ---
async def _run_mcp_query(command: str, args: list[str], prompt: str) -> dict:
    """Connects to an MCP stdio server, initializes session, lists tools, and executes a test/query."""
    if not MCP_AVAILABLE:
        return {"success": False, "error": "mcp python package not installed. Run `pip install mcp`."}
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=None
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools_list = [{"name": t.name, "description": t.description} for t in tools_resp.tools]
                return {
                    "success": True,
                    "tools": tools_list,
                    "server": f"{command} {' '.join(args)}"
                }
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_mcp_connection(command: str, args_str: str) -> dict:
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run_mcp_query(command, args, ""))
        loop.close()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================
# 💰 MODEL PRICING & DATABASE MODELS
# ============================================================

MODEL_PRICING = {
    "openai/gpt-oss-120b": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "context_window": 131072, "max_output": 65536},
    "openai/gpt-oss-20b": {"input": 0.075, "cached_input": 0.037, "output": 0.30, "context_window": 131072, "max_output": 65536},
    "aisingapore/Gemma-SEA-LION-v4-27B-IT": {"input": 0.0, "cached_input": 0.0, "output": 0.0, "context_window": 131072, "max_output": 65536},
    "aisingapore/Qwen-SEA-LION-v4-32B-IT": {"input": 0.0, "cached_input": 0.0, "output": 0.0, "context_window": 131072, "max_output": 65536},
}

def get_pricing(model_id: str) -> dict:
    return MODEL_PRICING.get(model_id, {"input": 0.0, "output": 0.0, "cached_input": 0.0, "context_window": 0, "max_output": 0})

def compute_cost_usd(model_id: str, uncached_input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
    p = get_pricing(model_id)
    return ((uncached_input_tokens / 1_000_000) * p["input"] +
            (cached_input_tokens / 1_000_000) * p["cached_input"] +
            (output_tokens / 1_000_000) * p["output"])

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
    return create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=120, pool_size=5, max_overflow=5)

@st.cache_resource
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)

def db_session():
    return get_session_factory()()

# ---------------------------------------------------------------------------
# 3. AI, SCRAPING & SERVICE LAYERS
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        return None

def fetch_job_details_from_url(url: str) -> dict:
    result = {"company": "", "role": "", "job_description": "", "job_url": url}
    if not url or not url.startswith("http"):
        return result
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return result
        soup = BeautifulSoup(response.text, "html.parser")
        if soup.find("meta", property="og:title"):
            result["role"] = soup.find("meta", property="og:title").get("content", "").strip()
        if soup.find("meta", property="og:site_name"):
            result["company"] = soup.find("meta", property="og:site_name").get("content", "").strip()
        for script in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            script.extract()
        text = soup.get_text(separator="\n", strip=True)
        result["job_description"] = "\n".join([line for line in text.splitlines() if line.strip()][:150])
    except Exception:
        pass
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
            return ChatGroq(model=model_id, temperature=0.7, api_key=GROQ_API_KEY), f"Groq ({model_id})"
        except Exception as e:
            return None, f"Groq init failed: {e}"
    if provider == "ai_singapore":
        api_key_to_use = SEALION_API_KEY or HF_TOKEN
        base_url = AI_SINGAPORE_BASE_URL if SEALION_API_KEY else "https://api-inference.huggingface.co/v1/"
        if not api_key_to_use:
            return None, "API key not set for Sea-Lion endpoint"
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_id, temperature=0.7, api_key=api_key_to_use, base_url=base_url), f"Provider ({model_id})"
        except Exception as e:
            return None, f"Provider init failed: {e}"
    return None, f"Unknown provider: {provider}"

def _extract_usage_from_langchain(result) -> tuple[int, int, int]:
    um = getattr(result, "usage_metadata", None)
    if um:
        in_tok = int(um.get("input_tokens", 0) or 0)
        out_tok = int(um.get("output_tokens", 0) or 0)
        details = um.get("input_token_details") or {}
        cached = int(details.get("cache_read", 0) or details.get("cached_tokens", 0) or 0)
        return in_tok, cached, out_tok
    return 0, 0, 0

def svc_log_llm_usage(*, provider, model_key, model_id, operation, input_tokens, output_tokens, latency_seconds, cached_input_tokens=0, success=True, error_message=""):
    total_prompt = int(input_tokens or 0)
    cached = int(cached_input_tokens or 0)
    uncached = max(total_prompt - cached, 0)
    output = int(output_tokens or 0)
    cost = compute_cost_usd(model_id, uncached, output, cached) if success else 0.0

    with db_session() as db:
        row = LLMUsage(
            provider=provider, model_key=model_key, model_id=model_id, operation=operation,
            input_tokens=uncached, cached_input_tokens=cached, output_tokens=output,
            total_tokens=uncached + cached + output, latency_seconds=float(latency_seconds or 0.0),
            cost_usd=cost, success=1 if success else 0, error_message=error_message or ""
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
    return float(util.cos_sim(emb[0], emb[1])[0][0])

def call_llm(prompt: str, operation: str = "generic") -> str:
    provider, model_id, label = get_active_model()
    llm, status = load_llm_for(provider, model_id)
    if llm is None:
        svc_log_llm_usage(provider=provider, model_key=label, model_id=model_id, operation=operation, input_tokens=0, cached_input_tokens=0, output_tokens=0, latency_seconds=0.0, success=False, error_message=str(status))
        return f"⚠️ {status}"

    start = time.time()
    try:
        result = llm.invoke(prompt)
        latency = time.time() - start
        in_tok, cached_tok, out_tok = _extract_usage_from_langchain(result)
        svc_log_llm_usage(provider=provider, model_key=label, model_id=model_id, operation=operation, input_tokens=in_tok, cached_input_tokens=cached_tok, output_tokens=out_tok, latency_seconds=latency, success=True)
        return getattr(result, "content", str(result))
    except Exception as e:
        latency = time.time() - start
        svc_log_llm_usage(provider=provider, model_key=label, model_id=model_id, operation=operation, input_tokens=0, cached_input_tokens=0, output_tokens=0, latency_seconds=latency, success=False, error_message=str(e))
        return f"⚠️ LLM call failed: {e}"

def tailor_resume(resume_text: str, job_description: str) -> str:
    prompt = f"""You are an objective professional resume writer. Tailor the resume based ONLY on the job description.\nJOB:\n{job_description}\n\nRESUME:\n{resume_text}\n\nTAILORED RESUME:"""
    return call_llm(prompt, operation="tailor_resume")

def generate_cover_letter(resume_text: str, job_description: str, company: str, role: str) -> str:
    prompt = f"""Write a professional cover letter for {role} at {company}.\nJOB:\n{job_description}\n\nRESUME:\n{resume_text}\n\nCOVER LETTER:"""
    return call_llm(prompt, operation="cover_letter")

def extract_pdf_text(uploaded_file) -> str:
    try:
        from PyPDF2 import PdfReader
        return "".join((page.extract_text() or "") for page in PdfReader(uploaded_file).pages)
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# 4. SERVICE LAYER (Applications)
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

def svc_update_status(app_id: int, new_status: str):
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        if app:
            app.status = new_status
            app.last_updated = datetime.now()
            db.commit()
            return True
    return False

def svc_delete_application(app_id: int):
    with db_session() as db:
        app = db.query(Application).filter(Application.id == app_id).first()
        if app:
            db.delete(app)
            db.commit()

def svc_list_llm_usage(days: int = 7):
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=days)
    with db_session() as db:
        rows = db.query(LLMUsage).filter(LLMUsage.called_at >= cutoff).order_by(LLMUsage.called_at.desc()).all()
        return [r.to_dict() for r in rows]

# ---------------------------------------------------------------------------
# 5. UI LAYER & MCP VIEW
# ---------------------------------------------------------------------------

def render_model_selector():
    st.sidebar.markdown("### 🧠 AI Model")
    options = list(MODEL_REGISTRY.keys())
    default_idx = options.index(DEFAULT_MODEL_KEY) if DEFAULT_MODEL_KEY in options else 0

    selected = st.sidebar.selectbox("Select model", options, index=default_idx, key="selected_model_widget", label_visibility="collapsed")
    st.session_state["selected_model_key"] = selected
    entry = MODEL_REGISTRY[selected]
    st.sidebar.caption(f"Provider: **{entry['provider']}**")


def render_sidebar():
    with st.sidebar:
        st.title("💼 AI Job Hunt OS")
        st.caption("AI-powered job search + MCP Client")
        st.markdown("---")
        nav = st.radio("Navigation", ["📊 Tracker", "➕ New Application", "🔌 MCP Client", "📈 Analytics", "🔍 Monitoring", "⚙️ Settings"], index=0, label_visibility="collapsed")
        st.markdown("---")
        search = st.text_input("🔍 Search company / role", "")
        st.markdown("---")
        render_model_selector()
    return nav, search


def view_mcp_client():
    st.title("🔌 Model Context Protocol (MCP) Client")
    st.caption("Connect your Streamlit application to local or remote MCP servers to inspect tools and resources.")

    if not MCP_AVAILABLE:
        st.warning("⚠️ The `mcp` Python package is not installed in your environment. Run `pip install mcp` to enable MCP tool calling.")
        return

    with st.form("mcp_form"):
        col1, col2 = st.columns([2, 3])
        with col1:
            cmd = st.text_input("Command", value=MCP_SERVER_COMMAND)
        with col2:
            args = st.text_input("Arguments (comma separated)", value=MCP_SERVER_ARGS)
        
        submitted = st.form_submit_button("Test MCP Connection & List Tools", type="primary")

    if submitted:
        with st.spinner("Connecting to MCP server via stdio..."):
            res = test_mcp_connection(cmd, args)
            if res["success"]:
                st.success(f"Connected successfully to MCP Server!")
                st.json(res["tools"])
            else:
                st.error(f"MCP Connection failed: {res['error']}")


def view_tracker(search: str):
    st.title("📊 Tracker")
    apps = svc_list_applications()
    df = pd.DataFrame(apps) if apps else pd.DataFrame(columns=["id", "company", "role", "status"])
    if search and not df.empty:
        df = df[df["company"].str.contains(search, case=False, na=False) | df["role"].str.contains(search, case=False, na=False)]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applications", len(df))
    c2.metric("Interviewing", int((df["status"] == "Interviewing").sum()) if not df.empty else 0)
    c3.metric("Offers", int((df["status"] == "Offer").sum()) if not df.empty else 0)
    st.markdown("---")
    st.dataframe(df, width="stretch", hide_index=True)


def view_analytics():
    st.title("📈 Analytics")
    apps = svc_list_applications()
    if not apps:
        st.info("No data yet.")
        return
    df = pd.DataFrame(apps)
    st.bar_chart(df["status"].value_counts())


def view_monitoring():
    st.title("🔍 LLM Monitoring")
    rows = svc_list_llm_usage(days=7)
    if not rows:
        st.info("No LLM calls recorded yet.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def view_settings():
    st.title("⚙️ Settings")
    st.write(f"- MCP Package Installed: {'✅ Yes' if MCP_AVAILABLE else '❌ No'}")
    st.write(f"- Supabase Connected: {'✅ Yes' if SUPABASE_DB_URL else '❌ No'}")


def main():
    if "selected_model_key" not in st.session_state:
        st.session_state["selected_model_key"] = DEFAULT_MODEL_KEY

    nav, search = render_sidebar()

    if nav == "📊 Tracker":
        view_tracker(search)
    elif nav == "➕ New Application":
        st.title("➕ New Application")
        st.info("Use the main Tracker view or quick import.")
    elif nav == "🔌 MCP Client":
        view_mcp_client()
    elif nav == "📈 Analytics":
        view_analytics()
    elif nav == "🔍 Monitoring":
        view_monitoring()
    elif nav == "⚙️ Settings":
        view_settings()


if __name__ == "__main__":
    main()

