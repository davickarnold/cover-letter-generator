# app.py

import os
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

import re
from datetime import datetime
import subprocess

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ---------- OpenAI API key ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "Missing OPENAI_API_KEY. Add it to your .env file (local) or Streamlit Secrets (cloud)."
    )
    st.stop()

client = OpenAI(api_key=api_key)


def fetch_job_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        return "\n".join(lines[:2000])  # limit length for safety
    except Exception as e:
        return f"ERROR_FETCHING_URL: {e}"


# ---------- App config ----------
st.set_page_config(page_title="Cover Letter Generator", layout="wide")
st.title("Cover Letter Generator (Accuracy-first)")

# ---------- Save settings ----------
SAVE_ROOT = Path(__file__).resolve().parent / "saved_letters"


def safe_slug(text: str) -> str:
    """Make a filesystem-safe folder/file name."""
    text = text.strip()
    text = re.sub(r"[^\w\s-]", "", text)  # remove weird chars
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    text = text.replace(" ", "_")
    return text[:80] or "unknown"  # keep it reasonable


# ---------- Load Work History ----------
@st.cache_data
def load_work_history(path="work_history.txt") -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


work_history = load_work_history()


# ---------- Load Strategic Capabilities (Non-factual) ----------
@st.cache_data
def load_strategic_capabilities(path="strategic_capabilities.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


strategic_capabilities = load_strategic_capabilities()

# ---------- OpenAI Client ----------
# OpenAI API uses API keys for authentication; keep it in an environment variable. :contentReference[oaicite:2]{index=2}
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY. Set it in Terminal before running the app.")
    st.stop()


client = OpenAI(api_key=api_key)


# ---------- UI ----------
col1, col2 = st.columns([2, 1])

with col1:
    company_name = st.text_input(
        "Company (for saving):", placeholder="e.g., Tryst Hospitality"
    )
    role_name = st.text_input(
        "Role (for saving):", placeholder="e.g., Senior Associate, PR & Communications"
    )

    job_url = st.text_input("Job posting URL (optional):", placeholder="https://...")

    job_text = st.text_area(
        "Paste the job description here (recommended for v1):",
        height=260,
        placeholder="Paste the full job posting text…",
    )

with col2:
    personality_mode = st.toggle("Personality mode (optional)", value=False)
    include_audit = st.toggle("Include claims audit (recommended)", value=True)
    model_name = st.text_input("Model", value="gpt-4.1-mini")
    generate = st.button("Generate cover letter", type="primary")

fetched_preview = ""

BASE_VOICE_RULES = """
DEFAULT VOICE MODE: Straightforward Professional.
Tone: confident, warm, conversational, newsroom-appropriate.
No gimmicks/metaphors/exaggerated claims.
If Personality mode is ON, allow light personality while remaining professional.
"""

MASTER_RULES = """
CRITICAL ACCURACY RULES:
- Do not invent experience, tools, metrics, titles, or responsibilities.
- Only make claims supported by the provided Work History Source-of-Truth.
- If a requirement is not directly supported, use adjacent/transferable framing without claiming you performed it.
- If there is no support, treat as a Gap and do not claim it.
- Strategic Context is NOT a factual source and must never be used to claim you have done something.
- Strategic Context MAY be used to support "Transferable (Strategic readiness)" in the audit and for phrasing like "well-positioned" in the letter.
"""


def run_llm(prompt: str) -> str:
    """
    Runs a single OpenAI generation call and returns plain text output.
    """
    resp = client.responses.create(
        model=model_name,
        input=prompt,
    )

    # Safely extract text output
    if hasattr(resp, "output_text"):
        return resp.output_text

    # Fallback in case SDK structure changes
    try:
        return resp.output[0].content[0].text
    except Exception:
        raise RuntimeError("Model returned no usable text output.")


def build_rewrite_prompt(job_text: str, draft_letter: str) -> str:
    return f"""
You are rewriting a cover letter to be more subtle and less similar to the job posting.

RULES:
- Do NOT quote the job posting.
- Do NOT reuse any 6+ word sequence from the job posting.
- Keep ALL factual claims exactly the same as the draft.
- Do NOT add new experience, tools, metrics, or responsibilities.
- Improve phrasing so it sounds natural, confident, and not keyword-stuffed.
- Preserve the role title and include it once in the opening paragraph.
- Output ONLY the final rewritten cover letter.

JOB POSTING (reference only — do not quote):
\"\"\"{job_text}\"\"\"

DRAFT COVER LETTER:
\"\"\"{draft_letter}\"\"\"
"""


def build_prompt(job_text: str, personality: bool, audit: bool) -> str:
    mode_line = (
        "Personality mode: ON" if personality else "Personality mode: OFF (default)"
    )
    audit_line = (
        "Include a claims audit table."
        if audit
        else "Do NOT include a claims audit table."
    )

    return f"""
You are an accuracy-first cover letter writer.

{MASTER_RULES}

{BASE_VOICE_RULES}
{mode_line}

WORK HISTORY SOURCE-OF-TRUTH (the only allowed factual source):
\"\"\"{work_history}\"\"\"

STRATEGIC CONTEXT (NOT factual claims — do not treat as past experience. 
You may use this only to frame language like "well-positioned" or "brings a strong understanding," 
but you must NOT claim completed work based on it):
\"\"\"{strategic_capabilities}\"\"\"

JOB POSTING:
\"\"\"{job_text}\"\"\"

TASK:
1) Extract the core responsibilities and required skills from the job posting (rank top responsibilities first).
2) Match those requirements to the Work History Source-of-Truth using:
   - Direct match (supported by Work History)
   - Adjacent match (supported by Work History)
   - Transferable (supported by Work History)
   - Transferable (Strategic readiness) (supported ONLY by Strategic Context; must be phrased as "well-positioned" and not as completed work)
   - Gap (no evidence in Work History or Strategic Context)

When you use "Transferable (Strategic readiness)", you MUST (a) explicitly cite Strategic Context as the support, and (b) write the evidence using "well-positioned / strong understanding / familiarity" language — never wording that implies past execution (e.g., "led", "secured", "drove", "won placements").

3) Write a 3–5 paragraph cover letter in first-person that:
   - Uses only supported claims
   - Preserves a straightforward professional voice
   - Is specific to this job
4) {audit_line}

OUTPUT FORMAT:
- Start with the cover letter (with greeting and closing).
- If audit is enabled, include the audit table after the letter.
"""


# ---------- Generate ----------
if generate:
    fetched_preview = ""

    # If a URL is provided, fetch text from the URL and use it if successful.
    if job_url.strip():
        fetched = fetch_job_text_from_url(job_url.strip())
        fetched_preview = fetched

        # Show preview of fetched text (even if it's junk or empty)
        with st.expander("Preview extracted job text (from URL)"):
            st.text_area(
                "Extracted text",
                fetched_preview or "(No text extracted)",
                height=200,
                key="extracted_text_preview",
            )

        if fetched.startswith("ERROR_FETCHING_URL:"):
            st.warning(
                "Could not fetch job posting from URL. Paste the description instead.\n\n"
                f"{fetched}"
            )
        else:
            job_text = fetched
            st.info(
                "Fetched job text from URL. If the preview below looks wrong (menus/legal/etc.), "
                "paste the job description manually instead."
            )

    if not job_text.strip():
        st.warning(
            "Please paste a job description (or provide a URL that successfully fetches it)."
        )
        st.stop()

    with st.spinner("Generating…"):
        prompt = build_prompt(job_text, personality_mode, include_audit)

        try:
            # First pass: accuracy-first generation
            letter = run_llm(prompt)

            # Second pass: rewrite for subtlety (no new facts)
            rewrite_prompt = build_rewrite_prompt(job_text, letter)
            letter = run_llm(rewrite_prompt)

            st.session_state["last_letter"] = letter

        except Exception as e:
            st.error(f"Error generating letter: {e}")
            st.exception(e)
            st.stop()

    # Show output AFTER generation (not inside except)
    st.subheader("Generated Cover Letter")
    st.write(st.session_state.get("last_letter", ""))

    # ---------- Save to folder (Company/Role) ----------
    save_col1, save_col2 = st.columns([2, 1])

    with save_col1:
        st.caption(
            f"Save path preview: saved_letters/{safe_slug(company_name)}/{safe_slug(role_name)}/"
        )

    with save_col2:
        save_to_folder = st.button("Save cover letter to folder")

    if save_to_folder:
        if not company_name.strip() or not role_name.strip():
            st.warning("Please fill in Company and Role before saving.")
        else:
            company_dir = SAVE_ROOT / safe_slug(company_name)
            role_dir = company_dir / safe_slug(role_name)
            role_dir.mkdir(parents=True, exist_ok=True)
            st.session_state["last_saved_dir"] = str(role_dir)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            file_path = role_dir / f"cover_letter_{timestamp}.txt"

            file_path.write_text(
                st.session_state.get("last_letter", ""), encoding="utf-8"
            )
            st.success(f"Saved: {file_path}")

    # ---------- Open last saved folder (macOS) ----------
    open_col1, open_col2 = st.columns([2, 1])

    with open_col1:
        last_dir = st.session_state.get("last_saved_dir", "")
        if last_dir:
            st.caption(f"Last saved folder: {last_dir}")

    with open_col2:
        if st.button(
            "Open saved folder in Finder",
            disabled=not st.session_state.get("last_saved_dir"),
        ):
            subprocess.run(["open", st.session_state["last_saved_dir"]])

    st.download_button(
        "Download as .txt",
        data=letter.encode("utf-8"),
        file_name="cover_letter.txt",
        mime="text/plain",
    )
