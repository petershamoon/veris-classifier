"""VERIS Classifier - Gradio Web Application.

Dual-mode inference:
- Primary: Fine-tuned Mistral-7B-Instruct on ZeroGPU (no API key needed)
- Fallback: OpenAI API (user provides their own key)
"""

# ── HfFolder compatibility shim ──────────────────────────────────────────
# Gradio's oauth.py imports HfFolder from huggingface_hub, but HfFolder was
# removed in huggingface_hub >= 0.24. This shim must run BEFORE importing
# Gradio so the import chain doesn't break.
try:
    from huggingface_hub import HfFolder  # noqa: F401
except ImportError:
    import huggingface_hub

    class _HfFolder:
        """Minimal shim for the removed HfFolder class."""

        @classmethod
        def get_token(cls):
            return huggingface_hub.get_token()

        @classmethod
        def save_token(cls, token):
            huggingface_hub.login(token=token)

    huggingface_hub.HfFolder = _HfFolder
# ──────────────────────────────────────────────────────────────────────────

import json
import logging
import os
import csv
import tempfile
import time
from importlib import metadata, util
from typing import Any

import gradio as gr
from dotenv import load_dotenv

from src.veris_classifier.classifier import (
    answer_question,
    classify_incident,
)
from src.veris_classifier.validator import validate_classification

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZeroGPU support — only available on HF Spaces
# ---------------------------------------------------------------------------
IS_SPACES = os.getenv("SPACE_ID") is not None

spaces = None
if IS_SPACES:
    try:
        import spaces as _spaces

        # Local `spaces/` directory can shadow the HF `spaces` package.
        if hasattr(_spaces, "GPU"):
            spaces = _spaces
        else:
            raise ImportError("Imported `spaces` module has no GPU decorator")
    except Exception:
        try:
            # Load the installed HF spaces package directly from site-packages.
            dist = metadata.distribution("spaces")
            module_path = dist.locate_file("spaces/__init__.py")
            spec = util.spec_from_file_location("hf_spaces_runtime", module_path)
            if spec is None or spec.loader is None:
                raise ImportError("Could not load spaces package spec")
            _spaces = util.module_from_spec(spec)
            spec.loader.exec_module(_spaces)
            if hasattr(_spaces, "GPU"):
                spaces = _spaces
            else:
                raise ImportError("Installed spaces package has no GPU decorator")
        except Exception as e:
            logger.warning(
                "HF Spaces GPU decorator unavailable (%s). Falling back to non-GPU wrappers.",
                e,
            )

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* Global */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hero header */
.hero-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(139, 92, 246, 0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.05rem !important;
    color: #94a3b8 !important;
    margin: 0 0 20px 0 !important;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #cbd5e1;
}

/* Stats bar */
.stats-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.stat-card {
    flex: 1;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tabs */
.tabs {
    border: none !important;
}
button.tab-nav {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 24px !important;
}

/* Input areas */
textarea {
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* Buttons */
.primary-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 32px !important;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
}

/* Code output */
.code-output {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* Examples */
.examples-table {
    border-radius: 10px !important;
    overflow: hidden;
}

/* Model info banner */
.model-banner {
    background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    color: #94a3b8;
}
.model-banner strong {
    color: #34d399;
}
.model-banner .fallback {
    color: #fbbf24;
}

/* Section headers */
.section-header {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    margin-bottom: 8px !important;
}
.section-desc {
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
    margin-bottom: 16px !important;
}

/* About page cards */
.about-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 24px;
    margin-top: 32px;
    border-top: 1px solid #1e293b;
    color: #64748b;
    font-size: 0.85rem;
}
.footer a {
    color: #60a5fa;
    text-decoration: none;
}

.status-card {
    border: 1px solid #334155;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 10px;
    padding: 8px 12px;
}

#table-controls .wrap {
    align-items: end;
}

/* Mobile */
@media (max-width: 900px) {
    .hero-section {
        padding: 28px 20px;
    }
    .hero-title {
        font-size: 1.75rem !important;
    }
    .stats-row {
        flex-wrap: wrap;
    }
    .stat-card {
        min-width: calc(50% - 8px);
    }
}

@media (max-width: 560px) {
    .stat-card {
        min-width: 100%;
    }
    .hero-badges {
        gap: 8px;
    }
    .primary-btn {
        width: 100% !important;
    }
    #table-controls .wrap {
        gap: 8px !important;
    }
}
"""

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
EXAMPLES_CLASSIFY = [
    ["A hospital employee lost their unencrypted laptop containing patient records while traveling. The device was left in a taxi and never recovered."],
    ["Russian organized crime group used stolen credentials to access the company's web application and exfiltrated 50,000 customer credit card numbers over several weeks."],
    ["An employee emailed a spreadsheet containing salary information for all staff to their personal Gmail account, violating company data handling policy."],
    ["Attackers sent phishing emails to the finance department. One employee clicked the link and entered credentials on a fake login page. The attackers then used those credentials to initiate wire transfers totaling $2.3 million."],
    ["A ransomware attack encrypted all file servers after an employee opened a malicious email attachment. The company was unable to access critical systems for 5 days."],
    ["During a routine office move, several boxes of paper documents containing customer Social Security numbers were accidentally left at the old building and found by the new tenant."],
]

EXAMPLES_QA = [
    ["What is the difference between hacking and misuse in VERIS?"],
    ["How do I classify a phishing attack that led to ransomware?"],
    ["What are the three actor types in VERIS?"],
    ["When should I mark data_disclosure as 'Potentially' vs 'Yes'?"],
    ["What is the A4 Grid and how is it used?"],
    ["How does VERIS handle incidents with multiple threat actors?"],
]


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
ZEROGPU_QUEUE_HINT = "No GPU was available after"
SPACES_PAGE_URL = "https://huggingface.co/spaces/vibesecurityguy/veris-classifier"
SPACE_HOST_URL = "https://vibesecurityguy-veris-classifier.hf.space"
ZEROGPU_RETRY_ATTEMPTS = 2
ZEROGPU_RETRY_DELAY_SECONDS = 3


def _is_zerogpu_queue_timeout(err: Exception) -> bool:
    """Detect ZeroGPU queue timeout errors from the spaces runtime."""
    return ZEROGPU_QUEUE_HINT in str(err)


def _spaces_user_logged_in(
    request: gr.Request | None,
    profile: gr.OAuthProfile | None = None,
) -> bool:
    """True when a Spaces OAuth user is attached to this request."""
    if profile is not None:
        return True
    if request is None:
        return False
    if getattr(request, "username", None):
        return True
    # Gradio/HF OAuth stores profile info in session; use it as fallback signal.
    session = getattr(request, "session", None)
    if isinstance(session, dict) and session.get("oauth_info"):
        return True
    return False


def _session_status_markdown(
    request: gr.Request | None = None,
    profile: gr.OAuthProfile | None = None,
) -> str:
    """Render current Spaces auth status for the user."""
    if not IS_SPACES:
        return ""

    if _spaces_user_logged_in(request, profile):
        username = None
        if profile is not None:
            username = (
                getattr(profile, "preferred_username", None)
                or getattr(profile, "name", None)
            )
        if not username and request is not None:
            username = getattr(request, "username", None)
        if username:
            return (
                f"**Session status:** Logged in as `{username}`. "
                "ZeroGPU requests will use your account quota."
            )
        return "**Session status:** Logged in. ZeroGPU requests will use your account quota."

    return (
        "**Session status:** Not logged in. Click sign in to attach this browser session "
        "to your Hugging Face quota."
    )


def _run_with_zerogpu_retry(call):
    """Retry queue-timeout failures once before returning an error."""
    last_error = None
    for attempt in range(1, ZEROGPU_RETRY_ATTEMPTS + 1):
        try:
            return call()
        except Exception as e:
            last_error = e
            if _is_zerogpu_queue_timeout(e) and attempt < ZEROGPU_RETRY_ATTEMPTS:
                logger.warning(
                    "ZeroGPU queue timeout (attempt %d/%d). Retrying in %ss.",
                    attempt,
                    ZEROGPU_RETRY_ATTEMPTS,
                    ZEROGPU_RETRY_DELAY_SECONDS,
                )
                time.sleep(ZEROGPU_RETRY_DELAY_SECONDS)
                continue
            raise
    raise last_error



def _use_hf_model() -> bool:
    """Check if we should use the fine-tuned HF model."""
    # On HF Spaces, always try the local model first
    if IS_SPACES:
        return True
    # Locally, use HF model if VERIS_USE_HF is set
    return os.getenv("VERIS_USE_HF", "").lower() in ("1", "true", "yes")


def classify(
    description: str,
    api_key: str,
    request: gr.Request | None = None,
    profile: gr.OAuthProfile | None = None,
) -> str:
    """Classify an incident — uses HF model on Spaces, OpenAI otherwise."""
    if not description.strip():
        return json.dumps({"error": "Please enter an incident description."}, indent=2)
    if IS_SPACES and not _spaces_user_logged_in(request, profile):
        return json.dumps(
            {
                "error": (
                    "Please log in on Hugging Face and open this app from "
                    f"{SPACES_PAGE_URL}. ZeroGPU quota is per logged-in user."
                )
            },
            indent=2,
        )

    use_hf = _use_hf_model()

    # Local-only override: allow OpenAI fallback when running outside Spaces.
    if api_key.strip() and not IS_SPACES:
        use_hf = False

    if use_hf:
        try:
            result = _run_with_zerogpu_retry(lambda: _classify_gpu(description))
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"HF model error: {e}")
            if _is_zerogpu_queue_timeout(e):
                return json.dumps(
                    {"error": "ZeroGPU queue is full right now. Try again in 1-2 minutes."},
                    indent=2,
                )
            if IS_SPACES:
                return json.dumps({"error": f"Model inference failed: {str(e)}"}, indent=2)
            # Local fallback path only.
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                return json.dumps({"error": f"Model inference failed: {str(e)}"}, indent=2)
    else:
        key = api_key.strip() or os.getenv("OPENAI_API_KEY", "")
        if not key:
            return json.dumps({"error": "Please provide an OpenAI API key or wait for the model to load."}, indent=2)

    # OpenAI fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        result = classify_incident(client=client, description=description)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def ask(
    question: str,
    api_key: str,
    request: gr.Request | None = None,
    profile: gr.OAuthProfile | None = None,
) -> str:
    """Answer a VERIS question — uses HF model on Spaces, OpenAI otherwise."""
    if not question.strip():
        return "*Please enter a question.*"
    if IS_SPACES and not _spaces_user_logged_in(request, profile):
        return (
            "**Error:** Please log in on Hugging Face and open this app from "
            f"{SPACES_PAGE_URL}. ZeroGPU quota is per logged-in user."
        )

    use_hf = _use_hf_model()

    if api_key.strip() and not IS_SPACES:
        use_hf = False

    if use_hf:
        try:
            return _run_with_zerogpu_retry(lambda: _ask_gpu(question))
        except Exception as e:
            logger.error(f"HF model error: {e}")
            if _is_zerogpu_queue_timeout(e):
                return "**Error:** ZeroGPU queue is full right now. Try again in 1-2 minutes."
            if IS_SPACES:
                return f"**Error:** Model inference failed: {str(e)}"
            # Local fallback path only.
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                return f"**Error:** Model inference failed: {str(e)}"
    else:
        key = api_key.strip() or os.getenv("OPENAI_API_KEY", "")
        if not key:
            return "*Please provide an OpenAI API key or wait for the model to load.*"

    # OpenAI fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        return answer_question(client=client, question=question)
    except Exception as e:
        return f"**Error:** {str(e)}"


def _dimension_from_path(path: str) -> str:
    root = path.split(".", 1)[0].split("[", 1)[0]
    return root.title() if root else "General"


def _flatten_for_table(value: Any, path: str, rows: list[list[str]]) -> None:
    """Flatten nested VERIS JSON into table rows."""
    if isinstance(value, dict):
        if not value:
            rows.append([_dimension_from_path(path), path or "root", "{}"])
            return
        for key, subvalue in value.items():
            subpath = f"{path}.{key}" if path else key
            _flatten_for_table(subvalue, subpath, rows)
        return

    if isinstance(value, list):
        if not value:
            rows.append([_dimension_from_path(path), path or "root", "[]"])
            return
        # Keep scalar lists compact in one row.
        if all(not isinstance(item, (dict, list)) for item in value):
            rows.append([_dimension_from_path(path), path or "root", ", ".join(map(str, value))])
            return
        for i, subvalue in enumerate(value):
            _flatten_for_table(subvalue, f"{path}[{i}]", rows)
        return

    rows.append([_dimension_from_path(path), path or "root", str(value)])


def _classification_rows_from_json(raw_json: str) -> list[list[str]]:
    """Build table rows from classifier JSON output string."""
    if not raw_json.strip():
        return []
    try:
        parsed = json.loads(raw_json)
    except Exception:
        return [["Error", "raw_output", raw_json]]

    rows: list[list[str]] = []
    _flatten_for_table(parsed, "", rows)
    return rows


def _validation_summary_markdown(raw_json: str) -> str:
    """Build validation summary for the classification output."""
    try:
        parsed = json.loads(raw_json)
    except Exception:
        return ""

    if not isinstance(parsed, dict) or parsed.get("error"):
        return "**Validation:** Skipped."

    result = validate_classification(parsed)
    lines = [f"**Validation:** {'Passed' if result.valid else 'Issues found'}"]
    if result.errors:
        lines.append("**Errors**")
        lines.extend(f"- {err}" for err in result.errors[:8])
        if len(result.errors) > 8:
            lines.append(f"- ... {len(result.errors) - 8} more")
    if result.warnings:
        lines.append("**Warnings**")
        lines.extend(f"- {warn}" for warn in result.warnings[:8])
        if len(result.warnings) > 8:
            lines.append(f"- ... {len(result.warnings) - 8} more")
    return "\n".join(lines)


def _filter_classification_rows(
    rows: list[list[str]],
    dimension_filter: str,
    errors_only: bool,
) -> list[list[str]]:
    """Filter table rows by dimension and optionally error-only rows."""
    filtered: list[list[str]] = []
    for row in rows:
        if len(row) != 3:
            continue
        dimension, field, value = row

        if dimension_filter != "All" and dimension != dimension_filter:
            continue

        if errors_only:
            blob = f"{dimension} {field} {value}".lower()
            if "error" not in blob:
                continue

        filtered.append(row)
    return filtered


def _render_classification_output(
    raw_json: str,
    output_format: str,
    all_rows: list[list[str]],
    dimension_filter: str,
    errors_only: bool,
):
    """Render classification as JSON code or filtered table."""
    filtered_rows = _filter_classification_rows(all_rows, dimension_filter, errors_only)
    show_table = output_format == "Table"

    if show_table:
        return (
            gr.update(value=raw_json, visible=False),
            gr.update(value=filtered_rows, visible=True),
            gr.update(visible=True),
            gr.update(visible=True, interactive=bool(filtered_rows)),
        )

    return (
        gr.update(value=raw_json, visible=True),
        gr.update(value=[], visible=False),
        gr.update(visible=False),
        gr.update(visible=False, interactive=False),
    )


def _apply_table_filters(
    all_rows: list[list[str]],
    dimension_filter: str,
    errors_only: bool,
):
    """Apply table-only filters without re-running inference."""
    filtered_rows = _filter_classification_rows(all_rows, dimension_filter, errors_only)
    return (
        gr.update(value=filtered_rows),
        gr.update(interactive=bool(filtered_rows)),
    )


def _build_filtered_csv(
    all_rows: list[list[str]],
    dimension_filter: str,
    errors_only: bool,
):
    """Create downloadable CSV file for filtered rows."""
    filtered_rows = _filter_classification_rows(all_rows, dimension_filter, errors_only)
    if not filtered_rows:
        return gr.update(value=None, visible=False)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        newline="",
        encoding="utf-8",
    ) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["Dimension", "Field", "Value"])
        writer.writerows(filtered_rows)
        csv_path = tmp.name

    return gr.update(value=csv_path, visible=True)


def classify_and_render(
    description: str,
    api_key: str,
    output_format: str,
    dimension_filter: str,
    errors_only: bool,
    request: gr.Request | None = None,
    profile: gr.OAuthProfile | None = None,
):
    """Run classification and return display-ready outputs."""
    raw_json = classify(description, api_key, request=request, profile=profile)
    all_rows = _classification_rows_from_json(raw_json)
    validation_md = _validation_summary_markdown(raw_json)
    code_update, table_update, controls_update, export_btn_update = _render_classification_output(
        raw_json,
        output_format,
        all_rows,
        dimension_filter,
        errors_only,
    )
    return (
        raw_json,
        all_rows,
        validation_md,
        code_update,
        table_update,
        controls_update,
        export_btn_update,
        gr.update(value=None, visible=False),
    )


# ---------------------------------------------------------------------------
# GPU-decorated functions for ZeroGPU
# ---------------------------------------------------------------------------

def _gpu_wrapper(duration: int):
    """Use HF ZeroGPU decorator when available; otherwise no-op."""

    def passthrough(fn):
        return fn

    if IS_SPACES and spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(duration=duration)
    return passthrough


if IS_SPACES:
    @_gpu_wrapper(duration=120)
    def _classify_gpu(description: str) -> dict:
        """Classify incident using the fine-tuned model on ZeroGPU."""
        return classify_incident(description=description, use_hf=True)

    @_gpu_wrapper(duration=120)
    def _ask_gpu(question: str) -> str:
        """Answer question using the fine-tuned model on ZeroGPU."""
        return answer_question(question=question, use_hf=True)
else:
    def _classify_gpu(description: str) -> dict:
        """Classify incident using the fine-tuned model locally."""
        return classify_incident(description=description, use_hf=True)

    def _ask_gpu(question: str) -> str:
        """Answer question using the fine-tuned model locally."""
        return answer_question(question=question, use_hf=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#e2e8f0",
    block_label_text_color_dark="#e2e8f0",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#334155",
    input_border_color_dark="#334155",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#64748b",
    border_color_primary="#3b82f6",
    border_color_primary_dark="#3b82f6",
    button_primary_background_fill="linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    shadow_spread="0px",
)


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="VERIS Incident Classifier",
        theme=THEME,
        css=CUSTOM_CSS,
    ) as app:
        session_status = None

        # --- Hero Header ---
        gr.HTML("""
        <div class="hero-section">
            <div style="position: relative; z-index: 1;">
                <div class="hero-title">VERIS Incident Classifier</div>
                <div class="hero-subtitle">
                    Transform security incident reports into structured
                    <a href="https://verisframework.org/" target="_blank" style="color: #60a5fa; text-decoration: none;">VERIS</a>
                    classifications using a fine-tuned AI model. No API key required.
                </div>
                <div class="hero-badges">
                    <span class="hero-badge">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                        Fine-tuned Mistral-7B
                    </span>
                    <span class="hero-badge">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                        10,000+ Real Incidents
                    </span>
                    <span class="hero-badge">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                        VCDB + QLoRA
                    </span>
                </div>
            </div>
        </div>
        """)

        # --- Stats Bar ---
        gr.HTML("""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-number">4</div>
                <div class="stat-label">Dimensions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">315</div>
                <div class="stat-label">A4 Grid Combos</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">10K+</div>
                <div class="stat-label">Trained Incidents</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">300+</div>
                <div class="stat-label">Enum Values</div>
            </div>
        </div>
        """)

        # --- Model Info Banner ---
        if IS_SPACES:
            gr.HTML("""
            <div class="model-banner">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
                <span>
                    <strong>Model:</strong> Fine-tuned
                    <a href="https://huggingface.co/vibesecurityguy/veris-classifier-v2" target="_blank" style="color: #34d399;">Mistral-7B-Instruct</a>
                    on ZeroGPU &mdash; no API key needed!
                </span>
            </div>
            """)
            with gr.Row():
                gr.Markdown(
                    "**Required:** Log in with Hugging Face so ZeroGPU usage "
                    "counts against your account quota."
                )
                login_btn = gr.LoginButton("Sign in with Hugging Face")
                # Gradio 4.44 can miss auto-activation in some Spaces contexts.
                login_btn.activate()
                gr.HTML(
                    f'<a href="{SPACE_HOST_URL}/login/huggingface" target="_top" '
                    'style="display:inline-block;padding:10px 14px;border-radius:8px;'
                    'border:1px solid #334155;color:#cbd5e1;text-decoration:none;font-weight:600;">'
                    "Direct sign-in (if button refreshes)</a>"
                )
            session_status = gr.Markdown(
                value="**Session status:** Checking...",
                elem_classes=["status-card"],
            )
        else:
            gr.HTML("""
            <div class="model-banner">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <span class="fallback">
                    <strong>Local Mode:</strong> Set <code>VERIS_USE_HF=true</code> to use the
                    fine-tuned model locally (requires GPU), or provide an OpenAI API key below.
                </span>
            </div>
            """)

        # --- API Key (local mode only) ---
        if IS_SPACES:
            api_key = gr.State("")
        else:
            with gr.Group():
                api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-... (required for OpenAI fallback)",
                    type="password",
                    info="Your key is never stored.",
                )

        # --- Main Tabs ---
        with gr.Tabs():

            # ---- TAB 1: Classify ----
            with gr.TabItem("Classify Incident", id="classify"):
                gr.HTML('<div class="section-header">Incident Classification</div>')
                gr.HTML('<div class="section-desc">Describe a security incident in plain English. The classifier will map it to the VERIS taxonomy across actors, actions, assets, and attributes.</div>')

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        incident_input = gr.Textbox(
                            label="Incident Description",
                            placeholder="e.g., An attacker used stolen credentials to log into our web portal and download customer records containing names, emails, and credit card numbers...",
                            lines=8,
                            max_lines=15,
                        )
                        classify_btn = gr.Button(
                            "Classify Incident",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"],
                        )

                    with gr.Column(scale=1):
                        output_format = gr.Radio(
                            choices=["JSON", "Table"],
                            value="JSON",
                            label="Output Format",
                            info="Switch between raw JSON and a flattened table view.",
                        )
                        last_classification_raw = gr.State("")
                        classification_rows = gr.State([])
                        validation_output = gr.Markdown(
                            label="Validation",
                            value="*Validation summary will appear after classification.*",
                        )
                        classification_output = gr.Code(
                            label="VERIS Classification (JSON)",
                            language="json",
                            lines=20,
                            elem_classes=["code-output"],
                        )
                        with gr.Row(visible=False, elem_id="table-controls") as table_controls:
                            dimension_filter = gr.Dropdown(
                                choices=["All", "Actor", "Action", "Asset", "Attribute", "Error", "General"],
                                value="All",
                                label="Filter Dimension",
                            )
                            errors_only = gr.Checkbox(
                                value=False,
                                label="Errors Only",
                            )
                            export_csv_btn = gr.Button(
                                "Generate CSV",
                                size="sm",
                                interactive=False,
                                visible=False,
                            )
                        csv_file = gr.File(
                            label="Download Filtered CSV",
                            visible=False,
                            interactive=False,
                        )
                        classification_table = gr.Dataframe(
                            headers=["Dimension", "Field", "Value"],
                            datatype=["str", "str", "str"],
                            row_count=(0, "dynamic"),
                            col_count=(3, "fixed"),
                            visible=False,
                            interactive=False,
                            wrap=True,
                            max_height=500,
                            label="VERIS Classification (Table)",
                        )

                gr.HTML('<div style="margin-top: 20px;"><div class="section-header">Try an Example</div></div>')
                gr.Examples(
                    examples=EXAMPLES_CLASSIFY,
                    inputs=incident_input,
                    label="",
                    examples_per_page=6,
                )

                classify_btn.click(
                    fn=classify_and_render,
                    inputs=[incident_input, api_key, output_format, dimension_filter, errors_only],
                    outputs=[
                        last_classification_raw,
                        classification_rows,
                        validation_output,
                        classification_output,
                        classification_table,
                        table_controls,
                        export_csv_btn,
                        csv_file,
                    ],
                )
                output_format.change(
                    fn=_render_classification_output,
                    inputs=[
                        last_classification_raw,
                        output_format,
                        classification_rows,
                        dimension_filter,
                        errors_only,
                    ],
                    outputs=[classification_output, classification_table, table_controls, export_csv_btn],
                )
                dimension_filter.change(
                    fn=_apply_table_filters,
                    inputs=[classification_rows, dimension_filter, errors_only],
                    outputs=[classification_table, export_csv_btn],
                )
                errors_only.change(
                    fn=_apply_table_filters,
                    inputs=[classification_rows, dimension_filter, errors_only],
                    outputs=[classification_table, export_csv_btn],
                )
                export_csv_btn.click(
                    fn=_build_filtered_csv,
                    inputs=[classification_rows, dimension_filter, errors_only],
                    outputs=[csv_file],
                )

            # ---- TAB 2: Q&A ----
            with gr.TabItem("Ask About VERIS", id="qa"):
                gr.HTML('<div class="section-header">VERIS Knowledge Base</div>')
                gr.HTML('<div class="section-desc">Ask anything about the VERIS framework — taxonomy, enumerations, classification guidance, the A4 Grid, or how specific incident types should be categorized.</div>')

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What's the difference between hacking and misuse in VERIS?",
                            lines=4,
                            max_lines=8,
                        )
                        ask_btn = gr.Button(
                            "Ask Question",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"],
                        )

                    with gr.Column(scale=1):
                        answer_output = gr.Markdown(
                            label="Answer",
                            value="*Your answer will appear here...*",
                        )

                gr.HTML('<div style="margin-top: 20px;"><div class="section-header">Common Questions</div></div>')
                gr.Examples(
                    examples=EXAMPLES_QA,
                    inputs=question_input,
                    label="",
                    examples_per_page=6,
                )

                ask_btn.click(
                    fn=ask,
                    inputs=[question_input, api_key],
                    outputs=answer_output,
                )

            # ---- TAB 3: About ----
            with gr.TabItem("About", id="about"):
                gr.HTML("""
                <div class="about-card">
                    <h3 style="color: #f1f5f9; margin-top: 0;">What is VERIS?</h3>
                    <p style="color: #94a3b8; line-height: 1.7;">
                        <strong style="color: #e2e8f0;">VERIS</strong> (Vocabulary for Event Recording and Incident Sharing)
                        is a structured taxonomy for describing security incidents, developed by the Verizon RISK Team.
                        It powers the annual <a href="https://www.verizon.com/business/resources/reports/dbir/" target="_blank" style="color: #60a5fa;">Verizon DBIR</a>
                        and provides a common language for the security community to share and analyze incident data.
                    </p>
                </div>

                <div class="about-card">
                    <h3 style="color: #f1f5f9; margin-top: 0;">The 4 A's Framework</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 12px;">
                        <thead>
                            <tr style="border-bottom: 1px solid #334155;">
                                <th style="text-align: left; padding: 10px 16px; color: #60a5fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Dimension</th>
                                <th style="text-align: left; padding: 10px 16px; color: #60a5fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Categories</th>
                                <th style="text-align: left; padding: 10px 16px; color: #60a5fa; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Question</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 600;">Actors</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">External, Internal, Partner</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">Who caused it?</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 600;">Actions</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">Malware, Hacking, Social, Misuse, Physical, Error, Environmental</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">What did they do?</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 600;">Assets</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">Server, Network, User Device, Terminal, Media, People</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">What was affected?</td>
                            </tr>
                            <tr>
                                <td style="padding: 12px 16px; color: #e2e8f0; font-weight: 600;">Attributes</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">Confidentiality, Integrity, Availability</td>
                                <td style="padding: 12px 16px; color: #94a3b8;">How was it affected?</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="about-card">
                    <h3 style="color: #f1f5f9; margin-top: 0;">About This Project</h3>
                    <p style="color: #94a3b8; line-height: 1.7;">
                        This classifier uses a <strong style="color: #e2e8f0;">fine-tuned Mistral-7B-Instruct</strong> model,
                        trained on <strong style="color: #e2e8f0;">10,000+ real security incidents</strong> from the
                        <a href="https://github.com/vz-risk/VCDB" target="_blank" style="color: #60a5fa;">VERIS Community Database (VCDB)</a>
                        plus 300+ VERIS Q&A pairs. The model was fine-tuned using QLoRA (4-bit quantization)
                        and runs for free on Hugging Face ZeroGPU.
                    </p>
                    <div style="display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap;">
                        <a href="https://verisframework.org/" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 8px 16px; color: #60a5fa; text-decoration: none; font-size: 0.9rem;">VERIS Framework</a>
                        <a href="https://github.com/vz-risk/VCDB" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 8px 16px; color: #a78bfa; text-decoration: none; font-size: 0.9rem;">VCDB GitHub</a>
                        <a href="https://www.verizon.com/business/resources/reports/dbir/" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; padding: 8px 16px; color: #34d399; text-decoration: none; font-size: 0.9rem;">Verizon DBIR</a>
                        <a href="https://huggingface.co/vibesecurityguy/veris-classifier-v2" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 8px; padding: 8px 16px; color: #fbbf24; text-decoration: none; font-size: 0.9rem;">Model on HF</a>
                    </div>
                </div>

                <div class="about-card">
                    <h3 style="color: #f1f5f9; margin-top: 0;">Technical Details</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 12px;">
                        <tbody>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 10px 16px; color: #94a3b8; width: 40%;">Base Model</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">mistralai/Mistral-7B-Instruct-v0.3</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 10px 16px; color: #94a3b8;">Fine-tuning Method</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">QLoRA (4-bit, r=16, alpha=32)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 10px 16px; color: #94a3b8;">Training Data</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">10,019 classification + 311 Q&A pairs</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 10px 16px; color: #94a3b8;">Training Epochs</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">3</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 10px 16px; color: #94a3b8;">Effective Batch Size</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">8 (2 x 4 gradient accumulation)</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px 16px; color: #94a3b8;">Inference</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">HF ZeroGPU (free A10G burst)</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)

        # --- Footer ---
        gr.HTML("""
        <div class="footer">
            Fine-tuned Mistral-7B-Instruct &middot; VERIS Framework &middot; VCDB &middot; QLoRA
            <br>
            <a href="https://github.com/pshamoon/veris-classifier">Source Code</a> &middot;
            <a href="https://huggingface.co/vibesecurityguy/veris-classifier-v2">Model</a> &middot;
            <a href="https://verisframework.org/">VERIS Docs</a> &middot;
            <a href="https://github.com/vz-risk/VCDB">VCDB</a>
        </div>
        """)

        if IS_SPACES and session_status is not None:
            app.load(
                fn=_session_status_markdown,
                outputs=[session_status],
                queue=False,
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
