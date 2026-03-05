"""VERIS Classifier - Gradio Web Application.

Dual-mode inference:
- Primary: Fine-tuned Qwen2.5-7B-Instruct on ZeroGPU (no API key needed)
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

import gradio as gr
from dotenv import load_dotenv

from src.veris_classifier.classifier import (
    HF_MODEL_ID,
    answer_question,
    classify_incident,
    load_hf_model,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZeroGPU support — only available on HF Spaces
# ---------------------------------------------------------------------------
IS_SPACES = os.getenv("SPACE_ID") is not None

if IS_SPACES:
    import spaces

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

def _use_hf_model() -> bool:
    """Check if we should use the fine-tuned HF model."""
    # On HF Spaces, always try the local model first
    if IS_SPACES:
        return True
    # Locally, use HF model if VERIS_USE_HF is set
    return os.getenv("VERIS_USE_HF", "").lower() in ("1", "true", "yes")


def classify(description: str, api_key: str) -> str:
    """Classify an incident — uses HF model on Spaces, OpenAI otherwise."""
    if not description.strip():
        return "Please enter an incident description."

    use_hf = _use_hf_model()

    # If user provided an API key, prefer OpenAI (explicit choice)
    if api_key.strip():
        use_hf = False

    if use_hf:
        try:
            result = _classify_gpu(description)
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"HF model error: {e}")
            # Fall through to OpenAI if available
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


def ask(question: str, api_key: str) -> str:
    """Answer a VERIS question — uses HF model on Spaces, OpenAI otherwise."""
    if not question.strip():
        return "*Please enter a question.*"

    use_hf = _use_hf_model()

    if api_key.strip():
        use_hf = False

    if use_hf:
        try:
            return _ask_gpu(question)
        except Exception as e:
            logger.error(f"HF model error: {e}")
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


# ---------------------------------------------------------------------------
# GPU-decorated functions for ZeroGPU
# ---------------------------------------------------------------------------

if IS_SPACES:
    @spaces.GPU(duration=120)
    def _classify_gpu(description: str) -> dict:
        """Classify incident using the fine-tuned model on ZeroGPU."""
        return classify_incident(description=description, use_hf=True)

    @spaces.GPU(duration=120)
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
                        Fine-tuned Qwen2.5-7B
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
                    <a href="https://huggingface.co/vibesecurityguy/veris-classifier-v1" target="_blank" style="color: #34d399;">Qwen2.5-7B-Instruct</a>
                    on ZeroGPU &mdash; no API key needed!
                </span>
            </div>
            """)
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

        # --- API Key (optional on Spaces) ---
        with gr.Group():
            api_key = gr.Textbox(
                label="OpenAI API Key (Optional)" if IS_SPACES else "OpenAI API Key",
                placeholder="sk-... (optional — the fine-tuned model runs for free)" if IS_SPACES else "sk-... (required for classification)",
                type="password",
                info="Your key is never stored. If provided, GPT-4o will be used instead of the fine-tuned model.",
            )

        # --- Main Tabs ---
        with gr.Tabs() as tabs:

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
                        classification_output = gr.Code(
                            label="VERIS Classification (JSON)",
                            language="json",
                            lines=20,
                            elem_classes=["code-output"],
                        )

                gr.HTML('<div style="margin-top: 20px;"><div class="section-header">Try an Example</div></div>')
                gr.Examples(
                    examples=EXAMPLES_CLASSIFY,
                    inputs=incident_input,
                    label="",
                    examples_per_page=6,
                )

                classify_btn.click(
                    fn=classify,
                    inputs=[incident_input, api_key],
                    outputs=classification_output,
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
                        This classifier uses a <strong style="color: #e2e8f0;">fine-tuned Qwen2.5-7B-Instruct</strong> model,
                        trained on <strong style="color: #e2e8f0;">10,000+ real security incidents</strong> from the
                        <a href="https://github.com/vz-risk/VCDB" target="_blank" style="color: #60a5fa;">VERIS Community Database (VCDB)</a>
                        plus 300+ VERIS Q&A pairs. The model was fine-tuned using QLoRA (4-bit quantization)
                        and runs for free on Hugging Face ZeroGPU.
                    </p>
                    <div style="display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap;">
                        <a href="https://verisframework.org/" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 8px 16px; color: #60a5fa; text-decoration: none; font-size: 0.9rem;">VERIS Framework</a>
                        <a href="https://github.com/vz-risk/VCDB" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 8px; padding: 8px 16px; color: #a78bfa; text-decoration: none; font-size: 0.9rem;">VCDB GitHub</a>
                        <a href="https://www.verizon.com/business/resources/reports/dbir/" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; padding: 8px 16px; color: #34d399; text-decoration: none; font-size: 0.9rem;">Verizon DBIR</a>
                        <a href="https://huggingface.co/vibesecurityguy/veris-classifier-v1" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; background: rgba(251, 191, 36, 0.1); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 8px; padding: 8px 16px; color: #fbbf24; text-decoration: none; font-size: 0.9rem;">Model on HF</a>
                    </div>
                </div>

                <div class="about-card">
                    <h3 style="color: #f1f5f9; margin-top: 0;">Technical Details</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 12px;">
                        <tbody>
                            <tr style="border-bottom: 1px solid #1e293b;">
                                <td style="padding: 10px 16px; color: #94a3b8; width: 40%;">Base Model</td>
                                <td style="padding: 10px 16px; color: #e2e8f0;">Qwen/Qwen2.5-7B-Instruct</td>
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
            Fine-tuned Qwen2.5-7B-Instruct &middot; VERIS Framework &middot; VCDB &middot; QLoRA
            <br>
            <a href="https://github.com/pshamoon/veris-classifier">Source Code</a> &middot;
            <a href="https://huggingface.co/vibesecurityguy/veris-classifier-v1">Model</a> &middot;
            <a href="https://verisframework.org/">VERIS Docs</a> &middot;
            <a href="https://github.com/vz-risk/VCDB">VCDB</a>
        </div>
        """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
