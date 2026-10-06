"""
InnovateX Solution Agent  -  OpenRouter (free models) AI agent + Streamlit marketing website.

Setup (macOS / Linux):
    export OPENROUTER_API_KEY="your-new-key"
Run:
    python -W ignore -m streamlit run app.py
"""
import csv
import inspect
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

# ======================================================================
# 1) SETTINGS & COMPANY DATA
# ======================================================================
# Free OpenRouter models, tried in order. When one is busy or removed, the next one is used.
# Free model names change often: check https://openrouter.ai/models?q=free
# or override with:  export OPENROUTER_MODELS="model-a:free,model-b:free"
MODELS = [m.strip() for m in os.environ.get("OPENROUTER_MODELS", "").split(",") if m.strip()] or [
    "deepseek/deepseek-v4-flash:free",
    "openai/gpt-oss-120b:free",
    "moonshotai/kimi-k2.6:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/free",  # router: picks any available free model that supports tool calling
]

MODEL_ATTEMPTS = 1       # tries per model (we move to the next model instead of waiting)
REQUEST_TIMEOUT = 25     # seconds before a try is abandoned
MAX_HISTORY = 10         # messages sent to the model on each turn
MAX_TOKENS = 1000        # cap on reply length (keeps answers fast)
MAX_TOOL_ROUNDS = 5

BASE_DIR = Path(__file__).parent
HISTORY_DIR = BASE_DIR / "history"
PROPOSALS_DIR = BASE_DIR / "proposals"
LEADS_FILE = BASE_DIR / "leads.csv"
BOOKINGS_FILE = BASE_DIR / "bookings.csv"

EMAIL = "innovatexs7@gmail.com"
PHONES = ["01018401874", "01270121354"]
CONTACT_TEXT = f"📧 Email: {EMAIL}\n📱 WhatsApp: {PHONES[0]} | {PHONES[1]}"
CONTACT_MD = CONTACT_TEXT.replace("\n", "  \n")  # Markdown needs two spaces to keep the line break


def wa_link(phone: str) -> str:
    """Egyptian mobile 010... -> https://wa.me/2010..."""
    return "https://wa.me/20" + phone.lstrip("0")


WHATSAPP_LINK = wa_link(PHONES[0])

CONTACT_KEYWORDS = (
    "contact", "phone", "whatsapp", "e-mail", "email", "reach you", "call you",
    "تواصل", "واتس", "ايميل", "إيميل", "بريد", "تليفون", "هاتف", "موبايل", "رقمكم", "أرقامكم", "ارقامكم",
)

COMPANY = {
    "name": "InnovateX Solution",
    "about": (
        "InnovateX Solution is an IT and software company delivering custom "
        "software, AI solutions, data analytics and digital products."
    ),
}

# base_price in USD, used only for rough estimates (edit to your real prices)
SERVICES = {
    "AI & Machine Learning": {
        "ar": "الذكاء الاصطناعي وتعلم الآلة",
        "icon": "🤖",
        "description": "Custom ML models, LLM apps, RAG chatbots and agentic AI systems.",
        "description_ar": "نماذج تعلم آلة مخصصة وتطبيقات LLM وشات بوت RAG وأنظمة وكلاء ذكاء اصطناعي.",
        "base_price": 1500,
    },
    "Data Science & Analytics": {
        "ar": "علوم البيانات والتحليلات",
        "icon": "📊",
        "description": "Data pipelines, dashboards, EDA, forecasting and business insights.",
        "description_ar": "خطوط بيانات ولوحات تحكم وتحليل استكشافي وتنبؤات ورؤى للأعمال.",
        "base_price": 1000,
    },
    "Web & Software Development": {
        "ar": "تطوير المواقع والبرمجيات",
        "icon": "💻",
        "description": "Websites, web apps, dashboards and APIs.",
        "description_ar": "مواقع وتطبيقات ويب ولوحات تحكم وواجهات برمجية.",
        "base_price": 1200,
    },
    "Cybersecurity Assessment": {
        "ar": "تقييم الأمن السيبراني",
        "icon": "🛡️",
        "description": "Authorized vulnerability scanning and automated security reports.",
        "description_ar": "فحص ثغرات مصرّح به وتقارير أمنية آلية.",
        "base_price": 900,
    },
    "Training & Bootcamps": {
        "ar": "التدريب والمعسكرات",
        "icon": "🎓",
        "description": "Hands-on data science and AI training for individuals and teams.",
        "description_ar": "تدريب عملي على علوم البيانات والذكاء الاصطناعي للأفراد والفرق.",
        "base_price": 400,
    },
}

PHASE_NAMES = ["Discovery & planning", "Design & architecture", "Development", "Testing & deployment"]
PHASE_SHARE = [0.15, 0.20, 0.45, 0.20]


def build_phases(weeks: int) -> list:
    return [{"phase": n, "approx_weeks": max(1, round(weeks * p))} for n, p in zip(PHASE_NAMES, PHASE_SHARE)]


def is_contact_request(text: str) -> bool:
    """True when the visitor asks for our contact details (answered directly, without the model)."""
    text = (text or "").lower()
    # A message that contains an email address or a long number is the visitor giving THEIR details
    # (booking / lead flow), so it must go to the agent.
    if "@" in text or re.search(r"\d{8,}", text):
        return False
    return any(k in text for k in CONTACT_KEYWORDS)


# ======================================================================
# 2) TOOLS (used by the AI agent AND by the step-by-step guide)
# ======================================================================
def _find_service(name: str):
    name = (name or "").strip().lower()
    for key, v in SERVICES.items():
        if name in (key.lower(), v["ar"].lower()):
            return key
    return None


def get_company_info(topic: str = "about") -> dict:
    """Get information about InnovateX Solution.

    Args:
        topic: One of "about" or "contact".
    """
    if (topic or "").lower() == "contact":
        return {"contact_text": CONTACT_TEXT}
    return {"name": COMPANY["name"], "about": COMPANY["about"]}


def list_services() -> dict:
    """List all services offered by InnovateX Solution (English and Arabic)."""
    return {
        k: {"description": v["description"], "arabic_name": v["ar"], "arabic_description": v["description_ar"]}
        for k, v in SERVICES.items()
    }


def estimate_project_cost(service: str, complexity: str = "medium", timeline_weeks: int = 4) -> dict:
    """Give a rough, non-binding price estimate in USD for a project.

    Args:
        service: Service name exactly as returned by list_services (English name).
        complexity: "low", "medium", or "high".
        timeline_weeks: Desired delivery time in weeks.
    """
    match = _find_service(service)
    if not match:
        return {"error": f"Unknown service. Available: {list(SERVICES)}"}
    mult = {"low": 0.7, "medium": 1.0, "high": 1.8}.get((complexity or "").lower(), 1.0)
    try:
        weeks = max(1, int(timeline_weeks))
    except (TypeError, ValueError):
        weeks = 4
    rush = 1.25 if weeks < 3 else 1.0
    est = SERVICES[match]["base_price"] * mult * rush * max(1, weeks / 4)
    return {
        "service": match,
        "complexity": complexity,
        "timeline_weeks": weeks,
        "estimated_range_usd": f"{int(est * 0.85)} - {int(est * 1.15)}",
        "note": "Rough estimate only. Final quote requires a short consultation.",
    }


def create_project_proposal(
    client_name: str, service: str, requirements: str, complexity: str = "medium", timeline_weeks: int = 4
) -> dict:
    """Create a structured project proposal (scope, phases, timeline, estimate) and save it.
    Use after the visitor described what they need.

    Args:
        client_name: Name of the client or company.
        service: Service name exactly as returned by list_services (English name).
        requirements: Short summary of the client's requirements.
        complexity: "low", "medium", or "high".
        timeline_weeks: Desired delivery time in weeks.
    """
    est = estimate_project_cost(service, complexity, timeline_weeks)
    if "error" in est:
        return est
    w = est["timeline_weeks"]
    proposal = {
        "client": client_name,
        "service": est["service"],
        "requirements": requirements,
        "phases": build_phases(w),
        "timeline_weeks": w,
        "estimated_range_usd": est["estimated_range_usd"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": est["note"],
    }
    PROPOSALS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", client_name or "client").strip("-")[:30] or "client"
    path = PROPOSALS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.json"
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "created", "proposal": proposal}


def book_consultation(name: str, contact: str, preferred_time: str, topic: str) -> dict:
    """Request a consultation call with the InnovateX team. Only call after the visitor gave
    their name, an email or WhatsApp number, and a preferred time.

    Args:
        name: Visitor name.
        contact: Visitor email or WhatsApp/phone number.
        preferred_time: Preferred day/time as the visitor wrote it.
        topic: What they want to discuss.
    """
    if not name or not contact:
        return {"status": "error", "message": "Name and contact are required."}
    _append_csv(BOOKINGS_FILE, ["timestamp", "name", "contact", "preferred_time", "topic"],
                [datetime.now().isoformat(timespec="seconds"), name, contact, preferred_time, topic])
    return {"status": "requested", "message": "Consultation request received. The team will confirm the time."}


def save_lead(name: str, email: str, project_description: str) -> dict:
    """Save a potential client's contact details so the team can follow up.
    Only call this after the visitor clearly shared their name and email.

    Args:
        name: Client name.
        email: Client email address.
        project_description: Short summary of what they need.
    """
    if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""):
        return {"status": "error", "message": "Invalid email address."}
    _append_csv(LEADS_FILE, ["timestamp", "name", "email", "project"],
                [datetime.now().isoformat(timespec="seconds"), name, email, project_description])
    return {"status": "saved", "message": "The InnovateX team will contact you soon."}


def _append_csv(path: Path, header: list, row: list):
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)


TOOLS = [get_company_info, list_services, estimate_project_cost,
         create_project_proposal, book_consultation, save_lead]

SYSTEM_PROMPT = f"""You are the InnovateX Solution Agent, the official AI assistant of {COMPANY['name']}.

Languages: You speak English and Arabic. ALWAYS reply in the language of the visitor's latest message
(Arabic -> Arabic, including Egyptian dialect; English -> English). Keep tool arguments in English.

Step-by-step help (most important): when a visitor asks about a service or a project, guide them ONE step
at a time. Ask only ONE short question per message and wait for the answer:
  1) Briefly explain the service they asked about (use list_services; 2-3 lines).
  2) Ask what they want to achieve.
  3) Ask how big it is: simple / standard / advanced.
  4) Ask the delivery time they want (in weeks).
  5) Give the rough estimate (estimate_project_cost) and offer a full proposal (create_project_proposal).
  6) Offer to book a consultation (book_consultation) or take their name + email/WhatsApp (save_lead).
End every message with a clear next step. Never ask for all details at once.

Other rules (use the tools, never invent facts):
- Explain the company with get_company_info.
- Contact details: when the visitor asks for contact, phone, number, WhatsApp or email, reply with EXACTLY
  this block (you may add one short friendly sentence before it):
{CONTACT_TEXT}
  Never invent or guess emails, phone numbers or links. The only valid contact details are the block above.
- Style: professional, warm, concise, easy to scan. Estimates are non-binding; never promise exact prices or
  deadlines. If asked something unrelated, answer briefly and steer back to how InnovateX can help.
"""


# ======================================================================
# 3) AGENT + HISTORY STORAGE
# ======================================================================
def get_api_key():
    """Read the key from the OPENROUTER_API_KEY env var or Streamlit secrets (never hardcode it)."""
    key = os.environ.get("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")
    if key:
        return key
    try:
        return st.secrets["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"]
    except Exception:
        return None


@st.cache_resource
def get_client(api_key: str):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=REQUEST_TIMEOUT,
        max_retries=0,  # no hidden SDK retries: retries are handled explicitly below
        default_headers={"HTTP-Referer": "https://innovatex.example", "X-Title": "InnovateX Solution Agent"},
    )


def _tool_schema(fn):
    """Build an OpenAI-style tool schema from a function's signature and docstring."""
    doc = inspect.getdoc(fn) or ""
    desc = doc.split("\n\nArgs:")[0].replace("\n", " ").strip()
    arg_docs = {}
    if "Args:" in doc:
        for line in doc.split("Args:")[1].splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                arg_docs[k.strip()] = v.strip()
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {"type": "integer" if p.annotation is int else "string",
                       "description": arg_docs.get(name, "")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOL_FUNCS = {f.__name__: f for f in TOOLS}
TOOL_SCHEMAS = [_tool_schema(f) for f in TOOLS]


def _is_busy(err) -> bool:
    msg = str(err).lower()
    return any(x in msg for x in ("429", "502", "503", "504", "rate", "unavailable", "timed out", "timeout"))


class ChatSession:
    """Streaming chat with tool calling over OpenRouter, with fallback across free models."""

    def __init__(self, api_key: str, history: list):
        self.client = get_client(api_key)
        self.history = [{"role": m["role"], "content": m["content"]} for m in history][-MAX_HISTORY:]

    def _run_stream(self, model: str, msgs: list, state: dict):
        """Yield text chunks. Handles tool calls between streamed rounds."""
        for _ in range(MAX_TOOL_ROUNDS):
            stream = self.client.chat.completions.create(
                model=model, messages=msgs, tools=TOOL_SCHEMAS, temperature=0.5,
                stream=True, max_tokens=MAX_TOKENS,
                extra_body={"reasoning": {"effort": "low"}},
            )
            text, calls = "", {}
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text += delta.content
                    yield delta.content
                for tc in delta.tool_calls or []:
                    c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        c["id"] = tc.id
                    if tc.function and tc.function.name:
                        c["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        c["args"] += tc.function.arguments

            calls = {i: c for i, c in calls.items() if c["name"]}
            if not calls:
                if not text.strip() and not state["text_shown"]:
                    raise ValueError("empty response")
                return
            state["text_shown"] = state["text_shown"] or bool(text.strip())

            msgs.append({"role": "assistant", "content": text, "tool_calls": [
                {"id": c["id"] or f"call_{i}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for i, c in calls.items()]})
            for i, c in calls.items():
                state["tools_ran"] = True
                try:
                    result = TOOL_FUNCS[c["name"]](**json.loads(c["args"] or "{}"))
                except Exception as e:
                    result = {"error": str(e)}
                msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}",
                             "content": json.dumps(result, ensure_ascii=False)})
        raise RuntimeError("too many tool rounds")

    def _remember(self, prompt: str, reply: str):
        self.history = (self.history + [{"role": "user", "content": prompt},
                                        {"role": "assistant", "content": reply}])[-MAX_HISTORY:]

    def stream_message(self, prompt: str):
        """Generator of text chunks. Moves to the next free model when one is busy. Never retries after a
        tool (e.g. a booking) already ran, so actions are never duplicated."""
        base = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history + \
               [{"role": "user", "content": prompt}]
        last_err = None
        for model in MODELS:
            for _attempt in range(MODEL_ATTEMPTS):
                state = {"text_shown": False, "tools_ran": False}
                full = ""
                try:
                    for piece in self._run_stream(model, list(base), state):
                        full += piece
                        yield piece
                    self._remember(prompt, full)
                    return
                except Exception as e:
                    last_err = e
                    if full:  # partial answer already shown: keep it, don't restart
                        self._remember(prompt, full)
                        return
                    if state["tools_ran"] or "401" in str(e):
                        raise  # an action already happened, or the key is bad: do not retry
                    time.sleep(1)
        raise last_err


def create_chat(api_key: str, messages: list):
    return ChatSession(api_key, messages)


def history_path(sid: str) -> Path:
    return HISTORY_DIR / f"{sid}.json"


def load_history(sid: str) -> list:
    try:
        return json.loads(history_path(sid).read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(sid: str, messages: list):
    HISTORY_DIR.mkdir(exist_ok=True)
    history_path(sid).write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")


def get_sid() -> str:
    sid = re.sub(r"[^a-zA-Z0-9-]", "", str(st.query_params.get("sid", "")))[:36]
    if not sid:
        sid = uuid.uuid4().hex
        st.query_params["sid"] = sid
    return sid


# ======================================================================
# 4) UI TEXT (English / Arabic)
# ======================================================================
T = {
    "en": {
        "badge": "AI • Data • Software • Security",
        "title": "Build smarter with AI &amp; software that works",
        "sub": "InnovateX Solution designs custom software, AI agents, data analytics and security solutions. "
               "Chat with our AI agent or follow the step-by-step guide to get an instant estimate and book a consultation.",
        "cta_wa": "💬 WhatsApp us", "cta_mail": "✉️ Email us",
        "pills": ["🤖 AI agent 24/7", "🌍 English &amp; العربية", "⚡ Instant estimates", "📅 Easy consultation booking"],
        "contact_title": "Talk to us", "email_lbl": "Email", "wa_lbl": "WhatsApp",
        "services": "Our services",
        "placeholder": "Type your message...",
        "greeting": "Hi! 👋 I'm the InnovateX Solution Agent. Tell me which service you're interested in and "
                    "I'll guide you step by step — or use the “Step-by-step guide” tab.",
        "contact_intro": "We'd love to hear from you! 😊",
        "clear": "🗑️ Clear chat", "download": "⬇️ Download chat",
        "saved": "💾 Your conversation is saved. Bookmark this page to continue later.",
        "theme": "🎨 Theme",
        "key_info": "The AI chat needs an OpenRouter API key (sidebar or OPENROUTER_API_KEY). "
                    "The step-by-step guide works without it.",
        "busy": "The AI service is very busy right now. Please try again in a minute, or use the "
                "“Step-by-step guide” tab to get your estimate right away.",
        "err": "Sorry, something went wrong", "thinking": "Thinking...",
        "footer": "© InnovateX Solution — Smart IT & software solutions",
        "quick": [("🧩 Our services", "What services do you offer?"),
                  ("💰 Get a quote", "I want a price estimate for a project"),
                  ("📅 Book a call", "I'd like to book a consultation"),
                  ("📞 Contact us", "What is your contact number?")],
        "guide": {
            "tab_chat": "💬 Chat with the agent", "tab_guide": "🧭 Step-by-step guide",
            "pick": "Pick the service you're interested in to start",
            "s1": "Step 1 of 4 — Your service", "s2": "Step 2 of 4 — How complex is your project?",
            "s3": "Step 3 of 4 — Desired delivery time", "s4": "Step 4 of 4 — Your estimate & proposal",
            "complexity": {"low": "Simple — small scope / MVP",
                           "medium": "Standard — typical business project",
                           "high": "Advanced — large scope / many integrations"},
            "weeks": "Weeks to deliver", "next": "Next ➜", "back": "⬅ Back",
            "est_range": "Estimated range", "phases": "Project phases", "unit": "weeks",
            "note": "Rough, non-binding estimate. Final quote after a short consultation.",
            "form_title": "Get your proposal and a callback",
            "name": "Your name", "contact": "Email or WhatsApp number",
            "when": "Preferred time to talk (optional)", "details": "Tell us about your project (optional)",
            "submit": "📨 Send my request",
            "done": "✅ Request sent! The InnovateX team will contact you soon.",
            "restart": "Start a new request", "need": "Please enter your name and contact.",
            "phase_names": ["Discovery & planning", "Design & architecture", "Development", "Testing & deployment"],
        },
    },
    "ar": {
        "badge": "ذكاء اصطناعي • بيانات • برمجيات • أمن سيبراني",
        "title": "ابنِ مستقبل أعمالك بحلول الذكاء الاصطناعي والبرمجة",
        "sub": "إنوفيتكس سوليوشن تصمّم برمجيات مخصصة ووكلاء ذكاء اصطناعي وتحليلات بيانات وحلولًا أمنية. "
               "تحدّث مع وكيلنا الذكي أو اتبع الدليل خطوة بخطوة للحصول على تقدير فوري وحجز استشارة.",
        "cta_wa": "💬 راسلنا على واتساب", "cta_mail": "✉️ راسلنا بالبريد",
        "pills": ["🤖 وكيل ذكي 24/7", "🌍 العربية &amp; English", "⚡ تقدير فوري للأسعار", "📅 حجز استشارة بسهولة"],
        "contact_title": "تواصل معنا", "email_lbl": "البريد الإلكتروني", "wa_lbl": "واتساب",
        "services": "خدماتنا",
        "placeholder": "اكتب رسالتك...",
        "greeting": "أهلًا! 👋 أنا وكيل إنوفيتكس سوليوشن. قل لي أي خدمة تهمّك وسأرشدك خطوة بخطوة، "
                    "أو استخدم تبويب «دليل خطوة بخطوة».",
        "contact_intro": "يسعدنا تواصلك معنا! 😊",
        "clear": "🗑️ مسح المحادثة", "download": "⬇️ تحميل المحادثة",
        "saved": "💾 محادثتك محفوظة. احفظ هذه الصفحة في المفضلة لتكمل لاحقًا.",
        "theme": "🎨 المظهر",
        "key_info": "محادثة الذكاء الاصطناعي تحتاج مفتاح OpenRouter API (الشريط الجانبي أو OPENROUTER_API_KEY). "
                    "أما الدليل خطوة بخطوة فيعمل بدونه.",
        "busy": "الخدمة مشغولة حاليًا. حاول بعد دقيقة، أو استخدم تبويب «دليل خطوة بخطوة» "
                "لتحصل على تقديرك فورًا.",
        "err": "عذرًا، حدث خطأ", "thinking": "جارٍ التفكير...",
        "footer": "© إنوفيتكس سوليوشن — حلول تقنية وبرمجية ذكية",
        "quick": [("🧩 خدماتنا", "ما هي الخدمات التي تقدمونها؟"),
                  ("💰 احصل على عرض سعر", "أريد تقدير سعر لمشروع"),
                  ("📅 احجز استشارة", "أريد حجز استشارة"),
                  ("📞 تواصل معنا", "ما هي أرقام التواصل؟")],
        "guide": {
            "tab_chat": "💬 تحدث مع الوكيل", "tab_guide": "🧭 دليل خطوة بخطوة",
            "pick": "اختر الخدمة التي تهمّك للبدء",
            "s1": "الخطوة 1 من 4 — الخدمة", "s2": "الخطوة 2 من 4 — ما مدى تعقيد مشروعك؟",
            "s3": "الخطوة 3 من 4 — مدة التسليم المطلوبة", "s4": "الخطوة 4 من 4 — التقدير والعرض",
            "complexity": {"low": "بسيط — نطاق صغير / نسخة أولية",
                           "medium": "متوسط — مشروع عمل معتاد",
                           "high": "متقدم — نطاق كبير / تكاملات كثيرة"},
            "weeks": "عدد الأسابيع للتسليم", "next": "التالي ➜", "back": "رجوع ⬅",
            "est_range": "التقدير التقريبي", "phases": "مراحل المشروع", "unit": "أسابيع",
            "note": "تقدير تقريبي وغير ملزم. العرض النهائي بعد استشارة قصيرة.",
            "form_title": "احصل على عرضك واطلب مكالمة",
            "name": "اسمك", "contact": "البريد الإلكتروني أو رقم واتساب",
            "when": "الوقت المناسب للتواصل (اختياري)", "details": "حدّثنا عن مشروعك (اختياري)",
            "submit": "📨 أرسل طلبي",
            "done": "✅ تم إرسال طلبك! سيتواصل معك فريق إنوفيتكس قريبًا.",
            "restart": "ابدأ طلبًا جديدًا", "need": "من فضلك اكتب اسمك ووسيلة التواصل.",
            "phase_names": ["الاكتشاف والتخطيط", "التصميم والبنية", "التطوير", "الاختبار والإطلاق"],
        },
    },
}

# Theme palettes: (dark, mid, light, accent)
THEMES = {
    "Ocean 🌊": ("#0b1437", "#1d2f7a", "#0aa5d6", "#00c2ff"),
    "Violet 💜": ("#1a0b37", "#5b21b6", "#c026d3", "#e879f9"),
    "Emerald 🌿": ("#052e2b", "#0f766e", "#16a34a", "#34d399"),
    "Sunset 🌅": ("#3b0a1e", "#be123c", "#f59e0b", "#fb923c"),
}

CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatInput textarea {font-family:'Inter','Cairo',sans-serif;}
#MainMenu, footer {visibility:hidden;}
.stApp {background: radial-gradient(1100px 520px at 8% -10%, @ACC@22, transparent),
                    radial-gradient(900px 520px at 100% 0%, @G3@1f, transparent);}
.block-container {max-width: 920px; padding-top: 1.2rem;}

@keyframes gradientShift {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes float {0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(20px) scale(1.06)}}
@keyframes fadeUp {from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:translateY(0)}}
@keyframes pulse {0%{box-shadow:0 0 0 0 #25D36688} 70%{box-shadow:0 0 0 14px #25D36600} 100%{box-shadow:0 0 0 0 #25D36600}}
@keyframes shine {to{background-position:200% center}}
@keyframes bob {0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)}}

.hero {position:relative; overflow:hidden; padding:2.3rem 2rem; border-radius:26px; color:#fff;
  background: linear-gradient(120deg,@G1@,@G2@,@G3@,@G2@,@G1@); background-size:300% 300%;
  animation: gradientShift 16s ease infinite; box-shadow: 0 22px 55px @G1@66; margin-bottom:1.1rem;}
.hero::before {content:""; position:absolute; width:360px; height:360px; right:-100px; top:-130px;
  background: radial-gradient(circle,#ffffff3a,transparent 70%); animation: float 8s ease-in-out infinite;}
.hero::after {content:""; position:absolute; width:280px; height:280px; left:-90px; bottom:-130px;
  background: radial-gradient(circle,@ACC@55,transparent 70%); animation: float 10s ease-in-out infinite reverse;}
.badge {display:inline-block; padding:.3rem .85rem; border-radius:999px; font-size:.78rem; letter-spacing:.04em;
  background:#ffffff22; border:1px solid #ffffff44; backdrop-filter: blur(6px); position:relative; z-index:1;}
.hero h1 {font-size:2.25rem; line-height:1.2; margin:.8rem 0 .5rem; font-weight:800; position:relative; z-index:1;}
.hero h1 span {background: linear-gradient(90deg,#fff,@ACC@,#fff,@ACC@); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: shine 6s linear infinite;}
.hero p {opacity:.93; font-size:1.02rem; max-width:640px; position:relative; z-index:1;}
.pills {display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; position:relative; z-index:1;}
.pill {padding:.35rem .8rem; border-radius:999px; background:#ffffff1f; border:1px solid #ffffff33; font-size:.85rem;
  animation: fadeUp .7s ease both;}
.pill:nth-child(2){animation-delay:.1s} .pill:nth-child(3){animation-delay:.2s} .pill:nth-child(4){animation-delay:.3s}
.cta {display:flex; flex-wrap:wrap; gap:.7rem; margin-top:1.2rem; position:relative; z-index:1;}
.cta a {text-decoration:none; padding:.68rem 1.25rem; border-radius:13px; font-weight:700; font-size:.95rem;
  transition: transform .15s ease, box-shadow .15s ease;}
.cta a:hover {transform: translateY(-3px); box-shadow:0 10px 24px rgba(0,0,0,.28);}
.cta .primary {background:#25D366; color:#062b13; animation: pulse 2.4s infinite;}
.cta .ghost {background:#ffffff1f; color:#fff; border:1px solid #ffffff66;}

.sec-title {font-weight:800; font-size:1.15rem; margin:.5rem 0 .7rem;}
.contact {display:grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap:.8rem; margin:0 0 1.2rem;}
.c-card {display:flex; align-items:center; gap:.8rem; padding:.9rem 1.1rem; border-radius:16px; text-decoration:none;
  color:inherit; border:1px solid @ACC@55; background: linear-gradient(160deg,@ACC@14,transparent);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease; animation: fadeUp .6s ease both;}
.c-card:nth-child(2){animation-delay:.1s} .c-card:nth-child(3){animation-delay:.2s}
.c-card:hover {transform: translateY(-4px); box-shadow:0 14px 30px @ACC@40; border-color:@ACC@;}
.c-card .ic {font-size:1.7rem; animation: bob 3s ease-in-out infinite;}
.c-card small {display:block; opacity:.65; font-size:.74rem; text-transform:uppercase; letter-spacing:.06em;}
.c-card b {font-size:1rem; word-break:break-all;}

.grid {display:grid; grid-template-columns: repeat(auto-fit,minmax(250px,1fr)); gap:.8rem; margin-bottom:1.3rem;}
.card {position:relative; overflow:hidden; padding:1rem 1.1rem; border-radius:16px; border:1px solid #8a9bd633;
  background: linear-gradient(160deg,#ffffff10,#ffffff04); opacity:0; animation: fadeUp .6s ease forwards;
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;}
.card:nth-child(1){animation-delay:.05s} .card:nth-child(2){animation-delay:.15s} .card:nth-child(3){animation-delay:.25s}
.card:nth-child(4){animation-delay:.35s} .card:nth-child(5){animation-delay:.45s}
.card::before {content:""; position:absolute; left:0; top:0; height:3px; width:100%;
  background: linear-gradient(90deg,@ACC@,@G3@); transform:scaleX(0); transform-origin:left; transition: transform .35s ease;}
.card:hover {transform: translateY(-5px); border-color:@ACC@88; box-shadow:0 14px 32px @ACC@33;}
.card:hover::before {transform:scaleX(1);}
.card .ic {font-size:1.7rem; display:inline-block; transition: transform .25s ease;}
.card:hover .ic {transform: scale(1.2) rotate(-6deg);}
.card h4 {margin:.3rem 0 .2rem; font-size:1rem;}
.card p {margin:0; font-size:.86rem; opacity:.8;}
.chosen {padding:.8rem 1rem; border-radius:14px; border:1px solid @ACC@66; background:@ACC@14; margin-bottom:.8rem;
  animation: fadeUp .4s ease both;}

[data-testid="stChatMessage"] {border-radius:16px; padding:.8rem 1rem; border:1px solid #8a9bd622;
  background:#ffffff08; margin-bottom:.5rem; animation: fadeUp .4s ease both;}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius:12px; font-weight:600; width:100%; border:1px solid @ACC@77; transition: all .18s ease;}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  border-color:@ACC@; transform: translateY(-2px); box-shadow:0 8px 18px @ACC@33;}
[data-testid="stChatInput"] {border-radius:16px;}
.stTabs [data-baseweb="tab"] {font-weight:700;}
.stTabs [data-baseweb="tab-highlight"] {background:@ACC@;}
.stProgress > div > div > div > div {background: linear-gradient(90deg,@ACC@,@G3@);}
.footer {text-align:center; opacity:.7; font-size:.82rem; margin:1.2rem 0 .4rem;}

[data-testid="stSidebar"] {background: linear-gradient(180deg,@G1@,@G2@);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] label *, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color:#eaf0ff;}
[data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] [data-testid^="stBaseLinkButton"] {background:#ffffff1a; color:#fff;
  border:1px solid #ffffff55;}
[data-testid="stSidebar"] .stButton > button *, [data-testid="stSidebar"] .stDownloadButton > button *,
[data-testid="stSidebar"] [data-testid^="stBaseLinkButton"] * {color:#fff;}
[data-testid="stSidebar"] input {color:#0b1437 !important;}

@media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important;}
  .card {opacity:1;}}
@media (max-width: 640px) {.hero {padding:1.5rem 1.2rem;} .hero h1 {font-size:1.6rem;}}
</style>
"""
RTL_CSS = """
<style>
.hero, .grid, .contact, .sec-title, .chosen, .footer, [data-testid="stChatMessage"], .pills, .cta,
[data-testid="stForm"], .stRadio, .stSlider, .stTabs [data-baseweb="tab-list"] {direction: rtl; text-align: right;}
html, body, [class*="css"], .stMarkdown, .stChatInput textarea {font-family:'Cairo','Inter',sans-serif;}
.stChatInput textarea, [data-testid="stForm"] input, [data-testid="stForm"] textarea {direction: rtl;}
.card::before {transform-origin:right;}
</style>
"""


def build_css(theme: str) -> str:
    g1, g2, g3, acc = THEMES.get(theme, THEMES["Ocean 🌊"])
    return (CSS_TEMPLATE.replace("@G1@", g1).replace("@G2@", g2).replace("@G3@", g3).replace("@ACC@", acc))


def contact_html(t: dict) -> str:
    cards = [f'<a class="c-card" href="mailto:{EMAIL}"><span class="ic">✉️</span>'
             f'<div><small>{t["email_lbl"]}</small><b>{EMAIL}</b></div></a>']
    for p in PHONES:
        cards.append(f'<a class="c-card" href="{wa_link(p)}" target="_blank"><span class="ic">📱</span>'
                     f'<div><small>{t["wa_lbl"]}</small><b>{p}</b></div></a>')
    return f'<div class="sec-title">{t["contact_title"]}</div><div class="contact">{"".join(cards)}</div>'


# ======================================================================
# 5) STEP-BY-STEP GUIDE (works without the AI model)
# ======================================================================
def _g_pick(service: str):
    st.session_state.g_service = service
    st.session_state.g_step = 1


def _g_go(step: int):
    st.session_state.g_step = step


def _g_restart():
    for k in ("g_service", "g_step", "g_complexity", "g_weeks"):
        st.session_state.pop(k, None)


def render_guide(t: dict, lang: str):
    g = t["guide"]
    st.session_state.setdefault("g_step", 0)
    st.session_state.setdefault("g_complexity", "medium")
    st.session_state.setdefault("g_weeks", 4)
    step = st.session_state.g_step
    service = st.session_state.get("g_service")
    st.progress(min(step + 1, 4) / 4 if step < 4 else 1.0)

    def sname(k):
        return SERVICES[k]["ar"] if lang == "ar" else k

    def sdesc(k):
        return SERVICES[k]["description_ar"] if lang == "ar" else SERVICES[k]["description"]

    if step == 0:
        st.markdown(f'<div class="sec-title">{g["pick"]}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (k, v) in enumerate(SERVICES.items()):
            with cols[i % 2]:
                st.button(f'{v["icon"]} {sname(k)}', key=f"g_svc_{i}", on_click=_g_pick, args=(k,))
                st.caption(sdesc(k))
        return

    if service in SERVICES:
        st.markdown(f'<div class="chosen"><b>{SERVICES[service]["icon"]} {sname(service)}</b><br>'
                    f'<small>{sdesc(service)}</small></div>', unsafe_allow_html=True)

    if step == 1:
        st.subheader(g["s2"])
        st.radio("complexity", ["low", "medium", "high"], key="g_complexity",
                 format_func=lambda x: g["complexity"][x], label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.button(g["back"], key="g_back1", on_click=_g_go, args=(0,))
        c2.button(g["next"], key="g_next1", on_click=_g_go, args=(2,))

    elif step == 2:
        st.subheader(g["s3"])
        st.slider(g["weeks"], 1, 24, key="g_weeks")
        c1, c2 = st.columns(2)
        c1.button(g["back"], key="g_back2", on_click=_g_go, args=(1,))
        c2.button(g["next"], key="g_next2", on_click=_g_go, args=(3,))

    elif step == 3:
        st.subheader(g["s4"])
        weeks, cx = st.session_state.g_weeks, st.session_state.g_complexity
        est = estimate_project_cost(service, cx, weeks)
        c1, c2 = st.columns(2)
        c1.metric(g["est_range"], "$" + est["estimated_range_usd"].replace(" - ", " – $"))
        c2.metric(g["weeks"], f'{est["timeline_weeks"]} {g["unit"]}')
        st.markdown(f"**{g['phases']}**")
        for name, ph in zip(g["phase_names"], build_phases(est["timeline_weeks"])):
            st.markdown(f"- {name} — ~{ph['approx_weeks']} {g['unit']}")
        st.caption(g["note"])
        st.divider()
        st.markdown(f"**{g['form_title']}**")
        with st.form("g_form"):
            name = st.text_input(g["name"], key="g_name")
            contact = st.text_input(g["contact"], key="g_contact")
            when = st.text_input(g["when"], key="g_when")
            details = st.text_area(g["details"], key="g_details", height=90)
            sent = st.form_submit_button(g["submit"], key="g_submit")
        if sent:
            if not name.strip() or not contact.strip():
                st.warning(g["need"])
            else:
                create_project_proposal(name.strip(), service, details.strip() or "-", cx, weeks)
                book_consultation(name.strip(), contact.strip(), when.strip() or "-", service)
                if re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", contact.strip()):
                    save_lead(name.strip(), contact.strip(), f"{service}: {details.strip() or '-'}")
                st.session_state.g_step = 4
                st.rerun()
        st.button(g["back"], key="g_back3", on_click=_g_go, args=(2,))

    else:  # step 4: done
        st.success(g["done"])
        st.markdown(CONTACT_MD)
        st.button(g["restart"], key="g_restart", on_click=_g_restart)


# ======================================================================
# 6) PAGE
# ======================================================================
st.set_page_config(page_title="InnovateX Solution Agent", page_icon="🤖", layout="centered")

st.session_state.setdefault("lang", "en")
st.session_state.setdefault("theme", "Ocean 🌊")

with st.sidebar:
    choice = st.radio("🌐 Language / اللغة", ["English", "العربية"],
                      index=0 if st.session_state.lang == "en" else 1, horizontal=True)
    st.session_state.lang = "en" if choice == "English" else "ar"
lang = st.session_state.lang
t = T[lang]

with st.sidebar:
    st.radio(t["theme"], list(THEMES), key="theme")

st.markdown(build_css(st.session_state.theme), unsafe_allow_html=True)
if lang == "ar":
    st.markdown(RTL_CSS, unsafe_allow_html=True)

api_key = get_api_key()
sid = get_sid()

if "messages" not in st.session_state:
    st.session_state.messages = load_history(sid)


def clear_chat():
    st.session_state.messages = []
    st.session_state.pop("chat", None)
    history_path(sid).unlink(missing_ok=True)


with st.sidebar:
    st.header(COMPANY["name"])
    st.markdown("📧 " + EMAIL)
    st.markdown("📱 " + " | ".join(PHONES))
    st.link_button(t["cta_wa"], WHATSAPP_LINK)
    st.divider()
    if not api_key:
        api_key = st.text_input("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", type="password")
    st.button(t["clear"], key="clear_side", on_click=clear_chat)
    transcript = "\n\n".join(f"[{m['role']}] {m['content']}" for m in st.session_state.messages) or "—"
    st.download_button(t["download"], transcript, file_name="innovatex-chat.txt")
    st.caption(t["saved"])

# ---------- Hero ----------
pills = "".join(f'<span class="pill">{p}</span>' for p in t["pills"])
st.markdown(
    f'<div class="hero"><span class="badge">{t["badge"]}</span>'
    f'<h1><span>{t["title"]}</span></h1><p>{t["sub"]}</p><div class="pills">{pills}</div>'
    f'<div class="cta"><a class="primary" href="{WHATSAPP_LINK}" target="_blank">{t["cta_wa"]}</a>'
    f'<a class="ghost" href="mailto:{EMAIL}">{t["cta_mail"]}</a></div></div>',
    unsafe_allow_html=True,
)

# ---------- Contact cards (email + both phone numbers) ----------
st.markdown(contact_html(t), unsafe_allow_html=True)

# ---------- Service cards ----------
cards = "".join(
    f'<div class="card"><div class="ic">{v["icon"]}</div>'
    f'<h4>{v["ar"] if lang == "ar" else k}</h4>'
    f'<p>{v["description_ar"] if lang == "ar" else v["description"]}</p></div>'
    for k, v in SERVICES.items()
)
st.markdown(f'<div class="sec-title">{t["services"]}</div><div class="grid">{cards}</div>',
            unsafe_allow_html=True)

# ---------- Chat + Guide tabs ----------
tab_chat, tab_guide = st.tabs([t["guide"]["tab_chat"], t["guide"]["tab_guide"]])

with tab_guide:
    render_guide(t, lang)

with tab_chat:
    if not api_key:
        st.info(t["key_info"])
    else:
        if "chat" not in st.session_state:
            st.session_state.chat = create_chat(api_key, st.session_state.messages)

        cols = st.columns(len(t["quick"]) + 1)
        for i, (label, text) in enumerate(t["quick"]):
            if cols[i].button(label, key=f"quick{i}"):
                st.session_state.pending = text
        cols[-1].button(t["clear"], key="clear_main", on_click=clear_chat)

        with st.chat_message("assistant"):
            st.markdown(t["greeting"])
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        prompt = st.chat_input(t["placeholder"])
        if not prompt and st.session_state.get("pending"):
            prompt = st.session_state.pop("pending")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                if is_contact_request(prompt):
                    # Answered directly from code: instant and always the exact, correct details
                    reply = f"{t['contact_intro']}\n\n{CONTACT_MD}"
                    st.markdown(reply)
                    st.session_state.messages += [{"role": "user", "content": prompt},
                                                  {"role": "assistant", "content": reply}]
                    save_history(sid, st.session_state.messages)
                    st.session_state.pop("chat", None)  # rebuilt from saved history on the next message
                else:
                    try:
                        reply = st.write_stream(st.session_state.chat.stream_message(prompt))
                        st.session_state.messages += [{"role": "user", "content": prompt},
                                                      {"role": "assistant", "content": reply}]
                        save_history(sid, st.session_state.messages)
                    except Exception as e:
                        st.warning(t["busy"] if _is_busy(e) else f'⚠️ {t["err"]}: {e}')
                        st.session_state.pop("chat", None)  # rebuilt cleanly from saved history next time

st.markdown(contact_html(t), unsafe_allow_html=True)
st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
