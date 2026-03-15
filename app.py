app_code = '''
import streamlit as st
import pickle
import numpy as np
import pdfplumber
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="محلل سياسات الخصوصية",
    page_icon="🔒",
    layout="wide"
)

# ======================================
# تحميل النموذج
# ======================================
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

# ======================================
# قراءة PDF
# ======================================
def read_pdf(file):
    texts = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs = [
                    p.strip() for p in text.split("\\n")
                    if len(p.strip()) > 40
                ]
                texts.extend(paragraphs)
    return texts

# ======================================
# الاسئلة القانونية الـ 21
# ======================================
QUESTIONS = [
    {"id": 1,  "text": "وضوح هوية المتحكم وبيانات الاتصال به والممثل المحلي في مصر"},
    {"id": 2,  "text": "الاساس القانوني للمعالجة لكل غرض موافقة عقد التزام قانوني"},
    {"id": 3,  "text": "الموافقة المسبقة الصريحة الحرة المستنيرة قبل جمع البيانات"},
    {"id": 4,  "text": "تحديد اغراض استخدام البيانات بدقة وبشكل غير مضلل"},
    {"id": 5,  "text": "الحد الادنى من البيانات جمع اقل كمية ضرورية فقط"},
    {"id": 6,  "text": "حق الوصول للبيانات والحصول على نسخة منها"},
    {"id": 7,  "text": "حق التصحيح وتعديل البيانات غير الدقيقة"},
    {"id": 8,  "text": "حق المحو والنسيان وحذف البيانات نهائيا"},
    {"id": 9,  "text": "حق سحب الموافقة في اي وقت بسهولة"},
    {"id": 10, "text": "حق الاعتراض على المعالجة لاغراض التسويق والبروفايل"},
    {"id": 11, "text": "الافصاح عن مشاركة البيانات مع اطراف ثالثة وفئاتهم"},
    {"id": 12, "text": "ضمانات الطرف الثالث والتزامه بنفس مستوى حماية البيانات"},
    {"id": 13, "text": "نقل البيانات عبر الحدود بترخيص رسمي وضمانات تعاقدية"},
    {"id": 14, "text": "التدابير الامنية التقنية والتنظيمية تشفير وصول حماية"},
    {"id": 15, "text": "الابلاغ عن خرق البيانات للمركز خلال 72 ساعة والمتضرر 3 ايام"},
    {"id": 16, "text": "مسؤول حماية البيانات DPO وبيانات التواصل معه"},
    {"id": 17, "text": "التراخيص والتصاريح من مركز حماية البيانات المصري"},
    {"id": 18, "text": "حماية بيانات الاطفال تحت 18 وموافقة ولي الامر"},
    {"id": 19, "text": "سجلات انشطة المعالجة للرقابة والتفتيش"},
    {"id": 20, "text": "وضوح اللغة العربية وسهولة الوصول للسياسة"},
    {"id": 21, "text": "آليات التظلم والشكوى لمركز حماية البيانات المصري"},
]

# ======================================
# تدريب النموذج
# ======================================
@st.cache_resource
def train_system(_law_texts, _penalty_texts, _model):
    if not _law_texts or not _penalty_texts:
        return None, None

    q_texts = [q["text"] for q in QUESTIONS]
    q_vectors = _model.encode(q_texts)

    law_vectors = _model.encode(
        _law_texts, show_progress_bar=False
    )
    penalty_vectors = _model.encode(
        _penalty_texts, show_progress_bar=False
    )

    from sklearn.metrics.pairwise import cosine_similarity
    training_data = []

    for q, q_vec in zip(QUESTIONS, q_vectors):
        sims = cosine_similarity([q_vec], law_vectors)[0]
        top_idx = np.argsort(sims)[-5:][::-1]
        for idx in top_idx:
            if sims[idx] > 0.35:
                training_data.append({
                    "text": _law_texts[idx],
                    "label": 1
                })

        sims = cosine_similarity([q_vec], penalty_vectors)[0]
        top_idx = np.argsort(sims)[-5:][::-1]
        for idx in top_idx:
            if sims[idx] > 0.35:
                training_data.append({
                    "text": _penalty_texts[idx],
                    "label": 0
                })

    if len(training_data) < 10:
        return None, None

    df = pd.DataFrame(training_data).drop_duplicates(
        subset=["text"]
    )
    X = _model.encode(df["text"].tolist())
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    nn.fit(X_scaled, y)

    return nn, scaler

# ======================================
# تحليل السؤال
# ======================================
def analyze(question, policy_texts, policy_vecs,
            law_texts, law_vecs, nn, scaler, model):

    q_vec = model.encode([question])

    sims = cosine_similarity(q_vec, policy_vecs)[0]
    top_idx = np.argsort(sims)[-3:][::-1]

    law_sims = cosine_similarity(q_vec, law_vecs)[0]
    law_idx = np.argmax(law_sims)
    law_ref = law_texts[law_idx]

    evidence_texts = [policy_texts[i] for i in top_idx]
    evidence_vecs = model.encode(evidence_texts)
    evidence_scaled = scaler.transform(evidence_vecs)
    probs = nn.predict_proba(evidence_scaled)
    avg_prob = np.mean(probs[:, 1])

    if avg_prob > 0.6:
        verdict = "تلتزم بهذا المتطلب"
        icon = "✅"
    elif avg_prob > 0.4:
        verdict = "تلتزم جزئيا"
        icon = "⚠️"
    else:
        verdict = "لا تلتزم"
        icon = "❌"

    return {
        "verdict": verdict,
        "icon": icon,
        "confidence": round(avg_prob * 100),
        "evidence": evidence_texts[0][:300],
        "law_ref": law_ref[:300]
    }

# ======================================
# الواجهة
# ======================================
model = load_model()

st.title("🔒 محلل سياسات الخصوصية")
st.markdown("### وفق القانون المصري 151 لسنة 2020")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### الخطوة 1: ارفعي ملفات القانون")
    law_files = st.file_uploader(
        "القانون + اللائحة + الشروح",
        type=["pdf"],
        accept_multiple_files=True,
        key="law"
    )

    st.markdown("### الخطوة 2: ارفعي سياسة الشركة")
    policy_file = st.file_uploader(
        "سياسة الخصوصية المراد تحليلها",
        type=["pdf"],
        key="policy"
    )

    st.markdown("### اسئلة مقترحة")
    suggestions = [
        "هل الشركة بتبيع بياناتي؟",
        "هل يحق لي حذف بياناتي؟",
        "هل الشركة بتشارك بياناتي مع اطراف تانية؟",
        "هل الشركة بتحمي بياناتي من الاختراق؟",
        "هل محتاج اوافق قبل جمع بياناتي؟",
        "هل الشركة بتحتفظ ببياناتي لفترة طويلة؟",
    ]
    for s in suggestions:
        if st.button(s, key=s):
            st.session_state["q"] = s

with col2:
    if law_files and policy_file:

        # قراءة القانون
        law_texts = []
        for f in law_files:
            law_texts.extend(read_pdf(f))

        # قراءة السياسة
        policy_texts = read_pdf(policy_file)

        if not law_texts:
            st.error("ملفات القانون فارغة")
        elif not policy_texts:
            st.error("ملف السياسة فارغ")
        else:
            with st.spinner("جاري تدريب النموذج..."):
                penalty_texts = []
                nn, scaler = train_system(
                    tuple(law_texts),
                    tuple(law_texts),
                    model
                )
                law_vecs = model.encode(law_texts)
                policy_vecs = model.encode(policy_texts)

            if nn is None:
                st.error("بيانات التدريب غير كافية")
            else:
                st.success("النموذج جاهز! اسأل اي سؤال")

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                default_q = st.session_state.get("q", "")
                question = st.chat_input("اكتبي سؤالك هنا...")

                if question or default_q:
                    q = question or default_q
                    st.session_state["q"] = ""

                    with st.chat_message("user"):
                        st.markdown(q)
                    st.session_state.messages.append({
                        "role": "user", "content": q
                    })

                    with st.chat_message("assistant"):
                        with st.spinner("جاري التحليل..."):
                            result = analyze(
                                q, policy_texts, policy_vecs,
                                law_texts, law_vecs,
                                nn, scaler, model
                            )
                        response = (
                            f"**الحكم:** {result[\'icon\']} "
                            f"{result[\'verdict\']}\\n\\n"
                            f"**نسبة الثقة:** "
                            f"{result[\'confidence\']}%\\n\\n"
                            f"**الدليل من السياسة:**\\n"
                            f"> {result[\'evidence\']}\\n\\n"
                            f"**المرجع القانوني:**\\n"
                            f"> {result[\'law_ref\']}"
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
    else:
        st.info("ارفعي الملفات من الشمال عشان تبدئي")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("تم انشاء app.py!")
