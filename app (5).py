import os
import sys
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# ------------------------- Configuration -------------------------
MODEL_PATH = "xgb_gdpr_model.joblib"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Real Fine Data (Embedded) -------------------------
REAL_FINES = [
    {"sector": "technology", "fine_eur": 746000000, "revenue_eur": 180000000000},
    {"sector": "technology", "fine_eur": 225000000, "revenue_eur": 117000000000},
    {"sector": "finance", "fine_eur": 50000000, "revenue_eur": 30000000000},
    {"sector": "healthcare", "fine_eur": 15000000, "revenue_eur": 5000000000},
    {"sector": "retail", "fine_eur": 20000000, "revenue_eur": 20000000000},
    {"sector": "telecom", "fine_eur": 40000000, "revenue_eur": 15000000000},
    {"sector": "public_sector", "fine_eur": 5000000, "revenue_eur": 1000000000},
]
REAL_FINES_DF = pd.DataFrame(REAL_FINES)

# ------------------------- GDPRDataFactory (Enhanced) -------------------------
class GDPRDataFactory:
    DEFAULT_SECTORS = [
        "technology", "finance", "healthcare", "retail", "education",
        "public_sector", "telecom", "energy", "other"
    ]
    ARTICLE_WEIGHTS = {
        "art30": 0.05, "art32": 0.25, "art33": 0.10,
        "art35": 0.10, "art37": 0.08
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _sample_sector(self, n: int) -> List[str]:
        sector_counts = REAL_FINES_DF['sector'].value_counts(normalize=True)
        weights = [sector_counts.get(s, 0.05) for s in self.DEFAULT_SECTORS]
        weights = np.array(weights) / np.sum(weights)
        return list(self.rng.choice(self.DEFAULT_SECTORS, size=n, p=weights))

    def _sample_revenue(self, n: int, sector: List[str]) -> np.ndarray:
        sector_revenue_mult = {
            "technology": 1.5, "finance": 2.0, "healthcare": 1.2, "retail": 1.0,
            "education": 0.5, "public_sector": 0.8, "telecom": 1.8, "energy": 1.6, "other": 0.7
        }
        base = self.rng.lognormal(mean=10.5, sigma=2.0, size=n)
        mult = np.array([sector_revenue_mult.get(s, 1.0) for s in sector])
        return np.clip(base * mult, 1e4, 5e10)

    def _sample_security_score(self, n: int, revenue: np.ndarray, sector: List[str]) -> np.ndarray:
        rev_norm = np.log1p(revenue) / np.log1p(revenue).max()
        sector_boost = {"finance": 1.5, "telecom": 1.3, "technology": 1.2}
        boost = np.array([sector_boost.get(s, 1.0) for s in sector])
        raw = 3 + 6 * rev_norm * boost + self.rng.normal(0, 1, size=n)
        return np.clip(np.round(raw), 1, 10).astype(int)

    def _sample_binary(self, n: int, prob_func, **kwargs) -> np.ndarray:
        probs = prob_func(n, **kwargs)
        return (self.rng.random(n) < probs).astype(int)

    def _prob_dpo(self, n: int, revenue: np.ndarray) -> np.ndarray:
        return 0.2 + 0.3 * (np.log1p(revenue) / np.log1p(revenue).max())

    def _prob_dpia(self, n: int, revenue: np.ndarray, sector: List[str]) -> np.ndarray:
        sector_risk = {"healthcare": 0.3, "finance": 0.25, "technology": 0.15}
        base = 0.1 + 0.2 * (np.log1p(revenue) / np.log1p(revenue).max())
        sector_adj = np.array([sector_risk.get(s, 0.0) for s in sector])
        return np.clip(base + sector_adj, 0.05, 0.9)

    def _prob_record(self, n: int, revenue: np.ndarray) -> np.ndarray:
        return 0.3 + 0.4 * (np.log1p(revenue) / np.log1p(revenue).max())

    def _generate_breaches(self, n: int, security_score: np.ndarray) -> np.ndarray:
        lam = 0.5 * (10 - security_score) / 10 + self.rng.random(n) * 0.2
        return self.rng.poisson(lam)

    def generate(self, n: int = 5000) -> pd.DataFrame:
        sector = self._sample_sector(n)
        revenue = self._sample_revenue(n, sector)
        security_score = self._sample_security_score(n, revenue, sector)
        dpo = self._sample_binary(n, self._prob_dpo, revenue=revenue)
        dpia = self._sample_binary(n, self._prob_dpia, revenue=revenue, sector=sector)
        record = self._sample_binary(n, self._prob_record, revenue=revenue)
        breaches = self._generate_breaches(n, security_score)

        df = pd.DataFrame({
            "sector": sector,
            "annual_revenue_eur": revenue,
            "art30_record": record,
            "art32_security_score": security_score,
            "art35_dpia": dpia,
            "art37_dpo": dpo,
            "art33_breach_count": breaches,
        })

        # Risk score calculation
        sec_risk = (10 - security_score) / 10 * self.ARTICLE_WEIGHTS["art32"] * 100
        dpo_risk = (1 - dpo) * self.ARTICLE_WEIGHTS["art37"] * 100
        dpia_risk = (1 - dpia) * self.ARTICLE_WEIGHTS["art35"] * 100
        record_risk = (1 - record) * self.ARTICLE_WEIGHTS["art30"] * 100
        breach_risk = breaches * self.ARTICLE_WEIGHTS["art33"] * 20

        sector_fine_avg = REAL_FINES_DF.groupby('sector')['fine_eur'].mean().to_dict()
        overall_avg = REAL_FINES_DF['fine_eur'].mean()
        sector_multiplier = np.array([sector_fine_avg.get(s, overall_avg) / overall_avg for s in sector])
        sector_multiplier = np.clip(sector_multiplier, 0.5, 2.0)

        raw_score = (sec_risk + dpo_risk + dpia_risk + record_risk + breach_risk) * sector_multiplier
        risk_score = np.clip(raw_score, 0, 100)
        df["risk_score"] = np.round(risk_score, 2)
        return df

# ------------------------- CompliancePredictor -------------------------
class CompliancePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def _prepare_Xy(self, df: pd.DataFrame, training: bool = False):
        features = [
            "art30_record", "art32_security_score", "art35_dpia",
            "art37_dpo", "art33_breach_count", "annual_revenue_eur"
        ]
        X = df[features].copy()
        if "sector" in df.columns:
            dummies = pd.get_dummies(df["sector"], prefix="sector")
            X = pd.concat([X, dummies], axis=1)
        if training:
            self.feature_names = X.columns.tolist()
        else:
            if self.feature_names:
                X = X.reindex(columns=self.feature_names, fill_value=0)
        y = df["risk_score"].values if "risk_score" in df else None
        return X, y

    def train(self, df: pd.DataFrame):
        X, y = self._prepare_Xy(df, training=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=4)
        param_grid = {"n_estimators": [50, 100], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
        grid = GridSearchCV(xgb_reg, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=0)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        joblib.dump(self.model, MODEL_PATH)
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        return {"best_params": grid.best_params_, "mse_test": mse, "r2_test": r2}

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.feature_names = getattr(self.model, "feature_names_in_", None)
            return True
        return False

    def predict(self, df: pd.DataFrame):
        if self.model is None:
            raise ValueError("Model not loaded")
        X, _ = self._prepare_Xy(df, training=False)
        return self.model.predict(X)

# ------------------------- LegalInterpreter -------------------------
class LegalInterpreter:
    REMEDIATION_MAP = {
        "art32_security_score": {
            "article": "Article 32 (Security of processing)",
            "advice": [
                "Implement encryption for data at rest and in transit.",
                "Enable multi-factor authentication (MFA).",
                "Conduct regular vulnerability scans.",
                "Establish an incident response plan."
            ]
        },
        "art37_dpo": {
            "article": "Article 37 (Designation of the Data Protection Officer)",
            "advice": [
                "Appoint a DPO if required.",
                "Ensure DPO independence and resources."
            ]
        },
        "art35_dpia": {
            "article": "Article 35 (Data Protection Impact Assessment)",
            "advice": [
                "Conduct a DPIA for high-risk processing.",
                "Consult the DPO and supervisory authority if needed."
            ]
        },
        "art33_breach_count": {
            "article": "Article 33 (Breach notification)",
            "advice": [
                "Improve breach detection and documentation.",
                "Train staff on breach reporting."
            ]
        },
        "art30_record": {
            "article": "Article 30 (Records of processing activities)",
            "advice": [
                "Maintain comprehensive and up-to-date records."
            ]
        }
    }

    def __init__(self, model):
        self.model = model
        self.feature_names = getattr(model, "feature_names_in_", None)
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except:
            self.explainer = None

    def top_drivers(self, X_row: pd.DataFrame, n: int = 3):
        if self.explainer is None:
            return []
        shap_values = self.explainer.shap_values(X_row)
        abs_sv = np.abs(shap_values)
        idx = np.argsort(-abs_sv)[:n]
        drivers = []
        for i in idx:
            drivers.append({
                "feature": self.feature_names[i],
                "shap_value": float(shap_values[i]),
                "contribution": "positive" if shap_values[i] > 0 else "negative"
            })
        return drivers

    def remediation_plan(self, top_drivers):
        plan = []
        for d in top_drivers:
            feature = d["feature"]
            if feature.startswith("sector_"):
                feature = "sector"
            for key, value in self.REMEDIATION_MAP.items():
                if key in feature:
                    plan.append(f"**{value['article']}**")
                    for tip in value['advice']:
                        plan.append(f"- {tip}")
                    break
            else:
                plan.append(f"Review feature '{feature}'.")
        # Remove duplicates
        unique = []
        for item in plan:
            if item not in unique:
                unique.append(item)
        return unique

    def fine_simulator(self, risk_score, revenue, sector):
        sector_fines = REAL_FINES_DF[REAL_FINES_DF['sector'] == sector]
        if not sector_fines.empty:
            ratio = (sector_fines['fine_eur'] / sector_fines['revenue_eur']).median()
        else:
            ratio = 0.02
        base = ratio * revenue * (risk_score / 100)
        prob = 1 / (1 + np.exp(-0.1 * (risk_score - 50)))
        return {
            "expected_fine_eur": round(base, 2),
            "min_fine_eur": round(base * 0.5, 2),
            "max_fine_eur": round(base * 1.5, 2),
            "fine_probability": round(prob, 4)
        }

# ------------------------- PDF Report -------------------------
def generate_pdf_report(company_dict, prediction, drivers, remediation, fine):
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="gdpr_report_")
    os.close(fd)
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "GDPR Compliance Risk Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Company Information")
    c.setFont("Helvetica", 10)
    y = height - 120
    for key, value in company_dict.items():
        c.drawString(70, y, f"{key}: {value}")
        y -= 15

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, "Risk Assessment")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(70, y, f"Risk Score: {prediction:.2f} / 100")
    y -= 15
    c.drawString(70, y, f"Expected Fine: €{fine['expected_fine_eur']:,.2f}")
    y -= 15
    c.drawString(70, y, f"Fine Probability: {fine['fine_probability']*100:.1f}%")

    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Main Risk Drivers")
    y -= 20
    c.setFont("Helvetica", 10)
    for d in drivers:
        c.drawString(70, y, f"- {d['feature']}: {d['contribution']} (SHAP: {d['shap_value']:.3f})")
        y -= 15

    y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Remediation Plan")
    y -= 20
    c.setFont("Helvetica", 10)
    for line in remediation:
        wrapped = simpleSplit(line, "Helvetica", 10, width - 100)
        for w in wrapped:
            c.drawString(70, y, w)
            y -= 12
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()
    return path

# ------------------------- Streamlit App -------------------------
st.set_page_config(page_title="AI GDPR Engine", layout="wide")
st.title("AI GDPR Engine — Compliance Scanner (Enterprise Demo)")

# Initialize or load model
predictor = CompliancePredictor()
if not predictor.load():
    st.warning("No pre-trained model found. Please train a model first using the 'Model Training' tab.")
interpreter = None
if predictor.model is not None:
    interpreter = LegalInterpreter(predictor.model)

tab1, tab2, tab3 = st.tabs(["Single Company Scan", "Bulk Upload", "Model Training"])

with tab1:
    st.header("Enter Company Details")
    col1, col2 = st.columns(2)
    with col1:
        sector = st.selectbox("Sector", GDPRDataFactory.DEFAULT_SECTORS)
        revenue = st.number_input("Annual Revenue (EUR)", min_value=10000.0, value=5000000.0, step=100000.0)
        subjects = st.number_input("Data Subjects Count", min_value=0, value=120000, step=1000)
        art30 = st.selectbox("Art.30 Record (0/1)", [0, 1], index=1)
        art32 = st.slider("Art.32 Security Score (1-10)", 1, 10, 5)
    with col2:
        art35 = st.selectbox("Art.35 DPIA (0/1)", [0, 1], index=0)
        art37 = st.selectbox("Art.37 DPO (0/1)", [0, 1], index=0)
        breaches = st.number_input("Art.33 Breach Count", min_value=0, value=1, step=1)

    if st.button("Scan Company"):
        if predictor.model is None:
            st.error("Model not loaded. Please train first.")
        else:
            with st.spinner("Analyzing..."):
                # Create DataFrame
                input_dict = {
                    "sector": sector,
                    "annual_revenue_eur": revenue,
                    "art30_record": art30,
                    "art32_security_score": art32,
                    "art35_dpia": art35,
                    "art37_dpo": art37,
                    "art33_breach_count": breaches
                }
                df_input = pd.DataFrame([input_dict])
                pred = predictor.predict(df_input)[0]
                X_row, _ = predictor._prepare_Xy(df_input)
                drivers = interpreter.top_drivers(X_row)
                remediation = interpreter.remediation_plan(drivers)
                fine = interpreter.fine_simulator(pred, revenue, sector)

                # Display
                col1, col2, col3 = st.columns(3)
                col1.metric("Risk Score", f"{pred:.1f}")
                col2.metric("Expected Fine", f"€{fine['expected_fine_eur']:,.0f}")
                col3.metric("Fine Probability", f"{fine['fine_probability']*100:.1f}%")

                st.subheader("Top Risk Drivers")
                st.json(drivers)

                st.subheader("Remediation Plan")
                for item in remediation:
                    st.markdown(item)

                # PDF
                if st.button("Generate PDF Report"):
                    pdf_path = generate_pdf_report(input_dict, pred, drivers, remediation, fine)
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name="compliance_report.pdf")

with tab2:
    st.header("Bulk Upload CSV")
    uploaded = st.file_uploader("Upload CSV with columns: sector, annual_revenue_eur, art30_record, art32_security_score, art35_dpia, art37_dpo, art33_breach_count", type=["csv"])
    if uploaded and predictor.model is not None:
        df_bulk = pd.read_csv(uploaded)
        st.dataframe(df_bulk.head())
        if st.button("Scan Bulk"):
            preds = predictor.predict(df_bulk)
            df_bulk["predicted_risk"] = preds
            st.dataframe(df_bulk)
            csv = df_bulk.to_csv(index=False)
            st.download_button("Download Results", csv, "scan_results.csv")
    elif predictor.model is None:
        st.warning("Model not loaded. Please train first.")

with tab3:
    st.header("Train New Model")
    n_samples = st.number_input("Number of synthetic samples", 1000, 20000, 5000, step=1000)
    if st.button("Train"):
        with st.spinner("Generating data and training..."):
            factory = GDPRDataFactory(seed=42)
            df = factory.generate(n=n_samples)
            stats = predictor.train(df)
            st.success("Training complete")
            st.json(stats)
            # Reload interpreter
            interpreter = LegalInterpreter(predictor.model)
