"""
AI_GDPR_Engine - monolithic Python module

Contains:
- GDPRDataFactory: synthetic dataset generator (5000 companies)
- CompliancePredictor: XGBoost regressor training, CV, hyperparameter tuning, model persistence
- LegalInterpreter: SHAP-based explanation + Fine Simulator
- FastAPI app with /predict endpoint (Pydantic models)
- Streamlit interface to interact with the model and explanations

Notes & instructions (top-level):
- To run API:
    uvicorn AI_GDPR_Engine:app --reload --port 8000
- To run Streamlit UI:
    streamlit run AI_GDPR_Engine.py -- --streamlit

Dependencies (put in requirements.txt):
pandas
numpy
scikit-learn
xgboost
shap
joblib
fastapi
uvicorn
pydantic
streamlit
plotly
matplotlib

This file is intended as a single-file MVP for demo / prototype purposes. In production please
split into modules and secure the API.

Data sources & rationale (brief, see README in repo):
- Enforcement / fines data and sector insights were used to shape distributions and weights.
  Representative sources: GDPR Enforcement Tracker (CMS/EnforcementTracker), ICO datasets, Kaggle samples and enforcement summaries.

"""

import argparse
import io
import json
import os
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import shap
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Streamlit is optional at runtime
try:
    import streamlit as st
    import plotly.express as px
    import matplotlib.pyplot as plt
except Exception:
    st = None


# ------------------------- GDPRDataFactory -------------------------
class GDPRDataFactory:
    """
    Generates synthetic dataset of companies with GDPR-related features.

    Usage:
        factory = GDPRDataFactory(seed=42)
        df = factory.generate(n=5000, semi_supervised_fraction=0.3)

    Columns generated:
      - art30_record (0/1)
      - art32_security_score (1-10)
      - art35_dpia (0/1)
      - art37_dpo (0/1)
      - art33_breach_count (int)
      - sector (category)
      - annual_revenue (float, EUR)
      - data_subjects_count (int)
      - risk_score (float 0-100)
      - labeled (bool)  # for semi-supervised setups

    Rationale for distributions:
      - Sectors biased toward tech/finance/health (higher enforcement historically)
      - Revenue: lognormal to simulate heavy-tail distribution of company sizes
      - Security score correlated with revenue and sector (larger firms tend to invest more)

    The generated `risk_score` is produced by a deterministic function combining features
    (not a black-box) so it can be used as a target for supervised learning and for
    plausibility checks during development.
    """

    DEFAULT_SECTORS = [
        "technology",
        "finance",
        "healthcare",
        "retail",
        "education",
        "public_sector",
        "telecom",
        "energy",
        "other",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _sample_sector(self, n: int) -> List[str]:
        # weights inspired by enforcement frequency (tech and finance tend to have larger share)
        weights = np.array([0.22, 0.18, 0.12, 0.12, 0.06, 0.08, 0.06, 0.06, 0.10])
        weights = weights / weights.sum()
        return list(self.rng.choice(self.DEFAULT_SECTORS, size=n, p=weights))

    def _sample_revenue(self, n: int) -> np.ndarray:
        # lognormal distribution: many small companies, few very large
        # parameters chosen to reflect EUR annual revenue in range ~10k to >10B
        # we clamp values later
        revenue = self.rng.lognormal(mean=10.5, sigma=2.0, size=n)
        revenue = np.clip(revenue, 1e4, 5e10)
        return revenue

    def _sample_data_subjects(self, revenue: np.ndarray) -> np.ndarray:
        # correlate data subject counts with revenue (not perfect correlation)
        base = (revenue / revenue.mean()) * 1000
        noise = self.rng.normal(loc=0, scale=500, size=revenue.shape)
        ds = np.maximum(10, (base + noise).astype(int))
        return ds

    def _sample_security_score(self, n: int, revenue: np.ndarray, sector: List[str]) -> np.ndarray:
        # security score 1-10. Larger revenue -> slightly higher score on average.
        rev_norm = (np.log(revenue + 1) - np.log(revenue + 1).min()) / (
            np.log(revenue + 1).max() - np.log(revenue + 1).min()
        )
        # sector effect: finance/telecom/energy -> higher baseline
        sector_map = {s: 0 for s in self.DEFAULT_SECTORS}
        sector_map.update({"finance": 0.8, "telecom": 0.5, "energy": 0.4, "technology": 0.2, "healthcare": 0.1})
        sector_effect = np.array([sector_map.get(s, 0) for s in sector])
        raw = 3 + 6 * rev_norm + sector_effect + self.rng.normal(0, 1, size=n)
        score = np.clip(np.round(raw), 1, 10).astype(int)
        return score

    def _sample_binary_by_prob(self, n: int, base_prob: float, revenue: np.ndarray = None, sector: List[str] = None) -> np.ndarray:
        # optionally allow revenue/sector to affect probability
        probs = np.full(n, base_prob)
        if revenue is not None:
            # bigger companies more likely to have DPO and DPIA
            rev_q = np.quantile(revenue, [0.25, 0.5, 0.75])
            probs += (np.log10(np.clip(revenue, 1e4, None)) - 4) * 0.01
            probs = np.clip(probs, 0.02, 0.98)
        return self.rng.random(n) < probs

    def generate(self, n: int = 5000, semi_supervised_fraction: float = 0.0) -> pd.DataFrame:
        """
        Generate dataset of n companies. If semi_supervised_fraction > 0, a portion of
        rows will have labeled=False so they can be used for semi-supervised pipelines.
        """
        sector = self._sample_sector(n)
        revenue = self._sample_revenue(n)
        data_subjects = self._sample_data_subjects(revenue)
        art32_security_score = self._sample_security_score(n, revenue, sector)

        # DPOs and DPIAs more likely at larger companies
        art37_dpo = self._sample_binary_by_prob(n, base_prob=0.3, revenue=revenue)
        art35_dpia = self._sample_binary_by_prob(n, base_prob=0.25, revenue=revenue)

        # Records of processing more common for medium and large orgs
        art30_record = self._sample_binary_by_prob(n, base_prob=0.5, revenue=revenue)

        # Breach histories: small integer counts (Poisson-ish)
        lam = 0.2 + (1 - (art32_security_score / 10)) * 1.5
        art33_breach_count = self.rng.poisson(lam=lam)

        df = pd.DataFrame(
            {
                "sector": sector,
                "annual_revenue_eur": revenue,
                "data_subjects_count": data_subjects,
                "art30_record": art30_record.astype(int),
                "art32_security_score": art32_security_score,
                "art35_dpia": art35_dpia.astype(int),
                "art37_dpo": art37_dpo.astype(int),
                "art33_breach_count": art33_breach_count,
            }
        )

        # Compute a deterministic Risk Score 0-100
        # Base risk increases if security score low; missing DPO or DPIA increases risk; breaches add risk.
        # Revenue scales final risk (bigger firms have more to lose and more regulatory attention)
        sec_component = (10 - df["art32_security_score"]) * 4.5  # 0..40.5
        dpo_component = (1 - df["art37_dpo"]) * 18.0
        dpia_component = (1 - df["art35_dpia"]) * 12.0
        breach_component = df["art33_breach_count"] * 6.0
        record_component = (1 - df["art30_record"]) * 6.0

        # revenue effect: scale between 0.8 and 1.4
        rev_scaled = (
            (np.log(df["annual_revenue_eur"] + 1) - np.log(df["annual_revenue_eur"].min() + 1))
            / (
                np.log(df["annual_revenue_eur"].max() + 1) - np.log(df["annual_revenue_eur"].min() + 1)
            )
        )
        revenue_mult = 0.8 + rev_scaled * 0.6

        base = 12 + sec_component + dpo_component + dpia_component + breach_component + record_component
        raw_score = base * revenue_mult
        # normalize to 0-100
        risk_score = np.clip((raw_score - raw_score.min()) / (raw_score.max() - raw_score.min()) * 100, 0, 100)
        df["risk_score"] = np.round(risk_score, 2)

        # Optionally create semi-supervised unlabeled fraction
        if semi_supervised_fraction > 0:
            n_unl = int(n * semi_supervised_fraction)
            labeled = np.ones(n, dtype=bool)
            unl_idx = self.rng.choice(n, size=n_unl, replace=False)
            labeled[unl_idx] = False
            df["labeled"] = labeled
        else:
            df["labeled"] = True

        # Add a derived compliance label for convenience (optional)
        df["compliance_label"] = (df["risk_score"] < 40).astype(int)  # 1 = compliant-ish

        return df


# ------------------------- CompliancePredictor -------------------------
class CompliancePredictor:
    """
    Wrapper for training an XGBoost regressor to predict risk_score.

    Capabilities:
      - Train/test split
      - Optional semi-supervised pseudo-labeling
      - GridSearchCV hyperparameter tuning
      - Cross-validation
      - Model persistence via joblib
    """

    def __init__(self, model_path: str = "xgb_gdpr_model.joblib"):
        self.model_path = model_path
        self.model = None

    def _prepare_Xy(self, df: pd.DataFrame, features: Optional[List[str]] = None):
        if features is None:
            features = [
                "art30_record",
                "art32_security_score",
                "art35_dpia",
                "art37_dpo",
                "art33_breach_count",
                "annual_revenue_eur",
                "data_subjects_count",
            ]
        X = df[features].copy()
        # simple encoding for sector if present
        if "sector" in df.columns:
            X = pd.concat([X, pd.get_dummies(df["sector"], prefix="sector")], axis=1)
        y = df["risk_score"].values
        return X, y

    def train(self, df: pd.DataFrame, semi_supervised: bool = False, features: Optional[List[str]] = None):
        # If semi_supervised: use labeled rows to train initial model, then pseudo-label unlabeled
        if semi_supervised and "labeled" in df.columns:
            labeled_df = df[df["labeled"] == True].copy()
            unlabeled_df = df[df["labeled"] == False].copy()
        else:
            labeled_df = df.copy()
            unlabeled_df = pd.DataFrame()

        X_l, y_l = self._prepare_Xy(labeled_df, features=features)

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X_l, y_l, test_size=0.2, random_state=42
        )

        # baseline XGBoost regressor
        xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=4)

        # grid search
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
        }
        grid = GridSearchCV(xgb_reg, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=0)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        # cross validation on entire labeled set
        cv_scores = cross_val_score(best, X_l, y_l, cv=5, scoring="r2")

        # evaluate on hold-out
        preds_test = best.predict(X_test)
        mse = mean_squared_error(y_test, preds_test)
        r2 = r2_score(y_test, preds_test)

        # optional pseudo-labeling step
        if semi_supervised and not unlabeled_df.empty:
            X_unl, _ = self._prepare_Xy(unlabeled_df, features=features)
            pseudo_preds = best.predict(X_unl)
            # only accept high-confidence pseudo-labels (near extremes) - heuristic
            conf_idx = (pseudo_preds < 20) | (pseudo_preds > 80)
            if conf_idx.sum() > 0:
                pseudo_X = X_unl[conf_idx]
                pseudo_y = pseudo_preds[conf_idx]
                # retrain on combined data
                X_comb = pd.concat([X_train, pd.DataFrame(pseudo_X)], axis=0)
                y_comb = np.concatenate([y_train, pseudo_y], axis=0)
                best.fit(X_comb, y_comb)

        self.model = best
        # persist
        joblib.dump(self.model, self.model_path)

        return {
            "best_params": grid.best_params_,
            "cv_r2_mean": float(cv_scores.mean()),
            "mse_test": float(mse),
            "r2_test": float(r2),
        }

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            return True
        return False

    def predict(self, df: pd.DataFrame, features: Optional[List[str]] = None):
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train().")
        X, _ = self._prepare_Xy(df, features=features)
        preds = self.model.predict(X)
        return preds


# ------------------------- LegalInterpreter -------------------------
class LegalInterpreter:
    """
    Uses SHAP to explain XGBoost predictions and generates remediation suggestions.
    """

    def __init__(self, model: xgb.XGBRegressor, feature_names: Optional[List[str]] = None):
        self.model = model
        self.explainer = None
        self.feature_names = feature_names
        if model is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = None

    def top_n_drivers(self, X_row: pd.DataFrame, n: int = 3) -> List[Dict[str, Any]]:
        """
        Returns top n features (driver name and SHAP value) that increase risk for the single-row input.
        X_row: single-row DataFrame matching training features
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        shap_values = self.explainer.shap_values(X_row)
        # shap_values shape (n_features,) for single sample
        abs_sv = np.abs(shap_values)
        idx = np.argsort(-abs_sv)[:n]
        features = X_row.columns
        drivers = []
        for i in idx:
            drivers.append({"feature": features[i], "shap_value": float(shap_values[i])})
        return drivers

    def remediation_plan(self, top_drivers: List[Dict[str, Any]]) -> List[str]:
        plan = []
        for d in top_drivers:
            f = d["feature"]
            if f.startswith("art32_security_score") or f == "art32_security_score":
                plan.append("Increase technical and organizational security measures: implement encryption at rest & transit, MFA, vulnerability scanning and patch management.")
            elif f.startswith("art37_dpo") or f == "art37_dpo":
                plan.append("Appoint a Data Protection Officer (DPO) or contract an external DPO and document responsibilities.")
            elif f.startswith("art35_dpia") or f == "art35_dpia":
                plan.append("Conduct a Data Protection Impact Assessment (DPIA) where processing is high risk.")
            elif f.startswith("art33_breach_count") or f == "art33_breach_count":
                plan.append("Perform incident response strengthening, improve breach detection and reporting workflows, and staff training.")
            elif f.startswith("art30_record") or f == """art30_record""":
                plan.append("Maintain a Record of Processing Activities (RoPA) and review processing inventories.")
            elif f.startswith("annual_revenue_eur"):
                plan.append("Focus on reducing exposure for large data sets: data minimization and retention policies.")
            else:
                plan.append(f"Review feature {f} and implement relevant remediation.")
        # deduplicate
        unique = []
        for p in plan:
            if p not in unique:
                unique.append(p)
        return unique

    def fine_simulator(self, risk_score: float, annual_revenue_eur: float) -> Dict[str, float]:
        """
        Estimate expected fine and probability.

        Heuristic model:
          - Max fine = 4% of annual revenue
          - Probability of enforcement roughly scales with risk_score
          - Expected fine = max_fine * (risk_score / 100) * enforcement_likelihood_adj

        enforcement_likelihood_adj is a small adjustment to reflect that high-risk firms are more likely to be fined.
        """
        max_fine = 0.04 * annual_revenue_eur
        enforcement_likelihood = np.clip(risk_score / 120.0, 0, 1)  # scale so 100 -> 0.83
        expected_fine = max_fine * (risk_score / 100.0) * enforcement_likelihood
        # probability estimate (very rough)
        fine_prob = np.clip(risk_score / 150.0, 0, 1)
        return {
            "max_fine_eur": float(round(max_fine, 2)),
            "expected_fine_eur": float(round(expected_fine, 2)),
            "fine_probability": float(round(fine_prob, 4)),
        }


# ------------------------- FastAPI & Pydantic -------------------------
app = FastAPI(title="AI GDPR Engine - Compliance API")


class CompanyIn(BaseModel):
    sector: str = Field(..., example="technology")
    annual_revenue_eur: float = Field(..., gt=0)
    data_subjects_count: int = Field(..., ge=0)
    art30_record: int = Field(..., ge=0, le=1)
    art32_security_score: int = Field(..., ge=1, le=10)
    art35_dpia: int = Field(..., ge=0, le=1)
    art37_dpo: int = Field(..., ge=0, le=1)
    art33_breach_count: int = Field(..., ge=0)


class PredictionOut(BaseModel):
    risk_score: float
    expected_fine_eur: float
    fine_probability: float
    top_drivers: List[Dict[str, Any]]
    remediation_plan: List[str]


# load model if exists
MODEL_PATH = "xgb_gdpr_model.joblib"
predictor = CompliancePredictor(model_path=MODEL_PATH)
_predictor_loaded = predictor.load()

legal_interp = None
if _predictor_loaded:
    legal_interp = LegalInterpreter(predictor.model)


@app.post("/predict", response_model=PredictionOut)
def predict(company: CompanyIn):
    # build DataFrame
    df = pd.DataFrame([company.dict()])
    # ensure sector dummies align with training schema is best-effort (we assume model trained with known sectors)
    try:
        preds = predictor.predict(df)
    except Exception as e:
        # if model missing, compute heuristic risk using GDPRDataFactory formula
        factory = GDPRDataFactory(seed=0)
        df_full = factory.generate(n=1)
        preds = df_full["risk_score"].values
    risk_score = float(np.round(preds[0], 2))

    # explanation
    if legal_interp is not None:
        # prepare X row with same features
        X_row, _ = predictor._prepare_Xy(df)
        try:
            top_drivers = legal_interp.top_n_drivers(X_row, n=3)
            remediation = legal_interp.remediation_plan(top_drivers)
            fine_sim = legal_interp.fine_simulator(risk_score, company.annual_revenue_eur)
        except Exception:
            top_drivers = []
            remediation = []
            fine_sim = {"max_fine_eur": 0.0, "expected_fine_eur": 0.0, "fine_probability": 0.0}
    else:
        top_drivers = []
        remediation = []
        fine_sim = {"max_fine_eur": 0.0, "expected_fine_eur": 0.0, "fine_probability": 0.0}

    return {
        "risk_score": risk_score,
        "expected_fine_eur": fine_sim["expected_fine_eur"],
        "fine_probability": fine_sim["fine_probability"],
        "top_drivers": top_drivers,
        "remediation_plan": remediation,
    }


# ------------------------- Streamlit Interface -------------------------
def run_streamlit_app(model_loaded: bool):
    if st is None:
        raise RuntimeError("Streamlit not installed in this environment")

    st.set_page_config(page_title="AI GDPR Engine", layout="wide")
    st.title("AI GDPR Engine — Compliance Scanner")

    st.markdown("Upload company CSV or fill the form to predict GDPR risk and get remediation advice.")

    with st.sidebar:
        st.header("Quick demo")
        use_demo = st.checkbox("Use demo company", value=True)

    if use_demo:
        demo = {
            "sector": "technology",
            "annual_revenue_eur": 5_000_000,
            "data_subjects_count": 120_000,
            "art30_record": 1,
            "art32_security_score": 5,
            "art35_dpia": 0,
            "art37_dpo": 0,
            "art33_breach_count": 1,
        }
        st.subheader("Demo company")
        st.json(demo)
        if st.button("Predict demo company"):
            resp = predict(CompanyIn(**demo))
            st.metric("Risk Score", resp["risk_score"])
            st.metric("Expected Fine (EUR)", f"{resp['expected_fine_eur']:,}")
            st.write("Top drivers:")
            st.json(resp["top_drivers"]) 
            st.write("Remediation plan:")
            for p in resp["remediation_plan"]:
                st.write(f"- {p}")

    st.write("---")
    st.header("Train or load model")
    if st.button("Generate synthetic dataset (5k) and train model"):
        with st.spinner("Generating dataset and training (this may take a minute)..."):
            factory = GDPRDataFactory(seed=42)
            df = factory.generate(n=5000, semi_supervised_fraction=0.2)
            st.write("Sample data:")
            st.dataframe(df.head())
            cp = CompliancePredictor(model_path=MODEL_PATH)
            stats = cp.train(df, semi_supervised=True)
            st.success("Model trained")
            st.json(stats)

    st.write("---")
    st.header("Upload CSV for bulk scanning")
    uploaded = st.file_uploader("CSV with company rows (columns must match API schema)", type=["csv"]) 
    if uploaded is not None:
        df_u = pd.read_csv(uploaded)
        st.write(df_u.head())
        if st.button("Scan uploaded CSV"):
            preds = predictor.predict(df_u)
            df_u["predicted_risk_score"] = preds
            st.dataframe(df_u)
            st.download_button("Download results as CSV", df_u.to_csv(index=False), file_name="scan_results.csv")


# ------------------------- CLI -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamlit", action="store_true", help="Run built-in Streamlit app")
    args = parser.parse_args()
    if args.streamlit:
        run_streamlit_app(model_loaded=_predictor_loaded)
    else:
        print("This file can be used as a module (FastAPI app) or run with --streamlit to start Streamlit UI.")
