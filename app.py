import os

import gradio as gr
import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = joblib.load("model.joblib")
prep, model = B["prep"], B["model"]
NUM, CAT, DEF, CH = B["NUM"], B["CAT"], B["defaults"], B["choices"]
THR_MED, THR_HIGH, M = B["thr_med"], B["thr_high"], B["metrics"]
FEATS = [str(f) for f in prep.get_feature_names_out()]
explainer = shap.TreeExplainer(model)

AGE_MID = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

ACTIONS = {
    "High": "Priority follow-up: medication reconciliation, discharge-planning review, follow-up visit or call within 48-72 h, consider transitional-care program.",
    "Medium": "Elevated risk: confirm follow-up appointment, patient education, phone check within 7 days.",
    "Low": "Standard discharge process and routine follow-up.",
}
COLORS = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#3c9d5d"}


def tier(p):
    return "High" if p >= THR_HIGH else ("Medium" if p >= THR_MED else "Low")


def prepare(df):
    df = df.copy()
    for c in NUM + CAT:
        if c not in df.columns:
            df[c] = DEF[c]
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(DEF[c])
    for c in CAT:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df[NUM + CAT]


def score(df):
    return model.predict_proba(prep.transform(prepare(df)))[:, 1]


def pos_shap(sv):
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def predict_one(
    age,
    gender,
    race,
    adm_type,
    adm_src,
    dis,
    stay,
    labs,
    procs,
    meds,
    n_out,
    n_emer,
    n_inp,
    n_diag,
    diag1,
    a1c,
    insulin,
    change,
    diab_med,
):
    r = dict(DEF)
    r.update(
        age_num=AGE_MID[age],
        gender=gender,
        race=race,
        admission_type=adm_type,
        admission_source=adm_src,
        discharge_group=dis,
        time_in_hospital=stay,
        num_lab_procedures=labs,
        num_procedures=procs,
        num_medications=meds,
        number_outpatient=n_out,
        number_emergency=n_emer,
        number_inpatient=n_inp,
        number_diagnoses=n_diag,
        diag_1_grp=diag1,
        A1Cresult=a1c,
        insulin=insulin,
        change=change,
        diabetesMed=diab_med,
    )
    r["prior_visits"] = n_out + n_emer + n_inp
    r["lab_per_day"] = labs / max(stay, 1)
    r["meds_per_day"] = meds / max(stay, 1)
    df = prepare(pd.DataFrame([r]))
    Xt = prep.transform(df)
    p = float(model.predict_proba(Xt)[0, 1])
    t = tier(p)

    html = f"""
    <div style='padding:18px;border-radius:12px;border:2px solid {COLORS[t]};'>
      <div style='font-size:14px;opacity:.7'>Estimated 30-day readmission risk</div>
      <div style='font-size:46px;font-weight:700;color:{COLORS[t]}'>{p * 100:.1f}%</div>
      <div style='display:inline-block;padding:4px 14px;border-radius:20px;background:{COLORS[t]};color:white;font-weight:600'>{t.upper()} RISK</div>
      <p style='margin-top:12px'><b>Average patient in training data:</b> {M["base_rate"] * 100:.1f}% &nbsp;|&nbsp;
         <b>Relative risk:</b> {p / M["base_rate"]:.1f}x</p>
      <p><b>Suggested action:</b> {ACTIONS[t]}</p>
      <p style='font-size:12px;opacity:.6'>Decision-support only. Not a substitute for clinical judgement.</p>
    </div>"""

    sv = pos_shap(explainer.shap_values(Xt))[0]
    top = np.argsort(-np.abs(sv))[:8][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(
        [FEATS[i].replace("_", " ") for i in top],
        sv[top],
        color=["#d9534f" if v > 0 else "#4c8bc4" for v in sv[top]],
    )
    ax.axvline(0, c="k", lw=0.8)
    ax.set_title("Top factors (red raises risk, blue lowers)")
    ax.set_xlabel("Impact on log-odds")
    plt.tight_layout()
    return html, fig


def predict_batch(path):
    if path is None:
        raise gr.Error("Upload a CSV first.")
    d = pd.read_csv(path)
    p = score(d)
    out = d.copy()
    out.insert(0, "risk_probability", np.round(p, 4))
    out.insert(1, "risk_tier", [tier(x) for x in p])
    out = out.sort_values("risk_probability", ascending=False)
    out.to_csv("readmission_predictions.csv", index=False)
    summ = (
        out["risk_tier"]
        .value_counts()
        .reindex(["High", "Medium", "Low"])
        .fillna(0)
        .astype(int)
    )
    msg = f"Scored {len(out)} patients: " + ", ".join(
        f"{k}: {v}" for k, v in summ.items()
    )
    return msg, out.head(100), "readmission_predictions.csv"


def card():
    return f"""
### Model card
**Task:** predict readmission within 30 days of discharge (diabetic inpatient encounters).
**Training data:** UCI Diabetes 130-US Hospitals (1999-2008), patient-level train/val/test split.

| Metric (held-out test set, {int(M["n_test"]):,} encounters) | Value |
|---|---|
| ROC-AUC (LightGBM) | **{M["auc"]:.3f}** (logistic baseline {M["auc_baseline"]:.3f}) |
| PR-AUC | {M["pr_auc"]:.3f} (random = {M["base_rate"]:.3f}) |
| Operating threshold | {M["threshold"]:.3f} |
| **Recall (sensitivity)** | **{M["recall"]:.1%}**, readmissions caught |
| Precision | {M["precision"]:.1%} |
| False negatives | {int(M["fn"])} of {int(M["tp"] + M["fn"])} readmissions missed |
| Share of patients flagged | {M["flagged_pct"]:.1%} |

**Limitations:** older US data, diabetic population only, no labs/vitals/notes/social factors. Validate locally, check calibration and
subgroup performance before any clinical use. The tool supports, never replaces, clinical decisions.
"""


with gr.Blocks(title="Readmission Risk Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 30-Day Readmission Risk Predictor\nEnter discharge-time information to estimate risk and see what drives it."
    )
    with gr.Tab("Single patient"):
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("**Demographics**")
                with gr.Row():
                    age = gr.Dropdown(list(AGE_MID), value="[70-80)", label="Age group")
                    gender = gr.Dropdown(
                        CH["gender"], value=DEF["gender"], label="Gender"
                    )
                    race = gr.Dropdown(CH["race"], value=DEF["race"], label="Race")
                gr.Markdown("**Admission and discharge**")
                with gr.Row():
                    adm_type = gr.Dropdown(
                        CH["admission_type"],
                        value=DEF["admission_type"],
                        label="Admission type",
                    )
                    adm_src = gr.Dropdown(
                        CH["admission_source"],
                        value=DEF["admission_source"],
                        label="Admission source",
                    )
                    dis = gr.Dropdown(
                        CH["discharge_group"],
                        value=DEF["discharge_group"],
                        label="Discharged to",
                    )
                diag1 = gr.Dropdown(
                    CH["diag_1_grp"],
                    value=DEF["diag_1_grp"],
                    label="Primary diagnosis group",
                )
                gr.Markdown("**Stay and treatment**")
                stay = gr.Slider(1, 14, value=4, step=1, label="Days in hospital")
                labs = gr.Slider(1, 130, value=43, step=1, label="Lab tests performed")
                procs = gr.Slider(0, 6, value=1, step=1, label="Procedures (non-lab)")
                meds = gr.Slider(
                    1, 80, value=16, step=1, label="Distinct medications given"
                )
                n_diag = gr.Slider(1, 16, value=7, step=1, label="Number of diagnoses")
                gr.Markdown("**Prior-year utilisation**")
                with gr.Row():
                    n_out = gr.Slider(0, 20, value=0, step=1, label="Outpatient visits")
                    n_emer = gr.Slider(0, 20, value=0, step=1, label="Emergency visits")
                    n_inp = gr.Slider(
                        0, 15, value=0, step=1, label="Inpatient admissions"
                    )
                gr.Markdown("**Diabetes management**")
                with gr.Row():
                    a1c = gr.Dropdown(
                        CH["A1Cresult"], value=DEF["A1Cresult"], label="HbA1c result"
                    )
                    insulin = gr.Dropdown(
                        CH["insulin"], value=DEF["insulin"], label="Insulin"
                    )
                with gr.Row():
                    change = gr.Dropdown(
                        CH["change"],
                        value=DEF["change"],
                        label="Medication change this stay",
                    )
                    diab_med = gr.Dropdown(
                        CH["diabetesMed"],
                        value=DEF["diabetesMed"],
                        label="On diabetes medication",
                    )
                btn = gr.Button("Predict risk", variant="primary", size="lg")
            with gr.Column(scale=4):
                out_html = gr.HTML()
                out_plot = gr.Plot()
        btn.click(
            predict_one,
            [
                age,
                gender,
                race,
                adm_type,
                adm_src,
                dis,
                stay,
                labs,
                procs,
                meds,
                n_out,
                n_emer,
                n_inp,
                n_diag,
                diag1,
                a1c,
                insulin,
                change,
                diab_med,
            ],
            [out_html, out_plot],
        )
    with gr.Tab("Batch (CSV)"):
        gr.Markdown(
            "Upload a CSV using the same columns as the template (`batch_template.csv`). Missing columns are filled with typical values. "
            "Results are ranked from highest to lowest risk."
        )
        f = gr.File(label="Patients CSV", type="filepath", file_types=[".csv"])
        go = gr.Button("Score patients", variant="primary")
        msg = gr.Markdown()
        tbl = gr.Dataframe(label="Top 100 by risk")
        dl = gr.File(label="Download full results")
        go.click(predict_batch, f, [msg, tbl, dl])
    with gr.Tab("Model card"):
        gr.Markdown(card())


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
