import os
import json
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.units import inch

# -------------------------------------------------------------------------
# PATHS SETUP
# -------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORTS_DIR / "project_goal_and_outcomes.pdf"

# -------------------------------------------------------------------------
# DOCUMENT SETUP
# -------------------------------------------------------------------------
doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=45,
    bottomMargin=45
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=18,
    leading=23,
    textColor=colors.HexColor("#0f172a"),
    spaceAfter=10
)

subtitle_style = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    alignment=TA_CENTER,
    fontSize=10.5,
    leading=15,
    textColor=colors.HexColor("#475569"),
    spaceAfter=15
)

h1_style = ParagraphStyle(
    "SectionH1",
    parent=styles["Heading2"],
    fontSize=13,
    leading=17,
    textColor=colors.HexColor("#0284c7"),
    spaceBefore=12,
    spaceAfter=6,
    keepWithNext=True
)

h2_style = ParagraphStyle(
    "SectionH2",
    parent=styles["Heading3"],
    fontSize=10.5,
    leading=14,
    textColor=colors.HexColor("#1e293b"),
    spaceBefore=8,
    spaceAfter=4,
    keepWithNext=True
)

body_style = ParagraphStyle(
    "DocBody",
    parent=styles["BodyText"],
    fontSize=9,
    leading=13.5,
    alignment=TA_JUSTIFY,
    textColor=colors.HexColor("#1e293b"),
    spaceAfter=6
)

bullet_style = ParagraphStyle(
    "DocBullet",
    parent=body_style,
    leftIndent=15,
    spaceAfter=4
)

callout_style = ParagraphStyle(
    "DocCallout",
    parent=body_style,
    fontSize=8.5,
    leading=12.5,
    textColor=colors.HexColor("#0369a1")
)

tbl_cell_style = ParagraphStyle(
    "TableCell",
    parent=styles["Normal"],
    fontSize=8,
    leading=10.5,
    textColor=colors.HexColor("#1e293b")
)

tbl_header_style = ParagraphStyle(
    "TableHeader",
    parent=tbl_cell_style,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#0f172a")
)

story = []

# =========================================================================
# HEADER BLOCK
# =========================================================================
story.append(Paragraph("Machine Learning for Drinking-Water Potability Screening", title_style))
story.append(Paragraph("<b>Executive Summary: Core Research Goal, Methodological Design &amp; Key Outcomes</b>", subtitle_style))
story.append(Spacer(1, 8))

# Callout Box: The Core Goal
goal_html = (
    "<b>PRIMARY RESEARCH GOAL:</b><br/>"
    "To establish a realistic, leakage-free performance baseline for machine-learning screening of "
    "drinking-water potability from routine physicochemical parameters, and to assess how missing-data handling "
    "and model choice affect the detection of unsafe water."
)
goal_table = Table([[Paragraph(goal_html, callout_style)]], colWidths=[6.8 * inch])
goal_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E0F2FE")),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#38BDF8")),
    ("PADDING", (0, 0), (-1, -1), 8),
]))
story.append(goal_table)
story.append(Spacer(1, 10))

# =========================================================================
# 1. THE RESEARCH GAP & MOTIVATION
# =========================================================================
story.append(Paragraph("1. Research Gap &amp; The '80%+ Accuracy Fallacy'", h1_style))
story.append(Paragraph(
    "Standard municipal laboratory testing is slow, expensive, and delayed by bacterial incubation times. "
    "While machine learning offers rapid risk screening, published literature on public drinking water datasets "
    "often reports artificially inflated accuracies (>80%–90%). Our audit reveals that these inflated metrics "
    "stem from <b>target leakage</b>—specifically, imputing missing values conditionally on the target column "
    "(e.g., <code>groupby('Potability')</code>) or fitting imputers and scalers prior to train-test splitting. "
    "In a real deployment, the safety status of an unmeasured water source is unknown. This project establishes "
    "a rigorous, leak-free benchmark that reflects genuine operational performance on unseen water sources.",
    body_style
))
story.append(Spacer(1, 6))

# =========================================================================
# 2. THREE SUPPORTING OBJECTIVES
# =========================================================================
story.append(Paragraph("2. Three Supporting Methodological Objectives", h1_style))

story.append(Paragraph("<b>Objective 1: Leakage-Free Classifier Benchmarking</b>", h2_style))
story.append(Paragraph(
    "Evaluate six diverse machine learning architectures (Random Forest, Hyperparameter-Tuned Random Forest, "
    "Support Vector Machine, K-Nearest Neighbors, Decision Tree, Logistic Regression) under a strict 70/15/15 "
    "stratified split (2,292 Train / 492 Validation / 492 Unseen Test). All imputers, feature engineering ratios, "
    "and scalers are fitted strictly on training data only.",
    bullet_style
))

story.append(Paragraph("<b>Objective 2: Quantifying Imputation Impact on Unsafe Water Detection</b>", h2_style))
story.append(Paragraph(
    "Because the primary public health hazard is a false negative (classifying contaminated water as safe), evaluate "
    "five imputation strategies (Mean, Median, KNN, BayesianRidge MICE, and ExtraTrees MICE) to measure their direct "
    "effect on model Recall and F1-score for detecting non-potable water.",
    bullet_style
))

story.append(Paragraph("<b>Objective 3: Explainable AI (SHAP) &amp; WHO Guideline Alignment</b>", h2_style))
story.append(Paragraph(
    "Compute TreeExplainer SHAP (SHapley Additive exPlanations) across all test samples to verify whether the model "
    "learns valid hydrochemical principles matching World Health Organization (WHO) and EPA international guidelines.",
    bullet_style
))
story.append(Spacer(1, 6))

# =========================================================================
# 3. SUMMARY OF KEY OUTCOMES (BENCHMARK DATASET)
# =========================================================================
story.append(Paragraph("3. Key Outcomes: Point-of-Use Drinking Water (N = 3,276)", h1_style))
story.append(Paragraph(
    "We lead with <b>Repeated Stratified 5x3 Cross-Validation</b> (15 folds, preprocessing refitted inside every fold) "
    "and report the held-out test split (n = 492; evaluated in prior project iterations, with ±4% Wilson CI width). "
    "Routine physicochemical parameters alone exhibit weak signal and overlapping distributions between potable (class 1) and unsafe (class 0) water:",
    body_style
))

# Load dynamic CV & Nadeau-Bengio stats
cv_csv_path = OUTPUTS_DIR / "cross_validation_results.csv"
stat_csv_path = OUTPUTS_DIR / "cv_statistical_tests.csv"
cv_note_txt = "Repeated Stratified 5x3 CV: SVM (0.7058 ± 0.0235), Tuned RF (0.6960 ± 0.0218), Plain RF (0.6912 ± 0.0196)."
if cv_csv_path.exists():
    df_cv_m = pd.read_csv(cv_csv_path)
    top3_parts = [f"{r['Model'].replace('Proposed Model - ', '')} ({r['CV_ROC_AUC_Mean']:.4f} ± {r['CV_ROC_AUC_Std']:.4f})" for _, r in df_cv_m.head(3).iterrows()]
    cv_note_txt = "Repeated Stratified 5x3 CV ROC-AUC: " + ", ".join(top3_parts) + "."
if stat_csv_path.exists():
    df_st_m = pd.read_csv(stat_csv_path)
    svm_row = df_st_m[df_st_m["Comparison"].str.startswith("SVM vs.")]
    rf_row = df_st_m[df_st_m["Comparison"].str.startswith("Random Forest vs.")]
    nb_parts = []
    if not svm_row.empty:
        nb_parts.append(f"SVM vs. Tuned RF: t={svm_row.iloc[0]['Nadeau_Bengio_t']:.4f}, p={svm_row.iloc[0]['Nadeau_Bengio_p']:.4f}")
    if not rf_row.empty:
        nb_parts.append(f"Plain RF vs. Tuned RF: t={rf_row.iloc[0]['Nadeau_Bengio_t']:.4f}, p={rf_row.iloc[0]['Nadeau_Bengio_p']:.4f}")
    if nb_parts:
        cv_note_txt += " Nadeau–Bengio corrected resampled t-test (df=14): " + "; ".join(nb_parts) + " (top models are statistically indistinguishable; hyperparameter tuning yielded no measurable gain)."

# Table: Kaggle Model Performance (Dynamically loaded from outputs/final_test_metrics.csv)
test_csv_path = OUTPUTS_DIR / "final_test_metrics.csv"
if test_csv_path.exists():
    df_test_m = pd.read_csv(test_csv_path)
    kaggle_table_data = [[
        Paragraph("<b>Model Architecture</b>", tbl_header_style),
        Paragraph("<b>Accuracy [95% CI]</b>", tbl_header_style),
        Paragraph("<b>Unsafe Recall (Spec)</b>", tbl_header_style),
        Paragraph("<b>False-Safe Rate</b>", tbl_header_style),
        Paragraph("<b>Potable F1</b>", tbl_header_style),
        Paragraph("<b>ROC-AUC</b>", tbl_header_style),
        Paragraph("<b>Brier</b>", tbl_header_style)
    ]]
    for _, row in df_test_m.iterrows():
        is_prop = "Proposed" in row["Model"]
        name_p = Paragraph(f"<b>{row['Model']}</b>" if is_prop else row["Model"], tbl_cell_style)
        acc_str = f"{row['Accuracy']*100:.2f}% {row['Acc_95_CI']}"
        acc_p = Paragraph(f"<b>{acc_str}</b>" if is_prop else acc_str, tbl_cell_style)
        u_rec_p = Paragraph(f"<b>{row['Unsafe_Recall_Spec']*100:.2f}%</b>" if is_prop else f"{row['Unsafe_Recall_Spec']*100:.2f}%", tbl_cell_style)
        fsr_p = Paragraph(f"<b>{row['False_Safe_Rate']*100:.2f}%</b>" if is_prop else f"{row['False_Safe_Rate']*100:.2f}%", tbl_cell_style)
        f1_p = Paragraph(f"<b>{row['F1-Score']*100:.2f}%</b>" if is_prop else f"{row['F1-Score']*100:.2f}%", tbl_cell_style)
        auc_p = Paragraph(f"<b>{row['ROC-AUC']:.4f}</b>" if is_prop else f"{row['ROC-AUC']:.4f}", tbl_cell_style)
        brier_p = Paragraph(f"<b>{row['Brier_Score']:.4f}</b>" if is_prop else f"{row['Brier_Score']:.4f}", tbl_cell_style)
        kaggle_table_data.append([name_p, acc_p, u_rec_p, fsr_p, f1_p, auc_p, brier_p])
    t_k = Table(kaggle_table_data, colWidths=[1.85 * inch, 1.45 * inch, 0.85 * inch, 0.8 * inch, 0.65 * inch, 0.65 * inch, 0.55 * inch])
    t_k.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_k)
story.append(Paragraph(f"<font size='7.5'><i>{cv_note_txt}</i></font>", body_style))

# Dynamic Frozen Threshold & Calibration Note
thr_csv_path = OUTPUTS_DIR / "test_threshold_comparison.csv"
cal_csv_path = OUTPUTS_DIR / "calibration_summary.csv"
thr_cal_parts = []
if thr_csv_path.exists():
    df_thr_t = pd.read_csv(thr_csv_path)
    if len(df_thr_t) >= 2:
        r0, r1 = df_thr_t.iloc[0], df_thr_t.iloc[1]
        thr_cal_parts.append(
            f"<b>Frozen Safety-First Score Cutoff (0.60, chosen on Val):</b> Cuts held-out test False-Safe Rate from "
            f"{r0['False_Safe_Rate']*100:.2f}% to {r1['False_Safe_Rate']*100:.2f}% (Unsafe Recall {r1['Unsafe_Recall_Spec']*100:.2f}%), "
            f"<b>at the operational cost of rejecting {r1['Safe_Water_Rejected_Rate']*100:.2f}% of genuinely safe samples</b> "
            f"(Potable Recall drops to {r1['Potable_Recall']*100:.2f}%). Because <code>class_weight='balanced'</code> distorts scores, 0.60 is a rank cutoff rather than a calibrated probability."
        )
if cal_csv_path.exists():
    df_cal_t = pd.read_csv(cal_csv_path)
    if len(df_cal_t) >= 3:
        b0, b1, b2 = df_cal_t.iloc[0], df_cal_t.iloc[1], df_cal_t.iloc[2]
        thr_cal_parts.append(
            f"<b>Calibration &amp; Brier Skill:</b> Base-rate prevalence Brier={b0['Brier_Score']:.4f}; raw balanced RF Brier={b1['Brier_Score']:.4f} "
            f"(BSS={b1['Brier_Skill_Score']:.4f}); isotonic calibration on Val improves test Brier to {b2['Brier_Score']:.4f} (BSS={b2['Brier_Skill_Score']:.4f})."
        )
if thr_cal_parts:
    story.append(Paragraph("<br/>".join(thr_cal_parts), body_style))
story.append(Spacer(1, 8))

# =========================================================================
# 4. OUTCOMES: IMPUTATION BENCHMARK & TARGET LEAKAGE AUDIT (OBJECTIVE 2)
# =========================================================================
story.append(Paragraph("4. Objective 2 Outcome: Imputation Ablation (CV/Validation) &amp; Target Leakage Audit", h1_style))
# Load dynamic leakage summary if available
rig_json_path = OUTPUTS_DIR / "cpcb_rigorous_summary.json"
leak_txt = "<b>79.47% accuracy and 0.8798 ROC-AUC</b> versus the clean baseline's <b>68.29% accuracy and 0.6735 ROC-AUC</b> (+11.18% Acc, +0.2063 AUC)"
if rig_json_path.exists():
    with open(rig_json_path, "r") as _fj:
        _rig = json.load(_fj)
    _lk = _rig.get("leakage_experiment", {})
    if _lk:
        leak_txt = (
            f"<b>{_lk['leaky_acc']:.2f}% accuracy and {_lk['leaky_auc']:.4f} ROC-AUC</b> versus the clean baseline's "
            f"<b>{_lk['clean_acc']:.2f}% accuracy and {_lk['clean_auc']:.4f} ROC-AUC</b> (+{_lk['gap_acc']:.2f}% Acc, +{_lk['gap_auc']:.4f} AUC)"
        )

story.append(Paragraph(
    "To avoid using the test set for design decisions, all five imputation strategies were benchmarked inside 5-fold "
    "Stratified CV on the training set (with the imputer refitted inside every fold) and on the validation set (n = 491):<br/>"
    "• <b>Statistical Finding:</b> Across 5-fold CV, ExtraTrees MICE (0.6964 ± 0.0210 AUC) and Median imputation (0.6984 ± 0.0152 AUC) "
    "are <b>statistically indistinguishable</b> (paired t-test: t = −0.51, p = 0.6386; Validation McNemar exact p = 1.0000). "
    "On Class-0 Unsafe Recall (Specificity), BayesianRidge MICE (81.33%) and Median (80.67%) slightly outperformed ExtraTrees MICE (78.00%). "
    "Thus, simple imputation performs on par with complex MICE on this weak-signal dataset.<br/>"
    "• <b>Target Leakage Demonstration (Class-Conditional Imputation):</b> To illustrate how published studies report inflated "
    "accuracies (&gt;80%), we compared <i>class-conditional mean imputation before splitting</i> (filling NaNs using the Potability group mean "
    f"on an 85/15 split) against a clean <code>SimpleImputer(mean)</code> fitted only on training data: the target-leaky pipeline reached {leak_txt}.",
    body_style
))

imp_csv_path = OUTPUTS_DIR / "imputation_ablation_study.csv"
if imp_csv_path.exists():
    df_imp_m = pd.read_csv(imp_csv_path)
    imp_table_data = [[
        Paragraph("<b>Imputation Strategy</b>", tbl_header_style),
        Paragraph("<b>5-Fold CV AUC</b>", tbl_header_style),
        Paragraph("<b>Val Accuracy</b>", tbl_header_style),
        Paragraph("<b>Val Unsafe Recall</b>", tbl_header_style),
        Paragraph("<b>Val False-Safe</b>", tbl_header_style),
        Paragraph("<b>Val Potable F1</b>", tbl_header_style),
        Paragraph("<b>Val AUC</b>", tbl_header_style)
    ]]
    for _, r in df_imp_m.iterrows():
        imp_table_data.append([
            Paragraph(str(r["Imputation Strategy"]), tbl_cell_style),
            Paragraph(f"{r['CV_AUC_Mean']:.4f} ± {r['CV_AUC_Std']:.4f}", tbl_cell_style),
            Paragraph(f"{r['Val_Accuracy']*100:.2f}%", tbl_cell_style),
            Paragraph(f"{r['Val_Unsafe_Recall_Spec']*100:.2f}%", tbl_cell_style),
            Paragraph(f"{r['Val_False_Safe_Rate']*100:.2f}%", tbl_cell_style),
            Paragraph(f"{r['Val_Potable_F1']*100:.2f}%", tbl_cell_style),
            Paragraph(f"{r['Val_ROC_AUC']:.4f}", tbl_cell_style)
        ])
    t_imp = Table(imp_table_data, colWidths=[1.8 * inch, 1.1 * inch, 0.75 * inch, 0.85 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch])
    t_imp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_imp)
story.append(Spacer(1, 8))

story.append(PageBreak())

# =========================================================================
# 5. OUTCOMES: EXPLAINABLE AI & WHO GUIDELINES (OBJECTIVE 3)
# =========================================================================
story.append(Paragraph("5. Objective 3 Outcome: SHAP Attribution &amp; Interpretive Caveats", h1_style))
shap_csv_path = OUTPUTS_DIR / "shap_feature_importance.csv"
shap_top_txt = "<code>pH</code> (0.0423), <code>Sulfate</code> (0.0407), and <code>ph_sulfate_ratio</code> (0.0299)"
if shap_csv_path.exists():
    df_sh = pd.read_csv(shap_csv_path)
    if len(df_sh) >= 3:
        shap_top_txt = (
            f"<code>{df_sh.iloc[0]['Feature']}</code> ({df_sh.iloc[0]['Mean_Absolute_SHAP']:.4f}), "
            f"<code>{df_sh.iloc[1]['Feature']}</code> ({df_sh.iloc[1]['Mean_Absolute_SHAP']:.4f}), and "
            f"<code>{df_sh.iloc[2]['Feature']}</code> ({df_sh.iloc[2]['Mean_Absolute_SHAP']:.4f})"
        )
story.append(Paragraph(
    f"TreeExplainer SHAP was evaluated on the 492 test samples. Mean |SHAP| magnitudes are modest (0.007–0.042), "
    f"consistent with a weak-signal model (ROC-AUC ≈ 0.68). The top attributed features are {shap_top_txt}.<br/>"
    "• <b>Correlated Feature Caveat:</b> Because <code>ph_sulfate_ratio</code> is a deterministic function of <code>ph</code> and "
    "<code>Sulfate</code>, SHAP distributes attribution across correlated inputs rather than proving independent causal synergy.<br/>"
    "• <b>WHO Guidance Context:</b> While extreme acidic/alkaline pH and elevated sulfate (&gt;400 mg/L) tend toward negative SHAP values, "
    "a ~67% accuracy classifier has <i>not</i> learned exact regulatory thresholds. Note that WHO Guidelines (4th ed.) set no formal "
    "health-based limit for sulfate (taste/laxative thresholds apply at 250–500 mg/L).",
    body_style
))
story.append(Spacer(1, 8))

# =========================================================================
# 6. SEPARATE REGIONAL CASE STUDY: INDIAN CPCB SURFACE WATER
# =========================================================================
story.append(Paragraph("6. Separate Regional Case Study: Indian CPCB Surface Water Screening (N = 683)", h1_style))
story.append(Paragraph(
    "As a <b>separate regional case study</b> (not a generalization test of the drinking-water potability model), we analyzed "
    "surface-water monitoring records from the Central Pollution Control Board (CPCB / NWMP) covering <b>683 unique lentic stations</b> "
    "across <b>26 Indian States/UTs</b> (488 Lakes, 98 Tanks, 97 Ponds; 0 duplicates). Binary labels are defined as Acceptable = 1 "
    "(WQI Excellent + Good = 369) vs. Degraded = 0 (WQI Poor + Very Poor = 314). Below are test results on 103 stations with "
    "<b>95% Wilson Score Confidence Intervals</b> and true <b>Macro OvR ROC-AUC</b>:",
    body_style
))

cpcb_table_data = [
    [
        Paragraph("<b>Evaluation Level &amp; Suite</b>", tbl_header_style),
        Paragraph("<b>Features Used</b>", tbl_header_style),
        Paragraph("<b>Test Accuracy [95% CI]</b>", tbl_header_style),
        Paragraph("<b>Precision</b>", tbl_header_style),
        Paragraph("<b>Recall</b>", tbl_header_style),
        Paragraph("<b>F1-Score</b>", tbl_header_style),
        Paragraph("<b>ROC-AUC</b>", tbl_header_style)
    ],
    [
        Paragraph("Binary (Full Lab Suite)", tbl_cell_style),
        Paragraph("18 Features (Formula Emulation Only)", tbl_cell_style),
        Paragraph("98.06% [93.19%–99.47%]", tbl_cell_style),
        Paragraph("98.21%", tbl_cell_style),
        Paragraph("98.21%", tbl_cell_style),
        Paragraph("98.21%", tbl_cell_style),
        Paragraph("0.9977", tbl_cell_style)
    ],
    [
        Paragraph("<b>Binary (Instant Field-Probe)</b>", tbl_cell_style),
        Paragraph("<b>11 Features (Partial WQI Overlap)</b>", tbl_cell_style),
        Paragraph("<b>92.23% [85.42%–96.01%]</b>", tbl_cell_style),
        Paragraph("<b>90.00%</b>", tbl_cell_style),
        Paragraph("<b>96.43%</b>", tbl_cell_style),
        Paragraph("<b>93.10%</b>", tbl_cell_style),
        Paragraph("<b>0.9837</b>", tbl_cell_style)
    ],
    [
        Paragraph("4-Class Grade (Full Lab Suite)", tbl_cell_style),
        Paragraph("18 Features (Formula Emulation Only)", tbl_cell_style),
        Paragraph("93.20% [86.63%–96.67%]", tbl_cell_style),
        Paragraph("93.31%", tbl_cell_style),
        Paragraph("93.42%", tbl_cell_style),
        Paragraph("93.30%", tbl_cell_style),
        Paragraph("0.9960 (OvR)", tbl_cell_style)
    ],
    [
        Paragraph("4-Class Grade (Instant Probe)", tbl_cell_style),
        Paragraph("11 Features (Partial WQI Overlap)", tbl_cell_style),
        Paragraph("69.90% [60.46%–77.91%]", tbl_cell_style),
        Paragraph("70.34%", tbl_cell_style),
        Paragraph("69.91%", tbl_cell_style),
        Paragraph("69.95%", tbl_cell_style),
        Paragraph("0.9297 (OvR)", tbl_cell_style)
    ]
]
t_cpcb = Table(cpcb_table_data, colWidths=[1.8 * inch, 1.8 * inch, 1.4 * inch, 0.5 * inch, 0.5 * inch, 0.5 * inch, 0.6 * inch])
t_cpcb.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ("PADDING", (0, 0), (-1, -1), 3.5),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(t_cpcb)
story.append(Spacer(1, 8))

# =========================================================================
# 7. CRITICAL METHODOLOGICAL DISCLOSURES
# =========================================================================
story.append(Paragraph("7. Methodological Disclosures &amp; Spatial Validation", h1_style))
story.append(Paragraph(
    "1. <b>Separate Task (Ambient Surface Water vs. Drinking Potability):</b> Consumer drinking-water potability in India is governed by "
    "BIS IS 10500:2012 (0 coliforms/100 mL). CPCB NWMP monitors ambient surface water (lakes/tanks/ponds) against WQI grades.<br/>"
    "2. <b>Circularity &amp; Partial Formula Overlap:</b> The 98.06% full-suite score (101/103) is circular WQI formula emulation. "
    "The 11-feature probe (<code>Temp_min/max</code>, <code>DO_min/max</code>, <code>pH_min/max</code>, <code>EC_min/max</code>, "
    "<code>Nitrate_min/max</code>, <code>pH_EC_ratio</code>) excludes 5-day BOD and coliforms, but <code>DO</code>, <code>pH</code>, "
    "<code>EC</code>, and <code>Nitrate</code> are WQI inputs (partial-formula prediction).<br/>"
    "3. <b>Data Source &amp; Spatial Validation:</b> Source: <i>Processed CPCB-NWMP 2024 Water Quality Dataset</i>, Mendeley Data "
    "(v2, 2024, CC BY 4.0, https://data.mendeley.com/datasets/6rhkyn9phm/2). Both 5-fold <code>GroupKFold</code> and 26-state "
    "<code>LeaveOneGroupOut</code> (LOSO) evaluations were run on identical folds across Binary RF, Logistic Regression, "
    "<code>DO_min &gt;= 4.0</code>, and 4-Class Regularized RF (all 11 probe features):",
    body_style
))
sp_csv_path = OUTPUTS_DIR / "cpcb_spatial_validation.csv"
if sp_csv_path.exists():
    df_sp = pd.read_csv(sp_csv_path)
    sp_tbl = [[
        Paragraph("<b>Spatial CV</b>", tbl_header_style),
        Paragraph("<b>Model (11 Probe)</b>", tbl_header_style),
        Paragraph("<b>Pooled OOF Acc [95% CI]</b>", tbl_header_style),
        Paragraph("<b>F1 / Macro-F1</b>", tbl_header_style),
        Paragraph("<b>ROC-AUC / OvR</b>", tbl_header_style),
    ]]
    for _, sr in df_sp.iterrows():
        auc_txt = f"{sr['Pooled_ROC_AUC_or_MacroOvR']:.4f}" if pd.notna(sr["Pooled_ROC_AUC_or_MacroOvR"]) else "—"
        sp_tbl.append([
            Paragraph(str(sr["CV_Scheme"]), tbl_cell_style),
            Paragraph(str(sr["Model"]), tbl_cell_style),
            Paragraph(f"{sr['Pooled_OOF_Acc']*100:.2f}% {sr['OOF_Acc_95_CI']}", tbl_cell_style),
            Paragraph(f"{sr['Pooled_F1_or_MacroF1']:.4f}", tbl_cell_style),
            Paragraph(auc_txt, tbl_cell_style),
        ])
    t_sp = Table(sp_tbl, colWidths=[1.35 * inch, 1.85 * inch, 1.65 * inch, 0.95 * inch, 1.0 * inch])
    t_sp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 3.0),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_sp)
    story.append(Paragraph(
        "<font size='7.5'><i>Spatial Insight: Binary RF (86.97%–87.26% OOF) and DO_min >= 4.0 (87.85%) are closely matched for binary triage, "
        "while 4-class field-probe accuracy drops to 49.93%–51.83% across unseen states—confirming 11-feature probes are viable for binary screening, "
        "not 4-tier WQI grading without 5-day BOD/coliforms.</i></font>",
        body_style
    ))
story.append(Spacer(1, 10))

# =========================================================================
# 8. CONCLUSION & PRACTICAL VALUE
# =========================================================================
story.append(Paragraph("8. Summary Takeaway for Academic &amp; Field Practice", h1_style))
summary_box = (
    "<b>FINAL RESEARCH TAKEAWAY:</b><br/>"
    "Rather than proposing an over-optimistic or leak-prone classifier, this study establishes a realistic, "
    "leakage-free baseline for drinking-water potability screening (66.67% accuracy [95% CI: 62.39%–70.69%], 0.6808 ROC-AUC; "
    "77.33% Unsafe-Water Recall / Specificity). Repeated Stratified CV and paired tests show that imputation strategies and top "
    "classifiers are statistically indistinguishable on this weak-signal dataset, while operating-point threshold tuning (e.g. 0.60) "
    "can reduce false-safe errors from 22.0% down to 4.3%. Separately, the Indian CPCB surface-water case study demonstrates "
    "87.78% spatial accuracy across unseen states using 11 instant field-probe parameters."
)
summary_table = Table([[Paragraph(summary_box, callout_style)]], colWidths=[6.8 * inch])
summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#4ADE80")),
    ("PADDING", (0, 0), (-1, -1), 8),
]))
story.append(summary_table)

# Build Document
doc.build(story)
print(f"SUCCESS: Summary PDF created at {PDF_PATH}")
