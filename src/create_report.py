from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import inch
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent
out = root / "reports" / "final_report.pdf"
outputs = root / "outputs"

doc = SimpleDocTemplate(
    str(out),
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=45,
    bottomMargin=45
)

styles = getSampleStyleSheet()

title = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=20,
    leading=25,
    spaceAfter=20
)

heading = ParagraphStyle(
    "HeadingCustom",
    parent=styles["Heading2"],
    fontSize=14,
    leading=18,
    spaceBefore=12,
    spaceAfter=8
)

body = ParagraphStyle(
    "BodyCustom",
    parent=styles["BodyText"],
    fontSize=10,
    leading=15,
    spaceAfter=8
)

small = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontSize=8,
    leading=11
)

story = []

story.append(Paragraph(
    "Machine Learning for Drinking-Water Potability Screening: Benchmarks, Missing Data, and Explainability",
    title
))

story.append(Paragraph(
    "<b>A Methodological, Leakage-Free Study on Drinking-Water Potability Screening</b><br/>"
    "Benchmarking Classifiers, Missing-Data Strategies, and SHAP Interpretability Across Global &amp; Regional Water Bodies",
    ParagraphStyle("sub", parent=body, alignment=TA_CENTER)
))
story.append(Spacer(1, 14))

story.append(Paragraph(
    "<b>Research Framework &amp; Core Objectives:</b><br/>"
    "• <b>Primary Goal:</b> Establish a realistic, leakage-free performance baseline for machine-learning screening of "
    "drinking-water potability from routine physicochemical parameters, evaluating how missing-data handling and model choice "
    "affect the detection of unsafe water.<br/>"
    "• <b>Objective 1:</b> Benchmark six classifiers under a strictly leakage-free protocol, with all imputation, feature engineering, "
    "and scaling fitted strictly on training data only.<br/>"
    "• <b>Objective 2:</b> Quantify the effect of five imputation strategies (Mean, Median, KNN, BayesianRidge MICE, ExtraTrees MICE) "
    "on recall and F1-score for the unsafe/safe decision threshold.<br/>"
    "• <b>Objective 3:</b> Interpret model decisions using TreeExplainer SHAP to identify key chemical drivers and compare learned boundaries "
    "with WHO drinking-water guidance.<br/>"
    "• <b>Regional Case Study:</b> Validate the framework on 683 lentic stations across 26 Indian states, demonstrating rapid on-site "
    "field-probe screening without 5-day laboratory delays.",
    body
))
story.append(Spacer(1, 10))

story.append(Paragraph("1. Abstract", heading))
story.append(Paragraph(
    "This project develops a machine learning based water quality classification "
    "and evaluation system to classify whether a water sample is Potable (Safe to Drink) or "
    "Non-Potable (Not Safe to Drink). The project uses the Kaggle Water Potability "
    "Dataset containing 3276 samples and nine physicochemical water-quality "
    "features. Preprocessing combines multivariate tree-based iterative imputation "
    "(IterativeImputer with ExtraTreesRegressor), non-linear domain feature engineering "
    "(pH-to-Sulfate interaction ratio), and z-score feature standardization fitted strictly "
    "on the training split. Multiple classification algorithms were evaluated, including "
    "Random Forest, Support Vector Machine (SVM), Logistic Regression, Decision "
    "Tree, and K-Nearest Neighbors (KNN). A Hyperparameter-Tuned Random Forest "
    "was developed as the proposed model using 5-fold Stratified GridSearchCV. An interactive "
    "Streamlit dashboard was also developed for dataset analysis, evaluation matrices, "
    "and model comparison.",
    body
))

story.append(Paragraph("2. Problem Statement", heading))
story.append(Paragraph(
    "Determining whether water is suitable for drinking is an important water "
    "quality classification problem. Manual analysis can require laboratory "
    "testing and expert interpretation. This project investigates how accurately "
    "machine learning models can classify water samples using physicochemical "
    "measurements and compares their performance across comprehensive evaluation metrics.",
    body
))

story.append(Paragraph("3. Objectives", heading))
for item in [
    "Analyze the Water Potability Dataset.",
    "Perform exploratory data analysis and leak-free preprocessing.",
    "Handle missing values in pH, Sulfate, and Trihalomethanes using multivariate tree-based imputation.",
    "Engineer a domain interaction feature (pH-to-Sulfate ratio) to capture non-linear chemical balance.",
    "Split the dataset into stratified training (70%), validation (15%), and testing (15%) sets.",
    "Train and compare multiple machine learning classification models.",
    "Implement Random Forest and SVM as compulsory models.",
    "Develop a Hyperparameter-Tuned Random Forest as the proposed model.",
    "Evaluate models using Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and 5-Fold Cross-Validation.",
    "Present all evaluation matrices and visual model comparisons in a Streamlit dashboard."
]:
    story.append(Paragraph("• " + item, body))

story.append(Paragraph("4. Dataset", heading))
dataset_data = [
    ["Property", "Value"],
    ["Dataset", "Kaggle Water Potability Dataset"],
    ["Total Samples", "3276"],
    ["Raw Input Features", "9 (29,484 cells; 1,434 missing values)"],
    ["Engineered Features", "1 (pH-to-Sulfate Ratio = pH / Sulfate)"],
    ["Total Processed Features", "10 (32,760 feature values; 36,036 total values with target)"],
    ["Target Column", "Potability"],
    ["Duplicate Rows", "0"],
    ["Classes", "2 (Non-Potable: 1998, Potable: 1278)"]
]
t = Table(dataset_data, colWidths=[2.2*inch, 3.5*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)
story.append(Spacer(1, 10))

story.append(Paragraph("5. Input Features", heading))
story.append(Paragraph(
    "The nine raw input features are: pH, Hardness, Solids, Chloramines, Sulfate, "
    "Conductivity, Organic Carbon, Trihalomethanes, and Turbidity. The preprocessing "
    "pipeline automatically derives the tenth feature, pH_sulfate_ratio (pH / Sulfate).",
    body
))
story.append(Paragraph(
    "<b>Target:</b> 0 = Non-Potable / Not Safe, 1 = Potable / Safe.",
    body
))

story.append(Paragraph("6. Data Preprocessing & Key Improvements", heading))
story.append(Paragraph(
    "1. <b>Multivariate Iterative Imputation:</b> Because Sulfate (~23.8%) and pH (~15.0%) have "
    "substantial missingness, replacing missing values with a single global median artificially concentrates "
    "non-potable records in the neutral safe-water zone. Using IterativeImputer with an ExtraTreesRegressor "
    "estimates missing values from the remaining physicochemical parameters while fitting strictly on X_train.<br/>"
    "2. <b>Domain Feature Engineering:</b> Adding the pH-to-Sulfate ratio (pH / Sulfate) inside the pipeline "
    "captures the joint acid-mineral balance that axis-aligned decision trees otherwise need many splits to approximate.<br/>"
    "3. <b>Feature Scaling:</b> StandardScaler normalizes all 10 features to zero mean and unit variance.",
    body
))

story.append(Paragraph("7. Machine Learning Models", heading))
models = [
    ["Model", "Purpose"],
    ["Random Forest", "Compulsory baseline model"],
    ["SVM", "Compulsory classification model"],
    ["Logistic Regression", "Comparison model"],
    ["Decision Tree", "Regularized comparison model (max_depth=8, min_samples_leaf=5)"],
    ["KNN", "Distance-weighted comparison model (n_neighbors=9)"],
    ["Tuned Random Forest", "Proposed improved model"]
]
t = Table(models, colWidths=[2.3*inch, 3.4*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)

story.append(Paragraph("8. Proposed Model", heading))
story.append(Paragraph(
    "The proposed model is a Hyperparameter-Tuned Random Forest. "
    "5-fold Stratified GridSearchCV was used to search for the optimal combination of model "
    "hyperparameters.",
    body
))
params = [
    ["Parameter", "Selected Value"],
    ["n_estimators", "400"],
    ["max_depth", "12"],
    ["max_features", "sqrt"],
    ["min_samples_leaf", "1"],
    ["min_samples_split", "4"],
    ["class_weight", "balanced"]
]
t = Table(params, colWidths=[2.5*inch, 3.2*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("PADDING", (0,0), (-1,-1), 6)
]))
story.append(t)

story.append(Paragraph("9. Validation Model Comparison (n = 491)", heading))
val_csv = outputs / "validation_results.csv"
if val_csv.exists():
    df_val = pd.read_csv(val_csv)
    val_rows = [["Model", "Accuracy", "Unsafe Recall (Spec)", "False-Safe Rate", "Potable F1", "ROC-AUC", "Brier"]]
    for _, r in df_val.iterrows():
        val_rows.append([
            r["Model"], f"{r['Accuracy']:.4f}", f"{r['Unsafe_Recall_Spec']:.4f}",
            f"{r['False_Safe_Rate']:.4f}", f"{r['F1-Score']:.4f}", f"{r['ROC-AUC']:.4f}", f"{r['Brier_Score']:.4f}"
        ])
    t = Table(val_rows, colWidths=[1.85*inch, .62*inch, .82*inch, .75*inch, .65*inch, .62*inch, .55*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("PADDING", (0,0), (-1,-1), 3.5)
    ]))
    story.append(t)

story.append(Paragraph(
    "On the validation split, the Hyperparameter-Tuned Random Forest achieved the highest validation "
    "ROC-AUC (0.6739) and Potable F1-Score (0.5356), though all top tree and kernel models fall within a narrow, "
    "overlapping performance band.",
    body
))

story.append(Paragraph("10. Repeated Stratified 5x3 Cross-Validation (Preprocessing Refitted Inside Every Fold)", heading))
cv_csv = outputs / "cross_validation_results.csv"
if cv_csv.exists():
    df_cv = pd.read_csv(cv_csv)
    cv_rows = [["Model", "Mean ROC-AUC", "Std. Dev.", "Mean Accuracy", "Mean Unsafe Recall (Spec)"]]
    for _, r in df_cv.iterrows():
        cv_rows.append([
            r["Model"], f"{r['CV_ROC_AUC_Mean']:.4f}", f"{r['CV_ROC_AUC_Std']:.4f}",
            f"{r['CV_Accuracy_Mean']:.4f}", f"{r['CV_Unsafe_Recall_Mean']:.4f}"
        ])
    t = Table(cv_rows, colWidths=[2.2*inch, .9*inch, .8*inch, .9*inch, 1.1*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("PADDING", (0,0), (-1,-1), 4)
    ]))
    story.append(t)

story.append(Paragraph(
    "A paired t-test across the 15 repeated CV folds between the highest-AUC model (SVM: 0.7058) and "
    "Tuned Random Forest (0.6960) yields t = 1.8781, p = 0.0814, confirming that the top classifiers are "
    "statistically indistinguishable at α = 0.05.",
    body
))

story.append(PageBreak())

story.append(Paragraph("11. Final Test Results (All Models on 492 Untouched Samples)", heading))
test_csv = outputs / "final_test_metrics.csv"
if test_csv.exists():
    df_test = pd.read_csv(test_csv)
    test_rows = [["Model", "Accuracy [95% CI]", "Unsafe Recall", "False-Safe", "Potable F1", "ROC-AUC", "Brier"]]
    for _, r in df_test.iterrows():
        test_rows.append([
            r["Model"], f"{r['Accuracy']:.4f} {r['Acc_95_CI']}", f"{r['Unsafe_Recall_Spec']:.4f}",
            f"{r['False_Safe_Rate']:.4f}", f"{r['F1-Score']:.4f}", f"{r['ROC-AUC']:.4f}", f"{r['Brier_Score']:.4f}"
        ])
    t = Table(test_rows, colWidths=[1.7*inch, 1.25*inch, .68*inch, .62*inch, .62*inch, .58*inch, .52*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 6.8),
        ("PADDING", (0,0), (-1,-1), 3.5)
    ]))
    story.append(t)
story.append(Spacer(1, 8))

story.append(Paragraph("Operating-Point Threshold Analysis (Validation Set)", heading))
thr_csv = outputs / "validation_threshold_sweep.csv"
if thr_csv.exists():
    df_thr = pd.read_csv(thr_csv)
    thr_rows = [["Threshold", "Accuracy", "Unsafe Recall (Spec)", "False-Safe Rate", "Potable Precision", "Potable Recall"]]
    for _, r in df_thr.iterrows():
        thr_rows.append([
            f"{r['Threshold']:.2f}", f"{r['Accuracy']:.4f}", f"{r['Unsafe_Recall_Spec']:.4f}",
            f"{r['False_Safe_Rate']:.4f}", f"{r['Potable_Precision']:.4f}", f"{r['Potable_Recall']:.4f}"
        ])
    t_thr = Table(thr_rows, colWidths=[0.8*inch, 0.8*inch, 1.1*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    t_thr.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("PADDING", (0,0), (-1,-1), 4)
    ]))
    story.append(t_thr)
    story.append(Paragraph(
        "For safety-critical screening, raising the decision threshold from 0.50 to 0.60 reduces the False-Safe Rate "
        "(passing unsafe water as safe) from 22.00% down to 4.33%, at the cost of lowering Potable Recall from 49.21% to 23.04%.",
        body
    ))

story.append(Paragraph("12. Final Confusion Matrix (Tuned Random Forest, n = 492)", heading))
cm_csv = outputs / "final_confusion_matrix.csv"
if cm_csv.exists():
    df_cm = pd.read_csv(cm_csv, index_col=0)
    cm_rows = [[""] + list(df_cm.columns)]
    for idx, r in df_cm.iterrows():
        cm_rows.append([idx] + [str(v) for v in r.values])
    t = Table(cm_rows, colWidths=[2.1*inch, 1.8*inch, 1.8*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("PADDING", (0,0), (-1,-1), 5)
    ]))
    story.append(t)

story.append(Paragraph(
    "On the untouched test set, the Tuned Random Forest correctly flagged 232 of 300 non-potable/unsafe samples "
    "(Unsafe Recall / Specificity = 77.33%), while 68 unsafe samples were falsely predicted as potable (False-Safe Rate = 22.67%).",
    body
))

for img_name, caption in [
    ("confusion_matrix.png", "Final Test Confusion Matrix"),
    ("roc_curve.png", "Final Test ROC Curve"),
    ("unsafe_pr_curve.png", "Unsafe Water (Class 0) Precision-Recall Curve"),
    ("model_comparison.png", "Overall Model Performance Comparison (Validation Set)")
]:
    img_path = outputs / img_name
    if img_path.exists():
        story.append(Paragraph(caption, heading))
        im = Image(str(img_path))
        im._restrictSize(5.8*inch, 3.6*inch)
        story.append(im)
        story.append(Spacer(1, 8))

story.append(PageBreak())

# ============================================================
# RESEARCH OBJECTIVE SECTIONS: Imputation, SHAP, CPCB Case Study
# ============================================================
story.append(Paragraph("13. Objective 2: Missing-Data Imputation Ablation (CV &amp; Validation)", heading))
story.append(Paragraph(
    "To avoid making design decisions on the test set, all five imputation strategies were benchmarked inside 5-fold "
    "Stratified CV on the training set (with the imputer refitted inside every fold) and evaluated on the validation split (n = 491):",
    body
))

imp_csv = outputs / "imputation_ablation_study.csv"
if imp_csv.exists():
    df_imp = pd.read_csv(imp_csv)
    imp_rows = [["Imputation Strategy", "5-Fold CV AUC", "Val Acc", "Val Unsafe Recall", "Val False-Safe", "Val Potable F1", "Val AUC"]]
    for _, r in df_imp.iterrows():
        imp_rows.append([
            r["Imputation Strategy"], f"{r['CV_AUC_Mean']:.4f} ± {r['CV_AUC_Std']:.4f}",
            f"{r['Val_Accuracy']:.4f}", f"{r['Val_Unsafe_Recall_Spec']:.4f}",
            f"{r['Val_False_Safe_Rate']:.4f}", f"{r['Val_Potable_F1']:.4f}", f"{r['Val_ROC_AUC']:.4f}"
        ])
    t_abl = Table(imp_rows, colWidths=[1.65*inch, 0.95*inch, 0.55*inch, 0.85*inch, 0.72*inch, 0.72*inch, 0.55*inch])
    t_abl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D6EAF8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F7")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 6.8),
        ("PADDING", (0,0), (-1,-1), 3.5)
    ]))
    story.append(t_abl)
story.append(Spacer(1, 8))

story.append(Paragraph(
    "<b>Statistical Finding:</b> Across 5-fold CV, ExtraTrees MICE (0.6964 ± 0.0210 AUC) and Median imputation (0.6984 ± 0.0152 AUC) "
    "are <b>statistically indistinguishable</b> (paired t-test: t = −0.5074, p = 0.6386; Validation McNemar exact p = 1.0000). "
    "On Class-0 Unsafe Recall (Specificity), BayesianRidge MICE (81.33%) and Median (80.67%) slightly outperformed ExtraTrees MICE (78.00%). "
    "Thus, simple univariate imputation performs on par with multivariate MICE on this dataset.",
    body
))
story.append(Spacer(1, 10))

story.append(Paragraph("14. Objective 3: SHAP Attribution &amp; Interpretive Caveats", heading))
story.append(Paragraph(
    "TreeExplainer SHAP was evaluated across the 492 test samples. Mean |SHAP| magnitudes are modest (0.007–0.042), "
    "reflecting the weak predictive signal of the dataset (ROC-AUC ≈ 0.68). Because <code>ph_sulfate_ratio</code> is a deterministic "
    "function of <code>ph</code> and <code>Sulfate</code>, SHAP distributes attribution across correlated inputs rather than proving "
    "independent chemical synergy.",
    body
))

for shap_img, shap_cap in [
    ("shap_bar_plot.png", "Global SHAP Feature Importance Ranking"),
    ("shap_summary_plot.png", "SHAP Beeswarm Summary Plot (Feature Attribution on Potability)")
]:
    p_img = outputs / shap_img
    if p_img.exists():
        story.append(Paragraph(shap_cap, heading))
        im = Image(str(p_img))
        im._restrictSize(5.8*inch, 3.4*inch)
        story.append(im)
        story.append(Spacer(1, 8))

story.append(PageBreak())

story.append(Paragraph("15. Methodological Rigor: Target Leakage Audit &amp; WHO Context", heading))
story.append(Paragraph(
    "<b>WHO Guidance Context:</b> While extreme acidic/alkaline pH and elevated sulfate (&gt;400 mg/L) tend toward negative SHAP values, "
    "a ~67% accuracy model has <i>not</i> learned exact regulatory thresholds. Note that WHO Guidelines for Drinking-Water Quality (4th ed.) "
    "set no formal health-based guideline value for sulfate (taste impairment occurs above 250 mg/L and laxative effects above 500 mg/L).",
    body
))
story.append(Paragraph(
    "<b>Target Leakage Demonstration (Class-Conditional Imputation):</b> Several published studies on this dataset report inflated "
    "accuracies (&gt;80%) by imputing missing values conditionally on the target label before splitting. To demonstrate this effect, "
    "we compared class-conditional mean imputation before splitting (filling NaNs using the Potability group mean on an 85/15 split) "
    "against a clean <code>SimpleImputer(mean)</code> fitted only on training data: the target-leaky pipeline yielded <b>79.47% test accuracy "
    "and 0.8798 ROC-AUC</b> versus the clean baseline's <b>68.29% accuracy and 0.6735 ROC-AUC</b> (+11.18% Acc, +0.2063 ROC-AUC).",
    body
))
story.append(Spacer(1, 12))

story.append(PageBreak())

story.append(Paragraph("16. Separate Regional Case Study: Indian CPCB Surface Water Screening (683 Stations, 26 States)", heading))
story.append(Paragraph(
    "As a <b>separate regional case study</b> (distinct from the point-of-use drinking-water potability task), the pipeline "
    "was applied to surface-water monitoring records from the Central Pollution Control Board (CPCB) National Water Quality "
    "Monitoring Programme (NWMP) spanning <b>683 unique monitoring stations (0 duplicates)</b> across <b>26 Indian States &amp; UTs</b> "
    "(488 Lakes, 98 Tanks, and 97 Ponds; zero missing values). Stations are partitioned into a 70/15/15 stratified split "
    "(478 Train / 102 Validation / 103 Unseen Test) for <b>Binary Screening</b> "
    "(Positive class Acceptable = 1: Excellent [156] + Good [213] = 369 vs. Degraded = 0: Poor [142] + Very Poor [172] = 314) and "
    "<b>4-Class WQI Grade Classification</b>. Source: <i>Processed CPCB-NWMP 2024 Water Quality Dataset and Analysis Code "
    "for Explainable Ensemble Classification of Indian Lakes, Ponds, and Tanks</i>, Mendeley Data (v2, 2024, CC BY 4.0), "
    "https://data.mendeley.com/datasets/6rhkyn9phm/2.",
    body
))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "<b>Methodological Disclosures &amp; Spatial Validation:</b><br/>"
    "1. <i>Separate Problem (Ambient Surface Water vs. Drinking Potability):</i> Consumer drinking-water potability in India is governed "
    "by BIS IS 10500:2012 (0 coliforms/100 mL). CPCB NWMP monitors ambient surface water against Water Quality Index (WQI) grades; "
    "thus, this is a separate ambient surface-water case study, not a generalization test of the drinking-water potability model.<br/>"
    "2. <i>Circularity (18 Features) vs. Partial-Formula Overlap (11 Features):</i> When fed all 18 parameters, the 98.06% score (101/103) "
    "[95% Wilson CI: 93.19%–99.47%] reflects <b>circular WQI formula emulation fidelity</b>. The <b>Instant Field-Probe Model (11 features)</b> "
    "excludes 5-day BOD and coliforms, achieving <b>92.23% accuracy [95% Wilson CI: 85.42%–96.01%] and 0.9837 ROC-AUC</b> on-site; however, "
    "because <code>DO</code>, <code>pH</code>, <code>EC</code>, and <code>Nitrate</code> are themselves WQI inputs, this represents "
    "<b>partial-formula prediction</b> rather than leak-free external forecasting.<br/>"
    "• <i>11 Instant Field-Probe Features:</i> <code>Temp_min</code>, <code>Temp_max</code>, <code>DO_min</code>, <code>DO_max</code>, "
    "<code>pH_min</code>, <code>pH_max</code>, <code>EC_min</code>, <code>EC_max</code>, <code>Nitrate_min</code>, <code>Nitrate_max</code>, "
    "and engineered <code>pH_EC_ratio</code>.<br/>"
    "• <i>Spatial Cross-Validation (5-Fold GroupKFold &amp; 26-State LOSO on Identical Folds):</i> Binary RF achieves <b>87.26% [84.55%, 89.56%]</b> "
    "pooled OOF accuracy (0.9224 ROC-AUC) in 5-fold GroupKFold and <b>86.97% [84.24%, 89.29%]</b> (0.9177 ROC-AUC) in Leave-One-State-Out, "
    "outperforming Logistic Regression (85.36% / 84.92% Acc; 0.8981 / 0.8888 ROC-AUC) and matching the DO_min &gt;= 4.0 mg/L heuristic "
    "(87.85% Acc on identical folds). Under spatial CV, 4-class field-probe accuracy drops to 49.93% (GroupKFold) and 51.83% (LOSO), "
    "confirming instant probes are suited for binary triage rather than 4-class grading without 5-day BOD/coliforms.",
    body
))
story.append(Spacer(1, 8))

import csv as _csv
from xml.sax.saxutils import escape as _xml_escape

def _build_cpcb_table(csv_path, col_widths, max_rows=None):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(_csv.reader(fh))
    if max_rows is not None and len(rows) > max_rows + 1:
        rows = rows[:max_rows + 1]
    formatted = []
    for r_idx, row in enumerate(rows):
        formatted_row = []
        for val in row:
            safe_val = _xml_escape(str(val))
            if r_idx == 0:
                formatted_row.append(Paragraph(f"<b>{safe_val}</b>", small))
            else:
                formatted_row.append(Paragraph(safe_val, small))
        formatted.append(formatted_row)
    tbl = Table(formatted, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6EAF8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl

bin_csv = outputs / "india_binary_metrics.csv"
if bin_csv.exists():
    story.append(Paragraph("<b>Binary Classification Unseen Test Metrics (Acceptable vs. Degraded, n = 103)</b>", body))
    story.append(_build_cpcb_table(
        bin_csv,
        [2.3 * inch, 0.84 * inch, 0.84 * inch, 0.84 * inch, 0.84 * inch, 0.84 * inch]
    ))
    story.append(Spacer(1, 8))

story.append(Paragraph(
    "On the 4-class CPCB water quality grade task, untuned Decision Tree and Random Forest reached 94.17% test accuracy "
    "(0.9426 / 0.9425 Macro-F1), while the GridSearchCV-tuned Random Forest achieved 93.20% accuracy (0.9330 Macro-F1, "
    "5-fold CV Macro-F1 0.9308), compared against SVM (80.58%), Logistic Regression (79.61%), and KNN (78.64%).",
    body
))
story.append(Spacer(1, 6))

probe_csv = outputs / "india_probe_vs_lab.csv"
if probe_csv.exists():
    story.append(Paragraph("<b>Instant Field-Probe Screening (11 Features - True Inference) vs. Full Laboratory Suite (18 Features - WQI Surrogate)</b>", body))
    story.append(_build_cpcb_table(
        probe_csv,
        [1.15 * inch, 1.45 * inch, 0.72 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.84 * inch]
    ))
    story.append(Spacer(1, 8))

state_csv = outputs / "india_state_summary.csv"
if state_csv.exists():
    story.append(Paragraph("<b>Top 10 Monitored Indian States: Station Volume, Degradation Rate &amp; Mean Stressors</b>", body))
    story.append(_build_cpcb_table(
        state_csv,
        [1.25 * inch, 0.72 * inch, 0.68 * inch, 0.75 * inch, 0.75 * inch, 0.78 * inch, 0.72 * inch, 0.85 * inch],
        max_rows=10
    ))
    story.append(Spacer(1, 8))

story.append(Paragraph(
    "<b>Regional &amp; Water-Body Insights:</b> States with the highest share of WQI-degraded stations include "
    "<b>Karnataka</b> (188 stations, 79.26% degraded, Mean BOD_max 35.31 mg/L, Mean FC_max 17,940,095 MPN/100mL) "
    "and <b>Telangana</b> (149 stations, 76.51% degraded, Mean DO_min 1.61 mg/L), followed by Jammu &amp; Kashmir "
    "(50.00% across 26 stations). Conversely, Tripura (0.00% across 18), Andhra Pradesh (5.26% across 19), "
    "Madhya Pradesh (6.25% across 32), and Assam (6.90% across 58) show predominantly acceptable water quality. "
    "Across lentic systems, Tanks exhibit 54.08% degradation (n=98), Lakes 47.54% (n=488), and Ponds 29.90% (n=97). "
    "(Note: These figures summarize observed degraded-station shares and measured dataset indicator averages rather than causal pollution-source attributions.)",
    body
))
story.append(Spacer(1, 8))

for img_name, caption in [
    ("india_state_pollution.png", "State-Wise CPCB Surface Water Quality Distribution"),
    ("india_shap_summary.png", "CPCB TreeExplainer SHAP Beeswarm Attribution (DO_BOD_ratio #1: 0.17527 mean |SHAP|)")
]:
    img_path = outputs / img_name
    if img_path.exists():
        story.append(Paragraph(f"<b>{caption}</b>", body))
        im = Image(str(img_path))
        im._restrictSize(6.5 * inch, 3.8 * inch)
        story.append(im)
        story.append(Spacer(1, 8))

story.append(Paragraph(
    "<b>SHAP Physics Validation:</b> On the full laboratory model, domain-engineered <b>DO_BOD_ratio</b> ranked #1 overall "
    "(0.17527 mean |SHAP|), outranking BOD_max (0.13693) and DO_min (0.07344). Under instant field-probe screening "
    "(without 5-day BOD or coliform assays), <b>DO_min</b> ranked #1 (0.20585) and engineered <b>pH_EC_ratio</b> ranked #4 "
    "(0.04765), retaining 92.23% binary screening accuracy (0.9837 ROC-AUC).",
    body
))
story.append(Spacer(1, 12))

story.append(PageBreak())

story.append(Paragraph("17. Streamlit Evaluation Dashboard", heading))
story.append(Paragraph(
    "An interactive Streamlit dashboard was developed with five sections focused "
    "on dataset exploration, evaluation matrices, and cross-regional validation: Dashboard, Dataset Inspection, "
    "Model Performance, Model Comparison, and India CPCB Case Study.",
    body
))

for img_name, caption in [
    ("dashboard.JPG", "Streamlit Dashboard"),
    ("dataset_summary.JPG", "Dataset Summary"),
    ("model_evaluation.JPG", "Final Model Evaluation"),
    ("model_comparison.JPG", "Model Comparison")
]:
    img_path = root / "screenshots" / img_name
    if img_path.exists():
        story.append(Paragraph(caption, heading))
        im = Image(str(img_path))
        im._restrictSize(6.5*inch, 4.0*inch)
        story.append(im)
        story.append(Spacer(1, 8))

story.append(PageBreak())

story.append(Paragraph("18. Conclusion &amp; Practical Implications", heading))
story.append(Paragraph(
    "This study establishes an honest, leakage-free machine-learning baseline for drinking-water potability screening "
    "and regional surface water assessment. On the global physicochemical benchmark, fitting all preprocessing inside each "
    "CV fold yields a realistic test accuracy of 66.67% [95% CI: 62.39%–70.69%] and 0.6808 ROC-AUC, with 77.33% Unsafe-Water "
    "Recall / Specificity (22.67% False-Safe Rate). Repeated Stratified CV and paired tests show that imputation strategies and top "
    "classifiers are statistically indistinguishable on this weak-signal dataset, while operating-point threshold tuning (e.g. 0.60) "
    "can reduce false-safe errors down to 4.33%. Separately, the Indian CPCB surface-water case study across 683 stations "
    "demonstrates 87.78% spatial accuracy across unseen states using 11 instant field-probe parameters.",
    body
))

story.append(Spacer(1, 20))
story.append(Paragraph(
    "<b>Selected Model:</b> Hyperparameter-Tuned Random Forest (statistically tied with SVM &amp; Plain RF in Repeated CV)<br/>"
    "<b>Kaggle Primary Test (n = 492):</b> 66.67% Accuracy [62.39%, 70.69%] | 77.33% Unsafe Recall | 0.6808 ROC-AUC<br/>"
    "<b>Indian CPCB Case Study (n = 683 stations, 26 states):</b> 92.23% Instant Field-Probe Test Acc | 87.78% State-Grouped CV Acc",
    body
))

doc.build(story)

print(f"FINAL REPORT CREATED: {out}")
