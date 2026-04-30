from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'
PLOTS_DIR = PROJECT_ROOT / 'outputs' / 'plots'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_STYLES = getSampleStyleSheet()

STYLE_TITLE = ParagraphStyle(
    'CustomTitle',
    parent=BASE_STYLES['Title'],
    fontName='Helvetica-Bold',
    fontSize=24,
    spaceAfter=12,
    alignment=TA_CENTER,
)
STYLE_SUBTITLE = ParagraphStyle(
    'CustomSubtitle',
    parent=BASE_STYLES['Normal'],
    fontName='Helvetica',
    fontSize=13,
    spaceAfter=8,
    alignment=TA_CENTER,
)
STYLE_H1 = ParagraphStyle(
    'CustomH1',
    parent=BASE_STYLES['Heading1'],
    fontName='Helvetica-Bold',
    fontSize=16,
    spaceBefore=16,
    spaceAfter=8,
)
STYLE_H2 = ParagraphStyle(
    'CustomH2',
    parent=BASE_STYLES['Heading2'],
    fontName='Helvetica-Bold',
    fontSize=13,
    spaceBefore=10,
    spaceAfter=6,
)
STYLE_BODY = ParagraphStyle(
    'CustomBody',
    parent=BASE_STYLES['Normal'],
    fontName='Helvetica',
    fontSize=10,
    spaceAfter=6,
    leading=14,
)
STYLE_CAPTION = ParagraphStyle(
    'CustomCaption',
    parent=BASE_STYLES['Normal'],
    fontName='Helvetica-Oblique',
    fontSize=9,
    spaceAfter=4,
    alignment=TA_CENTER,
)


def make_table_style(header_color=colors.HexColor('#CCCCCC')):
    from reportlab.platypus import TableStyle as TS
    return TS([
        ('BACKGROUND',     (0, 0), (-1, 0),  header_color),
        ('TEXTCOLOR',      (0, 0), (-1, 0),  colors.black),
        ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, 0),  10),
        ('FONTNAME',       (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE',       (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#F5F5F5'), colors.white]),
        ('GRID',           (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING',        (0, 0), (-1, -1), 6),
        ('ALIGN',          (0, 0), (-1, -1), 'CENTER'),
    ])


def safe_image(path_str, width=450, height=300):
    from reportlab.platypus import Image, Paragraph
    if path_str and Path(path_str).exists():
        try:
            img = Image(path_str, width=width, height=height)
            img.hAlign = 'CENTER'
            return img
        except Exception:
            pass
    return Paragraph(f"[Plot not available: {path_str}]", STYLE_CAPTION)


def generate_pdf(
    task: str = 'multiclass',
    include_eda: bool = True,
    include_confusion: bool = True,
    include_roc: bool = True,
    include_importance: bool = True,
    include_comparison: bool = True,
    model_source: str = 'local',
) -> Path:
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        PageBreak, HRFlowable,
    )
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from datetime import datetime
    from src.training.trainer import load_results

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = REPORTS_DIR / f'report_{task}_{timestamp}.pdf'

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    story = []
    SP = lambda n=12: Spacer(1, n)
    HR = lambda: HRFlowable(width='100%', thickness=0.5,
                             color=colors.grey, spaceAfter=8)

    # ── PAGE 1: Title ──────────────────────────────────────────────
    story += [
        SP(60),
        Paragraph("Cyber Risk Detection", STYLE_TITLE),
        Paragraph("using Machine Learning", STYLE_TITLE),
        SP(20),
        Paragraph("Comparative Analysis of ML Models on CICIDS2017 Dataset",
                  STYLE_SUBTITLE),
        SP(40),
        HR(),
        SP(10),
    ]

    all_results = load_results()
    filtered = [r for r in all_results if r['task'] == task]
    seen = {}
    for r in filtered:
        seen[r['model_name']] = r
    latest_results = list(seen.values())

    source_label = (
        'Pretrained (pretrained_models/)' if model_source == 'pretrained'
        else 'Trained locally (models/)'
    )
    meta_rows = [
        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Task', 'Binary Classification' if task == 'binary'
                 else 'Multiclass Classification (6 classes)'],
        ['Models evaluated', str(len(latest_results))],
        ['Model source', source_label],
        ['Dataset', 'CICIDS2017 (Canadian Institute for Cybersecurity)'],
        ['Institution', 'Yerevan State University'],
        ['Project', 'Diploma Thesis 2025'],
    ]
    meta_table = Table(meta_rows, colWidths=[5*cm, 11*cm])
    meta_table.setStyle(make_table_style())
    story += [meta_table, PageBreak()]

    # ── SECTION 1: Dataset Overview ────────────────────────────────
    story += [Paragraph("1. Dataset Overview", STYLE_H1), HR()]

    story += [Paragraph("""
    The CICIDS2017 dataset was created by the Canadian Institute for Cybersecurity
    and contains labeled network flow records captured over 5 days. It includes
    benign traffic and 14 attack types grouped into 6 classes for this analysis.
    Heartbleed (11 records) and Infiltration (36 records) were excluded due to
    insufficient samples for reliable model training.
    """, STYLE_BODY), SP()]

    class_info = [
        ['Class', 'Label', 'Approx. Records', 'Attack Type'],
        ['Benign',         '0', '2,273,097', 'Normal traffic'],
        ['DoS/DDoS',       '1', '380,688',   'Availability attacks'],
        ['Reconnaissance', '2', '158,930',   'Network scanning'],
        ['Brute Force',    '3', '13,835',    'Credential attacks'],
        ['Botnet',         '4', '1,966',     'Bot activity'],
        ['Web Attack',     '5', '2,180',     'SQLi, XSS, BF'],
    ]
    t = Table(class_info, colWidths=[3.5*cm, 2*cm, 4*cm, 6.5*cm])
    t.setStyle(make_table_style())
    story += [t, SP()]

    if include_eda:
        eda_path = PLOTS_DIR / 'eda_class_distribution.png'
        story += [
            SP(),
            safe_image(str(eda_path), width=440, height=250),
            Paragraph("Figure 1: Class distribution in CICIDS2017 dataset",
                      STYLE_CAPTION),
            SP(),
        ]

    story.append(PageBreak())

    # ── SECTION 2: Results Summary ─────────────────────────────────
    story += [Paragraph("2. Results Summary", STYLE_H1), HR()]

    if not latest_results:
        story += [Paragraph("No trained models found for this task.", STYLE_BODY)]
    else:
        story += [Paragraph(
            f"The following table summarises the performance of all "
            f"{len(latest_results)} trained models on the {task} task.",
            STYLE_BODY
        ), SP()]

        header = ['Model', 'Accuracy', 'Precision', 'Recall',
                  'F1 Score', 'AUC', 'Time (s)']
        table_data = [header]

        best_f1 = max(latest_results, key=lambda r: r['f1_macro'])

        for r in latest_results:
            auc_str = f"{r['roc_auc']:.4f}" if r.get('roc_auc') else 'N/A'
            row = [
                r['model_name'],
                f"{r['accuracy']:.4f}",
                f"{r['precision_macro']:.4f}",
                f"{r['recall_macro']:.4f}",
                f"{r['f1_macro']:.4f}",
                auc_str,
                f"{r['training_time_seconds']:.1f}",
            ]
            table_data.append(row)

        col_widths = [4*cm, 2.2*cm, 2.2*cm, 2*cm, 2.2*cm, 2*cm, 2*cm]
        summary_table = Table(table_data, colWidths=col_widths)
        style = make_table_style()
        best_idx = latest_results.index(best_f1) + 1
        style.add('FONTNAME',   (0, best_idx), (-1, best_idx), 'Helvetica-Bold')
        style.add('BACKGROUND', (0, best_idx), (-1, best_idx),
                  colors.HexColor('#D4EDDA'))
        summary_table.setStyle(style)
        story += [summary_table, SP()]

    story.append(PageBreak())

    # ── SECTION 3: Per-Model Analysis ──────────────────────────────
    story += [Paragraph("3. Per-Model Analysis", STYLE_H1), HR()]

    for i, r in enumerate(latest_results, 1):
        story += [
            Paragraph(f"3.{i} {r['model_name']}", STYLE_H2),
            SP(4),
        ]

        hp = {k: v for k, v in r.get('hyperparams', {}).items() if k != 'task'}
        if hp:
            hp_data = [['Hyperparameter', 'Value']] + \
                      [[str(k), str(v)] for k, v in hp.items()]
            hp_table = Table(hp_data, colWidths=[6*cm, 10*cm])
            hp_table.setStyle(make_table_style())
            story += [Paragraph("Hyperparameters:", STYLE_BODY), hp_table, SP(6)]

        auc_str = f"{r['roc_auc']:.4f}" if r.get('roc_auc') else 'N/A'
        m_data = [
            ['Metric', 'Value'],
            ['Accuracy',      f"{r['accuracy']:.4f}"],
            ['Precision',     f"{r['precision_macro']:.4f}"],
            ['Recall',        f"{r['recall_macro']:.4f}"],
            ['F1 Score',      f"{r['f1_macro']:.4f}"],
            ['ROC AUC',       auc_str],
            ['Training Time', f"{r['training_time_seconds']:.2f}s"],
            ['Sample Size',   str(r.get('sample_size', 'Full'))],
            ['SMOTE',         'ON' if r.get('use_smote') else 'OFF'],
        ]
        m_table = Table(m_data, colWidths=[6*cm, 10*cm])
        m_table.setStyle(make_table_style())
        story += [Paragraph("Performance Metrics:", STYLE_BODY), m_table, SP(8)]

        plot_paths = r.get('plot_paths', {})

        if include_confusion:
            cm_path = plot_paths.get('confusion_matrix')
            story += [
                safe_image(cm_path, width=460, height=230),
                Paragraph(f"Figure: {r['model_name']} Confusion Matrix",
                          STYLE_CAPTION),
                SP(6),
            ]

        if include_roc:
            roc_path = plot_paths.get('roc_curve')
            if roc_path and Path(roc_path).exists():
                story += [
                    safe_image(roc_path, width=380, height=250),
                    Paragraph(f"Figure: {r['model_name']} ROC Curve",
                              STYLE_CAPTION),
                    SP(6),
                ]

        if include_importance:
            fi_path = plot_paths.get('feature_importance')
            if fi_path and Path(fi_path).exists():
                story += [
                    safe_image(fi_path, width=420, height=280),
                    Paragraph(f"Figure: {r['model_name']} Feature Importances",
                              STYLE_CAPTION),
                    SP(6),
                ]

        story.append(PageBreak())

    # ── SECTION 4: Comparison ──────────────────────────────────────
    if include_comparison:
        story += [Paragraph("4. Model Comparison", STYLE_H1), HR()]

        cmp_metrics = PLOTS_DIR / 'comparison_metrics.png'
        cmp_time    = PLOTS_DIR / 'comparison_time.png'

        story += [
            safe_image(str(cmp_metrics), width=460, height=280),
            Paragraph("Figure: Metrics Comparison across all models", STYLE_CAPTION),
            SP(12),
            safe_image(str(cmp_time), width=460, height=240),
            Paragraph("Figure: Training Time Comparison", STYLE_CAPTION),
            SP(),
            PageBreak(),
        ]

    # ── SECTION 5: Conclusions ─────────────────────────────────────
    story += [Paragraph("5. Conclusions", STYLE_H1), HR()]

    if latest_results:
        best_acc = max(latest_results, key=lambda r: r['accuracy'])
        best_f1  = max(latest_results, key=lambda r: r['f1_macro'])
        best_rec = max(latest_results, key=lambda r: r['recall_macro'])
        fastest  = min(latest_results, key=lambda r: r['training_time_seconds'])

        conclusion_text = (
            f"This report presents a comparative analysis of {len(latest_results)} "
            f"machine learning models trained on the CICIDS2017 network intrusion "
            f"detection dataset for {task} classification."
            f"<br/><br/>"
            f"Among the evaluated models, <b>{best_f1['model_name']}</b> achieved "
            f"the highest F1 score of <b>{best_f1['f1_macro']:.4f}</b>, "
            f"making it the recommended model for balanced precision-recall "
            f"performance in cyber risk detection tasks. "
            f"<b>{best_acc['model_name']}</b> achieved the highest accuracy of "
            f"<b>{best_acc['accuracy']:.4f}</b>. "
            f"<b>{fastest['model_name']}</b> was the fastest to train at "
            f"<b>{fastest['training_time_seconds']:.1f} seconds</b>."
            f"<br/><br/>"
            f"The dataset exhibits significant class imbalance — benign traffic "
            f"accounts for over 80% of records, while attack classes such as "
            f"Botnet (~1,966 records) and Web Attack (~2,180 records) are "
            f"severely underrepresented. SMOTE oversampling was applied during "
            f"training to mitigate this imbalance."
            f"<br/><br/>"
            f"Future work may include evaluation on additional datasets such as "
            f"UNSW-NB15 to assess model generalizability, and exploration of "
            f"deep learning architectures for improved detection of rare attack "
            f"categories."
        )
        story += [Paragraph(conclusion_text, STYLE_BODY), SP(20)]

        rec_data = [
            ['Criterion', 'Recommended Model', 'Score'],
            ['Highest Accuracy', best_acc['model_name'],
             f"{best_acc['accuracy']:.4f}"],
            ['Highest F1 Score', best_f1['model_name'],
             f"{best_f1['f1_macro']:.4f}"],
            ['Highest Recall',   best_rec['model_name'],
             f"{best_rec['recall_macro']:.4f}"],
            ['Fastest Training', fastest['model_name'],
             f"{fastest['training_time_seconds']:.1f}s"],
        ]
        rec_table = Table(rec_data, colWidths=[5*cm, 6*cm, 5*cm])
        rec_table.setStyle(make_table_style())
        story += [rec_table]
    else:
        story += [Paragraph("No results available for conclusions.", STYLE_BODY)]

    doc.build(story)
    return out_path
