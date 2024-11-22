# modules/report_generator.py

import os
import logging
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus.tables import TableStyle
from reportlab.pdfgen.canvas import Canvas

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyDocTemplate(SimpleDocTemplate):
    """
    Custom Document Template that overrides the afterFlowable method
    to capture headings and add them to the Table of Contents.
    """

    def __init__(self, *args, **kwargs):
        super(MyDocTemplate, self).__init__(*args, **kwargs)
        self.toc = TableOfContents()
        self.toc.levelStyles = [
            ParagraphStyle(
                fontName='Helvetica-Bold',
                fontSize=14,
                name='TOCHeading1',
                leftIndent=20,
                firstLineIndent=-20,
                spaceBefore=5,
                leading=16
            ),
            ParagraphStyle(
                fontSize=12,
                name='TOCHeading2',
                leftIndent=40,
                firstLineIndent=-20,
                spaceBefore=0,
                leading=12
            ),
        ]

    def afterFlowable(self, flowable):
        """
        Registers headings with the Table of Contents.
        """
        if isinstance(flowable, Paragraph):
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == 'CustomHeading1':
                level = 0
            elif style == 'CustomHeading2':
                level = 1
            else:
                return
            # Register TOC entry
            self.notify('TOCEntry', (level, text, self.page))


class PageNumCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add page numbers"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            Canvas.showPage(self)
        Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont('Helvetica', 9)
        self.drawRightString(
            self._pagesize[0] - 50,
            20,
            f"Page {self._pageNumber} of {page_count}"
        )


def generate_report(
        metrics,
        class_weights,
        history,
        f1_callback,
        config,
        shap_explainer_train,
        shap_values_train,
        X_train_shap,
        X_test_shap,
        shap_plot_paths_train,
        shap_plot_paths_test,
        top_features_test,
        top_features_train,
        best_hyperparameters,
        hyperparameter_performance_paths,
        report_name='Binary_Classification_Report.pdf'  # Default report name
):
    """
    Generates a comprehensive PDF report for a binary classification project.

    Parameters:
    - metrics (dict): Contains various evaluation metrics and paths to related images.
    - class_weights (dict): Class weights used in the model.
    - history (dict): Training history data.
    - f1_callback (callable): Callback function for F1 score evaluation.
    - config (dict): Configuration settings, including the executive summary.
    - shap_explainer_train: SHAP explainer for the training set.
    - shap_values_train: SHAP values for the training set.
    - X_train_shap (pd.DataFrame): Training data used for SHAP.
    - X_test_shap (pd.DataFrame): Test data used for SHAP.
    - shap_plot_paths_train (list): Paths to SHAP plots for the training set.
    - shap_plot_paths_test (list): Paths to SHAP plots for the test set.
    - top_features_test (list): Top features based on SHAP values for the test set.
    - top_features_train (list): Top features based on SHAP values for the training set.
    - best_hyperparameters (dict): Hyperparameters used during model training.
    - hyperparameter_performance_paths (list): Paths to hyperparameter performance plots.
    - report_name (str): Name of the generated PDF report.
    """
    try:
        # Ensure the 'reports' directory exists
        os.makedirs('reports', exist_ok=True)
        report_path = os.path.join('reports', report_name)

        # Initialize the custom document template
        doc = MyDocTemplate(
            report_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        elements = []
        styles = getSampleStyleSheet()

        # Define custom styles
        custom_heading1 = ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            spaceAfter=12
        )
        custom_heading2 = ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            spaceAfter=10
        )
        custom_normal = ParagraphStyle(
            name='CustomNormal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=12
        )

        # Add Title Page
        title = Paragraph("Binary Classification Project Report", custom_heading1)
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))

        # Executive Summary
        exec_summary_text = config.get('report', {}).get('executive_summary', 'No Executive Summary Provided.')
        exec_summary = Paragraph(exec_summary_text, custom_normal)
        elements.append(exec_summary)
        elements.append(Spacer(1, 0.2 * inch))

        # Add Table of Contents
        elements.append(PageBreak())
        toc_title = Paragraph("Table of Contents", custom_heading1)
        elements.append(toc_title)
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(doc.toc)
        elements.append(PageBreak())

        # Function to add headings to the document
        def add_heading(text, style):
            heading = Paragraph(text, style)
            heading._bookmarkName = text
            elements.append(heading)
            elements.append(Spacer(1, 0.1 * inch))

        # Begin adding content sections

        # 1. Model Evaluation Metrics
        add_heading("Model Evaluation Metrics", custom_heading2)

        # Prepare metrics data for table
        metrics_data = [
            ['Metric', 'Value'],
            ['Accuracy', f"{metrics.get('accuracy', 0):.4f}"],
            ['Precision', f"{metrics.get('precision', 0):.4f}"],
            ['Recall', f"{metrics.get('recall', 0):.4f}"],
            ['F1 Score', f"{metrics.get('f1_score', 0):.4f}"],
            ['ROC AUC', f"{metrics.get('roc_auc', 0):.4f}"],
        ]

        # Create the metrics table
        metrics_table = Table(metrics_data, hAlign='LEFT', colWidths=[2 * inch, 4 * inch])
        metrics_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E90FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        metrics_table.setStyle(metrics_table_style)
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.2 * inch))

        # 2. Hyperparameters Used
        add_heading("Hyperparameters Used", custom_heading2)
        hyperparams_data = [['Hyperparameter', 'Value']] + [[k, str(v)] for k, v in best_hyperparameters.items()]
        hyperparams_table = Table(hyperparams_data, hAlign='LEFT', colWidths=[2 * inch, 4 * inch])
        hyperparams_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E90FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        hyperparams_table.setStyle(hyperparams_table_style)
        elements.append(hyperparams_table)
        elements.append(Spacer(1, 0.2 * inch))

        # 3. Confusion Matrix
        add_heading("Confusion Matrix", custom_heading2)
        cm_image_path = metrics.get('confusion_matrix_path', '')
        if cm_image_path and os.path.exists(cm_image_path):
            cm_image = Image(cm_image_path, width=6 * inch, height=4 * inch)
            elements.append(cm_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Confusion matrix image not found at path: {cm_image_path}")
            elements.append(Paragraph("Confusion matrix image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 4. Classification Report
        add_heading("Classification Report", custom_heading2)
        cr_image_path = metrics.get('classification_report_path', '')
        if cr_image_path and os.path.exists(cr_image_path):
            cr_image = Image(cr_image_path, width=6 * inch, height=4 * inch)
            elements.append(cr_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Classification report image not found at path: {cr_image_path}")
            elements.append(Paragraph("Classification report image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 5. ROC Curve
        add_heading("ROC Curve", custom_heading2)
        roc_image_path = metrics.get('roc_curve_path', '')
        if roc_image_path and os.path.exists(roc_image_path):
            roc_image = Image(roc_image_path, width=6 * inch, height=4 * inch)
            elements.append(roc_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"ROC curve image not found at path: {roc_image_path}")
            elements.append(Paragraph("ROC curve image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 6. Precision-Recall Curve
        add_heading("Precision-Recall Curve", custom_heading2)
        pr_image_path = metrics.get('precision_recall_curve_path', '')
        if pr_image_path and os.path.exists(pr_image_path):
            pr_image = Image(pr_image_path, width=6 * inch, height=4 * inch)
            elements.append(pr_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Precision-Recall curve image not found at path: {pr_image_path}")
            elements.append(Paragraph("Precision-Recall curve image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 7. Training History
        add_heading("Training History", custom_heading2)
        th_loss_image_path = metrics.get('training_loss_path', '')
        th_acc_image_path = metrics.get('training_accuracy_path', '')
        # Training Loss
        if th_loss_image_path and os.path.exists(th_loss_image_path):
            th_loss_image = Image(th_loss_image_path, width=6 * inch, height=4 * inch)
            elements.append(th_loss_image)
            elements.append(Spacer(1, 0.1 * inch))
        else:
            logger.warning(f"Training loss image not found at path: {th_loss_image_path}")
            elements.append(Paragraph("Training loss image not found.", custom_normal))
            elements.append(Spacer(1, 0.1 * inch))
        # Training Accuracy
        if th_acc_image_path and os.path.exists(th_acc_image_path):
            th_acc_image = Image(th_acc_image_path, width=6 * inch, height=4 * inch)
            elements.append(th_acc_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Training accuracy image not found at path: {th_acc_image_path}")
            elements.append(Paragraph("Training accuracy image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 8. Validation F1 Score Over Epochs
        add_heading("Validation F1 Score Over Epochs", custom_heading2)
        f1_image_path = metrics.get('validation_f1_score_path', '')
        if f1_image_path and os.path.exists(f1_image_path):
            f1_image = Image(f1_image_path, width=6 * inch, height=4 * inch)
            elements.append(f1_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Validation F1 score image not found at path: {f1_image_path}")
            elements.append(Paragraph("Validation F1 score image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 9. SHAP Summary Plots - Training Set
        add_heading("SHAP Summary Plot - Training Set", custom_heading2)
        shap_train_image_path = metrics.get('shap_summary_train_path', '')
        if shap_train_image_path and os.path.exists(shap_train_image_path):
            shap_train_image = Image(shap_train_image_path, width=6 * inch, height=4 * inch)
            elements.append(shap_train_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"SHAP summary training image not found at path: {shap_train_image_path}")
            elements.append(Paragraph("SHAP summary training image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 10. SHAP Summary Plots - Test Set
        add_heading("SHAP Summary Plot - Test Set", custom_heading2)
        shap_test_image_path = metrics.get('shap_summary_test_path', '')
        if shap_test_image_path and os.path.exists(shap_test_image_path):
            shap_test_image = Image(shap_test_image_path, width=6 * inch, height=4 * inch)
            elements.append(shap_test_image)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"SHAP summary test image not found at path: {shap_test_image_path}")
            elements.append(Paragraph("SHAP summary test image not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 11. SHAP Force and Waterfall Plots (Test Set)
        add_heading("SHAP Force and Waterfall Plots (Test Set)", custom_heading2)
        shap_plot_paths = metrics.get('shap_plot_paths', [])
        for path in shap_plot_paths:
            if os.path.exists(path):
                shap_plot_image = Image(path, width=6 * inch, height=4 * inch)
                elements.append(shap_plot_image)
                elements.append(Spacer(1, 0.2 * inch))
            else:
                logger.warning(f"SHAP plot image not found at path: {path}")
                elements.append(Paragraph(f"SHAP plot image not found at path: {path}", custom_normal))
                elements.append(Spacer(1, 0.2 * inch))

        # 12. Top Features Based on SHAP Values (Training Set)
        add_heading("Top 5 Features Based on SHAP Values (Training Set)", custom_heading2)
        top_features_data_train = [
                                      ['Rank', 'Feature Name'],
                                  ] + [[i + 1, feature] for i, feature in enumerate(top_features_train)]

        top_features_table_train = Table(top_features_data_train, hAlign='LEFT', colWidths=[1 * inch, 5 * inch])
        top_features_table_train_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E90FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        top_features_table_train.setStyle(top_features_table_train_style)
        elements.append(top_features_table_train)
        elements.append(Spacer(1, 0.2 * inch))

        # 13. Top Features Based on SHAP Values (Test Set)
        add_heading("Top 5 Features Based on SHAP Values (Test Set)", custom_heading2)
        top_features_data_test = [
                                     ['Rank', 'Feature Name'],
                                 ] + [[i + 1, feature] for i, feature in enumerate(top_features_test)]

        top_features_table_test = Table(top_features_data_test, hAlign='LEFT', colWidths=[1 * inch, 5 * inch])
        top_features_table_test_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E90FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        top_features_table_test.setStyle(top_features_table_test_style)
        elements.append(top_features_table_test)
        elements.append(Spacer(1, 0.2 * inch))

        # 14. Hyperparameter Combinations and Performance Table
        add_heading("Hyperparameter Combinations and Performance", custom_heading2)

        # Load hyperparameter performance data
        results_csv_path = os.path.join('results', 'hyperparameter_performance.csv')
        if os.path.exists(results_csv_path):
            df_hp = pd.read_csv(results_csv_path)
            df_hp.fillna('N/A', inplace=True)
            # Exclude the 'optimizer' column
            if 'optimizer' in df_hp.columns:
                df_hp.drop(columns=['optimizer'], inplace=True)
            df_hp = df_hp.astype(str)
            # Format the 'score' column to 4 decimal places
            df_hp['score'] = df_hp['score'].apply(lambda x: f"{float(x):.4f}")

            # Prepare table data
            table_data = [df_hp.columns.tolist()] + df_hp.values.tolist()

            # Create the table
            hp_table = Table(table_data, hAlign='LEFT', repeatRows=1)

            # Adjust column widths to fit the page
            num_cols = len(df_hp.columns)
            page_width = doc.width
            col_widths = [page_width / num_cols] * num_cols
            hp_table._argW = col_widths

            hp_table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E90FF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ])
            hp_table.setStyle(hp_table_style)
            elements.append(hp_table)
            elements.append(Spacer(1, 0.2 * inch))
        else:
            logger.warning(f"Hyperparameter performance CSV not found at path: {results_csv_path}")
            elements.append(Paragraph("Hyperparameter performance data not found.", custom_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # 15. Hyperparameter Performance Plots
        if hyperparameter_performance_paths:
            add_heading("Hyperparameter Performance Plots", custom_heading2)
            for path in hyperparameter_performance_paths:
                if os.path.exists(path):
                    hp_plot_image = Image(path, width=6 * inch, height=4 * inch)
                    elements.append(hp_plot_image)
                    elements.append(Spacer(1, 0.2 * inch))
                else:
                    logger.warning(f"Hyperparameter performance plot not found at path: {path}")
                    elements.append(
                        Paragraph(f"Hyperparameter performance plot not found at path: {path}", custom_normal))
                    elements.append(Spacer(1, 0.2 * inch))

        # Build the PDF with multiBuild to enable TOC and page numbers
        doc.multiBuild(elements, canvasmaker=PageNumCanvas)
        logger.info(f"PDF Report generated at {report_path}.")

    except Exception as e:
        logger.error(f"An error occurred while generating the report: {e}")
        logger.exception(e)
        raise
