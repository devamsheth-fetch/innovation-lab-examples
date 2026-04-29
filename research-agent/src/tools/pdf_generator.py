import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class ReportGenerator:
    def __init__(self, output_path: str = "research_report.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='CenterTitle', parent=self.styles['Heading1'], alignment=1))

    def generate_report(self, executive_summary: str, benchmark_data: list, lineage_data: list):
        doc = SimpleDocTemplate(self.output_path, pagesize=letter)
        elements = []

        elements.append(Paragraph("R2E2 Research Audit Report", self.styles['CenterTitle']))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Executive Summary", self.styles['Heading2']))
        elements.append(Paragraph(executive_summary, self.styles['Normal']))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Comparative Benchmarks", self.styles['Heading2']))
        if benchmark_data:
            table_data = [["Paper / Metric", "Value"]] + benchmark_data
            t = Table(table_data, colWidths=[300, 150])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(t)
        else:
            elements.append(Paragraph("No empirical benchmark data extracted.", self.styles['Normal']))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Citation Lineage (Mermaid Compatible)", self.styles['Heading2']))
        if lineage_data:
            mermaid_text = "graph TD<br/>"
            for edge in lineage_data:
                mermaid_text += f"{edge[0]} --> {edge[1]}<br/>"
            elements.append(Paragraph(mermaid_text, self.styles['Code']))
        else:
            elements.append(Paragraph("No citation lineage available.", self.styles['Normal']))

        doc.build(elements)

if __name__ == "__main__":
    generator = ReportGenerator()
    generator.generate_report(
        executive_summary="This report summarizes recent findings in agentic workflows.",
        benchmark_data=[["arxiv:2401.00001 - HotPotQA", "88%"], ["arxiv:2305.10601 - Latency", "12ms"]],
        lineage_data=[["Seed", "arxiv:2401.00001"], ["Seed", "arxiv:2305.10601"]]
    )
