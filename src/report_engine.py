from fpdf import FPDF
from datetime import datetime

def generate_pdf_report(target_id, status, threat_prob, nlp_summary, ai_logic, connection_breakdown, df_calls):
    report_text = f"""SPATIO-TEMPORAL AI INVESTIGATION REPORT
========================================================================
DATE GENERATED : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
NODE ID        : {target_id}
CLASSIFICATION : {status}
ANOMALY SCORE  : {threat_prob:.2f}%
========================================================================

[ EXECUTIVE SUMMARY ]
{nlp_summary}

------------------------------------------------------------------------
[ XAI STRUCTURAL EXPLANATION ]
{ai_logic}

------------------------------------------------------------------------
[ NODE LEVEL INTELLIGENCE ]
{connection_breakdown}

========================================================================
[ EXTRACTED EVIDENCE LOGS ]
------------------------------------------------------------------------
"""
    if df_calls is not None and not df_calls.empty:
        report_text += df_calls.to_string(index=False, justify='left')
    else:
        report_text += "No anomalous activity found above threshold.\n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    
    safe_text = report_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt=safe_text)
    
    file_path = f"data/AI_Investigation_Report_Node_{target_id}.pdf"
    pdf.output(file_path)
    return file_path
