import streamlit as st
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from scipy.sparse import hstack, csr_matrix

def load_data():
    incident_log = pd.read_csv('C:/Users/daphnee/Desktop/Work/Acads/FYP/Dataset/dashboard/generated_incidents.csv')
    incident_log=incident_log[incident_log.incident_state!='-100']
    # Reformat dates
    incident_log['opened_at'] = pd.to_datetime(incident_log['opened_at'], dayfirst=True, format='mixed')
    incident_log['closed_at'] = pd.to_datetime(incident_log['closed_at'], dayfirst=True, format='mixed')

    # Day of the week column, month and day
    incident_log['day_of_week'] = incident_log['opened_at'].dt.day_name()
    incident_log['month'] = incident_log['opened_at'].dt.month
    incident_log['day'] = incident_log['opened_at'].dt.day
    incident_log['year'] = incident_log['opened_at'].dt.year

    containment_df = pd.read_csv('C:/Users/daphnee/Desktop/Work/Acads/FYP/Dataset/dashboard/containment.csv')
    containment_df = containment_df.loc[:, ~containment_df.columns.str.contains("^Unnamed")]
    containment_df = containment_df.reset_index(drop=True)
    containment_df = containment_df.sort_values('containment_status',ascending=False)

    return incident_log, containment_df

def load_models():
    lgbm_reg = joblib.load('lightgbm_regressor.pkl')
    lgbm_clf = joblib.load('lightgbm_classifier.pkl')
    enc_reg = joblib.load('encoder_reg.pkl')
    scaler_reg = joblib.load('scaler_reg.pkl')
    enc_clf = joblib.load('encoder_clf.pkl')
    scaler_clf = joblib.load('scaler_clf.pkl')
    
    return lgbm_reg, lgbm_clf, enc_reg, scaler_reg, enc_clf, scaler_clf

def convert_to_hours_and_minutes(units):
    # Convert 8-hour units to hours and minutes
    total_hours = units * 8
    hours = int(total_hours)
    minutes = int((total_hours - hours) * 60)

    return total_hours, hours, minutes

def predict_incident(new_incident: dict, model, encoder, scaler):
    new_df = pd.DataFrame([new_incident])

    bool_fields = ['active', 'made_sla', 'knowledge', 'u_priority_confirmation']
    for field in bool_fields:
        if field in new_df.columns:
            val = new_df[field].iloc[0]
            # Convert to numpy boolean to match training data
            if isinstance(val, bool) and not isinstance(val, np.bool_):
                # Convert Python bool to numpy bool
                new_df[field] = np.True_ if val else np.False_
            elif isinstance(val, str):
                # Convert string to numpy bool
                if val.upper() in ['TRUE', '1', 'YES', 'T']:
                    new_df[field] = np.True_
                else:
                    new_df[field] = np.False_
            elif isinstance(val, (int, float)):
                # Convert numeric to numpy bool
                new_df[field] = np.True_ if val else np.False_
            # If it's already numpy bool, leave it as is

    if model == lgbm_reg:
        categorical_features = [
            'incident_state', 'active', 'made_sla', 'contact_type', 'location',
            'category', 'subcategory', 'knowledge', 'u_priority_confirmation', 
            'notify', 'day_of_week', 'month', 'day'
        ]
        numeric_features = [
            'reassignment_count', 'reopen_count', 'sys_mod_count',
            'impact_freq', 'urgency_freq', 'priority_freq'  # Frequency encoded features
        ]

        # Add frequency encoding for regression model
        for col in ['impact', 'urgency', 'priority']:
            if col in new_df.columns:
                # Create frequency encoded versions
                freq_map = {
                    '1': 100,  # High impact - rare
                    '2': 500,  # Medium impact - common  
                    '3': 50    # Low impact - rare
                }
                new_df[col + '_freq'] = new_df[col].map(freq_map).fillna(100)
    
    else:
        categorical_features = [
        'incident_state', 'active', 'made_sla', 'contact_type', 'location',
        'category', 'subcategory', 'impact', 'urgency', 'priority', 'knowledge',
        'u_priority_confirmation', 'notify', 'day_of_week', 'month', 'day'
        ]
        numeric_features = ['reassignment_count', 'reopen_count', 'sys_mod_count']

    # Ensure all columns exist with appropriate default values
    for col in categorical_features:
        if col not in new_df.columns:
            # For boolean fields, use numpy False as default
            if col in bool_fields:
                new_df[col] = np.False_
            else:
                new_df[col] = 'Unknown'
    
    for col in numeric_features:
        if col not in new_df.columns:
            new_df[col] = 0

    # --- Handle unseen categories ---
    for i, col in enumerate(categorical_features):
        known_values = encoder.categories_[i]
        # new_df[col] = new_df[col].apply(lambda x: x if x in known_values else known_values[0])
        # Special handling for numpy booleans
        if col in bool_fields and new_df[col].iloc[0] in [np.True_, np.False_]:
            # Numpy booleans should match the encoder categories
            if new_df[col].iloc[0] not in known_values:
                # Fall back to first known category
                new_df[col] = known_values[0] if len(known_values) > 0 else np.False_
        else:
            # For other types, use the existing logic
            new_df[col] = new_df[col].apply(lambda x: x if x in known_values else known_values[0])
    
    # --- Encode and scale ---
    try:
        cat_transformed = encoder.transform(new_df[categorical_features])
        num_scaled = scaler.transform(new_df[numeric_features])
        new_sparse = hstack([cat_transformed, csr_matrix(num_scaled)])
    
    except Exception as e:
        print(f"Error during encoding/scaling: {e}")
        print(f"Available columns: {new_df.columns.tolist()}")
        print(f"Expected categorical: {categorical_features}")
        print(f"Expected numeric: {numeric_features}")
        return None

    if model == lgbm_reg:

        # --- Predict resolution time and days --
        predicted_resolution_time = model.predict(new_sparse)[0]
        
        if predicted_resolution_time < 0:
            print(f"Negative prediction: {predicted_resolution_time:.4f}")
            # Use absolute value or minimum
            predicted_resolution_time = abs(predicted_resolution_time)
            # Ensure minimum reasonable time
            predicted_resolution_time = max(predicted_resolution_time, 0.125)
            print(f"Corrected to: {predicted_resolution_time:.4f}")
        
        predicted_days = predicted_resolution_time * (8 / 24)  # convert 8-hour units → days

        total_hours, hours, minutes = convert_to_hours_and_minutes(predicted_resolution_time)

        # --- Output results ---
        print(f"\nPredicted Resolution Time:")
        print(f"{predicted_resolution_time:.2f} (in 8-hour units)")
        print(f"{total_hours:.2f} hours")
        print(f"Approximately {hours}h {minutes}m")
        print(f"{predicted_days:.2f} days estimated to close incident")

        return predicted_resolution_time, predicted_days
    
    else:
        bin_edges = joblib.load('bin_edges.pkl')

        # -- Predict class and probability --
        predicted_class = model.predict(new_sparse)[0]
        predicted_proba = model.predict_proba(new_sparse)[0]
        
        # Get class confidence and handle low-confidence predictions
        max_proba = np.max(predicted_proba)
        confidence_threshold = 0.6

        # Decode the time range from bin edges
        if bin_edges is not None and predicted_class < len(bin_edges) - 1:
            lower = bin_edges[predicted_class]
            upper = bin_edges[predicted_class + 1]
            decoded_range = f"{lower:.1f}–{upper:.1f} days"
        else:
            decoded_range = "Unknown range"

        print(f"Predicted Resolution Range: {decoded_range}")
        print(f"Predicted Class: {predicted_class}")

        if max_proba < confidence_threshold:
            print(f"LOW CONFIDENCE WARNING: Maximum probability ({max_proba:.2f}) below threshold ({confidence_threshold})")
            # Show top 2 predictions for low confidence cases
            top2_indices = np.argsort(predicted_proba)[-2:][::-1]
            print("Top 2 predictions:")
            for idx in top2_indices:
                if idx < len(bin_edges) - 1:
                    lower = bin_edges[idx]
                    upper = bin_edges[idx + 1]
                    print(f"  Class {idx}: {lower:.1f}–{upper:.1f} days (prob: {predicted_proba[idx]:.3f})")

        print("\nClass Probabilities:")
        for i, p in enumerate(predicted_proba):
            if i < len(bin_edges) - 1:
                lower = bin_edges[i]
                upper = bin_edges[i + 1]
                range_str = f"{lower:.1f}–{upper:.1f} days"
            else:
                range_str = "Unknown range"
            print(f"  {range_str}: {p:.3f}")
        
        return predicted_class, predicted_proba, decoded_range

def predict_and_fill_closed_at(df, model, encoder, scaler):
    active_incidents = df[df['closed_at'].isna()].copy()
    
    if len(active_incidents) == 0:
        # Initialize the predicted column if it doesn't exist
        if 'closed_at_predicted' not in df.columns:
            df['closed_at_predicted'] = pd.NaT
            df['is_predicted'] = False
        return df
    
    print(f"📊 Predicting closure dates for {len(active_incidents)} active incidents...")
    
    predicted_closed_dates = []
    prediction_errors = 0
    
    for i, (idx, incident) in enumerate(active_incidents.iterrows()):
        if i % 10 == 0:  # Progress update every 10 incidents
            print(f"📊 Processing {i+1}/{len(active_incidents)} incidents...")
        
        try:
            incident_dict = incident.to_dict()
            
            # Enhanced data type handling
            bool_fields = ['active', 'made_sla', 'knowledge', 'u_priority_confirmation']
            for field in bool_fields:
                if field in incident_dict:
                    val = incident_dict[field]
                    if pd.isna(val):
                        incident_dict[field] = False
                    elif isinstance(val, (bool, np.bool_)):
                        incident_dict[field] = bool(val)
                    elif isinstance(val, (int, float)):
                        incident_dict[field] = bool(val)
                    elif isinstance(val, str):
                        incident_dict[field] = val.lower() in ['true', 'yes', '1', 't']
            
            # Predict with error handling
            prediction_result = predict_incident(incident_dict, model, encoder, scaler)
            
            if prediction_result is not None:
                if isinstance(prediction_result, tuple):
                    predicted_resolution_time = prediction_result[0]
                else:
                    predicted_resolution_time = prediction_result
                
                # Calculate closed_at with business hours consideration
                resolution_time_hours = predicted_resolution_time * 8
                opened_at = incident['opened_at']
                
                # Ensure opened_at is timezone-naive for calculations
                if hasattr(opened_at, 'tz') and opened_at.tz is not None:
                    opened_at = opened_at.tz_localize(None)
                
                predicted_closed_at = opened_at + pd.Timedelta(hours=resolution_time_hours)
                predicted_closed_dates.append(predicted_closed_at)
            else:
                # Fallback: use average resolution time or 1 day
                opened_at = incident['opened_at']
                if hasattr(opened_at, 'tz') and opened_at.tz is not None:
                    opened_at = opened_at.tz_localize(None)
                predicted_closed_at = opened_at + pd.Timedelta(days=1)
                predicted_closed_dates.append(predicted_closed_at)
                prediction_errors += 1
                
        except Exception as e:
            # Fallback for any prediction errors
            print(f"Prediction error for incident {idx}: {e}")
            opened_at = incident['opened_at']
            if hasattr(opened_at, 'tz') and opened_at.tz is not None:
                opened_at = opened_at.tz_localize(None)
            predicted_closed_at = opened_at + pd.Timedelta(days=1)
            predicted_closed_dates.append(predicted_closed_at)
            prediction_errors += 1
    
    # Update DataFrame
    df_predicted = df.copy()
    active_indices = active_incidents.index
    
    # Initialize columns if they don't exist
    if 'closed_at_predicted' not in df_predicted.columns:
        df_predicted['closed_at_predicted'] = pd.NaT
    if 'is_predicted' not in df_predicted.columns:
        df_predicted['is_predicted'] = False
    
    df_predicted.loc[active_indices, 'closed_at_predicted'] = predicted_closed_dates
    df_predicted.loc[active_indices, 'is_predicted'] = True
    
    if prediction_errors > 0:
        print(f"Generated predictions with {prediction_errors} fallbacks")
    else:
        print(f"Generated predictions for {len(active_incidents)} active incidents")
    
    return df_predicted

def calculate_recovery_time(incident_data):
    try:
        opened_at = incident_data.get('opened_at')
        closed_at = incident_data.get('closed_at')
        closed_at_predicted = incident_data.get('closed_at_predicted')
        
        # Case 1: Incident is closed - use actual closed_at
        if pd.notna(closed_at) and pd.notna(opened_at):
            recovery_duration = closed_at - opened_at
            total_hours = recovery_duration.total_seconds() / 3600
            total_days = total_hours / 24
            
            if total_hours < 1:
                minutes = total_hours * 60
                display = f"{minutes:.1f} minutes"
            elif total_hours < 24:
                display = f"{total_hours:.1f} hours"
            else:
                display = f"{total_days:.1f} days"
            
            return {
                'display': display,
                'total_hours': total_hours,
                'total_days': total_days,
                'is_actual': True,
                'status': 'completed'
            }
        
        # Case 2: Incident is active but has prediction
        elif pd.notna(closed_at_predicted) and pd.notna(opened_at):
            recovery_duration = closed_at_predicted - opened_at
            total_hours = recovery_duration.total_seconds() / 3600
            total_days = total_hours / 24
            
            if total_hours < 1:
                minutes = total_hours * 60
                display = f"{minutes:.1f} minutes (estimated)"
            elif total_hours < 24:
                display = f"{total_hours:.1f} hours (estimated)"
            else:
                display = f"{total_days:.1f} days (estimated)"
            
            return {
                'display': display,
                'total_hours': total_hours,
                'total_days': total_days,
                'is_actual': False,
                'status': 'predicted'
            }
        
        # Case 3: Incident is still active, no prediction
        else:
            return {
                'display': "Still in progress",
                'total_hours': None,
                'total_days': None,
                'is_actual': False,
                'status': 'active'
            }
            
    except Exception as e:
        print(f"Error calculating recovery time: {e}")
        return {
            'display': "Unable to calculate",
            'total_hours': None,
            'total_days': None,
            'is_actual': False,
            'status': 'error'
        }

def generate_report_content(report_data, preview=False):
    content = []
    
    # Header
    content.append("INCIDENT REPORT")
    content.append("=" * 50)
    content.append(f"Incident Name: {report_data['incident_number']}")
    content.append(f"Report Date: {report_data['report_date']}")
    content.append(f"Prepared By: {report_data['prepared_by']}")
    content.append(f"Report Version: {report_data['report_version']}")
    content.append("")
    
    # Identification
    content.append("Identification")
    content.append("-" * 20)
    content.append(f"Detected at: {report_data['detected_at']}")
    content.append(f"Detection Method: {report_data['detection_method']}")
    content.append(f"Category: {report_data['incident_data'].get('category', 'N/A')}")
    content.append(f"Priority: {report_data['incident_data'].get('priority', 'N/A')}")
    content.append("")
    
    # Containment
    content.append("Containment")
    content.append("-" * 20)
    
    if (report_data['containment_data'] is not None and 
        not report_data['containment_data'].empty):
        content.append("Contained Devices:")
        for _, device in report_data['containment_data'].iterrows():
            status = device.get('containment_status','Unknown')
            content.append(f"  - Device {device.get('device_no', 'N/A')} ({device.get('category', 'N/A')}): {status}")
    else:
        content.append("None")
    
    content.append(f"Containment Actions: {report_data['containment_notes']}")
    content.append("")
    
    # Eradication & Recovery
    content.append("Eradication & Recovery")
    content.append("-" * 20)
    content.append(f"Investigation notes: {report_data['investigation_notes']}")
    
    # Recovery Time (auto-calculated)
    recovery_status = report_data['recovery_time_info']['status']
    if recovery_status == 'completed':
        content.append(f"Recovery Time: {report_data['recovery_time_info']['display']} (Actual)")
    elif recovery_status == 'predicted':
        content.append(f"Recovery Time: {report_data['recovery_time_info']['display']}")
    else:
        content.append("Recovery Time: Still in progress")
    
    if report_data.get('recovery_details'):
        content.append(f"Recovery Process: {report_data['recovery_details']}")
    
    # Include detailed timeline if requested
    if report_data.get('include_timeline', False):
        content.append("")
        content.append("Timeline Details:")
                # Format opened_at
        opened_at = report_data['incident_data'].get('opened_at')
        if pd.notna(opened_at):
            if isinstance(opened_at, str):
                formatted_opened = opened_at
            else:
                formatted_opened = opened_at.strftime('%Y-%m-%d %H:%M:%S')
            content.append(f"  - Opened: {formatted_opened}")
        
        # Format closed_at if it exists
        closed_at = report_data['incident_data'].get('closed_at')
        if pd.notna(closed_at):
            if isinstance(closed_at, str):
                formatted_closed = closed_at
            else:
                formatted_closed = closed_at.strftime('%Y-%m-%d %H:%M:%S')
            content.append(f"  - Closed: {formatted_closed}")
        
        # Format predicted closure if it exists
        closed_at_predicted = report_data['incident_data'].get('closed_at_predicted')
        if pd.notna(closed_at_predicted):
            if isinstance(closed_at_predicted, str):
                formatted_predicted = closed_at_predicted
            else:
                formatted_predicted = closed_at_predicted.strftime('%Y-%m-%d %H:%M:%S')
            content.append(f"  - Predicted Closure: {formatted_predicted}")

    content.append("")
    
    # Lessons Learned
    content.append("Lessons Learned")
    content.append("-" * 20)
    content.append(f"Post Incident Notes: {report_data['post_incident_notes']}")
    content.append("")
    content.append("Recommended Improvements:")
    content.append(f"{report_data['improvement_actions']}")
    content.append("")
    
    # Footer
    content.append("=" * 50)
    content.append("End of Report")
    
    return "\n".join(content)

def generate_markdown_report(report_data):
    content = []
    
    # Header
    content.append(f"# Incident Report: {report_data['incident_number']}")
    content.append("")
    content.append(f"**Report Date:** {report_data['report_date']}  ")
    content.append(f"**Prepared By:** {report_data['prepared_by']}  ")
    content.append(f"**Version:** {report_data['report_version']}  ")
    content.append("")
    
    # Identification
    content.append("## Identification")
    content.append("")
    content.append(f"**Detected at:** {report_data['detected_at']}  ")
    content.append(f"**Detection Method:** {report_data['detection_method']}  ")
    content.append("")
    
    # Containment
    content.append("## Containment")
    content.append("")
    if (report_data['containment_data'] is not None and 
        not report_data['containment_data'].empty):
        content.append("**Contained Devices:**")
        content.append("")
        for _, device in report_data['containment_data'].iterrows():
            status = device.get('containment_status', 'Unknown')
            content.append(f"- {device.get('device_no', 'N/A')} ({device.get('category', 'N/A')}) - {status}")
        content.append("")
    else:
        content.append("-")
        content.append("")
    content.append(f"**Containment Actions:** {report_data['containment_notes']}  ")
    content.append("")
    
    # Eradication & Recovery
    content.append("## Eradication & Recovery")
    content.append("")
    content.append(f"**Investigation notes:** {report_data['investigation_notes']}  ")
    content.append("")
    
    # Recovery Time with status indicator
    recovery_status = report_data['recovery_time_info']['status']
    if recovery_status == 'completed':
        content.append(f"**Recovery Time:** {report_data['recovery_time_info']['display']}")
    elif recovery_status == 'predicted':
        content.append(f"**Recovery Time:** {report_data['recovery_time_info']['display']}")
    else:
        content.append("**Recovery Time:**Still in progress  ")
    
    if report_data.get('recovery_details'):
        content.append("")
        content.append(f"**Recovery Process:** {report_data['recovery_details']}  ")
    
    # Timeline details
    if report_data.get('include_timeline', False):
        content.append("")
        content.append("**Timeline Details:**")
        content.append("")

        # Format opened_at
        opened_at = report_data['incident_data'].get('opened_at')
        if pd.notna(opened_at):
            if isinstance(opened_at, str):
                formatted_opened = opened_at
            else:
                formatted_opened = opened_at.strftime('%Y-%m-%d %H:%M:%S')
            content.append(f"- **Opened:** {formatted_opened}")
        
        # Format closed_at if it exists
        closed_at = report_data['incident_data'].get('closed_at')
        if pd.notna(closed_at):
            if isinstance(closed_at, str):
                formatted_closed = closed_at
            else:
                formatted_closed = closed_at.strftime('%Y-%m-%d %H:%M:%S')
            content.append(f"- **Closed:** {formatted_closed}")
        
        # Format predicted closure if it exists
        closed_at_predicted = report_data['incident_data'].get('closed_at_predicted')
        if pd.notna(closed_at_predicted):
            if isinstance(closed_at_predicted, str):
                formatted_predicted = closed_at_predicted
            else:
                formatted_predicted = closed_at_predicted.strftime('%Y-%m-%d %H:%M:%S')
            content.append(f"- **Predicted Closure:** {formatted_predicted}")

    content.append("")
    
    # Lessons Learned
    content.append("## Lessons Learned")
    content.append("")
    content.append(f"**Post Incident Notes:** {report_data['post_incident_notes']}  ")
    content.append("")
    content.append("### Recommended Actions")
    content.append("")
    content.append(f"{report_data['improvement_actions']}")
    
    return "\n".join(content)

def preview_report():
    if 'report_data' not in st.session_state or not st.session_state.report_data:
        st.error("No report data available. Please fill in the report form first.")
        return
    
    report_data = st.session_state.report_data
    
    st.subheader("📄 Report Preview")
    st.markdown("---")
    
    # Generate preview content
    preview_content = generate_report_content(report_data, preview=True)
    
    # Display preview in expandable section
    with st.expander("Click to view report preview", expanded=True):
        st.text(preview_content)
    
    # Show download button for preview
    st.download_button(
        label="📥 Download Preview as TXT",
        data=preview_content,
        file_name=f"incident_report_{report_data['incident_number']}_preview.txt",
        mime="text/plain",
        width='stretch'
    )

def export_incident_report():
    if 'report_data' not in st.session_state or not st.session_state.report_data:
        st.error("No report data available. Please fill in the report form first.")
        return
    
    report_data = st.session_state.report_data
    export_format = report_data.get('export_format', 'TXT')
    incident_number = report_data.get('incident_number', 'UNKNOWN')
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == "TXT":
            report_content = generate_report_content(report_data)
            st.download_button(
                label="📥 Download TXT Report",
                data=report_content,
                file_name=f"incident_report_{incident_number}_{timestamp}.txt",
                mime="text/plain",
                width='stretch',
                key="download_txt"
            )
            
        elif export_format == "Markdown":
            report_content = generate_markdown_report(report_data)
            st.download_button(
                label="📥 Download Markdown Report",
                data=report_content,
                file_name=f"incident_report_{incident_number}_{timestamp}.md",
                mime="text/markdown",
                width='stretch',
                key="download_md"
            )
            
        elif export_format == "HTML":
            report_content = generate_html_report(report_data)
            st.download_button(
                label="📥 Download HTML Report",
                data=report_content,
                file_name=f"incident_report_{incident_number}_{timestamp}.html",
                mime="text/html",
                width='stretch',
                key="download_html"
            )
            
        elif export_format == "PDF":
            # For PDF, we'll generate HTML first and suggest conversion
            html_content = generate_html_report(report_data)
            st.warning("""
            **PDF Export Note:** 
            For full PDF export functionality, consider using libraries like `weasyprint` or `pdfkit`. 
            For now, you can:
            1. Download the HTML version and convert it to PDF using your browser
            2. Copy the text version and paste into a document editor
            """)
            
            # Offer HTML as alternative
            st.download_button(
                label="📥 Download HTML (Convert to PDF Manually)",
                data=html_content,
                file_name=f"incident_report_{incident_number}_{timestamp}.html",
                mime="text/html",
                width='stretch',
                key="download_pdf_fallback"
            )
        
        st.success(f"Report for {incident_number} is ready for download!")
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.info("Try generating a preview first to check for any issues.")

# Also update the generate_html_report function to be more robust:
def generate_html_report(report_data):
    try:
        content = []
        
        content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Incident Report</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 40px; 
                    line-height: 1.6;
                    color: #333;
                }
                h1 { 
                    color: #2c3e50; 
                    border-bottom: 2px solid #3498db; 
                    padding-bottom: 10px;
                }
                h2 { 
                    color: #34495e; 
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                }
                h3 {
                    color: #2c3e50;
                    margin-top: 20px;
                }
                .header { 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 5px;
                    border-left: 4px solid #3498db;
                    margin-bottom: 20px;
                }
                .section { 
                    margin: 25px 0; 
                    padding: 15px;
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .footer { 
                    margin-top: 40px; 
                    font-style: italic; 
                    color: #7f8c8d;
                    text-align: center;
                    border-top: 1px solid #ecf0f1;
                    padding-top: 20px;
                }
                .timeline {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .status-contained { color: #cd3849; font-weight: bold; }
                .status-not-contained { color: #dc793c; font-weight: bold; }
                .status-released { color: #3bb50b; font-weight: bold; }
                .status-unknown { color: #7f8c8d; font-weight: bold; }
            </style>
        </head>
        <body>
        """)
        
        # Header
        content.append(f'<h1>Incident Report: {report_data.get("incident_number", "N/A")}</h1>')
        content.append('<div class="header">')
        content.append(f'<p><strong>Report Date:</strong> {report_data.get("report_date", "N/A")}</p>')
        content.append(f'<p><strong>Prepared By:</strong> {report_data.get("prepared_by", "N/A")}</p>')
        content.append(f'<p><strong>Version:</strong> {report_data.get("report_version", "N/A")}</p>')
        content.append('</div>')
        
        # Identification
        content.append('<div class="section">')
        content.append('<h2>Identification</h2>')
        content.append(f'<p><strong>Detected at:</strong> {report_data.get("detected_at", "N/A")}</p>')
        content.append(f'<p><strong>Detection Method:</strong> {report_data.get("detection_method", "N/A")}</p>')
        content.append(f'<p><strong>Category:</strong> {report_data.get("incident_data", {}).get("category", "N/A")}</p>')
        content.append(f'<p><strong>Priority:</strong> {report_data.get("incident_data", {}).get("priority", "N/A")}</p>')
        content.append('</div>')
        
        # Containment
        content.append('<div class="section">')
        content.append('<h2>Containment</h2>')
        
        containment_data = report_data.get('containment_data')
        if (containment_data is not None and not containment_data.empty):
            content.append('<p><strong>Contained Devices:</strong></p>')
            content.append('<ul>')
            for _, device in containment_data.iterrows():
                status = device.get('containment_status', 'Unknown')
                # Map status to CSS class
                status_class_map = {
                    'Contained': 'status-contained',
                    'Not Contained': 'status-not-contained',
                    'Released': 'status-released'
                }
                status_class = status_class_map.get(status, 'status-unknown')
                content.append(f'<li>Device {device.get("device_no", "N/A")} ({device.get("category", "N/A")}): <span class="{status_class}">{status}</span></li>')
            content.append('</ul>')
        else:
            content.append('<p>None</p>')
        
        content.append(f'<p><strong>Containment Actions:</strong> {report_data.get("containment_notes", "N/A")}</p>')
        content.append('</div>')
        
        # Eradication & Recovery
        content.append('<div class="section">')
        content.append('<h2>Eradication & Recovery</h2>')
        content.append(f'<p><strong>Investigation notes:</strong> {report_data.get("investigation_notes", "N/A")}</p>')
        
        # Recovery Time with status
        recovery_info = report_data.get('recovery_time_info', {})
        recovery_status = recovery_info.get('status', 'unknown')
        status_class = f"status-{recovery_status}"
        
        if recovery_status == 'completed':
            content.append(f'<p><strong>Recovery Time:</strong> <span class="{status_class}">{recovery_info.get("display", "N/A")} (Actual)</span></p>')
        elif recovery_status == 'predicted':
            content.append(f'<p><strong>Recovery Time:</strong> <span class="{status_class}">{recovery_info.get("display", "N/A")}</span></p>')
        else:
            content.append(f'<p><strong>Recovery Time:</strong> <span class="{status_class}">Still in progress</span></p>')
        
        if report_data.get('recovery_details'):
            content.append(f'<p><strong>Recovery Process:</strong> {report_data.get("recovery_details", "")}</p>')
        
        # Timeline details
        if report_data.get('include_timeline', False):
            content.append('<div class="timeline">')
            content.append('<strong>Timeline Details:</strong>')
            content.append('<ul>')
            
            # Format opened_at
            opened_at = report_data['incident_data'].get('opened_at')
            if pd.notna(opened_at):
                if isinstance(opened_at, str):
                    formatted_opened = opened_at
                else:
                    formatted_opened = opened_at.strftime('%Y-%m-%d %H:%M:%S')
                content.append(f'<li><strong>Opened:</strong> {formatted_opened}</li>')
            
            # Format closed_at if it exists
            closed_at = report_data['incident_data'].get('closed_at')
            if pd.notna(closed_at):
                if isinstance(closed_at, str):
                    formatted_closed = closed_at
                else:
                    formatted_closed = closed_at.strftime('%Y-%m-%d %H:%M:%S')
                content.append(f'<li><strong>Closed:</strong> {formatted_closed}</li>')
            
            # Format predicted closure if it exists
            closed_at_predicted = report_data['incident_data'].get('closed_at_predicted')
            if pd.notna(closed_at_predicted):
                if isinstance(closed_at_predicted, str):
                    formatted_predicted = closed_at_predicted
                else:
                    formatted_predicted = closed_at_predicted.strftime('%Y-%m-%d %H:%M:%S')
                content.append(f'<li><strong>Predicted Closure:</strong> {formatted_predicted}</li>')
            
            content.append('</ul>')
            content.append('</div>')
        
        content.append('</div>')
        
        # Lessons Learned
        content.append('<div class="section">')
        content.append('<h2>Lessons Learned</h2>')
        content.append(f'<p><strong>Post Incident Notes:</strong> {report_data.get("post_incident_notes", "N/A")}</p>')
        content.append('<h3>Recommended Actions</h3>')
        content.append(f'<p>{report_data.get("improvement_actions", "N/A")}</p>')
        content.append('</div>')
        
        # Footer
        content.append('<div class="footer">')
        content.append(f'<p>Report generated on {report_data.get("report_date", "N/A")} by {report_data.get("prepared_by", "N/A")}</p>')
        content.append('<p>End of Report</p>')
        content.append('</div>')
        
        content.append('</body></html>')
        
        return "".join(content)
        
    except Exception as e:
        # Fallback to simple HTML if there's an error
        return f"""
        <html>
        <body>
            <h1>Error Generating Report</h1>
            <p>There was an error generating the HTML report: {str(e)}</p>
            <p>Please try exporting in TXT or Markdown format instead.</p>
        </body>
        </html>
        """
# Page config
st.set_page_config(
    page_title="Cybersecurity  Incident Response Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state for data loading
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_with_predictions' not in st.session_state:
    st.session_state.df_with_predictions = None
if 'containment_df' not in st.session_state:
    st.session_state.containment_df = None

if not st.session_state.data_loaded:
    with st.spinner("🔄 Loading data and generating predictions..."):
        # Load data
        df, containment_df = load_data()

        # Load models
        lgbm_reg, lgbm_clf, enc_reg, scaler_reg, enc_clf, scaler_clf = load_models()
        
        # Generate predictions for active incidents
        df_with_predictions = predict_and_fill_closed_at(df, lgbm_reg, enc_reg, scaler_reg)
        
        # Store in session state
        st.session_state.df_with_predictions = df_with_predictions
        st.session_state.data_loaded = True
        st.session_state.containment_df = containment_df
        st.session_state.lgbm_reg = lgbm_reg
        st.session_state.lgbm_clf = lgbm_clf
        st.session_state.enc_reg = enc_reg
        st.session_state.scaler_reg = scaler_reg
        st.session_state.enc_clf = enc_clf
        st.session_state.scaler_clf = scaler_clf
        
        st.success("✅ Data loaded and predictions generated!")

# Use the data from session state
df = st.session_state.df_with_predictions
containment_df = st.session_state.containment_df
lgbm_reg = st.session_state.lgbm_reg
lgbm_clf = st.session_state.lgbm_clf
enc_reg = st.session_state.enc_reg
scaler_reg = st.session_state.scaler_reg
enc_clf = st.session_state.enc_clf
scaler_clf = st.session_state.scaler_clf

# -- Streamlit UI

# Colour palatte (WCAG Accessibility)
# Base color = #2972FD
colors = {
    'c1': '#17d6b5',
    'c2': '#19addc',
    'c3': '#2972fd',
    'c4': '#5455ff',
    'c5': '#9d56ff'
}
color_scale = [colors['c1'], colors['c2'],colors['c3'],colors['c4'],colors['c5']]
cat_colors = list(colors.values())

sev_colors = {
    'c1': '#cd3849',
    'c2': '#dc793c',
    'c3': '#0d24cd',
    'c4': '#3bb50b',
}

priority_colors = {
    1: sev_colors['c1'], 
    2: sev_colors['c2'],
    3: sev_colors['c3'], 
    4: sev_colors['c4'],
}

st.markdown("""
    <h1 style='text-align: center; color: white;'>
        Cybersecurity Incident Response Dashboard
    </h1>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Navigate to:",
    [
        "Summary",
        "Preparation",
        "Identification",
        "Containment",
        "Eradication & Recovery",
        "Lessons Learned"
    ]
)

# Summary Page
if page == "Summary":
    st.markdown("<h2 style='text-align: center;'>Summary</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    df['month_name'] = df['opened_at'].dt.strftime('%B')

    # Ensure month order for consistent sorting
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

    # Sidebar filters
    year_filter = st.sidebar.multiselect(
        "Select Year", sorted(df['year'].dropna().unique()), default=sorted(df['year'].dropna().unique())
    )
    month_filter = st.sidebar.multiselect(
        "Select Month",
        [m for m in month_order if m in df['month_name'].unique()],
        default=[m for m in month_order if m in df['month_name'].unique()]
    )

    if 'priority' in df.columns:
        priority_filter = st.sidebar.multiselect(
            "Priority Level",
            options=sorted(df['priority'].dropna().unique()),
            default=sorted(df['priority'].dropna().unique())
        )
    else:
        priority_filter = []

    # Apply filters
    filtered_df = df[df['year'].isin(year_filter) & df['month_name'].isin(month_filter)]

    if priority_filter and 'priority' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['priority'].isin(priority_filter)]
    
    # KPIs with trends of previous statistics
    total_tickets = filtered_df['number'].nunique() if 'number' in filtered_df.columns else len(filtered_df)

    # Calculate Mean Time to Respond (MTTR)
    if 'opened_at' in filtered_df.columns and 'closed_at' in filtered_df.columns:
        filtered_df['response_time_hours'] = (filtered_df['closed_at'] - filtered_df['opened_at']).dt.total_seconds() / 3600

        # Exclude NaN values
        valid_response_times = filtered_df['response_time_hours'].dropna()
        if len(valid_response_times) > 0:
            mttr = valid_response_times.mean()

            # Format based on the magnitude
            if mttr < 1:
                mttr_display = f"{mttr * 60:.1f} min"
                mttr_value = mttr * 60  # in minutes for comparison
            elif mttr < 24:
                mttr_display = f"{mttr:.1f} hours"
                mttr_value = mttr
            else:
                mttr_display = f"{mttr / 24:.1f} days"
                mttr_value = mttr / 24  # in days for comparison
        else:
            mttr_display = "No data"
            mttr_value = 0
    else:
        mttr_display = "No time data"
        mttr_value = 0

    filtered_df['made_sla'] = filtered_df['made_sla'].map({'TRUE':1,'FALSE':0,True:1,False:0})
    if 'made_sla' in filtered_df.columns:
        total_sla_breached = (filtered_df['made_sla'] == 0).sum()
    else:
        total_sla_breached = 0

    prev_tickets = total_tickets * 0.9  # Example calculation
    prev_mttr = mttr_value
    prev_sla = total_sla_breached

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tickets", total_tickets, delta=f"{total_tickets - prev_tickets:+.0f}")
    col2.metric("Mean Time to Respond", mttr_display, delta=f"{mttr_value - prev_mttr:+.0f}")
    col3.metric("Total Breached SLA", total_sla_breached, delta=f"{total_sla_breached - prev_sla:+.0f}")

    if 'opened_at' in filtered_df.columns:
    # Group by year and month name, count incidents
        month_summary = (
            filtered_df.groupby(['year', 'month_name'])
            .size()
            .reset_index(name='count')
        )

        # Keep only months with incidents > 0
        month_summary = month_summary[month_summary['count'] > 0]
        month_summary['month_name'] = pd.Categorical(
            month_summary['month_name'], categories=month_order, ordered=True
        )
        month_summary = month_summary.sort_values(['year', 'month_name'])

        # Create color mapping for years using palette
        unique_years = sorted(month_summary['year'].unique())
        year_colors = {}
        for i, year in enumerate(unique_years):
            color_key = f'c{(i % 5) + 1}'
            year_colors[year] = colors[color_key]

        # Interactive line chart
        fig = px.line(
            month_summary,
            x='month_name',
            y='count',
            color='year',
            markers=True,
            title='Number of Incidents Reported by Month',
            color_discrete_map=year_colors,
            hover_data={'count': True, 'year': True}
        )

        fig.update_traces(
            mode='lines+markers',
            marker=dict(size=8, line=dict(width=2, color='white')),
            line=dict(width=3),
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Year: %{customdata[1]}<br>" +
                "Incidents: <b>%{y}</b>" +
                "<extra></extra>"
            ),
            customdata=month_summary[['count', 'year']].values
        )

        fig.update_layout(
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            xaxis=dict(
                title='Month',
                gridcolor='lightgray',
                title_font=dict(color='white')
            ),
            yaxis=dict(
                title='Incident Count',
                gridcolor='lightgray',
                title_font=dict(color='white')
            ),
            hovermode='x unified',
            legend=dict(
                title='Year',
                font=dict(color='white'),
            )
        )

        st.plotly_chart(fig, width='stretch')

    # Pie Charts
    colA, colB = st.columns(2)
    if 'category' in filtered_df.columns:
        cat_df = filtered_df['category'].value_counts().reset_index()
        cat_df.columns = ['Category', 'Count']

        pie1 = px.pie(
            cat_df, 
            names='Category', 
            values='Count', 
            title='Incidents by Category',
            color='Category',
            color_discrete_sequence=cat_colors,
            hover_data=['Count']
        )

        pie1.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont=dict(color='white',size=16),
            marker=dict(line=dict(color='white', width=1)),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        )

        pie1.update_layout(
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            legend=dict(
                font=dict(color='white', size=10),
                bordercolor='lightgray',
                borderwidth=1
            ),
            uniformtext_minsize=12,
            uniformtext_mode='hide'
        )


        colA.plotly_chart(pie1, width='stretch')

    if 'priority' in filtered_df.columns:
        prio_df = filtered_df['priority'].value_counts().reset_index()
        prio_df.columns = ['Priority', 'Count']

        pie2 = px.pie(
            prio_df, 
            names='Priority', 
            values='Count', 
            title='Incidents by Priority Level',
            color='Priority',
            color_discrete_map=priority_colors,
            hover_data=['Count'],
            category_orders={'Priority': [1, 2, 3, 4]}
        )

        pie2.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont=dict(color='white', size=16),
            marker=dict(line=dict(color='white', width=1)),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        )

        pie2.update_layout(
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            legend=dict(
                font=dict(color='white', size=10),
                bordercolor='lightgray',
                borderwidth=1,
                traceorder='normal'
            ),
            uniformtext_minsize=12,
            uniformtext_mode='hide'
        )

        colB.plotly_chart(pie2, width='stretch')

    # Incident table with search
    st.markdown("### Incident Records")

    display_df = filtered_df.copy()
    
    # Show prediction status
    active_count = len(df[df['closed_at'].isna()])
    predicted_count = len(df[df['is_predicted'] == True]) if 'is_predicted' in df.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidents", len(df))
    col2.metric("Active Incidents", active_count)
    col3.metric("With Predictions", predicted_count)


    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""

    # Search function
    search_term = st.text_input("Search for incidents...", placeholder="Search by description, category, etc.")

    if search_term:
        mask = np.column_stack([
            display_df[col].astype(str).str.contains(search_term, case=False, na=False) 
            for col in display_df.columns
        ]).any(axis=1)
        display_df = display_df[mask]

        if len(display_df) > 0:
            st.info(f"🔍 Found {len(display_df)} incidents matching '{st.session_state.search_term}'")
        else:
            st.warning(f"No incidents found matching '{st.session_state.search_term}'")
    
    base_columns = [
        'number', 'incident_state', 'caller_id', 'reassignment_count', 'category', 'subcategory', 
        'impact', 'urgency', 'priority', 'assignment_group', 'assigned_to',
        'opened_at', 'short_description'
    ]
    columns_to_display = base_columns.copy()
    
    opened_at_index = columns_to_display.index('opened_at')

    resolution_columns = []

    # Insert closed_at_predicted first (if it exists)
    if 'closed_at_predicted' in display_df.columns:
        columns_to_display.insert(opened_at_index + 1, 'closed_at_predicted')
        # Update the index since we added a column
        opened_at_index = columns_to_display.index('opened_at')
    
    # Insert closed_at next (if it exists)
    if 'closed_at' in display_df.columns:
        columns_to_display.insert(opened_at_index + 1, 'closed_at')

    available_cols = [col for col in columns_to_display if col in display_df.columns]
    
    # Format datetime columns
    datetime_cols = ['opened_at', 'closed_at', 'closed_at_predicted']
    for col in datetime_cols:
        if col in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime('%d/%m/%Y %H:%M')
    
    styled_df = display_df[available_cols].sort_values(by='incident_state', ascending=True)

    # Apply color coding
    def style_incident_rows(row):
        styles = [''] * len(row)
        
        # Highlight active incidents
        if 'incident_state' in row.index and row['incident_state'] == 'Active':
            styles = ['background-color: #8B0000'] * len(row)
        
        return styles
    
    try:
        styled_display = styled_df.style.apply(style_incident_rows, axis=1)
        st.dataframe(styled_display, width='stretch', height=400)

    except Exception as e:
        st.dataframe(styled_df, width='stretch', height=400)
        st.warning(f"Could not apply styling: {e}")

# Preparation Page
elif page == "Preparation":
    st.markdown("<h2 style='text-align: center;'>Preparation Stage</h2>", unsafe_allow_html=True)
    st.markdown("---")

    asset_data = {
    "Asset Type": ["Servers", "Databases", "Endpoints", "Network Devices"],
    "Readiness (%)": [95, 90, 85, 88],
    "Total Assets": [45, 25, 120, 30],
    "Pending Updates": [2, 1, 10, 3]
    }
    df_assets = pd.DataFrame(asset_data)

    col1,col2 = st.columns(2)
    with col1:
        st.metric("Monitored Assets","120")
        st.metric("Pending Patches","8")
    with col2:
        st.metric("Training Completion Rate","92%")
        st.metric("Open Vulnerabilities","15")

    fig = px.bar(
        df_assets,
        x="Asset Type",
        y="Readiness (%)",
        hover_data={
            "Total Assets": True,
            "Pending Updates": True,
            "Readiness (%)": True
        },
        labels={"x": "Asset Type", "y": "Readiness (%)"},
        title="Asset Readiness by Category",
        color="Readiness (%)",
        color_continuous_scale=cat_colors,
    )

    # Custom hover template
    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{x}</b>",
            "Readiness: %{y}%",
            "Total Assets: %{customdata[0]}",
            "Pending Updates: %{customdata[1]}",
            "<extra></extra>"
        ]),
        customdata=df_assets[['Total Assets', 'Pending Updates']].values
    )

    st.plotly_chart(fig, width='stretch')

# Identification Page
elif page == "Identification":
    st.markdown("<h2 style='text-align: center;'>Identification Stage</h2>", unsafe_allow_html=True)

    # Form for new incident
    # with st.form("prediction_form"):
    #     st.subheader("Enter Incident Details")

        # incident_state = st.selectbox("Incident State", ["New", "In Progress", "Resolved"])
        # active = st.selectbox("Active", ["True", "False"])
        # made_sla = st.selectbox("Made SLA", ["true", "false"])
        # contact_type = st.selectbox("Contact Type", ["Phone", "Email", "Self-service"])
        # location = st.text_input("Location", "Headquarters")
        # category = st.selectbox("Category", ["Network", "Hardware", "Software", "Security"])
        # subcategory = st.text_input("Subcategory", "Firewall")
        # impact = st.selectbox("Impact", ["1", "2", "3"])
        # urgency = st.selectbox("Urgency", ["1", "2", "3"])
        # priority = st.selectbox("Priority", ["1", "2", "3"])
        # knowledge = st.selectbox("Knowledge", ["True", "False"])
        # u_priority_confirmation = st.selectbox("Priority Confirmed", ["True", "False"])
        # notify = st.selectbox("Notify", ["True", "False"])
        # day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        # month = st.number_input("Month", 1, 12, 1)
        # day = st.number_input("Day", 1, 31, 1)
        # reassignment_count = st.number_input("Reassignment Count", 0, 10, 1)
        # reopen_count = st.number_input("Reopen Count", 0, 10, 0)
        # sys_mod_count = st.number_input("System Modification Count", 0, 50, 5)
        # model_choice = st.radio("Choose Model", ["Regression (Exact Time)", "Classification (Time Range)"])
        # submitted = st.form_submit_button("Predict Resolution Time")

    # Form for new incident
    with st.form("prediction_form"):
        st.subheader("Enter Incident Details")

        # Date and time selection for opened_at
        col1, col2 = st.columns(2)
        with col1:
            opened_date = st.date_input("Incident Date", datetime.now().date())
        with col2:
            opened_time = st.time_input("Incident Time", datetime.now().time())
        
        # Combine date and time
        opened_at = datetime.combine(opened_date, opened_time)
        # Extract components from datetime
        day_of_week = opened_at.strftime('%A')  # Full weekday name
        month = opened_at.month
        day = opened_at.day

        incident_state = st.selectbox("Incident State", ["New", "Active", "Resolved", "Closed"])
        active = st.selectbox("Active", ["True", "False"])
        made_sla = st.selectbox("Made SLA", ["True", "False"])
        contact_type = st.selectbox("Contact Type", ["Phone", "Email", "Self-service"])
        location = st.text_input("Location", "Headquarters")
        category = st.selectbox("Category", ["Network", "Hardware", "Software", "Security"])
        subcategory = st.text_input("Subcategory", "Firewall")
        impact = st.selectbox("Impact", ["1", "2", "3"])
        urgency = st.selectbox("Urgency", ["1", "2", "3"])
        priority = st.selectbox("Priority", ["1", "2", "3", "4"])
        knowledge = st.selectbox("Knowledge", ["True", "False"])
        u_priority_confirmation = st.selectbox("Priority Confirmed", ["True", "False"])
        notify = st.selectbox("Notify", ["True", "False"])
        reassignment_count = st.number_input("Reassignment Count", 0, 10, 1)
        reopen_count = st.number_input("Reopen Count", 0, 10, 0)
        sys_mod_count = st.number_input("System Modification Count", 0, 50, 5)
        model_choice = st.radio("Choose Model", ["Regression (Exact Time)", "Classification (Time Range)"])
        
        submitted = st.form_submit_button("Predict Resolution Time")

    if submitted:
        new_incident = {
            "incident_state": incident_state,
            "active": active,
            "made_sla": made_sla,
            "contact_type": contact_type,
            "location": location,
            "category": category,
            "subcategory": subcategory,
            "impact": impact,
            "urgency": urgency,
            "priority": priority,
            "knowledge": knowledge,
            "u_priority_confirmation": u_priority_confirmation,
            "notify": notify,
            "day_of_week": day_of_week,
            "month": month,
            "day": day,
            "reassignment_count": reassignment_count,
            "reopen_count": reopen_count,
            "sys_mod_count": sys_mod_count
        }

        if model_choice == "Regression (Exact Time)":
            time_8hr, time_days = predict_incident(new_incident, lgbm_reg, enc_reg, scaler_reg)
            total_hours, hours, minutes = convert_to_hours_and_minutes(time_8hr)

            # --- Output results ---
            st.success(f"Predicted Resolution Time: **{time_8hr:.2f} (8-hour units)** ≈ **{time_days:.2f} days**")
            st.write(f"Approximately {hours}h {minutes}m")
            
        else:
            cls, proba, range_str = predict_incident(new_incident, lgbm_clf, enc_clf, scaler_clf)
            st.success(f"Predicted Resolution Range: **{range_str}** (Class {cls})")
            st.write("Class Probabilities:", {f"Class {i}": f"{p:.3f}" for i, p in enumerate(proba)})

# Containment Page
elif page == "Containment":
    st.markdown("<h2 style='text-align: center;'>Containment Stage</h2>", unsafe_allow_html=True)
    st.markdown("---")

    left_col, right_col = st.columns([1,2])

    with left_col:
        st.subheader("Filters")
        selected_category = st.multiselect("Category",sorted(containment_df['category'].dropna().unique()))
        selected_status = st.multiselect("Containment Status",sorted(containment_df['containment_status'].dropna().unique()))
        selected_device = st.multiselect("Device Number",sorted(containment_df['device_no'].dropna().unique()))
        
        filtered_df = containment_df.copy()
        if selected_category:
            filtered_df = filtered_df[filtered_df['category'].isin(selected_category)]
        if selected_status:
            filtered_df = filtered_df[filtered_df['containment_status'].isin(selected_status)]
        if selected_device:
            filtered_df = filtered_df[filtered_df['device_no'].isin(selected_device)]
        
        st.markdown("---")

        #KPIs
        total_devices = len(filtered_df)
        st.markdown(f"**Total Devices**")
        st.markdown(f"<h1 style='color: #FFFFFF; margin: 0;'>{total_devices}</h1>", unsafe_allow_html=True)


    with right_col:
        # Pie chart
        status_counts = filtered_df['containment_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']

        status_colors = {
            'Contained': '#cd3849',
            'Not Contained': '#dc793c', 
            'Released': '#3bb50b'
        }

        fig_pie = px.pie(
            status_counts,
            names='Status',
            values='Count',
            title="Containment Status Overview",
            color='Status',
            color_discrete_map=status_colors,
            height=400,
            width=500
        )

        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent',
            textfont=dict(color='white', size=18),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            marker=dict(line=dict(color='white', width=1)),
            pull=[0.02,0.02]
        )
        fig_pie.update_layout(
            font=dict(color='white', size=10),
            title_font=dict(size=18, color='white'),
            legend=dict(
                font=dict(color='white', size=12),
                bordercolor='white',
                borderwidth=1,
                xanchor='left',
                yanchor='middle'
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )

        st.plotly_chart(fig_pie, width='stretch')
    
    # Category bar chart
    cat_count = (filtered_df.groupby('category').size().reset_index(name="Count"))

    # Create color mapping for categories using palette
    unique_categories = cat_count['category'].unique()
    category_colors = {}
    for i, category in enumerate(unique_categories):
        color_key = f'c{(i % 5) + 1}'
        category_colors[category] = colors[color_key]

    fig_bar = px.bar(
        cat_count,
        x='category',
        y='Count',
        title='Devices by Category',
        color='category',
        text='Count',
        color_discrete_map=category_colors
    )

    fig_bar.update_traces(
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        marker_line_color='white',
        marker_line_width=1.5
    )

    fig_bar.update_layout(
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(
            title='Category',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Number of Devices',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        showlegend=False
    )

    st.plotly_chart(fig_bar, width='stretch')

    # Table of Containment
    st.markdown("Containment Records")

    st.dataframe(
    filtered_df,
    width='stretch',
    hide_index=True,
    key="containment_table",
    column_config={
        "number": "Incident Number",
        "device_no": "Device Number", 
        "category": "Category",
        "model": "Device Model",
        "containment_status": "Containment Status"
    },
    height=300,
    selection_mode="single-row",
    )
    
# Eradication & Recovery Page
elif page == "Eradication & Recovery":
    st.markdown("<h2 style='text-align: center;'>Eradication & Recovery Stage</h2>", unsafe_allow_html=True)
    st.markdown("---")


    eradication_summary = df.groupby("category")["reassignment_count"].mean().reset_index()
    # Create color mapping for categories using palette

    unique_categories = eradication_summary['category'].unique()
    category_colors = {}
    for i, category in enumerate(unique_categories):
        color_key = f'c{(i % 5) + 1}'
        category_colors[category] = colors[color_key]
    
    fig1 = px.bar(
        eradication_summary, 
        x="category", 
        y="reassignment_count",
        title="Average Reassignments per Category",
        color="category",
        color_discrete_map=category_colors
    )

    fig1.update_traces(
        hovertemplate="<b>%{x}</b><br>Avg Reassignments: %{y:.2f}<extra></extra>",
        marker_line_color='white',
        marker_line_width=1.5
    )

    fig1.update_layout(
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(
            title='Category',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Average Reassignments',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        showlegend=False
    )
    st.plotly_chart(fig1, width='stretch')

    fig2 = px.histogram(
        df, 
        x='category', 
        color='priority', 
        barmode="group",
        title="Incident Frequency by Category and Priority",
        color_discrete_map=priority_colors,
        category_orders={"priority": [1, 2, 3, 4]}
    ).update_traces(
        marker_line_color='white',
        marker_line_width=1
    )

    fig2.update_layout(
        font=dict(color='white', size=12),
        title_font=dict(size=16, color='white'),
        xaxis=dict(
            title='Category',
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title='Number of Incidents',
            title_font=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.2)'
        ),
        legend=dict(
            title='Priority',
            font=dict(color='white', size=12),
            bordercolor='white',
            borderwidth=1,
            traceorder='normal'
        )
    )
    
    st.plotly_chart(fig2, width='stretch')
    
    # Select incident number
    st.subheader("View E&R")
    incident_no = sorted(df['number'].unique())
    selected_no = st.selectbox("Select Incident", incident_no)

    # Filter row
    selected_row = df[df['number'] == selected_no].iloc[0]

    # Resolution notes
    resolution_notes = selected_row.get('resolution_notes','No notes available.')

    st.subheader("Investigation Notes")

    st.markdown(
    f"""
    <div style="
        background-color: #0E1117;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 10px;
        height: 350px;
        overflow-y: auto;
        color: white;
        white-space: pre-wrap;
        font-family: monospace;
    ">
    {resolution_notes}
    </div>
    """,
    unsafe_allow_html=True
    )

# Lessons Learned Page
elif page == "Lessons Learned":
    st.markdown("<h2 style='text-align: center;'>Lessons Learned Stage</h2>", unsafe_allow_html=True)
    st.markdown("---")
    # notes = st.text_area("Post-Incident Notes", placeholder="Write lessons learned or improvements...")
    # if st.button("Save Notes"):
    #     st.success("Notes saved (simulation).")

    # Initialize session state for report data
    if 'report_data' not in st.session_state:
        st.session_state.report_data = {}
    
    # Section 1: Incident Selection
    st.subheader("1. Select Incident for Report")
    
    # Get list of incidents - prioritize closed incidents for complete reports
    incident_options = sorted(df['number'].unique())
    selected_incident = st.selectbox("Select Incident Number", incident_options)
    
    # Load containment data if available
    try:
        containment_df = pd.read_csv('C:/Users/daphnee/Desktop/Work/Acads/FYP/Dataset/dashboard/containment.csv')
        containment_df = containment_df.loc[:, ~containment_df.columns.str.contains("^Unnamed")]
    except:
        containment_df = pd.DataFrame()
    
    # When incident is selected, load its data
    if selected_incident:
        incident_data = df[df['number'] == selected_incident].iloc[0]
        
        # Get containment data for this incident
        containment_data = None
        if not containment_df.empty and 'number' in containment_df.columns:
            containment_data = containment_df[containment_df['number'] == selected_incident]
        
        # Calculate recovery time automatically
        recovery_time_info = calculate_recovery_time(incident_data)
        
        # Section 2: Report Generation
        st.subheader("2. Generate Incident Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Incident basic info (read-only)
            st.markdown("**Incident Details**")
            st.text(f"Number: {incident_data.get('number', 'N/A')}")
            st.text(f"State: {incident_data.get('incident_state', 'N/A')}")
            st.text(f"Category: {incident_data.get('category', 'N/A')}")
            st.text(f"Priority: {incident_data.get('priority', 'N/A')}")
            st.text(f"Opened: {incident_data.get('opened_at', 'N/A')}")
            
            # Show closure information
            if 'closed_at' in incident_data and pd.notna(incident_data['closed_at']):
                st.text(f"Closed: {incident_data['closed_at']}")
                st.success(f"Recovery Time: {recovery_time_info['display']}")
            elif 'closed_at_predicted' in incident_data and pd.notna(incident_data['closed_at_predicted']):
                st.text(f"Predicted Closure: {incident_data['closed_at_predicted']}")
                st.info(f"Estimated Recovery: {recovery_time_info['display']}")
            else:
                st.warning("Incident still active")
        
        with col2:
            # Report metadata
            report_date = st.date_input("Report Date", datetime.now().date())
            prepared_by = st.text_input("Prepared By", "IT Team")
            report_version = st.selectbox("Report Version", ["Draft", "Final"])
        
        # Section 3: Report Content
        st.subheader("3. Report Content")
        
        # Identification Section
        st.markdown("<u>Identification</u>", unsafe_allow_html=True)
        
        detected_at = st.text_input("Detected At", 
                                   value=incident_data.get('opened_at', 'N/A'),
                                   help="When the incident was first detected")
        
        detection_method = st.text_area("Detection Method",
                                       placeholder="How was this incident detected? (e.g., SIEM alert, user report, automated monitoring...)",
                                       help="Describe the method used to detect this incident")
        
        # Containment Section
        st.markdown("<u>Containment</u>", unsafe_allow_html=True)
        
        if containment_data is not None and not containment_data.empty:
            st.success(f"{len(containment_data)} device(s) contained for this incident")
            
            # Show contained devices in a table
            containment_display = containment_data[['device_no', 'category', 'model', 'containment_status']].copy()
            st.dataframe(containment_display, width='stretch')
            
            containment_notes = st.text_area("Containment Actions",
                                           placeholder="Describe the containment actions taken...",
                                           help="Details about how the incident was contained")
        else:
            st.info("No containment data found for this incident")
            containment_notes = st.text_area("Containment Actions", "None",
                                           help="No devices were contained for this incident")
        
        # Eradication & Recovery Section
        st.markdown("<u>Eradication & Recovery</u>", unsafe_allow_html=True)
        
        investigation_notes = st.text_area("Investigation Notes",
                                           value=incident_data.get('resolution_notes', 'None'),
                                           placeholder="Detailed investigation findings and eradication steps...",
                                           height=150,
                                           help="Technical details of the investigation and eradication process")
        
        # Recovery Time (auto-calculated, read-only)
        st.markdown("**Recovery Time (Auto-calculated)**")
        recovery_time_display = st.text_input(
            "Recovery Duration", 
            value=recovery_time_info['display'],
            disabled=True,
            help="Automatically calculated from incident timestamps"
        )
        
        # Additional recovery details
        recovery_details = st.text_area("Recovery Process Details",
                                       placeholder="Additional details about the recovery process...",
                                       height=100,
                                       help="Specific steps taken during recovery")
        
        # Lessons Learned Section
        st.markdown("<u>Lessons Learned</u>", unsafe_allow_html=True)
        
        post_incident_notes = st.text_area("Post Incident Notes",
                                        placeholder="Key lessons learned and recommendations for improvement...",
                                        height=150,
                                        help="What can be improved based on this incident?")
        
        improvement_actions = st.text_area("Recommended Actions",
                                           placeholder="Specific actions to prevent similar incidents...",
                                           height=100,
                                           help="Concrete steps for improvement")
        
        # Section 4: Export Options
        st.subheader("4. Export Report")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            export_format = st.selectbox("Export Format", ["TXT", "PDF", "HTML", "Markdown"])
        
        with export_col2:
            include_timeline = st.checkbox("Include Detailed Timeline", value=True)
        
        
        # Store report data in session state
        st.session_state.report_data = {
            'incident_number': selected_incident,
            'incident_data': incident_data,
            'containment_data': containment_data,
            'recovery_time_info': recovery_time_info,
            'report_date': report_date,
            'prepared_by': prepared_by,
            'report_version': report_version,
            'detected_at': detected_at,
            'detection_method': detection_method,
            'containment_notes': containment_notes,
            'investigation_notes': investigation_notes,
            'recovery_details': recovery_details,
            'post_incident_notes': post_incident_notes,
            'improvement_actions': improvement_actions,
            'export_format': export_format,
            'include_timeline': include_timeline
        }
        
        # Export buttons
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("Generate Report Preview", width='stretch'):
                preview_report()
        
        with col_export2:
            if st.button("Export Report", width='stretch', type="primary"):
                export_incident_report()



# Get the problem incident
problem_incident = df[df['number'] == 'INC0000120'].iloc[0]
incident_dict = problem_incident.to_dict()
predicted_time = predict_incident(incident_dict, lgbm_reg, enc_reg, scaler_reg)

# def debug_negative_predictions():
#     """Debug why the model produces negative predictions"""
#     print("=== NEGATIVE PREDICTION DEBUG ===")
    
#     # Test with the problem incident
#     problem_incident = df[df['number'] == 'INC0000120'].iloc[0]
#     incident_dict = problem_incident.to_dict()
    
#     # Convert booleans
#     bool_fields = ['active', 'made_sla', 'knowledge', 'u_priority_confirmation']
#     for field in bool_fields:
#         if field in incident_dict:
#             val = incident_dict[field]
#             if isinstance(val, bool) and not isinstance(val, np.bool_):
#                 incident_dict[field] = np.True_ if val else np.False_
    
#     # Create the feature matrix
#     new_df = pd.DataFrame([incident_dict])
    
#     # Apply frequency encoding
#     for col in ['impact', 'urgency', 'priority']:
#         if col in new_df.columns:
#             freq_map = {'1': 100, '2': 500, '3': 50}
#             new_df[col + '_freq'] = new_df[col].map(freq_map).fillna(100)
    
#     categorical_features = [
#         'incident_state', 'active', 'made_sla', 'contact_type', 'location',
#         'category', 'subcategory', 'knowledge', 'u_priority_confirmation', 
#         'notify', 'day_of_week', 'month', 'day'
#     ]
    
#     # Check if all features are properly encoded
#     print("Feature check:")
#     for col in categorical_features + ['impact_freq', 'urgency_freq', 'priority_freq']:
#         if col in new_df.columns:
#             print(f"  {col}: {new_df[col].iloc[0]} (type: {type(new_df[col].iloc[0])})")
    
#     # Get the prediction
#     cat_transformed = enc_reg.transform(new_df[categorical_features])
#     num_scaled = scaler_reg.transform(new_df[['reassignment_count', 'reopen_count', 'sys_mod_count', 'impact_freq', 'urgency_freq', 'priority_freq']])
#     new_sparse = hstack([cat_transformed, csr_matrix(num_scaled)])
    
#     prediction = lgbm_reg.predict(new_sparse)[0]
#     print(f"Raw prediction: {prediction}")
    
#     return prediction

# # Run the debug
# debug_negative_predictions()

# manual_incident = {
#     'incident_state': 'Active',
#     'active': True,
#     'reassignment_count': 0,
#     'reopen_count': 0,
#     'sys_mod_count': 0,
#     'made_sla': True,
#     'caller_id': "Caller 1254",
#     'opened_at': "20/4/2025  8:04:00 AM",
#     'contact_type': 'Phone',
#     'location': 'HQ',
#     'category': 'IAM',
#     'subcategory': 'Access Control',
#     'u_symptom': 'Symptom 534',
#     'impact': 2,
#     'urgency': 2,
#     'priority': 3,
#     'assignment_group': 'SOC',
#     'assigned_to': 'analyst3',
#     'knowledge': False,
#     'u_priority_confirmation': False,
#     'notify': 'Do Not Notify',
#     'resolved_by': 'Resolved by 62'
# }


# print("=== COMPARISON ===")

# # Compare key features
# key_features = ['incident_state', 'active', 'reassignment_count', 'reopen_count', 
#                 'sys_mod_count', 'made_sla', 'contact_type', 'location', 'category',
#                 'subcategory', 'impact', 'urgency', 'priority', 'knowledge',
#                 'u_priority_confirmation', 'notify']

# for feature in key_features:
#     manual_val = manual_incident.get(feature, 'MISSING')
#     df_val = incident_dict_from_df.get(feature, 'MISSING')
    
#     print(f"{feature}:")
#     print(f"  Manual: {manual_val} (type: {type(manual_val)})")
#     print(f"  DF:     {df_val} (type: {type(df_val)})")
#     print(f"  Equal:  {manual_val == df_val}")
#     print()

# # Common data type problems to check
# def check_data_issues(incident_dict, source_name):
#     print(f"=== {source_name} DATA ISSUES ===")
    
#     # Check numeric fields
#     numeric_fields = ['reassignment_count', 'reopen_count', 'sys_mod_count', 'impact', 'urgency', 'priority']
#     for field in numeric_fields:
#         if field in incident_dict:
#             val = incident_dict[field]
#             print(f"{field}: {val} (type: {type(val)})")
#             if not isinstance(val, (int, float, np.int64, np.float64)):
#                 print(f"  ⚠️  NOT NUMERIC: {type(val)}")
    
#     # Check boolean fields
#     bool_fields = ['active', 'made_sla', 'knowledge', 'u_priority_confirmation']
#     for field in bool_fields:
#         if field in incident_dict:
#             val = incident_dict[field]
#             print(f"{field}: {val} (type: {type(val)})")
    
#     # Check categorical fields for unexpected values
#     cat_fields = ['incident_state', 'contact_type', 'location', 'category', 'subcategory', 'notify']
#     for field in cat_fields:
#         if field in incident_dict:
#             val = incident_dict[field]
#             print(f"{field}: '{val}' (type: {type(val)})")

# # Check both
# check_data_issues(manual_incident, "MANUAL")
# print("\n" + "="*50 + "\n")
# check_data_issues(incident_dict_from_df, "DATAFRAME")
