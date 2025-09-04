import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from datetime import datetime
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Passive Fire Schedule Processor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .metrics-container {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def process_passive_fire_schedule(excel_file_content, csv_file_content):
    """
    Process passive fire schedule data from uploaded files
    """
    
    try:
        # Read the Excel file from uploaded content
        excel_df = pd.read_excel(io.BytesIO(excel_file_content), sheet_name=0)
        st.success(f"✓ Excel file loaded: {len(excel_df)} rows")
        
        # Read the CSV file from uploaded content
        csv_df = pd.read_csv(io.BytesIO(csv_file_content))
        st.success(f"✓ CSV file loaded: {len(csv_df)} rows")
        
    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None
    
    # Service type mapping
    service_lookup = {
        'FIR': 'Fire',
        'ELE': 'Electric', 
        'HYD': 'Hydraulic',
        'MEC': 'Mechanical',
        'STR': 'Structural'
    }
    
    def parse_service(title, csv_data):
        """Parse the service information from title string"""
        if pd.isna(title) or not title:
            return ''
        
        try:
            # Split by "|"
            parts = title.split('|')
            if len(parts) < 3:
                return ''
            
            # Extract service type from first part (last 3 characters)
            first_part = parts[0].strip()
            service_code = first_part[-3:]
            service_name = service_lookup.get(service_code, service_code)
            
            # Extract element ID from third part
            third_part = parts[2].strip()
            element_id_match = re.search(r'(\d+)\s*\[', third_part)
            if not element_id_match:
                return service_name + ' - '
            
            element_id = int(element_id_match.group(1))
            
            # Find matching row in CSV data
            matching_rows = csv_data[csv_data['Revizto→Authoring Tool Id'] == element_id]
            if matching_rows.empty:
                return service_name + ' - '
            
            matching_row = matching_rows.iloc[0]
            
            # Build service description
            service_desc = service_name + ' - '
            
            # Add System Classification (Column J)
            if pd.notna(matching_row.get('MECHANICAL→System Classification')):
                service_desc += str(matching_row['MECHANICAL→System Classification']) + ' '
            
            # Add Insulation Thickness (Column H)
            insul_thickness = matching_row.get('INSULATION→Insulation Thickness mm')
            if pd.notna(insul_thickness) and insul_thickness > 0:
                service_desc += str(int(insul_thickness)) + 'mm '
            
            # Add Item Type (Column B) or Item Name (Column A)
            item_type = matching_row.get('Item→Type')
            item_name = matching_row.get('Item→Name')
            
            if pd.notna(item_type) and not str(item_type).lower().startswith('standard'):
                service_desc += str(item_type)
            elif pd.notna(item_name):
                service_desc += str(item_name)
            
            # Add Insulation Type if exists (Column M)
            insul_type = matching_row.get('INSULATION→Insulation Type')
            if pd.notna(insul_type):
                thickness_val = insul_thickness if pd.notna(insul_thickness) else ''
                service_desc += f" with {thickness_val}mm {insul_type}"
            
            return service_desc.strip()
            
        except Exception as e:
            return ''
    
    def get_service_material(element_id, csv_data):
        """Get service material from CSV columns K or L"""
        if pd.isna(element_id):
            return ''
        
        matching_rows = csv_data[csv_data['Revizto→Authoring Tool Id'] == element_id]
        if matching_rows.empty:
            return ''
        
        matching_row = matching_rows.iloc[0]
        
        # Try Column K first (MECHANICAL→Material)
        mechanical_material = matching_row.get('MECHANICAL→Material')
        if pd.notna(mechanical_material):
            return str(mechanical_material)
        
        # Try Column L (MATERIALS→Structural Material)
        structural_material = matching_row.get('MATERIALS→Structural Material')
        if pd.notna(structural_material):
            return str(structural_material)
        
        return ''
    
    def extract_element_id(title):
        """Extract element ID from title for material and pipe size lookups"""
        if pd.isna(title) or not title:
            return None
        
        try:
            parts = title.split('|')
            if len(parts) >= 3:
                third_part = parts[2].strip()
                element_id_match = re.search(r'(\d+)\s*\[', third_part)
                if element_id_match:
                    return int(element_id_match.group(1))
        except:
            pass
        return None
    
    def extract_ceiling_info(title):
        """Extract the ceiling/wall information from the second split of title"""
        if pd.isna(title) or not title:
            return None
        
        try:
            parts = title.split('|')
            if len(parts) >= 2:
                return parts[1].strip()  # Second part
        except:
            pass
        return None
    
    def get_frr_info(ceiling_info, csv_data):
        """Get FRR information from CSV based on ceiling/wall info"""
        if not ceiling_info:
            return ''
        
        try:
            # Find rows where Item→Type matches the ceiling info
            matching_rows = csv_data[csv_data['Item→Type'].str.contains(ceiling_info, case=False, na=False)]
            
            if matching_rows.empty:
                # Try partial matching if exact match fails
                words = ceiling_info.split()
                for word in words:
                    if len(word) > 4:  # Only use meaningful words
                        partial_matches = csv_data[csv_data['Item→Type'].str.contains(word, case=False, na=False)]
                        if not partial_matches.empty:
                            matching_rows = partial_matches
                            break
            
            if matching_rows.empty:
                return ''
            
            # Get FRR values from Column D
            frr_values = matching_rows['Revit Type→WALL FRR'].dropna().unique()
            
            if len(frr_values) == 0:
                return ''
            elif len(frr_values) == 1:
                return str(frr_values[0])
            else:
                # Multiple different FRR values found - check if they're the same
                if len(set(str(v) for v in frr_values)) == 1:
                    return str(frr_values[0])  # Same values, return any
                else:
                    return f"WARNING: Multiple FRR values found: {', '.join(str(v) for v in frr_values)}"
        
        except Exception as e:
            return ''
    
    def get_separating_element(ceiling_info, csv_data):
        """Get separating element information from CSV with penetrated element category"""
        if not ceiling_info:
            return ''
        
        try:
            # Extract category (Ceilings/Walls/Floors) from ceiling_info
            category = ''
            if ceiling_info:
                category_parts = ceiling_info.split('-')
                if len(category_parts) > 0:
                    category = category_parts[0].strip()
            
            # Find rows where Item→Type matches the ceiling info
            matching_rows = csv_data[csv_data['Item→Type'].str.contains(ceiling_info, case=False, na=False)]
            
            if matching_rows.empty:
                # Try partial matching if exact match fails
                words = ceiling_info.split()
                for word in words:
                    if len(word) > 4:
                        partial_matches = csv_data[csv_data['Item→Type'].str.contains(word, case=False, na=False)]
                        if not partial_matches.empty:
                            matching_rows = partial_matches
                            break
            
            if matching_rows.empty:
                # Return just the category if no CSV match found
                return category if category else ''
            
            # Use first matching row
            matching_row = matching_rows.iloc[0]
            
            # Get WALL SYSTEM (Column E), if empty use Item→Type (Column B)
            wall_system = matching_row.get('Revit Type→WALL SYSTEM')
            if pd.isna(wall_system) or not wall_system:
                wall_system = matching_row.get('Item→Type', '')
            
            # Get WALL FRAMING (Column F)
            wall_framing = matching_row.get('Revit Type→WALL FRAMING')
            if pd.isna(wall_framing) or not wall_framing:
                # If no framing, return category + wall system (if available)
                if category:
                    if wall_system:
                        return f"{category} - {wall_system}"
                    else:
                        return category
                else:
                    return str(wall_system) if wall_system else ''
            
            # Build separating element description with category first
            if category:
                separating_element = f"{category} - {wall_system} - {wall_framing}"
            else:
                separating_element = f"{wall_system} - {wall_framing}"
            
            # Get WALL LINING (Column G) and add if exists
            wall_lining = matching_row.get('Revit Type→WALL LINING')
            if pd.notna(wall_lining) and wall_lining:
                separating_element += f" with {wall_lining}"
            
            return separating_element
            
        except Exception as e:
            return ''
    
    def get_pipe_size(element_id, csv_data):
        """Get pipe size from CSV column I"""
        if pd.isna(element_id):
            return ''
        
        matching_rows = csv_data[csv_data['Revizto→Authoring Tool Id'] == element_id]
        if matching_rows.empty:
            return ''
        
        matching_row = matching_rows.iloc[0]
        geometry_size = matching_row.get('GEOMETRY→Size')
        
        return str(geometry_size) if pd.notna(geometry_size) else ''
    
    # Process all data with progress bar
    processed_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, row in excel_df.iterrows():
        # Update progress
        progress = (index + 1) / len(excel_df)
        progress_bar.progress(progress)
        status_text.text(f"Processing record {index + 1} of {len(excel_df)}")
        
        revizto_id = row.get('ID', '')
        title = row.get('Title', '')
        
        # Parse service information
        service = parse_service(title, csv_df)
        
        # Extract element ID for lookups
        element_id = extract_element_id(title)
        
        # Get all required data
        service_material = get_service_material(element_id, csv_df)
        pipe_size = get_pipe_size(element_id, csv_df)
        
        # Extract ceiling/wall info for columns E & F
        ceiling_info = extract_ceiling_info(title)
        
        # Get FRR and separating element information
        frr_info = get_frr_info(ceiling_info, csv_df)
        separating_element = get_separating_element(ceiling_info, csv_df)
        
        processed_data.append({
            'Revizto ID': revizto_id,
            'Service': service,
            'Service Material': service_material,
            'Pipe Size(OD)': pipe_size,
            'FRR': frr_info,
            'Separating Element': separating_element
        })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Create final DataFrame
    result_df = pd.DataFrame(processed_data)
    
    return result_df

def export_dataframe(df, format_type, filename):
    """Export DataFrame to various formats"""
    
    if format_type == 'XLSX':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Passive Fire Schedule', index=False)
            # Auto-adjust column widths
            worksheet = writer.sheets['Passive Fire Schedule']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output.getvalue(), f"{filename}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    elif format_type == 'CSV':
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode('utf-8'), f"{filename}.csv", "text/csv"
    
    elif format_type == 'JSON':
        json_data = df.to_json(orient='records', indent=2)
        return json_data.encode('utf-8'), f"{filename}.json", "application/json"
    
    elif format_type == 'HTML':
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Passive Fire Schedule</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #1f77b4; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .summary {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Passive Fire Schedule</h1>
            <div class="summary">
                <strong>Processing Summary:</strong><br>
                Total Records: {len(df)}<br>
                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            {df.to_html(table_id='passive-fire-table', classes='table table-striped', index=False, escape=False)}
        </body>
        </html>"""
        return html_content.encode('utf-8'), f"{filename}.html", "text/html"

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔥 Passive Fire Schedule Processor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Output format selection
        output_format = st.selectbox(
            "Select Output Format",
            ["XLSX", "CSV", "JSON", "HTML"],
            index=0,
            help="Choose the format for your processed schedule"
        )
        
        # Output filename
        output_filename = st.text_input(
            "Output Filename",
            "Passive_Fire_Schedule_Processed",
            help="Enter filename (extension will be added automatically)"
        )
        
        st.markdown("---")
        
        # Help section
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **Step 1:** Upload your Excel file (Revizto export)
            
            **Step 2:** Upload your CSV file (Component database)
            
            **Step 3:** Click 'Process Files' button
            
            **Step 4:** Download your processed schedule
            
            **File Requirements:**
            - Excel: Must have 'ID' and 'Title' columns
            - CSV: Must have component and material data
            """)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📊 Excel File Upload")
        st.markdown("*Upload your Revizto issue export file (any .xlsx/.xls filename)*")
        
        excel_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            key="excel_upload",
            help="Upload any Excel file with Revizto clash detection data - filename doesn't matter, only content structure"
        )
        
        if excel_file:
            st.success(f"✓ Excel file loaded: {excel_file.name}")
            st.info(f"File size: {excel_file.size / 1024:.1f} KB")
            
            # Show required columns info
            with st.expander("📋 Required Excel Columns", expanded=False):
                st.markdown("""
                **Your Excel file must contain these columns:**
                - `ID` - Unique Revizto issue identifier
                - `Title` - Formatted as: "Service Type | Element | Component"
                
                **Example Title format:**
                ```
                ARC (Fyreline) vs HYD | Ceilings - 53-Clg_(CT10) | Pipe Fittings - Standard - 2570324 [172]
                ```
                """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📋 CSV File Upload")
        st.markdown("*Upload your component database file (any .csv filename)*")
        
        csv_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            key="csv_upload",
            help="Upload any CSV file with component data - filename doesn't matter, only content structure"
        )
        
        if csv_file:
            st.success(f"✓ CSV file loaded: {csv_file.name}")
            st.info(f"File size: {csv_file.size / 1024:.1f} KB")
            
            # Show required columns info
            with st.expander("📋 Required CSV Columns", expanded=False):
                st.markdown("""
                **Your CSV file must contain these columns:**
                - `Revizto→Authoring Tool Id` - Links to Excel file
                - `Item→Name` & `Item→Type` - Component details
                - `Revit Type→WALL FRR` - Fire resistance rating
                - `Revit Type→WALL SYSTEM/FRAMING/LINING` - Wall details
                - `GEOMETRY→Size` - Component dimensions
                - `MECHANICAL→System Classification/Material` - Service info
                - `MATERIALS→Structural Material` - Material specs
                - `INSULATION→Thickness/Type` - Insulation data
                """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if excel_file and csv_file:
        st.markdown("---")
        
        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            with st.spinner("Processing your files... This may take a few minutes."):
                try:
                    # Process the files
                    result_df = process_passive_fire_schedule(excel_file.read(), csv_file.read())
                    
                    if result_df is not None:
                        st.balloons()
                        st.success("🎉 Processing completed successfully!")
                        
                        # Display statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Records", len(result_df))
                        
                        with col2:
                            services_with_data = len(result_df[result_df['Service'].str.len() > 5])
                            st.metric("Services Processed", services_with_data)
                        
                        with col3:
                            frr_count = len(result_df[result_df['FRR'].str.len() > 0])
                            st.metric("FRR Records", frr_count)
                        
                        # Additional statistics
                        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            material_count = len(result_df[result_df['Service Material'].str.len() > 0])
                            st.metric("Material Data", material_count, f"{material_count/len(result_df)*100:.1f}%")
                        
                        with col2:
                            pipe_count = len(result_df[result_df['Pipe Size(OD)'].str.len() > 0])
                            st.metric("Pipe Size Data", pipe_count, f"{pipe_count/len(result_df)*100:.1f}%")
                        
                        with col3:
                            separating_count = len(result_df[result_df['Separating Element'].str.len() > 0])
                            st.metric("Separating Elements", separating_count, f"{separating_count/len(result_df)*100:.1f}%")
                        
                        with col4:
                            warning_count = len(result_df[result_df['FRR'].str.contains('WARNING', na=False)])
                            if warning_count > 0:
                                st.metric("⚠️ Warnings", warning_count, "Review Required")
                            else:
                                st.metric("✅ No Warnings", 0, "All Clear")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Service type distribution
                        service_types = {}
                        for service in result_df['Service']:
                            if service and pd.notna(service) and len(str(service).strip()) > 5:
                                service_type = str(service).split(' - ')[0]
                                service_types[service_type] = service_types.get(service_type, 0) + 1
                        
                        if service_types:
                            st.subheader("📊 Service Type Distribution")
                            service_df = pd.DataFrame(list(service_types.items()), columns=['Service Type', 'Count'])
                            service_df['Percentage'] = (service_df['Count'] / service_df['Count'].sum() * 100).round(1)
                            st.bar_chart(service_df.set_index('Service Type')['Count'])
                        
                        # Data preview
                        st.subheader("🔍 Data Preview")
                        st.dataframe(
                            result_df.head(10),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Export functionality
                        st.subheader("💾 Download Processed Data")
                        
                        try:
                            file_data, filename, mime_type = export_dataframe(result_df, output_format, output_filename)
                            
                            st.download_button(
                                label=f"📥 Download {output_format} File",
                                data=file_data,
                                file_name=filename,
                                mime=mime_type,
                                type="primary",
                                use_container_width=True
                            )
                            
                            st.success(f"Ready to download: {filename}")
                            
                        except Exception as e:
                            st.error(f"Error preparing download: {e}")
                        
                        # Warnings section
                        if warning_count > 0:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.subheader("⚠️ Review Required")
                            warning_records = result_df[result_df['FRR'].str.contains('WARNING', na=False)]
                            st.dataframe(warning_records[['Revizto ID', 'FRR']], use_container_width=True)
                            st.markdown("These records have conflicting FRR values and require manual review.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    else:
                        st.error("❌ Processing failed. Please check your files and try again.")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {e}")
                    st.exception(e)
    
    else:
        st.info("👆 Please upload both Excel and CSV files to begin processing.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; margin-top: 2rem;'>"
        "🔥 Passive Fire Schedule Processor | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()