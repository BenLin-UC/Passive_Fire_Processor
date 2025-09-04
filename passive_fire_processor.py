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
    .optional-upload {
        border: 2px dashed #28a745;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: #f0f8f0;
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

def load_lookup_table(file_content, file_type):
    """Load lookup table from CSV or Excel file"""
    try:
        if file_type == 'csv':
            return pd.read_csv(io.BytesIO(file_content))
        else:  # Excel
            return pd.read_excel(io.BytesIO(file_content), sheet_name=0)
    except Exception as e:
        st.warning(f"Could not load lookup table: {e}")
        return None

def extract_element_id_from_ceiling_info(ceiling_info):
    """
    Extract element ID from ceiling/wall info string
    Example: "Ceilings - 53-Clg_(CT10) 16mm Fyreline - 6681320" -> 6681320
    """
    if not ceiling_info:
        return None
    
    try:
        # Look for pattern: "- [digits]" at the end or before the last pipe
        # Pattern 1: Element ID at the end (e.g., "... - 6681320")
        match = re.search(r'-\s*(\d{6,})\s*$', ceiling_info)
        if match:
            return int(match.group(1))
        
        # Pattern 2: Element ID followed by more text (e.g., "... - 6681320 |")
        match = re.search(r'-\s*(\d{6,})\s*(?:\||$)', ceiling_info)
        if match:
            return int(match.group(1))
        
    except Exception as e:
        pass
    
    return None

def search_in_lookup_table(element_id, element_info, lookup_df, search_columns=None):
    """
    Search for element information in the lookup table
    Priority: 1. Element ID, 2. Element info text matching
    """
    if lookup_df is None or lookup_df.empty:
        return None
    
    # Default search columns if not specified
    if search_columns is None:
        # Try to identify ID columns and type columns
        id_columns = [col for col in lookup_df.columns if 'id' in col.lower() or 'authoring' in col.lower()]
        type_columns = [col for col in lookup_df.columns if 'type' in col.lower() or 'name' in col.lower()]
        search_columns = {'id': id_columns, 'type': type_columns}
    
    # First, try to find by element ID if provided
    if element_id:
        for col in search_columns.get('id', []):
            if col in lookup_df.columns:
                matches = lookup_df[lookup_df[col] == element_id]
                if not matches.empty:
                    return matches.iloc[0]
    
    # If no ID match, try text matching on element info
    if element_info:
        for col in search_columns.get('type', []):
            if col in lookup_df.columns:
                matches = lookup_df[lookup_df[col].str.contains(element_info, case=False, na=False)]
                if not matches.empty:
                    return matches.iloc[0]
    
    return None

def get_separating_element_enhanced(ceiling_info, csv_data, lookup_table=None):
    """
    Enhanced version that handles element IDs and lookup tables
    Priority order:
    1. Element ID lookup in lookup table (if available)
    2. Element ID lookup in main CSV
    3. Text-based matching in lookup table (if available)
    4. Text-based matching in main CSV
    5. Return original info if no matches found
    """
    if not ceiling_info:
        return ''
    
    try:
        # Extract element ID if present
        element_id = extract_element_id_from_ceiling_info(ceiling_info)
        
        # Parse the ceiling info to get category and details
        category = ''
        element_details = ''
        
        # Remove the element ID from the string for text matching
        clean_ceiling_info = ceiling_info
        if element_id:
            # Remove the element ID pattern from the string
            clean_ceiling_info = re.sub(r'-\s*\d{6,}\s*(?:\||$)', '', ceiling_info).strip()
        
        # Split to get category and details
        if '-' in clean_ceiling_info:
            parts = clean_ceiling_info.split('-', 1)
            category = parts[0].strip()
            element_details = parts[1].strip() if len(parts) > 1 else ''
        else:
            category = clean_ceiling_info.strip()
        
        # Determine element type
        element_type = category.lower()
        is_wall = 'wall' in element_type
        is_ceiling = 'ceiling' in element_type
        is_floor = 'floor' in element_type
        is_structural = any(term in element_type for term in ['column', 'beam', 'framing', 'structural'])
        
        # Priority 1: Check lookup table with element ID
        if lookup_table is not None and element_id:
            match = search_in_lookup_table(element_id, None, lookup_table)
            if match is not None:
                return format_element_description(category, match, is_wall, is_ceiling, is_floor, is_structural)
        
        # Priority 2: Check main CSV with element ID
        if element_id:
            id_columns = ['Revizto→Authoring Tool Id', 'Element ID', 'ID']
            for col in id_columns:
                if col in csv_data.columns:
                    matches = csv_data[csv_data[col] == element_id]
                    if not matches.empty:
                        return format_element_description(category, matches.iloc[0], is_wall, is_ceiling, is_floor, is_structural)
        
        # Priority 3: Check lookup table with text matching
        if lookup_table is not None:
            match = search_in_lookup_table(None, element_details or category, lookup_table)
            if match is not None:
                return format_element_description(category, match, is_wall, is_ceiling, is_floor, is_structural)
        
        # Priority 4: Check main CSV with text matching
        search_text = clean_ceiling_info if clean_ceiling_info else ceiling_info
        
        # Try exact match first
        matches = csv_data[csv_data['Item→Type'].str.contains(search_text, case=False, na=False)]
        
        # Try partial match if exact fails
        if matches.empty and element_details:
            matches = csv_data[csv_data['Item→Type'].str.contains(element_details, case=False, na=False, regex=False)]
        
        # Try individual words as last resort
        if matches.empty:
            words = search_text.split()
            for word in words:
                if len(word) > 4:
                    matches = csv_data[csv_data['Item→Type'].str.contains(word, case=False, na=False)]
                    if not matches.empty:
                        break
        
        if not matches.empty:
            return format_element_description(category, matches.iloc[0], is_wall, is_ceiling, is_floor, is_structural)
        
        # Priority 5: Return original info if no matches
        return ceiling_info
        
    except Exception as e:
        return ceiling_info

def format_element_description(category, data_row, is_wall, is_ceiling, is_floor, is_structural):
    """
    Format the element description based on element type and available data
    """
    try:
        if is_structural:
            # Handle structural elements (columns, beams, framing)
            return format_structural_description(category, data_row)
        elif is_wall:
            # Handle walls with wall-specific columns
            return format_wall_description(category, data_row)
        elif is_ceiling:
            # Handle ceilings without wall columns
            return format_ceiling_description(category, data_row)
        elif is_floor:
            # Handle floors similarly to ceilings
            return format_floor_description(category, data_row)
        else:
            # Unknown type - use generic formatting
            item_type = data_row.get('Item→Type', '')
            if pd.notna(item_type) and item_type:
                return str(item_type)
            return category
    except Exception as e:
        return category

def format_structural_description(category, data_row):
    """Format description for structural elements"""
    parts = [category]
    
    # Look for structural-specific columns
    structural_type = data_row.get('Structural Type', data_row.get('Type', ''))
    material = data_row.get('Material', data_row.get('Structural Material', ''))
    size = data_row.get('Size', data_row.get('Dimensions', ''))
    
    if pd.notna(structural_type) and structural_type:
        parts.append(str(structural_type))
    
    if pd.notna(material) and material:
        parts.append(str(material))
    
    result = ' - '.join(parts)
    
    if pd.notna(size) and size:
        result += f" ({size})"
    
    return result

def format_wall_description(category, data_row):
    """Format description for wall elements - with validation"""
    # First validate this is actually wall data
    item_type = data_row.get('Item→Type', '')
    if pd.notna(item_type):
        item_type_lower = str(item_type).lower()
        # Check if this is ceiling or floor data instead of wall
        if ('ceiling' in item_type_lower or 'clg' in item_type_lower or 
            'floor' in item_type_lower or 'flr' in item_type_lower or 'slab' in item_type_lower):
            # This is not wall data - return just the category
            return category
    
    parts = [category]
    
    # Get wall-specific properties
    wall_system = data_row.get('Revit Type→WALL SYSTEM', data_row.get('WALL SYSTEM', ''))
    wall_framing = data_row.get('Revit Type→WALL FRAMING', data_row.get('WALL FRAMING', ''))
    wall_lining = data_row.get('Revit Type→WALL LINING', data_row.get('WALL LINING', ''))
    
    if pd.notna(wall_system) and wall_system:
        parts.append(str(wall_system))
    elif pd.notna(item_type) and item_type:
        # Only use Item→Type if it's actually wall-related
        if any(term in str(item_type).lower() for term in ['wall', 'ext-', 'int-', 'fyreline', 'partition']):
            parts.append(str(item_type))
    
    if pd.notna(wall_framing) and wall_framing:
        parts.append(str(wall_framing))
    
    result = ' - '.join(parts)
    
    if pd.notna(wall_lining) and wall_lining:
        result += f" with {wall_lining}"
    
    return result

def format_ceiling_description(category, data_row):
    """Format description for ceiling elements"""
    item_type = data_row.get('Item→Type', '')
    
    if pd.notna(item_type) and item_type:
        # Check if the category is already present in the item_type to avoid "Ceilings - Ceilings..."
        if category.strip().lower() in item_type.lower():
            return item_type
        # Otherwise, prepend the category
        return f"{category} - {item_type}"
    
    # If no valid item_type is found in the matched row, return the original category.
    return category

def format_floor_description(category, data_row):
    """Format description for floor elements"""
    item_type = data_row.get('Item→Type', '')
    
    if pd.notna(item_type) and item_type:
        # Check if this is actually a floor type
        if 'floor' in str(item_type).lower() or 'flr' in str(item_type).lower():
            return str(item_type)
        else:
            # Don't use non-floor data for floor elements
            return f"{category} - {item_type}" if not category in str(item_type) else str(item_type)
    
    return category

def process_passive_fire_schedule(excel_file_content, csv_file_content, lookup_table_content=None, lookup_table_type=None):
    """
    Process passive fire schedule data from uploaded files
    Now with optional lookup table support
    """
    
    try:
        # Read the Excel file from uploaded content
        excel_df = pd.read_excel(io.BytesIO(excel_file_content), sheet_name=0)
        st.success(f"✔ Excel file loaded: {len(excel_df)} rows")
        
        # Read the CSV file from uploaded content
        csv_df = pd.read_csv(io.BytesIO(csv_file_content))
        st.success(f"✔ CSV file loaded: {len(csv_df)} rows")
        
        # Load optional lookup table
        lookup_df = None
        if lookup_table_content and lookup_table_type:
            lookup_df = load_lookup_table(lookup_table_content, lookup_table_type)
            if lookup_df is not None:
                st.success(f"✔ Lookup table loaded: {len(lookup_df)} rows")
        
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
    
    def get_frr_info(ceiling_info, csv_data, lookup_table=None):
        """Get FRR information with lookup table support"""
        if not ceiling_info:
            return ''
        
        try:
            # Extract element ID if present
            element_id = extract_element_id_from_ceiling_info(ceiling_info)
            
            # Check lookup table first if available
            if lookup_table is not None and element_id:
                match = search_in_lookup_table(element_id, None, lookup_table)
                if match is not None:
                    frr_columns = ['FRR', 'WALL FRR', 'Fire Rating', 'Revit Type→WALL FRR']
                    for col in frr_columns:
                        if col in match.index:
                            frr_value = match[col]
                            if pd.notna(frr_value) and frr_value:
                                return str(frr_value)
            
            # Clean ceiling info for text matching
            clean_ceiling_info = ceiling_info
            if element_id:
                clean_ceiling_info = re.sub(r'-\s*\d{6,}\s*(?:\||$)', '', ceiling_info).strip()
            
            # Find rows where Item→Type matches
            matching_rows = csv_data[csv_data['Item→Type'].str.contains(clean_ceiling_info, case=False, na=False)]
            
            if matching_rows.empty:
                # Try partial matching
                words = clean_ceiling_info.split()
                for word in words:
                    if len(word) > 4:
                        partial_matches = csv_data[csv_data['Item→Type'].str.contains(word, case=False, na=False)]
                        if not partial_matches.empty:
                            matching_rows = partial_matches
                            break
            
            if matching_rows.empty:
                return ''
            
            # Get FRR values
            frr_values = matching_rows['Revit Type→WALL FRR'].dropna().unique()
            
            if len(frr_values) == 0:
                return ''
            elif len(frr_values) == 1:
                return str(frr_values[0])
            else:
                # Multiple different FRR values found
                unique_values = set(str(v) for v in frr_values)
                if len(unique_values) == 1:
                    return str(frr_values[0])
                else:
                    return f"WARNING: Multiple FRR values found: {', '.join(str(v) for v in frr_values)}"
        
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
        
        # Get FRR and separating element information with lookup table support
        frr_info = get_frr_info(ceiling_info, csv_df, lookup_df)
        separating_element = get_separating_element_enhanced(ceiling_info, csv_df, lookup_df)
        
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
        
        # Advanced options
        with st.expander("⚙️ Advanced Options", expanded=False):
            st.markdown("""
            **Lookup Table Priority:**
            When enabled, the lookup table will be searched first for element data.
            
            **Search Order:**
            1. Element ID in lookup table
            2. Element ID in main CSV
            3. Text match in lookup table
            4. Text match in main CSV
            """)
        
        st.markdown("---")
        
        # Help section
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **Step 1:** Upload your Excel file (Revizto export)
            
            **Step 2:** Upload your CSV file (Component database)
            
            **Step 3:** (Optional) Upload lookup table for structural elements
            
            **Step 4:** Click 'Process Files' button
            
            **Step 5:** Download your processed schedule
            
            **File Requirements:**
            - Excel: Must have 'ID' and 'Title' columns
            - CSV: Must have component and material data
            - Lookup Table: Optional, for structural elements
            
            **Enhanced Features:**
            - Supports element IDs in ceiling/wall descriptions
            - Optional lookup table for structural elements
            - Smart fallback search strategy
            """)
    
    # Main content area - Required files
    st.subheader("📁 Required Files")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Excel File Upload")
        st.markdown("*Upload your Revizto issue export file*")
        
        excel_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            key="excel_upload",
            help="Upload Excel file with Revizto clash detection data"
        )
        
        if excel_file:
            st.success(f"✔ Excel file loaded: {excel_file.name}")
            st.info(f"File size: {excel_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📋 CSV File Upload")
        st.markdown("*Upload your component database file*")
        
        csv_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            key="csv_upload",
            help="Upload CSV file with component data"
        )
        
        if csv_file:
            st.success(f"✔ CSV file loaded: {csv_file.name}")
            st.info(f"File size: {csv_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Optional lookup table section
    st.subheader("📁 Optional Files")
    st.markdown('<div class="optional-upload">', unsafe_allow_html=True)
    st.markdown("### 🔍 Lookup Table (Optional)")
    st.markdown("*Upload a lookup table for structural elements (columns, beams, framing)*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        lookup_file = st.file_uploader(
            "Choose lookup table file",
            type=['csv', 'xlsx', 'xls'],
            key="lookup_upload",
            help="Optional: Upload CSV or Excel file with structural element data"
        )
        
        if lookup_file:
            st.success(f"✔ Lookup table loaded: {lookup_file.name}")
            st.info(f"File size: {lookup_file.size / 1024:.1f} KB")
            
            # Determine file type
            lookup_file_type = 'csv' if lookup_file.name.endswith('.csv') else 'excel'
            
            with st.expander("📋 Lookup Table Info", expanded=False):
                st.markdown("""
                **The lookup table will be searched first for:**
                - Structural elements (columns, beams, framing)
                - Elements with specific IDs
                - Additional element types not in main CSV
                
                **Recommended columns:**
                - Element ID or Authoring Tool Id
                - Element Type or Name
                - Material properties
                - Structural specifications
                - Fire ratings (if applicable)
                """)
        else:
            lookup_file_type = None
    
    with col2:
        if lookup_file:
            st.metric("Lookup Table", "Loaded", "Ready")
        else:
            st.metric("Lookup Table", "Not Loaded", "Optional")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if excel_file and csv_file:
        st.markdown("---")
        
        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            with st.spinner("Processing your files... This may take a few minutes."):
                try:
                    # Prepare lookup table data
                    lookup_table_content = None
                    if lookup_file:
                        lookup_table_content = lookup_file.read()
                    
                    # Process the files
                    result_df = process_passive_fire_schedule(
                        excel_file.read(), 
                        csv_file.read(),
                        lookup_table_content,
                        lookup_file_type
                    )
                    
                    if result_df is not None:
                        st.balloons()
                        st.success("🎉 Processing completed successfully!")
                        
                        # Display statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", len(result_df))
                        
                        with col2:
                            services_with_data = len(result_df[result_df['Service'].str.len() > 5])
                            st.metric("Services Processed", services_with_data)
                        
                        with col3:
                            frr_count = len(result_df[result_df['FRR'].str.len() > 0])
                            st.metric("FRR Records", frr_count)
                        
                        with col4:
                            if lookup_file:
                                st.metric("Lookup Table", "Used", "✓")
                            else:
                                st.metric("Lookup Table", "Not Used", "-")
                        
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
        "🔥 Passive Fire Schedule Processor v2.0 | Enhanced with Lookup Table Support"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()