# passive_fire_processor.py
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
    page_title="Passive Fire Schedule Processor v3.0",
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
    .lookup-section {
        border: 2px dashed #28a745;
        border-radius: 10px;
        padding: 15px;
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
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Normalization & Variant Engine
# ============================

def generate_name_variants(name: str):
    """
    Generate likely lookup variants for steel/structural names.

    Examples:
      - "410UB60" -> ["410UB60","410 UB 60","410UB 60","410 UB60"]
      - "125x6SHS" -> ["125x6SHS","125 x 125 x 6 SHS","125x125x6 SHS","125 x 6 SHS", ...]
    """
    if not name or pd.isna(name):
        return []

    s = str(name).strip()
    # canonical uppercase no-space form for detection
    canonical = re.sub(r'\s+', '', s).upper()
    canonical = canonical.replace('×', 'X')  # some sources use multiplication symbol
    variants = set()
    variants.add(s)
    variants.add(canonical)

# DHS Purlin
    m = re.search(r"(\\d+)\\s*DHS", canonical, re.IGNORECASE)
    if m:
        depth = m.group(1)
        variants.update({
            f"{depth} DHS",           # matches row 124 directly
            f"{depth}DHS",
            f"{depth} DHS Purlin",    # input form
            f"DHS {depth}",
            f"DHS{depth}"
        })
    # Also handle cases where there may be trailing text like "410UB60L" or "410UB60 A" 
    # capture numeric-UB/UC within a longer string
    m2 = re.search(r'(\d+)(UB|UC)(\d+)', canonical)
    if m2:
        depth, typ, weight = m2.groups()
        typ = typ.upper()
        variants.update({
            f"{depth}{typ}{weight}",
            f"{depth} {typ} {weight}",
            f"{depth}{typ} {weight}",
            f"{depth} {typ}{weight}"
        })
        # also include substring forms
        variants.add(canonical[m2.start():m2.end()])

    # --- SHS/RHS patterns
    # Cases:
    #  - 125x6SHS  (implies 125x125x6)
    #  - 125x125x6SHS
    #  - 125 X 6 SHS
    shs_match = re.match(r'^(\d+)X(\d+)(SHS|RHS)$', canonical)
    if shs_match:
        a, b, shape = shs_match.groups()
        shape = shape.upper()
        # Common variations to try:
        variants.update({
            f"{a}x{b}{shape}",
            f"{a} x {a} x {b} {shape}",
            f"{a}x{a}x{b} {shape}",
            f"{a} x {b} {shape}",
            f"{a}x{b} {shape}"
        })

    shs_match2 = re.match(r'^(\d+)X(\d+)X(\d+)(SHS|RHS)$', canonical)
    if shs_match2:
        a, b, c, shape = shs_match2.groups()
        shape = shape.upper()
        variants.update({
            f"{a}x{b}x{c}{shape}",
            f"{a} x {b} x {c} {shape}",
            f"{a}x{b}x{c} {shape}"
        })

    # --- CHS (circular hollow sections) or generic patterns like "Ø50" are not critical here,
    # but include a simple no-space / spaced form
    # Add spaced tokens for anything with letters and numbers e.g. "125X6SHS" -> "125 X 6 SHS"
    spaced = re.sub(r'([A-Z])', r' \1', canonical).strip()
    spaced = re.sub(r'\s+', ' ', spaced)
    variants.add(spaced)

    # Also include lower/upper, and remove duplicate whitespace variants:
    final_variants = set()
    for v in variants:
        if not v:
            continue
        final_variants.add(v)
        final_variants.add(v.upper())
        final_variants.add(v.title())
        # also add a cleaned space-normalized version
        final_variants.add(re.sub(r'\s+', ' ', v).strip())

    # return as a list, prefer preserving likely variants first by sorting length desc
    return sorted(final_variants, key=lambda x: (-len(x), x))

# ============================
# Lookup steel properties (variant-aware)
# ============================
def lookup_steel_properties(steel_name, size_lookup_df):
    """
    Look up steel properties from size lookup table
    Returns: (size_string, material)
    Uses generate_name_variants() to try a number of normalized forms.
    """
    if size_lookup_df is None or steel_name is None or pd.isna(steel_name):
        return ('', '')

    try:
        # prepare columns to check (case sensitive presence check)
        name_columns = ['Steel Name', 'Name', 'Steel', 'Item Name']
        # fallback to any column with 'name' in it if none of the above exist
        available_name_cols = [c for c in size_lookup_df.columns if c in name_columns]
        if not available_name_cols:
            available_name_cols = [c for c in size_lookup_df.columns if 'name' in c.lower()]

        if not available_name_cols:
            # no name-like columns; can't match
            return ('', '')

        variants = generate_name_variants(steel_name)

        # 1) Exact (normalized) matches first
        match_row = None
        for variant in variants:
            v_norm = variant.strip().lower()
            for col in available_name_cols:
                col_series = size_lookup_df[col].astype(str).fillna('').str.strip().str.lower()
                matches = size_lookup_df[col_series == v_norm]
                if not matches.empty:
                    match_row = matches.iloc[0]
                    break
            if match_row is not None:
                break

        # 2) Stricter partial matches using word boundaries (avoid matching '200' inside '1200')
        if match_row is None:
            for variant in variants:
                # use word-boundary anchored pattern to avoid accidental substring hits
                pat = rf"\b{re.escape(variant)}\b"
                for col in available_name_cols:
                    matches = size_lookup_df[size_lookup_df[col].astype(str).str.contains(pat, case=False, regex=True, na=False)]
                    if not matches.empty:
                        match_row = matches.iloc[0]
                        break
                if match_row is not None:
                    break


#        match_row = None
#
#        # Exact equality attempts (normalized)
#        for variant in variants:
#            v_norm = variant.strip().lower()
#            for col in available_name_cols:
#                # cast to str to avoid dtype issues, strip then lower
#                col_series = size_lookup_df[col].astype(str).fillna('').str.strip().str.lower()
#                matches = size_lookup_df[col_series == v_norm]
#                if not matches.empty:
#                    match_row = matches.iloc[0]
#                    break
#            if match_row is not None:
#                break
#
#        # Partial contains attempts if exact match failed
#        if match_row is None:
#            for variant in variants:
#                v_norm = variant.strip()
#                for col in available_name_cols:
#                    try:
#                        # contains with case-insensitive
#                        matches = size_lookup_df[size_lookup_df[col].astype(str).str.contains(re.escape(v_norm), case=False, na=False)]
#                    except Exception:
#                        # fallback generic contains
#                        matches = size_lookup_df[size_lookup_df[col].astype(str).str.contains(v_norm, case=False, na=False)]
#                    if not matches.empty:
#                        match_row = matches.iloc[0]
#                        break
#                if match_row is not None:
#                    break

        if match_row is not None:
            # Extract size information (try to find width/height-like columns)
            width_col = None
            height_col = None
            for col in match_row.index:
                if 'width' in col.lower():
                    width_col = col
                if 'height' in col.lower():
                    height_col = col
            size_str = ''
            if width_col and height_col:
                try:
                    width_val = float(str(match_row[width_col]).replace('mm','').strip())
                    height_val = float(str(match_row[height_col]).replace('mm','').strip())
                    size_str = f"{int(round(height_val))}x{int(round(width_val))}"
                except:
                    width = str(match_row[width_col]).replace('mm', '').strip()
                    height = str(match_row[height_col]).replace('mm', '').strip()
                    size_str = f"{height}x{width}"


            # Extract material information
            material = ''
            material_columns = ['Material', 'Materials', 'Type']
            for col in material_columns:
                if col in match_row.index and pd.notna(match_row[col]):
                    material = str(match_row[col])
                    break

            return (size_str, material)

    except Exception as e:
        # swallow errors to keep UI running; optionally log to st.warning in debug mode
        # st.warning(f"Lookup error for '{steel_name}': {e}")
        pass

    return ('', '')

# ============================
# Other utility functions (unchanged / slightly polished)
# ============================
def load_excel_with_sheet_selection(file_content):
    """
    Load Excel file with automatic second sheet selection if multiple sheets exist
    """
    try:
        # First, check how many sheets are in the file
        excel_file = pd.ExcelFile(io.BytesIO(file_content))
        sheet_names = excel_file.sheet_names

        if len(sheet_names) > 1:
            st.info(f"📋 Multiple worksheets detected ({len(sheet_names)} sheets). Automatically selecting sheet 2: '{sheet_names[1]}'")
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=1)
        else:
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=0)

        return df

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def load_lookup_table(file_content, file_type, table_name):
    """Load lookup table from CSV or Excel file"""
    try:
        if file_type == 'csv':
            df = pd.read_csv(io.BytesIO(file_content))
        else:  # Excel
            excel_file = pd.ExcelFile(io.BytesIO(file_content))
            if len(excel_file.sheet_names) > 1:
                st.info(f"📋 {table_name}: Multiple sheets detected. Using sheet 2: '{excel_file.sheet_names[1]}'")
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=1)
            else:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=0)

        return df

    except Exception as e:
        st.warning(f"Could not load {table_name}: {e}")
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
        match = re.search(r'-\s*(\d{6,})\s*$', ceiling_info)
        if match:
            return int(match.group(1))

        match = re.search(r'-\s*(\d{6,})\s*(?:\||$)', ceiling_info)
        if match:
            return int(match.group(1))

    except Exception as e:
        pass

    return None

def classify_material_by_keywords(text):
    """
    Classify material based on keyword detection (case-insensitive)
    Priority order: Steel, PVC, PP-R
    """
    if not text or pd.isna(text):
        return ''

    text_lower = str(text).lower()

    # Check for keywords in priority order
    if 'steel' in text_lower or 'gal' in text_lower or 'fp_gal' in text_lower:
        return 'Steel'
    elif 'pvc' in text_lower or 'plastic' in text_lower:
        return 'PVC'
    elif 'pp-r' in text_lower or 'ppr' in text_lower or 'raufusion' in text_lower:
        return 'PP-R'
    elif 'copper' in text_lower:
        return 'Copper'

    return ''

def extract_largest_pipe_size(size_text):
    """
    Extract the largest pipe size from a string containing multiple sizes
    Examples: 
    - "Ø32 mm-Ø32 mm" -> "Ø32"
    - "ø65-ø65-ø40-ø40-ø40" -> "ø65"
    """
    if not size_text or pd.isna(size_text):
        return ''

    size_str = str(size_text)

    # Split by common delimiters
    parts = re.split(r'[-,;/]', size_str)

    max_size = 0
    max_size_str = ''

    for part in parts:
        # Extract numeric value from each part
        numeric_match = re.search(r'[øØ∅]?\s*(\d+(?:\.\d+)?)', part)
        if numeric_match:
            size_value = float(numeric_match.group(1))
            if size_value > max_size:
                max_size = size_value
                # Preserve the original format with symbol
                symbol_match = re.search(r'([øØ∅])', part)
                if symbol_match:
                    max_size_str = f"{symbol_match.group(1)}{int(size_value)}"
                else:
                    max_size_str = str(int(size_value))

    return max_size_str

def lookup_material_mapping(material_text, materials_lookup_df):
    """
    Look up material mapping from materials lookup table
    """
    if materials_lookup_df is None or not material_text or pd.isna(material_text):
        return ''

    try:
        material_lower = str(material_text).lower()

        # Check for columns that might contain material mappings
        source_columns = ['Material Name', 'Source Material', 'Original', 'Input']
        target_columns = ['Mapped Material', 'Target Material', 'Output', 'Standard Material']

        source_col = None
        target_col = None

        # Find the source and target columns
        for col in materials_lookup_df.columns:
            if col in source_columns or 'name' in col.lower() or 'source' in col.lower():
                source_col = col
            if col in target_columns or 'mapped' in col.lower() or 'target' in col.lower():
                target_col = col

        # If we couldn't identify columns, try using first two columns
        if source_col is None and len(materials_lookup_df.columns) >= 2:
            source_col = materials_lookup_df.columns[0]
            target_col = materials_lookup_df.columns[1]

        if source_col and target_col:
            # Look for exact match first
            for _, row in materials_lookup_df.iterrows():
                if pd.notna(row[source_col]) and str(row[source_col]).lower() == material_lower:
                    if pd.notna(row[target_col]):
                        return str(row[target_col])

            # Try partial match
            for _, row in materials_lookup_df.iterrows():
                if pd.notna(row[source_col]) and str(row[source_col]).lower() in material_lower:
                    if pd.notna(row[target_col]):
                        return str(row[target_col])

    except Exception as e:
        pass

    return ''

def validate_element_type_match(category, data_row, is_wall, is_ceiling, is_floor):
    """
    Validate that the data row is appropriate for the element type
    """
    item_type = ''
    if hasattr(data_row, 'get'):
        item_type = str(data_row.get('Item→Type', '')).lower()
    elif 'Item→Type' in data_row.index:
        item_type = str(data_row['Item→Type']).lower()

    if not item_type:
        return True

    # Define indicators for different element types
    wall_indicators = ['wall', 'ext-', 'int-', 'fyreline', 'plasterboard', 'gypsum', 
                      'stud', 'party', 'external', 'internal']
    ceiling_indicators = ['ceiling', 'clg', 'soffit', 'suspended']
    floor_indicators = ['floor', 'flr', 'slab', 'deck', 'concrete']

    # Check what type the data actually represents
    has_wall_data = any(indicator in item_type for indicator in wall_indicators)
    has_ceiling_data = any(indicator in item_type for indicator in ceiling_indicators)
    has_floor_data = any(indicator in item_type for indicator in floor_indicators)

    # Validation rules
    if is_ceiling:
        if has_wall_data and not has_ceiling_data:
            return False
        return has_ceiling_data or (not has_wall_data and not has_floor_data)

    elif is_floor:
        if has_wall_data and not has_floor_data:
            return False
        return has_floor_data or (not has_wall_data and not has_ceiling_data)

    elif is_wall:
        if (has_ceiling_data or has_floor_data) and not has_wall_data:
            return False
        return True

    return True

def process_passive_fire_schedule(excel_file_content, csv_file_content, 
                                 size_lookup_content=None, size_lookup_type=None,
                                 materials_lookup_content=None, materials_lookup_type=None):
    """
    Process passive fire schedule data with dual lookup table support
    """

    try:
        # Read the Excel file with multi-sheet detection
        excel_df = load_excel_with_sheet_selection(excel_file_content)
        if excel_df is None:
            return None
        st.success(f"✔ Excel file loaded: {len(excel_df)} rows")

        # Read the CSV file
        csv_df = pd.read_csv(io.BytesIO(csv_file_content))
        st.success(f"✔ CSV file loaded: {len(csv_df)} rows")

        # Load optional lookup tables
        size_lookup_df = None
        if size_lookup_content and size_lookup_type:
            size_lookup_df = load_lookup_table(size_lookup_content, size_lookup_type, "Size Lookup Table")
            if size_lookup_df is not None:
                st.success(f"✔ Size Lookup Table loaded: {len(size_lookup_df)} rows")

        materials_lookup_df = None
        if materials_lookup_content and materials_lookup_type:
            materials_lookup_df = load_lookup_table(materials_lookup_content, materials_lookup_type, "Materials Lookup Table")
            if materials_lookup_df is not None:
                st.success(f"✔ Materials Lookup Table loaded: {len(materials_lookup_df)} rows")

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

            # Extract service type from first part
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

            # Add System Classification
            if pd.notna(matching_row.get('MECHANICAL→System Classification')):
                service_desc += str(matching_row['MECHANICAL→System Classification']) + ' '

            # Add Insulation Thickness
            insul_thickness = matching_row.get('INSULATION→Insulation Thickness mm')
            if pd.notna(insul_thickness) and insul_thickness > 0:
                service_desc += str(int(insul_thickness)) + 'mm '

            # Add Item Type or Item Name
            item_type = matching_row.get('Item→Type')
            item_name = matching_row.get('Item→Name')

            if pd.notna(item_type) and not str(item_type).lower().startswith('standard'):
                service_desc += str(item_type)
            elif pd.notna(item_name):
                service_desc += str(item_name)

            # Add Insulation Type if exists
            insul_type = matching_row.get('INSULATION→Insulation Type')
            if pd.notna(insul_type):
                thickness_val = insul_thickness if pd.notna(insul_thickness) else ''
                service_desc += f" with {thickness_val}mm {insul_type}"

            return service_desc.strip()

        except Exception as e:
            return ''

    def get_service_material_enhanced(element_id, service_text, csv_data, materials_lookup=None):
        """
        Get service material with keyword detection and lookup table support
        Priority: 1. Keyword detection, 2. CSV data, 3. Materials lookup table
        """
        # First, try keyword detection on the service text
        keyword_material = classify_material_by_keywords(service_text)
        if keyword_material:
            return keyword_material

        # If no element_id, skip CSV lookup
        if pd.isna(element_id):
            return ''

        # Try to get from CSV
        matching_rows = csv_data[csv_data['Revizto→Authoring Tool Id'] == element_id]
        if not matching_rows.empty:
            matching_row = matching_rows.iloc[0]

            # Try Column K first (MECHANICAL→Material)
            mechanical_material = matching_row.get('MECHANICAL→Material')
            if pd.notna(mechanical_material):
                # Apply keyword classification
                classified = classify_material_by_keywords(mechanical_material)
                if classified:
                    return classified

                # Try materials lookup
                if materials_lookup is not None:
                    mapped = lookup_material_mapping(mechanical_material, materials_lookup)
                    if mapped:
                        return mapped

                return str(mechanical_material)

            # Try Column L (MATERIALS→Structural Material)
            structural_material = matching_row.get('MATERIALS→Structural Material')
            if pd.notna(structural_material):
                # Apply keyword classification
                classified = classify_material_by_keywords(structural_material)
                if classified:
                    return classified

                # Try materials lookup
                if materials_lookup is not None:
                    mapped = lookup_material_mapping(structural_material, materials_lookup)
                    if mapped:
                        return mapped

                return str(structural_material)

        return ''

    def get_pipe_size_enhanced(element_id, csv_data, service_text, size_lookup=None):
        """
        Get pipe size with enhanced extraction logic
        Extracts largest value when multiple sizes are present
        """
        if pd.isna(element_id):
            return ''

        # Check if this is a structural element that needs size lookup
        if service_text and 'Structural' in service_text and size_lookup is not None:
            # Extract steel name from service text (e.g., "Structural - 410UB60")
            parts = service_text.split('-')
            if len(parts) >= 2:
                # take the second part and split out first token-like element
                steel_token = parts[1].strip()
                steel_name_candidate = re.split(r'[\s,;/\(\)]+', steel_token)[0]
                # pass to lookup routine (which will call generate_name_variants())
                size_str, _ = lookup_steel_properties(steel_name_candidate, size_lookup)
                if size_str:
                    return size_str

        # Regular pipe size extraction from CSV
        matching_rows = csv_data[csv_data['Revizto→Authoring Tool Id'] == element_id]
        if matching_rows.empty:
            return ''

        matching_row = matching_rows.iloc[0]
        geometry_size = matching_row.get('GEOMETRY→Size')

        if pd.notna(geometry_size):
            # Apply enhanced extraction logic
            return extract_largest_pipe_size(geometry_size)

        return ''

    def extract_element_id(title):
        """Extract element ID from title for lookups"""
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
                return parts[1].strip()
        except:
            pass
        return None

    def get_frr_info(ceiling_info, csv_data):
        """Get FRR information from CSV"""
        if not ceiling_info:
            return ''

        try:
            # Clean ceiling info for matching
            clean_info = ceiling_info
            element_id = extract_element_id_from_ceiling_info(ceiling_info)
            if element_id:
                clean_info = re.sub(r'-\s*\d{6,}\s*(?:\||$)', '', ceiling_info).strip()

            # Find matching rows
            matching_rows = csv_data[csv_data['Item→Type'].str.contains(clean_info, case=False, na=False)]

            if matching_rows.empty:
                words = clean_info.split()
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
                unique_values = set(str(v) for v in frr_values)
                if len(unique_values) == 1:
                    return str(frr_values[0])
                else:
                    return f"WARNING: Multiple FRR values found: {', '.join(str(v) for v in frr_values)}"

        except Exception as e:
            return ''

    def get_separating_element_enhanced(ceiling_info, csv_data):
        """Get separating element with strict type validation"""
        if not ceiling_info:
            return ''

        try:
            # Extract element ID if present
            element_id = extract_element_id_from_ceiling_info(ceiling_info)

            # Clean ceiling info
            clean_ceiling_info = ceiling_info
            if element_id:
                clean_ceiling_info = re.sub(r'-\s*\d{6,}\s*(?:\||$)', '', ceiling_info).strip()

            # Parse category and details
            category = ''
            element_details = ''

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

            # For ceilings and floors, don't use wall data
            if is_ceiling or is_floor:
                # Clean up any wall-related terms
                cleaned_info = clean_ceiling_info
                wall_terms = ['Fyreline', 'EXT-', 'INT-', 'Plasterboard', 'Gypsum']
                for term in wall_terms:
                    cleaned_info = re.sub(f'\\s*-?\\s*{re.escape(term)}.*$', '', cleaned_info, flags=re.IGNORECASE)
                return cleaned_info.strip()

            # For walls, find and use wall data
            if is_wall:
                matching_rows = csv_data[csv_data['Item→Type'].str.contains(clean_ceiling_info, case=False, na=False)]
                if not matching_rows.empty:
                    matching_row = matching_rows.iloc[0]
                    return format_wall_description(category, matching_row)

            return ceiling_info

        except Exception as e:
            return ceiling_info

    def format_wall_description(category, data_row):
        """Format wall description with wall-specific columns"""
        parts = [category]

        wall_system = data_row.get('Revit Type→WALL SYSTEM', '')
        wall_framing = data_row.get('Revit Type→WALL FRAMING', '')
        wall_lining = data_row.get('Revit Type→WALL LINING', '')

        if pd.notna(wall_system) and wall_system:
            parts.append(str(wall_system))

        if pd.notna(wall_framing) and wall_framing:
            parts.append(str(wall_framing))

        result = ' - '.join(parts)

        if pd.notna(wall_lining) and wall_lining:
            result += f" with {wall_lining}"

        return result

    # Process all data with priority processing
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

        # Get service material with enhanced logic
        service_material = get_service_material_enhanced(
            element_id, service, csv_df, materials_lookup_df
        )

        # Get pipe size with enhanced logic
        pipe_size = get_pipe_size_enhanced(
            element_id, csv_df, service, size_lookup_df
        )

        # Extract ceiling/wall info
        ceiling_info = extract_ceiling_info(title)

        # Get FRR and separating element information
        frr_info = get_frr_info(ceiling_info, csv_df)
        separating_element = get_separating_element_enhanced(ceiling_info, csv_df)

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

    # Second pass: Fill empty cells using lookup tables if available
    if size_lookup_df is not None or materials_lookup_df is not None:
        st.info("🔄 Performing second pass to fill empty cells using lookup tables...")

        for index, row in result_df.iterrows():
            # Fill empty Service Material using materials lookup
            if materials_lookup_df is not None and (pd.isna(row['Service Material']) or row['Service Material'] == ''):
                # Try to extract material from Service column
                service_text = row['Service']
                if service_text:
                    material = classify_material_by_keywords(service_text)
                    if material:
                        result_df.at[index, 'Service Material'] = material

            # Fill empty Pipe Size using size lookup for structural elements
            if size_lookup_df is not None and (pd.isna(row['Pipe Size(OD)']) or row['Pipe Size(OD)'] == ''):
                service_text = row['Service']
                if service_text and 'Structural' in service_text:
                    parts = service_text.split('-')
                    if len(parts) >= 2:
                        steel_token = parts[1].strip()
                        steel_name_candidate = re.split(r'[\s,;/\(\)]+', steel_token)[0]
                        size_str, material = lookup_steel_properties(steel_name_candidate, size_lookup_df)
                        if size_str:
                            result_df.at[index, 'Pipe Size(OD)'] = size_str
                        if material and (pd.isna(row['Service Material']) or row['Service Material'] == ''):
                            result_df.at[index, 'Service Material'] = material

    return result_df

def export_dataframe(df, format_type, filename):
    """Export DataFrame to various formats"""

    if format_type == 'XLSX':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Passive Fire Schedule', index=False)
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
    st.markdown('<h1 class="main-header">🔥 Passive Fire Schedule Processor v3.0</h1>', unsafe_allow_html=True)
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

        # Processing options
        with st.expander("⚙️ Processing Options", expanded=False):
            st.markdown("""
            **Material Classification Keywords:**
            - Steel: 'steel', 'gal', 'fp_gal'
            - PVC: 'pvc', 'plastic'
            - PP-R: 'pp-r', 'ppr', 'raufusion'
            - Copper: 'copper'

            **Size Extraction:**
            - Automatically extracts largest value
            - Handles multiple delimiters (-, /, ,)
            - Preserves diameter symbols (Ø, ø)
            """)

        st.markdown("---")

        # Help section
        with st.expander("📖 How to Use", expanded=False):
            st.markdown("""
            **Step 1:** Upload required files
            - Excel file (Revizto export)
            - CSV file (Component database)

            **Step 2:** Upload optional lookup tables
            - Size Lookup Table (for structural elements)
            - Materials Lookup Table (for material mapping)

            **Step 3:** Click 'Process Files' button

            **Step 4:** Download your processed schedule

            **Enhanced Features:**
            - Automatic second sheet selection for multi-sheet files
            - Keyword-based material classification
            - Intelligent size extraction (largest value)
            - Dual lookup table support
            - Two-pass processing for data completeness
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
            help="Multi-sheet files will use sheet 2 automatically"
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

    # Optional lookup tables section
    st.subheader("📁 Optional Lookup Tables")

    # Create two columns for the lookup tables
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="lookup-section">', unsafe_allow_html=True)
        st.markdown("### 📏 Size Lookup Table")
        st.markdown("*For structural element dimensions*")

        size_lookup_file = st.file_uploader(
            "Choose size lookup file",
            type=['csv', 'xlsx', 'xls'],
            key="size_lookup_upload",
            help="Upload CSV or Excel file with steel/structural specifications"
        )

        if size_lookup_file:
            st.success(f"✔ Size lookup: {size_lookup_file.name}")
            size_lookup_type = 'csv' if size_lookup_file.name.endswith('.csv') else 'excel'

            with st.expander("ℹ️ Size Lookup Info", expanded=False):
                st.markdown("""
                **Expected columns:**
                - Steel Name / Item Name
                - Overall Width / Width
                - Overall Height / Height
                - Material / Type

                **Usage:**
                - Structural items: "410UB60" → "410x60"
                - Includes material properties
                """)
        else:
            size_lookup_type = None
            st.info("No size lookup table uploaded")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="lookup-section">', unsafe_allow_html=True)
        st.markdown("### 🔧 Materials Lookup Table")
        st.markdown("*For material classification mapping*")

        materials_lookup_file = st.file_uploader(
            "Choose materials lookup file",
            type=['csv', 'xlsx', 'xls'],
            key="materials_lookup_upload",
            help="Upload CSV or Excel file with material mappings"
        )

        if materials_lookup_file:
            st.success(f"✔ Materials lookup: {materials_lookup_file.name}")
            materials_lookup_type = 'csv' if materials_lookup_file.name.endswith('.csv') else 'excel'

            with st.expander("ℹ️ Materials Lookup Info", expanded=False):
                st.markdown("""
                **Expected columns:**
                - Material Name / Source Material
                - Mapped Material / Target Material

                **Mapping examples:**
                - Plastic → PVC
                - RAUFUSION → PP-R
                - FP_GAL → Steel
                - GAL → Steel
                """)
        else:
            materials_lookup_type = None
            st.info("No materials lookup table uploaded")

        st.markdown('</div>', unsafe_allow_html=True)

    # Process button
    if excel_file and csv_file:
        st.markdown("---")

        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            with st.spinner("Processing your files... This may take a few minutes."):
                try:
                    # Prepare lookup table data
                    size_lookup_content = None
                    if size_lookup_file:
                        size_lookup_content = size_lookup_file.read()

                    materials_lookup_content = None
                    if materials_lookup_file:
                        materials_lookup_content = materials_lookup_file.read()

                    # Process the files
                    result_df = process_passive_fire_schedule(
                        excel_file.read(),
                        csv_file.read(),
                        size_lookup_content,
                        size_lookup_type,
                        materials_lookup_content,
                        materials_lookup_type
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
                            lookup_count = sum([1 for f in [size_lookup_file, materials_lookup_file] if f])
                            st.metric("Lookup Tables", lookup_count, "Used" if lookup_count > 0 else "None")

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

#                        # Comment out material distribution for now
#                        material_counts = result_df['Service Material'].value_counts()
#                        if not material_counts.empty:
#                            st.subheader("🔧 Material Distribution")
#                            material_df = pd.DataFrame({
#                                'Material': material_counts.index,
#                                'Count': material_counts.values,
#                                'Percentage': (material_counts.values / len(result_df) * 100).round(1)
#                            })
#                            st.bar_chart(material_df.set_index('Material')['Count'])

                        # ✅ New: Size Distribution
                        size_counts = result_df['Pipe Size(OD)'].value_counts()
                        if not size_counts.empty:
                            st.subheader("📏 Size Distribution")
                            size_df = pd.DataFrame({
                                'Size': size_counts.index,
                                'Count': size_counts.values,
                                'Percentage': (size_counts.values / len(result_df) * 100).round(1)
                            })
                            st.bar_chart(size_df.set_index('Size')['Count'])


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
        "🔥 Passive Fire Schedule Processor v3.0 | Enhanced with Dual Lookup Tables & Smart Processing"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()