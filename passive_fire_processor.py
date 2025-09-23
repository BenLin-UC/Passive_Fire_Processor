import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from datetime import datetime
import io
import base64
from typing import List, Dict, Tuple, Optional, Union
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Passive Fire Schedule Processor v4.3",
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
# ENHANCED DATA STRUCTURES & CONFIG
# ============================

class ColumnMapping:
    """Configurable column mapping for different data sources"""
    
    # STANDARDIZED CSV Column Names (these will be the target names after renaming)
    # The order/sequence corresponds to expected column positions 0-12
    csv_standardized_columns = [
        'Item_Name',                    # Position 0
        'Item_Type',                    # Position 1  
        'Element_ID',                   # Position 2
        'Fire_Rating_FRR',              # Position 3
        'Wall_System',                  # Position 4
        'Wall_Framing',                 # Position 5
        'Wall_Lining',                  # Position 6
        'Insulation_Thickness_mm',      # Position 7
        'Geometry_Size',                # Position 8
        'System_Classification',        # Position 9
        'Mechanical_Material',          # Position 10
        'Structural_Material',          # Position 11
        'Insulation_Type'               # Position 12
    ]
    
    # After standardization, these are the column names we'll use throughout the script
    csv_item_name: str = 'Item_Name'
    csv_item_type: str = 'Item_Type' 
    csv_id_column: str = 'Element_ID'
    csv_frr: str = 'Fire_Rating_FRR'
    csv_wall_system: str = 'Wall_System'
    csv_wall_framing: str = 'Wall_Framing'
    csv_wall_lining: str = 'Wall_Lining'
    csv_insulation_thickness: str = 'Insulation_Thickness_mm'
    csv_size: str = 'Geometry_Size'
    csv_system_classification: str = 'System_Classification'
    csv_mechanical_material: str = 'Mechanical_Material'
    csv_structural_material: str = 'Structural_Material'
    csv_insulation_type: str = 'Insulation_Type'
    
    # Size Lookup Table Columns
    size_lookup_name_cols: List[str] = ['Steel Name', 'Name', 'Steel', 'Item Name']
    size_lookup_width_cols: List[str] = ['Overall Width', 'Width', 'width']
    size_lookup_height_cols: List[str] = ['Overall Height', 'Height', 'height']
    size_lookup_material_cols: List[str] = ['Materials', 'Material', 'Type']
    
    # Materials Lookup Table Columns  
    materials_lookup_source_cols: List[str] = ['Material Name', 'Source Material', 'Original', 'Input']
    materials_lookup_target_cols: List[str] = ['Materials', 'Mapped Material', 'Target Material', 'Output', 'Standard Material']

# Global column mapping configuration
COLUMN_MAPPING = ColumnMapping()

# ============================
# SIMPLIFIED DATA RETRIEVAL ENGINE
# ============================

def similarity_score(a: str, b: str) -> float:
    """Calculate similarity score between two strings (0.0 to 1.0)"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def unified_data_retrieval(
    data_source: pd.DataFrame,
    element_ids: List[Union[int, str]],
    string_identifiers: List[str],
    target_columns: List[str],
    id_column: str,
    confidence_threshold: float = 0.75,
    enable_string_fallback: bool = True
) -> str:
    """
    Simplified data retrieval engine - returns value directly
    """
    if data_source is None or data_source.empty:
        return ''
    
    # Phase 1: Try exact ID matches
    if element_ids and id_column in data_source.columns:
        for element_id in element_ids:
            try:
                if isinstance(element_id, (int, float)) and not pd.isna(element_id):
                    matches = data_source[data_source[id_column] == element_id]
                elif isinstance(element_id, str) and element_id.strip():
                    # Try both exact match and contains for string IDs
                    exact_matches = data_source[data_source[id_column].astype(str) == str(element_id)]
                    if not exact_matches.empty:
                        matches = exact_matches
                    else:
                        matches = data_source[data_source[id_column].astype(str).str.contains(str(element_id), case=False, na=False)]
                else:
                    continue
                    
                if not matches.empty:
                    match_row = matches.iloc[0]
                    # Try each target column in order
                    for col in target_columns:
                        if col in match_row.index and pd.notna(match_row[col]) and str(match_row[col]).strip():
                            return str(match_row[col]).strip()
            except Exception:
                continue
    
    # Phase 2: Try string identifier exact matches
    if enable_string_fallback and string_identifiers:
        search_columns = [col for col in [COLUMN_MAPPING.csv_item_type, COLUMN_MAPPING.csv_item_name] 
                         if col in data_source.columns]
        
        for identifier in string_identifiers:
            if not identifier or not identifier.strip():
                continue
                
            identifier = identifier.strip()
            
            for search_col in search_columns:
                # Exact string matches
                exact_matches = data_source[
                    data_source[search_col].astype(str).str.contains(
                        re.escape(identifier), case=False, na=False, regex=True
                    )
                ]
                
                for _, match_row in exact_matches.iterrows():
                    # Calculate confidence based on string similarity
                    search_value = str(match_row[search_col]).strip()
                    conf = similarity_score(identifier, search_value)
                    
                    if conf >= confidence_threshold:
                        # Try each target column
                        for col in target_columns:
                            if col in match_row.index and pd.notna(match_row[col]) and str(match_row[col]).strip():
                                return str(match_row[col]).strip()
    
    return ''

# ============================
# ENHANCED CSV COLUMN STANDARDIZATION
# ============================

def standardize_csv_columns_by_position(df):
    """
    ENHANCED: Standardize CSV column names by position regardless of original names
    This ensures consistent column names across different projects
    """
    if df is None or df.empty:
        return df
    
    # Create a mapping from current position to standardized name
    column_mapping = {}
    standardized_df = df.copy()
    
    # Map columns by position to standardized names
    num_cols_to_rename = min(len(df.columns), len(COLUMN_MAPPING.csv_standardized_columns))
    
    for i in range(num_cols_to_rename):
        original_name = df.columns[i]
        standardized_name = COLUMN_MAPPING.csv_standardized_columns[i]
        column_mapping[original_name] = standardized_name
    
    # Apply the renaming
    standardized_df = standardized_df.rename(columns=column_mapping)
    
    # Report the standardization
    st.success(f"✅ CSV columns standardized by position! Renamed {len(column_mapping)} columns.")
    
    with st.expander("📊 Column Standardization Report", expanded=False):
        st.write("**Position-Based Column Mapping Applied:**")
        for i, (original, standardized) in enumerate(column_mapping.items()):
            st.write(f"Position {i}: `{original}` → `{standardized}`")
        
        if len(df.columns) > len(COLUMN_MAPPING.csv_standardized_columns):
            st.warning(f"⚠️ Found {len(df.columns) - len(COLUMN_MAPPING.csv_standardized_columns)} extra columns beyond position 12. These will be preserved with original names.")
        
        st.info("💡 This position-based mapping ensures the script works with different column naming conventions across projects.")
    
    return standardized_df

# ============================
# ENHANCED ELEMENT ID EXTRACTION
# ============================

def extract_element_ids(title, debug=False):
    """
    Enhanced element ID extraction with support for both numeric and string identifiers
    """
    if pd.isna(title) or not title:
        return []

    element_ids = []
    string_identifiers = []
    
    try:
        if debug: 
            st.write(f"DEBUG: Processing title: {title}")
        
        # Split by "|" and check each part for element IDs
        parts = title.split('|')
        
        for i, part in enumerate(parts):
            # Look for numeric IDs (6+ digits)
            id_matches = re.findall(r'-\s*(\d{6,})(?:\s*\||$)', part)
            element_ids.extend([int(match) for match in id_matches])
            
            # Also check for IDs before brackets
            bracket_matches = re.findall(r'(\d{6,})\s*\[', part)
            element_ids.extend([int(match) for match in bracket_matches])
            
            # Extract string identifiers for fallback - ENHANCED for steel names
            # Look for steel/structural patterns in the part
            steel_patterns = [
                r'\b(\d+\s*(?:UB|UC)\s*\d*)\b',      # 410UB60, 360UB57
                r'\b(\d+\s*DHS(?:\s*Purlin)?)\b',    # 200 DHS, 200 DHS Purlin  
                r'\b(\d+x\d+x\d+\s*(?:SHS|RHS))\b', # Structural sections
                r'\b(\d+x\d+\s*(?:SHS|RHS))\b',     # Square/rectangular sections
            ]
            
            for pattern in steel_patterns:
                matches = re.findall(pattern, part, re.IGNORECASE)
                for match in matches:
                    clean_match = re.sub(r'\s+', ' ', match.strip())
                    if clean_match and len(clean_match) > 2:
                        string_identifiers.append(clean_match)
            
            # Also extract general meaningful descriptive parts
            string_match = re.search(r'-\s*([^-\|]+?)\s*-\s*\d{6,}', part)
            if string_match:
                string_id = string_match.group(1).strip()
                if string_id and len(string_id) > 3 and not re.match(r'^\d+$', string_id):
                    # Don't add if it's already a steel pattern we caught above
                    if not any(steel_pattern in string_id.upper() for steel_pattern in ['UB', 'UC', 'DHS', 'SHS', 'RHS']):
                        string_identifiers.append(string_id)
            
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for id_val in element_ids:
            if id_val not in seen:
                seen.add(id_val)
                unique_ids.append(id_val)
        
        seen_strings = set()
        unique_strings = []
        for string_id in string_identifiers:
            string_clean = string_id.strip()
            if string_clean.upper() not in seen_strings and len(string_clean) > 2:
                seen_strings.add(string_clean.upper())
                unique_strings.append(string_clean)
        
        # Combine numeric IDs (priority) with string identifiers
        all_identifiers = unique_ids + unique_strings
        
        if debug:
            st.write(f"DEBUG: Found numeric IDs: {unique_ids}")
            st.write(f"DEBUG: Found string identifiers: {unique_strings}")
            
        return all_identifiers
        
    except Exception as e:
        if debug:
            st.write(f"DEBUG: Exception: {e}")
        return []

# ============================
# ENHANCED SIZE EXTRACTION WITH RECTANGULAR PRESERVATION
# ============================

def extract_largest_pipe_size(size_text):
    """
    FIXED: Enhanced size extraction that preserves rectangular dimensions
    """
    if not size_text or pd.isna(size_text):
        return ''

    size_str = str(size_text).strip()
    
    # First check for rectangular dimensions (preserve as-is)
    rect_patterns = [
        r'(\d+\s*x\s*\d+)(?:\s*mm)?',  # 450x450, 762x170
        r'(\d+\s*×\s*\d+)(?:\s*mm)?'   # Alternative multiplication symbol
    ]
    
    for pattern in rect_patterns:
        rect_matches = re.findall(pattern, size_str, re.IGNORECASE)
        if rect_matches:
            # Return the largest rectangular dimension (by area)
            max_area = 0
            max_rect = ''
            
            for rect in rect_matches:
                dims = re.findall(r'\d+', rect)
                if len(dims) >= 2:
                    area = int(dims[0]) * int(dims[1])
                    if area > max_area:
                        max_area = area
                        max_rect = f"{dims[0]}x{dims[1]}"
            
            if max_rect:
                return max_rect
    
    # If no rectangular dimensions, handle as before but with improved logic
    # Split by delimiters but be more careful
    parts = re.split(r'[-,;/]', size_str)
    
    max_size = 0
    max_size_str = ''

    for part in parts:
        # Extract numeric value with diameter symbol
        diameter_match = re.search(r'([øØ∅]?)\s*(\d+(?:\.\d+)?)', part)
        if diameter_match:
            symbol, value_str = diameter_match.groups()
            size_value = float(value_str)
            
            if size_value > max_size:
                max_size = size_value
                symbol = symbol or ''
                max_size_str = f"{symbol}{int(size_value)}"

    return max_size_str

# ============================
# ENHANCED STEEL PROPERTIES LOOKUP
# ============================

def generate_name_variants(name: str):
    """
    Generate steel name variants for lookup - enhanced version for "200 DHS Purlin" etc.
    """
    if not name or pd.isna(name):
        return []

    s = str(name).strip()
    canonical = re.sub(r'\s+', '', s).upper()
    canonical = canonical.replace('×', 'X')
    
    variants = set([s, canonical])

    # Enhanced DHS patterns - handle "200 DHS Purlin" better
    dhs_patterns = [
        r"(\d+)\s*DHS\s*PURLIN",  # "200 DHS Purlin"
        r"(\d+)\s*DHS",           # "200 DHS"
        r"DHS\s*(\d+)",           # "DHS 200"
    ]
    
    for pattern in dhs_patterns:
        match = re.search(pattern, canonical, re.IGNORECASE)
        if match:
            depth = match.group(1)
            variants.update({
                f"{depth} DHS",
                f"{depth}DHS", 
                f"DHS {depth}",
                f"DHS{depth}",
                f"{depth} DHS Purlin",
                f"{depth}DHS Purlin",
                f"DHS {depth} Purlin"
            })

    # UB/UC patterns
    ub_match = re.search(r'(\d+)(UB|UC)(\d+)', canonical)
    if ub_match:
        depth, typ, weight = ub_match.groups()
        variants.update({
            f"{depth}{typ}{weight}",
            f"{depth} {typ} {weight}",
            f"{depth}{typ} {weight}",
            f"{depth} {typ}{weight}"
        })

    # SHS/RHS patterns
    shs_match = re.match(r'^(\d+)X(\d+)(SHS|RHS)$', canonical)
    if shs_match:
        a, b, shape = shs_match.groups()
        variants.update({
            f"{a}x{b}{shape}",
            f"{a} x {a} x {b} {shape}",
            f"{a}x{a}x{b} {shape}",
            f"{a} x {b} {shape}",
            f"{a}x{b} {shape}"
        })

    # Add spaced variants
    spaced = re.sub(r'([A-Z])', r' \1', canonical).strip()
    spaced = re.sub(r'\s+', ' ', spaced)
    variants.add(spaced)

    # Normalize and clean variants
    final_variants = set()
    for v in variants:
        if v and v.strip():
            clean_v = re.sub(r'\s+', ' ', v.strip())
            final_variants.add(clean_v)
            final_variants.add(clean_v.upper())
            final_variants.add(clean_v.title())

    return sorted(final_variants, key=lambda x: (-len(x), x))

def lookup_steel_properties(steel_name, size_lookup_df):
    """
    Enhanced steel properties lookup with better variant matching
    Returns: (size_string, material)
    """
    if size_lookup_df is None or steel_name is None or pd.isna(steel_name):
        return ('', '')

    try:
        # Get available name columns
        available_name_cols = []
        for col in COLUMN_MAPPING.size_lookup_name_cols:
            if col in size_lookup_df.columns:
                available_name_cols.append(col)
        
        if not available_name_cols:
            # Fallback to any column with 'name' in it
            available_name_cols = [c for c in size_lookup_df.columns if 'name' in c.lower()]
            
        if not available_name_cols:
            return ('', '')

        variants = generate_name_variants(steel_name)

        # 1) Try exact matches first
        for variant in variants:
            for col in available_name_cols:
                col_values = size_lookup_df[col].astype(str).fillna('').str.strip()
                exact_matches = size_lookup_df[col_values.str.lower() == variant.lower()]
                
                if not exact_matches.empty:
                    match_row = exact_matches.iloc[0]
                    size_str, material = extract_size_and_material_from_lookup(match_row)
                    return (size_str, material)

        # 2) Try partial matches for steel names like "200 DHS Purlin"
        for variant in variants:
            for col in available_name_cols:
                for _, row in size_lookup_df.iterrows():
                    lookup_value = str(row[col]) if pd.notna(row[col]) else ""
                    if not lookup_value.strip():
                        continue
                    
                    # Check if variant is contained in lookup value or vice versa
                    if (variant.lower() in lookup_value.lower() or 
                        lookup_value.lower() in variant.lower()):
                        size_str, material = extract_size_and_material_from_lookup(row)
                        if size_str or material:
                            return (size_str, material)

    except Exception as e:
        st.warning(f"Lookup error for '{steel_name}': {e}")

    return ('', '')

def extract_size_and_material_from_lookup(match_row):
    """Extract size and material information from lookup table row"""
    size_str = ''
    material = ''
    
    # Extract size information
    width_col = height_col = None
    for col in COLUMN_MAPPING.size_lookup_width_cols:
        if col in match_row.index:
            width_col = col
            break
    for col in COLUMN_MAPPING.size_lookup_height_cols:
        if col in match_row.index:
            height_col = col
            break
            
    if width_col and height_col:
        try:
            width_val = str(match_row[width_col]).replace('mm','').strip()
            height_val = str(match_row[height_col]).replace('mm','').strip()
            
            # Try to convert to numbers for proper formatting
            try:
                width_num = float(width_val)
                height_num = float(height_val)
                size_str = f"{int(round(height_num))}x{int(round(width_num))}"
            except:
                size_str = f"{height_val}x{width_val}"
        except:
            pass

    # Extract material information
    for col in COLUMN_MAPPING.size_lookup_material_cols:
        if col in match_row.index and pd.notna(match_row[col]):
            material = str(match_row[col])
            break

    return size_str, material

# ============================
# ENHANCED MATERIAL CLASSIFICATION
# ============================

def classify_material_by_keywords(text):
    """Enhanced material classification with more comprehensive keywords"""
    if not text or pd.isna(text):
        return ''

    text_lower = str(text).lower()

    # Enhanced keyword patterns with priority
    material_patterns = [
        (r'\b(?:steel|gal|fp_gal|metal-steel)\b', 'Steel'),
        (r'\b(?:pvc|plastic)\b', 'PVC'),
        (r'\b(?:pp-r|ppr|raufusion)\b', 'PP-R'),
        (r'\b(?:copper|cu)\b', 'Copper'),
        (r'\b(?:cast iron|iron)\b', 'Cast Iron'),
        (r'\b(?:aluminium|aluminum|al)\b', 'Aluminium'),
    ]
    
    for pattern, material in material_patterns:
        if re.search(pattern, text_lower):
            return material

    return ''

# ============================
# ENHANCED LOOKUP MATERIAL MAPPING
# ============================

def lookup_material_mapping(material_text, materials_lookup_df):
    """Enhanced material mapping lookup"""
    if materials_lookup_df is None or not material_text or pd.isna(material_text):
        return ''

    try:
        # Get source and target columns
        source_col = None
        target_col = None

        for col in COLUMN_MAPPING.materials_lookup_source_cols:
            if col in materials_lookup_df.columns:
                source_col = col
                break
        
        for col in COLUMN_MAPPING.materials_lookup_target_cols:
            if col in materials_lookup_df.columns:
                target_col = col
                break

        # Fallback to first two columns
        if source_col is None and len(materials_lookup_df.columns) >= 2:
            source_col = materials_lookup_df.columns[0]
            target_col = materials_lookup_df.columns[1]

        if not source_col or not target_col:
            return ''

        material_lower = str(material_text).lower().strip()

        # Try exact matches first
        for _, row in materials_lookup_df.iterrows():
            if pd.notna(row[source_col]):
                source_value = str(row[source_col]).lower().strip()
                
                # Exact match
                if source_value == material_lower:
                    if pd.notna(row[target_col]):
                        return str(row[target_col]).strip()

        # Try partial matches
        for _, row in materials_lookup_df.iterrows():
            if pd.notna(row[source_col]):
                source_value = str(row[source_col]).strip()
                confidence = similarity_score(material_text, source_value)
                
                if confidence >= 0.75:
                    if pd.notna(row[target_col]):
                        return str(row[target_col]).strip()

    except Exception as e:
        st.warning(f"Material mapping error: {e}")

    return ''

def standardize_materials_with_lookup(result_df, materials_lookup_df):
    """
    NEW v4.3: Final material standardization step using Materials Lookup table
    
    This function takes the populated Material column and cross-references
    each material with the Materials Lookup table to simplify/standardize the output.
    """
    if materials_lookup_df is None or result_df is None:
        return result_df
    
    try:
        standardized_materials = []
        material_mapping_stats = {'mapped': 0, 'unmapped': 0, 'empty': 0}
        
        for index, row in result_df.iterrows():
            original_material = str(row.get('Material', '')).strip()
            
            if not original_material:
                standardized_materials.append('')
                material_mapping_stats['empty'] += 1
                continue
            
            # Try to find a mapping in the Materials Lookup table
            mapped_material = lookup_material_mapping(original_material, materials_lookup_df)
            
            if mapped_material and mapped_material != original_material:
                standardized_materials.append(mapped_material)
                material_mapping_stats['mapped'] += 1
            else:
                # Keep original if no mapping found
                standardized_materials.append(original_material)
                material_mapping_stats['unmapped'] += 1
        
        # Update the DataFrame
        result_df['Material'] = standardized_materials
        
        # Report the standardization results
        total_processed = len(result_df)
        st.success(f"✅ Final material standardization completed!")
        
        with st.expander("📊 Material Standardization Report", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mapped Materials", material_mapping_stats['mapped'], 
                         f"{(material_mapping_stats['mapped']/total_processed*100):.1f}%")
            with col2:
                st.metric("Unchanged Materials", material_mapping_stats['unmapped'],
                         f"{(material_mapping_stats['unmapped']/total_processed*100):.1f}%")
            with col3:
                st.metric("Empty Materials", material_mapping_stats['empty'],
                         f"{(material_mapping_stats['empty']/total_processed*100):.1f}%")
            
            st.info("💡 This final step standardizes materials using the Materials Lookup table for consistent output.")
        
        return result_df
        
    except Exception as e:
        st.warning(f"Error in final material standardization: {e}")
        return result_df

# ============================
# FIXED SERVICE PARSING - Issue 1 Fix
# ============================

def parse_service(title, csv_data):
    """
    FIXED: Parse service information with specific part-based logic
    
    Logic:
    1. Extract discipline from FIRST part: "ARC (Fyreline) vs HYD" → "HYD" → "Hydraulic"
    2. Extract element ID from LAST part: "Pipe Fittings - Standard - 2570300 [173]" → "2570300"
    3. Use element ID to lookup System Classification and Item Type from CSV
    4. Format as: "{Discipline} - {System Classification} {Item Type}"
    """
    if pd.isna(title) or not title:
        return ''

    try:
        # Service type mapping
        service_lookup = {
            'FIR': 'Fire', 'ELE': 'Electric', 'HYD': 'Hydraulic',
            'MEC': 'Mechanical', 'STR': 'Structural'
        }

        # Split by "|"
        parts = title.split('|')
        if len(parts) < 3:  # Need at least 3 parts for this logic
            return ''

        # STEP 1: Extract discipline from FIRST part
        first_part = parts[0].strip()
        service_code = ''
        
        # Look for service codes anywhere in first part
        for code in service_lookup.keys():
            if re.search(rf'\b{code}\b', first_part):
                service_code = code
                break
        
        # Fallback: try last 3 chars of first part
        if not service_code:
            service_code = first_part[-3:]
        
        service_name = service_lookup.get(service_code, service_code)

        # STEP 2: Extract element ID from LAST part
        last_part = parts[-1].strip()  # Get the last part
        element_id = None
        
        # Look for element ID patterns in last part
        id_matches = re.findall(r'-\s*(\d{6,})(?:\s*\[|$)', last_part)
        if id_matches:
            element_id = int(id_matches[0])
        else:
            # Also check for IDs before brackets
            bracket_matches = re.findall(r'(\d{6,})\s*\[', last_part)
            if bracket_matches:
                element_id = int(bracket_matches[0])

        if not element_id:
            return f"{service_name} - "

        # STEP 3: Lookup System Classification and Item Type using element ID
        matches = csv_data[csv_data[COLUMN_MAPPING.csv_id_column] == element_id]
        if not matches.empty:
            match_row = matches.iloc[0]
            
            # Get System Classification
            system_classification = ''
            if (COLUMN_MAPPING.csv_system_classification in match_row.index and 
                pd.notna(match_row[COLUMN_MAPPING.csv_system_classification])):
                system_classification = str(match_row[COLUMN_MAPPING.csv_system_classification]).strip()
            
            # Get Item Type (from CSV, not from title parts)
            item_type = ''
            if (COLUMN_MAPPING.csv_item_type in match_row.index and 
                pd.notna(match_row[COLUMN_MAPPING.csv_item_type])):
                item_type = str(match_row[COLUMN_MAPPING.csv_item_type]).strip()
            
            # ENHANCED: If Item Type is "Standard" (case insensitive), use Item Name instead
            if item_type.lower() == 'standard':
                # Get Item Name as replacement
                if (COLUMN_MAPPING.csv_item_name in match_row.index and 
                    pd.notna(match_row[COLUMN_MAPPING.csv_item_name])):
                    item_name = str(match_row[COLUMN_MAPPING.csv_item_name]).strip()
                    # Only use Item Name if it's not generic like "Pipe Types"
                    if item_name and item_name.lower() != 'pipe types':
                        item_type = item_name
                    else:
                        item_type = ''  # Clear it if it's generic
            
            # STEP 4: Build service description: {Discipline} - {System Classification} {Item Type}
            service_desc = service_name
            if system_classification and item_type:
                service_desc += f" - {system_classification} {item_type}"
            elif system_classification:
                service_desc += f" - {system_classification}"
            elif item_type:
                service_desc += f" - {item_type}"
            else:
                service_desc += " - "
                
            return service_desc

        return f"{service_name} - "

    except Exception as e:
        return ''

# ============================
# FIXED FRR RETRIEVAL - Issue 2 Fix
# ============================

def get_frr_info(ceiling_info, csv_data, element_ids=None, confidence_threshold=0.8):
    """
    FIXED: Get FRR info using Item Type from second part of title
    
    Logic:
    1. ceiling_info comes from SECOND part: "Ceilings - 53-Clg_(CT10) 16mm Fyreline"
    2. Extract Item Type: "53-Clg_(CT10) 16mm Fyreline" (after first " - ")
    3. Search CSV where Item_Type matches this extracted value
    4. Return FRR value from matching row
    """
    
    if not ceiling_info:
        return ''

    try:
        # STEP 1: Extract Item Type from ceiling_info (which is the second part)
        # ceiling_info format: "Ceilings - 53-Clg_(CT10) 16mm Fyreline"
        # We want: "53-Clg_(CT10) 16mm Fyreline"
        
        item_type_to_search = ''
        if ' - ' in ceiling_info:
            # Split on first " - " and take everything after
            parts = ceiling_info.split(' - ', 1)
            if len(parts) > 1:
                item_type_to_search = parts[1].strip()
        else:
            # If no " - " found, use the whole string
            item_type_to_search = ceiling_info.strip()

        if not item_type_to_search:
            return ''

        # STEP 2: Search CSV for Item Type matches
        matching_rows = pd.DataFrame()
        
        # Try exact match first
        exact_matches = csv_data[csv_data[COLUMN_MAPPING.csv_item_type].astype(str).str.strip() == item_type_to_search]
        if not exact_matches.empty:
            matching_rows = exact_matches
        else:
            # Try partial match (contains)
            partial_matches = csv_data[csv_data[COLUMN_MAPPING.csv_item_type].astype(str).str.contains(
                re.escape(item_type_to_search), case=False, na=False, regex=True
            )]
            if not partial_matches.empty:
                matching_rows = partial_matches
            else:
                # Try word-by-word search for better matching
                words = item_type_to_search.split()
                for word in words:
                    if len(word) > 4:  # Only significant words
                        word_matches = csv_data[csv_data[COLUMN_MAPPING.csv_item_type].astype(str).str.contains(
                            re.escape(word), case=False, na=False, regex=True
                        )]
                        if not word_matches.empty:
                            matching_rows = word_matches
                            break

        # STEP 3: Get FRR value from matching rows
        if not matching_rows.empty:
            frr_values = matching_rows[COLUMN_MAPPING.csv_frr].dropna().unique()
            
            if len(frr_values) == 0:
                return ''
            elif len(frr_values) == 1:
                return str(frr_values[0])
            else:
                # Multiple values found - return first or show warning
                unique_values = set(str(v) for v in frr_values)
                if len(unique_values) == 1:
                    return str(frr_values[0])
                else:
                    return f"WARNING: Multiple FRR values found: {', '.join(str(v) for v in frr_values)}"

        # STEP 4: Fallback - try element ID lookup if Item Type search failed
        if element_ids:
            for element_id in element_ids:
                if isinstance(element_id, (int, float)):
                    matches = csv_data[csv_data[COLUMN_MAPPING.csv_id_column] == element_id]
                    if not matches.empty:
                        match_row = matches.iloc[0]
                        frr_value = match_row.get(COLUMN_MAPPING.csv_frr)
                        if pd.notna(frr_value) and frr_value:
                            return str(frr_value)

        return ''

    except Exception as e:
        return ''

# ============================
# ENHANCED MATERIAL RETRIEVAL
# ============================

def get_service_material_enhanced(element_ids, service_text, csv_data, 
                                 materials_lookup=None, size_lookup=None):
    """
    UPDATED: Material retrieval with CSV-first priority
    
    Priority 1: CSV Material Columns (Mechanical_Material & Structural_Material)
    Priority 2: Keyword Detection on Service Text (fallback only)
    """
    
    if not element_ids:
        return ''

    # Extract string identifiers from element_ids
    string_identifiers = [str(eid) for eid in element_ids if isinstance(eid, str)]

    # Priority 1A: CSV mechanical material lookup
    mechanical_material = unified_data_retrieval(
        csv_data, element_ids, string_identifiers,
        [COLUMN_MAPPING.csv_mechanical_material],
        COLUMN_MAPPING.csv_id_column, confidence_threshold=0.7
    )

    if mechanical_material:
        # Try materials lookup table mapping first
        if materials_lookup is not None:
            lookup_result = lookup_material_mapping(mechanical_material, materials_lookup)
            if lookup_result:
                return lookup_result
        
        # If no mapping found, return full value as-is
        return mechanical_material

    # Priority 1B: CSV structural material lookup  
    structural_material = unified_data_retrieval(
        csv_data, element_ids, string_identifiers,
        [COLUMN_MAPPING.csv_structural_material],
        COLUMN_MAPPING.csv_id_column, confidence_threshold=0.7
    )

    if structural_material:
        # Try materials lookup table mapping first
        if materials_lookup is not None:
            lookup_result = lookup_material_mapping(structural_material, materials_lookup)
            if lookup_result:
                return lookup_result
        
        # If no mapping found, return full value as-is
        return structural_material

    # Priority 2: Keyword detection on service text (FALLBACK ONLY - when no CSV material data)
    keyword_material = classify_material_by_keywords(service_text)
    if keyword_material:
        return keyword_material

    # Priority 3: Size lookup table material (for structural elements - as additional fallback)
    if size_lookup is not None and service_text and 'Structural' in service_text:
        # Extract steel name from service text
        parts = service_text.split('-')
        if len(parts) >= 2:
            steel_part = parts[1].strip()
            steel_name = steel_part.split()[0] if steel_part.split() else steel_part
            
            # For cases like "200 DHS Purlin", get the full description
            if 'DHS' in steel_part.upper():
                dhs_match = re.search(r'(\d+\s*DHS(?:\s*Purlin)?)', steel_part, re.IGNORECASE)
                if dhs_match:
                    steel_name = dhs_match.group(1).strip()
            
            _, material = lookup_steel_properties(steel_name, size_lookup)
            if material:
                return material

    return ''

# ============================
# ENHANCED SIZE RETRIEVAL
# ============================

def get_pipe_size_enhanced(element_ids, csv_data, service_text, size_lookup=None):
    """Enhanced pipe size retrieval with rectangular preservation and material integration"""
    
    if not element_ids:
        return '', ''

    # Extract string identifiers
    string_identifiers = [str(eid) for eid in element_ids if isinstance(eid, str)]

    # Priority 1: Size lookup for structural elements
    if service_text and 'Structural' in service_text and size_lookup is not None:
        parts = service_text.split('-')
        if len(parts) >= 2:
            # Enhanced steel name extraction for cases like "Structural - 200 DHS Purlin"
            steel_part = parts[1].strip()
            steel_name = steel_part.split()[0] if steel_part.split() else steel_part
            
            # For DHS elements, get the full description
            if 'DHS' in steel_part.upper():
                dhs_match = re.search(r'(\d+\s*DHS(?:\s*Purlin)?)', steel_part, re.IGNORECASE)
                if dhs_match:
                    steel_name = dhs_match.group(1).strip()
            
            size_str, material = lookup_steel_properties(steel_name, size_lookup)
            
            if size_str:
                return size_str, material

    # Priority 2: CSV geometry size lookup
    geometry_size = unified_data_retrieval(
        csv_data, element_ids, string_identifiers,
        [COLUMN_MAPPING.csv_size],
        COLUMN_MAPPING.csv_id_column, confidence_threshold=0.7
    )

    if geometry_size:
        # Apply enhanced size extraction
        extracted_size = extract_largest_pipe_size(geometry_size)
        if extracted_size:
            return extracted_size, ''

    return '', ''

# ============================
# SIMPLIFIED SEPARATING ELEMENT RETRIEVAL
# ============================

def get_separating_element_enhanced(ceiling_info, csv_data, element_ids=None):
    """Simplified separating element retrieval - just return the middle section from title"""
    
    if not ceiling_info:
        return ''

    try:
        # Clean ceiling info (remove element ID at the end)
        clean_ceiling_info = re.sub(r'-\s*\d{6,}\s*(?:\||$)', '', ceiling_info).strip()
        
        # Return the cleaned ceiling info as-is (the middle section of title split by "|")
        return clean_ceiling_info

    except Exception as e:
        return ceiling_info  # Fallback to original

# ============================
# DATA VISUALIZATION FUNCTIONS
# ============================

def create_frr_distribution(df):
    """Create FRR distribution visualization"""
    if df is None or df.empty:
        return None, None
    
    # Filter out empty FRR values
    frr_data = df[df['FRR'].str.len() > 0]['FRR'].copy()
    
    if frr_data.empty:
        return None, None
    
    # Get value counts
    frr_counts = frr_data.value_counts().sort_index()
    
    # Create table
    frr_table = pd.DataFrame({
        'FRR Rating': frr_counts.index,
        'Count': frr_counts.values,
        'Percentage': (frr_counts.values / len(frr_data) * 100).round(1)
    })
    
    # Create bar chart
    fig = px.bar(
        frr_table, 
        x='FRR Rating', 
        y='Count',
        labels={'Count': 'Number of Records'},
        text='Count'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_title='Count'
    )
    
    return frr_table, fig

def create_material_distribution(df):
    """Create Material distribution visualization"""
    if df is None or df.empty:
        return None, None
    
    # Filter out empty material values
    material_data = df[df['Material'].str.len() > 0]['Material'].copy()
    
    if material_data.empty:
        return None, None
    
    # Get value counts
    material_counts = material_data.value_counts()
    
    # Create table
    material_table = pd.DataFrame({
        'Material': material_counts.index,
        'Count': material_counts.values,
        'Percentage': (material_counts.values / len(material_data) * 100).round(1)
    })
    
    # Create bar chart
    fig = px.bar(
        material_table, 
        x='Material', 
        y='Count',
        labels={'Count': 'Number of Records'},
        text='Count'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_title='Count',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return material_table, fig

# ============================
# UTILITY FUNCTIONS
# ============================

def load_excel_with_sheet_selection(file_content):
    """Load Excel file with automatic sheet selection"""
    try:
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

# ============================
# MAIN PROCESSING FUNCTION
# ============================

def process_passive_fire_schedule(excel_file_content, csv_file_content, 
                                 size_lookup_content=None, size_lookup_type=None,
                                 materials_lookup_content=None, materials_lookup_type=None):
    """Enhanced passive fire schedule processing with FIXED Service & FRR logic"""

    try:
        # Read the Excel file
        excel_df = load_excel_with_sheet_selection(excel_file_content)
        if excel_df is None:
            return None
        st.success(f"✅ Excel file loaded: {len(excel_df)} rows")

        # Read the CSV file  
        csv_df = pd.read_csv(io.BytesIO(csv_file_content))
        st.success(f"✅ CSV file loaded: {len(csv_df)} rows")

        # Standardize CSV column names (position-based renaming)
        csv_df = standardize_csv_columns_by_position(csv_df)

        # Load optional lookup tables
        size_lookup_df = None
        if size_lookup_content and size_lookup_type:
            size_lookup_df = load_lookup_table(size_lookup_content, size_lookup_type, "Size Lookup Table")
            if size_lookup_df is not None:
                st.success(f"✅ Size Lookup Table loaded: {len(size_lookup_df)} rows")

        materials_lookup_df = None
        if materials_lookup_content and materials_lookup_type:
            materials_lookup_df = load_lookup_table(materials_lookup_content, materials_lookup_type, "Materials Lookup Table")
            if materials_lookup_df is not None:
                st.success(f"✅ Materials Lookup Table loaded: {len(materials_lookup_df)} rows")

    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None

    # Process all data with FIXED logic
    processed_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Show processing examples for first few records
    show_examples = True
    example_count = 0
    max_examples = 3

    for index, row in excel_df.iterrows():
        # Update progress
        progress = (index + 1) / len(excel_df)
        progress_bar.progress(progress)
        status_text.text(f"Processing record {index + 1} of {len(excel_df)}")

        revizto_id = row.get('ID', '')
        title = row.get('Title', '')

        # Show processing examples for verification
        if show_examples and example_count < max_examples and title and '|' in title:
            with st.expander(f"📝 Processing Example {example_count + 1}: {revizto_id}", expanded=False):
                st.write("**Original Title:**")
                st.code(title)
                
                parts = title.split('|')
                st.write("**Title Parts:**")
                for i, part in enumerate(parts):
                    st.write(f"Part {i+1}: `{part.strip()}`")
                
                # Show service parsing logic
                if len(parts) >= 3:
                    st.write("**Service Logic:**")
                    st.write(f"• Discipline from Part 1: `{parts[0].strip()}`")
                    st.write(f"• Element ID from Part {len(parts)}: `{parts[-1].strip()}`")
                    
                    # Show FRR logic
                    st.write("**FRR Logic:**")
                    st.write(f"• Item Type extraction from Part 2: `{parts[1].strip()}`")
                
                example_count += 1

        # Extract element IDs and string identifiers
        element_ids = extract_element_ids(title)
        
        # FIXED: Parse service information using new logic
        service = parse_service(title, csv_df)

        # Get Material 
        material = get_service_material_enhanced(
            element_ids, service, csv_df, 
            materials_lookup_df, size_lookup_df
        )

        # Get pipe size (returns size and material)
        size, size_material = get_pipe_size_enhanced(
            element_ids, csv_df, service, size_lookup_df
        )

        # Use size lookup material if main material retrieval failed
        if not material and size_material:
            material = size_material

        # Extract ceiling/wall info (middle section from title split by "|")
        ceiling_info = ''
        title_parts = title.split('|')
        if len(title_parts) >= 2:
            ceiling_info = title_parts[1].strip()

        # FIXED: Get FRR information using new logic
        frr = get_frr_info(ceiling_info, csv_df, element_ids, confidence_threshold=0.85)

        # Get separating element (just return the middle section)
        separating_element = get_separating_element_enhanced(ceiling_info, csv_df, element_ids)

        # Compile results
        processed_data.append({
            'Revizto ID': revizto_id,
            'Service': service or '',
            'Material': material or '',
            'Size': size or '',
            'FRR': frr or '',
            'Separating Element': separating_element or '',
            'Reference': title
        })

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Create final DataFrame
    result_df = pd.DataFrame(processed_data)

    # NEW v4.3: Apply final material standardization using Materials Lookup table
    if materials_lookup_df is not None:
        result_df = standardize_materials_with_lookup(result_df, materials_lookup_df)

    return result_df

def export_dataframe(df, format_type, filename):
    """Export DataFrame to various formats"""

    if format_type == 'XLSX':
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Passive Fire Schedule', index=False)
            worksheet = writer.sheets['Passive Fire Schedule']
            
            # Auto-adjust column widths
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
            <title>Passive Fire Schedule v4.2</title>
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
            <h1>Passive Fire Schedule v4.2</h1>
            <div class="summary">
                <strong>Enhanced Data Visualization:</strong><br>
                Total Records: {len(df)}<br>
                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                Features: FRR & Material Distribution Analysis
            </div>
            {df.to_html(table_id='passive-fire-table', classes='table table-striped', index=False, escape=False)}
        </body>
        </html>"""
        return html_content.encode('utf-8'), f"{filename}.html", "text/html"

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🔥 Passive Fire Schedule Processor v4.3</h1>', unsafe_allow_html=True)
    st.markdown("**🔧 ENHANCED: Final Material Standardization with Materials Lookup Cross-Check**")
    st.markdown("---")

    # Show updates summary
    with st.expander("🔧 v4.3 Enhancements", expanded=False):
        st.markdown("""
        **✅ Final Material Standardization (NEW):**
        - **NEW**: Additional cross-check step after initial material population
        - **NEW**: Uses Materials Lookup table to standardize/simplify material names
        - **NEW**: Statistical reporting of mapping success rates
        - **IMPROVED**: Cleaner, more consistent material output

        **🔄 Material Processing Flow:**
        1. **Initial Population**: Get materials from CSV and Steel Lookup tables
        2. **Final Standardization**: Cross-reference with Materials Lookup table
        3. **Simplification**: Convert complex material names to standard terms
        4. **Reporting**: Show mapping statistics and success rates

        **📊 Previous Features (v4.2) Still Active:**
        - FRR Distribution table and bar chart
        - Material Distribution table and bar chart
        - Interactive visualizations with Plotly
        - Statistical summary with counts and percentages

        **🎯 Core Logic (v4.1) Still Active:**
        - Service column generation with discipline and element ID lookup
        - FRR retrieval using Item Type from second title part
        - Enhanced "Standard" Item Type handling
        """)

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
            "Passive_Fire_Schedule",
            help="Enter filename (extension will be added automatically)"
        )

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

            **Step 4:** Review processing examples & distributions

            **Step 5:** Download your processed schedule

            **🔧 v4.3 NEW FEATURES:**
            - **Final Material Standardization**: Extra cross-check with Materials Lookup
            - **Simplified Output**: Cleaner, more consistent material names
            - **Mapping Statistics**: See how many materials were standardized
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
            st.success(f"✅ Excel file loaded: {excel_file.name}")
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
            help="Upload CSV file with component data - column names will be auto-standardized by position"
        )

        if csv_file:
            st.success(f"✅ CSV file loaded: {csv_file.name}")
            st.info(f"File size: {csv_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)

    # Optional lookup tables section
    st.subheader("🔍 Optional Lookup Tables")

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
            st.success(f"✅ Size lookup: {size_lookup_file.name}")
            size_lookup_type = 'csv' if size_lookup_file.name.endswith('.csv') else 'excel'
        else:
            size_lookup_type = None
            st.info("No size lookup table uploaded")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="lookup-section">', unsafe_allow_html=True)
        st.markdown("### 🔧 Materials Lookup Table")
        st.markdown("*For material classification mapping & final standardization*")

        materials_lookup_file = st.file_uploader(
            "Choose materials lookup file",
            type=['csv', 'xlsx', 'xls'],
            key="materials_lookup_upload",
            help="Upload CSV or Excel file with material mappings - used for final standardization step"
        )

        if materials_lookup_file:
            st.success(f"✅ Materials lookup: {materials_lookup_file.name}")
            materials_lookup_type = 'csv' if materials_lookup_file.name.endswith('.csv') else 'excel'
        else:
            materials_lookup_type = None
            st.info("No materials lookup table uploaded - final standardization will be skipped")

        st.markdown('</div>', unsafe_allow_html=True)

    # Process button
    if excel_file and csv_file:
        st.markdown("---")

        if st.button("🚀 Process Files", type="primary", use_container_width=True):
            with st.spinner("Processing your files..."):
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

                        # NEW: FRR Distribution Analysis
                        st.subheader("🔥 FRR Distribution Analysis")
                        frr_table, frr_chart = create_frr_distribution(result_df)
                        
                        if frr_table is not None and frr_chart is not None:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.write("**FRR Summary Table:**")
                                st.dataframe(frr_table, use_container_width=True)
                            with col2:
                                st.write("**FRR Distribution:**")
                                st.plotly_chart(frr_chart, use_container_width=True)
                        else:
                            st.info("No FRR data available for distribution analysis")

                        # NEW: Material Distribution Analysis
                        st.subheader("🔧 Material Distribution Analysis")
                        material_table, material_chart = create_material_distribution(result_df)
                        
                        if material_table is not None and material_chart is not None:
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.write("**Material Summary Table:**")
                                st.dataframe(material_table, use_container_width=True)
                            with col2:
                                st.write("**Material Distribution:**")
                                st.plotly_chart(material_chart, use_container_width=True)
                        else:
                            st.info("No material data available for distribution analysis")

                        # Data preview
                        st.subheader("📊 Full Data Preview")
                        st.dataframe(
                            result_df.head(20),
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
        "🔥 Passive Fire Schedule Processor v4.3"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()