import streamlit as st
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types
from arcgis.gis import GIS
from arcgis.features import FeatureLayer

st.set_page_config(page_title="GoDurham Inventory Sync", layout="wide")
st.title("🚌 GoDurham Live Map Server Sync Hub")

IMAGES_DIR = Path("images_metadata")  # Folder containing left/center/right images

# Cache API clients to optimize load performance
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def get_arcgis_layer():
    try:
        # OAuth2 Application client auth bypassing 2-Factor challenges
        gis = GIS("https://dukeuniv.maps.arcgis.com", 
                  client_id=st.secrets["ARCGIS_CLIENT_ID"], 
                  client_secret=st.secrets["ARCGIS_CLIENT_SECRET"])
        return FeatureLayer(st.secrets["FEATURE_LAYER_URL"], gis=gis)
    except Exception as e:
        st.error(f"Failed to securely authenticate with Duke ArcGIS: {e}")
        return None

client = get_gemini_client()
layer = get_arcgis_layer()

# --- BACKEND DIRECT ARCGIS REST PUSH ---
def update_arcgis_live(stop_id, attribute_dict):
    """Directly queries and overrides attributes on the active map feature layer row."""
    if not layer:
        return False, "ArcGIS Database layer client not initialized."
    try:
        # Query the database for a row matching our exact Stop ID
        query_result = layer.query(where=f"stop_id = '{stop_id}'")
        
        if len(query_result.features) > 0:
            target_feature = query_result.features[0]
            
            # Map values directly into feature layer attribute cells
            for field_name, value in attribute_dict.items():
                target_feature.attributes[field_name] = value
                
            # Fire an asynchronous REST update to commit edits live
            layer.edit_features(updates=[target_feature])
            return True, "Successfully synced live with server map layer!"
        else:
            return False, f"Stop ID {stop_id} not found inside live ArcGIS hosted feature layer."
    except Exception as e:
        return False, f"ArcGIS REST Pipeline Error: {e}"


# --- FRONTEND INTERFACE UI ---
st.sidebar.header("Navigation Workspace")
available_stops = sorted(list(set([f.name.split('_')[0] for f in IMAGES_DIR.glob("*.jpg")])))

if not available_stops:
    st.warning("No imagery sequences found in `images_metadata/` staging folder.")
    st.stop()

selected_stop = st.sidebar.selectbox("Select Bus Stop ID:", available_stops)

# Gather and display target stop images side-by-side
stop_images = sorted(list(IMAGES_DIR.glob(f"{selected_stop}_*.jpg")))
st.subheader(f"Auditing Infrastructure Core: Stop {selected_stop}")

if stop_images:
    cols = st.columns(len(stop_images))
    for idx, img_path in enumerate(stop_images):
        with cols[idx]:
            st.image(str(img_path), caption=img_path.name, use_container_width=True)

# Fetch baseline data dynamically straight from ArcGIS database to populate our inputs
current_attributes = {}
if layer:
    try:
        query_baseline = layer.query(where=f"stop_id = '{selected_stop}'")
        if len(query_baseline.features) > 0:
            current_attributes = query_baseline.features[0].attributes
    except Exception:
        pass

st.divider()
st.markdown("### Feature Matrix Verification Panel")

with st.form("evaluation_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        surface_idx = ["Grass", "Concrete"].index(current_attributes.get("stop_surface", "Grass")) if current_attributes.get("stop_surface") in ["Grass", "Concrete"] else 0
        surface_val = st.selectbox("Stop Surface", ["Grass", "Concrete"], index=surface_idx)
        
        landing_options = ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"]
        landing_idx = landing_options.index(current_attributes.get("landing_type")) if current_attributes.get("landing_type") in landing_options else 0
        landing_val = st.selectbox("Landing Type", landing_options, index=landing_idx)
    with c2:
        sidewalk_options = ["Yes", "No", "NA"]
        sidewalk_idx = sidewalk_options.index(current_attributes.get("sidewalk_connection")) if current_attributes.get("sidewalk_connection") in sidewalk_options else 0
        sidewalk_val = st.selectbox("Sidewalk Connection", sidewalk_options, index=sidewalk_idx)
        
        pad_options = ["Two_doors", "One_door", "NA"]
        pad_idx = pad_options.index(current_attributes.get("landing_pad")) if current_attributes.get("landing_pad") in pad_options else 0
        pad_val = st.selectbox("Landing Pad", pad_options, index=pad_idx)
    with c3:
        shelter_val = st.number_input("Shelter Count", min_value=0, step=1, value=int(current_attributes.get("shelter_number", 0)))
        bench_val = st.number_input("Bench Count", min_value=0, step=1, value=int(current_attributes.get("bench_number", 0)))
    with c4:
        trash_val = st.number_input("Trash Can Count", min_value=0, step=1, value=int(current_attributes.get("trash_can_number", 0)))
        notes_val = st.text_area("Audit Operational Field Notes:", value=str(current_attributes.get("notes", "")))

    submit_button = st.form_submit_button("🔒 Save & Patch ArcGIS Online Live Map")

if submit_button:
    payload = {
        "stop_surface": surface_val,
        "landing_type": landing_val,
        "sidewalk_connection": sidewalk_val,
        "landing_pad": pad_val,
        "shelter_number": int(shelter_val),
        "bench_number": int(bench_val),
        "trash_can_number": int(trash_val),
        "notes": notes_val
    }
    
    with st.spinner("Pushing record modifications asynchronously to map server..."):
        success, message = update_arcgis_live(selected_stop, payload)
        
    if success:
        st.success(message)
    else:
        st.error(message)