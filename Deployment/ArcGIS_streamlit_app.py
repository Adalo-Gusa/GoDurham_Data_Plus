import streamlit as st
import os
import json
from pathlib import Path
from PIL import Image
from google import genai
from google.genai import types
import tempfile
import time
import requests

# ==========================================
# CONFIGURATION & API KEY INITIALIZATION
# ==========================================
MODEL_ID = "gemini-3.5-flash"
GEMINI_PROJECT = "dataplus-godurham"

try:
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
        tfile.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        tfile.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tfile.name
            
    client = genai.Client(
        vertexai=True,
        project=GEMINI_PROJECT,
        location="global"
    )
except Exception as e:
    st.error(f"Failed to initialize Gemini Client: {e}")

# ==========================================
# PROMPTS & SCHEMAS
# ==========================================
PROMPT_PASS_1 = """
You are an expert visual analyst evaluating a sequence of Street View images for a transit stop accessibility inventory.

CRITICAL INSTRUCTION: You must follow a strict TWO-PATH logical workflow. Attempt Path A first. Only proceed to Path B if Path A fails.

=== PATH A: SINGLE-IMAGE FAST-TRACK ===
Scan all provided images for clear, explicit transit infrastructure (a dedicated GoDurham bus stop sign, a bus shelter, or a dedicated transit bench).
(Note: Parked buses, generic yellow poles, utility poles, and bike racks DO NOT count).

If you find a clear view of the transit infrastructure AND the boarding area in ONE single image:
1. Set bus_stop_visible to "Yes".
2. Set best_view to the name of that specific winning image.
3. Classify ALL features (stop_surface, landing_type, amenities) using ONLY that single image. 
STILL: Synthesize all of the images to search for a cross walk. 
Stop here and output your JSON.

=== PATH B: PANORAMIC SYNTHESIS ===
If no single image provides a perfect view, you must synthesize the visual evidence from ALL images combined to evaluate the continuous environment.

Evaluate the synthesized environment and choose ONE of the following outcomes:

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama proves a bus stop exists (e.g., the transit sign is in the far-left image, but the concrete landing pad and bench are in the mid-left image).
- Action: Set bus_stop_visible to "Yes". Set best_view to the image containing the transit sign or the clearest part of the boarding pad. Synthesize the environment to count all features accurately. DO NOT flag for manual review.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360/panoramic area. There is absolutely no transit infrastructure (no bus sign, no shelter). You only see general street features (sidewalks, grass, utility poles).
- Action: Set bus_stop_visible to "No". Classify whatever generic features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction completely blocks the view of the curb. Do NOT assume a stop exists just because a bus is parked there.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."

IMPORTANT:
1. FIRST, determine the decimal confidence score (0.0 to 1.0) for: 
   bus_stop_visibility_confidence, shelter_confidence, bench_confidence, and trash_can_confidence.
   The scale is from 0.0 to 1.0. 0.0 means there is NO CHANCE of the component. 1.0 means without a doubt the component is there.
   You MUST output these as precise floats (e.g., 0.85, 0.42). DO NOT default to 1.0.

2. SECOND, use these calculated decimal scores to set the binary 'present' fields:
    bus_stop_visible: "Yes" if confidence > 0.0, else "No"
    shelter_present: 1 if shelter_confidence > 0.75, else 0
    bench_present: 1 if bench_confidence > 0.75, else 0
    trash_can_present: 1 if trash_can_confidence >= 0.4, else 0

Your output JSON must contain the specific decimal float for each confidence field, not just 1.0.

=== DEFINITIONS ===
lattidude: EXACT saved in variable: "lattitude"
longitude: EXACT saved in variable: "longitude"
1. shelter_number: Total integer count across ALL images.
2. bench_number: Total integer count across ALL images. If there is a shelter present there is at least one.
3. trash_can_number: Total integer count of ANY visible public trash cans. 
4. stop_surface: "Grass" or "Concrete"
5. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
6. sidewalk_connection: "Yes", "No", or "NA"
7. landing_pad: "Two_doors", "One_door", or "NA"
8. cross_walk: "Yes", or "No"
9. street_lighting: "Yes" or "No"

Return only JSON matching the schema.
"""

PROMPT_PASS_2 = """
You are an expert visual analyst evaluating a 360-degree panoramic sequence of a transit stop location. 
YOU ARE A PRECISE MACHINE THAT PRIORITIZES REPRODUCIBILITY. FOLLOW EVERY LINE AS IT COMES UP, THE PROMPT IS MEANT TO BE FOLLOWED IN ORDER.

INSTRUCTIONS:
You must SYNTHESIZE the visual evidence from ALL images combined to evaluate the continuous environment.

Classification Refresher:
Scan the provided images for clear, explicit transit infrastructure. If ANY of these components are found then the stop EXISTS.
1. A GoDurham bus stop sign (often attached to wooden utility poles or metal posts). Depending on the view you may be able to make out "Bus Stop" written on the stop if close enough and have a appropriate angle.
2. A bus shelter, will likely include a bench situated inside of it.
3. A public bench either included with the bus shelter, or in close proximity to a bus sign or bus shelter. 
4. A public trash can adjacent to the road that is near where pedestrians would walk.
(Note: Parked buses will obstruct view, bike racks, fire hydrants, and BARE utility poles or WITHOUT the up-right rectangle GoDurham bus sign DO NOT count).

Outcome 1: Synthesized "Yes" (Stop Exists)
- Condition: The combined panorama has ANY of the previously listed classification refreshers(1-4) are present throughout the entire synthesis. THEY DO NOT NEED TO BE IN CLOSE PROXIMITY TO EACH OTHER.
- Action: Set bus_stop_visible to "Yes". Set best_view to the EXACT string identifier of the image containing any of the components or the best representation of a component. (e.g., "n", "sw", "right"). Synthesize the environment across all images to count features.
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification, be sure to include the bus stop components in this note.

Outcome 2: Definitively "No" (Stop is Missing)
- Condition: You have viewed the entire 360 area and found ZERO of the components. (There is absolutely no sign, no shelter, no bench, AND no trash can).
- Action: Set bus_stop_visible to "No". Classify features you can see (sidewalks, grass). You MUST begin your notes with "MANUAL REVIEW REQUIRED: Definitively no bus stop infrastructure visible in any image."
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification, be sure to include the bus stop components in this note.

Outcome 3: "Unclear" (Blocked or Obscured)
- Condition: A parked vehicle (bus, car, truck), heavy foliage, or active construction blocks the view of the curb across ALL relevant images.
- Action: Set bus_stop_visible to "Unclear". Classify whatever background features you can see. You MUST begin your notes with "MANUAL REVIEW REQUIRED: View of the curb is blocked by a vehicle/object."
- Add your thought process into the 'notes' attribute in the JSON to help you make this classification

IMPORTANT:
1. FIRST, determine the decimal confidence score (0.0 to 1.0) for: 
   bus_stop_visibility_confidence, shelter_confidence, bench_confidence, and trash_can_confidence.
   The scale is from 0.0 to 1.0. 0.0 means there is NO CHANCE of the component. 1.0 means without a doubt the component is there.
   You MUST output these as precise floats (e.g., 0.85, 0.42). DO NOT default to 1.0.

2. SECOND, use these calculated decimal scores to set the binary 'present' fields:
    bus_stop_visible: "Yes" if confidence > 0.0, else "No"
    shelter_present: 1 if shelter_confidence > 0.75, else 0
    bench_present: 1 if bench_confidence > 0.75, else 0
    trash_can_present: 1 if trash_can_confidence >= 0.4, else 0

Your output JSON must contain the specific decimal float for each confidence field, not just 1.0.

DEFINITIONS:
lattidude: EXACT saved in variable: "lattitude"
longitude: EXACT saved in variable: "longitude"
1. shelter_number: Total integer count across ALL images.
2. bench_number: Total integer count across ALL images, if there is a shelter present there is at least one.
3. trash_can_number: Total integer count of ANY visible public trash cans. 
4. stop_surface: "Grass" or "Concrete"
5. landing_type: "Paved", "Unpaved", or "Unpaved_Grass_Strip_And_Sidewalk"
6. sidewalk_connection: "Yes", "No", or "NA"
7. landing_pad: "Two_doors", "One_door", or "NA"
8. cross_walk: "Yes", or "No"
9. street_lighting: "Yes" or "No"

Return only JSON matching the schema.
"""

response_schema = {
    "type": "object",
    "properties": {
        "stop_id": {"type": "string"},
        "selected_image_filename": {"type": "string"},
        "bus_stop_visibility_confidence": {"type": "number"},
        "bus_stop_visible": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
        "shelter_confidence": {"type": "number"},
        "shelter_present": {"type": "number"},
        "shelter_number": {"type": "integer"},
        "bench_confidence": {"type": "number"},
        "bench_present": {"type": "number"},
        "bench_number": {"type": "integer"},
        "trash_can_confidence": {"type": "number"},
        "trash_can_present": {"type": "number"},
        "trash_can_number": {"type": "integer"},
        "stop_surface": {"type": "string", "enum": ["Grass", "Concrete"]},
        "landing_type": {"type": "string", "enum": ["Paved", "Unpaved", "Unpaved_Grass_Strip_And_Sidewalk"]},
        "sidewalk_connection": {"type": "string", "enum": ["Yes", "No", "NA"]},
        "landing_pad": {"type": "string", "enum": ["Two_doors", "One_door", "NA"]},
        "cross_walk": {"type": "string", "enum": ["Yes", "No"]},
        "street_lighting": {"type": "string", "enum": ["Yes", "No"]},
        "date": {
            "type": "string",
            "description": "The exact year and month the image was taken explicitly stated in the file name."
        },
        "notes": {"type": "string"},
    },
    "required": [
        "stop_id", "selected_image_filename", "bus_stop_visibility_confidence", "bus_stop_visible", 
        "stop_surface", "landing_type", "sidewalk_connection", "landing_pad", 
        "shelter_confidence", "shelter_number", "shelter_present",
        "bench_confidence", "bench_number", "bench_present",  
        "trash_can_confidence","trash_can_number", "trash_can_present", "date",
        "street_lighting", "notes"
    ],
    "additionalProperties": False
}

# ==========================================
# ARCGIS TOKEN MANAGEMENT
# ==========================================
_token_cache = {"value": None, "expires_at": 0}

def get_arcgis_token() -> str:
    """Returns a valid token using secrets, auto-refreshing before expiry."""
    if _token_cache["value"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["value"]

    resp = requests.post(
        "https://www.arcgis.com/sharing/rest/oauth2/token",
        data={
            "client_id": st.secrets["ARCGIS_CLIENT_ID"],
            "client_secret": st.secrets["ARCGIS_CLIENT_SECRET"],
            "grant_type": "client_credentials",
            "f": "json",
        },
    ).json()

    if "access_token" not in resp:
        raise RuntimeError(f"Token fetch failed: {resp}")

    _token_cache["value"] = resp["access_token"]
    _token_cache["expires_at"] = time.time() + resp.get("expires_in", 7200)
    return _token_cache["value"]


# ==========================================
# BACKEND ARCGIS REST PUSH PIPELINE
# ==========================================
def push_to_arcgis_server(stop_id: str, gemini_results: dict, uploaded_files_list) -> tuple:
    """Locates feature, pushes attribute overrides, and mounts primary attachment."""
    try:
        layer_url = st.secrets["FEATURE_LAYER_URL"]
        token = get_arcgis_token()

        # 1. Query for the structural OBJECTID
        query_resp = requests.get(
            f"{layer_url}/query",
            params={
                "where": f"stop_id='{stop_id}'",
                "outFields": "OBJECTID",
                "f": "json",
                "token": token,
            },
        ).json()

        if "error" in query_resp:
            return False, f"Query failed: {query_resp['error']}"

        features = query_resp.get("features", [])
        if not features:
            return False, f"Stop ID {stop_id} missing on target map service layer."

        object_id = features[0]["attributes"]["OBJECTID"]

        # Helper function: ArcGIS strictly requires integer bits (1 or 0) for these DB fields
        def encode_binary(val):
            val_str = str(val).strip().lower()
            if val_str in ['yes', 'true', '1']: return 1
            if val_str in ['no', 'false', '0']: return 0
            if val_str == 'unclear': return 2
            try: return int(val)
            except: return 0

        # 2. Build explicit data mapping payload
        update_payload = [{
            "attributes": {
                "OBJECTID":                       int(object_id),
                "selected_image_filename": str(gemini_results.get("selected_image_filename", "")),
                "bus_stop_visible":               encode_binary(gemini_results.get("bus_stop_visible", "Yes")),
                "bus_stop_visibility_confidence": float(gemini_results.get("bus_stop_visibility_confidence", 0.0)),
                "shelter_present":                encode_binary(gemini_results.get("shelter_present", "No")),
                "shelter_confidence":             float(gemini_results.get("shelter_confidence", 0.0)),
                "shelter_number":                 int(gemini_results.get("shelter_number", 0)),
                "bench_present":                  encode_binary(gemini_results.get("bench_present", "No")),
                "bench_confidence":               float(gemini_results.get("bench_confidence", 0.0)),
                "bench_number":                   int(gemini_results.get("bench_number", 0)),
                "trash_can_present":              encode_binary(gemini_results.get("trash_can_present", "No")),
                "trash_can_confidence":           float(gemini_results.get("trash_can_confidence", 0.0)),
                "trash_can_number":               int(gemini_results.get("trash_can_number", 0)),
                "stop_surface":                   str(gemini_results.get("stop_surface", "Concrete")),
                "landing_type":                   str(gemini_results.get("landing_type", "Paved")),
                "sidewalk_connection":            str(gemini_results.get("sidewalk_connection", "Yes")),
                "landing_pad":                    str(gemini_results.get("landing_pad", "Two_doors")),
                "cross_walk":                     encode_binary(gemini_results.get("cross_walk", "No")),
                "street_lighting":                encode_binary(gemini_results.get("street_lighting", "No")),
                "date":                           str(gemini_results.get("date", "")),
                "notes":                          str(gemini_results.get("notes", "")),
            }
        }]

        # Inject spatial parameters into the push payload if provided dynamically
        if "lattitude" in gemini_results and gemini_results["lattitude"] != 0.0:
            update_payload[0]["attributes"]["lattitude"] = float(gemini_results["lattitude"])
        if "longitude" in gemini_results and gemini_results["longitude"] != 0.0:
            update_payload[0]["attributes"]["longitude"] = float(gemini_results["longitude"])

        # 3. Apply the edit
        edit_resp = requests.post(
            f"{layer_url}/applyEdits",
            data={
                "updates": json.dumps(update_payload),
                "f": "json",
                "token": token,
            },
        ).json()

        results = edit_resp.get("updateResults", [])
        if not results or not results[0].get("success"):
            err = results[0].get("error", {}) if results else edit_resp
            return False, f"REST applyEdits failed: {err}"

        # 4. Attachment upload (Pushes the first primary image in the sequence)
        if uploaded_files_list:
            for file_item in uploaded_files_list:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file_item.name).suffix
                ) as tmp_file:
                    tmp_file.write(file_item.getbuffer())
                    tmp_file_path = tmp_file.name

                with open(tmp_file_path, "rb") as f_attach:
                    attach_resp = requests.post(
                        f"{layer_url}/{object_id}/addAttachment",
                        params={"token": token, "f": "json"},
                        files={"attachment": (file_item.name, f_attach, file_item.type)},
                    ).json()

                os.unlink(tmp_file_path)

                if not attach_resp.get("addAttachmentResult", {}).get("success"):
                    st.warning(f"Map attachment rejected for {file_item.name}: {attach_resp}")

        return True, f"Perfect Sync! Stop {stop_id} items updated live on ArcGIS cloud layer."

    except Exception as e:
        return False, f"ArcGIS Live Data Stream Exception: {e}"


# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================
st.set_page_config(page_title="GoDurham Map Integration System", layout="centered")

st.title("🚌 GoDurham Bus Stop Classifier")
st.write("Upload sequential field heading photos to run a panoramic analysis with Gemini")

# Initialize session state variables
if "current_classification" not in st.session_state:
    st.session_state.current_classification = None
if "last_classified_stop" not in st.session_state:
    st.session_state.last_classified_stop = None

# Navigation Sidebar
st.sidebar.header("Configuration Panel")
selected_pass = st.sidebar.selectbox(
    "Choose Analysis Framework Prompt:",
    ["Pass 1: Fast-Track & Synthesis", "Pass 2: Continuous Panoramic Check"]
)
active_prompt = PROMPT_PASS_1 if "Pass 1" in selected_pass else PROMPT_PASS_2

# Form Entry Layout
col1, col2, col3 = st.columns(3)
with col1:
    stop_id = st.text_input("Enter Target Stop ID (e.g., 5203):")
with col2:
    input_lat = st.number_input("Latitude (Optional)", value=0.0, format="%.6f")
with col3:
    input_lon = st.number_input("Longitude (Optional)", value=0.0, format="%.6f")
    
input_date = st.date_input("Date of Image Capture")

uploaded_files = st.file_uploader("Upload Bus Stop Image Sequence Angles", type=["jpg", "jpeg", "png", "heic"], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    cols = st.columns(len(uploaded_files))
    for idx, file_item in enumerate(uploaded_files):
        opened_img = Image.open(file_item)
        with cols[idx]:
            st.image(opened_img, caption=file_item.name, use_container_width=True)

st.divider()

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("Run Gemini Classification", use_container_width=True):
        if not stop_id:
            st.warning("Please specify a Stop ID to target your feature rows.")
        elif not uploaded_files:
            st.warning("Please upload at least one image angle heading.")
        else:
            api_payload_list = [active_prompt]
            for file_item in uploaded_files:
                api_payload_list.append(Image.open(file_item))
                    
            with st.spinner('Running Multimodal Classification...'):
                try:
                    response = client.models.generate_content(
                        model=MODEL_ID,
                        contents=api_payload_list,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            temperature=0.1
                        )
                    )
                    
                    result_json = json.loads(response.text)
                    result_json["stop_id"] = str(stop_id)
                    result_json["selected_image_filename"] = ", ".join([f.name for f in uploaded_files])
                    result_json["date"] = input_date.strftime("%Y-%m-%d")
                    
                    if input_lat != 0.0:
                        result_json["lattitude"] = input_lat
                    if input_lon != 0.0:
                        result_json["longitude"] = input_lon
                        
                    st.session_state.current_classification = result_json
                    st.session_state.last_classified_stop = str(stop_id)
                    st.success("Analysis complete! Review the inventory data schema below before staging push.")
                    
                except Exception as e:
                    st.error(f"Operational Pipeline Disruption: {e}")

with btn_col2:
    if st.session_state.current_classification is not None:
        if st.button("Push & Sync with ArcGIS Map", type="primary", use_container_width=True):
            if str(stop_id) != st.session_state.last_classified_stop:
                st.error("Stop ID value entry mismatch! Run Step 1 again to evaluate the new stop code context.")
            else:
                with st.spinner('Publishing data fields and attachments to ArcGIS cloud...'):
                    sync_success, sync_msg = push_to_arcgis_server(
                        st.session_state.last_classified_stop, 
                        st.session_state.current_classification, 
                        uploaded_files
                    )
                    if sync_success:
                        st.success(sync_msg)
                        st.session_state.current_classification = None
                        st.rerun()
                    else:
                        st.error(f"Map Sync aborted: {sync_msg}")
    else:
        st.button("Push & Sync with ArcGIS Map", disabled=True, use_container_width=True)

if st.session_state.current_classification is not None:
    st.subheader(f"Staged Inventory Payload (Stop ID: {st.session_state.last_classified_stop}):")
    st.json(st.session_state.current_classification)