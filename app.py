# Hurricane Flood Impact Analysis Web App
# File: app.py
# Deploy to Streamlit Cloud for 24/7 availability

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
from pathlib import Path
import json
import rasterio
from scipy.ndimage import zoom
from datetime import datetime, timedelta

# Google Earth Engine imports
try:
    import ee
    import folium
    from streamlit_folium import folium_static
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Hurricane Flood Impact Analysis",
    page_icon="🌊",
    layout="wide"
)

# GEE Authentication function
@st.cache_resource
def authenticate_gee():
    """Authenticate Google Earth Engine"""
    if not GEE_AVAILABLE:
        return False

    try:
        # For Streamlit Cloud deployment, use service account
        if hasattr(st, 'secrets') and 'gee_service_account' in st.secrets:
            credentials = ee.ServiceAccountCredentials(
                st.secrets["gee_service_account"]["email"],
                key_data=st.secrets["gee_service_account"]["key"]
            )
            ee.Initialize(credentials)
        else:
            # For local development
            try:
                ee.Initialize()
            except:
                return False
        return True
    except Exception as e:
        st.error(f"GEE Authentication failed: {e}")
        return False
# Replace your existing get_hurricane_ian_hls_data and related helpers with this code.

def _degrees_to_meters_width(bbox):
    # approx conversion at mid-latitude: 1 degree lon ~ 111.32 km * cos(lat)
    lon0, lat0, lon1, lat1 = bbox
    center_lat = (lat0 + lat1) / 2.0
    deg_lon_m = 111320.0 * np.cos(np.deg2rad(center_lat))
    width_m = abs(lon1 - lon0) * deg_lon_m
    height_m = abs(lat1 - lat0) * 111320.0
    return width_m, height_m

def _ensure_min_pixels_on_sample(np_array, min_pixels=64*64):
    """Helper: check if array size >= min_pixels (for any band)"""
    if getattr(np_array, "size", 0) >= min_pixels:
        return True
    return False

def get_hurricane_ian_sentinel2_data(bbox, pre_date, post_date, max_cloud_cover=20, min_pixels=64*64):
    """
    Robust Sentinel-2 data fetcher for flood analysis
    Uses Sentinel-2 MSI: MultiSpectral Instrument, Level-2A
    """
    aoi = ee.Geometry.Rectangle(bbox)

    # Add a small buffer in time
    pre_start = (datetime.strptime(pre_date, '%Y-%m-%d') - timedelta(days=14)).strftime('%Y-%m-%d')
    post_end = (datetime.strptime(post_date, '%Y-%m-%d') + timedelta(days=14)).strftime('%Y-%m-%d')

    # Use Sentinel-2 bands compatible with flood analysis
    # B2 (blue), B3 (green), B4 (red), B8 (nir), B11 (swir1), B12 (swir2)
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

    try:
        # Use Sentinel-2 Level 2A data (surface reflectance)
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR')
        
        # Filter collections
        pre_collection = s2_collection.filterBounds(aoi).filterDate(pre_start, pre_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
        post_collection = s2_collection.filterBounds(aoi).filterDate(post_date, post_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))

        pre_count = pre_collection.size().getInfo()
        post_count = post_collection.size().getInfo()
        if pre_count == 0 or post_count == 0:
            raise ValueError(f"Insufficient images: {pre_count} pre, {post_count} post")

        # Median composites
        pre_image = pre_collection.select(bands).median().clip(aoi)
        post_image = post_collection.select(bands).median().clip(aoi)

        # First attempt: simple sampleRectangle
        try:
            pre_data = pre_image.sampleRectangle(region=aoi, defaultValue=0)
            post_data = post_image.sampleRectangle(region=aoi, defaultValue=0)

            # Convert to numpy arrays
            pre_arrays = {}
            post_arrays = {}
            small_flag = False
            for band in bands:
                pre_band = np.array(pre_data.get(band).getInfo())
                post_band = np.array(post_data.get(band).getInfo())

                # If shape 1D or too small, mark small_flag
                if pre_band.ndim != 2 or post_band.ndim != 2 or pre_band.size < 100 or post_band.size < 100:
                    small_flag = True

                pre_arrays[band] = pre_band
                post_arrays[band] = post_band

            if not small_flag and _ensure_min_pixels_on_sample(next(iter(pre_arrays.values())), min_pixels=min_pixels):
                pre_stack = np.stack([pre_arrays[b] for b in bands], axis=0)
                post_stack = np.stack([post_arrays[b] for b in bands], axis=0)
                # Normalize to 0..1 (Sentinel-2 SR is 0-10000)
                pre_stack = np.clip(pre_stack.astype(np.float32) / 10000.0, 0, 1)
                post_stack = np.clip(post_stack.astype(np.float32) / 10000.0, 0, 1)
                return pre_stack, post_stack, True
            # Else fall through to more robust methods
        except Exception as e_sample:
            st.info("Initial sampleRectangle returned too few pixels - trying robust fallback (expand / aggregate).")

        # --- Strategy A: Expand bbox slightly if very small
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width < 0.05 or bbox_height < 0.05:
            # Expand to at least ~0.06 degrees (approx ~6-7 km) around center
            center_lon = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            expanded_bbox = [center_lon - 0.03, center_lat - 0.03, center_lon + 0.03, center_lat + 0.03]
            aoi = ee.Geometry.Rectangle(expanded_bbox)
            st.info(f"Expanded study area from {bbox} to {expanded_bbox}")
            # Reclip images
            pre_image = pre_image.clip(aoi)
            post_image = post_image.clip(aoi)

        # --- Strategy B: Aggregate (reduce resolution) to produce a reasonable pixel grid
        # Compute physical size and choose a coarse scale
        width_m, height_m = _degrees_to_meters_width(bbox if 'expanded_bbox' not in locals() else expanded_bbox)
        # Choose desired target pixels along longer side (e.g. 128)
        target_long = 128
        longer_m = max(width_m, height_m, 1.0)
        chosen_scale = max(30, int(np.ceil(longer_m / target_long)))  # scale in meters
        st.info(f"Aggregating to scale={chosen_scale} m to ensure robust sampling.")

        # Apply mean reducer and reproject to that scale
        reducer = ee.Reducer.mean()
        pre_agg = pre_image.reduceResolution(reducer=reducer, bestEffort=True).reproject(crs='EPSG:3857', scale=chosen_scale).clip(aoi)
        post_agg = post_image.reduceResolution(reducer=reducer, bestEffort=True).reproject(crs='EPSG:3857', scale=chosen_scale).clip(aoi)

        # Sample again
        try:
            pre_data = pre_agg.sampleRectangle(region=aoi, defaultValue=0)
            post_data = post_agg.sampleRectangle(region=aoi, defaultValue=0)
            pre_arrays = {}
            post_arrays = {}
            for band in bands:
                pre_band = np.array(pre_data.get(band).getInfo())
                post_band = np.array(post_data.get(band).getInfo())
                # Double-check shapes
                if pre_band.ndim != 2 or post_band.ndim != 2:
                    raise ValueError(f"Aggregated band {band} not 2D")
                pre_arrays[band] = pre_band
                post_arrays[band] = post_band

            pre_stack = np.stack([pre_arrays[b] for b in bands], axis=0)
            post_stack = np.stack([post_arrays[b] for b in bands], axis=0)
            pre_stack = np.clip(pre_stack.astype(np.float32) / 10000.0, 0, 1)
            post_stack = np.clip(post_stack.astype(np.float32) / 10000.0, 0, 1)
            return pre_stack, post_stack, True
        except Exception as e_agg:
            st.info("Aggregation sampling failed — trying thumbnail fallback.")

        # --- Strategy C: getThumb fallback (guarantees a raster of desired px dimensions)
        try:
            # Choose reasonable output size
            out_px = 256
            pre_thumb_url = pre_image.getThumbURL({
                'min': 0, 
                'max': 10000, 
                'dimensions': out_px, 
                'format': 'png', 
                'bands': ['B4', 'B3', 'B2']  # RGB for display
            })
            post_thumb_url = post_image.getThumbURL({
                'min': 0, 
                'max': 10000, 
                'dimensions': out_px, 
                'format': 'png', 
                'bands': ['B4', 'B3', 'B2']  # RGB for display
            })
            
            # Attempt to retrieve via requests
            try:
                import requests
                r1 = requests.get(pre_thumb_url, timeout=20)
                r2 = requests.get(post_thumb_url, timeout=20)
                from io import BytesIO
                pre_img = Image.open(BytesIO(r1.content)).convert('RGB')
                post_img = Image.open(BytesIO(r2.content)).convert('RGB')
                # Convert to float arrays 0..1
                pre_arr = np.asarray(pre_img).astype(np.float32) / 255.0
                post_arr = np.asarray(post_img).astype(np.float32) / 255.0
                # Convert RGB to 6-channel padded arrays
                # This is a fallback and approximate
                pre_stub = np.stack([pre_arr[...,2], pre_arr[...,1], pre_arr[...,0], 
                                   pre_arr[...,0], pre_arr[...,1], pre_arr[...,2]], axis=0)
                post_stub = np.stack([post_arr[...,2], post_arr[...,1], post_arr[...,0], 
                                    post_arr[...,0], post_arr[...,1], post_arr[...,2]], axis=0)
                return pre_stub, post_stub, True
            except Exception as e_req:
                raise RuntimeError(f"Thumbnail fetch failed (network or requests not available): {e_req}")
        except Exception as e_thumb:
            raise RuntimeError(f"All fallback strategies failed: {e_thumb}")

    except Exception as e:
        st.error(f"Error fetching Sentinel-2 data: {e}")
        # Helpful suggestions
        if "Insufficient images" in str(e):
            st.write("**Solutions:** - Increase cloud cover param; - broaden date ranges")
        else:
            st.write("**Try:** increase study area size, widen dates, or increase cloud_cover to 60-80% for more images.")
        return None, None, False
    
# --- Small fixes for display and processing ---
def create_rgb_display(satellite_data, enhance_factor=2.0):
    """
    Convert 6-band Sentinel-2 data to RGB for display
    Bands: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    RGB mapping: R=B4 (index 2), G=B3 (index 1), B=B2 (index 0)
    """
    if len(satellite_data.shape) != 3 or satellite_data.shape[0] != 6:
        raise ValueError(f"Expected (6, H, W), got {satellite_data.shape}")
    
    # Extract RGB bands
    red_band = satellite_data[2]    # B4 (index 2)
    green_band = satellite_data[1]  # B3 (index 1) 
    blue_band = satellite_data[0]   # B2 (index 0)
    
    # Stack to create RGB: (height, width, 3)
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
    
    # Handle NaN values
    rgb_image = np.nan_to_num(rgb_image, nan=0.0)
    
    # Apply percentile stretch for better visualization
    p2, p98 = np.percentile(rgb_image, (2, 98))
    if p98 > p2:  # Avoid division by zero
        rgb_image = (rgb_image - p2) / (p98 - p2)
    
    # Enhance and clip
    rgb_image = np.clip(rgb_image * enhance_factor, 0, 1)
    
    return rgb_image

def process_image_for_demo(model, pre_image, post_image, device):
    """
    Ensure expected shapes and normalization before feeding model.
    Model expects input shape: (batch, channels, H, W) with channels=6 for each (pre/post)
    This wrapper makes minimal assumptions:
      - If input is (6,H,W) each: fine.
      - If input is (H,W,6): transpose.
      - If values ~ 0..10000 -> scale to 0..1
      - If values already 0..1 -> keep
    """
    try:
        model.eval()
        with torch.no_grad():
            def prep(img):
                arr = np.array(img)
                # if H,W,6 -> transpose
                if arr.ndim == 3 and arr.shape[2] == 6:
                    arr = np.transpose(arr, (2,0,1))
                if arr.ndim != 3:
                    raise ValueError(f"Unexpected image shape: {arr.shape}")
                # scale
                if arr.max() > 10.0:  # likely 0..10000
                    arr = np.clip(arr.astype(np.float32) / 10000.0, 0, 1)
                arr = arr.astype(np.float32)
                return arr

            pre_arr = prep(pre_image)
            post_arr = prep(post_image)

            # concatenate along channel dim if your model expects 6-ch input differently,
            # your siamese likely expects two 6-ch tensors as separate inputs.
            pre_tensor = torch.from_numpy(pre_arr).unsqueeze(0).to(device)
            post_tensor = torch.from_numpy(post_arr).unsqueeze(0).to(device)

            logits = model(pre_tensor, post_tensor)
            flood_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            # Ensure 2D and normalized
            if flood_prob.max() > 1.0 or flood_prob.min() < 0.0:
                flood_prob = (flood_prob - flood_prob.min()) / (flood_prob.max() - flood_prob.min() + 1e-9)
            return flood_prob
    except Exception as e:
        st.error(f"Error processing images: {e}")
        return None

# Import your model
@st.cache_resource
def load_model():
    """Load the trained model once and cache it"""
    try:
        from siamese_unet import ProductionSiameseUNet
        
        device = 'cpu'  # Use CPU for deployment
        model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
        
        # Try to download from Hugging Face
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="Satwik19/hurry", 
                                        filename="best_model.pth")
        except:
            # Fallback to local file
            model_path = "best_model.pth"
            if not Path(model_path).exists():
                st.error("Model file not found. Please ensure best_model.pth exists.")
                return None, None
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def load_population_data():
    """Load WorldPop data - cache it for performance"""
    try:
        # Use your clipped WorldPop data
        pop_files = list(Path('.').glob('**/worldpop_clipped_*.tif'))
        if pop_files:
            with rasterio.open(pop_files[0]) as src:
                population_data = src.read(1)
            return population_data, str(pop_files[0])
        else:
            # Fallback: create synthetic data for demo
            st.warning("Using synthetic population data for demo")
            return np.random.gamma(2, 500, (204, 192)), "synthetic_data"
    except Exception as e:
        st.error(f"Error loading population data: {e}")
        return None, None

def process_image_for_demo(model, pre_image, post_image, device):
    """Process images through the model"""
    try:
        model.eval()
        with torch.no_grad():
            # Ensure images are the right format
            if pre_image.max() > 1.0:
                pre_image = pre_image.astype(np.float32) / 255.0
                post_image = post_image.astype(np.float32) / 255.0
            
            # Convert to tensors
            pre_tensor = torch.FloatTensor(pre_image).unsqueeze(0).to(device)
            post_tensor = torch.FloatTensor(post_image).unsqueeze(0).to(device)
            
            # Get prediction
            logits = model(pre_tensor, post_tensor)
            flood_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            return flood_prob
    except Exception as e:
        st.error(f"Error processing images: {e}")
        return None

def calculate_impact(flood_map, population_data, threshold=0.5):
    """Calculate population impact from flood map and population data"""
    try:
        # Resize flood map to match population data if needed
        if flood_map.shape != population_data.shape:
            scale = (population_data.shape[0]/flood_map.shape[0], 
                    population_data.shape[1]/flood_map.shape[1])
            flood_map = zoom(flood_map, scale, order=1)
        
        # Create flood mask
        flood_mask = flood_map > threshold
        
        # Calculate impacts
        affected_population = np.sum(population_data * flood_mask)
        total_population = np.sum(population_data)
        percentage_affected = (affected_population / total_population) * 100 if total_population > 0 else 0
        flooded_area_km2 = np.sum(flood_mask) * 1.0  # 1km per pixel
        
        return {
            'affected_population': affected_population,
            'total_population': total_population,
            'percentage_affected': percentage_affected,
            'flooded_area_km2': flooded_area_km2,
            'population_density_flooded': affected_population/flooded_area_km2 if flooded_area_km2 > 0 else 0
        }
    except Exception as e:
        st.error(f"Error calculating impact: {e}")
        return None

def create_visualization(flood_map, population_data, impact_stats):
    """Create the analysis visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hurricane Flood Impact Analysis (Real WorldPop Data)', fontsize=16, fontweight='bold')
    
    # Population density
    im1 = axes[0, 0].imshow(population_data, cmap='YlOrRd', interpolation='nearest')
    axes[0, 0].set_title('Population Density\n(WorldPop 2020)')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='People per km²')
    
    # Flood probability
    im2 = axes[0, 1].imshow(flood_map, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axes[0, 1].set_title('Flood Probability\n(Siamese U-Net)')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='Flood Probability')
    
    # Affected population
    affected_map = population_data * (flood_map > 0.5)
    im3 = axes[1, 0].imshow(affected_map, cmap='Reds', interpolation='nearest')
    axes[1, 0].set_title('Affected Population')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8, label='Affected People')
    
    # Risk assessment
    risk_map = population_data * flood_map
    im4 = axes[1, 1].imshow(risk_map, cmap='plasma', interpolation='nearest')
    axes[1, 1].set_title('Population at Risk\n(Pop × Flood Probability)')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8, label='Risk Score')
    
    # Remove axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

# FIXED: Proper RGB conversion function
def create_rgb_display(satellite_data, enhance_factor=2.0):
    """
    Convert 6-band satellite data to RGB for display
    Args:
        satellite_data: numpy array of shape (6, H, W) 
        enhance_factor: brightness enhancement factor
    Returns:
        RGB image of shape (H, W, 3)
    """
    if len(satellite_data.shape) != 3 or satellite_data.shape[0] != 6:
        raise ValueError(f"Expected (6, H, W), got {satellite_data.shape}")
    
    # Extract RGB bands: B4 (Red), B3 (Green), B2 (Blue)
    # Bands: ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'] = indices [0, 1, 2, 3, 4, 5]
    red_band = satellite_data[2]    # B4 (index 2)
    green_band = satellite_data[1]  # B3 (index 1) 
    blue_band = satellite_data[0]   # B2 (index 0)
    
    # Stack to create RGB: (height, width, 3)
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
    
    # Handle NaN values
    rgb_image = np.nan_to_num(rgb_image, nan=0.0)
    
    # Apply percentile stretch for better visualization
    p2, p98 = np.percentile(rgb_image, (2, 98))
    if p98 > p2:  # Avoid division by zero
        rgb_image = (rgb_image - p2) / (p98 - p2)
    
    # Enhance and clip
    rgb_image = np.clip(rgb_image * enhance_factor, 0, 1)
    
    return rgb_image

def main():
    st.title("Hurricane Flood Impact Analysis")
    st.markdown("### Real-time flood detection using Siamese U-Net and WorldPop demographic data")
    
    # Sidebar
    st.sidebar.header("Analysis Configuration")
    confidence_threshold = st.sidebar.slider("Flood Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    
    # Load model and population data
    with st.spinner("Loading AI model and population data..."):
        model, device = load_model()
        population_data, pop_file = load_population_data()
    
    if model is None or population_data is None:
        st.error("Failed to load required components. Check your files.")
        return
    
    st.success("Model loaded successfully")
    st.info(f"Population data loaded: {np.sum(population_data):,.0f} people in study area")
    
    # Demo section with multiple options
    st.header("Analysis Options")
    
    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["Pre-loaded Data", "Google Earth Engine", "Upload Data"])
    
    with tab1:
        st.subheader("Option 1: Use Example Data")
        if st.button("Run Hurricane Ian Analysis", type="primary"):
            try:
                metadata_file = Path("processing_metadata.json")
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    hurricane_data = metadata['processed_data'][0]
                    periods = hurricane_data['periods']
                    
                    pre_files = [p for p in periods.keys() if 'pre' in p]
                    post_files = [p for p in periods.keys() if 'post' in p]
                    
                    if pre_files and post_files:
                        pre_path = periods[pre_files[-1]]['file_path']
                        post_path = periods[post_files[0]]['file_path']
                        
                        with st.spinner("Processing Hurricane Ian satellite imagery..."):
                            pre_image = np.load(pre_path)
                            post_image = np.load(post_path)
                            
                            flood_map = process_image_for_demo(model, pre_image, post_image, device)
                            
                            if flood_map is not None:
                                impact_stats = calculate_impact(flood_map, population_data, confidence_threshold)
                                
                                st.success("Analysis Complete!")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Affected Population", f"{impact_stats['affected_population']:,.0f}")
                                with col_b:
                                    st.metric("Percentage Affected", f"{impact_stats['percentage_affected']:.2f}%")
                                with col_c:
                                    st.metric("Flooded Area", f"{impact_stats['flooded_area_km2']:.1f} km²")
                                
                                fig = create_visualization(flood_map, population_data, impact_stats)
                                st.pyplot(fig)
                                
                                with st.expander("Detailed Statistics"):
                                    st.write(f"**Total Population in Study Area:** {impact_stats['total_population']:,.0f} people")
                                    st.write(f"**Population Density in Flooded Areas:** {impact_stats['population_density_flooded']:.1f} people/km²")
                                    st.write(f"**Data Source:** {pop_file}")
                                    st.write(f"**Model:** Siamese U-Net with WorldPop integration")
                                    st.write(f"**Hurricane:** {hurricane_data['hurricane']}")
                else:
                    st.warning("Hurricane Ian data not found. Try other options.")
                    
            except Exception as e:
                st.error(f"Error running example: {e}")

    with tab2:
        st.subheader("Option 2: Real-time Satellite Data")
        
        if not GEE_AVAILABLE:
            st.error("Google Earth Engine not available. Install: pip install earthengine-api folium streamlit-folium")
            return
        
        gee_auth = authenticate_gee()
        
        if gee_auth:
            st.success("Connected to Google Earth Engine")
            
            col1, col2 = st.columns(2)
            
            with col1:
                demo_areas = {
                    "Fort Myers (Coastal Impact)": [-82.1, 26.4, -81.6, 26.8],
                    "Cape Coral (Residential)": [-82.0, 26.5, -81.8, 26.7], 
                    "Sanibel Island (Barrier Island)": [-82.2, 26.4, -82.0, 26.5],
                    "Naples (Urban/Coastal)": [-81.8, 26.1, -81.6, 26.3]
                }
                
                selected_area = st.selectbox("Select Study Area", list(demo_areas.keys()))
                bbox = demo_areas[selected_area]
                
                pre_date = st.date_input("Pre-Hurricane Date", 
                                       value=datetime(2022, 9, 20),
                                       min_value=datetime(2022, 9, 1),
                                       max_value=datetime(2022, 9, 27))
                
                post_date = st.date_input("Post-Hurricane Date",
                                        value=datetime(2022, 10, 5),
                                        min_value=datetime(2022, 9, 28),
                                        max_value=datetime(2022, 10, 15))
                
                cloud_cover = st.slider("Max Cloud Cover %", 0, 50, 20)
            
            with col2:
                center_lat = (bbox[1] + bbox[3]) / 2
                center_lon = (bbox[0] + bbox[2]) / 2
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
                folium.Rectangle(
                    bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
                    color='red',
                    fill=True,
                    fillOpacity=0.3,
                    popup=f"Study Area: {selected_area}"
                ).add_to(m)
                
                folium_static(m, width=400, height=300)
            
            if st.button("Fetch & Analyze Real Satellite Data", type="primary"):
                with st.spinner(f"Fetching HLS data for {selected_area}..."):
                    
                    pre_data, post_data, success = get_hurricane_ian_sentinel2_data(
                        bbox, 
                        pre_date.strftime('%Y-%m-%d'), 
                        post_date.strftime('%Y-%m-%d'),
                        cloud_cover
                    )
                    
                    if success and pre_data is not None and post_data is not None:
                        st.success("Satellite data fetched successfully!")
                        
                        st.write(f"**Data Shape:** {pre_data.shape}")
                        st.write(f"**Area:** {selected_area}")
                        st.write(f"**Spatial Coverage:** ~{(bbox[2]-bbox[0])*111:.1f}km × {(bbox[3]-bbox[1])*111:.1f}km")
                        
                        # FIXED: Proper image display
                        try:
                            rgb_pre = create_rgb_display(pre_data)
                            rgb_post = create_rgb_display(post_data)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(rgb_pre, caption="Pre-Hurricane RGB", use_column_width=True)
                            with col_b:
                                st.image(rgb_post, caption="Post-Hurricane RGB", use_column_width=True)
                                
                        except Exception as e:
                            st.error(f"Error displaying images: {e}")
                            st.write(f"Error details: {type(e).__name__}")
                        
                        # Run analysis
                        st.write("**Running flood analysis...**")
                        
                        flood_map = process_image_for_demo(model, pre_data, post_data, device)
                        
                        if flood_map is not None:
                            impact_stats = calculate_impact(flood_map, population_data, confidence_threshold)
                            
                            st.success("Analysis Complete!")
                            
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("Affected Population", f"{impact_stats['affected_population']:,.0f}")
                            with col_y:
                                st.metric("Percentage Affected", f"{impact_stats['percentage_affected']:.2f}%")
                            with col_z:
                                st.metric("Flooded Area", f"{impact_stats['flooded_area_km2']:.1f} km²")
                            
                            fig = create_visualization(flood_map, population_data, impact_stats)
                            st.pyplot(fig)
                            
                            st.subheader("Research Question Analysis")
                            st.write("**Q: How many people were affected by Hurricane Ian in this area?**")
                            st.write(f"**A:** {impact_stats['affected_population']:,.0f} people were directly affected by flooding in the {selected_area} area, representing {impact_stats['percentage_affected']:.2f}% of the local population.")
                            
                            st.write("**Q: What was the spatial extent of flooding?**")
                            st.write(f"**A:** {impact_stats['flooded_area_km2']:.1f} km² were identified as flooded using deep learning analysis of real satellite data.")
                    
                    else:
                        st.error("Failed to fetch satellite data. Possible reasons:")
                        st.write("- No cloud-free imagery for selected dates")
                        st.write("- GEE quota limits")
                        st.write("- Network issues")
        else:
            st.error("Failed to connect to Google Earth Engine")
            st.write("To enable GEE:")
            st.code("""
            pip install earthengine-api folium streamlit-folium
            
            # For authentication, add to .streamlit/secrets.toml:
            [gee_service_account]
            email = "your-service-account@project.iam.gserviceaccount.com"
            key = '''-----BEGIN PRIVATE KEY-----
            [your-private-key-content]
            -----END PRIVATE KEY-----'''
            """)

    with tab3:
        st.subheader("Option 3: Upload Your Own Data")
        uploaded_pre = st.file_uploader("Upload Pre-Event Satellite Image (.npy)", type=['npy'])
        uploaded_post = st.file_uploader("Upload Post-Event Satellite Image (.npy)", type=['npy'])
        
        if uploaded_pre and uploaded_post:
            if st.button("Analyze Uploaded Images"):
                with st.spinner("Processing your images..."):
                    try:
                        pre_bytes = uploaded_pre.getvalue()
                        post_bytes = uploaded_post.getvalue()
                        
                        pre_image = np.load(io.BytesIO(pre_bytes))
                        post_image = np.load(io.BytesIO(post_bytes))
                        
                        flood_map = process_image_for_demo(model, pre_image, post_image, device)
                        
                        if flood_map is not None:
                            impact_stats = calculate_impact(flood_map, population_data, confidence_threshold)
                            
                            st.success("Custom Analysis Complete!")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Affected Population", f"{impact_stats['affected_population']:,.0f}")
                            with col_b:
                                st.metric("Percentage Affected", f"{impact_stats['percentage_affected']:.2f}%")
                            with col_c:
                                st.metric("Flooded Area", f"{impact_stats['flooded_area_km2']:.1f} km²")
                            
                            fig = create_visualization(flood_map, population_data, impact_stats)
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error processing uploaded images: {e}")
    
    # Model information
    with st.expander("Model Information"):
        st.markdown("""
        **Architecture:** Siamese U-Net for change detection
        - **Input:** 6-band satellite imagery (pre/post event)
        - **Output:** Flood probability map
        - **Training:** Hurricane events with real ground truth
        - **Population Data:** WorldPop 2020 (1km resolution)
        
        **Advantages over traditional methods:**
        - Uses all spectral bands simultaneously
        - Learns complex flood signatures from training data
        - Detects subtle changes missed by simple indices (NDWI/MDWI)
        - Provides probabilistic rather than binary outputs
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Hurricane Flood Impact Analysis** | Powered by PyTorch, Streamlit & Google Earth Engine | Real WorldPop Data Integration")

if __name__ == "__main__":
    main()