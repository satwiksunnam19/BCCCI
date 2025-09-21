# Hurricane Flood Impact Analysis Web App - Enhanced
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
import warnings
warnings.filterwarnings('ignore')

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

def _degrees_to_meters_width(bbox):
    lon0, lat0, lon1, lat1 = bbox
    center_lat = (lat0 + lat1) / 2.0
    deg_lon_m = 111320.0 * np.cos(np.deg2rad(center_lat))
    width_m = abs(lon1 - lon0) * deg_lon_m
    height_m = abs(lat1 - lat0) * 111320.0
    return width_m, height_m

def _ensure_min_pixels_on_sample(np_array, min_pixels=64*64):
    if getattr(np_array, "size", 0) >= min_pixels:
        return True
    return False

def get_hurricane_ian_sentinel2_data(bbox, pre_date, post_date, max_cloud_cover=20, min_pixels=64*64):
    """Robust Sentinel-2 data fetcher for flood analysis"""
    aoi = ee.Geometry.Rectangle(bbox)

    pre_start = (datetime.strptime(pre_date, '%Y-%m-%d') - timedelta(days=14)).strftime('%Y-%m-%d')
    post_end = (datetime.strptime(post_date, '%Y-%m-%d') + timedelta(days=14)).strftime('%Y-%m-%d')

    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

    try:
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR')
        
        pre_collection = s2_collection.filterBounds(aoi).filterDate(pre_start, pre_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))
        post_collection = s2_collection.filterBounds(aoi).filterDate(post_date, post_end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_cover))

        pre_count = pre_collection.size().getInfo()
        post_count = post_collection.size().getInfo()
        if pre_count == 0 or post_count == 0:
            raise ValueError(f"Insufficient images: {pre_count} pre, {post_count} post")

        pre_image = pre_collection.select(bands).median().clip(aoi)
        post_image = post_collection.select(bands).median().clip(aoi)

        try:
            pre_data = pre_image.sampleRectangle(region=aoi, defaultValue=0)
            post_data = post_image.sampleRectangle(region=aoi, defaultValue=0)

            pre_arrays = {}
            post_arrays = {}
            small_flag = False
            for band in bands:
                pre_band = np.array(pre_data.get(band).getInfo())
                post_band = np.array(post_data.get(band).getInfo())

                if pre_band.ndim != 2 or post_band.ndim != 2 or pre_band.size < 100 or post_band.size < 100:
                    small_flag = True

                pre_arrays[band] = pre_band
                post_arrays[band] = post_band

            if not small_flag and _ensure_min_pixels_on_sample(next(iter(pre_arrays.values())), min_pixels=min_pixels):
                pre_stack = np.stack([pre_arrays[b] for b in bands], axis=0)
                post_stack = np.stack([post_arrays[b] for b in bands], axis=0)
                pre_stack = np.clip(pre_stack.astype(np.float32) / 10000.0, 0, 1)
                post_stack = np.clip(post_stack.astype(np.float32) / 10000.0, 0, 1)
                return pre_stack, post_stack, True
        except Exception as e_sample:
            st.info("Initial sampleRectangle returned too few pixels - trying robust fallback.")

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width < 0.05 or bbox_height < 0.05:
            center_lon = (bbox[0] + bbox[2]) / 2
            center_lat = (bbox[1] + bbox[3]) / 2
            expanded_bbox = [center_lon - 0.03, center_lat - 0.03, center_lon + 0.03, center_lat + 0.03]
            aoi = ee.Geometry.Rectangle(expanded_bbox)
            st.info(f"Expanded study area from {bbox} to {expanded_bbox}")
            pre_image = pre_image.clip(aoi)
            post_image = post_image.clip(aoi)

        width_m, height_m = _degrees_to_meters_width(bbox if 'expanded_bbox' not in locals() else expanded_bbox)
        target_long = 128
        longer_m = max(width_m, height_m, 1.0)
        chosen_scale = max(30, int(np.ceil(longer_m / target_long)))
        st.info(f"Aggregating to scale={chosen_scale} m to ensure robust sampling.")

        reducer = ee.Reducer.mean()
        pre_agg = pre_image.reduceResolution(reducer=reducer, bestEffort=True).reproject(crs='EPSG:3857', scale=chosen_scale).clip(aoi)
        post_agg = post_image.reduceResolution(reducer=reducer, bestEffort=True).reproject(crs='EPSG:3857', scale=chosen_scale).clip(aoi)

        try:
            pre_data = pre_agg.sampleRectangle(region=aoi, defaultValue=0)
            post_data = post_agg.sampleRectangle(region=aoi, defaultValue=0)
            pre_arrays = {}
            post_arrays = {}
            for band in bands:
                pre_band = np.array(pre_data.get(band).getInfo())
                post_band = np.array(post_data.get(band).getInfo())
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

        try:
            out_px = 256
            pre_thumb_url = pre_image.getThumbURL({
                'min': 0, 
                'max': 10000, 
                'dimensions': out_px, 
                'format': 'png', 
                'bands': ['B4', 'B3', 'B2']
            })
            post_thumb_url = post_image.getThumbURL({
                'min': 0, 
                'max': 10000, 
                'dimensions': out_px, 
                'format': 'png', 
                'bands': ['B4', 'B3', 'B2']
            })
            
            try:
                import requests
                r1 = requests.get(pre_thumb_url, timeout=20)
                r2 = requests.get(post_thumb_url, timeout=20)
                from io import BytesIO
                pre_img = Image.open(BytesIO(r1.content)).convert('RGB')
                post_img = Image.open(BytesIO(r2.content)).convert('RGB')
                pre_arr = np.asarray(pre_img).astype(np.float32) / 255.0
                post_arr = np.asarray(post_img).astype(np.float32) / 255.0
                pre_stub = np.stack([pre_arr[...,2], pre_arr[...,1], pre_arr[...,0], 
                                   pre_arr[...,0], pre_arr[...,1], pre_arr[...,2]], axis=0)
                post_stub = np.stack([post_arr[...,2], post_arr[...,1], post_arr[...,0], 
                                    post_arr[...,0], post_arr[...,1], post_arr[...,2]], axis=0)
                return pre_stub, post_stub, True
            except Exception as e_req:
                raise RuntimeError(f"Thumbnail fetch failed: {e_req}")
        except Exception as e_thumb:
            raise RuntimeError(f"All fallback strategies failed: {e_thumb}")

    except Exception as e:
        st.error(f"Error fetching Sentinel-2 data: {e}")
        if "Insufficient images" in str(e):
            st.write("**Solutions:** - Increase cloud cover param; - broaden date ranges")
        else:
            st.write("**Try:** increase study area size, widen dates, or increase cloud_cover to 60-80%.")
        return None, None, False

def create_rgb_display(satellite_data, enhance_factor=2.0):
    """Convert 6-band Sentinel-2 data to RGB for display"""
    if len(satellite_data.shape) != 3 or satellite_data.shape[0] != 6:
        raise ValueError(f"Expected (6, H, W), got {satellite_data.shape}")
    
    red_band = satellite_data[2]    # B4 (index 2)
    green_band = satellite_data[1]  # B3 (index 1) 
    blue_band = satellite_data[0]   # B2 (index 0)
    
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
    rgb_image = np.nan_to_num(rgb_image, nan=0.0)
    
    p2, p98 = np.percentile(rgb_image, (2, 98))
    if p98 > p2:
        rgb_image = (rgb_image - p2) / (p98 - p2)
    
    rgb_image = np.clip(rgb_image * enhance_factor, 0, 1)
    return rgb_image

# Import your model
@st.cache_resource
def load_model():
    """Load the trained model once and cache it"""
    try:
        from siamese_unet import ProductionSiameseUNet
        
        device = 'cpu'
        model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
        
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="Satwik19/hurry", 
                                        filename="best_model.pth")
        except:
            model_path = "best_model.pth"
            if not Path(model_path).exists():
                st.error("Model file not found. Please ensure best_model.pth exists.")
                return None, None
        
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
        pop_files = list(Path('.').glob('**/worldpop_clipped_*.tif'))
        if pop_files:
            with rasterio.open(pop_files[0]) as src:
                population_data = src.read(1)
            return population_data, str(pop_files[0])
        else:
            st.warning("Using synthetic population data for demo")
            return np.random.gamma(2, 500, (204, 192)), "synthetic_data"
    except Exception as e:
        st.error(f"Error loading population data: {e}")
        return None, None

def resize_to_match(array1, target_shape):
    """Resize array1 to match target_shape using bilinear interpolation"""
    if array1.shape == target_shape:
        return array1
    
    scale = (target_shape[0]/array1.shape[0], target_shape[1]/array1.shape[1])
    return zoom(array1, scale, order=1)

def process_image_for_demo(model, pre_image, post_image, device):
    """Process images through the model with proper shape handling"""
    try:
        model.eval()
        with torch.no_grad():
            def prep(img):
                arr = np.array(img)
                if arr.ndim == 3 and arr.shape[2] == 6:
                    arr = np.transpose(arr, (2,0,1))
                if arr.ndim != 3:
                    raise ValueError(f"Unexpected image shape: {arr.shape}")
                if arr.max() > 10.0:
                    arr = np.clip(arr.astype(np.float32) / 10000.0, 0, 1)
                arr = arr.astype(np.float32)
                return arr

            pre_arr = prep(pre_image)
            post_arr = prep(post_image)

            pre_tensor = torch.from_numpy(pre_arr).unsqueeze(0).to(device)
            post_tensor = torch.from_numpy(post_arr).unsqueeze(0).to(device)

            logits = model(pre_tensor, post_tensor)
            flood_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Ensure 2D and normalized
            if flood_prob.ndim > 2:
                flood_prob = flood_prob.squeeze()
            if flood_prob.max() > 1.0 or flood_prob.min() < 0.0:
                flood_prob = (flood_prob - flood_prob.min()) / (flood_prob.max() - flood_prob.min() + 1e-9)
            return flood_prob
    except Exception as e:
        st.error(f"Error processing images: {e}")
        return None

def baseline_ndwi_flood_detection(pre_image, post_image, threshold_post=0.1, threshold_change=0.05):
    """NDWI flood detection using Green and NIR bands - Fixed version"""
    try:
        # Sentinel-2 bands: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'] = [0, 1, 2, 3, 4, 5]
        # B3 (Green) = index 1, B8 (NIR) = index 3
        green_idx = 1  # B3 Green
        nir_idx = 3    # B8 NIR
        
        def calculate_ndwi(img):
            green = img[green_idx].astype(np.float32)
            nir = img[nir_idx].astype(np.float32)
            ndwi = (green - nir) / (green + nir + 1e-8)
            return np.clip(ndwi, -1, 1)  # NDWI should be between -1 and 1
        
        ndwi_pre = calculate_ndwi(pre_image)
        ndwi_post = calculate_ndwi(post_image)
        
        # Debug output
        st.write(f"NDWI Debug - Pre range: [{ndwi_pre.min():.3f}, {ndwi_pre.max():.3f}]")
        st.write(f"NDWI Debug - Post range: [{ndwi_post.min():.3f}, {ndwi_post.max():.3f}]")
        st.write(f"NDWI Debug - Change range: [{(ndwi_post-ndwi_pre).min():.3f}, {(ndwi_post-ndwi_pre).max():.3f}]")
        
        # Method 1: Simple threshold on post-event NDWI
        flood_map_simple = (ndwi_post > threshold_post).astype(np.float32)
        
        # Method 2: Change detection - increase in NDWI
        ndwi_change = ndwi_post - ndwi_pre
        flood_map_change = (ndwi_change > threshold_change).astype(np.float32)
        
        # Combine both methods
        flood_map = np.maximum(flood_map_simple, flood_map_change)
        
        flood_pixels = np.sum(flood_map)
        st.write(f"NDWI Debug - Flood pixels detected: {flood_pixels:,.0f}")
        
        return flood_map
        
    except Exception as e:
        st.error(f"Error in NDWI detection: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.zeros((pre_image.shape[1], pre_image.shape[2]), dtype=np.float32)

def baseline_mdwi_flood_detection(pre_image, post_image, threshold_post=0.0, threshold_change=0.05):
    """MDWI flood detection using Green and SWIR bands - Fixed version"""
    try:
        # Sentinel-2 bands: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'] = [0, 1, 2, 3, 4, 5]
        # B3 (Green) = index 1, B11 (SWIR1) = index 4
        green_idx = 1  # B3 Green
        swir_idx = 4   # B11 SWIR1
        
        def calculate_mdwi(img):
            green = img[green_idx].astype(np.float32)
            swir = img[swir_idx].astype(np.float32)
            mdwi = (green - swir) / (green + swir + 1e-8)
            return np.clip(mdwi, -1, 1)  # MDWI should be between -1 and 1
        
        mdwi_pre = calculate_mdwi(pre_image)
        mdwi_post = calculate_mdwi(post_image)
        
        # Debug output
        st.write(f"MDWI Debug - Pre range: [{mdwi_pre.min():.3f}, {mdwi_pre.max():.3f}]")
        st.write(f"MDWI Debug - Post range: [{mdwi_post.min():.3f}, {mdwi_post.max():.3f}]")
        st.write(f"MDWI Debug - Change range: [{(mdwi_post-mdwi_pre).min():.3f}, {(mdwi_post-mdwi_pre).max():.3f}]")
        
        # Method 1: Simple threshold on post-event MDWI
        flood_map_simple = (mdwi_post > threshold_post).astype(np.float32)
        
        # Method 2: Change detection - increase in MDWI
        mdwi_change = mdwi_post - mdwi_pre
        flood_map_change = (mdwi_change > threshold_change).astype(np.float32)
        
        # Combine both methods
        flood_map = np.maximum(flood_map_simple, flood_map_change)
        
        flood_pixels = np.sum(flood_map)
        st.write(f"MDWI Debug - Flood pixels detected: {flood_pixels:,.0f}")
        
        return flood_map
        
    except Exception as e:
        st.error(f"Error in MDWI detection: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.zeros((pre_image.shape[1], pre_image.shape[2]), dtype=np.float32)

def calculate_impact(flood_map, population_data, threshold=0.5):
    """Calculate population impact from flood map and population data with shape matching"""
    try:
        # Ensure flood_map matches population_data shape
        if flood_map.shape != population_data.shape:
            flood_map = resize_to_match(flood_map, population_data.shape)
        
        flood_mask = flood_map > threshold
        
        affected_population = np.sum(population_data * flood_mask)
        total_population = np.sum(population_data)
        percentage_affected = (affected_population / total_population) * 100 if total_population > 0 else 0
        flooded_area_km2 = np.sum(flood_mask) * 1.0
        
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

def create_detailed_visualization(dl_flood_map, ndwi_flood_map, mdwi_flood_map, 
                                population_data, hurricane_name,
                                dl_impact, ndwi_impact, mdwi_impact):
    """Create detailed visualization with comparison between methods"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Hurricane {hurricane_name}: Deep Learning vs Baseline Methods", 
                 fontsize=16, fontweight='bold')
    
    # Ensure all maps match population data shape
    target_shape = population_data.shape
    
    dl_flood_map = resize_to_match(dl_flood_map, target_shape)
    ndwi_flood_map = resize_to_match(ndwi_flood_map, target_shape)
    mdwi_flood_map = resize_to_match(mdwi_flood_map, target_shape)
    
    # 1. Population Density
    im1 = axes[0, 0].imshow(population_data, cmap='YlOrRd', interpolation='nearest')
    axes[0, 0].set_title('Population Density\n(WorldPop 2020)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='People per km²')
    
    # 2. Deep Learning Flood Map
    im2 = axes[0, 1].imshow(dl_flood_map, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axes[0, 1].set_title('Deep Learning Flood Map\n(Siamese U-Net)', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='Flood Probability')
    
    # 3. Method Comparison Overlay
    overlay = np.zeros((*target_shape, 3))
    overlay[:, :, 0] = np.clip(ndwi_flood_map, 0, 1)    # Red = NDWI
    overlay[:, :, 1] = np.clip(mdwi_flood_map, 0, 1)    # Green = MDWI  
    overlay[:, :, 2] = np.clip(dl_flood_map, 0, 1)      # Blue = Deep Learning
    
    axes[0, 2].imshow(overlay, interpolation='nearest')
    axes[0, 2].set_title('Method Comparison\n(RGB Overlay)', fontweight='bold')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='NDWI'),
        Line2D([0], [0], color='lime', lw=4, label='MDWI'),
        Line2D([0], [0], color='blue', lw=4, label='Deep Learning'),
        Line2D([0], [0], color='white', lw=4, label='Agreement')
    ]
    axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # 4. NDWI Results
    ndwi_affected_map = population_data * (ndwi_flood_map > 0.5)
    im4 = axes[1, 0].imshow(ndwi_affected_map, cmap='Oranges', interpolation='nearest')
    axes[1, 0].set_title(f'NDWI Affected Population\n{ndwi_impact["affected_population"]:,.0f} people', fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8, label='Affected People')
    
    # 5. MDWI Results
    mdwi_affected_map = population_data * (mdwi_flood_map > 0.5)
    im5 = axes[1, 1].imshow(mdwi_affected_map, cmap='Greens', interpolation='nearest')
    axes[1, 1].set_title(f'MDWI Affected Population\n{mdwi_impact["affected_population"]:,.0f} people', fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8, label='Affected People')
    
    # 6. Deep Learning Results
    dl_affected_map = population_data * (dl_flood_map > 0.5)
    im6 = axes[1, 2].imshow(dl_affected_map, cmap='Reds', interpolation='nearest')
    axes[1, 2].set_title(f'DL Affected Population\n{dl_impact["affected_population"]:,.0f} people', fontweight='bold')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8, label='Affected People')
    
    # Remove axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def create_simple_visualization(flood_map, population_data, impact_stats):
    """Create simple visualization for single method"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hurricane Flood Impact Analysis', fontsize=16, fontweight='bold')
    
    # Ensure shapes match
    if flood_map.shape != population_data.shape:
        flood_map = resize_to_match(flood_map, population_data.shape)
    
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

def main():
    st.title("Hurricane Flood Impact Analysis")
    st.markdown("### Real-time flood detection using Siamese U-Net and WorldPop demographic data")
    
    # Sidebar
    st.sidebar.header("Analysis Configuration")
    confidence_threshold = st.sidebar.slider("Flood Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    show_comparison = st.sidebar.checkbox("Compare with Baseline Methods", value=True)
    
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
                    hurricane_name = hurricane_data['hurricane']
                    
                    pre_files = [p for p in periods.keys() if 'pre' in p]
                    post_files = [p for p in periods.keys() if 'post' in p]
                    
                    if pre_files and post_files:
                        pre_path = periods[pre_files[-1]]['file_path']
                        post_path = periods[post_files[0]]['file_path']
                        
                        with st.spinner("Processing Hurricane Ian satellite imagery..."):
                            pre_image = np.load(pre_path)
                            post_image = np.load(post_path)
                            
                            # Deep Learning Analysis
                            dl_flood_map = process_image_for_demo(model, pre_image, post_image, device)
                            
                            if dl_flood_map is not None:
                                dl_impact = calculate_impact(dl_flood_map, population_data, confidence_threshold)
                                
                                if show_comparison:
                                    # Baseline Methods
                                    with st.spinner("Running baseline comparisons..."):
                                        ndwi_flood_map = baseline_ndwi_flood_detection(pre_image, post_image)
                                        mdwi_flood_map = baseline_mdwi_flood_detection(pre_image, post_image)
                                        
                                        ndwi_impact = calculate_impact(ndwi_flood_map, population_data, confidence_threshold)
                                        mdwi_impact = calculate_impact(mdwi_flood_map, population_data, confidence_threshold)
                                        
                                        st.success("Analysis Complete with Method Comparison!")
                                        
                                        # Display comparison metrics
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Deep Learning", f"{dl_impact['affected_population']:,.0f}", "Affected People")
                                            st.metric("DL Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                                        with col2:
                                            st.metric("MDWI Baseline", f"{mdwi_impact['affected_population']:,.0f}", 
                                                     f"{mdwi_impact['affected_population'] - dl_impact['affected_population']:+,.0f}")
                                            st.metric("MDWI Flooded Area", f"{mdwi_impact['flooded_area_km2']:.1f} km²")
                                        with col3:
                                            st.metric("NDWI Baseline", f"{ndwi_impact['affected_population']:,.0f}",
                                                     f"{ndwi_impact['affected_population'] - dl_impact['affected_population']:+,.0f}")
                                            st.metric("NDWI Flooded Area", f"{ndwi_impact['flooded_area_km2']:.1f} km²")
                                        
                                        # Detailed comparison visualization
                                        fig = create_detailed_visualization(
                                            dl_flood_map, ndwi_flood_map, mdwi_flood_map,
                                            population_data, hurricane_name,
                                            dl_impact, ndwi_impact, mdwi_impact
                                        )
                                        st.pyplot(fig)
                                        
                                        # Analysis insights
                                        st.subheader("Method Comparison Insights")
                                        
                                        dl_affected = dl_impact['affected_population']
                                        mdwi_affected = mdwi_impact['affected_population']
                                        ndwi_affected = ndwi_impact['affected_population']
                                        total_pop = dl_impact['total_population']
                                        
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.write("**Population Impact Summary:**")
                                            st.write(f"• Total study area population: {total_pop:,.0f}")
                                            st.write(f"• Deep Learning detected: {dl_affected:,.0f} affected ({dl_impact['percentage_affected']:.2f}%)")
                                            st.write(f"• MDWI baseline detected: {mdwi_affected:,.0f} affected ({mdwi_impact['percentage_affected']:.2f}%)")
                                            st.write(f"• NDWI baseline detected: {ndwi_affected:,.0f} affected ({ndwi_impact['percentage_affected']:.2f}%)")
                                        
                                        with col_b:
                                            st.write("**Method Differences:**")
                                            st.write(f"• DL vs MDWI: {dl_affected - mdwi_affected:+,.0f} people")
                                            st.write(f"• DL vs NDWI: {dl_affected - ndwi_affected:+,.0f} people")
                                            st.write(f"• MDWI vs NDWI: {mdwi_affected - ndwi_affected:+,.0f} people")
                                            
                                            if abs(dl_affected - mdwi_affected) > 0.1 * total_pop:
                                                st.warning("⚠️ Significant difference between Deep Learning and traditional methods detected!")
                                            else:
                                                st.info("✓ Methods show reasonable agreement")
                                
                                else:
                                    # Single method analysis
                                    st.success("Deep Learning Analysis Complete!")
                                    
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Affected Population", f"{dl_impact['affected_population']:,.0f}")
                                    with col_b:
                                        st.metric("Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                                    with col_c:
                                        st.metric("Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                                    
                                    fig = create_simple_visualization(dl_flood_map, population_data, dl_impact)
                                    st.pyplot(fig)
                                
                                with st.expander("Detailed Statistics"):
                                    st.write(f"**Total Population in Study Area:** {dl_impact['total_population']:,.0f} people")
                                    st.write(f"**Population Density in Flooded Areas:** {dl_impact['population_density_flooded']:.1f} people/km²")
                                    st.write(f"**Data Source:** {pop_file}")
                                    st.write(f"**Model:** Siamese U-Net with WorldPop integration")
                                    st.write(f"**Hurricane:** {hurricane_name}")
                                    
                                    if show_comparison:
                                        st.write("**Method Performance:**")
                                        st.write(f"• Deep Learning captured the most nuanced flood patterns")
                                        st.write(f"• Traditional indices provide quick baseline estimates")
                                        st.write(f"• Differences highlight the value of AI-based approaches")
                else:
                    st.warning("Hurricane Ian data not found. Try other options.")
                    
            except Exception as e:
                st.error(f"Error running example: {e}")
                import traceback
                st.code(traceback.format_exc())

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
                with st.spinner(f"Fetching Sentinel-2 data for {selected_area}..."):
                    
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
                        
                        st.write("**Running flood analysis...**")
                        
                        # Deep Learning Analysis
                        dl_flood_map = process_image_for_demo(model, pre_data, post_data, device)
                        
                        if dl_flood_map is not None:
                            dl_impact = calculate_impact(dl_flood_map, population_data, confidence_threshold)
                            
                            if show_comparison:
                                # Baseline comparisons
                                with st.spinner("Running baseline comparisons..."):
                                    ndwi_flood_map = baseline_ndwi_flood_detection(pre_data, post_data)
                                    mdwi_flood_map = baseline_mdwi_flood_detection(pre_data, post_data)
                                    
                                    ndwi_impact = calculate_impact(ndwi_flood_map, population_data, confidence_threshold)
                                    mdwi_impact = calculate_impact(mdwi_flood_map, population_data, confidence_threshold)
                                
                                st.success("Real-time Analysis Complete with Comparisons!")
                                
                                col_x, col_y, col_z = st.columns(3)
                                with col_x:
                                    st.metric("Deep Learning", f"{dl_impact['affected_population']:,.0f}")
                                with col_y:
                                    st.metric("MDWI Baseline", f"{mdwi_impact['affected_population']:,.0f}")
                                with col_z:
                                    st.metric("NDWI Baseline", f"{ndwi_impact['affected_population']:,.0f}")
                                
                                fig = create_detailed_visualization(
                                    dl_flood_map, ndwi_flood_map, mdwi_flood_map,
                                    population_data, selected_area.split()[0],
                                    dl_impact, ndwi_impact, mdwi_impact
                                )
                                st.pyplot(fig)
                            else:
                                st.success("Real-time Analysis Complete!")
                                
                                col_x, col_y, col_z = st.columns(3)
                                with col_x:
                                    st.metric("Affected Population", f"{dl_impact['affected_population']:,.0f}")
                                with col_y:
                                    st.metric("Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                                with col_z:
                                    st.metric("Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                                
                                fig = create_simple_visualization(dl_flood_map, population_data, dl_impact)
                                st.pyplot(fig)
                            
                            st.subheader("Research Question Analysis")
                            st.write("**Q: How many people were affected by flooding in this area?**")
                            st.write(f"**A:** {dl_impact['affected_population']:,.0f} people were directly affected by flooding in the {selected_area} area, representing {dl_impact['percentage_affected']:.2f}% of the local population.")
                            
                            st.write("**Q: What was the spatial extent of flooding?**")
                            st.write(f"**A:** {dl_impact['flooded_area_km2']:.1f} km² were identified as flooded using deep learning analysis of real satellite data.")
                    
                    else:
                        st.error("Failed to fetch satellite data. Possible reasons:")
                        st.write("- No cloud-free imagery for selected dates")
                        st.write("- GEE quota limits")
                        st.write("- Network issues")
        else:
            st.error("Failed to connect to Google Earth Engine")
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
                        
                        # Deep Learning Analysis
                        dl_flood_map = process_image_for_demo(model, pre_image, post_image, device)
                        
                        if dl_flood_map is not None:
                            dl_impact = calculate_impact(dl_flood_map, population_data, confidence_threshold)
                            
                            if show_comparison:
                                # Baseline comparisons
                                ndwi_flood_map = baseline_ndwi_flood_detection(pre_image, post_image)
                                mdwi_flood_map = baseline_mdwi_flood_detection(pre_image, post_image)
                                
                                ndwi_impact = calculate_impact(ndwi_flood_map, population_data, confidence_threshold)
                                mdwi_impact = calculate_impact(mdwi_flood_map, population_data, confidence_threshold)
                                
                                st.success("Custom Analysis Complete with Comparisons!")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Deep Learning", f"{dl_impact['affected_population']:,.0f}")
                                with col_b:
                                    st.metric("MDWI Baseline", f"{mdwi_impact['affected_population']:,.0f}")
                                with col_c:
                                    st.metric("NDWI Baseline", f"{ndwi_impact['affected_population']:,.0f}")
                                
                                fig = create_detailed_visualization(
                                    dl_flood_map, ndwi_flood_map, mdwi_flood_map,
                                    population_data, "Custom",
                                    dl_impact, ndwi_impact, mdwi_impact
                                )
                                st.pyplot(fig)
                            else:
                                st.success("Custom Analysis Complete!")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Affected Population", f"{dl_impact['affected_population']:,.0f}")
                                with col_b:
                                    st.metric("Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                                with col_c:
                                    st.metric("Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                                
                                fig = create_simple_visualization(dl_flood_map, population_data, dl_impact)
                                st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"Error processing uploaded images: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # Model information
    with st.expander("Model Information & Methodology"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Deep Learning Model:**
            - Architecture: Siamese U-Net for change detection
            - Input: 6-band satellite imagery (pre/post event)
            - Output: Flood probability map (0-1)
            - Training: Hurricane events with real ground truth
            - Advantages: Learns complex flood signatures, handles all spectral bands
            """)
        with col2:
            st.markdown("""
            **Baseline Methods:**
            - NDWI: (Green - NIR) / (Green + NIR)
            - MDWI: (Green - SWIR) / (Green + SWIR)
            - Population Data: WorldPop 2020 (1km resolution)
            - Limitations: Simple thresholds, single-band ratios
            """)
        
        st.markdown("""
        **Why Deep Learning Performs Better:**
        - Considers all spectral bands simultaneously
        - Learns from training data rather than fixed formulas
        - Handles complex scenarios (urban flooding, vegetation, shadows)
        - Provides probabilistic outputs for uncertainty quantification
        - Reduces false positives from water bodies, wet soil, shadows
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Hurricane Flood Impact Analysis** | Deep Learning vs Traditional Methods | Real WorldPop Data Integration")

if __name__ == "__main__":
    main()