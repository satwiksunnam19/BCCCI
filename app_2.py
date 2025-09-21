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
from tqdm import tqdm
import gc
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

# Try to import WorldPop handler
try:
    from pop_data_donwload import ImprovedWorldPopHandler
    WORLDPOP_AVAILABLE = True
except ImportError:
    WORLDPOP_AVAILABLE = False
    st.warning("WorldPop handler not available. Using synthetic population data.")

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

# Function to fetch real satellite data
def get_hurricane_ian_hls_data(bbox, pre_date, post_date, max_cloud_cover=20):
    """Fetch Hurricane Ian HLS data from Google Earth Engine"""
    
    # Define area of interest
    aoi = ee.Geometry.Rectangle(bbox)
    
    # Define date ranges
    pre_start = (datetime.strptime(pre_date, '%Y-%m-%d') - timedelta(days=7)).strftime('%Y-%m-%d')
    post_end = (datetime.strptime(post_date, '%Y-%m-%d') + timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        # Get HLS collections
        hls_l30 = ee.ImageCollection('NASA/HLS/HLSL30/v002')
        hls_s30 = ee.ImageCollection('NASA/HLS/HLSS30/v002')
        hls_collection = hls_l30.merge(hls_s30)
        
        # Filter collections - FIXED: Use correct band names
        pre_collection = hls_collection.filterBounds(aoi).filterDate(pre_start, pre_date).filter(ee.Filter.lt('CLOUD_COVERAGE', max_cloud_cover))
        post_collection = hls_collection.filterBounds(aoi).filterDate(post_date, post_end).filter(ee.Filter.lt('CLOUD_COVERAGE', max_cloud_cover))
        
        # FIXED: Use correct band names for HLS (B2, B3, B4, etc., not B02, B03, B04)
        bands = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
        
        # Get median composites
        pre_image = pre_collection.select(bands).median().clip(aoi)
        post_image = post_collection.select(bands).median().clip(aoi)
        
        # Sample rectangle to get data
        pre_data = pre_image.sampleRectangle(region=aoi, defaultValue=0)
        post_data = post_image.sampleRectangle(region=aoi, defaultValue=0)
        
        # Convert to numpy arrays with shape validation
        pre_arrays = {}
        post_arrays = {}
        
        for band in bands:
            pre_band_data = np.array(pre_data.get(band).getInfo())
            post_band_data = np.array(post_data.get(band).getInfo())
            
            # Ensure we have 2D arrays, not 1D
            if pre_band_data.ndim == 1:
                raise ValueError(f"Pre-event data for band {band} is 1D: {pre_band_data.shape}. Area may be too small or no data available.")
            if post_band_data.ndim == 1:
                raise ValueError(f"Post-event data for band {band} is 1D: {post_band_data.shape}. Area may be too small or no data available.")
                
            pre_arrays[band] = pre_band_data
            post_arrays[band] = post_band_data
        
        # Stack arrays
        pre_stack = np.stack([pre_arrays[band] for band in bands], axis=0)
        post_stack = np.stack([post_arrays[band] for band in bands], axis=0)
        
        # Validate final shapes
        if pre_stack.ndim != 3 or pre_stack.shape[0] != 6:
            raise ValueError(f"Invalid pre_stack shape: {pre_stack.shape}, expected (6, H, W)")
        if post_stack.ndim != 3 or post_stack.shape[0] != 6:
            raise ValueError(f"Invalid post_stack shape: {post_stack.shape}, expected (6, H, W)")
        
        # Normalize (HLS is 0-10000 range)
        pre_stack = np.clip(pre_stack.astype(np.float32) / 10000.0, 0, 1)
        post_stack = np.clip(post_stack.astype(np.float32) / 10000.0, 0, 1)
        
        return pre_stack, post_stack, True
        
    except Exception as e:
        st.error(f"Error fetching HLS data: {e}")
        return None, None, False

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
def load_population_data(bbox=None):
    """Load WorldPop data - cache it for performance"""
    try:
        if WORLDPOP_AVAILABLE and bbox:
            # Try to use real WorldPop data
            downloader = ImprovedWorldPopHandler("hurricane_data")
            pop_file, pop_data = downloader.get_worldpop_data(bbox=bbox)
            
            if pop_file and pop_data is not None:
                return pop_data, f"Real WorldPop data: {pop_file}"
            elif pop_file:
                # Load from file
                with rasterio.open(pop_file) as src:
                    pop_data = src.read(1)
                return pop_data, f"Real WorldPop data: {pop_file}"
        
        # Try existing files
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
        return np.random.gamma(2, 500, (204, 192)), "synthetic_fallback_data"

def process_image_patches(model, pre_image, post_image, patch_size=256, overlap=64, device='cpu'):
    """Process large images in patches for better memory efficiency"""
    
    C, H, W = pre_image.shape
    stride = patch_size - overlap
    
    n_patches_h = max(1, ((H - patch_size) // stride) + 1)
    n_patches_w = max(1, ((W - patch_size) // stride) + 1)
    total_patches = n_patches_h * n_patches_w
    
    st.write(f"Processing in patches: {n_patches_h}×{n_patches_w} = {total_patches} total")
    
    flood_probabilities = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with torch.no_grad():
        patch_count = 0
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch_count += 1
                progress = patch_count / total_patches
                progress_bar.progress(progress)
                status_text.text(f"Processing patch {patch_count}/{total_patches}")
                
                try:
                    start_h = i * stride
                    end_h = min(start_h + patch_size, H)
                    start_w = j * stride  
                    end_w = min(start_w + patch_size, W)
                    
                    pre_patch = pre_image[:, start_h:end_h, start_w:end_w]
                    post_patch = post_image[:, start_h:end_h, start_w:end_w]
                    
                    actual_h, actual_w = pre_patch.shape[1], pre_patch.shape[2]
                    if actual_h < patch_size or actual_w < patch_size:
                        pad_h = max(0, patch_size - actual_h)
                        pad_w = max(0, patch_size - actual_w)
                        pre_patch = np.pad(pre_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                        post_patch = np.pad(post_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    
                    pre_tensor = torch.FloatTensor(pre_patch).unsqueeze(0).to(device)
                    post_tensor = torch.FloatTensor(post_patch).unsqueeze(0).to(device)
                    
                    logits = model(pre_tensor, post_tensor)
                    patch_probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
                    
                    if actual_h < patch_size or actual_w < patch_size:
                        patch_probs = patch_probs[:actual_h, :actual_w]
                    
                    flood_probabilities[start_h:end_h, start_w:end_w] += patch_probs
                    count_map[start_h:end_h, start_w:end_w] += 1.0
                    
                    del pre_tensor, post_tensor, logits
                    gc.collect()
                        
                except Exception as e:
                    st.warning(f"Patch {patch_count} failed: {e}")
                    continue
    
    progress_bar.empty()
    status_text.empty()
    
    flood_probabilities = np.divide(flood_probabilities, count_map, 
                                  out=np.zeros_like(flood_probabilities), 
                                  where=count_map!=0)
    
    return flood_probabilities

def baseline_flood_detection(pre_image, post_image, method='NDWI'):
    """Baseline flood detection using water indices"""
    
    if method == 'NDWI':
        # NDWI = (Green - NIR) / (Green + NIR)
        if pre_image.shape[0] < 4:
            green_idx, nir_idx = min(1, pre_image.shape[0]-1), min(3, pre_image.shape[0]-1)
        else:
            green_idx, nir_idx = 1, 3  # B3, B8A
            
        def calc_ndwi(img):
            green = img[green_idx].astype(np.float32)
            nir = img[nir_idx].astype(np.float32)
            return (green - nir) / (green + nir + 1e-8)
        
        ndwi_pre = calc_ndwi(pre_image)
        ndwi_post = calc_ndwi(post_image)
        
        flood_map = ((ndwi_post > 0.2) & (ndwi_pre < 0.1)).astype(np.float32)
        
    elif method == 'MDWI':
        # MDWI = (Green - SWIR) / (Green + SWIR)  
        if pre_image.shape[0] < 5:
            return baseline_flood_detection(pre_image, post_image, 'NDWI')
            
        def calc_mdwi(img):
            green = img[1].astype(np.float32)  # B3
            swir = img[4].astype(np.float32)   # B11
            return (green - swir) / (green + swir + 1e-8)
        
        mdwi_pre = calc_mdwi(pre_image)
        mdwi_post = calc_mdwi(post_image)
        
        flood_map = ((mdwi_post > 0.15) & (mdwi_pre < 0.1)).astype(np.float32)
    
    return flood_map

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

def create_comprehensive_analysis(pre_data, post_data, model, device, population_data, bbox, hurricane_name="Ian"):
    """Run comprehensive analysis like your local version"""
    
    st.subheader("Comprehensive Flood Impact Analysis")
    
    # Step 1: Generate Deep Learning flood map
    st.write("**Step 1: Generating Deep Learning Flood Map**")
    with st.spinner("Processing with Siamese U-Net..."):
        
        # Ensure proper data range
        if pre_data.max() > 1.0:
            pre_data = pre_data.astype(np.float32) / 10000.0
            post_data = post_data.astype(np.float32) / 10000.0
        
        # Use patch processing for large images
        if pre_data.shape[1] > 512 or pre_data.shape[2] > 512:
            dl_flood_map = process_image_patches(model, pre_data, post_data, 
                                               patch_size=256, device=device)
        else:
            # Process small images directly
            with torch.no_grad():
                pre_tensor = torch.FloatTensor(pre_data).unsqueeze(0).to(device)
                post_tensor = torch.FloatTensor(post_data).unsqueeze(0).to(device)
                
                logits = model(pre_tensor, post_tensor)
                dl_flood_map = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    st.success(f"Deep learning flood map generated! Shape: {dl_flood_map.shape}")
    
    # Step 2: Baseline methods
    st.write("**Step 2: Baseline Flood Detection Methods**")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner("Running NDWI..."):
            ndwi_flood_map = baseline_flood_detection(pre_data, post_data, 'NDWI')
        st.write(f"NDWI flood pixels: {np.sum(ndwi_flood_map):,.0f}")
    
    with col2:
        with st.spinner("Running MDWI..."):
            mdwi_flood_map = baseline_flood_detection(pre_data, post_data, 'MDWI')  
        st.write(f"MDWI flood pixels: {np.sum(mdwi_flood_map):,.0f}")
    
    # Step 3: Population impact calculation
    st.write("**Step 3: Population Impact Analysis**")
    
    def calc_impact(flood_map, pop_data, method_name):
        # Resize flood map to match population data
        if flood_map.shape != pop_data.shape:
            scale = (pop_data.shape[0]/flood_map.shape[0], 
                    pop_data.shape[1]/flood_map.shape[1])
            flood_map_resized = zoom(flood_map, scale, order=1)
        else:
            flood_map_resized = flood_map
            
        flood_mask = flood_map_resized > 0.5
        affected_pop = np.sum(pop_data * flood_mask)
        total_pop = np.sum(pop_data)
        percentage = (affected_pop / total_pop) * 100 if total_pop > 0 else 0
        flooded_area = np.sum(flood_mask) * 1.0  # km²
        
        return {
            'method': method_name,
            'affected_population': affected_pop,
            'percentage_affected': percentage,
            'flooded_area_km2': flooded_area,
            'total_population': total_pop
        }
    
    # Calculate impacts
    dl_impact = calc_impact(dl_flood_map, population_data, "Deep Learning")
    ndwi_impact = calc_impact(ndwi_flood_map, population_data, "NDWI")
    mdwi_impact = calc_impact(mdwi_flood_map, population_data, "MDWI")
    
    # Step 4: Results comparison
    st.write("**Step 4: Results Summary**")
    
    results_df = {
        'Method': ['Deep Learning (Siamese U-Net)', 'NDWI Baseline', 'MDWI Baseline'],
        'Affected Population': [f"{dl_impact['affected_population']:,.0f}", 
                               f"{ndwi_impact['affected_population']:,.0f}",
                               f"{mdwi_impact['affected_population']:,.0f}"],
        'Percentage Affected': [f"{dl_impact['percentage_affected']:.2f}%",
                               f"{ndwi_impact['percentage_affected']:.2f}%", 
                               f"{mdwi_impact['percentage_affected']:.2f}%"],
        'Flooded Area (km²)': [f"{dl_impact['flooded_area_km2']:.1f}",
                              f"{ndwi_impact['flooded_area_km2']:.1f}",
                              f"{mdwi_impact['flooded_area_km2']:.1f}"]
    }
    
    st.table(results_df)
    
    # Step 5: Visualization
    st.write("**Step 5: Comprehensive Visualization**")
    
    # Create the detailed visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Hurricane {hurricane_name} Flood Impact Analysis', fontsize=16, fontweight='bold')
    
    # Resize all maps to population data shape for consistency
    target_shape = population_data.shape
    
    def resize_for_viz(arr, target_shape):
        if arr.shape != target_shape:
            scale = (target_shape[0]/arr.shape[0], target_shape[1]/arr.shape[1])
            return zoom(arr, scale, order=1)
        return arr
    
    dl_flood_resized = resize_for_viz(dl_flood_map, target_shape)
    ndwi_resized = resize_for_viz(ndwi_flood_map, target_shape)
    mdwi_resized = resize_for_viz(mdwi_flood_map, target_shape)
    
    # 1. Population Density
    im1 = axes[0, 0].imshow(population_data, cmap='YlOrRd', interpolation='nearest')
    axes[0, 0].set_title('Population Density\n(WorldPop)', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, label='People/km²')
    
    # 2. Deep Learning Results
    im2 = axes[0, 1].imshow(dl_flood_resized, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axes[0, 1].set_title('Deep Learning Flood Map\n(Siamese U-Net)', fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, label='Flood Probability')
    
    # 3. Method Comparison
    overlay = np.zeros((*target_shape, 3))
    overlay[:, :, 0] = np.clip(ndwi_resized, 0, 1)    # Red = NDWI
    overlay[:, :, 1] = np.clip(mdwi_resized, 0, 1)    # Green = MDWI  
    overlay[:, :, 2] = np.clip(dl_flood_resized, 0, 1) # Blue = Deep Learning
    
    axes[0, 2].imshow(overlay, interpolation='nearest')
    axes[0, 2].set_title('Method Comparison\n(RGB Overlay)', fontweight='bold')
    
    # 4. Affected Population
    affected_map = population_data * (dl_flood_resized > 0.5)
    im4 = axes[1, 0].imshow(affected_map, cmap='Reds', interpolation='nearest')
    axes[1, 0].set_title('Affected Population\n(Deep Learning)', fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8, label='Affected People')
    
    # 5. Risk Assessment
    risk_map = population_data * dl_flood_resized
    im5 = axes[1, 1].imshow(risk_map, cmap='plasma', interpolation='nearest')
    axes[1, 1].set_title('Population at Risk\n(Pop × Flood Probability)', fontweight='bold')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8, label='Risk Score')
    
    # 6. Statistics Panel
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Impact Statistics', fontweight='bold')
    
    stats_text = f"""
HURRICANE {hurricane_name.upper()} ANALYSIS
Study Area: {bbox}
Total Population: {dl_impact['total_population']:,.0f} people

DEEP LEARNING RESULTS:
• Affected: {dl_impact['affected_population']:,.0f} people
• Percentage: {dl_impact['percentage_affected']:.2f}%
• Flooded Area: {dl_impact['flooded_area_km2']:.1f} km²

BASELINE COMPARISONS:
• NDWI: {ndwi_impact['affected_population']:,.0f} people
• MDWI: {mdwi_impact['affected_population']:,.0f} people

DIFFERENCES:
• DL vs NDWI: {dl_impact['affected_population'] - ndwi_impact['affected_population']:+,.0f}
• DL vs MDWI: {dl_impact['affected_population'] - mdwi_impact['affected_population']:+,.0f}
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Remove axes for cleaner look
    for ax in axes.flat[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig)
    
    # Return results for further analysis
    return {
        'dl_impact': dl_impact,
        'ndwi_impact': ndwi_impact,
        'mdwi_impact': mdwi_impact,
        'dl_flood_map': dl_flood_map,
        'population_data': population_data
    }

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
        population_data, pop_file = load_population_data(bbox=[-82.8, 25.8, -81.2, 27.5])  # Default Hurricane Ian area
    
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
                            
                            # Run comprehensive analysis
                            analysis_results = create_comprehensive_analysis(
                                pre_image, post_image, model, device, population_data, 
                                [-82.8, 25.8, -81.2, 27.5], hurricane_name
                            )
                            
                            if analysis_results:
                                st.success("Analysis Complete!")
                                
                                # Display results
                                dl_impact = analysis_results['dl_impact']
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Affected Population", f"{dl_impact['affected_population']:,.0f}")
                                with col_b:
                                    st.metric("Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                                with col_c:
                                    st.metric("Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                                
                                with st.expander("Detailed Statistics"):
                                    st.write(f"**Total Population in Study Area:** {dl_impact['total_population']:,.0f} people")
                                    st.write(f"**Data Source:** {pop_file}")
                                    st.write(f"**Model:** Siamese U-Net with WorldPop integration")
                                    st.write(f"**Hurricane:** {hurricane_name}")
                                    
                                    # Method comparison
                                    ndwi_impact = analysis_results['ndwi_impact']
                                    mdwi_impact = analysis_results['mdwi_impact']
                                    st.write("\n**Method Comparison:**")
                                    st.write(f"- Deep Learning: {dl_impact['affected_population']:,.0f} people")
                                    st.write(f"- MDWI Baseline: {mdwi_impact['affected_population']:,.0f} people")
                                    st.write(f"- NDWI Baseline: {ndwi_impact['affected_population']:,.0f} people")
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
                    "Fort Myers (Coastal Impact)": [-81.9, 26.5, -81.7, 26.7],
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
                    
                    pre_data, post_data, success = get_hurricane_ian_hls_data(
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
                        
                        # Check if we have actual imagery data
                        if pre_data.ndim < 3 or pre_data.shape[0] != 6:
                            st.error(f"Insufficient satellite data: shape {pre_data.shape}")
                            st.error("Possible causes:")
                            st.write("- Study area too small (try larger bounding box)")
                            st.write("- No cloud-free images available")
                            st.write("- Date range has no satellite coverage")
                            st.write("- All pixels masked as clouds/water")
                            return
                        
                        # Display satellite images
                        try:
                            # Create RGB images from satellite data
                            rgb_pre = create_rgb_display(pre_data)
                            rgb_post = create_rgb_display(post_data)
                            
                            # Display in columns
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(rgb_pre, caption="Pre-Hurricane RGB", use_column_width=True)
                            with col_b:
                                st.image(rgb_post, caption="Post-Hurricane RGB", use_column_width=True)
                                
                        except Exception as e:
                            st.error(f"Error displaying images: {e}")
                            st.write(f"Pre-data shape: {pre_data.shape if 'pre_data' in locals() else 'undefined'}")
                            st.write(f"Post-data shape: {post_data.shape if 'post_data' in locals() else 'undefined'}")
                            st.write(f"Error type: {type(e).__name__}")
                            
                            # Fallback display with individual bands
                            try:
                                st.subheader("Fallback: Individual Band Display")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.image(pre_data[0], caption="Pre - B2 (Blue)", use_column_width=True, clamp=True)
                                with col2:
                                    st.image(pre_data[1], caption="Pre - B3 (Green)", use_column_width=True, clamp=True)  
                                with col3:
                                    st.image(pre_data[2], caption="Pre - B4 (Red)", use_column_width=True, clamp=True)
                            except:
                                st.error("Could not display any satellite imagery")
                                return
                        
                        # Run comprehensive analysis instead of simple processing
                        st.write("**Running Comprehensive Hurricane Impact Analysis**")
                        
                        analysis_results = create_comprehensive_analysis(
                            pre_data, post_data, model, device, population_data, 
                            bbox, selected_area
                        )
                        
                        if analysis_results:
                            st.success("Comprehensive Analysis Complete!")
                            
                            # Display key metrics
                            dl_impact = analysis_results['dl_impact']
                            ndwi_impact = analysis_results['ndwi_impact']  
                            mdwi_impact = analysis_results['mdwi_impact']
                            
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("DL Affected Population", f"{dl_impact['affected_population']:,.0f}")
                            with col_y:
                                st.metric("DL Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                            with col_z:
                                st.metric("DL Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                                
                            # Research questions section
                            st.subheader("Research Question Analysis")
                            st.write("**Q: How many people were affected by Hurricane flooding in this area?**")
                            st.write(f"**A:** {dl_impact['affected_population']:,.0f} people were directly affected by flooding according to our Deep Learning analysis, representing {dl_impact['percentage_affected']:.2f}% of the local population.")
                            
                            st.write("**Q: How do Deep Learning methods compare to traditional indices?**")
                            dl_vs_ndwi = dl_impact['affected_population'] - ndwi_impact['affected_population']
                            dl_vs_mdwi = dl_impact['affected_population'] - mdwi_impact['affected_population']
                            st.write(f"**A:** Deep Learning detected {dl_vs_ndwi:+,.0f} more affected people than NDWI and {dl_vs_mdwi:+,.0f} more than MDWI, showing significant differences in flood detection capabilities.")
                            
                            st.write("**Q: What was the spatial extent of flooding?**")
                            st.write(f"**A:** {dl_impact['flooded_area_km2']:.1f} km² were identified as flooded using our Siamese U-Net model with real satellite data.")
                            
                            # Method comparison
                            st.subheader("Method Performance Comparison")
                            comparison_data = {
                                'Method': ['Deep Learning', 'MDWI', 'NDWI'],
                                'Affected Population': [
                                    int(dl_impact['affected_population']),
                                    int(mdwi_impact['affected_population']), 
                                    int(ndwi_impact['affected_population'])
                                ],
                                'Flooded Area (km²)': [
                                    round(dl_impact['flooded_area_km2'], 1),
                                    round(mdwi_impact['flooded_area_km2'], 1),
                                    round(ndwi_impact['flooded_area_km2'], 1)
                                ]
                            }
                            # st.table(comparison_data) impact_stats)
                            # st.pyplot(fig)
                            
                            # st.subheader("Research Question Analysis")
                            # st.write("**Q: How many people were affected by Hurricane Ian in this area?**")
                            # st.write(f"**A:** {impact_stats['affected_population']:,.0f} people were directly affected by flooding in the {selected_area} area, representing {impact_stats['percentage_affected']:.2f}% of the local population.")
                            
                            # st.write("**Q: What was the spatial extent of flooding?**")
                            # st.write(f"**A:** {impact_stats['flooded_area_km2']:.1f} km² were identified as flooded using deep learning analysis of real satellite data.")
                    
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
                        
                        # Run comprehensive analysis
                        analysis_results = create_comprehensive_analysis(
                            pre_image, post_image, model, device, population_data, 
                            "Custom Data", "User Upload"
                        )
                        
                        if analysis_results:
                            st.success("Custom Analysis Complete!")
                            
                            dl_impact = analysis_results['dl_impact']
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Affected Population", f"{dl_impact['affected_population']:,.0f}")
                            with col_b:
                                st.metric("Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                            with col_c:
                                st.metric("Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                    
                    except Exception as e:
                        st.error(f"Error processing uploaded images: {e}")
                        st.write(f"Error type: {type(e).__name__}")
                        st.write("Please ensure your files are in the correct format (.npy with 6 bands)")
    
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