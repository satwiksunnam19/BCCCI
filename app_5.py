# Google Earth Engine imports
try:
    import ee
    import folium
    from streamlit_folium import folium_static
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

# Earthaccess import
try:
    import earthaccess
    import pandas as pd
    import geopandas as gpd
    EARTHACCESS_AVAILABLE = True
except ImportError:
    EARTHACCESS_AVAILABLE = False

class ScalableHurricaneDataPipeline:
    """Simplified pipeline for Streamlit app"""
    
    def __init__(self, output_dir="hurricane_data", config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config = config or {}
        
    def authenticate_earthaccess(self):
        """Authenticate with NASA Earthdata"""
        try:
            earthaccess.login()
            return True
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return False
    
    def get_temporal_periods(self, hurricane_date, pre_days, post_days):
        """Generate simple pre/post periods"""
        hurricane_dt = datetime.strptime(hurricane_date, '%Y-%m-%d') if isinstance(hurricane_date, str) else hurricane_date
        
        pre_start = hurricane_dt - timedelta(days=pre_days)
        pre_end = hurricane_dt - timedelta(days=1)
        post_start = hurricane_dt + timedelta(days=1)
        post_end = hurricane_dt + timedelta(days=post_days)
        
        return [
            {'type': 'pre', 'start': pre_start, 'end': pre_end, 'name': 'pre_event'},
            {'type': 'post', 'start': post_start, 'end': post_end, 'name': 'post_event'}
        ]
    
    def search_hurricane_data(self, hurricane_config):
        """Search for HLS data"""
        hurricane_name = hurricane_config['name']
        hurricane_date = hurricane_config['date']
        bbox = hurricane_config['bbox']
        
        periods = self.get_temporal_periods(
            hurricane_date,
            hurricane_config['pre_days'],
            hurricane_config['post_days']
        )
        
        all_results = {}
        
        for period in periods:
            period_name = period['name']
            start_str = period['start'].strftime('%Y-%m-%d')
            end_str = period['end'].strftime('%Y-%m-%d')
            
            try:
                results = earthaccess.search_data(
                    short_name="HLSS30",
                    bounding_box=(*bbox,),
                    temporal=(start_str, end_str),
                    count=self.config.get('data_settings', {}).get('max_files_per_period', 10)
                )
                
                if results:
                    all_results[period_name] = {
                        'results': results,
                        'period_info': period,
                        'count': len(results)
                    }
                    
            except Exception as e:
                st.warning(f"Search failed for {period_name}: {e}")
                continue
        
        return all_results
    
    def download_period_data(self, period_data, hurricane_name):
        """Download data for a specific time period"""
        period_name = period_data['period_info']['name']
        results = period_data['results']
        
        hurricane_dir = self.output_dir / hurricane_name
        period_dir = hurricane_dir / period_name
        period_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            downloaded_files = earthaccess.download(
                results,
                local_path=str(period_dir)
            )
            return downloaded_files, period_dir
            
        except Exception as e:
            st.error(f"Download failed for {period_name}: {e}")
            return [], period_dir
    
    def create_enhanced_composite(self, image_folder, period_info):
        """Create composite from downloaded files"""
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob("*.tif")) + list(image_folder.glob("*.TIF"))
        
        if not image_files:
            return None, {}
        
        # Group by bands
        band_files = {}
        target_bands = self.config.get('data_settings', {}).get('target_bands', 
                                                                ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'])
        
        for img_file in image_files:
            filename = img_file.name
            for band in target_bands:
                if band in filename:
                    if band not in band_files:
                        band_files[band] = []
                    band_files[band].append(img_file)
                    break
        
        if not band_files:
            # Fallback to first 6 files
            composite = self._process_unstructured_files(image_files)
            metadata = {'method': 'unstructured', 'files': len(image_files)}
            return composite, metadata
        
        # Process each band
        composite_bands = []
        band_metadata = {}
        
        for band in target_bands:
            if band in band_files:
                band_images = []
                
                for band_file in band_files[band]:
                    processed = self._process_single_file(band_file)
                    if processed is not None:
                        band_images.append(processed)
                
                if band_images:
                    band_stack = np.stack(band_images, axis=0)
                    band_composite = np.nanmedian(band_stack, axis=0)
                    composite_bands.append(band_composite)
                    band_metadata[band] = {'files_used': len(band_images)}
                else:
                    dummy_band = np.full((512, 512), 0.1, dtype=np.float32)
                    composite_bands.append(dummy_band)
                    band_metadata[band] = {'status': 'dummy'}
            else:
                dummy_band = np.full((512, 512), 0.1, dtype=np.float32)
                composite_bands.append(dummy_band)
                band_metadata[band] = {'status': 'missing'}
        
        if composite_bands:
            composite = np.stack(composite_bands, axis=0)
            metadata = {
                'shape': composite.shape,
                'bands': target_bands,
                'band_details': band_metadata,
                'period_info': period_info
            }
            return composite, metadata
        
        return None, {}
    
    def _process_single_file(self, file_path):
        """Process a single satellite file"""
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                data = np.where(data == -9999, np.nan, data)
                data = np.where(data < 0, np.nan, data)
                data = data.astype(np.float32) * 0.0001  # HLS2 scale factor
                return data
        except Exception as e:
            return None
    
    def _process_unstructured_files(self, image_files):
        """Process files when band structure is unclear"""
        processed_images = []
        
        for img_file in image_files[:6]:
            processed = self._process_single_file(img_file)
            if processed is not None:
                processed_images.append(processed)
        
        if processed_images:
            while len(processed_images) < 6:
                processed_images.append(processed_images[-1])
            return np.stack(processed_images[:6], axis=0)
        
        return None
    
    def normalize_composite(self, composite):
        """Normalize composite using percentile stretch"""
        p2, p98 = np.nanpercentile(composite, (2, 98), axis=(1, 2), keepdims=True)
        normalized = (composite - p2) / (p98 - p2 + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        return normalized
    
    def quality_check_composite(self, composite, metadata):
        """Perform quality checks on composite"""
        issues = []
        
        # Check for excessive NaN values
        nan_percentage = np.sum(np.isnan(composite)) / composite.size * 100
        if nan_percentage > 20:
            issues.append(f"High NaN percentage: {nan_percentage:.1f}%")
        
        # Check value ranges
        target_bands = self.config.get('data_settings', {}).get('target_bands', 
                                                               ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'])
        for i, band in enumerate(target_bands):
            if i < composite.shape[0]:
                band_data = composite[i]
                min_val, max_val = np.nanmin(band_data), np.nanmax(band_data)
                
                if min_val == max_val:
                    issues.append(f"Band {band} has constant values")
                elif max_val > 2.0 or min_val < -0.5:
                    issues.append(f"Band {band} has unusual range: {min_val:.3f} to {max_val:.3f}")
        
        return issues# Hurricane Flood Impact Analysis Web App - Enhanced
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
    """Process images through the model with proper shape handling - matches training pipeline"""
    try:
        model.eval()
        with torch.no_grad():
            def prep_image(img):
                """Prepare image exactly like in training pipeline"""
                arr = np.array(img, dtype=np.float32)
                
                # Handle different input formats
                if arr.ndim == 3:
                    if arr.shape[2] == 6:  # (H, W, 6) -> (6, H, W)
                        arr = np.transpose(arr, (2, 0, 1))
                    elif arr.shape[0] != 6:  # Not in correct format
                        raise ValueError(f"Expected 6 bands, got {arr.shape}")
                elif arr.ndim != 3:
                    raise ValueError(f"Expected 3D array, got {arr.shape}")
                
                # Apply the same preprocessing as training pipeline
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Scale if needed (HLS data scaling)
                if arr.max() > 10.0:  # Likely raw satellite values
                    arr = arr * 0.0001  # HLS2 scale factor
                    arr = np.clip(arr, 0, 1)
                elif arr.max() > 1.0 and arr.max() <= 10.0:  # Already scaled but > 1
                    arr = np.clip(arr / arr.max(), 0, 1)
                
                # Apply normalization like in training (percentile stretch)
                p2, p98 = np.percentile(arr, (2, 98), axis=(1, 2), keepdims=True)
                normalized = (arr - p2) / (p98 - p2 + 1e-8)
                normalized = np.clip(normalized, 0, 1)
                
                return normalized.astype(np.float32)

            # Prepare both images
            pre_processed = prep_image(pre_image)
            post_processed = prep_image(post_image)
            
            # Ensure same spatial dimensions
            min_h = min(pre_processed.shape[1], post_processed.shape[1])
            min_w = min(pre_processed.shape[2], post_processed.shape[2])
            pre_processed = pre_processed[:, :min_h, :min_w]
            post_processed = post_processed[:, :min_h, :min_w]
            
            # Convert to tensors
            pre_tensor = torch.from_numpy(pre_processed).unsqueeze(0).to(device)
            post_tensor = torch.from_numpy(post_processed).unsqueeze(0).to(device)
            
            # Run inference
            logits = model(pre_tensor, post_tensor)
            flood_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # Ensure 2D output
            if flood_prob.ndim > 2:
                flood_prob = flood_prob.squeeze()
            
            return flood_prob
            
    except Exception as e:
        st.error(f"Error processing images: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def process_image_patches(model, pre_image, post_image, patch_size=256, overlap=64, device='cpu'):
    """Process large images in patches - same as population_analysis.py"""
    
    # Prepare images like in training
    def prep_for_patches(img):
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[2] == 6:
            arr = np.transpose(arr, (2, 0, 1))
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if arr.max() > 10.0:
            arr = arr * 0.0001
            arr = np.clip(arr, 0, 1)
        return arr.astype(np.float32)
    
    pre_processed = prep_for_patches(pre_image)
    post_processed = prep_for_patches(post_image)
    
    # Ensure same dimensions
    C, H, W = pre_processed.shape
    min_h = min(H, post_processed.shape[1])
    min_w = min(W, post_processed.shape[2])
    pre_processed = pre_processed[:, :min_h, :min_w]
    post_processed = post_processed[:, :min_h, :min_w]
    
    C, H, W = pre_processed.shape
    stride = patch_size - overlap
    
    n_patches_h = max(1, ((H - patch_size) // stride) + 1)
    n_patches_w = max(1, ((W - patch_size) // stride) + 1)
    
    st.write(f"Processing in patches: {n_patches_h}x{n_patches_w} = {n_patches_h * n_patches_w} total")
    
    flood_probabilities = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    progress_bar = st.progress(0)
    patch_count = 0
    total_patches = n_patches_h * n_patches_w
    
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch_count += 1
                progress_bar.progress(patch_count / total_patches)
                
                try:
                    start_h = i * stride
                    end_h = min(start_h + patch_size, H)
                    start_w = j * stride  
                    end_w = min(start_w + patch_size, W)
                    
                    pre_patch = pre_processed[:, start_h:end_h, start_w:end_w]
                    post_patch = post_processed[:, start_h:end_h, start_w:end_w]
                    
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
                        
                except Exception as e:
                    st.warning(f"Patch {patch_count} failed: {e}")
                    continue
    
    progress_bar.empty()
    flood_probabilities = np.divide(flood_probabilities, count_map, 
                                  out=np.zeros_like(flood_probabilities), 
                                  where=count_map!=0)
    
    st.success(f"Patch processing complete! Flood pixels detected: {np.sum(flood_probabilities > 0.5):,}")
    return flood_probabilities

def baseline_ndwi_flood_detection(pre_image, post_image, threshold_post=0.1, threshold_change=0.05):
    """NDWI flood detection using Green and NIR bands - Fixed version"""
    try:
        # Ensure proper format (6, H, W)
        def prep_baseline(img):
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[2] == 6:
                arr = np.transpose(arr, (2, 0, 1))
            return arr
        
        pre_image = prep_baseline(pre_image)
        post_image = prep_baseline(post_image)
        
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
        # Ensure proper format (6, H, W)  
        def prep_baseline(img):
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[2] == 6:
                arr = np.transpose(arr, (2, 0, 1))
            return arr
        
        pre_image = prep_baseline(pre_image)
        post_image = prep_baseline(post_image)
        
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
    
    st.sidebar.subheader("Processing Options")
    use_patch_processing = st.sidebar.checkbox("Use Patch Processing (for large images)", value=False)
    if use_patch_processing:
        patch_size = st.sidebar.slider("Patch Size", 128, 512, 256, 64)
        overlap = st.sidebar.slider("Patch Overlap", 32, 128, 64, 16)
    
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
                            if use_patch_processing:
                                st.info("Using patch-based processing for large images...")
                                dl_flood_map = process_image_patches(model, pre_image, post_image, 
                                                                   patch_size, overlap, device)
                            else:
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
        st.subheader("Option 2: Fetch New Hurricane Data & Analyze")
        
        if not EARTHACCESS_AVAILABLE:
            st.error("NASA Earthaccess not available. Install required packages:")
            st.code("pip install earthaccess pandas geopandas")
            return
        
        # Hurricane configuration interface
        st.write("Configure hurricane event to fetch and analyze:")
        
        # Predefined hurricane database
        hurricane_database = {
            "Hurricane Ian (2022) - Florida": {
                'name': 'Ian', 'date': '2022-09-28', 'bbox': [-82.8, 25.8, -81.2, 27.5],
                'description': 'Category 4 hurricane that devastated Southwest Florida'
            },
            "Hurricane Fiona (2022) - Puerto Rico": {
                'name': 'Fiona', 'date': '2022-09-18', 'bbox': [-67.5, 17.8, -65.2, 19.0],
                'description': 'Category 1 hurricane causing widespread flooding in Puerto Rico'
            },
            "Hurricane Ida (2021) - Louisiana": {
                'name': 'Ida', 'date': '2021-08-29', 'bbox': [-92.0, 29.0, -89.5, 31.0],
                'description': 'Category 4 hurricane that hit Louisiana as one of the strongest storms on record'
            },
            "Hurricane Laura (2020) - Louisiana/Texas": {
                'name': 'Laura', 'date': '2020-08-27', 'bbox': [-94.5, 29.0, -92.0, 31.5],
                'description': 'Category 4 hurricane with 150 mph winds hitting Louisiana-Texas border'
            },
            "Hurricane Sally (2020) - Alabama/Florida": {
                'name': 'Sally', 'date': '2020-09-16', 'bbox': [-88.0, 30.0, -86.0, 31.5],
                'description': 'Slow-moving hurricane causing catastrophic flooding'
            },
            "Hurricane Dorian (2019) - Bahamas/Carolinas": {
                'name': 'Dorian', 'date': '2019-09-06', 'bbox': [-79.0, 32.5, -75.0, 36.0],
                'description': 'Category 5 hurricane devastating the Bahamas, impacting US East Coast'
            },
            "Hurricane Michael (2018) - Florida Panhandle": {
                'name': 'Michael', 'date': '2018-10-10', 'bbox': [-86.0, 29.5, -84.0, 31.0],
                'description': 'Category 5 hurricane with 160 mph winds hitting Florida Panhandle'
            },
            "Hurricane Florence (2018) - North Carolina": {
                'name': 'Florence', 'date': '2018-09-14', 'bbox': [-79.5, 33.5, -76.5, 36.0],
                'description': 'Slow-moving Category 1 hurricane causing extreme flooding'
            },
            "Hurricane Irma (2017) - Florida Keys": {
                'name': 'Irma', 'date': '2017-09-10', 'bbox': [-82.0, 24.5, -80.0, 26.0],
                'description': 'Category 4 hurricane impacting the Florida Keys and peninsula'
            },
            "Hurricane Harvey (2017) - Texas": {
                'name': 'Harvey', 'date': '2017-08-25', 'bbox': [-96.5, 27.5, -94.0, 30.0],
                'description': 'Category 4 hurricane causing catastrophic flooding in Houston area'
            },
            "Hurricane Maria (2017) - Puerto Rico": {
                'name': 'Maria', 'date': '2017-09-20', 'bbox': [-67.5, 17.5, -65.0, 18.8],
                'description': 'Category 4 hurricane devastating Puerto Rico infrastructure'
            },
            "Hurricane Matthew (2016) - Florida/Carolinas": {
                'name': 'Matthew', 'date': '2016-10-07', 'bbox': [-81.5, 26.0, -79.0, 32.0],
                'description': 'Category 3 hurricane affecting Florida East Coast and Carolinas'
            },
            "Hurricane Sandy (2012) - New York/New Jersey": {
                'name': 'Sandy', 'date': '2012-10-29', 'bbox': [-75.0, 39.5, -73.0, 41.0],
                'description': 'Superstorm causing massive storm surge in NYC area'
            },
            "Hurricane Irene (2011) - North Carolina/New York": {
                'name': 'Irene', 'date': '2011-08-27', 'bbox': [-78.0, 34.0, -74.0, 42.0],
                'description': 'Category 1 hurricane causing flooding from NC to New England'
            },
            "Hurricane Ike (2008) - Texas": {
                'name': 'Ike', 'date': '2008-09-13', 'bbox': [-95.5, 28.5, -94.0, 30.0],
                'description': 'Category 2 hurricane with massive storm surge affecting Galveston'
            },
            "Hurricane Gustav (2008) - Louisiana": {
                'name': 'Gustav', 'date': '2008-09-01', 'bbox': [-91.5, 29.0, -89.5, 30.5],
                'description': 'Category 2 hurricane hitting Louisiana 3 years after Katrina'
            },
            "Hurricane Katrina (2005) - Louisiana/Mississippi": {
                'name': 'Katrina', 'date': '2005-08-29', 'bbox': [-90.5, 29.0, -88.5, 31.0],
                'description': 'Category 3 hurricane causing catastrophic flooding in New Orleans'
            },
            "Hurricane Rita (2005) - Texas/Louisiana": {
                'name': 'Rita', 'date': '2005-09-24', 'bbox': [-94.0, 29.0, -92.0, 30.5],
                'description': 'Category 3 hurricane affecting Texas-Louisiana border region'
            },
            "Hurricane Wilma (2005) - Florida": {
                'name': 'Wilma', 'date': '2005-10-24', 'bbox': [-81.5, 25.0, -79.5, 27.0],
                'description': 'Category 3 hurricane crossing South Florida'
            },
            "Hurricane Charley (2004) - Florida": {
                'name': 'Charley', 'date': '2004-08-13', 'bbox': [-82.5, 26.0, -81.0, 28.0],
                'description': 'Compact Category 4 hurricane hitting Southwest Florida'
            },
            "Hurricane Frances (2004) - Florida": {
                'name': 'Frances', 'date': '2004-09-05', 'bbox': [-81.0, 26.0, -79.5, 28.0],
                'description': 'Category 2 hurricane affecting Florida East Coast'
            },
            "Hurricane Jeanne (2004) - Florida": {
                'name': 'Jeanne', 'date': '2004-09-26', 'bbox': [-81.0, 26.5, -79.5, 28.5],
                'description': 'Category 3 hurricane following similar path to Frances'
            },
            "Custom Hurricane": {
                'name': '', 'date': '', 'bbox': [],
                'description': 'Enter your own hurricane parameters'
            }
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hurricane selection dropdown
            selected_hurricane = st.selectbox(
                "Select Hurricane Event:",
                list(hurricane_database.keys()),
                index=0
            )
            
            hurricane_info = hurricane_database[selected_hurricane]
            
            if selected_hurricane != "Custom Hurricane":
                # Pre-filled values
                hurricane_name = st.text_input("Hurricane Name", value=hurricane_info['name'])
                hurricane_date = st.date_input("Hurricane Date", 
                                             value=datetime.strptime(hurricane_info['date'], '%Y-%m-%d'),
                                             min_value=datetime(2000, 1, 1),
                                             max_value=datetime(2024, 12, 31))
                
                # Display description
                st.info(f"**{selected_hurricane}:** {hurricane_info['description']}")
                
                # Bounding box
                bbox_str = ', '.join(map(str, hurricane_info['bbox']))
                st.write("Study Area Bounding Box (lon_min, lat_min, lon_max, lat_max):")
                bbox_input = st.text_input("Bbox", value=bbox_str)
            else:
                # Custom input
                hurricane_name = st.text_input("Hurricane Name", value="CustomStorm")
                hurricane_date = st.date_input("Hurricane Date", 
                                             value=datetime(2022, 9, 18),
                                             min_value=datetime(2000, 1, 1),
                                             max_value=datetime(2024, 12, 31))
                
                st.write("Study Area Bounding Box (lon_min, lat_min, lon_max, lat_max):")
                bbox_input = st.text_input("Bbox", value="-67.5, 17.8, -65.2, 19.0", 
                                          help="Example: -67.5, 17.8, -65.2, 19.0 for Puerto Rico")
            
            try:
                bbox = [float(x.strip()) for x in bbox_input.split(',')]
                if len(bbox) != 4:
                    st.error("Please provide exactly 4 values")
                    return
            except:
                st.error("Invalid bbox format. Use: lon_min, lat_min, lon_max, lat_max")
                return
        
        with col2:
            pre_days = st.slider("Pre-event days", 7, 60, 21)
            post_days = st.slider("Post-event days", 7, 60, 21)
            max_files = st.slider("Max files per period", 3, 20, 5)
            cloud_threshold = st.slider("Max cloud cover", 10, 80, 30)
        
        if st.button("Fetch Hurricane Data & Analyze", type="primary"):
            # Create hurricane configuration
            hurricane_config = {
                'name': hurricane_name,
                'date': hurricane_date.strftime('%Y-%m-%d'),
                'bbox': bbox,
                'pre_days': pre_days,
                'post_days': post_days
            }
            
            data_config = {
                'hurricanes': [hurricane_config],
                'data_settings': {
                    'temporal_sampling': 'single',  # Single pre/post periods
                    'max_files_per_period': max_files,
                    'composite_method': 'median',
                    'target_bands': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'],
                }
            }
            
            # Initialize pipeline
            with st.spinner("Initializing data pipeline..."):
                try:
                    pipeline = ScalableHurricaneDataPipeline(
                        output_dir=f"hurricane_data_{hurricane_name.lower()}",
                        config=data_config
                    )
                    
                    # Authenticate with NASA Earthdata
                    if not pipeline.authenticate_earthaccess():
                        st.error("Failed to authenticate with NASA Earthdata")
                        return
                    
                    st.success("Authenticated with NASA Earthdata")
                except Exception as e:
                    st.error(f"Pipeline initialization failed: {e}")
                    return
            
            # Search and download data
            with st.spinner(f"Searching for {hurricane_name} satellite data..."):
                try:
                    search_results = pipeline.search_hurricane_data(hurricane_config)
                    if not search_results:
                        st.error("No satellite data found for the specified parameters")
                        return
                    
                    st.success(f"Found data for {len(search_results)} periods")
                    
                    # Display what was found
                    for period_name, period_data in search_results.items():
                        st.write(f"  - {period_name}: {period_data['count']} files")
                        
                except Exception as e:
                    st.error(f"Data search failed: {e}")
                    return
            
            # Download and process data
            processed_periods = {}
            for period_name, period_data in search_results.items():
                with st.spinner(f"Processing {period_name}..."):
                    try:
                        # Download
                        downloaded_files, period_dir = pipeline.download_period_data(
                            period_data, hurricane_name
                        )
                        
                        if not downloaded_files:
                            st.warning(f"No files downloaded for {period_name}")
                            continue
                        
                        st.write(f"Downloaded {len(downloaded_files)} files for {period_name}")
                        
                        # Create composite
                        composite, metadata = pipeline.create_enhanced_composite(
                            period_dir, period_data['period_info']
                        )
                        
                        if composite is None:
                            st.error(f"Failed to create composite for {period_name}")
                            continue
                        
                        # Quality check
                        issues = pipeline.quality_check_composite(composite, metadata)
                        if issues:
                            st.warning(f"Quality issues in {period_name}: {'; '.join(issues)}")
                        
                        # Normalize
                        normalized = pipeline.normalize_composite(composite)
                        
                        # Store for analysis
                        processed_periods[period_name] = {
                            'data': normalized,
                            'metadata': metadata,
                            'issues': issues
                        }
                        
                        st.success(f"Processed {period_name}: {normalized.shape}")
                        
                    except Exception as e:
                        st.error(f"Processing failed for {period_name}: {e}")
                        continue
            
            # Check if we have both pre and post data
            pre_periods = [p for p in processed_periods.keys() if 'pre' in p]
            post_periods = [p for p in processed_periods.keys() if 'post' in p]
            
            if not pre_periods or not post_periods:
                st.error("Need both pre-event and post-event data for analysis")
                return
            
            # Use the data for analysis
            pre_data = processed_periods[pre_periods[0]]['data']
            post_data = processed_periods[post_periods[0]]['data']
            
            st.success(f"Ready for analysis with {pre_periods[0]} and {post_periods[0]}")
            
            # Run flood analysis
            with st.spinner("Running flood analysis..."):
                try:
                    # Deep Learning Analysis
                    if use_patch_processing and (pre_data.shape[1] > 512 or pre_data.shape[2] > 512):
                        st.info("Using patch-based processing for large images...")
                        dl_flood_map = process_image_patches(model, pre_data, post_data, 
                                                           patch_size, overlap, device)
                    else:
                        dl_flood_map = process_image_for_demo(model, pre_data, post_data, device)
                    
                    if dl_flood_map is not None:
                        dl_impact = calculate_impact(dl_flood_map, population_data, confidence_threshold)
                        
                        if show_comparison:
                            # Baseline Methods
                            with st.spinner("Running baseline comparisons..."):
                                ndwi_flood_map = baseline_ndwi_flood_detection(pre_data, post_data)
                                mdwi_flood_map = baseline_mdwi_flood_detection(pre_data, post_data)
                                
                                ndwi_impact = calculate_impact(ndwi_flood_map, population_data, confidence_threshold)
                                mdwi_impact = calculate_impact(mdwi_flood_map, population_data, confidence_threshold)
                            
                            st.success(f"Analysis Complete for Hurricane {hurricane_name}!")
                            
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
                            st.subheader(f"Hurricane {hurricane_name} Impact Analysis")
                            
                            dl_affected = dl_impact['affected_population']
                            mdwi_affected = mdwi_impact['affected_population']
                            ndwi_affected = ndwi_impact['affected_population']
                            total_pop = dl_impact['total_population']
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write("**Population Impact Summary:**")
                                st.write(f"• Study area: {bbox}")
                                st.write(f"• Total population: {total_pop:,.0f}")
                                st.write(f"• Deep Learning detected: {dl_affected:,.0f} affected ({dl_impact['percentage_affected']:.2f}%)")
                                st.write(f"• MDWI detected: {mdwi_affected:,.0f} affected ({mdwi_impact['percentage_affected']:.2f}%)")
                                st.write(f"• NDWI detected: {ndwi_affected:,.0f} affected ({ndwi_impact['percentage_affected']:.2f}%)")
                            
                            with col_b:
                                st.write("**Data Quality & Processing:**")
                                st.write(f"• Pre-event period: {pre_periods[0]}")
                                st.write(f"• Post-event period: {post_periods[0]}")
                                st.write(f"• Data shape: {pre_data.shape}")
                                st.write(f"• Processing method: {'Patch-based' if use_patch_processing else 'Full image'}")
                                
                                if processed_periods[pre_periods[0]]['issues']:
                                    st.write("• Quality issues detected - see warnings above")
                        
                        else:
                            # Single method analysis
                            st.success(f"Deep Learning Analysis Complete for Hurricane {hurricane_name}!")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Affected Population", f"{dl_impact['affected_population']:,.0f}")
                            with col_b:
                                st.metric("Percentage Affected", f"{dl_impact['percentage_affected']:.2f}%")
                            with col_c:
                                st.metric("Flooded Area", f"{dl_impact['flooded_area_km2']:.1f} km²")
                            
                            fig = create_simple_visualization(dl_flood_map, population_data, dl_impact)
                            st.pyplot(fig)
                        
                        # Data summary
                        with st.expander("Processing Details"):
                            st.write(f"**Hurricane:** {hurricane_name} ({hurricane_date})")
                            st.write(f"**Study Area:** {bbox}")
                            st.write(f"**Pre-event data:** {pre_periods[0]} - {pre_data.shape}")
                            st.write(f"**Post-event data:** {post_periods[0]} - {post_data.shape}")
                            st.write(f"**Total Population:** {dl_impact['total_population']:,.0f} people")
                            st.write(f"**Population Density in Flooded Areas:** {dl_impact['population_density_flooded']:.1f} people/km²")
                            st.write(f"**Model:** Siamese U-Net with real-time satellite data")
                    
                    else:
                        st.error("Failed to generate flood map")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

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
                        if use_patch_processing:
                            st.info("Using patch-based processing for uploaded images...")
                            dl_flood_map = process_image_patches(model, pre_image, post_image, 
                                                               patch_size, overlap, device)
                        else:
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