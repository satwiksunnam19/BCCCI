# Hurricane Analysis with Real WorldPop Population Data
# File: population_analysis_real.py

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
from scipy.ndimage import zoom
import warnings
import rasterio
warnings.filterwarnings('ignore')

# Import the reliable downloader
from pop_data_donwload import ImprovedWorldPopHandler

class RealPopulationAnalyzer:
    """Population analyzer using real WorldPop data"""
    
    def __init__(self, bbox, output_dir="hurricane_data"):
        self.bbox = bbox
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.downloader = ImprovedWorldPopHandler(output_dir)
        self.population_file = None
        self.population_data = None
        
    def load_real_population_data(self, force_redownload=False):
        """Load real WorldPop population data"""
        
        print("Loading REAL WorldPop population data...")
        
        if force_redownload or self.population_file is None:
            # Use the correct method name from ImprovedWorldPopHandler
            self.population_file, self.population_data = self.downloader.get_worldpop_data(bbox=self.bbox)
        
        if self.population_file is None:
            raise Exception("Failed to download WorldPop data. Check internet connection and try again.")
        
        # If population_data is None (full dataset), load from file
        if self.population_data is None:
            try:
                with rasterio.open(self.population_file) as src:
                    self.population_data = src.read(1)
            except Exception as e:
                raise Exception(f"Failed to load population data from {self.population_file}: {e}")
        
        if self.population_data is None:
            raise Exception(f"Failed to load population data from {self.population_file}")
        
        print(f"Successfully loaded real WorldPop data")
        print(f"   File: {self.population_file}")
        print(f"   Total population in study area: {np.sum(self.population_data):,.0f}")
        print(f"   Data resolution: {self.population_data.shape}")
        
        return self.population_data

    def baseline_ndwi_flood_detection(self, pre_image, post_image, threshold_post=0.2, threshold_pre=0.1):
        """NDWI flood detection using Green and NIR bands"""
        print("Running baseline NDWI flood detection...")
        
        try:
            if pre_image.shape[0] < 4 or post_image.shape[0] < 4:
                print("NDWI requires at least 4 bands. Using available bands.")
                green_idx = min(1, pre_image.shape[0] - 1)
                nir_idx = min(3, pre_image.shape[0] - 1)
            else:
                green_idx = 1  # Green band
                nir_idx = 3    # NIR band
            
            def calculate_ndwi(img):
                green = img[green_idx].astype(np.float32)
                nir = img[nir_idx].astype(np.float32)
                return (green - nir) / (green + nir + 1e-8)
            
            ndwi_pre = calculate_ndwi(pre_image)
            ndwi_post = calculate_ndwi(post_image)
            
            # Flood: high NDWI post-event AND low NDWI pre-event
            flood_map = ((ndwi_post > threshold_post) & (ndwi_pre < threshold_pre)).astype(np.float32)
            
            flood_pixels = np.sum(flood_map)
            print(f"   NDWI detected {flood_pixels:,} flood pixels")
            return flood_map
            
        except Exception as e:
            print(f"Error in NDWI detection: {e}")
            return np.zeros((pre_image.shape[1], pre_image.shape[2]), dtype=np.float32)

    def baseline_mdwi_flood_detection(self, pre_image, post_image, threshold_post=0.15, threshold_pre=0.1):
        """MDWI flood detection using Green and SWIR bands"""
        print("Running baseline MDWI flood detection...")
        
        try:
            # Check for SWIR band (need 5+ bands)
            if pre_image.shape[0] < 5:
                print("MDWI requires SWIR band (5+ bands total). Falling back to NDWI.")
                return self.baseline_ndwi_flood_detection(pre_image, post_image)
            
            def calculate_mdwi(img):
                green = img[1].astype(np.float32)  # Green band
                swir = img[4].astype(np.float32)   # SWIR1 band
                return (green - swir) / (green + swir + 1e-8)
            
            mdwi_pre = calculate_mdwi(pre_image)
            mdwi_post = calculate_mdwi(post_image)
            
            flood_map = ((mdwi_post > threshold_post) & (mdwi_pre < threshold_pre)).astype(np.float32)
            
            flood_pixels = np.sum(flood_map)
            print(f"   MDWI detected {flood_pixels:,} flood pixels")
            return flood_map
            
        except Exception as e:
            print(f"Error in MDWI detection: {e}")
            print("   Falling back to NDWI method...")
            return self.baseline_ndwi_flood_detection(pre_image, post_image)

    def calculate_population_impact(self, flood_map, population_data, confidence_threshold=0.5):
        """Calculate population impact using real WorldPop data"""
        print(f"Calculating population impact using REAL WorldPop data...")
        
        try:
            # Ensure same shape
            if flood_map.shape != population_data.shape:
                print(f"   Resizing flood map from {flood_map.shape} to {population_data.shape}")
                scale = (population_data.shape[0]/flood_map.shape[0], 
                        population_data.shape[1]/flood_map.shape[1])
                flood_map = zoom(flood_map, scale, order=1)
            
            # Create binary flood mask
            flood_mask = flood_map > confidence_threshold
            
            # Calculate affected population
            affected_map = population_data * flood_mask
            total_affected = np.sum(affected_map)
            total_pop = np.sum(population_data)
            percentage_affected = (total_affected / total_pop) * 100 if total_pop > 0 else 0
            
            # Calculate flooded area (WorldPop is ~1km resolution)
            flooded_pixels = np.sum(flood_mask)
            flooded_area_km2 = flooded_pixels * 1.0  # Each pixel ≈ 1 km²
            
            print(f"   REAL POPULATION IMPACT RESULTS:")
            print(f"   • Total study area population: {total_pop:,.0f} people")
            print(f"   • Affected population: {total_affected:,.0f} people")
            print(f"   • Percentage affected: {percentage_affected:.2f}%")
            print(f"   • Flooded area: {flooded_area_km2:.1f} km²")
            print(f"   • Population density in flooded areas: {total_affected/flooded_area_km2:.1f} people/km²" if flooded_area_km2 > 0 else "")
            
            return {
                'affected_population': total_affected,
                'affected_percentage': percentage_affected,
                'flooded_area_km2': flooded_area_km2,
                'total_population': total_pop,
                'population_density_flooded': total_affected/flooded_area_km2 if flooded_area_km2 > 0 else 0,
                'data_source': 'WorldPop_Real_Data'
            }, affected_map
            
        except Exception as e:
            print(f"Error calculating population impact: {e}")
            return {
                'affected_population': 0,
                'affected_percentage': 0,
                'flooded_area_km2': 0,
                'total_population': np.sum(population_data) if population_data is not None else 0,
                'data_source': 'Error'
            }, np.zeros_like(population_data) if population_data is not None else np.zeros((100, 100))

    def create_detailed_visualization(self, dl_flood_map, ndwi_flood_map, mdwi_flood_map, 
                                    population_data, dl_affected_map, hurricane_name,
                                    dl_impact, ndwi_impact, mdwi_impact):
        """Create detailed visualization with real population data"""
        
        print("Creating detailed visualization with REAL population data...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Hurricane {hurricane_name}: Population Impact Analysis (Real WorldPop Data)", 
                     fontsize=16, fontweight='bold')
        
        # Ensure all maps are same size
        target_shape = population_data.shape
        
        def resize_map(arr, target_shape):
            if arr.shape != target_shape:
                scale = (target_shape[0]/arr.shape[0], target_shape[1]/arr.shape[1])
                return zoom(arr, scale, order=1)
            return arr
        
        dl_flood_map = resize_map(dl_flood_map, target_shape)
        ndwi_flood_map = resize_map(ndwi_flood_map, target_shape)
        mdwi_flood_map = resize_map(mdwi_flood_map, target_shape)
        dl_affected_map = resize_map(dl_affected_map, target_shape)
        
        # 1. Population Density (Real WorldPop Data)
        im1 = axes[0, 0].imshow(population_data, cmap='YlOrRd', interpolation='nearest')
        axes[0, 0].set_title('A) Real Population Density\n(WorldPop 2020)', fontweight='bold', pad=15)
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.set_label('People per km²', fontsize=10)
        
        # 2. Deep Learning Flood Map
        im2 = axes[0, 1].imshow(dl_flood_map, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
        axes[0, 1].set_title('B) Deep Learning Flood Map\n(Siamese U-Net)', fontweight='bold', pad=15)
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Flood Probability', fontsize=10)
        
        # 3. Method Comparison Overlay
        overlay = np.zeros((*target_shape, 3))
        overlay[:, :, 0] = np.clip(ndwi_flood_map, 0, 1)    # Red = NDWI
        overlay[:, :, 1] = np.clip(mdwi_flood_map, 0, 1)    # Green = MDWI  
        overlay[:, :, 2] = np.clip(dl_flood_map, 0, 1)      # Blue = Deep Learning
        
        axes[0, 2].imshow(overlay, interpolation='nearest')
        axes[0, 2].set_title('C) Method Comparison\n(RGB Overlay)', fontweight='bold', pad=15)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=4, label='NDWI'),
            Line2D([0], [0], color='lime', lw=4, label='MDWI'),
            Line2D([0], [0], color='blue', lw=4, label='Deep Learning'),
            Line2D([0], [0], color='white', lw=4, label='Agreement')
        ]
        axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # 4. Affected Population Heatmap
        im4 = axes[1, 0].imshow(dl_affected_map, cmap='Reds', interpolation='nearest')
        axes[1, 0].set_title('D) Affected Population\n(Deep Learning)', fontweight='bold', pad=15)
        cbar4 = plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        cbar4.set_label('Affected People', fontsize=10)
        
        # 5. Population vs Flood Risk
        # Create risk assessment map
        risk_map = population_data * dl_flood_map  # Population weighted by flood probability
        im5 = axes[1, 1].imshow(risk_map, cmap='plasma', interpolation='nearest')
        axes[1, 1].set_title('E) Population at Risk\n(Pop × Flood Probability)', fontweight='bold', pad=15)
        cbar5 = plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        cbar5.set_label('Risk Score', fontsize=10)
        
        # 6. Detailed Statistics
        axes[1, 2].axis('off')
        axes[1, 2].set_title('F) Detailed Impact Statistics', fontweight='bold', pad=15)
        
        # Create comprehensive statistics
        total_pop = dl_impact['total_population']
        dl_affected = dl_impact['affected_population']
        ndwi_affected = ndwi_impact['affected_population']
        mdwi_affected = mdwi_impact['affected_population']
        
        stats_text = f"""
HURRICANE {hurricane_name.upper()} - REAL DATA ANALYSIS
Data Source: {dl_impact['data_source']}
Study Area: {self.bbox}

POPULATION STATISTICS:
• Total Population: {total_pop:,.0f} people
• Population Density: {total_pop/np.prod(population_data.shape):.1f} people/km²

DEEP LEARNING RESULTS:
• Affected Population: {dl_affected:,.0f} people
• Percentage Affected: {dl_impact['affected_percentage']:.2f}%
• Flooded Area: {dl_impact['flooded_area_km2']:.1f} km²
• Density in Flooded Areas: {dl_impact['population_density_flooded']:.1f} people/km²

BASELINE COMPARISONS:
• MDWI Affected: {mdwi_affected:,.0f} people ({mdwi_impact['affected_percentage']:.2f}%)
• NDWI Affected: {ndwi_affected:,.0f} people ({ndwi_impact['affected_percentage']:.2f}%)

DIFFERENCES:
• DL vs MDWI: {dl_affected - mdwi_affected:+,.0f} people
• DL vs NDWI: {dl_affected - ndwi_affected:+,.0f} people

MODEL CONFIDENCE:
• Max Flood Prob: {np.max(dl_flood_map):.3f}
• Mean Flood Prob: {np.mean(dl_flood_map[dl_flood_map > 0.1]):.3f}
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Remove axes for cleaner look
        for ax in axes.flat[:5]:  # All except the text panel
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save with detailed filename
        output_path = self.output_dir / f"hurricane_{hurricane_name.lower()}_real_worldpop_analysis.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Detailed visualization saved: {output_path}")
        
        plt.show()
        
        return output_path

def process_image_patches(model, pre_image, post_image, patch_size=256, overlap=64, device='mps'):
    """Process large images in patches - same as before"""
    print(f"Processing image in patches (size={patch_size}, overlap={overlap})")
    
    C, H, W = pre_image.shape
    stride = patch_size - overlap
    
    n_patches_h = max(1, ((H - patch_size) // stride) + 1)
    n_patches_w = max(1, ((W - patch_size) // stride) + 1)
    
    print(f"   Image size: {H}×{W}")
    print(f"   Patches needed: {n_patches_h}×{n_patches_w} = {n_patches_h * n_patches_w} total")
    
    flood_probabilities = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    failed_patches = 0
    
    with torch.no_grad():
        patch_count = 0
        total_patches = n_patches_h * n_patches_w
        
        for i in tqdm(range(n_patches_h), desc="Processing rows"):
            for j in range(n_patches_w):
                patch_count += 1
                
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
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        
                except Exception as e:
                    failed_patches += 1
                    print(f"Patch {patch_count} failed: {e}")
                    continue
    
    flood_probabilities = np.divide(flood_probabilities, count_map, 
                                  out=np.zeros_like(flood_probabilities), 
                                  where=count_map!=0)
    
    print(f"Patch processing complete!")
    print(f"   Successful patches: {total_patches - failed_patches}/{total_patches}")
    print(f"   Flood pixels detected: {np.sum(flood_probabilities > 0.5):,}")
    print(f"   Max probability: {np.max(flood_probabilities):.3f}")
    
    return flood_probabilities

def generate_dl_flood_map_efficient(model_path, pre_image_path, post_image_path, 
                                  patch_size=256, device='mps'):
    """Generate flood map using trained Siamese U-Net model"""
    print("Loading trained Siamese U-Net model...")
    
    # Try different import options for the Siamese U-Net model
    model = None
    try:
        from siamese_unet import ProductionSiameseUNet
        model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
        print("Using ProductionSiameseUNet from siamese_unet.py")
    except ImportError:
        try:
            from siamese_unet_stable import ProductionSiameseUNet
            model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
            print("Using ProductionSiameseUNet from siamese_unet_stable.py")
        except ImportError:
            try:
                from siamese_unet import SiameseUNet
                model = SiameseUNet(in_channels=6)
                print("Using SiameseUNet from siamese_unet.py")
            except ImportError:
                try:
                    from siamese_unet_stable import SiameseUNet
                    model = SiameseUNet(in_channels=6)
                    print("Using SiameseUNet from siamese_unet_stable.py")
                except ImportError:
                    print("Cannot import any Siamese U-Net model. Available files:")
                    import os
                    for f in os.listdir('.'):
                        if f.endswith('.py') and 'siamese' in f:
                            print(f"  - {f}")
                    return None
    
    model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    print("Loading satellite imagery...")
    try:
        pre_image = np.load(pre_image_path)
        post_image = np.load(post_image_path)
        print(f"Pre-image shape: {pre_image.shape}")
        print(f"Post-image shape: {post_image.shape}")
        
        if pre_image.max() > 1.0:
            pre_image = pre_image.astype(np.float32) / 255.0
            post_image = post_image.astype(np.float32) / 255.0
            
    except Exception as e:
        print(f"Error loading images: {e}")
        return None
    
    flood_probabilities = process_image_patches(
        model, pre_image, post_image, 
        patch_size=patch_size, 
        overlap=64, 
        device=device
    )
    
    return flood_probabilities

def run_real_worldpop_analysis(force_redownload=False):
    """Run hurricane impact analysis with REAL WorldPop data"""
    print("\n" + "="*70)
    print("HURRICANE IMPACT ANALYSIS - REAL WORLDPOP DATA")
    print("="*70)
    
    # Configuration
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    patch_size = 256
    model_path = 'best_model.pth'
    bbox = [-82.8, 25.8, -81.2, 27.5]  # Hurricane Ian area
    
    print(f"Using device: {device}")
    print(f"Study area (bbox): {bbox}")
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"Trained model not found: {model_path}")
        return None
    
    # Load metadata
    metadata_file = Path("hurricane_data/processing_metadata.json")
    if not metadata_file.exists():
        print("Processing metadata not found.")
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        hurricane_data = metadata['processed_data'][0]
        hurricane_name = hurricane_data['hurricane']
        periods = hurricane_data['periods']
        
        pre_periods = [p for p in periods.keys() if 'pre' in p]
        post_periods = [p for p in periods.keys() if 'post' in p]
        
        if not pre_periods or not post_periods:
            print("Need both pre and post event data")
            return None
        
        pre_period = pre_periods[-1]
        post_period = post_periods[0]
        pre_path = periods[pre_period]['file_path']
        post_path = periods[post_period]['file_path']
        
        print(f"Analyzing Hurricane {hurricane_name}")
        print(f"   Pre-event: {pre_period}")
        print(f"   Post-event: {post_period}")
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None
    
    # 1. Initialize analyzer with REAL WorldPop data
    print(f"\nStep 1: Loading REAL WorldPop Population Data")
    print("-" * 50)
    
    analyzer = RealPopulationAnalyzer(bbox)
    
    try:
        population_data = analyzer.load_real_population_data(force_redownload=force_redownload)
        print(f"Real WorldPop data loaded successfully!")
    except Exception as e:
        print(f"Failed to load WorldPop data: {e}")
        print("\nSolutions:")
        print("   1. Check your internet connection")
        print("   2. Try running with force_redownload=True")
        print("   3. Consider using alternative data sources mentioned in the downloader")
        return None
    
    # 2. Generate DL flood map
    print(f"\nStep 2: Generating Deep Learning Flood Map")
    print("-" * 50)
    
    dl_flood_map = generate_dl_flood_map_efficient(
        model_path, pre_path, post_path, 
        patch_size=patch_size, device=device
    )
    
    if dl_flood_map is None:
        print("Failed to generate DL flood map")
        return None
    
    # Downsample if needed for population analysis
    if dl_flood_map.shape[0] > 1200 or dl_flood_map.shape[1] > 1200:
        print("Downsampling flood map to match population data resolution")
        target_size = min(1000, min(dl_flood_map.shape))
        scale = target_size / max(dl_flood_map.shape)
        dl_flood_map_ds = zoom(dl_flood_map, scale, order=1)
        print(f"   Downsampled to: {dl_flood_map_ds.shape}")
    else:
        dl_flood_map_ds = dl_flood_map
    
    # 3. Load satellite images for baseline methods
    print(f"\nStep 3: Loading Satellite Images for Baseline Methods")
    print("-" * 50)
    
    try:
        pre_image = np.load(pre_path)
        post_image = np.load(post_path)
        
        if pre_image.shape[1] > 1500 or pre_image.shape[2] > 1500:
            print("Downsampling baseline images for efficiency...")
            ds_scale = min(1000 / pre_image.shape[1], 1000 / pre_image.shape[2])
            pre_image = zoom(pre_image, (1, ds_scale, ds_scale), order=1)
            post_image = zoom(post_image, (1, ds_scale, ds_scale), order=1)
            print(f"   Downsampled to: {pre_image.shape}")
            
    except Exception as e:
        print(f"Error loading baseline images: {e}")
        return None
    
    # 4. Calculate population impacts
    print(f"\nStep 4: Calculating Population Impacts with REAL Data")
    print("-" * 50)
    
    # Deep Learning Model
    print("--- Deep Learning (Siamese U-Net) with Real WorldPop ---")
    dl_impact, dl_affected_map = analyzer.calculate_population_impact(
        dl_flood_map_ds, population_data.copy()
    )
    
    # NDWI Baseline
    print("\n--- NDWI Baseline with Real WorldPop ---")
    ndwi_flood_map = analyzer.baseline_ndwi_flood_detection(pre_image, post_image)
    ndwi_impact, _ = analyzer.calculate_population_impact(
        ndwi_flood_map, population_data.copy()
    )
    
    # MDWI Baseline
    print("\n--- MDWI Baseline with Real WorldPop ---")
    mdwi_flood_map = analyzer.baseline_mdwi_flood_detection(pre_image, post_image)
    mdwi_impact, _ = analyzer.calculate_population_impact(
        mdwi_flood_map, population_data.copy()
    )
    
    # 5. Print comprehensive results
    print("\n" + "="*70)
    print("FINAL RESULTS - REAL WORLDPOP POPULATION DATA")
    print("="*70)
    
    dl_affected = dl_impact['affected_population']
    ndwi_affected = ndwi_impact['affected_population']
    mdwi_affected = mdwi_impact['affected_population']
    total_pop = dl_impact['total_population']
    
    print(f"Study Area: Hurricane {hurricane_name} Impact Zone {bbox}")
    print(f"Data Source: {dl_impact['data_source']}")
    print(f"Total Population: {total_pop:,.0f} people")
    print(f"")
    print(f"DEEP LEARNING (Siamese U-Net):")
    print(f"   • Affected Population: {dl_affected:,.0f} people")
    print(f"   • Percentage Affected: {dl_impact['affected_percentage']:.2f}%")
    print(f"   • Flooded Area: {dl_impact['flooded_area_km2']:.1f} km²")
    print(f"   • Population Density in Flooded Areas: {dl_impact['population_density_flooded']:.1f} people/km²")
    print(f"")
    print(f"MDWI BASELINE:")
    print(f"   • Affected Population: {mdwi_affected:,.0f} people")
    print(f"   • Percentage Affected: {mdwi_impact['affected_percentage']:.2f}%")
    print(f"   • Flooded Area: {mdwi_impact['flooded_area_km2']:.1f} km²")
    print(f"")
    print(f"NDWI BASELINE:")
    print(f"   • Affected Population: {ndwi_affected:,.0f} people")
    print(f"   • Percentage Affected: {ndwi_impact['affected_percentage']:.2f}%")
    print(f"   • Flooded Area: {ndwi_impact['flooded_area_km2']:.1f} km²")
    print(f"")
    print(f"METHOD COMPARISONS:")
    print(f"   • DL vs MDWI difference: {dl_affected - mdwi_affected:+,.0f} people ({((dl_affected - mdwi_affected)/total_pop)*100:+.2f}%)")
    print(f"   • DL vs NDWI difference: {dl_affected - ndwi_affected:+,.0f} people ({((dl_affected - ndwi_affected)/total_pop)*100:+.2f}%)")
    print(f"   • MDWI vs NDWI difference: {mdwi_affected - ndwi_affected:+,.0f} people")
    print("="*70)
    
    # 6. Create detailed visualization
    print(f"\nStep 5: Creating Detailed Visualization")
    print("-" * 50)
    
    visualization_path = analyzer.create_detailed_visualization(
        dl_flood_map_ds, ndwi_flood_map, mdwi_flood_map, 
        population_data, dl_affected_map, hurricane_name,
        dl_impact, ndwi_impact, mdwi_impact
    )
    
    # 7. Generate summary for professor's questions
    print(f"\nPROFESSOR Q&A PREPARATION")
    print("-" * 50)
    
    summary_for_questions = {
        'total_population': total_pop,
        'dl_affected': dl_affected,
        'dl_percentage': dl_impact['affected_percentage'],
        'flooded_area_km2': dl_impact['flooded_area_km2'],
        'population_density_flooded': dl_impact['population_density_flooded'],
        'data_source': dl_impact['data_source'],
        'hurricane_name': hurricane_name,
        'bbox': bbox,
        'method_differences': {
            'dl_vs_mdwi': dl_affected - mdwi_affected,
            'dl_vs_ndwi': dl_affected - ndwi_affected,
        }
    }
    
    print("ANALYSIS COMPLETE WITH REAL WORLDPOP DATA!")
    print(f"Results ready for academic presentation")
    print(f"Visualization saved to: {visualization_path}")
    
    return {
        'hurricane': hurricane_name,
        'dl_impact': dl_impact,
        'ndwi_impact': ndwi_impact,
        'mdwi_impact': mdwi_impact,
        'population_data': population_data,
        'visualization_path': visualization_path,
        'summary': summary_for_questions,
        'population_file': analyzer.population_file
    }

if __name__ == "__main__":
    print("Starting REAL WorldPop Hurricane Impact Analysis...")
    
    try:
        # Set force_redownload=True if you want to download fresh data
        results = run_real_worldpop_analysis(force_redownload=False)
        
        if results:
            print(f"\nSUCCESS! Real WorldPop analysis complete!")
            print(f"Population file used: {results['population_file']}")
            print(f"Total population analyzed: {results['summary']['total_population']:,.0f}")
            print(f"Affected by {results['hurricane']}: {results['summary']['dl_affected']:,.0f} people")
            print(f"\nReady to answer your professor's questions with REAL population data!")
        else:
            print("\nAnalysis failed - check error messages above")
            
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()