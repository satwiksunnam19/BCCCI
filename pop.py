# DL Model Integrated Analysis - Memory Efficient Version
# File: dl_integrated_analysis_efficient.py

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from population_analysis import PopulationAnalyzer
from siamese_unet import ProductionSiameseUNet
import gc
from tqdm import tqdm

def process_image_patches(model, pre_image, post_image, patch_size=512, overlap=64, device='cpu'):
    """
    Process large images in patches to avoid memory issues
    
    Args:
        model: Trained Siamese U-Net model
        pre_image: Pre-event satellite image (C, H, W)
        post_image: Post-event satellite image (C, H, W)
        patch_size: Size of patches to process
        overlap: Overlap between patches to avoid edge artifacts
        device: Device to run inference on
    
    Returns:
        flood_probabilities: Full-size flood probability map
    """
    print(f"🔄 Processing image in patches (size={patch_size}, overlap={overlap})")
    
    C, H, W = pre_image.shape
    stride = patch_size - overlap
    
    # Calculate number of patches needed
    n_patches_h = ((H - patch_size) // stride) + 1
    n_patches_w = ((W - patch_size) // stride) + 1
    
    # If image is smaller than patch_size, adjust
    if H < patch_size:
        n_patches_h = 1
        patch_h = H
    else:
        patch_h = patch_size
        
    if W < patch_size:
        n_patches_w = 1
        patch_w = W
    else:
        patch_w = patch_size
    
    print(f"   Image size: {H}×{W}")
    print(f"   Patches needed: {n_patches_h}×{n_patches_w} = {n_patches_h * n_patches_w} total")
    
    # Initialize output arrays
    flood_probabilities = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)  # To handle overlaps
    
    model.eval()
    
    with torch.no_grad():
        patch_count = 0
        total_patches = n_patches_h * n_patches_w
        
        for i in tqdm(range(n_patches_h), desc="Processing rows"):
            for j in range(n_patches_w):
                patch_count += 1
                
                # Calculate patch boundaries
                start_h = i * stride
                end_h = min(start_h + patch_h, H)
                start_w = j * stride  
                end_w = min(start_w + patch_w, W)
                
                # Extract patches
                pre_patch = pre_image[:, start_h:end_h, start_w:end_w]
                post_patch = post_image[:, start_h:end_h, start_w:end_w]
                
                # Ensure patch is the right size (pad if necessary)
                actual_h, actual_w = pre_patch.shape[1], pre_patch.shape[2]
                if actual_h < patch_h or actual_w < patch_w:
                    # Pad the patch
                    pad_h = max(0, patch_h - actual_h)
                    pad_w = max(0, patch_w - actual_w)
                    
                    pre_patch = np.pad(pre_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    post_patch = np.pad(post_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                
                # Convert to tensors and add batch dimension
                pre_tensor = torch.FloatTensor(pre_patch).unsqueeze(0).to(device)
                post_tensor = torch.FloatTensor(post_patch).unsqueeze(0).to(device)
                
                # Run inference
                try:
                    logits = model(pre_tensor, post_tensor)
                    patch_probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
                    
                    # Remove padding if it was added
                    if actual_h < patch_h or actual_w < patch_w:
                        patch_probs = patch_probs[:actual_h, :actual_w]
                    
                    # Add to output (handling overlaps by averaging)
                    flood_probabilities[start_h:end_h, start_w:end_w] += patch_probs
                    count_map[start_h:end_h, start_w:end_w] += 1.0
                    
                except RuntimeError as e:
                    print(f"❌ Error processing patch {patch_count}/{total_patches}: {e}")
                    continue
                
                # Clean up tensors
                del pre_tensor, post_tensor, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Progress update
                if patch_count % 10 == 0:
                    print(f"   Processed {patch_count}/{total_patches} patches...")
    
    # Average overlapping regions
    flood_probabilities = np.divide(flood_probabilities, count_map, 
                                  out=np.zeros_like(flood_probabilities), 
                                  where=count_map!=0)
    
    print(f"✅ Patch processing complete!")
    print(f"   Flood pixels detected: {np.sum(flood_probabilities > 0.5):,}")
    print(f"   Max probability: {np.max(flood_probabilities):.3f}")
    print(f"   Mean probability: {np.mean(flood_probabilities):.3f}")
    
    return flood_probabilities

def generate_dl_flood_map_efficient(model_path, pre_image_path, post_image_path, 
                                  patch_size=512, device='mps'):
    """Generate flood map using trained Siamese U-Net model with memory-efficient processing"""
    print("🤖 Loading trained Siamese U-Net model...")
    
    # Load model architecture
    model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
    
    # Load trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load satellite images
    print("📡 Loading satellite imagery...")
    try:
        pre_image = np.load(pre_image_path)
        post_image = np.load(post_image_path)
        print(f"Pre-image shape: {pre_image.shape}")
        print(f"Post-image shape: {post_image.shape}")
        
        # Normalize if needed (assuming images are in [0, 1] range)
        if pre_image.max() > 1.0:
            pre_image = pre_image.astype(np.float32) / 255.0
            post_image = post_image.astype(np.float32) / 255.0
            
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return None
    
    # Process in patches to avoid memory issues
    flood_probabilities = process_image_patches(
        model, pre_image, post_image, 
        patch_size=patch_size, 
        overlap=64, 
        device=device
    )
    
    return flood_probabilities

def downsample_for_population_analysis(flood_map, target_size=(1000, 1000)):
    """Downsample flood map for population analysis if too large"""
    if flood_map.shape[0] > target_size[0] or flood_map.shape[1] > target_size[1]:
        print(f"🔽 Downsampling flood map from {flood_map.shape} to ~{target_size} for population analysis")
        from scipy.ndimage import zoom
        
        scale_h = target_size[0] / flood_map.shape[0]
        scale_w = target_size[1] / flood_map.shape[1]
        scale = min(scale_h, scale_w)  # Maintain aspect ratio
        
        downsampled = zoom(flood_map, scale, order=1)
        print(f"   Downsampled to: {downsampled.shape}")
        return downsampled, scale
    
    return flood_map, 1.0

def run_dl_comparison_analysis_efficient():
    """Run complete analysis using DL model vs baselines - Memory Efficient Version"""
    print("\n" + "="*60)
    print("🌪️ HURRICANE IMPACT ANALYSIS - DL vs BASELINES (Memory Efficient)")
    print("="*60)
    
    # Configuration - adjust these based on your system memory
    device = 'mps'  # Use CPU to avoid GPU memory issues
    patch_size = 256  # Smaller patches for large images
    model_path = 'best_model.pth'
    bbox = [-82.8, 25.8, -81.2, 27.5]  # Hurricane Ian area
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        print("💡 Make sure you have completed training and have best_model.pth")
        return None
    
    # Load metadata
    metadata_file = Path("hurricane_data/processing_metadata.json")
    if not metadata_file.exists():
        print("❌ Processing metadata not found.")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Get hurricane data
    hurricane_data = metadata['processed_data'][0]
    hurricane_name = hurricane_data['hurricane']
    periods = hurricane_data['periods']
    
    # Select pre and post periods
    pre_periods = [p for p in periods.keys() if 'pre' in p]
    post_periods = [p for p in periods.keys() if 'post' in p]
    
    if not pre_periods or not post_periods:
        print("❌ Need both pre and post event data")
        return None
    
    pre_period = pre_periods[-1]  # Latest pre-event
    post_period = post_periods[0]  # Earliest post-event
    
    pre_path = periods[pre_period]['file_path']
    post_path = periods[post_period]['file_path']
    
    print(f"📊 Analyzing Hurricane {hurricane_name}")
    print(f"   Pre-event period:  {pre_period}")
    print(f"   Post-event period: {post_period}")
    print(f"   Device: {device}")
    print(f"   Patch size: {patch_size}")
    
    # 1. Generate DL flood map with memory-efficient processing
    dl_flood_map = generate_dl_flood_map_efficient(
        model_path, pre_path, post_path, 
        patch_size=patch_size, device=device
    )
    
    if dl_flood_map is None:
        print("❌ Failed to generate DL flood map")
        return None
    
    print(f"✅ DL flood map generated: {dl_flood_map.shape}")
    
    # 2. Downsample for population analysis if needed
    dl_flood_map_ds, scale = downsample_for_population_analysis(dl_flood_map)
    
    # 3. Initialize analyzer and load population data
    print("\n👥 Loading population data...")
    analyzer = PopulationAnalyzer(bbox)
    population_data = analyzer.load_population_data()
    if population_data is None:
        print("❌ Failed to load population data")
        return None
    
    print(f"Population data shape: {population_data.shape}")
    print(f"Total population in study area: {np.sum(population_data):,.0f}")
    
    # 4. Load satellite images for baseline methods (downsample if needed)
    print("\n📡 Loading satellite images for baseline methods...")
    pre_image = np.load(pre_path)
    post_image = np.load(post_path)
    
    # Downsample baseline images if they're too large
    if pre_image.shape[1] > 2000 or pre_image.shape[2] > 2000:
        print("🔽 Downsampling baseline images for memory efficiency...")
        from scipy.ndimage import zoom
        ds_scale = min(1000 / pre_image.shape[1], 1000 / pre_image.shape[2])
        pre_image_ds = zoom(pre_image, (1, ds_scale, ds_scale), order=1)
        post_image_ds = zoom(post_image, (1, ds_scale, ds_scale), order=1)
        print(f"   Downsampled baseline images to: {pre_image_ds.shape}")
    else:
        pre_image_ds = pre_image
        post_image_ds = post_image
    
    # 5. Calculate impacts for all methods
    print("\n🔍 Calculating population impacts...")
    
    # DL Model Impact (using downsampled version)
    print("--- Deep Learning (Siamese U-Net) ---")
    dl_impact, dl_affected_map = analyzer.calculate_population_impact(
        dl_flood_map_ds, population_data.copy()
    )
    dl_affected = dl_impact['affected_population']
    dl_percentage = dl_impact['affected_percentage']
    
    # NDWI Baseline Impact
    print("--- NDWI Baseline ---")
    ndwi_flood_map = analyzer.baseline_ndwi_flood_detection(pre_image_ds, post_image_ds)
    ndwi_impact, ndwi_affected_map = analyzer.calculate_population_impact(
        ndwi_flood_map, population_data.copy()
    )
    ndwi_affected = ndwi_impact['affected_population']
    ndwi_percentage = ndwi_impact['affected_percentage']
    
    # MDWI Baseline Impact
    print("--- MDWI Baseline ---")
    mdwi_flood_map = analyzer.baseline_mdwi_flood_detection(pre_image_ds, post_image_ds)
    mdwi_impact, mdwi_affected_map = analyzer.calculate_population_impact(
        mdwi_flood_map, population_data.copy()
    )
    mdwi_affected = mdwi_impact['affected_population']
    mdwi_percentage = mdwi_impact['affected_percentage']
    
    # 6. Print comprehensive results
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE IMPACT COMPARISON")
    print("="*60)
    
    print(f"Hurricane {hurricane_name} Population Impact Results:")
    print(f"")
    print(f"🤖 Deep Learning (Siamese U-Net):")
    print(f"   • Affected Population: {dl_affected:,.0f} people")
    print(f"   • Percentage Affected: {dl_percentage:.2f}%")
    print(f"   • Flooded Area: {np.sum(dl_flood_map_ds > 0.5) * 0.0009 / (scale**2):.1f} km²")
    print(f"")
    print(f"📊 MDWI Baseline:")
    print(f"   • Affected Population: {mdwi_affected:,.0f} people")
    print(f"   • Percentage Affected: {mdwi_percentage:.2f}%")
    print(f"   • Flooded Area: {np.sum(mdwi_flood_map > 0.5) * 0.0009:.1f} km²")
    print(f"")
    print(f"📊 NDWI Baseline:")
    print(f"   • Affected Population: {ndwi_affected:,.0f} people")
    print(f"   • Percentage Affected: {ndwi_percentage:.2f}%")
    print(f"   • Flooded Area: {np.sum(ndwi_flood_map > 0.5) * 0.0009:.1f} km²")
    print(f"")
    print("📈 Method Comparison:")
    print(f"   • DL vs MDWI difference: {dl_affected - mdwi_affected:+,.0f} people")
    print(f"   • DL vs NDWI difference: {dl_affected - ndwi_affected:+,.0f} people")
    print("="*60)
    
    # Clean up memory
    del pre_image, post_image
    if 'pre_image_ds' in locals():
        del pre_image_ds, post_image_ds
    gc.collect()
    
    # 7. Create visualization (with downsampled data)
    print("\n🎨 Creating visualization...")
    create_efficient_visualization(
        dl_flood_map_ds, ndwi_flood_map, mdwi_flood_map, 
        population_data, dl_affected_map, 
        hurricane_name, dl_affected, ndwi_affected, mdwi_affected
    )
    
    return {
        'hurricane': hurricane_name,
        'dl_impact': dl_impact,
        'ndwi_impact': ndwi_impact,
        'mdwi_impact': mdwi_impact,
        'dl_flood_map': dl_flood_map_ds,  # Return downsampled version
        'population_data': population_data
    }

def create_efficient_visualization(dl_flood_map, ndwi_flood_map, mdwi_flood_map, 
                                 population_data, dl_affected_map, hurricane_name,
                                 dl_affected, ndwi_affected, mdwi_affected):
    """Create visualization with memory-efficient processing"""
    
    print("🎨 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Hurricane {hurricane_name}: Deep Learning vs Baseline Methods", 
                 fontsize=16, fontweight='bold')
    
    # Resize all maps to the same size for comparison
    target_shape = min(dl_flood_map.shape, ndwi_flood_map.shape, key=lambda x: x[0]*x[1])
    
    if population_data.shape != target_shape:
        from scipy.ndimage import zoom
        scale = (target_shape[0]/population_data.shape[0], 
                target_shape[1]/population_data.shape[1])
        population_data = zoom(population_data, scale, order=1)
    
    # Resize flood maps to match
    if dl_flood_map.shape != target_shape:
        from scipy.ndimage import zoom
        scale = (target_shape[0]/dl_flood_map.shape[0], 
                target_shape[1]/dl_flood_map.shape[1])
        dl_flood_map = zoom(dl_flood_map, scale, order=1)
    
    if ndwi_flood_map.shape != target_shape:
        from scipy.ndimage import zoom
        scale = (target_shape[0]/ndwi_flood_map.shape[0], 
                target_shape[1]/ndwi_flood_map.shape[1])
        ndwi_flood_map = zoom(ndwi_flood_map, scale, order=1)
        
    if mdwi_flood_map.shape != target_shape:
        from scipy.ndimage import zoom
        scale = (target_shape[0]/mdwi_flood_map.shape[0], 
                target_shape[1]/mdwi_flood_map.shape[1])
        mdwi_flood_map = zoom(mdwi_flood_map, scale, order=1)
    
    # 1. Population Density
    im1 = axes[0, 0].imshow(population_data, cmap='YlOrRd', interpolation='nearest')
    axes[0, 0].set_title('A) Population Density', fontweight='bold')
    axes[0, 0].set_xlabel('Longitude (pixels)')
    axes[0, 0].set_ylabel('Latitude (pixels)')
    plt.colorbar(im1, ax=axes[0, 0], label='People per pixel', shrink=0.8)
    
    # 2. Method Comparison Overlay (RGB)
    overlay = np.zeros((*target_shape, 3))
    overlay[:, :, 0] = ndwi_flood_map  # Red = NDWI
    overlay[:, :, 1] = mdwi_flood_map  # Green = MDWI  
    overlay[:, :, 2] = dl_flood_map    # Blue = Deep Learning
    
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title('B) Flood Detection Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Longitude (pixels)')
    axes[0, 1].set_ylabel('Latitude (pixels)')
    
    # Add legend for overlay
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='NDWI Baseline'),
        Line2D([0], [0], color='green', lw=3, label='MDWI Baseline'),
        Line2D([0], [0], color='blue', lw=3, label='Deep Learning'),
        Line2D([0], [0], color='white', lw=3, label='All Methods Agree')
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 3. DL Affected Population
    if dl_affected_map.shape != target_shape:
        from scipy.ndimage import zoom
        scale = (target_shape[0]/dl_affected_map.shape[0], 
                target_shape[1]/dl_affected_map.shape[1])
        dl_affected_map = zoom(dl_affected_map, scale, order=1)
    
    im3 = axes[1, 0].imshow(dl_affected_map, cmap='Reds', interpolation='nearest')
    axes[1, 0].set_title('C) Deep Learning - Affected Population', fontweight='bold')
    axes[1, 0].set_xlabel('Longitude (pixels)')
    axes[1, 0].set_ylabel('Latitude (pixels)')
    plt.colorbar(im3, ax=axes[1, 0], label='Affected People', shrink=0.8)
    
    # 4. Summary Statistics
    axes[1, 1].axis('off')
    axes[1, 1].set_title('D) Impact Comparison Summary', fontweight='bold')
    
    total_pop = np.sum(population_data)
    
    # Create summary text
    summary_text = f"""
HURRICANE {hurricane_name.upper()} IMPACT SUMMARY

Deep Learning (Siamese U-Net):
• Affected Population: {dl_affected:,.0f} people
• Percentage: {(dl_affected/total_pop)*100:.2f}%

MDWI Baseline:
• Affected Population: {mdwi_affected:,.0f} people  
• Percentage: {(mdwi_affected/total_pop)*100:.2f}%

NDWI Baseline:
• Affected Population: {ndwi_affected:,.0f} people
• Percentage: {(ndwi_affected/total_pop)*100:.2f}%

Method Differences:
• DL vs MDWI: {dl_affected - mdwi_affected:+,.0f} people
• DL vs NDWI: {dl_affected - ndwi_affected:+,.0f} people
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    # Remove axes for cleaner look
    for ax in axes.flat[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save visualization
    output_path = f"hurricane_{hurricane_name.lower()}_dl_comparison_efficient.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Lower DPI for memory
    print(f"💾 Visualization saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🚀 Starting Memory-Efficient Deep Learning Hurricane Impact Analysis...")
    
    try:
        results = run_dl_comparison_analysis_efficient()
        
        if results:
            print("\n✅ ANALYSIS COMPLETE!")
            print("The system successfully processed the large imagery using patch-based inference!")
            print("Results show population impact estimates using memory-efficient processing.")
        else:
            print("\n❌ Analysis failed. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Try reducing patch_size further if memory issues persist.")