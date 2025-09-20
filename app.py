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

# Configure page
st.set_page_config(
    page_title="Hurricane Flood Impact Analysis",
    page_icon="🌊",
    layout="wide"
)

# Import your model (make sure siamese_unet.py is in the repo)
@st.cache_resource
def load_model():
    """Load the trained model once and cache it"""
    try:
        from siamese_unet import ProductionSiameseUNet
        
        device = 'cpu'  # Use CPU for deployment
        model = ProductionSiameseUNet(in_channels=6, dropout_rate=0.1)
        # Download from Hugging Face
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id="Satwik19/hurry", 
                                    filename="best_model.pth")
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

def main():
    st.title("🌊 Hurricane Flood Impact Analysis")
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
    
    st.success(f"✅ Model loaded successfully")
    st.info(f"📊 Population data loaded: {np.sum(population_data):,.0f} people in study area")
    
    # Demo section with pre-loaded examples
    st.header("🎯 Quick Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option 1: Use Example Data")
        if st.button("🚀 Run Hurricane Ian Analysis", type="primary"):
            # Load your pre-processed Hurricane Ian data
            try:
                # Try to load your actual Hurricane Ian data
                metadata_file = Path("hurricane_data/processing_metadata.json")
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Get the first hurricane data
                    hurricane_data = metadata['processed_data'][0]
                    periods = hurricane_data['periods']
                    
                    # Find pre and post files
                    pre_files = [p for p in periods.keys() if 'pre' in p]
                    post_files = [p for p in periods.keys() if 'post' in p]
                    
                    if pre_files and post_files:
                        pre_path = periods[pre_files[-1]]['file_path']
                        post_path = periods[post_files[0]]['file_path']
                        
                        # Load and process
                        with st.spinner("Processing Hurricane Ian satellite imagery..."):
                            pre_image = np.load(pre_path)
                            post_image = np.load(post_path)
                            
                            # Downsample for web demo
                            if pre_image.shape[1] > 500:
                                scale = 500 / max(pre_image.shape[1], pre_image.shape[2])
                                pre_image = zoom(pre_image, (1, scale, scale), order=1)
                                post_image = zoom(post_image, (1, scale, scale), order=1)
                            
                            # Process through model
                            flood_map = process_image_for_demo(model, pre_image, post_image, device)
                            
                            if flood_map is not None:
                                # Calculate impact
                                impact_stats = calculate_impact(flood_map, population_data, confidence_threshold)
                                
                                # Display results
                                st.success("✅ Analysis Complete!")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Affected Population", f"{impact_stats['affected_population']:,.0f}")
                                with col_b:
                                    st.metric("Percentage Affected", f"{impact_stats['percentage_affected']:.2f}%")
                                with col_c:
                                    st.metric("Flooded Area", f"{impact_stats['flooded_area_km2']:.1f} km²")
                                
                                # Create visualization
                                fig = create_visualization(flood_map, population_data, impact_stats)
                                st.pyplot(fig)
                                
                                # Detailed stats
                                with st.expander("📊 Detailed Statistics"):
                                    st.write(f"**Total Population in Study Area:** {impact_stats['total_population']:,.0f} people")
                                    st.write(f"**Population Density in Flooded Areas:** {impact_stats['population_density_flooded']:.1f} people/km²")
                                    st.write(f"**Data Source:** {pop_file}")
                                    st.write(f"**Model:** Siamese U-Net with WorldPop integration")
                                    st.write(f"**Hurricane:** {hurricane_data['hurricane']}")
                        
                else:
                    st.warning("Hurricane Ian data not found. Upload your own data below.")
                    
            except Exception as e:
                st.error(f"Error running example: {e}")
    
    with col2:
        st.subheader("Option 2: Upload Your Own Data")
        uploaded_pre = st.file_uploader("Upload Pre-Event Satellite Image (.npy)", type=['npy'])
        uploaded_post = st.file_uploader("Upload Post-Event Satellite Image (.npy)", type=['npy'])
        
        if uploaded_pre and uploaded_post:
            if st.button("🔍 Analyze Uploaded Images"):
                with st.spinner("Processing your images..."):
                    try:
                        # Load uploaded files
                        pre_bytes = uploaded_pre.getvalue()
                        post_bytes = uploaded_post.getvalue()
                        
                        pre_image = np.load(io.BytesIO(pre_bytes))
                        post_image = np.load(io.BytesIO(post_bytes))
                        
                        # Process
                        flood_map = process_image_for_demo(model, pre_image, post_image, device)
                        
                        if flood_map is not None:
                            impact_stats = calculate_impact(flood_map, population_data, confidence_threshold)
                            
                            st.success("✅ Custom Analysis Complete!")
                            
                            # Display results similar to above
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
    with st.expander("🔬 Model Information"):
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
    st.markdown("**Hurricane Flood Impact Analysis** | Powered by PyTorch & Streamlit | Real WorldPop Data Integration")

if __name__ == "__main__":
    main()