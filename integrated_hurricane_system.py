# Integrated Hurricane Impact Analysis System
# File: integrated_hurricane_system.py

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from BCCCI.siamese_unet_stable import ProductionSiameseUNet
from population_analysis import PopulationAnalyzer

class HurricaneImpactAnalyzer:
    """
    Integrated system that combines Siamese U-Net flood detection 
    with population impact analysis to answer quantitative questions
    """
    
    def __init__(self, model_path, bbox, device='cpu'):
        self.bbox = bbox
        self.device = device
        
        # Load the trained Siamese U-Net model
        self.model = ProductionSiameseUNet(in_channels=6, n_classes=2)
        self.load_model(model_path)
        
        # Initialize population analyzer
        self.population_analyzer = PopulationAnalyzer(bbox)
        
        # Cache for results
        self.results_cache = {}
        
    def load_model(self, model_path):
        """Load the trained Siamese U-Net model"""
        try:
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                print(f"✓ Model loaded from {model_path}")
            else:
                print(f"❌ Model file not found: {model_path}")
                print("💡 Using untrained model - results will be poor")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Using untrained model - results will be poor")
    
    def predict_flood_map(self, pre_image, post_image):
        """
        Generate flood prediction map using the trained Siamese U-Net
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensors
            if isinstance(pre_image, np.ndarray):
                pre_tensor = torch.FloatTensor(pre_image).unsqueeze(0).to(self.device)
                post_tensor = torch.FloatTensor(post_image).unsqueeze(0).to(self.device)
            else:
                pre_tensor = pre_image.to(self.device)
                post_tensor = post_image.to(self.device)
            
            # Get predictions
            logits, confidence = self.model(pre_tensor, post_tensor)
            
            # Convert logits to probabilities and extract flood class
            probabilities = torch.softmax(logits, dim=1)
            flood_probability = probabilities[0, 1].cpu().numpy()  # Class 1 = flood
            confidence_map = confidence[0, 0].cpu().numpy()
            
            # Apply confidence weighting
            final_flood_map = flood_probability * confidence_map
            
        return final_flood_map, flood_probability, confidence_map
    
    def load_hurricane_data(self, hurricane_name=None, pre_period=None, post_period=None):
        """Load pre and post hurricane satellite data"""
        
        metadata_file = Path("hurricane_data/processing_metadata.json")
        if not metadata_file.exists():
            raise FileNotFoundError("No processing metadata found. Run the data pipeline first.")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find hurricane data
        hurricane_data = None
        for data in metadata['processed_data']:
            if hurricane_name is None or data['hurricane'] == hurricane_name:
                hurricane_data = data
                break
        
        if hurricane_data is None:
            raise ValueError(f"Hurricane {hurricane_name} not found in processed data")
        
        # Get periods
        periods = hurricane_data['periods']
        
        # Auto-select periods if not specified
        if pre_period is None:
            pre_periods = [p for p in periods.keys() if 'pre' in p]
            pre_period = pre_periods[-1] if pre_periods else None  # Latest pre-event
        
        if post_period is None:
            post_periods = [p for p in periods.keys() if 'post' in p]
            post_period = post_periods[0] if post_periods else None  # Earliest post-event
        
        if pre_period not in periods or post_period not in periods:
            raise ValueError(f"Periods not found: {pre_period}, {post_period}")
        
        # Load data
        pre_path = periods[pre_period]['file_path']
        post_path = periods[post_period]['file_path']
        
        pre_image = np.load(pre_path)
        post_image = np.load(post_path)
        
        return pre_image, post_image, hurricane_data['hurricane']
    
    def analyze_hurricane_impact(self, hurricane_name=None, pre_period=None, post_period=None):
        """
        Complete hurricane impact analysis
        Returns quantitative answers to common questions
        """
        print(f"🌪️ Analyzing hurricane impact for {hurricane_name or 'available hurricane'}...")
        
        # Load satellite data
        pre_image, post_image, actual_hurricane_name = self.load_hurricane_data(
            hurricane_name, pre_period, post_period
        )
        
        print(f"📊 Analyzing Hurricane {actual_hurricane_name}")
        print(f"   Pre-event period: {pre_period}")
        print(f"   Post-event period: {post_period}")
        
        # Generate flood map using Siamese U-Net
        flood_map, flood_prob, confidence = self.predict_flood_map(pre_image, post_image)
        
        # Load population data
        population_data, pop_transform, pop_crs = self.population_analyzer.load_population_data()
        
        # Calculate population impact
        impact_stats, affected_population = self.population_analyzer.calculate_population_impact(
            flood_map, population_data, None, pop_transform
        )
        
        # Generate baseline comparison
        baseline_flood, _, _ = self.population_analyzer.baseline_ndwi_flood_detection(
            pre_image, post_image
        )
        baseline_impact, _ = self.population_analyzer.calculate_population_impact(
            baseline_flood, population_data, None, pop_transform
        )
        
        # Calculate additional metrics
        pixel_area_km2 = 0.0009  # Approximate area per pixel in km²
        flooded_pixels = np.sum(flood_map > 0.5)
        high_confidence_pixels = np.sum((flood_map > 0.5) & (confidence > 0.7))
        
        # Compile comprehensive results
        results = {
            'hurricane_name': actual_hurricane_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'periods_analyzed': {'pre': pre_period, 'post': post_period},
            
            # Flood extent
            'total_flooded_area_km2': flooded_pixels * pixel_area_km2,
            'high_confidence_flooded_area_km2': high_confidence_pixels * pixel_area_km2,
            'flooded_pixels': int(flooded_pixels),
            
            # Population impact
            'total_population_in_area': int(impact_stats['total_population']),
            'affected_population': int(impact_stats['affected_population']),
            'affected_population_percentage': impact_stats['affected_percentage'],
            'population_density_in_flooded_areas': impact_stats['population_density_affected'],
            
            # Model confidence
            'average_flood_confidence': float(np.mean(confidence[flood_map > 0.5])) if flooded_pixels > 0 else 0.0,
            'high_confidence_flood_percentage': (high_confidence_pixels / flooded_pixels * 100) if flooded_pixels > 0 else 0.0,
            
            # Comparison with baseline
            'baseline_comparison': {
                'siamese_unet_affected': int(impact_stats['affected_population']),
                'baseline_ndwi_affected': int(baseline_impact['affected_population']),
                'difference': int(impact_stats['affected_population'] - baseline_impact['affected_population']),
                'siamese_unet_area_km2': impact_stats['flooded_area_km2'],
                'baseline_area_km2': baseline_impact['flooded_area_km2']
            }
        }
        
        # Cache results
        self.results_cache[actual_hurricane_name] = results
        
        return results, flood_map, population_data, affected_population
    
    def answer_question(self, question, hurricane_name=None):
        """
        Natural language interface to answer questions about hurricane impact
        """
        question = question.lower()
        
        # Get analysis results
        if hurricane_name and hurricane_name in self.results_cache:
            results = self.results_cache[hurricane_name]
        else:
            # Run analysis if not cached
            results, _, _, _ = self.analyze_hurricane_impact(hurricane_name)
        
        # Answer different types of questions
        if any(word in question for word in ['how many people', 'population affected', 'people impacted']):
            affected = results['affected_population']
            percentage = results['affected_population_percentage']
            return f"Hurricane {results['hurricane_name']} affected approximately {affected:,} people ({percentage:.2f}% of the population in the study area)."
        
        elif any(word in question for word in ['how much land', 'area flooded', 'flooded area', 'land flooded']):
            area_km2 = results['total_flooded_area_km2']
            high_conf_area = results['high_confidence_flooded_area_km2']
            return f"Hurricane {results['hurricane_name']} flooded approximately {area_km2:.2f} km² of land (with {high_conf_area:.2f} km² detected with high confidence)."
        
        elif any(word in question for word in ['confidence', 'how reliable', 'accuracy']):
            avg_confidence = results['average_flood_confidence']
            high_conf_percent = results['high_confidence_flood_percentage']
            return f"The flood detection model shows {avg_confidence:.1%} average confidence, with {high_conf_percent:.1f}% of detected floods having high confidence (>70%)."
        
        elif any(word in question for word in ['compare', 'comparison', 'baseline', 'difference']):
            siamese_affected = results['baseline_comparison']['siamese_unet_affected']
            baseline_affected = results['baseline_comparison']['baseline_ndwi_affected']
            difference = results['baseline_comparison']['difference']
            return f"Siamese U-Net detected {siamese_affected:,} affected people vs {baseline_affected:,} for baseline NDWI method (difference: {difference:,} people)."
        
        elif any(word in question for word in ['when', 'time period', 'dates']):
            pre_period = results['periods_analyzed']['pre']
            post_period = results['periods_analyzed']['post']
            return f"Analysis compared {pre_period} (pre-event) with {post_period} (post-event) periods."
        
        elif any(word in question for word in ['density', 'population density']):
            density = results['population_density_in_flooded_areas']
            return f"The average population density in flooded areas is {density:.1f} people per km²."
        
        else:
            # General summary
            return self.generate_summary(results)
    
    def generate_summary(self, results):
        """Generate a comprehensive summary of hurricane impacts"""
        
        summary = f"""
Hurricane {results['hurricane_name']} Impact Analysis Summary:

🌪️ FLOOD EXTENT:
   • Total flooded area: {results['total_flooded_area_km2']:.2f} km²
   • High-confidence area: {results['high_confidence_flooded_area_km2']:.2f} km²
   • Flooded pixels detected: {results['flooded_pixels']:,}

👥 POPULATION IMPACT:
   • Total people in study area: {results['total_population_in_area']:,}
   • Affected population: {results['affected_population']:,}
   • Percentage affected: {results['affected_population_percentage']:.2f}%
   • Population density in flooded areas: {results['population_density_in_flooded_areas']:.1f} people/km²

🎯 MODEL PERFORMANCE:
   • Average confidence: {results['average_flood_confidence']:.1%}
   • High-confidence detections: {results['high_confidence_flood_percentage']:.1f}%

📊 METHOD COMPARISON:
   • Siamese U-Net: {results['baseline_comparison']['siamese_unet_affected']:,} people affected
   • Baseline NDWI: {results['baseline_comparison']['baseline_ndwi_affected']:,} people affected
   • Difference: {results['baseline_comparison']['difference']:,} people

Analysis completed on {results['analysis_timestamp'][:16]}
        """
        return summary.strip()
    
    def create_impact_dashboard(self, results, flood_map, population_data, affected_population):
        """Create a comprehensive dashboard visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Hurricane {results['hurricane_name']} Impact Dashboard", fontsize=16, fontweight='bold')
        
        # 1. Flood detection map
        im1 = axes[0, 0].imshow(flood_map, cmap='Blues', vmin=0, vmax=1)
        axes[0, 0].set_title('Flood Detection (Siamese U-Net)')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0, 0], label='Flood Probability')
        
        # 2. Population density
        im2 = axes[0, 1].imshow(population_data, cmap='YlOrRd')
        axes[0, 1].set_title('Population Density')
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[0, 1], label='People per pixel')
        
        # 3. Affected population overlay
        im3 = axes[0, 2].imshow(affected_population, cmap='Reds')
        axes[0, 2].set_title('Affected Population')
        axes[0, 2].set_xlabel('X (pixels)')
        axes[0, 2].set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=axes[0, 2], label='Affected people')
        
        # 4. Impact statistics bar chart
        categories = ['Total Pop.', 'Affected', 'Flooded Area\n(km²)']
        values = [
            results['total_population_in_area'] / 1000,  # In thousands
            results['affected_population'],
            results['total_flooded_area_km2'] * 100  # Scale for visibility
        ]
        colors = ['lightblue', 'red', 'blue']
        
        bars = axes[1, 0].bar(categories, values, color=colors, alpha=0.7)
        axes[1, 0].set_title('Key Impact Metrics')
        axes[1, 0].set_ylabel('Scale varies by metric')
        
        # Add value labels on bars
        for bar, value, category in zip(bars, values, categories):
            height = bar.get_height()
            if 'Pop' in category:
                label = f'{value:.0f}k'
            elif 'Affected' in category:
                label = f'{value:.0f}'
            else:
                label = f'{value/100:.1f}'
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom')
        
        # 5. Method comparison
        methods = ['Siamese U-Net', 'Baseline NDWI']
        affected_counts = [
            results['baseline_comparison']['siamese_unet_affected'],
            results['baseline_comparison']['baseline_ndwi_affected']
        ]
        
        bars2 = axes[1, 1].bar(methods, affected_counts, color=['green', 'orange'], alpha=0.7)
        axes[1, 1].set_title('Method Comparison')
        axes[1, 1].set_ylabel('Affected Population')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars2, affected_counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{count:,}', ha='center', va='bottom')
        
        # 6. Summary text box
        summary_text = f"""
HURRICANE {results['hurricane_name'].upper()} IMPACT

Population Impact:
• {results['affected_population']:,} people affected
• {results['affected_population_percentage']:.1f}% of study area

Flood Extent:
• {results['total_flooded_area_km2']:.1f} km² flooded
• {results['flooded_pixels']:,} pixels detected

Model Confidence:
• {results['average_flood_confidence']:.0%} avg confidence
• {results['high_confidence_flood_percentage']:.0f}% high confidence

Population Density:
• {results['population_density_in_flooded_areas']:.0f} people/km² in floods
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].set_title('Impact Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = f"hurricane_{results['hurricane_name']}_impact_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved as: {dashboard_path}")
        
        return fig

def main():
    """Example usage of the integrated system"""
    
    # Configuration
    bbox = [-82.8, 25.8, -81.2, 27.5]  # Southwest Florida (Hurricane Ian area)
    model_path = "best_hurricane_model_final.pth"  # Path to your trained model
    
    # Initialize the integrated system
    analyzer = HurricaneImpactAnalyzer(model_path, bbox, device='cpu')
    
    # Run complete analysis
    try:
        results, flood_map, population_data, affected_population = analyzer.analyze_hurricane_impact(
            hurricane_name="Ian"  # or None to use the first available hurricane
        )
        
        # Answer example questions
        questions = [
            "How many people were affected by the hurricane?",
            "How much land was flooded?",
            "How reliable are these results?",
            "How does this compare to baseline methods?",
            "What is the population density in affected areas?"
        ]
        
        print("\n" + "="*60)
        print("HURRICANE IMPACT Q&A SYSTEM")
        print("="*60)
        
        for question in questions:
            print(f"\n❓ Q: {question}")
            answer = analyzer.answer_question(question, "Ian")
            print(f"💬 A: {answer}")
        
        # Generate full summary
        print("\n" + "="*60)
        print("COMPLETE IMPACT SUMMARY")
        print("="*60)
        summary = analyzer.generate_summary(results)
        print(summary)
        
        # Create dashboard
        analyzer.create_impact_dashboard(results, flood_map, population_data, affected_population)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have:")
        print("   1. Trained model file: best_hurricane_model_final.pth")
        print("   2. Processed satellite data from the pipeline")
        print("   3. Required dependencies installed")

if __name__ == "__main__":
    main()