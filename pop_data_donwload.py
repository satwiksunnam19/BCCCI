# Improved WorldPop Handler with Better Validation
# File: improved_worldpop_handler.py

import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import time
from tqdm import tqdm

class ImprovedWorldPopHandler:
    """Improved WorldPop handler with better validation for large datasets"""
    
    def __init__(self, output_dir="hurricane_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def download_with_progress(self, url, output_path, timeout=300):
        """Download file with progress tracking"""
        print(f"Downloading from: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading WorldPop",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            print(f"Download completed: {output_path}")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def smart_validate_geotiff(self, file_path):
        """Improved validation that samples multiple regions"""
        try:
            with rasterio.open(file_path) as src:
                print(f"Validating WorldPop file: {file_path}")
                print(f"  Dimensions: {src.width} x {src.height}")
                print(f"  Data type: {src.dtypes[0]}")
                print(f"  CRS: {src.crs}")
                print(f"  NoData value: {src.nodata}")
                
                # Sample multiple regions to get better coverage
                sample_size = 200
                samples = []
                
                # Sample from different parts of the image
                regions = [
                    (0, 0),  # Top-left
                    (src.width//2, src.height//2),  # Center
                    (src.width-sample_size, src.height-sample_size),  # Bottom-right
                    (src.width//4, src.height//4),  # Quarter points
                    (3*src.width//4, 3*src.height//4),
                ]
                
                for x_start, y_start in regions:
                    # Ensure we don't go out of bounds
                    x_start = max(0, min(x_start, src.width - sample_size))
                    y_start = max(0, min(y_start, src.height - sample_size))
                    
                    window = Window(x_start, y_start, sample_size, sample_size)
                    sample = src.read(1, window=window)
                    samples.append(sample)
                
                # Combine all samples
                combined_sample = np.concatenate([s.flatten() for s in samples])
                
                print(f"  Combined sample size: {combined_sample.shape}")
                print(f"  Sample min: {np.min(combined_sample)}")
                print(f"  Sample max: {np.max(combined_sample)}")
                print(f"  Sample mean: {np.mean(combined_sample)}")
                
                # More intelligent validation
                issues = []
                
                # Check for all negative values
                if np.all(combined_sample < 0):
                    issues.append("All sampled values are negative")
                
                # Check for reasonable population values (some positive, reasonable max)
                positive_values = combined_sample[combined_sample > 0]
                if len(positive_values) == 0:
                    # This might be OK for ocean areas, check total data
                    print("  No positive values in samples, checking full data statistics...")
                    
                    # Quick check: read a larger systematic sample
                    step = max(1, src.width // 100)  # Sample every N pixels
                    large_sample = src.read(1)[::step, ::step].flatten()
                    positive_in_large = np.sum(large_sample > 0)
                    
                    if positive_in_large == 0:
                        issues.append("No positive values found in systematic sampling")
                    else:
                        print(f"  Found {positive_in_large} positive values in systematic sample")
                else:
                    print(f"  Found {len(positive_values)} positive values in samples")
                    print(f"  Max positive value: {np.max(positive_values)}")
                
                # Check for extreme values
                if np.max(combined_sample) > 100000:
                    issues.append(f"Extremely high values detected (max: {np.max(combined_sample)})")
                
                if len(issues) == 0:
                    print("  Validation PASSED")
                    return True, []
                else:
                    print(f"  Issues found: {issues}")
                    return False, issues
                
        except Exception as e:
            return False, [f"Cannot read file: {e}"]
    
    def fix_population_data(self, input_file, output_file):
        """Fix WorldPop data issues"""
        print(f"Fixing population data...")
        
        try:
            with rasterio.open(input_file) as src:
                print(f"Processing {src.width} x {src.height} raster...")
                
                # Read data in chunks to handle large files
                profile = src.profile.copy()
                
                # Process data in blocks to manage memory
                block_size = 1000
                
                with rasterio.open(output_file, 'w', **profile) as dst:
                    total_population = 0
                    processed_blocks = 0
                    
                    for row_start in range(0, src.height, block_size):
                        row_end = min(row_start + block_size, src.height)
                        
                        for col_start in range(0, src.width, block_size):
                            col_end = min(col_start + block_size, src.width)
                            
                            # Read block
                            window = Window(col_start, row_start, 
                                          col_end - col_start, row_end - row_start)
                            block = src.read(1, window=window)
                            
                            # Fix the block
                            original_block = block.copy()
                            
                            # Handle NoData values
                            if src.nodata is not None:
                                block = np.where(block == src.nodata, 0, block)
                            
                            # Handle common NoData indicators
                            for nodata_val in [-99999, -9999, -999]:
                                block = np.where(block == nodata_val, 0, block)
                            
                            # Handle remaining negative values
                            block = np.where(block < 0, 0, block)
                            
                            # Handle extreme values (cap at 99.9th percentile)
                            if np.max(block) > 50000:
                                valid_values = block[block > 0]
                                if len(valid_values) > 0:
                                    cap_value = np.percentile(valid_values, 99.9)
                                    block = np.where(block > cap_value, cap_value, block)
                            
                            # Write fixed block
                            dst.write(block, 1, window=window)
                            
                            # Track progress
                            total_population += np.sum(block)
                            processed_blocks += 1
                            
                            if processed_blocks % 100 == 0:
                                print(f"  Processed {processed_blocks} blocks...")
                
                print(f"Data fixing completed")
                print(f"  Total population: {total_population:,.0f}")
                print(f"  Fixed data saved to: {output_file}")
                
                # Update profile for output
                profile.update({
                    'dtype': 'float32',
                    'nodata': 0
                })
                
                return True, total_population
                
        except Exception as e:
            print(f"Error fixing data: {e}")
            return False, 0
    
    def clip_to_bbox_smart(self, input_file, bbox, output_file):
        """Smart clipping that handles coordinate systems properly"""
        min_lon, min_lat, max_lon, max_lat = bbox
        
        print(f"Clipping to bbox: {bbox}")
        
        try:
            with rasterio.open(input_file) as src:
                # Use rasterio's built-in windowing
                from rasterio.windows import from_bounds
                
                # Get window for the bounding box
                window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
                
                # Read clipped data
                clipped_data = src.read(1, window=window)
                
                # Get transform for clipped area
                clipped_transform = src.window_transform(window)
                
                # Create output profile
                profile = src.profile.copy()
                profile.update({
                    'height': clipped_data.shape[0],
                    'width': clipped_data.shape[1],
                    'transform': clipped_transform
                })
                
                # Save clipped data
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(clipped_data, 1)
                
                population_in_area = np.sum(clipped_data)
                
                print(f"Clipped data saved: {output_file}")
                print(f"  Clipped dimensions: {clipped_data.shape}")
                print(f"  Population in study area: {population_in_area:,.0f}")
                print(f"  Non-zero pixels: {np.sum(clipped_data > 0):,}")
                
                return str(output_file), clipped_data
                
        except Exception as e:
            print(f"Error clipping data: {e}")
            return None, None
    
    def get_worldpop_data(self, bbox=None):
        """Main method to get WorldPop data"""
        
        print("=" * 60)
        print("IMPROVED WORLDPOP DATA HANDLER")
        print("=" * 60)
        
        source = {
            'name': 'WorldPop USA 2020',
            'url': 'https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/USA/usa_ppp_2020_1km_Aggregated_UNadj.tif',
            'filename': 'usa_worldpop_2020.tif'
        }
        
        raw_file = self.output_dir / source['filename']
        fixed_file = self.output_dir / f"fixed_{source['filename']}"
        
        # Download if needed
        if not raw_file.exists():
            print(f"Downloading {source['name']}...")
            if not self.download_with_progress(source['url'], raw_file):
                return None, None
        else:
            print(f"Using existing file: {raw_file}")
        
        # Validate and fix if needed
        is_valid, issues = self.smart_validate_geotiff(raw_file)
        
        working_file = raw_file
        
        if not is_valid:
            print(f"Issues found: {issues}")
            print("Applying fixes...")
            
            fix_success, total_pop = self.fix_population_data(raw_file, fixed_file)
            
            if fix_success and total_pop > 0:
                print(f"Data successfully fixed! Total population: {total_pop:,.0f}")
                working_file = fixed_file
                
                # Re-validate fixed file
                is_fixed_valid, _ = self.smart_validate_geotiff(fixed_file)
                if not is_fixed_valid:
                    print("Warning: Fixed file still has issues, but proceeding...")
            else:
                print("Fix failed, but attempting to proceed with original file...")
        else:
            print("Data validation passed!")
        
        # Clip to study area if bbox provided
        if bbox is not None:
            clipped_filename = f"worldpop_clipped_{abs(hash(str(bbox)))}.tif"
            clipped_file = self.output_dir / clipped_filename
            
            if clipped_file.exists():
                print(f"Using existing clipped file: {clipped_file}")
                with rasterio.open(clipped_file) as src:
                    clipped_data = src.read(1)
                return str(clipped_file), clipped_data
            else:
                return self.clip_to_bbox_smart(working_file, bbox, clipped_file)
        else:
            # Return full dataset (not recommended for analysis)
            print("Warning: Using full dataset without clipping")
            with rasterio.open(working_file) as src:
                # For large datasets, don't load everything into memory
                print("Large dataset detected - recommend using bbox clipping")
                return str(working_file), None

def test_improved_handler():
    """Test the improved WorldPop handler"""
    handler = ImprovedWorldPopHandler()
    
    # Test with Hurricane Ian bbox
    bbox = [-82.8, 25.8, -81.2, 27.5]
    
    print("Testing improved WorldPop handler...")
    population_file, population_data = handler.get_worldpop_data(bbox=bbox)
    
    if population_file and population_data is not None:
        print(f"\nSUCCESS!")
        print(f"Population file: {population_file}")
        print(f"Data shape: {population_data.shape}")
        print(f"Data range: {np.min(population_data):.1f} to {np.max(population_data):.1f}")
        print(f"Total population: {np.sum(population_data):,.0f}")
        print(f"Non-zero pixels: {np.sum(population_data > 0):,}")
        
        # Additional validation
        if np.sum(population_data) > 100000:  # Reasonable for this area
            print("Data appears valid for hurricane analysis!")
            return True
        else:
            print("Warning: Population total seems low for this area")
            return False
    else:
        print(f"\nFAILED to get usable population data")
        return False

if __name__ == "__main__":
    success = test_improved_handler()
    if success:
        print("\nReady to use with hurricane analysis!")
    else:
        print("\nTroubleshooting needed.")