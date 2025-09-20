# Enhanced Hurricane Data Pipeline - Scalable Version
# File: enhanced_data_pipeline.py

import earthaccess
import numpy as np
import rasterio
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ScalableHurricaneDataPipeline:
    """Enhanced pipeline for multiple hurricane events and temporal sampling"""
    
    def __init__(self, output_dir="hurricane_data", config_file=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Metadata tracking
        self.metadata = {
            'hurricanes': [],
            'processing_log': [],
            'band_statistics': {},
            'created': datetime.now().isoformat()
        }
        
    def _load_config(self, config_file):
        default_config = {
            'hurricanes': [
                {
                    'name': 'Ian',
                    'date': '2022-09-28',
                    'bbox': [-82.8, 25.8, -81.2, 27.5],
                    'pre_days': 30,
                    'post_days': 30
                }
            ],
            'data_settings': {
                'max_files_per_period': None,
                'temporal_sampling': 'weekly',
                'composite_method': 'median',
                'target_bands': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'],
                'resolution': 30,
                'patch_size': None
            },
            'processing': {
                'normalization_method': 'percentile',
                'cloud_threshold': 0.3,
                'water_threshold': 0.3,
                'quality_checks': True
            }
        }

        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                custom_config = json.load(f)

        for key, val in custom_config.items():
            # If default is dict and custom is dict → update
            if key in default_config and isinstance(default_config[key], dict) and isinstance(val, dict):
                default_config[key].update(val)
            # If default is list and custom is list → replace or extend
            elif key in default_config and isinstance(default_config[key], list) and isinstance(val, list):
                default_config[key] = val  # or default_config[key] + val if you want to append
            else:
                default_config[key] = val

        return default_config
    
    def authenticate_earthaccess(self):
        """Authenticate with NASA Earthdata"""
        try:
            earthaccess.login()
            print("✓ Successfully authenticated with NASA Earthdata")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    def get_temporal_periods(self, hurricane_date, pre_days, post_days, sampling='weekly'):
        """Generate temporal periods based on sampling strategy"""
        
        hurricane_dt = datetime.strptime(hurricane_date, '%Y-%m-%d') if isinstance(hurricane_date, str) else hurricane_date
        
        if sampling == 'single':
            # Original approach - single pre/post period
            pre_start = hurricane_dt - timedelta(days=pre_days)
            pre_end = hurricane_dt - timedelta(days=1)
            post_start = hurricane_dt + timedelta(days=1)
            post_end = hurricane_dt + timedelta(days=post_days)
            
            return [
                {'type': 'pre', 'start': pre_start, 'end': pre_end, 'name': 'pre_event'},
                {'type': 'post', 'start': post_start, 'end': post_end, 'name': 'post_event'}
            ]
        
        elif sampling == 'weekly':
            # Weekly sampling before and after hurricane
            periods = []
            
            # Pre-event weeks
            for week in range(4, 0, -1):  # 4 weeks before to 1 week before
                start = hurricane_dt - timedelta(days=week*7 + 3)
                end = hurricane_dt - timedelta(days=week*7 - 4)
                periods.append({
                    'type': 'pre',
                    'start': start,
                    'end': end,
                    'name': f'pre_week_{week}'
                })
            
            # Post-event weeks
            for week in range(1, 5):  # 1 week after to 4 weeks after
                start = hurricane_dt + timedelta(days=week*7 - 4)
                end = hurricane_dt + timedelta(days=week*7 + 3)
                periods.append({
                    'type': 'post',
                    'start': start,
                    'end': end,
                    'name': f'post_week_{week}'
                })
            
            return periods
        
        elif sampling == 'all':
            # Monthly sampling over larger period
            periods = []
            
            # 3 months before to 3 months after
            for month in range(-3, 4):
                if month == 0:  # Skip hurricane month
                    continue
                    
                start = hurricane_dt + timedelta(days=month*30 - 15)
                end = hurricane_dt + timedelta(days=month*30 + 15)
                
                period_type = 'pre' if month < 0 else 'post'
                name = f'{period_type}_month_{abs(month)}'
                
                periods.append({
                    'type': period_type,
                    'start': start,
                    'end': end,
                    'name': name
                })
            
            return periods
    
    def search_hurricane_data(self, hurricane_config):
        """Search for data for a specific hurricane with temporal periods"""
        
        hurricane_name = hurricane_config['name']
        hurricane_date = hurricane_config['date']
        bbox = hurricane_config['bbox']
        
        print(f"🔍 Searching data for Hurricane {hurricane_name}...")
        
        # Get temporal periods
        periods = self.get_temporal_periods(
            hurricane_date,
            hurricane_config['pre_days'],
            hurricane_config['post_days'],
            self.config['data_settings']['temporal_sampling']
        )
        
        all_results = {}
        
        for period in periods:
            period_name = period['name']
            start_str = period['start'].strftime('%Y-%m-%d')
            end_str = period['end'].strftime('%Y-%m-%d')
            
            print(f"   📅 {period_name}: {start_str} to {end_str}")
            
            try:
                results = earthaccess.search_data(
                    short_name="HLSS30",
                    bounding_box=(*bbox,),
                    temporal=(start_str, end_str),
                    count=self.config['data_settings']['max_files_per_period']
                )
                
                if results:
                    all_results[period_name] = {
                        'results': results,
                        'period_info': period,
                        'count': len(results)
                    }
                    print(f"      ✓ Found {len(results)} files")
                else:
                    print(f"      ⚠ No files found")
                    
            except Exception as e:
                print(f"      ❌ Search failed: {e}")
                continue
        
        return all_results
    
    def download_period_data(self, period_data, hurricane_name):
        """Download data for a specific time period"""
        
        period_name = period_data['period_info']['name']
        results = period_data['results']
        
        # Create hurricane-specific directory structure
        hurricane_dir = self.output_dir / hurricane_name
        period_dir = hurricane_dir / period_name
        period_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading {len(results)} files to {period_dir}...")
        
        try:
            downloaded_files = earthaccess.download(
                results,
                local_path=str(period_dir)
            )
            
            return downloaded_files, period_dir
            
        except Exception as e:
            print(f"❌ Download failed for {period_name}: {e}")
            return [], period_dir
    
    def create_enhanced_composite(self, image_folder, period_info):
        """Create composite with enhanced metadata tracking"""
        
        image_folder = Path(image_folder)
        image_files = list(image_folder.glob("*.tif")) + list(image_folder.glob("*.TIF"))
        
        if not image_files:
            print(f"❌ No files found in {image_folder}")
            return None, {}
        
        print(f"🔄 Creating composite from {len(image_files)} files...")
        
        # Group by bands
        band_files = {}
        target_bands = self.config['data_settings']['target_bands']
        
        for img_file in image_files:
            filename = img_file.name
            for band in target_bands:
                if band in filename:
                    if band not in band_files:
                        band_files[band] = []
                    band_files[band].append(img_file)
                    break
        
        if not band_files:
            print("⚠ Could not identify bands, using all files...")
            # Fallback to process all files
            composite = self._process_unstructured_files(image_files)
            metadata = {'method': 'unstructured', 'files': len(image_files)}
            return composite, metadata
        
        # Process each band
        composite_bands = []
        band_metadata = {}
        
        for band in target_bands:
            if band in band_files:
                band_images = []
                valid_files = []
                
                for band_file in band_files[band]:
                    processed = self._process_single_file(band_file)
                    if processed is not None:
                        band_images.append(processed)
                        valid_files.append(band_file.name)
                
                if band_images:
                    band_stack = np.stack(band_images, axis=0)
                    
                    # Create composite using specified method
                    method = self.config['data_settings']['composite_method']
                    if method == 'median':
                        band_composite = np.nanmedian(band_stack, axis=0)
                    elif method == 'mean':
                        band_composite = np.nanmean(band_stack, axis=0)
                    else:  # first
                        band_composite = band_stack[0]
                    
                    composite_bands.append(band_composite)
                    band_metadata[band] = {
                        'files_used': valid_files,
                        'method': method,
                        'stats': {
                            'min': float(np.nanmin(band_composite)),
                            'max': float(np.nanmax(band_composite)),
                            'mean': float(np.nanmean(band_composite))
                        }
                    }
                else:
                    # Create dummy band if no valid data
                    print(f"⚠ No valid data for band {band}, creating dummy")
                    dummy_band = np.full((512, 512), 0.1, dtype=np.float32)
                    composite_bands.append(dummy_band)
                    band_metadata[band] = {'status': 'dummy'}
            else:
                print(f"⚠ Band {band} not found")
                dummy_band = np.full((512, 512), 0.1, dtype=np.float32)
                composite_bands.append(dummy_band)
                band_metadata[band] = {'status': 'missing'}
        
        if composite_bands:
            composite = np.stack(composite_bands, axis=0)
            
            # Overall metadata
            metadata = {
                'shape': composite.shape,
                'bands': target_bands,
                'band_details': band_metadata,
                'composite_method': self.config['data_settings']['composite_method'],
                'period_info': period_info
            }
            
            return composite, metadata
        
        return None, {}
    
    def _process_single_file(self, file_path):
        """Process a single satellite file"""
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                
                # Handle fill values and scaling
                data = np.where(data == -9999, np.nan, data)
                data = np.where(data < 0, np.nan, data)
                data = data.astype(np.float32) * 0.0001  # HLS2 scale factor
                
                return data
        except Exception as e:
            print(f"⚠ Error processing {file_path}: {e}")
            return None
    
    def _process_unstructured_files(self, image_files):
        """Process files when band structure is unclear"""
        processed_images = []
        
        for img_file in image_files[:6]:  # Limit to 6 files
            processed = self._process_single_file(img_file)
            if processed is not None:
                processed_images.append(processed)
        
        if processed_images:
            # Ensure we have 6 bands
            while len(processed_images) < 6:
                processed_images.append(processed_images[-1])  # Duplicate last band
            
            return np.stack(processed_images[:6], axis=0)
        
        return None
    
    def normalize_composite(self, composite):
        """Normalize composite using configured method"""
        method = self.config['processing']['normalization_method']
        
        if method == 'percentile':
            p2, p98 = np.nanpercentile(composite, (2, 98), axis=(1, 2), keepdims=True)
            normalized = (composite - p2) / (p98 - p2 + 1e-8)
            normalized = np.clip(normalized, 0, 1)
        elif method == 'minmax':
            min_val = np.nanmin(composite, axis=(1, 2), keepdims=True)
            max_val = np.nanmax(composite, axis=(1, 2), keepdims=True)
            normalized = (composite - min_val) / (max_val - min_val + 1e-8)
        else:  # z-score
            mean_val = np.nanmean(composite, axis=(1, 2), keepdims=True)
            std_val = np.nanstd(composite, axis=(1, 2), keepdims=True)
            normalized = (composite - mean_val) / (std_val + 1e-8)
        
        return normalized
    
    def quality_check_composite(self, composite, metadata):
        """Perform quality checks on composite"""
        issues = []
        
        # Check for excessive NaN values
        nan_percentage = np.sum(np.isnan(composite)) / composite.size * 100
        if nan_percentage > 20:
            issues.append(f"High NaN percentage: {nan_percentage:.1f}%")
        
        # Check value ranges
        for i, band in enumerate(self.config['data_settings']['target_bands']):
            band_data = composite[i]
            min_val, max_val = np.nanmin(band_data), np.nanmax(band_data)
            
            if min_val == max_val:
                issues.append(f"Band {band} has constant values")
            elif max_val > 2.0 or min_val < -0.5:
                issues.append(f"Band {band} has unusual range: {min_val:.3f} to {max_val:.3f}")
        
        # Check spatial patterns
        if composite.shape[1] < 100 or composite.shape[2] < 100:
            issues.append("Composite is very small")
        
        return issues
    
    def run_scalable_pipeline(self):
        """Run the complete scalable pipeline"""
        print("🚀 Starting Scalable Hurricane Data Pipeline...")
        
        # Authenticate
        if not self.authenticate_earthaccess():
            print("❌ Authentication failed, cannot proceed with real data")
            return None
        
        all_processed_data = []
        processing_summary = {'successful': 0, 'failed': 0, 'total_composites': 0}
        
        # Process each hurricane
        for hurricane_config in self.config['hurricanes']:
            hurricane_name = hurricane_config['name']
            print(f"\n🌪 Processing Hurricane {hurricane_name}...")
            
            # Search for data
            search_results = self.search_hurricane_data(hurricane_config)
            
            if not search_results:
                print(f"❌ No data found for Hurricane {hurricane_name}")
                processing_summary['failed'] += 1
                continue
            
            hurricane_data = {'hurricane': hurricane_name, 'periods': {}}
            
            # Process each time period
            for period_name, period_data in search_results.items():
                print(f"\n📅 Processing period: {period_name}")
                
                # Download data
                downloaded_files, period_dir = self.download_period_data(period_data, hurricane_name)
                
                if not downloaded_files:
                    print(f"⚠ No files downloaded for {period_name}")
                    continue
                
                # Create composite
                composite, metadata = self.create_enhanced_composite(period_dir, period_data['period_info'])
                
                if composite is None:
                    print(f"❌ Failed to create composite for {period_name}")
                    continue
                
                # Quality check
                if self.config['processing']['quality_checks']:
                    issues = self.quality_check_composite(composite, metadata)
                    if issues:
                        print(f"⚠ Quality issues found: {'; '.join(issues)}")
                        metadata['quality_issues'] = issues
                
                # Normalize
                normalized = self.normalize_composite(composite)
                
                # Save composite
                output_file = self.output_dir / f"{hurricane_name}_{period_name}_processed.npy"
                np.save(output_file, normalized)
                
                # Store data
                hurricane_data['periods'][period_name] = {
                    'file_path': str(output_file),
                    'shape': normalized.shape,
                    'metadata': metadata
                }
                
                print(f"✅ Saved composite: {output_file}")
                processing_summary['total_composites'] += 1
            
            if hurricane_data['periods']:
                all_processed_data.append(hurricane_data)
                processing_summary['successful'] += 1
            else:
                processing_summary['failed'] += 1
        
        # Save metadata and processing log
        self.metadata['processed_data'] = all_processed_data
        self.metadata['processing_summary'] = processing_summary
        
        metadata_file = self.output_dir / "processing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"\n✅ Pipeline completed!")
        print(f"📊 Summary: {processing_summary['successful']} hurricanes, {processing_summary['total_composites']} composites")
        print(f"💾 Metadata saved to: {metadata_file}")
        
        return all_processed_data
    
    def get_training_data_list(self):
        """Get list of all processed composites for training"""
        
        metadata_file = self.output_dir / "processing_metadata.json"
        if not metadata_file.exists():
            print("❌ No processing metadata found. Run the pipeline first.")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        training_files = []
        
        for hurricane_data in metadata['processed_data']:
            hurricane_name = hurricane_data['hurricane']
            
            for period_name, period_info in hurricane_data['periods'].items():
                file_path = period_info['file_path']
                
                if Path(file_path).exists():
                    training_files.append({
                        'hurricane': hurricane_name,
                        'period': period_name,
                        'file_path': file_path,
                        'shape': period_info['shape'],
                        'type': 'pre' if 'pre' in period_name else 'post'
                    })
        
        return training_files

# Usage example
if __name__ == "__main__":
    # Create configuration for multiple hurricanes
    config = {
        'hurricanes': [
            {
                'name': 'Ian',
                'date': '2022-09-28',
                'bbox': [-82.8, 25.8, -81.2, 27.5],
                'pre_days': 30,
                'post_days': 30
            },
            Add more hurricanes here:
            {
                'name': 'Fiona',
                'date': '2022-09-18',
                'bbox': [-67.5, 17.8, -65.2, 19.0],  # Puerto Rico
                'pre_days': 21,
                'post_days': 21
            }
        ],
        'data_settings': {
            'temporal_sampling': 'weekly',  # Change to 'single' for original behavior
            'max_files_per_period': 5,
            'composite_method': 'median'
        }
    }
    
    # Save config
    config_file = "hurricane_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run pipeline
    pipeline = ScalableHurricaneDataPipeline(config_file=config_file)
    results = pipeline.run_scalable_pipeline()
    
    # Get training file list
    training_files = pipeline.get_training_data_list()
    print(f"\n📋 Available for training: {len(training_files)} composites")