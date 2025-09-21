# Hurricane Flood Impact Analysis

A comprehensive deep learning pipeline for detecting flood impacts from hurricanes using satellite imagery and population data.

## Overview

This project provides an end-to-end solution for analyzing flood impacts from hurricanes by combining:

- **Satellite Imagery Processing**: Downloads and processes HLS (Harmonized Landsat Sentinel-2) data from NASA Earthdata
- **Deep Learning Flood Detection**: Uses a Siamese U-Net model to detect flood changes from pre/post-hurricane imagery
- **Population Impact Analysis**: Integrates WorldPop demographic data to estimate affected populations
- **Interactive Web Application**: Streamlit-based interface for real-time analysis and visualization

## Key Features

- **Multi-source Data Integration**: Combines satellite data from NASA Earthdata with WorldPop population data
- **Advanced Deep Learning**: Implements a Siamese U-Net architecture for accurate flood detection
- **Comparative Analysis**: Compares deep learning results with traditional NDWI/MDWI methods
- **Scalable Processing**: Handles large satellite images through patch-based processing
- **Interactive Visualization**: Provides detailed maps and impact statistics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hurricane-flood-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up authentication:
   - For NASA Earthdata: Set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables
   - For Google Earth Engine: Follow authentication instructions in the app

## Usage

### Running the Web Application

```bash
streamlit run app_4.py
```

The application provides three analysis options:
1. **Pre-loaded Data**: Analyze Hurricane Ian with existing processed data
2. **Google Earth Engine**: Fetch new hurricane data and analyze
3. **Upload Data**: Use your own pre/post hurricane imagery

### Running the Data Pipeline

```bash
python data_pipeline_2.py
```

### Running Population Analysis

```bash
python population_analysis.py
```

## Project Structure

```
hurricane-flood-analysis/
├── app_4.py                 # Main Streamlit application
├── data_pipeline_2.py       # Enhanced data processing pipeline
├── population_analysis.py   # Population impact analysis
├── siamese_unet.py          # Deep learning model architecture
├── pop_data_donwload.py     # WorldPop data handler
├── hurricane_config.json    # Configuration for hurricane analysis
├── processing_metadata.json # Metadata from processed data
├── requirements.txt         # Python dependencies
└── hurricane_data/          # Directory for processed data
```

## Data Sources

- **Satellite Imagery**: NASA Harmonized Landsat Sentinel-2 (HLS) data
- **Population Data**: WorldPop 2020 demographic data
- **Hurricane Tracking**: Custom configuration for specific hurricane events

## Model Architecture

The project uses a **Siamese U-Net** model that:
- Takes 6-band satellite imagery as input (pre and post-hurricane)
- Outputs flood probability maps
- Is trained on real hurricane events with synthetic labels
- Outperforms traditional water index methods (NDWI/MDWI)

## Results

The analysis provides:
- Flood probability maps at 30m resolution
- Population impact estimates
- Comparative analysis between deep learning and traditional methods
- Detailed visualizations and statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for academic/research purposes. Please ensure proper attribution when using this code.
