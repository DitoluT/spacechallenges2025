import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from rasterio.warp import reproject, Resampling


#---------------------------------------------------------------------------------

# reading specific bands data from Sentinel-2 Level 1C format
def read_sent2_1c_bands(base_path: str, 
                        band_list:list=['B01', 'B02', 'B03', 'B4', 'B05', 'B06', 'B07', 
                                        'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']):
    """
    Read specific bands from a Sentinel-2 Level 1C dataset and resample to common resolution if needed
    
    Args:
        base_path (str): Base path name to the Sentinel-2 Level 1C product in the DATASETS folder
        band_list (list): List of bands to read (by default it reads all bands)
    
    Returns:
        dict: Dictionary with structure:
              {
                  'data': numpy array with shape (height, width, bands) containing stacked band data,
                  'profile': rasterio profile with geospatial metadata (unified for all bands),
                  'band_names': list of band names in the order they appear in the data array,
                  'band_order': dict mapping array index to band name,
                  'resampling_info': dict with resampling details (if resampling occurred)
              }
              All bands will be resampled to the highest resolution (lowest numerical value) and stacked together.
    """
    
    # Define native resolutions for Sentinel-2 Level 1C bands
    band_native_resolutions = {
        'B01': 60,  # Coastal aerosol
        'B02': 10,  # Blue
        'B03': 10,  # Green
        'B04': 10,  # Red
        'B05': 20,  # Vegetation Red Edge
        'B06': 20,  # Vegetation Red Edge
        'B07': 20,  # Vegetation Red Edge
        'B08': 10,  # NIR
        'B8A': 20,  # Vegetation Red Edge
        'B09': 60,  # Water vapour
        'B10': 60,  # Cirrus (available in L1C)
        'B11': 20,  # SWIR
        'B12': 20   # SWIR
    }

    base_path = os.path.join('DATASETS', base_path)
    # Find the granule folder (there should be only one)
    granule_path = os.path.join(base_path, 'GRANULE')
    granule_folders = [f for f in os.listdir(granule_path) if os.path.isdir(os.path.join(granule_path, f))]
    
    if not granule_folders:
        raise ValueError(f"No granule folder found in {granule_path}")               
    
    granule_folder = os.path.join(granule_path, granule_folders[0])  # Take the first (and usually only) granule
    img_data_path = os.path.join(granule_folder, 'IMG_DATA')
    
    if not os.path.exists(img_data_path):
        raise ValueError(f"IMG_DATA folder not found in {granule_folder}")
    
    bands_data = {}
    
    # Read each requested band
    for band in band_list:
        if band not in band_native_resolutions:
            print(f"Warning: Band {band} not recognized. Skipping...")
            continue
            
        # Search for band files in the IMG_DATA folder
        # Level 1C format: T{tile_id}_{timestamp}_{band}.jp2
        band_files = glob.glob(os.path.join(img_data_path, f'*_{band}.jp2'))
        
        if band_files:
            band_file = band_files[0]  # Take the first match
            print(f"Reading band {band} from: {band_file}")
            
            with rasterio.open(band_file) as src:
                bands_data[band] = {
                    'data': src.read(1).astype(np.float32),
                    'profile': src.profile,
                    'file_path': str(band_file),
                    'native_resolution': band_native_resolutions[band]
                }
        else:
            print(f"Warning: Band {band} not found in {img_data_path}")
    
    if not bands_data:
        return {'data': None, 'profile': None, 'band_names': [], 'band_order': {}}

    # RESAMPLING PART
    # Find the highest resolution (lowest numerical value) among the requested bands
    resolutions = [band_native_resolutions[band] for band in bands_data.keys()]
    target_resolution = min(resolutions)
    target_res_str = f"{target_resolution}m"
    
    print(f"Target resolution: {target_res_str}")
    
    # Find a reference band at the target resolution for spatial reference
    reference_band = None
    reference_profile = None
    for band_name, band_info in bands_data.items():
        if band_info['native_resolution'] == target_resolution:
            reference_band = band_name
            reference_profile = band_info['profile']
            break
    
    if reference_band is None:
        raise ValueError(f"No reference band found at target resolution {target_res_str}")
    
    print(f"Using band {reference_band} as reference for {target_res_str} resolution")
    
    # Check if resampling is needed
    needs_resampling = any(band_info['native_resolution'] != target_resolution 
                          for band_info in bands_data.values())
    
    if needs_resampling:
        print(f"Multiple resolutions detected. Resampling all bands to {target_res_str}")
        
        # Resample bands that don't match the target resolution
        resampled_bands_data = {}
        for band_name, band_info in bands_data.items():
            if band_info['profile']['width'] != reference_profile['width'] or \
               band_info['profile']['height'] != reference_profile['height']:
                
                print(f"Resampling band {band_name} from {band_info['native_resolution']}m to {target_res_str}")
                
                # Create output array with target dimensions
                resampled_data = np.zeros((reference_profile['height'], reference_profile['width']), dtype=np.float32)
                
                # Reproject the band data
                reproject(
                    source=band_info['data'],
                    destination=resampled_data,
                    src_transform=band_info['profile']['transform'],
                    src_crs=band_info['profile']['crs'],
                    dst_transform=reference_profile['transform'],
                    dst_crs=reference_profile['crs'],
                    resampling=Resampling.bilinear
                )
                
                # Update profile for resampled band
                resampled_profile = reference_profile.copy()
                resampled_profile.update({
                    'dtype': band_info['profile']['dtype'],
                    'nodata': band_info['profile'].get('nodata')
                })
                
                resampled_bands_data[band_name] = {
                    'data': resampled_data,
                    'profile': resampled_profile,
                    'file_path': band_info['file_path'],
                    'original_resolution': f"{band_info['native_resolution']}m",
                    'resampled_to': target_res_str
                }
            else:
                # Band already at target resolution
                resampled_bands_data[band_name] = band_info.copy()
                resampled_bands_data[band_name]['original_resolution'] = f"{band_info['native_resolution']}m"
                resampled_bands_data[band_name]['resampled_to'] = target_res_str
        
        bands_data = resampled_bands_data
        print(f"All bands resampled to {target_res_str} resolution")
    else:
        # All bands already at the same resolution, just add resolution info
        for band_name, band_info in bands_data.items():
            band_info['original_resolution'] = f"{band_info['native_resolution']}m"
            band_info['resampled_to'] = target_res_str

    # Stack all bands into a single numpy array and return with unified profile
    # Get the reference profile (all bands should have the same profile after resampling)
    reference_profile = next(iter(bands_data.values()))['profile']
    
    # Stack all band data into a 3D array (height, width, bands)
    band_arrays = []
    band_names = []
    for band_name in sorted(bands_data.keys()):  # Sort to ensure consistent order
        band_arrays.append(bands_data[band_name]['data'])
        band_names.append(band_name)
    
    stacked_data = np.stack(band_arrays, axis=2)
    
    # Create result structure
    result = {
        'data': stacked_data,
        'profile': reference_profile,
        'band_names': band_names,
        'band_order': {i: name for i, name in enumerate(band_names)}
    }
    
    # Add resampling information
    result['resampling_info'] = {
        band_name: {
            'original_resolution': band_info.get('original_resolution', 'unknown'),
            'resampled_to': band_info.get('resampled_to', 'unknown')
        }
        for band_name, band_info in bands_data.items()
    }
    
    return result


#--------------------------------------------------------------------------------

# saving data and profile info
def save_data_profile(bands_data, path:str, name:str):
    """
    Save the stacked bands data and profile
    
    Args:
        bands_data (dict): Result from read_sent2_1c_bands with 'data' and 'profile' keys
        path (str): Directory path to save files
        name (str): Base name for the files
    """
    os.makedirs(path, exist_ok=True)
    
    # save the stacked bands data
    np.save(os.path.join(path, f'{name}.npy'), bands_data['data'])
    
    # save the profile for the metadata (location etc.)
    with open(os.path.join(path, f'{name}_geospatial_profile.pkl'), 'wb') as f:
        pickle.dump(bands_data['profile'], f)
    
    # save band names and order information (NOT STRICTLY NEEDED BUT BETTER FOR LATER CLARITY)
    with open(os.path.join(path, f'{name}_band_info.pkl'), 'wb') as f:
        band_info = {
            'band_names': bands_data['band_names'],
            'band_order': bands_data['band_order']
        }
        if 'resampling_info' in bands_data:
            band_info['resampling_info'] = bands_data['resampling_info']
        pickle.dump(band_info, f)

    return os.path.join(path, f'{name}.npy')


#--------------------------------------------------------------------------------

# get ndvi = (nir - r)/(nir + r)
def get_ndvi_from_bands(bands_data):
    """
    Calculate NDVI from pre-extracted bands data
    
    Args:
        bands_data: Dictionary from read_sent2_1c_bands containing band data
    
    Returns:
        numpy.ndarray: NDVI data with shape (height, width, 1)
    """
    # Extract band data from stacked array
    band_order = bands_data['band_order']
    band_names = bands_data['band_names']
    
    # Find B08 (NIR) and B04 (Red) indices
    try:
        b08_idx = next(i for i, name in band_order.items() if name == 'B08')
        b04_idx = next(i for i, name in band_order.items() if name == 'B04')
    except StopIteration:
        raise ValueError("B08 (NIR) or B04 (Red) bands not found in the provided bands data")
    
    b08_data = bands_data['data'][:, :, b08_idx]
    b04_data = bands_data['data'][:, :, b04_idx]
    
    denom = b08_data + b04_data
    denom[denom==0] = 1e-6
    # avoid division by zero
    ndvi_img = (b08_data - b04_data)/denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    ndvi_img = ndvi_img[:, :, np.newaxis]

    return ndvi_img

# Backward compatibility function
def get_ndvi(img_path):
    """Legacy function for backward compatibility - reads bands and calculates NDVI"""
    nir_r = read_sent2_1c_bands(img_path, ['B08', 'B04'], ['10m', '10m'])
    return {'data': get_ndvi_from_bands(nir_r), 'profile': nir_r['profile']}


#--------------------------------------------------------------------------------

# Unified function to compute vegetation indices from single .npy patch file
def compute_vegetation_index_from_patch(patch_path: str, band_names: list, index_type: str = 'ndvi'):
    """
    Compute vegetation index (NDVI or NDMI) from a single .npy patch file containing stacked bands
    
    Args:
        patch_path (str): Path to the .npy patch file
        band_names (list): List of band names in the order they appear in the patch
        index_type (str): Type of index to compute ('ndvi' or 'ndmi')
    
    Returns:
        numpy.ndarray: Vegetation index data with shape (height, width, 1)
    """
    # Load the patch data
    patch_data = np.load(patch_path)
    
    # Define band requirements for each index
    index_configs = {
        'ndvi': {'bands': ['B08', 'B04'], 'names': ['NIR', 'Red']},
        'ndmi': {'bands': ['B08', 'B11'], 'names': ['NIR', 'SWIR']}
    }
    
    if index_type.lower() not in index_configs:
        raise ValueError(f"Unsupported index type: {index_type}. Supported types: {list(index_configs.keys())}")
    
    config = index_configs[index_type.lower()]
    required_bands = config['bands']
    band_descriptions = config['names']
    
    # Find required band indices
    try:
        band1_idx = band_names.index(required_bands[0])
        band2_idx = band_names.index(required_bands[1])
    except ValueError:
        raise ValueError(f"{required_bands[0]} ({band_descriptions[0]}) or {required_bands[1]} ({band_descriptions[1]}) bands not found in the provided band names")
    
    # Extract the specific bands
    band1_data = patch_data[:, :, band1_idx]
    band2_data = patch_data[:, :, band2_idx]
    
    # Compute the index: (band1 - band2) / (band1 + band2)
    denom = band1_data + band2_data
    denom[denom == 0] = 1e-6  # Avoid division by zero
    index_patch = (band1_data - band2_data) / denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    index_patch = index_patch[:, :, np.newaxis]
    
    return index_patch


def save_vegetation_index_patch(patch_path: str, band_names: list, output_dir: str, index_type: str = 'ndvi'):
    """
    Compute and save vegetation index for a single patch
    
    Args:
        patch_path (str): Path to the input .npy patch file
        band_names (list): List of band names in order
        output_dir (str): Directory to save the index patch
        index_type (str): Type of index to compute ('ndvi' or 'ndmi')
    
    Returns:
        str: Path to the saved index patch file
    """
    # Compute the vegetation index
    index_patch = compute_vegetation_index_from_patch(patch_path, band_names, index_type)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input filename
    input_filename = os.path.basename(patch_path)
    output_filename = input_filename.replace('_patch_', f'_{index_type.lower()}_')
    output_path = os.path.join(output_dir, output_filename)
    
    # Save index patch
    np.save(output_path, index_patch)
    
    return output_path

#--------------------------------------------------------------------------------

# get ndmi = (b08 - b11)/(b08 + b11)
def get_ndmi_from_bands(bands_data):
    """
    Calculate NDMI from pre-extracted bands data
    
    Args:
        bands_data: Dictionary from read_sentinel2_bands containing band data
    
    Returns:
        numpy.ndarray: NDMI data with shape (height, width, 1)
    """
    # Extract band data from stacked array
    band_order = bands_data['band_order']
    band_names = bands_data['band_names']
    
    # Find B08 (NIR) and B11 (SWIR) indices
    try:
        b08_idx = next(i for i, name in band_order.items() if name == 'B08')
        b11_idx = next(i for i, name in band_order.items() if name == 'B11')
    except StopIteration:
        raise ValueError("B08 (NIR) or B11 (SWIR) bands not found in the provided bands data")

    b08_data = bands_data['data'][:, :, b08_idx]
    b11_data = bands_data['data'][:, :, b11_idx]

    denom = b08_data + b11_data
    denom[denom==0] = 1e-6
    # avoid division by zero
    ndmi_img = (b08_data - b11_data)/denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    ndmi_img = ndmi_img[:, :, np.newaxis]

    return ndmi_img

# Backward compatibility function
def get_ndmi(img_path):
    """Legacy function for backward compatibility - reads bands and calculates NDMI"""
    nir_swir = read_sent2_1c_bands(img_path, ['B08', 'B11'])
    return {'data': get_ndmi_from_bands(nir_swir), 'profile': nir_swir['profile']}


#--------------------------------------------------------------------------------

# extract the single patches of dim 256x256 (or whatever needed) from the large image
def extract_patches_with_padding(image, name, patch_size, path):
    """
    Extract patches using padding strategy to ensure complete coverage.
    
    Args:
        image: Input image (H, W, C)
        name: The name of the location of the image (needed for data organizational purposes)
        patch_size: Size of each patch (height, width, channels)
        path: Location where to save patches
    """
    os.makedirs(path, exist_ok=True)

    h, w, c = image.shape
    ph, pw, pc = patch_size
    
    # Calculate padding needed
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    
    # Pad with reflection to maintain natural patterns
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    # Extract non-overlapping patches
    padded_h, padded_w, _ = padded_image.shape
    for i in range(0, padded_h, ph):
        for j in range(0, padded_w, pw):
            patch = padded_image[i:i+ph, j:j+pw, :]
            np.save(os.path.join(path, f'{name}_patch_{(i, j)}'), patch)

    print("All patched extracted correctly!")


#--------------------------------------------------------------------------------

# extract the dNBR map (of values between 0 and 1) and the dNBR binary map
def extract_data_labels_from_bands(pre_bands_data, post_bands_data, output_dir: str, thresh: float = 0.6):
    """
    Create data labels from pre-extracted bands data by considering the dNBR value
    
    Args:
        pre_bands_data: Dictionary from read_sentinel2_bands for pre-fire data
        post_bands_data: Dictionary from read_sentinel2_bands for post-fire data
        output_dir: Location where to save the results
        thresh: threshold of dNBR values to use to create binary map
    """
    
    nbr_imgs = {}
    
    # Process both datasets
    for i, bands_data in enumerate([pre_bands_data, post_bands_data]):
        band_order = bands_data['band_order']
        
        # Find B08 (NIR) and B12 (SWIR) indices
        try:
            b08_idx = next(i for i, name in band_order.items() if name == 'B08')
            b12_idx = next(i for i, name in band_order.items() if name == 'B12')
        except StopIteration:
            raise ValueError("B08 (NIR) or B12 (SWIR) bands not found in the provided bands data")
        
        b08_data = bands_data['data'][:, :, b08_idx]
        b12_data = bands_data['data'][:, :, b12_idx]

        denom = b08_data + b12_data
        denom[denom == 0] = 1e-6
        # avoid division by zero
        nbr_img = (b08_data - b12_data) / denom

        nbr_imgs[i] = nbr_img

    # get final dnbr image
    dnbr_img = nbr_imgs[0] - nbr_imgs[1]
    
    # normalize between 0 and 1 to get heatmap of probabilities (?)
    dnbr_img = (dnbr_img - np.min(dnbr_img)) / (np.max(dnbr_img) - np.min(dnbr_img))

    # Apply NBR masking: set dNBR to 0 where pre-fire 0.7 < NBR < 0.2 (vegetation which is not very humid, so more prone to fires)
    dnbr_img[nbr_imgs[0] < 0.2] = 0
    dnbr_img[nbr_imgs[0] > 0.7] = 0
    
    # Add channel dimension to make it 3D (height, width, 1)
    dnbr_img = dnbr_img[:, :, np.newaxis]

    # get also the binary map if needed, with a threshold
    dnbr_map = np.where(dnbr_img > thresh, 1, 0)

    # save this data
    os.makedirs(output_dir, exist_ok=True)

    # Save as numpy files
    np.save(os.path.join(output_dir, 'dnbr_normalized.npy'), dnbr_img)
    np.save(os.path.join(output_dir, 'dnbr_binary_map.npy'), dnbr_map)

    print("Data saved successfully!")
    print(f"Files saved in '{output_dir}' directory:")
    print("- dnbr_normalized.npy (NumPy array)")
    print("- dnbr_binary_map.npy (NumPy array)")

    # Visualize the heatmap and binary map side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display dNBR heatmap
    im1 = ax1.imshow(dnbr_img[:, :, 0], cmap='hot', vmin=0, vmax=1)
    ax1.set_title('dNBR Heatmap (Normalized)', fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Display binary map
    im2 = ax2.imshow(dnbr_map[:, :, 0], cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f'dNBR Binary Map (thresh={thresh})', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

    return dnbr_img

# Backward compatibility function
def extract_data_labels(img_list: list, output_dir: str, thresh: float = 0.6):
    """Legacy function for backward compatibility - reads images and calculates dNBR"""
    nbr_imgs = {}
    profiles = {}

    for i, img_path in enumerate(img_list):
        nir_swir = read_sent2_1c_bands(img_path, ['B08', 'B12'])

        b08_data = nir_swir['data'][:, :, 0]
        b08_profile = nir_swir['profile']

        b12_data = nir_swir['data'][:, :, 1]

        denom = b08_data + b12_data
        denom[denom == 0] = 1e-6
        # avoid division by zero
        nbr_img = (b08_data - b12_data) / denom

        nbr_imgs[i] = nbr_img
        profiles[i] = b08_profile

    # get final dnbr image
    dnbr_img = nbr_imgs[0] - nbr_imgs[1]
    
    # normalize between 0 and 1 to get heatmap of probabilities (?)
    dnbr_img = (dnbr_img - np.min(dnbr_img)) / (np.max(dnbr_img) - np.min(dnbr_img))

    # Apply NBR masking: set dNBR to 0 where pre-fire 0.7 < NBR < 0.2 (vegetation which is not very humid, so more prone to fires)
    dnbr_img[nbr_imgs[0] < 0.2] = 0
    dnbr_img[nbr_imgs[0] > 0.7] = 0

    # Add channel dimension to make it 3D (height, width, 1)
    dnbr_img = dnbr_img[:, :, np.newaxis]

    # get also the binary map if needed, with a threshold
    dnbr_map = np.where(dnbr_img > thresh, 1, 0)

    # save this data
    os.makedirs(output_dir, exist_ok=True)

    # Save as numpy files
    np.save(os.path.join(output_dir, 'dnbr_normalized.npy'), dnbr_img)
    np.save(os.path.join(output_dir, 'dnbr_binary_map.npy'), dnbr_map)

    print("Data saved successfully!")
    print(f"Files saved in '{output_dir}' directory:")
    print("- dnbr_normalized.npy (NumPy array)")
    print("- dnbr_binary_map.npy (NumPy array)")

    return dnbr_img


#--------------------------------------------------------------------------------

# Utility functions to process entire directories of patches
# Unified function to process directories of patches for vegetation indices
def process_patches_directory(patches_dir: str, band_names: list, output_dir: str, index_type: str = 'ndvi'):
    """
    Process all .npy patch files in a directory to compute vegetation indices
    
    Args:
        patches_dir (str): Directory containing .npy patch files
        band_names (list): List of band names in order they appear in patches
        output_dir (str): Directory to save index patches
        index_type (str): Type of index to compute ('ndvi' or 'ndmi')
    
    Returns:
        list: List of paths to saved index patch files
    """
    # Find all .npy files in the patches directory
    patch_files = glob.glob(os.path.join(patches_dir, '*.npy'))
    
    if not patch_files:
        print(f"No .npy files found in {patches_dir}")
        return []
    
    print(f"Processing {len(patch_files)} patches for {index_type.upper()} computation...")
    
    saved_paths = []
    for patch_file in patch_files:
        try:
            output_path = save_vegetation_index_patch(patch_file, band_names, output_dir, index_type)
            saved_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {patch_file}: {e}")
    
    print(f"Successfully processed {len(saved_paths)} patches for {index_type.upper()}")
    return saved_paths


