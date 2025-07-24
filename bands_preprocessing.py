import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from rasterio.warp import reproject, Resampling
import pickle

# reading all bands from data image
def read_all_bands(base_path: str):
    """
    Read all available Sentinel-2 bands and resample them to 10m resolution
    
    Args:
        base_path (str): Base path to the Sentinel-2 product
    
    Returns:
        dict: Dictionary with structure:
              {
                  'data': numpy array with shape (height, width, 12) containing all band data at 10m resolution,
                  'profile': rasterio profile with geospatial metadata (unified for all bands at 10m),
                  'band_names': list of all 12 band names in order,
                  'band_order': dict mapping array index to band name,
                  'resampling_info': dict with resampling details for each band
              }
    """
    
    # Define all available Sentinel-2 LEVEL 2A bands with their native resolutions
    # Note: can be changed as needed (e.g., if we use L1C)
    all_bands = [
        'B01',  # Coastal aerosol - 60m
        'B02',  # Blue - 10m
        'B03',  # Green - 10m
        'B04',  # Red - 10m
        'B05',  # Vegetation Red Edge - 20m
        'B06',  # Vegetation Red Edge - 20m
        'B07',  # Vegetation Red Edge - 20m
        'B08',  # NIR - 10m
        'B8A',  # Vegetation Red Edge - 20m
        'B09',  # Water vapour - 60m
        'B11',  # SWIR - 20m
        'B12'   # SWIR - 20m
    ]
    
    # Define native resolutions for each band
    native_resolutions = [
        '60m',  # B01
        '10m',  # B02
        '10m',  # B03
        '10m',  # B04
        '20m',  # B05
        '20m',  # B06
        '20m',  # B07
        '10m',  # B08
        '20m',  # B8A
        '60m',  # B09
        '20m',  # B11
        '20m'   # B12
    ]
    
    print(f"Reading all available Sentinel-2 bands from: {base_path}")
    print("Note: B10 (Cirrus) is not available in L2A products")
    print("All bands will be resampled to 10m resolution")
    
    # Use the existing read_sentinel2_bands function with all bands and their native resolutions
    result = read_sentinel2_bands(base_path, all_bands, native_resolutions)
    
    return result


#---------------------------------------------------------------------------------

# reading specific bands data
def read_sentinel2_bands(base_path:str, band_list:list, resolution_list:list):
    """
    Read specific bands from a Sentinel-2 dataset and resample to common resolution if needed
    
    Args:
        base_path (str): Base path to the Sentinel-2 product
        band_list (list): List of bands to read (e.g., ['B08', 'B12'])
        resolution_list (list): Resolution list to read from, for each band(10m, 20m, 60m)
    
    Returns:
        dict: Dictionary with structure:
              {
                  'data': numpy array with shape (height, width, bands) containing stacked band data,
                  'profile': rasterio profile with geospatial metadata (unified for all bands),
                  'band_names': list of band names in the order they appear in the data array,
                  'band_order': dict mapping array index to band name,
                  'resampling_info': dict with resampling details (if resampling occurred)
              }
              If multiple resolutions are detected, all bands will be resampled to the highest 
              resolution (lowest numerical value) and stacked together.
    """

    if not len(band_list) == len(resolution_list):
        raise ValueError(f"Bands and resolutions lists do not match.")               

    # Find the granule folder (there should be only one)
    granule_path = os.path.join(base_path, 'GRANULE')
    granule_folders = [f for f in os.listdir(granule_path) if os.path.isdir(os.path.join(granule_path, f))]
    
    if not granule_folders:
        raise ValueError(f"No granule folder found in {granule_path}")               
    
    granule_folder = os.path.join(granule_path, granule_folders[0])  # Take the first (and usually only) granule
    bands_data = {}
    
    for i, band in enumerate(band_list):
        img_data_path = os.path.join(granule_folder, 'IMG_DATA', f'R{resolution_list[i]}')
        # Search for band files in the folder
        band_files = glob.glob(os.path.join(img_data_path, f'*_{band}_{resolution_list[i]}.jp2'))
        
        if band_files:
            band_file = band_files[0]  # Take the first match
            print(f"Reading band {band} from: {band_file}")
            
            with rasterio.open(band_file) as src:
                bands_data[band] = {
                    'data': src.read(1).astype(np.float32),
                    'profile': src.profile,
                    'file_path': str(band_file)
                }
        else:
            print(f"Warning: Band {band} not found in {img_data_path}")
    

    # RESAMPLING PART
    # Check if all bands have the same resolution and resample if needed
    if len(set(resolution_list)) > 1:
        # Find the highest resolution (lowest numerical value)
        target_resolution = min([int(res.rstrip('m')) for res in resolution_list])
        target_res_str = f"{target_resolution}m"
        print(f"Multiple resolutions detected. Resampling all bands to {target_res_str}")
        
        # Find a reference band at the target resolution for spatial reference
        reference_band = None
        reference_profile = None
        for i, res in enumerate(resolution_list):
            if int(res.rstrip('m')) == target_resolution:
                reference_band = band_list[i]
                reference_profile = bands_data[reference_band]['profile']
                break
        
        # If no band exists at target resolution, create reference from the lowest resolution band
        if reference_band is None:
            # Find the band with the highest current resolution
            min_res_idx = resolution_list.index(min(resolution_list, key=lambda x: int(x.rstrip('m'))))
            reference_band = band_list[min_res_idx]
            reference_profile = bands_data[reference_band]['profile']
        
        # Resample all bands to target resolution
        resampled_bands_data = {}
        for band_name, band_info in bands_data.items():
            if band_info['profile']['width'] != reference_profile['width'] or \
               band_info['profile']['height'] != reference_profile['height']:
                
                print(f"Resampling band {band_name} to {target_res_str}")
                
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
                    'original_resolution': next(resolution_list[i] for i, b in enumerate(band_list) if b == band_name),
                    'resampled_to': target_res_str
                }
            else:
                # Band already at target resolution
                resampled_bands_data[band_name] = band_info.copy()
                resampled_bands_data[band_name]['original_resolution'] = next(resolution_list[i] for i, b in enumerate(band_list) if b == band_name)
                resampled_bands_data[band_name]['resampled_to'] = target_res_str
        
        bands_data = resampled_bands_data
        print(f"All bands resampled to {target_res_str} resolution")
    


    # Stack all bands into a single numpy array and return with unified profile
    if bands_data:
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
        
        # Add resampling information if available
        if 'resampled_to' in next(iter(bands_data.values())):
            result['resampling_info'] = {
                band_name: {
                    'original_resolution': band_info.get('original_resolution', 'unknown'),
                    'resampled_to': band_info.get('resampled_to', 'unknown')
                }
                for band_name, band_info in bands_data.items()
            }
        
        return result
    else:
        return {'data': None, 'profile': None, 'band_names': [], 'band_order': {}}


#--------------------------------------------------------------------------------

# saving data and profile info
def save_data_profile(bands_data, path:str, name:str):
    """
    Save the stacked bands data and profile
    
    Args:
        bands_data (dict): Result from read_sentinel2_bands with 'data' and 'profile' keys
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
        bands_data: Dictionary from read_sentinel2_bands containing band data
    
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
    nir_r = read_sentinel2_bands(img_path, ['B08', 'B04'], ['10m', '10m'])
    return {'data': get_ndvi_from_bands(nir_r), 'profile': nir_r['profile']}


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
    nir_swir = read_sentinel2_bands(img_path, ['B08', 'B11'], ['10m', '20m'])
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

    return dnbr_img

# Backward compatibility function
def extract_data_labels(img_list: list, output_dir: str, thresh: float = 0.6):
    """Legacy function for backward compatibility - reads images and calculates dNBR"""
    nbr_imgs = {}
    profiles = {}

    for i, img_path in enumerate(img_list):
        nir_swir = read_sentinel2_bands(img_path, ['B08', 'B12'], ['10m', '20m'])

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

# full processing pipeline to get all the patches data needed (chosen bands, ndvi, ndmi, labels in [0, 1])
def full_sentinel2_data_pipeline(dataset_name: str, 
                                   base_path: str = '/home/dario/Desktop/FlameSentinels',
                                   bands_to_process: list = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
                                   band_resolutions: list = ['10m', '10m', '10m', '10m', '20m', '20m'],
                                   patch_size: tuple = (256, 256),
                                   threshold: float = 0.6):
    """
    Complete processing pipeline for satellite data including bands, NDVI, NDMI, and labels extraction.
    
    Args:
        dataset_name (str): Common name for the dataset folder in which you should save (e.g., 'turkey', 'california', etc.). 

          Ideally the folder downloaded from Copernicus should be extracted and then renamed to <country>_pre or <country>_post
          And it should be placed into the base_path folder

        base_path (str): Base directory path where data folders are located
        bands_to_process (list): List of bands to process (to get full data input, together with ndvi and ndmi)
        band_resolutions (list): Corresponding resolutions for each band
        patch_size (tuple): Size of patches to extract (height, width)
        threshold (float): Threshold for binary dNBR classification
    """
    
    print(f"=== Starting processing pipeline for dataset: {dataset_name} ===")
    
    # Define paths
    post_fire_path = os.path.join(base_path, f'{dataset_name}_post')
    pre_fire_path = os.path.join(base_path, f'{dataset_name}_pre')
    full_img_path = os.path.join(base_path, f'{dataset_name}_full_img_results')
    
    # Output patch directories
    patches_bands_path = os.path.join(base_path, 'PATCHES_BANDS')
    patches_ndvi_path = os.path.join(base_path, 'PATCHES_NDVI') 
    patches_ndmi_path = os.path.join(base_path, 'PATCHES_NDMI')
    patches_labels_path = os.path.join(base_path, 'PATCHES_LABELS')
    
    # Step 1: Process bands data (read both pre and post fire data)
    print("\n--- Step 1: Processing bands data ---")
    print("Reading pre-fire bands...")
    pre_bands = read_sentinel2_bands(pre_fire_path, bands_to_process, band_resolutions)
    print("Reading post-fire bands...")
    post_bands = read_sentinel2_bands(post_fire_path, bands_to_process, band_resolutions)
    
    # Print information about the result
    print("Band names:", pre_bands['band_names'])
    print("Data shape:", pre_bands['data'].shape)
    print("Band order:", pre_bands['band_order'])
    if 'resampling_info' in pre_bands:
        print("Resampling info:", pre_bands['resampling_info'])
    
    # Save bands data (using pre-fire as reference)
    saved_npy_path = save_data_profile(pre_bands, full_img_path, f'{dataset_name}_pre')
    bands_data = np.load(saved_npy_path)
    
    # Extract patches from bands
    num_bands = len(bands_to_process)
    extract_patches_with_padding(bands_data, dataset_name, (*patch_size, num_bands), patches_bands_path)
    
    # Step 2: Process NDVI and NDMI using pre-extracted bands data
    print("\n--- Step 2: Processing NDVI and NDMI ---")
    
    # Try to use optimized functions first, fall back to path-based functions if bands are missing
    try:
        print("Attempting to compute NDVI from pre-extracted bands...")
        ndvi = get_ndvi_from_bands(pre_bands)  # Use optimized function
        print("✓ NDVI computed from pre-extracted bands")
    except ValueError as e:
        print(f"⚠ Warning: {e}")
        print("Falling back to reading bands directly from path for NDVI...")
        ndvi_result = get_ndvi(pre_fire_path)  # Fallback to backup function
        ndvi = ndvi_result['data']
        print("✓ NDVI computed from direct path reading")
    
    try:
        print("Attempting to compute NDMI from pre-extracted bands...")
        ndmi = get_ndmi_from_bands(pre_bands)  # Use optimized function
        print("✓ NDMI computed from pre-extracted bands")
    except ValueError as e:
        print(f"⚠ Warning: {e}")
        print("Falling back to reading bands directly from path for NDMI...")
        ndmi_result = get_ndmi(pre_fire_path)  # Fallback to backup function
        ndmi = ndmi_result['data']
        print("✓ NDMI computed from direct path reading")
    
    # Create temp directory and save indices
    os.makedirs(full_img_path, exist_ok=True)
    np.save(os.path.join(full_img_path, f'{dataset_name}_NDVI.npy'), ndvi)
    np.save(os.path.join(full_img_path, f'{dataset_name}_NDMI.npy'), ndmi)
    
    # Load and extract patches
    ndvi_np = np.load(os.path.join(full_img_path, f'{dataset_name}_NDVI.npy'))
    ndmi_np = np.load(os.path.join(full_img_path, f'{dataset_name}_NDMI.npy'))
    
    extract_patches_with_padding(ndvi_np, dataset_name, (*patch_size, 1), patches_ndvi_path)
    extract_patches_with_padding(ndmi_np, dataset_name, (*patch_size, 1), patches_ndmi_path)
    
    # Step 3: Process labels (dNBR) using pre-extracted bands data
    print("\n--- Step 3: Processing labels (dNBR) ---")
    
    # Try to use optimized function first, fall back to path-based function if bands are missing
    try:
        print("Attempting to compute dNBR from pre-extracted bands...")
        extract_data_labels_from_bands(pre_bands, post_bands, full_img_path, threshold)  # Use optimized function
        print("✓ dNBR computed from pre-extracted bands")
    except ValueError as e:
        print(f"⚠ Warning: {e}")
        print("Falling back to reading bands directly from paths for dNBR...")
        extract_data_labels([pre_fire_path, post_fire_path], full_img_path, threshold)  # Fallback to backup function
        print("✓ dNBR computed from direct path reading")
    
    # Load dNBR data and extract patches
    dnbr_normmap = np.load(os.path.join(full_img_path, 'dnbr_normalized.npy'))
    extract_patches_with_padding(dnbr_normmap, dataset_name, (*patch_size, 1), patches_labels_path)
    
    print(f"\n=== Processing pipeline completed for dataset: {dataset_name} ===")
    print(f"Generated patches in:")
    print(f"  - Bands: {patches_bands_path}")
    print(f"  - NDVI: {patches_ndvi_path}")
    print(f"  - NDMI: {patches_ndmi_path}")
    print(f"  - Labels: {patches_labels_path}")


if __name__ == '__main__':
    # Example usage: process the turkey dataset
    full_sentinel2_data_pipeline('turkey')
    
