import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

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
    
    # Define all available Sentinel-2 bands with their native resolutions
    # Note: B10 (Cirrus) is excluded as it's not available in L2A products
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

# reading bands data
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
    
    # save band names and order information
    with open(os.path.join(path, f'{name}_band_info.pkl'), 'wb') as f:
        band_info = {
            'band_names': bands_data['band_names'],
            'band_order': bands_data['band_order']
        }
        if 'resampling_info' in bands_data:
            band_info['resampling_info'] = bands_data['resampling_info']
        pickle.dump(band_info, f)

    return os.path.join(path, f'{name}.npy')

# get ndvi = (nir - r)/(nir + r)
def get_ndvi(img_path):
    nir_r = read_sentinel2_bands(img_path, ['B08', 'B04'], ['10m', '10m'])
    
    # Extract band data from stacked array
    band_order = nir_r['band_order']
    b08_idx = next(i for i, name in band_order.items() if name == 'B08')
    b04_idx = next(i for i, name in band_order.items() if name == 'B04')
    
    b08_data = nir_r['data'][:, :, b08_idx]
    b04_data = nir_r['data'][:, :, b04_idx]
    
    denom = b08_data + b04_data
    denom[denom==0] = 1e-6
    # avoid division by zero
    ndvi_img = (b08_data - b04_data)/denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    ndvi_img = ndvi_img[:, :, np.newaxis]

    return {'data': ndvi_img, 'profile': nir_r['profile']}

# get ndmi = (b08 - b11)/(b08 + b11)
def get_ndmi(img_path):
    nir_swir = read_sentinel2_bands(img_path, ['B08', 'B11'], ['10m', '20m'])
    
    # Extract band data from stacked array
    band_order = nir_swir['band_order']
    b08_idx = next(i for i, name in band_order.items() if name == 'B08')
    b11_idx = next(i for i, name in band_order.items() if name == 'B11')
    
    b08_data = nir_swir['data'][:, :, b08_idx]
    b11_data = nir_swir['data'][:, :, b11_idx]

    denom = b08_data + b11_data
    denom[denom==0] = 1e-6
    # avoid division by zero
    ndmi_img = (b08_data - b11_data)/denom
    
    # Add channel dimension to make it 3D (height, width, 1)
    ndmi_img = ndmi_img[:, :, np.newaxis]

    return {'data': ndmi_img, 'profile': nir_swir['profile']}



def extract_patches_with_padding(image, patch_size, path):
    """
    Extract patches using padding strategy to ensure complete coverage.
    
    Args:
        image: Input image (H, W, C)
        patch_size: Size of each patch (height, width, channels)
        path: Location where to save patches
    
    Returns:
        patches: List of patches
        positions: List of (row, col) positions for each patch
        padded_shape: Shape of the padded image
    """
    os.makedirs(path, exist_ok=True)

    h, w, c = image.shape
    print(image.shape)
    ph, pw, pc = patch_size
    
    # Calculate padding needed
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    
    # Pad with reflection to maintain natural patterns
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    patches = []
    positions = []
    
    # Extract non-overlapping patches
    padded_h, padded_w, _ = padded_image.shape
    for i in range(0, padded_h, ph):
        for j in range(0, padded_w, pw):
            patch = padded_image[i:i+ph, j:j+pw, :]
            np.save(os.path.join(path, f'patch_{(i, j)}'), patch)
            if patch.shape == patch_size:
                patches.append(patch)
                positions.append((i, j))
    
    return patches, positions, padded_image.shape


if __name__ == '__main__':


    turkey_post_bands = read_sentinel2_bands('/home/dario/Desktop/FlameSentinels/turkey_post', 
                                            ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
                                            ['10m', '10m', '10m', '10m', '20m', '20m'])
        
    # Print information about the result
    print("Band names:", turkey_post_bands['band_names'])
    print("Data shape:", turkey_post_bands['data'].shape)
    print("Band order:", turkey_post_bands['band_order'])
    if 'resampling_info' in turkey_post_bands:
        print("Resampling info:", turkey_post_bands['resampling_info'])

    saved_npy_path = save_data_profile(turkey_post_bands, '/home/dario/Desktop/FlameSentinels/turkey_np', 'turkey_post')

    bands_data = np.load(saved_npy_path)

    extract_patches_with_padding(bands_data, (256, 256, 6), '/home/dario/Desktop/FlameSentinels/PATCHES_BANDS')

    
        
    ndvi = get_ndvi('/home/dario/Desktop/FlameSentinels/turkey_post')
    ndmi = get_ndmi('/home/dario/Desktop/FlameSentinels/turkey_post')

    np.save(os.path.join('TEST_OUTPUT_NPY', f'NDVI.npy'), ndvi['data'])
    np.save(os.path.join('TEST_OUTPUT_NPY', f'NDMI.npy'), ndmi['data'])

    ndvi_np = np.load(os.path.join('TEST_OUTPUT_NPY', f'NDVI.npy'))
    ndmi_np = np.load(os.path.join('TEST_OUTPUT_NPY', f'NDMI.npy'))

    extract_patches_with_padding(ndvi_np,(256, 256, 1), '/home/dario/Desktop/FlameSentinels/PATCHES_NDVI')

    extract_patches_with_padding(ndmi_np, (256, 256, 1), '/home/dario/Desktop/FlameSentinels/PATCHES_NDMI')


