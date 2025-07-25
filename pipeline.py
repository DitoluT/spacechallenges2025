import os
import numpy as np
import matplotlib.pyplot as plt
from bands_preprocessing import *
from clouddetection import *



# full processing pipeline to get all the patches data needed (chosen bands, ndvi, ndmi, labels in [0, 1])
def full_sentinel2_data_pipeline(dataset_name: str, 
                                   base_path: str = '/home/dario/Desktop/FlameSentinels',
                                   patch_size: tuple = (256, 256),
                                   threshold: float = 0.6,
                                   cloud_threshold: float = 0.6):
    """
    Complete processing pipeline for satellite data including bands extraction, cloud filtering, 
    vegetation filtering, and NDVI/NDMI extraction.
    
    Args:
        dataset_name (str): Common name for the dataset folder (e.g., 'turkey', 'california', etc.)
        base_path (str): Base directory path where data folders are located
        patch_size (tuple): Size of patches to extract (height, width)
        threshold (float): Threshold for binary dNBR classification
        cloud_threshold (float): Cloud probability threshold for S2PixelCloudDetector (0.0-1.0)
    """
    
    print(f"=== Starting processing pipeline for dataset: {dataset_name} ===")
    
    # Define paths
    full_img_path = os.path.join(base_path, f'{dataset_name}_full_img_results')
    
    # Output patch directories
    patches_bands_path = os.path.join(base_path, 'PATCHES_BANDS')
    patches_ndvi_path = os.path.join(base_path, 'PATCHES_NDVI') 
    patches_ndmi_path = os.path.join(base_path, 'PATCHES_NDMI')
    patches_labels_path = os.path.join(base_path, 'PATCHES_LABELS')
    
    # Step 1: Extract data from folder and read all 13 bands
    print("\n--- Step 1: Reading all 13 bands data ---")
    all_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'B10']
    print("Reading pre-fire bands...")
    pre_bands = read_sent2_1c_bands(f'{dataset_name}_pre', all_bands)
    print("Reading post-fire bands...")
    post_bands = read_sent2_1c_bands(f'{dataset_name}_post', all_bands)
    
    # Print information about the result
    print("Band names:", pre_bands['band_names'])
    print("Data shape:", pre_bands['data'].shape)
    print("Band order:", pre_bands['band_order'])
    if 'resampling_info' in pre_bands:
        print("Resampling info:", pre_bands['resampling_info'])
    
    # Step 2: Create label map (dNBR)
    print("\n--- Step 2: Creating label map (dNBR) ---")
    os.makedirs(full_img_path, exist_ok=True)
    extract_data_labels_from_bands(pre_bands, post_bands, full_img_path, threshold)
    print("✓ dNBR label map created")

    
    
    # Step 3: Divide into patches of size (256, 256, 13)
    print("\n--- Step 3: Extracting patches (256, 256, 13) ---")
    bands_data = pre_bands['data']  # Use pre-fire data for patches
    num_bands = len(all_bands)
    extract_patches_with_padding(bands_data, dataset_name, (*patch_size, num_bands), patches_bands_path)
    
    # Also extract label patches
    dnbr_normmap = np.load(os.path.join(full_img_path, 'dnbr_normalized.npy'))
    extract_patches_with_padding(dnbr_normmap, dataset_name, (*patch_size, 1), patches_labels_path)
    print(f"✓ Patches extracted to {patches_bands_path}")
    

    """
    # Step 4: Apply cloud detection function
    print("\n--- Step 4: Applying cloud detection ---")
    print(f"Using S2PixelCloudDetector with threshold: {cloud_threshold}")
    cloud_results = patch_cloud_detection(patches_bands_path, cloud_threshold)
    if cloud_results is not None:
        visualize_patch_results(cloud_results)
        print(f"✓ Cloud detection completed. Clean patches: {cloud_results['clean_patches']}, Cloudy patches moved: {cloud_results['cloudy_patches']}")
    else:
        print("❌ Cloud detection failed or no patches found")
    """
    
    # Step 5: Placeholder for vegetation detection model
    # TODO: Implement vegetation detection model that removes patches without sufficient vegetation
    # This model should:
    # - Analyze vegetation indices (NDVI/NDMI) in each remaining patch
    # - Apply a vegetation threshold (e.g., minimum 30% vegetation coverage)
    # - Move patches with insufficient vegetation to a separate folder (e.g., "LOW_VEGETATION_PATCHES")
    # - Return statistics similar to cloud detection results
    # Example function call: vegetation_results = patch_vegetation_detection(patches_bands_path, vegetation_threshold=0.3)

    # Step 6: Placeholder for black pixel filtering
    # TODO: Implement filtering for black patches (0 or smth like that for all rgb values)
    # - check if pixel is fully black (part of image padding)
    # - if so, discard it
    
    # Step 7: Extract NDVI and NDMI for remaining patches
    print("\n--- Step 6: Extracting NDVI and NDMI for remaining clean patches ---")
    
    # Process remaining patches in the PATCHES_BANDS folder (after cloud filtering)
    remaining_patch_files = [f for f in os.listdir(patches_bands_path) if f.endswith('.npy')]
    
    if remaining_patch_files:
        print(f"Processing {len(remaining_patch_files)} remaining clean patches...")
        
        # Get band names for vegetation index computation
        band_names = pre_bands['band_names']
        
        # Process NDVI patches
        process_patches_directory(patches_bands_path, band_names, patches_ndvi_path, index_type='ndvi')
        print(f"✓ NDVI patches saved to {patches_ndvi_path}")
        
        # Process NDMI patches  
        process_patches_directory(patches_bands_path, band_names, patches_ndmi_path, index_type='ndmi')
        print(f"✓ NDMI patches saved to {patches_ndmi_path}")
        
    else:
        print("⚠ No clean patches remaining after cloud detection!")
    
    # Step 8: Print folder locations
    print(f"\n=== Processing pipeline completed for dataset: {dataset_name} ===")
    print("📁 OUTPUT FOLDER LOCATIONS:")
    print(f"  🔸 Clean bands patches (13 bands): {patches_bands_path}")
    print(f"  🔸 NDVI patches: {patches_ndvi_path}")
    print(f"  🔸 NDMI patches: {patches_ndmi_path}")
    print(f"  🔸 Label patches (dNBR): {patches_labels_path}")

    #if cloud_results:
    #    print(f"  🔸 Cloudy patches (moved): {cloud_results['cloudy_dir']}")
    print(f"  🔸 Full image results: {full_img_path}")
    
    return {
        'dataset_name': dataset_name,
        'bands_patches': patches_bands_path,
        'ndvi_patches': patches_ndvi_path,
        'ndmi_patches': patches_ndmi_path,
        'label_patches': patches_labels_path,
        'full_img_results': full_img_path,
        #'cloud_results': cloud_results
    }

if __name__ == '__main__':
    full_sentinel2_data_pipeline('greece')