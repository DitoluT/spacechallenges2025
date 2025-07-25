import os
import numpy as np
import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector


os.chdir('/home/dario/Desktop/FlameSentinels')

# Configuration
CLOUD_THRESHOLD = 0.4  # S2PixelCloudDetector threshold

# Initialize cloud detector
print("🔧 Initializing S2PixelCloudDetector...")
print("✅ Cloud detector ready!")

def patch_cloud_detection(patches_folder, cloud_threshold=0.6):
    """
    Process a folder of patches, detect clouds, and move cloudy patches to separate folder.
    Uses S2PixelCloudDetector's built-in threshold for cloud detection.
    
    Args:
        patches_folder: Path to folder containing .npy patch files
        cloud_threshold: Cloud probability threshold for S2PixelCloudDetector (0.0-1.0)
    
    Returns:
        results: Dictionary with processing statistics
    """
    
    # Initialize cloud detector inside the function
    detector = S2PixelCloudDetector(threshold=cloud_threshold, average_over=4, dilation_size=2, all_bands=True)

    print(f"\n🚀 PROCESSING PATCHES IN: {patches_folder}")
    print("=" * 60)
    
    # Create cloudy patches directory
    cloudy_dir = os.path.join(os.path.dirname(patches_folder), "CLOUDY_PATCHES")
    os.makedirs(cloudy_dir, exist_ok=True)
    
    # Get all patch files
    patch_files = [f for f in os.listdir(patches_folder) if f.endswith('.npy')]
    
    if not patch_files:
        print("❌ No .npy patch files found!")
        return None
    
    print(f"📂 Found {len(patch_files)} patch files")
    print(f"☁️  Processing with cloud threshold: {cloud_threshold}")
    
    # Process each patch
    clean_count = 0
    cloudy_count = 0
    patch_stats = []
    
    for idx, filename in enumerate(patch_files):
        patch_path = os.path.join(patches_folder, filename)
        
        # Load patch
        patch = np.load(patch_path)
        
        # Check if patch is cloudy using built-in detector threshold
        patch_batch = patch[np.newaxis, ...]  # Add batch dimension
        cloud_mask = detector.get_cloud_masks(patch_batch)[0]
        cloud_coverage = cloud_mask.sum() / cloud_mask.size
        is_cloudy = cloud_mask.any()  # If any pixels are detected as cloudy
        
        patch_info = {
            'filename': filename,
            'cloud_coverage': cloud_coverage,
            'status': 'cloudy' if is_cloudy else 'clean'
        }
        patch_stats.append(patch_info)
        
        if not is_cloudy:
            clean_count += 1
            status = "✅ CLEAN"
        else:
            # Move cloudy patch to cloudy directory
            cloudy_path = os.path.join(cloudy_dir, filename)
            os.rename(patch_path, cloudy_path)
            cloudy_count += 1
            status = "❌ CLOUDY → MOVED"
        
        print(f"   Patch {idx+1:3d}/{len(patch_files)}: {filename} | {cloud_coverage*100:5.1f}% clouds | {status}")
    
    # Compile results
    results = {
        'patches_folder': patches_folder,
        'total_patches': len(patch_files),
        'clean_patches': clean_count,
        'cloudy_patches': cloudy_count,
        'acceptance_rate': clean_count / len(patch_files) * 100,
        'cloudy_dir': cloudy_dir,
        'patch_stats': patch_stats,
        'threshold_used': cloud_threshold
    }
    
    # Display summary
    print(f"\n📊 PROCESSING RESULTS")
    print("=" * 40)
    print(f"Total patches processed: {results['total_patches']}")
    print(f"Clean patches (kept): {results['clean_patches']}")
    print(f"Cloudy patches (moved): {results['cloudy_patches']}")
    print(f"Acceptance rate: {results['acceptance_rate']:.1f}%")
    print(f"Cloud threshold used: {cloud_threshold}")
    print(f"Cloudy patches moved to: {cloudy_dir}/")
    print(f"\n✅ Cloud detection completed!")
    
    return results

# Visualize the patch cloud detection results
def visualize_patch_results(results):
    """
    Create visualizations showing the patch cloud detection results.
    """
    if results is None:
        print("❌ No results to visualize!")
        return
        
    print("\n🎨 CREATING VISUALIZATIONS...")
    
    # Get patch statistics
    patch_stats = results['patch_stats']
    clean_stats = [p for p in patch_stats if p['status'] == 'clean']
    cloudy_stats = [p for p in patch_stats if p['status'] == 'cloudy']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cloud coverage histogram
    coverages = [stat['cloud_coverage'] for stat in patch_stats]
    ax1.hist(coverages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    threshold_used = results.get('threshold_used', CLOUD_THRESHOLD)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                label=f'Visual Reference (50%)')
    ax1.set_xlabel('Cloud Coverage')
    ax1.set_ylabel('Number of Patches')
    ax1.set_title('Distribution of Cloud Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Status pie chart
    status_counts = [results['clean_patches'], results['cloudy_patches']]
    status_labels = ['Clean (Kept)', 'Cloudy (Moved)']
    colors = ['green', 'red']
    
    wedges, texts, autotexts = ax2.pie(status_counts, labels=status_labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax2.set_title('Patch Classification Results')
    
    # 3. Cloud coverage vs patch index
    patch_indices = range(len(patch_stats))
    clean_indices = [i for i, stat in enumerate(patch_stats) if stat['status'] == 'clean']
    cloudy_indices = [i for i, stat in enumerate(patch_stats) if stat['status'] == 'cloudy']
    clean_coverages = [patch_stats[i]['cloud_coverage'] for i in clean_indices]
    cloudy_coverages = [patch_stats[i]['cloud_coverage'] for i in cloudy_indices]
    
    ax3.scatter(clean_indices, clean_coverages, c='green', alpha=0.7, label='Clean', s=30)
    ax3.scatter(cloudy_indices, cloudy_coverages, c='red', alpha=0.7, label='Cloudy', s=30)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                label=f'Visual Reference (50%)')
    ax3.set_xlabel('Patch Index')
    ax3.set_ylabel('Cloud Coverage')
    ax3.set_title('Cloud Coverage by Patch')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics bar chart
    categories = ['Total\nPatches', 'Clean\nPatches', 'Cloudy\nPatches']
    values = [results['total_patches'], results['clean_patches'], results['cloudy_patches']]
    colors = ['lightblue', 'green', 'red']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_title('Processing Statistics')
    ax4.set_ylabel('Number of Patches')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Add acceptance rate text
    ax4.text(0.5, 0.95, f'Acceptance Rate: {results["acceptance_rate"]:.1f}%',
             transform=ax4.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n📈 DETAILED STATISTICS:")
    print("-" * 40)
    
    if clean_stats:
        clean_coverages = [s['cloud_coverage'] for s in clean_stats]
        print(f"Clean patches cloud coverage:")
        print(f"  Average: {np.mean(clean_coverages)*100:.1f}%")
        print(f"  Range: {np.min(clean_coverages)*100:.1f}% - {np.max(clean_coverages)*100:.1f}%")
        print(f"  Standard deviation: {np.std(clean_coverages)*100:.1f}%")
    
    if cloudy_stats:
        cloudy_coverages = [s['cloud_coverage'] for s in cloudy_stats]
        print(f"Cloudy patches cloud coverage:")
        print(f"  Average: {np.mean(cloudy_coverages)*100:.1f}%")
        print(f"  Range: {np.min(cloudy_coverages)*100:.1f}% - {np.max(cloudy_coverages)*100:.1f}%")
        print(f"  Standard deviation: {np.std(cloudy_coverages)*100:.1f}%")
    
    print(f"\nThreshold used: {results.get('threshold_used', CLOUD_THRESHOLD)}")
    print(f"Data efficiency: {results['acceptance_rate']:.1f}% of patches retained")
    print(f"Cloudy patches location: {results['cloudy_dir']}")
    print(f"Note: S2PixelCloudDetector uses built-in threshold for cloud detection")



if __name__ == '__main__':


    # Test with patch folder
    print("🎯 PATCH CLOUD DETECTION TEST")
    print("=" * 60)

    # Define patches folder
    PATCHES_FOLDER = "/home/dario/Desktop/FlameSentinels/PATCHES_BANDS"

    if os.path.exists(PATCHES_FOLDER):
        # Run the cloud detection
        results = patch_cloud_detection(PATCHES_FOLDER, CLOUD_THRESHOLD)
    else:
        print(f"❌ Patches folder not found: {PATCHES_FOLDER}")
        results = None



    # Create visualizations if we have results
    if results is not None:
        visualize_patch_results(results)