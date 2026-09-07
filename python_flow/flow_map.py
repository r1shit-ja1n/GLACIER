import numpy as np
import rasterio
from PIL import Image, ImageFilter
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
import scipy.ndimage as ndimage # NEW: For terrain-aware flooding

def simulate_glof_inundation(heightmap_path, output_path, glacier_x, glacier_y):
    print("Reading heightmap with Rasterio...")
    with rasterio.open(heightmap_path) as src:
        dem_array = src.read(1).astype(np.float64)
    
    rows, cols = dem_array.shape

    print("Setting up Landlab grid...")
    mg = RasterModelGrid((rows, cols), xy_spacing=(1.0, 1.0))
    z = mg.add_zeros('topographic__elevation', at='node')
    
    dem_flipped = np.flipud(dem_array)
    z[:] = dem_flipped.flatten()
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False)
    
    runoff = mg.add_zeros('water__unit_flux_in', at='node')
    landlab_y = (rows - 1) - glacier_y
    glacier_node = mg.grid_coords_to_node_id(landlab_y, glacier_x)
    runoff[glacier_node] = 1000000.0  
    
    print("Routing water down the mountains...")
    fa = FlowAccumulator(mg, flow_director='D8', depression_finder='DepressionFinderAndRouter')
    fa.run_one_step()
    
    discharge = mg.at_node['surface_water__discharge'].reshape((rows, cols))
    flood_path = np.flipud(discharge)
    
    # Get the exact 1-pixel center line of the flood
    main_river = flood_path > 0 
    
    # =================================================================
    # NEW: TERRAIN-AWARE VALLEY INUNDATION
    # =================================================================
    print("Simulating valley flooding based on terrain elevation...")
    
    # 1. Set how deep the flood is (Increase this for a more catastrophic flood)
    flood_depth = 10000.0 
    
    # 2. Create a Water Surface Elevation (WSE) array
    # The water level is the Terrain Elevation + the Flood Depth
    wse = np.where(main_river, dem_array + flood_depth, 0.0)
    
    # 3. Physically spill the water outward into the valleys
    max_spread = 50 # Maximum pixels the water is allowed to travel sideways
    
    for i in range(max_spread):
        # Push the water level outward by 1 pixel in all directions
        dilated_wse = ndimage.maximum_filter(wse, size=3)
        
        # CRITICAL MATH: Only allow the water to spread if the local water 
        # surface is higher than the surrounding terrain!
        wse = np.where(dilated_wse > dem_array, dilated_wse, wse)
        
    print("Baking final mask...")
    # Any pixel that has water surface > 0 is now flooded
    flood_mask = np.where(wse > 0, 255, 0).astype(np.uint8)
        
    out_img = Image.fromarray(flood_mask, 'L')
    
    # Soften the edges slightly so it blends perfectly into Babylon.js
    out_img = out_img.filter(ImageFilter.GaussianBlur(2)) 
    
    out_img.save(output_path)
    print(f"Success! True inundation map saved to {output_path}")

# Run it! (Make sure your coordinates are still correct)
simulate_glof_inundation("output_hh.tif", "flood_mask.png", 162, 135)