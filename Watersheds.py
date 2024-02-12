import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import label
import json
from tqdm import tqdm
import os
from scipy.interpolate import griddata
import pandas as pd




# Read layer names from the JSON file
for in_var, out_var, vd in zip(["tasmax", "tasmin"], ['tmax', 'tmin'], ['20220413', '20220413']):
    output_directories = [f"/home/ScoutJarman/Code/ILWA/data/rasters/LOCA/{out_var}_{i}/" for i in ["r1", "r2", "r3"]]
    
    r1_tiffs = [
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/historical/day/bearlake_{in_var}.ACCESS-CM2.historical.r1i1p1f1.1950-2014.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r1i1p1f1.2015-2044.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r1i1p1f1.2045-2074.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r1i1p1f1.2075-2100.LOCA_16thdeg_v{vd}.tif"
    ]
    r2_tiffs = [
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/historical/day/bearlake_{in_var}.ACCESS-CM2.historical.r2i1p1f1.1950-2014.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r2i1p1f1.2015-2044.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r2i1p1f1.2045-2074.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r2i1p1f1.2075-2100.LOCA_16thdeg_v{vd}.tif"
    ]
    r3_tiffs = [
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/historical/day/bearlake_{in_var}.ACCESS-CM2.historical.r3i1p1f1.1950-2014.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r3i1p1f1.2015-2044.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r3i1p1f1.2045-2074.LOCA_16thdeg_v{vd}.tif",
        f"/home/ScoutJarman/Code/ILWA/data/rasters/bear_lake/{in_var}/future/day/bearlake_{in_var}.ACCESS-CM2.ssp585.r3i1p1f1.2075-2100.LOCA_16thdeg_v{vd}.tif"
    ]

    for i, tiffs in enumerate([r1_tiffs, r2_tiffs, r3_tiffs]):
        # Check and make outputdirectory for r level
        output_directory = output_directories[i]
        if not os.path.exists(output_directory):
            print(f"Making {output_directory}")
            os.makedirs(output_directory)
        # Loop through each of the different tiff files
        for input_path in tiffs:
            with open(input_path + ".aux.json", 'r') as json_handle:
                layer_names = json.load(json_handle)['time']
            
            # # Open tiff and write each layer to new file
            # with rasterio.open(input_path) as src:
            #     # Loop through bands and layer names
            #     for band_idx, layer_name in tqdm(zip(range(1, src.count + 1), layer_names), total=src.count):
            #         # Read the band data
            #         band_data = src.read(band_idx)

            #         # Do nearest neighbor interpolation for lake values
            #         non_nan_indices = np.where(~np.isnan(band_data))
            #         points = np.column_stack((non_nan_indices[1], non_nan_indices[0]))
            #         values = band_data[non_nan_indices]
            #         rows, cols = band_data.shape
            #         grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
            #         interpolated_values = griddata(points, values, (grid_x, grid_y), method='nearest')
            #         band_data[np.isnan(band_data)] = interpolated_values[np.isnan(band_data)]

            #         year, month, day = layer_name.split("-")
            #         # Create a new raster file for each band with the corresponding layer name
            #         output_raster_path = f"{output_directory}/{out_var}{int(year)}{int(month):02d}{int(day):02d}.tif"

            #         # Write the band data to the new raster file with the same metadata
            #         with rasterio.open(output_raster_path, 'w', **src.profile) as dst:
            #             dst.write(band_data, 1)
        