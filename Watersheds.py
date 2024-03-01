import rasterio
import numpy as np
import json
from tqdm import tqdm
import os
from scipy.interpolate import griddata
import datetime

def get_tif_files(directory, var_n):
    pt_tif_files = []
    for filename in os.listdir(directory):
        if var_n in filename and filename.endswith(".tif"):
            pt_tif_files.append(os.path.join(directory, filename))
    return pt_tif_files

def generate_days(year):
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    delta = datetime.timedelta(days=1)

    days_in_year = []
    current_date = start_date
    while current_date <= end_date:
        days_in_year.append(current_date.strftime("%Y-%m-%d"))
        current_date += delta

    return days_in_year


# Read layer names from the JSON file
# for in_var, out_var in zip(["pr", "tasmax", "tasmin", "SWE"], ['ppt', 'tmax', 'tmin', 'swe']):
for in_var, out_var in zip(["SWE"], ['swe']):
    output_directories = [f"/data/ScoutJarman/LOCA/LOCA/{out_var}_{i}/" for i in ["rcp45", "rcp85"]]
    for directory in output_directories:
        if not os.path.exists(directory):
            print(f"Making {directory}")
            os.makedirs(directory)
    
    if in_var != "SWE":
        tiffs_45 = get_tif_files("/data/ScoutJarman/LOCA/NCAR/met/ACCESS1-0/historical/", in_var)
        tiffs_45 += get_tif_files(f"/data/ScoutJarman/LOCA/NCAR/met/ACCESS1-0/rcp45/", in_var)
        tiffs_85 = get_tif_files("/data/ScoutJarman/LOCA/NCAR/met/ACCESS1-0/historical/", in_var)
        tiffs_85 += get_tif_files(f"/data/ScoutJarman/LOCA/NCAR/met/ACCESS1-0/rcp85/", in_var)
    else:
        tiffs_45 = get_tif_files("/data/ScoutJarman/LOCA/NCAR/vic/ACCESS1-0/historical/", in_var)
        tiffs_45 += get_tif_files(f"/data/ScoutJarman/LOCA/NCAR/vic/ACCESS1-0/rcp45/", in_var)
        tiffs_85 = get_tif_files("/data/ScoutJarman/LOCA/NCAR/vic/ACCESS1-0/historical/", in_var)
        tiffs_85 += get_tif_files(f"/data/ScoutJarman/LOCA/NCAR/vic/ACCESS1-0/rcp85/", in_var)

    for i, tiffs in enumerate([tiffs_45, tiffs_85]):
        output_directory = output_directories[i]
        # Loop through each of the different tiff files
        for input_path in tiffs:
            if in_var != "SWE":
                with open(input_path + ".aux.json", 'r') as json_handle:
                    layer_names = json.load(json_handle)['time']
            else:
                year = int(input_path.split(".")[1])
                layer_names = generate_days(year)

            # Open tiff and write each layer to new file
            with rasterio.open(input_path) as src:
                # Loop through bands and layer names
                for band_idx, layer_name in tqdm(zip(range(1, src.count + 1), layer_names), total=src.count, desc=f"{input_path}"):
                    # Read the band data
                    band_data = src.read(band_idx)

                    # Do nearest neighbor interpolation for lake values
                    non_nan_indices = np.where(~np.isnan(band_data))
                    points = np.column_stack((non_nan_indices[1], non_nan_indices[0]))
                    values = band_data[non_nan_indices]
                    rows, cols = band_data.shape
                    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
                    interpolated_values = griddata(points, values, (grid_x, grid_y), method='nearest')
                    band_data[np.isnan(band_data)] = interpolated_values[np.isnan(band_data)]

                    year, month, day = layer_name.split("-")
                    # Create a new raster file for each band with the corresponding layer name
                    output_raster_path = f"{output_directory}/{out_var}{int(year)}{int(month):02d}{int(day):02d}.tif"

                    # Write the band data to the new raster file with the same metadata
                    with rasterio.open(output_raster_path, 'w', **src.profile) as dst:
                        dst.write(band_data, 1)
        