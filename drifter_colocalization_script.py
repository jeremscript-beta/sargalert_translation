import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from fpdf import FPDF
import numpy as np
import pygrib
from scipy.interpolate import griddata

# Paths to your data files
points_info_path = r'C:\Users\Utilisateur\Desktop\image_rapport\points_info.xlsx'
drifter_data_path = r"C:\Users\Utilisateur\Downloads\drifter_6hour_2022.csv"
output_path = r'C:\Users\Utilisateur\Desktop\image_rapport\points_info_drifter.xlsx'
grib_file_path = r"C:\Users\Utilisateur\Downloads\wind_model.grib"

drifter_speed = True  # Set this flag to control the velocity source
# If drifter_speed = True, speed will be taken from the drifter CSV column,
# else it will be deduced from the position at t-1 and t+1.

# Load points_info and drifter data
points_info = pd.read_excel(points_info_path)
drifter_data = pd.read_csv(drifter_data_path)

# Filter points_info for only May 2022
points_info = points_info[points_info['Date'].str.startswith('202205')]

# Filter drifter_data for May 2022
drifter_data['time'] = pd.to_datetime(drifter_data['time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
drifter_data_may = drifter_data[(drifter_data['time'] >= '2022-05-01') & (drifter_data['time'] < '2022-06-01')]

# Wind interpolation - open GRIB file and extract wind data
grib_data = pygrib.open(grib_file_path)
grib_u_wind, grib_v_wind = [], []
latitudes, longitudes = None, None

# Extract GRIB data for wind components
for message in grib_data:
    if 'U wind component' in message.parameterName:
        if latitudes is None and longitudes is None:
            latitudes, longitudes = message.latlons()  # Extract lat/lon grid
        grib_u_wind.append((message.validDate, message.values))
    elif 'V wind component' in message.parameterName:
        grib_v_wind.append((message.validDate, message.values))

grib_data.close()  # Close the GRIB file after extraction

# Function to interpolate wind data at specific lat/lon points
def interpolate_wind(grib_wind_data, lats, lons, point_lat, point_lon, time_diff_func):
    time_diffs = np.array([time_diff_func(t) for t, _ in grib_wind_data])
    before_indices = np.where(time_diffs <= 0)[0]
    after_indices = np.where(time_diffs >= 0)[0]
    
    if before_indices.size == 0 or after_indices.size == 0:
        # Cannot interpolate if we don't have both before and after times
        return np.nan
    
    before_idx = before_indices[-1]
    after_idx = after_indices[0]
    
    # Bilinear interpolation for the "before" and "after" wind grids
    u_before = griddata(
        (lats.flatten(), lons.flatten()), grib_wind_data[before_idx][1].flatten(),
        (point_lat, point_lon), method='linear'
    )
    u_after = griddata(
        (lats.flatten(), lons.flatten()), grib_wind_data[after_idx][1].flatten(),
        (point_lat, point_lon), method='linear'
    )
    
    # Time interpolation
    total_time_diff = time_diffs[after_idx] - time_diffs[before_idx]
    if total_time_diff == 0:
        return u_before  # Times are the same
    weight = -time_diffs[before_idx] / total_time_diff
    return u_before * (1 - weight) + u_after * weight

# Step to check drifter colocalization
def check_drifter_colocalization(points_info_row, drifter_data, max_time_diff=timedelta(hours=6), max_distance_km=50):
    point_time_str = points_info_row['Date'][:8]
    mod_time_str = points_info_row['Date'][9:13]
    myd_time_str = points_info_row['Date'][14:18]
    mod_time = datetime.strptime(point_time_str + mod_time_str, '%Y%m%d%H%M')
    myd_time = datetime.strptime(point_time_str + myd_time_str, '%Y%m%d%H%M')
    image_time = mod_time + (myd_time - mod_time) / 2
    point_coordinates = (points_info_row['Mod_Lat'], points_info_row['Mod_Lon'])
    best_match = None
    best_distance = float('inf')
    
    for _, drifter_row in drifter_data.iterrows():
        drifter_time = drifter_row['time']
        if pd.isnull(drifter_time):
            continue
        drifter_coordinates = (drifter_row['latitude'], drifter_row['longitude'])
        time_difference = abs(image_time - drifter_time)
        delta_t_min = time_difference.total_seconds() / 60  # Time difference in min
        distance = geodesic(point_coordinates, drifter_coordinates).km
        
        if time_difference <= max_time_diff and distance <= max_distance_km:
            drogue_lost_date = drifter_row.get('drogue_lost_date', None)
            drogue = 1
            if pd.notnull(drogue_lost_date):
                drogue_lost_time = pd.to_datetime(drogue_lost_date, format='%Y-%m-%d', errors='coerce')
                if drogue_lost_time <= drifter_time:
                    drogue = 0
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    'drifter_id': drifter_row['ID'],
                    'drifter_info': drifter_row,
                    'point_info': points_info_row,
                    'distance': distance,
                    'delta_t': delta_t_min,
                    'drogue': drogue,
                    'image_time': image_time
                }
    
    if best_match:
        return [best_match]
    return []

# Velocity calculation
def calculate_drifter_velocity(drifter_id, match_time, image_time, drifter_row):
    if drifter_speed:
        u_drifter = drifter_row['ve']
        v_drifter = drifter_row['vn']
        if pd.isna(u_drifter) or pd.isna(v_drifter):
            return None, None, None, None
        lat_start, lon_start = drifter_row['latitude'], drifter_row['longitude']
        lat_end, lon_end = lat_start, lon_start
    else:
        if image_time > match_time:
            time_start = match_time
            time_end = match_time + timedelta(hours=6)
        else:
            time_start = match_time - timedelta(hours=6)
            time_end = match_time
        drifter_start = drifter_data_may[(drifter_data_may['ID'] == drifter_id) & (drifter_data_may['time'] == time_start)]
        drifter_end = drifter_data_may[(drifter_data_may['ID'] == drifter_id) & (drifter_data_may['time'] == time_end)]
        if drifter_start.empty or drifter_end.empty:
            return None, None, None, None
        lat_start, lon_start = float(drifter_start.iloc[0]['latitude']), float(drifter_start.iloc[0]['longitude'])
        lat_end, lon_end = float(drifter_end.iloc[0]['latitude']), float(drifter_end.iloc[0]['longitude'])
        u_distance = geodesic((lat_start, lon_start), (lat_start, lon_end)).meters
        v_distance = geodesic((lat_start, lon_start), (lat_end, lon_start)).meters
        if lon_end < lon_start:
            u_distance = -u_distance
        if lat_end < lat_start:
            v_distance = -v_distance
        time_in_seconds = 6 * 3600
        u_drifter = u_distance / time_in_seconds
        v_drifter = v_distance / time_in_seconds
    return u_drifter, v_drifter, (lat_start, lon_start), (lat_end, lon_end)

# Initialize a list to store the new results
points_info_drifter = []

# Process matches and calculate drifter velocities and errors
for i in range(len(points_info)):
    for match in check_drifter_colocalization(points_info.iloc[i], drifter_data_may):
        drifter_id = match['drifter_info']['ID']
        match_time = match['drifter_info']['time']
        image_time = match['image_time']
        
        # Calculate drifter velocity
        u_drifter, v_drifter, drifter_start, drifter_end = calculate_drifter_velocity(drifter_id, match_time, image_time, match['drifter_info'])

        if u_drifter is not None and v_drifter is not None:
            u_LT = match['point_info']['u_LT']
            v_LT = match['point_info']['v_LT']
            
            # Interpolate wind data at the drifter location
            drifter_lat, drifter_lon = drifter_start
            drifter_time = pd.to_datetime(match['point_info']['Date'][:8], format='%Y%m%d')
            time_diff_func = lambda t: (t - drifter_time).total_seconds()
            u_wind_model = interpolate_wind(grib_u_wind, latitudes, longitudes, drifter_lat, drifter_lon, time_diff_func)
            v_wind_model = interpolate_wind(grib_v_wind, latitudes, longitudes, drifter_lat, drifter_lon, time_diff_func)
            
            # Combine match data with new drifter velocity data and interpolated wind data
            result = match['point_info'].copy()
            result['u_drifter'] = u_drifter
            result['v_drifter'] = v_drifter
            result['distance'] = match['distance']
            result['delta_t'] = match['delta_t']
            result['drogue'] = match['drogue']
            result['drifter_id'] = drifter_id
            result['drifter_start_lat'], result['drifter_start_lon'] = drifter_start
            result['drifter_end_lat'], result['drifter_end_lon'] = drifter_end
            result['u_wind_model'] = u_wind_model
            result['v_wind_model'] = v_wind_model
            points_info_drifter.append(result)

# Convert the results to a DataFrame
points_info_drifter_df = pd.DataFrame(points_info_drifter)

# Add units for each variable as the first row
units = {
    'Date': 'YYYYMMDD_hmod_hmyd',  # Example format for the date
    'Mod_Lat': 'degrees', 
    'Mod_Lon': 'degrees', 
    'Myd_Lat': 'degrees', 
    'Myd_Lon': 'degrees', 
    'u_LT': 'm/s',  # Eastward velocity (m/s)
    'v_LT': 'm/s',  # Northward velocity (m/s)
    'u_OF': 'm/s',  # Eastward velocity (m/s)
    'v_OF': 'm/s',
    'u_drifter': 'm/s',  # Drifter eastward velocity (m/s)
    'v_drifter': 'm/s',  # Drifter northward velocity (m/s)
    'distance': 'km',  # Distance between drifter and point (km)
    'delta_t': 'min',  # Time difference between satellite and drifter data (hours)
    'drogue': '1 = has drogue, 0 = lost drogue',
    'drifter_id': 'ID',  # ID of the drifter
    'drifter_start_lat': 'degrees', 
    'drifter_start_lon': 'degrees', 
    'drifter_end_lat': 'degrees', 
    'drifter_end_lon': 'degrees',
    'u_wind_model': 'm/s',  # Wind model eastward velocity (m/s)
    'v_wind_model': 'm/s',  # Wind model northward velocity (m/s)
    'Sat': 'sensor', 
}

# Convert the units dictionary into a DataFrame
units_df = pd.DataFrame([units])

# Concatenate the units row with the actual data
points_info_drifter_with_units = pd.concat([units_df, points_info_drifter_df], ignore_index=True)

# Save the new DataFrame with units row to Excel
points_info_drifter_with_units.to_excel(output_path, index=False)

print(f"Excel file with units saved to {output_path}")

# Function to plot vectors on a geographical map with dynamic borders and proper scale vector placement
def plot_vectors(mod_lat, mod_lon, drifter_start, drifter_end, u_drifter, v_drifter, u_LT, v_LT, u_wind_model, v_wind_model, drifter_id, save_path, title):
    lat_start, lon_start = drifter_start
    lat_end, lon_end = drifter_end
    
    # Convert coordinates to floats (if they are strings)
    lat_start = float(lat_start)
    lon_start = float(lon_start)
    lat_end = float(lat_end)
    lon_end = float(lon_end)
    
    # Calculate the barycenter of the two drifter points
    barycenter_lat = (lat_start + lat_end) / 2
    barycenter_lon = (lon_start + lon_end) / 2

    # Calculate the mean lat/lon and set the boundaries for the plot
    mean_lon = (mod_lon + barycenter_lon) / 2
    mean_lat = (mod_lat + barycenter_lat) / 2
    lon_min, lon_max = mean_lon - 2, mean_lon + 2
    lat_min, lat_max = mean_lat - 2, mean_lat + 2

    # Create the figure and axis with Cartopy projection
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set plot boundaries
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add coastlines and features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)
    
    # Set lat/lon ticks and labels
    ax.set_xticks(np.arange(np.floor(lon_min), np.ceil(lon_max), 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.floor(lat_min), np.ceil(lat_max), 1), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}°'))

    # Plot the drifter points
    ax.scatter([lon_start, lon_end], [lat_start, lat_end], color='purple', label='Drifter Points', s=8)

    # Plot the drifter velocity vector at the barycenter
    ax.quiver(barycenter_lon, barycenter_lat, u_drifter, v_drifter, color='blue', scale=1, scale_units='inches', label='Drifter')

    # Plot the LT vector at the MOD image location
    ax.quiver(mod_lon, mod_lat, u_LT, v_LT, color='red', scale=1, scale_units='inches', label='LT')
    
    # Plot the wind model vector at the drifter location
    ax.quiver(drifter_start[1], drifter_start[0], u_wind_model, v_wind_model, color='green', scale=1, scale_units='inches', label='Wind Model')

    # Add a scale vector (0.25 m/s) in the bottom-left corner
    scale_lon = lon_min + 0.05
    scale_lat = lat_min + 0.05
    ax.quiver(scale_lon, scale_lat, 0.25, 0, color='black', scale=1, scale_units='inches')  # Scale vector with u = 0.25, v = 0
    ax.text(scale_lon + 0.1, scale_lat - 0.1, '0.25 m/s', color='black')
    
    # Set the title, including drifter ID
    ax.set_title(f"{title}\nDrifter ID: {drifter_id}")
    
    # Plot a grid and set aspect ratio
    ax.grid(True)

    # Below the plot, add numerical values for LT, drifter, and wind model vectors
    plt.figtext(0.1, -0.05, f'LT Vector (u, v): ({u_LT:.4f}, {v_LT:.4f})', fontsize=10, color='red', ha='left')
    plt.figtext(0.1, -0.1, f'Drifter Vector (u, v): ({u_drifter:.4f}, {v_drifter:.4f})', fontsize=10, color='blue', ha='left')
    plt.figtext(0.1, -0.15, f'Wind Model Vector (u, v): ({u_wind_model:.4f}, {v_wind_model:.4f})', fontsize=10, color='green', ha='left')
    
    # Save the figure as a PNG file
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    
# Initialize a list to store the paths of generated plots
plot_paths = []

# Process matches and create geographical plots
for i, match in enumerate(points_info_drifter_df.to_dict('records')):
    mod_lat = match['Mod_Lat']
    mod_lon = match['Mod_Lon']
    drifter_start = (match['drifter_start_lat'], match['drifter_start_lon'])
    drifter_end = (match['drifter_end_lat'], match['drifter_end_lon'])
    u_drifter = match['u_drifter']
    v_drifter = match['v_drifter']
    u_LT = match['u_LT']
    v_LT = match['v_LT']
    u_wind_model = match['u_wind_model']
    v_wind_model = match['v_wind_model']
    
    # Ensure 'drifter_id' is available in match
    drifter_id = match.get('drifter_id', 'Unknown')  # Use 'Unknown' if the key 'drifter_id' doesn't exist
    
    # Create a unique filename for each plot
    plot_path = f'plot_{i}.png'
    
    # Call the plot function with modifications, including drifter ID
    plot_vectors(
        mod_lat, mod_lon, drifter_start, drifter_end,
        u_drifter, v_drifter, u_LT, v_LT,
        u_wind_model, v_wind_model, drifter_id,
        plot_path, f"Match {i+1}: {match['Date']}"
    )
    
    # Append the path to the list
    plot_paths.append(plot_path)

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add each plot as a new page in the PDF
for plot_path in plot_paths:
    pdf.add_page()
    pdf.image(plot_path, x=10, y=10, w=190)  # Add the plot image to the page

# Save the final PDF
pdf_output_path = r'C:\Users\Utilisateur\Desktop\image_rapport\points_info_plots.pdf'
pdf.output(pdf_output_path)

print(f"PDF saved to {pdf_output_path}")