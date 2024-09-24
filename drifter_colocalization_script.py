import pandas as pd 
from datetime import datetime, timedelta
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from fpdf import FPDF
import numpy as np

# Load data
points_info_path = r'C:\Users\Utilisateur\Desktop\image_rapport\points_info.xlsx'
drifter_data_path = r"C:\Users\Utilisateur\Downloads\drifter_6hour_2022.csv"
# Output path
output_path = r'C:\Users\Utilisateur\Desktop\image_rapport\points_info_drifter.xlsx'

points_info = pd.read_excel(points_info_path)
drifter_data = pd.read_csv(drifter_data_path)

# Filter points_info for only May 2022
points_info = points_info[points_info['Date'].str.startswith('202205')]

# Filter drifter_data for May 2022
drifter_data['time'] = pd.to_datetime(drifter_data['time'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')
drifter_data_may = drifter_data[(drifter_data['time'] >= '2022-05-01') & (drifter_data['time'] < '2022-06-01')]

def check_drifter_colocalization(points_info_row, drifter_data, max_time_diff=timedelta(hours=6), max_distance_km=50):
    # Extract time and coordinates from points_info
    point_time_str = points_info_row['Date'][:8]  # Extracting the date part YYYYMMDD
    mod_time_str = points_info_row['Date'][9:13]  # Extracting MOD time (HHMM)
    myd_time_str = points_info_row['Date'][14:18]  # Extracting MYD time (HHMM)

    # Convert mod and myd times to datetime objects
    mod_time = datetime.strptime(point_time_str + mod_time_str, '%Y%m%d%H%M')
    myd_time = datetime.strptime(point_time_str + myd_time_str, '%Y%m%d%H%M')

    # Calculate the mean (average) time between mod and myd
    image_time = mod_time + (myd_time - mod_time) / 2  # Mean time between MOD and MYD
    
    point_coordinates = (points_info_row['Mod_Lat'], points_info_row['Mod_Lon'])

    # Initialize variables to track the closest match
    best_match = None
    best_distance = float('inf')  # Set to a very large number initially
    
    # Loop through drifter_data to find a match
    for index, drifter_row in drifter_data.iterrows():
        drifter_time = drifter_row['time']
        if pd.isnull(drifter_time):
            continue  # Skip rows where time could not be parsed
        
        # Parse drifter coordinates
        drifter_coordinates = (drifter_row['latitude'], drifter_row['longitude'])
        
        # Calculate time difference
        time_difference = abs(image_time - drifter_time)
        delta_t_hours = time_difference.total_seconds() / 60 # Time difference in hours
        
        # Calculate geographic distance
        distance = geodesic(point_coordinates, drifter_coordinates).km
        
        # Check if time difference and distance are within the threshold
        if time_difference <= max_time_diff and distance <= max_distance_km:
            drogue_lost_date = drifter_row.get('drogue_lost_date', None)
            drogue = 1  # Default assumption: drifter still has drogue
            if pd.notnull(drogue_lost_date):
                drogue_lost_time = pd.to_datetime(drogue_lost_date, format='%Y-%m-%d', errors='coerce')
                # Check if the drogue was lost before or after the drifter_time
                if drogue_lost_time <= drifter_time:
                    drogue = 0  # The drifter lost its drogue
            
            # If this match is closer than the previous best, update best match
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    'drifter_id': drifter_row['ID'],
                    'drifter_info': drifter_row,
                    'point_info': points_info_row,
                    'distance': distance,  # Store the calculated distance
                    'delta_t': delta_t_hours,  # Time difference between satellite and drifter data
                    'drogue': drogue,  # Store drogue status: 1 = still has it, 0 = lost
                    'image_time': image_time  # Store the image time for velocity calculation
                }
    
    # Return the best match (if any)
    if best_match:
        return [best_match]  # Return as a list for consistency
    return []

def calculate_drifter_velocity(drifter_id, match_time, image_time):
    if image_time > match_time:
        # If the image time is greater, we use match_time and time_after
        time_start = match_time
        time_end = match_time + timedelta(hours=6)
    else:
        # If the image time is smaller, we use match_time and time_before
        time_start = match_time - timedelta(hours=6)
        time_end = match_time

    # Find the rows corresponding to start and end times
    drifter_start = drifter_data_may[(drifter_data_may['ID'] == drifter_id) & (drifter_data_may['time'] == time_start)]
    drifter_end = drifter_data_may[(drifter_data_may['ID'] == drifter_id) & (drifter_data_may['time'] == time_end)]

    if drifter_start.empty or drifter_end.empty:
        return None, None, None, None  # Return None if data is missing for any of the time points
    
    # Extract positions and convert to floats
    lat_start, lon_start = float(drifter_start.iloc[0]['latitude']), float(drifter_start.iloc[0]['longitude'])
    lat_end, lon_end = float(drifter_end.iloc[0]['latitude']), float(drifter_end.iloc[0]['longitude'])
    
    # Calculate East-West (longitude) and North-South (latitude) distances
    u_distance = geodesic((lat_start, lon_start), (lat_start, lon_end)).meters  # Longitude difference
    v_distance = geodesic((lat_start, lon_start), (lat_end, lon_start)).meters  # Latitude difference
    
    # Adjust the sign based on the direction of movement
    if lon_end < lon_start:
        u_distance = -u_distance  # Moving west
    if lat_end < lat_start:
        v_distance = -v_distance  # Moving south
    
    # Time difference in seconds
    time_in_seconds = 6 * 3600  # 6 hours = 21,600 seconds
    
    # Calculate velocity (in m/s)
    u_drifter = u_distance / time_in_seconds
    v_drifter = v_distance / time_in_seconds

    # Return the start and end points along with the calculated velocity
    return u_drifter, v_drifter, (lat_start, lon_start), (lat_end, lon_end)


# Initialize a list to store the new results
points_info_drifter = []

# Process matches and calculate drifter velocities and errors
for i in range(len(points_info)):  # Iterate over points_info
    for match in check_drifter_colocalization(points_info.iloc[i], drifter_data_may):
        drifter_id = match['drifter_info']['ID']  # Extract drifter ID from drifter_info
        match_time = match['drifter_info']['time']
        image_time = match['image_time']  # Extract the image_time from match
        
        # Calculate drifter velocity based on the image_time and match_time
        u_drifter, v_drifter, drifter_start, drifter_end = calculate_drifter_velocity(drifter_id, match_time, image_time)
        
        if u_drifter is not None and v_drifter is not None:
            # Extract LT results from points_info
            u_LT = match['point_info']['u_LT']
            v_LT = match['point_info']['v_LT']
            
            # Combine match data with new drifter velocity data and errors
            result = match['point_info'].copy()
            result['u_drifter'] = u_drifter
            result['v_drifter'] = v_drifter
            result['distance'] = match['distance']  # Distance between drifter and point
            result['delta_t'] = match['delta_t']  # Time difference between satellite and drifter data
            result['drogue'] = match['drogue']  # Drogue status
            result['drifter_id'] = drifter_id  # Add drifter_id
            result['drifter_start_lat'], result['drifter_start_lon'] = drifter_start  # Add drifter start coordinates
            result['drifter_end_lat'], result['drifter_end_lon'] = drifter_end  # Add drifter end coordinates
            points_info_drifter.append(result)

# Convert the results to a DataFrame
points_info_drifter_df = pd.DataFrame(points_info_drifter)

# Save to Excel
points_info_drifter_df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")

# Function to plot vectors on a geographical map with dynamic borders and proper scale vector placement
def plot_vectors(mod_lat, mod_lon, drifter_start, drifter_end, u_drifter, v_drifter, u_LT, v_LT, drifter_id, save_path, title):
    lat_start, lon_start = drifter_start
    lat_end, lon_end = drifter_end

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
    ax.scatter([lon_start, lon_end], [lat_start, lat_end], color='purple', label='Drifter Points',s=8)

    # Plot the drifter velocity vector at the barycenter
    ax.quiver(barycenter_lon, barycenter_lat, u_drifter, v_drifter, color='blue', scale=1, scale_units='inches', label='Drifter')

    # Plot the LT vector at the MOD image location
    ax.quiver(mod_lon, mod_lat, u_LT, v_LT, color='red', scale=1, scale_units='inches', label='LT')
    
    # Add a scale vector (0.25 m/s) in the bottom-left corner
    scale_lon = lon_min + 0.05
    scale_lat = lat_min + 0.05
    ax.quiver(scale_lon, scale_lat, 0.25, 0, color='black', scale=1, scale_units='inches')  # Scale vector with u = 0.25, v = 0
    ax.text(scale_lon + 0.1, scale_lat - 0.1, '0.25 m/s', color='black')
    
    # Set the title, including drifter ID
    ax.set_title(f"{title}\nDrifter ID: {drifter_id}")
    
    # Plot a grid and set aspect ratio
    ax.grid(True)

    # Below the plot, add numerical values for LT and drifter vectors
    plt.figtext(0.1, -0.05, f'LT Vector (u, v): ({u_LT:.4f}, {v_LT:.4f})', fontsize=10, color='red', ha='left')
    plt.figtext(0.1, -0.1, f'Drifter Vector (u, v): ({u_drifter:.4f}, {v_drifter:.4f})', fontsize=10, color='blue', ha='left')
    
    # Save the figure as a PNG file
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)
    
# Initialize a list to store the paths of generated plots
plot_paths = []

# Process matches and create geographical plots
for i, match in enumerate(points_info_drifter):
    mod_lat = match['Mod_Lat']
    mod_lon = match['Mod_Lon']
    drifter_start = (match['drifter_start_lat'], match['drifter_start_lon'])
    drifter_end = (match['drifter_end_lat'], match['drifter_end_lon'])
    u_drifter = match['u_drifter']
    v_drifter = match['v_drifter']
    u_LT = match['u_LT']
    v_LT = match['v_LT']
    
    # Ensure 'drifter_id' is available in match
    drifter_id = match.get('drifter_id', 'Unknown')  # Use 'Unknown' if the key 'drifter_id' doesn't exist
    
    # Create a unique filename for each plot
    plot_path = f'plot_{i}.png'
    
    # Call the plot function with modifications, including drifter ID
    plot_vectors(mod_lat, mod_lon, drifter_start, drifter_end, u_drifter, v_drifter, u_LT, v_LT, drifter_id, plot_path, f"Match {i+1}: {match['Date']}")
    
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