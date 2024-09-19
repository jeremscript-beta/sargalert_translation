% MATLAB script to reconstruct synthetic trajectories using velocity fields and plot observed and synthetic drifters

% Get the current directory
current_dir = fileparts(mfilename('fullpath'));

% List all .mat files in the directory (assumed to be drifter data)
mat_files = dir(fullfile(current_dir, 'drifter_*.mat'));

% Load the velocity field data from the .nc file
nc_file = fullfile(current_dir, 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i_1721739106099.nc');

lat_nc = ncread(nc_file, 'latitude'); % Latitude variable
lon_nc = ncread(nc_file, 'longitude'); % Longitude variable

% Read velocities from the NetCDF file
u = ncread(nc_file, 'uo', [1, 1, 1, 1], [Inf, Inf, 1, 60]);
v = ncread(nc_file, 'vo', [1, 1, 1, 1], [Inf, Inf, 1, 60]);

% Initialize figure
figure;
hold on;

% Loop through each .mat file to process drifter data
for i = 1:length(mat_files)
    % Load the drifter data
    mat_file = fullfile(current_dir, mat_files(i).name);
    data = load(mat_file);
    
    % Get the initial position of the drifter
    initial_lat = data.latim(1);
    initial_lon = data.lonim(1);
    
    % Plot the observed trajectory (use a unique marker)
    %plot(data.lonim, data.latim, 'DisplayName', ['Observed ID: ', mat_files(i).name], 'LineStyle', '-');

    % Initialize synthetic trajectory with initial position
    synthetic_lat = initial_lat;
    synthetic_lon = initial_lon;

    % Reconstruct the synthetic trajectory using the velocity field
    for t = 2:60
        % Find the nearest spatial index in the nc file
        [~, lat_idx] = min(abs(lat_nc - synthetic_lat(end)));
        [~, lon_idx] = min(abs(lon_nc - synthetic_lon(end)));
        
        % Get the velocity components at the current position and time
        u_current = u(lon_idx, lat_idx, 1, t);
        v_current = v(lon_idx, lat_idx, 1, t);
        
        % Update the position using simple Euler integration (adjust time step if necessary)
        dt = 6 * 3600; % Time step in seconds (6 hours)
        new_lat = synthetic_lat(end) + v_current * dt / 111320; % Convert to degrees (approx. 111.32 km per degree)
        new_lon = synthetic_lon(end) + u_current * dt / (111320 * cosd(synthetic_lat(end))); % Adjust for latitude
        
        % Append the new position to the synthetic trajectory
        synthetic_lat = [synthetic_lat; new_lat];
        synthetic_lon = [synthetic_lon; new_lon];
    end
    
    % Plot the synthetic trajectory with a different marker style
    plot(synthetic_lon, synthetic_lat, 'DisplayName', ['Synthetic ID: ', mat_files(i).name], 'LineStyle', '--');

    % Save the synthetic trajectory to a .mat file with a unique name
    latim = synthetic_lat;
    lonim = synthetic_lon;
    [~, name, ~] = fileparts(mat_files(i).name);
    new_name = ['synthetic_' name]; % Format name to avoid confusion with the observed data
    save(fullfile(current_dir, [new_name '.mat']), 'latim', 'lonim');
end


xlabel('Longitude');
ylabel('Latitude');
title('Observed and Synthetic Drifter Trajectories');
legend('show');
hold off;

disp('Observed and synthetic drifter trajectories plotted and saved.');