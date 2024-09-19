% Clear workspace and close all figures
clear all; close all; clc;

% Read GLOV4 files (6-hour 1/10 degree surface currents, 1-30 June 2022)
filecur = 'cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i_1721738266405.nc';

% Read latitude, longitude, and velocities from the NetCDF file
lat_cur = ncread(filecur, 'latitude');
lon_cur = ncread(filecur, 'longitude');
uRad = ncread(filecur, 'uo' , [1,1,1,1],[Inf,Inf,1,60]);
vRad = ncread(filecur, 'vo' , [1,1,1,1],[Inf,Inf,1,60]);

% Read the time variable from the NetCDF file
time = ncread(filecur, 'time' , 1 , 60);

% Convert time from seconds since epoch (01-Jan-1970) to datetime
epoch = datetime(1970, 1, 1, 0, 0, 0); % Define the epoch
time = epoch + seconds(time);

% Display the dates to verify they correspond to June 2022
disp(time);

% Read the mask from the TIFF file
mask_modis = imread('masque_modis_2022.tif');

% Manually define the latitude and longitude limits of the mask (based on your knowledge of the TIFF file)
% Adjust these limits based on the actual geographical extent of your TIFF file
lat_lim_mask = [5,20];  % Replace with actual latitude limits
lon_lim_mask = [-70, -50];  % Replace with actual longitude limits

% Create a meshgrid for the mask
[lon_mask, lat_mask] = meshgrid(linspace(lon_lim_mask(1), lon_lim_mask(2), size(mask_modis, 2)), ...
                                linspace(lat_lim_mask(1), lat_lim_mask(2), size(mask_modis, 1)));

% Define the bounding box for the study area
lat_min = 15; % Minimum latitude of the study area
lat_max = 18.5; % Maximum latitude of the study area
lon_min = -63; % Minimum longitude of the study area
lon_max = -52; % Maximum longitude of the study area

% Filter the latitude and longitude indices for the study area
lat_idx = find(lat_cur >= lat_min & lat_cur <= lat_max);
lon_idx = find(lon_cur >= lon_min & lon_cur <= lon_max);

% Extract the subset of data within the study area
lat_cur = lat_cur(lat_idx);
lon_cur = lon_cur(lon_idx);
uRad = uRad(lon_idx, lat_idx, :);
vRad = vRad(lon_idx, lat_idx, :);

% Interpolate the mask onto the current grid within the study area
[lon, lat] = meshgrid(lon_cur, lat_cur);
msk = interp2(lon_mask, lat_mask, double(mask_modis), lon, lat, 'nearest');

uRad = squeeze(uRad);
uRad = permute(uRad,[2,1,3]);
vRad = squeeze(vRad);
vRad = permute(vRad,[2,1,3]);

% Save the data to a .mat file
save('U_background.mat', "time", "lat", "lon", "uRad", "vRad", "msk");

disp('U_background.mat file created successfully.');

