#=============================Library===================

import numpy as np
import cv2
import xarray as xr
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_dilation ,gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import fonction_utile
from tqdm import tqdm
from pyproj import Geod
from abc import ABC, abstractmethod
from pathlib import Path
import random
from matplotlib.patches import ConnectionPatch
from datetime import datetime , timedelta
import pandas as pd
import glob 
import os 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
from joblib import Parallel, delayed
import numpy as np

#==========Classe Optical Flow========= 

class Modis_Goes_Image_Processor:
    def __init__(self):
                
        #==========Parameters=================
        #Mask
        self.seuil_goes_inf = 0.0002 #goes threshold for sarg mask
        self.seuil_goes_sup = 0.1 #goes threshold for sarg mask
        
        #Extention of ROI for more information (30 => 30 pixel more in each direction)
        self.extend = 30 
        
        #Threshold for agreement between Optical flow and linear translation
        self.angle_threshold = 25 #in degrees
        self.speed_threshold = 0.2 #in % (speed x % less or more)

        #Parameter for Linear_translation class
        self.pixel_res = 1   #Resolution in km
        self.max_sarg_speed = 1 #Max speed for sargasse in m/s
        self.matches_threshold = 3 #minimum number of key points pair in each image
        
        #Time period for goes : 1st date = 1st file
        #Smoothing = > 10h à 13h et 16h à 19h
        #Time delta = > 3h
        #by shifting time you can shift the smoothing and the time delta for goes utilisation
        self.start_time_mod = '2022-05-01T10:00:00.000000000'
        self.end_time_mod = '2022-05-01T13:00:00.000000000'
        self.start_time_myd = '2022-05-01T13:00:00.000000000'
        self.end_time_myd = '2022-05-01T16:00:00.000000000'
        self.start_time_mud = '2022-05-01T16:00:00.000000000'
        self.end_time_mud = '2022-05-01T19:00:00.000000000'
        self.delta_time_goes = 3 #time difference for goes
        self.delta_time_modis = 3 #time difference for goes
        #==========Chemins====================
        path_img_mod = r"C:\Users\DELL\Desktop\goes\modis\mod\2022"
        path_img_myd = r"C:\Users\DELL\Desktop\goes\modis\myd\2022"
        path_img_goes = r"C:\Users\DELL\Desktop\goes\goes_download"
        mask_output_goes = r"C:\Users\DELL\Desktop\goes\masque_goes_2022.tif"
        mask_output_modis = r"C:\Users\DELL\Desktop\goes\masque_modis_2022.tif"
        # Chemin vers le fichier Excel contenant les données existantes
        self.excel_path = r"C:\Users\DELL\Desktop\Jupyter\Projet_Sargalert\Notebooks\points_info.xlsx"
        self.validation_goes_path = r"C:\Users\DELL\Desktop\Jupyter\Projet_Sargalert\Notebooks\points_info_goes.xlsx"
        self.validation_modis_path = r"C:\Users\DELL\Desktop\Jupyter\Projet_Sargalert\Notebooks\points_info_modis.xlsx"
        #==========Output=====================
        self.output_path = r"C:\Users\DELL\Desktop\Jupyter\Projet_Sargalert\Notebooks\plot\output_"
        #Chemin d'accès aux images
        #Utilisation de glob pour lister les fichiers .nc dans les dossiers spécifiés
        files_mod = glob.glob(os.path.join(path_img_mod, "*.nc"))
        files_myd = [os.path.join(path_img_myd, file.split('\\')[-1].replace("MOD", "MYD", 1)) for file in files_mod]
        files_goes = sorted(glob.glob(os.path.join(path_img_goes, "*.nc")))
        self.path_img_mod = files_mod
        self.path_img_myd = files_myd
        self.path_img_goes = files_goes
        self.delta_t = self.delta_time_modis #temps entre les 2 images en heures par défault (change en fonction si calcul modis ou goes)
        
        #Initialisation des listes pour l'accumulation de données pour plusieurs jours
        self.regional_vector_list_OF = []
        self.regional_vector_list_LT = []
        self.angle_liste_OF_m = []
        self.speed_liste_OF_m = []
        self.angle_liste_LT_m = []
        self.speed_liste_LT_m = []
        self.angle_liste_OF_g = []
        self.speed_liste_OF_g = []
        self.angle_liste_LT_g = []
        self.speed_liste_LT_g = []
        self.centre = []
        
        #Définition du fichier traité
        self.number_file = 0 #numéro de fichier traité actuellement : 0 => 1er fichier modis et 1er fichier goes 

        #Etat des étapes de traitement
        self.read_modis_done = False
        self.read_goes_done = False
        self.preprocess_modis_done = False
        self.preprocess_goes_done = False
        self.combined_mask_done = False
        
        # Lire le masque GOES et le stocker dans l'objet
        goes_mask = cv2.imread(mask_output_goes, cv2.IMREAD_UNCHANGED)
        if goes_mask is None:
            raise FileNotFoundError(f"GOES mask file not found: {mask_output_goes}")
        self.masque_goes = np.flipud(goes_mask)

        # Lire le masque MODIS et le stocker dans l'objet
        modis_mask = cv2.imread(mask_output_modis, cv2.IMREAD_UNCHANGED)
        if modis_mask is None:
            raise FileNotFoundError(f"MODIS mask file not found: {mask_output_modis}")
        self.masque_modis = modis_mask

    def apply_mask(self, data, mask, pixel_distance=80, pixel_distance_2 = 50, latitudes=None):
        """
        Apply a masked on a numpy array or a pandas Dataframe
        parameters :
        Pixel_distance => Masked area around the coast (1 pixel = 1km)
        pixel_distance_2 = > Masked area around island (1 pixel = 1km)
        """
        if isinstance(data, pd.DataFrame):
            latitudes = data['latitude'].values
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise TypeError("data must be a NumPy array or a pandas DataFrame")
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a NumPy array")
        if isinstance(data, np.ndarray) and data.shape != mask.shape:
            raise ValueError("data and mask must have the same shape")
        if isinstance(data, pd.DataFrame) and data.shape[0] != mask.shape[0]:
            raise ValueError("data and mask must have the same number of rows")
        if latitudes is None:
            raise ValueError("latitudes must be provided if data is a NumPy array")
        if not self.combined_mask_done :
            modified_mask = mask.copy()
            mask_12_18 = np.zeros_like(mask)
            mask_other = np.ones_like(mask)
    
            for i in range(data.shape[0]):
                lat = latitudes[i]
                if 12 <= lat <= 18:
                    mask_12_18[i, :] = 1
                else:
                    mask_other[i, :] = 1
            
            struct_size_12_18 = pixel_distance_2
            struct_size_other = pixel_distance
    
            struct_12_18 = np.ones((struct_size_12_18, struct_size_12_18))
            struct_other = np.ones((struct_size_other, struct_size_other))
    
            dilated_mask_12_18 = binary_dilation((mask == 0) & mask_12_18, structure=struct_12_18).astype(mask.dtype)
            dilated_mask_other = binary_dilation((mask == 0) & mask_other, structure=struct_other).astype(mask.dtype)
    
            combined_mask = np.where(mask_12_18, dilated_mask_12_18, dilated_mask_other)
            self.combined_mask = combined_mask
            self.combined_mask_done = True
        
        data = np.where(self.combined_mask == 1, 0, data) #np.multiply
        return data
        
    
    def read_modis(self):
        """
        Lecture des données AFAI à partir d'un fichier MODIS NetCDF présent dans le dossier aux path_file de l'objet.
        """
        annee = 2022
        mois = 5
        heure_mod = "1030"
        heure_myd = "1330"
        self.date_modis = f'{annee}{mois:02}{self.number_file+1:02}_{heure_mod}_{heure_myd}'
        number_file  = self.number_file 
        fileMod = self.path_img_mod[number_file]
        fileMyd = self.path_img_myd[number_file]
        if not (fileMod or fileMyd) :
            raise FileNotFoundError("No NetCDF files found in the specified directory.")
            
        ds_mod = xr.open_dataset(fileMod)
        afai_mod = ds_mod['AFAI_deviation']
        afai_mod_sarg = ds_mod['AFAI_deviation_sargassum']
        afai_mod = afai_mod.where(afai_mod >= 0)
        afai_mod_sarg = afai_mod_sarg.where(afai_mod_sarg >= 0)
        ds_myd = xr.open_dataset(fileMyd)
        afai_myd = ds_myd['AFAI_deviation']
        afai_myd_sarg = ds_myd['AFAI_deviation_sargassum']
        afai_myd = afai_myd.where(afai_myd >= 0)
        afai_myd_sarg = afai_myd_sarg.where(afai_myd_sarg >= 0)
        self.original_file = ds_mod,ds_myd
        self.afai_mod = afai_mod
        self.afai_sarg_mod = afai_mod_sarg
        self.afai_myd = afai_myd
        self.afai_sarg_myd = afai_myd_sarg
        self.file_mod = fileMod
        # Create a binary mask for sargassum pixels
        sargassum_mask = (afai_mod_sarg >= 0).astype(int)
        
        # Label the connected regions in the sargassum mask
        labeled_mask= label(sargassum_mask)
        
        # Create a binary mask for filaments
        filament_mask = np.zeros_like(sargassum_mask)
        
        # Iterate through each labeled region and keep only those with 50 or more pixels
        for region_num in np.unique(labeled_mask):
            if region_num == 0:
                continue  # Skip background
            region = (labeled_mask == region_num)
            if region.sum() >= 30:
                filament_mask[region] = 1
        filament_mask = filament_mask.squeeze()
        # Keep only filaments with AFAI > 0.001
        filament_mask = filament_mask & (afai_mod_sarg.squeeze() > 0.001)
        self.masque_filament = filament_mask
        
        self.read_modis_done = True

    def read_goes(self):
        annee = 2022
        mois = 5
        heure_goes_mod = "1130"
        heure_goes_myd = "1430"
        heure_goes_mud = "1730"
        self.date_goes1 = f'{annee}{mois:02}{self.number_file+1:02}_{heure_goes_mod}_{heure_goes_myd}'
        self.date_goes2 = f'{annee}{mois:02}{self.number_file+1:02}_{heure_goes_myd}_{heure_goes_mud}'
        nombre_goes = self.number_file
        files_goes = self.path_img_goes
        
        if not files_goes:
            raise FileNotFoundError("No NetCDF files found in the specified directory.")
        
        ds = xr.open_dataset(files_goes[nombre_goes])
        self.ds = ds  # Assuming you want to store the dataset as an instance variable

        # Conversion des chaînes de caractères en objets datetime
        start_time_mod_dt = datetime.fromisoformat(self.start_time_mod)
        end_time_mod_dt = datetime.fromisoformat(self.end_time_mod)
        start_time_myd_dt = datetime.fromisoformat(self.start_time_myd)
        end_time_myd_dt = datetime.fromisoformat(self.end_time_myd)
        start_time_mud_dt = datetime.fromisoformat(self.start_time_mud)
        end_time_mud_dt = datetime.fromisoformat(self.end_time_mud)
        # Définition du décalage en jours
        days_offset = timedelta(days=nombre_goes)
        
        start_time_mod_dt += days_offset
        end_time_mod_dt += days_offset
        start_time_myd_dt += days_offset
        end_time_myd_dt += days_offset
        start_time_mud_dt += days_offset
        end_time_mud_dt += days_offset
        
        # Conversion des objets datetime modifiés en chaînes de caractères
        start_time_mod = start_time_mod_dt.isoformat()
        end_time_mod = end_time_mod_dt.isoformat()
        start_time_myd = start_time_myd_dt.isoformat()
        end_time_myd = end_time_myd_dt.isoformat()
        start_time_mud = start_time_mud_dt.isoformat()
        end_time_mud = end_time_mud_dt.isoformat()
        
        # Moyenne sur la dimension temporelle pour lisser les images
        fai_mod = ds['fai_anomaly'].sel(time=slice(start_time_mod, end_time_mod)).mean(dim='time')
        fai_sarg_mod = fai_mod.where((fai_mod > self.seuil_goes_inf) & (fai_mod < self.seuil_goes_sup))
        fai_myd = ds['fai_anomaly'].sel(time=slice(start_time_myd, end_time_myd)).mean(dim='time')
        fai_sarg_myd = fai_myd.where((fai_myd > self.seuil_goes_inf ) & (fai_myd < self.seuil_goes_sup))
        fai_mud = ds['fai_anomaly'].sel(time=slice(start_time_mud, end_time_mud)).mean(dim='time')
        fai_sarg_mud = fai_mud.where((fai_mud > self.seuil_goes_inf ) & (fai_mud < self.seuil_goes_sup))
        self.fai_mod = fai_mod
        self.fai_sarg_mod = fai_sarg_mod
        self.fai_myd = fai_myd
        self.fai_sarg_myd = fai_sarg_myd
        self.fai_mud = fai_mud
        self.fai_sarg_mud = fai_sarg_mud
        self.read_goes_done = True

    def preprocessing_modis(self):
        if not self.read_modis_done :
            self.read_modis()
        self.afai_mod_np = np.nan_to_num(self.afai_mod.values).squeeze()
        self.afai_sarg_mod_np = np.nan_to_num(self.afai_sarg_mod.values).squeeze()
        self.afai_myd_np = np.nan_to_num(self.afai_myd.values).squeeze()
        self.afai_sarg_myd_np = np.nan_to_num(self.afai_sarg_myd.values).squeeze()
        threshold=0
        data_array_mod = self.afai_sarg_mod
        data_array_myd = self.afai_sarg_myd
        se3 = np.ones((3, 3), np.uint8)
        se10 = np.ones((20, 20), np.uint8)
        #Image_mod
        img_mod = data_array_mod.isel(time=0).values
        img_masked_mod = np.where(img_mod > threshold, 1, 0)
        img_eroded_mod = binary_erosion(img_masked_mod, structure=se3).astype(img_masked_mod.dtype)
        img_dilated_mod = binary_dilation(img_eroded_mod, structure=se10).astype(img_eroded_mod.dtype)
        #Image_myd
        img_myd = data_array_myd.isel(time=0).values
        img_masked_myd = np.where(img_myd > threshold, 1, 0)
        img_eroded_myd = binary_erosion(img_masked_myd, structure=se3).astype(img_masked_myd.dtype)
        img_dilated_myd = binary_dilation(img_eroded_myd, structure=se10).astype(img_eroded_myd.dtype)
        self.image_process_mod = img_dilated_mod
        self.image_process_myd = img_dilated_myd
        img_intersection = np.logical_and(self.image_process_mod,self.image_process_myd).astype(np.uint8)
        self.intersected_images = img_intersection
        labeled_img = label(self.intersected_images)
        regions = regionprops(labeled_img)
        regions_info = [{'area': region.area, 'bbox': region.bbox} for region in regions if region.area >= 200]
        # Diviser les grandes régions
        new_regions_info = []
        for region in regions_info:
            minr, minc, maxr, maxc = region['bbox']
            height = maxr - minr
            width = maxc - minc
            area = height * width
            if area < 100000 :
                new_regions_info.extend(fonction_utile.divide_region(region))
            
        regions_info = sorted(new_regions_info, key=lambda x: -x['area'])

        self.regions_sarg = regions_info
        self.preprocess_modis_done = True
        
    def preprocessing_goes(self) :
        if not self.read_goes_done :
            self.read_goes() 
        masque_goes = self.masque_goes
        #pre_process_with_square
        threshold = 0
        data_array_mod = self.fai_sarg_mod
        data_array_myd = self.fai_sarg_myd
        data_array_mud = self.fai_sarg_mud
        
        se3 = np.ones((3, 3), np.uint8)
        se10 = np.ones((20, 20), np.uint8)
        # Extract latitudes from the dataset
        latitudes = data_array_mod['latitude'].values
        results = Parallel(n_jobs=-1)(delayed(self.apply_mask)(
            data_array.values, masque_goes, latitudes=latitudes) for data_array in [data_array_mod, data_array_myd,data_array_mud])

        img_mod, img_myd ,img_mud = results
        #Image_mod
        img_masked_mod = np.where(img_mod > threshold, 1, 0)
        img_eroded_mod = binary_erosion(img_masked_mod, structure=se3).astype(img_masked_mod.dtype)
        img_dilated_mod = binary_dilation(img_eroded_mod, structure=se10).astype(img_eroded_mod.dtype)
        #Image_myd
        img_masked_myd = np.where(img_myd > threshold, 1, 0)
        img_eroded_myd = binary_erosion(img_masked_myd, structure=se3).astype(img_masked_myd.dtype)
        img_dilated_myd = binary_dilation(img_eroded_myd, structure=se10).astype(img_eroded_myd.dtype)
        #Image_mud
        img_masked_mud = np.where(img_mud > threshold, 1, 0)
        img_eroded_mud = binary_erosion(img_masked_mud, structure=se3).astype(img_masked_mud.dtype)
        img_dilated_mud = binary_dilation(img_eroded_mud, structure=se10).astype(img_eroded_mud.dtype)

        #Intersect
        img_intersection = np.logical_and(img_dilated_mod,img_dilated_myd).astype(np.uint8)
        img_intersection2 = np.logical_and(img_dilated_myd,img_dilated_mud).astype(np.uint8)
        
        #Detection sargasse 1er période
        labeled_img = label(img_intersection)
        regions = regionprops(labeled_img)
        regions_info = [{'area': region.area, 'bbox': region.bbox} for region in regions if region.area >= 200]
        # Vérifier les dimensions et diviser les grandes régions
        # Diviser les grandes régions
        new_regions_info = []
        for region in regions_info:
            minr, minc, maxr, maxc = region['bbox']
            height = maxr - minr
            width = maxc - minc
            area = height * width
            if area < 100000 :
                new_regions_info.extend(fonction_utile.divide_region(region))
            
        regions_info1 = sorted(new_regions_info, key=lambda x: -x['area'])
        
        #Detection sargasse 2nd période
        labeled_img = label(img_intersection2)
        regions = regionprops(labeled_img)
        regions_info = [{'area': region.area, 'bbox': region.bbox} for region in regions if region.area >= 200]
        # Vérifier les dimensions et diviser les grandes régions
        # Diviser les grandes régions
        new_regions_info = []
        for region in regions_info:
            minr, minc, maxr, maxc = region['bbox']
            height = maxr - minr
            width = maxc - minc
            area = height * width
            if area < 100000 :
                new_regions_info.extend(fonction_utile.divide_region(region))
            
        regions_info2 = sorted(new_regions_info, key=lambda x: -x['area'])
        self.regions_sarg_goes = regions_info1
        self.regions_sarg_goes2 = regions_info2
         # Chargement numpy avec vérifications et impressions
        fai_mod_np = np.asarray(np.nan_to_num(self.fai_mod.values.squeeze()))
        fai_sarg_mod_np = np.asarray(np.nan_to_num(self.fai_sarg_mod.values.squeeze()))
        fai_myd_np = np.asarray(np.nan_to_num(self.fai_myd.values.squeeze()))
        fai_sarg_myd_np = np.asarray(np.nan_to_num(self.fai_sarg_myd.values.squeeze()))
        fai_mud_np = np.asarray(np.nan_to_num(self.fai_mud.values.squeeze()))
        fai_sarg_mud_np = np.asarray(np.nan_to_num(self.fai_sarg_mud.values.squeeze()))
        
        results = Parallel(n_jobs=-1)(delayed(self.apply_mask)(
            data_array, masque_goes, latitudes=latitudes) for data_array in [fai_mod_np,fai_sarg_mod_np,fai_myd_np,fai_sarg_myd_np,fai_mud_np,fai_sarg_mud_np])

        self.fai_mod_np , self.fai_sarg_mod_np , self.fai_myd_np , self.fai_sarg_myd_np, self.fai_mud_np , self.fai_sarg_mud_np = results
        self.preprocess_goes_done = True

class Modis_Goes_Translation(Modis_Goes_Image_Processor):
    
    def __init__(self):
        # Appel du constructeur de la classe parente pour conserver l'initialisation existante
        super().__init__()
        # Ajout de nouvelles propriétés spécifiques à OpticalFlowTracker ici si besoin

    def calculate_optical_flow_region(self, img1, img2, pyr_scale=None, levels=None, winsize=None, iterations=None, poly_n=None, poly_sigma=None):
        """
        Args:
            - img1 : np.ndarray, la première image en niveaux de gris et en format float32. Représente l'état initial pour le calcul du flux optique.
            - img2 : np.ndarray, la deuxième image en niveaux de gris et en format float32. Représente l'état final pour le calcul du flux optique.
            - pyr_scale : float, facteur de réduction de l'échelle de l'image dans la construction de la pyramide. Valeurs typiques sont entre 0 et 1.
            - levels : int, nombre de niveaux dans la pyramide. Plus le nombre est élevé, plus les détails fins sont capturés.
            - winsize : int, taille de la fenêtre d'analyse. Plus la taille est grande, plus la capacité de capturer les mouvements rapides est élevée, mais cela réduit la précision locale.
            - iterations : int, nombre d'itérations à chaque niveau de la pyramide. Plus le nombre est élevé, plus le calcul est précis.
            - poly_n : int, taille de la fenêtre du polynôme utilisé pour estimer le flux optique. Habituellement, 5 ou 7.
            - poly_sigma : float, écart-type de la gaussienne utilisée pour lisser les dérivées utilisées dans le calcul du flux optique.
        Calcul du flux optique entre deux images en utilisant l'algorithme de Farneback.
        """
        
        # Utiliser les valeurs optimisées si l'optimisation a été faite et aucun paramètre spécifique n'a été fourni
        if getattr(self, 'optimization_done', False):
            pyr_scale = pyr_scale if pyr_scale is not None else self.optimization_param.get('pyr_scale', 0.1)
            levels = levels if levels is not None else self.optimization_param.get('levels', 3)
            winsize = winsize if winsize is not None else self.optimization_param.get('winsize', 18)
            iterations = iterations if iterations is not None else self.optimization_param.get('iterations', 3)
            poly_n = poly_n if poly_n is not None else self.optimization_param.get('poly_n', 5)
            poly_sigma = poly_sigma if poly_sigma is not None else self.optimization_param.get('poly_sigma', 1.1)
        else:
            # Utiliser les valeurs par défaut si aucune valeur n'est fournie et l'optimisation n'a pas été faite
            pyr_scale = pyr_scale if pyr_scale is not None else 0.1
            levels = levels if levels is not None else 3
            winsize = winsize if winsize is not None else 18
            iterations = iterations if iterations is not None else 5
            poly_n = poly_n if poly_n is not None else 5
            poly_sigma = poly_sigma if poly_sigma is not None else 1.1
        
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        img1_norm = cv2.normalize(img1_float, None, 0, 255, cv2.NORM_MINMAX)
        img2_norm = cv2.normalize(img2_float, None, 0, 255, cv2.NORM_MINMAX)
        #img1_norm = img1_float
        #img2_norm = img2_float
        # Utiliser directement les variables qui ont été ajustées
        flow = cv2.calcOpticalFlowFarneback(img1_norm, img2_norm, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
        return flow

    def calculate_linear_translation_region(self , roi_mod ,roi_sarg_mod, roi_myd, roi_sarg_myd):
        """
        Calcul de la translation linéaire réalisé dans la ROI concerné avec l'algorithme SIFT
        """
        #Calcul linear translation
        plot_path = self.path_img_mod
        lats = self.afai_mod['latitude'].values
        lons = self.afai_mod['longitude'].values
        lat_lon_data = LatLonData(lats,lons)
        # Supposons que votre date soit sous forme de chaîne de caractères '2023-06-01'
        date_str = "2023-06-01"
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        LT = LinearTranslation(AFAI_src_dev=roi_mod, 
                                       AFAI_src_sarg=roi_sarg_mod,  # Assumons None pour simplifier
                                       AFAI_dst_dev=roi_myd, 
                                       AFAI_dst_sarg=roi_sarg_myd,  # Assumons None pour simplifier
                                       lat_lon_data=lat_lon_data, 
                                       plot_path=plot_path, 
                                       date=date, 
                                       drifter_drift=None,  # Assumons None pour simplifier
                                       wind_drift=None,
                                       delta_t = self.delta_t,
                                       pixel_res=self.pixel_res,
                                       max_sarg_speed =self.max_sarg_speed,
                                       matches_threshold=self.matches_threshold)  # Assumons None pour simplifier

        LT.resize(scale_percent=100)
        LT.match()
        matches_final = LT.final_matches 
        n_matches = LT.nb_matching_points 
        return matches_final,n_matches

    def validation_goes_modis(self) :
        # Load the data
        file_path = self.validation_goes_path 
        data = pd.read_excel(file_path)
        
        # Extract coordinates and vectors
        lats = data['Mod_Lat_m'].values
        lons = data['Mod_Lon_m'].values
        
        # Vectors
        u_m = data['u_m'].values
        v_m = data['v_m'].values
        u_OF = data['u_OF'].values
        v_OF = data['v_OF'].values
        u_LT = data['u_LT'].values
        v_LT = data['v_LT'].values

        self.read_goes()
        self.lat_lon_data = LatLonData(self.fai_mod['latitude'].values,self.fai_mod['longitude'].values)
        lat_lon_data = self.lat_lon_data
        lats_goes = self.fai_mod['latitude'].values
        lons_goes = self.fai_mod['longitude'].values
        # Initialise une figure avec projection cartographique
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        #ax.set_extent([lons_min, lons_max, lats_min, lats_max], crs=ccrs.PlateCarree())
        ax.set_extent([lons_goes.min()-1, lons_goes.max()+1, lats_goes.min()-1, lats_goes.max()+1], crs=ccrs.PlateCarree())
        # Ajouter les caractéristiques de la carte
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Plot the vectors
        for lat, lon, um, vm, uof, vof, ult, vlt in zip(lats, lons, u_m, v_m, u_OF, v_OF, u_LT, v_LT):
            if um != 0 or vm != 0:
                ax.quiver(lon, lat, um, vm, transform=ccrs.PlateCarree(), color='purple', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
            if uof != 0 or vof != 0:
                ax.quiver(lon, lat, uof, vof, transform=ccrs.PlateCarree(), color='green', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
            if ult != 0 or vlt != 0:
                ax.quiver(lon, lat, ult, vlt, transform=ccrs.PlateCarree(), color='red', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        
        # Add legends
        legend_of = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Optical Flow')
        legend_lt = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Linear Translation')
        legend_obs = mlines.Line2D([], [], color='purple', marker='_', linestyle='None', markersize=10, label='Observation')
        ax.legend(handles=[legend_of, legend_lt, legend_obs], loc='upper left')
        
        # Ajouter un vecteur d'échelle pour 1 m/s dans le coin en bas à droite à 69°W et 6°N
        scale_speed = 1
        ax.quiver(-69, 6, scale_speed, 0, transform=ccrs.PlateCarree(),
                  color='black', scale=20 , headwidth=1, headlength=0.5, headaxislength=0.5)
        # Ajouter un texte d'échelle au-dessus de la flèche d'échelle
        ax.text(-69, 6.1, 'Échelle: 1 m/s', transform=ccrs.PlateCarree(),
        verticalalignment='bottom', horizontalalignment='left')
        
        # Finalize and show the map
        plt.title("Comparaison des vecteurs Optical Flow (vert) et Linear Translation (rouge)")
        plt.savefig("observation_vectors_map_obs.png", bbox_inches='tight', dpi=300)
        plt.show()
        #===================MODIS=============
        # Load the data
        file_path = self.validation_modis_path 
        data = pd.read_excel(file_path)
        
        # Extract coordinates and vectors
        lats = data['Mod_Lat_m'].values
        lons = data['Mod_Lon_m'].values
        
        # Vectors
        u_m = data['u_m'].values
        v_m = data['v_m'].values
        u_OF = data['u_OF'].values
        v_OF = data['v_OF'].values
        u_LT = data['u_LT'].values
        v_LT = data['v_LT'].values

        self.read_modis()
        self.lat_lon_data = LatLonData(self.afai_mod['latitude'].values,self.afai_mod['longitude'].values)
        lat_lon_data = self.lat_lon_data
        lats_goes = self.afai_mod['latitude'].values
        lons_goes = self.afai_mod['longitude'].values
        # Initialise une figure avec projection cartographique
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        #ax.set_extent([lons_min, lons_max, lats_min, lats_max], crs=ccrs.PlateCarree())
        ax.set_extent([lons_goes.min()-1, lons_goes.max()+1, lats_goes.min()-1, lats_goes.max()+1], crs=ccrs.PlateCarree())
        # Ajouter les caractéristiques de la carte
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Plot the vectors
        for lat, lon, um, vm, uof, vof, ult, vlt in zip(lats, lons, u_m, v_m, u_OF, v_OF, u_LT, v_LT):
            if um != 0 or vm != 0:
                ax.quiver(lon, lat, um, vm, transform=ccrs.PlateCarree(), color='purple', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
            if uof != 0 or vof != 0:
                ax.quiver(lon, lat, uof, vof, transform=ccrs.PlateCarree(), color='green', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
            if ult != 0 or vlt != 0:
                ax.quiver(lon, lat, ult, vlt, transform=ccrs.PlateCarree(), color='red', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        
        # Add legends
        legend_of = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Optical Flow')
        legend_lt = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Linear Translation')
        legend_obs = mlines.Line2D([], [], color='purple', marker='_', linestyle='None', markersize=10, label='Observation')
        ax.legend(handles=[legend_of, legend_lt, legend_obs], loc='upper left')
        
        # Ajouter un vecteur d'échelle pour 1 m/s dans le coin en bas à droite à 69°W et 6°N
        scale_speed = 1
        ax.quiver(-69, 6, scale_speed, 0, transform=ccrs.PlateCarree(),
                  color='black', scale=20 , headwidth=1, headlength=0.5, headaxislength=0.5)
        # Ajouter un texte d'échelle au-dessus de la flèche d'échelle
        ax.text(-69, 6.1, 'Échelle: 1 m/s', transform=ccrs.PlateCarree(),
        verticalalignment='bottom', horizontalalignment='left')
        
        # Finalize and show the map
        plt.title("Comparaison des vecteurs Optical Flow (vert) et Linear Translation (rouge)")
        plt.savefig("observation_vectors_map_obs_modis.png", bbox_inches='tight', dpi=300)
        plt.show()   

    def plot_global_map(self):
        file_path = self.excel_path
        self.number_file = 5
        self.read_modis()
         # Load the data
        data = pd.read_excel(file_path)
    
        # Extract coordinates and vectors
        lats = data['Mod_Lat'].values
        lons = data['Mod_Lon'].values
        u_OF = data['u_OF'].values
        v_OF = data['v_OF'].values
        u_LT = data['u_LT'].values
        v_LT = data['v_LT'].values
        
        # Initialise une figure avec projection cartographique
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([-70, -50, 5, 20], crs=ccrs.PlateCarree())
        
        # Ajouter les caractéristiques de la carte
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Plot the filaments
        if hasattr(self, 'masque_filament'):
            lons_filaments, lats_filaments = np.meshgrid(self.afai_mod['longitude'].values, self.afai_mod['latitude'].values)
            ax.pcolormesh(lons_filaments, lats_filaments, self.masque_filament, transform=ccrs.PlateCarree(), cmap='Greens', alpha=0.5, shading='auto')
        
        # Plot the vectors
        for lat, lon, uof, vof, ult, vlt in zip(lats, lons, u_OF, v_OF, u_LT, v_LT):
            if uof != 0 or vof != 0:
                ax.quiver(lon, lat, uof, vof, transform=ccrs.PlateCarree(), color='green', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
            if ult != 0 or vlt != 0:
                ax.quiver(lon, lat, ult, vlt, transform=ccrs.PlateCarree(), color='red', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        
        # Add legends
        legend_of = mlines.Line2D([], [], color='green', marker='_', linestyle='None', markersize=10, label='Optical Flow')
        legend_lt = mlines.Line2D([], [], color='red', marker='_', linestyle='None', markersize=10, label='Linear Translation')
        ax.legend(handles=[legend_of, legend_lt], loc='upper left')
        
        # Ajouter un vecteur d'échelle pour 0.5 m/s dans le coin en bas à gauche à 69°W et 6°N
        scale_speed = 0.5
        ax.quiver(-69, 6, scale_speed, 0, transform=ccrs.PlateCarree(),
                  color='black', scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        # Ajouter un texte d'échelle juste au-dessus de la flèche d'échelle
        ax.text(-69, 6.1, 'Échelle: 0.5 m/s', transform=ccrs.PlateCarree(),
                verticalalignment='bottom', horizontalalignment='left')
        
        # Finalize and show the map
        plt.title("Comparaison des vecteurs Optical Flow (vert) et Linear Translation (rouge)")
        plt.savefig("displacement_vectors_map_with_filaments.png", bbox_inches='tight', dpi=300)
        plt.show()
        
    def goes_modis_translation(self , OF = False , LT = True , Modis = True , Goes = False) :
        extend = self.extend
        angle_threshold = self.angle_threshold 
        speed_threshold = self.speed_threshold 
        if not Modis and not Goes :
            print("choose goes or/and modis")
            return
        if not OF and not LT :
            print("choose OF or/and LT")
            return
        #On initialise les listes des vecteurs moyens pour chaques ROI
        #Vecteurs (x,y)
        regional_vector_list_OF = []
        regional_vector_list_LT = []
        
        #Orientations_modis
        angle_liste_OF_m = []
        angle_liste_LT_m = []
        
        #Orientations goes
        angle_liste_OF_g = []
        angle_liste_LT_g = []
        
        #Vitesses_goes
        speed_liste_OF_g = []
        speed_liste_LT_g = []
        
        #Vitesses_modis
        speed_liste_OF_m = []
        speed_liste_LT_m = []

        #Centre
        centre_liste = []

        #Point_info_exlsx
        new_data = []
        excel_path = self.excel_path
        base_path = self.output_path
        plot_path = base_path + f"day{self.number_file}_modis_goes_translation.pdf"
        #Read image
        if Modis :
            self.read_modis()
            #Prétraitement Modis
            self.preprocessing_modis()
        if Goes :
            self.read_goes()
            #Prétraitement Goes
            self.preprocessing_goes()
        # Tentative de charger les points existants
        try:
            df_existing = pd.read_excel(excel_path)
            print("Loaded existing data points.")
        except Exception as e:
            print(f"No existing data points: {e}")
            df_existing = pd.DataFrame(columns=['Date','Mod_Lat', 'Mod_Lon', 'Myd_Lat', 'Myd_Lon',"u_OF","v_OF","u_LT","v_LT","Sat"])

            # Fonction pour vérifier si des données similaires existent déjà
        def is_data_existing(date, sat, lat, lon, df_existing, threshold=0.2):
            existing = df_existing[(df_existing['Date'] == date) & (df_existing['Sat'] == sat)]
            if not existing.empty:
                for _, row in existing.iterrows():
                    if np.abs(row['Mod_Lat'] - lat) <= threshold and np.abs(row['Mod_Lon'] - lon) <= threshold:
                        return True
            return False
        #Creation pdf et des graphiques :
        with PdfPages(plot_path) as pdf:
            if Goes : 
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(self.fai_sarg_mod_np, cmap='viridis', origin='lower',vmin=0, vmax=0.001)
                for region in self.regions_sarg_goes:
                    minr, minc, maxr, maxc = region['bbox']
                    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=2)
                    ax.add_patch(rect)
                ax.set_title(f'régions de sargasse détectées goes')
                pdf.savefig(fig)
                plt.close(fig)
            if Modis :
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(self.afai_sarg_mod_np, cmap='viridis', origin='lower',vmin=0, vmax=0.001)
                for region in self.regions_sarg:
                    minr, minc, maxr, maxc = region['bbox']
                    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=2)
                    ax.add_patch(rect)
                ax.set_title(f'régions de sargasse détectées modis')
                ax.invert_yaxis()  # Inverser l'axe des y si nécessaire
                pdf.savefig(fig)
                plt.close(fig)
            #Graphique_goes :
            if Goes :
                for region in tqdm(self.regions_sarg_goes):
                    self.delta_t = self.delta_time_goes
                    #On réinitialise la variable permettant d'identifier le faisabilité de LT
                    LT_impossible = False
                    
                    #===============Définition des bords de la zone étudiée=================
                    minr, minc, maxr, maxc = region['bbox']
                    if extend != 0 :
                        minr = max(0, minr - extend)  # S'assurer que minr n'est pas négatif
                        minc = max(0, minc - extend)  # S'assurer que minc n'est pas négatif
                        maxr = min(self.fai_mod_np.shape[0], maxr + extend)  # Ne pas dépasser la hauteur de l'image
                        maxc = min(self.fai_mod_np.shape[1], maxc + extend)  # Ne pas dépasser la largeur de l'image
                    # Extraire les ROIs pour afai_mod, afai_myd, afai_sarg_mod et afai_sarg_myd
                    roi_mod = self.fai_mod_np[minr:maxr, minc:maxc]
                    roi_myd = self.fai_myd_np[minr:maxr, minc:maxc]
                    roi_sarg_mod = self.fai_sarg_mod_np[minr:maxr, minc:maxc]
                    roi_sarg_myd = self.fai_sarg_myd_np[minr:maxr, minc:maxc]
                    #Ajout centre
                    # Calculer le centre de la ROI
                    centre_x = (minc + maxc) // 2
                    centre_y = (minr + maxr) // 2
                    centre_liste.append((centre_x,centre_y))
                    
                    # ============= Calcul du flux optique et linear_translation pour la ROI=======
                    flow = self.calculate_optical_flow_region(roi_mod, roi_myd)
                    # Calcul de la linear translation pour la même ROI
                    matches, nb_matches = self.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
                    #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                    flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                    # Calculer le vecteur moyen pour le flux optique
                    mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                    if matches and len(matches) > 0:
                        #================Représentation graphique=================
                        #================Optical flow================
                        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                        X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                        step = 6  # Contrôle la densité des flèches
                        # Optical flow - MOD
                        ax[0, 0].imshow(roi_sarg_mod, cmap='viridis', origin='lower')
                        ax[0, 0].set_title('Optical Flow - MOD')
                        ax[0, 0].arrow(roi_mod.shape[1] // 2, roi_mod.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                        ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow_masked[::step, ::step, 0], flow_masked[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                         # Optical flow - MYD
                        ax[0, 1].imshow(roi_sarg_myd, cmap='viridis', origin='lower')
                        ax[0, 1].set_title('Optical Flow - MYD')
                        ax[0, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                        ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow_masked[::step, ::step, 0], flow_masked[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                        angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                        angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                        if angle_degrees_OF < 0 :
                            angle_degrees_OF += 360
                        speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                        ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
        
                        #================Linear Translation================
                        dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                        dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                        dx_mean = sum(dxs) / len(dxs)
                        dy_mean = sum(dys) / len(dys)
                        # Linear Translation - MOD - MYD
                        ax[1, 0].imshow(roi_sarg_mod, cmap='viridis', origin='lower')
                        ax[1, 0].set_title('Linear Translation - MOD')
                        ax[1, 1].imshow(roi_sarg_myd, cmap='viridis', origin='lower')
                        ax[1, 1].set_title('Linear Translation - MYD')
                        ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                         # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                        for pt_src, pt_dst in matches:
                            # Tracer un cercle sur le point source
                            ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                            # Tracer un cercle sur le point destination
                            ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                            # Créer une connexion entre les points source et destination
                            con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                      axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                            ax[1, 1].add_artist(con)
                            angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                            angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                            if angle_degrees_LT < 0 :
                                angle_degrees_LT += 360
                            speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                            ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                        pdf.savefig(fig)  # Save the current figure into a PDF page
                        plt.close(fig)
                        
                    else :
                        LT_impossible = True
                    write = True
                    #============Vérification si on garde le vecteur pour la carte===========
                    if LT_impossible or speed_LT == 0 :
                        #SI pas de LT possible => pas de prédiction
                        mean_flow_x, mean_flow_y = 0,0
                        dx_mean,dy_mean = 0,0
                        write = False
                    elif np.abs(angle_degrees_OF - angle_degrees_LT) > angle_threshold :
                        #Si pas cohérent on supprime => Utilisation OF pour vérification 
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        write = False
                    elif not (1-speed_threshold < (speed_OF / speed_LT) < 1+speed_threshold):
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        write = False
                    else :
                        angle_liste_OF_g.append(angle_degrees_OF)
                        speed_liste_OF_g.append(speed_OF)
                        angle_liste_LT_g.append(angle_degrees_LT)
                        speed_liste_LT_g.append(speed_LT)
                    mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                    dx_mean,dy_mean = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                    #List des vecteurs moyens pour chaque région en m/s
                    regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"goes"))
                    regional_vector_list_LT.append((dx_mean,dy_mean,"goes"))
                    if write : 
                        lat_lon_data = LatLonData(self.fai_mod['latitude'].values,self.fai_mod['longitude'].values)
                        date = self.date_goes1
                        centre_mod = fonction_utile.calculate_barycenter(matches[0])
                        centre_mod = (centre_mod[0] + minc, centre_mod[1] + minr)
                        centre_myd = fonction_utile.calculate_barycenter(matches[1])
                        centre_myd = (centre_myd[0] + minc, centre_myd[1] + minr)
                        Mod_Lat, Mod_Lon = lat_lon_data.get_lat_lon_coordinates(centre_mod[1],centre_mod[0])
                        Myd_Lat, Myd_Lon = lat_lon_data.get_lat_lon_coordinates(centre_myd[1],centre_myd[0])
                        u_OF = mean_flow_x
                        v_OF = mean_flow_y
                        u_LT = dx_mean
                        v_LT = dy_mean
                        sat = "Goes"
                        if not is_data_existing(date, sat, Mod_Lat, Mod_Lon, df_existing) :
                            new_data.append([date,Mod_Lat, Mod_Lon, Myd_Lat, Myd_Lon,u_OF,v_OF,u_LT,v_LT,sat])
                    
                for region in tqdm(self.regions_sarg_goes2):
                    #On réinitialise la variable permettant d'identifier le faisabilité de LT
                    LT_impossible = False
                    
                    #===============Définition des bords de la zone étudiée=================
                    minr, minc, maxr, maxc = region['bbox']
                    if extend != 0 :
                        minr = max(0, minr - extend)  # S'assurer que minr n'est pas négatif
                        minc = max(0, minc - extend)  # S'assurer que minc n'est pas négatif
                        maxr = min(self.fai_myd_np.shape[0], maxr + extend)  # Ne pas dépasser la hauteur de l'image
                        maxc = min(self.fai_myd_np.shape[1], maxc + extend)  # Ne pas dépasser la largeur de l'image
                    # Extraire les ROIs pour afai_mod, afai_myd, afai_sarg_mod et afai_sarg_myd
                    roi_mod = self.fai_myd_np[minr:maxr, minc:maxc]
                    roi_myd = self.fai_mud_np[minr:maxr, minc:maxc]
                    roi_sarg_mod = self.fai_sarg_myd_np[minr:maxr, minc:maxc]
                    roi_sarg_myd = self.fai_sarg_mud_np[minr:maxr, minc:maxc]
                    #Ajout centre
                    # Calculer le centre de la ROI
                    centre_x = (minc + maxc) // 2
                    centre_y = (minr + maxr) // 2
                    centre_liste.append((centre_x,centre_y))
                    
                    # ============= Calcul du flux optique et linear_translation pour la ROI=======
                    flow = self.calculate_optical_flow_region(roi_mod, roi_myd)
                    # Calcul de la linear translation pour la même ROI
                    matches, nb_matches = self.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
                    #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                    flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                    # Calculer le vecteur moyen pour le flux optique
                    mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                    if matches and len(matches) > 0:
                        #================Représentation graphique=================
                        #================Optical flow================
                        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                        X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                        step = 6  # Contrôle la densité des flèches
                        # Optical flow - MOD
                        ax[0, 0].imshow(roi_sarg_mod, cmap='viridis', origin='lower')
                        ax[0, 0].set_title('Optical Flow - MOD')
                        ax[0, 0].arrow(roi_mod.shape[1] // 2, roi_mod.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                        ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow_masked[::step, ::step, 0], flow_masked[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                         # Optical flow - MYD
                        ax[0, 1].imshow(roi_sarg_myd, cmap='viridis', origin='lower')
                        ax[0, 1].set_title('Optical Flow - MYD')
                        ax[0, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                        ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow_masked[::step, ::step, 0], flow_masked[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                        angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                        angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                        if angle_degrees_OF < 0 :
                            angle_degrees_OF += 360
                        speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                        ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
        
                        #================Linear Translation================
                        dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                        dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                        dx_mean = sum(dxs) / len(dxs)
                        dy_mean = sum(dys) / len(dys)
                        # Linear Translation - MOD - MYD
                        ax[1, 0].imshow(roi_sarg_mod, cmap='viridis', origin='lower')
                        ax[1, 0].set_title('Linear Translation - MOD')
                        ax[1, 1].imshow(roi_sarg_myd, cmap='viridis', origin='lower')
                        ax[1, 1].set_title('Linear Translation - MYD')
                        ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                         # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                        for pt_src, pt_dst in matches:
                            # Tracer un cercle sur le point source
                            ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                            # Tracer un cercle sur le point destination
                            ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                            # Créer une connexion entre les points source et destination
                            con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                      axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                            ax[1, 1].add_artist(con)
                            angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                            angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                            if angle_degrees_LT < 0 :
                                angle_degrees_LT += 360
                            speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                            ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                        pdf.savefig(fig)  # Save the current figure into a PDF page
                        plt.close(fig)
                        
                    else :
                        LT_impossible = True
                    write = True
                    #============Vérification si on garde le vecteur pour la carte===========
                    if LT_impossible or speed_LT == 0 :
                        #SI pas de LT possible => pas de prédiction
                        mean_flow_x, mean_flow_y = 0,0
                        dx_mean,dy_mean = 0,0
                        write = False
                    elif np.abs(angle_degrees_OF - angle_degrees_LT) > angle_threshold :
                        #Si pas cohérent on supprime => Utilisation OF pour vérification 
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        write = False
                    elif not (1-speed_threshold < (speed_OF / speed_LT) < 1+speed_threshold):
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        write = False
                    else :
                        angle_liste_OF_g.append(angle_degrees_OF)
                        speed_liste_OF_g.append(speed_OF)
                        angle_liste_LT_g.append(angle_degrees_LT)
                        speed_liste_LT_g.append(speed_LT)
                    mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                    dx_mean,dy_mean = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                    #List des vecteurs moyens pour chaque région en m/s
                    regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"goes"))
                    regional_vector_list_LT.append((dx_mean,dy_mean,"goes"))
                    if write : 
                        lat_lon_data = LatLonData(self.fai_mod['latitude'].values,self.fai_mod['longitude'].values)
                        date = self.date_goes2
                        centre_mod = fonction_utile.calculate_barycenter(matches[0])
                        centre_mod = (centre_mod[0] + minc, centre_mod[1] + minr)
                        centre_myd = fonction_utile.calculate_barycenter(matches[1])
                        centre_myd = (centre_myd[0] + minc, centre_myd[1] + minr)
                        Mod_Lat, Mod_Lon = lat_lon_data.get_lat_lon_coordinates(centre_mod[1],centre_mod[0])
                        Myd_Lat, Myd_Lon = lat_lon_data.get_lat_lon_coordinates(centre_myd[1],centre_myd[0])
                        u_OF = mean_flow_x
                        v_OF = mean_flow_y
                        u_LT = dx_mean
                        v_LT = dy_mean
                        sat = "Goes"
                        if not is_data_existing(date, sat, Mod_Lat, Mod_Lon, df_existing) :
                            new_data.append([date,Mod_Lat, Mod_Lon, Myd_Lat, Myd_Lon,u_OF,v_OF,u_LT,v_LT,sat])
                self.angle_liste_OF_g = self.angle_liste_OF_g + angle_liste_OF_g
                self.speed_liste_OF_g = self.speed_liste_OF_g + speed_liste_OF_g
                self.angle_liste_LT_g = self.angle_liste_LT_g + angle_liste_LT_g
                self.speed_liste_LT_g = self.speed_liste_LT_g + speed_liste_LT_g
                if Modis :
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(self.afai_mod_np, cmap='viridis', origin='lower')
                    for region in self.regions_sarg:
                        minr, minc, maxr, maxc = region['bbox']
                        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=2)
                        ax.add_patch(rect)
                    ax.set_title(f'Nouvelles régions de sargasse détectées modis')
                    pdf.savefig(fig)
                    plt.close(fig)
                lengh_goes = len(regional_vector_list_OF)

            if Modis :
                for region in tqdm(self.regions_sarg):
                    self.delta_t = self.delta_time_modis
                    #On réinitialise la variable permettant d'identifier le faisabilité de LT
                    LT_impossible = False
                    
                    #===============Définition des bords de la zone étudiée=================
                    minr, minc, maxr, maxc = region['bbox']
                    if extend != 0 :
                        minr = max(0, minr - extend)  # S'assurer que minr n'est pas négatif
                        minc = max(0, minc - extend)  # S'assurer que minc n'est pas négatif
                        maxr = min(self.afai_mod_np.shape[0], maxr + extend)  # Ne pas dépasser la hauteur de l'image
                        maxc = min(self.afai_mod_np.shape[1], maxc + extend)  # Ne pas dépasser la largeur de l'image
                    # Extraire les ROIs pour afai_mod, afai_myd, afai_sarg_mod et afai_sarg_myd
                    roi_mod = self.afai_mod_np[minr:maxr, minc:maxc]
                    roi_myd = self.afai_myd_np[minr:maxr, minc:maxc]
                    roi_sarg_mod = self.afai_sarg_mod_np[minr:maxr, minc:maxc]
                    roi_sarg_myd = self.afai_sarg_myd_np[minr:maxr, minc:maxc]
                    
                    #Ajout centre
                    # Calculer le centre de la ROI
                    centre_x = (minc + maxc) // 2
                    centre_y = (minr + maxr) // 2
                    centre_liste.append((centre_x,centre_y))
                    # ============= Calcul du flux optique et linear_translation pour la ROI=======
                    flow = self.calculate_optical_flow_region(roi_mod, roi_myd)
                    # Calcul de la linear translation pour la même ROI
                    matches, nb_matches = self.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
                    #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                    flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                    # Calculer le vecteur moyen pour le flux optique
                    mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                    
                    #================Représentation graphique=================
                    #================Optical flow================
                    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                    X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                    step = 6  # Contrôle la densité des flèches
                    # Optical flow - MOD
                    ax[0, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                    ax[0, 0].set_title('Optical Flow - MOD')
                    ax[0, 0].arrow(roi_mod.shape[1] // 2, roi_mod.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                    ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                     # Optical flow - MYD
                    ax[0, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                    ax[0, 1].set_title('Optical Flow - MYD')
                    ax[0, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                    ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                    angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                    angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                    if angle_degrees_OF < 0 :
                        angle_degrees_OF += 360
                    speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                    ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
        
                    #================Linear Translation================
                    if matches and len(matches) > 1:
                        dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                        dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                        dx_mean = sum(dxs) / len(dxs)
                        dy_mean = sum(dys) / len(dys)
                        # Linear Translation - MOD - MYD
                        ax[1, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                        ax[1, 0].set_title('Linear Translation - MOD')
                        ax[1, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                        ax[1, 1].set_title('Linear Translation - MYD')
                        ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                         # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                        for pt_src, pt_dst in matches:
                            # Tracer un cercle sur le point source
                            ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                            # Tracer un cercle sur le point destination
                            ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                            # Créer une connexion entre les points source et destination
                            con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                      axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                            ax[1, 1].add_artist(con)
                            angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                            angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                            if angle_degrees_LT < 0 :
                                angle_degrees_LT += 360
                            speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                            ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                    else : 
                         #Afficher un message indiquant qu'une translation linéaire est impossible
                        text_props = dict(ha='center', va='center', fontsize=12, color='black')
                        ax[1, 0].clear()  # Nettoyer l'axe si nécessaire
                        ax[1, 0].text(0.5, 0.5, "Translation linéaire impossible\nsur cette région", transform=ax[1, 0].transAxes, **text_props)
                        ax[1, 1].clear()  # Nettoyer l'axe si nécessaire
                        ax[1, 1].text(0.5, 0.5, "Translation linéaire impossible\nsur cette région", transform=ax[1, 1].transAxes, **text_props)
                        LT_impossible = True
                    pdf.savefig(fig)  # Save the current figure into a PDF page
                    plt.close(fig)
                    write = True
                    #============Vérification si on garde le vecteur pour la carte===========
                    if LT_impossible or speed_LT == 0 :
                        #SI pas de LT possible => pas de prédiction
                        mean_flow_x, mean_flow_y = 0,0
                        dx_mean,dy_mean = 0,0
                        write = False
                    elif np.abs(angle_degrees_OF - angle_degrees_LT) > angle_threshold :
                        #Si pas cohérent on supprime => Utilisation OF pour vérification 
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        write = False
                    elif not (1-speed_threshold < (speed_OF / speed_LT) < 1+speed_threshold):
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        write = False
                    else :
                        angle_liste_OF_m.append(angle_degrees_OF)
                        speed_liste_OF_m.append(speed_OF)
                        angle_liste_LT_m.append(angle_degrees_LT)
                        speed_liste_LT_m.append(speed_LT)
                    #transformation en m/s
                    mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                    dx_mean,dy_mean = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                    #List des vecteurs moyens pour chaque région en m/s
                    regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"modis"))
                    regional_vector_list_LT.append((dx_mean,dy_mean,"modis"))
                    if write : 
                        lat_lon_data = LatLonData(self.afai_mod['latitude'].values,self.afai_mod['longitude'].values)
                        date = self.date_modis
                        centre_mod = fonction_utile.calculate_barycenter(matches[0])
                        centre_mod = (centre_mod[0] + minc, centre_mod[1] + minr)
                        centre_myd = fonction_utile.calculate_barycenter(matches[1])
                        centre_myd = (centre_myd[0] + minc, centre_myd[1] + minr)
                        Mod_Lat, Mod_Lon = lat_lon_data.get_lat_lon_coordinates(centre_mod[1],centre_mod[0])
                        Myd_Lat, Myd_Lon = lat_lon_data.get_lat_lon_coordinates(centre_myd[1],centre_myd[0])
                        u_OF = mean_flow_x
                        v_OF = mean_flow_y
                        u_LT = dx_mean
                        v_LT = dy_mean
                        sat = "Modis"
                        if not is_data_existing(date, sat, Mod_Lat, Mod_Lon, df_existing) :
                            new_data.append([date,Mod_Lat, Mod_Lon, Myd_Lat, Myd_Lon,u_OF,v_OF,u_LT,v_LT,sat])
            self.angle_liste_OF_m = self.angle_liste_OF_m + angle_liste_OF_m
            self.speed_liste_OF_m = self.speed_liste_OF_m + speed_liste_OF_m
            self.angle_liste_LT_m = self.angle_liste_LT_m + angle_liste_LT_m
            self.speed_liste_LT_m = self.speed_liste_LT_m + speed_liste_LT_m
                
        #Ajout dans la liste des vecteurs de l'objet pour le calcul sur plusieurs jours
        self.regional_vector_list_OF = self.regional_vector_list_OF + regional_vector_list_OF
        self.regional_vector_list_LT = self.regional_vector_list_LT + regional_vector_list_LT
        self.centre = self.centre + centre_liste
        if not len(new_data) == 0 :
            df_new = pd.DataFrame(new_data, columns=['Date','Mod_Lat', 'Mod_Lon', 'Myd_Lat', 'Myd_Lon',"u_OF","v_OF","u_LT","v_LT","Sat"])
            # Fusionner les nouvelles données avec les données existantes
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            
            # Écriture des données combinées dans le fichier Excel
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                df_combined.to_excel(writer, sheet_name='All_Data', index=False)
            
        print("All regions processed and new data added to the Excel file.")
        # ============Créer la carte finale avec les vecteurs sur l'image============
        #============Définition Latitude et longitude et initialisation pour la carte final============
        if Modis :
            self.lat_lon_data = LatLonData(self.afai_mod['latitude'].values,self.afai_mod['longitude'].values)
            lat_lon_data = self.lat_lon_data
            lats = self.afai_mod['latitude'].values
            lons = self.afai_mod['longitude'].values
        if Goes :
            self.lat_lon_data_goes = LatLonData(self.fai_mod['latitude'].values,self.fai_mod['longitude'].values)
            lat_lon_data_goes = self.lat_lon_data_goes
            lats_goes = self.fai_mod['latitude'].values
            lons_goes = self.fai_mod['longitude'].values
            if not Modis :
                lats = lats_goes
                lons = lons_goes
                lat_lon_data = lat_lon_data_goes
        # Initialise une figure avec projection cartographique
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        #ax.set_extent([lons_min, lons_max, lats_min, lats_max], crs=ccrs.PlateCarree())
        ax.set_extent([lons.min()-1, lons.max()+1, lats.min()-1, lats.max()+1], crs=ccrs.PlateCarree())
        # Ajouter les caractéristiques de la carte
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        if OF :
            #==============Optical flow=============
            for index,((centre_x, centre_y), (vec_of_x, vec_of_y, image)) in enumerate(zip(centre_liste, regional_vector_list_OF)):
                if vec_of_x != 0 or vec_of_y != 0:
                    if image == "goes" :
                        color = "lightgreen"
                        centre_lat, centre_lon = lat_lon_data_goes.get_lat_lon_coordinates(centre_y, centre_x)
                    else :
                        color = "darkgreen"
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    ax.quiver(centre_lon, centre_lat, vec_of_x, vec_of_y, transform=ccrs.PlateCarree(), color=color, scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        #=====================Linear translation================
        if LT :
            for index,((centre_x, centre_y), (vec_lt_x, vec_lt_y, image)) in enumerate(zip(centre_liste, regional_vector_list_LT)):
                # Dessin des vecteurs Linear Translation
                if vec_lt_x != 0 or vec_lt_y != 0:
                    if image == "goes" :
                        color = '#FF6666'
                        centre_lat, centre_lon = lat_lon_data_goes.get_lat_lon_coordinates(centre_y, centre_x)
                    else :
                        color = '#8B0000'
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    ax.quiver(centre_lon, centre_lat, vec_lt_x, vec_lt_y, transform=ccrs.PlateCarree(), color=color, scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
                    
         #========Calcul des moyennes de vitesse et orientations pour la date============ 
        if Goes :
            #Moyenne vitesse et orientation
            mean_of_speed_g = np.mean(speed_liste_OF_g)
            mean_of_angle_g = np.mean(angle_liste_OF_g)
            mean_lt_speed_g = np.mean(speed_liste_LT_g)
            mean_lt_angle_g = np.mean(angle_liste_LT_g)
        if Modis : 
            #Moyenne vitesse et orientation
            mean_of_speed_m = np.mean(speed_liste_OF_m)
            mean_of_angle_m = np.mean(angle_liste_OF_m)
            mean_lt_speed_m = np.mean(speed_liste_LT_m)
            mean_lt_angle_m = np.mean(angle_liste_LT_m)
        
        #=============Ajout des légendes================
        if Goes and OF :
            legend_of_g = mlines.Line2D([], [], color="lightgreen", marker='_', linestyle='None',
                                      markersize=10, label=f'Optical Flow Goes')
        else :
            legend_of_g = ""
        if Goes and LT :    
            legend_lt_g = mlines.Line2D([], [], color='#FF6666', marker='_', linestyle='None',
                                      markersize=10, label=f'Linear Translation Goes')
        else : 
            legend_lt_g = ""
        if Modis and OF :
            legend_of_m = mlines.Line2D([], [], color="darkgreen", marker='_', linestyle='None',
                                      markersize=10, label=f'Optical Flow Modis')
        else :
            legend_of_m = ""
        if LT and Modis :
            legend_lt_m = mlines.Line2D([], [], color='#8B0000', marker='_', linestyle='None',
                                      markersize=10, label=f'Linear Translation Modis')
        else :
            legend_lt_m  = ""
            
        # Ajouter un vecteur d'échelle pour 1 m/s dans le coin en bas à droite à 69°W et 6°N
        scale_speed = 1
        ax.quiver(-69, 6, scale_speed, 0, transform=ccrs.PlateCarree(),
                  color='black', scale=20 , headwidth=1, headlength=0.5, headaxislength=0.5)
        # Ajouter un texte d'échelle au-dessus de la flèche d'échelle
        ax.text(-69, 6.1, 'Échelle: 1 m/s', transform=ccrs.PlateCarree(),
        verticalalignment='bottom', horizontalalignment='left')

        #============Terminer la création de la carte et exportation===============
        # Filtrer les handles non valides (par exemple, string vide)
        valid_handles = [handle for handle in [legend_of_m, legend_of_g, legend_lt_m, legend_lt_g] if isinstance(handle, plt.Line2D)]
        ax.legend(handles=valid_handles, loc='upper left')
        plt.title("Comparaison des vecteurs Optical Flow (vert) et Linear Translation (rouge)")
        plt.savefig(f"{base_path}day{self.number_file}_observation_vectors_map_obs.png", bbox_inches='tight', dpi=300)
        plt.show()


    def modis_translation_lissage(self ,OF = False , LT = True) :
        extend = self.extend
        angle_threshold = self.angle_threshold 
        speed_threshold = self.speed_threshold 
        #On initialise les listes des vecteurs moyens pour chaques ROI
        #Vecteurs (x,y)
        regional_vector_list_OF = []
        regional_vector_list_LT = []
        
        #Orientations_modis
        angle_liste_OF_m = []
        angle_liste_LT_m = []
        
        #Orientations goes
        angle_liste_OF_g = []
        angle_liste_LT_g = []
        
        #Vitesses_goes
        speed_liste_OF_g = []
        speed_liste_LT_g = []
        
        #Vitesses_modis
        speed_liste_OF_m = []
        speed_liste_LT_m = []

        #Centre
        centre_liste = []
        
        base_path = self.output_path
        plot_path = self.output_path + "modis_translation_lissage.pdf"
        #Read image
        self.read_modis()
        #Prétraitement modis
        self.preprocessing_modis()
        #Creation pdf et des graphiques :
        with PdfPages(plot_path) as pdf:
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.afai_mod_np, cmap='viridis', origin='lower')
            for region in self.regions_sarg:
                minr, minc, maxr, maxc = region['bbox']
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
            ax.set_title(f'Nouvelles régions de sargasse détectées modis')
            pdf.savefig(fig)
            plt.close(fig)
            
            for region in tqdm(self.regions_sarg):
                self.delta_t = self.delta_time_modis
                angle = True
                speed = True
                #On réinitialise la variable permettant d'identifier le faisabilité de LT
                LT_impossible = False
                
                #===============Définition des bords de la zone étudiée=================
                minr, minc, maxr, maxc = region['bbox']
                if extend != 0 :
                    minr = max(0, minr - extend)  # S'assurer que minr n'est pas négatif
                    minc = max(0, minc - extend)  # S'assurer que minc n'est pas négatif
                    maxr = min(self.afai_mod_np.shape[0], maxr + extend)  # Ne pas dépasser la hauteur de l'image
                    maxc = min(self.afai_mod_np.shape[1], maxc + extend)  # Ne pas dépasser la largeur de l'image
                # Extraire les ROIs pour afai_mod, afai_myd, afai_sarg_mod et afai_sarg_myd
                roi_mod = self.afai_mod_np[minr:maxr, minc:maxc]
                roi_myd = self.afai_myd_np[minr:maxr, minc:maxc]
                roi_sarg_mod = self.afai_sarg_mod_np[minr:maxr, minc:maxc]
                roi_sarg_myd = self.afai_sarg_myd_np[minr:maxr, minc:maxc]
                
                #Ajout centre
                # Calculer le centre de la ROI
                centre_x = (minc + maxc) // 2
                centre_y = (minr + maxr) // 2
                centre_liste.append((centre_x,centre_y))
                # ============= Calcul du flux optique et linear_translation pour la ROI=======
                flow = self.calculate_optical_flow_region(roi_mod, roi_myd)
                # Calcul de la linear translation pour la même ROI
                matches, nb_matches = self.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
                #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                # Calculer le vecteur moyen pour le flux optique
                mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                
                #================Représentation graphique=================
                #================Optical flow================
                fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                step = 6  # Contrôle la densité des flèches
                # Optical flow - MOD
                ax[0, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                ax[0, 0].set_title('Optical Flow - MOD')
                ax[0, 0].arrow(roi_mod.shape[1] // 2, roi_mod.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                 # Optical flow - MYD
                ax[0, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                ax[0, 1].set_title('Optical Flow - MYD')
                ax[0, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                if angle_degrees_OF < 0 :
                    angle_degrees_OF += 360
                speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
    
                #================Linear Translation================
                if matches and len(matches) > 1:
                    dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                    dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                    dx_mean = sum(dxs) / len(dxs)
                    dy_mean = sum(dys) / len(dys)
                    # Linear Translation - MOD - MYD
                    ax[1, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                    ax[1, 0].set_title('Linear Translation - MOD')
                    ax[1, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                    ax[1, 1].set_title('Linear Translation - MYD')
                    ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                     # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                    for pt_src, pt_dst in matches:
                        # Tracer un cercle sur le point source
                        ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                        # Tracer un cercle sur le point destination
                        ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                        # Créer une connexion entre les points source et destination
                        con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                  axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                        ax[1, 1].add_artist(con)
                        angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                        angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                        if angle_degrees_LT < 0 :
                            angle_degrees_LT += 360
                        speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                        ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                else : 
                     #Afficher un message indiquant qu'une translation linéaire est impossible
                    text_props = dict(ha='center', va='center', fontsize=12, color='black')
                    ax[1, 0].clear()  # Nettoyer l'axe si nécessaire
                    ax[1, 0].text(0.5, 0.5, "Translation linéaire impossible\nsur cette région", transform=ax[1, 0].transAxes, **text_props)
                    ax[1, 1].clear()  # Nettoyer l'axe si nécessaire
                    ax[1, 1].text(0.5, 0.5, "Translation linéaire impossible\nsur cette région", transform=ax[1, 1].transAxes, **text_props)
                    LT_impossible = True
                pdf.savefig(fig)  # Save the current figure into a PDF page
                plt.close(fig)
    
                #============Vérification si on garde le vecteur pour la carte===========
                if LT_impossible or speed_LT == 0 :
                    #SI pas de LT possible => pas de prédiction
                    mean_flow_x, mean_flow_y = 0,0
                    dx_mean,dy_mean = 0,0
                elif np.abs(angle_degrees_OF - angle_degrees_LT) > angle_threshold :
                    #Si pas cohérent on supprime => Utilisation OF pour vérification 
                    dx_mean,dy_mean = 0,0
                    mean_flow_x, mean_flow_y = 0,0
                    angle = False
                elif not (1-speed_threshold < (speed_OF / speed_LT) < 1+speed_threshold):
                    dx_mean,dy_mean = 0,0
                    mean_flow_x, mean_flow_y = 0,0
                    speed = False
                else :
                    angle_liste_OF_m.append(angle_degrees_OF)
                    speed_liste_OF_m.append(speed_OF)
                    angle_liste_LT_m.append(angle_degrees_LT)
                    speed_liste_LT_m.append(speed_LT)
                #transformation en m/s
                mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                dx_mean_m,dy_mean_m = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                #List des vecteurs moyens pour chaque région en m/s
                regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"modis"))
                regional_vector_list_LT.append((dx_mean_m,dy_mean_m,"modis"))

                if speed and angle and not LT_impossible :
                    roi_mod = fonction_utile.shift_image(roi_mod,dx_mean,dy_mean,18)
                    roi_myd = fonction_utile.shift_image(roi_myd,dx_mean,dy_mean,18)
                    roi_sarg_mod = fonction_utile.shift_image(roi_sarg_mod,dx_mean,dy_mean,18)
                    roi_sarg_myd = fonction_utile.shift_image(roi_sarg_myd,dx_mean,dy_mean,18)
                    #Ajout centre
                    # Calculer le centre de la ROI
                    centre_x = (minc + maxc) // 2
                    centre_y = (minr + maxr) // 2
                    centre_liste.append((centre_x,centre_y))
                    
                    # ============= Calcul du flux optique et linear_translation pour la ROI=======
                    flow = self.calculate_optical_flow_region(roi_mod, roi_myd)
                    # Calcul de la linear translation pour la même ROI
                    matches, nb_matches = self.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
                    #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                    flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                    # Calculer le vecteur moyen pour le flux optique
                    mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                    if matches and len(matches) > 0:
                        #================Représentation graphique=================
                        #================Optical flow================
                        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                        X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                        step = 6  # Contrôle la densité des flèches
                        # Optical flow - MOD
                        ax[0, 0].imshow(roi_sarg_mod, cmap='viridis', origin='lower')
                        ax[0, 0].set_title('Optical Flow - MOD')
                        ax[0, 0].arrow(roi_mod.shape[1] // 2, roi_mod.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                        ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                         # Optical flow - MYD
                        ax[0, 1].imshow(roi_sarg_myd, cmap='viridis', origin='lower')
                        ax[0, 1].set_title('Optical Flow - MYD')
                        ax[0, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                        ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                        angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                        angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                        if angle_degrees_OF < 0 :
                            angle_degrees_OF += 360
                        speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                        ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
        
                        #================Linear Translation================
                        dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                        dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                        dx_mean = sum(dxs) / len(dxs)
                        dy_mean = sum(dys) / len(dys)
                        # Linear Translation - MOD - MYD
                        ax[1, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                        ax[1, 0].set_title('Linear Translation - MOD')
                        ax[1, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                        ax[1, 1].set_title('Linear Translation - MYD')
                        ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                         # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                        for pt_src, pt_dst in matches:
                            # Tracer un cercle sur le point source
                            ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                            # Tracer un cercle sur le point destination
                            ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                            # Créer une connexion entre les points source et destination
                            con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                      axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                            ax[1, 1].add_artist(con)
                            angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                            angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                            if angle_degrees_LT < 0 :
                                angle_degrees_LT += 360
                            speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                            ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                        pdf.savefig(fig)  # Save the current figure into a PDF page
                        plt.close(fig)
                        
                    else :
                        LT_impossible = True
                    
                    #============Vérification si on garde le vecteur pour la carte===========
                    if LT_impossible or speed_LT == 0 :
                        #SI pas de LT possible => pas de prédiction
                        mean_flow_x, mean_flow_y = 0,0
                        dx_mean,dy_mean = 0,0
                    else :
                        angle_liste_OF_g.append(angle_degrees_OF)
                        speed_liste_OF_g.append(speed_OF)
                        angle_liste_LT_g.append(angle_degrees_LT)
                        speed_liste_LT_g.append(speed_LT)
                    mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                    dx_mean,dy_mean = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                    #List des vecteurs moyens pour chaque région en m/s
                    regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"modis_t"))
                    regional_vector_list_LT.append((dx_mean,dy_mean,"modis_t"))
        
        # ============Créer la carte finale avec les vecteurs sur l'image============
        #============Définition Latitude et longitude et initialisation pour la carte final============
        self.lat_lon_data = LatLonData(self.afai_mod['latitude'].values,self.afai_mod['longitude'].values)
        lat_lon_data = self.lat_lon_data
        lats = self.afai_mod['latitude'].values
        lons = self.afai_mod['longitude'].values
        # Initialise une figure avec projection cartographique
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        #ax.set_extent([lons_min, lons_max, lats_min, lats_max], crs=ccrs.PlateCarree())
        ax.set_extent([lons.min()-1, lons.max()+1, lats.min()-1, lats.max()+1], crs=ccrs.PlateCarree())
        # Ajouter les caractéristiques de la carte
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        if OF :
            #==============Optical flow=============
            for index,((centre_x, centre_y), (vec_of_x, vec_of_y, image)) in enumerate(zip(centre_liste, regional_vector_list_OF)):
                if vec_of_x != 0 or vec_of_y != 0:
                    if image == "modis_t" :
                        color = "lightgreen"
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    else :
                        color = "darkgreen"
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    ax.quiver(centre_lon, centre_lat, vec_of_x, vec_of_y, transform=ccrs.PlateCarree(), color=color, scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        #=====================Linear translation================
        if LT :
            for index,((centre_x, centre_y), (vec_lt_x, vec_lt_y, image)) in enumerate(zip(centre_liste, regional_vector_list_LT)):
                # Dessin des vecteurs Linear Translation
                if vec_lt_x != 0 or vec_lt_y != 0:
                    if image == "modis_t" :
                        color = '#FF6666'
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    else :
                        color = '#8B0000'
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    ax.quiver(centre_lon, centre_lat, vec_lt_x, vec_lt_y, transform=ccrs.PlateCarree(), color=color, scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
                    
         #========Calcul des moyennes de vitesse et orientations pour la date============ 
        #Moyenne vitesse et orientation
        mean_of_speed_g = np.mean(speed_liste_OF_g)
        mean_of_angle_g = np.mean(angle_liste_OF_g)
        mean_lt_speed_g = np.mean(speed_liste_LT_g)
        mean_lt_angle_g = np.mean(angle_liste_LT_g)
        #Moyenne vitesse et orientation
        mean_of_speed_m = np.mean(speed_liste_OF_m)
        mean_of_angle_m = np.mean(angle_liste_OF_m)
        mean_lt_speed_m = np.mean(speed_liste_LT_m)
        mean_lt_angle_m = np.mean(angle_liste_LT_m)
        
        #=============Ajout des légendes================
        legend_of_g = mlines.Line2D([], [], color="lightgreen", marker='_', linestyle='None',
                                  markersize=10, label=f'Optical Flow Modis lissé (Speed: {mean_of_speed_g:.2f} m/s, Orientation={mean_of_angle_g:.2f}°N)')
        legend_lt_g = mlines.Line2D([], [], color='#FF6666', marker='_', linestyle='None',
                                  markersize=10, label=f'Linear Translation Modis lissé (Speed : {mean_lt_speed_g:.2f} m/s, Orientation={mean_lt_angle_g:.2f}°N)')
        legend_of_m = mlines.Line2D([], [], color="darkgreen", marker='_', linestyle='None',
                                  markersize=10, label=f'Optical Flow Modis (Speed: {mean_of_speed_m:.2f} m/s, Orientation={mean_of_angle_m:.2f}°N)')
        legend_lt_m = mlines.Line2D([], [], color='#8B0000', marker='_', linestyle='None',
                                  markersize=10, label=f'Linear Translation Modis (Speed : {mean_lt_speed_m:.2f} m/s, Orientation={mean_lt_angle_m:.2f}°N)')
        # Ajouter un vecteur d'échelle pour 1 m/s dans le coin en bas à droite à 69°W et 6°N
        scale_speed = 1
        ax.quiver(-69, 6, scale_speed, 0, transform=ccrs.PlateCarree(),
                  color='black', scale=20 , headwidth=1, headlength=0.5, headaxislength=0.5)
        # Ajouter un texte d'échelle au-dessus de la flèche d'échelle
        ax.text(-69, 6.1, 'Échelle: 1 m/s', transform=ccrs.PlateCarree(),
        verticalalignment='bottom', horizontalalignment='left')

        #============Terminer la création de la carte et exportation===============
        if OF and LT :
            ax.legend(handles=[legend_of_m, legend_of_g, legend_lt_m ,legend_lt_g], loc='upper left')
        if OF and not LT :
            ax.legend(handles=[legend_of_m, legend_of_g], loc='upper left')
        if LT and not OF : 
            ax.legend(handles=[legend_lt_m ,legend_lt_g], loc='upper left')
        plt.title("modis_translation_lissage")
        plt.savefig(f"{base_path}day{self.number_file}_modis_translation_lissage_vectors_map_obs.png", bbox_inches='tight', dpi=300)
        plt.show()

    def modis_translation_lissage_region(self,OF = False , LT = True) :
        extend = self.extend
        angle_threshold = self.angle_threshold 
        speed_threshold = self.speed_threshold 
        #On initialise les listes des vecteurs moyens pour chaques ROI
        #Vecteurs (x,y)
        regional_vector_list_OF = []
        regional_vector_list_LT = []
        
        #Orientations_modis
        angle_liste_OF_m = []
        angle_liste_LT_m = []
        
        #Orientations goes
        angle_liste_OF_g = []
        angle_liste_LT_g = []
        
        #Vitesses_goes
        speed_liste_OF_g = []
        speed_liste_LT_g = []
        
        #Vitesses_modis
        speed_liste_OF_m = []
        speed_liste_LT_m = []

        #Centre
        centre_liste = []
        
        plot_path = self.output_path + "modis_translation_lissage_region.pdf"
        #Read image
        self.read_modis()
        #Prétraitement modis
        self.preprocessing_modis()
        #Creation pdf et des graphiques :
        with PdfPages(plot_path) as pdf:
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(self.afai_mod_np, cmap='viridis', origin='lower')
            for region in self.regions_sarg:
                minr, minc, maxr, maxc = region['bbox']
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
            ax.set_title(f'Nouvelles régions de sargasse détectées modis')
            pdf.savefig(fig)
            plt.close(fig)
            not_found = True
            for region in tqdm(self.regions_sarg):
                if not_found :
                    angle = True
                    speed = True
                    #On réinitialise la variable permettant d'identifier le faisabilité de LT
                    LT_impossible = False
                    
                    #===============Définition des bords de la zone étudiée=================
                    minr, minc, maxr, maxc = region['bbox']
                    if extend != 0 :
                        minr = max(0, minr - extend)  # S'assurer que minr n'est pas négatif
                        minc = max(0, minc - extend)  # S'assurer que minc n'est pas négatif
                        maxr = min(self.afai_mod_np.shape[0], maxr + extend)  # Ne pas dépasser la hauteur de l'image
                        maxc = min(self.afai_mod_np.shape[1], maxc + extend)  # Ne pas dépasser la largeur de l'image
                    # Extraire les ROIs pour afai_mod, afai_myd, afai_sarg_mod et afai_sarg_myd
                    roi_mod = self.afai_mod_np[minr:maxr, minc:maxc]
                    roi_myd = self.afai_myd_np[minr:maxr, minc:maxc]
                    roi_sarg_mod = self.afai_sarg_mod_np[minr:maxr, minc:maxc]
                    roi_sarg_myd = self.afai_sarg_myd_np[minr:maxr, minc:maxc]
                    
                    #Ajout centre
                    # Calculer le centre de la ROI
                    centre_x = (minc + maxc) // 2
                    centre_y = (minr + maxr) // 2
                    centre_liste.append((centre_x,centre_y))
                    # ============= Calcul du flux optique et linear_translation pour la ROI=======
                    flow = self.calculate_optical_flow_region(roi_mod, roi_myd)
                    # Calcul de la linear translation pour la même ROI
                    matches, nb_matches = self.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
                    #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                    flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                    # Calculer le vecteur moyen pour le flux optique
                    mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                    
                    #================Représentation graphique=================
                    #================Optical flow================
                    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                    X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                    step = 6  # Contrôle la densité des flèches
                    # Optical flow - MOD
                    ax[0, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                    ax[0, 0].set_title('Optical Flow - MOD')
                    ax[0, 0].arrow(roi_mod.shape[1] // 2, roi_mod.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                    ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                     # Optical flow - MYD
                    ax[0, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                    ax[0, 1].set_title('Optical Flow - MYD')
                    ax[0, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                    ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                    angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                    angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                    if angle_degrees_OF < 0 :
                        angle_degrees_OF += 360
                    speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                    ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
        
                    #================Linear Translation================
                    if matches and len(matches) > 3:
                        dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                        dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                        dx_mean = sum(dxs) / len(dxs)
                        dy_mean = sum(dys) / len(dys)
                        # Linear Translation - MOD - MYD
                        ax[1, 0].imshow(roi_mod, cmap='viridis', origin='lower')
                        ax[1, 0].set_title('Linear Translation - MOD')
                        ax[1, 1].imshow(roi_myd, cmap='viridis', origin='lower')
                        ax[1, 1].set_title('Linear Translation - MYD')
                        ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                         # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                        for pt_src, pt_dst in matches:
                            # Tracer un cercle sur le point source
                            ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                            # Tracer un cercle sur le point destination
                            ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                            # Créer une connexion entre les points source et destination
                            con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                      axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                            ax[1, 1].add_artist(con)
                            angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                            angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                            if angle_degrees_LT < 0 :
                                angle_degrees_LT += 360
                            speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                            ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                    else : 
                         #Afficher un message indiquant qu'une translation linéaire est impossible
                        text_props = dict(ha='center', va='center', fontsize=12, color='black')
                        ax[1, 0].clear()  # Nettoyer l'axe si nécessaire
                        ax[1, 0].text(0.5, 0.5, "Translation linéaire impossible\nsur cette région", transform=ax[1, 0].transAxes, **text_props)
                        ax[1, 1].clear()  # Nettoyer l'axe si nécessaire
                        ax[1, 1].text(0.5, 0.5, "Translation linéaire impossible\nsur cette région", transform=ax[1, 1].transAxes, **text_props)
                        LT_impossible = True
                    pdf.savefig(fig)  # Save the current figure into a PDF page
                    plt.close(fig)
        
                    #============Vérification si on garde le vecteur pour la carte===========
                    if LT_impossible or speed_LT == 0 :
                        #SI pas de LT possible => pas de prédiction
                        mean_flow_x, mean_flow_y = 0,0
                        dx_mean,dy_mean = 0,0
                    elif np.abs(angle_degrees_OF - angle_degrees_LT) > angle_threshold :
                        #Si pas cohérent on supprime => Utilisation OF pour vérification 
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        angle = False
                    elif not (1-speed_threshold < (speed_OF / speed_LT) < 1+speed_threshold):
                        dx_mean,dy_mean = 0,0
                        mean_flow_x, mean_flow_y = 0,0
                        speed = False
                    else :
                        angle_liste_OF_m.append(angle_degrees_OF)
                        speed_liste_OF_m.append(speed_OF)
                        angle_liste_LT_m.append(angle_degrees_LT)
                        speed_liste_LT_m.append(speed_LT)
                    #transformation en m/s
                    mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                    dx_mean_m,dy_mean_m = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                    #List des vecteurs moyens pour chaque région en m/s
                    regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"modis"))
                    regional_vector_list_LT.append((dx_mean_m,dy_mean_m,"modis"))
    
                    if speed and angle and not LT_impossible :
                        not_found = False
                        Dx = []
                        Dy = []
                        I = []
                        for i in tqdm(range(10,51)) :
                            #Creation du déplacement i/10 (1 à 5 pixels => 0.09 à 0.48 m/s)
                            
                            roi_mod_l = fonction_utile.lissage_image(roi_mod,0,i/10,18)
                            roi_sarg_mod_l = fonction_utile.lissage_image(roi_sarg_mod,0,i/10,18)
                            roi_myd_l = fonction_utile.shift_image(roi_mod_l,0,i/10)
                            roi_sarg_myd_l = fonction_utile.shift_image(roi_sarg_mod_l,0,i/10)
                            
                            #Ajout centre
                            # Calculer le centre de la ROI
                            centre_x = (minc + maxc) // 2
                            centre_y = (minr + maxr) // 2
                            centre_liste.append((centre_x,centre_y))
                            
                            # ============= Calcul du flux optique et linear_translation pour la ROI=======
                            flow = self.calculate_optical_flow_region(roi_mod_l, roi_myd_l)
                            # Calcul de la linear translation pour la même ROI
                            matches, nb_matches = self.calculate_linear_translation_region(roi_mod_l, roi_sarg_mod_l, roi_myd_l, roi_sarg_myd_l)
                            #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
                            flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
                            # Calculer le vecteur moyen pour le flux optique
                            mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
                            if matches and len(matches) > 0:
                                #================Représentation graphique=================
                                #================Optical flow================
                                fig, ax = plt.subplots(2, 2, figsize=(12, 12))
                                X, Y = np.meshgrid(np.arange(roi_mod.shape[1]), np.arange(roi_mod.shape[0]))
                                step = 6  # Contrôle la densité des flèches
                                # Optical flow - MOD
                                ax[0, 0].imshow(roi_sarg_mod_l, cmap='viridis', origin='lower')
                                ax[0, 0].set_title('Optical Flow - MOD')
                                ax[0, 0].arrow(roi_mod_l.shape[1] // 2, roi_mod_l.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                                ax[0, 0].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                                 # Optical flow - MYD
                                ax[0, 1].imshow(roi_sarg_myd_l, cmap='viridis', origin='lower')
                                ax[0, 1].set_title('Optical Flow - MYD')
                                ax[0, 1].arrow(roi_myd_l.shape[1] // 2, roi_myd_l.shape[0] // 2, mean_flow_x, mean_flow_y, head_width=2, head_length=3, fc='red', ec='red')
                                ax[0, 1].quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1], color='green', angles='xy', scale_units='xy', scale=1)
                                angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
                                angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
                                if angle_degrees_OF < 0 :
                                    angle_degrees_OF += 360
                                speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(self.delta_t))/ 3.6
                                ax[0, 1].text(0, -12, f'Speed: {speed_OF:.2f} m/s, Orientation:{angle_degrees_OF}°N', fontsize=11, color='red')
                
                                #================Linear Translation================
                                dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
                                dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
                                dx_mean = sum(dxs) / len(dxs)
                                dy_mean = sum(dys) / len(dys)
                                # Linear Translation - MOD - MYD
                                ax[1, 0].imshow(roi_mod_l, cmap='viridis', origin='lower')
                                ax[1, 0].set_title('Linear Translation - MOD')
                                ax[1, 1].imshow(roi_myd_l, cmap='viridis', origin='lower')
                                ax[1, 1].set_title('Linear Translation - MYD')
                                ax[1, 1].arrow(roi_myd.shape[1] // 2, roi_myd.shape[0] // 2, dx_mean, dy_mean,head_width=2, head_length=3, fc='red', ec='red')
                                 # Pour chaque paire de correspondances, tracer une connexion entre les subplots
                                for pt_src, pt_dst in matches:
                                    # Tracer un cercle sur le point source
                                    ax[1, 0].add_patch(plt.Circle((pt_src[0], pt_src[1]), 1, color='green', fill=True))
                                    # Tracer un cercle sur le point destination
                                    ax[1, 1].add_patch(plt.Circle((pt_dst[0], pt_dst[1]), 1, color='green', fill=True))
                                    # Créer une connexion entre les points source et destination
                                    con = ConnectionPatch(xyA=(pt_dst[0], pt_dst[1]), xyB=(pt_src[0], pt_src[1]), coordsA="data", coordsB="data",
                                                              axesA=ax[1, 1], axesB=ax[1, 0], color='green')
                                    ax[1, 1].add_artist(con)
                                    angle_radians_LT = np.arctan2(dy_mean, dx_mean)
                                    angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
                                    if angle_degrees_LT < 0 :
                                        angle_degrees_LT += 360
                                    speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(self.delta_t))/ 3.6
                                    ax[1, 1].text(0, -12, f'Speed: {speed_LT:.2f} m/s, Orientation:{angle_degrees_LT}°N', fontsize=11, color='red')
                                pdf.savefig(fig)  # Save the current figure into a PDF page
                                plt.close(fig)
                                
                            else :
                                LT_impossible = True
                            
                            #============Vérification si on garde le vecteur pour la carte===========
                            if LT_impossible or speed_LT == 0 :
                                #SI pas de LT possible => pas de prédiction
                                mean_flow_x, mean_flow_y = 0,0
                                dx_mean,dy_mean = 0,0
                            else :
                                angle_liste_OF_g.append(angle_degrees_OF)
                                speed_liste_OF_g.append(speed_OF)
                                angle_liste_LT_g.append(angle_degrees_LT)
                                speed_liste_LT_g.append(speed_LT)
                                
                            Dx.append([0,dx_mean])
                            Dy.append([i/10,dy_mean])
                            I.append(i)
                            mean_flow_x, mean_flow_y = (mean_flow_x/self.delta_t)/3.6, (mean_flow_y/self.delta_t)/3.6
                            dx_mean,dy_mean = (dx_mean/self.delta_t)/3.6, (dy_mean/self.delta_t)/3.6
                            #List des vecteurs moyens pour chaque région en m/s
                            regional_vector_list_OF.append((mean_flow_x, mean_flow_y,"modis_t"))
                            regional_vector_list_LT.append((dx_mean,dy_mean,"modis_t"))
                            
        # Séparation des valeurs de Dx et Dy en deux listes pour le tracé
        Dx_v, Dx_o = zip(*Dx)
        Dy_v, Dy_o = zip(*Dy)
        
        # Tracé des fonctions de Dx
        plt.figure(figsize=(10, 5))
        plt.plot(I, Dx_o, label="Dx observé", marker='o')
        plt.plot(I, Dx_v, label='Dx vrai', marker='x')
        plt.xlabel('I')
        plt.ylabel('Valeurs de Dx')
        plt.title('Plot de Dx')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(I, Dy_o, label="Dy observé", marker='o')
        plt.plot(I, Dy_v, label='Dy vrai', marker='x')
        plt.xlabel('I')
        plt.ylabel('Valeurs de Dy')
        plt.title('Plot de Dy')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def calculate_modis_goes_days(self, t0 = 0 , tf = 7, OF = False , LT = True , Modis = True , Goes = True) :
        if not OF and not LT :
            print("choisissez LT ou OF au moins")
            return

        #==========Boucle sur touts les jours à analyser============
        for i in tqdm(range(t0,tf)) :
            #Reset les états d'avancement
            self.read_modis_done = False
            self.read_goes_done = False
            self.preprocess_modis_done = False
            self.preprocess_goes_done = False
            #jour i
            self.number_file = i 
            #On calcul les rois et les déplacement associé pour cette date
            self.goes_modis_translation(OF = OF, LT = LT, Modis = Modis, Goes = Goes)

        base_path = self.output_path
        #=========Récupération des données pour chaques rois et chaque date
        regional_vector_list_OF = self.regional_vector_list_OF
        regional_vector_list_LT = self.regional_vector_list_LT
        centre_liste = self.centre
        
        #=========Initialisation de la carte===========
        #============Définition Latitude et longitude et initialisation pour la carte final============
        if Modis :
            self.lat_lon_data = LatLonData(self.afai_mod['latitude'].values,self.afai_mod['longitude'].values)
            lat_lon_data = self.lat_lon_data
            lats = self.afai_mod['latitude'].values
            lons = self.afai_mod['longitude'].values
        if Goes :
            self.lat_lon_data_goes = LatLonData(self.fai_mod['latitude'].values,self.fai_mod['longitude'].values)
            lat_lon_data_goes = self.lat_lon_data_goes
            lats_goes = self.fai_mod['latitude'].values
            lons_goes = self.fai_mod['longitude'].values
            if not Modis :
                lats = lats_goes
                lons = lons_goes
                lat_lon_data = lat_lon_data_goes
        # Initialise une figure avec projection cartographique
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        #ax.set_extent([lons_min, lons_max, lats_min, lats_max], crs=ccrs.PlateCarree())
        ax.set_extent([lons.min()-1, lons.max()+1, lats.min()-1, lats.max()+1], crs=ccrs.PlateCarree())
        # Ajouter les caractéristiques de la carte
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        if OF : 
           #==============Optical flow=============
            for index,((centre_x, centre_y), (vec_of_x, vec_of_y, image)) in enumerate(zip(centre_liste, regional_vector_list_OF)):
                if vec_of_x != 0 or vec_of_y != 0:
                    if image == "goes" :
                        color = "lightgreen"
                        centre_lat, centre_lon = lat_lon_data_goes.get_lat_lon_coordinates(centre_y, centre_x)
                    else :
                        color = "darkgreen"
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    ax.quiver(centre_lon, centre_lat, vec_of_x, vec_of_y, transform=ccrs.PlateCarree(), color=color, scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
        if LT :
            #=====================Linear translation================
            for index,((centre_x, centre_y), (vec_lt_x, vec_lt_y, image)) in enumerate(zip(centre_liste, regional_vector_list_LT)):
                # Dessin des vecteurs Linear Translation
                if vec_lt_x != 0 or vec_lt_y != 0:
                    if image == "goes" :
                        color = '#FF6666'
                        centre_lat, centre_lon = lat_lon_data_goes.get_lat_lon_coordinates(centre_y, centre_x)
                    else :
                        color = '#8B0000'
                        centre_lat, centre_lon = lat_lon_data.get_lat_lon_coordinates(centre_y, centre_x)
                    ax.quiver(centre_lon, centre_lat, vec_lt_x, vec_lt_y, transform=ccrs.PlateCarree(), color=color, scale=20, headwidth=1, headlength=0.5, headaxislength=0.5)
    
        #========Calcul des moyennes de vitesse et orientations pour la date============ 
        #Moyenne vitesse et orientation
        if Goes :
            if OF :
                mean_of_speed_g = np.mean(self.speed_liste_OF_g)
                mean_of_angle_g = np.mean(self.angle_liste_OF_g)
                print("Optical Flow Goes (Speed: ",mean_of_speed_g , ":m/s, Orientation=" ,mean_of_angle_g,"°N")
            if LT :
                mean_lt_speed_g = np.mean(self.speed_liste_LT_g)
                mean_lt_angle_g = np.mean(self.angle_liste_LT_g)
                print("Linear Translation Goes (Speed: ",mean_lt_speed_g , ":m/s, Orientation=" ,mean_lt_angle_g,"°N")
        if Modis : 
            if OF :
            #Moyenne vitesse et orientation
                mean_of_speed_m = np.mean(self.speed_liste_OF_m)
                mean_of_angle_m = np.mean(self.angle_liste_OF_m)
                print("Optical Flow Modis (Speed: ",mean_of_speed_m , ":m/s, Orientation=" ,mean_of_angle_m,"°N")
            if LT :
                mean_lt_speed_m = np.mean(self.speed_liste_LT_m)
                mean_lt_angle_m = np.mean(self.angle_liste_LT_m)
                print("Linear translation Modis (Speed: ",mean_lt_speed_m , ":m/s, Orientation=" ,mean_lt_angle_m,"°N")
                
        #=============Ajout des légendes================
        if Goes and OF :
            legend_of_g = mlines.Line2D([], [], color="lightgreen", marker='_', linestyle='None',
                                      markersize=10, label=f'Optical Flow Goes')
        else :
            legend_of_g = ""
        if Goes and LT :    
            legend_lt_g = mlines.Line2D([], [], color='#FF6666', marker='_', linestyle='None',
                                      markersize=10, label=f'Linear Translation Goes')
        else : 
            legend_lt_g = ""
        if Modis and OF :
            legend_of_m = mlines.Line2D([], [], color="darkgreen", marker='_', linestyle='None',
                                      markersize=10, label=f'Optical Flow Modis')
        else :
            legend_of_m = ""
        if LT and Modis :
            legend_lt_m = mlines.Line2D([], [], color='#8B0000', marker='_', linestyle='None',
                                      markersize=10, label=f'Linear Translation Modis')
        else :
            legend_lt_m  = ""
            
        # Ajouter un vecteur d'échelle pour 1 m/s dans le coin en bas à droite à 69°W et 6°N
        scale_speed = 1
        ax.quiver(-69, 6, scale_speed, 0, transform=ccrs.PlateCarree(),
                  color='black', scale=20 , headwidth=1, headlength=0.5, headaxislength=0.5)
        # Ajouter un texte d'échelle au-dessus de la flèche d'échelle
        ax.text(-69, 6.1, 'Échelle: 1 m/s', transform=ccrs.PlateCarree(),
        verticalalignment='bottom', horizontalalignment='left')

        #============Terminer la création de la carte et exportation===============
        # Filtrer les handles non valides (par exemple, string vide)
        valid_handles = [handle for handle in [legend_of_m, legend_of_g, legend_lt_m, legend_lt_g] if isinstance(handle, plt.Line2D)]
        ax.legend(handles=valid_handles, loc='upper left')
        plt.title("final vector map all days combined")
        plt.savefig(f"{base_path}_{t0}to{tf}days_observation_vectors_map_obs.png", bbox_inches='tight', dpi=300)
        plt.show()

#===========utilities class===========================
#===========Classe pour la méthode SIFT===============
class LatLonData :
    def __init__(self, lats, lons):
        self.lat_min = lats[-1]
        self.lat_max = lats[0]
        self.lon_min = lons[0]
        self.lon_max = lons[-1]
        self.delta_lat = (self.lat_max - self.lat_min) / (len(lats) - 1)
        self.delta_lon = (self.lon_max - self.lon_min) / (len(lons) - 1)

    def get_i_j_coordinates(self, lat, lon):
        i = int((self.lat_max - lat) / self.delta_lat)
        j = int((lon - self.lon_min) / self.delta_lon)
        return i, j

    def get_frac_i_j_coordinates(self, lat, lon):
        i = (self.lat_max - lat) / self.delta_lat
        j = (lon - self.lon_min) / self.delta_lon
        return i, j

    def get_lat_lon_coordinates(self, i, j):
        lat = self.lat_max - i * self.delta_lat
        lon = self.lon_min + j * self.delta_lon
        return lat, lon

class Matching(ABC):

    def __init__(self, AFAI_src_dev, AFAI_src_sarg, AFAI_dst_dev,
                 AFAI_dst_sarg, lat_lon_data, plot_path, date, drifter_drift, wind_drift=None):
        super().__init__()
        self.plot_path = plot_path
        self.date = date
        self.lat_lon_data = lat_lon_data
        self.AFAI_src_dev = AFAI_src_dev
        self.AFAI_src_sarg = AFAI_src_sarg
        self.AFAI_dst_dev = AFAI_dst_dev
        self.AFAI_dst_sarg = AFAI_dst_sarg
        self.AFAI_src_dev_8bit = cv2.normalize(AFAI_src_dev, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.AFAI_dst_dev_8bit = cv2.normalize(AFAI_dst_dev, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.drifter_drift = drifter_drift
        self.sarg_drift = None
        self.wind_drift = wind_drift
        self.matching_method = ""

    @abstractmethod
    def match(self):
        pass

    def draw_drift(self, axarr, drift, color, scale=3):
        for ax in axarr:
            ax.quiver(drift.get_start_pos()[0], drift.get_start_pos()[1],
                            drift.drift[0]*scale, -drift.drift[1]*scale, color=color, scale=1,
                            scale_units='xy')
            ax.add_patch(plt.Circle(drift.get_start_pos(), 0.5, color=color, linewidth=1, ec="black"))

    def draw_drifts(self, axarr,):
        self.draw_drift(axarr, self.sarg_drift, "red")
        # self.draw_drift(axarr, self.drifter_drift, "blue")
        # if self.wind_drift is not None:
        #     self.draw_drift(axarr, self.wind_drift, "green")


    def save_plot(self):
        # plt.suptitle(f"Method: {self.matching_method}\nSargassum Drift: {round(self.sarg_drift.get_distance(), 2)}"
        #              f",azimuth: {round(self.sarg_drift.get_azimuth(), 2)}\nDrifter Drift: "
        #              f"{round(self.drifter_drift.get_distance(), 2)}"
        #              f",azimuth: {round(self.drifter_drift.get_azimuth(), 2)}", fontsize=14)
        filename = fonction_utile.build_filename(self.date, f"{self.plot_path}YYYY-MM-DD_{self.matching_method}.png")
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close()

    @staticmethod
    def build_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_date_to_filename(template, date):
        return fonction_utile.build_filename(date, template)

    def get_sarg_drift(self):
        return self.sarg_drift

class Drift:
    """
    Class for drift data. It represents a drift between two geographic coordinates and its correspondent positions in a
    specific image.
    """
    def __init__(self, lat_lon_data, lat_start, lon_start, lat_end=None, lon_end=None,
                 azimuth=None, dist=None):
        self.lat_start = lat_start
        self.lon_start = lon_start
        self.lat_lon_data = lat_lon_data
        if azimuth is not None and dist is not None:
            self.dist = dist
            self.azimuth = azimuth
            self.lat_end = None
            self.lon_end = None
            self.compute_end_coordinates()
        else:
            self.lat_end = lat_end
            self.lon_end = lon_end
            self.azimuth = None
            self.dist = None
            self.compute_azimuth()

        self.i_start, self.j_start = lat_lon_data.get_frac_i_j_coordinates(lat_start, lon_start)
        self.i_end, self.j_end = lat_lon_data.get_frac_i_j_coordinates(self.lat_end, self.lon_end)
        self.drift = np.array([self.j_end - self.j_start, self.i_end - self.i_start])

    def compute_end_coordinates(self):
        geod = Geod(ellps="WGS84")
        lon_end, lat_end, _ = geod.fwd(self.lon_start, self.lat_start, self.azimuth, self.dist)
        self.lat_end = lat_end
        self.lon_end = lon_end

    def compute_azimuth(self):
        """
        Function for drift azimuth and distance computation.
        Returns
        -------
        """
        geod = Geod(ellps="WGS84")
        azimuth, _, dist = geod.inv(self.lon_start, self.lat_start,
                                    self.lon_end, self.lat_end)
        self.dist = dist
        self.azimuth = azimuth

    def get_start_pos(self):
        """
        Getter for x,y starting position in the image.
        Returns
        -------
            1D array of length 2: x, y for starting position
        """
        return np.array([self.j_start, self.i_start])

    def get_end_pos(self):
        """
                Getter for x,y ending position in the image.
                Returns
                -------
                    1D array of length 2: x,y for ending position
                """
        return np.array([self.j_end,self.i_end])

    def get_azimuth(self):
        azimuth = self.azimuth if self.azimuth > 0 else self.azimuth + 360
        return azimuth

    def get_distance(self):
        return self.dist / 1000

class KeyPoints(Matching):
    """
    Abstract class for key point extraction based on SIFT algorithm.
    """

    @abstractmethod
    def match(self):
        pass

    def __init__(self, AFAI_src_dev, AFAI_src_sarg, AFAI_dst_dev, AFAI_dst_sarg, lat_lon_data, plot_path, date, drifter_drift, wind_drift=None):
        super().__init__(AFAI_src_dev, AFAI_src_sarg, AFAI_dst_dev, AFAI_dst_sarg, lat_lon_data, plot_path, date, drifter_drift, wind_drift)
        self.kp_src = []
        self.kp_dst = []
        self.desc_src = []
        self.desc_dst = []

        self.final_matches = None

    def resize(self, scale_percent):
        """
        Image resizing function.
        Parameters
        ----------
        scale_percent
            Scale for resizing, in percent
        """
        width = int(self.AFAI_src_dev.shape[1] * scale_percent / 100)
        height = int(self.AFAI_src_dev.shape[0] * scale_percent / 100)
        dim = (width, height)
        self.AFAI_src_dev = cv2.resize(self.AFAI_src_dev, dim, interpolation=cv2.INTER_AREA)
        self.AFAI_dst_dev = cv2.resize(self.AFAI_dst_dev, dim, interpolation=cv2.INTER_AREA)
        self.AFAI_src_dev_8bit = cv2.normalize(self.AFAI_src_dev, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        self.AFAI_dst_dev_8bit = cv2.normalize(self.AFAI_dst_dev, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    def mask_kp(self):
        """
        Key points masking according to nan values.
        """
        new_kp_src = []
        new_kp_dst = []
        for i in range(len(self.kp_src)):
            pt_src = self.kp_src[i].pt
            if not self.check_nan(pt_src, self.AFAI_src_dev):
                new_kp_src.append(self.kp_src[i])
        for j in range(len(self.kp_dst)):
            pt_dst = self.kp_dst[j].pt
            if not self.check_nan(pt_dst, self.AFAI_dst_dev):
                new_kp_dst.append(self.kp_dst[j])
        self.kp_src = new_kp_src
        self.kp_dst = new_kp_dst

    @staticmethod
    def check_nan(pt, img, nb_pixel=2.5):
        """
        Function for checking nan presence in a radius around a point.
        Parameters
        ----------
        pt
            Array of length 2 for point coordinates.
        img
            2D array for the image.
        nb_pixel
            Float for the radius around the point, in pixels.
        Returns
        -------
            Boolean for nan presence
        """
        threshold = img[int(pt[1]), int(pt[0])] < .000179
        xmin = max(0, int(pt[1] - nb_pixel))
        xmax = min(int(pt[1] + nb_pixel), len(img))
        ymin = max(0, int(pt[0] - nb_pixel))
        ymax = min(int(pt[0] + nb_pixel), len(img[0]))
        nan = np.any(np.isnan(img[xmin:xmax, ymin:ymax]))
        return nan or threshold

    def compute_key_points(self):
        """
        SIFT detection of key points.
        """
        sift = cv2.SIFT_create()
        self.kp_src, self.desc_src = sift.detectAndCompute(self.AFAI_src_dev_8bit, None)
        self.kp_dst, self.desc_dst = sift.detectAndCompute(self.AFAI_dst_dev_8bit, None)

    def draw_key_points(self):
        """
        Display key points on the original images.
        """
        plt.figure()
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(self.AFAI_src_dev)
        axarr[1].imshow(self.AFAI_dst_dev)
        for kp in self.kp_src:
            rgb = (random.random(), random.random(), random.random())
            axarr[0].add_patch(plt.Circle(kp.pt, 1, color=rgb, linewidth=1, ec="black"))
        for kp in self.kp_dst:
            rgb = (random.random(), random.random(), random.random())
            axarr[1].add_patch(plt.Circle(kp.pt, 1, color=rgb, linewidth=1, ec="black"))
        plt.savefig(self.plot_path + "key_points.png", bbox_inches='tight', dpi=200)

    def draw_final_matches(self,pdf):
        """
        Display final matching between selected key points.
        """
        self.compute_position()
        f, axarr = plt.subplots(1, 2, figsize=(10,5))
        axarr[0].imshow(self.AFAI_src_dev)
        axarr[1].imshow(self.AFAI_dst_dev)
        self.draw_drifts(axarr)

        for src_pt, dst_pt in self.final_matches:
            rgb = (random.random(), random.random(), random.random())
            axarr[0].add_patch(plt.Circle(src_pt, 1, color=rgb, linewidth=1, ec="black"))
            axarr[1].add_patch(plt.Circle(dst_pt, 1, color=rgb, linewidth=1, ec="black"))
            con = ConnectionPatch(xyA=dst_pt, xyB=src_pt, coordsA="data", coordsB="data",
                                  axesA=axarr[1], axesB=axarr[0], color=rgb)
            axarr[1].add_artist(con)
        pdf.savefig()
        plt.close('all')
        #self.save_plot() ici normalement
    def compute_position(self):
        """
        Compute Sargassum geographic coordinates based on position in the image. Cast it as a drifter object.
        """
        barycenter_src, barycenter_dst = self.compute_barycenter()
        lat_barycenter_src, lon_barycenter_src = self.lat_lon_data.get_lat_lon_coordinates(barycenter_src[1],
                                                                                           barycenter_src[0])
        lat_barycenter_dst, lon_barycenter_dst = self.lat_lon_data.get_lat_lon_coordinates(barycenter_dst[1],
                                                                                           barycenter_dst[0])
        self.sarg_drift = Drift(self.lat_lon_data, lat_barycenter_src, lon_barycenter_src, lat_end=lat_barycenter_dst, lon_end=lon_barycenter_dst)

    def compute_barycenter(self):
        """
        Compute Sargassum position (in image coordinates) as the barycenter of the retained key points.
        """
        n = len(self.final_matches)
        barycenter_src = [0, 0]
        barycenter_dst = [0, 0]
        for src_pt, dst_pt in self.final_matches:
            barycenter_src[0] += src_pt[0] / n
            barycenter_src[1] += src_pt[1] / n
            barycenter_dst[0] += dst_pt[0] / n
            barycenter_dst[1] += dst_pt[1] / n
        return barycenter_src, barycenter_dst

class LinearTranslation(KeyPoints):
    """
    Class for key points matching with linear translation constraint.
    """

    def __init__(self, AFAI_src_dev, AFAI_src_sarg, AFAI_dst_dev, AFAI_dst_sarg, lat_lon_data, plot_path, date, drifter_drift,delta_t,pixel_res,max_sarg_speed,matches_threshold, wind_drift=None):
        super().__init__(AFAI_src_dev, AFAI_src_sarg, AFAI_dst_dev, AFAI_dst_sarg, lat_lon_data, plot_path, date, drifter_drift, wind_drift)
        self.affine = None
        self.matching_method = "SIFT linear translation"
        #Threshold for the number of matches minimum for linear_translation
        self.matches_threshold = matches_threshold
        self.delta_t = delta_t
        #Pixel resolution in km
        self.pixel_res = pixel_res
        #Maximal speed for sargasse in m/s
        self.max_sarg_speed = max_sarg_speed
        self.nb_matching_points = 0

    def match(self):
        """
        Global process.
        """
        self.compute_key_points()
        self.mask_kp()
        self.eliminate_duplicates()
        self.estimate_translation()

    def eliminate_duplicates(self):
        """
        Duplicated key points elimination.
        """
        new_kp_src = []
        new_kp_dst = []
        for i in range(len(self.kp_src)):
            if self.kp_src[i].pt not in [kp.pt for kp in new_kp_src]:
                new_kp_src.append(self.kp_src[i])
        for j in range(len(self.kp_dst)):
            if self.kp_dst[j] not in [kp.pt for kp in new_kp_dst]:
                new_kp_dst.append(self.kp_dst[j])

        self.kp_src = new_kp_src
        self.kp_dst = new_kp_dst

    def estimate_translation(self):
        """
        Estimate translation for every different pair of key points. Score it with the number of matching points. Select
        the best translation.
        """
        n_matches = 0
        matches_final = None
        for i in range(len(self.kp_src)):
            for j in range(len(self.kp_dst)):
                pt_src = self.kp_src[i].pt
                pt_dst = self.kp_dst[j].pt

                vec = (pt_dst[0] - pt_src[0], pt_dst[1] - pt_src[1])
                n_trans, matches = self.test_translation(vec)
                if n_trans > n_matches:
                    n_matches = n_trans
                    matches_final = matches
        if n_matches >= self.matches_threshold:
            self.final_matches = matches_final
            self.nb_matching_points = n_matches

    def test_translation(self, vec):
        """
        Evaluating the number of matches for a translation.
        Parameters
        ----------
        vec
            Translation vector.

        Returns
        -------
        count
            Number of  matches.
        Matches
            List of matching pairs of coordinates.
        """
        
        # Vérifier si le vecteur de translation est dans les limites spécifiées
        if np.sqrt(vec[0]**2 + vec[1]**2) > self.max_sarg_speed*(3.6/self.pixel_res)*self.delta_t : #check if max speed verified
            return 0, []  # Retourner immédiatement si le vecteur est hors limites
        count = 0
        matches = []
        for i in range(len(self.kp_src)):

            pt_src = self.kp_src[i].pt
            pt_trans = (pt_src[0] + vec[0], pt_src[1] + vec[1])
            for j in range(len(self.kp_dst)):
                pt_dst = self.kp_dst[j].pt
                dif = (pt_dst[0] - pt_trans[0], pt_dst[1] - pt_trans[1])
                dist = np.sqrt(dif[0]**2 + dif[1]**2)
                if dist < 1.5:
                    count += 1
                    matches.append((pt_src, pt_dst))
                    break
        return count, matches
