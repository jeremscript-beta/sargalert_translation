import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import Button
from openpyxl import load_workbook
from classes_drift import Modis_Goes_Translation , LatLonData
import fonction_utile

# Chemin vers le fichier Excel contenant les données existantes
excel_path = r"C:\Users\DELL\Desktop\Jupyter\Projet_Sargalert\Notebooks\points_info_modis.xlsx"  # Remplacez par le chemin réel du fichier

# Initialize an instance of OpticalFlowTracker
Translation = Modis_Goes_Translation()

# Supposons que vous travaillez sur la date5 (index 5)
Translation.number_file = 2
# Année et mois d'étude
angle_threshold = 25
speed_threshold = 0.2
matches_threshold = 2
annee = 2023
mois = 6
heure_mod = "1130"
heure_myd = "1430"
date = f'{annee}{mois:02}{Translation.number_file+1:02}_{heure_mod}_{heure_myd}'

Translation.read_modis()
Translation.preprocess_image_with_square_se()
Translation.intersect_images()
Translation.find_sargassum_polygons()
Translation.chargement_numpy()
Translation.lat_lon_data = LatLonData(Translation.afai_mod['latitude'].values,Translation.afai_mod['longitude'].values)
lat_lon_data = Translation.lat_lon_data
# Tentative de charger les points existants
try:
    df_existing = pd.read_excel(excel_path)
    print("Loaded existing data points.")
except Exception as e:
    print(f"No existing data points: {e}")
    df_existing = pd.DataFrame(columns=['Date', 'Mod_Lat_m', 'Mod_Lon_m', 'Myd_Lat_m', 'Myd_Lon_m', 'u_m', 'v_m','Mod_Lat', 'Mod_Lon', 'Myd_Lat', 'Myd_Lon',"u_OF","v_OF","u_LT","v_LT"])

regions = Translation.regions_sarg
points_info = []
regional_vector_list_OF = []
regional_vector_list_LT = []

for i, region in enumerate(regions):
    extend = 30
    minr, minc, maxr, maxc = region['bbox']
    minr = max(0, minr - extend)
    minc = max(0, minc - extend)
    maxr = min(Translation.afai_mod_np.shape[0], maxr + extend)
    maxc = min(Translation.afai_mod_np.shape[1], maxc + extend)
    roi_mod = Translation.afai_mod_np[minr:maxr, minc:maxc]
    roi_myd = Translation.afai_myd_np[minr:maxr, minc:maxc]
    roi_mod = roi_mod.astype(np.float32)
    roi_myd = roi_myd.astype(np.float32)
    roi_mod = cv2.normalize(roi_mod, None, 0, 255, cv2.NORM_MINMAX)
    roi_myd = cv2.normalize(roi_myd, None, 0, 255, cv2.NORM_MINMAX)
    roi_sarg_mod = Translation.afai_sarg_mod_np[minr:maxr, minc:maxc]
    roi_sarg_myd = Translation.afai_sarg_myd_np[minr:maxr, minc:maxc]
    liste_roi_mod = []
    liste_roi_myd = []
    selection_finished = False
    # ============= Calcul du flux optique et linear_translation pour la ROI=======
    flow = Translation.calculate_optical_flow_region(roi_mod, roi_myd)
    # Calcul de la linear translation pour la même ROI
    matches, nb_matches = Translation.calculate_linear_translation_region(roi_mod, roi_sarg_mod, roi_myd, roi_sarg_myd)
    #Filtrage Optical Flow pour ne garder que les points clefs pour le flux optique
    flow_masked = fonction_utile.filter_optical_flow_with_keypoints(flow, matches)
    # Calculer le vecteur moyen pour le flux optique
    mean_flow_x,mean_flow_y = fonction_utile.calculate_extreme_flow(flow_masked)
    if nb_matches > matches_threshold :
        angle_radians_OF = np.arctan2(mean_flow_y, mean_flow_x)
        angle_degrees_OF = 90 - int(round(np.degrees(angle_radians_OF)))
        if angle_degrees_OF < 0 :
            angle_degrees_OF += 360
        speed_OF = ((np.sqrt(mean_flow_x**2 + mean_flow_y**2))/(Translation.delta_t))/ 3.6
        dxs = [pt_dst[0] - pt_src[0] for pt_src, pt_dst in matches]
        dys = [pt_dst[1] - pt_src[1] for pt_src, pt_dst in matches]
        dx_mean = sum(dxs) / len(dxs)
        dy_mean = sum(dys) / len(dys)
        angle_radians_LT = np.arctan2(dy_mean, dx_mean)
        angle_degrees_LT = 90 - int(round(np.degrees(angle_radians_LT)))
        if angle_degrees_LT < 0 :
            angle_degrees_LT += 360
        speed_LT = (np.sqrt(dx_mean**2 + dy_mean**2)/(Translation.delta_t))/ 3.6
        if np.abs(angle_degrees_OF - angle_degrees_LT) < angle_threshold and (1-speed_threshold < (speed_OF / speed_LT) < 1+speed_threshold) :
            
            def onclick(event, roi_list, ax, color):
                if event.inaxes == ax:
                    x, y = event.xdata, event.ydata
                    roi_list.append((x, y))
                    ax.plot(x, y, 'o', color=color)
                    plt.draw()
        
            def finish_selection(event):
                global selection_finished
                selection_finished = True
                plt.close()
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
            ax1.imshow(roi_mod, cmap='viridis', origin="lower")
            ax1.set_title('Cliquez sur roi_mod')
            ax2.imshow(roi_myd, cmap='viridis', origin="lower")
            ax2.set_title('Cliquez sur roi_myd')
        
            ax_button = plt.axes([0.45, 0.01, 0.1, 0.05])
            button = Button(ax_button, 'Finir la sélection')
            button.on_clicked(finish_selection)
        
            fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, liste_roi_mod, ax1, "red"))
            fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, liste_roi_myd, ax2, "blue"))
        
            plt.show()
        
            while not selection_finished:
                plt.pause(0.1)
            if len(liste_roi_mod)>0 :    
                centre_mod = fonction_utile.calculate_barycenter(matches[0])
                centre_mod = (centre_mod[0] + minc, centre_mod[1] + minr)
                centre_myd = fonction_utile.calculate_barycenter(matches[1])
                centre_myd = (centre_myd[0] + minc, centre_myd[1] + minr)
                mod_points = fonction_utile.calculate_barycenter(liste_roi_mod)
                mod_points = (mod_points[0] + minc, mod_points[1] + minr)
                myd_points = fonction_utile.calculate_barycenter(liste_roi_myd)
                myd_points = (myd_points[0] + minc, myd_points[1] + minr)
                points_info.append((mod_points,myd_points))
                regional_vector_list_OF.append((mean_flow_x, mean_flow_y , centre_mod , centre_myd))
                regional_vector_list_LT.append((dx_mean, dy_mean , centre_mod , centre_myd))
            print(f"Processed region {i}: {len(liste_roi_mod)} MOD points, {len(liste_roi_myd)} MYD points.")

# Création des nouvelles données à ajouter au fichier Excel
new_data = []
for (mod_points, myd_points),(mean_flow_x, mean_flow_y , centre_mod , centre_myd),(dx_mean, dy_mean ,centre_mod , centre_myd) in zip(points_info,regional_vector_list_OF,regional_vector_list_LT):
    
    u = ((myd_points[0]-mod_points[0])/(Translation.delta_t))/ 3.6
    v = ((myd_points[1]-mod_points[1])/(Translation.delta_t))/ 3.6
    u_OF = mean_flow_x /(Translation.delta_t)/ 3.6
    v_OF = mean_flow_y /(Translation.delta_t)/ 3.6
    u_LT = dx_mean /(Translation.delta_t)/ 3.6
    v_LT = dy_mean /(Translation.delta_t)/ 3.6
    Mod_Lat_m, Mod_Lon_m = lat_lon_data.get_lat_lon_coordinates(mod_points[1],mod_points[0])
    Myd_Lat_m, Myd_Lon_m = lat_lon_data.get_lat_lon_coordinates(myd_points[1],myd_points[0])
    Mod_Lat, Mod_Lon = lat_lon_data.get_lat_lon_coordinates(centre_mod[1],centre_mod[0])
    Myd_Lat, Myd_Lon = lat_lon_data.get_lat_lon_coordinates(centre_myd[1],centre_myd[0])
    new_data.append([date, Mod_Lat_m, Mod_Lon_m, Myd_Lat_m, Myd_Lon_m, u, v,Mod_Lat, Mod_Lon, Myd_Lat, Myd_Lon,u_OF,v_OF,u_LT,v_LT])
df_new = pd.DataFrame(new_data, columns=['Date', 'Mod_Lat_m', 'Mod_Lon_m', 'Myd_Lat_m', 'Myd_Lon_m', 'u_m', 'v_m','Mod_Lat', 'Mod_Lon', 'Myd_Lat', 'Myd_Lon',"u_OF","v_OF","u_LT","v_LT"])

# Fusionner les nouvelles données avec les données existantes
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Écriture des données combinées dans le fichier Excel
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
    df_combined.to_excel(writer, sheet_name='All_Data', index=False)

print("All regions processed and new data added to the Excel file.")