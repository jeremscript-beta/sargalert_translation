import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.morphology import square, opening, closing
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_dilation ,gaussian_filter,map_coordinates
from skimage.draw import ellipse_perimeter
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import xarray as xr
import os
from datetime import datetime, timedelta
import time
from sklearn.neighbors import KDTree
import requests

#==========DATA EVALUATION==========
def calculate_barycenter(points):
    """
    Calcule le barycentre d'une liste de points en 2D.

    :param points: Liste de tuples ou de listes représentant les points (x, y).
    :return: Tuple représentant les coordonnées du barycentre (x_bar, y_bar).
    """
    if not points:
        raise ValueError("La liste de points ne peut pas être vide")

    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    x_bar = sum(x_coords) / len(points)
    y_bar = sum(y_coords) / len(points)

    return x_bar, y_bar
    
def add_blur_and_controlled_noise(image, blur_kernel=(5, 5), noise_scale=0.02, noise_freq=0.1, mean=0.000408, std=0.000236):
    """
    Ajoute un effet de flou et du bruit contrôlé à une image.

    Args:
    - image: L'image originale (numpy.ndarray).
    - blur_kernel: La taille du noyau utilisé pour le flou (tuple).
    - noise_scale: L'échelle du bruit à ajouter (float).
    - noise_freq: La fréquence des pixels affectés par le bruit (float entre 0 et 1).
    - mean: La moyenne du bruit gaussien (float).
    - std: L'écart-type du bruit gaussien (float).

    Returns:
    - image_with_noise: L'image avec effet de flou et bruit contrôlé ajouté (numpy.ndarray).
    """
    # seed 
    np.random.seed(42)
    # Appliquer un flou gaussien
    blurred_image = cv2.GaussianBlur(image, blur_kernel, 0)

    rows, cols = image.shape
    noise = np.random.normal(mean, std, (rows, cols))

    # Générer un masque aléatoire pour contrôler la fréquence du bruit
    mask = np.random.rand(rows, cols) < noise_freq

    # Appliquer le bruit seulement aux pixels sélectionnés par le masque
    image_with_noise = blurred_image.copy()
    image_with_noise[mask] += noise[mask]

    # S'assurer que les valeurs restent dans les limites acceptables pour une image et ne sont pas négatives
    image_with_noise = np.clip(image_with_noise, 0, 255)

    return image_with_noise.astype(image.dtype)

def evaluate_flow_from_d(estimated_flow, dx, dy):
    """
    Évalue la qualité d'un champ de vecteurs de flux optique estimé par rapport à des déplacements dx et dy attendus.
    
    Args:
    - estimated_flow: np.ndarray, champ de vecteurs de flux optique estimé.
    - dx: float, déplacement attendu en x.
    - dy: float, déplacement attendu en y.
    
    Returns:
    - mean_diff: float, la différence moyenne entre la moyenne des vecteurs estimés et le déplacement attendu.
    - std_deviation: float, l'écart-type des vecteurs estimés par rapport à la moyenne.
    """
    # Créer des masques pour les éléments non nuls
    nonzero_mask_x = estimated_flow[..., 0] != 0
    nonzero_mask_y = estimated_flow[..., 1] != 0
    nonzero_mask = nonzero_mask_x & nonzero_mask_y

    # Calcul de la moyenne des vecteurs de flux optique estimés en excluant les zéros
    mean_estimated_dx = np.mean(estimated_flow[nonzero_mask, 0])
    mean_estimated_dy = np.mean(estimated_flow[nonzero_mask, 1])
    
    # Calcul de la différence moyenne par rapport aux déplacements attendus
    mean_diff_dx = mean_estimated_dx - dx
    mean_diff_dy = mean_estimated_dy - dy
    mean_diff = np.sqrt(mean_diff_dx**2 + mean_diff_dy**2)
    
    # Calcul de l'écart-type des vecteurs estimés en excluant les zéros
    deviation_dx = estimated_flow[nonzero_mask, 0] - mean_estimated_dx
    deviation_dy = estimated_flow[nonzero_mask, 1] - mean_estimated_dy
    std_deviation = np.sqrt(np.mean(deviation_dx**2 + deviation_dy**2))
    
    return mean_diff, std_deviation

def lissage_image(image, dx, dy,coeff):
    """
    Déplace une image par un déplacement uniforme (dx, dy) avec interpolation.

    Args:
    - image : np.ndarray, l'image originale à déplacer.
    - dx : float, le déplacement horizontal de l'image.
    - dy : float, le déplacement vertical de l'image.
    - Coefficient de lissage
    Returns:
    - shifted_image : np.ndarray, l'image déplacée.
    """
    shifted_image = np.copy(image)
    # Création des coordonnées de grille pour l'image originale
    for i in range(coeff):
        height, width = shifted_image.shape
        yy, xx = np.mgrid[0:height, 0:width]
        # Appliquer le déplacement
        shifted_yy = yy - (dy/coeff)
        shifted_xx = xx - (dx/coeff)
        # Interpolation des nouvelles valeurs de pixels
        shifted_image = shifted_image + map_coordinates(image, [shifted_yy.ravel(), shifted_xx.ravel()], order=3, mode='reflect').reshape(height, width)
    return (shifted_image/coeff)

def shift_image(image, dx, dy):
    """
    Déplace une image par un déplacement uniforme (dx, dy) avec interpolation.

    Args:
    - image : np.ndarray, l'image originale à déplacer.
    - dx : float, le déplacement horizontal de l'image.
    - dy : float, le déplacement vertical de l'image.
    Returns:
    - shifted_image : np.ndarray, l'image déplacée.
    """
    shifted_image = np.copy(image)
    # Création des coordonnées de grille pour l'image originale
    height, width = shifted_image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    # Appliquer le déplacement
    shifted_yy = yy - dy
    shifted_xx = xx - dx
    # Interpolation des nouvelles valeurs de pixels
    shifted_image = map_coordinates(image, [shifted_yy.ravel(), shifted_xx.ravel()], order=3, mode='reflect').reshape(height, width)
    return shifted_image
    
def analyze_optical_flow_vectors(flow):
    # Séparation des composantes x et y
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    # Calcul des statistiques pour les composantes x et y
    stats_x = {'mean': np.mean(flow_x), 'median': np.median(flow_x),
               'std': np.std(flow_x), 'max': np.max(flow_x), 'min': np.min(flow_x)}
    stats_y = {'mean': np.mean(flow_y), 'median': np.median(flow_y),
               'std': np.std(flow_y), 'max': np.max(flow_y), 'min': np.min(flow_y)}

    # Affichage des statistiques
    print("Composante X:", stats_x)
    print("Composante Y:", stats_y)

    # Visualisation avec des histogrammes
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(flow_x.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Distribution des composantes X')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')

    plt.subplot(1, 2, 2)
    plt.hist(flow_y.flatten(), bins=50, color='red', alpha=0.7)
    plt.title('Distribution des composantes Y')
    plt.xlabel('Valeur')
    plt.ylabel('Fréquence')

    plt.tight_layout()
    plt.show()
    plt.close()

def calculate_extreme_flow(flow):
    extreme_x, extreme_y = 0, 0
    extreme_norm = 0  # Pour garder la trace de la norme la plus élevée

    # Parcourir chaque bloc de vecteurs
    for vec in flow:
        # Parcourir chaque vecteur dans le bloc
        for v in vec:
            x, y = v  # Extraction des composantes du vecteur
            current_norm = np.sqrt(x**2 + y**2)
            # Comparer la norme actuelle avec la norme extrême enregistrée
            if current_norm > extreme_norm:
                extreme_x, extreme_y = x, y
                extreme_norm = current_norm

    return extreme_x, extreme_y
            
    # Utiliser le max ou le min en fonction du signe de la médiane
    """
    if median_x > 0:
       extreme_x = np.max(flow_x)
    else:
       extreme_x = np.min(flow_x)
    
    if median_y > 0:
       extreme_y = np.max(flow_y)
    else:
        extreme_y = np.min(flow_y)
    """
    
    return extreme_x, extreme_y


# Pour la longitude, il faut connaître la latitude pour chaque point
# Nous allons créer une fonction pour cela
def deg_per_pixel_lon(lat):
    # Rayon moyen de la Terre en mètres
    earth_radius = 6371000
    return (1000 / (earth_radius * np.cos(np.deg2rad(lat)))) * (180 / np.pi)

def param_vs_displacement(param, best_param_list, range_dx= 7, range_dy= 7):
        """
        Trace un paramètre en fonction du déplacement total (dx + dy).
    
        Args:
        - param: Le nom du paramètre à tracer (str).
        - best_param_list: Liste contenant les meilleurs paramètres pour chaque combinaison de dx et dy.
        - range_dx: Plage des valeurs de dx utilisées lors de l'optimisation.
        - range_dy: Plage des valeurs de dy utilisées lors de l'optimisation.
        """
        displacement_values = []
        param_values = []
    
        # Construire les listes des déplacements totaux et des valeurs du paramètre correspondantes
        for dx in range(1,range_dx):
            for dy in range(1,range_dy):
                index = (dx - 1) * (range_dy - 1) + (dy - 1)
                best_params = best_param_list[index]
                displacement = dx + dy
                parameter = best_params.get(param)
                
                displacement_values.append(displacement)
                param_values.append(parameter)
    
        # Création du graphique
        filename = fr"C:\Users\DELL\Desktop\Jupyter\Projet_Sargalert\Notebooks\plot\Graphique_param_{param}_of.png"
        plt.figure(figsize=(10, 6))
        plt.scatter(displacement_values, param_values, color='blue', label=f'Optimal {param}')
        plt.title(f'Impact du déplacement total sur {param} optimal')
        plt.xlabel('Déplacement total (dx + dy)')
        plt.ylabel(f'Optimal {param}')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename, bbox_inches='tight', dpi=100)
        plt.show()
    
#==========SIFT METHOD========
def filter_optical_flow_with_keypoints(flow, matches):
    if matches and len(matches) > 0:
        # Création d'un masque de la même taille que le champ de vecteurs, initialisé à False
        keypoints_mask = np.zeros(flow.shape[:2], dtype=bool)
        
        # Marquage des positions des points clés dans le masque
        for pt_src, pt_dst in matches:
            x, y = pt_dst  # Utilisation des points destination comme référence
            keypoints_mask[int(y), int(x)] = True
    
        # Création d'un nouveau champ de vecteurs filtré
        filtered_flow = np.zeros_like(flow)
        filtered_flow[keypoints_mask] = flow[keypoints_mask]
    else :
        filtered_flow = np.zeros_like(flow)
    return filtered_flow
    
def match_keypoints_SIFT(image1, image2,color=False):
    if color :
        # Convertir les images en gris
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialiser SIFT detector
    sift = cv2.SIFT_create()

    # Détecter les points clés et calculer les descripteurs
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Matcher les descripteurs avec un matcher FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Appliquer le test de ratio pour filtrer les bons matchs
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    # Dessiner les matchs
    match_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(match_img)
    plt.show()

    return keypoints1, keypoints2, good_matches


def build_filename(date, template):
    """Build a file name according to a date and a template.

    Parameters
    ----------
    date : Datetime.date
        The date of the wanted file.
    template : String
        The template of the name, all date, month and year substring will be replace.

    Returns
    -------
    String
        The resulting filename.

    """
    name = template.replace("DD", f"{date.day:02}")
    name = name.replace("MM", f"{date.month:02}")
    name = name.replace("YYYY", f"{date.year}")
    return name


def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def add_coloc_field(dataframe, colocalisations):
    dataframe["coloc"] = 0
    for i, row in dataframe.iterrows():
        id = row["id"]
        date = row["date"]
        if np.any((colocalisations["id"] == id) & (colocalisations["date"] == date)):
            dataframe.loc[i, "coloc"] = 1
    return dataframe


def date_from_1993(seconds):
    origin = datetime(1993, 1, 1)
    obstime = origin + timedelta(seconds=seconds)
    return obstime


def compute_azimuth_from_vectors(vec1, vec2):
    x1, y1 = vec1
    x2, y2 = vec2
    angle = - np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2) * 180 / np.pi
    return angle


def compute_mean_drift(flow, frac=0.5):
        nb_vec = np.sum(~np.isnan(flow[..., 0]))
        vecs = flow[~np.isnan(flow)]
        vecs = vecs.reshape(nb_vec, 2)
        nb_vec_to_consider = int(nb_vec * frac)
        start_barycenter = np.array([[np.nanmean(flow[..., 0]), np.nanmean(flow[..., 1])]])
        kdtree = KDTree(vecs, leaf_size=30, metric='euclidean')
        query = kdtree.query(start_barycenter, k=nb_vec_to_consider, return_distance=False)
        coherent_vecs = vecs[query]
        drift = np.mean(coherent_vecs, axis=1)[0]
        return drift
#=======Old fonction=====

    
def create_ellipse_patch_with_irregularities(shape, center, axes_length, angle, mean_value=0.00043, std_value=0.00026, smoothness=2, irregularity_scale=0.1, fill_holes_scale=0.2, seed=None, blur=False):
    """
    Crée un patch elliptique avec un contour irrégulier et des valeurs variant à l'intérieur.

    Args:
    - shape: Tuple, la forme de l'image sur laquelle le patch sera appliqué.
    - center: Tuple, le centre de l'ellipse (y, x).
    - axes_length: Tuple, la longueur des axes de l'ellipse (axe_majeur, axe_mineur).
    - angle: Float, l'angle de rotation de l'ellipse en degrés.
    - mean_value: Float, la valeur moyenne des indices AFAI dans le patch.
    - std_value: Float, la valeur de l'écart-type des indices AFAI dans le patch.
    - smoothness: Int, le degré de lissage des contours de l'ellipse.
    - irregularity_scale: Float, échelle des irrégularités ajoutées au contour.
    - fill_holes_scale: Float, échelle de la "densité" des trous à l'intérieur de l'ellipse.

    Returns:
    - patch: Un tableau Numpy représentant l'image avec le patch de sargasse.
    """
    if seed is not None:
        np.random.seed(seed)
        
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = center
    a, b = axes_length

    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    x_ = x - center_x
    y_ = y - center_y
    x_rot = cos_angle * x_ + sin_angle * y_
    y_rot = -sin_angle * x_ + cos_angle * y_
    
    mask = ((x_rot**2) / (a**2)) + ((y_rot**2) / (b**2)) <= 1

    rr, cc = ellipse_perimeter(center_y, center_x, a, b, orientation=-angle_rad)
    rr = np.clip(rr + np.random.randint(-irregularity_scale*a, irregularity_scale*a, rr.shape), 0, shape[0]-1)
    cc = np.clip(cc + np.random.randint(-irregularity_scale*b, irregularity_scale*b, cc.shape), 0, shape[1]-1)
    irregular_mask = np.zeros(shape, dtype=bool)
    irregular_mask[rr, cc] = True
    irregular_mask = gaussian_filter(irregular_mask.astype(float), sigma=smoothness)

    patch = np.random.normal(mean_value, std_value, shape) * mask
    patch = gaussian_filter(patch, sigma=smoothness) * (irregular_mask > fill_holes_scale)

    # Appliquer un flou sur le patch si blur=True
    if blur:
        kernel_size = (5, 5)
        sigma = 1.5
        patch = cv2.GaussianBlur(patch, kernel_size, sigma)

    return patch

def flow_ideal(flow,dx,dy,ang) :
    flow_ideal = np.zeros_like(flow)
    dx_p = dx * np.cos(ang) - dy * np.sin(ang)
    dy_p = dx * np.sin(ang) + dy * np.cos(ang)
    for i in range(flow.shape[0]):  # Itération sur la dimension y
        for j in range(flow.shape[1]):  # Itération sur la dimension x
            if flow[i, j, 0] != 0 or flow[i, j, 1] != 0:
                if flow[i, j, 0]> dx_p  :
                    flow_ideal[i, j, 0] = dx_p
                if flow[i, j, 1]> dy_p  :
                    flow_ideal[i, j, 0] = dy_p
                if flow[i, j, 0]<1 :
                    flow_ideal[i, j, 0] = 0
                if flow[i, j, 1]<1 :
                    flow_ideal[i, j, 1] = 0
                else : 
                    flow_ideal[i, j, 0] = flow[i, j, 0]
                    flow_ideal[i, j, 1] = flow[i, j, 1]
    return flow_ideal

#Evaluation de la qualité d'un champ de vecteur à partir d'un champ de vecteur idéal
def evaluate_flow(estimated_flow, ideal_flow):
    """
    Évalue la qualité d'un champ de vecteurs de flux optique estimé par rapport à un champ idéal.
    
    Args:
    - estimated_flow: np.ndarray, champ de vecteurs de flux optique estimé.
    - ideal_flow: np.ndarray, champ de vecteurs de flux optique idéal (de référence).
    
    Returns:
    - error: float, l'erreur moyenne entre les flux optiques estimé et idéal.
    - magnitude_error: float, l'erreur de magnitude moyenne entre les flux optiques estimé et idéal.
    """
    # Calcul de la différence entre les deux champs de flux optique
    flow_diff = estimated_flow - ideal_flow
    
    # Calcul de l'erreur euclidienne (magnitude de la différence des vecteurs)
    error_magnitude = np.sqrt(flow_diff[..., 0] ** 2 + flow_diff[..., 1] ** 2)
    
    # Erreur moyenne sur tous les vecteurs
    error = np.mean(error_magnitude)
    
    # Calcul de l'erreur de magnitude pour chaque vecteur
    estimated_magnitude = np.sqrt(estimated_flow[..., 0] ** 2 + estimated_flow[..., 1] ** 2)
    ideal_magnitude = np.sqrt(ideal_flow[..., 0] ** 2 + ideal_flow[..., 1] ** 2)
    magnitude_error = np.mean(np.abs(estimated_magnitude - ideal_magnitude))
    
    return error, magnitude_error

def divide_region(region):
    minr, minc, maxr, maxc = region['bbox']
    height = maxr - minr
    width = maxc - minc
    area = height * width

    if area <= 8000:
        return [region]

    new_regions = []

    if height > width:
        midr = (minr + maxr) // 2
        new_regions.extend(divide_region({'area': (midr - minr) * (maxc - minc), 'bbox': (minr, minc, midr, maxc)}))
        new_regions.extend(divide_region({'area': (maxr - midr) * (maxc - minc), 'bbox': (midr, minc, maxr, maxc)}))
    else:
        midc = (minc + maxc) // 2
        new_regions.extend(divide_region({'area': (maxr - minr) * (midc - minc), 'bbox': (minr, minc, maxr, midc)}))
        new_regions.extend(divide_region({'area': (maxr - minr) * (maxc - midc), 'bbox': (minr, midc, maxr, maxc)}))

    return new_regions


def download_goes_netcdf(start_date, end_date, output_dir):
    """
    Télécharge un fichier NetCDF par jour pour la période donnée.
    
    Arguments:
    start_date -- date de début au format 'YYYY-MM-DD'
    end_date -- date de fin au format 'YYYY-MM-DD'
    output_dir -- répertoire où les fichiers seront téléchargés
    """
    base_url = "https://dataterra:odatis@tds-odatis.aviso.altimetry.fr/thredds/ncss/dataset-sargassum-abi-goes-global-nrt-hr"
    params = {
        "var": "fai_anomaly",
        "north": 20,
        "west": -70,
        "east": -50,
        "south": 5,
        "disableProjSubset": "on",
        "horizStride": 1,
        "accept": "netcdf"
    }
    
    # Convertir les dates en objets datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parcourir chaque jour dans la période
    current_dt = start_dt
    while current_dt <= end_dt:
        time_start = current_dt.replace(hour=10, minute=0, second=0).isoformat() + "Z"
        time_end = current_dt.replace(hour=19, minute=0, second=0).isoformat() + "Z"
        
        # Ajouter les paramètres de temps à l'URL
        params["time_start"] = time_start
        params["time_end"] = time_end
        
        # Construire l'URL de la requête
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            file_name = f"goes_{current_dt.strftime('%Y%m%d')}.nc"
            file_path = os.path.join(output_dir, file_name)
            
            # Écrire le fichier téléchargé
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            print(f"Téléchargé: {file_name}")
        else:
            print(f"Erreur lors du téléchargement pour le {current_dt.strftime('%Y-%m-%d')}: {response.status_code}")
        
        # Passer au jour suivant
        current_dt += timedelta(days=1)

