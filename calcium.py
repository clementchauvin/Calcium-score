# %pip install pydicom

import pydicom
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# +
def charger_volume_patient(dossier_patient):
    """
    Charge tous les fichiers DICOM d'un dossier et les assemble en 3D.
    """
    # 1. Lister les fichiers et les charger
    fichiers = [pydicom.dcmread(os.path.join(dossier_patient, f)) 
                for f in os.listdir(dossier_patient) if f.endswith('.dcm')]
    
    # 2. TRIER les coupes par position anatomique (crucial !)
    # On utilise l'attribut 'ImagePositionPatient' (le Z de la coupe)
    fichiers.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 3. Extraire les pixels et convertir en HU
    volume = []
    for ds in fichiers:
        img = ds.pixel_array.astype(float)
        slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
        intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
        volume.append(img * slope + intercept)
    
    # Transformer en array NumPy (Profondeur, Hauteur, Largeur)
    return np.array(volume)

# --- Exemple d'utilisation ---
# chemin_patient = "data/patient_001"
# volume_3d = charger_volume_patient(chemin_patient)
# print(f"Forme du volume : {volume_3d.shape}") 
# Résultat attendu : (60, 512, 512)


# +

def isoler_zone_coeur(volume_3d, taille_crop=256):
    """
    volume_3d: array de forme (Nb_coupes, 512, 512)
    Retourne un volume centré sur le médiastin.
    """
    # 1. Sélection des coupes (on garde le milieu du scan)
    # On prend 30 coupes autour du centre du volume
    nb_coupes = volume_3d.shape[0]
    centre_z = nb_coupes // 2
    debut_z = max(0, centre_z - 15)
    fin_z = min(nb_coupes, centre_z + 15)
    
    volume_coupe = volume_3d[debut_z:fin_z, :, :]
    
    # 2. Crop Spatial (Centrage sur le médiastin)
    # Sur une image 512x512, le coeur est souvent vers :
    # y (hauteur) : entre 150 et 400
    # x (largeur) : entre 128 et 384
    
    c_y, c_x = 220, 256  # Centre approximatif du médiastin
    r = taille_crop // 2 # Rayon du carré
    
    volume_final = volume_coupe[:, c_y-r:c_y+r, c_x-r:c_x+r]
    
    return volume_final

# Utilisation
# volume_coeur = isoler_zone_coeur(volume_patient_complet)
# print(volume_coeur.shape) # Doit afficher (2-, 256, 256)


# -

#Toutes les images n(ont pas le meme nombre de pixels do,nc marche pas
def smart_crop_coeur(ds, volume, taille_mm=(170,120)):
    """
    ds : l'objet pydicom d'une des coupes (pour avoir les métadonnées)
    volume_3d : le volume complet du patient
    taille_mm : la taille du carré en millimètres (200mm = 20cm est souvent suffisant)
    """

    # 1. Sélection des coupes (on garde le milieu du scan)
    # On prend 30 coupes autour du centre du volume
    nb_coupes = volume.shape[0]
    centre_z = nb_coupes // 2
    debut_z = max(0, centre_z - 15)
    fin_z = min(nb_coupes, centre_z + 15)
    
    volume_3d = volume[debut_z:fin_z, :, :]
    
    # 1. Obtenir la taille réelle d'un pixel (ex: [0.75, 0.75])
    pixel_spacing = ds.PixelSpacing 
    ps_y, ps_x = float(pixel_spacing[0]), float(pixel_spacing[1])
    
    # 2. Convertir la taille cible de mm vers pixels
    taille_pixels_y = int(taille_mm[1] / ps_y)
    taille_pixels_x = int(taille_mm[0] / ps_x)
    
    # 3. Trouver le centre du corps du patient
    # On prend la coupe du milieu, on seuille pour isoler le corps (HU > -500)
    coupe_milieu = volume_3d[len(volume_3d)//2]
    masque_corps = coupe_milieu > -500 
    
    # Calcul du centre de masse des pixels du corps
    coords = np.argwhere(masque_corps)
    centre_y, centre_x = coords.mean(axis=0).astype(int)
    
    # 4. Ajustement Anatomique : 
    # Le coeur est TOUJOURS au-dessus (antérieur) du centre de gravité
    # On remonte le centre du crop d'environ 20-30mm vers le haut du scanner
    decalage_haut = int(60 / ps_y) 
    centre_y_coeur = centre_y - decalage_haut
    
    # 5. Découpe (Crop)
    ry, rx = taille_pixels_y // 2, taille_pixels_x // 2
    
    # On s'assure de ne pas sortir des limites de l'image
    y1, y2 = max(0, centre_y_coeur-ry), min(volume_3d.shape[1], centre_y_coeur+ry)
    x1, x2 = max(0, centre_x-rx), min(volume_3d.shape[2], centre_x+rx)
    
    volume_final = volume_3d[:, y1:y2, x1:x2]
    
    return volume_final


volume_cadre_33 = smart_crop_coeur(pydicom.dcmread("P59_1.dcm"), charger_volume_patient("P59"))


# +
def verifier_crop(volume_original, volume_isole):
    # On choisit la coupe du milieu du nouveau volume pour vérifier
    coupe_index = volume_isole.shape[0] // 2
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. Affichage de l'original (on peut dessiner le rectangle du crop)
    ax[0].imshow(volume_original[volume_original.shape[0]//2], cmap='gray', vmin=-100, vmax=400)
    ax[0].set_title(f"Original (Full) - Coupe milieu")
    ax[0].axis('off')
    
    # 2. Affichage du crop (Zone Coeur)
    # On utilise la même fenêtre HU (-100 à 400) pour voir les tissus mous
    im = ax[1].imshow(volume_isole[15], cmap='gray', vmin=-100, vmax=400)
    ax[1].set_title(f"Crop Coeur - Coupe {coupe_index}")
    ax[1].axis('off')
    
    plt.colorbar(im, ax=ax[1], label="Unités Hounsfield (HU)")
    plt.tight_layout()
    plt.show()





















#pour smart crop
def preparer_et_cadrer_patient_smart(chemin_dossier, taille_h_mm=180, taille_w_mm=240):
    """Charge, trie et applique le crop intelligent sur un dossier patient."""
    # 1. Lecture et tri
    fichiers = [pydicom.dcmread(os.path.join(chemin_dossier, f)) 
                for f in os.listdir(chemin_dossier) if f.endswith('.dcm')]
    fichiers.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 2. Conversion HU et création du volume
    volume_hu = []
    for ds in fichiers:
        img = ds.pixel_array.astype(np.float32) # float32 pour gagner de la place
        slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
        intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
        volume_hu.append(img * slope + intercept)
    
    volume_3d = np.array(volume_hu)
    
    # 3. Utilisation de la fonction smart_crop (adaptée pour le rectangle)
    # On passe le ds de la première coupe pour les métadonnées
    volume_cadre = smart_crop_coeur(fichiers[0], volume_3d, taille_h_mm, taille_w_mm)
    
    return volume_cadre



def preparer_et_cadrer_patient(chemin_dossier):
    
    
    # 2. Conversion HU et création du volume
    volume_hu = []
    for ds in fichiers:
        img = ds.pixel_array.astype(np.float32) # float32 pour gagner de la place
        slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
        intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
        volume_hu.append(img * slope + intercept)
    
    volume_3d = np.array(volume_hu)
    
    # 3. Utilisation de la fonction i (adaptée pour le rectangle)
    
    volume_cadre = isoler_zone_coeur(volume_3d)

# +
# --- La Boucle Principale ---

dossier_source = "Patients" # Le dossier contenant P1, P2...
dossier_destination = "data_prete"

if not os.path.exists(dossier_destination):
    os.makedirs(dossier_destination)

# On liste tous les dossiers patients (1, 2, ...)
patients_folders_intermediare = [f for f in os.listdir(dossier_source) if f.isdigit()]
patients_folders = []
for patient_id in patients_folders_intermediare:
    # Chemin vers le premier dossier (ex: Patients/82)
    chemin_niveau_1 = os.path.join(dossier_source, patient_id)
    
    # Chemin vers le second dossier (ex: Patients/82/82)
    chemin_final_dicom = os.path.join(chemin_niveau_1, patient_id)
    patients_folders.append(chemin_final_dicom)



print(f"Lancement du traitement pour {len(patients_folders)} patients...")

for chemin_patient in patients_folders:
    volume = charger_volume_patient(chemin_patient)
    
    #  Lister tous les fichiers .dcm
    fichiers = [f for f in os.listdir(chemin_patient) if f.endswith('.dcm')]
    # 2. Trier par nom (ex: image1.dcm, image2.dcm...)
    fichiers.sort()
    # 3. Prendre le premier de la liste
    premier_fichier_nom = fichiers[0]
    # 4. Lire le fichier
    chemin_complet = os.path.join(chemin_patient, premier_fichier_nom)
    ds = pydicom.dcmread(chemin_complet)
    
    volume_final = isoler_zone_coeur(volume)

    
  
    
    # Sauvegarde individuelle (ex: P1.npy)
    # Si le chemin est 'Patients/82/82', os.path.basename donne '82'.
    p_id = os.path.basename(os.path.normpath(chemin_patient))
    nom_fichier = os.path.join(dossier_destination, f"{p_id}.npy")
    np.save(nom_fichier, volume_final)
        
       
   
print("\nTraitement terminé ! Toutes vos images sont cadrées et prêtes.")
# -









# +

# --- CONFIGURATION ---
dossier_npy = "/Users/locchauvinc/cours-info/data_prete/"
fichier_excel = "scores.xlsx"

df = pd.read_excel(fichier_excel)

# On nettoie les noms de colonnes (enlève les espaces comme dans 'filename ')
df.columns = df.columns.str.strip()

X_liste = []
y_liste = []

print(f"Recherche des fichiers dans : {dossier_npy}")

for index, row in df.iterrows():
    # 1. On transforme l'ID en entier puis en texte pour éviter le "1.0.npy"
    valeur_id = row['filename']
    id_patient = str(int(valeur_id)) 
    
    nom_fichier = f"{id_patient}.npy"
    chemin = os.path.join(dossier_npy, nom_fichier)
    
    if os.path.exists(chemin):
        vol = np.load(chemin).astype(np.float32)
        X_liste.append(vol.flatten() / 1000.0)
        
        # 2. On utilise 'total' qui est le nom de ta colonne Excel
        y_liste.append(row['total']) 
    else:
        # Debug : on affiche pourquoi ça ne marche pas pour les premiers
        if index < 5:
            print(f"❌ Introuvable : {nom_fichier}")

# --- FINALISATION ---
if len(y_liste) > 0:
    # Diagnostic rapide
    tailles_uniques = set([vol.shape[0] for vol in X_liste])
    print(f"Tailles de vecteurs trouvées : {tailles_uniques}")
    X = np.array(X_liste)
    y = np.array(y_liste).reshape(-1, 1)
    
    # On gère les scores vides (NaN) s'il y en a dans l'Excel
    y = np.nan_to_num(y) 
    
    score_max = y.max()
    y_norm = y / (score_max if score_max != 0 else 1)
    
    print(f"✅ Succès ! {len(X)} patients chargés.")
    print(f"Taille de X : {X.shape}")
else:
    print("⚠️ Toujours aucun fichier trouvé. Vérifie si tes fichiers .npy sont bien nommés '1.npy', '2.npy', etc.")


# +
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


# +
def hidden(X, W1, b1):
    return relu(np.dot(W1, X) + b1)

def output(h, W2, b2):
    return relu(np.dot(W2.T, h) + b2)
# +
# --- PARAMÈTRES ---
input_size = X.shape[1] # ex: 1 296 000
hidden_size = 64        # Nombre de neurones dans la couche cachée
learning_rate = 1e-7    # Très petit pour éviter les erreurs 'NaN'
epochs = 100            # Nombre de passages sur tout le dataset

# --- INITIALISATION DES POIDS ---
# He Initialization : on ne met pas de zéros sinon le réseau n'apprend rien
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, 1))

print(f"Début de l'entraînement sur {len(X)} patients...")

for epoch in range(epochs):
    epoch_loss = 0
    
    # On mélange les patients à chaque époque
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for i in indices:
        # --- 1. FORWARD PASS (Prédiction) ---
        xi = X[i:i+1] # Un seul patient (1, pixels)
        yi = y_norm[i:i+1]
        
        # Couche cachée
        z1 = np.dot(xi, W1) + b1
        a1 = relu(z1)
        
        # Couche de sortie
        prediction = np.dot(a1, W2) + b2
        
        # Calcul de l'erreur (MSE)
        loss = (prediction - yi)**2
        epoch_loss += loss[0][0]
        
        # --- 2. BACKWARD PASS (Calcul de la correction) ---
        # Dérivée par rapport à la sortie
        d_prediction = 2 * (prediction - yi)
        
        # Gradients pour la couche de sortie (W2, b2)
        dW2 = np.dot(a1.T, d_prediction)
        db2 = d_prediction
        
        # Propagation vers la couche cachée
        da1 = np.dot(d_prediction, W2.T)
        dz1 = da1 * relu_deriv(z1)
        
        # Gradients pour la couche d'entrée (W1, b1)
        dW1 = np.dot(xi.T, dz1)
        db1 = dz1
        
        # --- 3. MISE À JOUR DES POIDS ---
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
    if (epoch + 1) % 10 == 0:
        print(f"Époque {epoch+1}/{epochs} | Erreur Moyenne: {epoch_loss/len(X):.6f}")

print("Entraînement terminé !")
# -




