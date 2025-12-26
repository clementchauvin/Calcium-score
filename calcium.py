# %pip install pydicom

import pydicom
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


#DATA FORMATINg
def charger_volume_patient(dossier_patient):#To load all DICOM files of a patient and assemble them in 3D.
   
    # 1. List file and load them
    fichiers = [pydicom.dcmread(os.path.join(dossier_patient, f)) 
                for f in os.listdir(dossier_patient) if f.endswith('.dcm')]
    
    # 2. sort images by anatomic position
    
    fichiers.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 3. Convert pixels in HU
    volume = []
    for ds in fichiers:
        img = ds.pixel_array.astype(float)
        slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
        intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
        volume.append(img * slope + intercept)
    
    # Transform in array NumPy 
    return np.array(volume)




def isoler_zone_coeur(volume_3d, taille_crop=256):#To isolate the heart (keep only the images and the part of the CT's that are interesting
   
    # 1. Select the CT's with the heart
    
    nb_coupes = volume_3d.shape[0]
    centre_z = nb_coupes // 2
    debut_z = max(0, centre_z - 15)
    fin_z = min(nb_coupes, centre_z + 15)
    
    volume_coupe = volume_3d[debut_z:fin_z, :, :]
    
    # 2. Spatial Crop  (centering on the mediastinum)
    
    
    c_y, c_x = 220, 256  # Center of the heart
    r = taille_crop // 2 # radius of the square 
    
    volume_final = volume_coupe[:, c_y-r:c_y+r, c_x-r:c_x+r]
    
    return volume_final






def verifier_crop(volume_original, volume_isole):#To check on images some cuts done by the last function
    # On choisit la coupe du milieu du nouveau volume pour vérifier
    coupe_index = volume_isole.shape[0] // 2
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1. display the original
    ax[0].imshow(volume_original[volume_original.shape[0]//2], cmap='gray', vmin=-100, vmax=400)
    ax[0].set_title(f"Original (Full) - Coupe milieu")
    ax[0].axis('off')
    
    # 2. display the cut image
    # On utilise la même fenêtre HU (-100 à 400) pour voir les tissus mous
    im = ax[1].imshow(volume_isole[15], cmap='gray', vmin=-100, vmax=400)
    ax[1].set_title(f"Crop Coeur - Coupe {coupe_index}")
    ax[1].axis('off')
    
    plt.colorbar(im, ax=ax[1], label="Unités Hounsfield (HU)")
    plt.tight_layout()
    plt.show()


























#  Main loop

dossier_source = "Patients" # The file with P1, P2...
dossier_destination = "data_ready"

if not os.path.exists(dossier_destination):
    os.makedirs(dossier_destination)

# listing all the patients' file (1, 2, ...)
patients_folders_intermediare = [f for f in os.listdir(dossier_source) if f.isdigit()]
patients_folders = []
for patient_id in patients_folders_intermediare:
    # Path to the first file (ex: Patients/82)
    chemin_niveau_1 = os.path.join(dossier_source, patient_id)
    
    # Path to the second file (ex: Patients/82/82)
    chemin_final_dicom = os.path.join(chemin_niveau_1, patient_id)
    patients_folders.append(chemin_final_dicom)



print(f"Lancement du traitement pour {len(patients_folders)} patients...")

for chemin_patient in patients_folders:
    volume = charger_volume_patient(chemin_patient)
    
    volume_final = isoler_zone_coeur(volume)

    
  
    
    # Save Patient (ex: P1.npy)
    p_id = os.path.basename(os.path.normpath(chemin_patient))
    nom_fichier = os.path.join(dossier_destination, f"{p_id}.npy")
    np.save(nom_fichier, volume_final)
           
print("\nTraitement terminé ! Toutes vos images sont cadrées et prêtes.")








#  Prepare the entries and the expected results normalized
dossier_npy = "/Users/locchauvinc/cours-info/data_prete/"
fichier_excel = "scores.xlsx"

df = pd.read_excel(fichier_excel)

X_liste = []
y_liste = []

print(f"Recherche des fichiers dans : {dossier_npy}")

for index, row in df.iterrows():
    
    valeur_id = row['filename']
    id_patient = str(int(valeur_id)) 
    
    nom_fichier = f"{id_patient}.npy"
    chemin = os.path.join(dossier_npy, nom_fichier)
    
    if os.path.exists(chemin):
        vol = np.load(chemin).astype(np.float32)
        X_liste.append(vol.flatten() / 1000.0)
        
        # total is the column with the calcium score of patients
        y_liste.append(row['total']) 
    else:
        # Print unfound files
        print(f" Introuvable : {nom_fichier}")

 


size_vectors = set([vol.shape[0] for vol in X_liste])
print(f"Size of vectors X : {size_vectors}")
X = np.array(X_liste)
y = np.array(y_liste).reshape(-1, 1)
    
#if there are NaN in the excel
y = np.nan_to_num(y) 

#normalization
score_max = y.max()
y_norm = y / (score_max if score_max != 0 else 1)
    
print(f" Success ! {len(X)} loaded patients.")
print(f"Size of X : {X.shape}")


#END OF DATA FORMATING






#NEAURAL NETWORK



def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)



def hidden(X, W1, b1):
    return relu(np.dot(W1, X) + b1)

def output(h, W2, b2):
    return(np.dot(W2.T, h) + b2)



#  HYPERPARAMETERS 
input_size = X.shape[1] 
hidden_size = 64        
learning_rate = 0.0001    
epochs = 50            

#  INITIALISATION of weights and biases 

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, 1))

print(f"Début de l'entraînement sur {len(X)} patients...")

for epoch in range(epochs):
    epoch_loss = 0
    
    # Shuffle patients for each epoch
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for i in indices:
        #  1. FORWARD  
        xi = X[i:i+1] 
        yi = y_norm[i:i+1]
        
        # hidden layer
        z1 = np.dot(xi, W1) + b1
        a1 = relu(z1)
        
        # output
        prediction = np.dot(a1, W2) + b2
        
        # MSE
        loss = (prediction - yi)**2
        epoch_loss += loss[0][0]
        
        #  2. BACKWARDPROPAGATION 
        # Dérivée par rapport à la sortie
        d_prediction = 2 * (prediction - yi)
        
        # Gradiants 
        dW2 = np.dot(a1.T, d_prediction)
        db2 = d_prediction
        
        da1 = np.dot(d_prediction, W2.T)
        dz1 = da1 * relu_deriv(z1)
        
        dW1 = np.dot(xi.T, dz1)
        db1 = dz1
        
        # 3. UPDATES 
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
    
    print(f"Epoch {epoch+1}/{epochs} | Mean error: {epoch_loss/len(X):.6f}")

print("Training completed : Parameters are rrady to be used !!!!")






