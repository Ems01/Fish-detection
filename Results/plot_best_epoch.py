import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory contenente i file CSV
csv_folder = 'Results'
csv_files = [os.path.join(csv_folder, f'results{i}.csv') for i in range(1, 6)]

# Creazione di una cartella per i grafici se non esiste
output_folder = os.path.join(csv_folder, 'Plots')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lista per memorizzare i DataFrame
dfs = []

# Caricamento di tutti i file CSV, rimuovendo gli spazi extra dai nomi delle colonne
for file in csv_files:
    print(f"Trying to load: {file}")  # Stampa il percorso del file per il debug
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()  # Rimuove spazi all'inizio e alla fine dai nomi delle colonne
    dfs.append(df)

# Calcolare l'epoca ottimale per ogni fold
for idx, df in enumerate(dfs):
    # Estrazione delle metriche necessarie
    mAP50 = df['metrics/mAP50(B)']
    val_box_loss = df['val/box_loss']
    val_cls_loss = df['val/cls_loss']
    precision = df['metrics/precision(B)']
    recall = df['metrics/recall(B)']

    # Coefficienti per il calcolo del punteggio combinato
    coeff_box_loss = 2.0  # Maggiore peso per le perdite di box
    coeff_cls_loss = 2.0  # Maggiore peso per le perdite di classe
    coeff_precision = 1.5  # Peso per la precisione
    coeff_recall = 1.5  # Peso per il recall

    # Calcola un punteggio combinato
    combined_score = (mAP50 * 1.0 +  # Utilizza mAP50 con un peso di 1
                      precision * coeff_precision + 
                      recall * coeff_recall - 
                      (val_box_loss * coeff_box_loss) - 
                      (val_cls_loss * coeff_cls_loss))

    # Trova l'epoca con il punteggio massimo
    best_epoch = combined_score.idxmax()
    best_score = combined_score.max()

    print(f"Fold {idx + 1}: Best epoch = {best_epoch + 1}, Combined Score = {best_score:.4f}")

    # Plotting per ogni fold
    plt.figure(figsize=(12, 6))

    # Plot delle metriche
    plt.plot(mAP50, label='mAP50', color='blue', linestyle='--')
    plt.plot(val_box_loss, label='Validation Box Loss', color='green', linestyle='-')
    plt.plot(val_cls_loss, label='Validation Class Loss', color='orange', linestyle='-')
    plt.plot(precision, label='Precision', color='purple', linestyle='-.')
    plt.plot(recall, label='Recall', color='red', linestyle=':')

    # Aggiungi la linea verticale tratteggiata per l'epoca migliore
    plt.axvline(x=best_epoch, color='red', linestyle=':', label='Best Epoch')

    # Aggiungi titolo, legenda e etichette
    plt.title(f'Fold {idx + 1}: mAP50, Validation Box Loss, Class Loss, Precision & Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Value')

    # Aggiorna la legenda per includere l'epoca migliore
    plt.legend(loc='upper right')
    plt.legend(title=f'Best Epoch: {best_epoch + 1}', title_fontsize='13')

    # Salva il grafico
    plt.savefig(os.path.join(output_folder, f'fold_{idx + 1}_combined_score.png'))
    plt.close()

print("All plots saved in the 'Results/Plots' folder.")
