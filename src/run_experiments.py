import time
import subprocess

# ==========================================
# CONFIGURAZIONE ESPERIMENTI
# ==========================================
# Inserisci qui l'elenco di TUTTI gli IP PRIVATI dei tuoi Worker accesi.
# Assicurati che su ognuna di queste macchine stia girando 'python src/worker.py 50051'
WORKER_IPS = [
    "172.31.27.191",  # Worker 1
    "172.31.24.101",  # Worker 2
    "172.31.19.131",  # Worker 3
    "172.31.27.22",   # Worker 4
    "172.31.22.56"    # Worker 5
    # Aggiungi qui eventuali altri IP se usi 6, 7, 10 worker...
]

PORT = "50051"
TREES = 50
DATASETS = ["higgs", "taxi"]

def run_campaign():
    print("🚀 INIZIO CAMPAGNA DI TEST AUTOMATIZZATA 🚀")
    
    for dataset in DATASETS:
        print(f"\n{'='*55}")
        print(f" 🧪 TEST SCALABILITÀ: DATASET {dataset.upper()}")
        print(f"{'='*55}")
        
        # Prova con 1 worker, poi 2, poi 3, fino al massimo disponibile
        for num_workers in range(1, len(WORKER_IPS) + 1):
            print(f"\n---> Avvio test con {num_workers} Worker(s) <---")
            
            # Seleziona i primi 'num_workers' dalla lista
            current_ips = WORKER_IPS[:num_workers]
            
            # Costruisci la stringa "IP:PORTA IP:PORTA ..."
            worker_args = " ".join([f"{ip}:{PORT}" for ip in current_ips])
            
            # Costruisci il comando bash
            cmd = f"python src/master.py --dataset {dataset} --workers {worker_args} --trees {TREES}"
            print(f" Esecuzione comando:\n   {cmd}")
            
            # Esegue il comando e aspetta che finisca
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"ERRORE DURANTE IL RUN CON {num_workers} WORKER: {e}")
                print("Passo al prossimo test...")
            
            # Pausa fondamentale tra un test e l'altro (15 secondi)
            # Permette alle RAM delle macchine di svuotarsi e ai socket di rete di chiudersi correttamente
            print(f"\n⏳ Test completato. Pausa di 15 secondi per pulizia risorse...")
            time.sleep(15)

    print("\n CAMPAGNA DI TEST CONCLUSA CON SUCCESSO! Controlla il bucket S3 per i file CSV.")

if __name__ == "__main__":
    run_campaign()
