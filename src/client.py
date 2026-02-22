import boto3
import json
import argparse
import uuid

# HARDCODED! DA MODIFICARE?
QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/248593862537/JobRequestQueue.fifo"

def send_training_request(dataset, workers, trees):
    sqs = boto3.client('sqs', region_name='us-east-1')
    
    # Creiamo il "bigliettino" con le istruzioni per il Master
    messagge = {
        "dataset": dataset,
        "workers": workers,
        "trees": trees
    }
    
    try:
        # Invio del messaggio alla coda SQS
        response = sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(messagge),
            MessageGroupId="ML_Training_Jobs",
            MessageDeduplicationId=str(uuid.uuid4())
        )
        print(f" Richiesta inviata con successo a SQS!")
        print(f" Message ID: {response['MessageId']}")
        print(f" Dettagli: Dataset={dataset}, Alberi={trees}, Workers={len(workers)}")
    except Exception as e:
        print(f" Errore nell'invio del messaggio: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client per inviare Job di Training via SQS")
    parser.add_argument('--dataset', type=str, required=True, choices=['taxi', 'higgs'])
    parser.add_argument('--workers', nargs='+', required=True, help="Lista IP:Porta dei worker")
    parser.add_argument('--trees', type=int, default=10, help="Numero totale di alberi")
    
    args = parser.parse_args()
    send_training_request(args.dataset, args.workers, args.trees)
