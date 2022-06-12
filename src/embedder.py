from sentence_transformers import SentenceTransformer
from src.utils import save_embedding, load_embedding, load_logs
import os

def get_embedding(log_file: str, model_name: str, processed_flag=True, label_flag=False):
    """ Loads embedding if exists or computes and saves it
    Params:
        log_file (str): name of the log file
        model_name (str): name of the used sentence transformer model
        processed_flag (bool): flag if processed or unprocessed embeddings should be used
        label_flag (bool): flag if labelled or unlabelled log lines should be used
    Returns:
        embedding (Tensor): Tensor with embedded log lines
    """
    # try to load existing embedding, else compute and save
    try:
        embedding = load_embedding(log_file, model_name, processed_flag, label_flag)
    except:
        model = SentenceTransformer(model_name)
        # TODO: look at model.encode options (https://www.sbert.net/examples/applications/computing-embeddings/README.html?highlight=encode())
        logs = load_logs(f"./logs/{log_file}")
        embedding = model.encode(logs, show_progress_bar=True)

        save_embedding(embedding, log_file, model_name, processed_flag, label_flag)
    
    return embedding