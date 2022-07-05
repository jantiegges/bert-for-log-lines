from sentence_transformers import SentenceTransformer
from src.utils import save_embedding, load_embedding, load_logs
from sklearn.manifold import TSNE

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
        logs = load_logs(f"./logs/{log_file}")
        embedding = model.encode(logs, show_progress_bar=True)

        save_embedding(embedding, log_file, model_name, processed_flag, label_flag)
    
    return embedding

def get_tsne_embedding(embedding, n_dim, verbose=0, perplexity=30.0, n_iter=1000):
    """ computes and returns T-distributed Stochastic Neighbor Embedding for given dimension
    Params:
        embedding (Tensor): high dimension embedding
        n_dim (int): Dimension of the embedded space. 
        verbose (int): Verbosity level
        perplexity (float): The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms
        n_iter (int): Maximum number of iterations for the optimization. 
    Returns:
        tsne_results (array[n_samples, n_components]): Embedding of the training data in low-dimensional space.
    """
    # apply pca to reduce to lower dimension and then apply tsne?
    tsne = TSNE(n_components=n_dim, verbose=verbose, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(embedding)

    return tsne_results