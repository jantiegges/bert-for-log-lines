import os
import pickle
import numpy as np
from sentence_transformers import util

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_logs(file_name: str):
    """ Loads a log file into a list of strings 
    Params:
        file_name (str): name of the log file
    Returns:
        logs (List[str]): list with each line as one list element
    """
    with open(file_name) as f:
        logs = []
        for line in f:
            logs.append(line)
    return logs


def load_anomaly_logs(file_name: str):
    """ Loads only the anomaly logs of a log file into a list of strings 
    Params:
        file_name (str): name of the log file
    Returns:
        logs (List[str]): list with each line as one list element
    """
    with open(file_name) as f:
        logs = []
        for line in f:
            if not (line.startswith("-")):
                logs.append(line)
    return logs    
    

def save_embedding(data, log_file, model_name, processed_flag, label_flag):
    """ saves data to path in .pkl format
    Params:
        data (dict): data to save
        log_file (str): name of the log file
        model_name (str): name of the used sentence transformer model
        processed_flag (bool): flag if processed or unprocessed embeddings have been used
        label_flag (bool): flag if labelled or unlabelled log lines have been used
    """
    process_str = "processed" if processed_flag else "unprocessed"
    label_str = "labelled" if label_flag else "unlabelled"
    
    path = f"{THIS_DIR}/../embeddings/{log_file}_{process_str}_{label_str}_{model_name}"
    # write pickle representation of saved_data to file
    # write in highest protocol version --> best speed up
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_embedding(log_file, model_name, processed_flag, label_flag):
    """ loads a .pkl file from path
    Params:
        log_file (str): name of the log file
        model_name (str): name of the used sentence transformer model
        processed_flag (bool): flag if processed or unprocessed embeddings have been used
        label_flag (bool): flag if labelled or unlabelled log lines have been used
    Return:
        data: reconstituted object
    """
    process_str = "processed" if processed_flag else "unprocessed"
    label_str = "labelled" if label_flag else "unlabelled"
    
    path = f"{THIS_DIR}/../embeddings/{log_file}_{process_str}_{label_str}_{model_name}"

    # open in binary format for reading
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def get_cosine_distances(embedding, idc_anomaly, idc_no_anomaly):
    """ Computes the cosine distances for the entire embedding, normal to normal embedding, anomaly to anomaly embedding, normal to anomaly embedding
    Params:
        embedding: embedding
        idc_anomaly (list): all idc of lines with anomaly 
        idc_no_anomaly (list): all idc of lines with no anomaly 
    Returns:
        cos_scores_all, cos_scores_n2n, cos_scores_a2a, cos_scores_n2a (array[n, m]): cosine similarities
    """

    cos_scores = util.cos_sim(embedding, embedding)

    cos_scores_np = cos_scores.numpy()

    cos_scores_all = cos_scores_np
    cos_scores_n2n = cos_scores_np[np.ix_(idc_no_anomaly, idc_no_anomaly)]
    cos_scores_a2a = cos_scores_np[np.ix_(idc_anomaly, idc_anomaly)]
    cos_scores_n2a = cos_scores_np[np.ix_(idc_anomaly, idc_no_anomaly)]

    return cos_scores_all, cos_scores_n2n, cos_scores_a2a, cos_scores_n2a

def get_local_cosine_distances(embedding, idc_anomaly, idc_no_anomaly, epsilon):
    """ Computes the cosine distances for the entire embedding, normal to normal embedding, anomaly to anomaly embedding, normal to anomaly embedding
    Params:
        embedding: embedding
        idc_anomaly (list): all idc of lines with anomaly 
        idc_no_anomaly (list): all idc of lines with no anomaly 
        epsilon (int): number of lines before an after a line that should be taken into account
    Returns:
        cos_scores_all, cos_scores_n2n, cos_scores_a2a, cos_scores_n2a (list(array[n, m])): list of local cosine similarities for each line
    """

    cos_scores = util.cos_sim(embedding, embedding)

    cos_scores_np = cos_scores.numpy()

    cos_scores_all = []
    cos_scores_n2n = []
    cos_scores_a2a = []
    cos_scores_n2a = []

    start_idx = epsilon - 1

    for i in range(start_idx, cos_scores.shape[0] - epsilon - 1):
        idc_local = list(range(i - epsilon + 1, i + epsilon + 2))

        tmp = idc_local + idc_anomaly
        seen = set()
        idc_local_anomaly = [x for x in tmp if x in seen or seen.add(x)]  

        tmp = idc_local + idc_no_anomaly
        seen = set()
        idc_local_no_anomaly = [x for x in tmp if x in seen or seen.add(x)]  

        cos_scores_all.append(cos_scores_np[np.ix_([i], idc_local)])

        if i in idc_anomaly:
            cos_scores_a2a.append(cos_scores_np[np.ix_([i], idc_local_anomaly)])
            cos_scores_n2a.append(cos_scores_np[np.ix_([i], idc_local_no_anomaly)])
        else:
            cos_scores_n2n.append(cos_scores_np[np.ix_([i], idc_local_no_anomaly)])
            cos_scores_n2a.append(cos_scores_np[np.ix_([i], idc_local_anomaly)])


    return cos_scores_all, cos_scores_n2n, cos_scores_a2a, cos_scores_n2a


def get_distance_metrics(cos_scores):
    """ Computes min, max and mean distance of cosine similarities """
    min = cos_scores.min()
    max = cos_scores.max()
    mean = cos_scores.mean()

    return min, max, mean

def get_local_distance_metrics(cos_scores):
    """ Computes min, max and mean distance of cosine similarities for list of cos_scores """
    min_list = []
    max_list = []
    mean_list = []

    for i in cos_scores:
        try:
            min_list.append(i.min())
            max_list.append(i.max())
            mean_list.append(i.mean())
        except:
            pass

    min_list = np.asarray(min_list)
    max_list = np.asarray(max_list)
    mean_list = np.asarray(mean_list)

    min = min_list.min()
    max = max_list.max()
    mean = mean_list.mean()

    return min, max, mean