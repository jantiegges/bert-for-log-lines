import os
import pickle

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