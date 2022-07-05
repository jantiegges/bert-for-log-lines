from logparser import Drain
import csv


def pre_process(log_format, log_file, in_dir='./logs', out_dir='logparser-results/', depth=0.5, st=4):
    """ log line parser that saves result in csv file
    Params:
        log_format (str): Log format
        log_file (str): The input log file name
        in_dir (str): The input directory of log file
        out_dir (str): The output directory of parsing results
        depth (float): depth of all leaf nodes
        st (float): similarity threshold
    """

    # Regular expression list for optional preprocessing (default: [])
    regex      = [
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]

    parser = Drain.LogParser(log_format, indir=in_dir, outdir=out_dir,  depth=depth, st=st, rex=regex)
    parser.parse(log_file)


def delete_bgl_labels(log_file: str):
    """ gets rid of the labels of the bgl log lines and saves new csv
    Params:
        log_file (str): The input log file name
    """
    logs_no_label = []
    with open(f'logparser-results/{log_file}_structured.csv') as file:
        reader = csv.reader(file)

        for row in reader:
            line_no_label = row[2:]

            logs_no_label.append(line_no_label)

        logs_no_label.pop(0)

    with open(f'logparser-results/{log_file}_structured_no_labels.csv', 'w') as file:
        writer = csv.writer(file)

        for line in logs_no_label:
            writer.writerow(line)


def get_labels(log_file: str):
    """ gets labels from log datasets
    Params:
        log_file (str): The input log file name
    Returns:
        labels (list[str]): list with label of each line
    """

    labels = []

    with open(f'logparser-results/{log_file}_structured.csv') as file:
        reader = csv.reader(file)

        for row in reader:
            label = row[0]
            if row[1] != '-':
                label += ' - '
                label += row[1]

            labels.append(label)

        labels.pop(0)

    return labels


def get_binary_labels(labels: list):
    """ returns list with entry 0 for no anomaly and 1 for anomaly
    Params:
        labels (list): list with labels of log lines
    Returns:
        binary_labels (list): returns list with entry 0 for no anomaly and 1 for anomaly
    """
    # get idc of anomaly and non anomaly
    binary_labels = []

    for i, elem in enumerate(labels):
        if '-' in elem:
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    return binary_labels
    

def get_log_labels(log_file: str):
    """ returns list with actual labels of log lines
    Params:
        labels (list): list with labels of log lines
    Returns:
        binary_labels (list): returns list with entry 0 for no anomaly and 1 for anomaly
    """

    labels = []
    
    with open(f'logparser-results/{log_file}_structured.csv') as file:
        reader = csv.reader(file)

        for row in reader:
            labels.append(row[1])

        labels.pop(0)

    return labels


def get_idc(labels: list):
    """ returns idc of lines with anomaly and no anomaly
    Params:
        labels (list): list with labels of log lines
    Returns:
        idc_anomaly (list): all idc of lines with anomaly 
        idc_no_anomaly (list): all idc of lines with no anomaly 
    """
    # get idc of anomaly and non anomaly
    idc_anomaly = []
    idc_no_anomaly = []

    for i, elem in enumerate(labels):
        if '-' in elem:
            idc_anomaly.append(i)
        else:
            idc_no_anomaly.append(i)

    return idc_anomaly, idc_no_anomaly