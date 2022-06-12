from logparser import Drain
import csv

def pre_process(log_format, log_file, in_dir='./logs', out_dir='logparser-results/', depth=0.5, st=4):
    """ log line parser that saves result in csv file
    Params:
        log_format (string): Log format
        log_file (string): The input log file name
        in_dir (string): The input directory of log file
        out_dir (string): The output directory of parsing results
        st (float): similarity threshold
        depth (float): depth of all leaf nodes
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