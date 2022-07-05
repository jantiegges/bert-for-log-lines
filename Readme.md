## Log Line Embeddings with BERT

Log lines are of enormous importance to the operation of today's IT systems, and log line analysis using data-driven techniques has gained considerable popularity in recent years. A great amount of research has been done on models for detecting anomalies in logs to prevent or handle errors, all of which require log lines as input in vector form. However, the transformation of log lines into numerical vectors has mostly been neglected and has not kept pace with parallel developments in the Natural Language Processing (NLP) field, which is why this work has investigated the application of modern word embedding techniques to log lines. It has been shown that the application to the raw data without additional fine tuning and pre-processing can produce very meaningful results by mapping semantically similar lines close to each other. This has enormous potential to improve anomaly detection, since the models can obtain significantly better vector representations of the log lines as input, which capture semantic information better than the previously used methods.

### How to
1. Create virtualenv (recommendation is to use pyenv virtualenvs) 
2. Install dependencies from requirements.txt
3. Open the Notebook, select the right kernel and run the code.

GLHF!