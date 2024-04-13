import re
import math
import json
import nltk
import pandas as pd
from timeit import default_timer as timer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import ssl


def process_raw_text(lines, remove_stopwords=False, lemmatisation=False, stemming=False):
    tokens = []
    if remove_stopwords is True:
        # load stopwords into variable
        stop_words = nltk.corpus.stopwords.words('english')
    if lemmatisation is True:
        nltk.download('wordnet')
        # load lemmatiser into variable
        lemmatiser = WordNetLemmatizer()
    if stemming is True:
        # load stemmer into variable
        stemmer = SnowballStemmer("english")

    for line in lines:
        # make words lowercase
        line = line.strip().lower()

        # replace anything that is not A-Z or a-z or space with a space
        line = re.sub(r"[^a-zA-Z\s]", " ", line)

        # tokenise using nltk
        words = [w for w in nltk.word_tokenize(line)]
        if remove_stopwords is True:
            words = [w for w in words if w not in stop_words]
        if lemmatisation is True:
            words = [lemmatiser.lemmatize(w) for w in words]
        if stemming is True:
            words = [stemmer.stem(w) for w in words]
        tokens.append(words)

    return tokens


def generate_inverted_index(passage_dict):
    inverted_index_dict = {}
    for pid, tokens in passage_dict.items():
        for token in tokens:
            occurrence = tokens.count(token)
            if token not in inverted_index_dict.keys():
                inverted_index_dict[token] = {pid: occurrence}
            else:
                sub_dict = {pid: occurrence}  # passage id : number of token occurrence in the passage
                inverted_index_dict[token].update(sub_dict)

    return inverted_index_dict


def bm25(qid_pid_df, inv_index_dict, query_dict, passage_dict):
    df = pd.DataFrame()

    # define bm25 parameters
    k1 = 1.2
    k2 = 100
    b = 0.75
    n = len(passage_dict)

    # calculate average document length
    avdl = 0
    for passage_token_list in passage_dict.values():
        avdl += len(passage_token_list)
    avdl /= n

    qids = qid_pid_df.loc[:, 'qid'].unique()
    for qid in qids:
        output = []

        query_tokens = query_dict.get(qid)
        pids = qid_pid_df.loc[qid_pid_df['qid'] == qid, 'pid'].tolist()

        for pid in pids:
            score = 0

            passage_token_list = passage_dict.get(pid)
            dl = len(passage_token_list)
            k = k1 * ((1 - b) + b * dl / avdl)

            for query_token in (set(query_tokens) & set(passage_token_list)):
                pid_token_count_dict = inv_index_dict.get(query_token)
                ni = len(pid_token_count_dict)
                fi = pid_token_count_dict.get(pid)
                qfi = query_tokens.count(query_token)
                score += math.log(1 / ((ni + 0.5) / (n - ni + 0.5))) * ((k1 + 1) * fi / (k + fi)) * (
                        (k2 + 1) * qfi / (k2 + qfi))

            output.append((qid, pid, score))

        df_result = pd.DataFrame(data=output, columns=['qid', 'pid', 'score'])
        df_result = df_result.sort_values(by='score', ascending=False)
        df = pd.concat([df, df_result])

    df.columns = ['qid', 'pid', 'relevance_score']

    return df


def compute_map(pred_dict, relevance_dict):
    ap_sum = 0
    for qid, pid_score_dict in pred_dict.items():
        num_relevant_doc = 0
        ap = 0
        for rank, pid in enumerate(pid_score_dict.keys(), start=1):
            if pid in relevance_dict[qid] and relevance_dict[qid][pid] == "1.0":
                num_relevant_doc += 1
                ap += num_relevant_doc / rank
        ap /= min(len(relevance_dict[qid]), len(pid_score_dict))
        ap_sum += ap

    ap_sum /= len(pred_dict)

    return ap_sum


def compute_ndcg(pred_dict, relevance_dict):
    mean_ndcg = 0
    for qid, pid_score_dict in pred_dict.items():
        dcg = 0
        for rank, pid in enumerate(pid_score_dict.keys(), start=1):
            if pid in relevance_dict[qid] and relevance_dict[qid][pid] == "1.0":
                dcg += 1 / math.log2(rank + 1)

        idcg = 0
        for i in range(min(len(relevance_dict[qid]), len(pid_score_dict))):
            idcg += 1 / math.log2(i + 2)

        mean_ndcg += dcg / idcg

    return mean_ndcg / len(pred_dict)


if __name__ == "__main__":
    # if SSL certificate verify fails,
    # try using alternative WI-FI or use the try-except code below.
    # reference: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    start = timer()
    pd.options.mode.copy_on_write = True
    # download list of stopwords from nltk
    nltk.download('stopwords')

    # read data from validation dataset
    raw_validation_df = pd.read_csv("validation_data.tsv", sep="\t", dtype='string')

    # process all the passages in validation_data.csv
    validation_passage_tokens_list = process_raw_text(raw_validation_df.loc[:, 'passage'], remove_stopwords=True,
                                                      stemming=True)
    # create dict with pid as key and list of tokens as values
    validation_pid_passage_dict = dict(zip(raw_validation_df['pid'].tolist(), validation_passage_tokens_list))

    print("finished validation passage processing")

    # save processed passage to file
    with open("validation_processed_passage.json", 'w') as f:
        json.dump(validation_pid_passage_dict, f, indent=4)

    # build inverted index
    validation_inverted_index = generate_inverted_index(validation_pid_passage_dict)

    # save processed passage to file
    with open("validation_inverted_index.json", 'w') as f:
        json.dump(validation_inverted_index, f, indent=4)

    # process all the queries in validation_data.csv
    validation_query_tokens_list = process_raw_text(raw_validation_df.loc[:, 'queries'], remove_stopwords=True,
                                                    stemming=True)
    # create dict with pid as key and list of tokens as values
    validation_qid_query_dict = dict(zip(raw_validation_df['qid'].tolist(), validation_query_tokens_list))

    print("finished validation query processing")

    # save processed passage to file
    with open("validation_processed_query.json", 'w') as f:
        json.dump(validation_qid_query_dict, f, indent=4)

    df_qid_pid = raw_validation_df.loc[:, ['qid', 'pid']]

    df_bm25 = bm25(df_qid_pid, validation_inverted_index, validation_qid_query_dict, validation_pid_passage_dict)
    bm25_dict = df_bm25.groupby('qid').apply(lambda x: dict(zip(x.loc[:, "pid"], x.loc[:, "relevance_score"])),
                                             include_groups=False).to_dict()

    print("finish bm25")

    with open("bm25.json", 'w') as f:
        json.dump(bm25_dict, f, indent=4)

    validation_relevance_dict = raw_validation_df.loc[raw_validation_df['relevancy'] == "1.0"].groupby('qid').apply(
        lambda x: dict(zip(x.loc[:, "pid"], x.loc[:, "relevancy"])), include_groups=False).to_dict()

    with open("validation_relevance.json", 'w') as f:
        json.dump(validation_relevance_dict, f, indent=4)

    bm25_map = compute_map(bm25_dict, validation_relevance_dict)
    bm25_ndcg = compute_ndcg(bm25_dict, validation_relevance_dict)

    end = timer()
    time_taken = end - start

    print("BM25 MAP: ", bm25_map)
    print("BM25 NDCG: ", bm25_ndcg)

    print(f"Task 1 Process time: {round(time_taken / 60, 1)} minutes")

    del validation_passage_tokens_list, validation_inverted_index
    del validation_query_tokens_list, validation_qid_query_dict
    del df_bm25, bm25_dict
