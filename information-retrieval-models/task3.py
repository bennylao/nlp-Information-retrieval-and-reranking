import json
import math
import numpy as np
import pandas as pd
from timeit import default_timer as timer

import task1


def generate_passage_idf(inv_index_dict, passage_dict):
    idf_dict = {}
    num_passage = len(passage_dict)
    # calculate idf for each unique words
    for vocab in inv_index_dict:
        idf_dict[vocab] = math.log10(num_passage / len(inv_index_dict[vocab]))

    return idf_dict


def generate_passage_tf_idf(inv_index_dict, passage_dict, idf_dict):
    tf_idf_dict = {}
    for pid, tokens in passage_dict.items():
        tf_idf_dict[pid] = {}
        for token in tokens:
            tf = inv_index_dict.get(token).get(pid) / len(tokens)
            tf_idf = tf * idf_dict.get(token)
            tf_idf_dict[pid].update({token: tf_idf})

    return tf_idf_dict


def generate_query_tf_idf(query_dict, idf_dict):
    tf_idf_dict = {}
    for qid, tokens in query_dict.items():
        tf_idf_dict[qid] = {}
        for token in tokens:
            tf = tokens.count(token) / len(tokens)
            if token in idf_dict:
                tf_idf = tf * idf_dict.get(token)
                tf_idf_dict[qid].update({token: tf_idf})

    return tf_idf_dict


def calculate_cosine_similarity(qid_query_df, qid_pid_df, tfidf_query_dict, tfidf_passage_dict):

    df = pd.DataFrame()
    qids = qid_query_df.loc[:, 'qid'].tolist()

    for qid in qids:

        qid_tfidf_dict = tfidf_query_dict.get(qid)
        query_tokens = qid_tfidf_dict.keys()
        output = []

        pids = qid_pid_df.loc[qid_pid_df['qid'] == qid, 'pid'].tolist()
        for pid in pids:

            pid_tfidf_dict = tfidf_passage_dict.get(pid)
            passage_tokens = pid_tfidf_dict.keys()

            common_tokens = query_tokens & passage_tokens

            inner_product = 0
            for token in common_tokens:
                inner_product += tfidf_query_dict.get(qid).get(token) * tfidf_passage_dict.get(pid).get(token)
            denominator = np.linalg.norm(list(tfidf_query_dict.get(qid).values())) * np.linalg.norm(list(tfidf_passage_dict.get(pid).values()))
            cosine_similarity = inner_product / denominator

            output.append((qid, pid, cosine_similarity))

        df_result = pd.DataFrame(data=output, columns=['qid', 'pid', 'score'])
        df_top_100 = df_result.sort_values(by='score', ascending=False).head(100)

        df = pd.concat([df, df_top_100])

    return df


def bm25(qid_query_df, qid_pid_df, inv_index_dict, query_dict, passage_dict):

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

    qids = qid_query_df.loc[:, 'qid'].tolist()
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
                score += math.log(1/((ni + 0.5) / (n - ni + 0.5))) * ((k1 + 1) * fi / (k + fi)) * ((k2 + 1) * qfi / (k2 + qfi))

            output.append((qid, pid, score))

        df_result = pd.DataFrame(data=output, columns=['qid', 'pid', 'score'])
        df_top_100 = df_result.sort_values(by='score', ascending=False).head(100)
        df = pd.concat([df, df_top_100])

    return df


if __name__ == "__main__":

    start = timer()

    # load processed passage from file
    with open("processed_passage.json", 'r') as f:
        processed_passage_dict = json.load(f)
    # load inverted index from file
    with open("inverted_index.json", 'r') as f:
        inverted_index = json.load(f)

    idf_passage = generate_passage_idf(inverted_index, processed_passage_dict)
    tf_idf_passage = generate_passage_tf_idf(inverted_index, processed_passage_dict, idf_passage)

    df_query = pd.read_csv("test-queries.tsv", sep='\t', names=['qid', 'query'], dtype='string')
    query_tokens = task1.process_raw_text(df_query.loc[:, 'query'], remove_stopwords=True, stemming=True)
    qid_query_dict = dict(zip(df_query['qid'].tolist(), query_tokens))
    tf_idf_query = generate_query_tf_idf(qid_query_dict, idf_passage)

    df_qid_pid = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', names=['qid', 'pid', 'query', 'passage'],
                             dtype={'qid': 'string', 'pid': 'string'}).loc[:, ['qid', 'pid']]

    # tf-idf
    df_tfidf = calculate_cosine_similarity(df_query, df_qid_pid, tf_idf_query, tf_idf_passage)
    df_tfidf.to_csv("tfidf.csv", mode='w', index=False, header=False)

    # bm25
    df_bm25 = bm25(df_query, df_qid_pid, inverted_index, qid_query_dict, processed_passage_dict)
    df_bm25.to_csv("bm25.csv", mode='w', index=False, header=False)

    # save dictionaries to json file
    with open("idf_passage.json", 'w') as f:
        json.dump(idf_passage, f, indent=4)
    with open("tf_idf_passage.json", 'w') as f:
        json.dump(tf_idf_passage, f, indent=4)
    with open("processed_query.json", 'w') as f:
        json.dump(qid_query_dict, f, indent=4)
    with open("tf_idf_query.json", 'w') as f:
        json.dump(tf_idf_query, f, indent=4)

    end = timer()
    time_taken = end - start

    print(f"Task 3 Process time: {time_taken} seconds")  # usually takes 20 sec
