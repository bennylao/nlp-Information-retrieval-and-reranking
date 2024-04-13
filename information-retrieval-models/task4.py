import json
import math
import pandas as pd
from timeit import default_timer as timer


class SmoothingNotSupportError(Exception):
    def __init__(self, message="The smoothing is not supported!"):
        self.message = message
        super().__init__(self.message)


def query_likelihood_model(query_df, qid_pid_df, inverted_index_dict, query_dict, passage_dict, smoothing='laplace'):
    df = pd.DataFrame()

    qids = query_df.loc[:, 'qid'].tolist()

    if smoothing == 'laplace' or smoothing == 'lidstone':
        v = len(inverted_index_dict)
    elif smoothing == 'dirichlet':
        num_token_collection = 0
        for passage_token_list in passage_dict.values():
            num_token_collection += len(passage_token_list)
    else:
        raise SmoothingNotSupportError(f"The smoothing '{smoothing}' is not supported!")

    epsilon = 0.1
    mu = 50

    for qid in qids:
        output = []

        query_tokens = query_dict.get(qid)
        pids = qid_pid_df.loc[qid_pid_df['qid'] == qid, 'pid'].tolist()

        for pid in pids:
            score = 0
            passage_token_list = passage_dict.get(pid)
            # length of document
            d = len(passage_token_list)

            for query_token in query_tokens:

                pid_token_count_dict = inverted_index_dict.get(query_token)
                if query_token in passage_token_list:

                    m = pid_token_count_dict.get(pid)
                else:
                    m = 0

                if smoothing == 'laplace':
                    score += math.log((m + 1) / (d + v))
                if smoothing == 'lidstone':
                    score += math.log((m + epsilon) / (d + epsilon * v))
                if smoothing == 'dirichlet':
                    lambda_value = d / (d + mu)

                    if pid_token_count_dict is not None:
                        cqi = sum(pid_token_count_dict.values())
                    else:
                        cqi = 0

                    temp = lambda_value * m / d + (1 - lambda_value) * cqi / num_token_collection
                    if temp != 0:
                        score += math.log(temp)

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
    # load processed query from file
    with open("processed_query.json", 'r') as f:
        processed_query_dict = json.load(f)
    # load inverted index from file
    with open("inverted_index.json", 'r') as f:
        inverted_index = json.load(f)

    df_query = pd.read_csv("test-queries.tsv", sep='\t', names=['qid', 'query'], dtype='string')
    df_qid_pid = pd.read_csv("candidate-passages-top1000.tsv", sep='\t', names=['qid', 'pid', 'query', 'passage'],
                             dtype={'qid': 'string', 'pid': 'string'}).loc[:, ['qid', 'pid']]

    df_laplace = query_likelihood_model(df_query, df_qid_pid, inverted_index, processed_query_dict,
                                        processed_passage_dict, smoothing='laplace')
    df_laplace.to_csv("laplace.csv", mode='w', index=False, header=False)

    df_lidstone = query_likelihood_model(df_query, df_qid_pid, inverted_index, processed_query_dict,
                                         processed_passage_dict, smoothing='lidstone')
    df_lidstone.to_csv("lidstone.csv", mode='w', index=False, header=False)

    df_dirichlet = query_likelihood_model(df_query, df_qid_pid, inverted_index, processed_query_dict,
                                          processed_passage_dict, smoothing='dirichlet')
    df_dirichlet.to_csv("dirichlet.csv", mode='w', index=False, header=False)

    end = timer()
    time_taken = end - start

    print(f"Task 4 Process time: {time_taken} seconds")  # usually takes 20 sec
