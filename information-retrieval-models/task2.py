import pandas as pd
import json
from timeit import default_timer as timer

import task1


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


if __name__ == "__main__":

    start = timer()

    df_raw_text = pd.read_csv("candidate-passages-top1000.tsv", sep="\t", names=['qid', 'pid', 'query', 'passage'],
                              dtype='string')
    # process all the passages in candidate-passages-top1000 tsv
    tokens_list = task1.process_raw_text(df_raw_text.loc[:, 'passage'], remove_stopwords=True, stemming=True)
    # dict with pid as key and list of tokens as values
    pid_passage_dict = dict(zip(df_raw_text['pid'].tolist(), tokens_list))
    inverted_index = generate_inverted_index(pid_passage_dict)

    # save processed passage to file
    with open("processed_passage.json", 'w') as f:
        json.dump(pid_passage_dict, f, indent=4)
    # save inverted index to file
    with open("inverted_index.json", 'w') as f:
        json.dump(inverted_index, f, indent=4)

    end = timer()
    time_taken = end - start

    print(f"Task 2 Number of vocabulary: {len(inverted_index)}")
    print(f"Task 2 Process time: {time_taken} seconds")  # usually takes 80 sec
