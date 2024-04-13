import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from timeit import default_timer as timer
from task1 import compute_map, compute_ndcg


def prepare_input_data_lm(df, query_dict, passage_dict):
    processed_df = df.loc[:, ['qid', 'pid']]
    processed_df['query_embedding'] = processed_df.apply(lambda x: query_dict[x.loc['qid']], axis=1)
    processed_df['passage_embedding'] = processed_df.apply(lambda x: passage_dict[x.loc['pid']], axis=1)
    processed_df['x_train'] = processed_df.apply(
        lambda x: np.hstack((x.loc['passage_embedding'], x.loc['query_embedding'])), axis=1)

    return processed_df.loc[:, ['qid', 'pid', 'x_train']]


if __name__ == "__main__":
    start = timer()

    with open("train_query_ae.json", 'r') as f:
        train_query_ae_dict = json.load(f)
    with open("train_passage_ae.json", 'r') as f:
        train_passage_ae_dict = json.load(f)

    sub_train_data_df = pd.read_csv("sub_train_data.tsv", sep='\t', dtype='string')

    with open("validation_query_ae.json", 'r') as f:
        validation_query_ae_dict = json.load(f)

    with open("validation_passage_ae.json", 'r') as f:
        validation_passage_ae_dict = json.load(f)

    with open("validation_relevance.json", 'r') as f:
        validation_relevance_dict = json.load(f)

    # read data from validation dataset
    raw_validation_df = pd.read_csv("validation_data.tsv", sep="\t", dtype='string')

    validation_data_ae_df = prepare_input_data_lm(raw_validation_df, validation_query_ae_dict, validation_passage_ae_dict)
    x_valid = np.stack(validation_data_ae_df['x_train'].values)

    train_data_ae_df = prepare_input_data_lm(sub_train_data_df, train_query_ae_dict, train_passage_ae_dict)
    train_data_ae_df['qid'] = train_data_ae_df['qid'].astype(int)
    train_data_ae_df['y_train'] = sub_train_data_df.loc[:, 'relevancy'].astype(float).astype(int)

    train_data_ae_df = train_data_ae_df.sort_values(by='qid', ascending=True)

    x_train = np.stack(train_data_ae_df['x_train'].values)
    y_train = train_data_ae_df['y_train'].values

    # ParameterGrid
    param_grid = {
        'learning_rate': [0.1, 1],
        'alpha': [0, 0.1],
        'gamma': [0, 0.1, 1],
        'max_depth': [6, 7],
        'n_estimators': [100, 200]
    }

    best_ndcg = 0
    best_map = 0
    best_params_index = 0
    parameter_grid = ParameterGrid(param_grid)

    for i, params in enumerate(parameter_grid):
        print(params)
        ranker = xgb.XGBRanker(tree_method="hist", device="cuda", objective="rank:pairwise", lambdarank_pair_method="mean",
                               **params)
        ranker.fit(x_train, y_train, qid=train_data_ae_df['qid'])

        validation_data_ae_df['score'] = ranker.predict(x_valid)
        validation_data_ae_df = validation_data_ae_df.sort_values(by=['qid', 'score'], ascending=False, ignore_index=True)
        validation_lambdamart_dict = validation_data_ae_df.groupby('qid').apply(
            lambda x: dict(zip(x.loc[:, "pid"], x.loc[:, "score"])), include_groups=False).to_dict()

        validation_map = compute_map(validation_lambdamart_dict, validation_relevance_dict)

        validation_ndcg = compute_ndcg(validation_lambdamart_dict, validation_relevance_dict)

        print("map", validation_map)
        print("ndcg", validation_ndcg)

        if validation_ndcg > best_ndcg:
            best_ndcg = validation_ndcg
            best_map = validation_map
            best_params_index = i
        elif validation_ndcg == best_ndcg and validation_map > best_map:
            best_ndcg = validation_ndcg
            best_map = validation_map
            best_params_index = i

        validation_data_ae_df = validation_data_ae_df.drop(columns='score')
        del ranker

    print("LambdaMART MAP", best_map)
    print("LambdaMART NDCG", best_ndcg)
    print("LambdaMART Parameter Index", best_params_index)

    ranker = xgb.XGBRanker(tree_method="hist", device="cuda", objective="rank:pairwise", lambdarank_pair_method="mean",
                           **parameter_grid[best_params_index])
    ranker.fit(x_train, y_train, qid=train_data_ae_df['qid'])

    with open("test_passage_ae.json", 'r') as f:
        test_passage_ae_dict = json.load(f)

    with open("test_query_ae.json", 'r') as f:
        test_query_ae_dict = json.load(f)

    test_passages_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", names=['qid', 'pid', 'queries', 'passage'],
                                   dtype='string')
    test_data_ae_df = prepare_input_data_lm(test_passages_df, test_query_ae_dict, test_passage_ae_dict)

    lambdamart_test_data_df = pd.DataFrame()

    for qid in test_query_ae_dict.keys():
        qid_test_data_ae_df = test_data_ae_df.loc[test_data_ae_df['qid'] == qid, ['qid', 'pid', 'x_train']]
        x_test = np.stack(qid_test_data_ae_df['x_train'].values)
        qid_test_data_ae_df['score'] = ranker.predict(x_test)
        qid_test_data_ae_df = qid_test_data_ae_df.sort_values(by='score', ascending=False, ignore_index=True).head(100)
        qid_test_data_ae_df['rank'] = range(1, 1 + len(qid_test_data_ae_df))
        lambdamart_test_data_df = pd.concat([lambdamart_test_data_df, qid_test_data_ae_df])

    with open('LM.txt', 'w') as f:
        for _, row in lambdamart_test_data_df.iterrows():
            f.write(f"{row['qid']} A2 {row['pid']} {row['rank']} {row['score']} LM\n")

    end = timer()

    time_taken = end - start
    print(f"Task 3 Process time: {round(time_taken / 60, 1)} minutes")
