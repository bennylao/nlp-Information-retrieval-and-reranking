import json
import pandas as pd
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from task1 import compute_map, compute_ndcg
from task3 import prepare_input_data_lm


if __name__ == "__main__":
    start = timer()

    with open("train_query_ae.json", 'r') as f:
        train_query_ae_dict = json.load(f)
    with open("train_passage_ae.json", 'r') as f:
        train_passage_ae_dict = json.load(f)

    sub_train_data_df = pd.read_csv("sub_train_data.tsv", sep='\t', dtype='string')

    train_data_ae_df = prepare_input_data_lm(sub_train_data_df, train_query_ae_dict, train_passage_ae_dict)
    train_data_ae_df['qid'] = train_data_ae_df['qid'].astype(int)
    train_data_ae_df['y_train'] = sub_train_data_df.loc[:, 'relevancy'].astype(float).astype(int)

    train_data_ae_df = train_data_ae_df.sort_values(by='qid', ascending=True)

    x_train = np.stack(train_data_ae_df['x_train'].values)
    y_train = train_data_ae_df['y_train'].values

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[200, 1]),
        tf.keras.layers.SpatialDropout1D(0.4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.05, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=1)

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

    validation_data_ae_df['score'] = model.predict(x_valid)
    validation_data_ae_df = validation_data_ae_df.sort_values(by=['qid', 'score'], ascending=False, ignore_index=True)
    validation_rnn_dict = validation_data_ae_df.groupby('qid').apply(
        lambda x: dict(zip(x.loc[:, "pid"], x.loc[:, "score"])), include_groups=False).to_dict()

    validation_map = compute_map(validation_rnn_dict, validation_relevance_dict)

    validation_ndcg = compute_ndcg(validation_rnn_dict, validation_relevance_dict)

    print("map", validation_map)
    print("ndcg", validation_ndcg)

    with open("test_passage_ae.json", 'r') as f:
        test_passage_ae_dict = json.load(f)

    with open("test_query_ae.json", 'r') as f:
        test_query_ae_dict = json.load(f)

    test_passages_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", names=['qid', 'pid', 'queries', 'passage'],
                                   dtype='string')
    test_data_ae_df = prepare_input_data_lm(test_passages_df, test_query_ae_dict, test_passage_ae_dict)

    rnn_test_data_df = pd.DataFrame()

    for qid in test_query_ae_dict.keys():
        qid_test_data_ae_df = test_data_ae_df.loc[test_data_ae_df['qid'] == qid, ['qid', 'pid', 'x_train']]
        x_test = np.stack(qid_test_data_ae_df['x_train'].values)
        qid_test_data_ae_df['score'] = model.predict(x_test)
        qid_test_data_ae_df = qid_test_data_ae_df.sort_values(by='score', ascending=False, ignore_index=True).head(100)
        qid_test_data_ae_df['rank'] = range(1, 1 + len(qid_test_data_ae_df))
        rnn_test_data_df = pd.concat([rnn_test_data_df, qid_test_data_ae_df])

    with open('NN.txt', 'w') as f:
        for _, row in rnn_test_data_df.iterrows():
            f.write(f"{row['qid']} A2 {row['pid']} {row['rank']} {row['score']} NN\n")

    end = timer()
    time_taken = end - start
    print(f"Task 4 Process time: {round(time_taken / 60, 1)} minutes")
