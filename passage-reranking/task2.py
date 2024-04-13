import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from task1 import process_raw_text, compute_map, compute_ndcg
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


def down_sampling(df, random_seed):
    relevant_df = df[df['relevancy'] == "1.0"]
    irrelevant_df = df[df['relevancy'] == "0.0"]

    sub_irrelevant_df = []
    for qid, group in irrelevant_df.groupby('qid'):
        num_relevant = relevant_df[relevant_df['qid'] == qid].shape[0]
        max_samples = max(0, 100 - num_relevant)
        sub_irrelevant_df.append(group.sample(n=min(len(group), max_samples), random_state=random_seed))

    final_df = pd.concat([relevant_df] + sub_irrelevant_df)

    return final_df.sample(frac=1, random_state=random_seed, ignore_index=True)


def convert_average_embedding(tokens_dict, glove_model):
    ae_dict = {}
    vector_dim = 100
    for text_id, tokens_list in tokens_dict.items():
        if tokens_list:
            tokens_embedding = [
                glove_model[token] if token in glove_model else np.random.uniform(low=-1, high=1, size=vector_dim) for
                token in tokens_list]
            average_embedding = np.mean(tokens_embedding, axis=0).tolist()
        else:
            average_embedding = np.random.uniform(low=-1, high=1, size=vector_dim).tolist()
        ae_dict.update({text_id: average_embedding})

    return ae_dict


def prepare_input_data_logistic(df, query_dict, passage_dict):
    processed_df = df.loc[:, ['qid', 'pid']]
    processed_df['query_embedding'] = processed_df.apply(lambda x: query_dict[x.loc['qid']], axis=1)
    processed_df['passage_embedding'] = processed_df.apply(lambda x: passage_dict[x.loc['pid']], axis=1)
    processed_df['x_train'] = processed_df.apply(
        lambda x: np.hstack((1, x.loc['passage_embedding'], x.loc['query_embedding'])), axis=1)

    return processed_df.loc[:, ['qid', 'pid', 'x_train']]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_reg_train(x, y, epoch, lr, batch_size=5000):
    m = batch_size
    theta = np.ones((x[0].shape[0], 1))
    loss_epoch = []
    y = y.reshape(-1, 1)

    for i in range(epoch):
        j = 0
        loss_mini_batch = []
        while j < x.shape[0]:
            x_mini_batch = x[j:j + batch_size]
            y_mini_batch = y[j:j + batch_size]
            loss = -(1 / m) * np.sum(
                y_mini_batch * np.log(sigmoid(x_mini_batch.dot(theta))) + (1 - y_mini_batch) * np.log(
                    1 - sigmoid(x_mini_batch.dot(theta))))
            theta = theta - lr * (1 / m) * (x_mini_batch.T.dot(sigmoid(x_mini_batch.dot(theta)) - y_mini_batch))
            loss_mini_batch.append(loss)
            j += batch_size

        loss_epoch.append(np.mean(loss_mini_batch))
    print("complete training")
    return theta, loss_epoch


def plot_loss_function(labels, *losses):
    epoch = np.arange(1, len(losses[0]) + 1)

    plt.figure()
    for i, loss in enumerate(losses):
        plt.plot(epoch, loss, label=labels[i])
    plt.xlabel("Epochs", fontsize="x-large")
    plt.ylabel("Loss Function", fontsize="x-large")
    plt.legend()
    plt.savefig('logistic_regression_loss.pdf')


def logistic_reg_predict(x, coe):
    return sigmoid(x.dot(coe))


if __name__ == "__main__":
    start = timer()

    raw_train_df = pd.read_csv("train_data.tsv", sep="\t", dtype='string')
    sub_train_data_df = down_sampling(raw_train_df, 42)
    sub_train_data_df.to_csv("sub_train_data.tsv", mode="w", sep='\t', index=False)

    del raw_train_df

    # process all the passages in sub dataset of train_data.tsv
    train_passage_tokens_list = process_raw_text(sub_train_data_df.loc[:, 'passage'], remove_stopwords=True,
                                                 stemming=True)
    # create dict with pid as key and list of tokens as values
    train_pid_passage_dict = dict(zip(sub_train_data_df['pid'].tolist(), train_passage_tokens_list))

    print("finished train passage processing")

    # save processed passage to file
    with open("train_processed_passage.json", 'w') as f:
        json.dump(train_pid_passage_dict, f, indent=4)

    # process all the queries in sub dataset of train_data.tsv
    train_query_tokens_list = process_raw_text(sub_train_data_df.loc[:, 'queries'], remove_stopwords=True,
                                               stemming=True)
    # create dict with pid as key and list of tokens as values
    train_qid_query_dict = dict(zip(sub_train_data_df['qid'].tolist(), train_query_tokens_list))

    print("finished validation query processing")

    # save processed passage to file
    with open("train_processed_query.json", 'w') as f:
        json.dump(train_qid_query_dict, f, indent=4)

    del train_passage_tokens_list, train_query_tokens_list

    glove_input_file = 'glove.6B.100d.txt'
    word2vec_output_file = 'glove.6B.100d.word2vec.txt'
    glove2word2vec(glove_input_file, word2vec_output_file)

    glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    train_passage_ae_dict = convert_average_embedding(train_pid_passage_dict, glove)

    with open("train_passage_ae.json", 'w') as f:
        json.dump(train_passage_ae_dict, f, indent=4)

    train_query_ae_dict = convert_average_embedding(train_qid_query_dict, glove)

    with open("train_query_ae.json", 'w') as f:
        json.dump(train_query_ae_dict, f, indent=4)

    sub_train_data_df = pd.read_csv("sub_train_data.tsv", sep='\t', dtype='string')

    train_data_ae_df = prepare_input_data_logistic(sub_train_data_df, train_query_ae_dict, train_passage_ae_dict)
    train_data_ae_df['y_train'] = sub_train_data_df.loc[:, 'relevancy'].astype(float)
    train_data_ae_df.to_csv("train_data_embedding.tsv", sep='\t')

    x_train = np.stack(train_data_ae_df['x_train'].values)
    y_train = train_data_ae_df['y_train'].values

    coe_0001, loss_0001 = logistic_reg_train(x_train, y_train, epoch=500, lr=0.001, batch_size=5000)
    coe_001, loss_001 = logistic_reg_train(x_train, y_train, epoch=500, lr=0.01, batch_size=5000)
    coe_01, loss_01 = logistic_reg_train(x_train, y_train, epoch=500, lr=0.1, batch_size=5000)
    coe_1, loss_1 = logistic_reg_train(x_train, y_train, epoch=500, lr=1, batch_size=5000)
    coe_10, loss_10 = logistic_reg_train(x_train, y_train, epoch=500, lr=10, batch_size=5000)

    loss_labels = ["LR=0.001", "LR=0.01", "LR=0.1", "LR=1", "LR=10"]

    plot_loss_function(loss_labels, loss_0001, loss_001, loss_01, loss_1, loss_10)

    del coe_0001, loss_0001
    del coe_001, loss_001
    del coe_01, loss_01
    del coe_1, loss_1
    del coe_10, loss_10

    logistic_coe, logistic_loss = logistic_reg_train(x_train, y_train, epoch=100, lr=0.001, batch_size=5000)

    with open("logistic_coefficient.json", 'w') as f:
        json.dump(logistic_coe.tolist(), f, indent=4)

    with open("validation_processed_passage.json", 'r') as f:
        validation_pid_passage_dict = json.load(f)

    validation_passage_ae_dict = convert_average_embedding(validation_pid_passage_dict, glove)

    with open("validation_passage_ae.json", 'w') as f:
        json.dump(validation_passage_ae_dict, f, indent=4)

    with open("validation_processed_query.json", 'r') as f:
        validation_qid_query_dict = json.load(f)

    validation_query_ae_dict = convert_average_embedding(validation_qid_query_dict, glove)

    with open("validation_query_ae.json", 'w') as f:
        json.dump(validation_query_ae_dict, f, indent=4)

    # read data from validation dataset
    raw_validation_df = pd.read_csv("validation_data.tsv", sep="\t", dtype='string')

    validation_data_ae_df = prepare_input_data_logistic(raw_validation_df, validation_query_ae_dict, validation_passage_ae_dict)
    x_valid = np.stack(validation_data_ae_df['x_train'].values)
    validation_data_ae_df['score'] = logistic_reg_predict(x_valid, logistic_coe)

    validation_data_ae_df = validation_data_ae_df.sort_values(by=['qid', 'score'], ascending=False, ignore_index=True)

    validation_logistic_dict = validation_data_ae_df.groupby('qid').apply(
        lambda x: dict(zip(x.loc[:, "pid"], x.loc[:, "score"])), include_groups=False).to_dict()

    del validation_passage_ae_dict, validation_query_ae_dict
    del raw_validation_df, validation_data_ae_df

    with open("validation_relevance.json", 'r') as f:
        validation_relevance_dict = json.load(f)

    validation_map = compute_map(validation_logistic_dict, validation_relevance_dict)

    validation_ndcg = compute_ndcg(validation_logistic_dict, validation_relevance_dict)

    print("Logistic MAP", validation_map)
    print("Logistic NDCG", validation_ndcg)

    test_passages_df = pd.read_csv("candidate_passages_top1000.tsv", sep="\t",
                                   names=['qid', 'pid', 'queries', 'passage'],
                                   dtype='string')

    # process all the passages in sub dataset of train_data.tsv
    test_passage_tokens_list = process_raw_text(test_passages_df.loc[:, 'passage'], remove_stopwords=True,
                                                stemming=True)

    # create dict with pid as key and list of tokens as values
    test_pid_passage_dict = dict(zip(test_passages_df['pid'].tolist(), test_passage_tokens_list))

    print("finished train passage processing")

    # save processed passage to file
    with open("test_processed_passage.json", 'w') as f:
        json.dump(test_pid_passage_dict, f, indent=4)

    test_query_df = pd.read_csv("test-queries.tsv", sep='\t', names=['qid', 'queries'], dtype='string')

    # process all the queries in sub dataset of train_data.tsv
    test_query_tokens_list = process_raw_text(test_query_df.loc[:, 'queries'], remove_stopwords=True, stemming=True)
    # create dict with pid as key and list of tokens as values
    test_qid_query_dict = dict(zip(test_query_df['qid'].tolist(), test_query_tokens_list))

    print("finished validation query processing")

    # save processed passage to file
    with open("test_processed_query.json", 'w') as f:
        json.dump(test_qid_query_dict, f, indent=4)

    test_passage_ae_dict = convert_average_embedding(test_pid_passage_dict, glove)

    with open("test_passage_ae.json", 'w') as f:
        json.dump(test_passage_ae_dict, f, indent=4)

    test_query_ae_dict = convert_average_embedding(test_qid_query_dict, glove)

    with open("test_query_ae.json", 'w') as f:
        json.dump(test_query_ae_dict, f, indent=4)

    test_data_ae_df = prepare_input_data_logistic(test_passages_df, test_query_ae_dict, test_passage_ae_dict)

    logistic_test_data_df = pd.DataFrame()

    for qid in test_query_ae_dict.keys():
        qid_test_data_ae_df = test_data_ae_df.loc[test_data_ae_df['qid'] == qid, ['qid', 'pid', 'x_train']]
        x_test = np.stack(qid_test_data_ae_df['x_train'].values)
        qid_test_data_ae_df['score'] = logistic_reg_predict(x_test, logistic_coe)
        qid_test_data_ae_df = qid_test_data_ae_df.sort_values(by='score', ascending=False, ignore_index=True).head(100)
        qid_test_data_ae_df['rank'] = range(1, 1 + len(qid_test_data_ae_df))
        logistic_test_data_df = pd.concat([logistic_test_data_df, qid_test_data_ae_df])

    with open('LR.txt', 'w') as f:
        for _, row in logistic_test_data_df.iterrows():
            f.write(f"{row['qid']} A2 {row['pid']} {row['rank']} {row['score']} LR\n")

    end = timer()

    time_taken = end - start

    print(f"Task 2 Process time: {round(time_taken / 60, 1)} minutes")
