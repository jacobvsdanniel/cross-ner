import sys
import time
import math
import random
import argparse

import numpy as np
import tensorflow as tf

from model import Config, Model

clear_line = "\r" + " "*128 + "\r"

def load_data(config):
    """Read dataset and determine dataset related configuration.
    
    split_to_samplelist: {"train": sample_list, "validate": sample_list, "test": sample_list}
        sample_list: [sample]
            sample: (word_sequence, tokenlabel_sequence, chunk_to_entitytype)
    """
    if config.dataset == "ontonotes":
        import ontonotes as data_utils
    elif config.dataset == "conll2003":
        import conll2003 as data_utils
    elif config.dataset == "wnut2016":
        import wnut2016 as data_utils
    elif config.dataset == "wnut2017":
        import wnut2017 as data_utils
        
    (
        split_to_samplelist,
        config.tokenlabel_list,
        config.tokenlabel_to_index
    ) = data_utils.read_dataset()
    config.output_d = len(config.tokenlabel_list)
    
    (
        config.w_token_to_vector,
        config.w_embedding_d,
    ) = data_utils.load_embedding()
    
    return split_to_samplelist
    
def initialize_model(config):
    """Define Tensorflow graph and create model according to configuration."""
    
    model = Model(config)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    model.sess = tf.Session(config=tf_config)
    model.sess.run(tf.global_variables_initializer())
    return model
    
def make_batch_list(sample_list, batch_size, order):
    """Create mini-batches."""
    
    index_sample_list = list(enumerate(sample_list))
    if order == "original":
        pass
    elif order == "random":
        random.shuffle(index_sample_list)
    elif order == "length":
        index_sample_list = sorted(
            index_sample_list,
            key = lambda index_sample: len(index_sample[1][0])
        )
    else:
        assert False
        
    batch_list = []
    batch, samples = [], 0
    for index, sample in index_sample_list:
        batch.append((index, sample))
        samples += 1
        if samples >= batch_size:
            batch_list.append(batch)
            batch, samples = [], 0
    if batch:
        batch_list.append(batch)
        
    return batch_list
    
def compute_loss_for_dataset(model, total_sample_list, update_model=False):
    """Update model parameters for every sample once."""
    
    batch_list = make_batch_list(total_sample_list, 32, "random")
    total_samples = len(total_sample_list)
    samples = 0
    tokens = 0
    loss = 0
    
    for index_sample_list in batch_list:
        index_list, sample_list = zip(*index_sample_list)
        samples += len(sample_list)
        batch_tokens = np.sum(len(word_sequence) for word_sequence, _, _ in sample_list)
        tokens += batch_tokens
        
        token_loss = model.compute_loss_for_a_batch(sample_list, update_model)
        loss += batch_tokens * token_loss
        ppl = 2**(loss/tokens)
        
        print(clear_line + f"compute_loss_for_dataset(): ({samples}/{total_samples}) ppl={ppl:.2f}", end="", flush=True)
    print(clear_line, end="", flush=True)
    
    ppl = 2**(loss/tokens)
    return ppl
    
def predict_token_label_for_dataset(model, total_sample_list):
    """Predict the label for each token.
    
    global_prediction_list: [prediction]
        prediction: [token_label]
    """
    
    batch_list = make_batch_list(total_sample_list, 256, "length")
    total_samples = len(total_sample_list)
    global_prediction_list = [None] * total_samples
    samples = 0
    
    for index_sample_list in batch_list:
        index_list, sample_list = zip(*index_sample_list)
        samples += len(sample_list)
        
        prediction_list = model.predict_label_for_a_batch(sample_list)
        for b, index in enumerate(index_list):
            global_prediction_list[index] = prediction_list[b]
            
        print(clear_line + f"predict_dataset(): ({samples}/{total_samples})", end="", flush=True)
    print(clear_line, end="", flush=True)
    
    return global_prediction_list
    
def get_chunk_to_entitytype(tokenlabel_sequence):
    """Extract entities from token label sequence."""
    
    chunk_to_entitytype = {}
    left = None
    left_entity_type = None
    
    for index, token_label in enumerate(tokenlabel_sequence):
        entity_type, chunk_type = token_label.split(":")
        if chunk_type == "B":
            left = index
            left_entity_type = entity_type
        elif chunk_type == "I":
            if left is not None and left_entity_type != entity_type:
                left = None
                left_entity_type = None
        elif chunk_type == "O":
            left = None
            left_entity_type = None
        elif chunk_type == "E":
            if left is not None and left_entity_type == entity_type:
                chunk_to_entitytype[(left, index+1)] = entity_type
            left = None
            left_entity_type = None
        elif chunk_type == "S":
            chunk_to_entitytype[(index, index+1)] = entity_type
            left = None
            left_entity_type = None
        
    return chunk_to_entitytype
    
def evaluate_prediction(sample_list, prediction_list):
    """Compute scores for each metric across all samples.
    
    sample_list: [sample]
        sample: (word_sequence, tokenlabel_sequence, chunk_to_entitytype)
    prediction_list: [prediction]
        prediction: [token_label]
    P: precision, overlapped entities over predicted entities
    R: recall, overlapped entities over annotated entities
    F1: f1 score
    """
    
    samples = len(sample_list)
    overlap = 0
    golds = 0
    autos = 0
    
    for i, (_, _, gold_chunk_to_entitytype) in enumerate(sample_list):
        golds += len(gold_chunk_to_entitytype)
        
        auto_chunk_to_entitytype = get_chunk_to_entitytype(prediction_list[i])
        autos += len(auto_chunk_to_entitytype)
        
        for chunk, entity_type in auto_chunk_to_entitytype.items():
            if chunk in gold_chunk_to_entitytype and entity_type == gold_chunk_to_entitytype[chunk]:
                overlap += 1
                
    try:
        precision = overlap / autos
    except ZeroDivisionError:
        precision = 0
        
    try:
        recall = overlap / golds
    except ZeroDivisionError:
        recall = 0
        
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
        
    metric_to_score = {
        "P": precision*100,
        "R": recall*100,
        "F1": f1*100,
    }
    return metric_to_score
    
def train_script(config):
    """Update model parameters until it converges or reaches maximum epochs."""
    
    split_to_samplelist = load_data(config)
    model = initialize_model(config)
    
    saver = tf.train.Saver()
    metric_to_default = {
        "PPL": float("inf"),
        "P":   float("-inf"),
        "R":   float("-inf"),
        "F1":  float("-inf"),
    }
    record_epoch = 0
    record = {}
    for split in ["train", "validate"]:
        record[split] = {
            metric: default
            for metric, default in metric_to_default.items()
        }
        
    for epoch in range(1, config.max_epochs+1):
        print(f"\n<Epoch {epoch}>")
        
        start_time = time.time()
        train_ppl = compute_loss_for_dataset(model, split_to_samplelist["train"], update_model=True)
        elasped = time.time() - start_time
        print(f"[train] PPL={train_ppl:.2f} elapsed {elasped:.0f}s")
        
        start_time = time.time()
        prediction_list = predict_token_label_for_dataset(model, split_to_samplelist["validate"])
        metric_to_score = evaluate_prediction(split_to_samplelist["validate"], prediction_list)
        elapsed = time.time() - start_time
        print(f"[validate]", end="")
        for metric, score in metric_to_score.items():
            print(f" {metric}={score:.2f}", end="")
        print(f" elapsed={elapsed:.0f}s", end="")
        
        if record["validate"]["F1"] < metric_to_score["F1"]:
            record_epoch = epoch
            record["train"]["PPL"] = train_ppl
            for metric, score in metric_to_score.items():
                record["validate"][metric] = score
            saver.save(model.sess, f"./model/{config.name}")
        
        print(f" best=#{record_epoch}")
        
    saver.restore(model.sess, f"./model/{config.name}")
    
    prediction_list = predict_token_label_for_dataset(model, split_to_samplelist["train"])
    metric_to_score = evaluate_prediction(split_to_samplelist["train"], prediction_list)
    for metric, score in metric_to_score.items():
        record["train"][metric] = score
    record["validate"]["PPL"] = compute_loss_for_dataset(model, split_to_samplelist["validate"], update_model=False)
    
    print(f"\n<Best Epoch {record_epoch}>")
    for split, metric_to_score in record.items():
        print(f"[{split}]", end="")
        for metric, score in metric_to_score.items():
            print(f" {metric}={score:.2f}", end="")
        print()
        
    start_time = time.time()
    prediction_list = predict_token_label_for_dataset(model, split_to_samplelist["test"])
    metric_to_score = evaluate_prediction(split_to_samplelist["test"], prediction_list)
    elapsed = time.time() - start_time
    print(f"[test]", end="")
    for metric, score in metric_to_score.items():
        print(f" {metric}={score:.2f}", end="")
    print(f" elapsed={elapsed:.0f}s")
    return
    
def evaluate_script(config):
    """Compute the score of an existing model."""
    
    split_to_samplelist = load_data(config)
    sample_list = split_to_samplelist[config.split]
    model = initialize_model(config)
    
    saver = tf.train.Saver()
    saver.restore(model.sess, f"./model/{config.name}")
    
    start_time = time.time()
    ppl = compute_loss_for_dataset(model, sample_list, update_model=False)
    elapsed = time.time() - start_time
    print(f"[{config.split}] PPL={ppl:.2f}; elapsed={elapsed:.0f}s")
    
    start_time = time.time()
    prediction_list = predict_token_label_for_dataset(model, sample_list)
    metric_to_score = evaluate_prediction(sample_list, prediction_list)
    elapsed = time.time() - start_time
    print(f"[{config.split}]", end="")
    for metric, score in metric_to_score.items():
        print(f" {metric}={score:.2f}", end="")
    print(f" elapsed={elapsed:.0f}s")
    return
    
def predict_dataset_script(config):
    """Generate predictions for a existing dataset."""
    assert config.predict_file
    
    split_to_samplelist = load_data(config)
    sample_list = split_to_samplelist[config.split]
    model = initialize_model(config)
    
    saver = tf.train.Saver()
    saver.restore(model.sess, f"./model/{config.name}")
    
    start_time = time.time()
    prediction_list = predict_token_label_for_dataset(model, sample_list)
    metric_to_score = evaluate_prediction(sample_list, prediction_list)
    elapsed = time.time() - start_time
    print(f"[{config.split}]", end="")
    for metric, score in metric_to_score.items():
        print(f" {metric}={score:.2f}", end="")
    print(f" elapsed={elapsed:.0f}s")
    
    with open(config.predict_file, "w") as f:
        for i, (word_sequence, gold_tokenlabel_sequence, _) in enumerate(sample_list):
            auto_tokenlabel_sequence = prediction_list[i]
            f.write(" ".join(word_sequence) + "\n")
            f.write(" ".join(gold_tokenlabel_sequence) + "\n")
            f.write(" ".join(auto_tokenlabel_sequence) + "\n")
            f.write("\n")
    return
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", choices=["train", "evaluate", "predict"])
    parser.add_argument("-dataset", choices=["ontonotes", "conll2003", "wnut2016", "wnut2017"])
    parser.add_argument("-att", default="5")
    parser.add_argument("-bilstm", default="2-100")
    parser.add_argument("-cnn", default="1-20-2-20-3-20")
    parser.add_argument("-split", choices=["train", "validate", "test"])
    parser.add_argument("-predict_file")
    parser.add_argument("-epoch", default="400")
    parser.add_argument("-suffix", default="")
    parser.add_argument("-model_name", default="")
    arg = parser.parse_args()
    
    config = Config()
    config.name = (
        "model"
        + "_" + arg.dataset
        + "_Att-" + arg.att
        + "_BiLSTM-" + arg.bilstm
        + "_CNN-" + arg.cnn
    )
    if arg.suffix: config.name += "_" + arg.suffix
    if arg.model_name: config.name = arg.model_name
    
    config.dataset = arg.dataset
    config.attention_heads = int(arg.att)
    
    bilstm = arg.bilstm.split("-")
    if len(bilstm) == 3:
        config.is_cross_bilstm = True
        config.layers = int(bilstm[1])
        config.hidden_d = int(bilstm[2])
    elif len(bilstm) == 2:
        config.is_cross_bilstm = False
        config.layers = int(bilstm[0])
        config.hidden_d = int(bilstm[1])
    else:
        assert False
        
    cnn = [int(i) for i in arg.cnn.split("-")]
    config.cnn_window_to_kernels = {}
    for i in range(0, len(cnn), 2):
        config.cnn_window_to_kernels[cnn[i]] = cnn[i+1]
        
    config.split = arg.split
    config.predict_file = arg.predict_file
    config.max_epochs = int(arg.epoch)
    
    if arg.mode == "train":
        train_script(config)
    elif arg.mode == "evaluate":
        evaluate_script(config)
    elif arg.mode == "predict":
        predict_dataset_script(config)
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    