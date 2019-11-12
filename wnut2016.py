import os
import sys
from collections import defaultdict

import numpy as np

dataset = "wnut2016"
tokenlabel_list_file = os.path.join(dataset, "token_label_list.txt")

split_raw_file = {
    "train":    os.path.join(dataset, "raw_train.txt"),
    "validate": os.path.join(dataset, "raw_validate.txt"),
    "test":     os.path.join(dataset, "raw_test.txt")
}

split_data_file = {
    "train":    os.path.join(dataset, "train.txt"),
    "validate": os.path.join(dataset, "validate.txt"),
    "test":     os.path.join(dataset, "test.txt")
}

def get_chunk_to_entity_type_from_raw_label_sequence(label_sequence):
    chunk_to_entitytype = {}
    entity_type = None
    l = None
    
    for r, label in enumerate(label_sequence+["O"]):
        
        if label == "O":
            if entity_type:
                chunk_to_entitytype[(l,r)] = entity_type
                entity_type = None
                l = None
                
        elif label[0] == "B":
            if entity_type:
                chunk_to_entitytype[(l,r)] = entity_type
            entity_type = label[2:]
            l = r
            
        elif label[0] == "I":
            assert entity_type == label[2:]
            
        else:
            assert False
            
    return chunk_to_entitytype
    
def read_raw_data(conll_file):
    with open(conll_file, "r") as f:
        line_list = f.read().splitlines()
        
    sample_list = []
    token_sequence = []
    label_sequence = []
    
    for line in line_list:
        line = line.replace(chr(65279), "") # remove BOM
        
        if not line or line=="\t":
            if not token_sequence: continue
            chunk_to_entitytype = get_chunk_to_entity_type_from_raw_label_sequence(label_sequence)
            sample_list.append((token_sequence, chunk_to_entitytype))
            token_sequence = []
            label_sequence = []
        else:
            token, label = line.split("\t")
            token_sequence.append(token)
            label_sequence.append(label)
            
    assert not token_sequence
    return sample_list
    
def get_label_sequence(chunk_to_entitytype, length):
    label_sequence = ["NONE:O"] * length
    for (l,r), entity_type in chunk_to_entitytype.items():
        assert l<r
        if l==r-1:
            label_sequence[l] = f"{entity_type}:S"
        else:
            label_sequence[l] = f"{entity_type}:B"
            for i in range(l+1, r-1):
                label_sequence[i] = f"{entity_type}:I"
            label_sequence[r-1] = f"{entity_type}:E"
    return label_sequence
    
def extract_data():
    for split, raw_data_file in split_raw_file.items():
        print(f"\n[{split}]", end="", flush=True)
        
        sample_list = read_raw_data(raw_data_file)
        with open(split_data_file[split], "w") as f:
            for token_sequence, chunk_to_entitytype in sample_list:
                f.write(" ".join(token_sequence) + "\n")
                
                label_sequence = get_label_sequence(chunk_to_entitytype, len(token_sequence))
                f.write(" ".join(label_sequence) + "\n")
                
                for (l,r), entity_type in chunk_to_entitytype.items():
                    f.write(f"{l} {r} {entity_type}\n")
                f.write("\n")
                    
        samples = len(sample_list)
        tokens = sum(len(token_sequence) for token_sequence, _ in sample_list)
        entities = sum(len(chunk_to_entitytype) for _, chunk_to_entitytype in sample_list)
        
        print(f" samples={samples} tokens={tokens} entities={entities}")
        entitytype_to_count = defaultdict(lambda: 0)
        for _, chunk_to_entitytype in sample_list:
            for entity_type in chunk_to_entitytype.values():
                entitytype_to_count[entity_type] += 1
        for entity_type, count in sorted(entitytype_to_count.items(), key=lambda x: x[1], reverse=True):
            print(entity_type, count)
        
    return
    
def read_data(file_path):
    print(f"Reading from {file_path}... ", end="")
    
    with open(file_path, "r") as f:
        line_list = f.read().splitlines()
    
    sample_list = []
    index = 0
    
    while index < len(line_list):
        token_sequence = line_list[index].split(" ")
        tokenlabel_sequence = line_list[index+1].split(" ")
        index += 2
        
        chunk_to_entitytype = {}
        while True:
            line = line_list[index]
            index += 1
            if not line: break
            left, right, entity_type = line.split(" ")
            chunk_to_entitytype[(int(left),int(right))] = entity_type
        
        sample_list.append((token_sequence, tokenlabel_sequence, chunk_to_entitytype))
    
    sentences = len(sample_list)
    tokens = sum(len(token_sequence) for token_sequence, _, _ in sample_list)
    entities = sum(len(chunk_to_entitytype) for _, _, chunk_to_entitytype in sample_list)
    print(f"{sentences} sentences, {tokens} tokens, {entities} entities")
    return sample_list
    
def read_dataset(data_split_list = ["train", "validate", "test"], write_category_list=False):
    # Read data
    split_to_samplelist = {}
    for split in data_split_list:
        split_to_samplelist[split] = read_data(split_data_file[split])
        
    # Compute entity type distribution
    entitytype_to_count = defaultdict(lambda: 0)
    for split, sample_list in split_to_samplelist.items():
        for _, _, chunk_to_entitytype in sample_list:
            for _, entity_type in chunk_to_entitytype.items():
                entitytype_to_count[entity_type] += 1
                
    # Show entity type distribution
    entities = sum(entitytype_to_count.values())
    print(f"Total {entities} entities")
    print("-"*80)
    entity_type, count, ratio = "type", "count", "ratio"
    print(f"{entity_type:>13s} {count:>6s} {ratio:>6s}")
    print("-"*80)
    entitytype_count_list = sorted(entitytype_to_count.items(), key=lambda x: x[1], reverse=True)
    for entity_type, count in entitytype_count_list[:5]:
        ratio = count / entities
        print(f"{entity_type:>13s} {count:6d} {ratio:6.1%}")
    if write_category_list:
        with open(tokenlabel_list_file, "w") as f:
            f.write("NONE:O\n")
            for entity_type, count in entitytype_count_list:
                f.write(entity_type + ":B\n")
                f.write(entity_type + ":I\n")
                f.write(entity_type + ":E\n")
                f.write(entity_type + ":S\n")
                
    # Read label list
    with open(tokenlabel_list_file, "r") as f:
        tokenlabel_list = f.read().splitlines()
    tokenlabel_to_index = {token_label: index for index, token_label in enumerate(tokenlabel_list)}
    
    return split_to_samplelist, tokenlabel_list, tokenlabel_to_index
    
def load_embedding():
    # embedding_dir = "/share/home/jacobvsdanniel/glove"
    embedding_dir = "/share/home/jacobvsdanniel/NER/twitter_embedding/word2vec_twitter_model"
    print(f"Loading embedding from {embedding_dir}...", end="", flush=True)
    
    word_file = os.path.join(embedding_dir, "token.npy")
    word_list = np.load(word_file)
    
    vector_file = os.path.join(embedding_dir, "vector.npy")
    vector_list = np.load(vector_file)
    
    w_token_to_vector = dict(zip(word_list, vector_list))
    n = len(w_token_to_vector)
    d = vector_list.shape[1]
    
    print(f" [{n}, {d}]")
    return w_token_to_vector, d
    
def main():
    # extract_data()
    # read_dataset(write_category_list=True)
    
    read_dataset()
    # load_embedding()
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    
