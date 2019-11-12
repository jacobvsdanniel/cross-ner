import os
import sys
from collections import defaultdict

import numpy as np

dataset = "conll2003"
tokenlabel_list_file = os.path.join(dataset, "token_label_list.txt")

split_raw_file = {
    "train":    os.path.join(dataset, "eng.train"),
    "validate": os.path.join(dataset, "eng.testa"),
    "test":     os.path.join(dataset, "eng.testb")
}

split_data_file = {
    "train":    os.path.join(dataset, "train.txt"),
    "validate": os.path.join(dataset, "validate.txt"),
    "test":     os.path.join(dataset, "test.txt")
}

def get_chunk_to_entitytype(conll_tokenlabel_sequence):
    chunk_to_entitytype = {}
    
    left, entity_type = None, None
    for index, token_label in enumerate(conll_tokenlabel_sequence + ["O"]):
        if (token_label[0]=="O" or token_label[0]=="B") and entity_type:
            chunk_to_entitytype[(left, index)] = entity_type
            left, entity_type = None, None
        
        if token_label[0]=="B" or (token_label[0]=="I" and not entity_type):
            left, entity_type = index, token_label[2:]
        
    return chunk_to_entitytype
    
def get_tokenlabel_sequence(chunk_to_entitytype, sequence_length):
    tokenlabel_sequence = ["NONE:O"] * sequence_length
    
    for (left, right), entity_type in chunk_to_entitytype.items():
        if left == right-1:
            tokenlabel_sequence[left] = entity_type + ":S"
        else:
            tokenlabel_sequence[left] = entity_type + ":B"
            tokenlabel_sequence[right-1] = entity_type + ":E"
            for i in range(left+1, right-1):
                tokenlabel_sequence[i] = entity_type + ":I"
    return tokenlabel_sequence
    
def extract_data():
    for split in ["train", "validate", "test"]:
        print(f"Extracting [{split}] split... ", end="", flush=True)
        
        with open(split_raw_file[split], "r") as f:
            line_list = f.read().splitlines()
        
        with open(split_data_file[split], "w") as f:
            sentences, tokens, entities = 0, 0, 0
            token_sequence, tokenlabel_sequence = [], []
            
            for line in line_list:
                if not line:
                    if not token_sequence: continue
                    chunk_to_entitytype = get_chunk_to_entitytype(tokenlabel_sequence)
                    tokenlabel_sequence = get_tokenlabel_sequence(chunk_to_entitytype, len(token_sequence))
                    
                    f.write(" ".join(token_sequence) + "\n")
                    f.write(" ".join(tokenlabel_sequence) + "\n")
                    for (left, right), entity_type in sorted(chunk_to_entitytype.items(), key=lambda x: x[0][0]):
                        f.write(f"{left} {right} {entity_type}\n")
                    f.write("\n")
                    
                    sentences += 1
                    tokens += len(token_sequence)
                    entities += len(chunk_to_entitytype)
                    token_sequence, tokenlabel_sequence = [], []
                else:
                    if line[:10] == "-DOCSTART-":  continue
                    token, _, _, token_label = line.split()
                    token_sequence.append(token)
                    tokenlabel_sequence.append(token_label)
    
        print(f"{sentences} sentences, {tokens} tokens, {entities} entities")
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
    embedding_dir = "/share/home/jacobvsdanniel/glove"
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
    