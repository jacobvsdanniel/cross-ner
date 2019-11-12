import os
import sys
from collections import defaultdict

dataset = "wnut2017"
# dataset = "ontonotes"

def read_prediction_file(prediction_file):
    with open(prediction_file, "r") as f:
        line_list = f.read().splitlines()
        
    i = 0
    sample_list = []
    while i < len(line_list):
        word_sequence = line_list[i].split(" ")
        gold_sequence = line_list[i+1].split(" ")
        auto_sequence = line_list[i+2].split(" ")
        i += 4
        sample_list.append((word_sequence, gold_sequence, auto_sequence))
        
    return sample_list
    
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
    
def get_score(golds, autos, overlaps):
    try:
        precision = overlaps / autos
    except ZeroDivisionError:
        precision = 1
        
    try:
        recall = overlaps / golds
    except ZeroDivisionError:
        recall = 1
        
    try:
        f1 = 2*precision*recall / (precision+recall)
    except ZeroDivisionError:
        f1 = 0
        
    metric_to_score = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metric_to_score
    
def evaluate_entity_type_prediction(sample_list):
    entitytype_to_golds = defaultdict(lambda: 0)
    entitytype_to_autos = defaultdict(lambda: 0)
    entitytype_to_overlaps = defaultdict(lambda: 0)
    
    for _, gold_tokenlabel_sequence, auto_tokenlabel_sequence in sample_list:
        gold_chunk_to_entitytype = get_chunk_to_entitytype(gold_tokenlabel_sequence)
        auto_chunk_to_entitytype = get_chunk_to_entitytype(auto_tokenlabel_sequence)
        
        for chunk, entity_type in gold_chunk_to_entitytype.items():
            entitytype_to_golds[entity_type] += 1
            
        for chunk, entity_type in auto_chunk_to_entitytype.items():
            entitytype_to_autos[entity_type] += 1
            
            if chunk in gold_chunk_to_entitytype and gold_chunk_to_entitytype[chunk]==entity_type:
                entitytype_to_overlaps[entity_type] += 1
                
    # golds = sum(entitytype_to_golds.values())
    # autos = sum(entitytype_to_autos.values())
    # overlaps = sum(entitytype_to_overlaps.values())
    # precision, recall, f1 = get_score(golds, autos, overlaps)
    # print(f"f1={f1:.2%}")
    
    entitytype_metric_score = {}
    for entity_type in entitytype_to_golds:
        entitytype_metric_score[entity_type] = get_score(
            entitytype_to_golds[entity_type],
            entitytype_to_autos[entity_type],
            entitytype_to_overlaps[entity_type],
        )
    return entitytype_metric_score
    
def analyze_entity_type_performance():
    model_list = ["baseline", "cross", "att"]
    trials = 6
    
    entity_type_list = []
    
    for model in model_list:
        metric_entitytype_score = defaultdict(lambda: defaultdict(lambda: 0))
        
        for trial in range(1, 1+trials):
            prediction_file = os.path.join("prediction", dataset, f"{model}{trial}.txt")
            sample_list = read_prediction_file(prediction_file)
            trial_entitytype_metric_score = evaluate_entity_type_prediction(sample_list)
            
            for entity_type, metric_to_score in trial_entitytype_metric_score.items():
                for metric, score in metric_to_score.items():
                    metric_entitytype_score[metric][entity_type] += score
                    
        for metric, entitytype_to_score in metric_entitytype_score.items():
            score_file = os.path.join("prediction", dataset, f"{model}_{metric}.txt")
            with open(score_file, "w") as f:
                if not entity_type_list:
                    entity_type_list = sorted(entitytype_to_score)
                else:
                    assert entity_type_list == sorted(entitytype_to_score)
                for entity_type in entity_type_list:
                    score = entitytype_to_score[entity_type] / trials
                    f.write(f"{score:.2%}\n")
    
    entity_type_file = os.path.join("prediction", dataset, "entity_type.txt")
    with open(entity_type_file, "w") as f:
        for entity_type in entity_type_list:
            f.write(f"{entity_type}\n")
    return
    
def evaluate_entity_length_prediction(sample_list):
    entitylength_to_golds = defaultdict(lambda: 0)
    entitylength_to_autos = defaultdict(lambda: 0)
    entitylength_to_overlaps = defaultdict(lambda: 0)
    
    def map_length(length):
        # return length
        if length == 1:
            length = 1
        elif length == 2:
            length = 2
        elif length >= 3:
            length = 3
        else:
            assert False
        return length
    
    for _, gold_tokenlabel_sequence, auto_tokenlabel_sequence in sample_list:
        gold_chunk_to_entitytype = get_chunk_to_entitytype(gold_tokenlabel_sequence)
        auto_chunk_to_entitytype = get_chunk_to_entitytype(auto_tokenlabel_sequence)
        
        for chunk, entity_type in gold_chunk_to_entitytype.items():
            entity_length = chunk[1] - chunk[0]
            entitylength_to_golds[map_length(entity_length)] += 1
            
        for chunk, entity_type in auto_chunk_to_entitytype.items():
            entity_length = chunk[1] - chunk[0]
            entitylength_to_autos[map_length(entity_length)] += 1
            
            if chunk in gold_chunk_to_entitytype and gold_chunk_to_entitytype[chunk]==entity_type:
                entity_length = chunk[1] - chunk[0]
                entitylength_to_overlaps[map_length(entity_length)] += 1
                
    golds = sum(entitylength_to_golds.values())
    autos = sum(entitylength_to_autos.values())
    overlaps = sum(entitylength_to_overlaps.values())
    print(get_score(golds, autos, overlaps))
    
    entitylength_metric_score = {}
    for entity_length in entitylength_to_golds:
        entitylength_metric_score[entity_length] = get_score(
            entitylength_to_golds[entity_length],
            entitylength_to_autos[entity_length],
            entitylength_to_overlaps[entity_length],
        )
    return entitylength_metric_score
    
def analyze_entity_length_performance():
    model_list = ["baseline", "cross", "att"]
    trials = 6
    
    entity_length_list = []
    
    for model in model_list:
        metric_entitylength_score = defaultdict(lambda: defaultdict(lambda: 0))
        
        for trial in range(1, 1+trials):
            prediction_file = os.path.join("prediction", dataset, f"{model}{trial}.txt")
            sample_list = read_prediction_file(prediction_file)
            trial_entitylength_metric_score = evaluate_entity_length_prediction(sample_list)
            
            for entity_length, metric_to_score in trial_entitylength_metric_score.items():
                for metric, score in metric_to_score.items():
                    metric_entitylength_score[metric][entity_length] += score
                    
        for metric, entitylength_to_score in metric_entitylength_score.items():
            # score_file = os.path.join("prediction", dataset, f"length_{model}_{metric}.txt")
            score_file = os.path.join("prediction", dataset, f"length_range_{model}_{metric}.txt")
            with open(score_file, "w") as f:
                if not entity_length_list:
                    entity_length_list = sorted(entitylength_to_score)
                else:
                    assert entity_length_list == sorted(entitylength_to_score)
                for entity_length in entity_length_list:
                    score = entitylength_to_score[entity_length] / trials
                    f.write(f"{score:.2%}\n")
                    
    # entity_length_file = os.path.join("prediction", dataset, "entity_length.txt")
    entity_length_file = os.path.join("prediction", dataset, "entity_length_range.txt")
    with open(entity_length_file, "w") as f:
        for entity_length in entity_length_list:
            f.write(f"{entity_length}\n")
    return
    
def main():
    # analyze_entity_type_performance()
    analyze_entity_length_performance()
    return
    
if __name__ == "__main__":
    main()
    sys.exit()
    