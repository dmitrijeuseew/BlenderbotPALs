import pickle
import random
from transformers import BlenderbotTokenizer


path = "/archive/evseev/.deeppavlov/models/blenderbot"
tokenizer = BlenderbotTokenizer.from_pretrained(path)

def prepare_dialog_acts_train(batch_size):
    dialog_act_dict = {}
    fl = open("/archive/evseev/.deeppavlov/downloads/dialog_acts/dialog_acts_hist.pickle", 'rb')
    data = pickle.load(fl)
    for uttr, label in data["train"]:
        if uttr[0] != "<begin>":
            inputs1 = tokenizer([uttr[0]], return_tensors='pt')
            inputs2 = tokenizer([uttr[1]], return_tensors='pt')
            if len(inputs1["input_ids"][0]) < 128 and len(inputs2["input_ids"][0]) < 128:
                if label not in dialog_act_dict:
                    dialog_act_dict[label] = [uttr]
                else:
                    dialog_act_dict[label].append(uttr)
    batches = []
    for label in dialog_act_dict:
        elements = dialog_act_dict[label]
        num_batches = len(elements) // batch_size + int(len(elements) % batch_size > 0)
        for i in range(num_batches):
            batches.append((label, elements[i*batch_size:(i+1)*batch_size]))
    random.shuffle(batches)
    
    return batches


def prepare_dialog_acts_test(batch_size):
    dialog_act_dict = {}
    fl = open("/archive/evseev/.deeppavlov/downloads/dialog_acts/dialog_acts_hist.pickle", 'rb')
    data = pickle.load(fl)
    for uttr, label in data["test"]:
        if uttr[0] != "<begin>":
            inputs1 = tokenizer([uttr[0]], return_tensors='pt')
            inputs2 = tokenizer([uttr[1]], return_tensors='pt')
            if len(inputs1["input_ids"][0]) < 128 and len(inputs2["input_ids"][0]) < 128:
                if label not in dialog_act_dict:
                    dialog_act_dict[label] = [uttr]
                else:
                    dialog_act_dict[label].append(uttr)
    batches = []
    for label in dialog_act_dict:
        elements = dialog_act_dict[label]
        num_batches = len(elements) // batch_size + int(len(elements) % batch_size > 0)
        for i in range(5):
            batches.append((label, elements[i*batch_size:(i+1)*batch_size]))
    
    return batches
