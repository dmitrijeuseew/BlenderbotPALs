import torch
from transformers import BlenderbotTokenizer
from optim import Adam
from modeling_blenderbot import BlenderbotForConditionalGeneration
from deeppavlov import configs, build_model
from prepare_data import prepare_dialog_acts_train, prepare_dialog_acts_test


classifier = build_model("dialog_acts_hist.json", download=False)

device = torch.device("cuda:4")
print("device", device)

mname = 'facebook/blenderbot-400M-distill'
path = "/archive/evseev/.deeppavlov/models/blenderbot"
model = BlenderbotForConditionalGeneration.from_pretrained(path)
model.return_dict = None
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

tokenizer = BlenderbotTokenizer.from_pretrained(path)
optimizer = Adam(optimizer_grouped_parameters, 0.001, max_grad_norm=1.0)

model.setup_trainable_parts([0], train_base_model=False)

batches_train = prepare_dialog_acts_train(10)
batches_test = prepare_dialog_acts_test(10)

weights_path = "/archive/evseev/.deeppavlov/models/blenderbot/blenderbot.pth.tar"

model.train()

loss_sum = 0.0
prev_accuracy = 0.0
for n, batch in enumerate(batches_train):
    dialog_act, utt_batch = batch
    dialog_act = int(dialog_act)
    utt_input = [elem[0] for elem in utt_batch]
    utt_output = [elem[1] for elem in utt_batch]
    input_encoding = tokenizer.batch_encode_plus(utt_input, add_special_tokens = True, pad_to_max_length=True,
                                       return_attention_mask = True)
    output_encoding = tokenizer.batch_encode_plus(utt_output, add_special_tokens = True, pad_to_max_length=True,
                                       return_attention_mask = True)
    input_ids = input_encoding["input_ids"]
    attention_mask = input_encoding["attention_mask"]
    labels = output_encoding["input_ids"]
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)
    labels = torch.LongTensor(labels).to(device)
    
    model.choose_workers_in_branches([dialog_act])
    loss, *_ = model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     labels=labels,
                     return_dict=False)
    loss.backward()
    loss_sum += float(loss.detach())
    if n%100 == 0 and n > 0:
        print("train examples seen", round(n / len(batches_train), 2), "loss", round(loss_sum, 3))
        out = open("train_logs.txt", 'a')
        out.write(f"train examples seen {round(n / len(batches_train), 2)} loss {round(loss_sum, 3)}"+'\n')
        out.close()
        loss_sum = 0.0
        
    if n%100 == 0:
        correct = 0
        correct_dict = {}
        all_true_dialog_acts = []
        all_pred_dialog_acts = []
        for true_dialog_act, batch_test in batches_test:
            utt1_input = [elem[0] for elem in batch_test]
            utt2_input = [elem[1] for elem in batch_test]
            test_encoding = tokenizer.batch_encode_plus(utt1_input, add_special_tokens = True, pad_to_max_length=True,
                                                        return_attention_mask = True)
            test_input_ids = test_encoding["input_ids"]
            test_attention_mask = test_encoding["attention_mask"]
            test_input_ids = torch.LongTensor(test_input_ids).to(device)
            test_attention_mask = torch.LongTensor(test_attention_mask).to(device)
            model.choose_workers_in_branches([int(true_dialog_act)])
            with torch.no_grad():
                reply_ids = model.generate(input_ids=test_input_ids, attention_mask=test_attention_mask)
            
            reply_uttr = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
            clas_dialog_acts = classifier(utt1_input, reply_uttr)
            for clas_dialog_act in clas_dialog_acts:
                if clas_dialog_act == true_dialog_act:
                    correct += 1
                    if true_dialog_act in correct_dict:
                        correct_dict[true_dialog_act] += 1
                    else:
                        correct_dict[true_dialog_act] = 1
                all_pred_dialog_acts.append(int(clas_dialog_act))
                all_true_dialog_acts.append(int(true_dialog_act))
        
        cur_accuracy = round(correct / 200, 3)
        print("accuracy", cur_accuracy, correct_dict)
        out = open("train_logs.txt", 'a')
        out.write(f"accuracy {cur_accuracy} correct {correct_dict}"+'\n')
        out.close()
        if cur_accuracy > prev_accuracy:
            torch.save({
            "model_state_dict": model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            }, weights_path)
            model.to(device)
        prev_accuracy = cur_accuracy
            
    optimizer.step()
    optimizer.zero_grad()

