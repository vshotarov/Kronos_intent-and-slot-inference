from transformers import DistilBertTokenizer, DistilBertModel, logging
logging.set_verbosity_error()
import torch

from html.parser import HTMLParser

import sys
RECOGNIZED_SLOTS = ["location", "time"]
INTENTS = ["time.current","time.timer","weather.current","weather.future","other"]
SLOTS = ["PAD", "O", "B-location", "I-location", "B-time", "I-time"]
NUM_SLOTS = len(SLOTS)

class SlotParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        assert tag in RECOGNIZED_SLOTS, tag + " not recognized"

        self.last_tag = tag

    def handle_endtag(self, tag):
        self.last_tag = None

    def handle_data(self, data):
        self.cleaned += data
        if self.last_tag is None:
            self.tokens += ["O" for each in data.split()]
        else:
            self.tokens.append("B-" + self.last_tag)
            self.tokens += ["I-" + self.last_tag for each in data.split()[1:]]

    def tokenize(self, prompt):
        self.tokens = []
        self.cleaned = ""
        self.last_tag = None

        self.feed(prompt)

        return self.cleaned, self.tokens

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, encodings):
        super(Dataset, self).__init__()

        self.data = data
        self.encodings = encodings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, slots, intent = self.data[idx]
        _input = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        intent_id = INTENTS.index(intent)
        slot_ids = [SLOTS.index(slot) for slot in slots.split()]

        return _input, attention_mask, intent_id, slot_ids, prompt

def pad(data):
    inputs = []
    attention_masks = []
    intents = []
    slots = []
    prompts = []

    for _input, mask, intent_id, slot_ids, prompt in data:
        inputs.append(_input.unsqueeze(0))
        attention_masks.append(mask.unsqueeze(0))
        intents.append(intent_id)
        slot_ids = [0] + slot_ids + [0 for _ in range(len(_input) - len(slot_ids) - 1)]
        slots.append(torch.tensor(slot_ids).unsqueeze(0))
        prompts.append(prompt)

    inputs = torch.cat(inputs)
    attention_masks = torch.cat(attention_masks)
    intents = torch.tensor(intents)
    slots = torch.cat(slots)

    return inputs, attention_masks, intents, slots, prompts

slot_parser = SlotParser()

def prepare_data(tokenizer, dataset_file):
    data = []
    with open(dataset_file, "r") as f:
        for line in f.readlines():
            prompt, intent = line.strip().split(",")
            clean_prompt, slots = slot_parser.tokenize(prompt)
            clean_prompt_split = clean_prompt.split()
            tokenized_prompt = []
            tokenized_slots = []
            for word, slot in zip(clean_prompt_split, slots):
                bert_tokenized = tokenizer.tokenize(word)
                tokenized_prompt += bert_tokenized
                tokenized_slots.append(slot)
                for i in range(len(bert_tokenized) - 1):
                    tokenized_slots.append(slot.replace("B-","I-"))
            #data.append((" ".join(clean_prompt_split), " ".join(tokenized_slots), intent))
            data.append((" ".join(tokenized_prompt), " ".join(tokenized_slots), intent))

    return data

def train(tokenizer, bert, intent_model, slot_model, train_loader, test_loader, epochs=20):
    intent_model.train()
    slot_model.train()

    optimizer = torch.optim.Adam([p for p in intent_model.parameters()] +\
            [p for p in slot_model.parameters()], lr=1e-4)

    intent_criterion = torch.nn.CrossEntropyLoss()
    slot_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    best_test_loss = float("inf")

    for i in range(epochs):
        epoch_loss = 0
        for j, (input_ids, attention_mask, intents, slots, _) in enumerate(train_loader):
            bert_outputs = bert(input_ids, attention_mask=attention_mask)[0]

            intent_outputs = intent_model(bert_outputs[:,0])
            slot_outputs = slot_model(bert_outputs).view(-1, NUM_SLOTS)

            intent_loss = intent_criterion(intent_outputs, intents)
            slot_loss = slot_criterion(slot_outputs, slots.view(-1))

            total_loss = intent_loss + slot_loss
            total_loss.backward()

            optimizer.step()

            epoch_loss += total_loss.item()

            print("Train epoch %i: Batch %i : loss %f\r" % (i, j, epoch_loss / (j+1)), end="")
        print()

        if i % 5 == 0:
            with torch.no_grad():
                test_loss = 0
                for j, (input_ids, attention_mask, intents, slots, prompts) in enumerate(test_loader):
                    bert_outputs = bert(input_ids, attention_mask=attention_mask)[0]

                    intent_outputs = intent_model(bert_outputs[:,0])
                    slot_outputs = slot_model(bert_outputs)

                    intent_loss = intent_criterion(intent_outputs, intents)
                    slot_loss = slot_criterion(slot_outputs.view(-1, NUM_SLOTS), slots.view(-1))

                    total_loss = intent_loss + slot_loss
                    #total_loss.backward()

                    #optimizer.step()

                    test_loss += total_loss.item()

                    print("###### ")
                    print("Test: Batch %i : loss %f (i: %f, s: %f)" % (j, total_loss, intent_loss, slot_loss))

                    for i in range(intents.shape[0]):
                        this_intent_outputs = intent_outputs[i]
                        this_slot_outputs = slot_outputs[i][1:len(prompts[i].split())+1]
                        this_intent = torch.argmax(torch.nn.functional.softmax(this_intent_outputs, dim=0))
                        this_slots = torch.argmax(torch.nn.functional.softmax(this_slot_outputs, dim=1), dim=1)
                        print(prompts[i], "; intent: ", INTENTS[this_intent.item()],
                                "; slots: ", " ".join([SLOTS[x] for x in this_slots.detach()]),
                                torch.argmax(torch.nn.functional.softmax(slot_outputs[i], dim=1), dim=1),
                                slots[i])
                        print("this loss: ", slot_criterion(slot_outputs[i], slots[i]))
                print("Total test loss: %f" % (test_loss / (j+1)))

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(intent_model, "saved_intent_model.torch")
                    torch.save(slot_model, "saved_slot_model.torch")
                    print("Saved.")

    intent_model.eval()
    slot_model.eval()

    epoch_loss = 0
    for j, (input_ids, attention_mask, intents, slots, prompts) in enumerate(test_loader):
        bert_outputs = bert(input_ids, attention_mask=attention_mask)[0]

        intent_outputs = intent_model(bert_outputs[:,0])
        slot_outputs = slot_model(bert_outputs)

        intent_loss = intent_criterion(intent_outputs, intents)
        slot_loss = slot_criterion(slot_outputs.view(-1, NUM_SLOTS), slots.view(-1))

        total_loss = intent_loss + slot_loss
        #total_loss.backward()

        #optimizer.step()

        epoch_loss += total_loss.item()

        print("###### ")
        print("Test: Batch %i : loss %f" % (j, total_loss))

        for i in range(intents.shape[0]):
            this_intent_outputs = intent_outputs[i]
            this_slot_outputs = slot_outputs[i][1:len(prompts[i].split())+1]
            this_intent = torch.argmax(torch.nn.functional.softmax(this_intent_outputs, dim=0))
            this_slots = torch.argmax(torch.nn.functional.softmax(this_slot_outputs, dim=1), dim=1)
            print(prompts[i], "; intent: ", INTENTS[this_intent.item()],
                    "; slots: ", " ".join([SLOTS[x] for x in this_slots.detach()]))
    print("Total test loss: %f" % (epoch_loss / (j+1)))

def test(tokenizer, bert, intent_model, slot_model, train_loader, test_loader):
    intent_model.eval()
    slot_model.eval()

    intent_criterion = torch.nn.CrossEntropyLoss()
    slot_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    epoch_loss = 0
    for j, (input_ids, attention_mask, intents, slots, prompts) in enumerate(test_loader):
        bert_outputs = bert(input_ids, attention_mask=attention_mask)[0]

        intent_outputs = intent_model(bert_outputs[:,0])
        slot_outputs = slot_model(bert_outputs)

        intent_loss = intent_criterion(intent_outputs, intents)
        slot_loss = slot_criterion(slot_outputs.view(-1, NUM_SLOTS), slots.view(-1))

        total_loss = intent_loss + slot_loss
        #total_loss.backward()

        #optimizer.step()

        epoch_loss += total_loss.item()

        print("###### ")
        print("Test: Batch %i : loss %f" % (j, total_loss))

        for i in range(intents.shape[0]):
            this_intent_outputs = intent_outputs[i]
            this_slot_outputs = slot_outputs[i][1:len(prompts[i].split())+1]
            this_intent = torch.argmax(torch.nn.functional.softmax(this_intent_outputs, dim=0))
            this_slots = torch.argmax(torch.nn.functional.softmax(this_slot_outputs, dim=1), dim=1)
            print(prompts[i], "; intent: ", INTENTS[this_intent.item()],
                    "; slots: ", " ".join([SLOTS[x] for x in this_slots.detach()]))
    print("Total test loss: %f" % (epoch_loss / (j+1)))

def getIntentAndSlotModels(intent_path, slot_path):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)

    intent_model = torch.load(intent_path)
    slot_model = torch.load(slot_path)

    return tokenizer, model, intent_model, slot_model

def inferIntentAndSlots(x, models):
    encoding = models[0]([x], return_tensors="pt", padding=True)

    bert_outputs = models[1](encoding["input_ids"], attention_mask=encoding["attention_mask"])[0]

    intent_outputs = models[2](bert_outputs[:,0])
    slot_outputs = models[3](bert_outputs)

    tokenized = []
    x_split = x.split()
    while len(x_split):
        tokenized += models[0].tokenize(x_split.pop(0))

    i = 0
    this_intent_outputs = intent_outputs[i]
    this_slot_outputs = slot_outputs[i][1:encoding["input_ids"].shape[1]-1]
    this_intent = torch.argmax(torch.nn.functional.softmax(this_intent_outputs, dim=0))
    this_slots = torch.argmax(torch.nn.functional.softmax(this_slot_outputs, dim=1), dim=1)
    #print(" ".join(tokenized), "; intent: ", INTENTS[this_intent.item()],
    #        "; slots: ", " ".join([SLOTS[k] for k in this_slots.detach()]))
    recognized_slots = {}
    for word, slot in zip(tokenized, this_slots):
        if slot > 1:
            slot_type, slot_name = SLOTS[slot].split("-")
            if slot_name not in recognized_slots.keys():
                recognized_slots[slot_name] = []
            recognized_slots[slot_name].append(word)
    for slot_name,slot_value in recognized_slots.items():
        recognized_slots[slot_name] = " ".join(slot_value)

    return torch.nn.functional.softmax(intent_outputs[0], dim=0)[this_intent.item()], INTENTS[this_intent.item()], recognized_slots

if __name__ == "__main__":
    TRAIN_DATASET_FILE = sys.argv[1]
    TEST_DATASET_FILE = sys.argv[2]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)

    train_data = prepare_data(tokenizer, TRAIN_DATASET_FILE)
    test_data = prepare_data(tokenizer, TEST_DATASET_FILE)
    train_encodings = tokenizer([x[0] for x in train_data], return_tensors="pt", padding=True)
    test_encodings = tokenizer([x[0] for x in test_data], return_tensors="pt", padding=True)

    train_dataset = Dataset(train_data, train_encodings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, collate_fn=pad)

    test_dataset = Dataset(test_data, test_encodings)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=pad)

    intent_model = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768,len(INTENTS)))

    slot_model = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(768,len(SLOTS)))

    train(tokenizer, model, intent_model, slot_model, train_loader, test_loader, 100)

    torch.save(intent_model, "intent_model.torch")
    torch.save(slot_model, "slot_model.torch")
