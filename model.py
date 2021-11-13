from transformers import DistilBertTokenizer, DistilBertModel, logging
logging.set_verbosity_error()
import torch


class JointIntentAndSlotsModel(torch.nn.Module):
    def __init__(self, intents=[], recognized_slots=[]):
        super(JointIntentAndSlotsModel, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")

        self.recognized_slots = recognized_slots
        self.slots = ["PAD", "O"]
        for slot_name in recognized_slots:
            self.slots += ["B-" + slot_name, "I-" + slot_name]

        self.intents = intents

        self.num_intents = len(intents)
        self.num_slots = len(self.slots)

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", output_hidden_states=True)

        if intents and recognized_slots:
            self._init_models()

    def _init_models(self):
        self.intent_model = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(768,self.num_intents))

        self.slot_model = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(768,self.num_slots))

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[0]

        intent_outputs = self.intent_model(bert_outputs[:,0])
        slot_outputs = self.slot_model(bert_outputs)

        return intent_outputs, slot_outputs

    def parameters(self):
        for x in list(self.intent_model.parameters()) + list(self.slot_model.parameters()):
            yield x

    def eval(self):
        #super(JointIntentAndSlotsModel, self).eval()
        self.intent_model.eval()
        self.slot_model.eval()

    def train(self):
        #super(JointIntentAndSlotsModel, self).train()
        self.intent_model.train()
        self.slot_model.train()

    def state_dict(self):
        return {"intent":self.intent_model.state_dict(),
                "slot":self.slot_model.state_dict(),
                "intents":self.intents,"slots":self.slots}

    def load_state_dict(self, state_dict):
        self.intents = state_dict["intents"]
        self.slots = state_dict["slots"]
        self.num_intents = len(self.intents)
        self.num_slots = len(self.slots)
        self._init_models()

        self.intent_model.load_state_dict(state_dict["intent"])
        self.slot_model.load_state_dict(state_dict["slot"])

    def inferIntentAndSlots(self, x):
        encoding = self.tokenizer([x], return_tensors="pt", padding=True)
        bert_outputs = self.bert(
            encoding["input_ids"], attention_mask=encoding["attention_mask"])[0]

        intent_outputs = self.intent_model(bert_outputs[:,0])
        slot_outputs = self.slot_model(bert_outputs)

        tokenized = []
        x_split = x.split()
        while len(x_split):
            tokenized += self.tokenizer.tokenize(x_split.pop(0))

        i = 0
        this_intent_outputs = intent_outputs[i]
        this_slot_outputs = slot_outputs[i][1:encoding["input_ids"].shape[1]-1]
        this_intent = torch.argmax(torch.nn.functional.softmax(this_intent_outputs, dim=0))
        this_slots = torch.argmax(torch.nn.functional.softmax(this_slot_outputs, dim=1), dim=1)
        #print(" ".join(input_tokens), "; intent: ", self.intents[this_intent.item()],
        #        "; slots: ", " ".join([self.slots[k] for k in this_slots.detach()]))
        recognized_slots = {}
        for word, slot in zip(tokenized, this_slots):
            if slot > 1:
                slot_type, slot_name = self.slots[slot].split("-")
                if slot_name not in recognized_slots.keys():
                    recognized_slots[slot_name] = []
                recognized_slots[slot_name].append(word)

        for slot_name,slot_value in recognized_slots.items():
            joined_value = ""
            for v in slot_value:
                if "#" not in v:
                    if joined_value:
                        joined_value += " "
                    joined_value += v
                else:
                    joined_value += v.replace("#","")

            recognized_slots[slot_name] = joined_value

        return torch.nn.functional.softmax(intent_outputs[0], dim=0)[this_intent.item()],\
            self.intents[this_intent.item()], recognized_slots

