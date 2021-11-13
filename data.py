import torch
from html.parser import HTMLParser


class SlotParser(HTMLParser):
    def __init__(self, recognized_slots):
        super(SlotParser, self).__init__()
        self.recognized_slots = recognized_slots

    def handle_starttag(self, tag, attrs):
        assert tag in self.recognized_slots, tag + " not recognized"

        self.last_tag = tag

    def handle_endtag(self, tag):
        self.last_tag = None

    def handle_data(self, data):
        self.cleaned += data
        if self.last_tag is None:
            self.tokens += ["O" for each in data.split()]
        else:
            # NOTE: Maybe we should also have a separate token, for
            # only having one word part of the slot, rather than using
            # B- for both
            self.tokens.append("B-" + self.last_tag)
            self.tokens += ["I-" + self.last_tag for each in data.split()[1:]]

    def tokenize(self, prompt):
        self.tokens = []
        self.cleaned = ""
        self.last_tag = None

        self.feed(prompt)

        return self.cleaned, self.tokens


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, tokenizer, intents, slots, slot_names):
        super(Dataset, self).__init__()

        self.intents = intents
        self.slots = slots

        self.slot_parser = SlotParser(slot_names)

        self.data = self.tokenize_data(dataset_file, tokenizer)
        self.encodings = tokenizer(
            [x[0] for x in self.data], return_tensors="pt", padding=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, slots, intent = self.data[idx]
        _input = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        intent_id = self.intents.index(intent)
        slot_ids = [self.slots.index(slot) for slot in slots.split()]

        return _input, attention_mask, intent_id, slot_ids, prompt

    def tokenize_data(self, dataset_file, tokenizer):
        data = []
        with open(dataset_file, "r") as f:
            for line in f.readlines():
                prompt, intent = line.strip().split(",")
                clean_prompt, slots = self.slot_parser.tokenize(prompt)
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

