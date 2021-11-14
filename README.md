# Kronos - Intent and slot inference
The intent and slot inference module for [Kronos - my personal virtual assistant proof of concept](https://github.com/vshotarov/Kronos).

## Table of Contents
<ol>
<li><a href="#glossary-and-desired-behaviour">Glossary and Desired Behaviour</a></li>
<li><a href="#overview">Overview</a></li>
<li><a href="#training">Training</a></li>
</ol>

## Glossary and Desired Behaviour
**Intent** - describes the desired action from the input text. It will always be one of the provided intents during training. Common examples are querying the time, interacting with smart appliances, such as turning lights on/off, etc. If we are to think of the desired action from the text as a function, then the **intent** would be the function's name.

**Slots** - a lot of intents require further information to be useful. For example, querying the time could accept a location around the globe, turning the lights on and off might require specifying *on* or *off*, or maybe even a colour, etc. We call those additional pieces of information *slots*. Similar to the intents, we only recognized a finite list of slots, that the model has been trained on. In the examples above we have a location, and light's state/colour. If we are to think of the desired action from the text as a function, then the **slots** would be the function arguments.

A quick example:

```
intents: querying current time, toggling lights state, other
slots: world location, toggle state

Given the sentence: "What's the time in Barcelona", we would like to receive the following outputs:
	( intent = querying current time, slots = {world_location: "Barcelona"} )

Given the sentence: "Where is my cat", we woud like to receive the following outputs:
	( intent = other, slots = {} )
```

## Overview
The intent and slot inference module is responsible for converting a text sample into a pair of a recognized intent and a dictionary of slots.

The implementation follows the [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909v1) fairly closely, by building the intent inference model and the slot filling model as two separate heads to the underlying BERT\* model, which is quite obvious if we look at the code.

```python
def __init__(...)
		...
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", output_hidden_states=True)
		...
	...

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
```

\*I use a smaller version of BERT called [DistilBert](https://huggingface.co/transformers/model_doc/distilbert.html)

The input to the module is a text sample such as `whats the weather like in tokyo`, which is then [tokenized](https://huggingface.co/transformers/tokenizer_summary.html) and the output is a pair containing:

- a N dimensional vector of unnormalized log probabilities, where N is the number of intents the network has been trained on
- a W x M dimensional vector of unnormalized log probabilities per word, where W is the number of words in the input + 2 (bert adds a couple of extra tokens) and M is the number of slots the network has been trained on

Those are then used in the `JointIntentAndSlotsModel.inferIntentAndSlots()` method to produce:

- a confidence value representing how certain the network is that it's guessed the correct intent
- the intent name
- a dictionary containing all recognized slots

For an overview of how intent and slot inference fits in the full application have a look at [the main repo](https://github.com/vshotarov/Kronos#overview).

## Training
To train, run `train.py` with the relevant arguments.

```
usage: train.py [-h] [-ps PATH_TO_SAVE_MODEL] [-ne NUM_EPOCHS] [-ve VALIDATE_EVERY]
                intents_and_slots_file train_dataset test_dataset

Kronos virtual assistant - Joint intent and slot models trainer

The dataset .csv files need to have the following columns:
    text_with_slots,intent

The text_with_slots column is the sentence with the slots present

E.g. whats the time in <location>brazil</location>

There's an argument for providing a file containing all the intents
and slots. The formatting for that file is to have all the intents
separated with commas on the first line and on the second have
all the slots separated with commas.

E.g.

weather.current,weather.future,other
time,location

positional arguments:
  intents_and_slots_file
                        path to the file containing all the intents and slots
  train_dataset         path to the dataset .csv file for training
  test_dataset          path to the dataset .csv file for testing after training

optional arguments:
  -h, --help            show this help message and exit
  -ps PATH_TO_SAVE_MODEL, --path_to_save_model PATH_TO_SAVE_MODEL
                        path to save the trained model at. By default it's a file called
                        saved_model.torch in the current directory.
  -ne NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        how many epochs of training to run. By default it's 100.
  -ve VALIDATE_EVERY, --validate_every VALIDATE_EVERY
                        how often to validate in epochs. By default it's every 10.
```

Here's what my `intents_and_slots_file` looks like:

```
time.current,time.timer,weather.current,weather.future,other
location,time
```

and here's what a dataset file is expected to look like:

```
what is the time in <location>austria</location>,time.current
whats the time in <location>iceland</location>,time.current
how hot will it be <time>today</time>,weather.future
```

As you can see, we use HTML style notation for denoting the slots. To convert those to a representation appropriate for training for slots, we use an HTML parser as a preprocessor, which parses a sentence into a list of slot tokens. Here's an example:

The sentence `what is the time in <location>austria</location>` becomes `O O O O O B-location`, where the `O`s are the token for not having a slot and `B-location` denotes this word is the beginning of a location slot. Here's a list of all the tokens that would correspond to the above mentioned `intents_and_slots_file`:

- `O` - no slot
- `PAD` - during training we might have to pad some samples in order to fit in the batch and this token corresponds to padding
- `B-location` - the beginning of a location slot
- `I-location` - inside a location slot
- `B-time` - the beginning of a time slot
- `I-time` - inside a time slot

The reason we have a `B-` and `I-` versions of the slots is to be able to capture both multi word slots, but also multiple slots. E.g.

```
how far is    san       marino   from  brussels
 O   O  O  B-location I-location  O   B-location
```

