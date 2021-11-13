from model import JointIntentAndSlotsModel
from data import Dataset, pad
import torch
import sys
import argparse
import textwrap


def train(intents_and_slots_file, train_dataset, test_dataset,
        path_to_save_model, num_epochs, validate_every):
    with open(intents_and_slots_file, "r", newline="") as f:
        intents, recognized_slots = [x.strip("\n").split(",") for x in f.readlines()]

    joint_model = JointIntentAndSlotsModel(intents, recognized_slots)
    joint_model.train()
    slots = joint_model.slots

    train_loader = torch.utils.data.DataLoader(Dataset(
        train_dataset, joint_model.tokenizer, intents, slots, recognized_slots),
        batch_size=8, collate_fn=pad)

    test_loader = torch.utils.data.DataLoader(
        Dataset(test_dataset, joint_model.tokenizer, intents, slots, recognized_slots),
        batch_size=4, collate_fn=pad)

    optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-4)

    intent_criterion = torch.nn.CrossEntropyLoss()
    slot_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    best_test_loss = float("inf")

    test_model(joint_model, test_loader, intents, slots)

    for i in range(num_epochs):
        epoch_loss = 0
        for j, (input_ids, attention_mask, _intents, _slots, _) in enumerate(train_loader):
            intent_outputs, slot_outputs = joint_model(input_ids, attention_mask)

            intent_loss = intent_criterion(intent_outputs, _intents)
            slot_loss = slot_criterion(
                slot_outputs.view(-1, joint_model.num_slots), _slots.view(-1))

            total_loss = intent_loss + slot_loss
            total_loss.backward()

            optimizer.step()

            epoch_loss += total_loss.item()

            print("Train epoch %i: Batch %i : loss %f\r" % (i, j, epoch_loss / (j+1)), end="")
        print()

        if i % validate_every == 0:
            joint_model.eval()

            with torch.no_grad():
                test_loss = test_model(joint_model, test_loader, intents, slots)

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(joint_model.state_dict(), path_to_save_model)
                    print("Saved.")

            joint_model.train()

    joint_model.eval()

    test_model(joint_model, test_loader, intents, slots)

def test_model(joint_intents_and_slots_model, test_loader, intents, slots):
    intent_criterion = torch.nn.CrossEntropyLoss()
    slot_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    epoch_loss = 0
    with torch.no_grad():
        for j, (input_ids, attention_mask, _intents, _slots, prompts) in enumerate(test_loader):
            intent_outputs, slot_outputs = joint_intents_and_slots_model(
                input_ids, attention_mask)

            intent_loss = intent_criterion(intent_outputs, _intents)
            slot_loss = slot_criterion(slot_outputs.view(
                -1, joint_intents_and_slots_model.num_slots), _slots.view(-1))

            total_loss = intent_loss + slot_loss

            epoch_loss += total_loss.item()

            print("###### ")
            print("Test: Batch %i : loss %f" % (j, total_loss))

            for i in range(_intents.shape[0]):
                this_intent_outputs = intent_outputs[i]
                this_slot_outputs = slot_outputs[i][1:len(prompts[i].split())+1]
                this_intent = torch.argmax(torch.nn.functional.softmax(this_intent_outputs, dim=0))
                this_slots = torch.argmax(torch.nn.functional.softmax(this_slot_outputs, dim=1), dim=1)
                print(prompts[i], "; intent: ", intents[this_intent.item()],
                        "; slots: ", " ".join([slots[x] for x in this_slots.detach()]))
        print("Total test loss: %f" % (epoch_loss / (j+1)))

    return epoch_loss


class RawHelpFormatter(argparse.HelpFormatter):
    # https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text
    def _fill_text(self, text, width, indent):
                return "\n".join([textwrap.fill(line, width) for line in\
                    textwrap.indent(textwrap.dedent(text), indent).splitlines()])

if __name__ == "__main__":
    help_text = """
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

"""

    parser = argparse.ArgumentParser(
        description=help_text, formatter_class=RawHelpFormatter)
    parser.add_argument("intents_and_slots_file",
        help="path to the file containing all the intents and slots")
    parser.add_argument("train_dataset",
        help="path to the dataset .csv file for training")
    parser.add_argument("test_dataset",
        help="path to the dataset .csv file for testing after training")
    parser.add_argument("-ps", "--path_to_save_model", default="saved_model.torch",
        help="path to save the trained model at. By default it's a file called "
             "saved_model.torch in the current directory.")
    parser.add_argument("-ne", "--num_epochs", default=100, type=int,
        help="how many epochs of training to run. By default it's 100.")
    parser.add_argument("-ve", "--validate_every", default=10, type=int,
        help="how often to validate in epochs. By default it's every 10.")

    parsed_args = parser.parse_args()

    train(intents_and_slots_file=parsed_args.intents_and_slots_file,
        train_dataset=parsed_args.train_dataset,
        test_dataset=parsed_args.test_dataset,
        path_to_save_model=parsed_args.path_to_save_model,
        num_epochs=parsed_args.num_epochs,
        validate_every=parsed_args.validate_every)

