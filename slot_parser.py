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

