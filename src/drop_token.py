from allennlp.data.tokenizers import Token as OToken


class Token(OToken):
    def __init__(self, text: str = None,
        idx: int = None,
        lemma_: str = None,
        pos_: str = None,
        tag_: str = None,
        dep_: str = None,
        ent_type_: str = None,
        text_id: int = None,
        edx: int = None
    ):
        super(Token, self).__init__(text =text, idx=idx)
        self.edx = edx