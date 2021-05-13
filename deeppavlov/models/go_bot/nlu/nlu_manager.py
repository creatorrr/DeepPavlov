from logging import getLogger
from typing import List, Union, Dict

from deeppavlov import Chainer
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.models.bert.bert_classifier import BertClassifierModel
from deeppavlov.models.go_bot.nlu.dto.nlu_response import NLUResponse
from deeppavlov.models.go_bot.nlu.nlu_manager_interface import NLUManagerInterface

log = getLogger(__name__)


# todo add the ability to configure nlu loglevel in config (now the setting is shared across all the GO-bot)
# todo add each method input-output logging when proper loglevel level specified


class NLUManager(NLUManagerInterface):
    """
    NLUManager is a unit of the go-bot pipeline that handles the understanding of text.
    Given the text it provides tokenization, intents extraction and the slots extraction.
    (the whole go-bot pipeline is as follows: NLU, dialogue-state-tracking&policy-NN, NLG)
    """

    def __init__(self, tokenizer, slot_filler, intent_classifier, debug=False):
        self.debug = debug
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"tokenizer={tokenizer}, slot_filler={slot_filler}, "
                      f"intent_classifier={intent_classifier}, debug={debug}")
        # todo type hints
        self.tokenizer = tokenizer
        self.slot_filler = slot_filler
        self.intent_classifier = intent_classifier
        self.intents = []
        if isinstance(self.intent_classifier, Chainer):
            component = self.intent_classifier.get_main_component()
            if isinstance(component, BertClassifierModel):
                intent2labeltools = [el[-1] for el in self.intent_classifier.pipe if isinstance(el[-1], SimpleVocabulary)]
                if intent2labeltools:
                    self.intents = intent2labeltools[-1]._i2t
            else:
                self.intents = component.classes

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"tokenizer={tokenizer}, slot_filler={slot_filler}, "
                      f"intent_classifier={intent_classifier}, debug={debug}")

    def nlu(self, text: Union[str, Dict]) -> NLUResponse:
        """
        Extracts slot values and intents from text.

        Args:
            text: text to extract knowledge from

        Returns:
            an object storing the extracted slos and intents info
        """
        # todo meaningful type hints

        if isinstance(text, dict):
            if "text" in text:
                text_ = text["text"]
            else:
                log.debug("INSIDE nlu() key <text> is missing in the passed dict")
                text_ = ''
            tokens = self._tokenize_single_text_entry(text_)
        else:
            tokens = self._tokenize_single_text_entry(text)

        slots = None
        if isinstance(text, dict):
            if "slots" in text:
                slots = text["slots"]
            else:
                log.debug("INSIDE nlu() key <slots> is missing in the passed dict")
        else:
            if callable(self.slot_filler):
                slots = self._extract_slots_from_tokenized_text_entry(tokens)

        intents = []
        if isinstance(text, dict):
            if "intents" in text:
                intents = text["intents"]
            else:
                log.debug("INSIDE nlu() key <intents> is missing in the passed dict")
        else:
            if callable(self.intent_classifier):
                intents = self._extract_intents_from_tokenized_text_entry(tokens)

        return NLUResponse(slots, intents, tokens)

    def _extract_intents_from_tokenized_text_entry(self, tokens: List[str]):
        # todo meaningful type hints, relies on unannotated intent classifier
        classifier_output = self.intent_classifier([' '.join(tokens)])
        intent_features = classifier_output[1][0]
        return intent_features

    def _extract_slots_from_tokenized_text_entry(self, tokens: List[str]):
        # todo meaningful type hints, relies on unannotated slot filler
        return self.slot_filler([tokens])[0]

    def _tokenize_single_text_entry(self, text: str):
        # todo meaningful type hints, relies on unannotated tokenizer
        return self.tokenizer([text.lower().strip()])[0]

    def num_of_known_intents(self) -> int:
        """
        Returns:
            the number of intents known to the NLU module
        """
        return len(self.intents)
