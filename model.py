from typing import Any, Dict

from dpu_utils.mlutils import Vocabulary


class SyntacticModel():
    def __init__(self, hyperparameters: Dict[str, Any], vocab: Vocabulary) -> None:
        self.hyperparameters = hyperparameters
        self.vocab = vocab

    def get_default_hyperparameters(self):
        return None
