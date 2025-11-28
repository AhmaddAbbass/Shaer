import numpy as np
import sentencepiece as spm
from typing import Iterable, List


class SentencePieceTokenizer:
    """
    Minimal replacement for tkseem.SentencePieceTokenizer
    backed by the sentencepiece library.
    """

    def __init__(self) -> None:
        self._sp = spm.SentencePieceProcessor()

    def load_model(self, model_path: str) -> None:
        self._sp.load(model_path)

    def encode_sentences(self, sentences: Iterable[str], out_length: int | None = None):
        ids: List[List[int]] = [
            self._sp.encode(sentence, out_type=int) for sentence in sentences
        ]

        if out_length is not None:
            padded_ids: List[List[int]] = []
            for seq in ids:
                if len(seq) < out_length:
                    seq = seq + [0] * (out_length - len(seq))
                else:
                    seq = seq[:out_length]
                padded_ids.append(seq)
            ids = padded_ids

        return np.asarray(ids, dtype="int32")

