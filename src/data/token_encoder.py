import numpy as np


class TokenEncoder:
    def __init__(self, tokens: np.ndarray, map_token_to_id: dict[str, int], missing_token_id: int):
        self.alphabet: np.ndarray = tokens
        self.vectorized_map_token_to_id = np.vectorize(lambda x: map_token_to_id.get(x, missing_token_id))

    def encode(self, tokens: np.array) -> np.ndarray:
        """
        Encodes a sequence of tokens into their corresponding IDs.

        :param tokens: A numpy array of tokens to encode.
        :return: A numpy array of encoded token IDs.
        """
        return self.vectorized_map_token_to_id(tokens)

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Decodes a sequence of token IDs back into their corresponding tokens.

        :param token_ids: A numpy array of token IDs to decode.
        :return: A numpy array of decoded tokens.
        """
        return self.alphabet[token_ids]