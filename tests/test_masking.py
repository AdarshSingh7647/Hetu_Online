"""Tests for sitecustomize.py's think-content masking patch -- the one
piece of this tool that fails SILENTLY when wrong (mismatched tags mean
CotCond trains full-loss with no error, see that module's docstring).

sitecustomize.py only installs its patch when imported with
HETU_THINK_CONTENT_MASK=1 set AND `llamafactory` importable, and it patches
a real llamafactory class in place. To test the masking logic in isolation
(without requiring the real, heavy llamafactory package or a real
tokenizer), this stubs out just enough of llamafactory's module surface
for sitecustomize's import machinery, using a small fake tokenizer that
mimics BPE's key property: encode/decode is not simply "one token per
character", so char-offset-to-token-index mapping is genuinely exercised.
"""

import importlib
import os
import sys
import types
import unittest


IGNORE_INDEX = -100


class FakeTokenizer:
    """A minimal fake tokenizer where each token is a short string chunk,
    not a single character -- so char-offset math actually has to walk
    token boundaries rather than trivially aligning 1:1."""

    def __init__(self, chunks):
        # chunks: list of (token_id, chunk_text) pairs, in order.
        self._id_to_text = {i: text for i, text in enumerate(chunks)}
        self._text_to_id = {text: i for i, text in enumerate(chunks)}

    def encode_text(self, text):
        """Greedy longest-match tokenization for building test fixtures."""
        ids = []
        i = 0
        chunks_by_len = sorted(self._text_to_id.keys(), key=len, reverse=True)
        while i < len(text):
            for chunk in chunks_by_len:
                if text.startswith(chunk, i):
                    ids.append(self._text_to_id[chunk])
                    i += len(chunk)
                    break
            else:
                raise ValueError(f"no chunk matches at position {i}: {text[i:i+10]!r}")
        return ids

    def decode(self, ids):
        return "".join(self._id_to_text[i] for i in ids)


def _install_fake_llamafactory(monkeypatch_modules):
    """Registers minimal fake llamafactory.data.processor.supervised and
    llamafactory.extras.constants modules in sys.modules so
    sitecustomize.py's `from llamafactory... import ...` succeeds."""
    llamafactory = types.ModuleType("llamafactory")
    data_pkg = types.ModuleType("llamafactory.data")
    processor_pkg = types.ModuleType("llamafactory.data.processor")
    supervised_mod = types.ModuleType("llamafactory.data.processor.supervised")
    extras_pkg = types.ModuleType("llamafactory.extras")
    constants_mod = types.ModuleType("llamafactory.extras.constants")

    def _orig_encode_data_example(self, prompt, response, system, tools, images, videos, audios):
        return self._fixture_input_ids, self._fixture_labels

    class SupervisedDatasetProcessor:
        _encode_data_example = _orig_encode_data_example

    supervised_mod.SupervisedDatasetProcessor = SupervisedDatasetProcessor
    constants_mod.IGNORE_INDEX = IGNORE_INDEX

    modules = {
        "llamafactory": llamafactory,
        "llamafactory.data": data_pkg,
        "llamafactory.data.processor": processor_pkg,
        "llamafactory.data.processor.supervised": supervised_mod,
        "llamafactory.extras": extras_pkg,
        "llamafactory.extras.constants": constants_mod,
    }
    for name, mod in modules.items():
        monkeypatch_modules[name] = sys.modules.get(name)
        sys.modules[name] = mod
    return SupervisedDatasetProcessor


class TestThinkContentMasking(unittest.TestCase):
    def setUp(self):
        self._env_backup = dict(os.environ)
        self._modules_backup = {}
        self.SupervisedDatasetProcessor = _install_fake_llamafactory(self._modules_backup)
        sys.modules.pop("hetu_online.sitecustomize", None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)
        for name, original in self._modules_backup.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        sys.modules.pop("hetu_online.sitecustomize", None)

    def _load_patched_processor(self, think_open, think_close):
        os.environ["HETU_THINK_CONTENT_MASK"] = "1"
        os.environ["HETU_THINK_OPEN_TAG"] = think_open
        os.environ["HETU_THINK_CLOSE_TAG"] = think_close
        importlib.import_module("hetu_online.sitecustomize")
        return self.SupervisedDatasetProcessor

    def _run_patch(self, tokenizer, prompt_ids, response_text, think_open, think_close):
        processor_cls = self._load_patched_processor(think_open, think_close)
        response_ids = tokenizer.encode_text(response_text)
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

        instance = processor_cls.__new__(processor_cls)
        instance.tokenizer = tokenizer
        instance._fixture_input_ids = input_ids
        instance._fixture_labels = labels
        out_ids, out_labels = processor_cls._encode_data_example(
            instance, prompt=None, response=None, system=None, tools=None,
            images=None, videos=None, audios=None,
        )
        return out_ids, out_labels, response_ids

    def test_qwen3_tags_mask_only_reasoning_content(self):
        prompt_ids = [0]  # arbitrary single prompt token, masked in labels
        think_open = "<think>\n"
        think_close = "\n</think>\n\n"
        reasoning = "reasoning"
        answer = "42"
        chunks = ["<think>", "\n", "reasoning", "\n</think>\n\n", "4", "2"]
        tokenizer = FakeTokenizer(chunks)
        response_text = think_open + reasoning + think_close + answer

        out_ids, out_labels, response_ids = self._run_patch(
            tokenizer, prompt_ids, response_text, think_open, think_close
        )

        decoded_per_token = [tokenizer.decode([i]) for i in response_ids]
        response_start = len(prompt_ids)
        masked_texts = [
            decoded_per_token[i - response_start]
            for i in range(response_start, len(out_labels))
            if out_labels[i] == IGNORE_INDEX
        ]
        supervised_texts = [
            decoded_per_token[i - response_start]
            for i in range(response_start, len(out_labels))
            if out_labels[i] != IGNORE_INDEX
        ]
        self.assertEqual(masked_texts, ["reasoning"])
        # "<think>\n" tokenizes as two chunks with this fixture's token
        # vocabulary ("<think>" + "\n") -- both must stay supervised, since
        # the whole open tag is outside the masked span.
        self.assertEqual(supervised_texts, ["<think>", "\n", "\n</think>\n\n", "4", "2"])
        # Prompt labels must be untouched (still IGNORE_INDEX).
        self.assertEqual(out_labels[:response_start], [IGNORE_INDEX] * response_start)

    def test_gemma4_tags_mask_only_reasoning_content(self):
        think_open = "<|channel>thought\n"
        think_close = "<channel|>"
        reasoning = "reasoning"
        answer = "42"
        chunks = ["<|channel>thought\n", "reasoning", "<channel|>", "4", "2"]
        tokenizer = FakeTokenizer(chunks)
        response_text = think_open + reasoning + think_close + answer
        prompt_ids = [99]

        out_ids, out_labels, response_ids = self._run_patch(
            tokenizer, prompt_ids, response_text, think_open, think_close
        )
        decoded_per_token = [tokenizer.decode([i]) for i in response_ids]
        response_start = len(prompt_ids)
        masked_texts = [
            decoded_per_token[i - response_start]
            for i in range(response_start, len(out_labels))
            if out_labels[i] == IGNORE_INDEX
        ]
        supervised_texts = [
            decoded_per_token[i - response_start]
            for i in range(response_start, len(out_labels))
            if out_labels[i] != IGNORE_INDEX
        ]
        self.assertEqual(masked_texts, ["reasoning"])
        self.assertEqual(supervised_texts, ["<|channel>thought\n", "<channel|>", "4", "2"])

    def test_missing_tags_fails_open_to_full_supervision(self):
        """If tags aren't found (e.g. wrong model_family's tags passed),
        the patch must NOT crash and must NOT mask anything -- it fails
        open to full supervision, per sitecustomize.py's documented
        contract."""
        chunks = ["no", "think", "tags", "here", "4", "2"]
        tokenizer = FakeTokenizer(chunks)
        prompt_ids = [7]
        response_text = "no" + "think" + "tags" + "here" + "4" + "2"

        # Ask for gemma4 tags against qwen3-shaped text with no matching
        # tags anywhere.
        out_ids, out_labels, response_ids = self._run_patch(
            tokenizer, prompt_ids, response_text, "<|channel>thought\n", "<channel|>"
        )
        response_start = len(prompt_ids)
        # Every response token should remain fully supervised.
        self.assertTrue(all(lab != IGNORE_INDEX for lab in out_labels[response_start:]))


if __name__ == "__main__":
    unittest.main()
