"""
HETU_Online's think-content masking patch.

Auto-loaded IF this directory is on PYTHONPATH before llamafactory-cli
starts (Python imports `sitecustomize` at interpreter startup). This is
what makes CotCond training possible: LLaMA-Factory's stock
SupervisedDatasetProcessor only masks at whole-TURN granularity
(mask_history), it cannot mask a sub-span *inside* one turn. CotCond
needs exactly that: the model must emit its own "<think>...</think>{answer}"
turn, but should get zero gradient on the reasoning CONTENT between the
tags -- only the tag tokens and the final answer are supervised.

Ported from model-forge's Math_Reasoning cotcond patch (the more robust
of the two implementations that exist in model-forge -- passage_reranking's
think_mask_processor.py does the same thing via raw token-id search, which
only works when "<think>"/"</think>" are guaranteed single special tokens
for the tokenizer in use, e.g. Qwen3. That assumption breaks silently on
tokenizers where the tags are plain BPE text, e.g. GLM-Z1 -- confirmed:
standalone "</think>" encodes differently than "</think>" embedded in
surrounding text, because BPE merges are context-dependent. Decoding is
immune to this since it reconstructs the same text regardless of where
the tokenizer drew subword boundaries, so this version matches on DECODED
TEXT and maps character offsets back to token indices -- correct for any
tokenizer, not just ones where the tags happen to be atomic tokens.

Enable with:  HETU_THINK_CONTENT_MASK=1

Search is scoped to the response span only (labels != IGNORE_INDEX before
this patch runs), never the full prompt+response sequence -- if your
prompt text itself contains the literal string "<think>" (e.g. an
instruction telling the model to use think tags), an unscoped search
would match that mention first and mask nothing real.

If either tag is missing from an example (e.g. truncated by cutoff_len,
or a malformed example with no think block), this patch FAILS OPEN: it
leaves labels untouched, i.e. that example trains full-loss instead of
being masked. It never crashes and never silently zeroes an entire
example's loss.
"""
import os

_MASK_MARKER = os.environ.get("HETU_THINK_CONTENT_MASK")
if _MASK_MARKER == "1":
    try:
        from llamafactory.data.processor import supervised as _sup
        from llamafactory.extras.constants import IGNORE_INDEX as _IGNORE_INDEX

        _orig_encode_data_example = _sup.SupervisedDatasetProcessor._encode_data_example

        _THINK_OPEN = os.environ.get("HETU_THINK_OPEN_TAG", "<think>")
        _THINK_CLOSE = os.environ.get("HETU_THINK_CLOSE_TAG", "</think>")

        def _char_pos_to_token_idx(tokenizer, span_ids, char_target):
            """Smallest i such that decoding span_ids[:i] covers char_target
            decoded characters. O(n) incremental decode -- fine at
            per-example dataset-build time; not called during training."""
            for i in range(1, len(span_ids) + 1):
                if len(tokenizer.decode(span_ids[:i])) >= char_target:
                    return i
            return len(span_ids)

        def _think_content_mask_encode(self, prompt, response, system, tools, images, videos, audios):
            input_ids, labels = _orig_encode_data_example(
                self, prompt, response, system, tools, images, videos, audios
            )
            response_start = next((i for i, lab in enumerate(labels) if lab != _IGNORE_INDEX), None)
            if response_start is None:
                return input_ids, labels

            span_ids = input_ids[response_start:]
            full_decoded = self.tokenizer.decode(span_ids)
            open_char = full_decoded.find(_THINK_OPEN)
            close_char = full_decoded.rfind(_THINK_CLOSE)

            # Fail open to "fully supervised" -- never crash on an
            # unanticipated shape (truncated tag, no think block, etc).
            if open_char == -1 or close_char == -1 or close_char <= open_char:
                return input_ids, labels

            content_start = _char_pos_to_token_idx(self.tokenizer, span_ids, open_char + len(_THINK_OPEN))
            content_end = _char_pos_to_token_idx(self.tokenizer, span_ids, close_char)
            if content_end <= content_start:
                return input_ids, labels

            labels = list(labels)
            for i in range(response_start + content_start, response_start + content_end):
                labels[i] = _IGNORE_INDEX
            return input_ids, labels

        _sup.SupervisedDatasetProcessor._encode_data_example = _think_content_mask_encode
        print(f"[HETU_Online] think-content masking ENABLED "
              f"(tags={_THINK_OPEN!r}/{_THINK_CLOSE!r})")
    except Exception as _e:
        print(f"[HETU_Online] WARNING: failed to install think-content mask patch: {_e}")
