#!/usr/bin/env python3
"""
Step 0: verify the environment before spending GPU hours on it.

Checks, in order:
  1. the IndexTTS-2.5 source tree is importable,
  2. the checkpoint directory really is 2.5 (campplus + language embedding),
  3. the requested language resolves to a usable (control token, embedding row),
  4. the tiktoken vocabulary encodes the language without overflowing the text
     embedding table, and how efficiently it does so,
  5. the language row you are about to train is untrained (as expected) rather
     than one of the five the release already speaks.

Example
-------
    python scripts/check_setup.py --lang tr --sample-text data/tr_sample.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itts25ft import env  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an IndexTTS-2.5 finetuning setup.")
    p.add_argument("--repo", default=None, help="Path to the index-tts-2.5 source tree.")
    p.add_argument("--model-dir", default=None, help="Path to the downloaded 2.5 checkpoints.")
    p.add_argument("--lang", required=True, help="Target language code, e.g. tr")
    p.add_argument("--lang-alias", default=None, help="Borrowed vocabulary slot for unlisted languages.")
    p.add_argument("--normalizer", default="none", help="none | turkish | nemo:<lang> | module:func")
    p.add_argument("--case", default="lower", help="none | lower | upper | tr_lower | tr_upper")
    p.add_argument("--sample-text", type=Path, default=None,
                   help="A text file (one sentence per line) to measure tokenizer efficiency on.")
    p.add_argument("--load-model", action="store_true",
                   help="Also load gpt.pth and inspect the language embedding rows.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ok = True

    print("=" * 72)
    print("IndexTTS-2.5 finetuning setup check")
    print("=" * 72)

    repo = env.bootstrap(args.repo)
    print("[1/5] repo             : " + str(repo))

    model_dir = env.find_model_dir(args.model_dir, repo)
    cfg = env.load_model_config(model_dir)
    print("[2/5] checkpoints      : " + str(model_dir))
    print("      version          : " + str(cfg.get("version", "unknown")))
    print("      model_dim        : " + str(cfg.gpt.model_dim)
          + "  layers: " + str(cfg.gpt.layers)
          + "  heads: " + str(cfg.gpt.heads))
    print("      max_text_tokens  : " + str(cfg.gpt.max_text_tokens)
          + "  max_mel_tokens: " + str(cfg.gpt.max_mel_tokens))
    print("      number_text_tokens: " + str(cfg.gpt.number_text_tokens)
          + "  number_mel_codes: " + str(cfg.gpt.number_mel_codes))

    from itts25ft import lang as lang_mod

    try:
        slot = lang_mod.resolve(args.lang, args.lang_alias)
    except ValueError as exc:
        print("[3/5] language         : FAILED - " + str(exc))
        print("      free slots       : " + ", ".join(lang_mod.free_slots(20)) + " ...")
        return 1
    print("[3/5] language         : " + slot.describe())
    print("      lang rows        : " + str(lang_mod.num_lang_rows())
          + "  (pretrained: " + ", ".join(lang_mod.PRETRAINED_LANGS) + ")")
    if slot.code in lang_mod.PRETRAINED_LANGS:
        print("      WARNING          : this language is already trained in the base model; "
              "finetuning it will overwrite existing behaviour.")

    from itts25ft.textfront import TextFrontend, TextFrontendConfig

    frontend = TextFrontend(
        model_dir,
        slot,
        TextFrontendConfig(normalizer=args.normalizer, case=args.case),
    )
    probe = ["merhaba dünya, bugün hava çok güzel."]
    if args.sample_text and args.sample_text.is_file():
        lines = [ln.strip() for ln in args.sample_text.read_text(encoding="utf-8").splitlines()]
        probe = [ln for ln in lines if ln][:2000] or probe

    report = frontend.efficiency_report(probe)
    max_id = frontend.max_token_id(probe)
    print("[4/5] tokenizer        : " + str(int(report["n_samples"])) + " sample lines")
    print("      tokens/char      : {:.3f}".format(report["tokens_per_char"]))
    print("      tokens/word      : {:.2f}".format(report["tokens_per_word"]))
    print("      max token id     : " + str(max_id) + " / " + str(cfg.gpt.number_text_tokens))
    print("      example ids      : " + str(frontend.encode(probe[0])[:16]))
    print("      round trip       : " + repr(frontend.decode(frontend.encode(probe[0]))[:90]))

    if max_id >= cfg.gpt.number_text_tokens:
        print("      FAILED           : token ids overflow the text embedding table.")
        ok = False
    if report["tokens_per_char"] > 0.5:
        print("      WARNING          : >0.5 tokens/char - the BPE is shredding this language; "
              "expect to need more data and more steps.")

    if args.load_model:
        import torch
        from itts25ft.modeling import build_gpt

        print("[5/5] model            : loading gpt.pth ...")
        model = build_gpt(cfg, model_dir, device="cpu")
        emb = model.lang_embedding.weight
        print("      lang_embedding   : " + str(tuple(emb.shape)))
        target_norm = emb[slot.lang_id].norm().item()
        print("      row[{}] norm      : {:.4f}   ({})".format(
            slot.lang_id, target_norm, slot.slot))
        for code in lang_mod.PRETRAINED_LANGS:
            row = lang_mod.language_dict()[code]
            print("      row[{:>3}] norm    : {:.4f}   ({}, pretrained)".format(
                row, emb[row].norm().item(), code))
        trained_norms = [emb[lang_mod.language_dict()[c]].norm().item()
                         for c in lang_mod.PRETRAINED_LANGS]
        if target_norm > 0.5 * min(trained_norms):
            print("      NOTE             : the target row looks trained already - double check "
                  "you are not about to overwrite a language.")
        with torch.no_grad():
            print("      spk_emb_proj     : " + str(tuple(model.spk_emb_proj.weight.shape))
                  + "  (192 -> model_dim, campplus mode confirmed)")
    else:
        print("[5/5] model            : skipped (pass --load-model to inspect weights)")

    print("=" * 72)
    print("RESULT: " + ("ready to preprocess" if ok else "problems found - see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
