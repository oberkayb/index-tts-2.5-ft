#!/usr/bin/env python3
"""
Step 5: synthesise with the finetuned weights, including a cross-lingual check.

Two things this wrapper handles that a bare ``infer_v2_5`` call does not:

  * **Frontend parity.**  Upstream only knows how to normalise zh/en/ja/es/ar,
    so we normalise with the *same* :class:`TextFrontend` used at preprocessing
    time and then pass ``text_normalization=False``.  Train/infer mismatch here
    is the most common cause of "it trained fine but sounds wrong".
  * **Aliased languages.**  If you trained on a borrowed vocabulary slot, the
    ``<|xx|>`` token and the language row have to match what training used.

``--cross-lingual-check`` synthesises the same reference voice in the new
language *and* in the languages the base model shipped with, which is the
quickest way to hear whether the finetune damaged anything.

Example
-------
    python scripts/synthesize.py \
        --gpt-checkpoint runs/tr_lora/exported/gpt.pth \
        --lang tr --normalizer turkish --case tr_lower \
        --prompt-audio ref.wav --text "Merhaba, bugün hava çok güzel." \
        --output out/tr_test.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from itts25ft import env  # noqa: E402

#: Short probes in the languages the 2.5 release ships with, used by
#: --cross-lingual-check to confirm nothing regressed.
BASE_LANG_PROBES = {
    "en": "This is a quick check that English still sounds the way it used to.",
    "zh": "这是一个简单的检查，用来确认中文没有退化。",
    "ja": "これは日本語が劣化していないかを確認する短いテストです。",
    "es": "Esta es una prueba rápida para comprobar que el español sigue igual.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesise with a finetuned IndexTTS-2.5 GPT.")
    p.add_argument("--repo", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--gpt-checkpoint", type=Path, default=None,
                   help="Finetuned gpt.pth; omit to use the stock weights.")
    p.add_argument("--prompt-audio", type=Path, required=True, help="Reference voice.")
    p.add_argument("--emo-audio", type=Path, default=None, help="Separate emotion reference.")
    p.add_argument("--emo-alpha", type=float, default=1.0)
    p.add_argument("--text", default=None)
    p.add_argument("--text-file", type=Path, default=None, help="One utterance per line.")
    p.add_argument("--output", type=Path, default=Path("out/sample.wav"))
    p.add_argument("--lang", required=True)
    p.add_argument("--lang-alias", default=None)
    p.add_argument("--normalizer", default="none")
    p.add_argument("--case", default="lower")
    p.add_argument("--cross-lingual-check", action="store_true",
                   help="Also render the base languages with the same reference voice.")
    p.add_argument("--device", default=None)
    p.add_argument("--fp32", action="store_true", help="Disable bf16.")
    p.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--num-beams", type=int, default=3)
    p.add_argument("--repetition-penalty", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.text and not args.text_file and not args.cross_lingual_check:
        raise SystemExit("Provide --text, --text-file or --cross-lingual-check.")

    repo = env.bootstrap(args.repo)
    model_dir = env.find_model_dir(args.model_dir, repo)

    import torch
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.checkpoint import load_checkpoint

    from itts25ft.lang import resolve
    from itts25ft.textfront import TextFrontend, TextFrontendConfig

    if args.seed is not None:
        from itts25ft.utils import set_seed

        set_seed(args.seed)

    slot = resolve(args.lang, args.lang_alias)
    print(">> language: " + slot.describe())

    tts = IndexTTS2(
        cfg_path=str(Path(model_dir) / "config.yaml"),
        model_dir=str(model_dir),
        use_bf16=not args.fp32,
        device=args.device,
        # The acceleration engine snapshots GPT weights at construction time, so
        # swapping in a finetuned checkpoint afterwards would be ignored.
        use_accel=False,
        use_qwen_emo=False,
    )

    if args.gpt_checkpoint is not None:
        if not args.gpt_checkpoint.is_file():
            raise SystemExit("Checkpoint not found: " + str(args.gpt_checkpoint))
        print(">> loading finetuned GPT: " + str(args.gpt_checkpoint))
        load_checkpoint(tts.gpt, str(args.gpt_checkpoint))
        tts.gpt.eval()

    frontend = TextFrontend(
        model_dir, slot, TextFrontendConfig(normalizer=args.normalizer, case=args.case)
    )

    texts: List[str] = []
    if args.text:
        texts.append(args.text)
    if args.text_file and args.text_file.is_file():
        texts += [ln.strip() for ln in args.text_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    generation = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_beams": args.num_beams,
        "repetition_penalty": args.repetition_penalty,
    }
    output_dir = args.output.parent if args.output.suffix else args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    def render(text: str, lang_token: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tts.infer(
            spk_audio_prompt=str(args.prompt_audio),
            text=text,
            output_path=str(out_path),
            lang=lang_token,
            emo_audio_prompt=str(args.emo_audio) if args.emo_audio else None,
            emo_alpha=args.emo_alpha,
            verbose=args.verbose,
            max_text_tokens_per_segment=args.max_text_tokens_per_segment,
            # Normalisation already happened in our frontend; letting upstream
            # run its zh/en pipeline again would rewrite the text we trained on.
            text_normalization=False,
            **generation,
        )
        print("  wrote " + str(out_path))

    for index, raw in enumerate(texts):
        cleaned = frontend.clean(raw)
        if args.verbose:
            print("  text[" + str(index) + "] -> " + cleaned)
        target = args.output if len(texts) == 1 and args.output.suffix else (
            output_dir / (slot.code + "_" + str(index).zfill(3) + ".wav")
        )
        render(cleaned, slot.slot, target)

    if args.cross_lingual_check:
        print("")
        print(">> cross-lingual check with the same reference voice")
        check_dir = output_dir / "cross_lingual_check"
        for code, probe in BASE_LANG_PROBES.items():
            # Base languages keep upstream's own normalisation.
            check_path = check_dir / (code + ".wav")
            check_path.parent.mkdir(parents=True, exist_ok=True)
            tts.infer(
                spk_audio_prompt=str(args.prompt_audio),
                text=probe,
                output_path=str(check_path),
                lang=code,
                verbose=args.verbose,
                max_text_tokens_per_segment=args.max_text_tokens_per_segment,
                text_normalization=True,
                **generation,
            )
            print("  " + code + " -> " + str(check_path))
        print("")
        print("Listen to these against the stock model. Degradation here means the "
              "finetune leaked into the base languages: lower the LR, add replay "
              "manifests, or switch to --trainable-mode lora with a smaller rank.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
