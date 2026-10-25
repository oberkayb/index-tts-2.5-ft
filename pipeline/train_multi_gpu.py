#!/usr/bin/env python3
"""
İki GPU'lu (veri paralel) eğitim: index-tts-2.5-ft trainer'ının çok GPU'lu sürümü.

    python train_multi_gpu.py --ft-repo ..\\index-tts-2.5-ft --gpus 0,1 --rank-accum 5,3 <train.py argümanları>

Tek süreç, GPU başına bir model kopyası ve bir iş parçacığı:
  * Her adımda GPU'lar global batch'in kendi dilimini aynı anda işler (forward + backward).
  * Gradyanlar PCIe üzerinden GPU 0'a toplanır; optimizer yalnızca GPU 0'da çalışır,
    güncellenen ağırlıklar diğer kopyaya yazılır (4070'te Adam durumu tutulmaz).
  * text_embedding'in yalnızca eğitilen (maskeli) satırları taşınır.

Neden torch.distributed değil: Windows'ta NCCL yok; bu PyTorch derlemesi libuv'suz
olduğu için gloo da çalışmıyor ("makeDeviceForHostname(): unsupported gloo device").
Aynı makinedeki iki GPU için tek süreç zaten daha az kopya ister.

Doğruluk: bütün kopyalar aynı global batch dizisinden kendi dilimini alır; micro-batch
kaybı 1/Σaccum ile ölçeklenir ve gradyanlar toplanır. Bir optimizer adımı, tek GPU'da
--grad-accumulation Σaccum ile atılan adımla aynıdır. Checkpoint biçimi tek GPU
trainer'ıyla aynıdır: export.py ve --resume değişmeden çalışır.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402


def parse_own_args(argv: List[str]):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ft-repo", required=True, help="index-tts-2.5-ft klasörü")
    p.add_argument("--gpus", default="0,1", help="kullanılacak GPU'lar (nvidia-smi sırası); ilki ana GPU")
    p.add_argument("--rank-accum", default=None,
                   help="her GPU'nun adım başına micro-batch sayısı, örn. 5,3. Toplamı efektif birikimdir; "
                        "verilmezse --grad-accumulation eşit bölünür.")
    p.add_argument("--no-final-export", action="store_true", help="bitişte gpt.pth yazma (testler için)")
    p.add_argument("--report-json", default=None, help="süre/bellek/kayıp özetini bu dosyaya yaz")
    p.add_argument("--dump-grads", action="store_true",
                   help="doğrulama: ilk adımın toplanmış gradyanlarını kaydet ve çık")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="doğrulama kaybı art arda bu kadar ölçüm iyileşmezse dur (0 = kapalı)")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                   help="'iyileşme' sayılması için en az bu kadar düşüş")
    return p.parse_known_args(argv)


def load_train_module(ft_repo: Path):
    spec = importlib.util.spec_from_file_location("itts25ft_train", ft_repo / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RankSlice(torch.utils.data.Sampler):
    """Aynı global batch dizisinden bir GPU'nun payını verir.

    Her optimizer adımı Σaccum ardışık batch tüketir; GPU r bunların
    [offset_r, offset_r + accum_r) aralığını işler.
    """

    def __init__(self, base, accums: List[int], rank: int) -> None:
        self.base = base
        self.accums = accums
        self.rank = rank
        self.group = sum(accums)
        self.offset = sum(accums[:rank])
        self.skip = 0                     # devam ederken bu epoch'ta zaten atılmış adım sayısı

    def __iter__(self):
        batches = list(iter(self.base))
        for step in range(self.skip, len(batches) // self.group):
            start = step * self.group + self.offset
            yield from batches[start:start + self.accums[self.rank]]

    def __len__(self) -> int:
        return (len(self.base) // self.group) * self.accums[self.rank]


def main() -> int:
    own, rest = parse_own_args(sys.argv[1:])
    gpus = [int(g) for g in own.gpus.split(",") if g.strip()]
    world = len(gpus)
    devices = [torch.device("cuda", g) for g in gpus]

    T = load_train_module(Path(own.ft_repo).resolve())
    args = T.parse_args(rest)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accums = [int(a) for a in own.rank_accum.split(",")] if own.rank_accum else None
    if accums is None:
        base, extra = divmod(args.grad_accumulation, world)
        accums = [base + (1 if r < extra else 0) for r in range(world)]
    if len(accums) != world or min(accums) < 1:
        raise SystemExit("--rank-accum GPU sayısıyla eşleşmeli ve her değer >= 1 olmalı")
    total_accum = sum(accums)
    amp_dtype = T.resolve_amp_dtype(args.amp)
    if amp_dtype == torch.float16:
        raise SystemExit("Çok GPU'lu eğitim bf16 ya da fp32 ister (fp16 GradScaler desteklenmiyor).")

    T.set_seed(args.seed)
    repo = T.env.bootstrap(args.repo)
    model_dir = T.env.find_model_dir(args.model_dir, repo)
    cfg = T.env.load_model_config(model_dir)

    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from itts25ft.data import DatasetConfig, LengthBucketBatchSampler, ManifestSpec, PairedDataset, collate
    from itts25ft.lang import PRETRAINED_LANGS, language_dict, resolve
    from itts25ft.losses import LossConfig, compute_losses, total_loss
    from itts25ft.modeling import (
        UNTRAINED_ROW_NORM, EmbeddingRowGradMask, LanguageEmbeddingGradMask, TrainableSpec,
        apply_trainable_spec, build_gpt, describe_trainable, export_inference_checkpoint,
        init_language_row, parameter_groups,
    )

    # -- data -------------------------------------------------------------- #
    ds_cfg = DatasetConfig(
        emo_source=args.emo_source, prompt_emo_prob=args.prompt_emo_prob,
        max_text_tokens=args.max_text_tokens or int(cfg.gpt.max_text_tokens),
        max_code_tokens=args.max_code_tokens or int(cfg.gpt.max_mel_tokens),
    )
    train_ds = PairedDataset([ManifestSpec.parse(s) for s in args.train_manifests], ds_cfg)
    base_sampler = LengthBucketBatchSampler(train_ds, args.batch_size, bucket_multiplier=args.bucket_multiplier,
                                            seed=args.seed)
    samplers = [RankSlice(base_sampler, accums, r) for r in range(world)]
    loaders = [DataLoader(train_ds, batch_sampler=s, collate_fn=collate, num_workers=args.num_workers,
                          pin_memory=True, persistent_workers=args.num_workers > 0) for s in samplers]
    val_loader = None
    if args.val_manifests:
        val_ds = PairedDataset([ManifestSpec.parse(s) for s in args.val_manifests], ds_cfg)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                                num_workers=max(0, args.num_workers // 2), pin_memory=True)

    def steps_in_epoch(epoch: int) -> int:
        base_sampler.set_epoch(epoch)
        return sum(1 for _ in iter(base_sampler)) // total_accum

    steps_per_epoch = max(1, len(base_sampler) // total_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(">> {} GPU {}, micro-batch {} x birikim {} = efektif batch {} | {} çift, ~{} adım/epoch, {} adım".format(
        world, gpus, args.batch_size, accums, args.batch_size * total_accum, len(train_ds), steps_per_epoch,
        total_steps))

    # -- model (tek GPU trainer'ı ile aynı kurulum) -------------------------- #
    model = build_gpt(cfg, model_dir, device="cpu", checkpoint=args.base_checkpoint)
    slot = resolve(args.lang, args.lang_alias) if args.lang else None
    if slot is not None and slot.slot not in PRETRAINED_LANGS:
        norm = model.lang_embedding.weight[slot.lang_id].float().norm().item()
        if norm >= UNTRAINED_ROW_NORM:
            print("!! UYARI: taban checkpoint'te <|{}|> satırı zaten eğitilmiş (norm {:.2f}); stok model değil.".format(
                slot.slot, norm))
    if slot is not None:
        source_id = language_dict()[args.lang_init_from.lower()] if args.lang_init_from else None
        print(">> dil satırı: " + init_language_row(model, slot.lang_id, source_id, args.lang_init_noise))
    text_rows: List[int] = []
    if args.train_text_rows:
        if slot is None:
            raise SystemExit("--train-text-rows için --lang gerekli")
        text_rows = T.prepare_text_rows(args, model, train_ds, slot, model_dir, output_dir)
    spec = TrainableSpec(
        mode=args.trainable_mode, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_last_n_layers=args.lora_last_n_layers,
        train_last_n_layers=args.train_last_n_layers, train_text_embedding=args.train_text_embedding,
        train_text_rows=args.train_text_rows, train_heads=args.train_heads, preserve_emotion=args.preserve_emotion,
    )
    info = apply_trainable_spec(model, spec)
    print(">> eğitim modu: {} ({} LoRA modülü)".format(info["mode"], info["lora_modules"]))
    print(describe_trainable(model, limit=6))

    # Kopyalar CPU'da birebir aynı (LoRA dahil) oluşturulup GPU'lara taşınır.
    replicas = [model] + [copy.deepcopy(model) for _ in range(world - 1)]
    for replica, device in zip(replicas, devices):
        replica.to(device)
        # Doğrulama modunda dropout kapalı: iki koşunun gradyanı birebir karşılaştırılabilsin.
        replica.eval() if own.dump_grads else replica.train()
    main_model = replicas[0]

    masks = []
    for replica in replicas:
        if slot is not None and not args.train_all_lang_rows:
            masks.append(LanguageEmbeddingGradMask(replica, [slot.lang_id]))
        if args.train_text_rows:
            masks.append(EmbeddingRowGradMask(replica.text_embedding.weight, text_rows, name="text_embedding"))

    names = [n for n, p in main_model.named_parameters() if p.requires_grad]
    params = [[dict(r.named_parameters())[n] for n in names] for r in replicas]
    rows_per_device = [torch.tensor(text_rows, dtype=torch.long, device=d) for d in devices] if text_rows else None
    sparse = [n == "text_embedding.weight" and rows_per_device is not None for n in names]
    moved = sum((len(text_rows) * p.shape[1]) if s else p.numel() for p, s in zip(params[0], sparse))
    if world > 1:
        print(">> adım başına GPU'lar arası taşıma: {:.1f} M değer gradyan + {:.1f} M değer ağırlık".format(
            moved / 1e6, moved / 1e6))

    groups = parameter_groups(main_model, args.learning_rate, args.lang_lr_multiplier, args.weight_decay,
                              text_lr_multiplier=args.text_lr_multiplier if args.train_text_rows else None)
    optimizer = AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scheduler = T.build_scheduler(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
    loss_cfg = LossConfig(text_weight=args.text_loss_weight, mel_weight=args.mel_loss_weight,
                          label_smoothing=args.label_smoothing, spk_noise_std=args.spk_noise_std,
                          emo_dropout=args.emo_dropout)
    ckpt_manager = T.CheckpointManager(output_dir / "checkpoints", keep=args.keep_checkpoints)
    # Önceki koşulardan kalan step*.pt'ler de rotasyona girsin; yoksa her devamda birikirler.
    ckpt_manager._history = sorted(
        (output_dir / "checkpoints").glob("step*.pt"),
        key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    (output_dir / "train_args.json").write_text(json.dumps(
        dict({k: str(v) for k, v in vars(args).items()}, multi_gpu={"gpus": gpus, "rank_accum": accums}),
        indent=2, ensure_ascii=False), encoding="utf-8")

    @torch.no_grad()
    def push_weights(full: bool = False) -> None:
        """Ana kopyanın eğitilen ağırlıklarını diğer kopyalara yaz."""
        for r in range(1, world):
            for i, (p0, pr) in enumerate(zip(params[0], params[r])):
                if sparse[i] and not full:
                    pr.index_copy_(0, rows_per_device[r], p0.index_select(0, rows_per_device[0]).to(pr.device))
                else:
                    pr.copy_(p0.to(pr.device))

    @torch.no_grad()
    def gather_grads() -> None:
        """Diğer kopyaların gradyanlarını ana kopyaya topla."""
        for i, p0 in enumerate(params[0]):
            if p0.grad is None:
                p0.grad = torch.zeros_like(p0)
            for r in range(1, world):
                pr = params[r][i]
                if pr.grad is None:
                    continue
                if sparse[i]:
                    p0.grad.index_add_(0, rows_per_device[0],
                                       pr.grad.index_select(0, rows_per_device[r]).to(p0.device))
                else:
                    p0.grad.add_(pr.grad.to(p0.device))
                pr.grad = None

    # -- resume ----------------------------------------------------------- #
    global_step = 0
    baselines: Dict[str, float] = {}
    early: Dict[str, object] = {"best": None, "best_step": 0, "bad": 0}
    ckpt_dir = output_dir / "checkpoints"
    resume_path = ckpt_manager.latest() if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume_path and Path(resume_path).is_file():
        state = torch.load(resume_path, map_location="cpu")
        if state.get("finished"):
            print(">> {} tamamlanmış bir koşuya ait (adım {}); devam edilecek bir şey yok.".format(
                resume_path, state.get("step")))
            return 0
        main_model.load_state_dict(state["trainable"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state.get("step", 0))
        baselines = dict(state.get("baselines", {}))
        early.update(state.get("early_stop", {}))
        push_weights(full=True)
        print(">> {} adımından devam ({})".format(global_step, resume_path))

    # Kaldığı yeri bul: epoch'ların adım sayısı tohuma göre sabit, yani global_step'ten
    # hangi epoch'un kaçıncı adımında olduğumuz kesin hesaplanır ve o adımların verisi atlanır.
    start_epoch, skip = 0, global_step
    while start_epoch < args.epochs:
        n = steps_in_epoch(start_epoch)
        if skip < n:
            break
        skip -= n
        start_epoch += 1
    if global_step:
        print(">> epoch {}/{} içinde {} adım zaten tamam; o adımların verisi atlanıyor".format(
            start_epoch + 1, args.epochs, skip))

    def checkpoint_state(epoch: int, finished: bool = False) -> Dict[str, object]:
        return {
            "trainable": T.trainable_state_dict(main_model), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": None, "step": global_step, "epoch": epoch,
            "baselines": baselines, "spec": vars(spec), "multi_gpu": {"gpus": gpus, "rank_accum": accums},
            "early_stop": dict(early), "finished": finished,
        }

    def save_best(epoch: int) -> Path:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        target = ckpt_dir / "best.pt"
        tmp = target.with_suffix(".pt.tmp")
        torch.save(checkpoint_state(epoch), tmp)
        os.replace(tmp, target)
        return target

    def run_share(r: int, iterator, count: int):
        """GPU r'nin bu adımdaki micro-batch'leri: forward + backward (iş parçacığında)."""
        device = devices[r]
        torch.cuda.set_device(device)
        started = time.time()
        stats = []
        for _ in range(count):
            batch = next(iterator)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                text_loss, mel_loss, metrics = compute_losses(replicas[r], batch, device, loss_cfg, training=True)
                loss = total_loss(text_loss, mel_loss, loss_cfg)
            (loss / total_accum).backward()
            stats.append((loss.item(), mel_loss.item(), text_loss.item(), metrics["mel_top1"],
                          batch["text_ids"].size(0)))
        torch.cuda.synchronize(device)
        return stats, time.time() - started

    # Ctrl+C (ya da Ctrl+Break): o anki adım bitince checkpoint kaydedip temiz çık.
    stop_request = {"flag": False}

    def request_stop(signum, frame) -> None:
        if stop_request["flag"]:
            raise KeyboardInterrupt
        stop_request["flag"] = True
        print("\n>> Durdurma isteği alındı: adım bitince checkpoint kaydedilip çıkılacak "
              "(beklemeden çıkmak için tekrar Ctrl+C; o zaman son kayıttan devam edilir).", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    # Başka bir süreçten (örn. arayüzden) durdurma: output_dir/STOP dosyası.
    stop_file = output_dir / "STOP"
    stop_file.unlink(missing_ok=True)

    # -- train ------------------------------------------------------------ #
    history = []
    timing = {"compute": [0.0] * world, "sync": 0.0, "steps": 0}
    interval = {"compute": [0.0] * world, "sync": 0.0, "sums": [0.0] * 5}
    started = last_log = time.time()
    stop = completed = stopped_by_user = early_stopped = False
    epoch = start_epoch
    pool = futures.ThreadPoolExecutor(max_workers=world)
    print(">> eğitim başlıyor\n", flush=True)

    for epoch in range(start_epoch, args.epochs):
        first = epoch == start_epoch
        for sampler in samplers:
            sampler.skip = skip if first else 0
        n_steps = steps_in_epoch(epoch) - (skip if first else 0)
        iterators = [iter(loader) for loader in loaders]
        for _ in range(n_steps):
            jobs = [pool.submit(run_share, r, iterators[r], accums[r]) for r in range(world)]
            try:
                results = [job.result() for job in jobs]
            except Exception:
                if not stop_request["flag"]:
                    raise
                # Ctrl+C veri yükleyicilerini de durdurur; yarım kalan adım atılır, önceki adım kaydedilir.
                futures.wait(jobs)
                for ps in params:
                    for p in ps:
                        p.grad = None
                stopped_by_user = stop = True
                break
            sync_start = time.time()
            if world > 1:
                gather_grads()
            if own.dump_grads:
                dumped = {}
                for i, (name, p) in enumerate(zip(names, params[0])):
                    g = p.grad.index_select(0, rows_per_device[0]) if sparse[i] else p.grad
                    dumped[name] = g.detach().float().cpu()
                torch.save(dumped, output_dir / "grads_step1.pt")
                print(">> ilk adımın gradyanları kaydedildi: " + str(output_dir / "grads_step1.pt"))
                stop = True
                break
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params[0], args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if world > 1:
                push_weights()
                for device in devices:
                    torch.cuda.synchronize(device)
            sync_time = time.time() - sync_start
            global_step += 1

            for r, (stats, compute_time) in enumerate(results):
                interval["compute"][r] += compute_time
                timing["compute"][r] += compute_time
                for loss_v, mel_v, text_v, top1_v, size in stats:
                    for k, v in enumerate((loss_v * size, mel_v * size, text_v * size, top1_v * size, size)):
                        interval["sums"][k] += v
            interval["sync"] += sync_time
            timing["sync"] += sync_time
            timing["steps"] += 1

            if global_step % args.log_interval == 0:
                n = max(1.0, interval["sums"][4])
                values = {"loss": interval["sums"][0] / n, "mel_loss": interval["sums"][1] / n,
                          "text_loss": interval["sums"][2] / n, "mel_top1": interval["sums"][3] / n}
                elapsed = time.time() - last_log
                last_log = time.time()
                rate = args.log_interval / max(elapsed, 1e-6)
                lr = scheduler.get_last_lr()[0]
                per_gpu = " ".join("gpu{}:{:.2f}s/{:.1f}G".format(
                    gpus[r], interval["compute"][r] / args.log_interval,
                    torch.cuda.max_memory_allocated(devices[r]) / 1024 ** 3) for r in range(world))
                print("step {:>7} | ep {} | loss {:.4f} | mel {:.4f} | text {:.4f} | top1 {:.3f} | lr {:.2e} | "
                      "{:.2f} it/s | hesap {} | senkron {:.2f}s | eta {}".format(
                          global_step, epoch + 1, values["loss"], values["mel_loss"], values["text_loss"],
                          values["mel_top1"], lr, rate, per_gpu, interval["sync"] / args.log_interval,
                          T.human_time((total_steps - global_step) / max(rate, 1e-6))), flush=True)
                for key, value in values.items():
                    writer.add_scalar("train/" + key, value, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                history.append({"step": global_step, **values, "it_s": rate, "time": time.time() - started})
                if world > 1:
                    # Kopyalar ayrıştı mı? (senkron bozuksa burada yakalanır)
                    sums = [sum(p.detach().double().sum().item() for p in ps) for ps in params]
                    if any(abs(s - sums[0]) > 1e-6 * max(1.0, abs(sums[0])) for s in sums[1:]):
                        raise RuntimeError("GPU kopyaları ayrıştı: " + str(sums))
                interval = {"compute": [0.0] * world, "sync": 0.0, "sums": [0.0] * 5}

            if val_loader is not None and args.val_interval and global_step % args.val_interval == 0:
                val = T.evaluate(main_model, val_loader, devices[0], loss_cfg, amp_dtype)
                print("  val @ {:>7} | ".format(global_step) + " | ".join(
                    "{} {:.4f}".format(k, v) for k, v in sorted(val.items()) if "/" not in k), flush=True)
                for key, value in val.items():
                    writer.add_scalar("val/" + key, value, global_step)
                history.append({"step": global_step, "val": val})
                # Ezberleme bekçisi: görülmemiş konuşmacılardaki kayıp iyileşmiyorsa say, sınırda dur.
                current = float(val["loss"])
                if early["best"] is None or current < float(early["best"]) - own.early_stop_min_delta:
                    early.update(best=current, best_step=global_step, bad=0)
                    print("  en iyi doğrulama: {:.4f} -> {}".format(current, save_best(epoch)), flush=True)
                else:
                    early["bad"] = int(early["bad"]) + 1
                    print("  doğrulama iyileşmedi ({}/{}); en iyi {:.4f} @ adım {}".format(
                        early["bad"], own.early_stop_patience or "-", float(early["best"]), early["best_step"]),
                        flush=True)
                    if own.early_stop_patience and int(early["bad"]) >= own.early_stop_patience:
                        print("  ERKEN DURDURMA: doğrulama kaybı {} ölçümdür iyileşmiyor (ezberleme belirtisi). "
                              "En iyi checkpoint: best.pt (adım {}).".format(early["bad"], early["best_step"]),
                              flush=True)
                        early_stopped = stop = True

            if args.save_interval and global_step % args.save_interval == 0:
                print("  kaydedildi " + str(ckpt_manager.save(checkpoint_state(epoch), global_step)), flush=True)

            if not stop_request["flag"] and stop_file.is_file():
                print("\n>> STOP dosyası bulundu: checkpoint kaydedilip çıkılıyor.", flush=True)
                stop_request["flag"] = True
            if stop_request["flag"]:
                stopped_by_user = stop = True
            if args.max_steps and global_step >= args.max_steps:
                stop = True
            if stop:
                break
        if stop:
            break
    else:
        completed = True
    pool.shutdown(wait=True)

    if stopped_by_user and not own.dump_grads:
        stop_file.unlink(missing_ok=True)
        path = ckpt_manager.save(checkpoint_state(epoch), global_step)
        writer.close()
        print("\n>> Durduruldu; adım {} kaydedildi: {}".format(global_step, path))
        print(">> Devam etmek için aynı komutu (04_train.bat) tekrar çalıştır.")
        for mask in masks:
            mask.remove()
        return 0

    # -- finish ----------------------------------------------------------- #
    wall = time.time() - started
    final_val = None
    if val_loader is not None and not own.dump_grads:
        final_val = T.evaluate(main_model, val_loader, devices[0], loss_cfg, amp_dtype)
        print("\nson doğrulama: " + ", ".join("{} {:.4f}".format(k, v) for k, v in sorted(final_val.items())
                                             if "/" not in k))
        if early["best"] is None or float(final_val["loss"]) < float(early["best"]):
            early.update(best=float(final_val["loss"]), best_step=global_step)
            print("  en iyi doğrulama: {:.4f} -> {}".format(float(final_val["loss"]), save_best(epoch)))
    steps_done = max(1, timing["steps"])
    report = {
        "gpus": gpus, "rank_accum": accums, "batch_size": args.batch_size, "steps": global_step,
        "wall_seconds": wall, "seconds_per_step": wall / steps_done, "sync_s_per_step": timing["sync"] / steps_done,
        "per_gpu": [{"gpu": gpus[r], "peak_alloc_gb": torch.cuda.max_memory_allocated(devices[r]) / 1024 ** 3,
                     "peak_reserved_gb": torch.cuda.max_memory_reserved(devices[r]) / 1024 ** 3,
                     "compute_s_per_step": timing["compute"][r] / steps_done} for r in range(world)],
        "history": history, "final_val": final_val, "early_stop": dict(early),
        "completed": completed, "early_stopped": early_stopped,
    }
    print("\n>> {} adım {:.0f} sn ({:.2f} sn/adım, senkron {:.2f} sn/adım)".format(
        global_step, wall, report["seconds_per_step"], report["sync_s_per_step"]))
    for item in report["per_gpu"]:
        print("   GPU {gpu}: tepe {peak_alloc_gb:.1f} GB (rezerve {peak_reserved_gb:.1f} GB), "
              "hesap {compute_s_per_step:.2f} sn/adım".format(**item))
    if own.report_json:
        Path(own.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not own.dump_grads:
        # Bitmiş (ya da erken durmuş) koşu "finished" işaretlenir; --max-steps ile kesilen koşu devam ettirilebilir.
        finished = completed or early_stopped
        last_ckpt = ckpt_manager.save(checkpoint_state(epoch if not completed else args.epochs, finished=finished),
                                      global_step)
        if early["best"] is not None:
            print(">> en iyi checkpoint: {} (adım {}, val {:.4f}) -> 05_export.bat --best".format(
                ckpt_dir / "best.pt", early["best_step"], float(early["best"])))
        if own.no_final_export or args.final_export == "none":
            pass
        elif args.final_export == "adapter":
            from itts25ft.adapter import adapter_from_checkpoint

            base = Path(args.base_checkpoint) if args.base_checkpoint else model_dir / "gpt.pth"
            path = adapter_from_checkpoint(
                base, last_ckpt, output_dir / "adapters" / "{}_step{}.safetensors".format(args.lang, global_step),
                lang=args.lang, normalizer=args.normalizer, case=args.case, extra_meta={"step": global_step},
            )
            print("son adımın adaptörü: " + str(path))
        else:
            result = export_inference_checkpoint(main_model, output_dir / "exported" / "gpt.pth",
                                                 metadata={"step": global_step, "lang": args.lang, "multi_gpu": gpus})
            print("son adımın çıkarım ağırlığı: " + str(result["path"]))
    writer.close()
    for mask in masks:
        mask.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
