import os
import math
import logging
import sys
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.remove(PROJECT_ROOT)
except ValueError:
    pass
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.GradScaler.*is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="`torch.cuda.amp.autocast.*is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Grad strides do not match bucket view strides.*",
    category=UserWarning,
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

import datetime

from train import utils

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
training_deterministic = bool(getattr(hps.train, "deterministic", False))
if training_deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

from configs.config import get_training_dtype
from i18n.i18n import I18nAuto

i18n = I18nAuto()

training_dtype = get_training_dtype(
    prefer_bf16=bool(getattr(hps.train, "bf16", False))
)
training_is_half = training_dtype == torch.float16
training_use_amp = training_dtype in (torch.float16, torch.bfloat16)

from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = training_deterministic
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(training_deterministic)
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.module import commons
from train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.module.models import MultiPeriodDiscriminator
    from infer.module.models import SynthesizerTrnMs256NSFsid as NVC_Model_f0
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid_nono as NVC_Model_nof0,
    )
else:
    from infer.module.models import (
        SynthesizerTrnMs768NSFsid as NVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as NVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from train.mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from train.process_ckpt import savee
from train.lr_schedulers import create_lr_scheduler

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def load_pretrained_generator(model, path):
    target = model.module if hasattr(model, "module") else model
    saved_state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    current_state = target.state_dict()
    embedding_key = "emb_g.weight"
    if embedding_key in saved_state and embedding_key in current_state:
        saved_embedding = saved_state[embedding_key]
        current_embedding = current_state[embedding_key]
        if saved_embedding.shape != current_embedding.shape:
            compatible = (
                saved_embedding.dim() == current_embedding.dim()
                and saved_embedding.shape[1:] == current_embedding.shape[1:]
            )
            if compatible:
                expanded = current_embedding.clone()
                rows = min(saved_embedding.shape[0], current_embedding.shape[0])
                expanded[:rows].copy_(saved_embedding[:rows])
                saved_state[embedding_key] = expanded
    return target.load_state_dict(saved_state)


def main():
    n_gpus = torch.cuda.device_count()

    if n_gpus < 1:
        print(i18n("未检测到可用显卡，将使用CPU训练，耗时可能较长"))
        n_gpus = 1
    logger = utils.get_logger(hps.model_dir)
    logger.info(i18n("训练设备规则选择的精度：%s"), training_dtype)
    if n_gpus == 1:
        run(0, 1, hps, logger, False)
        return
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger, True),
        )
        children.append(subproc)
        subproc.start()

    for child in children:
        child.join()
    failed = [child.exitcode for child in children if child.exitcode != 0]
    if failed:
        raise RuntimeError("Training worker(s) failed with exit codes: %s" % failed)


def run(rank, n_gpus, hps, logger, use_ddp):
    try:
        return _run(rank, n_gpus, hps, logger, use_ddp)
    finally:
        if use_ddp and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run(rank, n_gpus, hps, logger, use_ddp):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    if use_ddp:
        backend = (
            "nccl"
            if torch.cuda.is_available() and dist.is_nccl_available()
            else "gloo"
        )
        dist.init_process_group(
            backend=backend,
            init_method="env://?use_libuv=False",
            world_size=n_gpus,
            rank=rank,
        )

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(
            hps.data.training_files,
            hps.data,
        )
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    num_workers = int(getattr(hps.train, "num_workers", 0))
    loader_options = {
        "dataset": train_dataset,
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
        "batch_sampler": train_sampler,
    }
    if num_workers > 0:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=int(getattr(hps.train, "prefetch_factor", 2)),
        )
    train_loader = DataLoader(**loader_options)
    fn_mel_loss = None
    if getattr(hps.train, "multiscale_mel", True):
        fn_mel_loss = MultiScaleMelSpectrogramLoss(hps.data.sampling_rate)
        if rank == 0:
            logger.info(i18n("Используется многомасштабная mel-потеря"))
    if hps.if_f0 == 1:
        net_g = NVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=training_is_half,
            sr=hps.sample_rate,
            gradient_checkpointing=bool(
                getattr(hps.train, "gradient_checkpointing", False)
            ),
        )
    else:
        if getattr(hps.train, "gradient_checkpointing", False) and rank == 0:
            logger.info(
                i18n("градиентный чекпоинтинг поддерживается только для f0-моделей")
            )
        net_g = NVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=training_is_half,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    if training_dtype == torch.bfloat16:
        # BF16 optimizer states with Kahan summation keep weight updates at
        # FP32-level quality (ported from Applio/torchao AnyPrecisionAdamW).
        from train.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer_class = AnyPrecisionAdamW
        if rank == 0:
            logger.info(i18n("Используется оптимизатор AnyPrecisionAdamW (BF16)"))
    else:
        optimizer_class = torch.optim.AdamW
    optim_g = optimizer_class(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = optimizer_class(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if use_ddp:
        if torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)

    discriminator_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
    generator_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
    if bool(discriminator_checkpoint) != bool(generator_checkpoint):
        raise RuntimeError(
            "Incomplete training checkpoint pair: generator=%r, discriminator=%r"
            % (generator_checkpoint, discriminator_checkpoint)
        )

    if generator_checkpoint:
        _, _, _, discriminator_epoch = utils.load_checkpoint(
            discriminator_checkpoint, net_d, optim_d
        )
        if rank == 0:
            logger.info(i18n("已恢复判别器检查点"))
        _, _, _, generator_epoch, restored_training_state = utils.load_checkpoint(
            generator_checkpoint,
            net_g,
            optim_g,
            return_training_state=True,
        )
        if discriminator_epoch != generator_epoch:
            raise RuntimeError(
                "Generator/discriminator checkpoint epochs differ: G=%s, D=%s"
                % (generator_epoch, discriminator_epoch)
            )
        epoch_str = generator_epoch + 1
        global_step = generator_epoch * len(train_loader)
    else:
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info(i18n("已加载生成器预训练模型：%s") % hps.pretrainG)
            logger.info(load_pretrained_generator(net_g, hps.pretrainG))
        if hps.pretrainD != "":
            if rank == 0:
                logger.info(i18n("已加载判别器预训练模型：%s") % hps.pretrainD)
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(
                            hps.pretrainD, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(
                            hps.pretrainD, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )

    configured_scheduler = getattr(hps.train, "lr_scheduler", None)
    scheduler_name = configured_scheduler or (
        "exponential" if generator_checkpoint else "step"
    )
    if rank == 0 and configured_scheduler is None and generator_checkpoint:
        logger.info(
            "Legacy checkpoint has no configured LR scheduler; preserving "
            "historical ExponentialLR behavior"
        )
    min_lr_ratio = float(getattr(hps.train, "min_lr_ratio", 0.01))
    scheduler_g, scheduler_interval = create_lr_scheduler(
        optim_g,
        scheduler_name,
        hps.total_epoch,
        len(train_loader),
        hps.train.lr_decay,
        min_lr_ratio,
    )
    scheduler_d, discriminator_interval = create_lr_scheduler(
        optim_d,
        scheduler_name,
        hps.total_epoch,
        len(train_loader),
        hps.train.lr_decay,
        min_lr_ratio,
    )
    if scheduler_interval != discriminator_interval:
        raise RuntimeError("Generator/discriminator scheduler intervals differ")
    if rank == 0:
        logger.info(
            "LR scheduler: %s (%s step, min_lr_ratio=%s)",
            scheduler_name,
            scheduler_interval,
            min_lr_ratio,
        )

    scaler = GradScaler(enabled=training_is_half)
    if generator_checkpoint:
        if utils.restore_training_state(
            restored_training_state,
            scaler,
            rank,
            [scheduler_g, scheduler_d],
        ):
            if rank == 0:
                logger.info(i18n("已恢复随机数生成器和AMP scaler状态"))
        elif rank == 0:
            logger.warning(
                "Checkpoint predates exact-resume state; RNG and AMP scaler were reset"
            )

    cache = []
    reference = None
    if rank == 0 and hps.if_f0 == 1:
        # Fixed reference batch used to write audible samples into TensorBoard
        # each save interval (idea ported from Applio).
        try:
            ref_info = next(iter(train_loader))
            ref_indices = (0, 1, 2, 3, 8)  # phone, lengths, pitch, pitchf, sid
            reference = tuple(
                (
                    ref_info[idx].cuda(rank, non_blocking=True)
                    if torch.cuda.is_available()
                    else ref_info[idx]
                )
                for idx in ref_indices
            )
        except StopIteration:
            reference = None
    for epoch in range(epoch_str, hps.total_epoch + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
                scheduler_interval,
                fn_mel_loss,
                reference,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
                scheduler_interval,
                fn_mel_loss,
                None,
            )
        if scheduler_interval == "epoch":
            scheduler_g.step()
            scheduler_d.step()
    if rank == 0:
        writer.close()
        writer_eval.close()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    cache,
    scheduler_interval,
    fn_mel_loss=None,
    reference=None,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=training_use_amp, dtype=training_dtype):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    sid,
                )
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if training_use_amp:
                y_hat_mel = y_hat_mel.to(training_dtype)
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=training_use_amp, dtype=training_dtype):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                if fn_mel_loss is not None:
                    # Multi-scale mel loss on the sliced waveforms (Applio).
                    loss_mel = (
                        fn_mel_loss(wave.float(), y_hat.float())
                        * hps.train.c_mel
                        / 3.0
                    )
                else:
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        if scheduler_interval == "batch":
            if math.isfinite(grad_norm_d):
                scheduler_d.step()
            if math.isfinite(grad_norm_g):
                scheduler_g.step()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    i18n("训练轮次：{} [{:.0f}%]").format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.float().cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.float().cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.float().cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    should_save = epoch % hps.save_every_epoch == 0
    gathered_training_state = None
    if should_save:
        local_training_state = utils.capture_training_state(
            scaler, schedulers, scheduler_interval
        )
        if dist.is_available() and dist.is_initialized():
            rank_states = [None] * dist.get_world_size() if rank == 0 else None
            dist.gather_object(local_training_state, rank_states, dst=0)
            if rank == 0:
                gathered_training_state = {"rank_states": rank_states}
        else:
            gathered_training_state = {"rank_states": [local_training_state]}

    if should_save and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                training_state=gathered_training_state,
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
                training_state=gathered_training_state,
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if reference is not None:
            # Audible progress: synthesize the fixed reference batch and write
            # it into TensorBoard alongside the spectrogram images.
            try:
                eval_net_g = net_g.module if hasattr(net_g, "module") else net_g
                with torch.no_grad():
                    with autocast(enabled=training_use_amp, dtype=training_dtype):
                        audio_out, *_ = eval_net_g.infer(*reference)
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    audios={
                        "gen/audio_reference": audio_out[0, :, :].float().cpu()
                    },
                    audio_sampling_rate=hps.data.sampling_rate,
                )
            except Exception as error:
                logger.warning(i18n("Не удалось записать аудио-эвалюацию: %s") % error)
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            save_result = savee(
                ckpt,
                hps.sample_rate,
                hps.if_f0,
                hps.name + "_e%s_s%s" % (epoch, global_step),
                epoch,
                hps.version,
                hps,
            )
            if save_result != i18n("成功"):
                raise RuntimeError(save_result)
            logger.info(
                i18n("正在保存检查点 %s_e%s：%s")
                % (hps.name, epoch, save_result)
            )

    if rank == 0:
        logger.info(i18n("====> 轮次：{} {}").format(epoch, epoch_recorder.record()))
    if epoch == hps.total_epoch and rank == 0:
        logger.info(i18n("训练已完成，正在保存最终模型"))

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        save_result = savee(
            ckpt,
            hps.sample_rate,
            hps.if_f0,
            hps.name,
            epoch,
            hps.version,
            hps,
        )
        if save_result != i18n("成功"):
            raise RuntimeError(save_result)
        logger.info(i18n("正在保存最终检查点：%s") % save_result)
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
