import multiprocessing
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    sys.path.remove(PROJECT_ROOT)
except ValueError:
    pass
sys.path.insert(0, PROJECT_ROOT)

from scipy import signal

inp_root = sys.argv[1]
sr = int(sys.argv[2])
n_p = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
manifest_path = sys.argv[7] if len(sys.argv) > 7 else ""
noise_reduction = len(sys.argv) > 8 and sys.argv[8].lower() == "true"
reduction_strength = float(sys.argv[9]) if len(sys.argv) > 9 else 0.75
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

os.environ["NVC_AUDIO_FORCE_CPU"] = "1"
from infer.audio import load_audio
from train.dataset.slicer2 import Slicer, split_audio_with_overlap
from i18n.i18n import I18nAuto
from tools.progress import should_report
from tools.multispeaker import ManifestError, load_manifest

i18n = I18nAuto()

# High-quality resampling for the 16 kHz feature copies when soxr is present.
try:
    import soxr  # noqa: F401

    RESAMPLE_TYPE = "soxr_vhq"
except ImportError:
    RESAMPLE_TYPE = None

NOISE_REDUCE = None
if noise_reduction:
    try:
        import noisereduce as NOISE_REDUCE
    except ImportError:
        NOISE_REDUCE = None

f = open("%s/preprocess.log" % exp_dir, "a", encoding="utf8")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class PreProcess:
    def __init__(self, sr, exp_dir, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, output_key, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
            println(
                i18n("[数据切分][跳过] 无效或异常音频片段：%s_%s | 峰值：%s")
                % (output_key, idx1, tmp_max)
            )
            return False
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, output_key, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        if RESAMPLE_TYPE is not None:
            audio_16k = librosa.resample(
                tmp_audio, orig_sr=self.sr, target_sr=16000, res_type=RESAMPLE_TYPE
            ).astype(np.float32)
        else:
            audio_16k = librosa.resample(
                tmp_audio, orig_sr=self.sr, target_sr=16000
            ).astype(np.float32)
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, output_key, idx1),
            16000,
            audio_16k,
        )
        return True

    def pipeline(self, path, output_key, progress_index, total):
        try:
            audio = load_audio(path, self.sr)
            if NOISE_REDUCE is not None:
                audio = NOISE_REDUCE.reduce_noise(
                    y=audio,
                    sr=self.sr,
                    prop_decrease=reduction_strength,
                ).astype(np.float32)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for sliced_audio in self.slicer.slice(audio):
                for tmp_audio in split_audio_with_overlap(
                    sliced_audio, self.sr, self.per, self.overlap
                ):
                    self.norm_write(tmp_audio, output_key, idx1)
                    idx1 += 1
            if should_report(progress_index, total):
                println(
                    i18n("[数据切分] 进度：%s/%s | %s")
                    % (progress_index + 1, total, os.path.basename(path))
                )
            return True
        except Exception:
            println(
                i18n("[数据切分][失败] %s\n%s")
                % (path, traceback.format_exc())
            )
            return False

    def pipeline_mp(self, infos):
        success = 0
        failed = 0
        for path, output_key, progress_index, total in infos:
            if self.pipeline(path, output_key, progress_index, total):
                success += 1
            else:
                failed += 1
        if infos:
            println(
                i18n("[数据切分] 子任务完成 | 成功：%s | 失败：%s")
                % (success, failed)
            )

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            names = sorted(
                name for name in os.listdir(inp_root)
                if os.path.isfile(os.path.join(inp_root, name))
            )
            total = len(names)
            infos = [
                ("%s/%s" % (inp_root, name), str(idx), idx, total)
                for idx, name in enumerate(names)
            ]
            worker_count = max(n_p, 1)
            worker_count = min(worker_count, max(total, 1))
            println(
                i18n("[数据切分] 待处理：%s | 进程数：%s")
                % (total, worker_count)
            )
            if noparallel:
                for i in range(worker_count):
                    self.pipeline_mp(infos[i::worker_count])
            else:
                ps = []
                for i in range(worker_count):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::worker_count],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(worker_count):
                    ps[i].join()
        except Exception:
            println(i18n("[数据切分][失败] %s") % traceback.format_exc())

    def pipeline_mp_manifest(self, manifest_entries, n_p):
        infos = [
            (entry["path"], entry["output_key"], idx, len(manifest_entries))
            for idx, entry in enumerate(manifest_entries)
        ]
        total = len(infos)
        worker_count = max(n_p, 1)
        worker_count = min(worker_count, max(total, 1))
        println(
            i18n("[数据切分] 多说话人待处理：%s | 进程数：%s")
            % (total, worker_count)
        )
        if noparallel:
            for i in range(worker_count):
                self.pipeline_mp(infos[i::worker_count])
            return
        ps = []
        for i in range(worker_count):
            p = multiprocessing.Process(
                target=self.pipeline_mp, args=(infos[i::worker_count],)
            )
            ps.append(p)
            p.start()
        for p in ps:
            p.join()


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per):
    pp = PreProcess(sr, exp_dir, per)
    println(i18n("[数据切分] 开始"))
    if manifest_path:
        try:
            manifest = load_manifest(exp_dir)
            for output_dir in (pp.gt_wavs_dir, pp.wavs16k_dir):
                for name in os.listdir(output_dir):
                    if name.startswith("ms") and name.endswith(".wav"):
                        try:
                            os.remove(os.path.join(output_dir, name))
                        except OSError:
                            pass
            pp.pipeline_mp_manifest(manifest["entries"], n_p)
        except ManifestError as error:
            println(i18n(error.key) % error.values)
            raise
    else:
        for output_dir in (pp.gt_wavs_dir, pp.wavs16k_dir):
            for name in os.listdir(output_dir):
                if name.startswith("ms") and name.endswith(".wav"):
                    try:
                        os.remove(os.path.join(output_dir, name))
                    except OSError:
                        pass
        pp.pipeline_mp_inp_dir(inp_root, n_p)
    println(i18n("[数据切分] 完成"))


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
