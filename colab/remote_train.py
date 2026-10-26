"""
Runs inside the Colab kernel (sent by colab/run_colab.sh through `colab exec --file`).

Unpacks the repository uploaded to /content/mri.tar.gz, installs the requirements, preprocesses the
data and then, depending on $PIPELINE:

    train  preprocess.py -> train.py -> evaluate.py for every architecture in $ARCHS
    cv     cv.py for every config in $CONFIGS (in parallel on the same GPU), cv.py --summarize, then
           retrains "baseline" and the best config on train/val (the best one saving a checkpoint per
           epoch) and evaluates both on the test split with the 0.5 and the validation-chosen thresholds
           (CV_ONLY=1 stops after the summary)
    scratch  cv.py for the baseline_scratch control (results/cv_scratch), then retrains it on train/val and
             evaluates it on the test split, like the final runs of the cv pipeline

Packs results/ (without checkpoints) into /content/mri/results.tar.gz and the checkpoints into
/content/mri/checkpoints.tar.
"""
import os
import json
import subprocess

WORKDIR = "/content/mri"
PIPELINE = os.environ.get("PIPELINE", "train")
ARCHS = os.environ.get("ARCHS", "resnet18 simplecnn").split()
CONFIGS = os.environ.get("CONFIGS", "baseline schedule schedule_wd schedule_wd_freeze").split()
SEED = os.environ.get("SEED", "42")
CV_SEEDS = os.environ.get("CV_SEEDS", "42 43 44")
EPOCHS = os.environ.get("EPOCHS", "60")


def run(cmd: str, log: str = "") -> None:
    """Runs a shell command, echoing its output to the kernel (and to `log`, if given)."""
    print(f"$ {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, cwd=WORKDIR, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    if log:
        os.makedirs(os.path.dirname(os.path.join(WORKDIR, log)), exist_ok=True)
        with open(os.path.join(WORKDIR, log), "w") as f:
            f.writelines(lines)
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_parallel(cmds: dict) -> None:
    """Starts every {log: cmd} at once, waits for all of them and prints each log at the end."""
    procs = {}
    for log, cmd in cmds.items():
        os.makedirs(os.path.dirname(os.path.join(WORKDIR, log)), exist_ok=True)
        print(f"$ {cmd} > {log}", flush=True)
        procs[log] = subprocess.Popen(f"{cmd} > {log} 2>&1", shell=True, cwd=WORKDIR)
    failed = [log for log, p in procs.items() if p.wait() != 0]
    for log in procs:
        with open(os.path.join(WORKDIR, log)) as f:
            print(f.read(), flush=True)
    if failed:
        raise RuntimeError(f"failed: {failed}")


def train_and_evaluate(name: str, extra: str = "") -> None:
    out = f"results/final/{name}"
    run(f"python train.py --config {name} --seed {SEED} --num_workers 4 --output_dir {out} {extra}",
        log=f"{out}/train.log")
    run(f"python evaluate.py --checkpoint {out}/model.pt --output_dir {out} --seed {SEED}",
        log=f"{out}/evaluate.log")
    run(f"python evaluate.py --checkpoint {out}/model.pt --output_dir {out} --seed {SEED} "
        f"--threshold val --suffix _valthr", log=f"{out}/evaluate_valthr.log")


os.makedirs(WORKDIR, exist_ok=True)
run("tar xzf /content/mri.tar.gz")
run("pip install -q -r requirements.txt")
run("mkdir -p results && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > results/colab_gpu.csv"
    " && python -c 'import torch, torchvision; print(\"torch\", torch.__version__, \"| torchvision\", torchvision.__version__,"
    " \"| CUDA\", torch.version.cuda)' | tee results/colab_versions.txt && cat results/colab_gpu.csv")
run(f"python preprocess.py --data_dir classification --output_dir data/processed --seed {SEED}",
    log="results/preprocess.log")
run("cp data/processed/summary.json data/processed/splits.csv results/")

if PIPELINE == "train":
    for arch in ARCHS:
        run(f"python train.py --arch {arch} --epochs {EPOCHS} --seed {SEED} --num_workers 4 --output_dir results/{arch}",
            log=f"results/{arch}/train.log")
        run(f"python evaluate.py --checkpoint results/{arch}/model.pt --output_dir results/{arch} --seed {SEED}",
            log=f"results/{arch}/evaluate.log")
elif PIPELINE == "cv":
    # Torch downloads the ImageNet weights once, before the parallel runs race for the same file.
    run("python -c 'from classifier import build_model; build_model(\"resnet18\")'")
    run_parallel({f"results/cv/{c}/cv.log": f"python cv.py --config {c} --seeds {CV_SEEDS} --output_dir results/cv"
                  for c in CONFIGS})
    run("python cv.py --summarize results/cv", log="results/cv/summarize.log")
    if os.environ.get("CV_ONLY") != "1":
        with open(os.path.join(WORKDIR, "results/cv/best_config.json")) as f:
            best = json.load(f)["config"]
        if best != "baseline":
            train_and_evaluate("baseline")
        train_and_evaluate(best, extra="--save_every_epoch")
elif PIPELINE == "scratch":
    run("python cv.py --config baseline_scratch --seeds {} --output_dir results/cv_scratch".format(CV_SEEDS),
        log="results/cv_scratch/baseline_scratch/cv.log")
    run("python cv.py --summarize results/cv_scratch", log="results/cv_scratch/summarize.log")
    train_and_evaluate("baseline_scratch")
else:
    raise ValueError(f"Unknown PIPELINE '{PIPELINE}'")

run("tar czf results.tar.gz --exclude='*.pt' results")
run("find results -name '*.pt' | tar cf checkpoints.tar -T -")
print("REMOTE_DONE", flush=True)
