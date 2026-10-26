"""
Runs inside the Colab kernel (sent by colab/run_colab.sh through `colab exec --file`).

Unpacks the repository uploaded to /content/mri.tar.gz, installs the requirements and runs
preprocess.py -> train.py -> evaluate.py for every architecture in $ARCHS, then packs
results/ and the processed split summary into /content/mri/results.tar.gz.
"""
import os
import subprocess

WORKDIR = "/content/mri"
ARCHS = os.environ.get("ARCHS", "resnet18 simplecnn").split()
SEED = os.environ.get("SEED", "42")
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


os.makedirs(WORKDIR, exist_ok=True)
run("tar xzf /content/mri.tar.gz")
run("pip install -q -r requirements.txt")
run("mkdir -p results && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > results/colab_gpu.csv"
    " && python -c 'import torch, torchvision; print(\"torch\", torch.__version__, \"| torchvision\", torchvision.__version__,"
    " \"| CUDA\", torch.version.cuda)' | tee results/colab_versions.txt && cat results/colab_gpu.csv")
run(f"python preprocess.py --data_dir classification --output_dir data/processed --seed {SEED}",
    log="results/preprocess.log")
run("cp data/processed/summary.json data/processed/splits.csv results/")
for arch in ARCHS:
    run(f"python train.py --arch {arch} --epochs {EPOCHS} --seed {SEED} --num_workers 4 --output_dir results/{arch}",
        log=f"results/{arch}/train.log")
    run(f"python evaluate.py --checkpoint results/{arch}/model.pt --output_dir results/{arch} --seed {SEED}",
        log=f"results/{arch}/evaluate.log")
run("tar czf results.tar.gz results")
print("REMOTE_DONE", flush=True)
