# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import time
import zipfile
from cog import BasePredictor, Input, Path as CogPath
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
import json
import os
import pathlib
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import faiss
import subprocess
import shutil
import glob
from zipfile import ZipFile
import traceback # Added for potential MiniBatchKMeans error logging

# Assume config and logger are handled appropriately in the full RVC environment
# For Cog, we'll just use print and basic logging
class DummyConfig:
    def __init__(self):
        self.n_cpu = os.cpu_count()
config = DummyConfig()

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# List of URLs and destinations (Already includes 32k models)
downloads = [
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D32k.pth",
        "assets/pretrained/D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D40k.pth",
        "assets/pretrained/D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D48k.pth",
        "assets/pretrained/D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G32k.pth",
        "assets/pretrained/G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G40k.pth",
        "assets/pretrained/G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G48k.pth",
        "assets/pretrained/G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D32k.pth",
        "assets/pretrained/f0D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D40k.pth",
        "assets/pretrained/f0D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D48k.pth",
        "assets/pretrained/f0D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G32k.pth",
        "assets/pretrained/f0G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G40k.pth",
        "assets/pretrained/f0G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G48k.pth",
        "assets/pretrained/f0G48k.pth",
    ),
    # v2 models
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D32k.pth",
        "assets/pretrained_v2/D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D40k.pth",
        "assets/pretrained_v2/D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D48k.pth",
        "assets/pretrained_v2/D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G32k.pth",
        "assets/pretrained_v2/G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G40k.pth",
        "assets/pretrained_v2/G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G48k.pth",
        "assets/pretrained_v2/G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D32k.pth",
        "assets/pretrained_v2/f0D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D40k.pth",
        "assets/pretrained_v2/f0D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D48k.pth",
        "assets/pretrained_v2/f0D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G32k.pth",
        "assets/pretrained_v2/f0G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G40k.pth",
        "assets/pretrained_v2/f0G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G48k.pth",
        "assets/pretrained_v2/f0G48k.pth",
    ),
    # Hubert and RMVPE
    (
        "https://weights.replicate.delivery/default/rvc/assets/hubert/hubert_base.pt",
        "assets/hubert/hubert_base.pt",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/rmvpe/rmvpe.pt",
        "assets/rmvpe/rmvpe.pt",
    ),
]


def infer_folder_name(base_path):
    # Print the current working directory and base path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Base path: {base_path}")

    # Check if the directory exists
    if not os.path.isdir(base_path):
        print(f"Directory does not exist: {base_path}")
        return None

    # List all directories in the base_path
    dirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    print(f"Found directories in {base_path}: {dirs}")

    # Return the first directory name
    return dirs[0] if dirs else None


def execute_command(command):
    print(f"Executing command: {command}")
    # Use Popen to capture and print output in real-time
    process = Popen(
        command,
        shell=True,
        stdout=PIPE,
        stderr=STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    output_lines = []
    for line in process.stdout:
        print(line.strip())
        output_lines.append(line)
    process.wait()
    output = "".join(output_lines)

    if process.returncode != 0:
        print(f"Error occurred executing command: {command}")
        print(f"Return code: {process.returncode}")
        print(f"Output/Error: {output}")
        # Potentially raise an exception here if needed
    else:
        print(f"Command executed successfully: {command}")

    return output, process.returncode # Return output and return code


def train_index(exp_dir1, version19):
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    print(f"Looking for features in: {feature_dir}") # Debug print
    if not os.path.exists(feature_dir):
        print(f"Feature directory not found: {feature_dir}")
        return "Please perform feature extraction first!" # Changed error message slightly

    listdir_res = list(os.listdir(feature_dir))
    print(f"Files in feature directory: {listdir_res}") # Debug print
    if len(listdir_res) == 0:
        print(f"No files found in feature directory: {feature_dir}")
        return "Please perform feature extraction first!"

    infos = []
    npys = []
    for name in sorted(listdir_res):
        if not name.endswith(".npy"): # Skip non-npy files if any
            continue
        try:
            phone = np.load("%s/%s" % (feature_dir, name))
            npys.append(phone)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            infos.append(f"Error loading {name}: {e}")

    if not npys:
         print("No .npy files were successfully loaded.")
         return "Feature extraction might have failed. No .npy files found or loaded."

    yield "Concatenating features..." # Progress update
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5: # If dataset is huge, maybe apply MiniBatchKMeans (optional, requires sklearn)
        try:
            from sklearn.cluster import MiniBatchKMeans
            infos.append("Dataset has %s features, applying MiniBatchKMeans to reduce to 10k centers." % big_npy.shape[0])
            yield "\n".join(infos)
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
            infos.append("KMeans finished.")
        except ImportError:
             infos.append("Skipping MiniBatchKMeans: scikit-learn not installed.")
             yield "\n".join(infos)
        except Exception as e:
            info = traceback.format_exc()
            logger.error(f"Error during MiniBatchKMeans: {info}")
            infos.append(f"Error during MiniBatchKMeans: {info}")
            yield "\n".join(infos) # Continue without KMeans if it fails

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), max(1, big_npy.shape[0] // 39)) # Ensure n_ivf is at least 1
    feature_dim = 256 if version19 == "v1" else 768
    infos.append(f"Feature dimension: {feature_dim}, Shape: {big_npy.shape}, n_ivf: {n_ivf}")
    yield "\n".join(infos)

    index = faiss.index_factory(feature_dim, "IVF%s,Flat" % n_ivf)
    infos.append("Training FAISS index...")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1 # Often needs adjustment, start with 1
    index.train(big_npy)
    trained_index_path = "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
    faiss.write_index(index, trained_index_path)
    infos.append(f"Index trained and saved to {trained_index_path}")


    infos.append("Adding features to index...")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])

    added_index_path = "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19)
    faiss.write_index(index, added_index_path)
    infos.append(
        "Successfully built index: %s" % os.path.basename(added_index_path)
    )
    yield "\n".join(infos) # Yield final message


def click_train(
    exp_dir1,
    sr2, # Now expects "32k", "40k", or "48k"
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # Generate filelist
    exp_dir = "%s/logs/%s" % (".", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )

    # Check directory existence before listing files
    if not os.path.exists(gt_wavs_dir):
        print(f"Ground truth waves directory not found: {gt_wavs_dir}")
        return f"Error: Ground truth waves directory not found: {gt_wavs_dir}"
    if not os.path.exists(feature_dir):
        print(f"Feature directory not found: {feature_dir}")
        return f"Error: Feature directory not found: {feature_dir}"

    gt_wav_files = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir) if name.endswith(".wav")])
    feature_files = set([name.split(".")[0] for name in os.listdir(feature_dir) if name.endswith(".npy")])
    print(f"Found {len(gt_wav_files)} wav files, {len(feature_files)} feature files.")

    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        if not os.path.exists(f0_dir) or not os.path.exists(f0nsf_dir):
             print(f"F0 directories not found (f0: {f0_dir}, f0nsf: {f0nsf_dir})")
             return "Error: F0 directories not found. Ensure F0 extraction was successful."
        f0_files = set([name.split(".")[0] for name in os.listdir(f0_dir) if name.endswith(".npy")])
        f0nsf_files = set([name.split(".")[0] for name in os.listdir(f0nsf_dir) if name.endswith(".npy")])
        print(f"Found {len(f0_files)} f0 files, {len(f0nsf_files)} f0nsf files.")
        names = gt_wav_files & feature_files & f0_files & f0nsf_files
    else:
        names = gt_wav_files & feature_files

    if not names:
        print("Error: No matching file sets found for training. Check preprocessing steps.")
        print(f" GT Wav files: {gt_wav_files}")
        print(f" Feature files: {feature_files}")
        if if_f0_3:
            print(f" F0 files: {f0_files}")
            print(f" F0nsf files: {f0nsf_files}")
        return "Error: No matching file sets found for training filelist. Check logs."

    print(f"Found {len(names)} matching sets for training.")
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"), # Windows path compatibility (keep?)
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768

    # Check if mute files exist before adding them
    mute_wav_path = f"logs/mute/0_gt_wavs/mute{sr2}.wav"
    mute_feature_path = f"logs/mute/3_feature{fea_dim}/mute.npy"
    mute_f0_path = "logs/mute/2a_f0/mute.wav.npy"
    mute_f0nsf_path = "logs/mute/2b-f0nsf/mute.wav.npy"

    mute_files_exist = os.path.exists(mute_wav_path) and os.path.exists(mute_feature_path)
    if if_f0_3:
        mute_files_exist = mute_files_exist and os.path.exists(mute_f0_path) and os.path.exists(mute_f0nsf_path)

    if mute_files_exist:
        print("Adding mute files to filelist...")
        if if_f0_3:
            for _ in range(2): # Add mute examples twice
                opt.append(
                    f"{mute_wav_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{spk_id5}"
                )
        else:
            for _ in range(2): # Add mute examples twice
                 opt.append(
                    f"{mute_wav_path}|{mute_feature_path}|{spk_id5}"
                 )
    else:
         print("Warning: Mute files not found, skipping addition to filelist.")
         print(f" Expected mute wav: {mute_wav_path}")
         print(f" Expected mute feature: {mute_feature_path}")
         if if_f0_3:
            print(f" Expected mute f0: {mute_f0_path}")
            print(f" Expected mute f0nsf: {mute_f0nsf_path}")


    shuffle(opt)
    filelist_path = "%s/filelist.txt" % exp_dir
    with open(filelist_path, "w") as f:
        f.write("\n".join(opt))
    print(f"Wrote {len(opt)} lines to {filelist_path}")

    # Replace logger.debug, logger.info with print statements
    print("Use gpus:", str(gpus16))
    if not pretrained_G14: # Check if empty or None
        print("No pretrained Generator specified.")
    if not pretrained_D15:
        print("No pretrained Discriminator specified.")

    # Select config path based on version and sample rate
    if version19 == "v1":
        config_path = f"configs/v1/{sr2}.json"
    else: # v2
        config_path = f"configs/v2/{sr2}.json"

    print(f"Using config path: {config_path}")

    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        try:
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(
                    config_data,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
                f.write("\n") # Add newline at the end
            print(f"Saved config to {config_save_path}")
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_path}")
            return f"Error: Base config file not found: {config_path}. Training cannot proceed."
        except Exception as e:
             print(f"Error processing config file {config_path}: {e}")
             return f"Error processing config file {config_path}: {e}"


    # Construct the training command
    cmd = (
        'python infer/modules/train/train.py'
        f' -e "{exp_dir1}"'         # Experiment directory name
        f' -sr {sr2}'              # Sample rate string ("32k", "40k", "48k")
        f' -f0 {1 if if_f0_3 else 0}' # F0 flag
        f' -bs {batch_size12}'     # Batch size
        f' -g {gpus16}'            # GPU ID (usually 0 for Cog)
        f' -te {total_epoch11}'    # Total epochs
        f' -se {save_epoch10}'     # Save frequency
        f' {"-pg "+pretrained_G14 if pretrained_G14 else ""}' # Pretrained Generator
        f' {"-pd "+pretrained_D15 if pretrained_D15 else ""}' # Pretrained Discriminator
        f' -l {1 if if_save_latest13 else 0}' # Save latest flag
        f' -c {1 if if_cache_gpu17 else 0}'   # Cache GPU flag
        f' -sw {1 if if_save_every_weights18 else 0}' # Save every weights flag
        f' -v {version19}'         # RVC version ("v1" or "v2")
    )

    print(f"Starting training command: {cmd}")
    # Use execute_command to run and print output
    output, returncode = execute_command(cmd)

    if returncode == 0:
        return "Training completed successfully. Check logs for details."
    else:
        print(f"Training command failed with return code {returncode}.")
        # The error should already be printed by execute_command
        return "Training failed. Check logs for error details."


def download_weights(url, dest):
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        start = time.time()
        try:
            # Using pget for potentially faster downloads if available and configured in Cog
            subprocess.check_call(["pget", "-f", url, dest], close_fds=False) # -f to overwrite potentially incomplete files
            print(f"Downloaded {dest} in {time.time() - start:.2f} seconds.")
        except FileNotFoundError:
            print("pget not found, falling back to aria2c...")
            try:
                # Fallback using aria2c if pget fails or isn't installed
                subprocess.check_call(
                    ["aria2c", "--console-log-level=warn", "-c", "-x", "16", "-s", "16", "-k", "1M", "-d", dest_dir, "-o", os.path.basename(dest), url],
                    close_fds=False
                )
                print(f"Downloaded {dest} using aria2c in {time.time() - start:.2f} seconds.")
            except FileNotFoundError:
                print("aria2c not found, consider installing it or pget in your Cog environment.")
                # Add a simple Python download fallback if needed, but it's slow for large files
            except subprocess.CalledProcessError as e:
                print(f"aria2c download failed for {url}: {e}")
        except subprocess.CalledProcessError as e:
            print(f"pget download failed for {url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during download of {url}: {e}")

    else:
        print(f"File already exists: {dest}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Loads the model into memory to make running multiple predictions efficient"""
        print("Downloading pretrained models...")
        # Running the downloads in parallel
        with ThreadPoolExecutor(max_workers=4) as executor: # Limit workers slightly
            futures = [
                executor.submit(download_weights, url, dest) for url, dest in downloads
            ]
            # Wait for all downloads to complete
            for future in futures:
                try:
                    future.result()  # Raise exceptions if downloads failed
                except Exception as e:
                    print(f"Error during weight download setup: {e}")
                    # Decide if this should be fatal
        print("Pretrained models downloaded.")

        # Check if essential directories exist, create if not
        os.makedirs("assets/weights", exist_ok=True)
        os.makedirs("logs/mute/0_gt_wavs", exist_ok=True)
        os.makedirs("logs/mute/3_feature256", exist_ok=True)
        os.makedirs("logs/mute/3_feature768", exist_ok=True)
        os.makedirs("logs/mute/2a_f0", exist_ok=True)
        os.makedirs("logs/mute/2b-f0nsf", exist_ok=True)
        # Consider pre-creating mute files here if they are static


    def delete_old_files(self):
        print("Deleting old experiment files...")
        # Delete 'dataset' folder if it exists
        if os.path.exists("dataset"):
            print("Deleting old dataset folder...")
            shutil.rmtree("dataset")

        # Delete 'Model' folder if it exists
        if os.path.exists("Model"):
            print("Deleting old Model folder...")
            shutil.rmtree("Model")

        # Clean assets/weights (trained models)
        weights_dir = "assets/weights"
        if os.path.exists(weights_dir):
             print(f"Cleaning {weights_dir} folder...")
             for filename in os.listdir(weights_dir):
                 file_path = os.path.join(weights_dir, filename)
                 try:
                     if os.path.isfile(file_path) or os.path.islink(file_path):
                         os.unlink(file_path)
                     elif os.path.isdir(file_path):
                         shutil.rmtree(file_path)
                 except Exception as e:
                     print(f'Failed to delete {file_path}. Reason: {e}')


        # Clean logs folder but keep 'mute' directory
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            print("Cleaning logs folder (keeping 'mute')...")
            for filename in os.listdir(logs_dir):
                file_path = os.path.join(logs_dir, filename)
                if filename == "mute":
                    continue  # Skip the 'mute' directory
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        print("Deletion of old files complete.")


    def predict(
        self,
        dataset_zip: CogPath = Input(
            description="Upload dataset zip. Structure: zip -> dataset/<model_name>/<audio_files>.wav"
        ),
        sample_rate: str = Input(
            description="Target sample rate for training.",
            default="48k",
            choices=["32k", "40k", "48k"] # Added 32k option
        ),
        version: str = Input(
            description="RVC Model version.",
            default="v2",
            choices=["v1", "v2"]
        ),
        f0method: str = Input(
            description="F0 extraction method. 'rmvpe_gpu' recommended for speed/quality.",
            default="rmvpe_gpu",
            choices=["pm", "dio", "harvest", "crepe", "rmvpe", "rmvpe_gpu", "fcpe"], # Added crepe, fcpe as common options
        ),
        epoch: int = Input(description="Number of training epochs.", default=20, ge=1), # Increased default, set minimum
        batch_size: int = Input(description="Batch size per GPU.", default=7, ge=1), # Changed to int, added minimum
        save_frequency: int = Input(description="Save checkpoint frequency in epochs. 0 to disable intermediate saves.", default=10, ge=0), # Added option to disable
        cache_gpu: bool = Input(description="Cache training data to GPU VRAM (faster training, more VRAM usage).", default=True),

    ) -> CogPath:
        """Runs the RVC training process."""
        self.delete_old_files() # Clean up before starting

        print("--- Step 0: Setup ---")
        start_time = time.time()

        # --- Unzip Dataset ---
        dataset_path_zip = str(dataset_zip)
        print(f"Extracting dataset from: {dataset_path_zip}")
        extract_path = "." # Extract to current directory
        with zipfile.ZipFile(dataset_path_zip, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extracted.")

        # --- Infer Model Name and Set Paths ---
        model_name = infer_folder_name("dataset")
        if not model_name:
             raise ValueError("Could not find model directory inside 'dataset' folder after unzipping. Expected structure: dataset/<model_name>/")
        print(f"Inferred model name: {model_name}")

        # Convert sample rate string to integer and back for consistency
        sr_map = {"32k": "32000", "40k": "40000", "48k": "48000"}
        sr_int = sr_map.get(sample_rate)
        if not sr_int:
            raise ValueError(f"Invalid sample rate selected: {sample_rate}")
        sr_str = sample_rate # Keep the "32k", "40k", "48k" string

        dataset_dir = f"dataset/{model_name}"
        exp_dir = model_name # Use model name as experiment directory name

        # --- Create Log Directories ---
        log_preprocess_dir = f"logs/{exp_dir}"
        log_feature_dir = f"logs/{exp_dir}"
        os.makedirs(log_preprocess_dir, exist_ok=True)
        os.makedirs(log_feature_dir, exist_ok=True)
        # Initialize log files (optional, commands will append)
        with open(f"{log_preprocess_dir}/preprocess.log", "w") as f: f.write("Preprocessing Log\n")
        with open(f"{log_feature_dir}/extract_f0_feature.log", "w") as f: f.write("F0/Feature Extraction Log\n")
        print(f"Log directories created/verified for experiment: {exp_dir}")

        print("\n--- Step 1: Preprocess Dataset ---")
        # Preprocessing command using the integer sample rate
        num_processes = max(1, os.cpu_count() // 2) # Use half CPU cores for preprocessing
        command_preprocess = f"python infer/modules/train/preprocess.py '{dataset_dir}' {sr_int} {num_processes} './logs/{exp_dir}' False 3.0"
        output, returncode = execute_command(command_preprocess)
        if returncode != 0:
            raise RuntimeError(f"Dataset preprocessing failed. Check logs in {log_preprocess_dir}/preprocess.log. Error: {output}")

        print("\n--- Step 2: Feature Extraction ---")
        # --- F0 Extraction ---
        print(f"Extracting F0 using: {f0method}")
        if f0method == "rmvpe_gpu":
            # Assumes GPU 0 is available
            command_f0 = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 './logs/{exp_dir}' True"
        elif f0method == "crepe":
             # Crepe extraction command (adjust n_cpu and gpu_idx as needed)
             command_f0 = f"python infer/modules/train/extract/extract_f0_crepe.py './logs/{exp_dir}' {num_processes} 8" # 8 is hop length default
        elif f0method == "fcpe":
             # FCPE extraction command (adjust n_cpu and gpu_idx as needed)
             command_f0 = f"python infer/modules/train/extract/extract_f0_fcpe.py 1 0 0 './logs/{exp_dir}'" # Assumes GPU 0
        else: # pm, dio, harvest, rmvpe (CPU versions)
            command_f0 = f"python infer/modules/train/extract/extract_f0_print.py './logs/{exp_dir}' {num_processes} {f0method}"
        output, returncode = execute_command(command_f0)
        if returncode != 0:
            raise RuntimeError(f"F0 extraction failed. Check logs in {log_feature_dir}/extract_f0_feature.log. Error: {output}")

        # --- Hubert Feature Extraction ---
        print(f"Extracting Hubert features (Version: {version})")
        # Assumes GPU 0 is available. device needs to be cuda:0 format
        command_feature = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 './logs/{exp_dir}' '{version}'"
        output, returncode = execute_command(command_feature)
        if returncode != 0:
            raise RuntimeError(f"Feature extraction failed. Check logs in {log_feature_dir}/extract_f0_feature.log. Error: {output}")

        print("\n--- Step 3: Train Index ---")
        index_train_generator = train_index(exp_dir, version)
        try:
            for status in index_train_generator:
                print(status) # Print progress from the generator
        except Exception as e:
             raise RuntimeError(f"Index training failed: {e}")


        print("\n--- Step 4: Train Model ---")
        # Select appropriate pretrained models based on version and sample rate (using sr_str "32k", "40k", "48k")
        pretrained_paths = {
            "v1": {
                "32k": ("assets/pretrained/f0G32k.pth", "assets/pretrained/f0D32k.pth"),
                "40k": ("assets/pretrained/f0G40k.pth", "assets/pretrained/f0D40k.pth"),
                "48k": ("assets/pretrained/f0G48k.pth", "assets/pretrained/f0D48k.pth"),
            },
            "v2": {
                "32k": ("assets/pretrained_v2/f0G32k.pth", "assets/pretrained_v2/f0G32k.pth"), # Corrected D path
                "40k": ("assets/pretrained_v2/f0G40k.pth", "assets/pretrained_v2/f0D40k.pth"),
                "48k": ("assets/pretrained_v2/f0G48k.pth", "assets/pretrained_v2/f0D48k.pth"),
            },
        }
        try:
            G_path, D_path = pretrained_paths[version][sr_str]
            print(f"Using pretrained models: G={G_path}, D={D_path}")
        except KeyError:
             raise ValueError(f"Could not find pretrained model paths for version={version}, sample_rate={sr_str}")

        # Check if pretrained files actually exist
        if not os.path.exists(G_path):
             raise FileNotFoundError(f"Pretrained Generator model not found: {G_path}")
        if not os.path.exists(D_path):
             raise FileNotFoundError(f"Pretrained Discriminator model not found: {D_path}")

        # Call the training function
        train_result = click_train(
            exp_dir1=exp_dir,
            sr2=sr_str,             # Pass sample rate string ("32k", etc.)
            if_f0_3=True,           # Always use F0 for training in this setup
            spk_id5=0,              # Speaker ID (assuming single speaker dataset)
            save_epoch10=save_frequency,
            total_epoch11=epoch,
            batch_size12=str(batch_size), # click_train expects string batch size
            if_save_latest13=True,  # Always save the latest epoch model
            pretrained_G14=G_path,
            pretrained_D15=D_path,
            gpus16=0,               # Use GPU 0
            if_cache_gpu17=cache_gpu,
            if_save_every_weights18=False, # Don't save every single weight usually
            version19=version,
        )
        print(f"Training function returned: {train_result}")
        # Check if training failed based on the return message (could be more robust)
        if "fail" in train_result.lower() or "error" in train_result.lower():
             raise RuntimeError(f"Model training failed. Check logs. Message: {train_result}")


        print("\n--- Step 5: Packaging Model ---")
        output_base_dir = f"Model/{exp_dir}"
        os.makedirs(output_base_dir, exist_ok=True)
        print(f"Created output directory: {output_base_dir}")

        # Find the latest saved model .pth file
        # Model files are saved in assets/weights/ by train.py
        # NOTE: train.py saves weights to `assets/weights/{exp_dir}_e{epoch}_s{step}.pth`
        # We need the *final* one. Or the one matching the last epoch.
        # Let's find the highest epoch number.
        weight_files = glob.glob(f"assets/weights/{exp_dir}_e{epoch}_*.pth")
        if not weight_files:
             # Fallback: Check for the latest file saved by -l 1 flag (G_{exp_dir}.pth)
             latest_g_path = f"assets/weights/G_{exp_dir}.pth"
             if os.path.exists(latest_g_path):
                  weight_files = [latest_g_path]
             else:
                  # Fallback search in logs dir if save structure is different
                  weight_files = glob.glob(f"logs/{exp_dir}/G_*.pth")
                  if not weight_files:
                       raise FileNotFoundError(f"Could not find trained model (.pth file) for epoch {epoch} in assets/weights/ or logs/{exp_dir}. Training might have failed to save.")

        # Select the last weight file found (usually the highest step if multiple exist for the target epoch)
        trained_model_path = sorted(weight_files)[-1]
        print(f"Found trained model: {trained_model_path}")
        final_model_name_in_zip = f"{exp_dir}.pth" # Standardize name in output zip
        shutil.copy(trained_model_path, os.path.join(output_base_dir, final_model_name_in_zip))
        print(f"Copied model to {output_base_dir}/{final_model_name_in_zip}")


        # --- Copy Index Files ---
        index_files = glob.glob(f"logs/{exp_dir}/added_*.index")
        if not index_files:
             raise FileNotFoundError(f"Could not find added index file (added_*.index) in logs/{exp_dir}")
        index_file_to_copy = index_files[0] # Should only be one 'added' index
        shutil.copy(index_file_to_copy, output_base_dir)
        print(f"Copied index file: {os.path.basename(index_file_to_copy)} to {output_base_dir}")

        # Copy total_fea.npy (optional but sometimes useful)
        total_fea_path = f"logs/{exp_dir}/total_fea.npy"
        if os.path.exists(total_fea_path):
             shutil.copy(total_fea_path, output_base_dir)
             print(f"Copied total_fea.npy to {output_base_dir}")


        # --- Create Zip File ---
        zip_file_path = f"{output_base_dir}.zip" # Place zip outside the folder it zips
        print(f"Creating zip file: {zip_file_path}")

        with ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
             # Add model .pth file
             model_arcname = final_model_name_in_zip
             zipf.write(os.path.join(output_base_dir, final_model_name_in_zip), arcname=model_arcname)
             print(f"Adding to zip: {model_arcname}")

             # Add index file
             index_arcname = os.path.basename(index_file_to_copy)
             zipf.write(os.path.join(output_base_dir, index_arcname), arcname=index_arcname)
             print(f"Adding to zip: {index_arcname}")

             # Add total_fea.npy if it exists
             if os.path.exists(os.path.join(output_base_dir, "total_fea.npy")):
                  fea_arcname = "total_fea.npy"
                  zipf.write(os.path.join(output_base_dir, fea_arcname), arcname=fea_arcname)
                  print(f"Adding to zip: {fea_arcname}")

        total_time = time.time() - start_time
        print(f"--- Training and Packaging Complete ---")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Output zip file created at: {zip_file_path}")

        return CogPath(zip_file_path)
