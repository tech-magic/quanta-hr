import os
from pathlib import Path
import boto3
import re
import shutil

from datetime import datetime, timezone

from transformers import TrainerCallback

# -----------
# VALIDATING DIRECTORY NAMES
# -----------
def validated_dir_name(text):
    # Replace all non-alphanumeric characters with underscores
    return re.sub(r'[^A-Za-z0-9]', '_', text)

# -----------
# UPLOAD/DOWNLOAD TO/FROM S3
# -----------

def upload_final_results(local_dir, bucket_name, s3_prefix):
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_path = f"{s3_prefix}/{rel_path}"
            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} → s3://{bucket_name}/{s3_path}")

def download_latest_s3_checkpoint(bucket, checkpoint_uploads_dir, local_checkpoint_base_dir):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=checkpoint_uploads_dir)
    if "Contents" not in response:
        return None

    checkpoints = sorted(
        [obj["Key"] for obj in response["Contents"] if "checkpoint-" in obj["Key"]],
        key=lambda k: int(k.split("checkpoint-")[-1].split("/")[0])
    )
    if not checkpoints:
        return None

    latest = checkpoints[-1].split("/")[1]
    local_ckpt_dir = os.path.join(local_checkpoint_base_dir, latest)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    print(f"Downloading latest S3 checkpoint: {latest}")
    for obj in [o for o in response["Contents"] if latest in o["Key"]]:
        local_path = os.path.join(local_checkpoint_base_dir, os.path.relpath(obj["Key"], checkpoint_uploads_dir))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, obj["Key"], local_path)
    return local_ckpt_dir

def get_latest_local_checkpoint(local_checkpoint_base_dir):
    checkpoints = sorted(Path(local_checkpoint_base_dir).glob("checkpoint-*"), key=os.path.getmtime)
    return str(checkpoints[-1]) if checkpoints else None

def download_s3_dir_if_changed(bucket_name, s3_prefix, local_dir):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            s3_mod_time = obj["LastModified"]
            rel_path = os.path.relpath(key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path):
                local_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
                if local_mod_time >= s3_mod_time:
                    continue
            print(f"⬇️ Downloading s3://{bucket_name}/{key} → {local_path}")
            s3.download_file(bucket_name, key, local_path)

# ---------------------------
# HANDLE LLM TRAINING CHECKPOINTS
# ---------------------------
class LLMTrainingCheckpointCallback(TrainerCallback):
    def __init__(self, bucket_name, s3_prefix, local_checkpoint_base_dir):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client("s3")
        self.local_checkpoint_base_dir = local_checkpoint_base_dir

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            return control
        print(f"Uploading checkpoint {checkpoint_dir} to S3...")
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, checkpoint_dir)
                s3_path = f"{self.s3_prefix}/{os.path.basename(checkpoint_dir)}/{rel_path}"
                self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
                print(f"Uploaded {local_path} → s3://{self.bucket_name}/{s3_path}")

        local_checkpoint_dir = os.path.join(self.local_checkpoint_base_dir, f"checkpoint-{state.global_step}")
        shutil.move(checkpoint_dir, local_checkpoint_dir)
        print(f"🚚 Moved checkpoint from: {checkpoint_dir} to: {local_checkpoint_dir}")