import os
import re
import shutil
from pathlib import Path
from datetime import datetime, timezone

from botocore.session import get_session

from minio import Minio
from transformers import TrainerCallback

# -------------------------------------------
# VALIDATING DIRECTORY NAMES
# -------------------------------------------
def validated_dir_name(text):
    return re.sub(r'[^A-Za-z0-9]', '_', text)


# -------------------------------------------
# CREATE MINIO CLIENT (AWS IAM / aws configure)
# -------------------------------------------
def get_minio_client():
    endpoint = os.getenv("S3_ENDPOINT", "s3.amazonaws.com")
    secure = True

    # Use botocore to load credentials from AWS CLI config or IAM role
    session = get_session()
    creds = session.get_credentials().get_frozen_credentials()

    client = Minio(
        endpoint,
        access_key=creds.access_key,
        secret_key=creds.secret_key,
        session_token=creds.token,
        secure=secure
    )
    return client

# -------------------------------------------
# UPLOAD DIRECTORY TO BLOB STORAGE
# -------------------------------------------
def upload_final_results(local_dir, blob_name, blob_prefix):
    client = get_minio_client()
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{blob_prefix}/{rel_path}".replace("\\", "/")
            client.fput_object(blob_name, blob_path, local_path)
            print(f"Uploaded {local_path} → s3://{blob_name}/{blob_path}")


# -------------------------------------------
# DOWNLOAD LATEST CHECKPOINT FROM BLOB STORAGE
# -------------------------------------------
def download_latest_s3_checkpoint(blob_name, checkpoint_uploads_dir, local_checkpoint_base_dir):
    client = get_minio_client()
    objects = list(client.list_objects(blob_name, prefix=checkpoint_uploads_dir, recursive=True))

    checkpoints = sorted(
        [obj.object_name for obj in objects if "checkpoint-" in obj.object_name],
        key=lambda k: int(k.split("checkpoint-")[-1].split("/")[0])
    )
    if not checkpoints:
        return None

    latest = checkpoints[-1].split("/")[1]
    local_ckpt_dir = os.path.join(local_checkpoint_base_dir, latest)
    os.makedirs(local_ckpt_dir, exist_ok=True)

    print(f"Downloading latest checkpoint: {latest}")
    for obj in objects:
        if latest in obj.object_name:
            rel_path = os.path.relpath(obj.object_name, checkpoint_uploads_dir)
            local_path = os.path.join(local_checkpoint_base_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            client.fget_object(blob_name, obj.object_name, local_path)
    return local_ckpt_dir


# -------------------------------------------
# GET LATEST LOCAL CHECKPOINT
# -------------------------------------------
def get_latest_local_checkpoint(local_checkpoint_base_dir):
    checkpoints = sorted(Path(local_checkpoint_base_dir).glob("checkpoint-*"), key=os.path.getmtime)
    return str(checkpoints[-1]) if checkpoints else None


# -------------------------------------------
# DOWNLOAD S3 DIRECTORY IF CHANGED LOCALLY
# -------------------------------------------
def download_s3_dir_if_changed(blob_name, blob_prefix, local_dir):
    client = get_minio_client()
    for obj in client.list_objects(blob_name, prefix=blob_prefix, recursive=True):
        blob_mod_time = obj.last_modified
        rel_path = os.path.relpath(obj.object_name, blob_prefix)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path):
            local_mod_time = datetime.fromtimestamp(os.path.getmtime(local_path), tz=timezone.utc)
            if local_mod_time >= blob_mod_time:
                continue

        print(f"⬇️ Downloading s3://{blob_name}/{obj.object_name} → {local_path}")
        client.fget_object(blob_name, obj.object_name, local_path)


# -------------------------------------------
# TRAINER CALLBACK TO UPLOAD CHECKPOINTS
# -------------------------------------------
class LLMTrainingCheckpointCallback(TrainerCallback):
    def __init__(self, blob_name, blob_prefix, local_checkpoint_base_dir):
        self.blob_name = blob_name
        self.blob_prefix = blob_prefix
        self.client = get_minio_client()
        self.local_checkpoint_base_dir = local_checkpoint_base_dir

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(checkpoint_dir):
            return control

        print(f"Uploading checkpoint {checkpoint_dir} to Blob Storage...")
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, checkpoint_dir)
                blob_path = f"{self.blob_prefix}/{os.path.basename(checkpoint_dir)}/{rel_path}".replace("\\", "/")
                self.client.fput_object(self.blob_name, blob_path, local_path)
                print(f"Uploaded {local_path} → s3://{self.blob_name}/{blob_path}")

        # Move checkpoint locally to keep directory clean
        local_checkpoint_dir = os.path.join(self.local_checkpoint_base_dir, f"checkpoint-{state.global_step}")
        shutil.move(checkpoint_dir, local_checkpoint_dir)
        print(f"🚚 Moved checkpoint from: {checkpoint_dir} to: {local_checkpoint_dir}")
        return control
