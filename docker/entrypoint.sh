#!/bin/bash
if [ "$DISABLE_STRICT_MODE" = "true" ]; then
    set +e +u
    set +o pipefail
    echo "Strict mode OFF"
else
    set -e -u -o pipefail
    trap 'echo "❌ Error on line $LINENO: $BASH_COMMAND"' ERR
    echo "Strict mode ON"
fi

echo "Current Working Directory -> $(pwd)"

# -------------------------------
# Workspace directories
# -------------------------------
WORKSPACE_DIR="/workspace"
TERRAFORM_DIR="$WORKSPACE_DIR/iac/terraform"
TERRAFORM_STATE_DIR="$WORKSPACE_DIR/tf_state"
ANSIBLE_DIR="$WORKSPACE_DIR/iac/ansible"
TARGET_DIR="$WORKSPACE_DIR/target"
TF_DATA_DIR="$TARGET_DIR/terraform_data"
LLM_DOWNLOAD_DIR_NAME="$WORKSPACE_DIR/model"

# -------------------------------
# LLM Settings
# -------------------------------

LLM_CONFIG_FILE="$WORKSPACE_DIR/data/config/qlora_config.json"
LLM_SETTINGS_JSON=$(cat "$LLM_CONFIG_FILE")

# -------------------------------
# Creating required directories (only if they don't exist)
# -------------------------------
echo "Creating required directories (only if they don't exist) -> $WORKSPACE_DIR"
mkdir -p "$LLM_DOWNLOAD_DIR_NAME"
mkdir -p "$TARGET_DIR"
mkdir -p "$TERRAFORM_STATE_DIR"
mkdir -p "$TF_DATA_DIR"

# ----------------
# Fetch values required to run Terraform
# ----------------
export TF_DATA="$TF_DATA_DIR"

LLM_MODEL_NAME=$(echo "$LLM_SETTINGS_JSON" | jq -r '.model.name')
COMPUTE_PLATFORM=$(echo "$LLM_SETTINGS_JSON" | jq -r '.compute.platform')
COMPUTE_INSTANCE_TYPE=$(echo "$LLM_SETTINGS_JSON" | jq -r '.compute.instance_type')
COMPUTE_BOOT_OS_QUERY=$(echo "$LLM_SETTINGS_JSON" | jq -r '.compute.boot_os_query')
LLM_STORAGE_BLOB_NAME=$(./sanitize_blob_name.py "$LLM_MODEL_NAME")

# ----------------
# Run Terraform
# ----------------

terraform -chdir="$TERRAFORM_DIR" init -input=false
terraform -chdir="$TERRAFORM_DIR" apply \
  -var="workspace_dir=$WORKSPACE_DIR" \
  -var="llm_model_name=$LLM_MODEL_NAME" \
  -var="llm_storage_blob_name=$LLM_STORAGE_BLOB_NAME" \
  -var="compute_platform=$COMPUTE_PLATFORM" \
  -var="compute_instance_type=$COMPUTE_INSTANCE_TYPE" \
  -var="compute_boot_os_query=$COMPUTE_BOOT_OS_QUERY" \
  -auto-approve -input=false \
  -state="$TERRAFORM_STATE_DIR/terraform.tfstate" \
  -lock=true \
  -lock-timeout=300s

# -------------------------------
# Extract Terraform outputs
# -------------------------------
terraform -chdir="$TERRAFORM_DIR" output \
  -state="$TERRAFORM_STATE_DIR/terraform.tfstate" \
  -json > "$TARGET_DIR/terraform_outputs.json"

# Flatten and save values only
jq 'map_values(.value)' "$TARGET_DIR/terraform_outputs.json" > "$TARGET_DIR/tf_vars.json"

# Optional: Show flattened outputs in console
cat "$TARGET_DIR/tf_vars.json"

# -------------------------------
# Extract Training Instance IP from Terraform JSON
# -------------------------------
TF_VARS_JSON="$TARGET_DIR/tf_vars.json"
TRAINING_INSTANCE_IP=$(jq -r '.instance_ip' "$TF_VARS_JSON")
echo "Training Instance IP -> $TRAINING_INSTANCE_IP"

# -------------------------------
# Extract AWS SSH key from Terraform JSON
# -------------------------------
SSH_PRIVATE_KEY=$(jq -r '.ssh_private_key' "$TF_VARS_JSON")
echo "SSH Private Key File -> $SSH_PRIVATE_KEY"

# -------------------------------
# Generate Ansible Config
# -------------------------------
ANSIBLE_CONFIG_FILE="$TARGET_DIR/ansible.cfg"
cat > "$ANSIBLE_CONFIG_FILE" <<EOF
[defaults]
host_key_checking = False
EOF

export ANSIBLE_CONFIG=$ANSIBLE_CONFIG_FILE
echo "ANSIBLE_CONFIG is set to -> $ANSIBLE_CONFIG"

# -------------------------------
# Generate Ansible Inventories
# -------------------------------
REMOTE_INVENTORY_FILE="$TARGET_DIR/remote_inventory.ini"
cat > "$REMOTE_INVENTORY_FILE" <<EOF
[aws]
aws_instance ansible_host=$TRAINING_INSTANCE_IP ansible_user=ubuntu ansible_ssh_private_key_file=$SSH_PRIVATE_KEY
EOF

echo "Generated remote ansible inventory -> $REMOTE_INVENTORY_FILE"

LOCAL_INVENTORY_FILE="$TARGET_DIR/local_inventory.ini"
cat > "$LOCAL_INVENTORY_FILE" <<EOF
[local]
localhost ansible_connection=local
EOF

echo "Generated local ansible inventory -> $LOCAL_INVENTORY_FILE"

# -------------------------------
# Wait until SSH is available in remote llm training host
# -------------------------------
echo "Checking SSH connectivity to $TRAINING_INSTANCE_IP on port 22..."
MAX_RETRIES=30
SLEEP_SEC=10
for i in $(seq 1 $MAX_RETRIES); do
    if nc -z -w5 "$TRAINING_INSTANCE_IP" 22; then
        echo "SSH port 22 is reachable on attempt $i."
        break
    else
        echo "SSH not available yet (attempt $i/$MAX_RETRIES). Retrying in $SLEEP_SEC seconds..."
        sleep $SLEEP_SEC
    fi

    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "ERROR: SSH port 22 did not become available after $MAX_RETRIES attempts."
        exit 1
    fi
done

# -------------------------------
# Syntax Check Ansible Playbooks
# -------------------------------
ANSIBLE_LOG_REMOTE_GPU="$TARGET_DIR/ansible_remote_gpu.log"
REMOTE_GPU_PLAYBOOK="$ANSIBLE_DIR/remote/remote_gpu_playbook.yml"

echo "Performing syntax checks for $REMOTE_GPU_PLAYBOOK"
ansible-playbook -vvv "$REMOTE_GPU_PLAYBOOK" \
  -i "$REMOTE_INVENTORY_FILE" \
  -e "llm_settings=$LLM_SETTINGS_JSON" \
  --extra-vars "@$TF_VARS_JSON" --syntax-check
echo "Successfully completed syntax checks for $REMOTE_GPU_PLAYBOOK"

ANSIBLE_LOG_LOCAL_CPU="$TARGET_DIR/ansible_local_cpu.log"
LOCAL_CPU_PLAYBOOK="$ANSIBLE_DIR/local/local_cpu_playbook.yml"

echo "Performing syntax checks for $LOCAL_CPU_PLAYBOOK"
ansible-playbook -vvv "$LOCAL_CPU_PLAYBOOK" \
    -i "$LOCAL_INVENTORY_FILE" \
    -e "work_dir=$WORKSPACE_DIR" \
    -e "llm_settings=$LLM_SETTINGS_JSON" \
    --extra-vars "@$TF_VARS_JSON" --syntax-check
echo "Successfully completed syntax checks for $LOCAL_CPU_PLAYBOOK"

# -------------------------------
# Run Ansible Playbook on Remote GPU Host with logging
# -------------------------------
echo "Running remote Ansible playbook, $REMOTE_GPU_PLAYBOOK at remote AWS machine with GPUs."
echo "Remote host execution output is available at -> $ANSIBLE_LOG_REMOTE_GPU"

ansible-playbook -vvv "$REMOTE_GPU_PLAYBOOK" \
  -i "$REMOTE_INVENTORY_FILE" \
  -e "llm_settings=$LLM_SETTINGS_JSON" \
  --extra-vars "@$TF_VARS_JSON" | tee "$ANSIBLE_LOG_REMOTE_GPU"

echo "✅ $REMOTE_GPU_PLAYBOOK completed successfully."

# -------------------------------
# Run Ansible Playbook on Local CPU Host with logging and retry logic
# -------------------------------
echo "Running local Ansible playbook, $LOCAL_CPU_PLAYBOOK at local machine with CPU."
echo "Local host execution output is available at -> $ANSIBLE_LOG_LOCAL_CPU"

ansible-playbook -vvv "$LOCAL_CPU_PLAYBOOK" \
    -i "$LOCAL_INVENTORY_FILE" \
    -e "work_dir=$WORKSPACE_DIR" \
    -e "llm_settings=$LLM_SETTINGS_JSON" \
    --extra-vars "@$TF_VARS_JSON" | tee -a "$ANSIBLE_LOG_LOCAL_CPU"

echo "✅ $LOCAL_CPU_PLAYBOOK completed successfully."

# -------------------------------
# Destroy Terraform resources
# -------------------------------
terraform -chdir="$TERRAFORM_DIR" destroy \
    -var="workspace_dir=$WORKSPACE_DIR" \
    -var="llm_model_name=$LLM_MODEL_NAME" \
    -var="llm_storage_blob_name=$LLM_STORAGE_BLOB_NAME" \
    -var="compute_platform=$COMPUTE_PLATFORM" \
    -var="compute_instance_type=$COMPUTE_INSTANCE_TYPE" \
    -var="compute_boot_os_query=$COMPUTE_BOOT_OS_QUERY" \
    -auto-approve -input=false \
    -state="$TERRAFORM_STATE_DIR/terraform.tfstate" \
    -lock=true \
    -lock-timeout=300s


