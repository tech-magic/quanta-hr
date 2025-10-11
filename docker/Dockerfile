ARG TARGETPLATFORM
FROM --platform=$TARGETPLATFORM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TERRAFORM_VERSION=1.9.5
ENV ANSIBLE_VERSION=9.5.1
ENV DISABLE_STRICT_MODE=false

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    netcat-openbsd \
    jq \
    unzip \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    unzip \
    sshpass \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install architecture-specific Terraform
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
      wget https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip; \
    else \
      wget https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_arm64.zip; \
    fi \
 && unzip terraform_*.zip \
 && mv terraform /usr/local/bin/terraform \
 && rm terraform_*.zip

# Install Ansible
RUN pip3 install ansible==${ANSIBLE_VERSION} boto3 botocore

COPY ./sanitize_blob_name.py /sanitize_blob_name.py
RUN chmod +x sanitize_blob_name.py

# Provide permissions to entrypoint script
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
