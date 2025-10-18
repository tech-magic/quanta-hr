# -------------------------------
# Assumptions:
# You've installed aws cli in your local and configured your AWS credentials using below commands.

# aws configure
# -------------------------------

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  required_version = ">= 1.5.0"
}

provider "aws" {
  region = "us-east-1"
}

# -------------------------------
# Pass the variable values from the command line
# -------------------------------
variable "workspace_dir" {
  type = string
}

variable "llm_model_name" {
  type = string
}

variable "llm_storage_blob_name" {
  type = string
}

variable "compute_platform" {
  type = string
}

variable "compute_instance_type" {
  type = string
}

variable "compute_boot_os_query" {
  type = string
}

# -------------------------------
# Generate a random integer (for unique naming)
# -------------------------------
resource "random_integer" "unique_id" {
  min = 10000
  max = 99999
}

# -------------------------------
# Declare Local Variables
# -------------------------------

locals {
  resource_tag_name = "${var.llm_model_name}-${random_integer.unique_id.result}"
  resource_tag_env = "QLoRA_Training"
  gpu_instance_type = var.compute_instance_type
}

# -------------------------------
# Get your public internet IP dynamically
# -------------------------------
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com"
}

# ----------------------------------------------------------------------
# Query for the latest AMI with Ubuntu + Deep Learning + GPU + PyTorch
# ----------------------------------------------------------------------
data "aws_ami" "deep_learning_ami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.compute_boot_os_query]
  }
}

# -------------------------------
# Generate AWS KeyPair
# -------------------------------

# Generate SSH key
resource "tls_private_key" "gpu_tls_private_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

# Register public key in AWS
resource "aws_key_pair" "gpu_aws_key_pair" {
  key_name   = "gpu_tls_private_key_${random_integer.unique_id.result}"
  public_key = tls_private_key.gpu_tls_private_key.public_key_openssh

  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

# Save private key locally
resource "local_file" "gpu_private_key_pem" {
  content         = tls_private_key.gpu_tls_private_key.private_key_pem
  filename        = "/workspace/target/aws-gpu-private-key-${random_integer.unique_id.result}.pem"
  file_permission = "0600"
}

# -------------------------------
# Declare Derived Local Variables
# -------------------------------

locals {
  my_ip_cidr = "${chomp(data.http.my_ip.body)}/32"
  deep_learning_ami = data.aws_ami.deep_learning_ami.id
  gpu_aws_key_pair_name = aws_key_pair.gpu_aws_key_pair.key_name
}

# -------------------------------
# S3 Bucket to store training data + final LLM model
# -------------------------------
resource "aws_s3_bucket" "llm_qlora_bucket" {
  bucket = "${var.llm_storage_blob_name}-${random_integer.unique_id.result}"
  acl    = "private"
  force_destroy = true
  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

resource "aws_s3_object" "llm_training_data" {
  for_each = fileset("${var.workspace_dir}/data", "**")

  bucket = aws_s3_bucket.llm_qlora_bucket.bucket
  key    = "data/${each.value}"
  source = "${var.workspace_dir}/data/${each.value}"
  etag   = filemd5("${var.workspace_dir}/data/${each.value}")
}

resource "aws_s3_object" "llm_training_config" {
  for_each = fileset("${var.workspace_dir}/config", "**")

  bucket = aws_s3_bucket.llm_qlora_bucket.bucket
  key    = "config/${each.value}"
  source = "${var.workspace_dir}/config/${each.value}"
  etag   = filemd5("${var.workspace_dir}/config/${each.value}")
}

resource "aws_s3_object" "llm_training_apps" {
  for_each = fileset("${var.workspace_dir}/apps", "**")

  bucket = aws_s3_bucket.llm_qlora_bucket.bucket
  key    = "apps/${each.value}"
  source = "${var.workspace_dir}/apps/${each.value}"
  etag   = filemd5("${var.workspace_dir}/apps/${each.value}")
}

# -------------------------------
# IAM Role and Policy for EC2
# -------------------------------
resource "aws_iam_role" "ec2_role" {
  name = "ec2-qlora-role-${random_integer.unique_id.result}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })

  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

resource "aws_iam_policy" "s3_access_policy" {
  name        = "ec2-s3-access-${random_integer.unique_id.result}"
  description = "Allow EC2 to read/write to S3 bucket"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.llm_qlora_bucket.arn,
          "${aws_s3_bucket.llm_qlora_bucket.arn}/*"
        ]
      }
    ]
  })

  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

resource "aws_iam_role_policy_attachment" "attach_policy" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

# -------------------------------
# Security Group (SSH restricted)
# -------------------------------
resource "aws_security_group" "gpu_sg" {
  name        = "gpu-sg-${random_integer.unique_id.result}"
  description = "Allow SSH from my IP only"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [local.my_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

# -------------------------------
# EC2 Instance for Training
# -------------------------------
resource "aws_instance" "gpu_instance" {
  ami                    = local.deep_learning_ami
  instance_type          = local.gpu_instance_type
  key_name               = local.gpu_aws_key_pair_name
  vpc_security_group_ids = [aws_security_group.gpu_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_size = 100    # GB, increase as needed
    volume_type = "gp3"
  }

  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "ec2_profile" {
  name = "ec2-qlora-profile-${random_integer.unique_id.result}"
  role = aws_iam_role.ec2_role.name

  tags = {
    Name        = local.resource_tag_name
    Environment = local.resource_tag_env
  }
}

# -------------------------------
# Outputs
# -------------------------------
output "instance_ip" {
  value = aws_instance.gpu_instance.public_ip
}

output "generated_s3_bucket" {
  value = aws_s3_bucket.llm_qlora_bucket.bucket
}

output "instance_id" {
  value = aws_instance.gpu_instance.id
}

output "my_ip_cidr" {
  value = local.my_ip_cidr
}

output "ssh_private_key" {
  value = local_file.gpu_private_key_pem.filename
}

output "workspace_dir" {
  value = var.workspace_dir
}

