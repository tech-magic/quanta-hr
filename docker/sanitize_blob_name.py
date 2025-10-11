#!/usr/bin/env python3
import re
import sys
import time

def sanitize_bucket_name(name: str) -> str:
    # 1. Lowercase
    bucket = name.lower()

    # 2. Replace invalid characters with '-'
    bucket = re.sub(r'[^a-z0-9-]', '-', bucket)

    # 3. Remove leading and trailing '-'
    bucket = bucket.strip('-')

    # 4. Collapse multiple consecutive '-' into one
    bucket = re.sub(r'-+', '-', bucket)

    # 5. Enforce length rules (an S3 bucket can have upto 63 chars, leaving a safe buffer at 50)
    if len(bucket) < 3:
        bucket = f"{bucket}{'x' * (3 - len(bucket))}"
    elif len(bucket) > 50:
        bucket = bucket[:50].rstrip('-')

    # 6. Fallback if empty after sanitization
    if not bucket:
        bucket = f"bucket-{int(time.time())}"

    return bucket

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sanitize_bucket_name.py <string>", file=sys.stderr)
        sys.exit(1)

    input_str = sys.argv[1]
    print(sanitize_bucket_name(input_str))
