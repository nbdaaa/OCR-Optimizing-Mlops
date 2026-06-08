"""
Delete data version(s) from the MinIO ocr-data bucket.

Usage:
    python scripts/delete_version.py v1
    python scripts/delete_version.py v1 v2 v3
    python scripts/delete_version.py --all      # delete every version
"""
import os
import sys

import boto3
from dotenv import load_dotenv

load_dotenv()


def _make_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )


def list_versions(client, bucket: str) -> list[str]:
    """Return all version names (top-level prefixes) in the bucket."""
    resp = client.list_objects_v2(Bucket=bucket, Delimiter="/")
    return [p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])]


def delete_version(client, bucket: str, version: str) -> int:
    """Delete all objects under {version}/ prefix. Returns number deleted."""
    prefix = f"{version}/"
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = resp.get("Contents", [])
    if not objects:
        print(f"  {version}: nothing to delete (prefix '{prefix}' empty)")
        return 0

    client.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
    )
    for obj in objects:
        print(f"    deleted {obj['Key']}")
    return len(objects)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/delete_version.py <version> [<version> ...] | --all")
        sys.exit(1)

    bucket = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")
    client = _make_client()

    if sys.argv[1] == "--all":
        versions = list_versions(client, bucket)
        if not versions:
            print("No versions found. Nothing to delete.")
            return
        print(f"Deleting ALL {len(versions)} version(s): {', '.join(versions)}")
    else:
        versions = sys.argv[1:]

    total = 0
    for version in versions:
        print(f"[{version}]")
        total += delete_version(client, bucket, version)

    print(f"\nDone — {total} object(s) deleted.")


if __name__ == "__main__":
    main()
