from google.cloud import storage
from google.oauth2 import service_account
import os

# Load credentials directly from file
credentials = service_account.Credentials.from_service_account_file(
    "./protonx-evaluation-b3a820c665ae.json"
)

# Initialize client with explicit credentials
client = storage.Client(credentials=credentials)

# Download
bucket_name = "protonx-ocr-data"
prefix = "data_part_1"
local_dir = "./"
os.makedirs(local_dir, exist_ok=True)

bucket = client.bucket(bucket_name)
for blob in bucket.list_blobs(prefix=prefix):
    filename = os.path.basename(blob.name)
    local_path = os.path.join(local_dir, filename)
    blob.download_to_filename(local_path)
    print(f"Downloaded: {blob.name} → {local_path}")