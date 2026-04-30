from google.cloud import storage
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    "./protonx-evaluation-b3a820c665ae.json"
)

client = storage.Client(credentials=credentials)

bucket = client.bucket("protonx-ocr-data")
blobs = list(bucket.list_blobs(prefix="data_part_1"))

print(f"Tổng số files: {len(blobs)}")