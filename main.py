import transcribe
import upload

uri = upload.upload_blob('test_bucket_cs206', 'test.flac', 'test.flac')
print(uri)
transcribe.transcribe_gcs(uri)
