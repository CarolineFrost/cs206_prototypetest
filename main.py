import transcribe
import upload

uri = upload.upload_blob('audio-bucket-206', 'test_interview.flac', 'test_interview.flac')
print(uri)
transcribe.transcribe_gcs(uri)
