# Azure Speech transcription helper

`transcribe.py` uploads audio to Azure Storage and uses Azure Speech-to-Text batch transcription with speaker attribution, emitting both normalized JSON and optional WebVTT.

## Setup

1. Create (or reuse) an Azure Speech resource and note its key + region.
2. (Optional) create a `.env` file so both scripts can pick up credentials automatically:

   ```bash
   cat <<'EOF' > .env
   AZURE_SPEECH_KEY="<speech key>"
   AZURE_SPEECH_REGION="<region>"
   AZURE_STORAGE_CONNECTION_STRING="<storage connection string>"
   EOF
   ```

3. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Ensure `ffmpeg`/`ffprobe` are available on your PATH so the script can auto-convert non-compliant audio/video (e.g., `brew install ffmpeg`).

## Transcription CLI (`transcribe.py`)

Use this when you need Azure's batch speaker-attribution service to process longer files or rely on server-side speaker metadata. Any input that isn't already PCM 16 kHz mono WAV (MP3s, AAC podcasts, MP4 videos, etc.) is automatically converted via `ffmpeg` before upload.

Extra requirement: set `AZURE_STORAGE_CONNECTION_STRING` (with `AccountName` and `AccountKey`) so the script can create a temporary blob container for uploads/results.

Example:

```bash
python transcribe.py --file path/to/audio.wav \
  --languages en-US pt-PT \
  --output batch-transcript.json \
  --vtt-output batch-transcript.vtt \
  -vv
```

Notable flags:

- `--min-speakers` / `--max-speakers`: hint Azure's speaker-attribution engine about the number of voices to expect (defaults: 2-10).
- `--sas-hours`: how long upload/download SAS URLs remain valid (default 2h).
- `--keep-artifacts`: retain the temporary storage container for troubleshooting.

### Quick JSON → VTT conversion

If Azure already produced a transcription JSON (e.g., downloaded from the portal), convert it locally without hitting the API:

```bash
python transcribe.py --json-input contenturl_0.json \
  --output segments.json \
  --vtt-output contenturl_0.vtt
```

In this mode no credentials are required; the script simply normalizes the JSON into the same segment format and emits WebVTT.
