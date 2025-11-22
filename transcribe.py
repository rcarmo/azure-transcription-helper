#!/usr/bin/env python3
"""Batch Azure Speech diarization via Speech-to-Text v3.1 API.

Workflow:
1. Upload the audio file to a temporary blob container (using AZURE_STORAGE_CONNECTION_STRING).
2. Submit a batch transcription job with diarization enabled.
3. Poll until the job completes, download the transcription JSON, and emit JSON/VTT locally.
4. Clean up the temporary container/blobs unless --keep-artifacts is set.

Environment variables (via .env or shell):
    AZURE_SPEECH_KEY
    AZURE_SPEECH_REGION
    AZURE_STORAGE_CONNECTION_STRING
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import subprocess
import tempfile
from urllib.parse import urlparse
from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    ContainerSasPermissions,
    generate_blob_sas,
    generate_container_sas,
)
from dotenv import load_dotenv

load_dotenv()

ISO_DURATION = re.compile(r"PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?")


@dataclass
class StorageContext:
    blob_service: BlobServiceClient
    account_name: str
    account_key: str
    container_name: str
    container_url: str


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Azure Speech diarization helper")
    parser.add_argument(
        "--file",
        help="Path to local audio/video file (required unless --json-input is used)",
    )
    parser.add_argument("--output", "-o", help="Path to write JSON result (default stdout)")
    parser.add_argument("--vtt-output", help="Optional WebVTT output path")
    parser.add_argument(
        "--json-input",
        help="Convert an existing Azure transcription JSON file directly to JSON/VTT outputs",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-US"],
        help="Locale(s) to consider. Multiple entries enable automatic language identification",
    )
    parser.add_argument(
        "--job-timeout",
        type=int,
        default=3600,
        help="Max seconds to wait for batch job completion (default 3600)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Seconds between polling the transcription job (default 15)",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Do not delete the temporary blob container and uploads",
    )
    parser.add_argument(
        "--sas-hours",
        type=int,
        default=2,
        help="Validity (hours) for generated SAS URLs (default 2)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=10,
        help="Upper bound of distinct speakers to look for (default 10)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=2,
        help="Lower bound of distinct speakers to look for (default 2)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (repeat for debug)",
    )
    args = parser.parse_args(argv)
    if not args.file and not args.json_input:
        parser.error("Either --file or --json-input must be provided")
    return args


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(message)s")


def parse_storage_connection_string(conn_str: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for chunk in conn_str.split(";"):
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        parts[k] = v
    required = {"AccountName", "AccountKey"}
    missing = required - parts.keys()
    if missing:
        raise ValueError(f"Storage connection string missing fields: {', '.join(missing)}")
    return parts


def create_storage_context(conn_str: str) -> StorageContext:
    parts = parse_storage_connection_string(conn_str)
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_name = f"diarize-{uuid.uuid4().hex}"[:63]
    container_client = blob_service.create_container(container_name)
    logging.info("Created temporary container %s", container_name)
    return StorageContext(
        blob_service=blob_service,
        account_name=parts["AccountName"],
        account_key=parts["AccountKey"],
        container_name=container_name,
        container_url=container_client.url,
    )


def upload_audio(ctx: StorageContext, local_path: Path) -> str:
    blob_client = ctx.blob_service.get_blob_client(ctx.container_name, local_path.name)
    with open(local_path, "rb") as fh:
        blob_client.upload_blob(fh, overwrite=True)
    logging.info("Uploaded %s to blob %s", local_path, blob_client.blob_name)
    return blob_client.blob_name


def _probe_audio_metadata(path: Path) -> tuple[Optional[str], Optional[int], Optional[int]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logging.debug("ffprobe failed for %s: %s", path, exc)
        return None, None, None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        return None, None, None
    codec = lines[0]
    try:
        sample_rate = int(float(lines[1]))
    except ValueError:
        sample_rate = None
    try:
        channels = int(float(lines[2]))
    except ValueError:
        channels = None
    return codec, sample_rate, channels


def ensure_wav_for_upload(source: Path) -> tuple[Path, Optional[Path]]:
    """Return a WAV path suitable for batch upload, converting via ffmpeg if needed."""

    codec, sample_rate, channels = _probe_audio_metadata(source)
    if (
        source.suffix.lower() == ".wav"
        and codec == "pcm_s16le"
        and sample_rate == 16000
        and channels == 1
    ):
        logging.debug("Audio already PCM 16 kHz mono WAV; skipping conversion")
        return source, None

    fd, temp_name = tempfile.mkstemp(suffix=".wav", prefix="transcribe_")
    os.close(fd)
    temp_path = Path(temp_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ]
    logging.info("Converting %s to PCM WAV via ffmpeg", source)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg conversion failed for {source}: {exc}") from exc

    return temp_path, temp_path


def build_sas_urls(ctx: StorageContext, blob_name: str, hours: int) -> Dict[str, str]:
    expiry = datetime.now(timezone.utc) + timedelta(hours=hours)
    blob_sas = generate_blob_sas(
        account_name=ctx.account_name,
        container_name=ctx.container_name,
        blob_name=blob_name,
        account_key=ctx.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    container_sas = generate_container_sas(
        account_name=ctx.account_name,
        container_name=ctx.container_name,
        account_key=ctx.account_key,
        permission=ContainerSasPermissions(read=True, write=True, list=True, add=True, create=True),
        expiry=expiry,
    )
    input_url = f"{ctx.container_url}/{blob_name}?{blob_sas}"
    destination_url = f"{ctx.container_url}?{container_sas}"
    return {"input": input_url, "destination": destination_url}


def submit_transcription(
    speech_region: str,
    speech_key: str,
    content_url: str,
    destination_url: str,
    languages: List[str],
    max_speakers: Optional[int],
    min_speakers: Optional[int],
) -> str:
    endpoint = f"https://{speech_region}.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions"
    headers = {"Ocp-Apim-Subscription-Key": speech_key, "Content-Type": "application/json"}
    properties: Dict[str, Any] = {
        "diarizationEnabled": True,
        "wordLevelTimestampsEnabled": True,
        "punctuationMode": "DictatedAndAutomatic",
        "profanityFilterMode": "Masked",
        "destinationContainerUrl": destination_url,
    }
    if max_speakers and max_speakers > 0 or (min_speakers and min_speakers > 0):
        speakers_payload: Dict[str, int] = {}
        if max_speakers and max_speakers > 0:
            speakers_payload["maxCount"] = max_speakers
        if min_speakers and min_speakers > 0:
            speakers_payload["minCount"] = min_speakers
        properties["diarization"] = {"speakers": speakers_payload}
    if len(languages) > 1:
        properties["languageIdentification"] = {
            "mode": "Continuous",
            "candidateLocales": languages,
        }
    payload = {
        "displayName": f"diarize-{uuid.uuid4().hex[:8]}",
        "description": "Batch diarization job",
        "locale": languages[0],
        "contentUrls": [content_url],
        "properties": properties,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if response.status_code not in (200, 201, 202):
        raise RuntimeError(f"Failed to submit transcription: {response.status_code} {response.text}")
    location = response.headers.get("Location")
    if not location:
        raise RuntimeError("Transcription submission missing Location header")
    logging.info("Submitted transcription job: %s", location)
    return location


def poll_transcription(
    location: str,
    speech_key: str,
    timeout_seconds: int,
    poll_interval: int,
) -> Dict[str, Any]:
    headers = {"Ocp-Apim-Subscription-Key": speech_key}
    deadline = time.time() + timeout_seconds
    while True:
        try:
            response = requests.get(location, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logging.warning("Transient error while polling job: %s", exc)
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for transcription job") from exc
            time.sleep(min(poll_interval, 5))
            continue
        data = response.json()
        status = data.get("status")
        logging.info("Job status: %s", status)
        if status in {"Succeeded", "Failed"}:
            if status == "Failed":
                raise RuntimeError(f"Transcription job failed: {json.dumps(data, indent=2)}")
            return data
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for transcription job")
        time.sleep(poll_interval)


def download_transcription_json(
    job_data: Dict[str, Any],
    speech_key: str,
    storage_ctx: StorageContext,
    retries: int = 5,
    backoff_seconds: int = 2,
) -> Dict[str, Any]:
    headers = {"Ocp-Apim-Subscription-Key": speech_key}
    files_link = job_data.get("links", {}).get("files")
    if not files_link and job_data.get("self"):
        files_link = f"{job_data['self']}/files"
    if not files_link:
        raise RuntimeError("Transcription job response does not expose file links")
    files_resp = requests.get(files_link, headers=headers, timeout=30)
    files_resp.raise_for_status()
    for entry in files_resp.json().get("values", []):
        if entry.get("kind") == "Transcription" and entry.get("links", {}).get("contentUrl"):
            url = entry["links"]["contentUrl"]
            logging.info("Downloading transcription result %s", url)
            blob_name = _extract_blob_name(url, storage_ctx.container_name)
            if blob_name:
                return _download_blob_from_storage(
                    storage_ctx,
                    blob_name,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                )
            # Fallback to HTTP download when blob path cannot be inferred
            last_error: Optional[requests.HTTPError] = None
            for attempt in range(retries):
                content_resp = requests.get(url, timeout=60)
                if content_resp.status_code == 404:
                    last_error = requests.HTTPError("404 Not Found", response=content_resp)
                    wait = backoff_seconds * (attempt + 1)
                    logging.debug(
                        "Result blob not ready yet (attempt %s/%s). Retrying in %ss",
                        attempt + 1,
                        retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                content_resp.raise_for_status()
                return content_resp.json()
            raise RuntimeError(
                f"Unable to download transcription JSON after {retries} attempts: {last_error}"
            )
    raise RuntimeError("No transcription JSON found in job output")


def _extract_blob_name(url: str, container_name: str) -> Optional[str]:
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    prefix = f"{container_name}/"
    if not path.startswith(prefix):
        return None
    return path[len(prefix) :]


def _download_blob_from_storage(
    ctx: StorageContext,
    blob_name: str,
    retries: int,
    backoff_seconds: int,
) -> Dict[str, Any]:
    blob_client = ctx.blob_service.get_blob_client(ctx.container_name, blob_name)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            stream = blob_client.download_blob()
            data = stream.readall()
            return json.loads(data)
        except Exception as exc:  # Broad catch to include ResourceNotFound
            last_exc = exc
            wait = backoff_seconds * (attempt + 1)
            logging.debug(
                "Blob %s not ready yet (attempt %s/%s). Retrying in %ss",
                blob_name,
                attempt + 1,
                retries,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Unable to download transcription blob {blob_name} after {retries} attempts: {last_exc}"
    )


def iso_duration_to_seconds(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = ISO_DURATION.fullmatch(value)
        if not match:
            return None
        hours = float(match.group("h") or 0)
        minutes = float(match.group("m") or 0)
        seconds = float(match.group("s") or 0)
        return hours * 3600 + minutes * 60 + seconds
    return None


def extract_segments(batch_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for phrase in batch_json.get("recognizedPhrases", []):
        nbest = (phrase.get("nBest") or [{}])[0]
        start = phrase.get("offsetInSeconds")
        duration = phrase.get("durationInSeconds")
        if start is None:
            start = iso_duration_to_seconds(phrase.get("offset"))
        if duration is None:
            duration = iso_duration_to_seconds(phrase.get("duration"))
        start = start or 0.0
        duration = duration or 0.0
        text = nbest.get("display") or nbest.get("lexical") or nbest.get("itn") or ""
        segments.append(
            {
                "speaker": phrase.get("speaker"),
                "channel": phrase.get("channel"),
                "language": phrase.get("language") or nbest.get("language"),
                "start": start,
                "duration": duration,
                "end": start + duration,
                "text": text.strip(),
            }
        )
    segments.sort(key=lambda s: s["start"])
    return segments


def write_json_output(segments: List[Dict[str, Any]], path: Optional[str]) -> None:
    payload = {"segments": segments}
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        sys.stdout.flush()


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{ms:03}"


def write_vtt_output(segments: List[Dict[str, Any]], path: Optional[str]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, 1):
            start = format_timestamp(seg.get("start", 0.0))
            end = format_timestamp(seg.get("end", seg.get("start", 0.0)))
            speaker = seg.get("speaker")
            speaker_label = f"Speaker {speaker}" if speaker is not None else "Speaker"
            handle.write(f"{idx}\n{start} --> {end}\n{speaker_label}: {seg.get('text', '')}\n\n")


def cleanup(ctx: StorageContext, keep: bool) -> None:
    if keep:
        logging.info("Keeping container %s", ctx.container_name)
        return
    try:
        ctx.blob_service.delete_container(ctx.container_name)
        logging.info("Deleted temporary container %s", ctx.container_name)
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logging.warning("Failed to delete container %s: %s", ctx.container_name, exc)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    if args.json_input:
        json_path = Path(args.json_input)
        if not json_path.is_file():
            logging.error("JSON file not found: %s", json_path)
            return 1
        with json_path.open(encoding="utf-8") as handle:
            batch_json = json.load(handle)
        if "segments" in batch_json:
            logging.debug("Detected normalized segments format in %s", json_path)
            segments = batch_json["segments"]
        else:
            segments = extract_segments(batch_json)
        logging.info("Converted %d segments from %s", len(segments), json_path)
        write_json_output(segments, args.output)
        write_vtt_output(segments, args.vtt_output)
        return 0

    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not speech_key or not speech_region:
        logging.error("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION")
        return 1
    if not storage_conn:
        logging.error("Missing AZURE_STORAGE_CONNECTION_STRING for blob uploads")
        return 1

    audio_path = Path(args.file)
    if not audio_path.is_file():
        logging.error("Audio file not found: %s", audio_path)
        return 1

    upload_path, temp_path = ensure_wav_for_upload(audio_path)

    ctx = create_storage_context(storage_conn)
    success = False
    service_error = False
    try:
        blob_name = upload_audio(ctx, upload_path)
        sas_urls = build_sas_urls(ctx, blob_name, args.sas_hours)
        location = submit_transcription(
            speech_region=speech_region,
            speech_key=speech_key,
            content_url=sas_urls["input"],
            destination_url=sas_urls["destination"],
            languages=args.languages,
            max_speakers=args.max_speakers,
            min_speakers=args.min_speakers,
        )
        job_data = poll_transcription(
            location=location,
            speech_key=speech_key,
            timeout_seconds=args.job_timeout,
            poll_interval=args.poll_interval,
        )
        batch_json = download_transcription_json(job_data, speech_key, storage_ctx=ctx)
        segments = extract_segments(batch_json)
        logging.info("Received %d diarized segments", len(segments))
        write_json_output(segments, args.output)
        write_vtt_output(segments, args.vtt_output)
        success = True
    except RuntimeError as exc:
        if "Transcription job failed" in str(exc):
            service_error = True
        raise
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
            logging.debug("Deleted temporary wav %s", temp_path)
        keep_container = args.keep_artifacts or (not success and not service_error)
        cleanup(ctx, keep_container)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
