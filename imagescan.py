"""Car damage detection using AWS Bedrock and Claude vision models."""

import argparse
import base64
import json
import logging
import os
import sys
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

# Defaults
DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
DEFAULT_DELAY_SECONDS = 60
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg")
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

ANALYSIS_PROMPT = (
    "Analyze this image of a car. Describe the car's appearance, including its "
    "color, make, and model if identifiable. Most importantly, carefully examine "
    "the entire vehicle and report any visible damage, scratches, dents, or "
    "imperfections. Be thorough in your inspection and report all findings."
)

logger = logging.getLogger(__name__)


def get_media_type(key: str) -> str:
    """Return the media type for an S3 object key based on its file extension."""
    ext = os.path.splitext(key)[1].lower()
    return MEDIA_TYPES.get(ext, "image/jpeg")


def get_image_from_s3(s3_client, bucket: str, key: str) -> Optional[str]:
    """Retrieve an image from S3 and return it as a base64-encoded string."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_content = response["Body"].read()
        return base64.b64encode(image_content).decode("utf-8")
    except ClientError as e:
        logger.error("Error retrieving image '%s' from S3: %s", key, e)
        return None


def invoke_with_retry(bedrock_client, model_id: str, body: str) -> dict:
    """Invoke a Bedrock model with exponential backoff retry on transient errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = bedrock_client.invoke_model(modelId=model_id, body=body)
            return json.loads(response["body"].read())
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("ThrottlingException", "ServiceUnavailableException") and attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Transient error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, MAX_RETRIES, wait, e)
                time.sleep(wait)
            else:
                raise
    # Should not reach here, but satisfy type checker
    raise RuntimeError("Exhausted retries")


def analyze_car_image(
    bedrock_client,
    image_base64: str,
    image_key: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> Optional[str]:
    """Analyze a car image for damage using Amazon Bedrock."""
    media_type = get_media_type(image_key)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": ANALYSIS_PROMPT,
                    },
                ],
            }
        ],
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
    })

    try:
        response_body = invoke_with_retry(bedrock_client, model_id, body)
        content = response_body.get("content")
        if content and len(content) > 0:
            return content[0].get("text")
        logger.error("Unexpected response structure: %s", response_body)
        return None
    except ClientError as e:
        logger.error("Error invoking Bedrock model: %s", e)
        return None


def list_image_keys(s3_client, bucket_name: str) -> list[str]:
    """List all image keys in an S3 bucket, handling pagination."""
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(SUPPORTED_EXTENSIONS):
                    keys.append(key)
    except ClientError as e:
        logger.error("Error listing objects in bucket '%s': %s", bucket_name, e)
    return keys


def process_images_in_bucket(
    bucket_name: str,
    model_id: str = DEFAULT_MODEL_ID,
    delay: int = DEFAULT_DELAY_SECONDS,
    output_file: Optional[str] = None,
) -> list[dict]:
    """Process all images in an S3 bucket and return analysis results."""
    s3_client = boto3.client("s3")
    bedrock_client = boto3.client("bedrock-runtime")

    image_keys = list_image_keys(s3_client, bucket_name)
    total = len(image_keys)

    if total == 0:
        logger.info("No images found in bucket '%s'.", bucket_name)
        return []

    logger.info("Found %d image(s) in bucket '%s'.", total, bucket_name)
    results: list[dict] = []

    for idx, key in enumerate(image_keys, start=1):
        logger.info("Processing image %d of %d: %s", idx, total, key)
        image_base64 = get_image_from_s3(s3_client, bucket_name, key)
        if not image_base64:
            continue

        analysis = analyze_car_image(bedrock_client, image_base64, key, model_id)
        result = {"image": key, "analysis": analysis}
        results.append(result)

        if analysis:
            print(f"\nAnalysis for {key}:\n{analysis}\n{'=' * 50}\n")

        if idx < total:
            logger.debug("Waiting %ds before next request...", delay)
            time.sleep(delay)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", output_file)

    return results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments with environment variable fallbacks."""
    parser = argparse.ArgumentParser(
        description="Analyze car images for damage using Amazon Bedrock."
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET"),
        help="S3 bucket name (env: S3_BUCKET)",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        help=f"Bedrock model ID (env: MODEL_ID, default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=int(os.environ.get("DELAY_SECONDS", str(DEFAULT_DELAY_SECONDS))),
        help=f"Delay in seconds between API calls (env: DELAY_SECONDS, default: {DEFAULT_DELAY_SECONDS})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the car damage detection script."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.bucket:
        logger.error("S3 bucket name is required. Use --bucket or set S3_BUCKET env var.")
        sys.exit(1)

    process_images_in_bucket(
        bucket_name=args.bucket,
        model_id=args.model_id,
        delay=args.delay,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
