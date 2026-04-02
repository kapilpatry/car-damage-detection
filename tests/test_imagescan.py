"""Unit tests for imagescan module."""

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import imagescan


# --- get_media_type ---

@pytest.mark.parametrize("key,expected", [
    ("photo.jpg", "image/jpeg"),
    ("photo.JPEG", "image/jpeg"),
    ("photo.png", "image/png"),
    ("photo.PNG", "image/png"),
    ("photo.unknown", "image/jpeg"),
])
def test_get_media_type(key, expected):
    assert imagescan.get_media_type(key) == expected


# --- get_image_from_s3 ---

def test_get_image_from_s3_success():
    raw = b"fake-image-bytes"
    s3 = MagicMock()
    s3.get_object.return_value = {"Body": BytesIO(raw)}

    result = imagescan.get_image_from_s3(s3, "bucket", "car.jpg")
    assert result == base64.b64encode(raw).decode("utf-8")
    s3.get_object.assert_called_once_with(Bucket="bucket", Key="car.jpg")


def test_get_image_from_s3_client_error():
    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}, "GetObject"
    )
    result = imagescan.get_image_from_s3(s3, "bucket", "missing.jpg")
    assert result is None


# --- invoke_with_retry ---

def _make_response_body(text: str) -> dict:
    return {"content": [{"text": text}]}


def test_invoke_with_retry_success():
    bedrock = MagicMock()
    response_payload = _make_response_body("damage found")
    bedrock.invoke_model.return_value = {
        "body": BytesIO(json.dumps(response_payload).encode())
    }

    result = imagescan.invoke_with_retry(bedrock, "model-id", "{}")
    assert result == response_payload


@patch("imagescan.time.sleep")
def test_invoke_with_retry_retries_on_throttle(mock_sleep):
    bedrock = MagicMock()
    throttle = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "InvokeModel",
    )
    success_payload = _make_response_body("ok")
    bedrock.invoke_model.side_effect = [
        throttle,
        throttle,
        {"body": BytesIO(json.dumps(success_payload).encode())},
    ]

    result = imagescan.invoke_with_retry(bedrock, "model-id", "{}")
    assert result == success_payload
    assert mock_sleep.call_count == 2


def test_invoke_with_retry_raises_non_transient():
    bedrock = MagicMock()
    bedrock.invoke_model.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "InvokeModel",
    )
    with pytest.raises(ClientError):
        imagescan.invoke_with_retry(bedrock, "model-id", "{}")


# --- analyze_car_image ---

def test_analyze_car_image_success():
    bedrock = MagicMock()
    response_payload = _make_response_body("Dent on left door")
    bedrock.invoke_model.return_value = {
        "body": BytesIO(json.dumps(response_payload).encode())
    }

    result = imagescan.analyze_car_image(bedrock, "base64data", "car.png")
    assert result == "Dent on left door"


def test_analyze_car_image_error():
    bedrock = MagicMock()
    bedrock.invoke_model.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "bad"}},
        "InvokeModel",
    )
    result = imagescan.analyze_car_image(bedrock, "base64data", "car.jpg")
    assert result is None


def test_analyze_car_image_unexpected_response():
    bedrock = MagicMock()
    bedrock.invoke_model.return_value = {
        "body": BytesIO(json.dumps({"content": []}).encode())
    }
    result = imagescan.analyze_car_image(bedrock, "base64data", "car.jpg")
    assert result is None


# --- list_image_keys ---

def test_list_image_keys():
    s3 = MagicMock()
    paginator = MagicMock()
    s3.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {"Contents": [
            {"Key": "car1.jpg"},
            {"Key": "car2.png"},
            {"Key": "readme.txt"},
        ]},
        {"Contents": [
            {"Key": "car3.JPEG"},
        ]},
    ]

    keys = imagescan.list_image_keys(s3, "bucket")
    assert keys == ["car1.jpg", "car2.png", "car3.JPEG"]


def test_list_image_keys_empty_bucket():
    s3 = MagicMock()
    paginator = MagicMock()
    s3.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{}]

    keys = imagescan.list_image_keys(s3, "bucket")
    assert keys == []


# --- parse_args ---

def test_parse_args_defaults():
    args = imagescan.parse_args(["--bucket", "my-bucket"])
    assert args.bucket == "my-bucket"
    assert args.model_id == imagescan.DEFAULT_MODEL_ID
    assert args.delay == imagescan.DEFAULT_DELAY_SECONDS
    assert args.output is None
    assert args.verbose is False


def test_parse_args_all_options():
    args = imagescan.parse_args([
        "--bucket", "b",
        "--model-id", "m",
        "--delay", "5",
        "--output", "out.json",
        "--verbose",
    ])
    assert args.bucket == "b"
    assert args.model_id == "m"
    assert args.delay == 5
    assert args.output == "out.json"
    assert args.verbose is True


def test_parse_args_env_fallback(monkeypatch):
    monkeypatch.setenv("S3_BUCKET", "env-bucket")
    monkeypatch.setenv("MODEL_ID", "env-model")
    monkeypatch.setenv("DELAY_SECONDS", "10")
    args = imagescan.parse_args([])
    assert args.bucket == "env-bucket"
    assert args.model_id == "env-model"
    assert args.delay == 10


# --- main ---

def test_main_missing_bucket(capsys):
    with pytest.raises(SystemExit) as exc_info:
        imagescan.main(["--delay", "1"])
    assert exc_info.value.code == 1
