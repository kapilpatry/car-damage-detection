import boto3
import json
import base64
from botocore.exceptions import ClientError
import time

def get_image_from_s3(bucket, key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()
        return base64.b64encode(image_content).decode('utf-8')
    except ClientError as e:
        print(f"Error retrieving image from S3: {e}")
        return None

def analyze_car_image(image_base64):
    bedrock = boto3.client('bedrock-runtime')

    prompt = """Analyze this image of a car. Describe the car's appearance, including its color, make, and model if identifiable. 
    Most importantly, carefully examine the entire vehicle and report any visible damage, scratches, dents, or imperfections. 
    Be thorough in your inspection and report all findings."""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "top_p": 0.9,
    })

    try:
        response = bedrock.invoke_model(
            modelId= "anthropic.claude-3-5-sonnet-20240620-v1:0",
            #"anthropic.claude-3-sonnet-20240229-v1:0",
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']
    except ClientError as e:
        print(f"Error invoking Bedrock model: {e}")
        return None

def process_images_in_bucket(bucket_name):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing image: {key}")
                image_base64 = get_image_from_s3(bucket_name, key)
                if image_base64:
                    analysis = analyze_car_image(image_base64)
                    if analysis:
                        print(f"Analysis for {key}:")
                        print(analysis)
                        print("\n" + "="*50 + "\n")
                    # Delay for 1 minute before the next request
                    time.sleep(60)
    except ClientError as e:
        print(f"Error listing objects in bucket: {e}")

if __name__ == "__main__":
    bucket_name = "kp-acc-image-detect"  # Replace with your actual S3 bucket name #  
    process_images_in_bucket(bucket_name)