# Car Damage Analysis with AWS Generative AI

This Python script automates the process of analyzing car images for damage and imperfections using Amazon Bedrock Anthropic Claude 3.5 Sonnet. It streamlines the workflow of retrieving images from an S3 bucket, processing them with Amazon Bedrock's Anthropic Claude 3.5 Sonnet model, and outputting detailed analysis results.

## Features

- Retrieves car images from a specified AWS S3 bucket
- Utilizes Amazon Bedrock API with Anthropic Claude 3.5 Sonnet model for advanced image analysis
- Detects and describes visible damage or imperfections on cars
- Processes multiple images in batch
- Outputs detailed analysis results for each image

## Requirements

- Python 3.x
- Boto3 (AWS SDK for Python)
- AWS account with access to S3 and Amazon Bedrock
- Request access to Anthropic Claude Sonnet 3.5 in Amazon Bedrock
- Properly configured AWS credentials

## How It Works

1. Connects to the specified S3 bucket
2. Retrieves images from the bucket
3. Sends each image to Amazon Bedrock's Claude 3.5 Sonnet model for analysis
4. Processes the AI-generated descriptions of damage and imperfections
5. Prints or logs the analysis results for each image

## Usage

1. Upload the images in one of these formats: .jpg, .jpeg, .png
2. Execute the imagescan.py from your terminal

## Output

The script provides detailed output for each analyzed image, including:
- Image filename
- Detected damage areas
- Severity of damage
- Description of imperfections

## Why Claude 3.5 Sonnet?

Anthropic's Claude 3.5 Sonnet model, available through Amazon Bedrock, offers:
- Advanced visual understanding capabilities
- Detailed and nuanced descriptions of car damage
- High accuracy in identifying various types of imperfections
- Efficient processing for large batches of images

## License

[Specify the license for your project]

---

This summary highlights the use of Amazon Bedrock's Anthropic Claude 3.5 Sonnet model as the foundation for your image analysis task. It provides an overview of your script's functionality, key features, and includes sections that you can easily fill in with specific details about usage, configuration, and licensing. This version gives potential users or collaborators a clear understanding of what your script does, how it leverages advanced AI capabilities, and the benefits of using the Claude 3.5 Sonnet model for car damage analysis.
