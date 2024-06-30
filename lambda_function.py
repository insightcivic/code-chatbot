import json
import logging
import boto3
from botocore.exceptions import ClientError

# Set up logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize the Bedrock client
bedrock = boto3.client('bedrock-runtime')

def query_claude_haiku(message):
    try:
        logger.info(f"Sending message to Claude Haiku: {message}")
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            })
        )
        logger.info(f"Response received: {response}")
        response_body = json.loads(response['body'].read().decode('utf-8'))
        logger.info(f"Response body: {response_body}")
        claude_response = response_body['content'][0]['text']
        logger.info(f"Received response from Claude Haiku: {claude_response}")
        return claude_response
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS ClientError: {error_code} - {error_message}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error querying Claude Haiku: {str(e)}")
        raise

def lambda_handler(event, context):
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        body = json.loads(event['body'])
        user_message = body['message']
        logger.info(f"Extracted user message: {user_message}")
        response = query_claude_haiku(user_message)
        logger.info(f"Sending response: {response}")
        return {
            'statusCode': 200,
            'body': json.dumps({'message': response})
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error', 'details': str(e)})
        }
