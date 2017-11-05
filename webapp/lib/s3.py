import tempfile
import uuid

import boto3
import cv2
import os

config = {
    "AWS_BUCKET":            os.environ.get("AS_AWS_BUCKET", "actress-search"),
    "AWS_KEY_PREFIX":        os.environ.get("AS_AWS_KEY_PREFIX", "requests"),
    "AWS_ACCESS_KEY_ID":     os.environ.get("AS_AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AS_AWS_SECRET_ACCESS_KEY", "")
}

def upload_image_to_s3(img):
    with tempfile.TemporaryDirectory() as dir:
        image_file = os.path.join(dir, "image.jpg")
        cv2.imwrite(image_file, img)

        Bucket = config["AWS_BUCKET"]
        Key = "{0}/{1}.jpg".format(config["AWS_KEY_PREFIX"], str(uuid.uuid4()))

        s3 = boto3.client(
            "s3",
            aws_access_key_id = config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key = config["AWS_SECRET_ACCESS_KEY"]
        )
        s3.upload_file(image_file, Bucket=Bucket, Key=Key)
        s3.put_object_acl(ACL='public-read', Bucket=Bucket, Key=Key)