import asyncio
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, status
from app.core.config import settings
from loguru import logger


class S3Service:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.AWS_S3_BUCKET
        self.region = settings.AWS_REGION

    def public_url(self, key: str) -> str:
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{key}"

    async def ensure_object_folder(self, object_id: str) -> str:
        """
        Creates a placeholder object in S3 so the object_id folder
        is visible in the AWS Console.
        Key: inspections/{object_id}/.keep
        Safe to call multiple times — overwrites the tiny placeholder.
        Returns the folder prefix.
        """
        folder_prefix = f"inspections/{object_id.upper()}/"
        placeholder_key = f"{folder_prefix}.keep"
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.put_object(
                    Bucket=self.bucket,
                    Key=placeholder_key,
                    Body=b"",
                    ContentType="application/octet-stream",
                )
            )
            logger.info(f"S3 folder ensured: {folder_prefix}")
        except ClientError as e:
            # Non-fatal — log and continue
            logger.warning(f"S3 folder placeholder failed [{placeholder_key}]: {e}")
        return folder_prefix

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload raw bytes to S3 (public-read) and return the public URL."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                    # ACL="public-read",
                )
            )
            url = self.public_url(key)
            logger.info(f"S3 ✓ → {key}")
            return url
        except ClientError as e:
            logger.error(f"S3 upload failed [{key}]: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"S3 upload failed: {str(e)}",
            )

    async def download_bytes(self, key: str) -> bytes:
        """Download raw bytes from S3."""
        loop = asyncio.get_running_loop()
        try:
            def _download():
                response = self.client.get_object(Bucket=self.bucket, Key=key)
                return response["Body"].read()
            return await loop.run_in_executor(None, _download)
        except ClientError as e:
            logger.error(f"S3 download failed [{key}]: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"S3 download failed: {str(e)}",
            )

    async def delete_prefix(self, prefix: str) -> int:
        """
        Deletes all objects in the bucket that start with the given prefix.
        Returns the number of objects deleted.
        """
        loop = asyncio.get_running_loop()
        def _delete():
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

            deleted_count = 0
            for page in pages:
                if "Contents" in page:
                    # AWS S3 delete_objects takes max 1000 keys per request
                    objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
                    self.client.delete_objects(
                        Bucket=self.bucket,
                        Delete={"Objects": objects_to_delete, "Quiet": True}
                    )
                    deleted_count += len(objects_to_delete)
            return deleted_count

        try:
            deleted_count = await loop.run_in_executor(None, _delete)
            logger.info(f"S3 ✓ Deleted {deleted_count} objects under prefix '{prefix}'")
            return deleted_count
        except ClientError as e:
            logger.error(f"S3 delete_prefix failed [{prefix}]: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"S3 deletion failed: {str(e)}",
            )


s3_service = S3Service()
