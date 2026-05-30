"""FastAPI dependency providers for shared services.

Replaces direct req.app.state access with typed dependency injection.
"""

import concurrent.futures

from fastapi import Request

from api.services.image_storage import ImageStorage
from api.services.recognition import RecognitionService
from src.indexer import SKUIndexer
from src.patch_store import PatchStore
from src.reference_processor import ReferenceProcessor


def get_indexer(request: Request) -> SKUIndexer:
    return request.app.state.indexer


def get_processor(request: Request) -> ReferenceProcessor:
    return request.app.state.processor


def get_recognition_service(request: Request) -> RecognitionService:
    return request.app.state.recognition_service


def get_image_storage(request: Request) -> ImageStorage:
    return request.app.state.image_storage


def get_inference_executor(request: Request) -> concurrent.futures.ThreadPoolExecutor:
    return request.app.state.inference_executor


def get_patch_store(request: Request) -> PatchStore | None:
    return request.app.state.patch_store


def get_device(request: Request) -> str:
    return request.app.state.device
