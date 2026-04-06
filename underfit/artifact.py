"""Artifact model for Underfit."""

from __future__ import annotations

import base64
import binascii
import hashlib
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import Message
from email.utils import formatdate
from pathlib import Path
from typing import Any, Union
from urllib.parse import unquote, urlparse

from underfit.media import Media

PathLike = Union[str, Path]
BytesLike = Union[bytes, bytearray, memoryview]
MAX_PATH_BYTES = 1024
MAX_PATH_SEGMENT_BYTES = 255


@dataclass(frozen=True)
class ArtifactCreate:
    """Represent the payload used to create an artifact."""

    name: str
    type: str  # noqa: A003
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class ArtifactPathUpload:
    """Represent a queued file upload sourced from a local file."""

    path: str
    source_path: str


@dataclass(frozen=True)
class ArtifactDataUpload:
    """Represent a queued file upload sourced from in-memory data."""

    path: str
    data: str


ArtifactUpload = Union[ArtifactPathUpload, ArtifactDataUpload]


@dataclass(frozen=True)
class ArtifactReference:
    """Represent an external artifact reference."""

    url: str
    size: int | None = None
    sha256: str | None = None
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class ArtifactManifest:
    """Represent the finalized artifact manifest."""

    files: list[str]
    references: list[ArtifactReference]


class Artifact:
    """Represent an artifact collection.

    Examples:
        >>> from underfit import Artifact, Image
        >>> artifact = Artifact("eval", type="report")
        >>> artifact.add_file("./metrics.json")
        >>> artifact.add_media(Image("./plots/loss.png"), name="loss-curve")
        >>> artifact.add_url("https://example.com/model-card")
    """

    def __init__(
        self,
        name: str,
        type: str,  # noqa: A002
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize an artifact.

        Args:
            name: Artifact name.
            type: Artifact type such as ``"dataset"`` or ``"model"``.
            metadata: Optional metadata dictionary.

        Raises:
            TypeError: If ``name`` or ``type`` are not strings or metadata is not a mapping.
            ValueError: If ``name`` or ``type`` are empty.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(type, str):
            raise TypeError("type must be a string")
        if not name.strip():
            raise ValueError("name must be non-empty")
        if not type.strip():
            raise ValueError("type must be non-empty")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        self.name = name
        self.type = type
        self.metadata = dict(metadata or {})
        self._upload_files: list[ArtifactUpload] = []
        self._references: list[ArtifactReference] = []
        self._used_paths: set[str] = set()
        self._next_media_index = 1

    def add_file(self, local_path: PathLike, *, name: str | None = None) -> str:
        """Add a file to the artifact.

        Args:
            local_path: Path to a local file.
            name: Optional artifact-relative file path.

        Returns:
            The resolved artifact file path.

        Raises:
            FileNotFoundError: If the file path does not exist.
            TypeError: If ``local_path`` is not path-like.
            ValueError: If ``local_path`` is not a file or the name is invalid.
        """
        path = self._coerce_existing_path(local_path)
        if not path.is_file():
            raise ValueError(f"local_path must point to a file: {path}")

        artifact_path = self._resolve_artifact_path(path.name, name)
        self._upload_files.append(ArtifactPathUpload(path=artifact_path, source_path=str(path)))
        return artifact_path

    def add_dir(self, local_path: PathLike, *, name: str | None = None) -> str:
        """Add a directory tree to the artifact.

        Args:
            local_path: Path to a local directory.
            name: Optional artifact-relative directory path.

        Returns:
            The resolved artifact directory path prefix.

        Raises:
            FileNotFoundError: If the directory path does not exist.
            TypeError: If ``local_path`` is not path-like.
            ValueError: If ``local_path`` is not a directory or the name is invalid.
        """
        path = self._coerce_existing_path(local_path)
        if not path.is_dir():
            raise ValueError(f"local_path must point to a directory: {path}")

        artifact_path = self._normalize_artifact_path(path.name if name is None else name)
        files = sorted((child for child in path.rglob("*") if child.is_file()), key=lambda child: child.as_posix())
        for child in files:
            child_path = self._normalize_artifact_path(f"{artifact_path}/{child.relative_to(path).as_posix()}")
            self._reserve_artifact_path(child_path)
            self._upload_files.append(ArtifactPathUpload(path=child_path, source_path=str(child)))
        return artifact_path

    def add_media(self, obj: Media, *, name: str | None = None) -> str:
        """Add a media object to the artifact.

        Args:
            obj: Media object from ``underfit.media``.
            name: Optional artifact-relative file path.

        Returns:
            The resolved artifact file path.

        Raises:
            TypeError: If ``obj`` is not a supported media object.
            ValueError: If the file path is invalid.
        """
        if not isinstance(obj, Media):
            raise TypeError("obj must be an underfit media object implementing the Media protocol")

        payload = obj.to_payload()
        default_name = self._default_media_name(payload)
        artifact_path = self._resolve_artifact_path(default_name, name)
        if name is None:
            self._next_media_index += 1
        self._upload_files.append(self._build_media_upload(payload, artifact_path))
        return artifact_path

    def add_bytes(self, data: BytesLike, *, name: str | None = None) -> str:
        """Add raw bytes to the artifact.

        Args:
            data: Bytes payload to upload.
            name: Optional artifact-relative file path.

        Returns:
            The resolved artifact file path.

        Raises:
            TypeError: If ``data`` is not bytes-like.
            ValueError: If the file path is invalid.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")

        artifact_path = self._resolve_artifact_path("checkpoint.bin", name)
        encoded = base64.b64encode(bytes(data)).decode("ascii")
        self._upload_files.append(ArtifactDataUpload(path=artifact_path, data=encoded))
        return artifact_path

    def add_url(self, url: str) -> None:
        """Add a URL reference to the artifact manifest.

        URL references are stored as external pointers and are not uploaded as
        artifact files.

        Args:
            url: URL to reference.

        Raises:
            FileNotFoundError: If a ``file://`` URL points to a missing file.
            TypeError: If ``url`` is not a string.
            ValueError: If the URL is invalid, a ``file://`` URL has an
                authority, or a ``file://`` URL does not point to a file.
        """
        if not isinstance(url, str):
            raise TypeError("url must be a string")
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError("url must include a scheme")
        scheme = parsed.scheme.lower()
        if scheme == "file":
            self._references.append(self._file_reference(url, parsed))
        elif scheme in {"http", "https"}:
            self._references.append(self._http_reference(url))
        else:
            self._references.append(ArtifactReference(url=url))

    def create_request(self) -> ArtifactCreate:
        """Return the typed payload used to create the artifact."""
        return ArtifactCreate(name=self.name, type=self.type, metadata=self.metadata or None)

    def uploads(self) -> list[ArtifactUpload]:
        """Return files that should be uploaded before finalizing the artifact."""
        return list(self._upload_files)

    def manifest(self) -> ArtifactManifest:
        """Return the typed manifest payload used to finalize the artifact."""
        refs = list({ref.url: ref for ref in self._references}.values())
        return ArtifactManifest(files=[upload.path for upload in self._upload_files], references=refs)

    def _coerce_existing_path(self, value: PathLike) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError("local_path must be a path string or Path")

        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        return path

    def _resolve_artifact_path(self, default_name: str, name: str | None) -> str:
        artifact_path = self._normalize_artifact_path(default_name if name is None else name)
        self._reserve_artifact_path(artifact_path)
        return artifact_path

    def _reserve_artifact_path(self, artifact_path: str) -> None:
        for existing in self._used_paths:
            if (
                existing == artifact_path
                or existing.startswith(f"{artifact_path}/")
                or artifact_path.startswith(f"{existing}/")
            ):
                raise ValueError(f"entry path already exists: {artifact_path}")
        self._used_paths.add(artifact_path)

    def _build_media_upload(self, payload: dict[str, Any], artifact_path: str) -> ArtifactUpload:
        if "path" in payload and payload["path"] is not None:
            path = self._coerce_existing_path(payload["path"])
            if not path.is_file():
                raise ValueError(f"media path must point to a file: {path}")
            return ArtifactPathUpload(path=artifact_path, source_path=str(path))
        if "data" in payload and payload["data"] is not None:
            return ArtifactDataUpload(path=artifact_path, data=payload["data"])
        if "html" in payload and payload["html"] is not None:
            encoded = base64.b64encode(str(payload["html"]).encode("utf-8")).decode("ascii")
            return ArtifactDataUpload(path=artifact_path, data=encoded)
        raise ValueError("media payload is missing content")

    def _default_media_name(self, payload: dict[str, Any]) -> str:
        suffix = ""
        if payload.get("_type") == "html":
            suffix = ".html"
        elif file_type := payload.get("file_type"):
            suffix = f".{file_type}"
        return f"media-{self._next_media_index}{suffix}"

    def _normalize_artifact_path(self, path: str) -> str:
        if not isinstance(path, str):
            raise TypeError("name must be a string")

        path = unicodedata.normalize("NFC", path)
        if not path or path.startswith("/"):
            raise ValueError("name must be a valid relative path")
        if any(ch == "\\" or (ch.isspace() and ch != " ") or unicodedata.category(ch).startswith("C") for ch in path):
            raise ValueError("name must be a valid relative path")
        if len(path.encode()) > MAX_PATH_BYTES:
            raise ValueError("name must be shorter than 1024 bytes")
        for segment in path.split("/"):
            if not segment or segment in {".", ".."} or segment != segment.strip(" ") or segment.endswith("."):
                raise ValueError("name must be a valid relative path")
            if len(segment.encode()) > MAX_PATH_SEGMENT_BYTES:
                raise ValueError("name must be shorter than 255 bytes per segment")
        return path

    def _http_reference(self, url: str) -> ArtifactReference:
        request = urllib.request.Request(url, method="HEAD")  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                return self._headers_reference(url, response.headers)
        except urllib.error.HTTPError as exc:
            return self._headers_reference(url, exc.headers)
        except urllib.error.URLError:
            return ArtifactReference(url=url)

    def _file_reference(self, url: str, parsed: Any) -> ArtifactReference:
        if parsed.netloc:
            raise ValueError("file:// URLs must not include an authority")
        path_text = urllib.request.url2pathname(unquote(parsed.path))
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"path must point to a file: {path}")

        stat = path.stat()
        return ArtifactReference(
            url=url,
            size=stat.st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            last_modified=formatdate(stat.st_mtime, usegmt=True),
        )

    def _headers_reference(self, url: str, headers: Message | None) -> ArtifactReference:
        if headers is None:
            return ArtifactReference(url=url)
        return ArtifactReference(
            url=url,
            size=self._header_size(headers.get("Content-Length")),
            sha256=self._header_sha256(headers),
            etag=headers.get("ETag"),
            last_modified=headers.get("Last-Modified"),
        )

    def _header_size(self, value: str | None) -> int | None:
        try:
            return int(value) if value is not None and int(value) >= 0 else None
        except ValueError:
            return None

    def _header_sha256(self, headers: Message) -> str | None:
        for header_name in ("X-Checksum-Sha256", "X-Amz-Checksum-Sha256"):
            if value := headers.get(header_name):
                candidate = value.strip()
                if len(candidate) == 64 and all(char in "0123456789abcdefABCDEF" for char in candidate):
                    return candidate.lower()
                try:
                    decoded = base64.b64decode(candidate, validate=True)
                    return decoded.hex() if len(decoded) == 32 else candidate
                except binascii.Error:
                    return candidate
        return None
