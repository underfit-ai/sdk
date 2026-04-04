"""Artifact model for Underfit."""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Mapping
from email.message import Message
from email.utils import formatdate
from pathlib import Path
from typing import Any, Union
from urllib.parse import unquote, urlparse

from underfit.media import Audio, Html, Image, Video

PathLike = Union[str, Path]
BytesLike = Union[bytes, bytearray, memoryview]
MediaObject = Union[Html, Image, Video, Audio]
MAX_PATH_BYTES = 1024
MAX_PATH_SEGMENT_BYTES = 255


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
        self._upload_files: list[dict[str, Any]] = []
        self._references: list[dict[str, Any]] = []
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
        self._upload_files.append({"path": artifact_path, "source_path": str(path)})
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
            child_path = self._join_artifact_path(artifact_path, child.relative_to(path).as_posix())
            self._reserve_artifact_path(child_path)
            self._upload_files.append({"path": child_path, "source_path": str(child)})
        return artifact_path

    def add_media(self, obj: MediaObject, *, name: str | None = None) -> str:
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
        if not isinstance(obj, (Html, Image, Video, Audio)):
            raise TypeError(
                "obj must be an underfit.media.Html, underfit.media.Image, "
                "underfit.media.Video, or underfit.media.Audio"
            )

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
        self._upload_files.append({"path": artifact_path, "data": encoded})
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

        metadata = self._reference_metadata(url, parsed)
        self._references.append({"url": url, **metadata})

    def create_payload(self) -> dict[str, Any]:
        """Return the payload used to create the artifact."""
        return {"name": self.name, "type": self.type, "metadata": copy.deepcopy(self.metadata) or None}

    def upload_files(self) -> list[dict[str, Any]]:
        """Return files that should be uploaded before finalizing the artifact."""
        return copy.deepcopy(self._upload_files)

    def upload_manifest(self) -> list[dict[str, Any]]:
        """Return files that should be uploaded before finalizing the artifact."""
        return self.upload_files()

    def finalize_manifest(self) -> dict[str, Any]:
        """Return the manifest payload used to finalize the artifact."""
        refs = list({ref["url"]: copy.deepcopy(ref) for ref in self._references}.values())
        return {"files": [upload["path"] for upload in self._upload_files], "references": refs}

    def to_payload(self) -> dict[str, Any]:
        """Return a serializable artifact payload."""
        return {
            "_type": "artifact",
            "name": self.name,
            "artifact_type": self.type,
            "metadata": copy.deepcopy(self.metadata),
            "files": self.upload_files(),
            "manifest": self.finalize_manifest(),
        }

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

    def _build_media_upload(self, payload: dict[str, Any], artifact_path: str) -> dict[str, Any]:
        if "path" in payload and payload["path"] is not None:
            path = self._coerce_existing_path(payload["path"])
            if not path.is_file():
                raise ValueError(f"media path must point to a file: {path}")
            return {"path": artifact_path, "source_path": str(path)}
        if "data" in payload and payload["data"] is not None:
            return {"path": artifact_path, "data": payload["data"]}
        if "html" in payload and payload["html"] is not None:
            encoded = base64.b64encode(str(payload["html"]).encode("utf-8")).decode("ascii")
            return {"path": artifact_path, "data": encoded}
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

    def _join_artifact_path(self, *parts: str) -> str:
        return self._normalize_artifact_path("/".join(part.strip("/") for part in parts if part))

    def _reference_metadata(self, url: str, parsed: Any) -> dict[str, Any]:
        scheme = parsed.scheme.lower()
        if scheme == "file":
            return self._file_reference_metadata(parsed)
        if scheme in {"http", "https"}:
            return self._http_reference_metadata(url)
        return self._empty_reference_metadata()

    def _http_reference_metadata(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, method="HEAD")  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                return self._headers_reference_metadata(response.headers)
        except urllib.error.HTTPError as exc:
            return self._headers_reference_metadata(exc.headers)
        except urllib.error.URLError:
            return self._empty_reference_metadata()

    def _file_reference_metadata(self, parsed: Any) -> dict[str, Any]:
        if parsed.netloc:
            raise ValueError("file:// URLs must not include an authority")
        path_text = urllib.request.url2pathname(unquote(parsed.path))
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"path must point to a file: {path}")

        stat = path.stat()
        return {
            "size": stat.st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "etag": None,
            "last_modified": formatdate(stat.st_mtime, usegmt=True),
        }

    def _headers_reference_metadata(self, headers: Message | None) -> dict[str, Any]:
        if headers is None:
            return self._empty_reference_metadata()
        size = self._header_size(headers.get("Content-Length"))
        return {
            "size": size,
            "sha256": self._header_sha256(headers),
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
        }

    def _header_size(self, value: str | None) -> int | None:
        if value is None:
            return None
        try:
            size = int(value)
        except ValueError:
            return None
        return size if size >= 0 else None

    def _header_sha256(self, headers: Message) -> str | None:
        for header_name in ("X-Checksum-Sha256", "X-Amz-Checksum-Sha256"):
            if value := headers.get(header_name):
                return self._normalize_sha256(value)
        if digest := headers.get("Digest"):
            for item in digest.split(","):
                algorithm, _, value = item.strip().partition("=")
                if algorithm.lower() in {"sha-256", "sha256"} and value:
                    return self._normalize_sha256(value.strip().strip('"'))
        return None

    def _normalize_sha256(self, value: str) -> str:
        candidate = value.strip()
        if len(candidate) == 64 and all(char in "0123456789abcdefABCDEF" for char in candidate):
            return candidate.lower()
        try:
            decoded = base64.b64decode(candidate, validate=True)
        except binascii.Error:
            return candidate
        return decoded.hex() if len(decoded) == 32 else candidate

    def _empty_reference_metadata(self) -> dict[str, Any]:
        return {"size": None, "sha256": None, "etag": None, "last_modified": None}
