"""Artifact model for Underfit."""

from __future__ import annotations

import base64
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

from underfit.media import Audio, Html, Image, Video

PathLike = Union[str, Path]
BytesLike = Union[bytes, bytearray, memoryview]
MediaObject = Union[Html, Image, Video, Audio]


class Artifact:
    """Represent an artifact collection.

    Args:
        name: Artifact name.
        type: Artifact type such as ``"dataset"`` or ``"model"``.
        metadata: Optional metadata dictionary.

    Raises:
        TypeError: If ``name`` or ``type`` are not strings or metadata is not a mapping.
        ValueError: If ``name`` or ``type`` are empty.

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
        self._upload_entries: list[dict[str, Any]] = []
        self._references: list[dict[str, Any]] = []
        self._used_entry_names: set[str] = set()
        self._next_media_index = 1

    def add_file(self, local_path: PathLike, *, name: str | None = None) -> str:
        """Add a file entry that will be uploaded by the backend uploader.

        Args:
            local_path: Path to a local file.
            name: Optional artifact entry name.

        Returns:
            The resolved artifact entry name.

        Raises:
            FileNotFoundError: If the file path does not exist.
            TypeError: If ``local_path`` is not path-like.
            ValueError: If ``local_path`` is not a file or the name is invalid.
        """
        path = self._coerce_existing_path(local_path)
        if not path.is_file():
            raise ValueError(f"local_path must point to a file: {path}")

        entry_name = self._resolve_entry_name(path.name, name)
        self._upload_entries.append({"kind": "file", "name": entry_name, "path": str(path)})
        return entry_name

    def add_dir(self, local_path: PathLike, *, name: str | None = None) -> str:
        """Add a directory entry that will be uploaded by the backend uploader.

        Args:
            local_path: Path to a local directory.
            name: Optional artifact entry name.

        Returns:
            The resolved artifact entry name.

        Raises:
            FileNotFoundError: If the directory path does not exist.
            TypeError: If ``local_path`` is not path-like.
            ValueError: If ``local_path`` is not a directory or the name is invalid.
        """
        path = self._coerce_existing_path(local_path)
        if not path.is_dir():
            raise ValueError(f"local_path must point to a directory: {path}")

        entry_name = self._resolve_entry_name(path.name, name)
        self._upload_entries.append({"kind": "directory", "name": entry_name, "path": str(path)})
        return entry_name

    def add_media(self, obj: MediaObject, *, name: str | None = None) -> str:
        """Add a media object that will be uploaded by the backend uploader.

        Args:
            obj: Media object from ``underfit.media``.
            name: Optional artifact entry name.

        Returns:
            The resolved artifact entry name.

        Raises:
            TypeError: If ``obj`` is not a supported media object.
            ValueError: If the entry name is invalid.
        """
        if not isinstance(obj, (Html, Image, Video, Audio)):
            raise TypeError(
                "obj must be an underfit.media.Html, underfit.media.Image, "
                "underfit.media.Video, or underfit.media.Audio"
            )

        default_name = f"media-{self._next_media_index}"
        entry_name = self._resolve_entry_name(default_name, name)
        if name is None:
            self._next_media_index += 1
        self._upload_entries.append({"kind": "media", "name": entry_name, "payload": obj.to_payload()})
        return entry_name

    def add_bytes(self, data: BytesLike, *, name: str | None = None) -> str:
        """Add a raw-bytes entry that will be uploaded by the backend uploader.

        Args:
            data: Bytes payload to upload.
            name: Optional artifact entry name.

        Returns:
            The resolved artifact entry name.

        Raises:
            TypeError: If ``data`` is not bytes-like.
            ValueError: If the entry name is invalid.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")

        entry_name = self._resolve_entry_name("checkpoint.bin", name)
        encoded = base64.b64encode(bytes(data)).decode("ascii")
        self._upload_entries.append({"kind": "bytes", "name": entry_name, "data": encoded})
        return entry_name

    def add_url(self, url: str, *, name: str | None = None) -> None:
        """Add a URL reference entry.

        URL references are stored as external pointers and are not included in
        the upload manifest.

        Args:
            url: URL to reference.
            name: Optional reference name.

        Raises:
            TypeError: If ``url`` is not a string.
            ValueError: If the URL or name is invalid.
        """
        if not isinstance(url, str):
            raise TypeError("url must be a string")
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError("url must include a scheme")

        reference: dict[str, Any] = {"url": url}
        if name is not None:
            reference["name"] = self._resolve_entry_name(None, name)

        self._references.append(reference)

    def upload_manifest(self) -> list[dict[str, Any]]:
        """Return entries that should be uploaded by the backend uploader."""
        return copy.deepcopy(self._upload_entries)

    def to_payload(self) -> dict[str, Any]:
        """Return a serializable artifact payload."""
        payload: dict[str, Any] = {
            "_type": "artifact",
            "name": self.name,
            "artifact_type": self.type,
            "metadata": copy.deepcopy(self.metadata),
            "manifest": self.upload_manifest(),
            "references": copy.deepcopy(self._references),
        }
        return payload

    def _coerce_existing_path(self, value: PathLike) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError("local_path must be a path string or Path")

        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        return path

    def _resolve_entry_name(self, default_name: str | None, name: str | None) -> str:
        candidate = default_name if name is None else name
        if candidate is None:
            raise ValueError("name must be provided")
        if not isinstance(candidate, str):
            raise TypeError("name must be a string")
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("name must be non-empty")
        if candidate in self._used_entry_names:
            raise ValueError(f"entry name already exists: {candidate}")

        self._used_entry_names.add(candidate)
        return candidate
