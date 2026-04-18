"""Artifact model for Underfit."""

from __future__ import annotations

import base64
import binascii
import hashlib
import subprocess
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import Message
from email.utils import formatdate
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Union
from urllib.parse import unquote, urlparse
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from underfit.media import Audio, Html, Image, Media, Video

PathLike = Union[str, Path]
BytesLike = Union[bytes, bytearray, memoryview]
PathFilter = Callable[[Path], bool]
MAX_PATH_BYTES = 1024
MAX_PATH_SEGMENT_BYTES = 255


@dataclass(frozen=True)
class ArtifactPathUpload:
    """Represent a queued file upload sourced from a local file."""
    path: str
    source_path: str


@dataclass(frozen=True)
class ArtifactDataUpload:
    """Represent a queued file upload sourced from in-memory data."""
    path: str
    data: bytes


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


ArtifactUpload = Union[ArtifactPathUpload, ArtifactDataUpload]


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
        step: int | None = None,
    ) -> None:
        """Initialize an artifact.

        Args:
            name: Artifact name.
            type: Artifact type such as ``"dataset"`` or ``"model"``.
            metadata: Optional metadata dictionary.
            step: Optional global step for the artifact.

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
        self.step = step
        self._upload_files: list[ArtifactUpload] = []
        self._references: list[ArtifactReference] = []
        self._used_paths: set[str] = set()
        self._next_media_index = 1

    @classmethod
    def from_code(
        cls,
        root_path: PathLike | None = None,
        *,
        name: str | None = None,
        include: PathFilter | None = None,
        exclude: PathFilter | None = None,
    ) -> Artifact:
        """Build a source-code artifact from a directory tree."""
        root = Path.cwd() if root_path is None else Path(root_path)
        if not root.exists():
            raise FileNotFoundError(f"root_path does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"root_path must point to a directory: {root}")
        artifact = cls(name or "source-code", "code")
        resolved_root = root.resolve()
        include_match = include or (lambda path: path.suffix == ".py")
        paths = sorted((path for path in resolved_root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
        matched_paths = [path for path in paths if include_match(path) and (exclude is None or not exclude(path))]
        if matched_paths:
            buffer = BytesIO()
            with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
                for path in matched_paths:
                    info = ZipInfo(path.relative_to(resolved_root).as_posix())
                    info.compress_type = ZIP_DEFLATED
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    archive.writestr(info, path.read_bytes())
            artifact.add_bytes(buffer.getvalue(), name=f"{artifact.name}.zip")
        return artifact

    @classmethod
    def from_git(cls, repo_path: PathLike | None = None, *, name: str | None = None) -> Artifact:
        """Build an artifact describing the current git state."""
        repo = Path.cwd() if repo_path is None else Path(repo_path)
        if not repo.exists():
            raise FileNotFoundError(f"repo_path does not exist: {repo}")
        if not repo.is_dir():
            raise ValueError(f"repo_path must point to a directory: {repo}")
        repo_root = Path(_git_text(repo, ["rev-parse", "--show-toplevel"]))
        status = _git_text(repo_root, ["status", "--porcelain=v2", "--branch"], ok_returncodes=(0, 128))
        head = None
        branch = None
        untracked_files: list[str] = []
        for line in status.splitlines():
            if line.startswith("# branch.oid "):
                value = line.removeprefix("# branch.oid ")
                head = None if value == "(initial)" else value
            elif line.startswith("# branch.head "):
                value = line.removeprefix("# branch.head ")
                branch = None if value == "(detached)" else value
            elif line.startswith("? "):
                untracked_files.append(line[2:])
        patch = (
            _git_output(repo_root, ["diff", "--binary", "HEAD"])
            if head is not None
            else _join_bytes([
                _git_output(repo_root, ["diff", "--binary", "--cached"]),
                _git_output(repo_root, ["diff", "--binary"]),
            ])
        )
        artifact = cls(name or "git-state", "git", metadata={
            "commit": head,
            "branch": branch,
            "is_dirty": bool(patch) or bool(untracked_files),
            "untracked_files": untracked_files,
        })
        artifact.add_bytes(patch, name="working-tree.patch")
        return artifact

    @classmethod
    def from_model(
        cls, checkpoint: PathLike | BytesLike, *, name: str | None = None, step: int | None = None,
    ) -> Artifact:
        """Build a model artifact from bytes or a filesystem path."""
        artifact = cls(name or "model-checkpoint", "model", step=step)
        if isinstance(checkpoint, (bytes, bytearray, memoryview)):
            artifact.add_bytes(checkpoint, name="checkpoint.bin")
            return artifact
        if not isinstance(checkpoint, (str, Path)):
            raise TypeError("checkpoint must be a path string, Path, or bytes-like object")
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")
        if path.is_file():
            artifact.add_file(path)
        elif path.is_dir():
            artifact.add_dir(path)
        else:
            raise ValueError(f"checkpoint path must point to a file or directory: {path}")
        return artifact

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
        if not isinstance(obj, (Audio, Html, Image, Video)):
            raise TypeError("obj must be an underfit media object implementing the Media protocol")

        default_name = self._default_media_name(obj)
        artifact_path = self._resolve_artifact_path(default_name, name)
        if name is None:
            self._next_media_index += 1
        self._upload_files.append(self._build_media_upload(obj, artifact_path))
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
        self._upload_files.append(ArtifactDataUpload(path=artifact_path, data=bytes(data)))
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

    def _build_media_upload(self, payload: Media, artifact_path: str) -> ArtifactUpload:
        if isinstance(payload, (Audio, Image, Video)):
            return ArtifactDataUpload(path=artifact_path, data=payload.data)
        if isinstance(payload, Html):
            return ArtifactDataUpload(path=artifact_path, data=payload.data)
        raise TypeError("payload must be an underfit media payload")

    def _default_media_name(self, payload: Media) -> str:
        suffix = ""
        if isinstance(payload, Html):
            suffix = ".html"
        elif file_type := getattr(payload, "file_type", None):
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
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
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


def _git_output(repo_path: Path, args: list[str], *, ok_returncodes: tuple[int, ...] = (0,)) -> bytes:
    try:
        result = subprocess.run(["git", *args], cwd=repo_path, capture_output=True, check=False)  # noqa: S603,S607
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed") from exc
    if result.returncode not in ok_returncodes:
        command = " ".join(["git", *args])
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"{command} failed with exit code {result.returncode}")
    return result.stdout


def _git_text(repo_path: Path, args: list[str], *, ok_returncodes: tuple[int, ...] = (0,)) -> str:
    return _git_output(repo_path, args, ok_returncodes=ok_returncodes).decode("utf-8").strip()


def _join_bytes(chunks: list[bytes]) -> bytes:
    output = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        if output and not output.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(chunk)
    return bytes(output)
