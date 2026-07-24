/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
use std::time::SystemTime;

use crate::FinalizerError;

const DIGEST_DOMAIN: &[u8] = b"cuda-oxide/artifact-finalizer/digest/v1";
// Bump this recipe version whenever tool invocation, option translation,
// input ordering, output validation, or other output-affecting semantics
// change. Cache keys and the cargo-oxide/backend handshake rely on it.
const RECIPE: &[u8] = b"cuda-oxide/artifact-finalizer/recipe/v1";

/// Exact compiler inputs discovered alongside the loaded CUDA tools.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolProvenance {
    /// SHA-256 of the exact loaded libNVVM file, if it can be proven.
    pub libnvvm_sha256: Option<[u8; 32]>,
    /// SHA-256 of the exact loaded nvJitLink file, if it can be proven.
    pub nvjitlink_sha256: Option<[u8; 32]>,
    /// SHA-256 of the exact libdevice bytes added to libNVVM.
    pub libdevice_sha256: [u8; 32],
}

/// Stable identity of the finalizer algorithm itself.
pub fn recipe_digest() -> [u8; 32] {
    Sha256::digest(RECIPE).into()
}

pub(crate) fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Run one CUDA-tool operation between exact checks of its retained DSO.
///
/// An unavailable initial digest is allowed for runtime fallback, whose cache
/// is disabled separately. When an exact digest exists (and therefore may be
/// part of Cargo's fingerprint or a cache key), the complete retained file is
/// rehashed immediately before and after the operation. The post-check runs
/// even when the tool call itself returned an error.
pub(crate) fn with_revalidated_tool_identity<T>(
    tool: &'static str,
    expected: Option<[u8; 32]>,
    mut current_digest: impl FnMut() -> Option<[u8; 32]>,
    operation: impl FnOnce() -> Result<T, FinalizerError>,
) -> Result<T, FinalizerError> {
    let Some(expected) = expected else {
        return operation();
    };
    if current_digest() != Some(expected) {
        return Err(FinalizerError::ToolIdentityChanged { tool });
    }

    let result = operation();
    if current_digest() != Some(expected) {
        return Err(FinalizerError::ToolIdentityChanged { tool });
    }
    result
}

/// Hash a retained CUDA-tool descriptor and reject a concurrent replacement.
pub(crate) fn digest_file_handle(file: &File) -> io::Result<[u8; 32]> {
    digest_file_handle_with_post_read(file, || {})
}

fn digest_file_handle_with_post_read(
    file: &File,
    post_read: impl FnOnce(),
) -> io::Result<[u8; 32]> {
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "tool fingerprint input is not a regular file",
        ));
    }
    let snapshot = FileSnapshot::capture(&metadata)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;

        let mut offset = 0_u64;
        loop {
            let read = match file.read_at(&mut buffer, offset) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            offset = offset
                .checked_add(read as u64)
                .ok_or_else(|| io::Error::other("tool file length overflow"))?;
        }
    }

    #[cfg(not(unix))]
    {
        let mut reader = file.try_clone()?;
        reader.seek(SeekFrom::Start(0))?;
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }

    let digest = hasher.finalize().into();
    post_read();
    if FileSnapshot::capture(&file.metadata()?)? != snapshot {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tool file changed while it was fingerprinted",
        ));
    }
    Ok(digest)
}

#[derive(Debug, Eq, PartialEq)]
struct FileSnapshot {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_time: (i64, i64),
}

impl FileSnapshot {
    fn capture(metadata: &fs::Metadata) -> io::Result<Self> {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_time: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

/// Unambiguous ordered digest used for recipes, provenance, and artifacts.
pub(crate) struct StableDigest {
    hasher: Sha256,
}

impl StableDigest {
    pub(crate) fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(DIGEST_DOMAIN);
        Self { hasher }
    }

    pub(crate) fn field(mut self, tag: &str, value: impl AsRef<[u8]>) -> Self {
        let tag = tag.as_bytes();
        let value = value.as_ref();
        self.hasher.update([1]);
        self.hasher.update(length_prefix(tag.len()));
        self.hasher.update(tag);
        self.hasher.update(length_prefix(value.len()));
        self.hasher.update(value);
        self
    }

    pub(crate) fn finish(mut self) -> [u8; 32] {
        self.hasher.update([0xff]);
        self.hasher.finalize().into()
    }
}

fn length_prefix(length: usize) -> [u8; 8] {
    u64::try_from(length)
        .expect("digest fields cannot exceed u64::MAX bytes")
        .to_be_bytes()
}

pub(crate) fn common_provenance_digest(
    libnvvm: &[u8; 32],
    nvjitlink: &[u8; 32],
    libdevice: &[u8; 32],
) -> [u8; 32] {
    StableDigest::new()
        .field("recipe", recipe_digest())
        .field("libnvvm-sha256", libnvvm)
        .field("libnvjitlink-sha256", nvjitlink)
        .field("libdevice-sha256", libdevice)
        .finish()
}

pub(crate) fn compiler_provenance_digest(libnvvm: &[u8; 32], libdevice: &[u8; 32]) -> [u8; 32] {
    StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"nvvm-ir-to-ltoir")
        .field("libnvvm-sha256", libnvvm)
        .field("libdevice-sha256", libdevice)
        .finish()
}

pub(crate) fn linker_provenance_digest(nvjitlink: &[u8; 32]) -> [u8; 32] {
    StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ltoir-to-output")
        .field("libnvjitlink-sha256", nvjitlink)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn provenance_is_route_specific_and_content_sensitive() {
        let nvvm = [1; 32];
        let linker = [2; 32];
        let libdevice = [3; 32];
        assert_ne!(
            compiler_provenance_digest(&nvvm, &libdevice),
            linker_provenance_digest(&linker)
        );
        assert_ne!(
            common_provenance_digest(&nvvm, &linker, &libdevice),
            common_provenance_digest(&nvvm, &linker, &[4; 32])
        );
    }

    #[test]
    fn stable_digest_distinguishes_field_boundaries_and_order() {
        let left = StableDigest::new()
            .field("input", b"ab")
            .field("input", b"c")
            .finish();
        let different_boundaries = StableDigest::new()
            .field("input", b"a")
            .field("input", b"bc")
            .finish();
        let reversed = StableDigest::new()
            .field("input", b"c")
            .field("input", b"ab")
            .finish();
        assert_ne!(left, different_boundaries);
        assert_ne!(left, reversed);
    }

    #[test]
    fn post_call_tool_change_rejects_the_operation_result() {
        let expected = [7; 32];
        let changed = [8; 32];
        let checks = Cell::new(0_u32);
        let operation_calls = Cell::new(0_u32);

        let error = with_revalidated_tool_identity(
            "test CUDA tool",
            Some(expected),
            || {
                let check = checks.get();
                checks.set(check + 1);
                Some(if check == 0 { expected } else { changed })
            },
            || {
                operation_calls.set(operation_calls.get() + 1);
                Ok(b"must not be accepted".to_vec())
            },
        )
        .expect_err("a post-call identity change must discard successful output");

        assert!(matches!(
            error,
            FinalizerError::ToolIdentityChanged {
                tool: "test CUDA tool"
            }
        ));
        assert_eq!(checks.get(), 2, "identity must be checked on both sides");
        assert_eq!(operation_calls.get(), 1, "the seam models a post-call race");
    }

    #[test]
    fn tool_digest_stays_bound_to_open_file_after_path_replacement() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-artifact-finalizer-provenance-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        let replacement = directory.join("replacement.so");
        std::fs::write(&path, b"tool version one").unwrap();
        std::fs::write(&replacement, b"tool version two").unwrap();

        let opened = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        std::fs::rename(&replacement, &path).unwrap();

        assert_eq!(
            digest_file_handle(&opened).unwrap(),
            digest_bytes(b"tool version one")
        );
        assert_eq!(
            digest_file_handle(&File::open(&path).unwrap()).unwrap(),
            digest_bytes(b"tool version two")
        );

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn tool_digest_changes_after_in_place_content_change() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-artifact-finalizer-in-place-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        std::fs::write(&path, b"tool bytes version one").unwrap();
        #[cfg(unix)]
        let original_inode = {
            use std::os::unix::fs::MetadataExt;
            path.metadata().unwrap().ino()
        };
        let original = digest_file_handle(&File::open(&path).unwrap()).unwrap();

        std::fs::write(&path, b"tool bytes version two").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(path.metadata().unwrap().ino(), original_inode);
        }
        let changed = digest_file_handle(&File::open(&path).unwrap()).unwrap();

        assert_eq!(original, digest_bytes(b"tool bytes version one"));
        assert_eq!(changed, digest_bytes(b"tool bytes version two"));
        assert_ne!(original, changed);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn metadata_change_between_read_and_validation_rejects_digest() {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "cuda-artifact-finalizer-mid-hash-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        std::fs::write(&path, b"tool bytes before hash").unwrap();
        let opened = File::open(&path).unwrap();

        let error = digest_file_handle_with_post_read(&opened, || {
            std::fs::write(&path, b"tool bytes changed during hash and now longer").unwrap();
        })
        .unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(
            error.to_string(),
            "tool file changed while it was fingerprinted"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
