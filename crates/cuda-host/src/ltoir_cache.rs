/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Content-addressed cache support for the libNVVM + nvJitLink pipeline.
//!
//! A cache hit is accepted only after the entry manifest, cubin, and optional
//! LTOIR have all been checked. Callers always receive owned bytes and must
//! load those bytes, rather than reopening a cache path after validation.

use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
#[cfg(not(unix))]
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

const CACHE_KEY_DOMAIN: &[u8] = b"cuda-oxide/ltoir-cubin-cache/key/v1";
const CACHE_DIRECTORY: &str = "ltoir-cubin-cache";
const CACHE_VERSION: &str = "v1";
const MANIFEST_FILE: &str = "manifest.bin";
const CUBIN_FILE: &str = "image.cubin";
const LTOIR_FILE: &str = "image.ltoir";
const MANIFEST_MAGIC: &[u8] = b"cuda-oxide-ltoir-cache-entry\0";
const MANIFEST_VERSION: u32 = 1;
const DIGEST_LENGTH: usize = 32;
const MAX_CACHE_ARTIFACT_LENGTH: u64 = 1 << 30;
const ELF64_HEADER_LENGTH: usize = 64;
const ELF64_PROGRAM_HEADER_LENGTH: u16 = 56;
const ELF64_SECTION_HEADER_LENGTH: u16 = 64;
const MANIFEST_LENGTH: usize =
    MANIFEST_MAGIC.len() + 4 + DIGEST_LENGTH + 8 + DIGEST_LENGTH + 1 + 8 + DIGEST_LENGTH;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
#[derive(Debug, PartialEq, Eq)]
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

/// Hash bytes exactly as they appear in a cache input.
pub(crate) fn digest_bytes(bytes: &[u8]) -> [u8; DIGEST_LENGTH] {
    Sha256::digest(bytes).into()
}

/// Hash the complete contents of a file.
///
/// Tool fingerprints should be taken from the same loaded library that is
/// subsequently used to build the artifact. If that exact file cannot be
/// identified or read, callers should skip the cache.
#[cfg(test)]
pub(crate) fn digest_file(path: &Path) -> io::Result<[u8; DIGEST_LENGTH]> {
    let file = File::open(path)?;
    digest_file_handle(&file)
}

/// Hash the complete contents of an already-open file without changing its
/// shared cursor.
///
/// The retained CUDA-tool descriptor is the provenance boundary: pathname
/// replacement cannot redirect these reads to another inode.
pub(crate) fn digest_file_handle(file: &File) -> io::Result<[u8; DIGEST_LENGTH]> {
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
    let after = FileSnapshot::capture(&file.metadata()?)?;
    if after != snapshot {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tool file changed while it was fingerprinted",
        ));
    }
    Ok(digest)
}

/// Builds an unambiguous cache key from ordered, tagged fields.
///
/// Both the tag and value are length-prefixed. Field order is significant,
/// which preserves option ordering and prevents inputs such as `(ab, c)` and
/// `(a, bc)` from producing the same byte stream before hashing.
pub(crate) struct CacheKeyBuilder {
    hasher: Sha256,
}

impl CacheKeyBuilder {
    pub(crate) fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(CACHE_KEY_DOMAIN);
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

    pub(crate) fn finish(mut self) -> [u8; DIGEST_LENGTH] {
        self.hasher.update([0xff]);
        self.hasher.finalize().into()
    }
}

impl Default for CacheKeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn length_prefix(length: usize) -> [u8; 8] {
    u64::try_from(length)
        .expect("cache key fields cannot exceed u64::MAX bytes")
        .to_be_bytes()
}

/// Fresh artifacts returned by the expensive compiler/linker pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BuiltArtifacts {
    pub(crate) cubin: Vec<u8>,
    pub(crate) ltoir: Option<Vec<u8>>,
}

impl BuiltArtifacts {
    pub(crate) fn new(cubin: Vec<u8>, ltoir: Option<Vec<u8>>) -> Self {
        Self { cubin, ltoir }
    }
}

/// Owned, fully verified artifacts returned to the loader.
///
/// The immutable paths are present only when a complete cache entry was read
/// or published successfully. They are informational and useful for copying
/// compatibility sidecars; the loader should consume the owned bytes.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CacheResult {
    pub(crate) cubin: Vec<u8>,
    pub(crate) ltoir: Option<Vec<u8>>,
    pub(crate) immutable_cubin_path: Option<PathBuf>,
    pub(crate) immutable_ltoir_path: Option<PathBuf>,
    pub(crate) cache_hit: bool,
}

impl CacheResult {
    fn uncached(artifacts: BuiltArtifacts) -> Self {
        Self {
            cubin: artifacts.cubin,
            ltoir: artifacts.ltoir,
            immutable_cubin_path: None,
            immutable_ltoir_path: None,
            cache_hit: false,
        }
    }

    fn from_stored(stored: StoredArtifacts, cache_hit: bool) -> Self {
        Self {
            cubin: stored.artifacts.cubin,
            ltoir: stored.artifacts.ltoir,
            immutable_cubin_path: Some(stored.cubin_path),
            immutable_ltoir_path: stored.ltoir_path,
            cache_hit,
        }
    }
}

/// Return a verified cache entry or build it once under a per-key file lock.
///
/// `source_dir` is the directory containing the source artifact. Cache data is
/// kept below `source_dir/.oxide-artifacts/ltoir-cubin-cache/v1/`. Every cache
/// I/O error is treated as an optimization failure: the builder still runs and
/// its result is returned with no immutable path. Errors from `build` itself
/// are always propagated.
pub(crate) fn cache_or_build<E, F>(
    source_dir: &Path,
    key: &[u8; DIGEST_LENGTH],
    build: F,
) -> Result<CacheResult, E>
where
    F: FnOnce() -> Result<BuiltArtifacts, E>,
{
    let paths = CachePaths::new(source_dir, key);

    if let Ok(Some(stored)) = read_valid_entry(&paths.entry, key) {
        return Ok(CacheResult::from_stored(stored, true));
    }

    let _lock = match acquire_key_lock(&paths) {
        Ok(lock) => lock,
        Err(_) => return build().map(CacheResult::uncached),
    };

    // Another process may have published this key while this process waited.
    if let Ok(Some(stored)) = read_valid_entry(&paths.entry, key) {
        return Ok(CacheResult::from_stored(stored, true));
    }

    let artifacts = build()?;
    if !looks_like_cubin(&artifacts.cubin) {
        return Ok(CacheResult::uncached(artifacts));
    }

    match publish_entry(&paths, key, &artifacts) {
        Ok(stored) => Ok(CacheResult::from_stored(stored, false)),
        Err(_) => Ok(CacheResult::uncached(artifacts)),
    }
}

struct CachePaths {
    entries: PathBuf,
    locks: PathBuf,
    temporary: PathBuf,
    entry: PathBuf,
    lock: PathBuf,
}

impl CachePaths {
    fn new(source_dir: &Path, key: &[u8; DIGEST_LENGTH]) -> Self {
        let root = source_dir
            .join(".oxide-artifacts")
            .join(CACHE_DIRECTORY)
            .join(CACHE_VERSION);
        let entries = root.join("entries");
        let locks = root.join("locks");
        let temporary = root.join("tmp");
        let key_hex = hex_digest(key);

        Self {
            entry: entries.join(&key_hex),
            lock: locks.join(format!("{key_hex}.lock")),
            entries,
            locks,
            temporary,
        }
    }
}

fn acquire_key_lock(paths: &CachePaths) -> io::Result<File> {
    fs::create_dir_all(&paths.entries)?;
    fs::create_dir_all(&paths.locks)?;
    fs::create_dir_all(&paths.temporary)?;

    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&paths.lock)?;
    lock.lock()?;
    Ok(lock)
}

#[derive(Debug)]
struct StoredArtifacts {
    artifacts: BuiltArtifacts,
    cubin_path: PathBuf,
    ltoir_path: Option<PathBuf>,
}

fn read_valid_entry(
    entry: &Path,
    expected_key: &[u8; DIGEST_LENGTH],
) -> io::Result<Option<StoredArtifacts>> {
    let metadata = match fs::symlink_metadata(entry) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    if !metadata.file_type().is_dir() {
        return Ok(None);
    }

    let manifest_path = entry.join(MANIFEST_FILE);
    let Some(manifest_bytes) = read_entry_file(&manifest_path, MANIFEST_LENGTH as u64)? else {
        return Ok(None);
    };
    let Some(manifest) = Manifest::decode(&manifest_bytes) else {
        return Ok(None);
    };
    if &manifest.key != expected_key {
        return Ok(None);
    }

    let expected_names: &[&str] = if manifest.has_ltoir {
        &[MANIFEST_FILE, CUBIN_FILE, LTOIR_FILE]
    } else {
        &[MANIFEST_FILE, CUBIN_FILE]
    };
    if !directory_contains_exact_regular_files(entry, expected_names)? {
        return Ok(None);
    }

    let cubin_path = entry.join(CUBIN_FILE);
    let Some(cubin) = read_entry_file(&cubin_path, manifest.cubin_length)? else {
        return Ok(None);
    };
    if digest_bytes(&cubin) != manifest.cubin_digest || !looks_like_cubin(&cubin) {
        return Ok(None);
    }

    let (ltoir, ltoir_path) = if manifest.has_ltoir {
        let path = entry.join(LTOIR_FILE);
        let Some(bytes) = read_entry_file(&path, manifest.ltoir_length)? else {
            return Ok(None);
        };
        if digest_bytes(&bytes) != manifest.ltoir_digest {
            return Ok(None);
        }
        (Some(bytes), Some(path))
    } else {
        if manifest.ltoir_length != 0 || manifest.ltoir_digest != [0; DIGEST_LENGTH] {
            return Ok(None);
        }
        (None, None)
    };

    Ok(Some(StoredArtifacts {
        artifacts: BuiltArtifacts { cubin, ltoir },
        cubin_path,
        ltoir_path,
    }))
}

fn read_entry_file(path: &Path, expected_length: u64) -> io::Result<Option<Vec<u8>>> {
    match read_regular_file_exact(path, expected_length) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::InvalidData
                    | io::ErrorKind::UnexpectedEof
                    | io::ErrorKind::IsADirectory
            ) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn read_regular_file_exact(path: &Path, expected_length: u64) -> io::Result<Vec<u8>> {
    if expected_length > MAX_CACHE_ARTIFACT_LENGTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact exceeds the maximum supported size",
        ));
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact is not a regular file",
        ));
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // O_NOFOLLOW prevents a final symlink from escaping the cache, while
        // O_NONBLOCK prevents a raced FIFO/device replacement from blocking.
        options.custom_flags(0o400000 | 0o4000);
    }
    let file = options.open(path).map_err(|error| {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if error.raw_os_error() == Some(40) {
            return io::Error::new(io::ErrorKind::InvalidData, "cache artifact is a symlink");
        }
        error
    })?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != expected_length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact is not a regular file of the expected size",
        ));
    }

    let expected_length = usize::try_from(expected_length).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact does not fit in memory on this platform",
        )
    })?;
    let mut bytes = Vec::with_capacity(expected_length);
    file.take(expected_length as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() != expected_length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cache artifact changed size while it was read",
        ));
    }
    Ok(bytes)
}

fn directory_contains_exact_regular_files(dir: &Path, expected: &[&str]) -> io::Result<bool> {
    let mut names = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Ok(false);
        }
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            return Ok(false);
        };
        names.push(name);
    }

    names.sort_unstable();
    let mut expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    expected.sort_unstable();
    Ok(names == expected)
}

#[derive(Debug)]
struct Manifest {
    key: [u8; DIGEST_LENGTH],
    cubin_length: u64,
    cubin_digest: [u8; DIGEST_LENGTH],
    has_ltoir: bool,
    ltoir_length: u64,
    ltoir_digest: [u8; DIGEST_LENGTH],
}

impl Manifest {
    fn for_artifacts(key: &[u8; DIGEST_LENGTH], artifacts: &BuiltArtifacts) -> io::Result<Self> {
        let cubin_length = u64::try_from(artifacts.cubin.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "cubin exceeds cache format limit",
            )
        })?;
        if cubin_length > MAX_CACHE_ARTIFACT_LENGTH {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cubin exceeds cache size limit",
            ));
        }
        let (has_ltoir, ltoir_length, ltoir_digest) = match &artifacts.ltoir {
            Some(ltoir) => {
                let length = u64::try_from(ltoir.len()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "LTOIR exceeds cache format limit",
                    )
                })?;
                if length > MAX_CACHE_ARTIFACT_LENGTH {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "LTOIR exceeds cache size limit",
                    ));
                }
                (true, length, digest_bytes(ltoir))
            }
            None => (false, 0, [0; DIGEST_LENGTH]),
        };

        Ok(Self {
            key: *key,
            cubin_length,
            cubin_digest: digest_bytes(&artifacts.cubin),
            has_ltoir,
            ltoir_length,
            ltoir_digest,
        })
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MANIFEST_LENGTH);
        bytes.extend_from_slice(MANIFEST_MAGIC);
        bytes.extend_from_slice(&MANIFEST_VERSION.to_be_bytes());
        bytes.extend_from_slice(&self.key);
        bytes.extend_from_slice(&self.cubin_length.to_be_bytes());
        bytes.extend_from_slice(&self.cubin_digest);
        bytes.push(u8::from(self.has_ltoir));
        bytes.extend_from_slice(&self.ltoir_length.to_be_bytes());
        bytes.extend_from_slice(&self.ltoir_digest);
        debug_assert_eq!(bytes.len(), MANIFEST_LENGTH);
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != MANIFEST_LENGTH || !bytes.starts_with(MANIFEST_MAGIC) {
            return None;
        }

        let mut cursor = MANIFEST_MAGIC.len();
        let version = take_array::<4>(bytes, &mut cursor).map(u32::from_be_bytes)?;
        if version != MANIFEST_VERSION {
            return None;
        }
        let key = take_array::<DIGEST_LENGTH>(bytes, &mut cursor)?;
        let cubin_length = take_array::<8>(bytes, &mut cursor).map(u64::from_be_bytes)?;
        let cubin_digest = take_array::<DIGEST_LENGTH>(bytes, &mut cursor)?;
        let has_ltoir = match *bytes.get(cursor)? {
            0 => false,
            1 => true,
            _ => return None,
        };
        cursor += 1;
        let ltoir_length = take_array::<8>(bytes, &mut cursor).map(u64::from_be_bytes)?;
        let ltoir_digest = take_array::<DIGEST_LENGTH>(bytes, &mut cursor)?;
        if cursor != bytes.len() {
            return None;
        }

        Some(Self {
            key,
            cubin_length,
            cubin_digest,
            has_ltoir,
            ltoir_length,
            ltoir_digest,
        })
    }
}

fn take_array<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Option<[u8; N]> {
    let end = cursor.checked_add(N)?;
    let value = bytes.get(*cursor..end)?.try_into().ok()?;
    *cursor = end;
    Some(value)
}

fn publish_entry(
    paths: &CachePaths,
    key: &[u8; DIGEST_LENGTH],
    artifacts: &BuiltArtifacts,
) -> io::Result<StoredArtifacts> {
    let manifest = Manifest::for_artifacts(key, artifacts)?;
    let mut temporary = PendingDirectory::create(&paths.temporary, key)?;

    write_synced_file(&temporary.path.join(CUBIN_FILE), &artifacts.cubin)?;
    if let Some(ltoir) = &artifacts.ltoir {
        write_synced_file(&temporary.path.join(LTOIR_FILE), ltoir)?;
    }
    write_synced_file(&temporary.path.join(MANIFEST_FILE), &manifest.encode())?;
    sync_directory(&temporary.path)?;

    match read_valid_entry(&paths.entry, key) {
        Ok(Some(existing)) if existing.artifacts == *artifacts => return Ok(existing),
        Ok(Some(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "cache key already contains different verified output",
            ));
        }
        Ok(None) => remove_cache_path_if_present(&paths.entry)?,
        Err(error) => return Err(error),
    }

    fs::rename(&temporary.path, &paths.entry)?;
    temporary.published = true;
    sync_directory(&paths.entries)?;

    // Read through the same verification path used for future hits. This also
    // ensures the returned immutable paths refer to the bytes being returned.
    read_valid_entry(&paths.entry, key)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "published cache entry failed verification",
        )
    })
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

fn remove_cache_path_if_present(path: &Path) -> io::Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };

    if metadata.file_type().is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

struct PendingDirectory {
    path: PathBuf,
    published: bool,
}

impl PendingDirectory {
    fn create(root: &Path, key: &[u8; DIGEST_LENGTH]) -> io::Result<Self> {
        let key = hex_digest(key);
        for _ in 0..128 {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(".{key}.{}.{}.tmp", std::process::id(), id));
            match fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        published: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }

        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique cache staging directory",
        ))
    }
}

impl Drop for PendingDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn looks_like_cubin(bytes: &[u8]) -> bool {
    // nvJitLink cubins are 64-bit, little-endian ET_EXEC ELF files for
    // EM_CUDA (190). Validate all declared tables and file-backed ranges so a
    // truncated ELF prefix never becomes a reusable cache entry.
    if bytes.len() < ELF64_HEADER_LENGTH
        || !bytes.starts_with(b"\x7fELF")
        || bytes[4] != 2
        || bytes[5] != 1
        || bytes[6] != 1
        || read_u16(bytes, 16) != Some(2)
        || read_u16(bytes, 18) != Some(190)
        // CUDA cubins use non-zero, CUDA-specific e_version values rather
        // than consistently using the generic ELF EV_CURRENT value (1).
        || read_u32(bytes, 20).is_none_or(|version| version == 0)
        || read_u16(bytes, 52) != Some(ELF64_HEADER_LENGTH as u16)
    {
        return false;
    }

    let Some(program_offset) = read_u64(bytes, 32) else {
        return false;
    };
    let Some(section_offset) = read_u64(bytes, 40) else {
        return false;
    };
    let Some(program_entry_size) = read_u16(bytes, 54) else {
        return false;
    };
    let Some(program_count) = read_u16(bytes, 56) else {
        return false;
    };
    let Some(section_entry_size) = read_u16(bytes, 58) else {
        return false;
    };
    let Some(section_count) = read_u16(bytes, 60) else {
        return false;
    };
    let Some(section_name_index) = read_u16(bytes, 62) else {
        return false;
    };

    if program_count == u16::MAX || (program_count == 0 && section_count == 0) {
        return false;
    }
    let program_table = if program_count == 0 {
        if program_offset != 0 || !matches!(program_entry_size, 0 | ELF64_PROGRAM_HEADER_LENGTH) {
            return false;
        }
        None
    } else {
        if program_entry_size != ELF64_PROGRAM_HEADER_LENGTH {
            return false;
        }
        table_bounds(
            program_offset,
            program_entry_size,
            program_count,
            bytes.len(),
        )
    };
    let section_table = if section_count == 0 {
        // Extended section numbering is unnecessary for cubins and would
        // require trusting section zero before its bounds were known.
        if section_offset != 0
            || section_name_index != 0
            || !matches!(section_entry_size, 0 | ELF64_SECTION_HEADER_LENGTH)
        {
            return false;
        }
        None
    } else {
        if section_entry_size != ELF64_SECTION_HEADER_LENGTH
            || (section_name_index != 0 && section_name_index >= section_count)
        {
            return false;
        }
        table_bounds(
            section_offset,
            section_entry_size,
            section_count,
            bytes.len(),
        )
    };
    if program_count != 0 && program_table.is_none()
        || section_count != 0 && section_table.is_none()
    {
        return false;
    }

    if let Some((start, _)) = program_table {
        for index in 0..usize::from(program_count) {
            let header = start + index * usize::from(program_entry_size);
            let Some(file_offset) = read_u64(bytes, header + 8) else {
                return false;
            };
            let Some(file_size) = read_u64(bytes, header + 32) else {
                return false;
            };
            if !file_range_is_valid(file_offset, file_size, bytes.len()) {
                return false;
            }
        }
    }

    if let Some((start, _)) = section_table {
        for index in 0..usize::from(section_count) {
            let header = start + index * usize::from(section_entry_size);
            let Some(section_type) = read_u32(bytes, header + 4) else {
                return false;
            };
            let Some(file_offset) = read_u64(bytes, header + 24) else {
                return false;
            };
            let Some(file_size) = read_u64(bytes, header + 32) else {
                return false;
            };
            // SHT_NOBITS occupies memory but has no bytes in the ELF image.
            if section_type != 8 && !file_range_is_valid(file_offset, file_size, bytes.len()) {
                return false;
            }
        }
    }

    true
}

fn table_bounds(
    offset: u64,
    entry_size: u16,
    count: u16,
    file_len: usize,
) -> Option<(usize, usize)> {
    if offset < ELF64_HEADER_LENGTH as u64 {
        return None;
    }
    let length = u64::from(entry_size).checked_mul(u64::from(count))?;
    let end = offset.checked_add(length)?;
    if end > file_len as u64 {
        return None;
    }
    Some((usize::try_from(offset).ok()?, usize::try_from(end).ok()?))
}

fn file_range_is_valid(offset: u64, length: u64, file_len: usize) -> bool {
    offset
        .checked_add(length)
        .is_some_and(|end| end <= file_len as u64)
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn hex_digest(digest: &[u8; DIGEST_LENGTH]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(DIGEST_LENGTH * 2);
    for byte in digest {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            for _ in 0..128 {
                let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "cuda-oxide-ltoir-cache-{name}-{}-{id}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("failed to create test directory: {error}"),
                }
            }
            panic!("failed to allocate unique test directory");
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fake_cubin_with_version(marker: u8, elf_version: u32) -> Vec<u8> {
        let mut cubin = vec![0_u8; 120];
        cubin[..4].copy_from_slice(b"\x7fELF");
        cubin[4] = 2;
        cubin[5] = 1;
        cubin[6] = 1;
        cubin[15] = marker;
        cubin[16..18].copy_from_slice(&2_u16.to_le_bytes());
        cubin[18..20].copy_from_slice(&190_u16.to_le_bytes());
        cubin[20..24].copy_from_slice(&elf_version.to_le_bytes());
        cubin[32..40].copy_from_slice(&64_u64.to_le_bytes());
        cubin[52..54].copy_from_slice(&(ELF64_HEADER_LENGTH as u16).to_le_bytes());
        cubin[54..56].copy_from_slice(&ELF64_PROGRAM_HEADER_LENGTH.to_le_bytes());
        cubin[56..58].copy_from_slice(&1_u16.to_le_bytes());

        // One PT_LOAD program header spanning the small synthetic file.
        let cubin_len = cubin.len() as u64;
        cubin[64..68].copy_from_slice(&1_u32.to_le_bytes());
        cubin[72..80].copy_from_slice(&0_u64.to_le_bytes());
        cubin[96..104].copy_from_slice(&cubin_len.to_le_bytes());
        cubin[104..112].copy_from_slice(&cubin_len.to_le_bytes());
        cubin[112..120].copy_from_slice(&8_u64.to_le_bytes());
        cubin
    }

    fn fake_cubin(marker: u8) -> Vec<u8> {
        fake_cubin_with_version(marker, 1)
    }

    fn artifacts(marker: u8) -> BuiltArtifacts {
        BuiltArtifacts::new(fake_cubin(marker), Some(vec![b'L', b'T', b'O', marker]))
    }

    fn truncated_cubin_prefix() -> Vec<u8> {
        let mut cubin = vec![0_u8; 20];
        cubin[..4].copy_from_slice(b"\x7fELF");
        cubin[4] = 2;
        cubin[5] = 1;
        cubin[6] = 1;
        cubin[16..18].copy_from_slice(&2_u16.to_le_bytes());
        cubin[18..20].copy_from_slice(&190_u16.to_le_bytes());
        cubin
    }

    fn test_key() -> [u8; DIGEST_LENGTH] {
        CacheKeyBuilder::new()
            .field("pipeline", b"test-v1")
            .field("route", b"nvvm-ir-to-cubin")
            .field("arch", b"sm_80")
            .finish()
    }

    #[test]
    fn sha256_matches_published_vectors_and_incremental_updates() {
        assert_eq!(
            hex_digest(&digest_bytes(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_digest(&digest_bytes(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(&digest_bytes(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex_digest(&digest_bytes(&vec![b'a'; 1_000_000])),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );

        let mut incremental = Sha256::new();
        incremental.update(b"a");
        incremental.update(b"b");
        incremental.update(b"c");
        let incremental: [u8; DIGEST_LENGTH] = incremental.finalize().into();
        assert_eq!(incremental, digest_bytes(b"abc"));

        for length in [55, 56, 63, 64, 65, 127, 128, 129] {
            let bytes = (0..length).map(|value| value as u8).collect::<Vec<_>>();
            let mut chunked = Sha256::new();
            for chunk in bytes.chunks(7) {
                chunked.update(chunk);
            }
            let chunked: [u8; DIGEST_LENGTH] = chunked.finalize().into();
            assert_eq!(chunked, digest_bytes(&bytes), "length {length}");
        }
    }

    #[test]
    fn digest_file_tracks_in_place_changes_and_inode_replacement() {
        let dir = TestDirectory::new("digest-file");
        let path = dir.path().join("tool.so");
        fs::write(&path, b"exact tool bytes").unwrap();
        let first = digest_file(&path).unwrap();
        assert_eq!(first, digest_bytes(b"exact tool bytes"));
        assert_eq!(digest_file(&path).unwrap(), first);

        fs::write(&path, b"changed-tool-dat").unwrap();
        let changed = digest_file(&path).unwrap();
        assert_eq!(changed, digest_bytes(b"changed-tool-dat"));
        assert_ne!(
            changed, first,
            "same-path content changes must change the digest"
        );

        let replacement = dir.path().join("replacement.so");
        fs::write(&replacement, b"replaced-tool-da").unwrap();
        fs::remove_file(&path).unwrap();
        fs::rename(&replacement, &path).unwrap();
        let replaced = digest_file(&path).unwrap();
        assert_eq!(replaced, digest_bytes(b"replaced-tool-da"));
        assert_ne!(
            replaced, changed,
            "inode replacement must change the digest"
        );
    }

    #[test]
    fn digest_file_handle_stays_bound_to_open_inode_after_replacement() {
        let dir = TestDirectory::new("digest-open-file");
        let path = dir.path().join("tool.so");
        let replacement = dir.path().join("replacement.so");
        fs::write(&path, b"tool version one").unwrap();
        fs::write(&replacement, b"tool version two").unwrap();

        let opened = File::open(&path).unwrap();
        fs::remove_file(&path).unwrap();
        fs::rename(&replacement, &path).unwrap();

        assert_eq!(
            digest_file_handle(&opened).unwrap(),
            digest_bytes(b"tool version one")
        );
        assert_eq!(
            digest_file(&path).unwrap(),
            digest_bytes(b"tool version two")
        );
    }

    #[test]
    fn cubin_validation_rejects_truncated_headers_and_out_of_bounds_tables() {
        assert!(!looks_like_cubin(&truncated_cubin_prefix()));
        assert!(looks_like_cubin(&fake_cubin(1)));

        let mut out_of_bounds = fake_cubin(2);
        out_of_bounds[32..40].copy_from_slice(&119_u64.to_le_bytes());
        assert!(!looks_like_cubin(&out_of_bounds));
    }

    #[test]
    fn cubin_validation_accepts_cuda_specific_elf_versions() {
        // Observed in cubins from multiple CUDA toolchain generations.
        for version in [0x73, 0x80] {
            assert!(
                looks_like_cubin(&fake_cubin_with_version(1, version)),
                "CUDA ELF version {version:#x} should be accepted"
            );
        }

        assert!(!looks_like_cubin(&fake_cubin_with_version(1, 0)));
    }

    #[test]
    fn cache_reads_reject_oversized_lengths_before_allocating() {
        let dir = TestDirectory::new("oversized");
        let path = dir.path().join("image.cubin");
        fs::write(&path, b"small").unwrap();
        let error = read_regular_file_exact(&path, MAX_CACHE_ARTIFACT_LENGTH + 1).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn cache_reads_do_not_follow_artifact_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = TestDirectory::new("symlink");
        let outside = dir.path().join("outside");
        let link = dir.path().join("image.cubin");
        fs::write(&outside, b"outside bytes").unwrap();
        symlink(&outside, &link).unwrap();

        assert!(read_regular_file_exact(&link, 13).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn symlinked_cache_artifact_is_a_miss_and_is_repaired() {
        use std::os::unix::fs::symlink;

        let dir = TestDirectory::new("symlink-repair");
        let key = test_key();
        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(20))).unwrap();
        let cubin_path = first.immutable_cubin_path.unwrap();
        let outside = dir.path().join("outside");
        fs::write(&outside, b"outside bytes").unwrap();
        fs::remove_file(&cubin_path).unwrap();
        symlink(&outside, &cubin_path).unwrap();

        let repaired = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(21))).unwrap();
        assert!(!repaired.cache_hit);
        assert_eq!(repaired.cubin, fake_cubin(21));
        assert!(repaired.immutable_cubin_path.is_some());
        assert_eq!(fs::read(outside).unwrap(), b"outside bytes");
    }

    #[test]
    fn key_changes_for_every_input_and_is_unambiguous() {
        let fields: [(&str, &[u8]); 8] = [
            ("pipeline", b"pipeline-v1"),
            ("route", b"nvvm-ir-to-cubin"),
            ("arch", b"sm_80"),
            ("module", b"kernel"),
            ("option", b"-lto"),
            ("source", b"source digest"),
            ("libdevice", b"libdevice digest"),
            ("tools", b"exact tool digests"),
        ];

        let build_key = |changed: Option<usize>| {
            fields
                .iter()
                .enumerate()
                .fold(CacheKeyBuilder::new(), |builder, (index, (tag, value))| {
                    if changed == Some(index) {
                        builder.field(tag, [*value, b" changed"].concat())
                    } else {
                        builder.field(tag, *value)
                    }
                })
                .finish()
        };

        let original = build_key(None);
        let changed = (0..fields.len()).map(|index| build_key(Some(index)));
        let all = std::iter::once(original)
            .chain(changed)
            .collect::<HashSet<_>>();
        assert_eq!(all.len(), fields.len() + 1);

        let left = CacheKeyBuilder::new()
            .field("part", b"ab")
            .field("part", b"c")
            .finish();
        let right = CacheKeyBuilder::new()
            .field("part", b"a")
            .field("part", b"bc")
            .finish();
        let reordered = CacheKeyBuilder::new()
            .field("part", b"c")
            .field("part", b"ab")
            .finish();
        assert_ne!(left, right);
        assert_ne!(left, reordered);
    }

    #[test]
    fn second_request_is_a_verified_hit() {
        let dir = TestDirectory::new("hit");
        let key = test_key();
        let builds = AtomicU64::new(0);

        let first = cache_or_build(dir.path(), &key, || {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(artifacts(1))
        })
        .unwrap();
        assert!(!first.cache_hit);
        assert_eq!(first.cubin, fake_cubin(1));
        assert_eq!(first.ltoir, Some(vec![b'L', b'T', b'O', 1]));
        assert!(
            first
                .immutable_cubin_path
                .as_ref()
                .is_some_and(|path| path.is_file())
        );
        assert!(
            first
                .immutable_ltoir_path
                .as_ref()
                .is_some_and(|path| path.is_file())
        );

        let second = cache_or_build(dir.path(), &key, || {
            builds.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(artifacts(2))
        })
        .unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.cubin, first.cubin);
        assert_eq!(second.ltoir, first.ltoir);
        assert_eq!(second.immutable_cubin_path, first.immutable_cubin_path);
        assert_eq!(second.immutable_ltoir_path, first.immutable_ltoir_path);
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn standalone_ltoir_route_caches_without_intermediate() {
        let dir = TestDirectory::new("without-intermediate");
        let key = test_key();
        let output = BuiltArtifacts::new(fake_cubin(3), None);

        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(output.clone())).unwrap();
        assert_eq!(first.ltoir, None);
        assert_eq!(first.immutable_ltoir_path, None);
        assert!(first.immutable_cubin_path.is_some());

        let second = cache_or_build(dir.path(), &key, || -> Result<_, ()> {
            panic!("valid entry should avoid rebuilding")
        })
        .unwrap();
        assert!(second.cache_hit);
        assert_eq!(second.cubin, output.cubin);
        assert_eq!(second.ltoir, None);
    }

    #[test]
    fn corrupt_cubin_digest_is_a_miss_and_is_repaired() {
        let dir = TestDirectory::new("corrupt-cubin");
        let key = test_key();
        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(4))).unwrap();
        let cubin_path = first.immutable_cubin_path.unwrap();

        let mut corrupted = fake_cubin(4);
        corrupted[32] ^= 0x80;
        fs::write(&cubin_path, corrupted).unwrap();

        let repaired = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(5))).unwrap();
        assert!(!repaired.cache_hit);
        assert_eq!(repaired.cubin, fake_cubin(5));
        assert!(repaired.immutable_cubin_path.is_some());

        let hit = cache_or_build(dir.path(), &key, || -> Result<_, ()> {
            panic!("repaired entry should be reusable")
        })
        .unwrap();
        assert!(hit.cache_hit);
        assert_eq!(hit.cubin, fake_cubin(5));
    }

    #[test]
    fn truncated_ltoir_and_manifest_are_misses() {
        let dir = TestDirectory::new("truncated");
        let key = test_key();
        let first = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(6))).unwrap();
        fs::write(first.immutable_ltoir_path.unwrap(), b"L").unwrap();

        let second = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(7))).unwrap();
        assert!(!second.cache_hit);
        assert_eq!(second.ltoir, Some(vec![b'L', b'T', b'O', 7]));

        let manifest = second
            .immutable_cubin_path
            .as_ref()
            .unwrap()
            .parent()
            .unwrap()
            .join(MANIFEST_FILE);
        fs::write(manifest, &MANIFEST_MAGIC[..8]).unwrap();

        let third = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(8))).unwrap();
        assert!(!third.cache_hit);
        assert_eq!(third.cubin, fake_cubin(8));
    }

    #[test]
    fn complete_entries_swapped_between_keys_are_rejected() {
        let dir = TestDirectory::new("swapped-entries");
        let key_a = CacheKeyBuilder::new().field("source", b"A").finish();
        let key_b = CacheKeyBuilder::new().field("source", b"B").finish();
        cache_or_build(dir.path(), &key_a, || Ok::<_, ()>(artifacts(20))).unwrap();
        cache_or_build(dir.path(), &key_b, || Ok::<_, ()>(artifacts(21))).unwrap();

        let paths_a = CachePaths::new(dir.path(), &key_a);
        let paths_b = CachePaths::new(dir.path(), &key_b);
        let swap = paths_a.entries.join("swap");
        fs::rename(&paths_a.entry, &swap).unwrap();
        fs::rename(&paths_b.entry, &paths_a.entry).unwrap();
        fs::rename(&swap, &paths_b.entry).unwrap();

        let repaired_a = cache_or_build(dir.path(), &key_a, || Ok::<_, ()>(artifacts(22))).unwrap();
        let repaired_b = cache_or_build(dir.path(), &key_b, || Ok::<_, ()>(artifacts(23))).unwrap();
        assert!(!repaired_a.cache_hit);
        assert!(!repaired_b.cache_hit);
        assert_eq!(repaired_a.cubin, fake_cubin(22));
        assert_eq!(repaired_b.cubin, fake_cubin(23));
    }

    #[test]
    fn partial_entries_and_staging_artifacts_are_ignored() {
        let dir = TestDirectory::new("partial");
        let key = test_key();
        let paths = CachePaths::new(dir.path(), &key);
        fs::create_dir_all(&paths.entry).unwrap();
        fs::write(paths.entry.join(CUBIN_FILE), fake_cubin(9)).unwrap();
        fs::create_dir_all(&paths.temporary).unwrap();
        let abandoned = paths.temporary.join("abandoned.tmp");
        fs::create_dir(&abandoned).unwrap();
        fs::write(abandoned.join(CUBIN_FILE), fake_cubin(10)).unwrap();

        let result = cache_or_build(dir.path(), &key, || Ok::<_, ()>(artifacts(11))).unwrap();
        assert!(!result.cache_hit);
        assert_eq!(result.cubin, fake_cubin(11));
        assert!(abandoned.exists());
    }

    #[test]
    fn same_key_concurrency_builds_once() {
        let dir = Arc::new(TestDirectory::new("concurrent"));
        let key = test_key();
        let builds = Arc::new(AtomicU64::new(0));
        let barrier = Arc::new(Barrier::new(8));
        let mut threads = Vec::new();

        for _ in 0..8 {
            let dir = Arc::clone(&dir);
            let builds = Arc::clone(&builds);
            let barrier = Arc::clone(&barrier);
            threads.push(thread::spawn(move || {
                barrier.wait();
                cache_or_build(dir.path(), &key, || {
                    builds.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(30));
                    Ok::<_, ()>(artifacts(12))
                })
                .unwrap()
            }));
        }

        let results = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(results.iter().filter(|result| result.cache_hit).count(), 7);
        assert!(results.iter().all(|result| result.cubin == fake_cubin(12)));
        assert!(
            results
                .iter()
                .all(|result| result.ltoir == Some(vec![b'L', b'T', b'O', 12]))
        );
    }

    #[test]
    fn cache_setup_failure_still_returns_fresh_bytes() {
        let dir = TestDirectory::new("setup-failure");
        fs::write(dir.path().join(".oxide-artifacts"), b"not a directory").unwrap();
        let key = test_key();
        let builds = AtomicU64::new(0);

        for marker in [13, 14] {
            let result = cache_or_build(dir.path(), &key, || {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(artifacts(marker))
            })
            .unwrap();
            assert_eq!(result.cubin, fake_cubin(marker));
            assert_eq!(result.immutable_cubin_path, None);
            assert_eq!(result.immutable_ltoir_path, None);
            assert!(!result.cache_hit);
        }
        assert_eq!(builds.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn publication_failure_after_build_still_returns_fresh_bytes() {
        let dir = TestDirectory::new("publication-failure");
        let key = test_key();
        let paths = CachePaths::new(dir.path(), &key);

        let result = cache_or_build(dir.path(), &key, || {
            // Lock acquisition has already created this directory. Replace it
            // with a file so the atomic rename itself cannot be published.
            fs::remove_dir_all(&paths.entries).unwrap();
            fs::write(&paths.entries, b"block publication").unwrap();
            Ok::<_, ()>(artifacts(15))
        })
        .unwrap();

        assert_eq!(result.cubin, fake_cubin(15));
        assert_eq!(result.ltoir, Some(vec![b'L', b'T', b'O', 15]));
        assert_eq!(result.immutable_cubin_path, None);
        assert_eq!(result.immutable_ltoir_path, None);
        assert!(!result.cache_hit);
    }

    #[test]
    fn non_cubin_output_is_never_cached() {
        let dir = TestDirectory::new("invalid-output");
        let key = test_key();
        let builds = AtomicU64::new(0);

        for _ in 0..2 {
            let result = cache_or_build(dir.path(), &key, || {
                builds.fetch_add(1, Ordering::Relaxed);
                Ok::<_, ()>(BuiltArtifacts::new(
                    truncated_cubin_prefix(),
                    Some(vec![1, 2, 3]),
                ))
            })
            .unwrap();
            assert_eq!(result.immutable_cubin_path, None);
        }
        assert_eq!(builds.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn builder_errors_are_not_hidden_by_cache_fallback() {
        let dir = TestDirectory::new("builder-error");
        let key = test_key();
        let result = cache_or_build(dir.path(), &key, || {
            Err::<BuiltArtifacts, _>("compile failed")
        });
        assert_eq!(result.unwrap_err(), "compile failed");
    }
}
