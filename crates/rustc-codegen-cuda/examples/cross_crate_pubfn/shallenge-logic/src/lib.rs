//! Pure-arithmetic, GPU-agnostic shallenge logic. Equivalent to
//! `vanity-miner-rs/logic/`. No `cuda_device` import, no
//! `cuda-macros` dep, no `#[device]` / `#[kernel]` / `#[cuda_module]`
//! annotations, no `#[inline]` on the public surface.
//!
//! Everything below is reproduced verbatim from
//! `examples/shallenge_repro/src/main.rs`; the only thing different
//! is that it lives in a separate path-dep no_std crate.

#![no_std]

// ============================================================================
// PRNG: splitmix64 + Xoroshiro128**
// ============================================================================

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub struct Xoroshiro128StarStar {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128StarStar {
    pub fn seed_from_u64(seed: u64) -> Self {
        let s0 = splitmix64(seed);
        let s1 = splitmix64(s0);
        Self { s0, s1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s0.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let s0 = self.s0;
        let mut s1 = self.s1;
        s1 ^= s0;
        self.s0 = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
        self.s1 = s1.rotate_left(37);
        result
    }

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

pub const BASE64_CHARS: &[u8] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn generate_base64_nonce(thread_idx: usize, rng_seed: u64, nonce: &mut [u8]) {
    let mixed_seed = splitmix64(rng_seed.wrapping_add(thread_idx as u64));
    let mut rng = Xoroshiro128StarStar::seed_from_u64(mixed_seed);
    for byte in nonce.iter_mut() {
        let idx = (rng.next_u32() % 64) as usize;
        *byte = BASE64_CHARS[idx];
    }
}

// ============================================================================
// SHA-256 (one-shot, 32-byte input)
// ============================================================================

pub const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}
fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

pub fn sha256_32_from_bytes(input: &[u8; 32]) -> [u8; 32] {
    let mut input_words = [0u32; 8];
    for i in 0..8 {
        input_words[i] = u32::from_le_bytes([
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
        ]);
    }

    let mut w = [0u32; 64];
    for i in 0..8 {
        w[i] = input_words[i].to_be();
    }
    w[8] = 0x80000000;
    w[14] = 0;
    w[15] = 256;
    for n in 16..64 {
        w[n] = small_sigma1(w[n - 2])
            .wrapping_add(w[n - 7])
            .wrapping_add(small_sigma0(w[n - 15]))
            .wrapping_add(w[n - 16]);
    }

    let mut a = H0[0];
    let mut b = H0[1];
    let mut c = H0[2];
    let mut d = H0[3];
    let mut e = H0[4];
    let mut f = H0[5];
    let mut g = H0[6];
    let mut h = H0[7];

    for round in 0..64 {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[round])
            .wrapping_add(w[round]);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    let hash_words = [
        H0[0].wrapping_add(a),
        H0[1].wrapping_add(b),
        H0[2].wrapping_add(c),
        H0[3].wrapping_add(d),
        H0[4].wrapping_add(e),
        H0[5].wrapping_add(f),
        H0[6].wrapping_add(g),
        H0[7].wrapping_add(h),
    ];

    let mut output = [0u8; 32];
    for i in 0..8 {
        let bytes = hash_words[i].to_be_bytes();
        output[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }
    output
}

// ============================================================================
// Shallenge
// ============================================================================

pub fn shallenge(
    username: &[u8],
    username_len: usize,
    nonce: &[u8],
    nonce_len: usize,
) -> [u8; 32] {
    let mut input = [0u8; 32];
    let mut pos = 0;
    input[pos..pos + username_len].copy_from_slice(&username[..username_len]);
    pos += username_len;
    input[pos] = b'/';
    pos += 1;
    input[pos..pos + nonce_len].copy_from_slice(&nonce[..nonce_len]);
    sha256_32_from_bytes(&input)
}

pub fn compare_hashes(a: &[u8; 32], b: &[u8; 32]) -> i32 {
    for i in 0..32 {
        if a[i] < b[i] {
            return -1;
        }
        if a[i] > b[i] {
            return 1;
        }
    }
    0
}

pub struct ShallengeRequest<'a> {
    pub username: &'a [u8],
    pub username_len: usize,
    pub target_hash: &'a [u8; 32],
    pub thread_idx: usize,
    pub rng_seed: u64,
}

pub struct ShallengeResult {
    pub hash: [u8; 32],
    pub nonce: [u8; 64],
    pub nonce_len: usize,
    pub is_better: bool,
}

pub fn generate_and_check_shallenge(request: &ShallengeRequest) -> ShallengeResult {
    let mut nonce = [0u8; 21];
    generate_base64_nonce(request.thread_idx, request.rng_seed, &mut nonce);
    let hash = shallenge(request.username, request.username_len, &nonce, 21);
    let is_better = compare_hashes(&hash, request.target_hash) < 0;
    let mut result_nonce = [0u8; 64];
    result_nonce[..21].copy_from_slice(&nonce);
    ShallengeResult {
        hash,
        nonce: result_nonce,
        nonce_len: 21,
        is_better,
    }
}
