/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA Virtual Memory Management (VMM) API wrappers.
//!
//! The VMM APIs provide fine-grained control over physical memory allocation,
//! virtual address reservation, and mapping. Unlike `cuMemAlloc`, which bundles
//! all three steps, VMM separates them so that physical memory from one device
//! can be mapped into another device's virtual address space -- the foundation
//! for P2P symmetric heaps.
//!
//! All handle types are RAII: `PhysicalAllocation` releases via `cuMemRelease`,
//! `VirtualReservation` frees via `cuMemAddressFree`, and `Mapping` unmaps
//! via `cuMemUnmap`. Drop order matters -- mappings must be dropped before the
//! physical allocation or virtual reservation they reference.

use crate::error::{DriverError, IntoResult};
use cuda_bindings::CUdeviceptr;
use std::mem::MaybeUninit;

/// Sets the device ordinal on a `CUmemLocation_st`.
///
/// CUDA 13.2 wraps `id` in an anonymous union (`__bindgen_anon_1.id`), while
/// older versions expose it directly. The memory layout is identical -- `id` is
/// always at offset 4 (after the `type_` enum). Writing via pointer works across
/// both layouts.
unsafe fn set_mem_location_device(
    loc: &mut cuda_bindings::CUmemLocation_st,
    device: cuda_bindings::CUdevice,
) {
    loc.type_ = cuda_bindings::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;
    unsafe {
        let base = loc as *mut _ as *mut u8;
        (base.add(4) as *mut i32).write(device);
    }
}

/// A physical memory allocation created by `cuMemCreate`.
///
/// Owns the underlying `CUmemGenericAllocationHandle`. The allocation lives on
/// a specific device and can be mapped into any device's VA space that has been
/// granted access.
///
/// Dropping this releases the physical memory. All `Mapping`s referencing this
/// allocation must be dropped first.
pub struct PhysicalAllocation {
    handle: cuda_bindings::CUmemGenericAllocationHandle,
    size: usize,
}

impl PhysicalAllocation {
    /// Allocates `size` bytes of physical memory on `device`.
    ///
    /// `size` must be a multiple of the allocation granularity for the device
    /// (query via [`allocation_granularity`]).
    pub fn new(device: cuda_bindings::CUdevice, size: usize) -> Result<Self, DriverError> {
        let mut prop: cuda_bindings::CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
        prop.type_ = cuda_bindings::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
        unsafe { set_mem_location_device(&mut prop.location, device) };

        let mut handle = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuMemCreate(handle.as_mut_ptr(), size, &prop, 0).result()?;
            Ok(Self {
                handle: handle.assume_init(),
                size,
            })
        }
    }

    /// Returns the raw `CUmemGenericAllocationHandle`.
    pub fn handle(&self) -> cuda_bindings::CUmemGenericAllocationHandle {
        self.handle
    }

    /// Returns the allocation size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PhysicalAllocation {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemRelease(self.handle).result();
        }
    }
}

/// A reserved virtual address range created by `cuMemAddressReserve`.
///
/// Owns a contiguous VA range `[base, base + size)`. Physical memory can be
/// mapped into this range via [`Mapping::new`]. The range is freed on drop.
///
/// All `Mapping`s within this range must be dropped before the reservation.
pub struct VirtualReservation {
    base: CUdeviceptr,
    size: usize,
}

impl VirtualReservation {
    /// Reserves `size` bytes of virtual address space.
    ///
    /// The driver chooses the base address. `size` must be a multiple of the
    /// allocation granularity. `alignment` can be 0 to let the driver choose.
    pub fn new(size: usize, alignment: usize) -> Result<Self, DriverError> {
        let mut base = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuMemAddressReserve(base.as_mut_ptr(), size, alignment, 0, 0)
                .result()?;
            Ok(Self {
                base: base.assume_init(),
                size,
            })
        }
    }

    /// Returns the base device pointer of the reserved range.
    pub fn base(&self) -> CUdeviceptr {
        self.base
    }

    /// Returns the reserved size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for VirtualReservation {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemAddressFree(self.base, self.size).result();
        }
    }
}

/// A mapping of physical memory into a virtual address range.
///
/// Created by [`Mapping::new`], which calls `cuMemMap` to bind a
/// `PhysicalAllocation` (or a portion of it) to a region within a
/// `VirtualReservation`. Dropped via `cuMemUnmap`.
pub struct Mapping {
    va: CUdeviceptr,
    size: usize,
}

impl Mapping {
    /// Maps `size` bytes of `phys` at `offset` into virtual address `va`.
    ///
    /// `va` must lie within a `VirtualReservation`. `offset` is the byte
    /// offset into the physical allocation (typically 0 for full mappings).
    /// `size` must be a multiple of the allocation granularity.
    pub fn new(
        va: CUdeviceptr,
        size: usize,
        phys: &PhysicalAllocation,
        offset: usize,
    ) -> Result<Self, DriverError> {
        unsafe {
            cuda_bindings::cuMemMap(va, size, offset, phys.handle(), 0).result()?;
        }
        Ok(Self { va, size })
    }

    /// Returns the virtual address this mapping occupies.
    pub fn va(&self) -> CUdeviceptr {
        self.va
    }

    /// Returns the mapped size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuMemUnmap(self.va, self.size).result();
        }
    }
}

/// Sets read/write access on a virtual address range for one or more devices.
///
/// After calling `cuMemMap`, the mapping is not yet accessible. This function
/// grants the specified `devices` read/write permission on the range
/// `[va, va + size)`.
///
/// Typically called once after all mappings within a reservation are established.
pub fn set_access(
    va: CUdeviceptr,
    size: usize,
    devices: &[cuda_bindings::CUdevice],
) -> Result<(), DriverError> {
    let descs: Vec<cuda_bindings::CUmemAccessDesc_st> = devices
        .iter()
        .map(|&dev| {
            let mut desc: cuda_bindings::CUmemAccessDesc_st = unsafe { std::mem::zeroed() };
            unsafe { set_mem_location_device(&mut desc.location, dev) };
            desc.flags = cuda_bindings::CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            desc
        })
        .collect();

    unsafe { cuda_bindings::cuMemSetAccess(va, size, descs.as_ptr(), descs.len()) }.result()
}

/// Queries the minimum allocation granularity for VMM operations on `device`.
///
/// All sizes passed to [`PhysicalAllocation::new`], [`VirtualReservation::new`],
/// and [`Mapping::new`] must be multiples of this value.
pub fn allocation_granularity(device: cuda_bindings::CUdevice) -> Result<usize, DriverError> {
    let mut prop: cuda_bindings::CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
    prop.type_ = cuda_bindings::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
    unsafe { set_mem_location_device(&mut prop.location, device) };

    let mut granularity = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemGetAllocationGranularity(
            granularity.as_mut_ptr(),
            &prop,
            cuda_bindings::CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        .result()?;
        Ok(granularity.assume_init())
    }
}

/// Rounds `size` up to the nearest multiple of `granularity`.
pub fn align_size(size: usize, granularity: usize) -> usize {
    (size + granularity - 1) & !(granularity - 1)
}
