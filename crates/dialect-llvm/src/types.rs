/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! [Type]s defined in the LLVM dialect.

use combine::{Parser, between, optional, token};
use pliron::builtin::type_interfaces::{FloatTypeInterface, FunctionTypeInterface};
use pliron::derive::{pliron_type, type_interface_impl};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    input_err_noloc,
    irfmt::{
        parsers::{delimited_list_parser, location, spaced},
        printers::{enclosed, list_with_sep},
    },
    location::Located,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, ListSeparator, Printable},
    result::Result,
    r#type::{Type, TypeObj, TypePtr},
    utils::apfloat::{self, GetSemantics, Semantics},
    verify_err_noloc,
};
use thiserror::Error;

use std::hash::Hash;

/// LLVM IR `half` type.
#[pliron_type(name = "llvm.half", format, generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct HalfType;

#[type_interface_impl]
impl FloatTypeInterface for HalfType {
    fn get_semantics(&self) -> Semantics {
        <apfloat::Half as GetSemantics>::get_semantics()
    }
}

/// Represents a c-like struct type.
/// Limitations and warnings on its usage are similar to the upstream MLIR LLVM dialect.
/// `<https://mlir.llvm.org/docs/Dialects/LLVM/#structure-types>`
///   1. Anonymous (aka unnamed) structs cannot be recursive.
///   2. Named structs are uniqued *only* by name, and may be recursive.
///   3. LLVM calls anonymous structs as literal structs and
///      named structs as identified structs.
///   4. Named structs may be opaque, i.e., no body specificed.
///      Recursive types may be created by first creating an opaque struct
///      and later setting its fields (body).
#[pliron_type(name = "llvm.struct")]
#[derive(Debug)]
pub struct StructType {
    name: Option<Identifier>,
    fields: Option<Vec<Ptr<TypeObj>>>,
}

impl StructType {
    /// Get or create a named `StructType`.
    /// If `fields` is `None`, it indicates an opaque struct.
    /// A body can be added to opaque structs by calling this again later.
    /// Returns an error if all of the below conditions are true:
    ///   a. The name is already registered
    ///   b. The body is already set (i.e, the struct is not oqaue)
    ///   c. The fields provided here don't match with the existing body.
    /// Since named structs only rely on the name for uniqueness,
    /// It is not an error to provide `fields` as `None` even when
    /// the named struct already exists and has its body set.
    pub fn get_named(
        ctx: &mut Context,
        name: Identifier,
        fields: Option<Vec<Ptr<TypeObj>>>,
    ) -> Result<TypePtr<Self>> {
        let self_ptr = Type::register_instance(
            Self {
                name: Some(name.clone()),
                // Uniquing happens only on the name, so this doesn't matter.
                fields: None,
            },
            ctx,
        );
        // Verify that we created a new or equivalent existing type.
        let mut self_ref = self_ptr.to_ptr().deref_mut(ctx);
        let self_ref = self_ref.downcast_mut::<Self>().unwrap();
        assert!(self_ref.name.as_ref().unwrap() == &name);
        if let Some(fields) = fields {
            // We've been provided fields to be set.
            if let Some(existing_fields) = &self_ref.fields {
                // Fields were already set before, ensure they're same as the given ones.
                if existing_fields != &fields {
                    input_err_noloc!(StructErr::ExistingMismatch(name.into()))?;
                }
            } else {
                // Set the fields now.
                self_ref.fields = Some(fields);
            }
        }
        Ok(self_ptr)
    }

    /// Get or create a new unnamed (anonymous) struct.
    /// These are finalized upon creation, and uniqued based on the fields.
    pub fn get_unnamed(ctx: &mut Context, fields: Vec<Ptr<TypeObj>>) -> TypePtr<Self> {
        Type::register_instance(
            Self {
                name: None,
                fields: Some(fields),
            },
            ctx,
        )
    }

    /// If a named struct already exists, get a pointer to it.
    #[must_use]
    pub fn get_existing_named(ctx: &Context, name: &Identifier) -> Option<TypePtr<Self>> {
        Type::get_instance(
            Self {
                name: Some(name.clone()),
                // Named structs are uniqued only on the name.
                fields: None,
            },
            ctx,
        )
    }

    /// If an unnamed struct already exists, get a pointer to it.
    #[must_use]
    pub fn get_existing_unnamed(ctx: &Context, fields: Vec<Ptr<TypeObj>>) -> Option<TypePtr<Self>> {
        Type::get_instance(
            Self {
                name: None,
                fields: Some(fields),
            },
            ctx,
        )
    }

    /// Does this struct not have its body set?
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        self.fields.is_none()
    }

    /// Is this a named struct?
    #[must_use]
    pub fn is_named(&self) -> bool {
        self.name.is_some()
    }

    /// Get this struct's name, if it has one.
    #[must_use]
    pub fn name(&self) -> Option<Identifier> {
        self.name.clone()
    }

    /// Get type of the idx'th field.
    #[must_use]
    pub fn field_type(&self, field_idx: usize) -> Ptr<TypeObj> {
        self.fields
            .as_ref()
            .expect("field_type shouldn't be called on opaque types")[field_idx]
    }

    /// Get the number of fields this struct has
    #[must_use]
    pub fn num_fields(&self) -> usize {
        self.fields
            .as_ref()
            .expect("num_fields shouldn't be called on opaque types")
            .len()
    }

    /// Get an iterator over the fields of this struct
    pub fn fields(&self) -> impl Iterator<Item = Ptr<TypeObj>> + '_ {
        self.fields
            .as_ref()
            .expect("fields shouldn't be called on opaque types")
            .iter()
            .copied()
    }
}

#[derive(Debug, Error)]
pub enum StructErr {
    #[error("struct cannot be both opaque and anonymous")]
    OpaqueAndAnonymousErr,
    #[error("struct {0} already exists and is different")]
    ExistingMismatch(String),
}

impl Verify for StructType {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        if self.name.is_none() && self.fields.is_none() {
            verify_err_noloc!(StructErr::OpaqueAndAnonymousErr)?;
        }
        Ok(())
    }
}

impl Printable for StructType {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        use std::cell::RefCell;

        write!(f, "<")?;
        // Ugly, but also the simplest way to avoid infinite recursion.
        // Upstream MLIR does the same: see LLVMTypeSyntax::printStructType.
        thread_local! {
            // We use a vec instead of a HashMap hoping that this isn't
            // going to be large, in which case vec would be faster.
            static IN_PRINTING: RefCell<Vec<Identifier>>  = const { RefCell::new(vec![]) };
        }
        if let Some(name) = &self.name {
            let in_printing = IN_PRINTING.with(|f| f.borrow().contains(name));
            if in_printing {
                return write!(f, "{}>", name.clone());
            }
            IN_PRINTING.with(|f| f.borrow_mut().push(name.clone()));
            write!(f, "{name}")?;
            if !self.is_opaque() {
                write!(f, " ")?;
            }
        }

        if let Some(fields) = &self.fields {
            enclosed(
                "{ ",
                " }",
                list_with_sep(fields, ListSeparator::CharSpace(',')),
            )
            .fmt(ctx, state, f)?;
        }

        // Done processing this struct. Remove it from the stack.
        if let Some(name) = &self.name {
            assert!(IN_PRINTING.with(|f| f.borrow().last().unwrap() == name));
            IN_PRINTING.with(|f| f.borrow_mut().pop());
        }
        write!(f, ">")
    }
}

impl Hash for StructType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match &self.name {
            Some(name) => name.hash(state),
            None => self
                .fields
                .as_ref()
                .expect("Anonymous struct must have its fields set")
                .iter()
                .for_each(|field_type| {
                    field_type.hash(state);
                }),
        }
    }
}

impl PartialEq for StructType {
    fn eq(&self, other: &Self) -> bool {
        match (&self.name, &other.name) {
            (Some(name), Some(other_name)) => name == other_name,
            (None, None) => self.fields == other.fields,
            _ => false,
        }
    }
}

impl Parsable for StructType {
    type Arg = ();
    type Parsed = TypePtr<Self>;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed>
    where
        Self: Sized,
    {
        let body_parser = || {
            // Parse multiple type annotated fields separated by ',', all of it delimited by braces.
            delimited_list_parser('{', '}', ',', Ptr::<TypeObj>::parser(()))
        };

        let named = spaced((location(), Identifier::parser(())))
            .and(spaced(optional(body_parser())))
            .map(|((loc, name), body_opt)| (loc, Some(name), body_opt));
        let anonymous = spaced((location(), body_parser()))
            .map(|(loc, body)| (loc, None::<Identifier>, Some(body)));

        // A struct type is named or anonymous.
        let mut struct_parser = between(token('<'), token('>'), named.or(anonymous));

        let (loc, name_opt, body_opt) = struct_parser.parse_stream(state_stream).into_result()?.0;
        let ctx = &mut state_stream.state.ctx;
        if let Some(name) = name_opt {
            Self::get_named(ctx, name, body_opt)
                .map_err(|mut err| {
                    err.set_loc(loc);
                    err
                })
                .into_parse_result()
        } else {
            Ok(Self::get_unnamed(
                ctx,
                body_opt.expect("Without a name, a struct type must have a body."),
            ))
            .into_parse_result()
        }
    }
}

impl Eq for StructType {}

/// CUDA/PTX address space constants (matches NVPTX backend).
pub mod address_space {
    /// Generic address space (can alias any memory)
    pub const GENERIC: u32 = 0;
    /// Global device memory (VRAM)
    pub const GLOBAL: u32 = 1;
    /// Per-block shared memory (fast scratchpad)
    pub const SHARED: u32 = 3;
    /// Read-only constant memory (cached)
    pub const CONSTANT: u32 = 4;
    /// Per-thread local memory (stack/spill)
    pub const LOCAL: u32 = 5;
    /// Tensor Memory - Blackwell+ (`sm_100`+) tcgen05 operands
    pub const TMEM: u32 = 6;
}

/// An opaque pointer with address space, corresponding to LLVM's pointer type.
///
/// In LLVM IR: `ptr addrspace(N)` where N is the address space.
/// - addrspace(0): Generic pointer (default)
/// - addrspace(1): Global memory
/// - addrspace(3): Shared memory
/// - addrspace(4): Constant memory
/// - addrspace(5): Local/private memory
/// - addrspace(6): Tensor Memory - Blackwell+ tcgen05 operands
#[pliron_type(
    name = "llvm.ptr",
    format = "`addrspace(` $address_space `)`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct PointerType {
    pub address_space: u32,
}

impl PointerType {
    /// Get or create a pointer type with the specified address space.
    pub fn get(ctx: &mut Context, address_space: u32) -> TypePtr<Self> {
        Type::register_instance(Self { address_space }, ctx)
    }

    /// Get or create a pointer in generic address space (0).
    pub fn get_generic(ctx: &mut Context) -> TypePtr<Self> {
        Self::get(ctx, address_space::GENERIC)
    }

    /// Get or create a pointer in shared memory address space (3).
    pub fn get_shared(ctx: &mut Context) -> TypePtr<Self> {
        Self::get(ctx, address_space::SHARED)
    }

    /// Get or create a pointer in global memory address space (1).
    pub fn get_global(ctx: &mut Context) -> TypePtr<Self> {
        Self::get(ctx, address_space::GLOBAL)
    }

    /// Get or create a pointer in tensor memory address space (6) - Blackwell+ tcgen05.
    pub fn get_tmem(ctx: &mut Context) -> TypePtr<Self> {
        Self::get(ctx, address_space::TMEM)
    }

    /// Get, if it already exists, a pointer type with the given address space.
    #[must_use]
    pub fn get_existing(ctx: &Context, address_space: u32) -> Option<TypePtr<Self>> {
        Type::get_instance(Self { address_space }, ctx)
    }

    /// Returns the address space of this pointer type.
    #[must_use]
    pub fn address_space(&self) -> u32 {
        self.address_space
    }

    /// Check if this pointer is in shared memory (addrspace 3).
    #[must_use]
    pub fn is_shared(&self) -> bool {
        self.address_space == address_space::SHARED
    }

    /// Check if this pointer is in tensor memory (addrspace 6) - Blackwell+ tcgen05.
    #[must_use]
    pub fn is_tmem(&self) -> bool {
        self.address_space == address_space::TMEM
    }
}

/// Array type, corresponding to LLVM's array type.
#[pliron_type(
    name = "llvm.array",
    format = "`[` $size ` x ` $elem `]`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct ArrayType {
    elem: Ptr<TypeObj>,
    size: u64,
}

impl ArrayType {
    /// Get or create a new array type.
    pub fn get(ctx: &mut Context, elem: Ptr<TypeObj>, size: u64) -> TypePtr<Self> {
        Type::register_instance(Self { elem, size }, ctx)
    }
    /// Get, if it already exists, an array type.
    #[must_use]
    pub fn get_existing(ctx: &Context, elem: Ptr<TypeObj>, size: u64) -> Option<TypePtr<Self>> {
        Type::get_instance(Self { elem, size }, ctx)
    }

    /// Get array element type.
    #[must_use]
    pub fn elem_type(&self) -> Ptr<TypeObj> {
        self.elem
    }

    /// Get array size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Vector type, corresponding to LLVM's vector type.
/// Vectors support `extractelement` with runtime indices, unlike arrays.
#[pliron_type(
    name = "llvm.vector",
    format = "`<` $size ` x ` $elem `>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct VectorType {
    elem: Ptr<TypeObj>,
    size: u64,
}

impl VectorType {
    /// Get or create a new vector type.
    pub fn get(ctx: &mut Context, elem: Ptr<TypeObj>, size: u64) -> TypePtr<Self> {
        Type::register_instance(Self { elem, size }, ctx)
    }

    /// Get, if it already exists, a vector type.
    #[must_use]
    pub fn get_existing(ctx: &Context, elem: Ptr<TypeObj>, size: u64) -> Option<TypePtr<Self>> {
        Type::get_instance(Self { elem, size }, ctx)
    }

    /// Get vector element type.
    #[must_use]
    pub fn elem_type(&self) -> Ptr<TypeObj> {
        self.elem
    }

    /// Get vector size (number of elements).
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }
}

#[pliron_type(name = "llvm.void", format, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct VoidType;

impl VoidType {
    /// Get or create a new void type.
    pub fn get(ctx: &mut Context) -> TypePtr<Self> {
        Type::register_instance(Self {}, ctx)
    }
}

#[pliron_type(
    name = "llvm.func",
    format = "`<` $res `(` vec($args, CharSpace(`,`)) `) variadic = ` $is_var_arg `>`",
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct FuncType {
    res: Ptr<TypeObj>,
    args: Vec<Ptr<TypeObj>>,
    is_var_arg: bool,
}

#[derive(Debug, Error)]
pub enum FuncTypeErr {
    #[error("Expected at most one result")]
    TooManyResults,
}

impl FuncType {
    /// Get or create a new Func type.
    pub fn get(
        ctx: &mut Context,
        res: Ptr<TypeObj>,
        args: Vec<Ptr<TypeObj>>,
        is_var_arg: bool,
    ) -> TypePtr<Self> {
        Type::register_instance(
            Self {
                res,
                args,
                is_var_arg,
            },
            ctx,
        )
    }

    /// Result type
    #[must_use]
    pub fn result_type(&self) -> Ptr<TypeObj> {
        self.res
    }

    /// Is this a variadic function type?
    #[must_use]
    pub fn is_var_arg(&self) -> bool {
        self.is_var_arg
    }
}

#[type_interface_impl]
impl FunctionTypeInterface for FuncType {
    fn arg_types(&self) -> Vec<Ptr<TypeObj>> {
        self.args.clone()
    }
    fn res_types(&self) -> Vec<Ptr<TypeObj>> {
        vec![self.res]
    }
}

pub fn register(ctx: &mut Context) {
    HalfType::register(ctx);
    VoidType::register(ctx);
    ArrayType::register(ctx);
    VectorType::register(ctx);
    StructType::register(ctx);
    PointerType::register(ctx);
    FuncType::register(ctx);
}

#[cfg(test)]
mod tests {

    use crate as llvm;
    use combine::{Parser, eof, token};
    use expect_test::expect;
    use pliron::derive::pliron_type;

    use crate::types::{FuncType, StructType, VoidType};
    use pliron::{
        builtin::types::{IntegerType, Signedness},
        context::{Context, Ptr},
        identifier::Identifier,
        irfmt::parsers::{spaced, type_parser},
        location,
        parsable::{self, Parsable, ParseResult, StateStream, state_stream_from_iterator},
        printable::{self, Printable},
        result::Result,
        r#type::{Type, TypeObj, TypePtr},
    };

    #[test]
    fn test_struct() -> Result<()> {
        let mut ctx = Context::new();
        let int64_ptr = IntegerType::get(&mut ctx, 64, Signedness::Signless).into();
        let linked_list_id: Identifier = "LinkedList".try_into().unwrap();
        let linked_list_2_id: Identifier = "LinkedList2".try_into().unwrap();

        // Create an opaque struct since we want a recursive type.
        let list_struct: Ptr<TypeObj> =
            StructType::get_named(&mut ctx, linked_list_id.clone(), None)?.into();
        assert!(
            list_struct
                .deref(&ctx)
                .downcast_ref::<StructType>()
                .unwrap()
                .is_opaque()
        );
        let list_struct_ptr = TypedPointerType::get(&mut ctx, list_struct).into();
        let fields = vec![int64_ptr, list_struct_ptr];
        // Set the struct body now.
        StructType::get_named(&mut ctx, linked_list_id.clone(), Some(fields))?;
        assert!(
            !list_struct
                .deref(&ctx)
                .downcast_ref::<StructType>()
                .unwrap()
                .is_opaque()
        );

        let list_struct_2 = StructType::get_existing_named(&ctx, &linked_list_id)
            .unwrap()
            .into();
        assert!(list_struct == list_struct_2);
        assert!(StructType::get_existing_named(&ctx, &linked_list_2_id).is_none());

        assert_eq!(
            list_struct.disp(&ctx).to_string(),
            "llvm.struct <LinkedList { builtin.integer i64, llvm.typed_ptr <llvm.struct <LinkedList>> }>"
        );

        let head_fields = vec![int64_ptr, list_struct_ptr];
        let head_struct = StructType::get_unnamed(&mut ctx, head_fields.clone());
        let head_struct2 = StructType::get_existing_unnamed(&ctx, head_fields).unwrap();
        assert!(head_struct == head_struct2);
        assert!(StructType::get_existing_unnamed(&ctx, vec![int64_ptr, list_struct]).is_none());

        Ok(())
    }

    /// A pointer type that knows the type it points to.
    /// This used to be in LLVM earlier, but the latest version
    /// is now type-erased (https://llvm.org/docs/OpaquePointers.html)
    #[pliron_type(name = "llvm.typed_ptr", verifier = "succ")]
    #[derive(Hash, PartialEq, Eq, Debug)]
    pub struct TypedPointerType {
        to: Ptr<TypeObj>,
    }

    impl TypedPointerType {
        /// Get or create a new pointer type.
        pub fn get(ctx: &mut Context, to: Ptr<TypeObj>) -> TypePtr<Self> {
            Type::register_instance(TypedPointerType { to }, ctx)
        }
        /// Get, if it already exists, a pointer type.
        pub fn get_existing(ctx: &Context, to: Ptr<TypeObj>) -> Option<TypePtr<Self>> {
            Type::get_instance(TypedPointerType { to }, ctx)
        }

        /// Get the pointee type.
        pub fn get_pointee_type(&self) -> Ptr<TypeObj> {
            self.to
        }
    }

    impl Printable for TypedPointerType {
        fn fmt(
            &self,
            ctx: &Context,
            _state: &printable::State,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            write!(f, "<{}>", self.to.disp(ctx))
        }
    }

    impl Parsable for TypedPointerType {
        type Arg = ();
        type Parsed = TypePtr<Self>;

        fn parse<'a>(
            state_stream: &mut StateStream<'a>,
            _arg: Self::Arg,
        ) -> ParseResult<'a, Self::Parsed>
        where
            Self: Sized,
        {
            combine::between(token('<'), token('>'), spaced(type_parser()))
                .parse_stream(state_stream)
                .map(|pointee_ty| TypedPointerType::get(state_stream.state.ctx, pointee_ty))
                .into()
        }
    }

    #[test]
    fn test_pointer_types() {
        let mut ctx = Context::new();
        let int32_1_ptr = IntegerType::get(&mut ctx, 32, Signedness::Signed);
        let int64_ptr = IntegerType::get(&mut ctx, 64, Signedness::Signed).into();

        let int64pointer_ptr = TypedPointerType { to: int64_ptr };
        let int64pointer_ptr = Type::register_instance(int64pointer_ptr, &mut ctx);
        assert_eq!(
            int64pointer_ptr.disp(&ctx).to_string(),
            "llvm.typed_ptr <builtin.integer si64>"
        );
        assert!(int64pointer_ptr == TypedPointerType::get(&mut ctx, int64_ptr));

        assert!(
            int64_ptr
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .unwrap()
                .width()
                == 64
        );

        assert!(IntegerType::get_existing(&ctx, 32, Signedness::Signed).unwrap() == int32_1_ptr);
        assert!(TypedPointerType::get_existing(&ctx, int64_ptr).unwrap() == int64pointer_ptr);
        assert!(int64pointer_ptr.deref(&ctx).get_pointee_type() == int64_ptr);
    }

    #[test]
    fn test_pointer_type_parsing() {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        TypedPointerType::register(&mut ctx);

        let state_stream = state_stream_from_iterator(
            "llvm.typed_ptr <builtin.integer si64>".chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );

        let res = type_parser().parse(state_stream).unwrap().0;
        assert_eq!(
            &res.disp(&ctx).to_string(),
            "llvm.typed_ptr <builtin.integer si64>"
        );
    }

    #[test]
    fn test_struct_type_parsing() {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);
        TypedPointerType::register(&mut ctx);

        let state_stream = state_stream_from_iterator(
            "llvm.struct <LinkedList { builtin.integer i64, llvm.typed_ptr <llvm.struct <LinkedList>> }>"
                .chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );

        let res = type_parser().parse(state_stream).unwrap().0;
        assert_eq!(
            &res.disp(&ctx).to_string(),
            "llvm.struct <LinkedList { builtin.integer i64, llvm.typed_ptr <llvm.struct <LinkedList>> }>"
        );

        // Test parsing an opaque struct.
        let test_string = "llvm.struct <ExternStruct>";
        let state_stream = state_stream_from_iterator(
            test_string.chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );
        let res = type_parser().parse(state_stream).unwrap().0;
        assert_eq!(&res.disp(&ctx).to_string(), test_string);
        {
            let res = res.deref(&ctx);
            let res = res.downcast_ref::<StructType>().unwrap();
            assert!(res.is_opaque() && res.is_named());
        }

        // Test parsing an unnamed struct.
        let test_string = "llvm.struct <{ builtin.integer i8 }>";
        let state_stream = state_stream_from_iterator(
            test_string.chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );
        let res = type_parser().parse(state_stream).unwrap().0;
        assert_eq!(&res.disp(&ctx).to_string(), test_string);
        {
            let res = res.deref(&ctx);
            let res = res.downcast_ref::<StructType>().unwrap();
            assert!(!res.is_opaque() && !res.is_named());
        }
    }

    #[test]
    fn test_struct_type_errs() {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);

        let state_stream = state_stream_from_iterator(
            "llvm.struct < My1 { builtin.integer i8 } >".chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );
        let _ = type_parser().parse(state_stream).unwrap().0;

        let state_stream = state_stream_from_iterator(
            "llvm.struct < My1 { builtin.integer i16 } >".chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );

        let res = type_parser().parse(state_stream);
        let err_msg = format!("{}", res.err().unwrap());

        let expected_err_msg = expect![[r#"
            Parse error at line: 1, column: 15
            struct My1 already exists and is different
        "#]];
        expected_err_msg.assert_eq(&err_msg);
    }

    #[test]
    fn test_functype_parsing() {
        let mut ctx = Context::new();
        llvm::register(&mut ctx);

        let si32 = IntegerType::get(&mut ctx, 32, Signedness::Signed);

        let input = "llvm.func <llvm.void (builtin.integer si32) variadic = false>";
        let state_stream = state_stream_from_iterator(
            input.chars(),
            parsable::State::new(&mut ctx, location::Source::InMemory),
        );

        let res = type_parser().and(eof()).parse(state_stream).unwrap().0.0;

        let void_ty = VoidType::get(&mut ctx);
        assert!(res == FuncType::get(&mut ctx, void_ty.to_ptr(), vec![si32.into()], false).into());
        assert_eq!(input, &res.disp(&ctx).to_string());
    }
}
