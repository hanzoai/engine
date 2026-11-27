// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Path B: RISC Zero-style general deterministic compact register zkVM over
//! Goldilocks `F_p` / `F_{p^2}` with `ImageID` program+weights binding, Merkle
//! execution trace commitments, and algebraic transition proofs.
//!
//! Supported deterministic ISA instructions:
//! `LOAD` (immediate / public input / committed weight), `ADD`, `SUB`, `MUL`,
//! `SRA` (exact arithmetic right shift), `CMP_GT` (exact signed greater-than),
//! `SELECT` (boolean-gated mux), and `HALT`.

use core::fmt;

use crate::circuit::{
    compute_weights_sha256, prove_circuit, prove_circuit_unchecked_for_adversarial_test,
    setup_circuit_keys, verify_circuit_proof, CircuitProof, CircuitProvingKey, CircuitSpec,
    CircuitVerifyingKey, CircuitVersion, LogicalWire, R1csBuilder, SIGNED_BIAS, SIGNED_BIT_WIDTH,
};
use crate::field::Fp;
use crate::merkle::{verify_merkle_proof, LigeroParams, MerkleProof, MerkleTree};
use crate::sha256::{sha256, Sha256, Transcript};
use crate::ZkmlError;

/// Default domain separation tag for Path B general zkVM receipts.
pub const ZKVM_DOMAIN_TAG: &str = "hanzo.zkml.zkvm.v1";

/// Number of general-purpose registers (`R0..R7`).
pub const NUM_REGS: usize = 8;

/// Number of Fiat-Shamir random trace rows spot-opened against `trace_merkle_root`.
pub const TRACE_SPOT_QUERIES: usize = 8;

/// Semantic version of the deterministic zkVM ISA.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ZkVmVersion {
    /// Major ISA version.
    pub major: u16,
    /// Minor ISA version.
    pub minor: u16,
    /// Patch ISA version.
    pub patch: u16,
}

impl ZkVmVersion {
    /// Constructs an ISA version.
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Current v1.0.0 ISA version.
    pub const V1: Self = Self::new(1, 0, 0);

    /// Canonical 6-byte little-endian encoding.
    pub fn to_le_bytes(self) -> [u8; 6] {
        let mut out = [0u8; 6];
        out[0..2].copy_from_slice(&self.major.to_le_bytes());
        out[2..4].copy_from_slice(&self.minor.to_le_bytes());
        out[4..6].copy_from_slice(&self.patch.to_le_bytes());
        out
    }
}

impl fmt::Display for ZkVmVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Source operand for [`Instruction::Load`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LoadSource {
    /// Signed constant immediate.
    Imm(i64),
    /// Public journal input slot `journal_input[idx]`.
    Input(usize),
    /// Committed weight/table slot `weights[idx]`.
    Weight(usize),
}

/// Deterministic register zkVM instruction set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// `R[dst] = source` (immediate, public journal input, or committed weight).
    Load {
        /// Destination register `0..8`.
        dst: u8,
        /// Source value or slot.
        src: LoadSource,
    },
    /// `R[dst] = R[a] + R[b]`.
    Add {
        /// Destination register.
        dst: u8,
        /// First source register.
        a: u8,
        /// Second source register.
        b: u8,
    },
    /// `R[dst] = R[a] - R[b]`.
    Sub {
        /// Destination register.
        dst: u8,
        /// First source register.
        a: u8,
        /// Second source register.
        b: u8,
    },
    /// `R[dst] = R[a] * R[b]`.
    Mul {
        /// Destination register.
        dst: u8,
        /// First source register.
        a: u8,
        /// Second source register.
        b: u8,
    },
    /// `R[dst] = R[a] >> shift` (exact signed floor division by `2^shift`, `1 <= shift <= 16`).
    Sra {
        /// Destination register.
        dst: u8,
        /// Source register.
        a: u8,
        /// Shift amount in `1..=16`.
        shift: u8,
    },
    /// `R[dst] = 1` if `R[a] > R[b]` (signed) else `0`.
    CmpGt {
        /// Destination register.
        dst: u8,
        /// Left operand register.
        a: u8,
        /// Right operand register.
        b: u8,
    },
    /// `R[dst] = if R[cond] == 1 { R[a] } else { R[b] }` (enforces `R[cond] in {0, 1}`).
    Select {
        /// Destination register.
        dst: u8,
        /// Boolean selector register.
        cond: u8,
        /// Value register when `cond == 1`.
        a: u8,
        /// Value register when `cond == 0`.
        b: u8,
    },
    /// Halts execution and emits `R[r]` for each `r` in `out_regs` to `journal_output`.
    Halt {
        /// Ordered register indices committed as public journal outputs.
        out_regs: Vec<u8>,
    },
}

impl Instruction {
    /// Stable opcode byte tag for canonical serialization and trace hashing.
    pub fn opcode_tag(&self) -> u8 {
        match self {
            Self::Load { .. } => 0x01,
            Self::Add { .. } => 0x02,
            Self::Sub { .. } => 0x03,
            Self::Mul { .. } => 0x04,
            Self::Sra { .. } => 0x05,
            Self::CmpGt { .. } => 0x06,
            Self::Select { .. } => 0x07,
            Self::Halt { .. } => 0x08,
        }
    }

    /// Appends canonical encoding of this instruction to `h`.
    fn hash_into(&self, h: &mut Sha256) {
        h.update(&[self.opcode_tag()]);
        match self {
            Self::Load { dst, src } => {
                h.update(&[*dst]);
                match *src {
                    LoadSource::Imm(v) => {
                        h.update(&[0x00]);
                        h.update(&v.to_le_bytes());
                    }
                    LoadSource::Input(idx) => {
                        h.update(&[0x01]);
                        h.update(&(idx as u64).to_le_bytes());
                    }
                    LoadSource::Weight(idx) => {
                        h.update(&[0x02]);
                        h.update(&(idx as u64).to_le_bytes());
                    }
                }
            }
            Self::Add { dst, a, b } | Self::Sub { dst, a, b } | Self::Mul { dst, a, b } => {
                h.update(&[*dst, *a, *b]);
            }
            Self::Sra { dst, a, shift } => {
                h.update(&[*dst, *a, *shift]);
            }
            Self::CmpGt { dst, a, b } => {
                h.update(&[*dst, *a, *b]);
            }
            Self::Select { dst, cond, a, b } => {
                h.update(&[*dst, *cond, *a, *b]);
            }
            Self::Halt { out_regs } => {
                h.update(&(out_regs.len() as u32).to_le_bytes());
                h.update(out_regs);
            }
        }
    }
}

/// Deterministic straight-line / predicated zkVM bytecode program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkVmProgram {
    /// Human-readable program identifier.
    pub program_id: String,
    /// Number of expected public journal input integers.
    pub num_inputs: usize,
    /// Number of expected public journal output integers.
    pub num_outputs: usize,
    /// Number of expected committed weight table entries.
    pub num_weights: usize,
    /// Ordered instructions (must end with [`Instruction::Halt`]).
    pub instructions: Vec<Instruction>,
}

impl ZkVmProgram {
    /// Validates static instruction well-formedness (register bounds, operand bounds,
    /// and terminating `HALT`).
    pub fn validate(&self) -> Result<(), ZkmlError> {
        if self.instructions.is_empty() {
            return Err(ZkmlError::VmExecutionError {
                detail: "program has zero instructions".to_string(),
            });
        }
        let check_reg = |r: u8| -> Result<(), ZkmlError> {
            if (r as usize) >= NUM_REGS {
                Err(ZkmlError::VmExecutionError {
                    detail: format!("register R{r} out of range 0..{NUM_REGS}"),
                })
            } else {
                Ok(())
            }
        };

        for (pc, instr) in self.instructions.iter().enumerate() {
            let is_last = pc + 1 == self.instructions.len();
            match instr {
                Instruction::Load { dst, src } => {
                    check_reg(*dst)?;
                    match *src {
                        LoadSource::Imm(_) => {}
                        LoadSource::Input(idx) if idx >= self.num_inputs => {
                            return Err(ZkmlError::VmExecutionError {
                                detail: format!("input index {idx} >= {}", self.num_inputs),
                            });
                        }
                        LoadSource::Weight(idx) if idx >= self.num_weights => {
                            return Err(ZkmlError::VmExecutionError {
                                detail: format!("weight index {idx} >= {}", self.num_weights),
                            });
                        }
                        _ => {}
                    }
                    if is_last {
                        return Err(ZkmlError::VmExecutionError {
                            detail: "last instruction must be HALT".to_string(),
                        });
                    }
                }
                Instruction::Add { dst, a, b }
                | Instruction::Sub { dst, a, b }
                | Instruction::Mul { dst, a, b }
                | Instruction::CmpGt { dst, a, b } => {
                    check_reg(*dst)?;
                    check_reg(*a)?;
                    check_reg(*b)?;
                    if is_last {
                        return Err(ZkmlError::VmExecutionError {
                            detail: "last instruction must be HALT".to_string(),
                        });
                    }
                }
                Instruction::Sra { dst, a, shift } => {
                    check_reg(*dst)?;
                    check_reg(*a)?;
                    if *shift == 0 || *shift > 16 {
                        return Err(ZkmlError::VmExecutionError {
                            detail: format!("SRA shift {shift} must be in 1..=16"),
                        });
                    }
                    if is_last {
                        return Err(ZkmlError::VmExecutionError {
                            detail: "last instruction must be HALT".to_string(),
                        });
                    }
                }
                Instruction::Select { dst, cond, a, b } => {
                    check_reg(*dst)?;
                    check_reg(*cond)?;
                    check_reg(*a)?;
                    check_reg(*b)?;
                    if is_last {
                        return Err(ZkmlError::VmExecutionError {
                            detail: "last instruction must be HALT".to_string(),
                        });
                    }
                }
                Instruction::Halt { out_regs } => {
                    if !is_last {
                        return Err(ZkmlError::VmExecutionError {
                            detail: "HALT must be the final instruction".to_string(),
                        });
                    }
                    if out_regs.len() != self.num_outputs {
                        return Err(ZkmlError::VmExecutionError {
                            detail: format!(
                                "HALT emits {} outputs, program declares {}",
                                out_regs.len(),
                                self.num_outputs
                            ),
                        });
                    }
                    for &r in out_regs {
                        check_reg(r)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Computes the canonical SHA-256 digest of the bytecode alone.
    pub fn program_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"hanzo.zkml.zkvm.program.v1");
        h.update(&(self.program_id.len() as u64).to_le_bytes());
        h.update(self.program_id.as_bytes());
        h.update(&(self.num_inputs as u64).to_le_bytes());
        h.update(&(self.num_outputs as u64).to_le_bytes());
        h.update(&(self.num_weights as u64).to_le_bytes());
        h.update(&(self.instructions.len() as u64).to_le_bytes());
        for instr in &self.instructions {
            instr.hash_into(&mut h);
        }
        h.finalize()
    }

    /// Computes the RISC Zero-style `ImageID` binding VM version, domain tag, bytecode,
    /// committed weights digest + Merkle root, and declared `max_cycles`.
    pub fn compute_image_id(
        &self,
        vm_version: ZkVmVersion,
        domain_tag: &str,
        weights_sha256: &[u8; 32],
        weights_root: &[u8; 32],
        max_cycles: usize,
    ) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"hanzo.zkml.zkvm.image_id.v1");
        h.update(&vm_version.to_le_bytes());
        h.update(&(domain_tag.len() as u64).to_le_bytes());
        h.update(domain_tag.as_bytes());
        h.update(&self.program_digest());
        h.update(weights_sha256);
        h.update(weights_root);
        h.update(&(max_cycles as u64).to_le_bytes());
        h.finalize()
    }
}

/// One row of the zkVM execution trace matrix (cycle `t`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRow {
    /// Zero-based cycle index.
    pub cycle: usize,
    /// Opcode byte executed at this cycle.
    pub opcode: u8,
    /// Register file before this instruction.
    pub regs_before: [i64; NUM_REGS],
    /// Register file after this instruction.
    pub regs_after: [i64; NUM_REGS],
    /// Auxiliary witness value (quotient for `SRA` or signed delta for `CMP_GT`).
    pub aux_a: i64,
    /// Auxiliary witness value (remainder for `SRA` or comparison result for `CMP_GT`).
    pub aux_b: i64,
}

impl TraceRow {
    /// Canonical byte serialization used for Merkle trace commitment leaves.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 1 + NUM_REGS * 16 + 16);
        out.extend_from_slice(&(self.cycle as u64).to_le_bytes());
        out.push(self.opcode);
        for &r in &self.regs_before {
            out.extend_from_slice(&r.to_le_bytes());
        }
        for &r in &self.regs_after {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out.extend_from_slice(&self.aux_a.to_le_bytes());
        out.extend_from_slice(&self.aux_b.to_le_bytes());
        out
    }

    /// Exact serialized byte length.
    pub const fn byte_len() -> usize {
        8 + 1 + NUM_REGS * 16 + 16
    }
}

/// Spot-opened execution trace row with its Merkle authentication path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRowOpening {
    /// Opened row contents.
    pub row: TraceRow,
    /// Inclusion proof against `ZkVmReceipt::trace_merkle_root`.
    pub merkle_proof: MerkleProof,
}

impl TraceRowOpening {
    /// Appends canonical wire bytes to `out`.
    pub fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.row.to_bytes());
        self.merkle_proof.write_bytes(out);
    }

    /// Exact serialized byte length.
    pub fn byte_len(&self) -> usize {
        TraceRow::byte_len() + self.merkle_proof.byte_len()
    }
}

/// Result of compiling a [`ZkVmProgram`] into its algebraic AIR/R1CS trace constraint system.
fn compile_zkvm_trace_spec(
    program: &ZkVmProgram,
    vm_version: ZkVmVersion,
    domain_tag: &str,
) -> Result<CircuitSpec, ZkmlError> {
    program.validate()?;
    let mut b = R1csBuilder::new();

    let pub_in: Vec<LogicalWire> = (0..program.num_inputs)
        .map(|_| b.alloc_public_input())
        .collect();
    let pub_out: Vec<LogicalWire> = (0..program.num_outputs)
        .map(|_| b.alloc_public_output())
        .collect();
    let w_wires: Vec<LogicalWire> = (0..program.num_weights).map(|_| b.alloc_weight()).collect();

    // Allocate initial register state (all constrained to 0).
    let mut curr_regs = [LogicalWire::One; NUM_REGS];
    for reg in &mut curr_regs {
        let w = b.alloc_witness();
        b.enforce_linear_zero(vec![(w, Fp::ONE)]);
        *reg = w;
    }

    for instr in &program.instructions {
        let mut next_regs = [LogicalWire::One; NUM_REGS];
        for reg in &mut next_regs {
            *reg = b.alloc_witness();
        }

        let preserve_all_except = |b: &mut R1csBuilder,
                                   prev: &[LogicalWire; NUM_REGS],
                                   next: &[LogicalWire; NUM_REGS],
                                   skip: Option<usize>| {
            for r in 0..NUM_REGS {
                if Some(r) != skip {
                    b.enforce_linear_eq(vec![(prev[r], Fp::ONE)], next[r]);
                }
            }
        };

        match instr {
            Instruction::Load { dst, src } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                match *src {
                    LoadSource::Imm(v) => {
                        let fp = Fp::from_signed_checked(v)
                            .ok_or(ZkmlError::FieldRangeExceeded { value: v })?;
                        b.enforce_linear_eq(vec![(LogicalWire::One, fp)], next_regs[d]);
                    }
                    LoadSource::Input(idx) => {
                        b.enforce_linear_eq(vec![(pub_in[idx], Fp::ONE)], next_regs[d]);
                    }
                    LoadSource::Weight(idx) => {
                        b.enforce_linear_eq(vec![(w_wires[idx], Fp::ONE)], next_regs[d]);
                    }
                }
            }
            Instruction::Add { dst, a, b: rb } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                b.enforce_linear_eq(
                    vec![
                        (curr_regs[*a as usize], Fp::ONE),
                        (curr_regs[*rb as usize], Fp::ONE),
                    ],
                    next_regs[d],
                );
            }
            Instruction::Sub { dst, a, b: rb } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                b.enforce_linear_eq(
                    vec![
                        (curr_regs[*a as usize], Fp::ONE),
                        (curr_regs[*rb as usize], -Fp::ONE),
                    ],
                    next_regs[d],
                );
            }
            Instruction::Mul { dst, a, b: rb } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                b.enforce_mul(
                    curr_regs[*a as usize],
                    curr_regs[*rb as usize],
                    next_regs[d],
                );
            }
            Instruction::Sra { dst, a, shift } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                let rem = b.alloc_witness();
                let _ = b.enforce_unsigned_bits(rem, *shift as usize);
                let scale = Fp::new(1u64 << (*shift as u32));
                // next_regs[d] * 2^shift + rem == curr_regs[a]
                b.enforce_linear_eq(
                    vec![(next_regs[d], scale), (rem, Fp::ONE)],
                    curr_regs[*a as usize],
                );
            }
            Instruction::CmpGt { dst, a, b: rb } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                // a > b  <=>  (a - b - 1) >= 0  <=>  (a - b - 1) + 2^23 in [2^23, 2^24 - 1]
                let mut sum_terms = Vec::with_capacity(SIGNED_BIT_WIDTH + 3);
                let mut pow2 = Fp::ONE;
                let mut top_bit = LogicalWire::One;
                for k in 0..SIGNED_BIT_WIDTH {
                    let bit = b.alloc_witness();
                    b.enforce_bool(bit);
                    sum_terms.push((bit, pow2));
                    if k + 1 == SIGNED_BIT_WIDTH {
                        top_bit = bit;
                    }
                    pow2 += pow2;
                }
                sum_terms.push((curr_regs[*a as usize], -Fp::ONE));
                sum_terms.push((curr_regs[*rb as usize], Fp::ONE));
                sum_terms.push((LogicalWire::One, Fp::ONE - Fp::new(SIGNED_BIAS as u64)));
                b.enforce_linear_zero(sum_terms);
                b.enforce_linear_eq(vec![(top_bit, Fp::ONE)], next_regs[d]);
            }
            Instruction::Select {
                dst,
                cond,
                a,
                b: rb,
            } => {
                let d = *dst as usize;
                preserve_all_except(&mut b, &curr_regs, &next_regs, Some(d));
                let c_w = curr_regs[*cond as usize];
                let a_w = curr_regs[*a as usize];
                let b_w = curr_regs[*rb as usize];
                b.enforce_bool(c_w);
                // cond * (a - b) = dst - b
                b.add_constraint(
                    vec![(c_w, Fp::ONE)],
                    vec![(a_w, Fp::ONE), (b_w, -Fp::ONE)],
                    vec![(next_regs[d], Fp::ONE), (b_w, -Fp::ONE)],
                );
            }
            Instruction::Halt { out_regs } => {
                preserve_all_except(&mut b, &curr_regs, &next_regs, None);
                for (out_idx, &reg) in out_regs.iter().enumerate() {
                    b.enforce_linear_eq(vec![(curr_regs[reg as usize], Fp::ONE)], pub_out[out_idx]);
                }
            }
        }

        curr_regs = next_regs;
    }

    Ok(b.build(
        format!("zkvm::{}", program.program_id),
        CircuitVersion::new(vm_version.major, vm_version.minor, vm_version.patch),
        domain_tag,
    ))
}

/// Public verifying key for a compiled [`ZkVmProgram`] pinned to an `ImageID`.
#[derive(Clone, Debug)]
pub struct ZkVmVerifyingKey {
    /// Expected RISC Zero-style program + weights + cycle-bound `ImageID`.
    pub image_id: [u8; 32],
    /// zkVM ISA version.
    pub vm_version: ZkVmVersion,
    /// Domain separation tag.
    pub domain_tag: String,
    /// Declared cycle bound.
    pub max_cycles: usize,
    /// Public bytecode program.
    pub program: ZkVmProgram,
    /// Underlying algebraic trace verifying key.
    pub trace_vk: CircuitVerifyingKey,
}

/// Proving key for a compiled [`ZkVmProgram`] + committed weights.
#[derive(Clone, Debug)]
pub struct ZkVmProvingKey {
    /// Public verifying key.
    pub vk: ZkVmVerifyingKey,
    /// Underlying algebraic trace proving key.
    pub trace_pk: CircuitProvingKey,
}

/// Compiles a [`ZkVmProgram`] and binds its `ImageID` to `weights` and `max_cycles`.
pub fn setup_zkvm_keys(
    program: ZkVmProgram,
    weights: &[i64],
    max_cycles: usize,
    ligero_params: LigeroParams,
) -> Result<(ZkVmProvingKey, ZkVmVerifyingKey), ZkmlError> {
    program.validate()?;
    if program.instructions.len() > max_cycles {
        return Err(ZkmlError::CycleBoundExceeded {
            cycles: program.instructions.len(),
            max_cycles,
        });
    }
    if weights.len() != program.num_weights {
        return Err(ZkmlError::WeightsCommitmentMismatch);
    }

    let vm_version = ZkVmVersion::V1;
    let spec = compile_zkvm_trace_spec(&program, vm_version, ZKVM_DOMAIN_TAG)?;
    let (trace_pk, trace_vk) = setup_circuit_keys(spec, weights, ligero_params)?;
    let image_id = program.compute_image_id(
        vm_version,
        ZKVM_DOMAIN_TAG,
        &trace_vk.weights_sha256,
        &trace_vk.weights_commitment.root,
        max_cycles,
    );

    let vk = ZkVmVerifyingKey {
        image_id,
        vm_version,
        domain_tag: ZKVM_DOMAIN_TAG.to_string(),
        max_cycles,
        program,
        trace_vk,
    };
    let pk = ZkVmProvingKey {
        vk: vk.clone(),
        trace_pk,
    };
    Ok((pk, vk))
}

/// Non-interactive zkVM execution receipt verifiable statelessly against `ImageID`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkVmReceipt {
    /// Bound `ImageID` (program + weights + VM version + cycle bound).
    pub image_id: [u8; 32],
    /// zkVM ISA version.
    pub vm_version: ZkVmVersion,
    /// Domain separation tag.
    pub domain_tag: String,
    /// Number of VM cycles executed.
    pub cycles_executed: usize,
    /// Declared cycle bound.
    pub max_cycles: usize,
    /// SHA-256 Merkle root committing to all execution trace rows.
    pub trace_merkle_root: [u8; 32],
    /// Fiat-Shamir spot-opened trace rows authenticated against `trace_merkle_root`.
    pub trace_spot_openings: Vec<TraceRowOpening>,
    /// Algebraic Spartan + Ligero proof of the full trace transition & boundary constraints.
    pub trace_proof: CircuitProof,
}

impl ZkVmReceipt {
    /// Serializes the receipt into canonical deterministic bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(&self.image_id);
        out.extend_from_slice(&self.vm_version.to_le_bytes());
        out.extend_from_slice(&(self.domain_tag.len() as u32).to_le_bytes());
        out.extend_from_slice(self.domain_tag.as_bytes());
        out.extend_from_slice(&(self.cycles_executed as u64).to_le_bytes());
        out.extend_from_slice(&(self.max_cycles as u64).to_le_bytes());
        out.extend_from_slice(&self.trace_merkle_root);
        out.extend_from_slice(&(self.trace_spot_openings.len() as u32).to_le_bytes());
        for open in &self.trace_spot_openings {
            open.write_bytes(&mut out);
        }
        let proof_bytes = self.trace_proof.to_bytes();
        out.extend_from_slice(&(proof_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&proof_bytes);
        out
    }

    /// Exact serialized wire byte length.
    pub fn byte_len(&self) -> usize {
        32 + 6
            + 4
            + self.domain_tag.len()
            + 8
            + 8
            + 32
            + 4
            + self
                .trace_spot_openings
                .iter()
                .map(TraceRowOpening::byte_len)
                .sum::<usize>()
            + 4
            + self.trace_proof.byte_len()
    }

    /// Canonical SHA-256 digest of the serialized receipt bytes.
    pub fn digest(&self) -> [u8; 32] {
        sha256(&self.to_bytes())
    }
}

/// Summary returned upon successful stateless zkVM receipt verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkVmVerificationReport {
    /// Verified `ImageID`.
    pub image_id: [u8; 32],
    /// Verified program identifier.
    pub program_id: String,
    /// Verified cycles executed.
    pub cycles_executed: usize,
    /// Number of algebraic trace constraints verified.
    pub trace_constraints_verified: usize,
    /// Exact serialized receipt byte length.
    pub receipt_bytes: usize,
    /// Canonical receipt SHA-256 digest.
    pub receipt_digest: [u8; 32],
}

/// Deterministic output of executing a [`ZkVmProgram`]:
/// `(journal_output, trace_rows, witness_fp)`.
pub type ZkVmExecutionOutput = (Vec<i64>, Vec<TraceRow>, Vec<Fp>);

/// Executes a [`ZkVmProgram`] deterministically, returning `(journal_output, trace_rows, witness_fp)`.
pub fn execute_zkvm_program(
    program: &ZkVmProgram,
    journal_input: &[i64],
    weights: &[i64],
) -> Result<ZkVmExecutionOutput, ZkmlError> {
    program.validate()?;
    if journal_input.len() != program.num_inputs {
        return Err(ZkmlError::PublicInputMismatch {
            detail: format!(
                "expected {} journal inputs, got {}",
                program.num_inputs,
                journal_input.len()
            ),
        });
    }
    if weights.len() != program.num_weights {
        return Err(ZkmlError::WeightsCommitmentMismatch);
    }

    let mut regs = [0i64; NUM_REGS];
    let mut trace = Vec::with_capacity(program.instructions.len());
    let mut witness = Vec::new();
    let mut journal_output = Vec::new();

    // Initial register witness values (all 0).
    for _ in 0..NUM_REGS {
        witness.push(Fp::ZERO);
    }

    for (cycle, instr) in program.instructions.iter().enumerate() {
        let before = regs;
        let mut after = regs;
        let mut aux_a = 0i64;
        let mut aux_b = 0i64;

        let mut step_aux_witness = Vec::new();

        match instr {
            Instruction::Load { dst, src } => {
                let val = match *src {
                    LoadSource::Imm(v) => v,
                    LoadSource::Input(idx) => journal_input[idx],
                    LoadSource::Weight(idx) => weights[idx],
                };
                after[*dst as usize] = val;
            }
            Instruction::Add { dst, a, b } => {
                after[*dst as usize] = before[*a as usize]
                    .checked_add(before[*b as usize])
                    .ok_or(ZkmlError::FieldRangeExceeded { value: i64::MAX })?;
            }
            Instruction::Sub { dst, a, b } => {
                after[*dst as usize] = before[*a as usize]
                    .checked_sub(before[*b as usize])
                    .ok_or(ZkmlError::FieldRangeExceeded { value: i64::MIN })?;
            }
            Instruction::Mul { dst, a, b } => {
                after[*dst as usize] = before[*a as usize]
                    .checked_mul(before[*b as usize])
                    .ok_or(ZkmlError::FieldRangeExceeded { value: i64::MAX })?;
            }
            Instruction::Sra { dst, a, shift } => {
                let x = before[*a as usize];
                let q = x >> (*shift as u32);
                let rem = x - (q << (*shift as u32));
                after[*dst as usize] = q;
                aux_a = q;
                aux_b = rem;

                step_aux_witness.push(
                    Fp::from_signed_checked(rem)
                        .ok_or(ZkmlError::FieldRangeExceeded { value: rem })?,
                );
                for k in 0..(*shift as usize) {
                    step_aux_witness.push(Fp::new(((rem as u64) >> k) & 1));
                }
            }
            Instruction::CmpGt { dst, a, b } => {
                let diff_m1 = before[*a as usize] - before[*b as usize] - 1;
                let shifted = diff_m1 + SIGNED_BIAS;
                if !(0..(1i64 << SIGNED_BIT_WIDTH)).contains(&shifted) {
                    return Err(ZkmlError::FieldRangeExceeded { value: diff_m1 });
                }
                let res = i64::from(before[*a as usize] > before[*b as usize]);
                after[*dst as usize] = res;
                aux_a = diff_m1;
                aux_b = res;
                for k in 0..SIGNED_BIT_WIDTH {
                    step_aux_witness.push(Fp::new(((shifted as u64) >> k) & 1));
                }
            }
            Instruction::Select { dst, cond, a, b } => {
                let c = before[*cond as usize];
                if c != 0 && c != 1 {
                    return Err(ZkmlError::VmExecutionError {
                        detail: format!("SELECT condition register R{cond} = {c} is not boolean"),
                    });
                }
                after[*dst as usize] = if c == 1 {
                    before[*a as usize]
                } else {
                    before[*b as usize]
                };
            }
            Instruction::Halt { out_regs } => {
                for &r in out_regs {
                    journal_output.push(before[r as usize]);
                }
            }
        }

        // Push next_regs witnesses first, then step-specific auxiliary witnesses
        // (matching the exact allocation order in compile_zkvm_trace_spec).
        for &r_val in &after {
            witness.push(
                Fp::from_signed_checked(r_val)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: r_val })?,
            );
        }
        witness.extend_from_slice(&step_aux_witness);

        trace.push(TraceRow {
            cycle,
            opcode: instr.opcode_tag(),
            regs_before: before,
            regs_after: after,
            aux_a,
            aux_b,
        });
        regs = after;
    }

    Ok((journal_output, trace, witness))
}

fn spot_query_indices(
    image_id: &[u8; 32],
    domain_tag: &str,
    trace_root: &[u8; 32],
    proof_digest: &[u8; 32],
    journal_input: &[i64],
    journal_output: &[i64],
    num_cycles: usize,
) -> Vec<usize> {
    let mut t = Transcript::new(domain_tag);
    t.absorb_bytes("image_id", image_id);
    t.absorb_bytes("trace_merkle_root", trace_root);
    t.absorb_bytes("trace_proof_digest", proof_digest);
    t.absorb_i64_slice("journal_input", journal_input);
    t.absorb_i64_slice("journal_output", journal_output);
    t.challenge_distinct_indices("trace_spot_queries", TRACE_SPOT_QUERIES, num_cycles)
}

/// Executes `pk.vk.program` on `journal_input` and produces `(journal_output, ZkVmReceipt)`.
pub fn prove_zkvm(
    pk: &ZkVmProvingKey,
    journal_input: &[i64],
) -> Result<(Vec<i64>, ZkVmReceipt), ZkmlError> {
    let vk = &pk.vk;
    let (journal_output, trace, witness) =
        execute_zkvm_program(&vk.program, journal_input, &pk.trace_pk.weights_i64)?;
    let cycles_executed = trace.len();
    if cycles_executed > vk.max_cycles {
        return Err(ZkmlError::CycleBoundExceeded {
            cycles: cycles_executed,
            max_cycles: vk.max_cycles,
        });
    }

    let trace_proof = prove_circuit(&pk.trace_pk, journal_input, &journal_output, &witness)?;
    let leaves: Vec<Vec<u8>> = trace.iter().map(TraceRow::to_bytes).collect();
    let tree = MerkleTree::from_leaves(&leaves);
    let trace_merkle_root = tree.root();

    let proof_digest = trace_proof.digest();
    let query_rows = spot_query_indices(
        &vk.image_id,
        &vk.domain_tag,
        &trace_merkle_root,
        &proof_digest,
        journal_input,
        &journal_output,
        cycles_executed,
    );
    let mut trace_spot_openings = Vec::with_capacity(query_rows.len());
    for idx in query_rows {
        trace_spot_openings.push(TraceRowOpening {
            row: trace[idx].clone(),
            merkle_proof: tree.prove(idx),
        });
    }

    let receipt = ZkVmReceipt {
        image_id: vk.image_id,
        vm_version: vk.vm_version,
        domain_tag: vk.domain_tag.clone(),
        cycles_executed,
        max_cycles: vk.max_cycles,
        trace_merkle_root,
        trace_spot_openings,
        trace_proof,
    };
    Ok((journal_output, receipt))
}

/// Generates a zkVM receipt with a deliberately corrupted trace witness for adversarial tests.
pub fn prove_zkvm_with_corrupted_trace_for_test(
    pk: &ZkVmProvingKey,
    journal_input: &[i64],
    corrupt_witness_idx: usize,
) -> Result<(Vec<i64>, ZkVmReceipt), ZkmlError> {
    let vk = &pk.vk;
    let (journal_output, trace, mut witness) =
        execute_zkvm_program(&vk.program, journal_input, &pk.trace_pk.weights_i64)?;
    if corrupt_witness_idx < witness.len() {
        witness[corrupt_witness_idx] += Fp::ONE;
    }
    let trace_proof = prove_circuit_unchecked_for_adversarial_test(
        &pk.trace_pk,
        journal_input,
        &journal_output,
        &witness,
    )?;
    let leaves: Vec<Vec<u8>> = trace.iter().map(TraceRow::to_bytes).collect();
    let tree = MerkleTree::from_leaves(&leaves);
    let trace_merkle_root = tree.root();
    let proof_digest = trace_proof.digest();
    let query_rows = spot_query_indices(
        &vk.image_id,
        &vk.domain_tag,
        &trace_merkle_root,
        &proof_digest,
        journal_input,
        &journal_output,
        trace.len(),
    );
    let trace_spot_openings = query_rows
        .into_iter()
        .map(|idx| TraceRowOpening {
            row: trace[idx].clone(),
            merkle_proof: tree.prove(idx),
        })
        .collect();
    Ok((
        journal_output,
        ZkVmReceipt {
            image_id: vk.image_id,
            vm_version: vk.vm_version,
            domain_tag: vk.domain_tag.clone(),
            cycles_executed: trace.len(),
            max_cycles: vk.max_cycles,
            trace_merkle_root,
            trace_spot_openings,
            trace_proof,
        },
    ))
}

/// Verifies local semantic consistency of an opened [`TraceRow`] against `program.instructions[row.cycle]`.
fn verify_spot_row_semantics(
    program: &ZkVmProgram,
    journal_input: &[i64],
    journal_output: &[i64],
    row: &TraceRow,
) -> bool {
    let Some(instr) = program.instructions.get(row.cycle) else {
        return false;
    };
    if row.opcode != instr.opcode_tag() {
        return false;
    }
    if row.cycle == 0 && row.regs_before != [0i64; NUM_REGS] {
        return false;
    }

    let check_unchanged = |skip: Option<usize>| -> bool {
        (0..NUM_REGS).all(|r| Some(r) == skip || row.regs_after[r] == row.regs_before[r])
    };

    match instr {
        Instruction::Load { dst, src } => {
            let d = *dst as usize;
            if !check_unchanged(Some(d)) {
                return false;
            }
            match *src {
                LoadSource::Imm(v) => row.regs_after[d] == v,
                LoadSource::Input(idx) => row.regs_after[d] == journal_input[idx],
                LoadSource::Weight(_) => true, // Weight value itself is verified algebraically against weights_commitment.
            }
        }
        Instruction::Add { dst, a, b } => {
            let d = *dst as usize;
            check_unchanged(Some(d))
                && row.regs_after[d] == row.regs_before[*a as usize] + row.regs_before[*b as usize]
        }
        Instruction::Sub { dst, a, b } => {
            let d = *dst as usize;
            check_unchanged(Some(d))
                && row.regs_after[d] == row.regs_before[*a as usize] - row.regs_before[*b as usize]
        }
        Instruction::Mul { dst, a, b } => {
            let d = *dst as usize;
            check_unchanged(Some(d))
                && row.regs_after[d] == row.regs_before[*a as usize] * row.regs_before[*b as usize]
        }
        Instruction::Sra { dst, a, shift } => {
            let d = *dst as usize;
            let expected = row.regs_before[*a as usize] >> (*shift as u32);
            let rem = row.regs_before[*a as usize] - (expected << (*shift as u32));
            check_unchanged(Some(d))
                && row.regs_after[d] == expected
                && row.aux_a == expected
                && row.aux_b == rem
                && rem >= 0
                && rem < (1i64 << (*shift as u32))
        }
        Instruction::CmpGt { dst, a, b } => {
            let d = *dst as usize;
            let expected = i64::from(row.regs_before[*a as usize] > row.regs_before[*b as usize]);
            check_unchanged(Some(d)) && row.regs_after[d] == expected && row.aux_b == expected
        }
        Instruction::Select { dst, cond, a, b } => {
            let d = *dst as usize;
            let c = row.regs_before[*cond as usize];
            if c != 0 && c != 1 {
                return false;
            }
            let expected = if c == 1 {
                row.regs_before[*a as usize]
            } else {
                row.regs_before[*b as usize]
            };
            check_unchanged(Some(d)) && row.regs_after[d] == expected
        }
        Instruction::Halt { out_regs } => {
            check_unchanged(None)
                && out_regs
                    .iter()
                    .enumerate()
                    .all(|(i, &r)| row.regs_before[r as usize] == journal_output[i])
        }
    }
}

/// Independently and statelessly verifies a [`ZkVmReceipt`] against `expected_image_id`,
/// `vk`, `journal_input`, and `journal_output`.
///
/// Rejects wrong `ImageID`, wrong VM version, wrong weights, wrong journal inputs/outputs,
/// tampered trace steps, violated cycle bounds, and cross-domain replays.
pub fn verify_zkvm_receipt(
    expected_image_id: &[u8; 32],
    vk: &ZkVmVerifyingKey,
    journal_input: &[i64],
    journal_output: &[i64],
    receipt: &ZkVmReceipt,
) -> Result<ZkVmVerificationReport, ZkmlError> {
    if receipt.domain_tag != vk.domain_tag {
        return Err(ZkmlError::DomainSeparationMismatch {
            expected: vk.domain_tag.clone(),
            actual: receipt.domain_tag.clone(),
        });
    }
    if receipt.vm_version != vk.vm_version {
        return Err(ZkmlError::VmVersionMismatch {
            expected: vk.vm_version.to_string(),
            actual: receipt.vm_version.to_string(),
        });
    }
    if receipt.cycles_executed > vk.max_cycles
        || receipt.max_cycles != vk.max_cycles
        || receipt.cycles_executed != vk.program.instructions.len()
    {
        return Err(ZkmlError::CycleBoundExceeded {
            cycles: receipt.cycles_executed,
            max_cycles: vk.max_cycles,
        });
    }

    // Recompute ImageID independently from vk.program + committed weights + cycle bound.
    let recomputed_image_id = vk.program.compute_image_id(
        vk.vm_version,
        &vk.domain_tag,
        &vk.trace_vk.weights_sha256,
        &vk.trace_vk.weights_commitment.root,
        vk.max_cycles,
    );
    if &recomputed_image_id != expected_image_id
        || &vk.image_id != expected_image_id
        || &receipt.image_id != expected_image_id
    {
        return Err(ZkmlError::ImageIdMismatch);
    }

    // 1. Verify full algebraic Spartan + Ligero trace transition & boundary proof.
    let trace_report = verify_circuit_proof(
        &vk.trace_vk,
        journal_input,
        journal_output,
        &receipt.trace_proof,
    )?;

    // 2. Verify Fiat-Shamir spot-opened trace rows against trace_merkle_root.
    let expected_rows = spot_query_indices(
        expected_image_id,
        &vk.domain_tag,
        &receipt.trace_merkle_root,
        &trace_report.proof_digest,
        journal_input,
        journal_output,
        receipt.cycles_executed,
    );
    if receipt.trace_spot_openings.len() != expected_rows.len() {
        return Err(ZkmlError::TraceVerificationFailed {
            detail: "spot opening count mismatch".to_string(),
        });
    }
    for (opening, &expected_cycle) in receipt.trace_spot_openings.iter().zip(expected_rows.iter()) {
        if opening.row.cycle != expected_cycle
            || opening.merkle_proof.leaf_index != expected_cycle
            || opening.merkle_proof.leaf_count != receipt.cycles_executed
        {
            return Err(ZkmlError::TraceVerificationFailed {
                detail: format!("unexpected trace row index {}", opening.row.cycle),
            });
        }
        let payload = opening.row.to_bytes();
        if !verify_merkle_proof(&receipt.trace_merkle_root, &payload, &opening.merkle_proof) {
            return Err(ZkmlError::TraceVerificationFailed {
                detail: format!("Merkle proof failed at cycle {expected_cycle}"),
            });
        }
        if !verify_spot_row_semantics(&vk.program, journal_input, journal_output, &opening.row) {
            return Err(ZkmlError::TraceVerificationFailed {
                detail: format!("local trace semantics failed at cycle {expected_cycle}"),
            });
        }
    }

    Ok(ZkVmVerificationReport {
        image_id: *expected_image_id,
        program_id: vk.program.program_id.clone(),
        cycles_executed: receipt.cycles_executed,
        trace_constraints_verified: trace_report.constraints_verified,
        receipt_bytes: receipt.byte_len(),
        receipt_digest: receipt.digest(),
    })
}

/// Helper to compute the raw VFS SHA-256 of weights used in a zkVM program.
pub fn zkvm_weights_sha256(weights: &[i64]) -> [u8; 32] {
    compute_weights_sha256(weights)
}

// ============================================================================
// Reference Compiled ZkVmPrograms (Matching Path A Workloads Head-to-Head)
// ============================================================================

/// Builds the Kai 6-feature 3-way routing policy decision program for the general zkVM.
///
/// Weights layout: `[w0..w5, bias, thresh_escalate_minus_1, thresh_reject_minus_1]` (9 entries).
/// Outputs: `[score, decision]` (`0 = LOCAL_FAST`, `1 = VERIFIED_ESCROW`, `2 = REJECT`).
pub fn build_zkvm_kai_policy_program() -> (ZkVmProgram, Vec<i64>) {
    let mut ins = Vec::new();
    // R0 = bias (weight 6)
    ins.push(Instruction::Load {
        dst: 0,
        src: LoadSource::Weight(6),
    });
    for i in 0..6 {
        ins.push(Instruction::Load {
            dst: 1,
            src: LoadSource::Input(i),
        });
        ins.push(Instruction::Load {
            dst: 2,
            src: LoadSource::Weight(i),
        });
        ins.push(Instruction::Mul { dst: 3, a: 1, b: 2 });
        ins.push(Instruction::Add { dst: 0, a: 0, b: 3 });
    }
    // R4 = (score > thresh_escalate - 1)  <=>  score >= thresh_escalate
    ins.push(Instruction::Load {
        dst: 1,
        src: LoadSource::Weight(7),
    });
    ins.push(Instruction::CmpGt { dst: 4, a: 0, b: 1 });
    // R5 = (score > thresh_reject - 1)  <=>  score >= thresh_reject
    ins.push(Instruction::Load {
        dst: 2,
        src: LoadSource::Weight(8),
    });
    ins.push(Instruction::CmpGt { dst: 5, a: 0, b: 2 });
    // R6 = decision = R4 + R5
    ins.push(Instruction::Add { dst: 6, a: 4, b: 5 });
    ins.push(Instruction::Halt {
        out_regs: vec![0, 6],
    });

    let prog = ZkVmProgram {
        program_id: "hanzo.zkvm.kai_policy_gate_6f".to_string(),
        num_inputs: 6,
        num_outputs: 2,
        num_weights: 9,
        instructions: ins,
    };
    // Match KaiPolicyGateV1 reference weights (thresholds stored as t - 1 for strict CmpGt).
    let weights = vec![12, -2, 3, 15, -4, -5, 100, 1199, 2799];
    (prog, weights)
}

/// Builds the deterministic 8-rubric evaluation scorer program for the general zkVM.
///
/// Exercises `LOAD`, `SUB`, `MUL`, `ADD`, `SRA` (fixed-point rescaling), `CMP_GT`, `HALT`.
/// Inputs: 8 observed scores + 8 calibration targets (`16` inputs).
/// Outputs: `[weighted_sum, l2_err, net_score_scaled, pass_flag]`.
pub fn build_zkvm_eval_scorer_program() -> (ZkVmProgram, Vec<i64>) {
    let mut ins = Vec::new();
    // R0 = weighted_sum, R1 = l2_err
    for i in 0..8 {
        ins.push(Instruction::Load {
            dst: 2,
            src: LoadSource::Input(i),
        });
        ins.push(Instruction::Load {
            dst: 3,
            src: LoadSource::Weight(i),
        });
        ins.push(Instruction::Mul { dst: 4, a: 2, b: 3 });
        ins.push(Instruction::Add { dst: 0, a: 0, b: 4 });

        ins.push(Instruction::Load {
            dst: 5,
            src: LoadSource::Input(8 + i),
        });
        ins.push(Instruction::Sub { dst: 6, a: 2, b: 5 });
        ins.push(Instruction::Mul { dst: 7, a: 6, b: 6 });
        ins.push(Instruction::Add { dst: 1, a: 1, b: 7 });
    }
    // net_raw = weighted_sum * 16 - penalty_w * l2_err
    ins.push(Instruction::Load {
        dst: 2,
        src: LoadSource::Imm(16),
    });
    ins.push(Instruction::Mul { dst: 3, a: 0, b: 2 });
    ins.push(Instruction::Load {
        dst: 4,
        src: LoadSource::Weight(8),
    });
    ins.push(Instruction::Mul { dst: 5, a: 1, b: 4 });
    ins.push(Instruction::Sub { dst: 6, a: 3, b: 5 });
    // Fixed-point rescale via exact SRA by 2 bits (divide by 4).
    ins.push(Instruction::Sra {
        dst: 2,
        a: 6,
        shift: 2,
    });
    // Pass flag: net_scaled > (min_pass_scaled - 1)
    ins.push(Instruction::Load {
        dst: 3,
        src: LoadSource::Weight(9),
    });
    ins.push(Instruction::CmpGt { dst: 4, a: 2, b: 3 });
    ins.push(Instruction::Halt {
        out_regs: vec![0, 1, 2, 4],
    });

    let prog = ZkVmProgram {
        program_id: "hanzo.zkvm.deterministic_eval_scorer_8r".to_string(),
        num_inputs: 16,
        num_outputs: 4,
        num_weights: 10,
        instructions: ins,
    };
    let weights = vec![4, 5, 3, 6, 2, 5, 4, 3, 2, 5999];
    (prog, weights)
}

/// Builds a compact 8-feature -> 4-class quantized linear classifier with fixed
/// preprocessing, `SRA` rescaling, and `CMP_GT` + `SELECT` exact argmax tournament.
///
/// Exercises ALL 8 zkVM instructions (`LOAD`, `ADD`, `SUB`, `MUL`, `SRA`, `CMP_GT`, `SELECT`, `HALT`).
/// Outputs: `[logit_0, logit_1, logit_2, logit_3, best_class]`.
pub fn build_zkvm_compact_classifier_program() -> (ZkVmProgram, Vec<i64>) {
    // Weights layout:
    //   prep_mean[8] (0..8), w[4*8] (8..40), bias[4] (40..44) => 44 weights.
    let mut ins = Vec::new();
    // Compute 4 logits into R0, R1, R2, R3 using R4..R7 as temporaries.
    for c in 0..4usize {
        ins.push(Instruction::Load {
            dst: c as u8,
            src: LoadSource::Weight(40 + c),
        });
        for i in 0..8usize {
            ins.push(Instruction::Load {
                dst: 4,
                src: LoadSource::Input(i),
            });
            ins.push(Instruction::Load {
                dst: 5,
                src: LoadSource::Weight(i),
            });
            ins.push(Instruction::Sub { dst: 4, a: 4, b: 5 });
            ins.push(Instruction::Load {
                dst: 6,
                src: LoadSource::Weight(8 + c * 8 + i),
            });
            ins.push(Instruction::Mul { dst: 7, a: 4, b: 6 });
            ins.push(Instruction::Add {
                dst: c as u8,
                a: c as u8,
                b: 7,
            });
        }
        // Quantized fixed-point rescale by >> 2.
        ins.push(Instruction::Sra {
            dst: c as u8,
            a: c as u8,
            shift: 2,
        });
    }

    // Argmax tournament across R0..R3:
    //   R4 = best_val (init R0), R5 = best_idx (init 0), R6 = cmp flag, R7 = candidate idx.
    ins.push(Instruction::Load {
        dst: 5,
        src: LoadSource::Imm(0),
    });
    ins.push(Instruction::Add { dst: 4, a: 0, b: 5 }); // R4 = R0 + 0

    for c in 1..4u8 {
        ins.push(Instruction::CmpGt { dst: 6, a: c, b: 4 });
        ins.push(Instruction::Select {
            dst: 4,
            cond: 6,
            a: c,
            b: 4,
        });
        ins.push(Instruction::Load {
            dst: 7,
            src: LoadSource::Imm(i64::from(c)),
        });
        ins.push(Instruction::Select {
            dst: 5,
            cond: 6,
            a: 7,
            b: 5,
        });
    }

    ins.push(Instruction::Halt {
        out_regs: vec![0, 1, 2, 3, 5],
    });

    let prog = ZkVmProgram {
        program_id: "hanzo.zkvm.compact_classifier_linear_8x4".to_string(),
        num_inputs: 8,
        num_outputs: 5,
        num_weights: 44,
        instructions: ins,
    };

    let mut weights = Vec::with_capacity(44);
    weights.extend_from_slice(&[2, -1, 3, 0, -2, 1, 2, -1]);
    for c in 0..4 {
        for i in 0..8 {
            let v = (((c * 13 + i * 7 + 5) % 15) as i64) - 7;
            weights.push(if v == 0 { 3 } else { v });
        }
    }
    weights.extend_from_slice(&[12, -8, 20, 4]);
    (prog, weights)
}
