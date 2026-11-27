// Copyright 2026 Hanzo AI, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Path A: EZKL-style model-specific R1CS arithmetic circuits over Goldilocks
//! `F_p` / `F_{p^2}` proved via Spartan multilinear Sumcheck + Ligero RS-Merkle PCS.
//!
//! Includes three fixed, deterministic, open-specification circuits targeting the
//! tractable scope of `hanzoai/engine#68`:
//! 1. [`CompactClassifierV1`]: fixed preprocessing + quantized 8->8 exact-ReLU + 8->4 argmax MLP.
//! 2. [`KaiPolicyGateV1`]: Kai deterministic 3-way routing/guardrail decision circuit.
//! 3. [`DeterministicEvalScorerV1`]: deterministic weighted rubric + L2 penalty + threshold scorer.

use core::fmt;

use crate::field::{
    eq_eval_point, eq_eval_table, eval_deg2_univariate, eval_deg3_univariate, eval_mle_fp, Fp, Fp2,
};
use crate::merkle::{
    commit_vector_ligero, open_mle_ligero, verify_mle_ligero, LigeroCommitment, LigeroOpeningProof,
    LigeroParams, LigeroProverState,
};
use crate::sha256::{sha256, Sha256, Transcript};
use crate::ZkmlError;

/// Default domain separation tag for Path A model-specific circuits.
pub const CIRCUIT_DOMAIN_TAG: &str = "hanzo.zkml.circuit.v1";

/// Bit-width for exact signed range/ReLU/comparison decompositions (`[-2^23, 2^23 - 1]`).
pub const SIGNED_BIT_WIDTH: usize = 24;

/// Bias added to signed values before unsigned 24-bit decomposition (`2^23`).
pub const SIGNED_BIAS: i64 = 1i64 << (SIGNED_BIT_WIDTH - 1);

/// Semantic version of a compiled R1CS circuit specification.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CircuitVersion {
    /// Major version (breaking constraint/layout changes).
    pub major: u16,
    /// Minor version (compatible circuit extensions).
    pub minor: u16,
    /// Patch version.
    pub patch: u16,
}

impl CircuitVersion {
    /// Constructs a semantic circuit version.
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Encodes as 6 canonical little-endian bytes.
    pub fn to_le_bytes(self) -> [u8; 6] {
        let mut out = [0u8; 6];
        out[0..2].copy_from_slice(&self.major.to_le_bytes());
        out[2..4].copy_from_slice(&self.minor.to_le_bytes());
        out[4..6].copy_from_slice(&self.patch.to_le_bytes());
        out
    }
}

impl fmt::Display for CircuitVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Logical wire reference used during circuit construction before layout assignment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LogicalWire {
    /// Constant `1` wire (always physical index `0`).
    One,
    /// Public input wire `0..num_public_inputs`.
    PublicInput(usize),
    /// Public output wire `0..num_public_outputs`.
    PublicOutput(usize),
    /// Committed weight wire `0..num_weights`.
    Weight(usize),
    /// Private witness wire `0..num_witness`.
    Witness(usize),
}

/// Sparse R1CS constraint: `(sum_i a_i * z_i) * (sum_j b_j * z_j) - (sum_k c_k * z_k) = 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csConstraint {
    /// Left linear combination `A_row`.
    pub a: Vec<(usize, Fp)>,
    /// Right linear combination `B_row`.
    pub b: Vec<(usize, Fp)>,
    /// Output linear combination `C_row`.
    pub c: Vec<(usize, Fp)>,
}

/// Logical constraint prior to physical wire layout resolution.
#[derive(Clone, Debug)]
struct LogicalConstraint {
    a: Vec<(LogicalWire, Fp)>,
    b: Vec<(LogicalWire, Fp)>,
    c: Vec<(LogicalWire, Fp)>,
}

/// Ergonomic R1CS circuit builder that automatically computes power-of-two aligned
/// wire partitions (`public | weights | witness`) for Spartan + Ligero.
#[derive(Clone, Debug)]
pub struct R1csBuilder {
    num_public_inputs: usize,
    num_public_outputs: usize,
    num_weights: usize,
    num_witness: usize,
    constraints: Vec<LogicalConstraint>,
}

impl R1csBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self {
            num_public_inputs: 0,
            num_public_outputs: 0,
            num_weights: 0,
            num_witness: 0,
            constraints: Vec::new(),
        }
    }

    /// Allocates a new public input wire.
    pub fn alloc_public_input(&mut self) -> LogicalWire {
        let w = LogicalWire::PublicInput(self.num_public_inputs);
        self.num_public_inputs += 1;
        w
    }

    /// Allocates a new public output wire.
    pub fn alloc_public_output(&mut self) -> LogicalWire {
        let w = LogicalWire::PublicOutput(self.num_public_outputs);
        self.num_public_outputs += 1;
        w
    }

    /// Allocates a new committed weight wire.
    pub fn alloc_weight(&mut self) -> LogicalWire {
        let w = LogicalWire::Weight(self.num_weights);
        self.num_weights += 1;
        w
    }

    /// Allocates a new private witness wire.
    pub fn alloc_witness(&mut self) -> LogicalWire {
        let w = LogicalWire::Witness(self.num_witness);
        self.num_witness += 1;
        w
    }

    /// Adds a general R1CS constraint `a * b = c`.
    pub fn add_constraint(
        &mut self,
        a: Vec<(LogicalWire, Fp)>,
        b: Vec<(LogicalWire, Fp)>,
        c: Vec<(LogicalWire, Fp)>,
    ) {
        self.constraints.push(LogicalConstraint { a, b, c });
    }

    /// Enforces `a * b = out`.
    pub fn enforce_mul(&mut self, a: LogicalWire, b: LogicalWire, out: LogicalWire) {
        self.add_constraint(vec![(a, Fp::ONE)], vec![(b, Fp::ONE)], vec![(out, Fp::ONE)]);
    }

    /// Enforces `sum_terms = rhs`.
    pub fn enforce_linear_eq(&mut self, lhs: Vec<(LogicalWire, Fp)>, rhs: LogicalWire) {
        self.add_constraint(lhs, vec![(LogicalWire::One, Fp::ONE)], vec![(rhs, Fp::ONE)]);
    }

    /// Enforces a linear combination equals zero: `sum_terms = 0`.
    pub fn enforce_linear_zero(&mut self, terms: Vec<(LogicalWire, Fp)>) {
        self.add_constraint(terms, vec![(LogicalWire::One, Fp::ONE)], vec![]);
    }

    /// Enforces `bit * (1 - bit) = 0`, constraining `bit in {0, 1}`.
    pub fn enforce_bool(&mut self, bit: LogicalWire) {
        self.add_constraint(
            vec![(bit, Fp::ONE)],
            vec![(LogicalWire::One, Fp::ONE), (bit, -Fp::ONE)],
            vec![],
        );
    }

    /// Enforces `0 <= val < 2^bit_width` via boolean bit decomposition.
    /// Returns the allocated bit wires (LSB-first).
    pub fn enforce_unsigned_bits(
        &mut self,
        val: LogicalWire,
        bit_width: usize,
    ) -> Vec<LogicalWire> {
        let mut bits = Vec::with_capacity(bit_width);
        let mut linear = Vec::with_capacity(bit_width);
        let mut pow2 = Fp::ONE;
        for _ in 0..bit_width {
            let b = self.alloc_witness();
            self.enforce_bool(b);
            linear.push((b, pow2));
            bits.push(b);
            pow2 += pow2;
        }
        self.enforce_linear_eq(linear, val);
        bits
    }

    /// Enforces exact signed ReLU `out = max(0, x)` for `x in [-2^23, 2^23 - 1]`.
    /// Returns `(sign_bit, bit_wires)` where `sign_bit = 1` iff `x >= 0`.
    pub fn enforce_exact_relu(
        &mut self,
        x: LogicalWire,
        out: LogicalWire,
    ) -> (LogicalWire, Vec<LogicalWire>) {
        let mut bits = Vec::with_capacity(SIGNED_BIT_WIDTH);
        let mut sum_terms = Vec::with_capacity(SIGNED_BIT_WIDTH + 2);
        let mut pow2 = Fp::ONE;
        for _ in 0..SIGNED_BIT_WIDTH {
            let b = self.alloc_witness();
            self.enforce_bool(b);
            sum_terms.push((b, pow2));
            bits.push(b);
            pow2 += pow2;
        }
        // sum_k 2^k b_k - x - 2^23 = 0
        sum_terms.push((x, -Fp::ONE));
        sum_terms.push((LogicalWire::One, -Fp::new(SIGNED_BIAS as u64)));
        self.enforce_linear_zero(sum_terms);

        let sign_bit = bits[SIGNED_BIT_WIDTH - 1];
        self.enforce_mul(sign_bit, x, out);
        (sign_bit, bits)
    }

    /// Enforces exact signed comparison `ge_bit = (a >= b)` for `a - b in [-2^23, 2^23 - 1]`.
    /// Returns the allocated 24 bit wires (where `bits[23]` equals `ge_bit`).
    pub fn enforce_cmp_ge(
        &mut self,
        a: LogicalWire,
        b: LogicalWire,
        ge_bit: LogicalWire,
    ) -> Vec<LogicalWire> {
        let mut bits = Vec::with_capacity(SIGNED_BIT_WIDTH);
        let mut sum_terms = Vec::with_capacity(SIGNED_BIT_WIDTH + 3);
        let mut pow2 = Fp::ONE;
        for _ in 0..SIGNED_BIT_WIDTH {
            let bit = self.alloc_witness();
            self.enforce_bool(bit);
            sum_terms.push((bit, pow2));
            bits.push(bit);
            pow2 += pow2;
        }
        // sum_k 2^k b_k - (a - b) - 2^23 = 0
        sum_terms.push((a, -Fp::ONE));
        sum_terms.push((b, Fp::ONE));
        sum_terms.push((LogicalWire::One, -Fp::new(SIGNED_BIAS as u64)));
        self.enforce_linear_zero(sum_terms);

        let top_bit = bits[SIGNED_BIT_WIDTH - 1];
        self.enforce_linear_eq(vec![(top_bit, Fp::ONE)], ge_bit);
        bits
    }

    /// Compiles logical wires and constraints into an immutable [`CircuitSpec`].
    pub fn build(
        self,
        circuit_id: impl Into<String>,
        version: CircuitVersion,
        domain_tag: impl Into<String>,
    ) -> CircuitSpec {
        let pub_len = 1 + self.num_public_inputs + self.num_public_outputs;
        let min_quarter = pub_len
            .max(self.num_weights.max(1))
            .max(2)
            .next_power_of_two();
        let min_half = self.num_witness.max(4).next_power_of_two();
        let half = (min_quarter * 2).max(min_half);
        let quarter = half / 2;
        let total_vars = half * 2;
        debug_assert!(total_vars.is_power_of_two() && total_vars >= 8);

        let resolve = |w: LogicalWire| -> usize {
            match w {
                LogicalWire::One => 0,
                LogicalWire::PublicInput(i) => 1 + i,
                LogicalWire::PublicOutput(i) => 1 + self.num_public_inputs + i,
                LogicalWire::Weight(i) => quarter + i,
                LogicalWire::Witness(i) => half + i,
            }
        };

        let constraints = self
            .constraints
            .into_iter()
            .map(|c| R1csConstraint {
                a: c.a
                    .into_iter()
                    .map(|(w, coeff)| (resolve(w), coeff))
                    .collect(),
                b: c.b
                    .into_iter()
                    .map(|(w, coeff)| (resolve(w), coeff))
                    .collect(),
                c: c.c
                    .into_iter()
                    .map(|(w, coeff)| (resolve(w), coeff))
                    .collect(),
            })
            .collect();

        CircuitSpec {
            circuit_id: circuit_id.into(),
            version,
            domain_tag: domain_tag.into(),
            num_public_inputs: self.num_public_inputs,
            num_public_outputs: self.num_public_outputs,
            num_weights: self.num_weights,
            num_witness: self.num_witness,
            quarter_size: quarter,
            half_size: half,
            total_vars,
            constraints,
        }
    }
}

impl Default for R1csBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable compiled R1CS circuit specification with canonical wire partitioning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitSpec {
    /// Human-readable circuit identifier.
    pub circuit_id: String,
    /// Semantic version of the circuit.
    pub version: CircuitVersion,
    /// Domain separation tag binding proofs to this protocol domain.
    pub domain_tag: String,
    /// Number of public input wires.
    pub num_public_inputs: usize,
    /// Number of public output wires.
    pub num_public_outputs: usize,
    /// Number of committed weight wires.
    pub num_weights: usize,
    /// Number of private witness wires.
    pub num_witness: usize,
    /// Size of region `00` (public) and region `01` (weights): `N / 4`.
    pub quarter_size: usize,
    /// Size of region `1*` (private witness): `N / 2`.
    pub half_size: usize,
    /// Total wire space `N = 2^nu` (`>= 8`).
    pub total_vars: usize,
    /// Sparse R1CS constraints.
    pub constraints: Vec<R1csConstraint>,
}

impl CircuitSpec {
    /// Number of Boolean variables `nu = log2(total_vars)`.
    pub const fn log_vars(&self) -> usize {
        self.total_vars.trailing_zeros() as usize
    }

    /// Padded number of constraints `M = 2^mu >= 2`.
    pub fn padded_constraints(&self) -> usize {
        self.constraints.len().max(2).next_power_of_two()
    }

    /// Number of Boolean variables `mu = log2(M)`.
    pub fn log_constraints(&self) -> usize {
        self.padded_constraints().trailing_zeros() as usize
    }

    /// Total non-zero sparse entries across `A`, `B`, and `C`.
    pub fn total_nnz(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| c.a.len() + c.b.len() + c.c.len())
            .sum()
    }

    /// Computes the canonical SHA-256 digest of the entire circuit specification
    /// (ID, version, domain tag, wire geometry, and every sparse constraint term).
    pub fn compute_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"hanzo.zkml.circuit_spec.v1");
        h.update(&(self.circuit_id.len() as u64).to_le_bytes());
        h.update(self.circuit_id.as_bytes());
        h.update(&self.version.to_le_bytes());
        h.update(&(self.domain_tag.len() as u64).to_le_bytes());
        h.update(self.domain_tag.as_bytes());
        for &dim in &[
            self.num_public_inputs,
            self.num_public_outputs,
            self.num_weights,
            self.num_witness,
            self.quarter_size,
            self.half_size,
            self.total_vars,
            self.constraints.len(),
        ] {
            h.update(&(dim as u64).to_le_bytes());
        }
        for c in &self.constraints {
            for list in [&c.a, &c.b, &c.c] {
                h.update(&(list.len() as u64).to_le_bytes());
                for &(idx, coeff) in list {
                    h.update(&(idx as u64).to_le_bytes());
                    h.update(&coeff.to_le_bytes());
                }
            }
        }
        h.finalize()
    }

    /// Assembles the full physical wire vector `z in F_p^N` from the three slices.
    pub fn assemble_wire_vector(
        &self,
        public_inputs: &[Fp],
        public_outputs: &[Fp],
        weights: &[Fp],
        witness: &[Fp],
    ) -> Result<Vec<Fp>, ZkmlError> {
        if public_inputs.len() != self.num_public_inputs {
            return Err(ZkmlError::PublicInputMismatch {
                detail: format!(
                    "expected {} public inputs, got {}",
                    self.num_public_inputs,
                    public_inputs.len()
                ),
            });
        }
        if public_outputs.len() != self.num_public_outputs {
            return Err(ZkmlError::PublicOutputMismatch {
                detail: format!(
                    "expected {} public outputs, got {}",
                    self.num_public_outputs,
                    public_outputs.len()
                ),
            });
        }
        if weights.len() != self.num_weights {
            return Err(ZkmlError::WeightsCommitmentMismatch);
        }
        if witness.len() != self.num_witness {
            return Err(ZkmlError::UnsatisfiedConstraint {
                constraint_index: 0,
            });
        }

        let mut z = vec![Fp::ZERO; self.total_vars];
        z[0] = Fp::ONE;
        z[1..1 + self.num_public_inputs].copy_from_slice(public_inputs);
        let out_start = 1 + self.num_public_inputs;
        z[out_start..out_start + self.num_public_outputs].copy_from_slice(public_outputs);
        z[self.quarter_size..self.quarter_size + self.num_weights].copy_from_slice(weights);
        z[self.half_size..self.half_size + self.num_witness].copy_from_slice(witness);
        Ok(z)
    }

    /// Checks whether every R1CS constraint holds on wire vector `z`.
    pub fn check_satisfaction(&self, z: &[Fp]) -> Result<(), ZkmlError> {
        if z.len() != self.total_vars || z[0] != Fp::ONE {
            return Err(ZkmlError::UnsatisfiedConstraint {
                constraint_index: 0,
            });
        }
        for (i, c) in self.constraints.iter().enumerate() {
            let a = eval_lc(&c.a, z);
            let b = eval_lc(&c.b, z);
            let cv = eval_lc(&c.c, z);
            if a * b != cv {
                return Err(ZkmlError::UnsatisfiedConstraint {
                    constraint_index: i,
                });
            }
        }
        Ok(())
    }
}

fn eval_lc(terms: &[(usize, Fp)], z: &[Fp]) -> Fp {
    let mut acc = Fp::ZERO;
    for &(idx, coeff) in terms {
        acc += coeff * z[idx];
    }
    acc
}

/// Computes the canonical VFS SHA-256 digest of a signed weight vector.
pub fn compute_weights_sha256(weights: &[i64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"hanzo.zkml.weights.v1");
    h.update(&(weights.len() as u64).to_le_bytes());
    for &w in weights {
        h.update(&w.to_le_bytes());
    }
    h.finalize()
}

/// Converts signed integer slice into canonical Goldilocks `Fp` elements with range checking.
pub fn signed_slice_to_fp(vals: &[i64]) -> Result<Vec<Fp>, ZkmlError> {
    let mut out = Vec::with_capacity(vals.len());
    for &v in vals {
        let fp = Fp::from_signed_checked(v).ok_or(ZkmlError::FieldRangeExceeded { value: v })?;
        out.push(fp);
    }
    Ok(out)
}

/// Public verifying key for a compiled circuit + committed model weights.
///
/// The verifier holds ONLY this key (never raw private witness or secret keys) and
/// can verify proofs statelessly with zero Hanzo-operated infrastructure.
#[derive(Clone, Debug)]
pub struct CircuitVerifyingKey {
    /// Circuit identifier.
    pub circuit_id: String,
    /// Semantic circuit version.
    pub version: CircuitVersion,
    /// Domain separation tag.
    pub domain_tag: String,
    /// Canonical SHA-256 digest of `spec`.
    pub circuit_digest: [u8; 32],
    /// Canonical VFS SHA-256 digest of the raw committed weights.
    pub weights_sha256: [u8; 32],
    /// Ligero RS-Merkle commitment to the weight slice (`quarter_size` elements).
    pub weights_commitment: LigeroCommitment,
    /// Ligero PCS parameters enforced by the verifier.
    pub ligero_params: LigeroParams,
    /// Public sparse R1CS specification.
    pub spec: CircuitSpec,
}

/// Proving key pairing the public verifying key with cached Ligero weights state.
#[derive(Clone, Debug)]
pub struct CircuitProvingKey {
    /// Public verifying key.
    pub vk: CircuitVerifyingKey,
    /// Committed signed weights.
    pub weights_i64: Vec<i64>,
    /// Committed field weights.
    pub weights_fp: Vec<Fp>,
    /// Precomputed Ligero prover state for the weight slice.
    pub weights_state: LigeroProverState,
}

/// Generates a deterministic `(CircuitProvingKey, CircuitVerifyingKey)` pair for a
/// compiled [`CircuitSpec`] and quantized weights.
pub fn setup_circuit_keys(
    spec: CircuitSpec,
    weights: &[i64],
    ligero_params: LigeroParams,
) -> Result<(CircuitProvingKey, CircuitVerifyingKey), ZkmlError> {
    if weights.len() != spec.num_weights {
        return Err(ZkmlError::WeightsCommitmentMismatch);
    }
    let weights_fp = signed_slice_to_fp(weights)?;
    let mut padded_weights = vec![Fp::ZERO; spec.quarter_size];
    padded_weights[..weights_fp.len()].copy_from_slice(&weights_fp);

    let weights_state = commit_vector_ligero(&padded_weights, ligero_params);
    let weights_sha256 = compute_weights_sha256(weights);
    let circuit_digest = spec.compute_digest();

    let vk = CircuitVerifyingKey {
        circuit_id: spec.circuit_id.clone(),
        version: spec.version,
        domain_tag: spec.domain_tag.clone(),
        circuit_digest,
        weights_sha256,
        weights_commitment: weights_state.commitment,
        ligero_params,
        spec,
    };
    let pk = CircuitProvingKey {
        vk: vk.clone(),
        weights_i64: weights.to_vec(),
        weights_fp,
        weights_state,
    };
    Ok((pk, vk))
}

/// Non-interactive Spartan Sumcheck + Ligero RS-Merkle proof for a model-specific R1CS circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitProof {
    /// Circuit identifier.
    pub circuit_id: String,
    /// Semantic circuit version.
    pub version: CircuitVersion,
    /// Domain separation tag.
    pub domain_tag: String,
    /// Circuit digest bound into the transcript.
    pub circuit_digest: [u8; 32],
    /// Raw weights VFS SHA-256 bound into the transcript.
    pub weights_sha256: [u8; 32],
    /// Weights Ligero Merkle root bound into the transcript.
    pub weights_root: [u8; 32],
    /// Ligero commitment to the private witness slice (`half_size` elements).
    pub witness_commitment: LigeroCommitment,
    /// Outer degree-3 Sumcheck round evaluations (`mu` rounds).
    pub outer_sumcheck_rounds: Vec<[Fp2; 4]>,
    /// Claimed `A_z_tilde(r_x)`.
    pub claimed_va: Fp2,
    /// Claimed `B_z_tilde(r_x)`.
    pub claimed_vb: Fp2,
    /// Claimed `C_z_tilde(r_x)`.
    pub claimed_vc: Fp2,
    /// Inner degree-2 Sumcheck round evaluations (`nu` rounds).
    pub inner_sumcheck_rounds: Vec<[Fp2; 3]>,
    /// Claimed multilinear evaluation of the weight slice at `r_y[..nu-2]`.
    pub claimed_z_weights_eval: Fp2,
    /// Claimed multilinear evaluation of the private witness slice at `r_y[..nu-1]`.
    pub claimed_z_witness_eval: Fp2,
    /// Ligero opening proof against `vk.weights_commitment`.
    pub weights_opening: LigeroOpeningProof,
    /// Ligero opening proof against `proof.witness_commitment`.
    pub witness_opening: LigeroOpeningProof,
}

impl CircuitProof {
    /// Serializes the proof into canonical deterministic bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(&(self.circuit_id.len() as u32).to_le_bytes());
        out.extend_from_slice(self.circuit_id.as_bytes());
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.domain_tag.len() as u32).to_le_bytes());
        out.extend_from_slice(self.domain_tag.as_bytes());
        out.extend_from_slice(&self.circuit_digest);
        out.extend_from_slice(&self.weights_sha256);
        out.extend_from_slice(&self.weights_root);
        self.witness_commitment.write_bytes(&mut out);

        out.extend_from_slice(&(self.outer_sumcheck_rounds.len() as u32).to_le_bytes());
        for round in &self.outer_sumcheck_rounds {
            for &v in round {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.claimed_va.to_le_bytes());
        out.extend_from_slice(&self.claimed_vb.to_le_bytes());
        out.extend_from_slice(&self.claimed_vc.to_le_bytes());

        out.extend_from_slice(&(self.inner_sumcheck_rounds.len() as u32).to_le_bytes());
        for round in &self.inner_sumcheck_rounds {
            for &v in round {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.claimed_z_weights_eval.to_le_bytes());
        out.extend_from_slice(&self.claimed_z_witness_eval.to_le_bytes());
        self.weights_opening.write_bytes(&mut out);
        self.witness_opening.write_bytes(&mut out);
        out
    }

    /// Exact canonical wire byte length of this proof.
    pub fn byte_len(&self) -> usize {
        4 + self.circuit_id.len()
            + 6
            + 4
            + self.domain_tag.len()
            + 32
            + 32
            + 32
            + (32 + 24)
            + 4
            + self.outer_sumcheck_rounds.len() * 64
            + 48
            + 4
            + self.inner_sumcheck_rounds.len() * 48
            + 32
            + self.weights_opening.byte_len()
            + self.witness_opening.byte_len()
    }

    /// Canonical SHA-256 digest of the serialized proof bytes.
    pub fn digest(&self) -> [u8; 32] {
        sha256(&self.to_bytes())
    }
}

/// Summary returned upon successful stateless circuit proof verification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitVerificationReport {
    /// Verified circuit identifier.
    pub circuit_id: String,
    /// Verified circuit version.
    pub version: CircuitVersion,
    /// Number of R1CS constraints verified.
    pub constraints_verified: usize,
    /// Exact proof byte size.
    pub proof_bytes: usize,
    /// Proof SHA-256 digest.
    pub proof_digest: [u8; 32],
}

fn init_circuit_transcript(
    vk: &CircuitVerifyingKey,
    public_inputs: &[i64],
    public_outputs: &[i64],
    witness_commitment: &LigeroCommitment,
) -> Transcript {
    let mut t = Transcript::new(&vk.domain_tag);
    t.absorb_bytes("circuit_id", vk.circuit_id.as_bytes());
    t.absorb_bytes("version", &vk.version.to_le_bytes());
    t.absorb_bytes("circuit_digest", &vk.circuit_digest);
    t.absorb_bytes("weights_sha256", &vk.weights_sha256);
    let mut w_commit_bytes = Vec::with_capacity(56);
    vk.weights_commitment.write_bytes(&mut w_commit_bytes);
    t.absorb_bytes("weights_commitment", &w_commit_bytes);
    t.absorb_i64_slice("public_inputs", public_inputs);
    t.absorb_i64_slice("public_outputs", public_outputs);
    let mut wit_commit_bytes = Vec::with_capacity(56);
    witness_commitment.write_bytes(&mut wit_commit_bytes);
    t.absorb_bytes("witness_commitment", &wit_commit_bytes);
    t
}

/// Generates a verifiable Spartan + Ligero proof for an R1CS witness.
///
/// Fails with [`ZkmlError::UnsatisfiedConstraint`] if the witness violates any R1CS constraint.
pub fn prove_circuit(
    pk: &CircuitProvingKey,
    public_inputs: &[i64],
    public_outputs: &[i64],
    witness: &[Fp],
) -> Result<CircuitProof, ZkmlError> {
    let pub_in_fp = signed_slice_to_fp(public_inputs)?;
    let pub_out_fp = signed_slice_to_fp(public_outputs)?;
    let z = pk
        .vk
        .spec
        .assemble_wire_vector(&pub_in_fp, &pub_out_fp, &pk.weights_fp, witness)?;
    pk.vk.spec.check_satisfaction(&z)?;
    prove_circuit_inner(pk, public_inputs, public_outputs, &z)
}

/// Runs the circuit prover WITHOUT locally enforcing R1CS satisfaction.
///
/// Intended strictly for adversarial verification tests confirming that the stateless
/// verifier cryptographically rejects forged witnesses and violated constraints.
pub fn prove_circuit_unchecked_for_adversarial_test(
    pk: &CircuitProvingKey,
    public_inputs: &[i64],
    public_outputs: &[i64],
    witness: &[Fp],
) -> Result<CircuitProof, ZkmlError> {
    let pub_in_fp = signed_slice_to_fp(public_inputs)?;
    let pub_out_fp = signed_slice_to_fp(public_outputs)?;
    let z = pk
        .vk
        .spec
        .assemble_wire_vector(&pub_in_fp, &pub_out_fp, &pk.weights_fp, witness)?;
    prove_circuit_inner(pk, public_inputs, public_outputs, &z)
}

fn prove_circuit_inner(
    pk: &CircuitProvingKey,
    public_inputs: &[i64],
    public_outputs: &[i64],
    z: &[Fp],
) -> Result<CircuitProof, ZkmlError> {
    let vk = &pk.vk;
    let spec = &vk.spec;
    let m = spec.padded_constraints();
    let mu = spec.log_constraints();
    let n = spec.total_vars;
    let nu = spec.log_vars();

    // Commit to private witness slice (region `1*`: z[half_size..total_vars]).
    let witness_slice = &z[spec.half_size..spec.total_vars];
    let witness_state = commit_vector_ligero(witness_slice, vk.ligero_params);

    let mut transcript =
        init_circuit_transcript(vk, public_inputs, public_outputs, &witness_state.commitment);

    // Evaluate A*z, B*z, C*z padded to M = 2^mu.
    let mut az = vec![Fp2::ZERO; m];
    let mut bz = vec![Fp2::ZERO; m];
    let mut cz = vec![Fp2::ZERO; m];
    for (i, c) in spec.constraints.iter().enumerate() {
        az[i] = Fp2::from_base(eval_lc(&c.a, z));
        bz[i] = Fp2::from_base(eval_lc(&c.b, z));
        cz[i] = Fp2::from_base(eval_lc(&c.c, z));
    }

    // --- Outer Sumcheck (degree 3, mu rounds) ---
    let tau = transcript.challenge_fp2_vec("outer_tau", mu);
    let mut eq_tau = eq_eval_table(&tau);
    let mut outer_rounds = Vec::with_capacity(mu);
    let mut rx = Vec::with_capacity(mu);

    let two = Fp2::from_u64(2);
    let three = Fp2::from_u64(3);
    for _round in 0..mu {
        let half_len = eq_tau.len() / 2;
        let mut evals = [Fp2::ZERO; 4];
        for i in 0..half_len {
            let e0 = eq_tau[2 * i];
            let e1 = eq_tau[2 * i + 1];
            let de = e1 - e0;

            let a0 = az[2 * i];
            let a1 = az[2 * i + 1];
            let da = a1 - a0;

            let b0 = bz[2 * i];
            let b1 = bz[2 * i + 1];
            let db = b1 - b0;

            let c0 = cz[2 * i];
            let c1 = cz[2 * i + 1];
            let dc = c1 - c0;

            // t = 0
            evals[0] += e0 * (a0 * b0 - c0);
            // t = 1
            evals[1] += e1 * (a1 * b1 - c1);
            // t = 2
            let e2 = e0 + two * de;
            let a2 = a0 + two * da;
            let b2 = b0 + two * db;
            let c2 = c0 + two * dc;
            evals[2] += e2 * (a2 * b2 - c2);
            // t = 3
            let e3 = e0 + three * de;
            let a3 = a0 + three * da;
            let b3 = b0 + three * db;
            let c3 = c0 + three * dc;
            evals[3] += e3 * (a3 * b3 - c3);
        }
        transcript.absorb_fp2_slice("outer_round", &evals);
        let r_k = transcript.challenge_fp2("outer_r");
        outer_rounds.push(evals);
        rx.push(r_k);

        // Fold tables LSB-first.
        for i in 0..half_len {
            eq_tau[i] = eq_tau[2 * i] + r_k * (eq_tau[2 * i + 1] - eq_tau[2 * i]);
            az[i] = az[2 * i] + r_k * (az[2 * i + 1] - az[2 * i]);
            bz[i] = bz[2 * i] + r_k * (bz[2 * i + 1] - bz[2 * i]);
            cz[i] = cz[2 * i] + r_k * (cz[2 * i + 1] - cz[2 * i]);
        }
        eq_tau.truncate(half_len);
        az.truncate(half_len);
        bz.truncate(half_len);
        cz.truncate(half_len);
    }

    let claimed_va = az[0];
    let claimed_vb = bz[0];
    let claimed_vc = cz[0];

    // --- Inner Sumcheck (degree 2, nu rounds) ---
    transcript.absorb_fp2("claimed_va", claimed_va);
    transcript.absorb_fp2("claimed_vb", claimed_vb);
    transcript.absorb_fp2("claimed_vc", claimed_vc);
    let lam_a = transcript.challenge_fp2("inner_lam_a");
    let lam_b = transcript.challenge_fp2("inner_lam_b");
    let lam_c = transcript.challenge_fp2("inner_lam_c");

    // Build M_rx(y) = sum_i eq(rx, i) * (lam_a A_{i,y} + lam_b B_{i,y} + lam_c C_{i,y}).
    let eq_rx = eq_eval_table(&rx);
    let mut m_rx = vec![Fp2::ZERO; n];
    for (i, c) in spec.constraints.iter().enumerate() {
        let w_i = eq_rx[i];
        let wa = w_i * lam_a;
        let wb = w_i * lam_b;
        let wc = w_i * lam_c;
        for &(col, coeff) in &c.a {
            m_rx[col] += wa.mul_base(coeff);
        }
        for &(col, coeff) in &c.b {
            m_rx[col] += wb.mul_base(coeff);
        }
        for &(col, coeff) in &c.c {
            m_rx[col] += wc.mul_base(coeff);
        }
    }

    let mut z_table: Vec<Fp2> = z.iter().map(|&v| Fp2::from_base(v)).collect();
    let mut inner_rounds = Vec::with_capacity(nu);
    let mut ry = Vec::with_capacity(nu);

    for _round in 0..nu {
        let half_len = m_rx.len() / 2;
        let mut evals = [Fp2::ZERO; 3];
        for i in 0..half_len {
            let m0 = m_rx[2 * i];
            let m1 = m_rx[2 * i + 1];
            let dm = m1 - m0;

            let z0 = z_table[2 * i];
            let z1 = z_table[2 * i + 1];
            let dz = z1 - z0;

            evals[0] += m0 * z0;
            evals[1] += m1 * z1;
            evals[2] += (m0 + two * dm) * (z0 + two * dz);
        }
        transcript.absorb_fp2_slice("inner_round", &evals);
        let r_k = transcript.challenge_fp2("inner_r");
        inner_rounds.push(evals);
        ry.push(r_k);

        for i in 0..half_len {
            m_rx[i] = m_rx[2 * i] + r_k * (m_rx[2 * i + 1] - m_rx[2 * i]);
            z_table[i] = z_table[2 * i] + r_k * (z_table[2 * i + 1] - z_table[2 * i]);
        }
        m_rx.truncate(half_len);
        z_table.truncate(half_len);
    }

    // --- Ligero RS-Merkle PCS openings for committed weights and private witness ---
    let (claimed_z_weights_eval, weights_opening) = open_mle_ligero(
        &pk.weights_state,
        &ry[..nu - 2],
        vk.ligero_params,
        &mut transcript,
    );
    let (claimed_z_witness_eval, witness_opening) = open_mle_ligero(
        &witness_state,
        &ry[..nu - 1],
        vk.ligero_params,
        &mut transcript,
    );

    Ok(CircuitProof {
        circuit_id: vk.circuit_id.clone(),
        version: vk.version,
        domain_tag: vk.domain_tag.clone(),
        circuit_digest: vk.circuit_digest,
        weights_sha256: vk.weights_sha256,
        weights_root: vk.weights_commitment.root,
        witness_commitment: witness_state.commitment,
        outer_sumcheck_rounds: outer_rounds,
        claimed_va,
        claimed_vb,
        claimed_vc,
        inner_sumcheck_rounds: inner_rounds,
        claimed_z_weights_eval,
        claimed_z_witness_eval,
        weights_opening,
        witness_opening,
    })
}

/// Independently and statelessly verifies a [`CircuitProof`] against a [`CircuitVerifyingKey`],
/// public inputs, and public outputs.
///
/// Cryptographically rejects wrong weights, wrong outputs, wrong circuit versions or
/// digests, violated intermediate constraints, wrong inputs, and cross-domain replays.
pub fn verify_circuit_proof(
    vk: &CircuitVerifyingKey,
    public_inputs: &[i64],
    public_outputs: &[i64],
    proof: &CircuitProof,
) -> Result<CircuitVerificationReport, ZkmlError> {
    if proof.domain_tag != vk.domain_tag || vk.spec.domain_tag != vk.domain_tag {
        return Err(ZkmlError::DomainSeparationMismatch {
            expected: vk.domain_tag.clone(),
            actual: proof.domain_tag.clone(),
        });
    }
    if proof.circuit_id != vk.circuit_id || proof.version != vk.version {
        return Err(ZkmlError::CircuitVersionMismatch {
            expected: format!("{}@{}", vk.circuit_id, vk.version),
            actual: format!("{}@{}", proof.circuit_id, proof.version),
        });
    }
    if proof.circuit_digest != vk.circuit_digest || vk.spec.compute_digest() != vk.circuit_digest {
        return Err(ZkmlError::CircuitDigestMismatch);
    }
    if proof.weights_sha256 != vk.weights_sha256 || proof.weights_root != vk.weights_commitment.root
    {
        return Err(ZkmlError::WeightsCommitmentMismatch);
    }

    let spec = &vk.spec;
    if public_inputs.len() != spec.num_public_inputs {
        return Err(ZkmlError::PublicInputMismatch {
            detail: format!(
                "expected {} inputs, got {}",
                spec.num_public_inputs,
                public_inputs.len()
            ),
        });
    }
    if public_outputs.len() != spec.num_public_outputs {
        return Err(ZkmlError::PublicOutputMismatch {
            detail: format!(
                "expected {} outputs, got {}",
                spec.num_public_outputs,
                public_outputs.len()
            ),
        });
    }

    let pub_in_fp = signed_slice_to_fp(public_inputs)?;
    let pub_out_fp = signed_slice_to_fp(public_outputs)?;

    let mu = spec.log_constraints();
    let nu = spec.log_vars();
    if proof.outer_sumcheck_rounds.len() != mu {
        return Err(ZkmlError::SumcheckFailed {
            stage: "outer_round_count",
        });
    }
    if proof.inner_sumcheck_rounds.len() != nu {
        return Err(ZkmlError::SumcheckFailed {
            stage: "inner_round_count",
        });
    }
    if proof.witness_commitment.vector_len() != spec.half_size
        || vk.weights_commitment.vector_len() != spec.quarter_size
    {
        return Err(ZkmlError::PcsVerificationFailed {
            slice: "commitment_shape",
        });
    }

    let mut transcript =
        init_circuit_transcript(vk, public_inputs, public_outputs, &proof.witness_commitment);

    // --- Verify Outer Sumcheck ---
    let tau = transcript.challenge_fp2_vec("outer_tau", mu);
    let mut claimed_sum = Fp2::ZERO;
    let mut rx = Vec::with_capacity(mu);
    for &evals in &proof.outer_sumcheck_rounds {
        if evals[0] + evals[1] != claimed_sum {
            return Err(ZkmlError::SumcheckFailed {
                stage: "outer_round_sum",
            });
        }
        transcript.absorb_fp2_slice("outer_round", &evals);
        let r_k = transcript.challenge_fp2("outer_r");
        claimed_sum = eval_deg3_univariate(evals, r_k);
        rx.push(r_k);
    }

    let outer_expected =
        eq_eval_point(&tau, &rx) * (proof.claimed_va * proof.claimed_vb - proof.claimed_vc);
    if outer_expected != claimed_sum {
        return Err(ZkmlError::SumcheckFailed {
            stage: "outer_final_check",
        });
    }

    // --- Verify Inner Sumcheck ---
    transcript.absorb_fp2("claimed_va", proof.claimed_va);
    transcript.absorb_fp2("claimed_vb", proof.claimed_vb);
    transcript.absorb_fp2("claimed_vc", proof.claimed_vc);
    let lam_a = transcript.challenge_fp2("inner_lam_a");
    let lam_b = transcript.challenge_fp2("inner_lam_b");
    let lam_c = transcript.challenge_fp2("inner_lam_c");

    let mut inner_claimed =
        lam_a * proof.claimed_va + lam_b * proof.claimed_vb + lam_c * proof.claimed_vc;
    let mut ry = Vec::with_capacity(nu);
    for &evals in &proof.inner_sumcheck_rounds {
        if evals[0] + evals[1] != inner_claimed {
            return Err(ZkmlError::SumcheckFailed {
                stage: "inner_round_sum",
            });
        }
        transcript.absorb_fp2_slice("inner_round", &evals);
        let r_k = transcript.challenge_fp2("inner_r");
        inner_claimed = eval_deg2_univariate(evals, r_k);
        ry.push(r_k);
    }

    // --- Verify Ligero PCS openings for weights and private witness ---
    if !verify_mle_ligero(
        &vk.weights_commitment,
        &ry[..nu - 2],
        proof.claimed_z_weights_eval,
        &proof.weights_opening,
        vk.ligero_params,
        &mut transcript,
    ) {
        return Err(ZkmlError::PcsVerificationFailed { slice: "weights" });
    }
    if !verify_mle_ligero(
        &proof.witness_commitment,
        &ry[..nu - 1],
        proof.claimed_z_witness_eval,
        &proof.witness_opening,
        vk.ligero_params,
        &mut transcript,
    ) {
        return Err(ZkmlError::PcsVerificationFailed { slice: "witness" });
    }

    // Compute public slice MLE directly from verifier's public inputs & outputs.
    let mut z_pub = Vec::with_capacity(1 + pub_in_fp.len() + pub_out_fp.len());
    z_pub.push(Fp::ONE);
    z_pub.extend_from_slice(&pub_in_fp);
    z_pub.extend_from_slice(&pub_out_fp);
    let z_pub_eval = eval_mle_fp(&z_pub, &ry[..nu - 2]);

    // Combine Z_tilde(ry) across the top-2-bit wire regions:
    //   00 -> public, 01 -> weights, 1* -> witness.
    let r_top = ry[nu - 1];
    let r_sub = ry[nu - 2];
    let one_m_top = Fp2::ONE - r_top;
    let one_m_sub = Fp2::ONE - r_sub;
    let z_ry = one_m_top * one_m_sub * z_pub_eval
        + one_m_top * r_sub * proof.claimed_z_weights_eval
        + r_top * proof.claimed_z_witness_eval;

    // Directly evaluate M_rx(ry) in O(nnz(A, B, C)).
    let eq_rx = eq_eval_table(&rx);
    let eq_ry = eq_eval_table(&ry);
    let mut m_rx_ry = Fp2::ZERO;
    for (i, c) in spec.constraints.iter().enumerate() {
        let w_i = eq_rx[i];
        let mut row_a = Fp2::ZERO;
        for &(col, coeff) in &c.a {
            row_a += eq_ry[col].mul_base(coeff);
        }
        let mut row_b = Fp2::ZERO;
        for &(col, coeff) in &c.b {
            row_b += eq_ry[col].mul_base(coeff);
        }
        let mut row_c = Fp2::ZERO;
        for &(col, coeff) in &c.c {
            row_c += eq_ry[col].mul_base(coeff);
        }
        m_rx_ry += w_i * (lam_a * row_a + lam_b * row_b + lam_c * row_c);
    }

    if m_rx_ry * z_ry != inner_claimed {
        return Err(ZkmlError::SumcheckFailed {
            stage: "inner_final_check",
        });
    }

    Ok(CircuitVerificationReport {
        circuit_id: vk.circuit_id.clone(),
        version: vk.version,
        constraints_verified: spec.constraints.len(),
        proof_bytes: proof.byte_len(),
        proof_digest: proof.digest(),
    })
}

/// Helper to push the 24-bit unsigned decomposition of `val >= 0` into `witness`.
fn push_unsigned_bits(witness: &mut Vec<Fp>, val: i64, bit_width: usize) {
    assert!(val >= 0, "expected non-negative value, got {val}");
    let u = val as u64;
    for k in 0..bit_width {
        let bit = (u >> k) & 1;
        witness.push(Fp::new(bit));
    }
}

/// Helper to push the 24-bit biased signed decomposition of `val in [-2^23, 2^23-1]` into `witness`.
fn push_signed_bits(witness: &mut Vec<Fp>, val: i64) {
    let shifted = val + SIGNED_BIAS;
    assert!(
        (0..(1i64 << SIGNED_BIT_WIDTH)).contains(&shifted),
        "signed value {val} out of 24-bit decomposition range"
    );
    push_unsigned_bits(witness, shifted, SIGNED_BIT_WIDTH);
}

// ============================================================================
// Concrete Circuit 1: CompactClassifierV1 (Fixed Preprocessing + 8->8 ReLU -> 4 Argmax)
// ============================================================================

/// Input dimension of [`CompactClassifierV1`].
pub const CLASSIFIER_IN_DIM: usize = 8;
/// Hidden dimension of [`CompactClassifierV1`].
pub const CLASSIFIER_HIDDEN_DIM: usize = 8;
/// Output class count of [`CompactClassifierV1`].
pub const CLASSIFIER_NUM_CLASSES: usize = 4;

/// Fixed-point quantized 2-layer compact classifier with in-circuit preprocessing,
/// exact 24-bit ReLU activations, and exact 4-class argmax verification.
#[derive(Clone, Debug)]
pub struct CompactClassifierV1 {
    /// Compiled R1CS circuit specification.
    pub spec: CircuitSpec,
}

impl CompactClassifierV1 {
    /// Compiles the deterministic `CompactClassifierV1` circuit specification.
    pub fn compile() -> Self {
        let mut b = R1csBuilder::new();

        // Public inputs: 8 raw quantized features.
        let x_in: Vec<LogicalWire> = (0..CLASSIFIER_IN_DIM)
            .map(|_| b.alloc_public_input())
            .collect();

        // Public outputs: 4 class logits + 1 exact argmax class index (`0..3`).
        let logits_out: Vec<LogicalWire> = (0..CLASSIFIER_NUM_CLASSES)
            .map(|_| b.alloc_public_output())
            .collect();
        let pred_class_out = b.alloc_public_output();

        // Committed weights:
        //   prep_mean[8], prep_scale[8], w1[8*8], b1[8], w2[4*8], b2[4] => 124 weights.
        let prep_mean: Vec<LogicalWire> =
            (0..CLASSIFIER_IN_DIM).map(|_| b.alloc_weight()).collect();
        let prep_scale: Vec<LogicalWire> =
            (0..CLASSIFIER_IN_DIM).map(|_| b.alloc_weight()).collect();
        let w1: Vec<LogicalWire> = (0..CLASSIFIER_HIDDEN_DIM * CLASSIFIER_IN_DIM)
            .map(|_| b.alloc_weight())
            .collect();
        let b1: Vec<LogicalWire> = (0..CLASSIFIER_HIDDEN_DIM)
            .map(|_| b.alloc_weight())
            .collect();
        let w2: Vec<LogicalWire> = (0..CLASSIFIER_NUM_CLASSES * CLASSIFIER_HIDDEN_DIM)
            .map(|_| b.alloc_weight())
            .collect();
        let b2: Vec<LogicalWire> = (0..CLASSIFIER_NUM_CLASSES)
            .map(|_| b.alloc_weight())
            .collect();

        // Step 1: Fixed deterministic preprocessing: x_prep[i] = prep_scale[i] * (x_in[i] - prep_mean[i]).
        let mut x_prep = Vec::with_capacity(CLASSIFIER_IN_DIM);
        for i in 0..CLASSIFIER_IN_DIM {
            let centered = b.alloc_witness();
            b.enforce_linear_eq(vec![(x_in[i], Fp::ONE), (prep_mean[i], -Fp::ONE)], centered);
            let scaled = b.alloc_witness();
            b.enforce_mul(prep_scale[i], centered, scaled);
            x_prep.push(scaled);
        }

        // Step 2: Hidden layer linear transform + exact 24-bit ReLU.
        let mut h_act = Vec::with_capacity(CLASSIFIER_HIDDEN_DIM);
        for j in 0..CLASSIFIER_HIDDEN_DIM {
            let mut sum_terms = Vec::with_capacity(CLASSIFIER_IN_DIM + 1);
            sum_terms.push((b1[j], Fp::ONE));
            for i in 0..CLASSIFIER_IN_DIM {
                let prod = b.alloc_witness();
                b.enforce_mul(w1[j * CLASSIFIER_IN_DIM + i], x_prep[i], prod);
                sum_terms.push((prod, Fp::ONE));
            }
            let pre = b.alloc_witness();
            b.enforce_linear_eq(sum_terms, pre);
            let post = b.alloc_witness();
            let _ = b.enforce_exact_relu(pre, post);
            h_act.push(post);
        }

        // Step 3: Output layer linear transform.
        for c in 0..CLASSIFIER_NUM_CLASSES {
            let mut sum_terms = Vec::with_capacity(CLASSIFIER_HIDDEN_DIM + 1);
            sum_terms.push((b2[c], Fp::ONE));
            for j in 0..CLASSIFIER_HIDDEN_DIM {
                let prod = b.alloc_witness();
                b.enforce_mul(w2[c * CLASSIFIER_HIDDEN_DIM + j], h_act[j], prod);
                sum_terms.push((prod, Fp::ONE));
            }
            b.enforce_linear_eq(sum_terms, logits_out[c]);
        }

        // Step 4: Exact argmax proof via one-hot selector + non-negative gaps (max_logit - logit[c] >= 0).
        let mut one_hot = Vec::with_capacity(CLASSIFIER_NUM_CLASSES);
        let mut sum_oh = Vec::with_capacity(CLASSIFIER_NUM_CLASSES);
        let mut idx_sum = Vec::with_capacity(CLASSIFIER_NUM_CLASSES);
        let mut max_terms = Vec::with_capacity(CLASSIFIER_NUM_CLASSES);
        for (c, &logit_w) in logits_out.iter().enumerate() {
            let sel = b.alloc_witness();
            b.enforce_bool(sel);
            one_hot.push(sel);
            sum_oh.push((sel, Fp::ONE));
            idx_sum.push((sel, Fp::new(c as u64)));
            let sel_logit = b.alloc_witness();
            b.enforce_mul(sel, logit_w, sel_logit);
            max_terms.push((sel_logit, Fp::ONE));
        }
        // sum(one_hot) == 1
        sum_oh.push((LogicalWire::One, -Fp::ONE));
        b.enforce_linear_zero(sum_oh);
        // pred_class == sum(c * one_hot[c])
        b.enforce_linear_eq(idx_sum, pred_class_out);
        // max_logit == sum(one_hot[c] * logits[c])
        let max_logit = b.alloc_witness();
        b.enforce_linear_eq(max_terms, max_logit);

        // For every class c: gap_c = max_logit - logits[c] >= 0 (24-bit unsigned).
        for &logit_w in &logits_out {
            let gap = b.alloc_witness();
            b.enforce_linear_eq(vec![(max_logit, Fp::ONE), (logit_w, -Fp::ONE)], gap);
            let _ = b.enforce_unsigned_bits(gap, SIGNED_BIT_WIDTH);
        }

        let spec = b.build(
            "hanzo.compact_classifier.mlp_8x8x4",
            CircuitVersion::new(1, 0, 0),
            CIRCUIT_DOMAIN_TAG,
        );
        Self { spec }
    }

    /// Deterministic reference open model weights (124 signed fixed-point values).
    pub fn reference_weights() -> Vec<i64> {
        let mut w = Vec::with_capacity(124);
        // prep_mean[8]
        w.extend_from_slice(&[2, -1, 3, 0, -2, 1, 2, -1]);
        // prep_scale[8]
        w.extend_from_slice(&[2, 2, 1, 3, 2, 1, 2, 2]);
        // w1[8 * 8] deterministic small signed weights
        for j in 0..CLASSIFIER_HIDDEN_DIM {
            for i in 0..CLASSIFIER_IN_DIM {
                let raw = (((j * 7 + i * 5 + 3) % 11) as i64) - 5;
                w.push(if raw == 0 { 2 } else { raw });
            }
        }
        // b1[8]
        w.extend_from_slice(&[5, -8, 12, -3, 7, -6, 4, -10]);
        // w2[4 * 8]
        for c in 0..CLASSIFIER_NUM_CLASSES {
            for j in 0..CLASSIFIER_HIDDEN_DIM {
                let raw = (((c * 11 + j * 3 + 1) % 9) as i64) - 4;
                w.push(if raw == 0 { 1 } else { raw });
            }
        }
        // b2[4]
        w.extend_from_slice(&[10, -5, 15, 0]);
        w
    }

    /// Executes the classifier deterministically and generates `(public_outputs, witness_fp)`.
    pub fn evaluate_witness(
        inputs: &[i64],
        weights: &[i64],
    ) -> Result<(Vec<i64>, Vec<Fp>), ZkmlError> {
        if inputs.len() != CLASSIFIER_IN_DIM {
            return Err(ZkmlError::PublicInputMismatch {
                detail: format!("expected {CLASSIFIER_IN_DIM} features"),
            });
        }
        if weights.len() != 124 {
            return Err(ZkmlError::WeightsCommitmentMismatch);
        }

        let prep_mean = &weights[0..8];
        let prep_scale = &weights[8..16];
        let w1 = &weights[16..80];
        let b1 = &weights[80..88];
        let w2 = &weights[88..120];
        let b2 = &weights[120..124];

        let mut witness = Vec::new();

        // Step 1: Preprocessing.
        let mut x_prep = [0i64; CLASSIFIER_IN_DIM];
        for i in 0..CLASSIFIER_IN_DIM {
            let centered = inputs[i] - prep_mean[i];
            let scaled = prep_scale[i] * centered;
            witness.push(
                Fp::from_signed_checked(centered)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: centered })?,
            );
            witness.push(
                Fp::from_signed_checked(scaled)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: scaled })?,
            );
            x_prep[i] = scaled;
        }

        // Step 2: Hidden layer + exact ReLU.
        let mut h_act = [0i64; CLASSIFIER_HIDDEN_DIM];
        for j in 0..CLASSIFIER_HIDDEN_DIM {
            let mut acc = b1[j];
            for i in 0..CLASSIFIER_IN_DIM {
                let prod = w1[j * CLASSIFIER_IN_DIM + i] * x_prep[i];
                witness.push(
                    Fp::from_signed_checked(prod)
                        .ok_or(ZkmlError::FieldRangeExceeded { value: prod })?,
                );
                acc += prod;
            }
            let post = acc.max(0);
            witness.push(
                Fp::from_signed_checked(acc).ok_or(ZkmlError::FieldRangeExceeded { value: acc })?,
            );
            witness.push(
                Fp::from_signed_checked(post)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: post })?,
            );
            push_signed_bits(&mut witness, acc);
            h_act[j] = post;
        }

        // Step 3: Output logits.
        let mut logits = [0i64; CLASSIFIER_NUM_CLASSES];
        for c in 0..CLASSIFIER_NUM_CLASSES {
            let mut acc = b2[c];
            for j in 0..CLASSIFIER_HIDDEN_DIM {
                let prod = w2[c * CLASSIFIER_HIDDEN_DIM + j] * h_act[j];
                witness.push(
                    Fp::from_signed_checked(prod)
                        .ok_or(ZkmlError::FieldRangeExceeded { value: prod })?,
                );
                acc += prod;
            }
            logits[c] = acc;
        }

        // Step 4: Argmax (lowest index breaks ties deterministically).
        let mut best_c = 0usize;
        let mut best_val = logits[0];
        for (c, &v) in logits.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_c = c;
            }
        }
        for (c, &logit) in logits.iter().enumerate() {
            let sel = if c == best_c { 1i64 } else { 0i64 };
            let sel_logit = sel * logit;
            witness.push(Fp::new(sel as u64));
            witness.push(
                Fp::from_signed_checked(sel_logit)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: sel_logit })?,
            );
        }
        witness.push(
            Fp::from_signed_checked(best_val)
                .ok_or(ZkmlError::FieldRangeExceeded { value: best_val })?,
        );
        for &logit in &logits {
            let gap = best_val - logit;
            witness.push(
                Fp::from_signed_checked(gap).ok_or(ZkmlError::FieldRangeExceeded { value: gap })?,
            );
            push_unsigned_bits(&mut witness, gap, SIGNED_BIT_WIDTH);
        }

        let mut public_outputs = Vec::with_capacity(CLASSIFIER_NUM_CLASSES + 1);
        public_outputs.extend_from_slice(&logits);
        public_outputs.push(best_c as i64);
        Ok((public_outputs, witness))
    }
}

// ============================================================================
// Concrete Circuit 2: KaiPolicyGateV1 (Kai 3-Way Deterministic Routing Decision)
// ============================================================================

/// Feature count for [`KaiPolicyGateV1`].
pub const KAI_FEATURE_DIM: usize = 6;

/// Kai deterministic routing / guardrail policy decision circuit.
///
/// Computes `score = bias + sum_i w_i * x_i` and proves exact 3-way routing classification:
/// - `0` = `ROUTE_LOCAL_FAST` (`score < thresh_escalate`)
/// - `1` = `ROUTE_VERIFIED_ESCROW` (`thresh_escalate <= score < thresh_reject`)
/// - `2` = `REJECT_POLICY_VIOLATION` (`score >= thresh_reject`)
#[derive(Clone, Debug)]
pub struct KaiPolicyGateV1 {
    /// Compiled R1CS circuit specification.
    pub spec: CircuitSpec,
}

impl KaiPolicyGateV1 {
    /// Compiles the deterministic `KaiPolicyGateV1` circuit specification.
    pub fn compile() -> Self {
        let mut b = R1csBuilder::new();

        let x: Vec<LogicalWire> = (0..KAI_FEATURE_DIM)
            .map(|_| b.alloc_public_input())
            .collect();
        let score_out = b.alloc_public_output();
        let decision_out = b.alloc_public_output();

        // Weights: w[6], bias, thresh_escalate, thresh_reject => 9 weights.
        let w: Vec<LogicalWire> = (0..KAI_FEATURE_DIM).map(|_| b.alloc_weight()).collect();
        let bias = b.alloc_weight();
        let thresh_escalate = b.alloc_weight();
        let thresh_reject = b.alloc_weight();

        let mut sum_terms = Vec::with_capacity(KAI_FEATURE_DIM + 1);
        sum_terms.push((bias, Fp::ONE));
        for i in 0..KAI_FEATURE_DIM {
            let prod = b.alloc_witness();
            b.enforce_mul(w[i], x[i], prod);
            sum_terms.push((prod, Fp::ONE));
        }
        b.enforce_linear_eq(sum_terms, score_out);

        let is_escalate = b.alloc_witness();
        let _ = b.enforce_cmp_ge(score_out, thresh_escalate, is_escalate);
        let is_reject = b.alloc_witness();
        let _ = b.enforce_cmp_ge(score_out, thresh_reject, is_reject);

        b.enforce_linear_eq(
            vec![(is_escalate, Fp::ONE), (is_reject, Fp::ONE)],
            decision_out,
        );

        let spec = b.build(
            "hanzo.kai.policy_gate_6f",
            CircuitVersion::new(1, 0, 0),
            CIRCUIT_DOMAIN_TAG,
        );
        Self { spec }
    }

    /// Reference deterministic Kai policy weights `[w0..w5, bias, thresh_escalate, thresh_reject]`.
    pub fn reference_weights() -> Vec<i64> {
        vec![12, -2, 3, 15, -4, -5, 100, 1200, 2800]
    }

    /// Evaluates the Kai policy gate and builds `(public_outputs = [score, decision], witness)`.
    pub fn evaluate_witness(
        inputs: &[i64],
        weights: &[i64],
    ) -> Result<(Vec<i64>, Vec<Fp>), ZkmlError> {
        if inputs.len() != KAI_FEATURE_DIM {
            return Err(ZkmlError::PublicInputMismatch {
                detail: format!("expected {KAI_FEATURE_DIM} telemetry features"),
            });
        }
        if weights.len() != KAI_FEATURE_DIM + 3 {
            return Err(ZkmlError::WeightsCommitmentMismatch);
        }

        let mut witness = Vec::new();
        let mut score = weights[KAI_FEATURE_DIM];
        for i in 0..KAI_FEATURE_DIM {
            let prod = weights[i] * inputs[i];
            witness.push(
                Fp::from_signed_checked(prod)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: prod })?,
            );
            score += prod;
        }

        let thresh_escalate = weights[KAI_FEATURE_DIM + 1];
        let thresh_reject = weights[KAI_FEATURE_DIM + 2];
        let is_escalate = i64::from(score >= thresh_escalate);
        let is_reject = i64::from(score >= thresh_reject);

        witness.push(Fp::new(is_escalate as u64));
        push_signed_bits(&mut witness, score - thresh_escalate);
        witness.push(Fp::new(is_reject as u64));
        push_signed_bits(&mut witness, score - thresh_reject);

        let decision = is_escalate + is_reject;
        Ok((vec![score, decision], witness))
    }
}

// ============================================================================
// Concrete Circuit 3: DeterministicEvalScorerV1 (Rubric + L2 Penalty + Pass Gate)
// ============================================================================

/// Number of rubric dimensions in [`DeterministicEvalScorerV1`].
pub const EVAL_RUBRIC_DIM: usize = 8;

/// Deterministic evaluation / grading function circuit computing a weighted rubric
/// score, quadratic L2 calibration penalty, net score, and binary pass/fail decision.
#[derive(Clone, Debug)]
pub struct DeterministicEvalScorerV1 {
    /// Compiled R1CS circuit specification.
    pub spec: CircuitSpec,
}

impl DeterministicEvalScorerV1 {
    /// Compiles the deterministic `DeterministicEvalScorerV1` circuit specification.
    pub fn compile() -> Self {
        let mut b = R1csBuilder::new();

        // Public inputs: 8 observed scores + 8 calibration targets = 16 inputs.
        let scores: Vec<LogicalWire> = (0..EVAL_RUBRIC_DIM)
            .map(|_| b.alloc_public_input())
            .collect();
        let targets: Vec<LogicalWire> = (0..EVAL_RUBRIC_DIM)
            .map(|_| b.alloc_public_input())
            .collect();

        // Public outputs: [weighted_sum, l2_err, net_score, pass_flag].
        let weighted_sum_out = b.alloc_public_output();
        let l2_err_out = b.alloc_public_output();
        let net_score_out = b.alloc_public_output();
        let pass_flag_out = b.alloc_public_output();

        // Weights: rubric_w[8], l2_penalty_w, min_pass_score => 10 weights.
        let rubric_w: Vec<LogicalWire> = (0..EVAL_RUBRIC_DIM).map(|_| b.alloc_weight()).collect();
        let l2_penalty_w = b.alloc_weight();
        let min_pass_score = b.alloc_weight();

        let mut w_terms = Vec::with_capacity(EVAL_RUBRIC_DIM);
        for i in 0..EVAL_RUBRIC_DIM {
            let prod = b.alloc_witness();
            b.enforce_mul(rubric_w[i], scores[i], prod);
            w_terms.push((prod, Fp::ONE));
        }
        b.enforce_linear_eq(w_terms, weighted_sum_out);

        let mut sq_terms = Vec::with_capacity(EVAL_RUBRIC_DIM);
        for i in 0..EVAL_RUBRIC_DIM {
            let diff = b.alloc_witness();
            b.enforce_linear_eq(vec![(scores[i], Fp::ONE), (targets[i], -Fp::ONE)], diff);
            let sq = b.alloc_witness();
            b.enforce_mul(diff, diff, sq);
            sq_terms.push((sq, Fp::ONE));
        }
        b.enforce_linear_eq(sq_terms, l2_err_out);

        let penalty_term = b.alloc_witness();
        b.enforce_mul(l2_penalty_w, l2_err_out, penalty_term);
        b.enforce_linear_eq(
            vec![(weighted_sum_out, Fp::new(16)), (penalty_term, -Fp::ONE)],
            net_score_out,
        );

        let _ = b.enforce_cmp_ge(net_score_out, min_pass_score, pass_flag_out);

        let spec = b.build(
            "hanzo.eval.deterministic_scorer_8r",
            CircuitVersion::new(1, 0, 0),
            CIRCUIT_DOMAIN_TAG,
        );
        Self { spec }
    }

    /// Reference deterministic rubric weights `[w0..w7, l2_penalty_w, min_pass_score]`.
    pub fn reference_weights() -> Vec<i64> {
        vec![4, 5, 3, 6, 2, 5, 4, 3, 2, 24_000]
    }

    /// Evaluates the deterministic scorer and builds `(public_outputs, witness)`.
    pub fn evaluate_witness(
        inputs: &[i64],
        weights: &[i64],
    ) -> Result<(Vec<i64>, Vec<Fp>), ZkmlError> {
        if inputs.len() != 2 * EVAL_RUBRIC_DIM {
            return Err(ZkmlError::PublicInputMismatch {
                detail: format!("expected {} eval inputs", 2 * EVAL_RUBRIC_DIM),
            });
        }
        if weights.len() != EVAL_RUBRIC_DIM + 2 {
            return Err(ZkmlError::WeightsCommitmentMismatch);
        }

        let scores = &inputs[0..EVAL_RUBRIC_DIM];
        let targets = &inputs[EVAL_RUBRIC_DIM..2 * EVAL_RUBRIC_DIM];
        let mut witness = Vec::new();

        let mut weighted_sum = 0i64;
        for i in 0..EVAL_RUBRIC_DIM {
            let prod = weights[i] * scores[i];
            witness.push(
                Fp::from_signed_checked(prod)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: prod })?,
            );
            weighted_sum += prod;
        }

        let mut l2_err = 0i64;
        for i in 0..EVAL_RUBRIC_DIM {
            let diff = scores[i] - targets[i];
            let sq = diff * diff;
            witness.push(
                Fp::from_signed_checked(diff)
                    .ok_or(ZkmlError::FieldRangeExceeded { value: diff })?,
            );
            witness.push(
                Fp::from_signed_checked(sq).ok_or(ZkmlError::FieldRangeExceeded { value: sq })?,
            );
            l2_err += sq;
        }

        let penalty_term = weights[EVAL_RUBRIC_DIM] * l2_err;
        witness.push(Fp::from_signed_checked(penalty_term).ok_or(
            ZkmlError::FieldRangeExceeded {
                value: penalty_term,
            },
        )?);

        let net_score = weighted_sum * 16 - penalty_term;
        let min_pass = weights[EVAL_RUBRIC_DIM + 1];
        let pass_flag = i64::from(net_score >= min_pass);
        push_signed_bits(&mut witness, net_score - min_pass);

        Ok((vec![weighted_sum, l2_err, net_score, pass_flag], witness))
    }
}
