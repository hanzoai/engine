# BSD-3-Eco — the Lux Ecosystem Source License

**Governed by the Web3 AI Foundation ([w3a.foundation](https://w3a.foundation))**

Copyright (c) 2026 Lux Industries Inc., Hanzo AI Inc., Zoo Labs Foundation Inc.

This license covers the Proof-of-AI / Proof-of-Inference work — the Rust prover, the
Go verifier and precompiles, and the Solidity settlement contracts — and applies
alongside the BSD-3-Clause terms below.

---

## 1. BSD-3-Clause (the open base)

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that:

1. Redistributions of source code retain the above copyright notice, this list of
   conditions, and the following disclaimer.
2. Redistributions in binary form reproduce the above copyright notice, this list of
   conditions, and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
3. Neither the names of the copyright holders nor of their contributors may be used to
   endorse or promote derived products without specific prior written permission.

## 2. The "Eco" clause — protocol open, implementation ecosystem-bound

**The PROTOCOL is open.** The Proof-of-Inference scheme, the wire formats, the
on-chain verification, the trustless relay, and the economic design are specified in
the public papers and may be implemented by **anyone, on any EVM**, to mine AI
independently. We want every chain to be able to verify AI work.

**The EXACT IMPLEMENTATIONS in this repository** (this Rust/Go/Solidity code, verbatim
or as a derivative) are granted for **production use only on a chain that descends
from the Lux primary network** — a sovereign L1/L2/L3 that bootstraps from and settles
to the Lux primary network (Lux, Hanzo, Zoo, Pars, and their descendants). This keeps
AI settlement reconciled to **one global A-Chain root**, so no proof is double-minted
across chains, and keeps the stack composable, orthogonal, and brand-neutral.

This is one-and-one-way-only by design: there is exactly one global AIVM root, and the
exact code is the canonical client of it.

## 3. What you get

A chain in the ecosystem receives, as public goods:

- the right to mine AI on the **unified AIVM** with fair-launch AICoin issuance under
  the shared halving cap;
- **global reconciliation** to the A-Chain root — `AIComputeRegistry` guarantees a unit
  of work mints exactly once network-wide (no cross-chain double-mint);
- the **post-quantum trustless relay** (`AChainRootOracle`, Warp ML-DSA quorum) and the
  native **compute-proof precompile** — verified, not trusted, permissionless;
- the white-label branding model — your chain, your brand, the same shared infrastructure.

## 4. Commercial / off-ecosystem use

Production use of the exact implementations on a chain that does **not** descend from
the Lux primary network requires a commercial license.

**Contact:** `licensing@w3a.foundation`

---

## Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
