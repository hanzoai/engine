#pragma once

// Shared paged-attention function constants, declared once with globally
// unique indices. Multiple kernels (pagedattention, reshape_and_cache,
// gather_kv_cache) reference these; declaring them here and #including keeps
// the indices consistent whether the kernels are compiled as separate
// translation units (precompiled metallib) or concatenated into one source
// (runtime fallback). A kernel only needs to set the constants it actually
// uses at pipeline creation; unused ones may remain unset.
constant bool use_partitioning [[function_constant(10)]];
constant bool use_alibi [[function_constant(20)]];
constant bool use_fp8_scales [[function_constant(30)]];
constant bool use_sinks [[function_constant(40)]];
