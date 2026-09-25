//! Branch: a side Lane forked from, and joined back into, a main Lane by two reused events.
//!
//! This is vLLM's shared-expert overlap (moe_runner.py): fork an aux stream before the router
//! gate GEMM, run the shared expert on it while the main stream routes and runs the routed
//! experts, join before the combine. Events are capture-safe, so a captured forward keeps the
//! two branches parallel on replay.
//!
//! Contract: callers allocate every buffer either branch touches on the main stream before
//! `run` and drop it after, and cudarc event tracking is off (the engine synchronizes these two
//! streams itself; tracked slices would add their own cross-stream waits and break capture).
//! One Branch serves every layer.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use hanzo_ml::cuda_backend::cudarc::driver::{sys, CudaContext, CudaEvent};
use hanzo_ml::cuda_backend::WrapErr;
use hanzo_ml::Result;

use super::lane::Lane;

pub struct Branch {
    side: Lane,
    open: CudaEvent,
    close: CudaEvent,
    runs: AtomicUsize,
}

impl Branch {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        if ctx.is_event_tracking() {
            hanzo_ml::bail!(
                "Branch needs cudarc event tracking off (disable it before any weights load)"
            );
        }
        // Non-blocking, default priority: vLLM's aux stream.
        let side = Lane::new(ctx.new_stream().w()?)?;
        let flags = Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
        Ok(Self {
            side,
            open: ctx.new_event(flags).w()?,
            close: ctx.new_event(flags).w()?,
            runs: AtomicUsize::new(0),
        })
    }

    pub fn side(&self) -> &Lane {
        &self.side
    }

    /// How many times `run` has forked.
    pub fn runs(&self) -> usize {
        self.runs.load(Ordering::Relaxed)
    }

    /// Fork `side` onto the side Lane after everything already queued on `main`, run `work` on
    /// main, and make main wait for the side before returning. The join is recorded even when a
    /// closure fails, so a capture always ends joined; the first error is returned.
    pub fn run<A, B>(
        &self,
        main: &Lane,
        side: impl FnOnce(&Lane) -> Result<A>,
        work: impl FnOnce() -> Result<B>,
    ) -> Result<(A, B)> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        self.open.record(main.stream()).w()?;
        self.side.stream().wait(&self.open).w()?;
        let a = side(&self.side);
        let b = work();
        let closed = self
            .close
            .record(self.side.stream())
            .and_then(|()| main.stream().wait(&self.close))
            .w();
        let a = a?;
        let b = b?;
        closed?;
        Ok((a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::lane::tests::{device, host, stream};
    use half::bf16;
    use hanzo_ml::cuda_backend::cudarc::driver::sys::{
        CUgraphInstantiate_flags, CUstreamCaptureMode,
    };
    use hanzo_ml::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use hanzo_ml::Device;

    const N: usize = 4096;

    fn nan(s: &Arc<CudaStream>, n: usize) -> CudaSlice<bf16> {
        s.clone_htod(&vec![bf16::NAN; n]).unwrap()
    }

    fn operands(s: &Arc<CudaStream>) -> (CudaSlice<bf16>, CudaSlice<bf16>) {
        let a: Vec<bf16> = (0..N * N).map(|i| bf16::from_f32(((i % 97) as f32 - 48.) / 64.)).collect();
        let b: Vec<bf16> = (0..N * N).map(|i| bf16::from_f32(((i % 89) as f32 - 44.) / 64.)).collect();
        (s.clone_htod(&a).unwrap(), s.clone_htod(&b).unwrap())
    }

    fn copy(lane: &Lane, src: &CudaSlice<bf16>, dst: &mut CudaSlice<bf16>) -> Result<()> {
        let s = lane.stream();
        let (sp, _a) = src.device_ptr(s);
        let (dp, _b) = dst.device_ptr_mut(s);
        unsafe { sys::cuMemcpyDtoDAsync_v2(dp, sp, src.len() * 2, s.cu_stream()) }
            .result()
            .w()
    }

    /// (a) The side branch sees what main queued before the fork.
    #[test]
    fn side_after_main() {
        let dev = device();
        let s = stream(&dev);
        let main = Lane::new(s.clone()).unwrap();
        let branch = Branch::new(s.context()).unwrap();
        let (a, b) = operands(&s);
        let mut x = nan(&s, N * N);
        let mut y = nan(&s, N * N);
        s.synchronize().unwrap();
        main.linear(&a, &b, &mut x, N, N, N).unwrap();
        branch
            .run(&main, |side| copy(side, &x, &mut y), || Ok(()))
            .unwrap();
        s.synchronize().unwrap();
        let (hx, hy) = (host(&s, &x), host(&s, &y));
        let nans = hy.iter().filter(|v| bf16::from_bits(**v).is_nan()).count();
        assert_eq!(nans, 0, "side copied before main's GEMM finished");
        assert!(hx == hy);
    }

    /// (b) Main, after run, sees what the side branch wrote.
    #[test]
    fn main_after_side() {
        let dev = device();
        let s = stream(&dev);
        let main = Lane::new(s.clone()).unwrap();
        let branch = Branch::new(s.context()).unwrap();
        let (a, b) = operands(&s);
        let mut want = nan(&s, N * N);
        main.linear(&a, &b, &mut want, N, N, N).unwrap();
        let want = host(&s, &want);
        let mut z = nan(&s, N * N);
        s.synchronize().unwrap();
        branch
            .run(&main, |side| side.linear(&a, &b, &mut z, N, N, N), || Ok(()))
            .unwrap();
        let got = host(&s, &z);
        assert!(got == want, "main read the side's output before the side finished");
    }

    unsafe fn nodes(g: sys::CUgraph) -> (Vec<sys::CUgraphNode>, Vec<(sys::CUgraphNode, sys::CUgraphNode)>) {
        let mut n = 0usize;
        sys::cuGraphGetNodes(g, std::ptr::null_mut(), &mut n).result().unwrap();
        let mut v = vec![std::ptr::null_mut(); n];
        sys::cuGraphGetNodes(g, v.as_mut_ptr(), &mut n).result().unwrap();
        let mut e = 0usize;
        sys::cuGraphGetEdges_v2(g, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), &mut e)
            .result()
            .unwrap();
        let mut from = vec![std::ptr::null_mut(); e];
        let mut to = vec![std::ptr::null_mut(); e];
        sys::cuGraphGetEdges_v2(g, from.as_mut_ptr(), to.as_mut_ptr(), std::ptr::null_mut(), &mut e)
            .result()
            .unwrap();
        (v, from.into_iter().zip(to).collect())
    }

    /// (c) Captured on main: memset A; run(side: memset B, work: memset C); memset D. The graph
    /// orders A before B, leaves B and C unordered, and orders both before D; replay == eager.
    #[test]
    fn capture_shape() {
        let dev = device();
        let s = stream(&dev);
        let main = Lane::new(s.clone()).unwrap();
        let branch = Branch::new(s.context()).unwrap();
        let mut bufs: Vec<CudaSlice<u32>> = (0..4).map(|_| s.alloc_zeros::<u32>(1 << 20).unwrap()).collect();
        let ptrs: Vec<sys::CUdeviceptr> = bufs.iter_mut().map(|b| b.device_ptr_mut(&s).0).collect();
        let set = |lane: &Lane, i: usize, v: u32| -> Result<()> {
            unsafe { sys::cuMemsetD32Async(ptrs[i], v, 1 << 20, lane.stream().cu_stream()) }
                .result()
                .w()
        };
        let body = |v: u32| {
            set(&main, 0, v).unwrap();
            branch
                .run(&main, |side| set(side, 1, v + 1), || set(&main, 2, v + 2))
                .unwrap();
            set(&main, 3, v + 3).unwrap();
        };
        s.synchronize().unwrap();
        s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).unwrap();
        body(10);
        let g = s
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .unwrap()
            .unwrap();
        let (vs, edges) = unsafe { nodes(g.cu_graph()) };
        let mut which = std::collections::HashMap::new();
        for &n in &vs {
            let mut p: sys::CUDA_MEMSET_NODE_PARAMS = unsafe { std::mem::zeroed() };
            if unsafe { sys::cuGraphMemsetNodeGetParams(n, &mut p) }.result().is_ok() {
                which.insert(n, ptrs.iter().position(|&q| q == p.dst).unwrap());
            }
        }
        assert_eq!(which.len(), 4, "graph has {} nodes, {} memsets", vs.len(), which.len());
        let reach = |from: usize, to: usize| {
            let start = *which.iter().find(|(_, &i)| i == from).unwrap().0;
            let goal = *which.iter().find(|(_, &i)| i == to).unwrap().0;
            let mut stack = vec![start];
            let mut seen = std::collections::HashSet::new();
            while let Some(n) = stack.pop() {
                if n == goal {
                    return true;
                }
                if seen.insert(n) {
                    stack.extend(edges.iter().filter(|(f, _)| *f == n).map(|(_, t)| *t));
                }
            }
            false
        };
        assert!(reach(0, 1), "A before B");
        assert!(!reach(1, 2) && !reach(2, 1), "B and C unordered");
        assert!(reach(1, 3) && reach(2, 3), "B and C before D");
        g.launch().unwrap();
        s.synchronize().unwrap();
        let replay: Vec<Vec<u32>> = bufs.iter().map(|b| s.clone_dtoh(b).unwrap()).collect();
        body(10);
        s.synchronize().unwrap();
        let eager: Vec<Vec<u32>> = bufs.iter().map(|b| s.clone_dtoh(b).unwrap()).collect();
        assert!(replay == eager);
        assert!((0..4).all(|i| eager[i].iter().all(|&v| v == 10 + i as u32)));
    }

    /// (d) A context that still tracks events is refused.
    #[test]
    fn tracked_context_refused() {
        let dev = Device::new_cuda(0).unwrap();
        let s = stream(&dev);
        assert!(s.context().is_event_tracking());
        let err = Branch::new(s.context()).err().expect("tracked context accepted");
        assert!(err.to_string().contains("event tracking"), "{err}");
    }

    /// (e) A side closure that fails inside a capture still joins, so the capture ends cleanly.
    #[test]
    fn failing_side_still_joins() {
        let dev = device();
        let s = stream(&dev);
        let main = Lane::new(s.clone()).unwrap();
        let branch = Branch::new(s.context()).unwrap();
        s.synchronize().unwrap();
        s.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).unwrap();
        let r = branch.run(&main, |_| -> Result<()> { hanzo_ml::bail!("side failed") }, || Ok(()));
        assert!(r.is_err());
        let g = s.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);
        assert!(g.is_ok(), "capture did not end joined: {:?}", g.err());
    }
}
