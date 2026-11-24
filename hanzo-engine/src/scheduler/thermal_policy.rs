use std::time::Instant;

/// Hardware telemetry metrics polled from device drivers (NVML, SMC, etc.)
#[derive(Debug, Clone)]
pub struct ThermalState {
    pub temperature_celsius: f32,
    pub power_draw_watts: f32,
    pub gpu_utilization_percent: f32,
    pub timestamp: Instant,
}

/// Dynamic adjustments applied to the scheduler based on thermal conditions
#[derive(Debug, Clone)]
pub struct SchedulerAdjustments {
    pub max_prefill_tokens_multiplier: f32,
    pub max_decode_tokens_multiplier: f32,
    pub speculative_depth_limit: usize,
}

impl Default for SchedulerAdjustments {
    fn default() -> Self {
        Self {
            max_prefill_tokens_multiplier: 1.0,
            max_decode_tokens_multiplier: 1.0,
            speculative_depth_limit: 5, // Default speculative verification depth
        }
    }
}

pub trait ThermalPolicy {
    fn update_state(&mut self, state: ThermalState);
    fn get_adjustments(&self) -> SchedulerAdjustments;
}

/// A threshold-based thermal policy that throttles compute to prevent OS-level frequency collapse.
pub struct ThresholdThermalPolicy {
    throttle_temp: f32,
    critical_temp: f32,
    current_state: Option<ThermalState>,
}

impl ThresholdThermalPolicy {
    pub fn new(throttle_temp: f32, critical_temp: f32) -> Self {
        Self {
            throttle_temp,
            critical_temp,
            current_state: None,
        }
    }
}

impl ThermalPolicy for ThresholdThermalPolicy {
    fn update_state(&mut self, state: ThermalState) {
        self.current_state = Some(state);
    }

    fn get_adjustments(&self) -> SchedulerAdjustments {
        let mut adj = SchedulerAdjustments::default();
        
        if let Some(state) = &self.current_state {
            if state.temperature_celsius >= self.critical_temp {
                // Critical thermal throttling: drastic reduction in compute intensity
                adj.max_prefill_tokens_multiplier = 0.25;
                adj.max_decode_tokens_multiplier = 0.50;
                adj.speculative_depth_limit = 0; // Disable speculative decoding entirely (compute heavy)
            } else if state.temperature_celsius >= self.throttle_temp {
                // Mild throttling: linear scaling based on how close to critical
                let temp_range = self.critical_temp - self.throttle_temp;
                let excess_temp = state.temperature_celsius - self.throttle_temp;
                let throttle_factor = 1.0 - (excess_temp / temp_range) * 0.5; // Throttle up to 50%
                
                adj.max_prefill_tokens_multiplier = throttle_factor;
                // Prioritize decode over prefill during mild throttling to maintain streaming UX
                adj.max_decode_tokens_multiplier = throttle_factor.max(0.75); 
                adj.speculative_depth_limit = 2; // Reduce speculative depth to save power
            }
        }
        
        adj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_throttling_behavior() {
        let mut policy = ThresholdThermalPolicy::new(75.0, 90.0);
        
        // Normal state
        policy.update_state(ThermalState {
            temperature_celsius: 60.0,
            power_draw_watts: 150.0,
            gpu_utilization_percent: 90.0,
            timestamp: Instant::now(),
        });
        let adj = policy.get_adjustments();
        assert_eq!(adj.max_prefill_tokens_multiplier, 1.0);
        assert_eq!(adj.speculative_depth_limit, 5);

        // Mild throttling
        policy.update_state(ThermalState {
            temperature_celsius: 82.5, // 50% through the throttle range
            power_draw_watts: 200.0,
            gpu_utilization_percent: 99.0,
            timestamp: Instant::now(),
        });
        let adj = policy.get_adjustments();
        assert_eq!(adj.max_prefill_tokens_multiplier, 0.75); // 25% throttled
        assert_eq!(adj.speculative_depth_limit, 2);

        // Critical throttling
        policy.update_state(ThermalState {
            temperature_celsius: 92.0,
            power_draw_watts: 250.0,
            gpu_utilization_percent: 100.0,
            timestamp: Instant::now(),
        });
        let adj = policy.get_adjustments();
        assert_eq!(adj.max_prefill_tokens_multiplier, 0.25);
        assert_eq!(adj.speculative_depth_limit, 0);
    }
}
