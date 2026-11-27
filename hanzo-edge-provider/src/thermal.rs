// Copyright 2026 Hanzo AI Inc.
// SPDX-License-Identifier: Apache-2.0

//! First-principles RC thermal network governor, progressive DVFS throttling,
//! battery state-of-charge integration, and deterministic 4-guardrail automatic
//! market withdrawal policy.
//!
//! ## Thermal Physics
//! Lumped single-pole RC thermal ODE integrated with explicit sub-stepping:
//!
//! ```text
//! dT / dt = (P_watts - (T - T_amb) / R_th) / C_th
//! ```
//!
//! Where `P_watts` is active package power, `T_amb` is ambient temperature (°C),
//! `R_th` is package-to-ambient thermal resistance (°C/W), and `C_th` is lumped
//! thermal capacitance (J/°C).

use crate::device::EdgeDeviceProfile;
use std::fmt;

/// Progressive DVFS thermal throttle states enforced by [`ThermalPowerGovernor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThrottleLevel {
    /// Nominal unthrottled operation (`1.00x` clocks, `1.00x` active power cap).
    Full,
    /// Soft thermal throttle triggered at `T >= soft_throttle_temp_c` (`0.70x` clocks, `0.62x` power).
    ThermalThrottle70,
    /// Hard thermal throttle triggered at `T >= hard_throttle_temp_c` (`0.45x` clocks, `0.40x` power).
    ThermalThrottle45,
    /// Hard thermal/safety withdrawal triggered at `T >= withdraw_temp_c` (`0.00x`, market exit).
    AutoWithdraw,
}

impl ThrottleLevel {
    /// Canonical wire/manifest string label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::ThermalThrottle70 => "thermal_throttle_70",
            Self::ThermalThrottle45 => "thermal_throttle_45",
            Self::AutoWithdraw => "auto_withdraw",
        }
    }

    /// Parses a wire/manifest throttle state label.
    #[must_use]
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Self::Full),
            "thermal_throttle_70" => Some(Self::ThermalThrottle70),
            "thermal_throttle_45" => Some(Self::ThermalThrottle45),
            "auto_withdraw" => Some(Self::AutoWithdraw),
            _ => None,
        }
    }

    /// Relative clock/memory-bandwidth scaling factor under this DVFS state.
    #[must_use]
    pub const fn clock_scale(self) -> f64 {
        match self {
            Self::Full => 1.00,
            Self::ThermalThrottle70 => 0.70,
            Self::ThermalThrottle45 => 0.45,
            Self::AutoWithdraw => 0.00,
        }
    }

    /// Relative dynamic power scaling factor (cubic/quadratic voltage-frequency reduction).
    #[must_use]
    pub const fn power_scale(self) -> f64 {
        match self {
            Self::Full => 1.00,
            Self::ThermalThrottle70 => 0.62,
            Self::ThermalThrottle45 => 0.40,
            Self::AutoWithdraw => 0.00,
        }
    }

    /// Calibrated live per-token DVFS slowdown sleep (in microseconds) used during
    /// live loopback CPU serving so progressive throttling visibly increases measured ITL.
    #[must_use]
    pub const fn live_pacing_delay_us(self) -> u64 {
        match self {
            Self::Full => 0,
            Self::ThermalThrottle70 => 220,
            Self::ThermalThrottle45 => 520,
            Self::AutoWithdraw => 0,
        }
    }
}

impl fmt::Display for ThrottleLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Deterministic owner-protection guardrail reasons that trigger automatic market withdrawal.
#[derive(Debug, Clone, PartialEq)]
pub enum WithdrawalReason {
    /// Guardrail 1: Package or skin temperature reached or exceeded `withdraw_temp_c`.
    ThermalCeilingExceeded {
        /// Observed temperature in °C.
        temp_c: f64,
        /// Configured hard withdrawal threshold in °C.
        limit_c: f64,
    },
    /// Guardrail 2: Battery state-of-charge hit floor or AC unplugged with unsafe power draw.
    BatteryFloorReached {
        /// Current battery state-of-charge (%).
        battery_pct: f64,
        /// Configured owner battery floor (%).
        min_battery_pct: f64,
        /// Whether AC wall power is currently connected.
        on_ac_power: bool,
    },
    /// Guardrail 3: Availability window elapsed or user resumed foreground activity.
    AvailabilityWindowExpired {
        /// Current logical Unix timestamp (seconds).
        now_unix: u64,
        /// Advertised availability window end timestamp (seconds).
        window_end_unix: u64,
        /// Whether owner foreground preemption triggered immediate exit.
        user_foreground_active: bool,
    },
    /// Guardrail 4: Free unified RAM dropped below mandatory user-reserved headroom.
    UserMemoryPressure {
        /// Observed free unified memory in GiB.
        free_ram_gib: f64,
        /// Mandatory user-reserved headroom floor in GiB.
        min_user_headroom_gib: f64,
    },
}

impl WithdrawalReason {
    /// Machine-readable short guardrail category code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::ThermalCeilingExceeded { .. } => "thermal_ceiling_exceeded",
            Self::BatteryFloorReached { .. } => "battery_floor_reached",
            Self::AvailabilityWindowExpired { .. } => "availability_window_expired",
            Self::UserMemoryPressure { .. } => "user_memory_pressure",
        }
    }
}

impl fmt::Display for WithdrawalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThermalCeilingExceeded { temp_c, limit_c } => {
                write!(
                    f,
                    "thermal_ceiling_exceeded(temp={temp_c:.2}C >= limit={limit_c:.2}C)"
                )
            }
            Self::BatteryFloorReached {
                battery_pct,
                min_battery_pct,
                on_ac_power,
            } => write!(
                f,
                "battery_floor_reached(soc={battery_pct:.2}% <= min={min_battery_pct:.2}%, on_ac={on_ac_power})"
            ),
            Self::AvailabilityWindowExpired {
                now_unix,
                window_end_unix,
                user_foreground_active,
            } => write!(
                f,
                "availability_window_expired(now={now_unix}, end={window_end_unix}, fg_active={user_foreground_active})"
            ),
            Self::UserMemoryPressure {
                free_ram_gib,
                min_user_headroom_gib,
            } => write!(
                f,
                "user_memory_pressure(free={free_ram_gib:.2}GiB < min={min_user_headroom_gib:.2}GiB)"
            ),
        }
    }
}

/// Snapshot of thermal, power, battery, and memory telemetry for a governor step.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetrySnapshot {
    /// Current temperature in °C.
    pub temp_c: f64,
    /// Instantaneous active package power in Watts.
    pub active_watts: f64,
    /// Cumulative energy consumed since governor start in Joules.
    pub cumulative_joules: f64,
    /// Current battery state-of-charge (%).
    pub battery_pct: f64,
    /// Whether AC power is connected.
    pub on_ac_power: bool,
    /// Observed free unified memory in GiB.
    pub free_ram_gib: f64,
    /// Active DVFS throttle level.
    pub throttle_level: ThrottleLevel,
    /// Active withdrawal trigger, if any guardrail has tripped.
    pub withdrawal_reason: Option<WithdrawalReason>,
}

/// Live RC thermal/power governor and 4-guardrail automatic withdrawal controller.
#[derive(Debug, Clone)]
pub struct ThermalPowerGovernor {
    profile: EdgeDeviceProfile,
    /// Current package or skin temperature in °C.
    pub temp_c: f64,
    /// Peak temperature observed during the provider session in °C.
    pub peak_temp_c: f64,
    /// Current battery state-of-charge (% in `[0.0, 100.0]`).
    pub battery_pct: f64,
    /// Configured owner minimum battery floor (%).
    pub min_battery_pct: f64,
    /// Whether AC wall power is currently connected.
    pub on_ac_power: bool,
    /// Advertised power cap override in Watts.
    pub watts_cap: f64,
    /// Current observed free unified memory in GiB.
    pub free_ram_gib: f64,
    /// Mandatory user-reserved memory headroom floor in GiB.
    pub min_user_headroom_gib: f64,
    /// Advertised availability window start Unix timestamp (seconds).
    pub availability_start_unix: u64,
    /// Advertised availability window end Unix timestamp (seconds).
    pub availability_end_unix: u64,
    /// Current logical Unix clock (seconds).
    pub now_unix: u64,
    /// Whether user foreground activity is preempting background inference.
    pub user_foreground_active: bool,
    /// Active DVFS throttle state.
    pub throttle_level: ThrottleLevel,
    /// Whether DVFS throttling is enabled (always `true` in production; `false` only in negative-regime tests).
    pub dvfs_enabled: bool,
    /// Thermal acceleration multiplier for short deterministic live bursts (`1.0` = real-time seconds).
    pub thermal_time_scale: f64,
    /// Cumulative energy consumed in Joules.
    pub cumulative_joules: f64,
    /// Active withdrawal reason if any guardrail tripped.
    pub withdrawal_reason: Option<WithdrawalReason>,
}

impl ThermalPowerGovernor {
    /// Creates a new governor initialized at warm idle temperature for `profile`.
    #[must_use]
    pub fn new(
        profile: EdgeDeviceProfile,
        on_ac_power: bool,
        initial_battery_pct: f64,
        availability_start_unix: u64,
        availability_end_unix: u64,
    ) -> Self {
        let initial_temp =
            profile.ambient_temp_c + profile.idle_watts * profile.thermal_resistance_c_per_w * 0.5;
        let watts_cap = profile.active_power_cap_watts(on_ac_power);
        let free_ram_gib = profile.user_reserved_memory_gib + 2.0;
        let min_user_headroom_gib = profile.user_reserved_memory_gib;
        let min_battery_pct = profile.default_min_battery_pct;
        Self {
            profile,
            temp_c: initial_temp,
            peak_temp_c: initial_temp,
            battery_pct: initial_battery_pct.clamp(0.0, 100.0),
            min_battery_pct,
            on_ac_power,
            watts_cap,
            free_ram_gib,
            min_user_headroom_gib,
            availability_start_unix,
            availability_end_unix,
            now_unix: availability_start_unix,
            user_foreground_active: false,
            throttle_level: ThrottleLevel::Full,
            dvfs_enabled: true,
            thermal_time_scale: 1.0,
            cumulative_joules: 0.0,
            withdrawal_reason: None,
        }
    }

    /// Returns the underlying hardware profile.
    #[must_use]
    pub const fn profile(&self) -> &EdgeDeviceProfile {
        &self.profile
    }

    /// Returns whether the provider has tripped any automatic withdrawal guardrail.
    #[must_use]
    pub const fn is_withdrawn(&self) -> bool {
        self.withdrawal_reason.is_some()
    }

    /// Computes instantaneous active package power in Watts given `load_fraction` in `[0, 1]`.
    #[must_use]
    pub fn current_power_watts(&self, load_fraction: f64) -> f64 {
        let load = load_fraction.clamp(0.0, 1.0);
        let cap = self
            .watts_cap
            .min(self.profile.active_power_cap_watts(self.on_ac_power));
        let dynamic_headroom = (cap - self.profile.idle_watts).max(0.0);
        let scale = if self.dvfs_enabled {
            self.throttle_level.power_scale()
        } else {
            1.0
        };
        self.profile.idle_watts + dynamic_headroom * load * scale
    }

    /// Evaluates all four deterministic automatic market withdrawal guardrails without mutating time.
    pub fn check_guardrails(&mut self) -> Option<WithdrawalReason> {
        if let Some(existing) = &self.withdrawal_reason {
            return Some(existing.clone());
        }

        // Guardrail 1: Hard thermal ceiling.
        if self.temp_c >= self.profile.withdraw_temp_c {
            self.throttle_level = ThrottleLevel::AutoWithdraw;
            let reason = WithdrawalReason::ThermalCeilingExceeded {
                temp_c: self.temp_c,
                limit_c: self.profile.withdraw_temp_c,
            };
            self.withdrawal_reason = Some(reason.clone());
            return Some(reason);
        }

        // Guardrail 2: Battery floor when off AC (or critical battery even if AC drops).
        if !self.on_ac_power && self.battery_pct <= self.min_battery_pct {
            self.throttle_level = ThrottleLevel::AutoWithdraw;
            let reason = WithdrawalReason::BatteryFloorReached {
                battery_pct: self.battery_pct,
                min_battery_pct: self.min_battery_pct,
                on_ac_power: self.on_ac_power,
            };
            self.withdrawal_reason = Some(reason.clone());
            return Some(reason);
        }

        // Guardrail 3: Availability window expiry or user foreground activity preemption.
        if self.now_unix >= self.availability_end_unix || self.user_foreground_active {
            self.throttle_level = ThrottleLevel::AutoWithdraw;
            let reason = WithdrawalReason::AvailabilityWindowExpired {
                now_unix: self.now_unix,
                window_end_unix: self.availability_end_unix,
                user_foreground_active: self.user_foreground_active,
            };
            self.withdrawal_reason = Some(reason.clone());
            return Some(reason);
        }

        // Guardrail 4: User memory pressure below mandatory user-reserved headroom.
        if self.free_ram_gib < self.min_user_headroom_gib {
            self.throttle_level = ThrottleLevel::AutoWithdraw;
            let reason = WithdrawalReason::UserMemoryPressure {
                free_ram_gib: self.free_ram_gib,
                min_user_headroom_gib: self.min_user_headroom_gib,
            };
            self.withdrawal_reason = Some(reason.clone());
            return Some(reason);
        }

        None
    }

    /// Advances the RC thermal network and battery SoC over `dt_seconds` at `load_fraction`,
    /// returning the updated [`TelemetrySnapshot`].
    ///
    /// Note: `joules_delta` integrated into `cumulative_joules` uses `dt_seconds` (wall or
    /// modeled interval), while thermal integration uses `dt_seconds * thermal_time_scale`
    /// so short live loopback CPU bursts can exercise full multi-minute thermal ramps
    /// deterministically.
    pub fn step(&mut self, dt_seconds: f64, load_fraction: f64) -> TelemetrySnapshot {
        let dt = dt_seconds.max(0.0);
        let watts = self.current_power_watts(load_fraction);

        // Energy accounting (exact: Joules = Watts * seconds).
        let joules_step = watts * dt;
        self.cumulative_joules += joules_step;

        // Battery drain when running off AC wall power.
        if !self.on_ac_power && self.profile.battery_capacity_wh > 0.0 {
            let thermal_dt = dt * self.thermal_time_scale;
            let wh_used = (watts * thermal_dt) / 3600.0;
            let pct_drop = (wh_used / self.profile.battery_capacity_wh) * 100.0;
            self.battery_pct = (self.battery_pct - pct_drop).max(0.0);
        }

        // Sub-stepped RC thermal integration: dT/dt = (P - (T - T_amb)/R_th) / C_th.
        let thermal_dt = dt * self.thermal_time_scale;
        let substeps = ((thermal_dt / 0.25).ceil() as usize).clamp(1, 200);
        let sub_dt = thermal_dt / (substeps as f64);
        for _ in 0..substeps {
            let cooling_watts = (self.temp_c - self.profile.ambient_temp_c)
                / self.profile.thermal_resistance_c_per_w;
            let d_temp =
                ((watts - cooling_watts) / self.profile.thermal_capacitance_j_per_c) * sub_dt;
            self.temp_c += d_temp;
        }
        if self.temp_c > self.peak_temp_c {
            self.peak_temp_c = self.temp_c;
        }

        // Progressive DVFS state transition with 3°C cooling hysteresis.
        if self.dvfs_enabled && self.withdrawal_reason.is_none() {
            if self.temp_c >= self.profile.withdraw_temp_c {
                self.throttle_level = ThrottleLevel::AutoWithdraw;
            } else if self.temp_c >= self.profile.hard_throttle_temp_c {
                self.throttle_level = ThrottleLevel::ThermalThrottle45;
            } else if self.temp_c >= self.profile.soft_throttle_temp_c {
                if self.throttle_level < ThrottleLevel::ThermalThrottle70 {
                    self.throttle_level = ThrottleLevel::ThermalThrottle70;
                }
            } else if self.temp_c <= self.profile.soft_throttle_temp_c - 3.0 {
                self.throttle_level = ThrottleLevel::Full;
            }
        }

        let withdrawal_reason = self.check_guardrails();
        TelemetrySnapshot {
            temp_c: self.temp_c,
            active_watts: watts,
            cumulative_joules: self.cumulative_joules,
            battery_pct: self.battery_pct,
            on_ac_power: self.on_ac_power,
            free_ram_gib: self.free_ram_gib,
            throttle_level: self.throttle_level,
            withdrawal_reason,
        }
    }

    /// Simulates an AC unplug event (switches to battery cap immediately).
    pub fn unplug_ac(&mut self) {
        self.on_ac_power = false;
        self.watts_cap = self.watts_cap.min(self.profile.battery_cap_watts);
    }

    /// Updates observed free unified memory in GiB (e.g., user opens a heavy IDE/browser).
    pub fn set_free_ram_gib(&mut self, free_ram_gib: f64) {
        self.free_ram_gib = free_ram_gib.max(0.0);
    }

    /// Advances the logical Unix clock.
    pub fn advance_clock(&mut self, delta_seconds: u64) {
        self.now_unix = self.now_unix.saturating_add(delta_seconds);
    }

    /// Sets whether user foreground activity is active.
    pub fn set_user_foreground_active(&mut self, active: bool) {
        self.user_foreground_active = active;
    }
}

/// Summary of Negative Regime 1: unthrottled sustained decode vs DVFS-governed decode.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalRunawayComparison {
    /// Device profile evaluated.
    pub device_label: &'static str,
    /// Duration simulated in seconds.
    pub duration_seconds: f64,
    /// Peak temperature without DVFS throttling (°C).
    pub unthrottled_peak_temp_c: f64,
    /// Hard safety ceiling (°C).
    pub safety_ceiling_c: f64,
    /// Peak temperature with progressive DVFS throttling enabled (°C).
    pub governed_peak_temp_c: f64,
    /// Whether unthrottled execution breached the thermal safety ceiling.
    pub unthrottled_breached_ceiling: bool,
    /// Whether the governed run stayed safe or cleanly withdrew at the ceiling.
    pub governed_stayed_safe: bool,
}

/// Evaluates Negative Regime 1 (`neg_unthrottled_thermal_runaway`) under sustained burst load.
///
/// Models a thin laptop/mobile envelope with elevated burst power (`1.18x` sustained cap)
/// running continuous decode without DVFS throttling vs with progressive DVFS + auto-withdraw.
#[must_use]
pub fn simulate_unthrottled_thermal_runaway(
    profile: &EdgeDeviceProfile,
    duration_seconds: f64,
) -> ThermalRunawayComparison {
    // Unthrottled device runs at 1.18x nominal sustained cap without DVFS or auto-withdraw.
    let mut unthrottled_profile = profile.clone();
    unthrottled_profile.sustained_ac_watts *= 1.18;
    let mut unthrottled = ThermalPowerGovernor::new(
        unthrottled_profile,
        true,
        100.0,
        1_700_000_000,
        1_700_010_000,
    );
    unthrottled.dvfs_enabled = false;
    // Bypass withdrawal to measure true physical runaway temperature if guardrails are disabled.
    unthrottled.profile.withdraw_temp_c = 150.0;

    let mut governed_profile = profile.clone();
    governed_profile.sustained_ac_watts *= 1.18;
    let mut governed =
        ThermalPowerGovernor::new(governed_profile, true, 100.0, 1_700_000_000, 1_700_010_000);

    let steps = (duration_seconds.ceil() as usize).max(1);
    for _ in 0..steps {
        unthrottled.step(1.0, 1.0);
        if !governed.is_withdrawn() {
            governed.step(1.0, 1.0);
        }
    }

    ThermalRunawayComparison {
        device_label: profile.label,
        duration_seconds,
        unthrottled_peak_temp_c: unthrottled.peak_temp_c,
        safety_ceiling_c: profile.withdraw_temp_c,
        governed_peak_temp_c: governed.peak_temp_c,
        unthrottled_breached_ceiling: unthrottled.peak_temp_c > profile.withdraw_temp_c,
        governed_stayed_safe: governed.peak_temp_c <= profile.withdraw_temp_c + 0.5,
    }
}

/// Summary of Negative Regime 3: battery drain without automatic withdrawal vs clean exit + handover.
#[derive(Debug, Clone, PartialEq)]
pub struct BatteryDrainComparison {
    /// Device profile evaluated.
    pub device_label: &'static str,
    /// Starting battery state-of-charge (%).
    pub start_battery_pct: f64,
    /// Owner safety floor (%).
    pub owner_floor_pct: f64,
    /// Ending battery state-of-charge (%) without withdrawal guardrail.
    pub unguarded_end_battery_pct: f64,
    /// Ending battery state-of-charge (%) with automatic battery-floor withdrawal.
    pub guarded_end_battery_pct: f64,
    /// Seconds served before clean withdrawal triggered.
    pub guarded_withdrawal_time_s: f64,
    /// Whether the unguarded run breached the owner's battery floor.
    pub unguarded_breached_floor: bool,
    /// Whether the guarded run preserved the owner's battery floor.
    pub guarded_preserved_floor: bool,
}

/// Evaluates Negative Regime 3 (`neg_battery_drain_without_withdrawal`) on battery power.
#[must_use]
pub fn simulate_battery_drain_regime(
    profile: &EdgeDeviceProfile,
    start_battery_pct: f64,
    duration_seconds: f64,
) -> BatteryDrainComparison {
    let mut unguarded = ThermalPowerGovernor::new(
        profile.clone(),
        false,
        start_battery_pct,
        1_700_000_000,
        1_700_100_000,
    );
    unguarded.min_battery_pct = 0.0; // Guardrail disabled.
    unguarded.watts_cap = profile.sustained_ac_watts; // Uncapped AC draw on battery.
    unguarded.profile.battery_cap_watts = profile.sustained_ac_watts;

    let mut guarded = ThermalPowerGovernor::new(
        profile.clone(),
        false,
        start_battery_pct,
        1_700_000_000,
        1_700_100_000,
    );

    let steps = (duration_seconds.ceil() as usize).max(1);
    let mut withdraw_time = duration_seconds;
    for sec in 0..steps {
        unguarded.step(1.0, 1.0);
        if !guarded.is_withdrawn() {
            guarded.step(1.0, 1.0);
            if guarded.is_withdrawn() {
                withdraw_time = (sec + 1) as f64;
            }
        }
    }

    BatteryDrainComparison {
        device_label: profile.label,
        start_battery_pct,
        owner_floor_pct: profile.default_min_battery_pct,
        unguarded_end_battery_pct: unguarded.battery_pct,
        guarded_end_battery_pct: guarded.battery_pct,
        guarded_withdrawal_time_s: withdraw_time,
        unguarded_breached_floor: unguarded.battery_pct < profile.default_min_battery_pct,
        guarded_preserved_floor: guarded.battery_pct >= profile.default_min_battery_pct - 0.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::EdgeDeviceClass;

    #[test]
    fn thermal_governor_progressively_throttles_and_withdraws() {
        let mut prof = EdgeDeviceClass::AppleSiliconM4Pro24G.profile();
        // Elevated thermal resistance simulates blocked fan / lap placement to reach 95C
        // even after ThermalThrottle45 cuts dynamic power to 40% (14.7W * 5.5 + 24C = 104.85C).
        prof.thermal_resistance_c_per_w = 5.5;
        let mut gov = ThermalPowerGovernor::new(prof, true, 100.0, 1_700_000_000, 1_700_010_000);
        let mut saw_70 = false;
        let mut saw_45 = false;
        for _ in 0..600 {
            let snap = gov.step(1.0, 1.0);
            if snap.throttle_level == ThrottleLevel::ThermalThrottle70 {
                saw_70 = true;
            }
            if snap.throttle_level == ThrottleLevel::ThermalThrottle45 {
                saw_45 = true;
            }
            if snap.withdrawal_reason.is_some() {
                break;
            }
        }
        assert!(saw_70, "expected soft DVFS throttle");
        assert!(saw_45, "expected hard DVFS throttle");
        assert!(gov.is_withdrawn(), "expected automatic thermal withdrawal");
    }
}
