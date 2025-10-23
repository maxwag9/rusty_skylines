use std::fmt;

/// Shared control flags for driving the simulation update cadence.
///
/// The UI mutates these values and the simulation logic can read them
/// each frame to decide whether to advance time and how quickly.
#[derive(Debug, Clone)]
pub struct SimulationControls {
    running: bool,
    speed: SimulationSpeed,
}

impl SimulationControls {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn toggle_running(&mut self) {
        self.running = !self.running;
    }

    pub fn set_running(&mut self, running: bool) {
        self.running = running;
    }

    pub fn speed(&self) -> SimulationSpeed {
        self.speed
    }

    pub fn set_speed(&mut self, speed: SimulationSpeed) {
        self.speed = speed;
    }

    /// Convenience helper for applying the configured multiplier to a delta time.
    pub fn scale_dt(&self, dt: f32) -> f32 {
        if self.running {
            dt * self.speed.multiplier()
        } else {
            0.0
        }
    }

    pub fn speed_multiplier(&self) -> f32 {
        self.speed.multiplier()
    }
}

impl Default for SimulationControls {
    fn default() -> Self {
        Self {
            running: true,
            speed: SimulationSpeed::One,
        }
    }
}

/// Discrete speed presets surfaced by the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationSpeed {
    One,
    Two,
    Five,
}

impl SimulationSpeed {
    pub fn multiplier(self) -> f32 {
        match self {
            SimulationSpeed::One => 1.0,
            SimulationSpeed::Two => 2.0,
            SimulationSpeed::Five => 5.0,
        }
    }
}

impl fmt::Display for SimulationSpeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulationSpeed::One => write!(f, "1x"),
            SimulationSpeed::Two => write!(f, "2x"),
            SimulationSpeed::Five => write!(f, "5x"),
        }
    }
}
