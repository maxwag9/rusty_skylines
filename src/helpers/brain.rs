use std::mem::take;
use std::time::Instant;
use rand::prelude::ThreadRng;
use rand::RngExt;
use crate::helpers::hsv::lerp;
use crate::helpers::MRHANDS;
use crate::resources::Time;
use crate::ui::ui_editor::Ui;

#[derive(Clone, Copy, Default, Debug)]
pub enum MemoryType {
    Spoke,
    HasThought,
    Paused,
    #[default]
    Screamed,
    FirstPlayerEncounter,
    LastPlayerEncounter,
    Silenced,
    Sleep,
}
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub memory: MemoryType,
    pub time: Instant,
}
impl Default for Memory {
    fn default() -> Self {
        Self::new(MemoryType::Paused)
    }
}
impl Memory {
    pub fn new(memory: MemoryType) -> Memory {
        Memory {
            memory,
            time: Instant::now(),
        }
    }
}
pub struct SpeakContext {}
#[derive(Default)]
pub enum BrainAction {
    #[default]
    Pause,
    Think,
    Speak(SpeakContext),
    Sleep,

}
pub struct BrainPack<'a> {
    ui: &'a mut Ui,
    time: &'a Time,
    rng: &'a mut ThreadRng,
}
pub struct Thought {
    time: Instant,
    suicide: f32,
    proud: f32,
    cool: f32,
    last_memory: Memory,
    thinking_thoughts: Option<Vec<Thought>>,
    piss_urge: f32,
    shit_urge: f32,
    damping: f32,
    ignorance: f32,
}
#[derive(Default)]
pub struct BodyState {
    piss: f32,
    shit: f32,
    thc: f32,
    alcohol: f32,
    nicotine: f32,
    fungus: f32,
    pain: f32,
    brain_points: u32,
}
#[derive(Default)]
pub struct BrainStatistics {
    piss_pissed: f64,
    shit_shat: f64,
    weed_cigars_smoked: u32,
    beers_gulped: u32,
    cigarettes_smoked: u32, // Fucking disgusting
    max_fungus_infection_spread: f32,
    hours_slept: f32,
}
pub struct Brain {
    name: String,
    birthday: Instant,
    memories: Vec<Memory>,
    next_action: BrainAction,
    thoughts: Vec<Thought>,
    body_state: BodyState,
    statistics: BrainStatistics,
    pharynx: SpeechModule,
}
impl Brain {
    pub fn new() -> Brain {
        Brain {
            name: MRHANDS.to_string(),
            birthday: Instant::now(),
            memories: vec![Memory::new(MemoryType::Screamed)],
            next_action: BrainAction::Pause,
            thoughts: vec![],
            body_state: BodyState::default(),
            statistics: BrainStatistics::default(),
            pharynx: SpeechModule::new(),
        }
    }
    pub fn update_body(&mut self, pack: &mut BrainPack) {
        let bs = &mut self.body_state;
        bs.piss += pack.time.render_dt * 0.005; // 60FPS * 0.01667 * 0.05 = 0.005 per second
        bs.fungus *= pack.time.render_dt * 1.0005;
        fn conditional_pain(n: f32) -> f32 {
            if n != 0.0 { 1.0 - n.cbrt() } else { 0.0 }
        }
        bs.pain = bs.fungus * 0.2 + conditional_pain(bs.nicotine) * 0.005 + conditional_pain(-bs.alcohol) * 0.05 + conditional_pain(-bs.thc) * 0.05 + bs.piss * 0.1 + bs.shit * 0.2;
    }
    /// Think
    pub fn schizophrenia(&mut self, pack: &mut BrainPack) -> Thought {
        let bs = &mut self.body_state;
        let last_memory = self.memories.last().copied().unwrap_or_default();
        let mem_len = self.memories.len();
        let pause_amount: usize = self.memories.get(mem_len.saturating_sub(11)..mem_len.saturating_sub(1))
            .map(|mems| {
                mems.iter().filter(|m| matches!(m.memory, MemoryType::Paused)).count()
            }).unwrap_or_default();
        let suicide = {
            bs.pain * pause_amount as f32 * 0.1
        };
        Thought {
            time: Instant::now(),
            suicide,
            proud: 0.0,
            cool: 0.0,
            last_memory,
            thinking_thoughts: None,
            piss_urge: 0.0,
            shit_urge: 0.0,
            damping: 0.0,
            ignorance: 0.0,
        }
    }
    /// Pause
    pub fn enlightenment(&mut self, pack: &mut BrainPack) {}
    /// Speak
    pub fn public_schizophenria(&mut self, pack: &mut BrainPack) {}
    /// Scream
    pub fn loud_public_schizophenria(&mut self, pack: &mut BrainPack) {}
    /// Exactly what it says on the packaging.
    pub fn square_rooted_thought(&mut self, pack: &mut BrainPack) {
        let idx = self.memories.len().isqrt();
        let Some(memory) = self.memories.get(idx) else {
            return;
        };
    }

    pub fn weigh_thoughts(&mut self, pack: &mut BrainPack) {
        let mut scrap_action = pack.rng.random_range(-1.0..0.85);
        let thoughts_len = self.thoughts.len();
        let thoughts = self.thoughts.get(thoughts_len.saturating_sub(2)..thoughts_len.saturating_sub(1)).clone().unwrap_or_default();
        for thought in thoughts {
            scrap_action += -thought.proud * 0.5 + thought.suicide * 0.5 - thought.cool * 0.2;
        };


        if scrap_action > 0.8 {
            self.next_action = BrainAction::Pause;
        }
    }
    pub fn execute_actions(&mut self, pack: &mut BrainPack) {
        let action = take(&mut self.next_action);
        match action {
            BrainAction::Pause => {
                self.memories.push(Memory::new(MemoryType::Paused));
                self.try_fungus(pack)
            }
            BrainAction::Think => {
                let thought = self.schizophrenia(pack);
                self.thoughts.push(thought);
                self.weigh_thoughts(pack);
            }
            BrainAction::Speak(ctx) => {}
            _ => {}
        }
    }
    fn try_fungus(&mut self, pack: &mut BrainPack) {
        self.body_state.fungus *= 1.001;
        if self.body_state.fungus == 0.0 && pack.rng.random_bool((0.001 * self.body_state.shit as f64).clamp(0.0, 0.01)) {
            self.body_state.fungus = 0.0001; // Fuck. Infected...
        }
    }
}


struct SpeechModule { // Larynx

}
impl SpeechModule {
    pub fn new() -> SpeechModule {
        SpeechModule {}
    }
    pub fn speak_string(ctx: &SpeakContext, pack: &mut BrainPack) {}
}

pub struct VascularSystem {
    blood_pressure: f32,
    vascular_integrity: f32,
    aneurysm: f32,
}
