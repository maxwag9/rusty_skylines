use crate::resources::HOURS_PER_YEAR;
use crate::world::roads::road_structs::{CrossingPoint, RoadType};
use crate::world::statisticals::demography::{AgeGroups, LifeStage, MAX_AGE};
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(Serialize, Deserialize, Clone)]
pub struct Economy {
    pub money: i64, // Dollars? Euros? Depends on everything in my game, depends on supply chains, supply and demand, the third housing crisis and the obese president ruling the country!
                    // TAXES!! Handled in district updates
}
impl Default for Economy {
    fn default() -> Self {
        // Get it? Like Defaulting on your debt
        Self {
            money: 2_000_000, // ;(
        }
    }
}

impl Economy {
    #[inline]
    pub fn can_buy(&mut self, cost: i64) -> bool {
        if self.money > cost { true } else { false }
    }
    #[inline]
    pub fn buy(&mut self, cost: i64) -> bool {
        if self.money > cost {
            self.money -= cost;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn add_money(&mut self, amount: i64) {
        self.money += amount;
    }
}

pub fn calculate_road_cost(
    road_type: &RoadType,
    crossings: &[CrossingPoint],
    road_length: f64,
) -> u64 {
    let base_per_meter = road_type.cost_per_meter();

    let mut total = base_per_meter * road_length;

    // crossings = infrastructure complexity cost
    total *= 1.0 + crossings.len() as f64 * 0.15;

    total as u64
}

/// Tax configuration: what each class earns and how much of it the state takes.
///
/// Stored on `Demography` so it can be tuned per-scenario (e.g. a flat-tax
/// reform or austerity policy) without touching the distribution logic.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TaxConfig {
    /// Annual pre-tax income in € for a fully-productive worker of each class.
    /// [Poor, Working, Middle, Upper, Wealthy]
    pub annual_gross_income: [u64; NUM_INCOME_CLASSES],

    /// Effective tax rate per class (0.0 – 1.0).
    pub tax_rate: [f64; NUM_INCOME_CLASSES],
}

impl Default for TaxConfig {
    fn default() -> Self {
        Self {
            //                         Poor    Work     Midd      Uppr       Wlth
            annual_gross_income: [2_000, 8_000, 20_000, 60_000, 200_000],
            tax_rate: [0.05, 0.12, 0.22, 0.32, 0.42],
        }
    }
}

impl TaxConfig {
    /// Daily tax revenue from one fully-productive worker of income class `ic`.
    #[inline]
    pub fn secondly_tax_per_worker(&self, ic: usize, day_length: f64) -> f64 {
        (self.annual_gross_income[ic] as f64 * self.tax_rate[ic] / (HOURS_PER_YEAR * day_length))
            * 20.0
    }
}

pub const NUM_INCOME_CLASSES: usize = 5;

/// Annual Markov transition matrix for income mobility.
/// `matrix[from][to]` = annual probability of moving from class `from` to `to`.
/// Every row must sum to exactly 1.0.
pub type IncomeTransitionMatrix = [[f64; NUM_INCOME_CLASSES]; NUM_INCOME_CLASSES];

/// Broad economic stratum a citizen or age cohort belongs to.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, EnumIter)]
pub enum IncomeClass {
    Poor = 0,
    Working = 1,
    Middle = 2,
    Upper = 3,
    Wealthy = 4,
}

impl IncomeClass {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::Poor),
            1 => Some(Self::Working),
            2 => Some(Self::Middle),
            3 => Some(Self::Upper),
            4 => Some(Self::Wealthy),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Poor => "Poor",
            Self::Working => "Working",
            Self::Middle => "Middle",
            Self::Upper => "Upper",
            Self::Wealthy => "Wealthy",
        }
    }

    /// Initial population split: [Poor, Working, Middle, Upper, Wealthy]
    pub fn base_distribution() -> [f64; NUM_INCOME_CLASSES] {
        [0.20, 0.35, 0.30, 0.10, 0.05]
    }
}

/// Tracks the statistical distribution of income classes across age cohorts.
///
/// ## Design: fractions, not counts
///
/// `AgeGroups` already owns the canonical head-counts per age. Duplicating them
/// here would create a synchronisation burden on every birth, death, and
/// graduation. Instead, this struct stores only *fractions*:
///
///   `shares[ic][age]` = fraction of the age-`age` cohort in income class `ic`
///
/// Each column (one per age) sums to 1.0. Actual head-counts are derived on
/// demand:
///
///   count(ic, age) ≈ shares[ic][age] × age_groups.get(age)
///
/// ### Why deaths and graduations are "free"
///
/// - **Deaths** are Poisson-sampled from the *whole* cohort, proportionally to
///   every income class inside it — so the within-cohort ratios are preserved.
///   No hook needed.
/// - **Graduations** similarly pull a proportional slice from the cohort, so the
///   graduating sub-group has the same income distribution as the parent.
///   No hook needed.
///
/// ### The two hooks that *do* matter
///
/// - **`on_births`** — newborns are assigned their parents' income distribution
///   (weighted across the working-age population) and blended into age-0.
/// - **`apply_annual_transitions`** — a per-LifeStage Markov matrix drives
///   income mobility once per simulated year.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IncomeDistribution {
    /// `shares[ic][age]` ∈ [0, 1]. Each column sums to 1.0.
    shares: [[f64; MAX_AGE]; NUM_INCOME_CLASSES],

    /// Annual Markov transition matrices, keyed by LifeStage.
    pub transitions: HashMap<LifeStage, IncomeTransitionMatrix>,
}

impl Default for IncomeDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl IncomeDistribution {
    pub fn new() -> Self {
        let base = IncomeClass::base_distribution();
        let mut shares = [[0.0f64; MAX_AGE]; NUM_INCOME_CLASSES];
        for age in 0..MAX_AGE {
            for ic in 0..NUM_INCOME_CLASSES {
                shares[ic][age] = base[ic];
            }
        }
        Self {
            shares,
            transitions: default_transitions(),
        }
    }

    // ── Hot-path hooks ───────────────────────────────────────────────────────

    /// Call **after** births have been applied to `age_groups[0]`.
    ///
    /// Newborns inherit the current working-age income mix, then are blended
    /// into the existing age-0 cohort by a population-weighted average.
    pub fn on_births(&mut self, births: u32, age_groups: &AgeGroups) {
        if births == 0 {
            return;
        }
        let new_total = age_groups.get(0usize) as f64;
        if new_total == 0.0 {
            return;
        }
        let old_total = (new_total - births as f64).max(0.0);
        let parent_dist = self.working_age_distribution(age_groups);

        for ic in 0..NUM_INCOME_CLASSES {
            self.shares[ic][0] =
                (self.shares[ic][0] * old_total + parent_dist[ic] * births as f64) / new_total;
        }
        self.normalize_col(0);
    }

    // ── Annual update ────────────────────────────────────────────────────────

    /// Apply one year of income-class Markov transitions.
    ///
    /// Call once per simulated year (e.g. `day_counter % DAYS_PER_YEAR as u32 == 0`).
    /// Each age cohort's share column is multiplied by the transition matrix for
    /// its LifeStage:
    ///
    ///   new_share[to] = Σ_from  old_share[from] × M[from][to]
    pub fn apply_annual_transitions(&mut self) {
        for age in 0..MAX_AGE {
            let stage = LifeStage::from_usize(age);
            let m = self.transitions[&stage];

            let old: [f64; NUM_INCOME_CLASSES] = std::array::from_fn(|ic| self.shares[ic][age]);
            let mut new_col = [0.0f64; NUM_INCOME_CLASSES];

            for from in 0..NUM_INCOME_CLASSES {
                for to in 0..NUM_INCOME_CLASSES {
                    new_col[to] += old[from] * m[from][to];
                }
            }

            for ic in 0..NUM_INCOME_CLASSES {
                self.shares[ic][age] = new_col[ic];
            }
            self.normalize_col(age);
        }
    }

    // ── Query API ────────────────────────────────────────────────────────────

    /// Sample a random income class for a citizen of the given age, weighted
    /// by that cohort's current share distribution.
    ///
    /// Mirrors `Demography::get_random_age` — O(NUM_INCOME_CLASSES).
    pub fn get_random_income_class(&self, rng: &mut impl Rng, age: usize) -> IncomeClass {
        let mut roll: f64 = rng.random();
        for ic in 0..NUM_INCOME_CLASSES - 1 {
            roll -= self.shares[ic][age];
            if roll <= 0.0 {
                return IncomeClass::from_index(ic).unwrap();
            }
        }
        // Absorbing last class catches any floating-point underrun
        IncomeClass::from_index(NUM_INCOME_CLASSES - 1).unwrap()
    }

    /// Fraction of the age-`age` cohort in each income class.
    /// Returns `[Poor, Working, Middle, Upper, Wealthy]`, all in [0, 1], summing to 1.
    #[inline]
    pub fn fractions_for_age(&self, age: usize) -> [f64; NUM_INCOME_CLASSES] {
        std::array::from_fn(|ic| self.shares[ic][age])
    }

    /// Population-weighted fractions across every age group combined.
    pub fn overall_fractions(&self, age_groups: &AgeGroups) -> [f64; NUM_INCOME_CLASSES] {
        let total = age_groups.whole_population() as f64;
        if total == 0.0 {
            return IncomeClass::base_distribution();
        }
        let mut out = [0.0f64; NUM_INCOME_CLASSES];
        for age in 0..MAX_AGE {
            let w = age_groups.get(age) as f64 / total;
            for ic in 0..NUM_INCOME_CLASSES {
                out[ic] += self.shares[ic][age] * w;
            }
        }
        out
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    /// Population-weighted income distribution of working-age citizens (ages 9–19).
    fn working_age_distribution(&self, age_groups: &AgeGroups) -> [f64; NUM_INCOME_CLASSES] {
        let mut totals = [0.0f64; NUM_INCOME_CLASSES];
        let mut pop = 0.0f64;

        for age in 9..=19 {
            let n = age_groups.get(age) as f64;
            pop += n;
            for ic in 0..NUM_INCOME_CLASSES {
                totals[ic] += self.shares[ic][age] * n;
            }
        }

        if pop == 0.0 {
            return IncomeClass::base_distribution();
        }
        totals.map(|t| t / pop)
    }

    /// Re-normalise a column so it sums to 1.0.
    /// Falls back to base_distribution if the column is degenerate.
    fn normalize_col(&mut self, age: usize) {
        let sum: f64 = (0..NUM_INCOME_CLASSES).map(|ic| self.shares[ic][age]).sum();
        if sum > 1e-12 {
            (0..NUM_INCOME_CLASSES).for_each(|ic| self.shares[ic][age] /= sum);
        } else {
            let base = IncomeClass::base_distribution();
            (0..NUM_INCOME_CLASSES).for_each(|ic| self.shares[ic][age] = base[ic]);
        }
    }
}

// ── Default transition matrices ───────────────────────────────────────────────

fn default_transitions() -> HashMap<LifeStage, IncomeTransitionMatrix> {
    LifeStage::iter()
        .map(|stage| {
            let m = stage_transition_matrix(&stage);
            (stage, m)
        })
        .collect()
}

/// Annual transition matrix for each LifeStage.
/// Rows = from-class, columns = to-class. Every row sums to 1.0.
fn stage_transition_matrix(stage: &LifeStage) -> IncomeTransitionMatrix {
    //                          Poor    Work    Midd    Uppr    Wlth
    match stage {
        // Dependants: income is almost entirely inherited; near-zero own mobility.
        LifeStage::Infant | LifeStage::Child => [
            [0.98, 0.02, 0.00, 0.00, 0.00], // Poor
            [0.01, 0.98, 0.01, 0.00, 0.00], // Working
            [0.00, 0.01, 0.98, 0.01, 0.00], // Middle
            [0.00, 0.00, 0.01, 0.98, 0.01], // Upper
            [0.00, 0.00, 0.00, 0.01, 0.99], // Wealthy
        ],

        // Entering the workforce — highest upward mobility.
        LifeStage::YoungAdult => [
            [0.80, 0.18, 0.02, 0.00, 0.00], // Poor
            [0.04, 0.78, 0.17, 0.01, 0.00], // Working
            [0.01, 0.07, 0.80, 0.11, 0.01], // Middle
            [0.00, 0.01, 0.06, 0.83, 0.10], // Upper
            [0.00, 0.00, 0.01, 0.05, 0.94], // Wealthy
        ],

        // Peak working life — moderate, mildly asymmetric mobility.
        LifeStage::Adult => [
            [0.87, 0.11, 0.02, 0.00, 0.00], // Poor
            [0.03, 0.87, 0.09, 0.01, 0.00], // Working
            [0.01, 0.04, 0.88, 0.06, 0.01], // Middle
            [0.00, 0.01, 0.03, 0.90, 0.06], // Upper
            [0.00, 0.00, 0.01, 0.02, 0.97], // Wealthy
        ],

        // Retirement — mostly fixed, slight downward drift from reduced income.
        LifeStage::Elder => [
            [0.96, 0.04, 0.00, 0.00, 0.00], // Poor
            [0.02, 0.95, 0.03, 0.00, 0.00], // Working
            [0.01, 0.02, 0.94, 0.03, 0.00], // Middle
            [0.00, 0.01, 0.02, 0.94, 0.03], // Upper
            [0.00, 0.00, 0.01, 0.02, 0.97], // Wealthy
        ],
    }
}
