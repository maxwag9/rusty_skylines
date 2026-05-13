use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Economy {
    pub money: f64, // Dollars? Euros? Depends on everything in my game, depends on supply chains, supply and demand, the third housing crisis and the obese president ruling the country!
}
impl Default for Economy {
    fn default() -> Self {
        // Get it? Like Defaulting on your debt
        Self {
            money: 0.0 // ;(
        }
    }
}
