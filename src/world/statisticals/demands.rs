pub struct ZoningDemand {
    pub residential: f32,
    pub commercial: f32,
    pub industrial: f32,
    pub office: f32,
}

pub struct Demands {
    pub zoning_demand: ZoningDemand,
}
