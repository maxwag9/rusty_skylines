use crate::cars::car_subsystem::CarSubsystem;
use crate::events::*;
use crate::positions::{ChunkCoord, WorldPos};
use crate::terrain::roads::road_subsystem::RoadRenderSubsystem;

#[derive(Debug)]
pub enum CarChangeEvent {
    UpdatePosition { pos: WorldPos },
    UpdateLaneS { lane_s: f32 }, // and more, but separate, not in one event, because I don't want to carry non-changing info with me if I only want to change one little thing
    MoveChunk { from: ChunkCoord, to: ChunkCoord },
    Despawn,
}

pub fn run_car_events(
    event: Event,
    car_subsystem: &mut CarSubsystem,
    road_render_subsystem: &RoadRenderSubsystem,
) {
    match event {
        Event::CarNavigate(car_chunks) => { // owned car chunks so I can consume them in the thread.
            // idk spawn threads, send and receive stuff, how?! And this must make as many threads as my cpu has, kinda (
            //        let threads = num_cpus::get_physical().saturating_sub(1).max(1); i think)
            // send the car_chunks to update, then it will update async and receive will take the output cars and car chunks (car chunks to replace the ones in the car storage,
            // a car chunk can only be sent to one thread at a time. Right? That would make sense.)
            // thread::spawn(
            //     { // only takes car_subsystem: &CarSubsystem, no mut, takes the chunks owned and calculates the car movements,
            //         // using road_render_subsystem: &RoadRenderSubsystem for road info. No mutable access inside, no mutex and arc crap.
            //         // Inside, the cars get new future positions and gives events to outside so I can receive the carchangeevents here to update the cars
            //         // outside the thread here mutably once I received the carchangeevents. I won't make a copy of the car in the thread of course, but only pass down changes to be applied outside here!
            //         let car_chunks = car_chunks;
            //         //blablabla
            //     }
            // );
        }
        _ => {}
    }
}
