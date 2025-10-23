# rusty_skylines
Cities Skylines but better.

## UI module overview

The `src/ui/` tree wraps [`egui`](https://github.com/emilk/egui) for declarative HUDs on top of the wgpu render target:

* `mod.rs` wires the public API and keeps the submodules discoverable.
* `system.rs` hosts `UiSystem`, which manages the `egui` context, collects window input via `egui_winit`, and renders draw commands with `egui_wgpu`. It exposes a simple frame lifecycle of `begin_frame`, `draw_controls`, and `render` to keep integration with the renderer explicit.
* `layout.rs`, `widgets.rs`, and `theme.rs` provide small helpers for anchored overlays, shared styling, and reusable button primitives. As the project grows, drop-in menus or icons can live alongside these helpers without touching the core system.

`UiSystem::draw_controls` paints the floating transport widget used today. It mutates a shared [`SimulationControls`](src/simulation_controls.rs) struct so gameplay/simulation code can react to pauses or speed multipliers without depending on any `egui` types. Future UI features—toolbar icons, inspectors, minimaps—can build on the same pattern by adding new widgets to `widgets.rs` and invoking them from additional `UiSystem` methods.
