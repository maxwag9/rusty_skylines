use crate::simulation_controls::{SimulationControls, SimulationSpeed};
use egui::{Align2, Context, Id, Margin, Vec2};
use egui_wgpu::ScreenDescriptor;
use egui_winit::EventResponse;
use wgpu::{CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::event::WindowEvent;
use winit::window::Window;

use super::layout;
use super::theme::UiTheme;
use super::widgets;

pub struct UiSystem {
    ctx: Context,
    winit_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    theme: UiTheme,
    window_size: winit::dpi::PhysicalSize<u32>,
}

impl UiSystem {
    pub fn new(
        window: &Window,
        device: &Device,
        surface_format: TextureFormat,
        initial_size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        let ctx = Context::default();
        let viewport_id = egui::ViewportId::ROOT;
        let mut winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport_id,
            window,
            Some(window.scale_factor() as f32),
            None,
        );
        let renderer = egui_wgpu::Renderer::new(device, surface_format, None, 1);

        Self {
            ctx,
            winit_state,
            renderer,
            theme: UiTheme::default(),
            window_size: initial_size,
        }
    }

    pub fn handle_event(&mut self, event: &WindowEvent) -> EventResponse {
        self.winit_state.on_event(&self.ctx, event)
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.winit_state.take_egui_input(window);
        self.ctx.begin_frame(raw_input);
    }

    pub fn draw_controls(&mut self, controls: &mut SimulationControls) {
        layout::anchored_panel(
            Id::new("simulation.controls"),
            Align2::RIGHT_BOTTOM,
            Vec2::new(-24.0, -24.0),
        )
        .show(&self.ctx, |ui| {
            let frame = egui::Frame::none()
                .fill(self.theme.panel_fill)
                .stroke(self.theme.panel_stroke)
                .rounding(self.theme.rounding)
                .inner_margin(Margin::same(12.0));

            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    if widgets::play_pause_button(ui, controls.is_running()).clicked() {
                        controls.toggle_running();
                    }

                    ui.separator();

                    for speed in [
                        SimulationSpeed::One,
                        SimulationSpeed::Two,
                        SimulationSpeed::Five,
                    ] {
                        let active = controls.speed() == speed;
                        if widgets::speed_button(ui, speed, active).clicked() {
                            controls.set_speed(speed);
                            if !controls.is_running() {
                                controls.set_running(true);
                            }
                        }
                    }
                });
            });
        });
    }

    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        window: &Window,
        encoder: &mut CommandEncoder,
        target_view: &TextureView,
    ) {
        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
        } = self.ctx.end_frame();

        self.winit_state
            .handle_platform_output(window, &self.ctx, platform_output);
        let paint_jobs = self.ctx.tessellate(shapes);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.window_size.width, self.window_size.height],
            pixels_per_point: self.winit_state.pixels_per_point(),
        };

        for (id, image_delta) in textures_delta.set {
            self.renderer
                .update_texture(device, queue, id, &image_delta);
        }

        self.renderer
            .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        self.renderer
            .render(encoder, target_view, &paint_jobs, &screen_descriptor);

        for id in textures_delta.free {
            self.renderer.free_texture(&id);
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = new_size;
    }

    pub fn wants_pointer_input(&self) -> bool {
        self.ctx.wants_pointer_input()
    }

    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }
}
