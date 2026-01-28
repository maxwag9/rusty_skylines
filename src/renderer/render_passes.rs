use crate::data::Settings;
use wgpu::*;

pub struct RenderPassConfig {
    pub background_color: Color,
}

impl RenderPassConfig {
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            background_color: Color {
                r: settings.background_color[0] as f64,
                g: settings.background_color[1] as f64,
                b: settings.background_color[2] as f64,
                a: settings.background_color[3] as f64,
            },
        }
    }
}

pub fn create_color_attachment<'a>(
    msaa_hdr_view: &'a TextureView,
    resolved_hdr_view: &'a TextureView,
    msaa_samples: u32,
    background_color: Color,
) -> RenderPassColorAttachment<'a> {
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_hdr_view,
            resolve_target: Some(resolved_hdr_view),
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(background_color),
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: resolved_hdr_view,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(background_color),
                store: StoreOp::Store,
            },
        }
    }
}
pub fn create_normal_attachment<'a>(
    msaa_normal_view: &'a TextureView,
    resolved_normal_view: &'a TextureView,
    msaa_samples: u32,
) -> RenderPassColorAttachment<'a> {
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_normal_view,
            resolve_target: Some(resolved_normal_view),
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: resolved_normal_view,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        }
    }
}
pub fn create_depth_attachment(depth_view: &TextureView) -> RenderPassDepthStencilAttachment<'_> {
    RenderPassDepthStencilAttachment {
        view: depth_view,
        depth_ops: Some(Operations {
            load: LoadOp::Clear(1.0),
            store: StoreOp::Store,
        }),
        stencil_ops: Some(Operations {
            load: LoadOp::Clear(0),
            store: StoreOp::Store,
        }),
    }
}
