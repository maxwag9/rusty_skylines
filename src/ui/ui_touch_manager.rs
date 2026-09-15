// ui_touch_manager.rs
//! Touch interaction system with event-driven architecture
//!
//! Responsibilities:
//! - Convert raw input into UI touch events
//! - Track hover/press/drag states per element
//! - Emit high-level interaction events
//! - Coordinate selection state
//!
//! Does NOT handle:
//! - Rendering, saving/loading, undo storage, element creation/deletion

use crate::data::Settings;
use crate::renderer::ui_text_rendering::anchor_to;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::ui::selections::SelectionManager;
use crate::ui::ui_editor::{GuiOptions, TouchableElement, Ui, get_element};
use crate::ui::ui_edits::SizeProperty;
use crate::ui::ui_runtime::UiRuntimes;
use crate::ui::vertex::{
    ElementKind, UiButtonCircle, UiButtonHandle, UiButtonPolygon, UiButtonRect, UiButtonText,
    UiElement,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::Duration;
use tracing::error;

/// Configuration for touch behavior - data-driven, easy to tweak
#[derive(Clone, Debug)]
pub struct TouchConfig {
    /// Pixels mouse must move before press becomes drag
    pub drag_threshold: f32,
    /// Time window for double-click detection
    pub double_click_time: Duration,
    /// Press duration before "held" state fires
    pub hold_time: Duration,
    /// Snap grid size in pixels
    pub snap_grid_size: f32,
    /// Whether snapping is enabled
    pub snap_enabled: bool,
    /// Modifier for multi-select (true = Ctrl held)
    pub multi_select_active: bool,
    /// Modifier for additive select (true = Shift held)
    pub additive_select_active: bool,
    pub zoom_states: HashMap<ElementRef, ZoomState>,
}

impl Default for TouchConfig {
    fn default() -> Self {
        Self {
            drag_threshold: 4.0,
            double_click_time: Duration::from_millis(300),
            hold_time: Duration::from_millis(150),
            snap_grid_size: 10.0,
            snap_enabled: false,
            multi_select_active: false,
            additive_select_active: false,
            zoom_states: HashMap::new(),
        }
    }
}

/// Reference to an element (menu/layer/id)
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ElementRef {
    pub menu: String,
    pub layer: String,
    pub id: String,
    pub kind: ElementKind,
}
impl fmt::Display for ElementRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{} ({})",
            self.menu, self.layer, self.id, self.kind
        )
    }
}
impl From<&ElementRef> for ElementRef {
    fn from(value: &ElementRef) -> Self {
        value.clone()
    }
}
impl ElementRef {
    pub fn action(&self, ui: &Ui) -> Vec<String> {
        match get_element(&ui.menus, self) {
            Some(e) => e.string_actions(),
            None => vec![],
        }
    }
}

impl Default for ElementRef {
    fn default() -> ElementRef {
        ElementRef {
            menu: "m".into(),
            layer: "l".into(),
            id: "e".into(),
            kind: ElementKind::None,
        }
    }
}

impl ElementRef {
    pub fn new(menu: &str, layer: &str, id: &str, kind: ElementKind) -> Self {
        Self {
            menu: menu.to_string(),
            layer: layer.to_string(),
            id: id.to_string(),
            kind,
        }
    }
}

/// Result of a hit test
#[derive(Clone, Debug)]
pub struct HitTestResult {
    pub element_ref: ElementRef,
    pub affected_element: Option<ElementRef>,
    pub z_order: u32,
    pub element_order: usize,
    /// Distance from element center (useful for tie-breaking)
    pub distance: f32,
    /// For polygon: which vertex was hit, if any
    pub vertex_index: Option<usize>,
    pub text_being_edited: Option<bool>,
}

impl HitTestResult {
    /// Ordering key for determining top hit (higher = on top)
    pub fn priority(&self) -> (u32, usize) {
        (self.z_order, self.element_order)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ButtonState {
    pub pressed: bool,
    pub just_pressed: bool,
    pub just_released: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MouseButtons {
    pub left: ButtonState,
    pub right: ButtonState,
    pub middle: ButtonState,
    pub back: ButtonState,
    pub forward: ButtonState,
}
impl MouseButtons {
    pub fn pressed(&self) -> bool {
        self.left.pressed
            || self.right.pressed
            || self.middle.pressed
            || self.back.pressed
            || self.forward.pressed
    }

    pub fn just_pressed(&self) -> bool {
        self.left.just_pressed
            || self.right.just_pressed
            || self.middle.just_pressed
            || self.back.just_pressed
            || self.forward.just_pressed
    }

    pub fn just_released(&self) -> bool {
        self.left.just_released
            || self.right.just_released
            || self.middle.just_released
            || self.back.just_released
            || self.forward.just_released
    }
}

#[derive(Clone, Debug)]
pub enum UiEvent {
    ElementEvent(ElementEvent),
    GlobalEvent(GlobalEvent),
}
#[derive(Clone, Debug)]
pub enum ElementEvent {
    // Hover events
    HoverEnter,
    Hovering,
    HoverExit,
    Nothing,
    // Press/release events
    Press {
        position: [f32; 2],
        vertex_index: Option<usize>,
    },
    Down {
        position: [f32; 2],
        vertex_index: Option<usize>,
    },
    Release {
        position: [f32; 2],
        was_drag: bool,
    },
    Click {
        position: [f32; 2],
    },
    DoubleClick {
        position: [f32; 2],
    },

    // Drag events
    DragStart {
        start_position: [f32; 2],
        vertex_index: Option<usize>,
    },
    DragMove {
        current_position: [f32; 2],
        delta: [f32; 2],
        total_delta: [f32; 2],
    },
    DragEnd {
        start_position: [f32; 2],
        end_position: [f32; 2],
        vertex_index: Option<usize>,
    },

    ScrollOnElement {
        delta: f32,
    },

    // Selection events
    SelectionRequested {
        additive: bool,
        multi: bool,
    },

    TextEditRequested,
    TextEditEnded,

    Activated,
    Deactivated,
}

#[derive(Clone, Debug)]
pub enum GlobalEvent {
    DeselectAllRequested,
    BoxSelectStart { start: [f32; 2] },
    BoxSelectMove { current: [f32; 2] },
    BoxSelectEnd { start: [f32; 2], end: [f32; 2] },

    // Navigation events
    NavigateDirection { direction: NavigationDirection },
    StartUp,
    ScreenResize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavigationDirection {
    Up,
    Down,
    Left,
    Right,
}

// ============================================================================
// TRAITS
// ============================================================================

/// Trait for elements that can be hit-tested
pub trait Touchable {
    fn kind(&self) -> ElementKind;
    fn hit_test(&self, point: [f32; 2]) -> Option<TouchableHit>;
    fn center(&self) -> [f32; 2];
    fn z_order(&self) -> u32;
    fn is_active(&self) -> bool;
    fn is_pressable(&self) -> bool;
    fn is_editable(&self, override_mode: bool) -> bool;
    fn sizes(&self) -> Vec<SizeProperty>;
    fn main_size(&self) -> SizeProperty;
}

/// Result of hitting a touchable element
#[derive(Clone, Debug)]
pub struct TouchableHit {
    pub distance: f32,
    pub vertex_index: Option<usize>,
}

/// Trait for elements that can be dragged
pub trait Draggable: Touchable {
    fn drag_anchor(&self, vertex_index: Option<usize>) -> [f32; 2];
    fn can_snap(&self) -> bool;
}

// ============================================================================
// TRAIT IMPLEMENTATIONS
// ============================================================================

impl Touchable for UiButtonCircle {
    fn kind(&self) -> ElementKind {
        ElementKind::Circle
    }

    fn hit_test(&self, point: [f32; 2]) -> Option<TouchableHit> {
        let dx = point[0] - self.x;
        let dy = point[1] - self.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist <= self.radius {
            Some(TouchableHit {
                distance: dist,
                vertex_index: None,
            })
        } else {
            None
        }
    }

    fn center(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    fn z_order(&self) -> u32 {
        0 // Circles don't have individual z-order; layer handles this
    }

    fn is_active(&self) -> bool {
        self.misc.active
    }

    fn is_pressable(&self) -> bool {
        self.misc.touchable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn sizes(&self) -> Vec<SizeProperty> {
        vec![
            SizeProperty::Radius(self.radius),
            SizeProperty::Border(self.border_thickness),
            SizeProperty::InsideBorder(self.inside_border_thickness),
        ]
    }

    fn main_size(&self) -> SizeProperty {
        SizeProperty::Radius(self.radius)
    }
}

impl Draggable for UiButtonCircle {
    fn drag_anchor(&self, _vertex_index: Option<usize>) -> [f32; 2] {
        [self.x, self.y]
    }

    fn can_snap(&self) -> bool {
        true
    }
}

impl Touchable for UiButtonPolygon {
    fn kind(&self) -> ElementKind {
        ElementKind::Polygon
    }

    fn hit_test(&self, point: [f32; 2]) -> Option<TouchableHit> {
        if self.scaled_vertices().is_empty() {
            return None;
        }

        const VERTEX_RADIUS: f32 = 10.0;

        // Check vertex hits first
        for (i, v) in self.scaled_vertices().iter().enumerate() {
            let dx = point[0] - v.pos[0];
            let dy = point[1] - v.pos[1];
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < VERTEX_RADIUS {
                return Some(TouchableHit {
                    distance: dist,
                    vertex_index: Some(i),
                });
            }
        }

        // Check polygon interior/edge
        let sdf = polygon_sdf(point[0], point[1], &self.scaled_vertices());
        let inside = sdf < 0.0;
        let near_edge = sdf.abs() < 8.0;

        if inside || near_edge {
            Some(TouchableHit {
                distance: sdf.abs(),
                vertex_index: None,
            })
        } else {
            None
        }
    }

    fn center(&self) -> [f32; 2] {
        self.center()
    }

    fn z_order(&self) -> u32 {
        0
    }

    fn is_active(&self) -> bool {
        self.misc.active
    }

    fn is_pressable(&self) -> bool {
        self.misc.touchable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn sizes(&self) -> Vec<SizeProperty> {
        vec![SizeProperty::PolygonScale(self.scale)]
    }

    fn main_size(&self) -> SizeProperty {
        SizeProperty::PolygonScale(self.scale)
    }
}

impl Draggable for UiButtonPolygon {
    fn drag_anchor(&self, vertex_index: Option<usize>) -> [f32; 2] {
        if let Some(idx) = vertex_index {
            if let Some(v) = self.scaled_vertices().get(idx) {
                return [v.pos[0], v.pos[1]];
            }
        }
        self.center()
    }

    fn can_snap(&self) -> bool {
        true
    }
}

impl Touchable for UiButtonRect {
    fn kind(&self) -> ElementKind {
        ElementKind::Rect
    }

    fn hit_test(&self, point: [f32; 2]) -> Option<TouchableHit> {
        let half_w = self.w * 0.5;
        let half_h = self.h * 0.5;

        // Convert normalized roundness (0.0-1.0) to absolute radius
        // Maximum radius is the smaller half-dimension (makes it a circle/capsule at 1.0)
        let max_round = half_w.min(half_h);
        let roundness = self.roundness * max_round;

        // SDF for rounded rectangle
        let sdf = sd_rounded_box(point, [self.x, self.y], [half_w, half_h], roundness);

        let inside = sdf < 0.0;
        let near_edge = sdf.abs() < 1.0;

        if inside || near_edge {
            Some(TouchableHit {
                distance: sdf.abs(),
                vertex_index: None,
            })
        } else {
            None
        }
    }

    fn center(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    fn z_order(&self) -> u32 {
        0
    }

    fn is_active(&self) -> bool {
        self.misc.active
    }

    fn is_pressable(&self) -> bool {
        self.misc.touchable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn sizes(&self) -> Vec<SizeProperty> {
        vec![
            SizeProperty::Rect(self.size()),
            SizeProperty::Border(self.border_thickness),
        ]
    }
    fn main_size(&self) -> SizeProperty {
        SizeProperty::Rect(self.size())
    }
}

impl Draggable for UiButtonRect {
    fn drag_anchor(&self, _vertex_index: Option<usize>) -> [f32; 2] {
        // Rects only drag from center, no vertex manipulation
        [self.x, self.y]
    }

    fn can_snap(&self) -> bool {
        true
    }
}

impl Touchable for UiButtonText {
    fn kind(&self) -> ElementKind {
        ElementKind::Text
    }

    fn hit_test(&self, point: [f32; 2]) -> Option<TouchableHit> {
        let pos = anchor_to(self.anchor, [self.x, self.y], self.width, self.height);
        let pad = 2f32;
        let x0 = pos[0];
        let y0 = pos[1];
        let x1 = x0 + self.width + pad;
        let y1 = y0 + self.height + pad;

        if point[0] >= x0 && point[0] <= x1 && point[1] >= y0 && point[1] <= y1 {
            let cx = (x0 + x1) / 2.0;
            let cy = (y0 + y1) / 2.0;
            let dist = ((point[0] - cx).powi(2) + (point[1] - cy).powi(2)).sqrt();
            Some(TouchableHit {
                distance: dist,
                vertex_index: None,
            })
        } else {
            None
        }
    }

    fn center(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    fn z_order(&self) -> u32 {
        0
    }

    fn is_active(&self) -> bool {
        self.misc.active
    }

    fn is_pressable(&self) -> bool {
        self.misc.touchable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn sizes(&self) -> Vec<SizeProperty> {
        vec![
            SizeProperty::Pt(self.pt),
            SizeProperty::Border(self.border_width),
            SizeProperty::Rect([self.width, self.height]),
        ]
    }

    fn main_size(&self) -> SizeProperty {
        SizeProperty::Rect([self.width, self.height])
    }
}

impl Draggable for UiButtonText {
    fn drag_anchor(&self, _vertex_index: Option<usize>) -> [f32; 2] {
        self.center()
    }

    fn can_snap(&self) -> bool {
        true
    }
}

impl Touchable for UiButtonHandle {
    fn kind(&self) -> ElementKind {
        ElementKind::Handle
    }

    fn hit_test(&self, point: [f32; 2]) -> Option<TouchableHit> {
        let dx = point[0] - self.x;
        let dy = point[1] - self.y;
        let dist2 = dx * dx + dy * dy;

        let width_ratio = self.handle_misc.handle_width;
        let half_thick = 0.5 * self.radius * width_ratio;
        let inner = self.radius - half_thick;
        let outer = self.radius + half_thick;
        let margin = (self.radius * 0.15).max(10.0);
        let inner_grab = (inner - margin).max(0.0);
        let outer_grab = outer + margin;

        if dist2 >= inner_grab * inner_grab && dist2 <= outer_grab * outer_grab {
            Some(TouchableHit {
                distance: dist2.sqrt(),
                vertex_index: None,
            })
        } else {
            None
        }
    }

    fn center(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    fn z_order(&self) -> u32 {
        0
    }

    fn is_active(&self) -> bool {
        self.misc.active
    }

    fn is_pressable(&self) -> bool {
        self.misc.touchable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn sizes(&self) -> Vec<SizeProperty> {
        vec![SizeProperty::Radius(self.radius)]
    }

    fn main_size(&self) -> SizeProperty {
        SizeProperty::Radius(self.radius)
    }
}

impl Draggable for UiButtonHandle {
    fn drag_anchor(&self, _vertex_index: Option<usize>) -> [f32; 2] {
        [self.x, self.y]
    }

    fn can_snap(&self) -> bool {
        false // Handles typically follow their parent
    }
}

// PER-ELEMENT STATE MACHINE

/// State machine state for a single element
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementTouchState {
    Idle,
    Hovered,
    Pressed { frame_count: u32 },
    Dragging,
}

impl Default for ElementTouchState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Runtime data for a single element's touch state
#[derive(Clone, Debug, Default)]
pub struct ElementTouchData {
    pub state: ElementTouchState,
    pub press_position: Option<[f32; 2]>,
    pub press_time: f32,
    pub last_click_time: f32,
    pub vertex_index: Option<usize>,
}

impl ElementTouchData {
    pub fn reset(&mut self) {
        self.state = ElementTouchState::Idle;
        self.press_position = None;
        self.vertex_index = None;
    }

    pub fn is_down(&self) -> bool {
        matches!(
            self.state,
            ElementTouchState::Pressed { .. } | ElementTouchState::Dragging
        )
    }
}

// ============================================================================
// HIT DETECTOR
// ============================================================================

/// Pure hit detection logic - no state mutation
pub struct HitDetector;

impl HitDetector {
    /// Find the topmost hit element at a point
    pub fn find_top_hit(
        point: [f32; 2],
        elements: &[TouchableElement],
        editor_mode: bool,
        override_mode: bool,
    ) -> Option<HitTestResult> {
        elements
            .iter()
            .filter_map(|element| {
                Self::test_element(
                    point,
                    element.menu,
                    element.layer,
                    element.order,
                    element.idx,
                    element.element,
                    editor_mode,
                    override_mode,
                )
            })
            .max_by_key(|candidate| candidate.priority())
    }

    /// Test a single element for hit
    fn test_element(
        point: [f32; 2],
        menu_name: &str,
        layer_name: &str,
        layer_order: u32,
        element_order: usize,
        element: &UiElement,
        editor_mode: bool,
        override_mode: bool,
    ) -> Option<HitTestResult> {
        let (id, kind, active, pressable, editable) = match element {
            UiElement::Circle(c) => (
                c.id.clone(),
                ElementKind::Circle,
                c.misc.active,
                c.misc.touchable,
                &c.misc.editable,
            ),
            UiElement::Polygon(p) => (
                p.id.clone(),
                ElementKind::Polygon,
                p.misc.active,
                p.misc.touchable,
                &p.misc.editable,
            ),
            UiElement::Text(t) => (
                t.id.clone(),
                ElementKind::Text,
                t.misc.active,
                t.misc.touchable,
                &t.misc.editable,
            ),
            UiElement::Handle(h) => (
                h.id.clone(),
                ElementKind::Handle,
                h.misc.active,
                h.misc.touchable,
                &h.misc.editable,
            ),
            UiElement::Outline(_) => return None, // Outlines aren't interactive
            UiElement::Rect(r) => (
                r.id.clone(),
                ElementKind::Rect,
                r.misc.active,
                r.misc.touchable,
                &r.misc.editable,
            ),
            UiElement::Advanced(_) => return None,
        };

        // Skip inactive or non-interactive elements
        if !active {
            return None;
        }

        // if !override_mode && !editable {
        //     return None;
        // }

        // Skip handles in non-editor mode
        if kind == ElementKind::Handle && !editor_mode {
            return None;
        }

        let mut text_being_edited = None;
        let mut affected_element = None;
        let hit = match element {
            UiElement::Circle(c) => c.hit_test(point),
            UiElement::Polygon(p) => p.hit_test(point),
            UiElement::Text(t) => {
                text_being_edited = Some(t.being_edited);
                t.hit_test(point)
            }
            UiElement::Handle(h) => {
                affected_element = h.parent.clone();
                h.hit_test(point)
            }
            UiElement::Outline(_) => return None,
            UiElement::Rect(r) => r.hit_test(point),
            UiElement::Advanced(_) => return None,
        }?;

        Some(HitTestResult {
            element_ref: ElementRef::new(menu_name, layer_name, id.as_str(), kind),
            affected_element,
            z_order: layer_order,
            element_order,
            distance: hit.distance,
            vertex_index: hit.vertex_index,
            text_being_edited,
        })
    }

    /// Find all elements within a box selection region
    pub fn find_in_box(
        start: [f32; 2],
        end: [f32; 2],
        elements: &Vec<TouchableElement>,
    ) -> Vec<ElementRef> {
        let min_x = start[0].min(end[0]);
        let max_x = start[0].max(end[0]);
        let min_y = start[1].min(end[1]);
        let max_y = start[1].max(end[1]);

        let mut results = Vec::new();

        for touchable_element in elements {
            let (id, kind, center) = match touchable_element.element {
                UiElement::Circle(c) if c.misc.active => {
                    (c.id.clone(), ElementKind::Circle, c.center())
                }
                UiElement::Polygon(p) if p.misc.active => {
                    (p.id.clone(), ElementKind::Polygon, p.center())
                }
                UiElement::Text(t) if t.misc.active => {
                    (t.id.clone(), ElementKind::Text, t.center())
                }
                UiElement::Rect(r) if r.misc.active => {
                    (r.id.clone(), ElementKind::Rect, r.center())
                }
                _ => continue,
            };

            if center[0] >= min_x && center[0] <= max_x && center[1] >= min_y && center[1] <= max_y
            {
                results.push(ElementRef::new(
                    touchable_element.menu,
                    touchable_element.layer,
                    id.as_str(),
                    kind,
                ));
            }
        }

        results
    }
}

// ============================================================================
// DRAG COORDINATOR
// ============================================================================

/// Manages drag operations
#[derive(Clone, Debug, Default)]
pub struct DragCoordinator {
    /// Currently dragging element
    pub active_drag: Option<ActiveDrag>,
}

#[derive(Clone, Debug)]
pub struct ActiveDrag {
    pub element: ElementRef,
    pub affected_element: Option<ElementRef>,
    pub buttons: MouseButtons,
    pub start_position: [f32; 2],
    pub current_position: [f32; 2],
    pub offset: [f32; 2],
    pub vertex_index: Option<usize>,
    pub threshold_exceeded: bool,
}

impl ActiveDrag {
    pub fn total_delta(&self) -> [f32; 2] {
        [
            self.current_position[0] - self.start_position[0],
            self.current_position[1] - self.start_position[1],
        ]
    }

    pub fn delta_from_last(&self, new_pos: [f32; 2]) -> [f32; 2] {
        [
            new_pos[0] - self.current_position[0],
            new_pos[1] - self.current_position[1],
        ]
    }
}

impl DragCoordinator {
    pub fn new() -> Self {
        Self { active_drag: None }
    }

    /// Begin a potential drag operation
    pub fn begin(
        &mut self,
        element: ElementRef,
        affected_element: Option<ElementRef>,
        buttons: MouseButtons,
        mouse_pos: [f32; 2],
        anchor: [f32; 2],
        vertex_index: Option<usize>,
    ) {
        let offset = [mouse_pos[0] - anchor[0], mouse_pos[1] - anchor[1]];

        self.active_drag = Some(ActiveDrag {
            element,
            affected_element,
            buttons,
            start_position: mouse_pos,
            current_position: mouse_pos,
            offset,
            vertex_index,
            threshold_exceeded: false,
        });
    }

    /// Update drag with new position, returns events if any
    pub fn update(&mut self, mouse_pos: [f32; 2], config: &TouchConfig) -> Vec<ElementEvent> {
        let mut events = Vec::new();

        let Some(drag) = &mut self.active_drag else {
            return events;
        };

        let dx = mouse_pos[0] - drag.start_position[0];
        let dy = mouse_pos[1] - drag.start_position[1];
        let distance = (dx * dx + dy * dy).sqrt();

        // Check if we've exceeded drag threshold
        if !drag.threshold_exceeded && distance >= config.drag_threshold {
            drag.threshold_exceeded = true;
            let drag_element = drag.element.clone();
            if drag_element.kind != ElementKind::Handle {
                events.push(ElementEvent::DragStart {
                    start_position: drag.start_position,
                    vertex_index: drag.vertex_index,
                });
            }
        }

        if drag.threshold_exceeded {
            let delta = drag.delta_from_last(mouse_pos);
            let total_delta = [
                mouse_pos[0] - drag.start_position[0],
                mouse_pos[1] - drag.start_position[1],
            ];

            events.push(ElementEvent::DragMove {
                current_position: drag.current_position,
                delta,
                total_delta,
            });
        }

        drag.current_position = mouse_pos;

        events
    }

    /// End drag operation, returns DragEnd event if threshold was exceeded
    pub fn end(&mut self) -> Option<(ElementRef, ElementEvent)> {
        let drag = self.active_drag.take()?;

        if drag.threshold_exceeded {
            Some((
                drag.element,
                ElementEvent::DragEnd {
                    start_position: drag.start_position,
                    end_position: drag.current_position,
                    vertex_index: drag.vertex_index,
                },
            ))
        } else {
            None
        }
    }

    /// Check if currently dragging
    pub fn is_dragging(&self) -> bool {
        self.active_drag
            .as_ref()
            .map(|d| d.threshold_exceeded)
            .unwrap_or(false)
    }

    /// Get the currently dragged element
    pub fn dragging_element(&self) -> Option<&ElementRef> {
        self.active_drag
            .as_ref()
            .filter(|d| d.threshold_exceeded)
            .map(|d| &d.element)
    }

    /// Apply snapping to a position
    pub fn apply_snap(pos: [f32; 2], config: &TouchConfig) -> [f32; 2] {
        if !config.snap_enabled {
            return pos;
        }

        let grid = config.snap_grid_size;
        [
            (pos[0] / grid).round() * grid,
            (pos[1] / grid).round() * grid,
        ]
    }

    /// Cancel current drag without emitting end event
    pub fn cancel(&mut self) {
        self.active_drag = None;
    }
}

// ============================================================================
// EDITOR TOUCH EXTENSION
// ============================================================================

/// Editor-specific touch handling behavior
#[derive(Clone, Debug, Default)]
pub struct EditorTouchExtension {
    /// Whether editor mode is active
    pub enabled: bool,
    /// Currently editing text element
    pub editing_text: Option<ElementRef>,
    /// Whether actively dragging text selection
    pub dragging_text_selection: bool,
    /// Original radius when resize started
    pub original_radius: f32,
    /// Active vertex being dragged (for polygons)
    pub active_vertex: Option<usize>,
}

impl EditorTouchExtension {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Default::default()
        }
    }

    /// Process scroll event for resizing (editor mode only)
    pub fn process_scroll(
        &mut self,
        element: &ElementRef,
        scroll_delta: f32,
    ) -> Option<ElementEvent> {
        if scroll_delta == 0.0 {
            return None;
        }

        Some(ElementEvent::ScrollOnElement {
            delta: scroll_delta,
        })
    }
}

// ============================================================================
// EVENT QUEUE
// ============================================================================

/// Queue of touch events to be processed by subscribers
#[derive(Clone, Debug, Default)]
pub struct TouchEventQueue {
    events: VecDeque<ElementEvent>,
    capacity: usize,
}

impl TouchEventQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, event: ElementEvent) {
        if self.events.len() >= self.capacity {
            self.grow();
        }

        self.events.push_back(event);
    }

    fn grow(&mut self) {
        const MAX_CAPACITY: usize = 4096;

        if self.capacity >= MAX_CAPACITY {
            error!("TouchEvent queue is full!");
            return;
        }

        let new_capacity = (self.capacity * 2).min(MAX_CAPACITY);

        self.events.reserve(new_capacity - self.capacity);
        self.capacity = new_capacity;
    }

    pub fn push_all(&mut self, events: impl IntoIterator<Item = ElementEvent>) {
        for event in events {
            self.push(event);
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = ElementEvent> + '_ {
        self.events.drain(..)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ElementEvent> {
        self.events.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// GLOBAL INTERACTION STATE

/// High-level interaction mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionMode {
    None,
    Hovering,
    Pressing,
    Dragging,
    BoxSelecting,
    TextEditing,
}

impl Default for InteractionMode {
    fn default() -> Self {
        Self::None
    }
}

/// Central coordinator for all UI touch interactions
pub struct UiTouchManager {
    // Configuration
    pub config: TouchConfig,

    // Sub-components
    pub selection: SelectionManager,
    pub drag: DragCoordinator,
    pub editor: EditorTouchExtension,
    pub events: HashMap<ElementRef, Vec<ElementEvent>>,
    pub global_events: Vec<GlobalEvent>,
    pub runtimes: UiRuntimes,
    // State
    element_states: HashMap<String, ElementTouchData>,
    current_hover: Option<ElementRef>,
    interaction_mode: InteractionMode,

    // Timing
    accumulated_time: f32,
    pub options: GuiOptions,
    pub element_actives: HashMap<ElementRef, bool>,
    pub add_screen_resize_event: bool,
    pub top_hit: Option<HitTestResult>,
}

impl UiTouchManager {
    pub fn new(settings: &Settings) -> Self {
        Self {
            config: TouchConfig::default(),
            selection: SelectionManager::new(),
            drag: DragCoordinator::new(),
            editor: EditorTouchExtension::new(settings.editor_mode),
            events: HashMap::new(),
            global_events: vec![],
            runtimes: UiRuntimes::new(),
            element_states: HashMap::new(),
            current_hover: None,
            interaction_mode: InteractionMode::None,
            accumulated_time: 0.0,
            options: GuiOptions {
                override_mode: settings.override_mode,
                show_gui: settings.show_gui,
            },
            element_actives: HashMap::new(),
            add_screen_resize_event: true,
            top_hit: None,
        }
    }
    pub fn push_event<E>(&mut self, element_ref: E, event: ElementEvent)
    where
        E: Into<ElementRef>,
    {
        self.events
            .entry(element_ref.into())
            .or_default()
            .push(event);
    }
    /// Update touch manager with new input
    pub fn update(
        &mut self,
        dt: f32,
        input: &Input,
        elements: &Vec<TouchableElement>,
        time: &Time,
    ) {
        self.accumulated_time += dt;
        self.selection.reset_frame_flags();
        self.events.clear();
        self.global_events.clear();
        // Update config from input modifiers
        self.config.multi_select_active = input.ctrl;
        self.config.additive_select_active = input.shift;

        // Find what we're hitting
        let top_hit = HitDetector::find_top_hit(
            input.mouse.pos.to_array(),
            &elements,
            self.editor.enabled,
            self.options.override_mode,
        );
        self.top_hit = top_hit;
        //println!("{:?}", top_hit);
        // Process hover changes
        self.process_hover();

        // Process press/release/drag
        self.process_press_release(input);

        // Process scroll
        if input.mouse.scroll_delta.y != 0.0 {
            self.process_scroll(input.mouse.scroll_delta.y);
        }

        // Handle box selection if active
        if self.selection.is_box_selecting() && input.mouse.buttons.pressed() {
            // self.events.get_mut().push(TouchEvent::BoxSelectMove {
            //     current: input.position,
            // }); TODO: IDK!!
        }
    }

    /// Process hover state changes
    fn process_hover(&mut self) {
        let new_hover = self.top_hit.as_ref().map(|h| h.element_ref.clone());

        // Check if hover target changed
        if self.current_hover != new_hover {
            // Exit old hover
            if let Some(old) = self.current_hover.clone() {
                self.push_event(old.clone(), ElementEvent::HoverExit);
                if let Some(state) = self.element_states.get_mut(&old.id) {
                    if state.state == ElementTouchState::Hovered {
                        state.state = ElementTouchState::Idle;
                    }
                }
            }

            // Enter new hover
            if let Some(new) = &new_hover {
                self.push_event(new.clone(), ElementEvent::HoverEnter);
                let state = self.element_states.entry(new.id.clone()).or_default();
                if state.state == ElementTouchState::Idle {
                    state.state = ElementTouchState::Hovered;
                }
            }

            self.current_hover = new_hover;

            // Update interaction mode
            if self.interaction_mode == InteractionMode::None
                || self.interaction_mode == InteractionMode::Hovering
            {
                self.interaction_mode = if self.current_hover.is_some() {
                    InteractionMode::Hovering
                } else {
                    InteractionMode::None
                };
            }
        } else {
            if let Some(current_hover) = self.current_hover.clone() {
                self.push_event(current_hover, ElementEvent::Hovering);
            }
        }
    }

    /// Process press and release events
    fn process_press_release(&mut self, input: &Input) {
        // Just pressed
        if input.mouse.buttons.just_pressed() {
            self.handle_press(input);
        }

        // Held (potential drag)
        if input.mouse.buttons.pressed() && !input.mouse.buttons.just_pressed() {
            self.handle_held(input);
        }

        // Just released
        if input.mouse.buttons.just_released() {
            self.handle_release(input);
        }
    }

    /// Handle mouse press
    fn handle_press(&mut self, input: &Input) {
        if let Some(hit) = self.top_hit.clone() {
            let element = &hit.element_ref;
            let mouse_pos = input.mouse.pos.to_array();
            // Update element state
            let state = self.element_states.entry(element.id.clone()).or_default();
            state.state = ElementTouchState::Pressed { frame_count: 0 };
            state.press_position = Some(mouse_pos);
            state.press_time = self.accumulated_time;
            state.vertex_index = hit.vertex_index;

            // Emit press event
            self.push_event(
                element,
                ElementEvent::Press {
                    position: mouse_pos,
                    vertex_index: hit.vertex_index,
                },
            );
            self.push_event(
                element,
                ElementEvent::Down {
                    position: mouse_pos,
                    vertex_index: hit.vertex_index,
                },
            );
            // Begin potential drag
            if !hit.text_being_edited.unwrap_or(false) {
                let anchor = mouse_pos; // Could get from element's drag anchor
                self.drag.begin(
                    element.clone(),
                    hit.affected_element.clone(),
                    input.mouse.buttons,
                    mouse_pos,
                    anchor,
                    hit.vertex_index,
                );
            }

            // Handle selection
            let selection_event = if self.config.multi_select_active {
                ElementEvent::SelectionRequested {
                    additive: false,
                    multi: true,
                }
            } else if self.config.additive_select_active {
                ElementEvent::SelectionRequested {
                    additive: true,
                    multi: false,
                }
            } else {
                ElementEvent::SelectionRequested {
                    additive: false,
                    multi: false,
                }
            };
            self.push_event(hit.element_ref.clone(), selection_event);

            self.interaction_mode = InteractionMode::Pressing;
        } else {
            // Clicked on empty space
            if !self.config.additive_select_active {
                self.global_events.push(GlobalEvent::DeselectAllRequested);
            }

            // Begin box select if in editor mode
            if self.editor.enabled {
                let mouse_pos = input.mouse.pos.to_array();
                self.selection.begin_box_select(mouse_pos);
                self.global_events
                    .push(GlobalEvent::BoxSelectStart { start: mouse_pos });
                self.interaction_mode = InteractionMode::BoxSelecting;
            }
        }
    }

    /// Handle mouse held
    fn handle_held(&mut self, input: &Input) {
        // Update drag
        let drag_events = self.drag.update(input.mouse.pos.to_array(), &self.config);

        if !drag_events.is_empty() {
            self.interaction_mode = InteractionMode::Dragging;
            if let Some(active_drag_element) =
                self.drag.active_drag.as_ref().map(|d| d.element.clone())
            {
                for drag_event in drag_events {
                    self.push_event(&active_drag_element, drag_event);
                }
            }
        }

        // Update pressed element state
        let mut events = Vec::new();

        for state in self.element_states.values_mut() {
            if let ElementTouchState::Pressed { frame_count } = &mut state.state {
                *frame_count += 1;

                if let Some(hit) = self.top_hit.as_ref() {
                    events.push((
                        hit.element_ref.clone(),
                        ElementEvent::Down {
                            position: input.mouse.pos.to_array(),
                            vertex_index: hit.vertex_index,
                        },
                    ));
                }
            }
        }

        for (element, event) in events {
            self.push_event(element, event);
        }
    }

    /// Handle mouse release
    fn handle_release(&mut self, input: &Input) {
        // End drag
        if let Some((drag_element, drag_end)) = self.drag.end() {
            self.push_event(drag_element, drag_end);
        }

        // End box select
        if let Some(start) = self.selection.end_box_select() {
            self.global_events.push(GlobalEvent::BoxSelectEnd {
                start,
                end: input.mouse.pos.to_array(),
            });
        }

        // Process element releases
        let mut releases = Vec::new();
        for (id, state) in &mut self.element_states {
            if state.is_down() {
                let was_drag = self.drag.is_dragging();
                releases.push((id.clone(), state.press_position, was_drag));
                state.reset();
            }
        }

        for (id, _press_pos, was_drag) in releases {
            if let Some(hover) = self.current_hover.clone() {
                if hover.id == id {
                    self.push_event(
                        hover.clone(),
                        ElementEvent::Release {
                            position: input.mouse.pos.to_array(),
                            was_drag,
                        },
                    );

                    // If it wasn't a drag, it's a click
                    if !was_drag {
                        let state = self.element_states.get(&id);
                        let time_since_last_click = state
                            .map(|s| self.accumulated_time - s.last_click_time)
                            .unwrap_or(f32::MAX);

                        if time_since_last_click < self.config.double_click_time.as_secs_f32() {
                            self.push_event(
                                hover,
                                ElementEvent::DoubleClick {
                                    position: input.mouse.pos.to_array(),
                                },
                            );
                        } else {
                            self.push_event(
                                hover,
                                ElementEvent::Click {
                                    position: input.mouse.pos.to_array(),
                                },
                            );
                        }

                        // Update last click time
                        if let Some(state) = self.element_states.get_mut(&id) {
                            state.last_click_time = self.accumulated_time;
                        }
                    }
                }
            }
        }

        self.interaction_mode = if self.current_hover.is_some() {
            InteractionMode::Hovering
        } else {
            InteractionMode::None
        };
    }

    /// Process scroll input
    fn process_scroll(&mut self, delta: f32) {
        if let Some(hover) = &self.current_hover {
            if let Some(event) = self.editor.process_scroll(&hover, delta) {
                self.push_event(hover.clone(), event);
            }
        }
    }

    /// Process keyboard navigation
    pub fn process_navigation(&mut self, direction: NavigationDirection) {
        self.global_events
            .push(GlobalEvent::NavigateDirection { direction });
    }

    // QUERIES

    /// Get current interaction mode
    pub fn interaction_mode(&self) -> InteractionMode {
        self.interaction_mode
    }

    /// Check if any element is being dragged
    pub fn is_dragging(&self) -> bool {
        self.drag.is_dragging()
    }

    /// Get currently hovered element
    pub fn hovered(&self) -> Option<&ElementRef> {
        self.current_hover.as_ref()
    }

    /// Get element touch state
    pub fn element_state(&self, id: &str) -> Option<&ElementTouchData> {
        self.element_states.get(id)
    }

    /// Check if element is being pressed
    pub fn is_pressed(&self, id: &str) -> bool {
        self.element_states
            .get(id)
            .map(|s| s.is_down())
            .unwrap_or(false)
    }

    /// Set editor mode
    pub fn set_editor_mode(&mut self, enabled: bool) {
        self.editor.enabled = enabled;
    }

    /// Toggle snap
    pub fn toggle_snap(&mut self) {
        self.config.snap_enabled = !self.config.snap_enabled;
    }

    /// Set snap grid size
    pub fn set_snap_grid(&mut self, size: f32) {
        self.config.snap_grid_size = size.max(1.0);
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Compute polygon SDF (signed distance function)
fn polygon_sdf(px: f32, py: f32, vertices: &[crate::ui::vertex::UiVertex]) -> f32 {
    if vertices.is_empty() {
        return f32::MAX;
    }

    let n = vertices.len();
    let mut d = f32::MAX;
    let mut s = 1.0_f32;

    let mut j = n - 1;
    for i in 0..n {
        let vi = &vertices[i].pos;
        let vj = &vertices[j].pos;

        let ex = vj[0] - vi[0];
        let ey = vj[1] - vi[1];
        let wx = px - vi[0];
        let wy = py - vi[1];

        let dot_we = wx * ex + wy * ey;
        let dot_ee = ex * ex + ey * ey;
        let t = (dot_we / dot_ee).clamp(0.0, 1.0);

        let bx = wx - ex * t;
        let by = wy - ey * t;
        let dist2 = bx * bx + by * by;

        d = d.min(dist2);

        let c0 = py >= vi[1];
        let c1 = py < vj[1];
        let c2 = ex * wy > ey * wx;

        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            s = -s;
        }

        j = i;
    }

    s * d.sqrt()
}

/// Signed distance function for a rounded rectangle.
/// Returns negative values inside, positive outside.
///
/// - `p`: test point
/// - `center`: rectangle center
/// - `half_size`: half width and half height
/// - `r`: corner radius
pub fn sd_rounded_box(p: [f32; 2], center: [f32; 2], half_size: [f32; 2], r: f32) -> f32 {
    let dx = (p[0] - center[0]).abs() - half_size[0] + r;
    let dy = (p[1] - center[1]).abs() - half_size[1] + r;

    let outside_dist = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    let inside_dist = dx.max(dy).min(0.0);

    outside_dist + inside_dist - r
}

#[derive(Debug, Clone)]
pub struct ZoomState {
    pub zoom_target: f32,
    pub zoom_current: f32,
    pub last_used: f64, // time in seconds
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::selections::SelectionManager;

    #[test]
    fn test_selection_manager_basic() {
        let mut sm = SelectionManager::new();

        let elem = ElementRef::new("menu", "layer", "elem1", ElementKind::Circle);
        sm.select_single(elem.clone());

        assert!(sm.is_selected(&elem));
        assert_eq!(sm.count(), 1);
    }

    #[test]
    fn test_selection_manager_multi() {
        let mut sm = SelectionManager::new();

        let elem1 = ElementRef::new("menu", "layer", "elem1", ElementKind::Circle);
        let elem2 = ElementRef::new("menu", "layer", "elem2", ElementKind::Circle);

        sm.select_single(elem1.clone());
        sm.add_to_selection(elem2.clone());

        assert!(sm.is_selected(&elem2));
        assert_eq!(sm.count(), 2);
    }

    #[test]
    fn test_selection_manager_toggle() {
        let mut sm = SelectionManager::new();

        let elem = ElementRef::new("menu", "layer", "elem1", ElementKind::Circle);
        sm.select_single(elem.clone());
        sm.toggle_selection(elem.clone());

        assert!(!sm.is_selected(&elem));
        assert_eq!(sm.count(), 0);
    }

    #[test]
    fn test_drag_coordinator() {
        let mut dc = DragCoordinator::new();
        let config = TouchConfig {
            drag_threshold: 5.0,
            ..Default::default()
        };

        let elem = ElementRef::new("menu", "layer", "elem1", ElementKind::Circle);
        dc.begin(
            elem.clone(),
            Some(elem),
            MouseButtons::default(),
            [100.0, 100.0],
            [100.0, 100.0],
            None,
        );

        // Move less than threshold
        let events = dc.update([102.0, 102.0], &config);
        assert!(events.is_empty());
        assert!(!dc.is_dragging());

        // Move past threshold
        let events = dc.update([110.0, 110.0], &config);
        assert!(!events.is_empty());
        assert!(dc.is_dragging());

        // End drag
        let end_event = dc.end();
        assert!(end_event.is_some());
    }

    #[test]
    fn test_touch_config_snap() {
        let config = TouchConfig {
            snap_enabled: true,
            snap_grid_size: 10.0,
            ..Default::default()
        };

        let snapped = DragCoordinator::apply_snap([12.3, 17.8], &config);
        assert_eq!(snapped, [10.0, 20.0]);
    }

    // #[test]
    // fn test_event_queue() {
    //     let mut queue = TouchEventQueue::new(3);
    //
    //     queue.push(ElementEvent::DeselectAllRequested);
    //     queue.push(ElementEvent::DeselectAllRequested);
    //     queue.push(ElementEvent::DeselectAllRequested);
    //     queue.push(ElementEvent::DeselectAllRequested); // Should evict first
    //
    //     assert_eq!(queue.len(), 3);
    // }
}
