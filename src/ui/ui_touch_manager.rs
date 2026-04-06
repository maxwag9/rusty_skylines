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
use crate::renderer::ui_text_rendering::{Anchor, anchor_to};
use crate::ui::selections::SelectionManager;
use crate::ui::ui_editor::{GuiOptions, TouchableElement, Ui, get_element};
use crate::ui::ui_runtime::UiRuntimes;
use crate::ui::vertex::{
    ElementKind, UiButtonCircle, UiButtonHandle, UiButtonPolygon, UiButtonRect, UiButtonText,
    UiElement,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
// ============================================================================
// CONFIGURATION
// ============================================================================

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

// ============================================================================
// CORE TYPES
// ============================================================================

/// Reference to an element (menu/layer/id)
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ElementRef {
    pub menu: String,
    pub layer: String,
    pub id: String,
    pub kind: ElementKind,
}

impl ElementRef {
    pub fn action(&self, ui: &Ui) -> Vec<String> {
        match get_element(&ui.menus, self) {
            Some(e) => e.actions(),
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
    pub actions: Vec<String>,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct ButtonState {
    pub pressed: bool,
    pub just_pressed: bool,
    pub just_released: bool,
}

#[derive(Clone, Copy, Debug, Default)]
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
/// Mouse/touch input snapshot
#[derive(Clone, Copy, Debug, Default)]
pub struct InputSnapshot {
    pub position: [f32; 2],
    pub scroll_delta: f32,

    pub buttons: MouseButtons,

    pub ctrl_held: bool,
    pub shift_held: bool,
    pub alt_held: bool,
}

// ============================================================================
// EVENTS
// ============================================================================

/// All possible touch events emitted by the system
#[derive(Clone, Debug)]
pub enum TouchEvent {
    // Hover events
    HoverEnter {
        element: ElementRef,
        actions: Vec<String>,
    },
    Hovering {
        element: ElementRef,
        actions: Vec<String>,
    },
    HoverExit {
        element: ElementRef,
        actions: Vec<String>,
    },

    Nothing {
        element: ElementRef,
        actions: Vec<String>,
    },
    // Press/release events
    Press {
        element: ElementRef,
        position: [f32; 2],
        vertex_index: Option<usize>,
        actions: Vec<String>,
        buttons: MouseButtons,
    },
    Down {
        element: ElementRef,
        position: [f32; 2],
        vertex_index: Option<usize>,
        actions: Vec<String>,
        buttons: MouseButtons,
    },
    Release {
        element: ElementRef,
        position: [f32; 2],
        was_drag: bool,
        actions: Vec<String>,
        buttons: MouseButtons,
    },
    Click {
        element: ElementRef,
        position: [f32; 2],
        actions: Vec<String>,
        buttons: MouseButtons,
    },
    DoubleClick {
        element: ElementRef,
        position: [f32; 2],
        actions: Vec<String>,
        buttons: MouseButtons,
    },

    // Drag events
    DragStart {
        element: ElementRef,
        start_position: [f32; 2],
        vertex_index: Option<usize>,
    },
    DragMove {
        element: ElementRef,
        actions: Vec<String>,
        buttons: MouseButtons,
        current_position: [f32; 2],
        delta: [f32; 2],
        total_delta: [f32; 2],
    },
    DragEnd {
        element: ElementRef,
        start_position: [f32; 2],
        end_position: [f32; 2],
        vertex_index: Option<usize>,
    },

    // Selection events
    SelectionRequested {
        element: ElementRef,
        additive: bool,
        multi: bool,
    },
    DeselectAllRequested,
    BoxSelectStart {
        start: [f32; 2],
    },
    BoxSelectMove {
        current: [f32; 2],
    },
    BoxSelectEnd {
        start: [f32; 2],
        end: [f32; 2],
    },

    // Scroll/resize events
    ScrollOnElement {
        element: ElementRef,
        delta: f32,
        actions: Vec<String>,
    },

    // Text editing events
    TextEditRequested {
        element: ElementRef,
    },
    TextEditEnded {
        element: ElementRef,
    },

    // Navigation events
    NavigateDirection {
        direction: NavigationDirection,
    },
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
    fn action(&self) -> Vec<String>;
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
        self.misc.pressable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn action(&self) -> Vec<String> {
        self.actions.clone()
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
        self.misc.pressable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn action(&self) -> Vec<String> {
        self.actions.clone()
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
        self.misc.pressable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn action(&self) -> Vec<String> {
        self.actions.clone()
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
        let pos = anchor_to(
            self.anchor.unwrap_or(Anchor::Center),
            [self.x, self.y],
            self.width,
            self.height,
        );
        let pad = 2f32;
        let x0 = pos[0] + self.top_left_offset[0];
        let y0 = pos[1] + self.top_left_offset[1];
        let x1 = x0 + self.width + self.top_right_offset[0] + pad;
        let y1 = y0 + self.height + self.bottom_left_offset[1] + pad;

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
        self.misc.pressable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn action(&self) -> Vec<String> {
        self.actions.clone()
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
        self.misc.pressable
    }

    fn is_editable(&self, override_mode: bool) -> bool {
        self.misc.editable.editable(override_mode)
    }

    fn action(&self) -> Vec<String> {
        vec![]
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
        elements: &Vec<TouchableElement>,
        editor_mode: bool,
        override_mode: bool,
    ) -> Option<HitTestResult> {
        let mut best: Option<HitTestResult> = None;

        for element in elements {
            let hit = Self::test_element(
                point,
                element.menu,
                element.layer,
                element.order,
                element.idx,
                element.element,
                editor_mode,
                override_mode,
            );

            if let Some(candidate) = hit {
                match &best {
                    Some(current) if candidate.priority() > current.priority() => {
                        best = Some(candidate);
                    }
                    None => {
                        best = Some(candidate);
                    }
                    _ => {}
                }
            }
        }

        best
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
        let (id, kind, active, pressable, editable, actions) = match element {
            UiElement::Circle(c) => (
                c.id.clone(),
                ElementKind::Circle,
                c.misc.active,
                c.misc.pressable,
                &c.misc.editable,
                c.actions.clone(),
            ),
            UiElement::Polygon(p) => (
                p.id.clone(),
                ElementKind::Polygon,
                p.misc.active,
                p.misc.pressable,
                &p.misc.editable,
                p.actions.clone(),
            ),
            UiElement::Text(t) => (
                t.id.clone(),
                ElementKind::Text,
                t.misc.active,
                t.misc.pressable,
                &t.misc.editable,
                t.actions.clone(),
            ),
            UiElement::Handle(h) => (
                h.id.clone(),
                ElementKind::Handle,
                h.misc.active,
                h.misc.pressable,
                &h.misc.editable,
                vec![],
            ),
            UiElement::Outline(_) => return None, // Outlines aren't interactive
            UiElement::Rect(r) => (
                r.id.clone(),
                ElementKind::Rect,
                r.misc.active,
                r.misc.pressable,
                &r.misc.editable,
                r.actions.clone(),
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
            actions,
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
    pub actions: Vec<String>,
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
        actions: Vec<String>,
        buttons: MouseButtons,
        mouse_pos: [f32; 2],
        anchor: [f32; 2],
        vertex_index: Option<usize>,
    ) {
        let offset = [mouse_pos[0] - anchor[0], mouse_pos[1] - anchor[1]];

        self.active_drag = Some(ActiveDrag {
            element,
            affected_element,
            actions,
            buttons,
            start_position: mouse_pos,
            current_position: mouse_pos,
            offset,
            vertex_index,
            threshold_exceeded: false,
        });
    }

    /// Update drag with new position, returns events if any
    pub fn update(&mut self, mouse_pos: [f32; 2], config: &TouchConfig) -> Vec<TouchEvent> {
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
                events.push(TouchEvent::DragStart {
                    element: drag_element,
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

            events.push(TouchEvent::DragMove {
                element: drag.element.clone(),
                actions: drag.actions.clone(),
                buttons: drag.buttons,
                current_position: drag.current_position,
                delta,
                total_delta,
            });
        }

        drag.current_position = mouse_pos;

        events
    }

    /// End drag operation, returns DragEnd event if threshold was exceeded
    pub fn end(&mut self) -> Option<TouchEvent> {
        let drag = self.active_drag.take()?;

        if drag.threshold_exceeded {
            Some(TouchEvent::DragEnd {
                element: drag.element,
                start_position: drag.start_position,
                end_position: drag.current_position,
                vertex_index: drag.vertex_index,
            })
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

    /// Enter text editing mode
    pub fn begin_text_edit(&mut self, element: ElementRef) -> TouchEvent {
        self.editing_text = Some(element.clone());
        TouchEvent::TextEditRequested { element }
    }

    /// Exit text editing mode
    pub fn end_text_edit(&mut self) -> Option<TouchEvent> {
        self.editing_text
            .take()
            .map(|element| TouchEvent::TextEditEnded { element })
    }

    /// Check if currently editing text
    pub fn is_editing_text(&self) -> bool {
        self.editing_text.is_some()
    }

    /// Check if editing specific element
    pub fn is_editing(&self, element: &ElementRef) -> bool {
        self.editing_text.as_ref() == Some(element)
    }

    /// Process scroll event for resizing (editor mode only)
    pub fn process_scroll(
        &mut self,
        element: &ElementRef,
        action: &Vec<String>,
        scroll_delta: f32,
    ) -> Option<TouchEvent> {
        if !self.enabled || scroll_delta == 0.0 {
            return None;
        }

        Some(TouchEvent::ScrollOnElement {
            element: element.clone(),
            delta: scroll_delta,
            actions: action.clone(),
        })
    }
}

// ============================================================================
// EVENT QUEUE
// ============================================================================

/// Queue of touch events to be processed by subscribers
#[derive(Clone, Debug, Default)]
pub struct TouchEventQueue {
    events: VecDeque<TouchEvent>,
    capacity: usize,
}

impl TouchEventQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, event: TouchEvent) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    pub fn push_all(&mut self, events: impl IntoIterator<Item = TouchEvent>) {
        for event in events {
            self.push(event);
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = TouchEvent> + '_ {
        self.events.drain(..)
    }

    pub fn iter(&self) -> impl Iterator<Item = &TouchEvent> {
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

// ============================================================================
// GLOBAL INTERACTION STATE
// ============================================================================

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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CurrentHover {
    pub element: ElementRef,
    pub actions: Vec<String>,
}

/// Central coordinator for all UI touch interactions
pub struct UiTouchManager {
    // Configuration
    pub config: TouchConfig,

    // Sub-components
    pub selection: SelectionManager,
    pub drag: DragCoordinator,
    pub editor: EditorTouchExtension,
    pub events: TouchEventQueue,
    pub runtimes: UiRuntimes,
    // State
    element_states: HashMap<String, ElementTouchData>,
    current_hover: Option<CurrentHover>,
    interaction_mode: InteractionMode,
    pub(crate) last_input: InputSnapshot,

    // Timing
    accumulated_time: f32,
    pub options: GuiOptions,
}

impl UiTouchManager {
    pub fn new(settings: &Settings) -> Self {
        Self {
            config: TouchConfig::default(),
            selection: SelectionManager::new(),
            drag: DragCoordinator::new(),
            editor: EditorTouchExtension::new(settings.editor_mode),
            events: TouchEventQueue::new(64),
            runtimes: UiRuntimes::new(),
            element_states: HashMap::new(),
            current_hover: None,
            interaction_mode: InteractionMode::None,
            last_input: InputSnapshot::default(),
            accumulated_time: 0.0,
            options: GuiOptions {
                override_mode: settings.override_mode,
                show_gui: settings.show_gui,
            },
        }
    }

    /// Update touch manager with new input
    pub fn update(&mut self, dt: f32, input: InputSnapshot, elements: &Vec<TouchableElement>) {
        self.accumulated_time += dt;
        self.selection.reset_frame_flags();
        self.events.clear();

        // Update config from input modifiers
        self.config.multi_select_active = input.ctrl_held;
        self.config.additive_select_active = input.shift_held;

        // Find what we're hitting
        let top_hit = HitDetector::find_top_hit(
            input.position,
            &elements,
            self.editor.enabled,
            self.options.override_mode,
        );

        // Process hover changes
        self.process_hover(&top_hit);

        // Process press/release/drag
        self.process_press_release(&input, &top_hit);

        // Process scroll
        if input.scroll_delta != 0.0 {
            self.process_scroll(input.scroll_delta);
        }

        // Handle box selection if active
        if self.selection.is_box_selecting() && input.buttons.pressed() {
            self.events.push(TouchEvent::BoxSelectMove {
                current: input.position,
            });
        }
        for touchable_element in elements {
            self.events.push(TouchEvent::Nothing {
                element: ElementRef::new(
                    touchable_element.menu,
                    touchable_element.layer,
                    touchable_element.element.id(),
                    touchable_element.element.kind(),
                ),
                actions: touchable_element.element.actions(),
            });
        }
        self.last_input = input;
    }

    /// Process hover state changes
    fn process_hover(&mut self, top_hit: &Option<HitTestResult>) {
        let new_hover = top_hit.as_ref().map(|h| CurrentHover {
            element: h.element_ref.clone(),
            actions: h.actions.clone(),
        });

        // Check if hover target changed
        if self.current_hover != new_hover {
            // Exit old hover
            if let Some(old) = &self.current_hover {
                self.events.push(TouchEvent::HoverExit {
                    element: old.element.clone(),
                    actions: old.actions.clone(),
                });
                if let Some(state) = self.element_states.get_mut(&old.element.id) {
                    if state.state == ElementTouchState::Hovered {
                        state.state = ElementTouchState::Idle;
                    }
                }
            }

            // Enter new hover
            if let Some(new) = &new_hover {
                self.events.push(TouchEvent::HoverEnter {
                    element: new.element.clone(),
                    actions: new.actions.clone(),
                });
                let state = self
                    .element_states
                    .entry(new.element.id.clone())
                    .or_default();
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
            if let Some(current_hover) = &self.current_hover {
                self.events.push(TouchEvent::Hovering {
                    element: current_hover.element.clone(),
                    actions: current_hover.actions.clone(),
                });
            }
        }
    }

    /// Process press and release events
    fn process_press_release(&mut self, input: &InputSnapshot, top_hit: &Option<HitTestResult>) {
        // Just pressed
        if input.buttons.just_pressed() {
            self.handle_press(input, top_hit);
        }

        // Held (potential drag)
        if input.buttons.pressed() && !input.buttons.just_pressed() {
            self.handle_held(input, top_hit);
        }

        // Just released
        if input.buttons.just_released() {
            self.handle_release(input);
        }
    }

    /// Handle mouse press
    fn handle_press(&mut self, input: &InputSnapshot, top_hit: &Option<HitTestResult>) {
        if let Some(hit) = top_hit {
            let element = &hit.element_ref;

            // Update element state
            let state = self.element_states.entry(element.id.clone()).or_default();
            state.state = ElementTouchState::Pressed { frame_count: 0 };
            state.press_position = Some(input.position);
            state.press_time = self.accumulated_time;
            state.vertex_index = hit.vertex_index;

            // Emit press event
            self.events.push(TouchEvent::Press {
                element: element.clone(),
                position: input.position,
                vertex_index: hit.vertex_index,
                actions: hit.actions.clone(),
                buttons: input.buttons,
            });
            self.events.push(TouchEvent::Down {
                element: element.clone(),
                position: input.position,
                vertex_index: hit.vertex_index,
                actions: hit.actions.clone(),
                buttons: input.buttons,
            });
            // Begin potential drag
            if !hit.text_being_edited.unwrap_or(false) {
                let anchor = input.position; // Could get from element's drag anchor
                self.drag.begin(
                    element.clone(),
                    hit.affected_element.clone(),
                    hit.actions.clone(),
                    input.buttons,
                    input.position,
                    anchor,
                    hit.vertex_index,
                );
            }

            // Handle selection
            let selection_event = if self.config.multi_select_active {
                TouchEvent::SelectionRequested {
                    element: element.clone(),
                    additive: false,
                    multi: true,
                }
            } else if self.config.additive_select_active {
                TouchEvent::SelectionRequested {
                    element: element.clone(),
                    additive: true,
                    multi: false,
                }
            } else {
                TouchEvent::SelectionRequested {
                    element: element.clone(),
                    additive: false,
                    multi: false,
                }
            };
            self.events.push(selection_event);

            self.interaction_mode = InteractionMode::Pressing;
        } else {
            // Clicked on empty space
            if !self.config.additive_select_active {
                self.events.push(TouchEvent::DeselectAllRequested);
            }

            // Begin box select if in editor mode
            if self.editor.enabled {
                self.selection.begin_box_select(input.position);
                self.events.push(TouchEvent::BoxSelectStart {
                    start: input.position,
                });
                self.interaction_mode = InteractionMode::BoxSelecting;
            }
        }
    }

    /// Handle mouse held
    fn handle_held(&mut self, input: &InputSnapshot, top_hit: &Option<HitTestResult>) {
        // Update drag
        let drag_events = self.drag.update(input.position, &self.config);

        if !drag_events.is_empty() {
            self.interaction_mode = InteractionMode::Dragging;
            self.events.push_all(drag_events);
        }

        // Update pressed element state
        for state in self.element_states.values_mut() {
            if let ElementTouchState::Pressed { frame_count } = &mut state.state {
                *frame_count += 1;
                if let Some(hit) = top_hit {
                    let element = &hit.element_ref;
                    self.events.push(TouchEvent::Down {
                        element: element.clone(),
                        position: input.position,
                        vertex_index: hit.vertex_index,
                        actions: hit.actions.clone(),
                        buttons: input.buttons,
                    });
                }
            }
        }
    }

    /// Handle mouse release
    fn handle_release(&mut self, input: &InputSnapshot) {
        // End drag
        if let Some(drag_end) = self.drag.end() {
            self.events.push(drag_end);
        }

        // End box select
        if let Some(start) = self.selection.end_box_select() {
            self.events.push(TouchEvent::BoxSelectEnd {
                start,
                end: input.position,
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
            if let Some(hover) = &self.current_hover {
                if hover.element.id == id {
                    self.events.push(TouchEvent::Release {
                        element: hover.element.clone(),
                        position: input.position,
                        was_drag,
                        actions: hover.actions.clone(),
                        buttons: input.buttons,
                    });

                    // If it wasn't a drag, it's a click
                    if !was_drag {
                        let state = self.element_states.get(&id);
                        let time_since_last_click = state
                            .map(|s| self.accumulated_time - s.last_click_time)
                            .unwrap_or(f32::MAX);

                        if time_since_last_click < self.config.double_click_time.as_secs_f32() {
                            self.events.push(TouchEvent::DoubleClick {
                                element: hover.element.clone(),
                                position: input.position,
                                actions: hover.actions.clone(),
                                buttons: input.buttons,
                            });
                        } else {
                            self.events.push(TouchEvent::Click {
                                element: hover.element.clone(),
                                position: input.position,
                                actions: hover.actions.clone(),
                                buttons: input.buttons,
                            });
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
            if self.editor.enabled {
                if let Some(event) =
                    self.editor
                        .process_scroll(&hover.element, &hover.actions, delta)
                {
                    self.events.push(event);
                }
            }
        }
    }

    /// Process keyboard navigation
    pub fn process_navigation(&mut self, direction: NavigationDirection) {
        self.events
            .push(TouchEvent::NavigateDirection { direction });
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
    pub fn hovered(&self) -> Option<&CurrentHover> {
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

    /// Drain all events for processing
    pub fn drain_events(&mut self) -> impl Iterator<Item = TouchEvent> + '_ {
        self.events.drain()
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
            vec![],
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

    #[test]
    fn test_event_queue() {
        let mut queue = TouchEventQueue::new(3);

        queue.push(TouchEvent::DeselectAllRequested);
        queue.push(TouchEvent::DeselectAllRequested);
        queue.push(TouchEvent::DeselectAllRequested);
        queue.push(TouchEvent::DeselectAllRequested); // Should evict first

        assert_eq!(queue.len(), 3);
    }
}
