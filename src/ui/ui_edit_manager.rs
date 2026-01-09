//! Undo/Redo system for the UI Editor
//!
//! Supports Ctrl+Z (undo) and Ctrl+Shift+Z (redo)
//!
//! Uses Command pattern with concrete types implementing the Command trait.
//! Each command stores before/after state for explicit undo/redo semantics.

use crate::ui::input::MouseState;
use crate::ui::menu::Menu;
use crate::ui::ui_edits::*;
use crate::ui::ui_runtime::UiRuntime;
use crate::ui::ui_touch_manager::{ElementRef, UiTouchManager};
use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::*;
use std::any::Any;
use std::collections::{HashMap, VecDeque};

/// Maximum number of undo steps to keep
const MAX_UNDO_HISTORY: usize = 100;

/// Coalescing timeout in seconds - actions within this window are merged
const COALESCE_TIMEOUT: f32 = 0.3;

// ============================================================================
// Command Trait
// ============================================================================

/// Command trait for undoable actions
pub trait Command: Any {
    /// Apply undo (restore before state)
    fn undo(
        &self,
        touch_manager: &mut UiTouchManager,
        ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    );

    /// Apply redo (apply after state)
    fn redo(
        &self,
        touch_manager: &mut UiTouchManager,
        ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    );

    /// Human-readable description for UI
    fn description(&self) -> String;

    /// For downcasting
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Try to merge another command into this one (for coalescing).
    /// Returns true if merged successfully.
    fn try_coalesce(&mut self, _other: &dyn Command) -> bool {
        false
    }

    /// Clone into a Box (required for redo stack)
    fn clone_box(&self) -> Box<dyn Command>;
}

impl dyn Command {
    pub fn is<T: Command + 'static>(&self) -> bool {
        self.as_any().is::<T>()
    }

    pub fn downcast_ref<T: Command + 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }

    pub fn downcast_mut<T: Command + 'static>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut::<T>()
    }
}

impl Clone for Box<dyn Command> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ============================================================================
// Concrete Command Types
// ============================================================================

/// Command for moving a single element
#[derive(Clone, Debug)]
pub struct MoveElementCommand {
    pub affected_element: ElementRef,
    pub before: (f32, f32),
    pub after: (f32, f32),
}

impl Command for MoveElementCommand {
    fn undo(
        &self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_element_position(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.affected_element.kind,
            self.before,
        );
    }

    fn redo(
        &self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_element_position(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.affected_element.kind,
            self.after,
        );
    }

    fn description(&self) -> String {
        format!("Move '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn try_coalesce(&mut self, other: &dyn Command) -> bool {
        let Some(other) = other.downcast_ref::<Self>() else {
            return false;
        };
        if self.affected_element == other.affected_element {
            self.after = other.after;
            true
        } else {
            false
        }
    }

    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for resizing an element
#[derive(Clone, Debug)]
pub struct ResizeElementCommand {
    pub affected_element: ElementRef,
    pub before: f32,
    pub after: f32,
}

impl Command for ResizeElementCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_element_size(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.affected_element.kind,
            self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_element_size(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.affected_element.kind,
            self.after,
        );
    }

    fn description(&self) -> String {
        format!("Resize '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn try_coalesce(&mut self, other: &dyn Command) -> bool {
        let Some(other) = other.downcast_ref::<Self>() else {
            return false;
        };
        if self.affected_element == other.affected_element {
            self.after = other.after;
            true
        } else {
            false
        }
    }

    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for modifying a circle's full state
#[derive(Clone, Debug)]
pub struct ModifyCircleCommand {
    pub affected_element: ElementRef,
    pub before: UiButtonCircle,
    pub after: UiButtonCircle,
}

impl Command for ModifyCircleCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        replace_circle(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        replace_circle(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.after,
        );
    }

    fn description(&self) -> String {
        format!("Modify circle '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for modifying text's full state
#[derive(Clone, Debug)]
pub struct ModifyTextCommand {
    pub affected_element: ElementRef,
    pub before: UiButtonText,
    pub after: UiButtonText,
}

impl Command for ModifyTextCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        replace_text(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        replace_text(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.after,
        );
    }

    fn description(&self) -> String {
        format!("Modify text '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for modifying polygon's full state
#[derive(Clone, Debug)]
pub struct ModifyPolygonCommand {
    pub affected_element: ElementRef,
    pub before: UiButtonPolygon,
    pub after: UiButtonPolygon,
}

impl Command for ModifyPolygonCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        replace_polygon(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        replace_polygon(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.after,
        );
    }

    fn description(&self) -> String {
        format!("Modify polygon '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for creating an element
#[derive(Clone, Debug)]
pub struct CreateElementCommand {
    pub affected_element: ElementRef,
    pub element: UiElement,
}

impl Command for CreateElementCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        delete_element(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.element,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        let _ = create_element(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            self.element.clone(),
            mouse,
        );
    }

    fn description(&self) -> String {
        format!("Create {}", self.element.kind_name())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for deleting an element
#[derive(Clone, Debug)]
pub struct DeleteElementCommand {
    pub affected_element: ElementRef,
    pub element: UiElement,
}

impl Command for DeleteElementCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        let _ = create_element(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            self.element.clone(),
            mouse,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        delete_element(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.element,
        );
    }

    fn description(&self) -> String {
        format!("Delete {}", self.element.kind_name())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for changing z-index
#[derive(Clone, Debug)]
pub struct ChangeZIndexCommand {
    pub affected_element: ElementRef,
    pub delta: i32,
}

impl Command for ChangeZIndexCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        change_z_index(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            -self.delta,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        change_z_index(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.delta,
        );
    }

    fn description(&self) -> String {
        format!("Change Z-index of '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for changing layer order
#[derive(Clone, Debug)]
pub struct ChangeLayerOrderCommand {
    pub affected_element: ElementRef,
    pub before: u32,
    pub after: u32,
}

impl Command for ChangeLayerOrderCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_layer_order(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_layer_order(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            self.after,
        );
    }

    fn description(&self) -> String {
        format!("Reorder layer '{}'", self.affected_element.layer)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for text content editing
#[derive(Clone, Debug)]
pub struct TextEditCommand {
    pub affected_element: ElementRef,
    pub before_text: String,
    pub after_text: String,
    pub before_template: String,
    pub after_template: String,
    pub before_caret: usize,
    pub after_caret: usize,
}

impl Command for TextEditCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_text_content(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            &self.before_text,
            &self.before_template,
            self.before_caret,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_text_content(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            &self.after_text,
            &self.after_template,
            self.after_caret,
        );
    }

    fn description(&self) -> String {
        format!("Edit text '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for moving a polygon vertex
#[derive(Clone, Debug)]
pub struct MoveVertexCommand {
    pub affected_element: ElementRef,
    pub vertex_index: usize,
    pub before: [f32; 2],
    pub after: [f32; 2],
}

impl Command for MoveVertexCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_vertex_position(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.vertex_index,
            self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_vertex_position(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.vertex_index,
            self.after,
        );
    }

    fn description(&self) -> String {
        format!(
            "Move vertex {} of '{}'",
            self.vertex_index, self.affected_element.id
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn try_coalesce(&mut self, other: &dyn Command) -> bool {
        let Some(other) = other.downcast_ref::<Self>() else {
            return false;
        };
        if self.affected_element == other.affected_element
            && self.vertex_index == other.vertex_index
        {
            self.after = other.after;
            true
        } else {
            false
        }
    }

    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Color properties that can be changed
#[derive(Clone, Debug, PartialEq)]
pub enum ColorProperty {
    Fill,
    Border,
    InsideBorder,
    Glow,
    TextColor,
    DashColor,
    SubDashColor,
}

impl std::fmt::Display for ColorProperty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fill => write!(f, "fill"),
            Self::Border => write!(f, "border"),
            Self::InsideBorder => write!(f, "inside_border"),
            Self::Glow => write!(f, "glow"),
            Self::TextColor => write!(f, "text"),
            Self::DashColor => write!(f, "dash"),
            Self::SubDashColor => write!(f, "sub_dash"),
        }
    }
}

/// Command for changing a color property
#[derive(Clone, Debug)]
pub struct ChangeColorCommand {
    pub affected_element: ElementRef,
    pub property: ColorProperty,
    pub before: [f32; 4],
    pub after: [f32; 4],
}

impl Command for ChangeColorCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_element_color(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.affected_element.kind,
            &self.property,
            self.before,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        set_element_color(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.affected_element.id,
            self.affected_element.kind,
            &self.property,
            self.after,
        );
    }

    fn description(&self) -> String {
        format!(
            "Change {} color of '{}'",
            self.property, self.affected_element.id
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for duplicating an element
#[derive(Clone, Debug)]
pub struct DuplicateElementCommand {
    pub affected_element: ElementRef,
    pub new_element: UiElement,
}

impl Command for DuplicateElementCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        delete_element(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            &self.new_element,
        );
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        let _ = create_element(
            menus,
            &self.affected_element.menu,
            &self.affected_element.layer,
            self.new_element.clone(),
            mouse,
        );
    }

    fn description(&self) -> String {
        format!("Duplicate '{}'", self.affected_element.id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command for deselecting all elements
#[derive(Clone, Debug)]
pub struct DeselectAllCommand {
    pub(crate) primary: Option<ElementRef>,
    pub(crate) secondary: Vec<ElementRef>,
}

impl Command for DeselectAllCommand {
    fn undo(
        &self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        touch_manager
            .selection
            .select_from_overwrite(menus, &self.primary, &self.secondary);
    }

    fn redo(
        &self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        touch_manager.selection.deselect_all(menus);
    }

    fn description(&self) -> String {
        "Deselected all Ui Elements".to_string()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Data for a single element move within MoveMultipleCommand
#[derive(Clone, Debug)]
pub struct ElementMove {
    pub affected_element: ElementRef,
    pub before: (f32, f32),
    pub after: (f32, f32),
}

/// Command for moving multiple elements at once
#[derive(Clone, Debug)]
pub struct MoveMultipleCommand {
    pub moves: Vec<ElementMove>,
}

impl Command for MoveMultipleCommand {
    fn undo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        for m in &self.moves {
            set_element_position(
                menus,
                &m.affected_element.menu,
                &m.affected_element.layer,
                &m.affected_element.id,
                m.affected_element.kind,
                m.before,
            );
        }
    }

    fn redo(
        &self,
        _touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        _mouse: &MouseState,
    ) {
        for m in &self.moves {
            set_element_position(
                menus,
                &m.affected_element.menu,
                &m.affected_element.layer,
                &m.affected_element.id,
                m.affected_element.kind,
                m.after,
            );
        }
    }

    fn description(&self) -> String {
        format!("Move {} elements", self.moves.len())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn try_coalesce(&mut self, other: &dyn Command) -> bool {
        let Some(other) = other.downcast_ref::<Self>() else {
            return false;
        };

        if self.moves.len() != other.moves.len() {
            return false;
        }

        // Verify same elements in same order
        let matches = self
            .moves
            .iter()
            .zip(&other.moves)
            .all(|(a, b)| a.affected_element == b.affected_element);

        if matches {
            for (existing, new) in self.moves.iter_mut().zip(&other.moves) {
                existing.after = new.after;
            }
            true
        } else {
            false
        }
    }

    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

/// Command that batches multiple commands as a single undo step
pub struct BatchCommand {
    pub commands: Vec<Box<dyn Command>>,
    pub description_text: String,
}

impl Clone for BatchCommand {
    fn clone(&self) -> Self {
        Self {
            commands: self.commands.iter().map(|c| c.clone_box()).collect(),
            description_text: self.description_text.clone(),
        }
    }
}

impl std::fmt::Debug for BatchCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchCommand")
            .field("description", &self.description_text)
            .field("count", &self.commands.len())
            .finish()
    }
}

impl Command for BatchCommand {
    fn undo(
        &self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        // Undo in reverse order
        for cmd in self.commands.iter().rev() {
            cmd.undo(touch_manager, _ui_runtime, menus, _variables, mouse);
        }
    }

    fn redo(
        &self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        // Redo in forward order
        for cmd in &self.commands {
            cmd.redo(touch_manager, _ui_runtime, menus, _variables, mouse);
        }
    }

    fn description(&self) -> String {
        self.description_text.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

///MANAGER!! Karen >:( ///

struct TimestampedCommand {
    command: Box<dyn Command>,
    timestamp: f32,
}

struct BatchBuilder {
    commands: Vec<Box<dyn Command>>,
    description: String,
}

/// Manages undo/redo history with coalescing support
pub struct UiEditManager {
    undo_stack: VecDeque<TimestampedCommand>,
    redo_stack: Vec<Box<dyn Command>>,
    in_undo_redo: bool,
    pending_batch: Option<BatchBuilder>,
    saved_position: Option<usize>,
    history_position: usize,
    last_action_time: f32,
    current_time: f32,
}

impl Default for UiEditManager {
    fn default() -> Self {
        Self::new()
    }
}

impl UiEditManager {
    pub fn new() -> Self {
        Self {
            undo_stack: VecDeque::with_capacity(MAX_UNDO_HISTORY),
            redo_stack: Vec::with_capacity(MAX_UNDO_HISTORY / 2),
            in_undo_redo: false,
            pending_batch: None,
            saved_position: Some(0),
            history_position: 0,
            last_action_time: 0.0,
            current_time: 0.0,
        }
    }

    /// Update internal time (call each frame with dt)
    pub fn update(&mut self, dt: f32) {
        self.current_time += dt;
    }

    /// Start a batch operation
    pub fn begin_batch(&mut self, description: impl Into<String>) {
        if self.pending_batch.is_none() {
            self.pending_batch = Some(BatchBuilder {
                commands: Vec::new(),
                description: description.into(),
            });
        }
    }

    /// End batch and commit as single undo step
    pub fn end_batch(&mut self) {
        if let Some(batch) = self.pending_batch.take() {
            if batch.commands.is_empty() {
                return;
            }
            if batch.commands.len() == 1 {
                self.push_internal(batch.commands.into_iter().next().unwrap());
            } else {
                self.push_internal(Box::new(BatchCommand {
                    commands: batch.commands,
                    description_text: batch.description,
                }));
            }
        }
    }

    /// Cancel current batch without committing
    pub fn cancel_batch(&mut self) {
        self.pending_batch = None;
    }

    /// Check if currently building a batch
    pub fn is_batching(&self) -> bool {
        self.pending_batch.is_some()
    }

    /// Push a command (boxed)
    pub fn push(&mut self, command: Box<dyn Command>) {
        if self.in_undo_redo {
            return;
        }

        if let Some(ref mut batch) = self.pending_batch {
            batch.commands.push(command);
        } else {
            if self.try_coalesce(&*command) {
                return;
            }
            self.push_internal(command);
        }
    }

    /// Convenience: push without explicit Box
    pub fn push_command<C: Command + 'static>(&mut self, command: C) {
        self.push(Box::new(command));
    }

    /// Try to merge with previous command
    fn try_coalesce(&mut self, command: &dyn Command) -> bool {
        if self.current_time - self.last_action_time > COALESCE_TIMEOUT {
            return false;
        }

        let Some(last) = self.undo_stack.back_mut() else {
            return false;
        };

        if last.command.try_coalesce(command) {
            last.timestamp = self.current_time;
            self.last_action_time = self.current_time;
            true
        } else {
            false
        }
    }

    /// Execute a command immediately and push it onto the undo stack
    pub fn execute(
        &mut self,
        command: Box<dyn Command>,
        touch_manager: &mut UiTouchManager,
        ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        if self.in_undo_redo {
            return;
        }

        // Execute the command immediately
        command.redo(touch_manager, ui_runtime, menus, variables, mouse);

        // Now push it for undo capability
        if let Some(ref mut batch) = self.pending_batch {
            batch.commands.push(command);
        } else {
            if self.try_coalesce(&*command) {
                return; // Merged with previous, command dropped
            }
            self.push_internal(command);
        }
    }

    /// Convenience: execute without explicit Box
    pub fn execute_command<C: Command + 'static>(
        &mut self,
        command: C,
        touch_manager: &mut UiTouchManager,
        ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) {
        self.execute(
            Box::new(command),
            touch_manager,
            ui_runtime,
            menus,
            variables,
            mouse,
        );
    }

    fn push_internal(&mut self, command: Box<dyn Command>) {
        self.redo_stack.clear();

        self.undo_stack.push_back(TimestampedCommand {
            command,
            timestamp: self.current_time,
        });

        self.history_position += 1;
        self.last_action_time = self.current_time;

        while self.undo_stack.len() > MAX_UNDO_HISTORY {
            self.undo_stack.pop_front();
            if let Some(ref mut saved) = self.saved_position {
                if *saved > 0 {
                    *saved -= 1;
                } else {
                    self.saved_position = None;
                }
            }
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty() && !self.in_undo_redo
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty() && !self.in_undo_redo
    }

    pub fn undo_description(&self) -> Option<String> {
        self.undo_stack.back().map(|t| t.command.description())
    }

    pub fn redo_description(&self) -> Option<String> {
        self.redo_stack.last().map(|c| c.description())
    }

    /// Perform undo. Returns description of what was undone.
    pub fn undo(
        &mut self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) -> Option<String> {
        if !self.can_undo() {
            return None;
        }

        self.in_undo_redo = true;

        let timestamped = self.undo_stack.pop_back()?;
        let desc = timestamped.command.description();

        timestamped
            .command
            .undo(touch_manager, _ui_runtime, menus, _variables, mouse);

        self.history_position = self.history_position.saturating_sub(1);
        self.redo_stack.push(timestamped.command);

        self.in_undo_redo = false;
        Some(desc)
    }

    /// Perform redo. Returns description of what was redone.
    pub fn redo(
        &mut self,
        touch_manager: &mut UiTouchManager,
        _ui_runtime: &mut UiRuntime,
        menus: &mut HashMap<String, Menu>,
        _variables: &mut UiVariableRegistry,
        mouse: &MouseState,
    ) -> Option<String> {
        if !self.can_redo() {
            return None;
        }

        self.in_undo_redo = true;

        let command = self.redo_stack.pop()?;
        let desc = command.description();

        command.redo(touch_manager, _ui_runtime, menus, _variables, mouse);

        self.history_position += 1;
        self.undo_stack.push_back(TimestampedCommand {
            command,
            timestamp: self.current_time,
        });

        self.in_undo_redo = false;
        Some(desc)
    }

    /// Access last command for direct mutation during drag operations
    pub fn last_command_mut<T: Command + 'static>(&mut self) -> Option<&mut T> {
        if self.in_undo_redo {
            return None;
        }
        self.undo_stack
            .back_mut()?
            .command
            .as_any_mut()
            .downcast_mut::<T>()
    }

    pub fn mark_saved(&mut self) {
        self.saved_position = Some(self.history_position);
    }

    pub fn is_dirty(&self) -> bool {
        self.saved_position != Some(self.history_position)
    }

    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.pending_batch = None;
        self.history_position = 0;
        self.saved_position = Some(0);
    }

    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_coalescing() {
        let mut cmd1 = MoveElementCommand {
            affected_element: ElementRef::default(),
            before: (0.0, 0.0),
            after: (5.0, 5.0),
        };

        let cmd2 = MoveElementCommand {
            affected_element: ElementRef::default(),
            before: (5.0, 5.0),
            after: (10.0, 10.0),
        };

        assert!(cmd1.try_coalesce(&cmd2));
        assert_eq!(cmd1.before, (0.0, 0.0));
        assert_eq!(cmd1.after, (10.0, 10.0));
    }

    #[test]
    fn test_no_coalesce_different_element() {
        let mut cmd1 = MoveElementCommand {
            affected_element: ElementRef::default(),
            before: (0.0, 0.0),
            after: (5.0, 5.0),
        };
        let mut affected_2 = ElementRef::default();
        affected_2.id = "e2".into();
        let cmd2 = MoveElementCommand {
            affected_element: affected_2,
            before: (0.0, 0.0),
            after: (10.0, 10.0),
        };

        assert!(!cmd1.try_coalesce(&cmd2));
    }

    #[test]
    fn test_batch_undo_order() {
        let mut affected_2 = ElementRef::default();
        affected_2.id = "b".into();
        let batch = BatchCommand {
            commands: vec![
                Box::new(MoveElementCommand {
                    affected_element: ElementRef::default(),
                    before: (0.0, 0.0),
                    after: (1.0, 1.0),
                }),
                Box::new(MoveElementCommand {
                    affected_element: affected_2,
                    before: (0.0, 0.0),
                    after: (2.0, 2.0),
                }),
            ],
            description_text: "Move multiple".into(),
        };

        assert_eq!(batch.description(), "Move multiple");
        assert_eq!(batch.commands.len(), 2);
    }

    #[test]
    fn test_dirty_tracking() {
        let mut manager = UiEditManager::new();
        assert!(!manager.is_dirty());

        manager.push_command(CreateElementCommand {
            affected_element: ElementRef::default(),
            element: UiElement::Circle(UiButtonCircle::default()),
        });

        assert!(manager.is_dirty());
        manager.mark_saved();
        assert!(!manager.is_dirty());
    }
}
