use crate::ui::ui_editor::{ElementKind, UiButtonLoader};

#[derive(Debug, Clone)]
pub struct SelectedUiElement {
    pub menu_name: String,
    pub layer_name: String,
    pub element_id: String,
    pub active: bool,
    pub just_deselected: bool,
    pub dragging: bool,
    pub element_type: ElementKind,
    pub just_selected: bool,
    pub action_name: String,
    pub input_box: bool,
}

impl SelectedUiElement {
    pub(crate) fn default() -> SelectedUiElement {
        Self {
            menu_name: "no menu".to_string(),
            layer_name: "no layer".to_string(),
            element_id: "no element".to_string(),
            active: false,
            just_deselected: true,
            dragging: false,
            element_type: ElementKind::None,
            just_selected: false,
            action_name: "None".to_string(),
            input_box: false,
        }
    }
}

pub fn select_ui_element(loader: &mut UiButtonLoader, s: SelectedUiElement) {
    loader.ui_runtime.selected_ui_element_multi.clear();
    loader.ui_runtime.selected_ui_element_primary = make_selected_element(&s);
    loader.variables.set_string(
        "selected_menu",
        loader
            .ui_runtime
            .selected_ui_element_primary
            .menu_name
            .to_string(),
    );
    loader.variables.set_string(
        "selected_layer",
        loader
            .ui_runtime
            .selected_ui_element_primary
            .layer_name
            .to_string(),
    );
    loader.variables.set_string(
        "selected_ui_element_id",
        loader
            .ui_runtime
            .selected_ui_element_primary
            .element_id
            .to_string(),
    );
    loader.update_selection()
}

pub fn select_to_multi(loader: &mut UiButtonLoader, s: SelectedUiElement) {
    loader
        .ui_runtime
        .selected_ui_element_multi
        .push(make_selected_element(&s));
}

pub fn select_move_primary_to_multi(loader: &mut UiButtonLoader, s: SelectedUiElement) {
    loader
        .ui_runtime
        .selected_ui_element_multi
        .push(loader.ui_runtime.selected_ui_element_primary.clone());

    loader.ui_runtime.selected_ui_element_primary = make_selected_element(&s);
    println!(
        "Selected multi: {:?}, Selected primary: {:?}",
        loader.ui_runtime.selected_ui_element_multi[0].element_id,
        loader.ui_runtime.selected_ui_element_primary.element_id
    );
    loader.update_selection()
}

pub fn make_selected_element(s: &SelectedUiElement) -> SelectedUiElement {
    SelectedUiElement {
        menu_name: s.menu_name.clone(),
        layer_name: s.layer_name.clone(),
        element_id: s.element_id.clone(),
        active: true,
        just_deselected: if s.element_type == ElementKind::None {
            true
        } else {
            false
        },
        dragging: s.dragging,
        element_type: s.element_type,
        just_selected: if s.element_type == ElementKind::None {
            false
        } else {
            true
        },
        action_name: s.action_name.clone(),
        input_box: s.input_box,
    }
}

pub fn deselect_everything(loader: &mut UiButtonLoader) {
    let menu_name = loader
        .ui_runtime
        .selected_ui_element_primary
        .menu_name
        .clone();
    let layer_name = loader
        .ui_runtime
        .selected_ui_element_primary
        .layer_name
        .clone();
    let element_id = loader
        .ui_runtime
        .selected_ui_element_primary
        .element_id
        .clone();
    loader.ui_runtime.selected_ui_element_primary = SelectedUiElement::default();
    loader.ui_runtime.selected_ui_element_primary.menu_name = menu_name.clone();
    loader.ui_runtime.selected_ui_element_primary.layer_name = layer_name.clone();
    loader.ui_runtime.selected_ui_element_primary.element_id = element_id.clone();
    loader.ui_runtime.editing_text = false;
    for (_, menu) in loader.menus.iter_mut() {
        for layer in menu.layers.iter_mut() {
            for text in &mut layer.texts {
                text.being_edited = false;
            }
        }
    }
    println!("deselection");
    loader.ui_runtime.selected_ui_element_multi.clear();
    loader.update_selection()
}
