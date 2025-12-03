use crate::ui::ui_editor::{ElementKind, SelectedUiElement, UiButtonLoader};

pub fn select_ui_element(loader: &mut UiButtonLoader, s: SelectedUiElement) {
    loader.ui_runtime.selected_ui_element_multi.clear();
    loader.ui_runtime.selected_ui_element_primary = make_selected_element(&s);
    loader.variables.set(
        "selected_menu",
        format!(
            "{}",
            loader.ui_runtime.selected_ui_element_primary.menu_name
        ),
    );
    loader.variables.set(
        "selected_layer",
        format!(
            "{}",
            loader.ui_runtime.selected_ui_element_primary.layer_name
        ),
    );
    loader.variables.set(
        "selected_ui_element_id",
        format!(
            "{}",
            loader.ui_runtime.selected_ui_element_primary.element_id
        ),
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
    }
}

pub fn deselect_everything(loader: &mut UiButtonLoader) {
    loader.ui_runtime.selected_ui_element_primary = SelectedUiElement::default();
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
