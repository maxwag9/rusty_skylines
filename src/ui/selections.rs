use crate::ui::ui_editor::{ElementKind, SelectedUiElement, UiButtonLoader};

pub fn select_ui_element(
    loader: &mut UiButtonLoader,
    menu_name: String,
    layer_name: String,
    element_id: String,
    dragging: bool,
    element_kind: ElementKind,
    action_name: String,
) {
    loader.ui_runtime.selected_ui_element_multi.clear();
    loader.ui_runtime.selected_ui_element_primary = SelectedUiElement {
        menu_name,
        layer_name,
        element_id,
        active: true,
        just_deselected: if element_kind == ElementKind::None {
            true
        } else {
            false
        },
        dragging,
        element_type: element_kind,
        just_selected: if element_kind == ElementKind::None {
            false
        } else {
            true
        },
        action_name,
    };
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

pub fn select_to_multi(
    loader: &mut UiButtonLoader,
    menu_name: String,
    layer_name: String,
    element_id: String,
    dragging: bool,
    element_kind: ElementKind,
    action_name: String,
) {
    loader
        .ui_runtime
        .selected_ui_element_multi
        .push(SelectedUiElement {
            menu_name,
            layer_name,
            element_id,
            active: true,
            just_deselected: if element_kind == ElementKind::None {
                true
            } else {
                false
            },
            dragging,
            element_type: element_kind,
            just_selected: if element_kind == ElementKind::None {
                false
            } else {
                true
            },
            action_name,
        });
}

pub fn select_move_primary_to_multi(
    loader: &mut UiButtonLoader,
    menu_name: String,
    layer_name: String,
    element_id: String,
    dragging: bool,
    element_kind: ElementKind,
    action_name: String,
) {
    loader
        .ui_runtime
        .selected_ui_element_multi
        .push(loader.ui_runtime.selected_ui_element_primary.clone());

    loader.ui_runtime.selected_ui_element_primary = SelectedUiElement {
        menu_name,
        layer_name,
        element_id,
        active: true,
        just_deselected: if element_kind == ElementKind::None {
            true
        } else {
            false
        },
        dragging,
        element_type: element_kind,
        just_selected: if element_kind == ElementKind::None {
            false
        } else {
            true
        },
        action_name,
    };
    println!(
        "Selected multi: {:?}, Selected primary: {:?}",
        loader.ui_runtime.selected_ui_element_multi[0].element_id,
        loader.ui_runtime.selected_ui_element_primary.element_id
    );
    loader.update_selection()
}

pub fn deselect_everything(loader: &mut UiButtonLoader) {
    loader.ui_runtime.selected_ui_element_primary = SelectedUiElement::default();
    loader.ui_runtime.editing_text = false;
    println!("deselection");
    loader.ui_runtime.selected_ui_element_multi.clear();
}
