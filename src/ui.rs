pub trait UiBuilder {
    fn build(&mut self, ui: &mut imgui::Ui);
}
