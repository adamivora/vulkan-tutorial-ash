use crate::frame_data::FrameData;

pub trait UiBuilder {
    fn build(&mut self, ui: &mut imgui::Ui);
    fn frame_data(&self) -> FrameData;
}
