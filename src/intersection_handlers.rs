use crate::low_level::IntersectionHandler;

pub struct DefaultIntersectionHandler;

unsafe impl IntersectionHandler for DefaultIntersectionHandler {
    fn source(&self) -> String {
        include_str!("default_intersection_handler.wesl").to_string()
    }
}
