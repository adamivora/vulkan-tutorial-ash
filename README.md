
The [Vulkan tutorial](https://vulkan-tutorial.com/) reimplemented in Rust, with some extras and an opinionated choice of underlying crates. Tested on arm64 macOS and Asahi Linux.

## Extras
- All allocations done using [vk-mem](https://github.com/gwihlidal/vk-mem-rs).
- A Rust reimplementation of the imgui CameraWidget from [nvpro_core](https://github.com/nvpro-samples/nvpro_core), includes keyboard + mouse camera control. 
- A minimal .obj loader is included.

## Crates
- [ash](https://github.com/ash-rs/ash)
- [winit](https://github.com/rust-windowing/winit)
- [imgui](https://github.com/imgui-rs/)
- [glam](https://github.com/bitshifter/glam-rs)

---

<img width="1024" alt="Screenshot 2024-12-26 at 11 25 58" src="https://github.com/user-attachments/assets/00c9ddc9-134e-4b76-9242-5aa0d4082b0c" />
