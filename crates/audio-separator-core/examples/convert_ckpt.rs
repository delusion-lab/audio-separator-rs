//! 一次性转换工具：`cargo run -p audio-separator-core --example convert_ckpt -- <ckpt> <out.safetensors>`
//! 用于验证与预转换 torch `.ckpt` / `.pt` 权重为 safetensors。

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("用法: convert_ckpt <ckpt 路径> <输出 safetensors 路径>");
        std::process::exit(2);
    }
    match audio_separator_core::weights::convert_ckpt_to_safetensors(
        Path::new(&args[1]),
        Path::new(&args[2]),
    ) {
        Ok(()) => println!("转换成功: {} -> {}", args[1], args[2]),
        Err(e) => {
            eprintln!("转换失败: {e}");
            std::process::exit(1);
        }
    }
}
