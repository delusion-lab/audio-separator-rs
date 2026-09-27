//! 音频读写：symphonia 解码（多格式）、rubato 重采样、hound WAV 编码。

use std::path::Path;

use symphonia::core::audio::{SampleBuffer, SignalSpec};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::error::{Error, Result};

/// 解码结果：交织采样（f32，-1..1）+ 采样率 + 声道数（已归一化到立体声）。
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// 交织采样（L/R 交替）。
    pub samples: Vec<f32>,
    /// 采样率。
    pub sample_rate: u32,
    /// 声道数（恒为 2）。
    pub channels: u16,
}

/// 解码音频文件为立体声 f32 交织采样。
///
/// 单声道复制为双声道；超过 2 声道取前两声道。
pub fn decode(path: &Path) -> Result<DecodedAudio> {
    let file = std::fs::File::open(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("打开音频文件失败 {}: {e}", path.display()),
        ))
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| Error::Format(format!("无法解析音频 {}: {e}", path.display())))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| Error::Format(format!("音频 {} 没有可用音轨", path.display())))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| Error::Format(format!("无法创建解码器: {e}")))?;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut spec: Option<SignalSpec> = None;
    let mut out: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // 文件读完（EOF）或任何读取错误均结束；EOF 为正常路径。
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|e| Error::Format(format!("解码失败: {e}")))?;
        let s = decoded.spec();
        spec = Some(*s);
        let channels = s.channels.count();
        if sample_buf.is_none() {
            sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, *s));
        }
        if let Some(buf) = &mut sample_buf {
            buf.copy_interleaved_ref(decoded);
            out.extend_from_slice(buf.samples());
        }
        if channels != 2 {
            // 后续以立体声处理，仅在首个缓冲记录声道数
            let _ = spec;
        }
    }

    if out.is_empty() {
        return Err(Error::Format(format!("音频 {} 无解码数据", path.display())));
    }
    let channels = spec.as_ref().map(|s| s.channels.count()).unwrap_or(2) as u16;
    let sample_rate = spec.as_ref().map(|s| s.rate).unwrap_or(44100);

    Ok(DecodedAudio {
        samples: to_stereo(out, channels),
        sample_rate,
        channels: 2,
    })
}

/// 交织采样归一化为立体声：单声道复制、>2 声道取前 2。
fn to_stereo(interleaved: Vec<f32>, channels: u16) -> Vec<f32> {
    match channels {
        0 | 1 => {
            let mut out = Vec::with_capacity(interleaved.len() * 2);
            for &s in &interleaved {
                out.push(s);
                out.push(s);
            }
            out
        }
        2 => interleaved,
        n => {
            let ch = n as usize;
            let frames = interleaved.len() / ch;
            let mut out = Vec::with_capacity(frames * 2);
            for f in 0..frames {
                out.push(interleaved[f * ch]);
                out.push(interleaved[f * ch + 1]);
            }
            out
        }
    }
}

/// 若 `from != to`，用 rubato SincFixedIn 将交织采样重采样到目标采样率。
pub fn resample_to(samples: &[f32], channels: u16, from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(samples.to_vec());
    }
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    let ratio = to as f64 / from as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let ch = channels as usize;
    let frames = samples.len() / ch;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, 4096, ch)
        .map_err(|e| Error::Backend(format!("创建重采样器失败: {e}")))?;

    // 分声道输入
    let mut input: Vec<Vec<f32>> = vec![Vec::with_capacity(frames); ch];
    for f in 0..frames {
        for c in 0..ch {
            input[c].push(samples[f * ch + c]);
        }
    }
    let out_ch = resampler
        .process(&input, None)
        .map_err(|e| Error::Backend(format!("重采样失败: {e}")))?;
    if out_ch.is_empty() || out_ch[0].is_empty() {
        return Err(Error::Backend("重采样结果为空".to_string()));
    }
    let out_frames = out_ch[0].len();
    let mut out_i = Vec::with_capacity(out_frames * ch);
    for f in 0..out_frames {
        for c in 0..ch {
            out_i.push(out_ch[c][f]);
        }
    }
    Ok(out_i)
}

/// 峰值归一化（避免写出时削波）。
pub fn normalize(samples: &mut [f32]) {
    let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    if peak > 1.0 {
        let g = 1.0 / peak;
        for s in samples {
            *s *= g;
        }
    }
}

/// 交织 f32 采样写为 16-bit PCM WAV。
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| Error::Format(format!("创建 {} 失败: {e}", path.display())))?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer
            .write_sample(v)
            .map_err(|e| Error::Format(format!("写入 {} 失败: {e}", path.display())))?;
    }
    writer
        .finalize()
        .map_err(|e| Error::Format(format!("完成 {} 失败: {e}", path.display())))?;
    Ok(())
}

/// 按输出格式写音频文件（M5：WAV16/32、FLAC16/24、MP3；M4A 本地暂不支持）。
pub fn write_audio(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    format: crate::model::OutputFormat,
) -> Result<()> {
    match format {
        crate::model::OutputFormat::Wav16 => write_wav(path, samples, sample_rate, channels),
        crate::model::OutputFormat::Wav32 => write_wav_f32(path, samples, sample_rate, channels),
        crate::model::OutputFormat::Flac16 => write_flac(path, samples, sample_rate, channels, 16),
        crate::model::OutputFormat::Flac24 => write_flac(path, samples, sample_rate, channels, 24),
        crate::model::OutputFormat::Mp3 => write_mp3(path, samples, sample_rate, channels),
        crate::model::OutputFormat::M4a => Err(Error::Backend(
            "本地后端暂不支持 M4A 输出（请使用 MVSEP 后端，或改选 wav/flac/mp3）".to_string(),
        )),
    }
}

/// 交织 f32 采样写为 32-bit float WAV。
fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    use hound::{SampleFormat, WavSpec, WavWriter};
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)
        .map_err(|e| Error::Format(format!("创建 {} 失败: {e}", path.display())))?;
    for &s in samples {
        writer
            .write_sample(s.clamp(-1.0, 1.0))
            .map_err(|e| Error::Format(format!("写入 {} 失败: {e}", path.display())))?;
    }
    writer
        .finalize()
        .map_err(|e| Error::Format(format!("完成 {} 失败: {e}", path.display())))?;
    Ok(())
}

/// 交织 f32 采样写为 FLAC（16/24-bit，纯 Rust flacenc 编码器）。
fn write_flac(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    bits: usize,
) -> Result<()> {
    use flacenc::bitsink::ByteSink;
    use flacenc::component::BitRepr;
    use flacenc::config::Encoder as FlacConfig;
    use flacenc::error::Verify;
    use flacenc::source::MemSource;

    // f32 → 目标位深整数（24-bit 时映射到 24 位有效范围）
    let scale: f32 = if bits == 24 {
        (1i64 << 23) as f32
    } else {
        i16::MAX as f32
    };
    let i32_samples: Vec<i32> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * scale) as i32)
        .collect();

    let config = FlacConfig::default()
        .into_verified()
        .map_err(|e| Error::Format(format!("FLAC 配置无效: {e:?}")))?;
    let source = MemSource::from_samples(
        &i32_samples,
        channels as usize,
        bits,
        sample_rate as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| Error::Format(format!("FLAC 编码失败: {e:?}")))?;
    let mut sink = ByteSink::with_capacity(stream.count_bits());
    stream
        .write(&mut sink)
        .map_err(|e| Error::Format(format!("FLAC 位流写入失败: {e:?}")))?;
    std::fs::write(path, sink.into_inner())
        .map_err(|e| Error::Io(std::io::Error::new(e.kind(), format!("写入 {} 失败: {e}", path.display()))))?;
    Ok(())
}

/// 交织 f32 采样写为 MP3（320kbps CBR，纯 Rust rusty_mp3 编码器）。
fn write_mp3(path: &Path, samples: &[f32], sample_rate: u32, channels: u16) -> Result<()> {
    use rusty_mp3::error::Error as Mp3Error;
    use rusty_mp3::{Mp3Encoder, Mp3EncoderConfig};

    let config = Mp3EncoderConfig {
        bitrate_kbps: 320,
        vbr_quality: None,
    };
    let mut encoder = Mp3Encoder::new(config);
    encoder
        .push_pcm_f32(samples, channels, sample_rate)
        .map_err(|e| Error::Format(format!("MP3 编码输入失败: {e:?}")))?;
    encoder.finish();

    let mut bytes: Vec<u8> = Vec::new();
    loop {
        match encoder.next_packet() {
            Ok(p) => bytes.extend_from_slice(&p),
            Err(Mp3Error::Eof) => break,
            Err(Mp3Error::Again) => continue,
            Err(e) => return Err(Error::Format(format!("MP3 编码失败: {e:?}"))),
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| Error::Io(std::io::Error::new(e.kind(), format!("写入 {} 失败: {e}", path.display()))))?;
    Ok(())
}
