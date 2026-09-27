//! 受限 pickle 虚拟机：安全解析 PyTorch 权重存档的 `data.pkl`。
//!
//! 只实现协议 2 的子集 + torch 权重序列化的白名单 GLOBAL（`collections.OrderedDict`、
//! `torch._utils._rebuild_tensor_v2`、各类 `torch.*Storage`），遇到白名单之外的
//! 全局对象立即报错——不执行任意 Python 代码（规避 CVE-2025-49839 一类 pickle 风险）。
//!
//! opcode 字节值以 CPython `Lib/pickle.py` 权威表为准：
//! 协议 2 中 GLOBAL=`'c'`(0x63)、REDUCE=`'R'`(0x52)、BINUNICODE=`'X'`(0x58)、
//! TUPLE=`'t'`(0x74)、TUPLE1/2/3=`\x85\x86\x87`、BINPUT=`'q'`(0x71) 等。

use crate::error::{Error, Result};

/// 张量数据类型（torch storage 类型）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    F64,
    I32,
    I64,
    U8,
    Bool,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::U8 | Dtype::Bool => 1,
        }
    }
}

/// torch storage 记录（BINPERSID 持久对象）。
#[derive(Debug, Clone)]
pub struct Storage {
    /// zip 内分片名（如 `data/0` 的 `0`）。
    pub id: String,
    pub device: String,
    pub numel: u64,
    pub dtype: Dtype,
}

/// 重建的 tensor 记录（`_rebuild_tensor_v2`）。
#[derive(Debug, Clone)]
pub struct TensorRec {
    pub storage: Option<Storage>,
    pub offset: usize,
    pub size: Vec<u64>,
    pub stride: Vec<u64>,
}

/// pickle 值模型（受限）。
#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// 白名单类实例（OrderedDict 类、storage 类等）。
    Class(String),
    Storage(Storage),
    Tensor(TensorRec),
}

impl Value {
    fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
}

/// 解析协议 2 pickle 字节流，返回顶层值。
pub fn parse(data: &[u8]) -> Result<Value> {
    parse_with_trace(data, false)
}

/// 调试入口：trace=true 时打印每个 opcode 执行轨迹。
pub fn parse_with_trace(data: &[u8], trace: bool) -> Result<Value> {
    let mut vm = Vm {
        data,
        pos: 0,
        stack: Vec::new(),
        memo: Vec::new(),
        last_op: String::new(),
        trace,
    };
    vm.run()
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: Vec<Value>,
    last_op: String,
    trace: bool,
}

macro_rules! bail {
    ($($t:tt)*) => { return Err(Error::Model(format!("pickle parse failed: {}", format!($($t)*)))) };
}

impl<'a> Vm<'a> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| Error::Model("pickle data truncated".to_string()))?;
        self.pos += 1;
        Ok(b)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self
            .data
            .get(self.pos..self.pos + n)
            .ok_or_else(|| Error::Model("pickle data truncated".to_string()))?;
        self.pos += n;
        Ok(s)
    }
    fn u8_le(&mut self) -> Result<u8> {
        Ok(self.byte()?)
    }
    fn u16_le(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32_le(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64_le(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    #[allow(dead_code)] // M2-D bs_polarformer 权重（含 LONG tensor）转换时使用
    fn i64_le(&mut self) -> Result<i64> {
        Ok(self.u64_le()? as i64)
    }
    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().ok_or_else(|| {
            Error::Model(format!(
                "pickle stack underflow (opcode: {}) @pos={}",
                self.last_op, self.pos
            ))
        })
    }
    fn push(&mut self, v: Value) {
        self.stack.push(v);
    }

    fn run(&mut self) -> Result<Value> {
        let mut marks: Vec<usize> = Vec::new();
        loop {
            let op = self.byte()?;
            let op_name = opcode_name(op);
            self.last_op = op_name.to_string();
            if self.trace {
                eprintln!(
                    "pickle@{:6} {:<22} stack={:5} marks={}",
                    self.pos - 1,
                    op_name,
                    self.stack.len(),
                    marks.len()
                );
            }
            match op {
                // ---------- 基础（协议 1/2 通用 ASCII opcode） ----------
                b'.' => {
                    // STOP
                    if self.stack.len() != 1 {
                        bail!("STOP with non-single-element stack: {}", self.stack.len());
                    }
                    return Ok(self.pop()?);
                }
                b'0' => {
                    // POP
                    self.pop()?;
                }
                b'1' => {
                    // POP_MARK：丢弃到最近 MARK
                    let m = marks.pop().ok_or_else(|| Error::Model("POP_MARK missing MARK".into()))?;
                    self.stack.truncate(m);
                }
                b'2' => {
                    // DUP
                    let v = self.pop()?;
                    self.push(v.clone());
                    self.push(v);
                }
                b'F' => {
                    let line = self.read_ascii_line()?;
                    let f: f64 = line
                        .parse()
                        .map_err(|_| Error::Model("FLOAT parse failed".into()))?;
                    self.push(Value::Float(f));
                }
                b'I' => {
                    let line = self.read_ascii_line()?;
                    let i: i64 = line.trim_end_matches('L').parse().map_err(|_| {
                        Error::Model("INT parse failed".to_string())
                    })?;
                    self.push(Value::Int(i));
                }
                b'J' => {
                    // BININT：4 字节有符号
                    let v = self.u32_le()? as i32 as i64;
                    self.push(Value::Int(v));
                }
                b'K' => {
                    let v = self.u8_le()?;
                    self.push(Value::Int(v as i64));
                }
                b'L' => {
                    let line = self.read_ascii_line()?;
                    let i: i64 = line.trim_end_matches('L').parse().map_err(|_| {
                        Error::Model("LONG parse failed".to_string())
                    })?;
                    self.push(Value::Int(i));
                }
                b'M' => {
                    let v = self.u16_le()?;
                    self.push(Value::Int(v as i64));
                }
                b'N' => self.push(Value::None),
                b'P' => {
                    // PERSID（ASCII 持久 id）——torch 存档一般不用，安全拒绝
                    bail!("PERSID not supported");
                }
                b'Q' => {
                    // BINPERSID：torch 2.x persistent id 为 5 元组
                    // ('storage', storage_type_class, root_key, location, numel)
                    let idv = self.pop()?;
                    let (id, device, numel, dtype) = match &idv {
                        Value::Tuple(items) if items.len() == 5 => {
                            let dtype = match &items[1] {
                                Value::Class(c) => dtype_from_storage_class(c),
                                _ => Dtype::F32,
                            };
                            let id = items[2]
                                .as_str()
                                .ok_or_else(|| Error::Model("persistent root_key not a string".into()))?
                                .to_string();
                            let device = items[3]
                                .as_str()
                                .ok_or_else(|| Error::Model("persistent location not a string".into()))?
                                .to_string();
                            let numel = items[4]
                                .as_int()
                                .ok_or_else(|| Error::Model("persistent numel not an integer".into()))?;
                            if numel < 0 {
                                bail!("persistent numel is negative");
                            }
                            (id, device, numel as u64, dtype)
                        }
                        // 旧版 torch：3 元组 (root_key, location, numel)
                        Value::Tuple(items) if items.len() == 3 => {
                            let id = items[0]
                                .as_str()
                                .ok_or_else(|| Error::Model("persistent id not a string".into()))?
                                .to_string();
                            let device = items[1]
                                .as_str()
                                .ok_or_else(|| Error::Model("persistent device not a string".into()))?
                                .to_string();
                            let numel = items[2]
                                .as_int()
                                .ok_or_else(|| Error::Model("persistent numel not an integer".into()))?;
                            let dtype = self
                                .stack
                                .iter()
                                .rev()
                                .find_map(|v| match v {
                                    Value::Class(c) if c.starts_with("torch ") => {
                                        Some(dtype_from_storage_class(c))
                                    }
                                    _ => None,
                                })
                                .unwrap_or(Dtype::F32);
                            (id, device, numel as u64, dtype)
                        }
                        Value::Str(id) => (id.clone(), "cpu".to_string(), 0, Dtype::F32),
                        other => bail!("BINPERSID shape mismatch: {other:?}"),
                    };
                    self.push(Value::Storage(Storage {
                        id,
                        device,
                        numel,
                        dtype,
                    }));
                }
                b'R' => {
                    // REDUCE
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let result = self.reduce(&callable, &args)?;
                    self.push(result);
                }
                b'S' => {
                    let s = self.read_ascii_line()?;
                    self.push(Value::Bytes(s.into_bytes()));
                }
                b'T' => {
                    let n = self.u32_le()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.push(Value::Bytes(b));
                }
                b'U' => {
                    let n = self.u8_le()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.push(Value::Bytes(b));
                }
                b'V' => {
                    let s = self.read_ascii_line()?;
                    self.push(Value::Str(s));
                }
                b'X' => {
                    let n = self.u32_le()? as usize;
                    let b = self.take(n)?;
                    let s = std::str::from_utf8(b)
                        .map_err(|_| Error::Model("BINUNICODE not UTF-8".into()))?;
                    self.push(Value::Str(s.to_string()));
                }
                b'a' => {
                    // APPEND
                    let v = self.pop()?;
                    let mut list = self.pop()?;
                    match &mut list {
                        Value::List(l) => l.push(v),
                        _ => bail!("APPEND target is not a list"),
                    }
                    self.push(list);
                }
                b'b' => {
                    // BUILD
                    let state = self.pop()?;
                    let mut obj = self.pop()?;
                    match (&mut obj, state) {
                        (Value::Dict(d), Value::Dict(s)) => d.extend(s),
                        (Value::Class(_), Value::Dict(_)) => {
                            // OrderedDict() 后 BUILD state：忽略（torch 用 REDUCE 构建）
                        }
                        (obj_v, Value::None) => {
                            let _ = obj_v;
                        }
                        _ => bail!("BUILD type mismatch"),
                    }
                    self.push(obj);
                }
                b'c' => {
                    // GLOBAL：module\nname
                    let module = self.read_ascii_line()?;
                    let name = self.read_ascii_line()?;
                    let global = format!("{module} {name}");
                    self.push(Value::Class(global));
                }
                b'd' => {
                    // DICT：MARK 间键值对建 dict
                    let m = marks.pop().ok_or_else(|| Error::Model("DICT missing MARK".into()))?;
                    let items: Vec<Value> = self.stack.split_off(m);
                    let mut d = Vec::new();
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                    self.push(Value::Dict(d));
                }
                b'e' => {
                    // APPENDS
                    let m = marks.pop().ok_or_else(|| Error::Model("APPENDS missing MARK".into()))?;
                    let items: Vec<Value> = self.stack.split_off(m);
                    let mut list = self.pop()?;
                    match &mut list {
                        Value::List(l) => l.extend(items),
                        _ => bail!("APPENDS target is not a list"),
                    }
                    self.push(list);
                }
                b'g' => {
                    // GET（ASCII 索引）
                    let line = self.read_ascii_line()?;
                    let i: usize = line
                        .parse()
                        .map_err(|_| Error::Model("GET index parse failed".into()))?;
                    let v = self
                        .memo
                        .get(i)
                        .cloned()
                        .ok_or_else(|| Error::Model(format!("GET miss in memo[{i}]")))?;
                    self.push(v);
                }
                b'h' => {
                    // BINGET
                    let i = self.u8_le()? as usize;
                    let v = self
                        .memo
                        .get(i)
                        .cloned()
                        .ok_or_else(|| Error::Model(format!("BINGET miss in memo[{i}]")))?;
                    self.push(v);
                }
                b'i' => {
                    bail!("INST not supported (instantiation outside whitelist)");
                }
                b'j' => {
                    // LONG_BINGET
                    let i = self.u32_le()? as usize;
                    let v = self
                        .memo
                        .get(i)
                        .cloned()
                        .ok_or_else(|| Error::Model(format!("LONG_BINGET miss in memo[{i}]")))?;
                    self.push(v);
                }
                b'l' => {
                    // LIST
                    let m = marks.pop().ok_or_else(|| Error::Model("LIST missing MARK".into()))?;
                    let items: Vec<Value> = self.stack.split_off(m);
                    self.push(Value::List(items));
                }
                b'o' => {
                    bail!("OBJ not supported (instantiation outside whitelist)");
                }
                b'p' => {
                    // PUT（ASCII 索引）
                    let line = self.read_ascii_line()?;
                    let i: usize = line
                        .parse()
                        .map_err(|_| Error::Model("PUT index parse failed".into()))?;
                    let v = self.pop()?;
                    if self.memo.len() <= i {
                        self.memo.resize(i + 1, Value::None);
                    }
                    self.memo[i] = v.clone();
                    self.push(v);
                }
                b'q' => {
                    // BINPUT
                    let i = self.u8_le()? as usize;
                    let v = self.pop()?;
                    if self.memo.len() <= i {
                        self.memo.resize(i + 1, Value::None);
                    }
                    self.memo[i] = v.clone();
                    self.push(v);
                }
                b'r' => {
                    // LONG_BINPUT
                    let i = self.u32_le()? as usize;
                    let v = self.pop()?;
                    if self.memo.len() <= i {
                        self.memo.resize(i + 1, Value::None);
                    }
                    self.memo[i] = v.clone();
                    self.push(v);
                }
                b's' => {
                    // SETITEM
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let mut d = self.pop()?;
                    match &mut d {
                        Value::Dict(dd) => dd.push((k, v)),
                        _ => bail!("SETITEM target is not a dict"),
                    }
                    self.push(d);
                }
                b't' => {
                    // TUPLE
                    let m = marks.pop().ok_or_else(|| Error::Model("TUPLE missing MARK".into()))?;
                    let items: Vec<Value> = self.stack.split_off(m);
                    self.push(Value::Tuple(items));
                }
                b'u' => {
                    // SETITEMS
                    let m = marks.pop().ok_or_else(|| Error::Model("SETITEMS missing MARK".into()))?;
                    let items: Vec<Value> = self.stack.split_off(m);
                    let mut d = self.pop()?;
                    match &mut d {
                        Value::Dict(dd) => {
                            let mut it = items.into_iter();
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                dd.push((k, v));
                            }
                        }
                        _ => bail!("SETITEMS target is not a dict"),
                    }
                    self.push(d);
                }
                b'B' => {
                    let n = self.u32_le()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.push(Value::Bytes(b));
                }
                b'C' => {
                    let n = self.u8_le()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.push(Value::Bytes(b));
                }
                b'}' => self.push(Value::Dict(Vec::new())), // EMPTY_DICT
                b']' => self.push(Value::List(Vec::new())),  // EMPTY_LIST
                b')' => self.push(Value::Tuple(Vec::new())), // EMPTY_TUPLE
                // 协议 2
                0x80 => {
                    let v = self.u8_le()?;
                    if v > 4 {
                        bail!("unsupported pickle protocol version {v}");
                    }
                }
                b'(' => marks.push(self.stack.len()), // MARK（协议 1/2 通用）
                0x81 => {
                    // NEWOBJ
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let result = self.newobj(&cls, &args)?;
                    self.push(result);
                }
                0x85 => {
                    let v = self.pop()?;
                    self.push(Value::Tuple(vec![v]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Tuple(vec![a, b, c]));
                }
                0x88 => self.push(Value::Bool(true)),
                0x89 => self.push(Value::Bool(false)),
                0x8A => {
                    // LONG1：1 字节长度的小端大整数
                    let n = self.u8_le()? as usize;
                    let b = self.take(n)?;
                    self.push(Value::Int(le_unsigned(b)));
                }
                0x8B => {
                    // LONG4
                    let n = self.u32_le()? as usize;
                    let b = self.take(n)?;
                    self.push(Value::Int(le_unsigned(b)));
                }
                0x94 => {
                    // MEMOIZE
                    let v = self.pop()?;
                    self.memo.push(v.clone());
                    self.push(v);
                }
                // ---------- 协议 4（只接受 FRAME 跳过；其余安全拒绝） ----------
                0x95 => {
                    let _len = self.u64_le()?;
                }
                other => bail!("unsupported pickle opcode {op_name}(0x{other:02x})"),
            }
        }
    }

    fn read_ascii_line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(Error::Model("unterminated ASCII line".to_string()));
        }
        let s = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| Error::Model("ASCII line is not ASCII".to_string()))?
            .to_string();
        self.pos += 1; // 跳过 \n
        Ok(s)
    }

    fn reduce(&mut self, callable: &Value, args: &Value) -> Result<Value> {
        let name = match callable {
            Value::Class(c) => c.clone(),
            _ => "?".to_string(),
        };
        let args_vec = match args {
            Value::Tuple(t) | Value::List(t) => t.clone(),
            _ => bail!("REDUCE argument is not a tuple"),
        };
        match name.as_str() {
            "collections OrderedDict" => {
                let mut items = Vec::new();
                let mut it = args_vec.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    items.push((k, v));
                }
                Ok(Value::Dict(items))
            }
            "torch._utils _rebuild_tensor_v2"
            | "torch._utils _rebuild_parameter_v2"
            | "torch._utils _rebuild_parameter" => {
                // 宽松提取：args 中第一个 Storage、第一个 int（offset）、
                // 第一/第二个 tuple<int>（size/stride）
                let mut storage = None;
                let mut ints = Vec::new();
                let mut tuples = Vec::new();
                for a in &args_vec {
                    match a {
                        Value::Storage(s) => storage = Some(s.clone()),
                        Value::Int(i) => ints.push(*i),
                        Value::Tuple(t) | Value::List(t)
                            if !t.is_empty() && t.iter().all(|v| v.as_int().is_some()) =>
                        {
                            tuples.push(t.iter().map(|v| v.as_int().unwrap()).collect::<Vec<i64>>())
                        }
                        _ => {}
                    }
                }
                if tuples.len() < 2 {
                    bail!("_rebuild_tensor_v2 missing size/stride");
                }
                let size = tuples[0].iter().map(|&x| x as u64).collect::<Vec<_>>();
                let stride = tuples[1].iter().map(|&x| x as u64).collect::<Vec<_>>();
                let offset = ints.first().copied().unwrap_or(0).max(0) as usize;
                Ok(Value::Tensor(TensorRec {
                    storage,
                    offset,
                    size,
                    stride,
                }))
            }
            "torch FloatStorage"
            | "torch HalfStorage"
            | "torch BFloat16Storage"
            | "torch DoubleStorage"
            | "torch IntStorage"
            | "torch LongStorage"
            | "torch ByteStorage"
            | "torch CharStorage"
            | "torch BoolStorage" => {
                // 旧协议 storage 直接 REDUCE：(id, device, numel)
                let mut id = String::new();
                let mut device = "cpu".to_string();
                let mut numel = 0u64;
                for a in &args_vec {
                    if let Some(s) = a.as_str() {
                        if id.is_empty() {
                            id = s.to_string();
                        } else {
                            device = s.to_string();
                        }
                    } else if let Some(i) = a.as_int() {
                        numel = i.max(0) as u64;
                    }
                }
                let dtype = dtype_from_storage_class(&name);
                Ok(Value::Storage(Storage {
                    id,
                    device,
                    numel,
                    dtype,
                }))
            }
            "builtins dict" => Ok(Value::Dict(Vec::new())),
            "builtins list" => Ok(Value::List(Vec::new())),
            "builtins set" => Ok(Value::Tuple(Vec::new())),
            "builtins frozenset" => Ok(Value::Tuple(Vec::new())),
            _ => bail!("GLOBAL outside whitelist: {name} (callable={callable:?})"),
        }
    }

    fn newobj(&mut self, cls: &Value, args: &Value) -> Result<Value> {
        let name = match cls {
            Value::Class(c) => c.clone(),
            _ => "?".to_string(),
        };
        match name.as_str() {
            "collections OrderedDict" => self.reduce(cls, args),
            "builtins dict" => self.reduce(cls, args),
            "builtins list" => self.reduce(cls, args),
            _ => bail!("NEWOBJ outside whitelist: {name}"),
        }
    }
}

fn le_unsigned(b: &[u8]) -> i64 {
    let mut v: i64 = 0;
    for (i, byte) in b.iter().enumerate().take(8) {
        v |= (*byte as i64) << (8 * i);
    }
    v
}

fn dtype_from_storage_class(global: &str) -> Dtype {
    match global {
        "torch FloatStorage" | "torch.FloatStorage" => Dtype::F32,
        "torch HalfStorage" | "torch.HalfStorage" => Dtype::F16,
        "torch BFloat16Storage" | "torch.BFloat16Storage" => Dtype::Bf16,
        "torch DoubleStorage" | "torch.DoubleStorage" => Dtype::F64,
        "torch IntStorage" | "torch.IntStorage" => Dtype::I32,
        "torch LongStorage" | "torch.LongStorage" => Dtype::I64,
        "torch ByteStorage" | "torch.ByteStorage" | "torch CharStorage" | "torch.CharStorage" => {
            Dtype::U8
        }
        "torch BoolStorage" | "torch.BoolStorage" => Dtype::Bool,
        _ => Dtype::F32,
    }
}

fn opcode_name(op: u8) -> &'static str {
    match op {
        b'.' => "STOP",
        b'0' => "POP",
        b'1' => "POP_MARK",
        b'2' => "DUP",
        b'F' => "FLOAT",
        b'I' => "INT",
        b'J' => "BININT",
        b'K' => "BININT1",
        b'L' => "LONG",
        b'M' => "BININT2",
        b'N' => "NONE",
        b'P' => "PERSID",
        b'Q' => "BINPERSID",
        b'R' => "REDUCE",
        b'S' => "STRING",
        b'T' => "BINSTRING",
        b'U' => "SHORT_BINSTRING",
        b'V' => "UNICODE",
        b'X' => "BINUNICODE",
        b'a' => "APPEND",
        b'b' => "BUILD",
        b'c' => "GLOBAL",
        b'd' => "DICT",
        b'e' => "APPENDS",
        b'g' => "GET",
        b'h' => "BINGET",
        b'i' => "INST",
        b'j' => "LONG_BINGET",
        b'l' => "LIST",
        b'o' => "OBJ",
        b'p' => "PUT",
        b'q' => "BINPUT",
        b'r' => "LONG_BINPUT",
        b's' => "SETITEM",
        b't' => "TUPLE",
        b'u' => "SETITEMS",
        b'B' => "BINBYTES",
        b'C' => "SHORT_BINBYTES",
        b'}' => "EMPTY_DICT",
        b']' => "EMPTY_LIST",
        b')' => "EMPTY_TUPLE",
        b'(' => "MARK",
        0x80 => "PROTO",
        0x81 => "NEWOBJ",
        0x85 => "TUPLE1",
        0x86 => "TUPLE2",
        0x87 => "TUPLE3",
        0x88 => "NEWTRUE",
        0x89 => "NEWFALSE",
        0x8A => "LONG1",
        0x8B => "LONG4",
        0x8F => "EMPTY_SET",
        0x90 => "FROZENSET",
        0x91 => "ADDITEMS",
        0x93 => "NEXT_BUFFER",
        0x94 => "MEMOIZE",
        0x95 => "FRAME",
        _ => "UNKNOWN",
    }
}
