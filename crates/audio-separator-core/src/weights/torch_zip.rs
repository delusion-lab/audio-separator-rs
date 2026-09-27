//! torch zip 存档容器读取：定位 `data.pkl`、`byteorder` 与 `data/<n>` 分片。

use std::cell::RefCell;
use std::io::Read;
use std::path::Path;

use crate::error::{Error, Result};
use crate::weights::pickle::{parse, Value};

/// 解析后的 torch 存档。
pub struct TorchArchive {
    /// (tensor 名, TensorRec)。
    pub tensors: Vec<(String, crate::weights::pickle::TensorRec)>,
    pub byteorder: String,
    /// 按 storage id 读取分片字节（little/big endian 原样返回）。
    read_storage: RefCell<Box<dyn FnMut(&str) -> Result<Vec<u8>>>>,
}

impl TorchArchive {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let zip = zip::ZipArchive::new(file)
            .map_err(|e| Error::Model(format!("ckpt is not a valid zip: {e}")))?;

        // 定位 data.pkl（目录前缀如 last_bs_roformer/、archive/）
        let names: Vec<String> = zip.file_names().map(|s| s.to_string()).collect();
        let pkl_name = names
            .iter()
            .find(|n| n.ends_with("data.pkl"))
            .cloned()
            .ok_or_else(|| Error::Model("archive missing data.pkl".to_string()))?;
        let prefix = pkl_name.trim_end_matches("data.pkl").to_string();

        let zip = RefCell::new(zip);
        let byteorder = {
            let mut z = zip.borrow_mut();
            match z.by_name(&format!("{prefix}byteorder")) {
                Ok(mut f) => {
                    let mut s = String::new();
                    f.read_to_string(&mut s)?;
                    s
                }
                Err(_) => "little".to_string(),
            }
        };

        let mut pkl = Vec::new();
        {
            let mut z = zip.borrow_mut();
            z.by_name(&pkl_name)
                .map_err(|e| Error::Model(format!("failed to read {pkl_name}: {e}")))?
                .read_to_end(&mut pkl)?;
        }
        let top = parse(&pkl)?;
        let dict = match top {
            Value::Dict(items) => items,
            _ => return Err(Error::Model("top level is not an OrderedDict".to_string())),
        };
        let mut tensors = Vec::new();
        for (k, v) in dict {
            if let Value::Tensor(t) = v {
                let name = match &k {
                    Value::Str(s) => s.clone(),
                    Value::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
                    _ => continue,
                };
                tensors.push((name, t));
            }
        }

        let read_storage = RefCell::new(Box::new(move |id: &str| -> Result<Vec<u8>> {
            let name = format!("{prefix}data/{id}");
            let mut z = zip.borrow_mut();
            let mut out = Vec::new();
            z.by_name(&name)
                .map_err(|e| Error::Model(format!("failed to read weight shard {name}: {e}")))?
                .read_to_end(&mut out)?;
            Ok(out)
        }) as Box<dyn FnMut(&str) -> Result<Vec<u8>>>);

        Ok(Self {
            tensors,
            byteorder,
            read_storage,
        })
    }

    /// 读取某个 tensor 的 storage 原始字节。
    pub fn read_storage(&self, id: &str) -> Result<Vec<u8>> {
        let mut f = self.read_storage.borrow_mut();
        f(id)
    }
}
