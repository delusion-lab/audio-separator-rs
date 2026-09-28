//! 模型排名/推荐数据（独立于模型清单维护）。
//!
//! 排名数据与模型清单（[`crate::model::ModelList`]）分离存放：模型清单回答
//! 「有哪些模型、去哪下载」，排名文件回答「社区/平台推荐什么、评分如何」。
//! 两者通过 `name` 关联（社区条目沿用模型清单名称；MVSEP 条目为平台算法名，
//! 与清单名称通常不一致，独立展示）。
//!
//! 数据来源：
//! - `community`：deton24 community guide（UVR-MDX-Demucs-GSEP）维护的社区推荐排名；
//! - `mvsep`：MVSEP 平台 multisong leaderboard 快照（公开页面抓取，按需更新）。

use serde::{Deserialize, Serialize};

/// 排名文件（本地 JSON 路径或远程 URL 均可，加载机制与模型清单一致）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RankingsList {
    /// 排名文件格式版本。
    #[serde(default)]
    pub version: u32,
    /// 社区指南推荐条目（deton24 community guide）。
    #[serde(default)]
    pub community: Vec<CommunityRankEntry>,
    /// MVSEP multisong leaderboard 快照条目。
    #[serde(default)]
    pub mvsep: Vec<MvsepRankEntry>,
}

impl RankingsList {
    /// 按模型名查社区推荐条目。
    pub fn community_of(&self, name: &str) -> Option<&CommunityRankEntry> {
        self.community.iter().find(|e| e.name == name)
    }

    /// 按算法名查 MVSEP 排行条目。
    pub fn mvsep_of(&self, name: &str) -> Option<&MvsepRankEntry> {
        self.mvsep.iter().find(|e| e.name == name)
    }
}

/// 社区指南推荐条目。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommunityRankEntry {
    /// 模型名（与模型清单 `name` 一致）。
    pub name: String,
    /// 在该用途分类中的推荐序号（从 1 开始）。
    #[serde(default)]
    pub rank: Option<u32>,
    /// 用途分类，如 "2 stems > instrumentals"、"drums"、"de-reverb"。
    #[serde(default)]
    pub category: Option<String>,
    /// 社区评测指标（可选）。
    #[serde(default)]
    pub metrics: Option<CommunityMetrics>,
    /// 数据来源标识。
    #[serde(default)]
    pub source: Option<String>,
    /// 模型下载链接或文档链接。
    #[serde(default)]
    pub url: Option<String>,
}

/// 评测指标（community 与 mvsep 共用同一套 multisong 口径）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommunityMetrics {
    #[serde(default)]
    pub sdr: Option<f64>,
    #[serde(default)]
    pub fullness: Option<f64>,
    #[serde(default)]
    pub bleedless: Option<f64>,
}

/// MVSEP multisong leaderboard 条目。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MvsepRankEntry {
    /// 平台算法名（如 "BS Roformer 124 bands (2026.07.10, MVSep.com)"）。
    pub name: String,
    /// 质量检查条目 ID（`/quality_checker/entry/<id>`）。
    #[serde(default)]
    pub entry_id: Option<u64>,
    /// 质量检查结果页 URL。
    #[serde(default)]
    pub url: Option<String>,
    /// multisong 各声部 SDR（dB）。
    #[serde(default)]
    pub sdr: Option<MvsepSdr>,
    /// 出现在各排序视图中的名次（一个模型可同时在多个视图上榜）。
    #[serde(default)]
    pub views: Vec<MvsepViewRank>,
}

/// multisong 数据集各声部 SDR。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MvsepSdr {
    #[serde(default)]
    pub bass: Option<f64>,
    #[serde(default)]
    pub drums: Option<f64>,
    #[serde(default)]
    pub other: Option<f64>,
    #[serde(default)]
    pub vocals: Option<f64>,
    #[serde(default)]
    pub instrumental: Option<f64>,
}

/// 某个排序视图（instrum/vocals/bass/drums/other）中的名次。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MvsepViewRank {
    /// 视图名：instrum / vocals / bass / drums / other。
    pub view: String,
    /// 该视图中的名次（从 1 开始）。
    #[serde(default)]
    pub rank: u32,
}
