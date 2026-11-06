# P2I-GAN 重構工作計畫

## 原則

所有工作只在本機進行：不 push、不 fetch、不動任何 remote，tag 也只打在 local。重構後的程式之後會搬到另一個 repo，這個 repo 的遠端不再使用。

工作在 local 分支 `refactor` 上進行，`main` 保持原狀。每個工作包至少一個獨立 commit，測試與文件隨該包一起改。

舊開發者目錄 `/home/NAS/homes/brick-10015` 只讀。新資料來源 `isaac-10009` 也只讀且不複製。

## 新資料與約束

新 target 是 UKMO NIMROD 雷達合成降雨率，路徑 `/home/NAS/house/isaac-10009/Data/RPP/NIMROD/NIMROD_ZARR`。一個月一個 zarr v2 store，`201001.zarr` 到 `202601.zarr` 共 193 個，每月約 1 GB。這個路徑在 D 包完成前只記在這裡，D 之後寫進 config。

| 項目 | 值 |
| --- | --- |
| 變數 | `rainrate`，維度 (time, Northing, Easting) |
| 型態與單位 | int16，`mm/h*32`；-1 = 無資料 |
| 網格 | 2175 × 1725，1 km，BNG 座標 |
| 時間 | 5 分鐘；202007 月 8928 幀無缺 |
| chunk | (12, 2175, 1725)，一小時全圖 |

三個約束決定資料管線的形狀：

1. 約 69% 的網格是無資料，且幾乎固定。128×128 全有效的裁切位置約 76 萬個，1024×1024 則一個都沒有。
2. chunk 是整小時全圖，直接訓練太慢，必須先裁切並重新 chunk。
3. 雨量站位置固定，且 `UPPos` 學的是位置，所以訓練要用固定地理區域，不能隨機裁切。

Input 是多個固定雨量站的時間序列，實際資料尚未取得。缺測等拿到真實資料再處理。

## 工作包

共 13 包，依字母順序進行；前置欄是必須先完成的包。規模 S 約半天到一天，M 數天，L 含長時間訓練。

| 包 | 內容 | 前置 | 規模 | 狀態 |
| --- | --- | --- | --- | --- |
| A 基線凍結 | local tag；複製 brick 依賴到 `datasets/`、`weights/`；路徑改為 repo 相對；`scripts/tools` 收進 repo；核對 MLflow 紀錄的訓練參數與現行 config；黃金測試 | 無 | S | Done |
| B 清理 | 刪死碼與 mp4 路線、未用依賴、`.codex`；`.gitignore`；README 先刪錯誤內容 | A | S | Done |
| C 升級 | Python 3.12 與套件、lock、pytest、ruff；黃金測試改容差比對 | B | M | Not started |
| D Config | pydantic schema、`extra="forbid"`、全部 config 重寫；寫入新資料路徑 | C | M | Not started |
| E Trainer | 修 `train.py:217`、`val_every_epochs`、resume、LR 排程；實驗紀錄改用 W&B（抽出 logger 介面） | D | S | Not started |
| F 編碼與指標 | 單一 `RainCodec` 存進 checkpoint；指標重寫並接上驗證；刪三份舊 transform | D | M | Not started |
| G 模型泛化 | T 參數化、H/W 檢查、合併 wopos、DK/STDK 的 k 由遮罩推導 | D | M | Not started |
| H1 資料探勘 | 各年缺幀、無資料區變動、雷達網年代差異、雨量分布 | C | S | Not started |
| H2 前處理 | 裁固定區域、重新 chunk、事件篩選、年份切分索引、無資料遮罩 | H1、F、區域決定 | M | Not started |
| H3 Dataset | 單一 Zarr Dataset、單位與尺寸檢查、地理固定遮罩 | H2、D | M | Not started |
| I 雨量站介面 | 站點表轉遮罩；推論端從時序組輸入張量；單位驗證 | H3、F | M | Not started |
| J 新資料訓練 | overfit 閘門 → 小規模 → 全量；與 DK/STDK 比較 | E、F、G、H3 | L | Not started |
| K 文件收尾 | 架構、資料格式、限制與權重適用範圍 | J | S | Not started |

## 相依與順序

A 必須最先：黃金測試要在舊環境建立，後續才能證明行為未變。B 先於 C，因為 decord 不移除就升不了級；C 先於其他包，新碼才不必為 zarr 2/3、NumPy 1/2 各寫一次。

工作包表的「前置」欄只列直接相依。D 完成後 E、F、G 可平行；H1 在 C 後即可開始，建議儘早做。黃金測試只保護到 E 為止；G 若統一 `InputBlock` 的 IDW 精度（見進度紀錄 2026-11-06），也會改變輸出；F、G 起會有意改變行為，舊權重之後只能靠 tag 重現。

## 已定預設與待決

以下預設可以推翻，但在沒有新資訊前照此執行。

- **切分**：train 2010–2020、test 2021–2022（可與舊事件比較）、val 2023–2024、2025 保留為最後留出集。H1 若發現雷達網斷層再調整。
- **模型尺度**：第一版用 `x = dBZ/85`，與舊尺度相同但不量化，保住 `weighted_l1` 的 `x_max=0.70`。換尺度另做受控實驗。
- **儲存**：新資料保留 int16 原值，讀取時轉浮點，不再用 uint8。
- **事件篩選**：門檻改為 config 參數。舊腳本註解寫 20%、code 是 0.3，不直接沿用。
- **遮罩**：真實站點到手前用可替換的合成固定站點。
- **權重**：舊權重不移植到新區域，只用於基線測試。
- **Zarr**：新管線全用 Zarr，HDF5 只留給舊基線。

待決事項：

- [ ] 雨量站座標（Easting/Northing 或經緯度）與目標區域大小。這是 H2 的卡點；若届時未定，先用一塊全有效 128×128 區域跑通流程，之後重做。
- [ ] 雨量計資料的單位與時間解析度。模型用雷達點值訓練、用雨量計推論，兩者偏差在 I 包量測。
- [ ] 舊資料的原始編碼腳本沒有找到；`v=48·log10(R/0.036)` 即 3×dBZ 是推導，尚未由程式確認。

## 進度紀錄

| 日期 | 包 | 結果 |
| --- | --- | --- |
| 2026-11-06 | A | local 分支 `refactor`、local tag `baseline-v0`（7fec131）；路徑改為 repo 相對；`scripts/tools` 收進 repo；pytest 加入 dev 依賴 |
| 2026-11-06 | A | 訓練重現核對：v9 的 MLflow 參數與現行 config 一致；訓練相關檔案修改時間都早於訓練開始（2026-04-04 05:45） |
| 2026-11-06 | A | commit 3c7afb5。四組權重嚴格載入通過，副本與原檔逐位元組相同 |
| 2026-11-06 | B | commit 9f97e99，淨刪約 950 行。新舊程式在 CPU 上對四個模型的資料讀取、生成器輸出、loss 共 48 個陣列逐值相同 |
| 2026-11-06 | A | RTX 3080 上，`baseline-v0` 與 `refactor` 對 MIDAS 10 個事件的 GPU 推論逐位元相同；與作者存檔差最大 1.97、平均 ≤0.009（0–255 尺度），原因未確認，推測是 GPU 型號或驅動不同。新黃金參考 `datasets/golden/v9_midas_gauge.zarr`，commit 82a2f1b |
| 2026-11-06 | A | CPU 推論不可信：`InputBlock` 的 IDW 在 CUDA 用 float16、CPU 用 float32，4 個最近雨量站的選擇在約 8.6% 的網格點不同，單一視窗輸出差可達 324。權重是在 GPU 路徑上訓練的，所以只有 GPU 輸出代表模型行為。統一精度屬於 G 包，會改變舊權重的輸出 |
