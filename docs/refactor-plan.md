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

**訓練設定（2026-11-07 確定）。** 輸入與輸出都是雷達：輸入只取雨量站所在像素的雷達值，當成 N 條時間序列；label 是整張雷達場。推論時再把雷達序列換成雨量計序列。

雨量站表在 `datasets/gauges_in_domain.csv`（不進 git）：192 站，EA 143、MO_MIDAS_hourly 40、NRW 9。`PixelE`、`PixelN` 是吸附到 NIMROD 1 km 像素中心的座標（吸附距離 65–682 m），與 NIMROD 網格逐點對齊。192 站落在 176 個不同像素，16 個像素有 2 站；以雷達當輸入時同像素的兩站值相同。站點範圍 126 × 126 km，對應 NIMROD 的 row 1085–1210、col 724–849，128×128 的固定區域可以全部包含。

現行程式已經是這個形式：`create_mask` 把固定遮罩上的雷達值留下、其餘歸零，label 是整張場；config 裡的 `keep`、`block_sizes`、`interval` 沒有被讀取。差別在站點換成真實的 176 個像素（舊遮罩 79 個）、區域換成上述固定裁切，以及 dataset 改為直接回傳 N 條序列與站點座標，由模型端決定怎麼鋪成網格（H3、I、G）。

各站的紀錄年份不同（最早一站 1977 年就停止），以雷達當輸入時不受影響；推論時的缺測在 I 包處理。

## 工作包

共 13 包，依字母順序進行；前置欄是必須先完成的包。規模 S 約半天到一天，M 數天，L 含長時間訓練。

| 包 | 內容 | 前置 | 規模 | 狀態 |
| --- | --- | --- | --- | --- |
| A 基線凍結 | local tag；複製 brick 依賴到 `datasets/`、`weights/`；路徑改為 repo 相對；`scripts/tools` 收進 repo；核對 MLflow 紀錄的訓練參數與現行 config；黃金測試 | 無 | S | Done |
| B 清理 | 刪死碼與 mp4 路線、未用依賴、`.codex`；`.gitignore`；README 先刪錯誤內容 | A | S | Done |
| C 升級 | Python 3.14 與套件、lock、pytest、ruff；torch 固定 2.14.1+cu126；黃金測試改容差比對並涵蓋 NIMROD | B | M | Done |
| D Config | pydantic schema、`extra="forbid"`、全部 config 重寫；寫入新資料路徑 | C | M | Done |
| E Trainer | 修 `train.py:217`、`val_every_epochs`、resume、LR 排程；實驗紀錄改用 W&B（抽出 logger 介面） | D | S | In progress |
| F 編碼與指標 | 單一 `RainCodec` 存進 checkpoint；指標重寫並接上驗證；刪三份舊 transform | D | M | In progress |
| G 模型泛化 | T 參數化、H/W 檢查、合併 wopos、DK/STDK 的 k 由遮罩推導；InputBlock 重新設計（見下節） | D | M | Not started |
| H1 資料探勘 | 各年缺幀、無資料區變動、雷達網年代差異、雨量分布 | C | S | In progress |
| H2 前處理 | 裁固定區域、重新 chunk、事件篩選、年份切分索引、無資料遮罩 | H1、F、區域決定 | M | Not started |
| H3 Dataset | 單一 Zarr Dataset、單位與尺寸檢查、地理固定遮罩 | H2、D | M | In progress |
| I 雨量站介面 | 站點表轉遮罩；推論端從時序組輸入張量；單位驗證 | H3、F | M | Not started |
| J 新資料訓練 | overfit 閘門 → 小規模 → 全量；與 DK/STDK 比較 | E、F、G、H3 | L | Not started |
| K 文件收尾 | 架構、資料格式、限制與權重適用範圍 | J | S | Not started |

## 相依與順序

A 必須最先：黃金測試要在舊環境建立，後續才能證明行為未變。B 先於 C，因為 decord 不移除就升不了級；C 先於其他包，新碼才不必為 zarr 2/3、NumPy 1/2 各寫一次。

工作包表的「前置」欄只列直接相依。D 完成後 E、F、G 可平行；H1 在 C 後即可開始，建議儘早做。黃金測試只保護到 E 為止；G 若統一 `InputBlock` 的 IDW 精度（見進度紀錄 2026-11-06），也會改變輸出；F、G 起會有意改變行為，舊權重之後只能靠 tag 重現。

## G 包：InputBlock 的設計問題

以下是在 v9 與 MIDAS `event_01` 上量到的現況。改動任何一項都會改變舊權重的輸出，所以只在新資料重新訓練時一起做。

**結構。** 輸入是 `(B, T, H, W)` 的稀疏影像，只有 79 個雨量站像素有值。第一步把每個像素的 T 幀當成長度 16 的向量，經過 2 層 `AttentionBlock`（類別預設 4 層，兩個 P2I 模型都傳 `depth=2`）：`out = relu(x · (1 + Conv1d_16→16(x)))`。這是時間軸上的乘法閘門，所有像素共用權重。因為是乘法，值為 0 的觀測永遠維持 0。第二步對雨量站像素的值做 3D IDW，鋪成完整的場。之後的網路只看得到這張 IDW 場：原始觀測與遮罩不再往下傳，輸出端也沒有加回 IDW 場的殘差。

**時空距離尺度。** x、y、t 各自除以 (W−1)、(H−1)、(T−1) 正規化到 0–1，再算歐氏距離。在 128×128、T=16 下，1 幀等於約 8.5 像素。這個比例只由視窗大小決定，沒有物理意義；改成 T=20 或 256×256 會悄悄改變時空權衡。新設計應該用明確的尺度參數，例如每幀對應多少公里。

**鄰居實際組成。** 時間差只增加 √(d²+δt²)−d 的距離：離站 0 像素時同站前後幀遠 8.5 像素，13 像素時只遠 2.6 像素，38 像素時不到 1 像素。所以離站越遠，同一站的前後幀越容易擠掉其他站。實測 k=4 時，75% 的網格點只用到 2 個雨量站，19% 用到 3 個，4% 用到 4 個；最常見的組合是最近站的 t−1、t、t+1 加上第二近站的 t。IDW 場因此呈現近似 Voronoi 的多邊形色塊。另外，同一站在 t−1 與 t+1 和網格點的距離完全相等，選哪一個由浮點誤差決定。

**τ（`tau=0.05`）。** 權重是 `1/(d+τ)²`。沒有 τ 時，站上的權重無限大，IDW 在站點會精確等於觀測，近站也壓倒遠站；例如 d=0.01 與 0.1 的權重比是 100。加上 τ=0.05（約 6 像素）後同一比值只剩約 6，場較平滑，但站點像素不再等於觀測：站上自己只佔約 70% 權重，其餘來自同站前後幀與鄰站。τ 和平均站距（約 14 像素，0.11）同一量級，所以距離的影響被大幅攤平。

**精度與平手。** CUDA 上 IDW 的座標轉成 float16（距離在 autocast 下以 float32 計算），CPU 全用 float32；約 8.6% 的網格點兩者選到不同的 4 個鄰居，CPU 推論因此和訓練時的行為不一致。根本原因是規則網格讓大量網格點的第 4、第 5 近鄰距離完全相等，勝負取決於座標捨入；torch 2.15 改了 float16 `linspace` 的捨入，就讓每個視窗約 2.7% 的網格點換了鄰居，P2I 輸出局部差到 86。詳見 `docs/reports/p2i-numerical-instability.md`。

待決：缺測處理（重要）：舊程式完全沒有處理缺測，篩選規則也不檢查缺測；原始 NIMROD 轉 uint8 的腳本找不到，缺測可能被當成 0 雨量。第一版排除含缺測的視窗，之後要做的實驗包括讓 `AttentionBlock` 略過缺測、label 缺測在 loss 中遮掉、逐點與整站 dropout。時空尺度參數化、k 與 τ 的取值或改成可學、網格與雨量站座標同一路線計算並一律 float32 以上、在平手處連續的權重（取代硬性 top-k）、是否把原始觀測與遮罩一併送進後續網路（例如加一個觀測通道），以及這些在新資料上的消融實驗。

## 已定預設與待決

以下預設可以推翻，但在沒有新資訊前照此執行。

- **切分**：train 2010–2020、test 2021–2022（可與舊事件比較）、val 2023–2024、2025 保留為最後留出集。H1 若發現雷達網斷層再調整。
- **模型尺度**：第一版用 `x = dBZ/85`，與舊尺度相同但不量化，保住 `weighted_l1` 的 `x_max=0.70`。換尺度另做受控實驗。
- **儲存**：新資料保留 int16 原值，讀取時轉浮點，不再用 uint8。
- **事件篩選**：沿用舊規則，門檻做成 config 參數：視窗內任一幀，中心區有 ≥30% 像素非零（v ≥ 1，即 R > 0.036 mm/h）且至少一個像素 v ≥ 84（R ≥ 2.0 mm/h）。舊規則以 40 幀的 1024×1024 檔案為單位；新管線改為對裁切後的研究區逐視窗計算，換門檻只需重算索引，不必另建資料集。
- **DK、STDK**：第一版不移植，只保留舊版供黃金測試。
- **原則**：先以最少改動讓新資料的訓練跑通；屬於實驗性質的改動（IDW 尺度、閘門、dropout、DK/STDK）之後再做。
- **算力**：可用 H200；模型小，3080 也能訓練。預算可以放寬。
- **遮罩**：真實站點到手前用可替換的合成固定站點。
- **權重**：舊權重不移植到新區域，只用於基線測試。
- **站點數**：P2I 對輸入點數沒有限制（`InputBlock` 取遮罩上所有非零的 (t, y, x)）；DK、STDK 把 79 寫死在 `topk` 與 MLP 輸入寬度（`T × 79`），換成 176 站或隨機丟站都要改 G 包。
- **Zarr**：新管線全用 Zarr，HDF5 只留給舊基線。

待決事項：

- [x] 雨量站座標與目標區域：`datasets/gauges_in_domain.csv`，176 個像素。研究區以 Scout Moor 與 Manchester Airport 的中點取整（E 382000、N 402000）為中心、各方向 64 km：E 318000–446000、N 338000–466000，即 NIMROD row 1084–1211、col 723–850。中心之後可能改，改了要重新訓練。
- [x] 時間解析度：保留雷達的 5 分鐘，T=16 即 80 分鐘視窗。
- [x] 年份：train 2020–2023、val 2024、test 2025。
- [x] label 的缺測像素在 loss 中遮掉。
- [ ] 輸入碰到缺測：IDW 本身不受影響（遮罩是逐 (t, y, x)，缺的點直接不參與），問題只在 `AttentionBlock` 的閘門混合 16 幀、看不到遮罩，會把缺測當成 0 雨量。第一版先跳過輸入有缺測的視窗；閘門改成會略過缺測屬於 G 的實驗。H1 量到缺測比例後再確認。
- [x] 第一版固定 176 個站點像素，不丟站。整站丟棄與逐點丟棄兩種 dropout 都列為之後的實驗（逐點丟棄需要先改 `AttentionBlock`）。
- [x] 模型尺度沿用 P2I：`v = 48·log10(R/0.036)`、`x = v/255`（R ≤ 0.036 mm/h 為 0）。作者的解碼 `R = 0.036·10^(v/48)` 見 brick `Nimrod_2d_data/NIMROD_Reader/nimrod_video_export.ipynb`；新管線以浮點計算、不量化成 uint8。
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
| 2026-11-06 | C | commit 6489ba7。Python 3.12、torch 2.14.1+cu126、numpy 2.5、zarr 3.4、mlflow 3.18、scipy 1.18、pandas 3.1；ruff 加入並修正全 repo。輸出 store 改為 zarr v3；mlflow 3 的 file store 需設 `MLFLOW_ALLOW_FILE_STORE=true` |
| 2026-11-06 | C | 升級後在 3080 上對黃金參考：DK、STDK 逐位元相同；v9 最大差 2.1、平均 0.006；wopos v9 最大差 1.65、平均 0.008。黃金測試容差 P2I 設 (2.5, 0.02)、DK/STDK 設為 0。全套 15 個測試通過 |
| 2026-11-06 | C | 以新環境重跑作者的 exp1：gauge、radar 的所有指標與作者 `metrics.json` 相對差 ≤1.1e-5。nimrod 的 `metrics.json`（4/10）早於作者 4/11 修改 PSS 並移除 DTSSIM/NSE 的 commit，無法比對；其中 MAE、RMSE、SSIM 仍相同 |
| 2026-11-06 | C | 事故：import 檢查誤載入 `scripts/tools/preprocess.py`，其模組層級以 `mode="w"` 開啟 `datasets/nimrod/nimrod_train.zarr`，刪掉本地副本 12686 個事件中的 3699 個。已自 brick 原檔補回：1,662,133 個檔案名稱與大小全部一致，無任何檔案的修改時間晚於事故前。brick 原檔與 isaac 資料未受影響。`scripts/tools` 之後只做 `py_compile`，不 import |
| 2026-11-07 | C | 依新容器改為 Python 3.14。Python 3.14 的 DataLoader 改用 forkserver，dataset 的 `Lambda(lambda ...)` 無法 pickle，改成模組層級函式 |
| 2026-11-07 | C | torch 2.15.0+cu132 讓 v9 對黃金參考最大差 63–86。逐層拆解定位到 `InputBlock`：torch 2.15 改了 float16 `linspace` 的捨入，網格座標移動一個最小單位，平手的鄰居選擇翻轉。torch 固定 `>=2.14.1,<2.15`、cu126，報告見 `docs/reports/p2i-numerical-instability.md` |
| 2026-11-07 | C | Python 3.14 + torch 2.14.1+cu126 直接對作者輸出：gauge 10 事件、nimrod 20 事件，DK、STDK 逐位元相同；v9 最大差 2.16、wopos v9 最大差 1.81，平均 ≤0.013。作者的「P2I-GAN」輸出由 wopos v9 權重產生。作者 nimrod 驗證集複製到 `datasets/nimrod_h5/author_val`；黃金測試加入 NIMROD 與 wopos 對作者的比較，共 20 個測試全過，約 9.5 分鐘 |
| 2026-11-07 | — | 確定訓練設定：輸入是雨量站像素上的雷達序列，label 是整張雷達場。雨量站表 `datasets/gauges_in_domain.csv` 與 NIMROD 網格逐點對齊；D 與 H1 同步進行 |
| 2026-11-07 | — | 研究區、訓練年份 2020–2023、5 分鐘解析度、label 缺測遮罩確定。以真實 176 站與現行 IDW 尺度（1 幀 ≈ 8.5 km），4 個鄰居來自 1/2/3/4 站的網格點比例為 0.3/32.6/29.0/38.1% |
| 2026-11-07 | D | `p2igan_bench/config/schema.py`：pydantic schema，未知 key 一律報錯；`train.py` 與 `infer.py` 載入時驗證，`Trainer` 建構時也驗證。程式仍收同結構的 dict，行為不變。刪掉 config 中程式從未讀取的 key（`hole_weight`、`valid_weight`、`fractal_weight`、`temp_weight`、`use_fractal`、`decay`、`d2glr`、`save_freq`、`valid_freq`、`niter_steady`、`use_test`、`base_channels`、`comod`、`data.*.name`、遮罩的 `keep`/`block_sizes`/`interval`）。新增 `kind: "domain"` 的資料區段定義（store、年份、站點表、視窗長度、篩選門檻），dataset 尚未實作 |
| 2026-11-07 | H1/H2 | `scripts/data/build_domain_store.py`：從 isaac 的月檔裁出研究區，寫成單一 `datasets/nimrod_domain.zarr`（2020–2025，int16 原值、規則 5 分鐘時間軸、缺的時間點補 -1 並標記），附逐幀統計（缺測像素數、濕像素數、最大值）。各月 `time` 的起點是該月月初，已換成絕對時間 |
| 2026-11-07 | H1 | `datasets/nimrod_domain.zarr` 建好（1.7 GB，72 個月）。2020–2025 沒有缺時間點；有缺測像素的幀每年 16–200 幀（<0.2%），多數是整幀缺；雨量站像素有缺的幀共 346 幀（0.05%）。研究區沒有無資料區。超過 100 mm/h 的像素幀共 29,097 個，空間上分散（最多的 10 個像素只佔 0.7%），看不出固定雜波 |
| 2026-11-07 | H1 | 舊篩選規則（任一幀 ≥30% 濕、峰值 ≥2 mm/h）並排除含缺測幀的 16 幀視窗：train 89,369 個（stride 1；不重疊 6,043）、val 23,458、test 17,893，約佔全部視窗的 17–22%。放寬到 ≥10% 濕約為兩倍。因為缺測很少，第一版直接排除含缺測的視窗，label 不會有缺測，loss 遮罩先不需要 |
| 2026-11-08 | H3 | `p2igan_bench/data/domain_dataset.py`：從研究區 store 依門檻切 16 幀視窗、排除含缺測的視窗，輸入為 176 個站點像素，輸出格式與舊 `Dataset` 相同，模型、loss、`Trainer` 不變。所需幀載入共享記憶體（train 3.4 GB，載入約 4.5 分鐘）。val、test 用 stride 16（1,584、1,207 個視窗） |
| 2026-11-08 | J | 第一次訓練 `p2igan_bench/config/p2igan_gan_domain.json`：v9 的 loss 與最佳化設定、200,000 iterations、batch 12，3080 約 1.06 it/s，預估 52 小時。300 步試跑 rec loss 0.70 → 0.32、val 0.45 → 0.42 |
| 2026-11-08 | E | `Trainer` 可從 `save_dir/latest.pt` 續跑（含 MLflow run id、`best_val`）；修掉關閉驗證時 `val_loss` 未定義。新增 `cudnn_benchmark`、`amp: "bf16"`、`compile` 三個開關，預設關閉；compile 不含 discriminator（舊式 spectral norm）與 `InputBlock`。`scripts/train_jobs/` 仿 circqpe 提供 slurm 與本機訓練腳本 |
| 2026-11-08 | G | 量測發現 `InputBlock` 佔生成器一步的 82%：每個樣本都對 262,144 個網格點重算 kNN。鄰居與權重只取決於座標，改為依點集快取；輸出與梯度逐位元相同，forward 677 → 12 ms，全套測試 40 個通過（4 分鐘）。3080 上一步：無開關 4.09 it/s、cudnn+compile 4.15、加 bf16 7.28 |
| 2026-11-08 | J | 300 步試跑之外的第一次長跑在 5,600 步停止（保留於 `weights/p2igan_gan_domain_v0_trial`），以 cudnn+compile 重新開始 `weights/domain-v1`，預估 13–14 小時 |
| 2026-11-08 | — | 新 repo `NTU-CompHydroMet-Lab/P2I-GAN-ScoutMoor` 設為 `origin`，`main` 與 `refactor` 一致；作者原 repo 改名為 `original`，不再推送。Slurm 暫緩，模型小，本機 3080 足夠 |
| 2026-11-08 | E | 實驗紀錄改用 W&B，取代 MLflow（仿 circqpe）：`p2igan_bench/run_logger.py`，config 的 `log_to_wandb`、`wandb_project`，W&B run id 存進 checkpoint 供續跑；step 指標以 `global_step`、epoch 指標以 `epoch` 為 x 軸，每個 epoch 上傳範例圖。不再把 checkpoint 上傳成 artifact |
| 2026-11-08 | E/G | Discriminator 改用 `parametrizations.spectral_norm`，可以 compile；載入時把舊鍵名（`weight_orig`/`weight_u`/`weight_v`）改成新鍵名，v9 與 domain-v1 的舊 checkpoint 都能嚴格載入，eval 輸出與舊版逐位元相同。compile 與 bf16 同時開啟可正常訓練與續跑 |
| 2026-11-08 | F | `p2igan_bench/codec.py`：`RainCodec`（P2I 對數尺度，`v < 1` 解碼為 0），domain dataset 改用它，checkpoint 存 `codec`。`p2igan_bench/metrics/rain.py`：`RainScores` 以 mm/h 累積 MAE、RMSE、bias、POD/FAR/CSI（模型沒看到的像素）與 FSS（全場），驗證時一併計算並記到 W&B；刪掉沒人用、尺度也錯的舊 `metrics/metric.py` 與 torchmetrics 依賴。`experiments/` 的兩份舊轉換仍保留，以免改變作者分析的數字 |
| 2026-11-08 | F/J | `scripts/eval_events.py` 與 `p2igan_bench/config/eval/pilot_events.json`：對指定事件評估 checkpoint 並與同站點的 IDW 比較，輸出指標、報告與圖。重跑 epoch 7 快照，與先前試跑的數字最大差 0.002（差在 Darragh 有缺測的影格整幀排除） |
