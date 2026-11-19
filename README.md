# EB 二樓整層 3D Gaussian Splatting（光達里程計 + T265 輔助）

用一台移動機器人錄的 12 段 ROS2 bag（4 個區域 × 3 個相機高度），重建整層二樓的 3D Gaussian Splatting 地圖。
**光達－慣性里程計（FAST-LIO2, Livox Mid-360）為主要位姿來源，Intel RealSense T265 為輔助**，影像來自 ZED2 左眼，訓練使用 gsplat 的 MCMC。

![三個高度同一位置（由上到下 h3 / h2 / h1）](docs/images/hall_lidar_version.jpg)

> **v2（2026-11-19）**：關鍵影格加密一倍、整層切成 8 個分區、每張影像的曝光補償、光達牆面約束、各分區色彩校正。
> 排除曝光差異後 PSNR 30.96 → **31.33**（兩版都沒訓練過的同一批影像：30.29 → **31.29**），細節保留率 38% → 49%。詳見[v2 改進](#v2-改進2026-11-19)。

## 下載影片與模型

影片與模型檔太大，放在 [Releases](../../releases)（最新為 **v2.1**：Isaac Sim 碰撞場景＋各高度單獨影片；v2.0 為 v2 模型本身；v1.0 為上一版）：

**v2.1**

| 檔案 | 內容 |
|---|---|
| `floor2_h1_front*.mp4`、`floor2_h2_front*.mp4`、`floor2_h3_front*.mp4` | 修正後路線（route v3）的各高度單獨影片；原速 0.6 m/s、`_x1.5`、`_x2` |
| `EB2F_v2.usdz.part1`、`.part2` | Isaac Sim 場景（v2 高斯＋碰撞），2.9 GB，因單檔 2 GB 上限切成兩段，下載後 `cat EB2F_v2.usdz.part1 EB2F_v2.usdz.part2 > EB2F_v2.usdz` |
| `collision_floor.ply`、`collision_obstacles.ply`、`collision.usdc` | 碰撞網格（地板／牆面家具天花板），單獨使用 |

**v2.0**

| 檔案 | 內容 |
|---|---|
| `floor2_3heights_front*.mp4` | 繞整層一圈，三個高度上下同步（上 h3、中 h2、下 h1）；原速 0.6 m/s、`_x1.5`、`_x2` |
| `floor2_h3_front*.mp4` | 同一路線，只有 h3 |
| `floor2_v2_compressed.ply` | v2 整層模型（1,227 萬個高斯），SuperSplat 壓縮 PLY，620 MB |
| `floor2_v2_west.ply`、`floor2_v2_east.ply` | v2 整層模型的標準 3DGS PLY，因 GitHub 單檔 2 GB 上限，以 x = 23.9 m 切成東西兩半 |

v2 影片的虛擬相機：水平視角 75°（ZED 原生約 101°，畫面邊緣都是被斜看的牆）、離牆至少 0.55 m、下傾 4°。

## 成果

在 415 張**所有分區都沒拿去訓練**的影像上評估：

| 版本 | 位姿來源 | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| 舊版 | T265 為主 + 光達合併 + COLMAP BA | 28.64 | 0.928 | 0.288 |
| **目前版本** | **FAST-LIO2 為主，T265 輔助** | **30.51** | **0.938** | **0.266** |

各區、各高度 PSNR：28.5～32.6（`results/eval_lidar_primary.json`）。

| 光達為主時的整層格局（4 區剛性 ICP，無尺度校正） | T265 誤差分布（BA 修正量，紅 = T265 偏差大） |
|---|---|
| ![](docs/images/floor_layout_lidar.png) | ![](docs/images/t265_error_map.png) |

## Isaac Sim 場景與碰撞（v2.1）

把 v2 整層模型和碰撞網格打包成一個 `.usdz`，在 Isaac Sim 6.1 直接開啟，按 Play 就有重力與碰撞。

| 實拍（上）vs Isaac Sim 渲染（下），p1 大廳 | p4 北走廊（視角跨過分塊接縫） |
|---|---|
| ![](docs/images/isaac_vs_photo_p1hall.jpg) | ![](docs/images/isaac_vs_photo_p4.jpg) |

**場景結構**（Z 軸朝上，單位公尺；整個場景上移 0.19 m，地板約在 z = 0）
- `/World/EB2F/Gaussians/Part0..3`：`ParticleField3DGaussianSplat`（OpenUSD 26.03 schema），1,227 萬個高斯，SH 3 階，float。
- `/World/EB2F/Collision/Floor`、`Obstacles`：隱形網格，`PhysicsCollisionAPI`＋三角網格碰撞。
- `/World/PhysicsScene`：重力 −Z。

**碰撞怎麼做**（`scripts/make_collision_mesh.py`）

![碰撞俯視圖：顏色 = 地板高度，黑點 = 0–1.5 m 的障礙物](docs/images/isaac_collision_topdown.jpg)

- **地板**：前處理時光達地圖的地板已被挖掉，換成合成地板，所以地板高度改從高斯模型量測：取各區錄影路徑 1 m 內、不透明且平躺的高斯的中位數（p1 −0.204、p2 −0.183、p3 −0.191、p4 −0.182 m），交界處平滑過渡。可走範圍 = 離路徑 2 m 內、0.15–1.5 m 高度沒有光達點、且和路徑相連的格子（約 400 m²）。
- **牆、家具、天花板**：4 區光達點（剛性 place_T 轉到整層座標，去掉合成地板點）→ 5 cm 體素 → 去除孤立點 → 膨脹 1 格 → marching cubes → 簡化到 60 萬個三角形。

**實測**（`scripts/isaac_test_usdz.py`，結果見 `results/isaac/physics_render_test.txt`）：從實拍相機位置渲染和照片一致；球掉落後停在地板上（誤差 1 cm 內）；球以 4 m/s 丟向牆面會停下並反彈，不會穿牆。

**打包注意事項**（`scripts/make_isaac_usdz.py`）
- 不用 Isaac 內建的 `omni.kit.converter.gsplat`：它逐點用 Python 迴圈、寫成 ASCII `.usda`，1,200 萬個高斯不實際。這裡改用 numpy → `Vt` 陣列直接寫二進位 `.usdc`。
- **單一 prim 的 SH 陣列超過 2 GB 時，渲染結果全黑，而且沒有任何錯誤訊息**，所以沿 x 軸切成 4 個 prim，畫面上看不到接縫。
- 半精度屬性（`scalesh` 等）能讓檔案小一半，但畫面明顯變髒（和 float 版相比 PSNR 29 dB），所以使用 float。

**開啟**：`python scripts/isaac_open_usdz.py EB2F_v2.usdz`，視角會放在 p1 大廳 h2 高度；或在 Isaac Sim 選 File → Open。顯示記憶體約需 9 GB，載入約 3 分鐘。
想看碰撞網格：Viewport 眼睛圖示 → Show By Type → Physics → Colliders。

**限制**：碰撞只涵蓋錄影路徑附近，門後的房間沒有地板；南側走廊實際只有約 1.6 m 寬。

## v2 改進（2026-11-19）

| 項目 | 做法 | 效果 |
|---|---|---|
| 關鍵影格加密 | 每 5 cm 或 2.5° 一張（原本 12 cm／5°），共 11,187 張 | BA 重投影誤差 0.86 → 0.82 px |
| 8 個分區 | 每個區域沿走廊主軸從中位數切兩半（`scripts/make_part_map.py`） | 各分區在 300 萬高斯的上限內有更多容量 |
| 曝光補償 | 訓練時每張影像學一組 RGB 增益＋偏移，只作用於 loss（`--exposure-comp`） | 各錄影段、各高度的亮度一致，結構不再遷就曝光 |
| 光達牆面約束 | 光達平面 5 cm 體素（擴張 10 cm）內的高斯：中心拉回平面、最薄軸對齊法線、變薄（`--wall-normals`，`scripts/make_surface_normals.py`） | 牆面高斯離牆距離中位數 3.6 → 1.1 cm；只作用在光達平面附近，地板下的反光高斯不受影響 |
| 色彩校正 | 曝光補償使每個分區的整體色調可自由漂移；以 200 張實拍擬合每區增益／偏移並寫回 SH（`scripts/colour_calibrate.py`） | 分區之間色調一致、整體顏色回到實拍 |

**評估**（`scripts/eval_colourfit.py`，`results/v2_dense_8part/`）。因為 v2 刻意輸出統一曝光，直接算的 PSNR 會下降（例如 p3_h2 那段錄影本身拍得比較亮），所以另外計算「每張影像先擬合整體增益／偏移，再算 PSNR」的公平指標：

| | v1（4 分區） | v2（8 分區） |
|---|---|---|
| 排除曝光差異的 PSNR（各自的保留影像） | 30.96 | **31.33** |
| 同上，兩版都沒訓練過的同一批 45 張影像 | 30.29 | **31.29** |
| 直接 PSNR | 30.51 | 26.61 |
| LPIPS | 0.266 | 0.270 |
| 細節保留率（渲染／實拍的細尺度能量） | 38% | **49%** |

| v1（上）與 v2（下），影片路線上相同位置 |
|---|
| ![](docs/images/v1_vs_v2_a.jpg) |
| ![](docs/images/v1_vs_v2_b.jpg) |

![統一曝光：左實拍、右 v2 渲染](docs/images/v2_unified_exposure.jpg)

**模糊的來源**：實拍影像的清晰度和亮度高度相關（暗的 p2、p4 只有亮的 p1 的一半），和移動速度幾乎無關，代表主因是光線不足造成的增益雜訊與相機降噪，不是動態模糊。要再明顯提升，需要重新錄影：補光、固定曝光（≤ 8–10 ms）與白平衡，並在走廊左右各走一趟，讓側牆有正面視角。

### 為什麼改成光達為主
`results/lio_vs_t265.txt`：T265 在走廊裡的路徑長度比 FAST-LIO 多 7～20%，並有多次重定位跳躍（最大 3.1 m，正好在出現「兩台飲水機」鬼影的大廳）。
FAST-LIO 在繞回起點的錄影段閉合誤差只有 3～25 cm；COLMAP BA 對光達位姿的修正中位數 3～4 cm（T265 時是 7～22 cm）。

| T265 為主的版本（大廳模糊、鬼影） | 光達為主的版本 |
|---|---|
| ![](docs/images/hall_t265_versions.jpg) | ![](docs/images/hall_lidar_version.jpg) |

## 流程

```
ROS2 bag (/livox/lidar, /livox/imu, /t265/odom, /zed/.../image/compressed, /tf_static)
 │
 ├─ FAST-LIO2 (fast_lio/run/run_fastlio.sh)              每段錄影的光達-慣性里程計 (TUM)
 ├─ 調平：路徑擬合沿走廊坡度 + 各段光達地圖的天花板平面    results/lio_trajectories/*.tum
 ├─ scripts/bag_to_colmap.py --prep_only --odom_dir      關鍵影格、相機位姿、10 s 光達子地圖
 ├─ scripts/merge_heights.py --rigid_only --planar       同一區域三個高度合併（x, y, z, yaw）
 ├─ scripts/refine_colmap.py (pycolmap 4)                SIFT + 位姿引導配對 + 帶位置先驗的 BA（含 ZED 安裝角 2.3°）
 ├─ scripts/place_layout_lio.py                          4 個區域放進同一座標系（剛性 ICP）
 ├─ scripts/build_floor2.py --strict                     整層資料集，只留 BA 修正 < 0.3 m 且 < 3° 的影像
 ├─ scripts/subset_dataset.py (VIS_R=12)                 依可見度切分區（v1：4 區；v2：PART_MAP=part_map_8.json 切 8 區）
 ├─ scripts/make_depth_dataset.py                        光達深度監督（≥1 m），移除沒有影像看過的初始點
 ├─ scripts/run_lio_train.sh / run_liod_train.sh         gsplat MCMC：每區 300 萬高斯、6 萬步、depth loss（v2 加 --exposure-comp --wall-normals）
 ├─ scripts/colour_calibrate.py                          v2：各分區色彩校正（曝光補償的整體色調）
 ├─ scripts/merge_partitions.py                          分區合併（交界 1.5 m 淡入淡出）＋評估
 └─ scripts/render_floor_video.py / make_route.py        沿錄影路線的平滑影片（三高度同步）
```

### 依賴
- ROS2 Jazzy、[FAST-LIO (ROS2 branch)](https://github.com/hku-mars/FAST_LIO) + `patches/fast_lio_ros2_jazzy.patch`（C++17、移除未使用的 pcl_ros、補 tf2_ros）
- `fast_lio/livox_ros_driver2_msgs/`：只含 `CustomMsg`/`CustomPoint` 訊息定義的 `livox_ros_driver2` 套件（不需 Livox SDK）
- [gsplat](https://github.com/nerfstudio-project/gsplat)（`patches/gsplat_base_commit.txt`）+ `patches/gsplat_simple_trainer.patch`（`--max-scale`、`--surface-grid`、`--exposure-comp`、`--wall-normals`）
- conda 環境：`gs`（torch, gsplat, open3d, mcap）與 `colmap`（pycolmap 4.x）

### 重要的訓練設定與教訓
- MCMC：`--visible-adam`、`--opacity-reg 0.001 --scale-reg 0.001`（暗走廊用預設 0.01 會崩潰）、`--no-normalize-world-space` 時 `noise_lr = 5e4·1.21/scene_scale²`、`depth_lambda = 0.05/scene_scale`
- 單一模型裝不下整層（300 萬高斯不夠）→ 4 個分區分別訓練再合併
- 不要用：bilateral grid、上限 400 萬＋packed（崩潰）、硬性光達表面約束（地板反光需要地板下的高斯，會崩潰）、合併後再刪高斯（PSNR −2 dB）
- 地板比 base_link 低約 0.2 m（ZED 在 h1 時離地約 0.23 m）

### 已知限制
只有一台朝前的相機、走廊偏暗：貼近鏡頭的側牆、近處家具（如大廳長桌）會有拖影，光滑地板的反光從錄影沒走過的位置看會糊；從走廊外或上方俯瞰時看得到模型外圍的雜亂高斯（訓練只約束「從走廊內看」的畫面）。

## 檔案
- `scripts/`：全部流程（含已不使用的實驗腳本）
- `fast_lio/`：FAST-LIO 離線執行腳本、參數、訊息套件
- `patches/`：對 FAST-LIO 與 gsplat 的修改
- `results/`：評估、光達軌跡、對齊參數
- `docs/`：圖片與渲染對照（左：實拍，右：渲染）
