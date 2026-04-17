# Readme pls

本repo未包含原始 Yelp 数据、预处理后的 Parquet 数据、以及训练/评估产生的模型产物，太大了传不上去

对应大文件目录/后缀已在 [.gitignore](file:///c:/Users/debuf/Desktop/CityU_SDSC6001_Project_BOC_Team25/.gitignore) 中忽略：

- `Yelp_data/`（原始数据）
- `outputs/`（预处理产物，主要是 Parquet）
- `artifacts/`（训练/评估产物）
- `*.parquet`, `*.pt`, `*.ckpt`, `*.pth`（大文件后缀）

***

## 1) 原始数据应放置位置

将 Yelp Academic Dataset 的原始 JSON 文件放在项目根目录下的：

```
Yelp_data/
  yelp_academic_dataset_business.json
  yelp_academic_dataset_review.json
  yelp_academic_dataset_user.json
  yelp_academic_dataset_checkin.json
  yelp_academic_dataset_tip.json
```

***

## 2) 预处理产物（outputs/）会生成什么

运行 `data_prep/` 下的 Step1\~Step5 后，所有预处理产物会写入项目根目录：

```
outputs/
```

### Step 1: filter & split

脚本：`data_prep/step1_filter_and_split.py`

- `outputs/business_filtered.parquet`
- `outputs/reviews_filtered.parquet`
- `outputs/users_train.parquet`
- `outputs/users_val.parquet`
- `outputs/users_test.parquet`
- `outputs/split_stats.txt`

### Step 2: BERT encoding

脚本：`data_prep/step2_bert_encoding.py`

- `outputs/business_bert_embeddings.parquet`

### Step 3: geo features

脚本：`data_prep/step3_geo_features.py`

- `outputs/business_geo_features.parquet`

### Step 4: graph construction

脚本：`data_prep/step4_graph_construction.py`

- `outputs/graph_user_business_edges.parquet`
- `outputs/graph_user_social_edges.parquet`
- `outputs/graph_stats.txt`

### Step 5: aspect assignment

脚本：`data_prep/step5_aspect_assignment.py`

- `outputs/reviews_with_aspects.parquet`
- `outputs/aspect_distribution.txt`

***

## 3) 训练/评估产物（artifacts/）会生成什么

训练与评估脚本位于 `scripts/`，产物输出到：

```
artifacts/
```

主要包含：

- Baselines：`baseline_metrics_{val,test}.json`
- 每个实验的评估指标：`*_{val,test}.json`
- 每个实验的训练曲线：`*_history.csv`
- 汇总表：`report_results.csv`（由 `scripts/summarize_results.py` 汇总生成）

此外，我们还有 report 生成脚本，会把表格与曲线图输出到：

```
report/
  tables/
  figures/
```

***

## 4) 如何在本地重新生成

### 4.1 生成 outputs/（Step1\~5）

在项目根目录运行：

```powershell
python data_prep\run_pipeline.py --start-step 1 --end-step 5
```

#

