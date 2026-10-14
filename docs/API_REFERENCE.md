# API Reference

> REST API documentation for the GraphSAGE AML Fraud Detection backend.

**Base URL:** `http://localhost:8000`

---

## Endpoints

### 1. Predict Transactions

**`POST /api/predict`**

Upload a transaction CSV file and receive fraud predictions for each transaction.

#### Request

- **Content-Type:** `multipart/form-data`
- **Body:** File upload (field name: `file`)

#### Required CSV Columns

| Column | Type | Description |
|---|---|---|
| `Timestamp` | string | Transaction timestamp (`YYYY/MM/DD HH:MM`) |
| `From Bank` | string | Sender's bank identifier |
| `Account` | string | Sender's account identifier |
| `To Bank` | string | Recipient's bank identifier |
| `Account.1` | string | Recipient's account identifier |
| `Amount Received` | float | Amount received by recipient |
| `Amount Paid` | float | Amount paid by sender |
| `Receiving Currency` | string | Currency received |
| `Payment Currency` | string | Currency used for payment |
| `Payment Format` | string | Payment method (ACH, Wire, Cash, etc.) |

#### Optional CSV Columns

| Column | Type | Description |
|---|---|---|
| `Is Laundering` | int (0/1) | Ground truth label (for evaluation) |

#### Response

```json
{
  "nodes": [
    {"id": "8000ECA930"}
  ],
  "links": [
    {
      "source": "8000ECA930",
      "target": "8000F5E710",
      "prediction": 0.0234,
      "isFraud": false,
      "amount": 1500.50,
      "timestamp": "2022/09/01 08:30",
      "currency": "US Dollar"
    }
  ],
  "threshold": 0.95,
  "actualFraudCount": 3,
  "modelUsed": "cycle_attn"
}
```

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `nodes` | array | Unique account nodes with `id` |
| `links` | array | Transaction edges with predictions |
| `links[].prediction` | float | Fraud probability (0.0 – 1.0) |
| `links[].isFraud` | boolean | Whether prediction exceeds threshold |
| `threshold` | float | Classification threshold used |
| `actualFraudCount` | int | Ground truth fraud count (0 if column absent) |
| `modelUsed` | string | `"cycle_attn"` or `"baseline"` |

#### Example (cURL)

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@data/samples/sample_trans_4_with_fraud.csv"
```

---

### 2. Get Ablation Results

**`GET /api/ablation`**

Returns evaluation metrics for all trained model variants.

#### Response

```json
{
  "baseline": {
    "roc_auc": 0.9852,
    "pr_auc": 0.3235,
    "best_threshold": 0.95,
    "fraud_recall": 0.874,
    "fraud_precision": 0.023,
    "ring_recall_any": 0.987,
    "ring_recall_majority": 0.978,
    "ring_recall_complete": 0.769,
    "mean_ring_edge_coverage": 0.834
  },
  "cycle_rw": { ... },
  "cycle_attn": { ... }
}
```

#### Example

```bash
curl "http://localhost:8000/api/ablation"
```

---

### 3. Get Model Info

**`GET /api/model_info`**

Returns information about the currently loaded model.

#### Response

```json
{
  "active_model": "cycle_attn",
  "threshold": 0.95,
  "device": "mps"
}
```

#### Example

```bash
curl "http://localhost:8000/api/model_info"
```

---

## Error Responses

| Status | Response | Cause |
|---|---|---|
| 200 | `{"error": "Model not loaded..."}` | No trained model found in `outputs/` |
| 200 | `{"error": "Missing required columns..."}` | CSV missing required columns |
| 422 | Validation Error | Invalid request format |

---

## Running the Server

```bash
# Development (with auto-reload)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## CORS Configuration

The API allows all origins (`*`) by default for development. For production, update the `allow_origins` list in `src/api/main.py`.
