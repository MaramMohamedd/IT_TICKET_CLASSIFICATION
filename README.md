# Intelligent Programming Final Project: IT Service Ticket Classification System

## 📌 Project Overview
This project implements an intelligent IT service ticket classification system using two complementary AI approaches: **Case-Based Reasoning (CBR)** and a **GRU-based Deep Learning model**. The system categorizes IT issue descriptions, retrieves similar historical cases, and provides actionable solutions through an interactive Streamlit web application.

<img width="1919" height="824" alt="Screenshot 2025-12-15 124029" src="https://github.com/user-attachments/assets/c118256c-8950-4a11-a468-7fae9fa93bae" />

---

## 📂 Dataset

**Source:** [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)

**Size:** 47,837 rows × 2 columns  
**Columns:**
- `description`: Text description of the IT issue
- `topic category I`: Category label (e.g., Hardware, HR Support, Access, etc.)

**Class Distribution:**
- The dataset exhibits moderate class imbalance, with Hardware being the most frequent category.

<img width="1000" height="500" alt="class_distribution" src="https://github.com/user-attachments/assets/d8b2a7a0-de35-4c17-afed-d112e8648d23" />

---

## 🔧 Preprocessing Pipeline

### 1. **Base Preprocessor** (`base_preprocessor.py`)
- Removes nulls and duplicates
- Cleans text: lowercasing, URL/email removal, punctuation handling

### 2. **CBR Preprocessor** (`cbr_preprocessor.py`)
- Uses **TF-IDF vectorization** with max_features=15,000 and ngram_range=(1,2)
- Saves vectorizer for inference on new data

### 3. **Sequence Preprocessor** (`sequence_preprocessor.py`)
- Tokenizes text into integer sequences
- Pads/truncates sequences to fixed length (120 tokens)
- Builds vocabulary with OOV handling for GRU model

---

## 🤖 Models Implemented

### 1. **CBR Model (Case-Based Reasoning)**
- **Purpose:** Retrieves similar historical cases for new queries
- **Method:**
  - TF-IDF vectorization
  - k-Nearest Neighbors (k=3) with cosine similarity
- **Features:**
  - Real-time similarity search
  - Dynamic case addition to knowledge base
  - Model persistence for deployment
- **Case Base Update:** Uses JSONL format for efficient incremental updates

### 2. **GRU Model (Gated Recurrent Unit)**
- **Purpose:** Classifies IT issues into categories using deep learning
- **Architecture:**
  - Embedding Layer (128-dim)
  - GRU Layer (128 units)
  - Dense Layers with Dropout
  - Softmax output for multi-class classification
- **Training:**
  - Early stopping with patience=3
  - Sparse categorical crossentropy loss
  - Adam optimizer

---

## 💾 Data Storage Strategy

### **Case Base Updates with JSONL**
We implemented a **JSONL (JSON Lines)** approach for storing and updating the case base instead of traditional CSV files:

```python
# JSONL - Just append one line
with open('feedback.jsonl', 'a') as f:
    f.write(json.dumps(new_feedback) + '\n')  # Fast, atomic operation

# CSV - Must rewrite entire file (inefficient for large datasets)
df = pd.read_csv('feedback.csv')           # Load ALL data
df = df.append(new_feedback)               # Add new row  
df.to_csv('feedback.csv', index=False)     # Rewrite ENTIRE file
```

**Why JSONL?**
- ✅ **Efficiency:** Append-only operations are O(1) vs O(n) for CSV rewrites
- ✅ **Atomicity:** Single-line writes are safer in concurrent environments
- ✅ **Scalability:** Handles large datasets without memory overhead
- ✅ **Fault Tolerance:** Partial writes don't corrupt existing data
- ✅ **Streaming Friendly:** Easy to process line-by-line


### **Case Base Update Process**
1. **User Feedback Collection:** Users mark solutions as helpful/not helpful
2. **Periodic Retraining:** System can retrain models with updated case base
3. **Feedback Analysis:** User feedback used to improve model confidence scoring

---

## 📊 Training Results

**GRU Model Performance:**
- Final validation accuracy: ~85–86%
- Training stopped at epoch 7 due to early stopping
- Model shows strong learning capability with IT-specific text patterns

**CBR Model Performance:**
- Efficient similarity matching with cosine similarity scores
- Real-time retrieval of top-3 similar cases
- Dynamic learning from new cases via JSONL updates

---

## 🚀 Streamlit Application Features

### **1. Dual-Model Comparison**
- Side-by-side display of CBR and GRU predictions
- Shows similarity scores (CBR) and confidence levels (GRU)
  <img width="1578" height="747" alt="Screenshot 2025-12-15 124253" src="https://github.com/user-attachments/assets/ecbd9464-a109-4d63-a0b9-221b95700820" />


### **2. Interactive Visualizations**
- Pie chart of case category distribution
- Bar chart of similarity scores for top 3 CBR matches
<img width="921" height="771" alt="Screenshot 2025-12-15 125343" src="https://github.com/user-attachments/assets/287bdd14-af5f-4ddd-92f2-5c7e29ec6725" />

### **3. Feedback System**
- Users can mark solutions as "Helpful" or "Not Helpful"
- Feedback stored in `feedback.jsonl` for future model improvements
- Incremental updates without reloading entire dataset
<img width="1533" height="749" alt="Screenshot 2025-12-15 124420" src="https://github.com/user-attachments/assets/76a922f4-a32b-4800-a8ae-dd994889ddec" />

### **4. Knowledge Base Management**
- Add new cases directly through the UI
- Solutions for recurring issues are preserved
<img width="1606" height="552" alt="Screenshot 2025-12-15 124652" src="https://github.com/user-attachments/assets/32662aa9-2ea0-4f9c-a1c4-f3ef509c5737" />

### **5. Combined Recommendations**
- System suggests reviewing both models when confidence is moderate
- Provides actionable next steps based on historical data


### Final evaluation : 
CBR model out performed the GRU model in the generalization stage
---

## 🗂️ Project Structure

```
project/
├── models/
│   ├── preprocessing/
│   │   ├── base_preprocessor.py
│   │   ├── cbr_preprocessor.py
│   │   └── sequence_preprocessor.py
│   ├── cbr_model.py
│   └── gru_model.py
├── data
│  
├── artifacts/
│   ├── gru_model.h5
│   ├── cbr_tfidf.joblib
│   └── tokenizer.joblib
├── streamv4.py                       # Streamlit application
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Usage

### 1. Clone Repository
```bash
git clone <repository-url>
cd intelligent-ticket-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run streamv4.py
```

### 4. Update Case Base (Programmatically)
```python
import json
from datetime import datetime

new_case = {
    "case_id": f"CASE_{int(datetime.now().timestamp())}",
    "description": "Printer not connecting to network",
    "solution": "Check network cable and printer IP settings",
    "category": "Hardware",
    "timestamp": datetime.now().isoformat(),
    "helpful_votes": 0,
    "unhelpful_votes": 0
}

# Efficient append operation
with open('data/case_base.jsonl', 'a') as f:
    f.write(json.dumps(new_case) + '\n')
```

---


## 📈 Future Improvements

1. **Hybrid Model:** Combine CBR retrieval with GRU classification for improved accuracy
2. **Real-Time Learning:** Automatically incorporate user feedback into model retraining
3. **Multi-Language Support:** Extend to non-English IT tickets
4. **Deployment:** Dockerize application for cloud deployment
5. **JSONL Analytics:** Implement streaming analytics on case base updates
6. **Automated Quality Scoring:** Use feedback to rank case quality automatically
7. **Update & Retraining Pipeline**
   through : 
1. **Daily Updates:** New cases accumulate in `case_base.jsonl`
2. **Weekly Retraining:** 
   - Load all cases from JSONL files
   - Retrain TF-IDF vectorizer on updated corpus
   - Update k-NN similarity index
3. **Model Versioning:** Track performance improvements per update cycle


---

## 📚 References

- Scikit-learn: TF-IDF, NearestNeighbors
- TensorFlow/Keras: GRU, Embedding, EarlyStopping
- Streamlit: Web app framework
- JSONL Format: Efficient line-delimited JSON storage
- Kaggle Dataset: IT Service Ticket Classification

---

## 📄 License
This project is developed for academic purposes as part of the Intelligent Programming course. All rights reserved by the development team.

---

**✨ Developed with ❤️ by Ola, Jana, and Maram**
