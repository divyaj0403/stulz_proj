# 🛠 Equipment Failure Dashboard with Predictive Maintenance

This project is a **Streamlit web application** that provides **interactive data analysis** and **predictive maintenance** insights for equipment failure management. It helps identify failure-prone models and components, perform root cause analysis using Apriori algorithm, and predict future equipment failures using Random Survival Forest models.

---

## 📌 Features

### 🔍 Data Analysis
- Upload equipment failure CSV files
- Filter data by model numbers and part names
- Visualize top models and parts by failure rate
- Root Cause Analysis using **Apriori Algorithm** for association rules

### 🔧 Predictive Maintenance
- Predict "Time to Failure" for equipment using **Random Survival Forest**
- Calculate **Failure Risk Scores**
- Analyze high-risk spare parts
- Visualize failure time distribution and risk score comparisons
- Suggest spare part optimization based on risk level

---

## 🚀 Tech Stack & Tools

- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: 
  - Association Rules: `mlxtend`
  - Predictive Modeling: `sksurv` (scikit-survival)
- **Modeling**: Random Survival Forests

---

## 📂 Folder Structure

stulz-proj/ │ ├── app.py # Main Streamlit Application ├── requirements.txt # Required dependencies └── sample_data.csv # Example CSV data


---

## ⚙️ How to Run Locally

```bash
1. Clone the Repository
git clone https://github.com/your-username/stulz-proj.git
cd stulz-proj

2. Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows

3. Install Dependencies
bash
Copy
Edit
pip install --no-cache-dir -r requirements.txt

4. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
```
---
## ⚙️ Screenshots

---
📬 Contact
For questions or collaborations, feel free to reach out via LinkedIn or create an issue in the repository.



