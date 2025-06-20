# 🛠 Equipment Failure Dashboard with Predictive Maintenance and Customer Service Chatbot

This project is a **Streamlit web application** that provides **interactive data analysis**, **predictive maintenance insights**, and an integrated **customer service chatbot** for equipment failure management. It helps identify failure-prone models and components, perform root cause analysis using the Apriori algorithm, predict future equipment failures using Random Survival Forest models, and assist users with real-time answers through a conversational chatbot.

---

## 📌 Features

### 🔍 Data Analysis

- Upload equipment failure CSV files  
- Filter data by model numbers and part names  
- Visualize top models and parts by failure rate  
- Root Cause Analysis using **Apriori Algorithm** for association rules  

### 🔧 Predictive Maintenance

- Predict **"Time to Failure"** for equipment using **Random Survival Forest**  
- Calculate **Failure Risk Scores**  
- Analyze high-risk spare parts  
- Visualize failure time distribution and risk score comparisons  
- Suggest spare part optimization based on risk level  

### 🤖 Customer Service Chatbot

- Conversational interface integrated into the dashboard  
- Built using **Gemini Flash 2.0**, optimized for fast and accurate responses  
- Trained on **intent-based Q&A pairs** generated from original equipment failure data  
- Capable of answering queries related to:  
  - *Equipment models and their failure trends*  
  - *Specific part failure reasons*  
  - *Predictive maintenance insights and usage instructions*  
  - *Navigation and feature help within the dashboard*  

---

## 🚀 Tech Stack & Tools

- **Frontend**: Streamlit  
- **Visualization**: Plotly, Matplotlib, Seaborn  
- **Data Manipulation**: Pandas, NumPy  
- **Machine Learning**:  
  - **Association Rules**: `mlxtend`  
  - **Predictive Modeling**: `sksurv` (scikit-survival)  
  - **Modeling**: Random Survival Forests  
- **Chatbot Integration**:  
  - **LLM**: Gemini Flash 2.0  
  - **Training Data**: Custom intent-based Q&A generated from domain-specific datasets  
  - **NLP Preprocessing**: Tokenization, embedding generation, and similarity matching  


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

**PdM System**
![Screenshot 2025-06-20 121417](https://github.com/user-attachments/assets/8c6db8fb-6fc4-4638-970c-41fbbe3fad6d)
![Screenshot 2025-06-20 121427](https://github.com/user-attachments/assets/778dc190-2ff3-41b1-b3e5-db4dd7a2d4ef)
![Screenshot 2025-06-20 121444](https://github.com/user-attachments/assets/bcea7b96-ddc7-4efc-9ad4-0561d847d5c7)
![Screenshot 2025-06-20 122148](https://github.com/user-attachments/assets/77ee980f-6411-4bb5-a18e-92a6cc4ee265)
![Screenshot 2025-06-20 122201](https://github.com/user-attachments/assets/96472f51-b2e0-4825-b24f-e28c294ec4f8)
![Screenshot 2025-06-20 122217](https://github.com/user-attachments/assets/5cff620a-0546-4842-87a3-84751ed8883d)
![Screenshot 2025-06-20 122228](https://github.com/user-attachments/assets/d930777c-31ad-4214-a2ca-fd58f18565af)
![Screenshot 2025-06-19 152922](https://github.com/user-attachments/assets/2982d94e-2b28-4bdb-93c1-89bb39395cb9)

**Customer Service Chatbot**

![Screenshot 2025-06-20 125351](https://github.com/user-attachments/assets/3f8b74a0-e355-45a3-ac76-e84dbf366b59)
![Screenshot 2025-06-20 125400](https://github.com/user-attachments/assets/0bdaa483-593d-4adf-9cdb-dbd10a820283)
![Screenshot 2025-06-20 125417](https://github.com/user-attachments/assets/60366901-0275-42f0-9e8d-0cdc09cc8982)

---
📬 Contact
For questions or collaborations, feel free to reach out via LinkedIn or create an issue in the repository.



