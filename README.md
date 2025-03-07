# 🩺 AI-Powered Medical Image Analysis  

## 📌 Overview  
This project is a **medical image analysis tool** built with **Streamlit, OpenCV, and the Groq API**. Users can upload **medical images (X-rays, MRIs, etc.)**, perform **local edge detection and histogram analysis**, and get **AI-generated medical insights**.  

## 🚀 Features  
✅ **Medical Image Analysis**  
- Uses OpenCV for **edge detection, histogram analysis, and texture variance detection**.  
- Highlights potential abnormalities with contour visualization.  

✅ **AI-Powered Medical Insights (Groq API)**  
- Provides **detailed explanations, symptoms, causes, and next steps** based on image analysis.  
- Supports **LLaMA-3, Mixtral, and Gemma** models.  
- Implements **API rate limit handling with exponential backoff**.  

✅ **Smart Caching & Performance Optimization**  
- Uses **image hashing** for caching previous analysis results.  
- Provides **local fallback analysis** when the API is unavailable.  
- Supports **model selection** from the Streamlit sidebar.  

## 🛠️ Tech Stack  
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Computer Vision:** OpenCV, NumPy, PIL  
- **AI Models:** Groq API (LLaMA-3, Mixtral, Gemma)  
- **Caching & Optimization:** JSON-based local caching  

## 📂 Project Structure  

## 🔧 Installation & Setup  

### 1️⃣ **Clone the Repository**  
```sh
git clone https://github.com/your-username/medical-image-analysis.git
cd medical-image-analysis
2️⃣ Install Dependencies
sh
Copy
Edit
pip install streamlit numpy opencv-python pillow requests

3️⃣ Run the Application
sh
Copy
Edit
streamlit run main.py
