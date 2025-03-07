# medical-diagonsis
🩺 AI-Powered Medical Image Analysis
📌 Overview
This project is a medical image analysis tool built using Streamlit, OpenCV, and the Groq API. It allows users to upload medical images (X-rays, MRIs, etc.), performs local edge detection and histogram analysis, and provides AI-generated medical insights.

🚀 Features
✅ Medical Image Analysis:

Uses OpenCV to detect abnormalities in medical images.
Provides edge detection, histogram analysis, and texture variance detection.
Draws contours and highlights regions of interest.
✅ AI-Powered Insights (Groq API)

Generates medical explanations, symptoms, causes, and next steps.
Uses LLM models (LLaMA-3, Mixtral, Gemma, etc.) to process findings.
Implements API rate limit handling with exponential backoff.
✅ Smart Caching & Performance Optimization

Uses image hashing to cache previous analysis results.
Local fallback analysis when API is unavailable.
Supports model selection via Streamlit sidebar.
🛠️ Tech Stack
Frontend: Streamlit
Computer Vision: OpenCV, NumPy, PIL
AI Models: Groq API (LLaMA-3, Mixtral, Gemma)
Caching & Optimization: JSON-based local caching
📂 Project Structure
graphql
Copy
Edit
📦 medical-image-analysis
 ┣ 📜 main.py          # Main Streamlit app
 ┣ 📜 README.md        # Documentation
 ┗ 📂 cache/           # Cached API responses  
🔧 Installation & Setup
1️⃣ Install Dependencies
sh
Copy
Edit
pip install streamlit numpy opencv-python pillow requests
2️⃣ Run the App
sh
Copy
Edit
streamlit run main.py
⚠ Disclaimer
This tool does not provide medical diagnoses. It is intended for educational purposes only. Always consult a healthcare professional for medical advice.
