import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time
import os
import json
from datetime import datetime
import hashlib
import requests
import random

# Configure Groq API
st.sidebar.header("API Configuration")
groq_api_key = "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"
MODEL_NAME = "llama3-70b-8192"  # Default Groq model

# Add language selection for Indian languages
st.sidebar.header("Language Settings")
LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi", 
    "বাংলা (Bengali)": "bn",
    "తెలుగు (Telugu)": "te",
    "मराठी (Marathi)": "mr",
    "தமிழ் (Tamil)": "ta",
    "ગુજરાતી (Gujarati)": "gu",
    "ಕನ್ನಡ (Kannada)": "kn",
    "മലയാളം (Malayalam)": "ml",
    "ਪੰਜਾਬੀ (Punjabi)": "pa",
    "ଓଡ଼ିଆ (Odia)": "or",
    "অসমীয়া (Assamese)": "as",
}

selected_language = st.sidebar.selectbox(
    "Select Language", 
    options=list(LANGUAGES.keys()),
    index=0  # Default to English
)

lang_code = LANGUAGES[selected_language]

# Translations dictionary - extend with more phrases as needed
TRANSLATIONS = {
    # English translations (default)
    "en": {
        "app_title": "🩺 Medical Image Analysis Assistant",
        "upload_text": "Upload a medical image for AI analysis. Note: This is for educational purposes only.",
        "upload_button": "Upload a Medical Image (X-ray, MRI, etc.)",
        "analyze_button": "Analyze Image",
        "loading_text": "Performing analysis...",
        "results_header": "Detection Results:",
        "viz_caption": "Analysis Visualization",
        "medical_analysis": "Medical Analysis:",
        "local_analysis": "Local Analysis (No API):",
        "questions_tab": "Medical Questions",
        "questions_header": "Medical Question Assistant",
        "questions_description": "Ask general medical questions and get AI-powered responses.",
        "question_input": "Ask any medical question:",
        "get_answer": "Get Answer",
        "api_key_warning": "Please enter a Groq API key in the sidebar to use this feature.",
        "method_choice": "Choose analysis method:",
        "method_local_ai": "Local Analysis + AI Insights",
        "method_local_only": "Local Analysis Only (No API)",
        "disclaimer": "⚠ Important Disclaimer: This tool is for educational purposes only and should not be used for diagnosis. Always consult with a qualified healthcare professional for medical advice and interpretation of medical images."
    },
    # Hindi translations
    "hi": {
        "app_title": "🩺 चिकित्सा छवि विश्लेषण सहायक",
        "upload_text": "एआई विश्लेषण के लिए चिकित्सा छवि अपलोड करें। नोट: यह केवल शैक्षिक उद्देश्यों के लिए है।",
        "upload_button": "एक चिकित्सा छवि अपलोड करें (एक्स-रे, एमआरआई, आदि)",
        "analyze_button": "छवि का विश्लेषण करें",
        "loading_text": "विश्लेषण किया जा रहा है...",
        "results_header": "पहचान परिणाम:",
        "viz_caption": "विश्लेषण विज़ुअलाइज़ेशन",
        "medical_analysis": "चिकित्सा विश्लेषण:",
        "local_analysis": "स्थानीय विश्लेषण (कोई एपीआई नहीं):",
        "questions_tab": "चिकित्सा प्रश्न",
        "questions_header": "चिकित्सा प्रश्न सहायक",
        "questions_description": "सामान्य चिकित्सा प्रश्न पूछें और एआई-संचालित प्रतिक्रियाएँ प्राप्त करें।",
        "question_input": "कोई भी चिकित्सा प्रश्न पूछें:",
        "get_answer": "उत्तर प्राप्त करें",
        "api_key_warning": "इस सुविधा का उपयोग करने के लिए कृपया साइडबार में Groq API कुंजी दर्ज करें।",
        "method_choice": "विश्लेषण विधि चुनें:",
        "method_local_ai": "स्थानीय विश्लेषण + एआई अंतर्दृष्टि",
        "method_local_only": "केवल स्थानीय विश्लेषण (कोई एपीआई नहीं)",
        "disclaimer": "⚠ महत्वपूर्ण अस्वीकरण: यह उपकरण केवल शैक्षिक उद्देश्यों के लिए है और निदान के लिए इसका उपयोग नहीं किया जाना चाहिए। चिकित्सा सलाह और चिकित्सा छवियों की व्याख्या के लिए हमेशा योग्य स्वास्थ्य देखभाल पेशेवर से परामर्श करें।"
    },
    # Add translations for other languages here
    # Bengali translations
    "bn": {
        "app_title": "🩺 মেডিকেল ইমেজ বিশ্লেষণ সহকারী",
        "upload_text": "এআই বিশ্লেষণের জন্য একটি মেডিকেল ইমেজ আপলোড করুন। দ্রষ্টব্য: এটি শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে।",
        "upload_button": "একটি মেডিকেল ইমেজ আপলোড করুন (এক্স-রে, এমআরআই, ইত্যাদি)",
        # Add more Bengali translations as needed
        "disclaimer": "⚠ গুরুত্বপূর্ণ দাবিত্যাগ: এই টুলটি শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে এবং রোগ নির্ণয়ের জন্য ব্যবহার করা উচিত নয়। চিকিৎসা পরামর্শ এবং মেডিকেল ইমেজ ব্যাখ্যার জন্য সর্বদা একজন যোগ্য স্বাস্থ্যসেবা পেশাদারের সাথে পরামর্শ করুন।"
    },
    # Telugu translations
    "te": {
        "app_title": "🩺 వైద్య చిత్ర విశ్లేషణ సహాయకుడు",
        "upload_text": "AI విశ్లేషణ కోసం ఒక వైద్య చిత్రాన్ని అప్‌లోడ్ చేయండి. గమనిక: ఇది విద్యా ప్రయోజనాల కోసం మాత్రమే.",
        # Add more Telugu translations as needed
        "disclaimer": "⚠ ముఖ్యమైన నిరాకరణ: ఈ సాధనం విద్యా ప్రయోజనాల కోసం మాత్రమే మరియు రోగనిర్ధారణకు ఉపయోగించకూడదు. వైద్య సలహా మరియు వైద్య చిత్రాల వివరణ కోసం ఎల్లప్పుడూ అర్హత గల ఆరోగ్య నిపుణులను సంప్రదించండి."
    },
}

# Get translation based on selected language, fallback to English if not available
def translate(key):
    if lang_code in TRANSLATIONS and key in TRANSLATIONS[lang_code]:
        return TRANSLATIONS[lang_code][key]
    return TRANSLATIONS["en"][key]  # Fallback to English

# Verify API key is working
if groq_api_key:
    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        test_payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": MODEL_NAME
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                json=test_payload, 
                                headers=headers)
        if response.status_code == 200:
            st.sidebar.success("✅ Groq API key is valid")
        else:
            st.sidebar.error(f"❌ API key error: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"❌ API key error: {str(e)}")

# Improved local model approach with multiple detection methods
def detect_medical_condition_local(image):
    """
    Perform enhanced local medical condition detection using multiple OpenCV techniques.
    """
    try:
        # Convert PIL image to OpenCV format
        img_array = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create a grayscale version for analysis
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Multiple analysis techniques
        # 1. Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # 2. Histogram analysis - look for unusual distributions
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()
        hist_std = np.std(hist_normalized)
        
        # 3. Texture analysis with GLCM-like approach (simplified)
        texture_variance = np.var(gray)
        
        # Create visualization
        # Draw contours for visualization
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis_img = img_cv.copy()
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
        
        # Add histogram to visualization
        hist_img = np.zeros((200, 256, 3), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(hist_img, (i, 200), (i, 200 - int(hist[i])), (255, 0, 0), 1)
        
        # Combine visualizations
        combined_height = vis_img.shape[0] + hist_img.shape[0]
        combined_width = max(vis_img.shape[1], hist_img.shape[1])
        combined_vis = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_vis[:vis_img.shape[0], :vis_img.shape[1]] = vis_img
        combined_vis[vis_img.shape[0]:, :hist_img.shape[1]] = hist_img
        
        # Convert back to PIL for display
        vis_pil = Image.fromarray(cv2.cvtColor(combined_vis, cv2.COLOR_BGR2RGB))
        
        # More sophisticated determination based on multiple factors
        abnormality_score = 0
        reasons = []
        
        if edge_ratio > 0.08:
            abnormality_score += 1
            reasons.append("High edge density")
        
        if hist_std > 0.015:
            abnormality_score += 1
            reasons.append("Unusual histogram distribution")
        
        if texture_variance > 2000:
            abnormality_score += 1
            reasons.append("High texture variance")
            
        # Create a more detailed and specific condition message
        if abnormality_score >= 2:
            condition = f"Potential abnormality detected ({', '.join(reasons)})"
            condition += f" - Edge ratio: {edge_ratio:.3f}, Histogram std: {hist_std:.4f}, Texture var: {texture_variance:.1f}"
        else:
            condition = f"No significant abnormalities detected - Edge ratio: {edge_ratio:.3f}, Histogram std: {hist_std:.4f}, Texture var: {texture_variance:.1f}"
            
        return condition, vis_pil
        
    except Exception as e:
        st.error(f"Error in local detection: {str(e)}")
        return f"Error in image processing: {str(e)}", None

# Improved cache function that better handles uniqueness
def get_better_image_hash(image):
    """Create a more reliable hash for the image"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
    img_bytes = img_byte_arr.getvalue()
    return hashlib.md5(img_bytes).hexdigest()

# Add a cache for API responses to avoid rate limits
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_medical_insights(condition_key, lang="en"):
    """Check if we have a cached response for this condition in the specified language"""
    cache_file = f"cache_{condition_key[:40].replace(' ', '')}{lang}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            return cache_data.get('response')
    
    return None

def save_to_cache(condition_key, response, lang="en"):
    """Save API response to cache file with language tag"""
    cache_file = f"cache_{condition_key[:40].replace(' ', '')}{lang}.json"
    cache_data = {
        'response': response,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

def exponential_backoff(retries, max_retries=5, initial_delay=1):
    """Calculate delay with exponential backoff and jitter"""
    if retries >= max_retries:
        return None  # Stop retrying
    delay = initial_delay * (2 ** retries) + (random.random() * 0.5)  # Add jitter
    return min(delay, 60)  # Cap at 60 seconds

def get_medical_insights(condition, lang_code="en", max_retries=3, retry_delay=2):
    """Fetch medical insights from Groq API with rate limit handling, caching, and language support."""
    
    # Skip API call if we already know this is an error condition
    if condition.startswith("Error") or condition == "Model loading failed":
        return f"Cannot provide medical insights: {condition}"
    
    # Generate a cache key from the condition
    condition_key = condition.lower().strip()
    
    # Check cache first, including language
    cached_response = get_cached_medical_insights(condition_key, lang_code)
    if cached_response:
        return f"{cached_response}\n\n(This response was retrieved from cache)"
    
    # If no API key is provided
    if not groq_api_key:
        return "Please enter a Groq API key in the sidebar to get medical insights."
    
    for attempt in range(max_retries):
        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            
            # Construct the prompt in the selected language
            if lang_code == "en":
                prompt = f"""
                You are a medical assistant. Based on the medical image analysis, the following was detected: {condition}.
                
                Please provide:
                1. A brief explanation of what this finding might indicate (be specific to the finding, not generic)
                2. Common symptoms that might be associated with this specific finding
                3. Possible causes related to the specific metrics mentioned
                4. Recommended next steps
                5. Important disclaimers about the limitations of AI-based diagnosis
                
                Format your response in a clear, structured way with headers for each section.
                Be very clear that this is NOT a diagnosis and the patient should consult a medical professional.
                Be specific to the details in the condition message and avoid generic responses that could apply to any condition.
                """
            else:
                prompt = f"""
                You are a medical assistant. Based on the medical image analysis, the following was detected: {condition}.
                
                Please provide:
                1. A brief explanation of what this finding might indicate (be specific to the finding, not generic)
                2. Common symptoms that might be associated with this specific finding
                3. Possible causes related to the specific metrics mentioned
                4. Recommended next steps
                5. Important disclaimers about the limitations of AI-based diagnosis
                
                Format your response in a clear, structured way with headers for each section.
                Be very clear that this is NOT a diagnosis and the patient should consult a medical professional.
                Be specific to the details in the condition message and avoid generic responses that could apply to any condition.
                
                Provide your response in {selected_language} language.
                """
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": f"You are a helpful medical assistant providing educational information. Respond in {selected_language} language."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 2048
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            # Save to cache with language tag
            save_to_cache(condition_key, result, lang_code)
            
            # Update usage tracker
            today = datetime.now().strftime("%Y-%m-%d")
            usage_key = f"usage_count_{today}"
            if usage_key not in st.session_state:
                st.session_state[usage_key] = 0
            st.session_state[usage_key] += 1
            
            return result
        
        except Exception as e:
            error_message = str(e)
            # Check if this is a rate limit error
            if "429" in error_message or "rate limit" in error_message.lower():
                backoff_time = exponential_backoff(attempt, max_retries, retry_delay)
                if backoff_time is not None:
                    time.sleep(backoff_time)
                    continue
                else:
                    return """
                    Rate Limit Exceeded
                    
                    The API service is currently experiencing high demand. Please try again in a few minutes.
                    
                    In the meantime, please note that any AI-based detection should be confirmed by a healthcare professional.
                    """
            else:
                return f"Error getting insights: {error_message}"

# Function to provide local fallback analysis when API is unavailable
def get_local_analysis(condition):
    """Provide basic analysis without using API"""
    if "abnormality" in condition.lower():
        # Extract specific reasons if present
        reasons = []
        if "Edge ratio:" in condition:
            try:
                edge_ratio = float(condition.split("Edge ratio:")[1].split(",")[0])
                reasons.append(f"Edge ratio of {edge_ratio:.3f}")
            except:
                pass
                
        if "Histogram std:" in condition:
            try:
                hist_std = float(condition.split("Histogram std:")[1].split(",")[0])
                reasons.append(f"Histogram standard deviation of {hist_std:.4f}")
            except:
                pass
                
        if "Texture var:" in condition:
            try:
                texture_var = float(condition.split("Texture var:")[1].split(",")[0])
                reasons.append(f"Texture variance of {texture_var:.1f}")
            except:
                pass
        
        reason_text = ", ".join(reasons) if reasons else "unspecified pattern"
        
        return f"""
        Local Analysis: Potential Abnormality Detected
        
        The local detection algorithm has identified areas of the image with unusual patterns based on: {reason_text}. This could indicate:
        
        1. Possible Findings: The system detected areas with unusual characteristics that may require professional review
        2. Limitations: This is a basic analysis using simple computer vision techniques and is not a diagnosis
        3. Next Steps: Consult with a healthcare professional for proper interpretation
        4. Important Note: This analysis is performed locally without AI assistance and should be considered preliminary only
        """
    else:
        return """
        Local Analysis: No Significant Findings
        
        The local detection algorithm did not identify unusual patterns. This indicates:
        
        1. Basic Assessment: No significant anomalies detected using simple image analysis
        2. Limitations: This is a basic analysis that can miss subtle findings
        3. Next Steps: If you have symptoms or concerns, consult with a healthcare professional
        4. Important Note: Even "normal" findings on basic analysis may not rule out conditions requiring medical attention
        """

# Clear cache button to force fresh analysis
def clear_cache():
    cache_files = [f for f in os.listdir() if f.startswith("cache_")]
    for file in cache_files:
        os.remove(file)
    st.success("Cache cleared! Next analysis will generate fresh results.")

# Streamlit UI
st.title(translate("app_title"))

# Add cache clearing option
if st.sidebar.button("Clear Analysis Cache"):
    clear_cache()

# Add API test button
if st.sidebar.button("Test API Connection"):
    if not groq_api_key:
        st.sidebar.error("Please enter a Groq API key first")
    else:
        try:
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }
            test_payload = {
                "messages": [{"role": "user", "content": "Hello, can you respond with 'API is working'?"}],
                "model": MODEL_NAME
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                    json=test_payload, 
                                    headers=headers)
            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"]
                st.sidebar.success(f"API Test Response: {response_text}")
            else:
                st.sidebar.error(f"API Test Failed: {response.status_code} - {response.text}")
        except Exception as e:
            st.sidebar.error(f"API Test Failed: {str(e)}")

# Add usage tracker
st.sidebar.markdown("### API Usage Tracker")
today = datetime.now().strftime("%Y-%m-%d")
usage_key = f"usage_count_{today}"
if usage_key not in st.session_state:
    st.session_state[usage_key] = 0
st.sidebar.text(f"Requests today: {st.session_state[usage_key]}")

# Create tabs
tab1, tab2 = st.tabs(["Medical Image Analysis", translate("questions_tab")])

with tab1:
    st.write(translate("upload_text"))
    
    uploaded_file = st.file_uploader(translate("upload_button"), type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        analysis_method = st.radio(
            translate("method_choice"),
            [translate("method_local_ai"), translate("method_local_only")]
        )
        
        if st.button(translate("analyze_button")):
            if analysis_method == translate("method_local_ai"):
                with st.spinner(translate("loading_text")):
                    detected_condition, result_image = detect_medical_condition_local(image)
                    
                    st.subheader(translate("results_header"))
                    st.info(f"Detected: {detected_condition}")
                    
                    # Show detection visualization
                    if result_image is not None:
                        st.image(result_image, caption=translate("viz_caption"), use_container_width=True)
                
                with st.spinner(translate("loading_text")):
                    insights = get_medical_insights(detected_condition, lang_code)
                    st.subheader(translate("medical_analysis"))
                    st.write(insights)
            
            else:  # Local Analysis Only
                with st.spinner(translate("loading_text")):
                    detected_condition, result_image = detect_medical_condition_local(image)
                    
                    st.subheader(translate("results_header"))
                    st.info(f"Detected: {detected_condition}")
                    
                    # Show detection visualization
                    if result_image is not None:
                        st.image(result_image, caption=translate("viz_caption"), use_container_width=True)
                    
                    # Get local insights without API
                    local_insights = get_local_analysis(detected_condition)
                    st.subheader(translate("local_analysis"))
                    st.write(local_insights)

with tab2:
    st.subheader(translate("questions_header"))
    st.write(translate("questions_description"))
    
    user_query = st.text_input(translate("question_input"))
    if st.button(translate("get_answer")) and user_query:
        if not groq_api_key:
            st.warning(translate("api_key_warning"))
        else:
            with st.spinner(translate("loading_text")):
                try:
                    # Create a cache key for the question
                    cache_key = f"question_{hashlib.md5(user_query.encode()).hexdigest()[:10]}"
                    
                    # Check cache
                    cached_response = get_cached_medical_insights(cache_key, lang_code)
                    if cached_response:
                        st.write(f"{cached_response}\n\n(This response was retrieved from cache)")
                    else:
                        # Prepare API request
                        headers = {
                            "Authorization": f"Bearer {groq_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        medical_prompt = f"""
                        You are a helpful medical information assistant. Provide educational information about the following medical question, 
                        making sure to include appropriate disclaimers about not being a replacement for professional medical advice:
                        
                        {user_query}
                        
                        Provide your response in {selected_language} language.
                        """
                        
                        payload = {
                            "model": MODEL_NAME,
                            "messages": [
                                {"role": "system", "content": f"You are a helpful medical information assistant. Respond in {selected_language} language."},
                                {"role": "user", "content": medical_prompt}
                            ],
                            "temperature": 0.4,
                            "max_tokens": 2048
                        }
                        
                        response = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            json=payload,
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            result = response.json()["choices"][0]["message"]["content"]
                        else:
                            raise Exception(f"API error: {response.status_code} - {response.text}")
                        
                        # Save to cache
                        save_to_cache(cache_key, result, lang_code)
                        
                        # Update usage tracker
                        st.session_state[usage_key] += 1
                        
                        st.write(result)
                except Exception as e:
                    st.error(f"Error type: {type(e)._name_}")
                    st.error(f"Detailed error: {str(e)}")
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        st.error("Rate limit exceeded. Please try again in a few minutes.")

# Model selector
st.sidebar.markdown("### Model Settings")
model_option = st.sidebar.selectbox(
    "Select Groq Model",
    ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    index=0
)
MODEL_NAME = model_option  # Update the model name based on selection

st.sidebar.markdown("---")
st.sidebar.markdown("### About This App")
st.sidebar.markdown("""
This application demonstrates medical image analysis using:
1. Local edge detection for basic image analysis
2. Groq API for AI-powered medical insights
3. Response caching to minimize API calls
4. Support for multiple Indian languages

No actual diagnosis is provided - this is for educational purposes only.
""")

st.markdown("---")
st.markdown(translate("disclaimer"))
