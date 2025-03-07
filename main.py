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
            st.sidebar.success("Groq API key is valid and working")
        else:
            st.sidebar.error(f"API key error: {response.status_code} - {response.text}")
    except Exception as e:
        st.sidebar.error(f"API key error: {str(e)}")

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
def get_cached_medical_insights(condition_key):
    """Check if we have a cached response for this condition"""
    cache_file = f"cache_{condition_key[:40].replace(' ', '_')}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            return cache_data.get('response')
    
    return None

def save_to_cache(condition_key, response):
    """Save API response to cache file"""
    cache_file = f"cache_{condition_key[:40].replace(' ', '_')}.json"
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

def get_medical_insights(condition, max_retries=3, retry_delay=2):
    """Fetch medical insights from Groq API with rate limit handling and caching."""
    
    # Skip API call if we already know this is an error condition
    if condition.startswith("Error") or condition == "Model loading failed":
        return f"Cannot provide medical insights: {condition}"
    
    # Generate a cache key from the condition
    condition_key = condition.lower().strip()
    
    # Check cache first
    cached_response = get_cached_medical_insights(condition_key)
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
            
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant providing educational information."},
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
            
            # Save to cache
            save_to_cache(condition_key, result)
            
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
st.title("🩺 Medical Image Analysis Assistant")

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
tab1, tab2 = st.tabs(["Medical Image Analysis", "Medical Questions"])

with tab1:
    st.write("Upload a medical image for AI analysis. Note: This is for educational purposes only.")
    
    uploaded_file = st.file_uploader("Upload a Medical Image (X-ray, MRI, etc.)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        analysis_method = st.radio(
            "Choose analysis method:",
            ["Local Analysis + AI Insights", "Local Analysis Only (No API)"]
        )
        
        if st.button("Analyze Image"):
            if analysis_method == "Local Analysis + AI Insights":
                with st.spinner("Performing local analysis..."):
                    detected_condition, result_image = detect_medical_condition_local(image)
                    
                    st.subheader("Detection Results:")
                    st.info(f"Detected: {detected_condition}")
                    
                    # Show detection visualization
                    if result_image is not None:
                        st.image(result_image, caption="Analysis Visualization", use_container_width=True)
                
                with st.spinner("Getting Medical Insights..."):
                    insights = get_medical_insights(detected_condition)
                    st.subheader("Medical Analysis:")
                    st.write(insights)
            
            else:  # Local Analysis Only
                with st.spinner("Performing local analysis..."):
                    detected_condition, result_image = detect_medical_condition_local(image)
                    
                    st.subheader("Detection Results:")
                    st.info(f"Detected: {detected_condition}")
                    
                    # Show detection visualization
                    if result_image is not None:
                        st.image(result_image, caption="Analysis Visualization", use_container_width=True)
                    
                    # Get local insights without API
                    local_insights = get_local_analysis(detected_condition)
                    st.subheader("Local Analysis (No API):")
                    st.write(local_insights)

with tab2:
    st.subheader("Medical Question Assistant")
    st.write("Ask general medical questions and get AI-powered responses.")
    
    user_query = st.text_input("Ask any medical question:")
    if st.button("Get Answer") and user_query:
        if not groq_api_key:
            st.warning("Please enter a Groq API key in the sidebar to use this feature.")
        else:
            with st.spinner("Analyzing your question..."):
                try:
                    # Create a cache key for the question
                    cache_key = f"question_{hashlib.md5(user_query.encode()).hexdigest()[:10]}"
                    
                    # Check cache
                    cached_response = get_cached_medical_insights(cache_key)
                    if cached_response:
                        st.write(f"{cached_response}\n\n(This response was retrieved from cache)")
                    else:
                        st.write("Sending request to Groq API...")  # Debug message
                        
                        # Prepare API request
                        headers = {
                            "Authorization": f"Bearer {groq_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        medical_prompt = f"""
                        You are a helpful medical information assistant. Provide educational information about the following medical question, 
                        making sure to include appropriate disclaimers about not being a replacement for professional medical advice:
                        
                        {user_query}
                        """
                        
                        payload = {
                            "model": MODEL_NAME,
                            "messages": [
                                {"role": "system", "content": "You are a helpful medical information assistant."},
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
                        
                        st.write("Response received from API")  # Debug message
                        
                        if response.status_code == 200:
                            result = response.json()["choices"][0]["message"]["content"]
                        else:
                            raise Exception(f"API error: {response.status_code} - {response.text}")
                        
                        # Save to cache
                        save_to_cache(cache_key, result)
                        
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

No actual diagnosis is provided - this is for educational purposes only.
""")

st.markdown("---")
st.markdown("⚠ Important Disclaimer: This tool is for educational purposes only and should not be used for diagnosis. Always consult with a qualified healthcare professional for medical advice and interpretation of medical images.")
