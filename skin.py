import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pickle
import os
from pathlib import Path

# Import your preprocessing class
class SkinTonePredictor:
    """Skin tone predictor using the best model"""
    
    def __init__(self, model_path='best_model.pkl', preprocessing_dir='preprocessed_ml_data'):
        self.classes = ['dark', 'light', 'mid-dark', 'mid-light']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load model and preprocessing
        self.load_model_and_preprocessing(model_path, preprocessing_dir)
    
    def load_model_and_preprocessing(self, model_path, preprocessing_dir):
        """Load model and preprocessing tools"""
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler and PCA
            with open(os.path.join(preprocessing_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(preprocessing_dir, 'pca.pkl'), 'rb') as f:
                self.pca = pickle.load(f)
            
            # Load metadata
            with open(os.path.join(preprocessing_dir, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
            
        except FileNotFoundError as e:
            st.error(f"Error loading model files: {e}")
            return False
        
        return True
    
    def extract_hog_features(self, image):
        """Extract HOG features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        return features.flatten()
    
    def extract_color_histogram(self, image, bins=32):
        """Extract color histogram from image"""
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
        
        hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def extract_combined_features(self, image):
        """Extract combined features (HOG + color histogram)"""
        hog_features = self.extract_hog_features(image)
        color_features = self.extract_color_histogram(image)
        
        return np.concatenate([hog_features, color_features])
    
    def extract_features(self, image_input):
        """Extract features from image"""
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert('RGB')
            image = np.array(image)
        else:
            image = image_input
            if image.shape[-1] > 3:
                image = image[:, :, :3]
        
        image = cv2.resize(image, (128, 128))
        features = self.extract_combined_features(image)
        
        return features
    
    def predict_skin_tone(self, image):
        """Predict skin tone from image"""
        try:
            features = self.extract_features(image)
            
            expected_features = 34116
            if abs(len(features) - expected_features) > 100:
                st.error(f"Incorrect feature dimension: {len(features)} instead of {expected_features}")
                return None, None
            
            features_scaled = self.scaler.transform([features])
            features_pca = self.pca.transform(features_scaled)
            
            prediction = self.model.predict(features_pca)[0]
            skin_tone = self.idx_to_class[prediction]
            
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_pca)[0]
                confidence = probabilities[prediction]
            
            return skin_tone, confidence
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

def load_custom_css():
    """Load custom CSS for luxury styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(100deg,rgba(26, 26, 26, 0.8) 30%, rgba(212, 175, 55, 0.1) 0%, rgba(26, 26, 26, 0.8) 30%);
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 5px 60px #877439 ;
        animation: fadeInUp 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 50%, rgba(212, 175, 55, 0.15) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
.brand-title {
    font-family: Playfair Display;
    font-size: 2rem; 
    color: #d4af37; 
    margin-bottom: 1rem;
}
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    .brand-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: #f8f9fa;
        margin-top: 1rem;
        font-weight: 300;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }
    
    .tagline {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #d4af37;
        margin-top: 2rem;
        font-style: italic;
        position: relative;
        z-index: 1;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: #d4af37;
        text-align: center;
        margin: 3rem 0 2rem 0;
        font-weight: 600;
        letter-spacing: 0.05em;
        animation: fadeIn 1s ease-out;
    }
    
    /* Upload Section */
    .upload-container {
        background: rgba(45, 45, 45, 0.6);
        padding: 3rem;
        border-radius: 20px;
        border: 2px solid rgba(212, 175, 55, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        animation: fadeInLeft 1s ease-out;
    }
    
    .upload-container:hover {
        border-color: rgba(212, 175, 55, 0.6);
        box-shadow: 0 15px 40px rgba(212, 175, 55, 0.2);
        transform: translateY(-5px);
    }
    
    /* Gender Selection */
    .gender-card {
        background: rgba(45, 45, 45, 0.8);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(212, 175, 55, 0.2);
        text-align: center;
        cursor: pointer;
        transition: all 0.4s ease;
        margin: 1rem;
    }
    
    .gender-card:hover {
        border-color: #d4af37;
        box-shadow: 0 10px 30px rgba(212, 175, 55, 0.3);
        transform: translateY(-8px) scale(1.05);
    }
    
    .gender-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 10px rgba(212, 175, 55, 0.3));
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #d4af37 0%, #f4d47c 100%);
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 1rem 3rem;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        box-shadow: 0 10px 30px rgba(212, 175, 55, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #f4d47c 0%, #d4af37 100%);
        box-shadow: 0 15px 40px rgba(212, 175, 55, 0.6);
        transform: translateY(-3px);
    }
    
    /* Results Section */
    .result-card {
        background: rgba(45, 45, 45, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid rgba(212, 175, 55, 0.3);
        margin: 2rem 0;
        animation: fadeInUp 1s ease-out;
        backdrop-filter: blur(10px);
    }
    
    .skin-tone-result {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        color: #d4af37;
        text-align: center;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        animation: slideIn 0.8s ease-out;
    }
    
    /* Image Gallery */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .gallery-item {
        position: relative;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.4s ease;
        border: 2px solid rgba(212, 175, 55, 0.2);
    }
    
    .gallery-item:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 20px 50px rgba(212, 175, 55, 0.4);
        border-color: #d4af37;
    }
    
    .gallery-item img {
        width: 100%;
        height: auto;
        display: block;
        transition: transform 0.4s ease;
    }
    
    .gallery-item:hover img {
        transform: scale(1.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Text Styles */
    .luxury-text {
        font-family: 'Inter', sans-serif;
        color: #f8f9fa;
        font-size: 1.1rem;
        line-height: 1.8;
        font-weight: 300;
    }
    
    .gold-accent {
        color: #d4af37;
        font-weight: 600;
    }
    
    /* Divider */
    .luxury-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #d4af37 50%, transparent 100%);
        margin: 3rem 0;
        animation: shimmerLine 3s ease-in-out infinite;
    }
    
    @keyframes shimmerLine {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Custom file uploader */
    .stFileUploader {
        background: rgba(45, 45, 45, 0.4);
        border-radius: 15px;
        border: 2px dashed rgba(212, 175, 55, 0.3);
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #d4af37;
        background: rgba(45, 45, 45, 0.6);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: rgba(45, 45, 45, 0.8);
        color: #f8f9fa;
        border: 2px solid rgba(212, 175, 55, 0.3);
        border-radius: 10px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(212, 175, 55, 0.2);
        color: #d4af37;
        border-left: 4px solid #d4af37;
        font-family: 'Inter', sans-serif;
    }
    
    /* Image container */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        border: 2px solid rgba(212, 175, 55, 0.2);
    }
    
    </style>
    """, unsafe_allow_html=True)

def render_hero():
    """Render hero section"""
    st.markdown("""
    <div class="hero-container">
        <h1 class="brand-title">Glowly</h1>
        <p class="brand-subtitle">Luxury Skin Tone Analysis</p>
        <p class="tagline">Discover Your Perfect Palette</p>
     </div>
    """, unsafe_allow_html=True)

def render_divider():
    """Render luxury divider"""
    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

def display_makeup_recommendations(skin_tone):
    """Display makeup recommendations for women"""
    st.markdown('<h2 class="section-header">Your Personalized Makeup Collection</h2>', unsafe_allow_html=True)
    
    makeup_path = f"image/makeup/{skin_tone}"
    
    if os.path.exists(makeup_path):
        cols = st.columns(5)
        for i in range(1, 6):
            img_path = os.path.join(makeup_path, f"m{i}.jpg")
            if os.path.exists(img_path):
                with cols[i-1]:
                    st.image(img_path, use_container_width=True)
                    st.markdown(f'<p style="text-align: center; color: #d4af37; font-family: Inter; margin-top: 0.5rem;"></p>', unsafe_allow_html=True)
    else:
        st.warning(f"Makeup images not found for {skin_tone} skin tone at path: {makeup_path}")
    
    render_divider()
    st.markdown('<h2 class="section-header">Your Recommended Clothing</h2>', unsafe_allow_html=True)
    
    cloth_path = f"image/cloth/female/{skin_tone}/c1.jpg"
    if os.path.exists(cloth_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cloth_path, use_container_width=True)
    else:
        st.warning(f"Clothing image not found at: {cloth_path}")

def display_clothing_recommendations(skin_tone):
    """Display clothing recommendations for men"""
    st.markdown('<h2 class="section-header">Your Recommended Clothing</h2>', unsafe_allow_html=True)
    
    cloth_path = f"image/cloth/male/{skin_tone}/c1.jpg"
    
    if os.path.exists(cloth_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cloth_path, use_container_width=True)
    else:
        st.warning(f"Clothing image not found at: {cloth_path}")

def display_small_palette(skin_tone, gender):
    """Display small color palette under the analyze button"""
    if gender == "Woman":
        palette_path = f"image/cloth/female/{skin_tone}/c2.png"
    else:
        palette_path = f"image/cloth/male/{skin_tone}/c2.png"
    
    if os.path.exists(palette_path):
        col1, col2, col3 = st.columns([4, 20, 4])
        with col2:
            st.markdown('<p style="text-align: center; color: #d4af37; font-family: Playfair Display; font-size: 1.5rem; margin-top: 1rem; margin-bottom: 0.5rem;">Your Color Palette</p>', unsafe_allow_html=True)
            st.image(palette_path, use_container_width=True)
    else:
        st.warning(f"Color palette image not found at: {palette_path}")

def main():
    st.set_page_config(
        page_title="Glowly - Luxury Skin Tone Analysis",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Render hero
    render_hero()
    
    # Initialize predictor
    try:
        predictor = SkinTonePredictor()
    except Exception as e:
        st.error(f"Error initializing predictor: {e}")
        return
    
    # Main content
    render_divider()
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<h2 class="section-header">Upload Your Photo</h2>', unsafe_allow_html=True)
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a clear photo of your face",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a well-lit photo for accurate analysis",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True, caption="Your Photo")
            
            # Store image in session state
            st.session_state['uploaded_image'] = np.array(image)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Select Your Gender</h2>', unsafe_allow_html=True)
        
       # Gender selection with custom styling
        gender_col1, gender_col2 = st.columns(2)
        
        with gender_col1:
            st.markdown("""
            <div class="gender-card">
                <div class="gender-icon">👩‍💼</div>
                <p style="color: #f8f9fa; font-family: Inter; font-size: 1.2rem; font-weight: 500;">Woman</p>
            </div>
            """, unsafe_allow_html=True)
            
        
        with gender_col2:
            st.markdown("""
            <div class="gender-card">
                <div class="gender-icon">👨‍💼</div>
                <p style="color: #f8f9fa; font-family: Inter; font-size: 1.2rem; font-weight: 500;">Man</p>
            </div>
            """, unsafe_allow_html=True)
        
        gender = st.selectbox(
            "Select your gender",
            ["Woman", "Man"],
            label_visibility="collapsed"
        )
        
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        # Analyze button
        col1, col2, col3 = st.columns([5, 15, 5])
        with col2:
         if uploaded_file is not None:
            if st.button("✨ Analyze My Skin Tone", type="primary"):
                with st.spinner("Analyzing your unique skin tone..."):
                    image_array = st.session_state.get('uploaded_image')
                    
                    if image_array is not None:
                        skin_tone, confidence = predictor.predict_skin_tone(image_array)
                        
                        if skin_tone is not None:
                            st.session_state['skin_tone'] = skin_tone
                            st.session_state['confidence'] = confidence
                            st.session_state['gender'] = gender
                            st.rerun()
            
            if 'skin_tone' in st.session_state and 'gender' in st.session_state:
                display_small_palette(st.session_state['skin_tone'], st.session_state['gender'])
    
    # Display results if available
    if 'skin_tone' in st.session_state:
        render_divider()
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="skin-tone-result">{st.session_state["skin_tone"]}</h2>', unsafe_allow_html=True)
        
        if st.session_state.get('confidence'):
            confidence_pct = st.session_state['confidence'] * 100
            st.markdown(f'<p style="text-align: center; color: #f8f9fa; font-family: Inter; font-size: 1.2rem;">Confidence: <span class="gold-accent">{confidence_pct:.1f}%</span></p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        render_divider()
        
        if st.session_state['gender'] == "Woman":
            display_makeup_recommendations(st.session_state['skin_tone'])
        else:
            display_clothing_recommendations(st.session_state['skin_tone'])
        
    
    
    # Footer
    render_divider()
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #f8f9fa; font-family: Inter;">
        <p style="font-family: Playfair Display; font-size: 2rem; color: #d4af37; margin-bottom: 1rem;">GLOWLY</p>
        <p style="font-size: 0.9rem; opacity: 0.7;">Where Science Meets Luxury</p>
        <p style="font-size: 0.8rem; opacity: 0.5; margin-top: 1rem;">© 2025 Glowly. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
