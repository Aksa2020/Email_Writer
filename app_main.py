import streamlit as st
import re
import requests
import os

# Fix torch classes issue
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

try:
    import torch
    # Force CPU mode to avoid torch.classes issues
    torch.set_default_tensor_type(torch.FloatTensor)
except ImportError:
    torch = None

import warnings
warnings.filterwarnings('ignore')

# 🔑 Load secrets from Streamlit
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", None)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# Set page config
st.set_page_config(
    page_title="Professional Email Writer",
    page_icon="✉️",
    layout="wide"
)

# Initialize session state
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = ""
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False


def create_professional_prompt(context, message_type, tone, key_points):
    """Create well-structured prompts optimized for different models"""
    if message_type == "email":
        prompt = f"""Task: Write a professional business email.

Context: {context}
Tone: {tone}
Key points: {key_points}

Format:
Subject: [Clear subject line]
Dear [Name],

[Professional opening sentence]
[Main content with key points]
[Professional closing]

Best regards,
[Your name]

Write the complete email:"""

    elif message_type == "message":
        prompt = f"""Task: Write a professional business message.

Context: {context}
Tone: {tone}
Key points: {key_points}

Write a direct professional message (no email format):"""

    else:  # letter
        prompt = f"""Task: Write a formal business letter.

Context: {context}
Tone: {tone}
Key points: {key_points}

Format:
[Date]
[Address]
Dear [Name],

[Formal letter content]

Sincerely,
[Your name]

Write the complete letter:"""

    return prompt


def generate_with_gamma3(prompt):
    """Generate content using Hugging Face Gamma 3 model"""
    try:
        url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1
            },
            "options": {"wait_for_model": True}
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No content generated")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
        elif response.status_code == 503:
            return "Model is loading, please try again in a moment..."
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error calling Hugging Face Gamma 3 API: {str(e)}"


def generate_with_flan_t5_api(prompt):
    """Generate content using Hugging Face Flan-T5 API"""
    try:
        url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No content generated")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
        elif response.status_code == 503:
            return "Model is loading, please try again in a moment..."
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error calling Hugging Face Flan-T5 API: {str(e)}"


def generate_with_groq(prompt):
    """Generate content using Groq API"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a professional business writer. Write clear, professional communications."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error calling Groq API: {str(e)}"


@st.cache_resource
def load_local_flan_t5():
    """Load local Flan-T5 model"""
    try:
        if torch is None:
            st.error("PyTorch not installed.")
            return None, None

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        model = model.to('cpu')
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model

    except Exception as e:
        st.error(f"Local model failed: {str(e)}")
        return None, None


def generate_with_local_flan_t5(prompt, tokenizer, model):
    """Generate content using local Flan-T5"""
    try:
        device = 'cpu'
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                min_length=50,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                num_beams=3,
                repetition_penalty=1.2,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        return f"Generation failed: {str(e)}"


def post_process_content(content):
    """Clean and format the generated content"""
    if not content or content.startswith("Error"):
        return content

    content = re.sub(r'\b(\w+\s+){3,}\1', r'\1', content)
    sentences = content.split('. ')
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    content = '. '.join(sentences)

    if not content.endswith(('.', '!', '?')):
        content += '.'

    return content


# ---------- Streamlit UI ----------
st.title("✉️ Professional Email & Message Writer")
st.markdown("*Transform your ideas into polished, professional communication*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    model_option = st.selectbox(
        "AI Model",
        ["Local Flan-T5 (Free)", "Flan-T5 API", "Groq API", "Gamma 3 API"]
    )

    if "API" in model_option:
        if "Flan-T5" in model_option or "Gamma" in model_option:
            if HUGGINGFACE_API_KEY != "hf_your_huggingface_api_key_here":
                st.success("🔑 Hugging Face API Key: Configured")
            else:
                st.warning("⚠️ Hugging Face API Key: Not configured")
        elif "Groq" in model_option:
            if GROQ_API_KEY != "gsk_your_groq_api_key_here":
                st.success("🔑 Groq API Key: Configured")
            else:
                st.warning("⚠️ Groq API Key: Not configured")

    st.markdown("---")
    message_type = st.selectbox("Message Type", ["email", "message", "letter"])
    tone = st.selectbox("Tone", ["formal", "friendly", "urgent", "apologetic", "grateful", "informative"])


# Input
col1, col2 = st.columns([1, 1])
with col1:
    st.header("📝 Input")
    context = st.text_area("Context/Background", placeholder="Describe the situation...", height=150)
    key_points = st.text_area("Key Points", placeholder="List the main points...", height=100)
    generate_btn = st.button("🚀 Generate Professional Content", type="primary", use_container_width=True)

# Output
with col2:
    st.header("📧 Generated Content")

    if generate_btn and context and key_points:
        prompt = create_professional_prompt(context, message_type, tone, key_points)
        with st.spinner("Crafting your professional message..."):
            generated_content = ""

            if "Local" in model_option:
                tokenizer, model = load_local_flan_t5()
                if tokenizer and model:
                    generated_content = generate_with_local_flan_t5(prompt, tokenizer, model)
                    st.success("✅ Generated using local Flan-T5")
                else:
                    st.error("❌ Local model not available.")
            elif "Flan-T5 API" in model_option:
                generated_content = generate_with_flan_t5_api(prompt)
            elif "Gamma 3 API" in model_option:
                generated_content = generate_with_gamma3(prompt)
            elif "Groq API" in model_option:
                generated_content = generate_with_groq(prompt)

            if generated_content and not generated_content.startswith("Error"):
                st.session_state.generated_content = post_process_content(generated_content)
                st.session_state.edit_mode = False
            else:
                st.error(generated_content)

    if st.session_state.generated_content:
        col_edit, col_copy = st.columns(2)
        with col_edit:
            if st.button("✏️ Edit"):
                st.session_state.edit_mode = not st.session_state.edit_mode
        with col_copy:
            if st.button("📋 Copy"):
                st.success("Content ready to copy!")

        if st.session_state.edit_mode:
            edited_content = st.text_area("Edit your content:", value=st.session_state.generated_content, height=300)
            if st.button("💾 Save Changes"):
                st.session_state.generated_content = edited_content
                st.session_state.edit_mode = False
                st.rerun()
        else:
            st.text_area("Generated Content", value=st.session_state.generated_content, height=300, disabled=True)

            word_count = len(st.session_state.generated_content.split())
            char_count = len(st.session_state.generated_content)
            col_metrics1, col_metrics2 = st.columns(2)
            col_metrics1.metric("Word Count", word_count)
            col_metrics2.metric("Character Count", char_count)
