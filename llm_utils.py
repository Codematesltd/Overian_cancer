import together

TOGETHER_API_KEY = "914d4aeccac6e2e89aca8caad3df4e66db83372955616d348cb85bafd0231fb9"
together.api_key = TOGETHER_API_KEY

def get_llm_context(prediction, risk_level, recommendation):
    prompt = f"You are a medical AI assistant. Based on the following ovarian cancer risk prediction, provide a short, clear explanation for the patient.\nRisk Level: {risk_level}\nProbability: {prediction}%\nRecommendation: {recommendation}"
    response = together.Complete.create(
        prompt=prompt,
        model="togethercomputer/llama-2-7b-chat",
        max_tokens=128,
        temperature=0.7
    )
    return response['output']['choices'][0]['text'] if 'output' in response else "No context available."

def analyze_histopath_image(image_path):
    prompt = f"You are a medical AI assistant. Analyze the following histopathological image for ovarian cancer risk. Provide a short summary for the patient. Image path: {image_path}"
    response = together.Complete.create(
        prompt=prompt,
        model="togethercomputer/llama-2-7b-chat",
        max_tokens=128,
        temperature=0.7
    )
    return response['output']['choices'][0]['text'] if 'output' in response else "No summary available."
