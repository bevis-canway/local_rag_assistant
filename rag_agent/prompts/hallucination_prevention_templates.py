"""
防止幻觉的提示词模板
专门用于减少大模型生成回答时的幻觉现象
"""

# 防止幻觉的提示词模板
HALLUCINATION_PREVENTION_TEMPLATES = {
    "strict_factual_response": """You are a fact-checking assistant. Your response must strictly adhere to the provided context.

IMPORTANT RULES:
- Only include information that is explicitly stated in the provided context
- If the answer is not available in the context, clearly state "I cannot find this information in the provided documents"
- Do not make inferences or assumptions beyond what is directly stated
- Do not fabricate, infer, or hallucinate information
- Acknowledge limitations when the context does not provide sufficient information

Context:
{context}

Question: {query}

Response:""",
    
    "document_based_verification": """Verify each statement in your response against the provided documents.

For each statement you make:
1. Identify which document(s) support this statement
2. Quote or reference the specific information
3. If you cannot find support for a statement, do not include it

Documents:
{context}

Question: {query}

Verified Response:""",
    
    "confidence_aware_generation": """Generate a response based on the provided context, but indicate confidence levels.

For each part of your answer:
- High confidence: Information directly stated in documents
- Medium confidence: Reasonable inferences from documents
- Low confidence: Information not clearly supported by documents

If any part of your answer has low confidence, clearly state this limitation.

Context:
{context}

Question: {query}

Response with Confidence Levels:""",
    
    "fact_extraction_first": """First, extract all relevant facts from the documents, then form your response.

Step 1: Extract relevant facts from the following documents:
{context}

Step 2: Identify which facts are relevant to the question: {query}

Step 3: Form your response using only the extracted facts, clearly indicating when information is not available in the documents.

Response:""",
    
    "contradiction_check": """Before providing your answer, check for contradictions with the provided context.

1. Identify all claims you're about to make
2. Verify each claim against the provided documents
3. If any claim contradicts the documents, do not include it
4. If any claim is not supported by the documents, acknowledge this limitation

Context:
{context}

Question: {query}

Verified Response:"""
}