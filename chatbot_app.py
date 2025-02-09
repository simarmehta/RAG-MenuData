
import re
import json
import pickle
import faiss
import uvicorn
import openai
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

openai.api_key = "YOUR_API_KEY"#REPLACE WITH OPENAI API KEY HERE
embedding_model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(embedding_model_name)

faiss_index_file = 'faiss_index.bin'
metadata_file = 'metadata_mapping.pkl'
internal_index = faiss.read_index(faiss_index_file)
with open(metadata_file, 'rb') as f:
    internal_metadata = pickle.load(f)

external_index_file = 'faiss_external.index'
external_mapping_file = 'faiss_external_mapping.json'
external_index = faiss.read_index(external_index_file)
with open(external_mapping_file, 'r', encoding='utf-8') as f:
    external_metadata = json.load(f)
if isinstance(external_metadata, dict):
    external_metadata = [external_metadata[k] for k in sorted(external_metadata.keys(), key=lambda x: int(x))]

app = FastAPI(title="RAG Chatbot API")


class QueryRequest(BaseModel):
    query: str
    #storing the hisrtory
    history: list = []

class QueryResponse(BaseModel):
    query: str
    answer: str
    internal_results: list
    external_results: list
    internal_distances: list
    external_distances: list
    history: list

#Function to extract restaurant name
def extract_restaurant_name(sentence: str) -> str:
    match = re.search(r"Restaurant:\s*([^\.]+)\.", sentence)
    if match:
        return match.group(1).strip().lower()
    return None

# fucntion to deduplicate by restaurant
def deduplicate_by_restaurant(records, distances, top_k=10):
    
    best_by_restaurant = {}
    for record, dist in zip(records, distances):
        restaurant_name = extract_restaurant_name(record.get('sentence', ''))
        if not restaurant_name:
            continue
        if (restaurant_name not in best_by_restaurant or dist < best_by_restaurant[restaurant_name]['distance']):
            best_by_restaurant[restaurant_name] = {'record': record, 'distance': dist}
    deduped = list(best_by_restaurant.values())
    deduped.sort(key=lambda x: x['distance'])
    final = deduped[:top_k]
    final_records = [x['record'] for x in final]
    final_distances = [x['distance'] for x in final]
    return final_records, final_distances



#partial deduplication in case the context isnt clear to provide a response that is remotely acccurate
def partial_deduplicate_by_restaurant(records, distances, top_k=10, max_per_restaurant=5):
    
    by_restaurant = {}
    for record, dist in zip(records, distances):
        restaurant_name = extract_restaurant_name(record.get('sentence', '')) or "unknown"
        by_restaurant.setdefault(restaurant_name, []).append((record, dist))
    
    for rest in by_restaurant:
        by_restaurant[rest].sort(key=lambda x: x[1])
        by_restaurant[rest] = by_restaurant[rest][:max_per_restaurant]
    
    combined = []
    for recs in by_restaurant.values():
        combined.extend(recs)
    combined.sort(key=lambda x: x[1])
    
    final = combined[:top_k]
    final_records = [r[0] for r in final]
    final_distances = [r[1] for r in final]
    return final_records, final_distances

#classify deduplication mode by using llm
def classify_dedup_mode_with_llm(query: str) -> str:
    prompt = f"""Determine the deduplication mode for the following query.
Output one of the following tokens only:
- BY_RESTAURANT (when the user wants one distinct record per restaurant)
- NO_DEDUP (when the user wants full details from a single restaurant)
- PARTIAL (a balanced approach)

Query: "{query}"

Answer:"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that classifies query intents."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0
    )
    classification = response.choices[0].message.content.strip()
    if "BY_RESTAURANT" in classification.upper():
        return "by_restaurant"
    elif "NO_DEDUP" in classification.upper():
        return "no_dedup"
    else:
        return "partial"

#retreive internal records
def retrieve_internal_raw(query: str, initial_k: int = 1000):
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    # faiss.normalize_L2(query_embedding)
    distances, indices = internal_index.search(query_embedding, initial_k)
    raw_records = [internal_metadata[i] for i in indices[0] if i < len(internal_metadata)]
    raw_distances = distances[0].tolist()
    return raw_records, raw_distances

#retreive external record
def retrieve_external_records(query: str, initial_k: int = 10, final_k: int = 5):
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = external_index.search(query_embedding, initial_k)
    raw_records = [external_metadata[i] for i in indices[0] if i < len(external_metadata)]
    raw_distances = distances[0].tolist()
    return raw_records[:final_k], raw_distances[:final_k]

#creating a prompt that would provide the outpt based on the context
def assemble_prompt(query: str, internal_records: list, external_records: list,
                    analytics_summary: str = "", fallback=False, conversation_context="") -> str:
    # conversation history if available.
    conversation_text = f"Conversation History:\n{conversation_context}\n\n" if conversation_context else ""
    print(conversation_text)
    external_context = "\n".join(f"External Record: {rec.get('paragraph', '')}" for rec in external_records)
    internal_context = "\n".join(f"Record {rec.get('item_id', 'N/A')}: {rec.get('sentence', '')}" for rec in internal_records)
    
    instructions = (
        "You are an expert restaurant and cuisine assistant. When answering a query, please follow these instructions:"

        "1. Cuisine Summary:"
        "- Begin by summarizing the basic details about the requested cuisine using any relevant external data."
        "- If no relevant external data is available, omit mentioning external data and rely on internal data instead."

        "2. Restaurant Listings:"
        "- List relevant restaurants based on internal data."
        "- If internal records are not available, do not reference internal data; instead, include any available details from external data."

        "3. Data Disclaimer:"
        "- If the data is limited or contains conflicting information, include an appropriate disclaimer in your response."

    #     "You are an expert restaurant and cuisine assistant. First, answer the query by summarizing "
    #     "basic details about the cuisine based on the external data, if there is no relevant then do not mention anything about external data move to internal data ; then, list relevant restaurants "
    #     "from the internal data, if there is no internal record/data just dont mention anything about internal data,  mention external data. If data is limited or conflicting, include a disclaimer."
     )
    if fallback:
        instructions += " Note: The data available is limited, so include a disclaimer in your answer."
    
    prompt = f"{conversation_text}User Query: {query}\n\n"
    if match := re.search(r'\b\d{5}\b', query):
        prompt += f"Note: The user's location is {match.group()}. Only consider restaurants with this zip code.\n\n" #zip code functinality if there were restaurants in different zip codes to filter by location
    prompt += f"External Cuisine Context:\n{external_context}\n\n"
    prompt += f"Internal Restaurant Context:\n{internal_context}\n\n"
    if analytics_summary:#analytics summary in order to get analytics if prices were justifiably provided in the dataset
        prompt += f"Analytics Summary:\n{analytics_summary}\n\n"
    prompt += f"Instructions: {instructions}\n\n"
    prompt += "Provide your answer by first listing key points from the external context if ther is any , then list the internal restaurant information, if there is any.\n\nAnswer:"
    return prompt


def append_references(answer: str, internal_records: list, external_records: list) -> str:
    internal_refs = [f"Internal Record ID: {rec.get('item_id', 'N/A')}" for rec in internal_records]
    external_refs = [f"External Record: {rec.get('paragraph', '')[:50]}..." for rec in external_records]
    return answer + "\n\nReferences:\n" + "\n".join(internal_refs + external_refs)

#call llm api endpoint based on the prompt assembled
def call_llm(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert restaurant and cuisine assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.6,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error calling OpenAI API: {e}"
    return answer
#comp analysis optional function to provide analytics based on the dataset
def perform_comparative_analysis(query: str, records: list):
    if "compare" in query.lower():
        prices = []
        for record in records:
            try:
                price = float(record.get('price', 0))
                if price > 0:
                    prices.append(price)
            except Exception:
                continue
        if prices:
            avg_price = sum(prices) / len(prices)
            return f"The computed average price is ${avg_price:.2f}."
    return ""

#using an llm to classify whetehr it is external or internal dataset
def classify_internal_vs_external(query: str) -> dict:
    prompt = f"""For the following query, classify the intent into two categories:
    
    1. INTERNAL: details about a restaurant (menus, reviews, location, etc.)
    2. EXTERNAL: general context about cuisine (fun facts, history, flavor, etc.)

Output your answer in JSON format with two boolean fields: "internal" and "external".

Query: "{query}"
    
Answer:"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that classifies query intents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0
        )
        classification = response.choices[0].message.content.strip()
        result = json.loads(classification)
        return {
            "internal": bool(result.get("internal", False)),
            "external": bool(result.get("external", False))
        }
    except Exception as e:
        # Fallback to keyword-based logic if LLM classification fails
        query_lower = query.lower()
        internal = any(kw in query_lower for kw in ["restaurant", "menu", "review", "eat", "serve"])
        external = any(kw in query_lower for kw in ["cuisine", "fun fact", "history", "flavor"])
        return {"internal": internal, "external": external}

#function that brings it all together
def process_query(user_query: str, history: list, dedup_mode: str = None):
    if not user_query.strip():
        fallback_message = "No query provided. Please ask a question about restaurants or cuisine."
        return fallback_message, history, ""
    
    # Build conversation context from the history.
    conversation_context = "\n".join([f"User: {entry['user']}\nBot: {entry['bot']}" for entry in history])
    print(conversation_context)
    
    if dedup_mode is None:
        dedup_mode = classify_dedup_mode_with_llm(user_query)
    
    external_records, external_distances = retrieve_external_records(user_query, initial_k=10, final_k=5)
    raw_internal_records, raw_internal_distances = retrieve_internal_raw(user_query, initial_k=1000)
    
    if dedup_mode == "by_restaurant":
        final_internal_records, final_internal_distances = deduplicate_by_restaurant(
            raw_internal_records, raw_internal_distances, top_k=10
        )
    elif dedup_mode == "no_dedup":
        final_internal_records = raw_internal_records[:10]
        final_internal_distances = raw_internal_distances[:10]
    else:
        final_internal_records, final_internal_distances = partial_deduplicate_by_restaurant(
            raw_internal_records, raw_internal_distances, top_k=10, max_per_restaurant=5
        )
    print(dedup_mode)
    
    # Use the classifier to decide which records to include.
    intent = classify_internal_vs_external(user_query)
    print(intent)
    if intent["internal"] and not intent["external"]:
        used_internal_records = final_internal_records
        used_internal_distances = final_internal_distances
        used_external_records = []
        used_external_distances = []
    elif intent["external"] and not intent["internal"]:
        used_internal_records = []
        used_internal_distances = []
        used_external_records = external_records
        used_external_distances = external_distances
    else:
        used_internal_records = final_internal_records
        used_internal_distances = final_internal_distances
        used_external_records = external_records
        used_external_distances = external_distances
    
    analytics_summary = perform_comparative_analysis(user_query, used_internal_records) if used_internal_records else ""
    
    prompt = assemble_prompt(
        user_query,
        used_internal_records,
        used_external_records,
        analytics_summary,
        fallback=False,
        conversation_context=conversation_context
    )
    print(prompt)
    
    answer = call_llm(prompt)
    answer = append_references(answer, used_internal_records, used_external_records)
    
    # conversation history.
    history.append({"user": user_query, "bot": answer})
    return answer, history, prompt, used_internal_records, used_external_records, used_internal_distances, used_external_distances

#API endpoint
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    answer, updated_history, prompt_used, used_internal_records, used_external_records, used_internal_distances, used_external_distances = process_query(request.query, request.history)
    return QueryResponse(
        query=request.query,
        answer=answer,
        internal_results=used_internal_records,
        external_results=used_external_records,
        internal_distances=used_internal_distances,
        external_distances=used_external_distances,
        history=updated_history
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



























