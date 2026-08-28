# ============================================================
# MARKETRON RAG RETRIEVAL TEST
# ============================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.rag_engine import retrieve_marketing_context


TEST_QUERIES = {
    "High Value": (
        "High-value customers with high frequency, recent purchases, "
        "and high monetary value. Find useful marketing tactics for "
        "loyalty, VIP treatment, exclusivity, retention, cross-selling, "
        "and increasing long-term customer value."
    ),

    "Promising": (
        "Promising customers with recent engagement, moderate purchase "
        "frequency and meaningful spending. Find marketing tactics for "
        "nurturing, repeat purchases, personalization, engagement, "
        "and moving customers toward stronger loyalty."
    ),

    "At Risk": (
        "At-risk customers with declining engagement and increasing "
        "recency. Find practical retention, reactivation, win-back, "
        "behavioral segmentation, email, loyalty and urgency tactics."
    ),

    "Churned/Lost": (
        "Churned or long-inactive customers with very high recency. "
        "Find practical win-back, re-engagement, retention, lapsed-"
        "customer and email campaign tactics."
    ),
}


# ============================================================
# RUN RETRIEVAL TESTS
# ============================================================

print("=" * 70)
print("MARKETRON RAG RETRIEVAL TEST")
print("=" * 70)

for segment_name, query in TEST_QUERIES.items():

    print(f"\n{'=' * 70}")
    print(f"SEGMENT: {segment_name}")
    print(f"{'=' * 70}")

    print(f"\nQUERY:\n{query}\n")

    context = retrieve_marketing_context(query)

    if not context:
        print("NO CONTEXT RETRIEVED.")
        continue

    print("RETRIEVED CONTEXT:\n")
    print(context)

print("\n" + "=" * 70)
print("RAG RETRIEVAL TEST COMPLETE")
print("=" * 70)

# 🔄 Attempting Gemini: gemini-3.5-flash-lite (Key 1)...
# Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.
# ❌ ERROR on Gemini gemini-3.5-flash-lite: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-3.5-flash-lite (Key 2)...
# ❌ ERROR on Gemini gemini-3.5-flash-lite: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-3.5-flash-lite (Key 3)...
# ❌ ERROR on Gemini gemini-3.5-flash-lite: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-2.5-flash (Key 1)...
# ❌ ERROR on Gemini gemini-2.5-flash: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-2.5-flash (Key 2)...
# ❌ ERROR on Gemini gemini-2.5-flash: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-2.5-flash (Key 3)...
# ❌ ERROR on Gemini gemini-2.5-flash: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-2.5-flash-lite (Key 1)...
# ❌ ERROR on Gemini gemini-2.5-flash-lite: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-2.5-flash-lite (Key 2)...
# ❌ ERROR on Gemini gemini-2.5-flash-lite: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Gemini: gemini-2.5-flash-lite (Key 3)...
# ❌ ERROR on Gemini gemini-2.5-flash-lite: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Manually set deadline 8s is too short. Minimum allowed deadline is 10s.', 'status': 'INVALID_ARGUMENT'}}
# 🔄 Attempting Groq: llama3-70b-8192 (Key 1)...
# ❌ ERROR on Groq llama3-70b-8192: Error code: 400 - {'error': {'message': 'The model `llama3-70b-8192` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: llama3-70b-8192 (Key 2)...
# ❌ ERROR on Groq llama3-70b-8192: Error code: 400 - {'error': {'message': 'The model `llama3-70b-8192` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: llama3-70b-8192 (Key 3)...
# ❌ ERROR on Groq llama3-70b-8192: Error code: 400 - {'error': {'message': 'The model `llama3-70b-8192` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: mixtral-8x7b-32768 (Key 1)...
# ❌ ERROR on Groq mixtral-8x7b-32768: Error code: 400 - {'error': {'message': 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: mixtral-8x7b-32768 (Key 2)...
# ❌ ERROR on Groq mixtral-8x7b-32768: Error code: 400 - {'error': {'message': 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: mixtral-8x7b-32768 (Key 3)...
# ❌ ERROR on Groq mixtral-8x7b-32768: Error code: 400 - {'error': {'message': 'The model `mixtral-8x7b-32768` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: gemma2-9b-it (Key 1)...
# ❌ ERROR on Groq gemma2-9b-it: Error code: 400 - {'error': {'message': 'The model `gemma2-9b-it` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: gemma2-9b-it (Key 2)...
# ❌ ERROR on Groq gemma2-9b-it: Error code: 400 - {'error': {'message': 'The model `gemma2-9b-it` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}
# 🔄 Attempting Groq: gemma2-9b-it (Key 3)...
# ❌ ERROR on Groq gemma2-9b-it: Error code: 400 - {'error': {'message': 'The model `gemma2-9b-it` has been decommissioned and is no longer supported. Please refer to https://console.groq.com/docs/deprecations for a recommendation on which model to use instead.', 'type': 'invalid_request_error', 'code': 'model_decommissioned'}}