import dspy
import boto3
import json
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

# Paper: https://arxiv.org/abs/2403.08295
# Code: https://github.com/google-deepmind/dsp

# Configure Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Set up DSPy to use Claude 3 via Bedrock
class AWSAnthropicLM(dspy.LM):
    def __init__(self):
        super().__init__(model="anthropic.claude-3-sonnet-20240229-v1:0")
    
    def basic_request(self, prompt, **kwargs):
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 500,
                "temperature": 0.7,
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        return json.loads(response['body'].read())['completion']

dspy.settings.configure(lm=AWSAnthropicLM())

# Load a small subset of the SQuAD dataset
dataset = load_dataset("squad", split="train[:1000]")

# Print a sample to understand the structure
print("Dataset sample:")
print(json.dumps(dataset[0], indent=2))

# Define RAG pipeline with Chain of Thought
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define signature with improved prompting
        signature = "context: str, question: str -> reasoning: str, answer: str"
        self.generate_answer = dspy.ChainOfThought(signature)
        # Add instructions for better reasoning
        self.generate_answer.predict.instructions = """Given a context and question, let's solve this step by step:
1. First, carefully read and understand the question
2. Then, locate the relevant information in the context
3. Finally, provide a clear, concise answer

Your reasoning should be clear and your answer should be precise and directly answer the question."""

    def forward(self, context, question):
        result = self.generate_answer(context=context, question=question)
        return result

rag = RAG()

# Define Chain of Draft RAG
class CoD_RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define signatures following the paper's methodology
        draft_signature = "context: str, question: str -> draft_thoughts: str, key_points: str"
        answer_signature = "draft_thoughts: str, key_points: str -> answer: str"
        
        self.generate_draft = dspy.Predict(draft_signature)
        self.generate_answer = dspy.Predict(answer_signature)
        
        # Instructions following the paper's CoD methodology
        self.generate_draft.instructions = """Given a context and question, create a minimal draft:
1. Use concise notation and symbols where possible (e.g., →, =, ∈)
2. Extract only essential information using abbreviated forms
3. Focus on key entities and relationships
4. Identify the target information the question seeks
5. Use mathematical or logical notation when applicable

Format your response as:
draft_thoughts: [your minimal notation/symbols]
key_points: [list of crucial facts]"""
        
        self.generate_answer.instructions = """Based on the draft thoughts and key points:
1. Interpret the minimal notation
2. Construct a precise answer using only necessary information
3. Ensure the answer directly addresses the question
4. Be concise but complete
5. Use the exact wording from the context when appropriate"""

    def forward(self, context, question):
        # Generate minimal draft and key points
        draft_result = self.generate_draft(context=context, question=question)
        # Use both draft thoughts and key points to generate final answer
        result = self.generate_answer(
            draft_thoughts=draft_result.draft_thoughts,
            key_points=draft_result.key_points
        )
        return result

cod_rag = CoD_RAG()

# Update evaluation function to better track metrics from the paper
def evaluate_model(model, dataset, num_samples=5):  # Reduced samples for testing
    metrics = {
        'correct': 0,
        'total_tokens': 0,
        'total_time': 0,
        'total_steps': 0,
        'responses': [],
        'draft_lengths': [],
        'answer_lengths': []
    }
    
    for i in tqdm(range(num_samples)):
        example = dataset[i]
        try:
            start_time = time.time()
            result = model(context=example['context'], question=example['question'])
            end_time = time.time()
            
            if isinstance(model, RAG):
                # For CoT, track full reasoning
                tokens = len(str(result.reasoning).split()) + len(str(result.answer).split())
                steps = len(str(result.reasoning).split('\n'))
                metrics['draft_lengths'].append(0)  # No draft for CoT
                metrics['answer_lengths'].append(len(str(result.answer).split()))
            else:  # CoD_RAG
                # For CoD, track both draft and answer separately
                draft_tokens = len(str(result.draft_thoughts).split()) if hasattr(result, 'draft_thoughts') else 0
                answer_tokens = len(str(result.answer).split())
                tokens = draft_tokens + answer_tokens
                steps = 2
                metrics['draft_lengths'].append(draft_tokens)
                metrics['answer_lengths'].append(answer_tokens)
            
            metrics['total_tokens'] += tokens
            metrics['total_time'] += (end_time - start_time)
            metrics['total_steps'] += steps
            
            # Improved answer comparison based on paper's evaluation
            predicted = str(result.answer).strip().lower()
            reference_answers = [ans.strip().lower() for ans in example['answers']['text']]
            
            # Store detailed response for analysis
            metrics['responses'].append({
                'question': example['question'],
                'predicted': predicted,
                'actual': reference_answers,
                'tokens': tokens,
                'time': end_time - start_time,
                'steps': steps,
                'draft_length': metrics['draft_lengths'][-1],
                'answer_length': metrics['answer_lengths'][-1]
            })
            
            # Exact match or substantial overlap
            is_correct = any(
                predicted == ref_answer or
                (len(predicted.split()) > 2 and predicted in ref_answer) or
                (len(ref_answer.split()) > 2 and ref_answer in predicted)
                for ref_answer in reference_answers
            )
            
            if is_correct:
                metrics['correct'] += 1
                print(f"\nCorrect answer for example {i}:")
                print(f"Question: {example['question']}")
                print(f"Predicted: {predicted}")
                print(f"Reference: {reference_answers}")
                
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    # Calculate aggregate metrics as per paper
    num_samples = max(len(metrics['responses']), 1)
    return {
        'accuracy': metrics['correct'] / num_samples,
        'avg_tokens': metrics['total_tokens'] / num_samples,
        'avg_time': metrics['total_time'] / num_samples,
        'avg_steps': metrics['total_steps'] / num_samples,
        'token_efficiency': metrics['total_tokens'] / (metrics['correct'] + 1e-6),
        'avg_draft_length': sum(metrics['draft_lengths']) / num_samples,
        'avg_answer_length': sum(metrics['answer_lengths']) / num_samples,
        'responses': metrics['responses']
    }

# Evaluate both models
print("\nEvaluating Chain of Thought (CoT)...")
cot_metrics = evaluate_model(rag, dataset)

print("\nEvaluating Chain of Draft (CoD)...")
cod_metrics = evaluate_model(cod_rag, dataset)

# Print detailed results
print("\nDetailed Comparison:")
print(f"{'Metric':<20} {'CoT':>10} {'CoD':>10}")
print("-" * 42)
print(f"{'Accuracy':<20} {cot_metrics['accuracy']:>10.2%} {cod_metrics['accuracy']:>10.2%}")
print(f"{'Avg Tokens':<20} {cot_metrics['avg_tokens']:>10.1f} {cod_metrics['avg_tokens']:>10.1f}")
print(f"{'Avg Time (s)':<20} {cot_metrics['avg_time']:>10.3f} {cod_metrics['avg_time']:>10.3f}")
print(f"{'Avg Steps':<20} {cot_metrics['avg_steps']:>10.1f} {cod_metrics['avg_steps']:>10.1f}")
print(f"{'Token Efficiency':<20} {cot_metrics['token_efficiency']:>10.1f} {cod_metrics['token_efficiency']:>10.1f}")

# Enhanced plotting
plt.style.use('default')  # Use default style instead of seaborn
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Set a consistent color palette
colors = ['#2ecc71', '#3498db']

# Accuracy comparison
labels = ['Chain of Thought', 'Chain of Draft']
accuracies = [cot_metrics['accuracy'], cod_metrics['accuracy']]
ax1.bar(labels, accuracies, color=colors)
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# Token usage comparison
tokens = [cot_metrics['avg_tokens'], cod_metrics['avg_tokens']]
ax2.bar(labels, tokens, color=colors)
ax2.set_ylabel('Average Tokens')
ax2.set_title('Token Usage Comparison')
ax2.grid(True, alpha=0.3)

# Time comparison
times = [cot_metrics['avg_time'], cod_metrics['avg_time']]
ax3.bar(labels, times, color=colors)
ax3.set_ylabel('Average Time (s)')
ax3.set_title('Processing Time Comparison')
ax3.grid(True, alpha=0.3)

# Token efficiency comparison
efficiencies = [cot_metrics['token_efficiency'], cod_metrics['token_efficiency']]
ax4.bar(labels, efficiencies, color=colors)
ax4.set_ylabel('Tokens per Correct Answer')
ax4.set_title('Token Efficiency Comparison')
ax4.grid(True, alpha=0.3)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Save detailed results
results = {
    'cot': cot_metrics,
    'cod': cod_metrics,
    'timestamp': datetime.datetime.now().isoformat()
}
with open('comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)